"""Source contracts only. These do not compile or execute the Rust copier."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class GroupedBounceRangesSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = ROOT / "crates/spark-runtime/src/fast_weights"
        cls.source = (directory / "bounce_copy.rs").read_text()
        cls.native = (directory / "bounce_copy_tests.rs").read_text()
        cls.legacy = cls.source.split("pub(super) fn copy(", 1)[1].split(
            "pub(super) fn copy_into_owned_range(", 1
        )[0]
        cls.entry = cls.source.split("pub(super) fn copy_into_owned_range(", 1)[1].split(
            "fn copy_once(", 1
        )[0]
        cls.once = cls.source.split("fn copy_once(", 1)[1].split(
            "pub(super) fn drain(", 1
        )[0]
        cls.wait = cls.source.split("fn wait_slot_bounded(", 1)[1]

    def test_legacy_copy_delegates_to_zero_offset(self):
        self.assertIn("self.copy_into_owned_range(src, dst, 0)", self.legacy)
        self.assertNotIn("self.gpu", self.legacy)

    def test_complete_arithmetic_is_checked_before_any_chunk_submission(self):
        before_loop = self.once.split("while offset < src.len()", 1)[0]
        self.assertIn(".checked_add(src.len())", before_loop)
        self.assertIn(".checked_add(u64::try_from(end)?)", before_loop)
        self.assertIn(".checked_add(offset)", self.once)
        self.assertIn(".checked_add(u64::try_from(destination_offset)?)", self.once)
        self.assertNotIn(" as u64", self.once)

    def test_every_chunk_lease_and_flight_retains_the_owning_base(self):
        compact = " ".join(self.once.split())
        self.assertIn(
            ".retain_h2d_destination(owning_base, destination_offset, len)", compact
        )
        self.assertEqual(self.once.count("destination: owning_base"), 2)
        self.assertNotIn("destination: chunk_dst", self.once)
        self.assertLess(self.once.index("retain_h2d_destination"), self.once.index("Flight::Untracked"))
        self.assertLess(self.once.index("Flight::Untracked"), self.once.index("copy_h2d_pinned_async"))
        self.assertLess(self.once.index("record_event"), self.once.index("Flight::Recorded"))

    def test_range_entry_preserves_sticky_poison_on_errors_or_panics(self):
        self.assertIn("!self.poisoned", self.entry)
        self.assertLess(self.entry.index("self.poisoned = true"), self.entry.index("self.copy_once"))
        self.assertIn("self.copy_once(src, owning_base, view_offset)", self.entry)
        self.assertIn("if result.is_ok()", self.entry)
        self.assertEqual(self.entry.count("self.poisoned = false"), 1)

    def test_uncertain_completion_and_base_release_contract_is_unchanged(self):
        self.assertIn("Flight::Untracked { event, destination } => bail!", self.wait)
        self.assertIn("Flight::Recorded { event, destination } => (event, destination)", self.wait)
        self.assertLess(self.wait.index("gpu.poll_event(event)?"), self.wait.index("gpu.release_h2d_destination(destination)?"))
        self.assertIn("source/event/destination retained", self.wait)
        self.assertIn("std::mem::forget(buffer)", self.source)
        self.assertIn("self.poisoned = was_poisoned", self.source)
        self.assertNotIn("synchronize(", self.source)

    def test_strict_native_regressions_are_defined_but_not_executed(self):
        for name in (
            "grouped_views_copy_exact_tails_without_touching_guards_or_freeing_aliases",
            "grouped_range_arithmetic_overflow_submits_no_prefix_and_poison_is_sticky",
            "grouped_invalid_base_or_first_chunk_range_submits_nothing",
            "empty_owned_range_uses_no_backend_lease_or_submission",
            "grouped_uncertain_sibling_retains_only_its_base_lease_and_host_slot",
            "grouped_poll_failure_drains_completed_sibling_but_retains_uncertain_base",
            "grouped_timeout_can_later_release_base_without_allowing_retry",
            "grouped_submission_panic_keeps_base_and_host_slot_quarantined",
        ):
            self.assertIn("fn " + name, self.native)
        self.assertIn('self.inner.read_alloc(ptr).is_some(), "not a live owning base"', " ".join(self.native.split()))
        self.assertIn('expect("submission requires its own retained range")', self.native)
        self.assertIn('assert!(self.refs(base) > 0, "submission requires a live base lease")', self.native)
        self.assertIn("Some(dst.0)", self.native)


if __name__ == "__main__":
    unittest.main()
