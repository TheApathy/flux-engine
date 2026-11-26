"""Source wiring only. Native mock regressions require a compiler window."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "crates/spark-runtime/src"


def compact(text):
    return " ".join(text.split()).replace(" .", ".").replace("!( ", "!(")


class PinnedBounceSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bounce = (RUNTIME / "fast_weights/bounce_copy.rs").read_text()
        cls.backend = (RUNTIME / "cuda_backend/gpu_impl.rs").read_text()
        cls.table = (RUNTIME / "cuda_backend/allocation_lifetime.rs").read_text()
        cls.gpu = (RUNTIME / "gpu.rs").read_text()

    def test_poison_precedes_calls_and_no_retry_loop_remains(self):
        copy = self.bounce.split("pub(super) fn copy(", 1)[1].split("fn copy_once(", 1)[0]
        self.assertLess(copy.index("self.poisoned = true"), copy.index("self.copy_once"))
        self.assertIn("ensure!(!self.poisoned", compact(copy))
        self.assertIn("if result.is_ok()", copy)
        self.assertNotIn("MAX_RETRIES", self.bounce)
        self.assertNotIn("attempt +=", self.bounce)

    def test_event_and_exact_destination_ownership_precede_async_submission(self):
        copy = self.bounce.split("fn copy_once(", 1)[1].split("pub(super) fn drain(", 1)[0]
        markers = ["self.gpu.create_event()?", "self.gpu.retain_h2d_destination(owning_base, destination_offset, len)",
                   "slot.flight = Flight::Untracked", "self.gpu.copy_h2d_pinned_async",
                   "self.gpu.record_event", "slot.flight = Flight::Recorded"]
        indexes = [compact(copy).index(marker) for marker in markers]
        self.assertEqual(indexes, sorted(indexes))
        self.assertNotIn(".take()", copy)
        self.assertIn(".checked_add(u64::try_from(destination_offset)?)", copy)
        self.assertIn("destination: owning_base", copy)

    def test_slot_wait_precedes_any_overwrite(self):
        copy = self.bounce.split("fn copy_once(", 1)[1]
        self.assertLess(copy.index("wait_slot_bounded"), copy.index("copy_from_slice"))

    def test_unrecorded_events_are_never_polled(self):
        wait = self.bounce.split("fn wait_slot_bounded(", 1)[1]
        untracked = wait.split("Flight::Untracked", 1)[1].split("Flight::Recorded", 1)[0]
        self.assertIn("bail!", untracked)
        self.assertNotIn("poll_event", untracked)
        self.assertIn("Flight::Recorded", wait.split("gpu.poll_event", 1)[0])

    def test_only_positive_completion_releases_sources_and_destination(self):
        wait = self.bounce.split("fn wait_slot_bounded(", 1)[1]
        markers = ["if gpu.poll_event(event)?", "slot.flight = Flight::Idle",
                   "gpu.release_h2d_destination(destination)?", "gpu.destroy_event(event)?"]
        indexes = [wait.index(marker) for marker in markers]
        self.assertEqual(indexes, sorted(indexes))
        timeout = wait.split("if Instant::now() >= deadline", 1)[1]
        self.assertNotIn("Flight::Idle", timeout)
        self.assertNotIn("release_h2d_destination", timeout)

    def test_drop_retains_uncertain_sources_and_keeps_backend_borrowed(self):
        drop = self.bounce.split("impl Drop for Slot", 1)[1].split("pub(super) struct", 1)[0]
        self.assertIn("self.flight != Flight::Idle", drop)
        self.assertIn("std::mem::forget(buffer)", drop)
        self.assertIn("gpu: &'a dyn GpuBackend", self.bounce)
        self.assertIn("impl Drop for PinnedBounceCopier<'_>", self.bounce)
        self.assertIn("AssertUnwindSafe(|| self.drain())", self.bounce)

    def test_drain_attempts_every_slot_with_shared_deadline_and_catches_panics(self):
        drain = self.bounce.split("pub(super) fn drain(", 1)[1].split("impl Drop for PinnedBounceCopier", 1)[0]
        self.assertLess(drain.index("let deadline ="), drain.index("for slot in &mut self.slots"))
        self.assertIn("wait_slot_bounded(self.gpu, slot, deadline)", drain)
        self.assertIn("catch_unwind", drain)
        self.assertIn("first_error", drain)
        self.assertLess(drain.index("first_error = Some(error)"), drain.index("return Err(error)"))

    def test_backend_exact_base_and_checked_range_retention_share_allocation_mutex(self):
        retention = self.backend.split("fn retain_h2d_destination(", 1)[1].split("fn stream_is_capturing", 1)[0]
        self.assertIn("self.live_allocations", compact(retention))
        self.assertIn(".get_mut(&base.0)", retention)
        self.assertIn(".retain(offset, bytes)", retention)
        self.assertIn(".release()", retention)
        self.assertIn(".checked_add(bytes)", retention)
        for marker in ("offset.checked_add(bytes)", "end <= self.bytes", "pending_h2d.checked_add(1)", "pending_h2d.checked_sub(1)"):
            self.assertIn(marker, compact(self.table).replace(" .", "."))

    def test_individual_free_claims_under_mutex_before_any_driver_free(self):
        free = self.backend.split("fn free(&self", 1)[1].split("fn free_all_allocations", 1)[0]
        self.assertLess(compact(free).index("self.live_allocations"), compact(free).index("take_for_free"))
        self.assertLess(free.index("take_for_free"), free.index("cuMemFree_v2"))
        self.assertIn("self.restore_allocation", free)
        take = self.table.split("pub(super) fn take_for_free", 1)[1].split("pub(super) fn take_releasable", 1)[0]
        self.assertLess(take.index("allocation.pending_h2d == 0"), take.index("live.remove(&base)"))

    def test_bulk_free_atomically_partitions_and_reports_retained_destinations(self):
        bulk = self.backend.split("fn free_all_allocations", 1)[1].split("fn copy_h2d", 1)[0]
        self.assertIn("super::allocation_lifetime::take_releasable(&mut live)", bulk)
        self.assertLess(bulk.index("take_releasable"), bulk.index("let status = unsafe { cuMemFree_v2"))
        self.assertIn("if failures > 0 || retained > 0", bulk)
        self.assertIn("self.restore_allocation(dptr, allocation)", bulk)
        self.assertIn("allocation.pending_h2d == 0", self.table)

    def test_loader_uses_borrowed_copier_without_blocking_sync_or_process_exit(self):
        loader = (RUNTIME / "fast_weights/mod.rs").read_text()
        self.assertIn("tensors.iter().map(|tensor| tensor.len).max().unwrap_or(0)", loader)
        self.assertIn(
            "PinnedBounceCopier::new(gpu, max_tensor_bytes, expert_scalar_layout.allocation_bytes)?",
            compact(loader),
        )
        self.assertIn("max_tensor_bytes.max(scalar_slab_bytes).clamp(1,CHUNK_BYTES)", "".join(self.bounce.split()))
        self.assertIn("bounce.copy(host, *base)?", loader)
        self.assertIn("bounce.copy(src, p)?", loader)
        self.assertIn("bounce.drain()?", loader)
        for forbidden in ("gpu.synchronize(", "process::abort", "process::exit", "cuStreamSynchronize"):
            self.assertNotIn(forbidden, self.bounce)
        self.assertIn("No-op defaults are valid only for", self.gpu)

    def test_single_chunk_slot_selection_preserves_multichunk_and_empty_paths(self):
        selector = self.bounce.split("fn shard_slot_count(", 1)[1].split("#[derive", 1)[0]
        self.assertIn("max_tensor_bytes.max(scalar_slab_bytes)", selector)
        self.assertRegex(selector, r"if largest_payload > 0 && largest_payload <= CHUNK_BYTES\s*\{\s*1\s*\}\s*else\s*\{\s*SLOTS")
        self.assertIn("Self::with_limits(gpu, slot_count, chunk_bytes, EVENT_TIMEOUT)?", self.bounce)
        self.assertIn("requested_pool_bytes = slot_count * chunk_bytes", self.bounce)

    def test_eleven_native_regressions_are_defined_not_executed(self):
        native = (RUNTIME / "fast_weights/bounce_copy_tests.rs").read_text() + self.table
        self.assertIn("self.inner.memset_async(ptr, value, bytes, stream)", native)
        names = (
            "successful_copy_drains_and_releases_each_source_event_and_destination_reference",
            "timeout_keeps_event_and_bytes_and_forbids_retry_or_overwrite",
            "submission_and_record_errors_never_poll_unrecorded_events_or_release_dma_owners",
            "create_or_retain_failure_submits_nothing_and_drops_idle_source_normally",
            "poll_error_retains_owners_but_completed_destroy_or_release_errors_do_not_retain_source",
            "caught_submission_panic_poison_prevents_reuse_and_drop_retains_dma_owners",
            "drain_attempts_later_slots_after_error_or_panic_in_first_slot",
            "later_proven_completion_releases_quarantine_but_never_unpoisons_failed_copier",
            "exact_base_and_checked_subrange_are_required_before_submission",
            "retained_allocation_survives_both_free_paths_until_every_reference_completes",
            "checked_ranges_and_reference_arithmetic_fail_without_changing_ownership",
        )
        for name in names:
            self.assertRegex(native, r"#\[test\]\s*fn " + re.escape(name) + r"\(\)")


if __name__ == "__main__":
    unittest.main()
