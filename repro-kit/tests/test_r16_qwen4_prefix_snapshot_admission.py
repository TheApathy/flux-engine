# SPDX-License-Identifier: AGPL-3.0-only
"""Source-only checks for Qwen4 PLE prefix-snapshot admission and C2+ HSS guards.

Verifies that `SsmSnapshotPool` persists PLE convolution history alongside GDN
state (`ple_conv_snapshots` / `decode_ple_conv_snapshots`), admitting C1 and C2+
Qwen4 prefix caching while keeping C2+ `--high-speed-swap` fail-closed.
"""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "crates/spark-server/src/main_modules/qwen4_concurrency.rs"
SNAPSHOT = ROOT / "crates/spark-model/src/model/ssm_snapshot.rs"


def assert_c1_admitted_and_c2_hss_guarded(source: str) -> None:
    production = source.split("#[cfg(test)]", 1)[0]
    early_c1 = production.index("if c.max_batch_size <= 1 {")
    c2_guard = re.search(
        r"if\s+c\.high_speed_swap\s*"
        r"\{\s*return Err\((?P<error>[^}]+)\);\s*\}",
        production,
    )
    assert c2_guard is not None, "missing C2+ HSS-lifecycle rejection"
    assert early_c1 < c2_guard.start(), "C2+ HSS rejection must sit after C1 admission"
    assert "high-speed-swap" in c2_guard["error"]


class Qwen4PrefixSnapshotAdmission(unittest.TestCase):
    def test_snapshot_pool_owns_ple_conv_marconi_and_decode_regions(self):
        snap_source = SNAPSHOT.read_text()
        self.assertIn("ple_conv_snapshots: Vec<DevicePtr>", snap_source)
        self.assertIn("decode_ple_conv_snapshots: Vec<DevicePtr>", snap_source)
        self.assertIn("attach_ple_conv_layers", snap_source)

    def test_production_admits_c1_and_guards_c2_plus_high_speed_swap(self):
        assert_c1_admitted_and_c2_hss_guarded(POLICY.read_text())

    def test_source_check_detects_missing_c2_guard(self):
        mutated = POLICY.read_text().replace(
            "if c.high_speed_swap",
            "if false",
            1,
        )
        with self.assertRaises(AssertionError):
            assert_c1_admitted_and_c2_hss_guarded(mutated)

    def test_loader_runs_production_policy_before_gpu_initialization(self):
        loader = (ROOT / "crates/spark-server/src/main_modules/serve_load.rs").read_text()
        admission = loader.index("super::qwen4_concurrency::validate_qwen4_concurrency(")
        initialization = loader.index("serve_phases::init_gpu_backend(")
        self.assertLess(admission, initialization)
        for marker in (
            "prefix_caching: args.enable_prefix_caching",
            "ssm_cache_slots: args.ssm_cache_slots",
        ):
            self.assertIn(marker, loader[admission:initialization])

    def test_rust_regressions_call_the_production_policy(self):
        tests = POLICY.read_text().split("#[cfg(test)]", 1)[1]
        for name in (
            "c1_prefix_snapshots_admitted_with_ple_conv_history_snapshots",
            "c1_without_paired_prefix_snapshots_keeps_full_prefill_admitted",
            "concurrent_prefix_and_swap_lifecycles_admitted_across_1m_and_multispark",
        ):
            body = tests.split(f"fn {name}()", 1)[1].split("#[test]", 1)[0]
            self.assertIn("validate_qwen4_concurrency(c)", body)
        self.assertIn("[(false, 0), (true, 0), (false, 16)]", tests)

    def test_logical_ceiling_is_not_labeled_runtime_qualified(self):
        source = POLICY.read_text()
        self.assertIn("const MAX_ADMITTED_CONCURRENCY: usize = 128;", source)
        self.assertIn("admitted C{} ceiling", source)
        self.assertNotIn("MAX_QUALIFIED_CONCURRENCY", source)


if __name__ == "__main__":
    unittest.main()
