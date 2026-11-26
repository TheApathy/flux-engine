# SPDX-License-Identifier: AGPL-3.0-only
"""Offline source contracts for startup-memory policy.

These checks inspect Rust source and existing recipe controls. They do not
execute Rust, compile native code, query a GPU, or qualify a runtime peak.
"""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


def source(relative):
    return (ROOT / relative).read_text(encoding="utf-8")


RUNTIME = "crates/spark-runtime/src/startup_ledger.rs"
LEDGER = "crates/spark-server/src/main_modules/serve_phases/startup_ledger.rs"
TRANSPOSE = "crates/spark-model/src/factory/m2_setup.rs"
CACHE_POLICY = "crates/spark-model/src/layers/qwen4_fast_proj_cache_floor.rs"
CACHE = "crates/spark-model/src/layers/qwen4_fast_proj.rs"
PREFLIGHT = "crates/spark-server/src/main_modules/serve_phases/preflight.rs"


class StartupMemoryPolicySourceTests(unittest.TestCase):
    def test_reduced_floor_requires_explicit_override_and_checked_arithmetic(self):
        text = source(RUNTIME)
        parser = text.split("pub fn parse_floor_bytes_with_override(", 1)[1]
        parser = parser.split("/// The declared floor", 1)[0]
        self.assertIn(".checked_mul(MIB)", parser)
        self.assertIn("bytes >= QUALIFICATION_FLOOR_BYTES || override_selected", parser)
        self.assertIn("NONQUALIFYING", parser)
        self.assertIn("parse_floor_bytes_with_override(value, false)", text)
        self.assertIn('for raw in ["0", "1", "6655"]', text)

    def test_environment_reader_does_not_bypass_override_policy(self):
        text = source(RUNTIME).split("pub fn memavailable_floor_bytes()", 1)[1]
        text = text.split("/// True when", 1)[0]
        self.assertIn("let override_selected = floor_override_selected();", text)
        self.assertIn("parse_floor_bytes_with_override(Some(&value), override_selected)", text)
        self.assertIn("parse_floor_bytes_with_override(None, override_selected)", text)

    def test_meminfo_rejects_wrong_units_extra_fields_and_duplicate_measurements(self):
        text = source(RUNTIME).split("pub fn parse_mem_available_bytes(", 1)[1]
        text = text.split("/// Real host", 1)[0]
        self.assertIn('fields.next() != Some("kB")', text)
        self.assertIn("fields.next().is_some()", text)
        self.assertIn("matching_lines.next().is_some()", text)
        self.assertIn("kb.checked_mul(1024)", text)

    def test_ledger_validates_floor_and_override_before_pass(self):
        text = source(LEDGER).split("pub(crate) fn decision(", 1)[1]
        text = text.split("/// The table", 1)[0]
        self.assertLess(text.index("FloorDecision::InvalidFloor"), text.index("match self.mem_available_bytes"))
        self.assertLess(text.index("if self.override_selected"), text.index("if available >= self.floor_bytes"))
        self.assertIn("self.floor_bytes.saturating_sub(available)", text)
        ledger = source(LEDGER)
        self.assertIn("OVERRIDDEN by {FLOOR_OVERRIDE_SELECTOR}=1, NONQUALIFYING", ledger)

    def test_unified_guard_precedes_first_transpose_entry(self):
        text = source(TRANSPOSE).split("} else if unified_layout {", 1)[1]
        self.assertLess(
            text.index("ensure_unified_pass_above_floor("),
            text.index("layer.transpose_moe_for_prefill_unified("),
        )
        guard = source(TRANSPOSE).split("fn ensure_unified_pass_above_floor(", 1)[1]
        guard = guard.split("fn unified_pass_floor_decision(", 1)[0]
        self.assertIn("startup_ledger::memavailable_floor_bytes()?", guard)
        self.assertIn('cfg!(target_os = "linux")', guard)
        self.assertIn("None if is_linux && !override_selected => anyhow::bail!", guard)
        self.assertIn("/proc/meminfo has no readable MemAvailable", guard)

    def test_optional_cache_never_uses_backend_fallback_for_missing_linux_meminfo(self):
        text = source(CACHE_POLICY).split("pub fn cache_available_bytes(", 1)[1]
        text = text.split("#[cfg(test)]", 1)[0]
        self.assertIn("None if is_linux => Ok(None)", text)
        self.assertIn("None => backend_fallback().map(Some)", text)
        self.assertLess(text.index("None if is_linux"), text.index("None => backend_fallback"))
        self.assertIn("must not query backend memory", source(CACHE_POLICY))

    def test_optional_cache_returns_before_device_allocation_without_telemetry(self):
        text = source(CACHE).split("match cache_floor::evaluate_cache_admission(", 1)[1]
        text = text.split("let (w8, w_scale, w_amax)", 1)[0]
        self.assertIn('cfg!(target_os = "linux")', text)
        self.assertIn("cache_floor::CacheAdmitDecision::UnmeasurableHost =>", text)
        self.assertIn("skipping optional FP8 weight-cache allocation", text)
        self.assertLess(text.index("return Ok(None)"), text.index(".alloc(bytes)"))

    def test_projection_counts_eager_qsa_shadows_and_declares_remaining_omissions(self):
        text = source(PREFLIGHT)
        self.assertIn("qsa_fp8_shadow_bytes: breakdown.qsa_fp8_shadow_bytes", text)
        self.assertIn(".saturating_add(self.qsa_fp8_shadow_bytes)", text)
        self.assertIn("eager QSA shadows (addressable upper bound)", text)
        self.assertIn("partial estimate, not a peak bound", text)
        self.assertIn("additional prefill scratch, lazy request caches", text)
        self.assertNotIn("Everything still to be allocated before the server listens", text)

    def test_eager_shadow_order_test_matches_the_measured_target_cache_constructor(self):
        constructor = "|| PagedKvCache::new(kv_config, num_kv_blocks, gpu.as_ref())"
        factory = source("crates/spark-model/src/factory/build.rs")
        test = source("crates/spark-model/src/factory/qwen4_prefill_reserve.rs")
        self.assertIn(constructor, factory)
        self.assertIn('.find("' + constructor + '")', test)
        self.assertLess(
            factory.index(constructor),
            factory.index("maybe_reserve_qwen4_target_shadows(&config, gpu.as_ref(), &kv_cache)"),
        )

    def test_default_4g_loader_and_recipe_cache_controls_remain(self):
        cli = source("crates/spark-server/src/cli.rs")
        self.assertIn("#[arg(long, default_value_t = 4096)]\n    pub oom_guard_mb", cli)
        recipe = source("repro-kit/recipes/flux-v9-max-prefill.env")
        # Single-node max-prefill recipe pins a 6 GiB reserve; 2x-spark-cluster.env keeps 7 GiB.
        self.assertIn("ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_WEIGHT_CACHE_RESERVE_GB=6\n", recipe)
        policy = source(CACHE_POLICY)
        self.assertIn(".max(self.floor_bytes.saturating_add(self.margin_bytes))", policy)
        self.assertIn("assert_eq!(recipe_policy().stop_line_bytes(), 8704 * MIB)", policy)

    def test_focused_rust_unit_definitions_cover_boundary_cases(self):
        cases = {
            RUNTIME: [
                "reduced_floors_require_the_explicit_nonqualifying_override",
                "malformed_memavailable_telemetry_is_not_a_measurement",
            ],
            TRANSPOSE: ["unified_pass_refuses_missing_linux_telemetry_before_allocating"],
            CACHE_POLICY: [
                "unavailable_linux_telemetry_skips_cache_without_backend_fallback",
                "non_linux_telemetry_preserves_backend_fallback_and_errors",
            ],
            "crates/spark-server/src/main_modules/serve_phases/startup_ledger_tests.rs": [
                "reduced_floor_never_produces_an_ordinary_pass",
                "explicit_override_is_nonqualifying_even_with_positive_headroom",
            ],
            "crates/spark-server/src/main_modules/serve_phases/preflight_projection_tests.rs": [
                "projection_sums_modeled_terms_including_eager_qsa_shadows",
            ],
        }
        for path, names in cases.items():
            text = source(path)
            for name in names:
                with self.subTest(path=path, test=name):
                    self.assertIn("fn " + name + "()", text)

    def test_retained_layouts_guard_full_pass_and_recheck_each_layer(self):
        text = source(TRANSPOSE)
        for start, end, budget, allocation in [
            ("if hybrid_fits {", "} else if unified_layout {", "full_pass_bytes", "layer.transpose_moe_for_prefill_hybrid("),
            ("} else if free >= full_backend_need {", "} else if free >= gate_up_backend_need {", "full_pass_bytes", "layer.transpose_moe_for_prefill("),
            ("} else if free >= gate_up_backend_need {", "// ", "gate_up_pass_bytes", "layer.transpose_moe_gate_up_for_prefill("),
        ]:
            branch = text.split(start, 1)[1]
            if budget == "gate_up_pass_bytes":
                branch = branch.split("let n_per_expert_packed", 1)[0]
            else:
                branch = branch.split(end, 1)[0]
            with self.subTest(layout=start):
                self.assertLess(branch.index("retained_budget." + budget), branch.index(allocation))
                self.assertGreaterEqual(branch.count("ensure_retained_pass_above_floor("), 2)
                self.assertLess(branch.index("for layer in layers.iter_mut()"), branch.rindex("ensure_retained_pass_above_floor("))
                self.assertLess(branch.rindex("ensure_retained_pass_above_floor("), branch.index(allocation))

    def test_unified_budget_uses_full_layer_not_model_retained_set_or_net_reclaim(self):
        text = source(TRANSPOSE)
        branch = text.split("} else if unified_layout {", 1)[1].split("} else if free >= full_backend_need {", 1)[0]
        self.assertEqual(branch.count("ensure_unified_pass_above_floor(gpu, retained_budget.unified_layer_bytes)?"), 2)
        self.assertIn("unified_layer_bytes: checked_transpose_sum(&[full_layer, host_transient, 512])?", text)
        self.assertNotIn("cost_full / config.num_hidden_layers", branch)
        self.assertNotIn("full_pass_bytes", branch)

    def test_all_layouts_share_linux_override_policy_and_preserve_backend_guards(self):
        text = source(TRANSPOSE)
        guard = text.split("fn ensure_retained_pass_above_floor(", 1)[1].split("fn unified_pass_telemetry_decision(", 1)[0]
        self.assertIn("startup_ledger::memavailable_floor_bytes()?", guard)
        self.assertIn("checked_transpose_sum(&[allocation_bytes, safety])?", guard)
        self.assertIn('cfg!(target_os = "linux")', guard)
        self.assertIn("let hybrid_fits = hybrid_layout && free >= hybrid_backend_need;", text)
        self.assertIn("} else if free >= full_backend_need {", text)
        self.assertIn("} else if free >= gate_up_backend_need {", text)
        self.assertIn("let safety: usize = 2 * 1024 * 1024 * 1024;", text)
        self.assertIn("checked_transpose_product(&[2, cost_full])?", text)

    def test_retained_budget_counts_shared_tables_host_padding_and_checked_scratch(self):
        text = source(TRANSPOSE)
        budget = text.split("fn retained_transpose_budget(", 1)[1].split("/// Pre-flight memory audit", 1)[0]
        self.assertIn("config.num_experts, 8 + 8 + 4", budget)
        self.assertIn("checked_transpose_product(&[2, shared_gate])?", budget)
        self.assertIn("host_transient", budget)
        self.assertIn("checked_transpose_product(&[full_layer, layer_count])?", budget)
        self.assertIn("checked_transpose_product(&[gate_up_layer, layer_count])?", budget)
        scratch = text.split("let n_per_expert_packed =", 1)[1].split("tracing::info!(", 1)[0]
        self.assertIn("checked_transpose_product(&[local_experts, n_per_expert_packed])?", scratch)
        self.assertIn("checked_transpose_product(&[local_experts, n_per_expert_scale])?", scratch)
        self.assertIn("scratch_allocation_bytes == retained_budget.down_scratch_bytes", scratch)
        self.assertLess(scratch.index("ensure_retained_pass_above_floor("), scratch.index("gpu.alloc(scratch_packed_bytes)"))
        self.assertEqual(scratch.count("gpu.alloc(scratch_pointer_table_bytes)?"), 2)
        for name in [
            "retained_budget_counts_padded_shared_weights_tables_and_down_scratch",
            "transpose_budget_rejects_size_padding_and_aggregate_overflow",
            "every_layout_refuses_missing_linux_telemetry_unless_explicitly_overridden",
            "each_retained_branch_and_down_scratch_has_a_preallocation_guard",
        ]:
            self.assertIn("fn " + name + "()", text)


if __name__ == "__main__":
    unittest.main()
