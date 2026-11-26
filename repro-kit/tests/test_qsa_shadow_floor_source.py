# SPDX-License-Identifier: AGPL-3.0-only
"""Source-only QSA shadow admission checks. No Rust or GPU code is executed.

The Rust mock/policy tests are defined beside the allocator, but are deliberately
not run by this CPU suite. Mutation probes check that these source contracts fail
closed when admission, telemetry, arithmetic, or override wiring is weakened.
"""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "crates/spark-model/src/layers/qwen4_qsa/prefill_fp8.rs"


def section(text, start, end):
    return text.split(start, 1)[1].split(end, 1)[0]


def policy_contract(text):
    policy = section(text, "fn shadow_floor_admission(", "fn ensure_shadow_allocation_above_floor(")
    required = [
        "floor >= startup_ledger::QUALIFICATION_FLOOR_BYTES || override_selected",
        ".checked_add(temporary_bytes)",
        ".and_then(|bytes| bytes.checked_add(floor))",
        "if override_selected",
        "return Ok(ShadowFloorAdmission::Overridden)",
        "let available = mem_available.ok_or_else",
        "available >= required",
        "Ok(ShadowFloorAdmission::Enforced)",
        "NONQUALIFYING",
    ]
    for needle in required:
        if needle not in policy:
            raise AssertionError(f"missing QSA floor policy contract: {needle}")
    order = [
        policy.index(".checked_add(temporary_bytes)"),
        policy.index("if override_selected"),
        policy.index("let available = mem_available.ok_or_else"),
        policy.index("available >= required"),
        policy.index("Ok(ShadowFloorAdmission::Enforced)"),
    ]
    if order != sorted(order):
        raise AssertionError("override, telemetry, and checked-cost decision order changed")


def allocation_contract(text):
    allocator = section(text, "fn shadow_with_mode_admission(", "fn validate_target_shadow_geometry(")
    order = [
        allocator.index("return Ok((s.k8, s.v8))"),
        allocator.index("admit(layer_bytes)?"),
        allocator.index("gpu.alloc(layer_bytes)"),
        allocator.index("gpu.memset(k8, 0, layer_bytes)"),
        allocator.index("map.insert("),
    ]
    if order != sorted(order):
        raise AssertionError("shadow admission must follow a cache hit and precede allocation")


class QsaShadowFloorSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SOURCE.read_text(encoding="utf-8")

    def test_checked_floor_policy_requires_real_telemetry_and_explicit_override(self):
        policy_contract(self.text)

    def test_host_reader_has_no_backend_or_platform_telemetry_bypass(self):
        guard = section(
            self.text,
            "fn ensure_shadow_allocation_above_floor(",
            "/// The layer's K8/(optional V8) shadow pools",
        )
        self.assertIn("startup_ledger::memavailable_floor_bytes()?", guard)
        self.assertIn("startup_ledger::mem_available_bytes()", guard)
        self.assertIn("startup_ledger::floor_override_selected()", guard)
        self.assertIn("let temporary_bytes = 0;", guard)
        self.assertIn("admission == ShadowFloorAdmission::Overridden", guard)
        self.assertIn("OVERRIDDEN, NONQUALIFYING", guard)
        for forbidden in ["free_memory", "cfg!", "unwrap_or", "saturating_"]:
            self.assertNotIn(forbidden, guard)

    def test_public_lazy_and_eager_entry_points_use_the_real_guard(self):
        lazy = section(self.text, "pub fn shadow(", "fn shadow_with_mode_admission(")
        eager = section(self.text, "fn reserve_shadows_for_cache(", "fn reserve_shadows_with_mode_admission(")
        self.assertIn("&mut ensure_shadow_allocation_above_floor", lazy)
        self.assertIn("&mut ensure_shadow_allocation_above_floor", eager)

    def test_per_layer_admission_precedes_allocation_and_page_materialization(self):
        allocation_contract(self.text)
        allocator = section(self.text, "fn shadow_with_mode_admission(", "fn validate_target_shadow_geometry(")
        self.assertIn(".checked_mul(block_bytes)", allocator)
        self.assertIn(".checked_mul(planes)", allocator)
        self.assertIn("let mut map = shadows()", allocator)
        self.assertIn("s.num_blocks == kv_cache.num_blocks() && s.block_bytes == block_bytes", allocator)

    def test_aggregate_admission_counts_only_missing_physical_layers_before_loop(self):
        eager = section(self.text, "fn reserve_shadows_with_mode_admission(", "/// Eagerly allocate")
        self.assertIn("map.get(&key)", eager)
        self.assertIn("shadow.num_blocks == kv_cache.num_blocks() && shadow.block_bytes == block_bytes", eager)
        self.assertIn("missing.checked_add(1)", eager)
        compact = " ".join(eager.split())
        self.assertIn(
            "let pending_bytes = checked_shadow_reserve_bytes_for_mode( mode, kv_cache.block_size(), kv_cache.num_blocks(), missing_layers, )?;",
            compact,
        )
        self.assertIn("if pending_bytes > 0 {\n        admit(pending_bytes)?;", eager)
        self.assertLess(eager.index("admit(pending_bytes)?"), eager.index("shadow_with_mode_admission(gpu, layer, kv_cache, mode, admit)?"))
        self.assertIn("for layer in 0..kv_cache.num_layers()", eager)

    def test_target_geometry_and_selected_mode_semantics_are_preserved(self):
        target = section(self.text, "pub fn reserve_target_shadows(", "/// Record that the prep")
        self.assertIn("let Some(active_mode) = mode()? else {\n        return Ok(0);\n    };", target)
        self.assertLess(target.index("validate_target_shadow_geometry("), target.index("reserve_shadows_for_cache("))
        geometry = section(self.text, "fn validate_target_shadow_geometry(", "fn reserve_shadows_for_cache(")
        self.assertIn("expected_layers > 0 && kv_cache.num_layers() == expected_layers", geometry)
        self.assertIn("kv_cache.dtype_for_layer(layer) == KvCacheDtype::Bf16", geometry)
        self.assertIn("kv_cache.config().dims_for_layer(layer) == (KV_HEADS, HEAD_DIM)", geometry)
        policy = section(self.text, "enum ShadowFloorAdmission", "fn validate_target_shadow_geometry(")
        for forbidden in ["set_var", "gpu.free("]:
            self.assertNotIn(forbidden, policy)

    def test_existing_target_factory_wiring_enters_guarded_reserve(self):
        factory = (ROOT / "crates/spark-model/src/factory/build.rs").read_text(encoding="utf-8")
        bridge = (ROOT / "crates/spark-model/src/factory/qwen4_prefill_reserve.rs").read_text(encoding="utf-8")
        self.assertLess(
            factory.index("|| PagedKvCache::new(kv_config, num_kv_blocks, gpu.as_ref())"),
            factory.index("maybe_reserve_qwen4_target_shadows(&config, gpu.as_ref(), &kv_cache)"),
        )
        self.assertIn("prefill_fp8::reserve_target_shadows(", bridge)

    def test_rust_definitions_cover_refusal_partial_progress_retry_and_overrides(self):
        for name in [
            "shadow_floor_checks_exact_boundary_and_temporary_cost",
            "shadow_floor_missing_telemetry_refuses_and_override_is_always_nonqualifying",
            "shadow_floor_overflow_refuses_even_when_overridden",
            "aggregate_shadow_refusal_allocates_and_materializes_nothing",
            "shadow_rechecks_each_layer_and_retry_charges_only_missing_layers",
            "lazy_shadow_refusal_precedes_device_allocation",
        ]:
            with self.subTest(name=name):
                self.assertIn("fn " + name + "()", self.text)
        self.assertIn("vec![24_576, 12_288, 12_288]", self.text)
        self.assertIn("vec![12_288, 12_288]", self.text)

    def test_checked_cost_contract_rejects_saturating_and_missing_temporary_mutations(self):
        for before, after in [
            (".checked_add(temporary_bytes)", ".saturating_add(temporary_bytes)"),
            (".and_then(|bytes| bytes.checked_add(floor))", ".map(|bytes| bytes.saturating_add(floor))"),
            ("floor >= startup_ledger::QUALIFICATION_FLOOR_BYTES || override_selected", "true"),
            ("let available = mem_available.ok_or_else", "let available = Some(usize::MAX).ok_or_else"),
            ("return Ok(ShadowFloorAdmission::Overridden)", "return Ok(ShadowFloorAdmission::Enforced)"),
        ]:
            with self.subTest(mutation=before):
                self.assertIn(before, self.text)
                with self.assertRaises(AssertionError):
                    policy_contract(self.text.replace(before, after, 1))

    def test_layer_contract_rejects_removed_and_postallocation_admission(self):
        with self.assertRaises(ValueError):
            allocation_contract(self.text.replace("admit(layer_bytes)?;", "", 1))
        mutated = self.text.replace("admit(layer_bytes)?;", "", 1)
        mutated = mutated.replace("let v8 = if planes == 2 {", "admit(layer_bytes)?;\n    let v8 = if planes == 2 {", 1)
        with self.assertRaises(AssertionError):
            allocation_contract(mutated)

    def test_guards_document_sampling_and_unmeasured_overhead_limits(self):
        self.assertIn("not an atomic reservation against other users", self.text)
        self.assertIn("Driver/registry overhead is not a measured bound", self.text)


if __name__ == "__main__":
    unittest.main()
