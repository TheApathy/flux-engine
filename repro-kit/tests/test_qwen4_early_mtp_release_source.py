# SPDX-License-Identifier: AGPL-3.0-only
"""Offline source contracts for Plain-only early unused-MTP release.

These inspect source, not Rust execution. The mock-backed Rust unit definitions
are checked for coverage but remain uncompiled and unexecuted by this suite.
No model payload, GPU, process, or native-build actions are required.
"""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
BUILD = "crates/spark-model/src/factory/build.rs"


def source(relative):
    return (ROOT / relative).read_text(encoding="utf-8")


def function(text, name):
    body = text.split("fn " + name + "(", 1)[1]
    return body.split("\nfn ", 1)[0]


def validate_individual_fallback(loader):
    """Keep the real non-grouped allocation/copy/audit path source-bound."""
    compact = re.sub(r"\s+", "", loader)
    start = compact.split("letgrouped_destination=", 1)[1]
    fallback = start.split("}else{letallocation=matchgpu.alloc(meta.len){", 1)[1]
    fallback = fallback.split("out.insert(", 1)[0]
    for expected in (
        "Ok(p)=>{",
        "bounce.copy(src,p)?;",
        "(p,false)",
        "Err(_)=>{",
        "letp=gpu.alloc_managed(meta.len)?;",
        "gpu.copy_h2d_managed(src,p)?;",
        "(p,true)",
        "allocation_audit.record(meta.len,allocation.1);allocation",
    ):
        if expected not in fallback:
            raise ValueError("individual fallback lost allocation, copy, error propagation or audit")


class Qwen4EarlyMtpReleaseSourceTests(unittest.TestCase):
    def test_gate_reuses_existing_plain_selector_and_returns_before_measurement(self):
        gate = function(source(BUILD), "maybe_release_unused_qwen4_mtp_early")
        self.assertIn(
            "!matches!(\n"
            "        select_qwen4_release_mode(\n"
            "            is_qwen4,\n"
            "            use_native_mtp,\n"
            "            has_dflash,\n"
            "            false,\n"
            "            release_plain,\n"
            "            false,\n"
            "        ),\n"
            "        Qwen4ReleaseMode::Plain | Qwen4ReleaseMode::Dflash\n"
            "    )",
            gate,
        )
        self.assertLess(gate.index("return Ok((0, 0))"), gate.index("startup_ledger::measure("))
        self.assertIn('"release-unused-mtp-early"', gate)
        self.assertIn("negative = reclaimed", gate)
        self.assertIn("|| release_unused_qwen4_mtp(store, gpu)", gate)
        selector = source("crates/spark-model/src/factory/qwen4_release_mode.rs")
        self.assertIn("if !is_qwen4", selector)
        self.assertIn("if !has_dflash && release_plain", selector)
        self.assertIn("else if has_dflash && release_plain", selector)
        self.assertIn("Qwen4ReleaseMode::Dflash", selector)
        self.assertLess(selector.index("if use_speculative"), selector.index("Qwen4ReleaseMode::Plain"))

    def test_only_mtp_aliases_are_synced_then_released_through_owner(self):
        release = function(source(BUILD), "release_unused_qwen4_mtp")
        self.assertIn('.filter(|name| name.starts_with("mtp."))', release)
        empty = release.index("if names.is_empty()")
        sync = release.index("gpu.synchronize(gpu.default_stream())?;")
        first_lookup = release.index("let weight = store.get(name)?;")
        owner_release = release.index("store.release_consumed_alias(name, weight.ptr, weight.dtype, gpu)?")
        self.assertLess(empty, sync)
        self.assertLess(sync, first_lookup)
        self.assertLess(first_lookup, owner_release)
        self.assertNotIn("gpu.free(", release)
        self.assertNotIn("lm_head", release)
        self.assertIn(".checked_add(released)", release)

    def test_early_release_is_after_target_load_before_transpose_and_mtp_probes(self):
        build = source(BUILD)
        target = build.index("|| loader.load_layers(store, &config, gpu.as_ref(), &attn_layer_dtypes)")
        call = build.index("let (early_mtp_tensors, early_mtp_bytes)")
        transpose = build.index("|| maybe_run_minimax_m2_moe_transpose(&config, gpu.as_ref(), &mut layers)")
        mtp_probe = build.index("let mtp_weights = loader.load_mtp_weights_multi(")
        self.assertLess(target, call)
        self.assertLess(call, transpose)
        self.assertLess(transpose, mtp_probe)
        actual_call = build[call:transpose]
        self.assertIn("config.is_qwen4_exp(),", actual_call)
        self.assertIn("use_native_mtp,", actual_call)
        self.assertIn("dflash_args.is_some(),", actual_call)
        self.assertIn('std::env::var("ATLAS_QWEN4_RELEASE_UNUSED").ok().as_deref() == Some("1")', actual_call)
        self.assertIn('"QWEN4_RELEASE_UNUSED_MTP_EARLY before target MoE transpose"', actual_call)
        self.assertIn("requested_bytes = early_mtp_bytes", actual_call)

    def test_late_plain_path_is_idempotent_and_preserves_head_after_quantization(self):
        build = source(BUILD)
        plain_start = build.index("Qwen4ReleaseMode::Plain => {")
        dflash_start = build.index("Qwen4ReleaseMode::Dflash => {", plain_start)
        native_start = build.index("Qwen4ReleaseMode::NativeMtp => {", dflash_start)
        plain = build[plain_start:dflash_start]
        dflash = build[dflash_start:native_start]
        self.assertIn("gpu.synchronize(stream)?;", plain)
        self.assertIn("release_unused_qwen4_mtp(store, gpu.as_ref())?", plain)
        self.assertIn("release_unused_qwen4_bf16_lm_head(", plain)
        self.assertIn("lm_head_nvfp4.is_some(),", plain)
        self.assertIn("gpu.synchronize(stream)?;", dflash)
        self.assertIn("release_unused_qwen4_mtp(store, gpu.as_ref())?", dflash)
        self.assertIn("can_release_qwen4_dflash_bf16_lm_head(", dflash)
        self.assertIn("release_unused_qwen4_bf16_lm_head(", dflash)
        self.assertLess(build.index("let lm_head_nvfp4 ="), plain_start)
        # Definition + Plain + Dflash + NativeMtp + memlean-dense (Qwen4ReleaseMode::None)
        # + #[cfg(test)] unit-test call (pinned BF16 head kept) = 6.
        self.assertEqual(build.count("release_unused_qwen4_bf16_lm_head("), 6)
        self.assertNotIn("BF16) is never read", plain)

    def test_native_release_keeps_live_proposer_gate_and_existing_packed_keys(self):
        build = source(BUILD)
        selection = build.split("let release_mode = select_qwen4_release_mode(", 1)[1]
        selection = selection.split("let release_mem_before", 1)[0]
        self.assertIn("qwen4_mtp_proposer.is_some(),", selection)
        self.assertIn('std::env::var("ATLAS_QWEN4_RELEASE_SPEC_UNUSED")', selection)
        native = build.split("Qwen4ReleaseMode::NativeMtp => {", 1)[1].split("Qwen4ReleaseMode::None =>", 1)[0]
        self.assertIn('"mtp.layers.0.mlp.experts.gate_up_proj"', native)
        self.assertIn('"mtp.layers.0.mlp.experts.down_proj"', native)
        self.assertIn("store.release_consumed_alias(name, ptr, dtype, gpu.as_ref())?", native)

    def test_tombstones_hide_later_probes_and_failed_free_rolls_back(self):
        store = source("crates/spark-runtime/src/weights.rs")
        release = store.split("pub fn release_consumed_alias(", 1)[1].split("/// Merge", 1)[0]
        self.assertIn("self.reserve_consumed_alias(name, expected_ptr, expected_dtype)?", release)
        self.assertIn("if let Err(error) = gpu.free(ptr)", release)
        self.assertIn("self.rollback_consumed(name);", release)
        self.assertLess(release.index("self.rollback_consumed(name);"), release.index("return Err(error)"))
        contains = store.split("pub fn contains(", 1)[1].split("/// Number", 1)[0]
        names = store.split("pub fn names(", 1)[1].split("/// Total", 1)[0]
        self.assertIn("!self.released.lock().contains(name)", contains)
        self.assertIn("!self.released.lock().contains(name.as_str())", names)
        probe = source("crates/spark-model/src/weight_loader/qwen35.rs")
        probe = probe.split("fn load_mtp_weights(", 1)[1]
        self.assertIn('if !store.contains("mtp.fc.weight")', probe)
        self.assertIn("return Ok(None);", probe)

    def test_mtp_aliases_are_excluded_from_shared_scalar_slab_ownership(self):
        loader = source("crates/spark-runtime/src/fast_weights/mod.rs")
        candidate = function(loader, "is_expert_scalar_slab_candidate")
        self.assertIn('meta.name.starts_with("model.language_model.layers.")', candidate)
        self.assertIn('!meta.name.starts_with("mtp.")', candidate)
        self.assertIn("expert_scalar_layout.offsets[*idx].is_none()", loader)
        validate_individual_fallback(loader)

    def test_individual_mtp_fallback_rejects_missing_copy_managed_or_audit(self):
        loader = source("crates/spark-runtime/src/fast_weights/mod.rs")
        for original, mutant in (
            ("bounce.copy(src, p)?;", "bounce.copy(src, p).ok();"),
            ("gpu.alloc_managed(meta.len)?", "gpu.alloc_managed(meta.len).unwrap_or_default()"),
            ("gpu.copy_h2d_managed(src, p)?;", "gpu.copy_h2d_managed(src, p).ok();"),
            ("allocation_audit.record(meta.len, allocation.1);", ""),
        ):
            self.assertTrue(original in loader, "mutation must affect the actual loader")
            with self.subTest(original=original), self.assertRaises(ValueError):
                validate_individual_fallback(loader.replace(original, mutant, 1))

    def test_mock_unit_definitions_cover_gates_sync_idempotence_and_partial_failure(self):
        tests = source(BUILD).split("mod early_mtp_release_tests {", 1)[1]
        for name in [
            "early_mtp_release_gate_is_exhaustively_plain_only",
            "early_mtp_release_syncs_tombstones_and_late_repeat_is_noop",
            "early_mtp_release_sync_failure_does_not_free_or_tombstone",
            "early_mtp_release_partial_free_failure_rolls_back_only_failed_alias",
        ]:
            self.assertIn("fn " + name + "()", tests)
        self.assertIn("gpu.fail_free_at.store(2, Ordering::SeqCst)", tests)
        self.assertIn("let remaining_bytes = store.get(&remaining[0]).unwrap().byte_size();", tests)
        self.assertIn("(1, remaining_bytes)", tests)


if __name__ == "__main__":
    unittest.main()
