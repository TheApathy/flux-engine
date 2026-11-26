#!/usr/bin/env python3
"""Offline source contract for Qwen4 native multi-sequence decode (C2-C128 / QWEN4_NATIVE_MAX_ROWS=256).

These checks do not compile kernels or establish runtime correctness. They keep
the fail-closed source and documentation boundaries reviewable while GPU gates
remain outstanding.
"""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


def source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


class NativeConcurrencyContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dispatch = source(
            "crates/spark-model/src/model/trait_impl/decode_qwen4_native_batch.rs"
        )
        cls.route = source("crates/spark-model/src/model/trait_impl/decode_a2.rs")
        cls.attention = source(
            "crates/spark-model/src/layers/qwen3_attention/trait_impl/multi_seq/qwen4_native.rs"
        )
        cls.qkv = source(
            "crates/spark-model/src/layers/qwen3_attention/trait_impl/multi_seq/qkv.rs"
        )
        cls.attention_tail = source(
            "crates/spark-model/src/layers/qwen3_attention/trait_impl/multi_seq/attn.rs"
        )
        cls.policy = source(
            "crates/spark-model/src/layers/qwen3_attention/trait_impl/multi_seq/qwen4_native_policy.rs"
        )
        cls.ssm = source(
            "crates/spark-model/src/layers/qwen3_ssm/qwen4_native_multi_seq.rs"
        )
        cls.ssm_dispatch = source(
            "crates/spark-model/src/layers/qwen3_ssm/trait_decode_multi_seq.rs"
        )
        cls.ple = source(
            "crates/spark-model/src/layers/qwen4_ple/runtime/verify_batch.rs"
        )
        cls.logits = source(
            "crates/spark-server/src/scheduler/decode_logits_seq.rs"
        )
        cls.lifecycle = source(
            "crates/spark-server/src/scheduler/lifecycle.rs"
        )
        cls.probe = source("repro-kit/qwen4_native_concurrency_probe.py")
        cls.qualification = source("repro-kit/QWEN4_R9_QUALIFICATION.md")
        cls.native_recipe = source(
            "repro-kit/recipes/flux-qwen4-native-262k-bf16.env"
        )

    def test_only_exact_active_widths_c2_through_c128_are_native(self) -> None:
        limit = "crate::layers::QWEN4_NATIVE_MAX_ROWS"
        self.assertIn(f"rows >= 2 && rows <= {limit}", self.dispatch)
        self.assertIn(f"(2..={limit}).contains(&num_seqs)", self.attention)
        self.assertIn(f"rows >= 2 && rows <= {limit}", self.ssm)
        self.assertIn(f"(2..={limit}).contains(&rows)", self.ple)
        self.assertIn(
            "pub(crate) const QWEN4_NATIVE_MAX_ROWS: usize = 256;",
            source("crates/spark-model/src/layers/mod.rs"),
        )
        self.assertIn("C33-C128", self.qualification)
        self.assertIn("select_native_target_width(selected, rows)", self.dispatch)
        self.assertIn("supports active widths C2-C128 only", self.dispatch)

    def test_qsa_horizon_and_long_context_eligibility_and_opt_in_route(self) -> None:
        eligibility = self.dispatch.split(
            "pub(super) fn qwen4_native_batch_request_eligible", 1
        )[1].split("impl TransformerModel", 1)[0]
        self.assertIn("!seq.qwen4_qsa_required", eligibility)
        self.assertIn("next < QSA_BOUNDARY", eligibility)
        self.assertIn("qsa_native::native_attn_route(qsa_mode, true, seq.seq_len).is_ok()", eligibility)
        native_call = self.route.index(
            "decode_qwen4_native_batch_dispatch(tokens, seqs)"
        )
        serial_call = self.route.index("decode_batch_serial_rows(tokens, seqs)")
        self.assertLess(native_call, serial_call)
        self.assertIn("cannot reach the sparse-QSA boundary", self.policy)

    def test_target_dtype_topology_and_graph_contracts_fail_closed(self) -> None:
        for marker in (
            "target_only || native_mtp_bootstrap || dflash_bootstrap",
            "native Qwen4 decode is target-only; disable speculative decoding",
            "native Qwen4 decode requires TP1..TP4 (+ matching EP) with PP1, or PP>=2 with TP1/EP1",
            "native Qwen4 decode requires resident BF16 KV (or NVFP4) without high-speed swap",
            "validate_kv_block_ownership_prefix_cache(",
            "native Qwen4 decode does not admit prefix-cache ownership",
            "profiled Qwen4 requests must use the generic profile-aware decoder",
        ):
            self.assertIn(marker, self.dispatch)
        self.assertIn("!ctx.graph_capture", self.attention)
        self.assertIn("!ctx.graph_capture", self.ssm)

    def test_state_and_capacity_proofs_precede_first_allocation(self) -> None:
        body = self.dispatch.split(
            "pub(in crate::model) fn decode_qwen4_native_batch_dispatch", 1
        )[1].split("#[cfg(test)]", 1)[0]
        first_allocation = body.index("ensure_blocks_through_decode(")
        for marker in (
            "let receipt_enabled = qwen4_native_batch_receipt_enabled()?;",
            "preflight_embed_qwen4_batch(rows)?",
            "ple.preflight_active_decode_rows(",
            "preflight_qwen4_native_target(rows",
            "validate_kv_rows_before_allocation(",
            "validate_kv_block_ownership(",
            "blocks_needed <= kv_cache.num_free_blocks()",
        ):
            self.assertLess(body.index(marker), first_allocation, marker)
        self.assertIn("aliases another request's recurrent state", self.ssm)
        self.assertIn("alias persistent slot", self.ple)

    def test_documentation_does_not_equate_capacity_with_native_batching(self) -> None:
        self.assertIn(
            "Context capacity and native concurrency are independent claims",
            self.qualification,
        )
        self.assertIn("does not batch their long-context decode", self.qualification)
        self.assertIn("Do not infer native C128", self.qualification)

    def test_exact_attention_launch_reduction_is_fail_closed_and_receipted(self) -> None:
        for selector in (
            "ATLAS_ATTN_QKV_EXACT_STRIDED",
            "ATLAS_ATTN_QK_NORM_BATCHED",
        ):
            self.assertIn(f"{selector} must be exactly 0 or 1", self.qkv)
            self.assertIn(f"ENGAGED {selector}: multi_seq", self.qkv)
            self.assertIn(f"{selector}=1", self.native_recipe)
        self.assertIn(
            "ATLAS_ATTN_QK_NORM_BATCHED=1 requires rms_norm_qk_batch3",
            self.qkv,
        )
        self.assertIn(
            "ATLAS_ATTN_GATE_BATCHED must be exactly 0 or 1",
            self.attention_tail,
        )
        self.assertIn(
            "ENGAGED ATLAS_ATTN_GATE_BATCHED: multi_seq",
            self.attention_tail,
        )
        self.assertIn("ATLAS_ATTN_GATE_BATCHED=1", self.native_recipe)
        self.assertIn(
            "ATLAS_ATTN_KV_WRITE_BATCHED must be exactly 0 or 1",
            self.attention_tail,
        )
        self.assertIn(
            "ENGAGED ATLAS_ATTN_KV_WRITE_BATCHED: multi_seq",
            self.attention_tail,
        )
        self.assertIn("ATLAS_ATTN_KV_WRITE_BATCHED=1", self.native_recipe)
        self.assertIn("requires exact token and full-logit parity", self.native_recipe)

    def test_native_ssm_pointer_table_reuse_is_exact_and_fail_closed(self) -> None:
        self.assertIn(
            "ATLAS_QWEN4_NATIVE_SSM_PTR_CACHE must be exact value 0 or 1",
            self.ssm,
        )
        self.assertIn(
            "ATLAS_QWEN4_NATIVE_SSM_PTR_CACHE must be valid UTF-8 and exactly 0 or 1",
            self.ssm,
        )
        self.assertIn("Err(std::env::VarError::NotUnicode(_))", self.ssm)
        selector = self.ssm.split("fn native_ssm_ptr_cache_selected()", 1)[1].split(
            "pub(super) const fn qwen4_native_multi_seq_width", 1
        )[0]
        self.assertNotIn(".ok()", selector)
        self.assertIn(
            "refresh_native_multi_seq_ptr_table_cached(states, rows",
            self.ssm,
        )
        self.assertIn("QWEN4_NATIVE_SSM_PTR_CACHE_ENGAGED", self.ssm)
        self.assertIn("ATLAS_QWEN4_NATIVE_SSM_PTR_CACHE=1", self.native_recipe)

        cached = self.ssm_dispatch.split(
            "pub(crate) fn refresh_native_multi_seq_ptr_table_cached", 1
        )[1]
        self.assertIn("multi_seq_ptr_uploaded_rows", cached)
        self.assertIn("== num_seqs", cached)
        self.assertIn("slots[row] != ssm_state.h_state.0", cached)
        self.assertIn(
            "slots[num_seqs + row] != ssm_state.conv_state.0", cached
        )
        self.assertIn(
            "self.refresh_multi_seq_ptr_table(states, num_seqs, gpu, stream)?",
            cached,
        )
        tail = self.ssm_dispatch.split(
            "pub(crate) fn refresh_multi_seq_ptr_table", 1
        )[1]
        upload, _ = tail.split(
            "pub(crate) fn refresh_native_multi_seq_ptr_table_cached", 1
        )
        self.assertLess(upload.index("store(0"), upload.index("copy_h2d_pinned_async"))
        self.assertLess(
            upload.index("copy_h2d_pinned_async"),
            upload.index("store(num_seqs"),
        )

    def test_qualification_receipts_bind_route_session_logits_and_state(self) -> None:
        self.assertIn("ATLAS_QWEN4_NATIVE_BATCH_RECEIPT", self.dispatch)
        self.assertIn("must be exact value 0 or 1", self.dispatch)
        self.assertIn("QWEN4_NATIVE_BATCH_FRAME rows={rows} sessions={sessions}", self.dispatch)
        self.assertIn("QWEN4_SERIAL_BATCH_FRAME rows={n} sessions={sessions}", self.route)
        self.assertIn(
            "LOGITS_FNV session={session_hash:016x} idx={output_index}",
            self.logits,
        )
        self.assertIn("sha256={sha256_hex}", self.logits)
        self.assertIn("QWEN4_NATIVE_MTP_REQUEST_RECEIPT", self.lifecycle)
        self.assertIn("state_scope=live-gdn-ple-plus-logical-bf16-kv", self.lifecycle)
        self.assertIn("release_qualified\": False", self.probe)
        self.assertIn("reference receipt", self.probe)


if __name__ == "__main__":
    unittest.main()
