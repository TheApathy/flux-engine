# SPDX-License-Identifier: AGPL-3.0-only
"""Source wiring and mutation checks only, not Rust execution or QSA parity."""

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "crates/spark-model/src"
TRAIT = MODEL / "layer/transformer_layer.rs"
ATTENTION = MODEL / "layers/qwen3_attention/trait_impl.rs"
DECODER = MODEL / "layers/qwen3_attention/trait_impl/decode_inner.rs"
FORWARD = MODEL / "layers/qwen3_attention/decode/attention_forward.rs"
PREFILL = MODEL / "layers/qwen3_attention/trait_impl/prefill_inner.rs"
VERIFY = MODEL / "model/trait_impl/verify_d.rs"
QSA = MODEL / "layers/qwen4_qsa.rs"


def compact(source):
    return re.sub(r"\s+", "", re.sub(r"//[^\n]*", "", source))


def body(source, marker):
    source = compact(source)
    marker = compact(marker)
    start = source.index("{", source.index(marker))
    depth = 1
    for end in range(start + 1, len(source)):
        depth += (source[end] == "{") - (source[end] == "}")
        if depth == 0:
            return source[start + 1:end]
    raise AssertionError("unclosed source body: " + marker)


GUARD = (
    "letqsa_journal=metadata.qwen4_qsa_required"
    "&&crate::layers::qwen4_qsa::verify_journal_rows(k);"
)
PREPARE = "layer.prepare_qwen4_qsa_verify(kv_cache,self.gpu.as_ref())?;"
SNAPSHOT = (
    "layer.snapshot_qwen4_qsa_verify_row("
    "kv_cache,token_metadata.slot,row,self.gpu.as_ref(),stream,)?;"
)
DECODE = "layer.decode_qwen4_verified_row("


def oracle(source):
    return body(compact(source), "ifverify_route==Qwen4VerifyLayerRoute::RowSerialOracle")


def validate_oracle(source):
    source = oracle(source)
    for required in (
        GUARD,
        "ifqsa_journal{" + PREPARE + "}",
        "ifqsa_journal{" + SNAPSHOT + "}",
        "slot:metadata.slot.offset(row*8)",
        "attn_metadata:Some(token_metadata)",
        DECODE,
    ):
        if required not in source:
            raise AssertionError("missing oracle contract: " + required)
    order = [
        source.index(GUARD),
        source.index(PREPARE),
        source.index("forrowin0..k{"),
        source.index("lettoken_metadata="),
        source.index("lettoken_ctx="),
        source.index(SNAPSHOT),
        source.index(DECODE),
    ]
    if order != sorted(order):
        raise AssertionError("prepare/snapshot must precede row mutation")
    for hook in (PREPARE, SNAPSHOT):
        if source.count(hook) != 1:
            raise AssertionError("journal hook must appear once: " + hook)


def validate_transactional_decode(attention, decoder):
    verified = compact(body(attention, "fn decode_qwen4_verified_row("))
    plain = compact(body(attention, "fn decode("))
    kv_fallback = compact(body(decoder, "fn decode_kv_only_inner("))
    arguments = (
        "hidden,residual,state,kv_cache,seq_len,{transactional},"
        "block_table,disk_block_ids,disk_last_offloaded_per_layer,ctx,stream,"
    )
    for source, transactional in ((verified, "true"), (plain, "false"), (kv_fallback, "false")):
        required = "self.decode_inner(" + arguments.format(transactional=transactional) + ")"
        if not source.endswith(required):
            raise AssertionError("wrong common decoder transaction flag: " + transactional)
    common = compact(body(decoder, "fn decode_inner("))
    for input_name in ("mixed", "normed"):
        call = "self.attention_forward(" + input_name + ",seq_len,transactional_qsa,"
        if call not in common:
            raise AssertionError("transactional QSA flag not passed to attention: " + input_name)


class QsaOracleJournalSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trait = TRAIT.read_text(encoding="utf-8")
        cls.attention = ATTENTION.read_text(encoding="utf-8")
        cls.decoder = DECODER.read_text(encoding="utf-8")
        cls.verify = VERIFY.read_text(encoding="utf-8")
        cls.qsa = QSA.read_text(encoding="utf-8")

    def test_hooks_are_object_safe_default_noops(self):
        for name in ("prepare_qwen4_qsa_verify", "snapshot_qwen4_qsa_verify_row"):
            method = self.trait.split("fn " + name + "(", 1)[1].split("{", 1)[0]
            self.assertIn("&self", method)
            self.assertNotIn("Self", method)
            self.assertNotIn("where", method)
            self.assertIn("-> Result<()>", method)
            self.assertEqual(compact(body(self.trait, "fn " + name + "(")), "Ok(())")

    def test_prepare_delegates_and_propagates_missing_indexer_error(self):
        hook = compact(body(self.attention, "fn prepare_qwen4_qsa_verify("))
        self.assertIn("self.qwen4_qsa.as_ref().ok_or_else(", hook)
        self.assertIn("requiresanindexer", hook)
        self.assertIn("))?;", hook)
        self.assertTrue(hook.endswith("qsa.prepare_verify_k5_transaction(kv_cache,gpu)"))

    def test_snapshot_preserves_slot_row_backend_stream_and_result(self):
        hook = compact(body(self.attention, "fn snapshot_qwen4_qsa_verify_row("))
        self.assertIn("self.qwen4_qsa.as_ref().ok_or_else(", hook)
        self.assertIn("requiresanindexer", hook)
        self.assertIn("))?;", hook)
        self.assertTrue(hook.endswith("qsa.snapshot_verify_k5_row(kv_cache,slot,row,gpu,stream)"))

    def test_oracle_prepare_and_undo_snapshot_precede_decode(self):
        validate_oracle(self.verify)

    def test_verified_row_uses_transactional_decode_and_plain_defaults_stay_false(self):
        validate_transactional_decode(self.attention, self.decoder)
        default = compact(body(self.trait, "fn decode_qwen4_verified_row("))
        self.assertEqual(
            default,
            "self.decode(hidden,residual,state,kv_cache,seq_len,block_table,"
            "disk_block_ids,disk_last_offloaded_per_layer,ctx,stream,)",
        )

    def test_transactional_flag_disables_existing_qsa_side_stream(self):
        forward = compact(FORWARD.read_text(encoding="utf-8"))
        self.assertIn(
            "letqsa_overlap=ifdevice_row.is_none()&&meta.qwen4_qsa_required&&!transactional_qsa",
            forward,
        )

    def test_all_common_decoder_callers_have_explicit_transaction_flags(self):
        callers = {}
        for path in (MODEL / "layers/qwen3_attention").rglob("*.rs"):
            source = compact(path.read_text(encoding="utf-8"))
            count = source.count("self.decode_inner(")
            if count:
                callers[str(path.relative_to(MODEL))] = count
        self.assertEqual(callers, {
            "layers/qwen3_attention/trait_impl.rs": 2,
            "layers/qwen3_attention/trait_impl/decode_inner.rs": 2,
            "layers/qwen3_attention/trait_impl/prefill_inner.rs": 1,
            "layers/qwen3_attention/trait_impl/prefill_suppressed.rs": 1,
        })
        decode_inner = compact(
            (MODEL / "layers/qwen3_attention/trait_impl/decode_inner.rs").read_text(encoding="utf-8")
        )
        self.assertIn(
            "returnself.decode_inner(hidden,residual,&mutempty_state,kv_cache,"
            "seq_lens[0],false,block_tables[0],disk_block_ids[0],"
            "disk_last_offloaded_per_layer[0],&single_ctx,stream,);",
            decode_inner,
        )
        prefill = compact(PREFILL.read_text(encoding="utf-8"))
        self.assertIn(
            "self.decode_inner(hidden.offset(t*row_bytes),residual.offset(t*row_bytes),"
            "_state,kv_cache,seq_len_start+t,false,block_table,disk_block_ids,"
            "disk_last_offloaded_per_layer,&token_ctx,stream,)?;",
            prefill,
        )
        prefill_suppressed = compact(
            (MODEL / "layers/qwen3_attention/trait_impl/prefill_suppressed.rs").read_text(
                encoding="utf-8"
            )
        )
        self.assertIn(
            "self.decode_inner(hidden.offset(t*row_bytes),residual.offset(t*row_bytes),"
            "state,kv_cache,seq_len_start+t,false,block_table,disk_block_ids,"
            "disk_last_offloaded_per_layer,&token_ctx,stream,)?;",
            prefill_suppressed,
        )

    def test_untransactional_verified_row_and_dropped_flag_are_rejected(self):
        attention = compact(self.attention)
        decoder = compact(self.decoder)
        for before, after in (
            ("seq_len,true,block_table", "seq_len,false,block_table"),
            ("seq_len,false,block_table", "seq_len,true,block_table"),
        ):
            with self.subTest(mutation=before):
                self.assertIn(before, attention)
                with self.assertRaises(AssertionError):
                    validate_transactional_decode(attention.replace(before, after, 1), self.decoder)
        for input_name in ("mixed", "normed"):
            before = "self.attention_forward(" + input_name + ",seq_len,transactional_qsa,"
            self.assertIn(before, decoder)
            with self.assertRaises(AssertionError):
                validate_transactional_decode(self.attention, decoder.replace(before, before.replace("transactional_qsa", "false"), 1))

    def test_commit_predicate_and_diagnostic_opt_out_are_reused(self):
        predicate = compact(body(self.qsa, "fn verify_journal_rows("))
        self.assertIn("rows==VERIFY_K5_ROWS||((2..=VERIFY_MAX_ROWS).contains(&rows)&&all_k)", predicate)
        self.assertIn('std::env::var("ATLAS_QWEN4_QSA_JOURNAL_ALL_K").as_deref()!=Ok("0")', predicate)
        self.assertIn(GUARD, oracle(self.verify))
        self.assertNotIn("(2..=5)", oracle(self.verify))

    def test_metadata_and_recurrent_snapshots_stay_in_original_order(self):
        source = oracle(self.verify)
        for required in (
            "positions:metadata.positions.offset(row*4)",
            "positions_h:metadata.positions_h.offset(row*4)",
            "positions_w:metadata.positions_w.offset(row*4)",
            "seq_len:metadata.seq_len.offset(row*4)",
            "block_table:metadata.block_table.offset(row*mb*4)",
            "hidden.offset(row*row_stride)",
            "residual.offset(row*row_stride)",
            "seq.seq_len+row",
            "h_inter.offset(row*self.ssm_pool.h_bytes)",
            "conv_inter.offset(row*self.ssm_pool.conv_bytes)",
        ):
            self.assertIn(required, source)
        self.assertLess(source.index(DECODE), source.index("ssm.h_state"))
        self.assertLess(source.index("ssm.h_state"), source.index("ssm.conv_state"))

    def test_route_selector_does_not_enable_unqualified_batching(self):
        route = compact(body(self.verify, "fn qwen4_verify_layer_route("))
        self.assertIn("4ifk5_hybrid_enabled&&k4_hybrid_enabled=>", route)
        self.assertIn("4|6..=9ifk5_hybrid_enabled&&all_k_hybrid_enabled=>", route)
        self.assertIn("_=>Qwen4VerifyLayerRoute::RowSerialOracle", route)

    def test_removed_or_unguarded_hook_is_rejected(self):
        source = compact(self.verify)
        for before, after in (
            (PREPARE, ""),
            (SNAPSHOT, ""),
            (DECODE, "layer.decode("),
            ("ifqsa_journal{" + PREPARE, "iftrue{" + PREPARE),
            ("ifqsa_journal{" + SNAPSHOT, "iftrue{" + SNAPSHOT),
        ):
            with self.subTest(mutation=before):
                self.assertIn(before, source)
                with self.assertRaises(AssertionError):
                    validate_oracle(source.replace(before, after, 1))

    def test_wrong_slot_row_stream_or_swallowed_error_is_rejected(self):
        source = compact(self.verify)
        for before, after in (
            (SNAPSHOT, SNAPSHOT.replace("token_metadata.slot", "metadata.slot")),
            (SNAPSHOT, SNAPSHOT.replace(",row,", ",0,")),
            (SNAPSHOT, SNAPSHOT.replace(",stream,", ",0,")),
            (SNAPSHOT, SNAPSHOT.replace(")?;", ").ok();")),
            (PREPARE, PREPARE.replace(")?;", ").ok();")),
            (GUARD, GUARD.replace("metadata.qwen4_qsa_required", "true")),
            (GUARD, GUARD.replace("verify_journal_rows(k)", "verify_journal_rows(5)")),
        ):
            with self.subTest(mutation=before):
                self.assertIn(before, source)
                with self.assertRaises(AssertionError):
                    validate_oracle(source.replace(before, after, 1))

    def test_post_decode_snapshot_is_rejected(self):
        source = compact(self.verify)
        moved = "ifqsa_journal{" + SNAPSHOT + "}"
        self.assertEqual(source.count(moved), 1)
        source = source.replace(moved, "", 1)
        decode_end = "&token_ctx,stream,)?;"
        self.assertIn(decode_end, source)
        source = source.replace(decode_end, decode_end + moved, 1)
        with self.assertRaises(AssertionError):
            validate_oracle(source)


if __name__ == "__main__":
    unittest.main()
