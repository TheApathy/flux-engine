#!/usr/bin/env python3
"""Offline source contract for serial native-MTP K5 at Qwen4 C2-C8 (S0).

`ATLAS_QWEN4_K5_CMULTI=1` admits K5 at C2-C8 by running each request's exact
C1 verify -> commit transaction back to back. These checks pin what a compiler
cannot see: the unset gate keeps the target-only refusal, cross-request
verifiers stay refused, the QSA journal owner and tree checks precede every
frame effect, and admission reserves `num_drafts` verify lookahead rows for all
active speculative proposers (`0` for plain decode). They do not compile the
crates or establish GPU parity.
"""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "crates/spark-model/src"
SERVER = ROOT / "crates/spark-server/src"

TARGET_ONLY = (
    '"Qwen4 concurrency is target-only unless native K5 is selected with '
    'ATLAS_QWEN4_K5_CMULTI=1 or ATLAS_QWEN4_BATCHED_K5=1; disable DFlash, '
    'self-speculative and ngram speculation"'
)


def source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def compact(text: str) -> str:
    return "".join(ch for ch in text if not ch.isspace())


def body_after(text: str, needle: str) -> str:
    start = text.index(needle)
    opening = text.index("{", start)
    depth = 0
    for offset, ch in enumerate(text[opening:]):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[opening + 1 : opening + offset]
    raise AssertionError(f"unbalanced body after {needle}")


def ordered(text: str, markers) -> None:
    cursor = 0
    for marker in markers:
        found = text.find(marker, cursor)
        if found < 0:
            raise AssertionError(f"missing or out of order: {marker}")
        cursor = found + len(marker)


class SerialK5SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.gate = source(MODEL / "model/qwen4_k5_cmulti.rs")
        cls.guard = source(SERVER / "main_modules/qwen4_concurrency.rs")
        cls.serve_load = source(SERVER / "main_modules/serve_load.rs")
        cls.verify = source(MODEL / "model/trait_impl/verify_d.rs")
        cls.commit = source(MODEL / "model/trait_impl/async_chkpt.rs")
        cls.types = source(MODEL / "model/types.rs")
        cls.ctor = source(MODEL / "model/impl_a1.rs")
        cls.admission = source(SERVER / "scheduler/admission.rs")
        cls.scheduler = source(SERVER / "scheduler/mod.rs")
        cls.mtp_step = source(SERVER / "scheduler/mtp_step.rs")
        cls.verify_step = source(SERVER / "scheduler/verify_dflash_step.rs")
        cls.layers = source(MODEL / "layers/mod.rs")
        cls.dflash_batched = source(SERVER / "scheduler/verify_dflash_batched_step.rs")
        cls.kgamma = body_after(
            cls.verify, "pub(super) fn decode_verify_graphed_kgamma_dispatch("
        )

    def test_gate_is_exact_default_off_and_bounded(self) -> None:
        parse = body_after(self.gate, "pub fn parse_k5_cmulti(")
        self.assertIn('None | Some(Some("0")) => Ok(false),', parse)
        self.assertIn('Some(Some("1")) => Ok(true),', parse)
        self.assertIn("Some(Some(_)) => Err(", parse)
        self.assertIn("Some(None) => Err(", parse)
        self.assertIn('QWEN4_K5_CMULTI_ENV: &str = "ATLAS_QWEN4_K5_CMULTI";', self.gate)
        self.assertIn("QWEN4_K5_CMULTI_MAX_CONCURRENCY: usize = 8;", self.gate)
        self.assertIn("QWEN4_K5_CMULTI_NUM_DRAFTS: usize = 4;", self.gate)
        enforced = body_after(self.gate, "pub(crate) fn k5_cmulti_enforced()")
        self.assertIn("!matches!(k5_cmulti_selected(), Ok(false))", enforced)

    def test_unset_gate_keeps_the_target_only_refusal(self) -> None:
        validate = body_after(self.guard, "pub(crate) fn validate_qwen4_concurrency(")
        speculative = body_after(validate, "if c.speculative && !c.batched_k5 {")
        ordered(
            compact(speculative),
            [
                "if!c.serial_k5.gate{returnErr(" + compact(TARGET_ONLY) + ".into(),);}",
                "validate_serial_k5(c.max_batch_size,c.serial_k5)?;",
            ],
        )
        self.assertEqual(self.guard.count(TARGET_ONLY), 2)  # production + exact-text test
        # The C1 early return and every concurrent lifecycle guard remain ordered.
        ordered(
            validate,
            [
                "if c.max_batch_size <= 1 {",
                "--kv-cache-dtype bf16 or nvfp4",
                "if c.speculative && !c.batched_k5 {",
                "--high-speed-swap disabled",
                "TP=1, TP=2, or TP=4",
                "EP=1, EP=2, or EP=4",
                "1..=4 distributed ranks",
                "1M YaRN ceiling",
                "checkpoint's native context",
                "--swap-space-gb 0",
            ],
        )

    def test_gate_admits_only_native_mtp_k5_at_c2_c8_without_batched_verify(self) -> None:
        serial = body_after(self.guard, "fn validate_serial_k5(")
        ordered(
            serial,
            [
                "if !k5.native_mtp_only {",
                "if max_batch_size > QWEN4_K5_CMULTI_MAX_CONCURRENCY {",
                "if k5.num_drafts != QWEN4_K5_CMULTI_NUM_DRAFTS {",
                "if k5.cross_request_verify {",
                "Ok(())",
            ],
        )
        wiring = compact(body_after(self.serve_load, "serial_k5: super::qwen4_concurrency::SerialK5Request {"))
        for marker in (
            "gate:spark_model::model::qwen4_k5_cmulti::k5_cmulti_selected().map_err(anyhow::Error::msg)?,",
            "native_mtp_only:args.speculative&&!args.dflash&&!args.self_speculative&&!args.ngram_speculative,",
            "num_drafts:args.num_drafts,",
            "spark_model::layers::mtp_k2_batch_cseq_enabled()",
            "spark_model::layers::mtp_k3_batch_cseq_enabled()",
            'std::env::var("ATLAS_DFLASH_BATCHED_VERIFY").ok().as_deref()==Some("1")',
        ):
            self.assertIn(marker, wiring)
        # Startup reads match the scheduler selectors exactly.
        self.assertIn('std::env::var("ATLAS_MTP_K2_BATCH_CSEQ").ok().as_deref() == Some("1")', self.layers)
        self.assertIn('std::env::var("ATLAS_MTP_K3_BATCH_CSEQ").ok().as_deref() == Some("1")', self.layers)
        self.assertIn(
            'std::env::var("ATLAS_DFLASH_BATCHED_VERIFY").ok().as_deref() == Some("1")',
            self.dflash_batched,
        )

    def test_journal_owner_and_tree_checks_precede_every_frame_effect(self) -> None:
        entry = self.kgamma
        tree = entry.index("check_tree_singletons_unset(")
        admit = entry.index(".admit_frame(owner)")
        scope = entry.index("if qwen4_qsa_verify_frame_required(")
        arm = entry.index(".arm(owner)")
        self.assertLess(entry.index("check_partial_head_admission("), tree)
        self.assertLess(tree, admit)
        self.assertLess(admit, scope)
        self.assertLess(scope, arm)
        for effect in (
            "seq.rotary_positions.tail_scalar(seq.seq_len)?;",
            "self.ssm_h_to_f16_dispatch(seq)?;",
            "self.pre_verify_copy_async(seq)?;",
            "self.kv_cache.lock()",
            "self.embed(",
        ):
            self.assertLess(arm, entry.index(effect), effect)
        self.assertIn("super::super::qwen4_k5_cmulti::k5_cmulti_enforced()", entry)
        # Arming is scoped to journaled QSA frames.
        arm_guard = compact(entry[entry.rindex("if let Some(owner) = serial_cmulti_owner", 0, arm):arm])
        self.assertIn("&&qwen4_qsa_verify_frame_required(self.config.is_qwen4_exp(),seq.qwen4_qsa_required,k,)", arm_guard)

    def test_commit_and_abort_check_then_release_the_journal(self) -> None:
        dispatch = body_after(self.commit, "pub(in crate::model) fn commit_qwen4_qsa_k5_only_dispatch(")
        ordered(
            dispatch,
            [
                "layer.commit_qwen4_qsa_rows(num_accepted, k, self.gpu.as_ref(), stream)?;",
                "if self.qwen4_k5_cmulti_enforced() {",
                "self.qwen4_qsa_journal_owner.lock().release();",
            ],
        )
        commit = body_after(self.commit, "pub(in crate::model) fn commit_verify_state_async_dispatch(")
        ordered(
            commit,
            [
                "self.check_qsa_journal_commit(seq)?;",
                "if num_accepted == 0 {",
                "self.commit_qwen4_qsa_k5_only_dispatch(num_accepted, k)?;",
            ],
        )
        abort = body_after(self.commit, "pub(in crate::model) fn abort_verify_state_dispatch(")
        ordered(
            compact(abort),
            [
                "self.check_qsa_journal_commit(seq).and_then(|()|self.commit_qwen4_qsa_k5_only_dispatch(0,k));",
                "VerifyAbortGuard::enter();",
                "self.pre_verify_copy_async_dispatch(seq)",
            ],
        )
        check = body_after(self.commit, "fn check_qsa_journal_commit(")
        self.assertIn("if !self.qwen4_k5_cmulti_enforced() {", check)
        self.assertIn("slot: seq.slot_idx,", check)
        self.assertIn("session: seq.session_hash,", check)
        self.assertIn(
            "pub(super) qwen4_qsa_journal_owner: Mutex<super::qwen4_k5_cmulti::QsaJournalOwnership>,",
            self.types,
        )
        self.assertIn("qwen4_qsa_journal_owner: Mutex::new(Default::default()),", self.ctor)

    def test_serial_per_request_verify_commit_loop_is_reused_as_is(self) -> None:
        loop = self.mtp_step[self.mtp_step.index("// ── Phase B (default per-seq path)"):]
        ordered(loop, ["for &idx in &verify_idxs {", "let a = &mut active[idx];", "dispatch_verify(model, a, &drafts, num_drafts, think, depth_policy);"])
        # Inside one request's generic K5 verify: commit, then hidden save,
        # then proposer trim, before the loop advances to the next request.
        ordered(
            self.verify_step,
            [
                "model.commit_verify_state_async_with_slot(",
                "model.save_hidden_for_mtp_sequence(row, &a.seq, 0)",
                "model.trim_proposer_state(&mut a.seq, num_accepted, 0)",
            ],
        )
        self.assertNotIn("qwen4_k5_cmulti", self.mtp_step)

    def test_admission_reserves_verify_lookahead_for_active_speculation(self) -> None:
        resolve = body_after(self.admission, "pub(super) fn resolve_verify_lookahead_tokens(")
        self.assertEqual(compact(resolve), "num_drafts")
        legacy = body_after(self.admission, "pub(super) fn seq_commitment_blocks(")
        self.assertEqual(
            compact(legacy),
            "seq_commitment_blocks_with_lookahead(demand,watermark,max_seq_len,block_size,0)",
        )
        extended = body_after(self.admission, "pub(super) fn seq_commitment_blocks_with_lookahead(")
        ordered(extended, ["depth.min(max_seq_len.max(demand.current_tokens))", "blocks_for_tokens(depth.saturating_add(lookahead), block_size)"])
        gate = body_after(self.admission, "pub(super) fn gate_admissions_with_reserved(")
        self.assertIn("gate_admissions_with_blocks(", gate)
        self.assertIn("verify_lookahead,", gate)
        gate_blocks = body_after(self.admission, "fn gate_admissions_with_blocks(")
        self.assertIn("committed_blocks_with_lookahead(", gate_blocks)
        self.assertIn("admit_count_with_lookahead(", gate_blocks)
        self.assertEqual(gate_blocks.count("verify_lookahead"), 2)
        wiring = compact(self.scheduler)
        self.assertIn(
            "letverify_lookahead=admission::resolve_verify_lookahead_tokens("
            "spark_model::model::qwen4_k5_cmulti::k5_cmulti_selected()==Ok(true)"
            "||spark_model::layers::qwen4_verify_lanes::batched_k5_selected().unwrap_or(false),"
            "use_mtp&&model.proposer_arm_kind(0)=="
            "Some(spark_model::speculative::DraftProposerKind::Qwen4NativeMtp),"
            "num_drafts,);",
            wiring,
        )
        self.assertIn("block_size,verify_lookahead,);", wiring)

    def test_cpu_tests_cover_the_documented_matrix(self) -> None:
        for name in (
            "fn serial_k5_gate_admits_native_mtp_k5_at_c2_through_c8()",
            "fn unset_selectors_keep_the_target_only_refusal()",
            "fn serial_k5_gate_refuses_other_proposers_widths_and_batched_verifiers()",
            "fn serial_k5_gate_admits_nvfp4_yarn_prefix_swap_and_tp2_while_rejecting_invalid_bounds()",
        ):
            self.assertIn(name, self.guard)
        for name in (
            "fn gate_is_exact_and_default_off()",
            "fn journal_refuses_cross_owner_prepare_until_released()",
            "fn every_tree_singleton_is_refused()",
        ):
            self.assertIn(name, self.gate)
        hidden = source(MODEL / "model/qwen4_mtp_hidden.rs")
        self.assertIn("fn serial_k5_c8_handoffs_never_cross_slots_or_sessions()", hidden)
        for name in (
            "fn verify_lookahead_is_reserved_for_all_active_speculative_proposers()",
            "fn zero_lookahead_is_byte_identical_to_the_legacy_reservation()",
            "fn k5_lookahead_can_queue_the_last_request_at_c8()",
        ):
            self.assertIn(name, self.admission)
        self.assertTrue(re.search(r"pub mod qwen4_k5_cmulti;", source(MODEL / "model/mod.rs")))


if __name__ == "__main__":
    unittest.main()
