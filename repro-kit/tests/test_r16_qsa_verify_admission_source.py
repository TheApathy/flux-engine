# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded source/mutation checks. No Rust, model or device execution."""

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "crates/spark-model/src"
VERIFY = MODEL / "model/trait_impl/verify_d.rs"
QSA = MODEL / "layers/qwen4_qsa.rs"
TREE = MODEL / "model/trait_impl/mod.rs"


def compact(source):
    return re.sub(r"\s+", "", re.sub(r"//[^\n]*", "", source))


def body(source, marker):
    source, marker = compact(source), compact(marker)
    start = source.index("{", source.index(marker))
    depth = 1
    for end in range(start + 1, len(source)):
        depth += (source[end] == "{") - (source[end] == "}")
        if depth == 0:
            return source[start + 1:end]
    raise AssertionError("unclosed source body")


SCOPE = "ifqwen4_qsa_verify_frame_required(self.config.is_qwen4_exp(),seq.qwen4_qsa_required,k,)"
ADMISSION = (
    "qwen4_qsa_verify_frame_admission("
    "crate::layers::qwen4_qsa::verify_journal_rows(k),"
    "tree_tokens!=0||host_parents_present||device_parents_present,"
    ").map_err(anyhow::Error::msg)?;"
)


def validate_helper(source):
    required = body(source, "fn qwen4_qsa_verify_frame_required(")
    if required != "target_is_qwen4&&qsa_required&&rows>1":
        raise AssertionError("admission scope must preserve nonQwen/nonQSA/K0/K1")
    helper = body(source, "fn qwen4_qsa_verify_frame_admission(")
    expected = compact('''
        if !journal_available {
            return Err("Qwen4 QSA verify requires undo journaling for its exact physical width");
        }
        if tree_frame_active {
            return Err("Qwen4 QSA verify transaction currently requires a flat candidate chain");
        }
        Ok(())
    ''')
    if helper != expected:
        raise AssertionError("frame admission is not the pure fail-closed policy")


def validate_entry(source):
    entry = body(source, "fn decode_verify_graphed_kgamma_dispatch(")
    scoped = body(entry, SCOPE)
    expected = compact('''
        let tree_tokens = *self.ddtree_num_tree_tokens.lock();
        let host_parents_present = !self.ddtree_parent_ids_host.lock().is_empty();
        let device_parents_present = self.ddtree_parent_ids_dev.lock().is_some();
    ''') + ADMISSION
    if scoped != expected or entry.count(ADMISSION) != 1:
        raise AssertionError("admission must bind actual physical width and all active tree metadata")
    guard = SCOPE + "{" + scoped + "}"
    if entry.count(guard) != 1:
        raise AssertionError("target/QSA/multirow scope is not exact")
    early = entry.index(guard)
    empty = entry.index("ifk==0{returnOk(Vec::new());}")
    if not empty < early:
        raise AssertionError("empty verify no-op must remain before admission")
    for effect in (
        "seq.rotary_positions.tail_scalar(seq.seq_len)?;",
        "exact_bool_env(QWEN4_K16_BATCHED_VERIFY_ENV)?;",
        "crate::model::k16_route_receipt::mark_batched_entry(",
        "qwen4_legacy_k2_admission(",
        "self.ssm_h_to_f16_dispatch(seq)?;",
        "crate::model::k1_stage_diag::begin_batch(",
        "self.pre_verify_copy_async(seq)?;",
        "crate::full_profile::begin_step();",
        "self.kv_cache.lock()",
        "self.embed(",
        "ensure_blocks_through_decode(",
        "self.copy_h2d_group_fast(",
    ):
        if effect not in entry or early >= entry.index(effect):
            raise AssertionError("admission must precede " + effect)
    if "self.decode_verify_graphed_dispatch(" in entry:
        raise AssertionError("Qwen4 fixed-K2 recursive back-edge must stay absent")
    if "Qwen4QSAK5transactioncurrentlyrequiresaflatcandidatechain" in entry:
        raise AssertionError("late width-specific tree guard remains")
    route_guard = (
        "!(self.config.is_qwen4_exp()&&seq.qwen4_qsa_required&&k==5"
        "&&qwen4_layer_route!=Some(Qwen4VerifyLayerRoute::QualifiedBatched))"
    )
    if route_guard not in entry:
        raise AssertionError("existing qualified K5 route restriction was removed")


class QsaVerifyAdmissionSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = VERIFY.read_text(encoding="utf-8")
        cls.qsa = QSA.read_text(encoding="utf-8")
        cls.tree = TREE.read_text(encoding="utf-8")

    def test_pure_fail_closed_admission_policy(self):
        validate_helper(self.source)

    def test_exact_scope_width_tree_inputs_and_pre_effect_ordering(self):
        validate_entry(self.source)

    def test_canonical_journal_predicate_and_rollback_consumer_agree(self):
        predicate = body(self.qsa, "fn verify_journal_rows(")
        self.assertIn("rows==VERIFY_K5_ROWS||((2..=VERIFY_MAX_ROWS).contains(&rows)&&all_k)", predicate)
        self.assertIn('std::env::var("ATLAS_QWEN4_QSA_JOURNAL_ALL_K").as_deref()!=Ok("0")', predicate)
        constants = compact((MODEL / "layers/mod.rs").read_text(encoding="utf-8"))
        self.assertIn("constQWEN4_HYBRID_MAX_ROWS:usize=9;", constants)
        commit = body((MODEL / "model/trait_impl/async_chkpt.rs").read_text(encoding="utf-8"),
                      "fn commit_qwen4_qsa_k5_only_dispatch(")
        self.assertIn("if!crate::layers::qwen4_qsa::verify_journal_rows(k){returnOk(());}", commit)
        self.assertIn("layer.commit_qwen4_qsa_rows(num_accepted,k,self.gpu.as_ref(),stream)?;", commit)

    def test_tree_metadata_set_clear_and_local_flat_injection_match_admission(self):
        setter = body(self.tree, "fn set_ddtree_parent_ids(")
        clearer = body(self.tree, "fn clear_ddtree_parent_ids(")
        for expected in ("*self.ddtree_parent_ids_dev.lock()=Some(self.ddtree_parent_ids_persistent);",
                         "*self.ddtree_num_tree_tokens.lock()=kernel_parents.len();",
                         "*self.ddtree_parent_ids_host.lock()=kernel_parents;"):
            self.assertIn(expected, setter)
        for expected in ("*self.ddtree_parent_ids_dev.lock()=None;", "*self.ddtree_num_tree_tokens.lock()=0;",
                         "self.ddtree_parent_ids_host.lock().clear();"):
            self.assertIn(expected, clearer)
        entry = body(self.source, "fn decode_verify_graphed_kgamma_dispatch(")
        self.assertIn("letmutddtree_parent_ids_dev=*self.ddtree_parent_ids_dev.lock();", entry)
        self.assertNotIn("*self.ddtree_parent_ids_dev.lock()=Some(", entry)

    def test_no_nonqsa_route_selector_or_existing_k5_restriction_changes(self):
        selector = body(self.source, "fn qwen4_verify_layer_route(")
        self.assertIn("if!identity.target_is_qwen4{returnOk(None);}", selector)
        self.assertIn("16ifk16_batched_requested=>Qwen4VerifyLayerRoute::ExperimentalK16Batched", selector)
        self.assertIn("_=>Qwen4VerifyLayerRoute::RowSerialOracle", selector)
        validate_entry(self.source)

    def test_native_pure_definitions_exist_without_environment_mutation(self):
        tests = body(self.source, "mod qwen4_verify_route_tests")
        for name in (
            "qsa_transaction_accepts_only_journaled_flat_frames",
            "qsa_gamma15_and_wider_frames_have_no_undo_journal_admission",
            "qsa_k5_still_requires_flat_frames_when_always_journaled",
            "qsa_shortened_and_wider_journal_frames_all_refuse_tree_metadata",
            "qsa_disabled_journal_never_admits_a_rejectable_suffix",
        ):
            test = body(tests, "fn " + name + "(")
            self.assertIn("qwen4_qsa_verify_frame_admission(", test)
            self.assertNotIn("set_var(", test)
            self.assertNotIn("remove_var(", test)
        self.assertIn("[10,11,15,16,17,32,128,usize::MAX]", tests)
        scope = body(tests, "fn qsa_transaction_scope_leaves_nonqwen_nonqsa_and_single_rows_unchanged(")
        for expected in ("qwen4_qsa_verify_frame_required(false,false,physical_k)",
                         "qwen4_qsa_verify_frame_required(false,true,physical_k)",
                         "qwen4_qsa_verify_frame_required(true,false,physical_k)",
                         "qwen4_qsa_verify_frame_required(true,true,physical_k)", "physical_k>1"):
            self.assertIn(expected, scope)

    def test_nonqwen_nonqsa_and_k0_k1_scope_expansion_is_rejected(self):
        source = compact(self.source)
        exact = "target_is_qwen4&&qsa_required&&rows>1"
        for changed in ("qsa_required&&rows>1", "target_is_qwen4&&rows>1",
                        "target_is_qwen4&&qsa_required", "target_is_qwen4&&qsa_required&&rows==5",
                        "target_is_qwen4&&qsa_required&&rows>9"):
            with self.subTest(changed=changed), self.assertRaises(AssertionError):
                validate_helper(source.replace(exact, changed, 1))

    def test_helper_fail_open_mutations_are_rejected(self):
        source = compact(self.source)
        for before, after in (
            ("if!journal_available{", "iffalse{"),
            ("iftree_frame_active{", "iffalse{"),
            ("if!journal_available{", "ifjournal_available{"),
            ("iftree_frame_active{", "if!tree_frame_active{"),
        ):
            self.assertIn(before, source)
            with self.subTest(mutation=before), self.assertRaises(AssertionError):
                validate_helper(source.replace(before, after, 1))

    def test_wrong_scope_width_predicate_and_swallowed_error_are_rejected(self):
        source = compact(self.source)
        for before, after in (
            (SCOPE, SCOPE.replace("self.config.is_qwen4_exp()", "true")),
            (SCOPE, SCOPE.replace("seq.qwen4_qsa_required", "true")),
            (SCOPE, SCOPE.replace(",k,", ",5,")),
            (SCOPE, SCOPE.replace(",k,", ",16,")),
            (ADMISSION, ADMISSION.replace("verify_journal_rows(k)", "verify_journal_rows(5)")),
            (ADMISSION, ADMISSION.replace("verify_journal_rows(k)", "verify_journal_rows(9)")),
            (ADMISSION, ADMISSION.replace("crate::layers::qwen4_qsa::verify_journal_rows(k)", "true")),
            (ADMISSION, ADMISSION.replace(".map_err(anyhow::Error::msg)?;", ".ok();")),
        ):
            self.assertIn(before, source)
            with self.subTest(mutation=before), self.assertRaises((AssertionError, ValueError)):
                validate_entry(source.replace(before, after, 1))

    def test_missing_tree_channels_and_and_instead_of_or_are_rejected(self):
        source = compact(self.source)
        for field in ("tree_tokens!=0||", "host_parents_present||", "||device_parents_present"):
            self.assertIn(field, ADMISSION)
            with self.subTest(field=field), self.assertRaises(AssertionError):
                validate_entry(source.replace(ADMISSION, ADMISSION.replace(field, ""), 1))
        with self.assertRaises(AssertionError):
            validate_entry(source.replace(ADMISSION, ADMISSION.replace("||", "&&"), 1))

    def test_removed_or_post_effect_admission_is_rejected(self):
        source = compact(self.source)
        entry = body(source, "fn decode_verify_graphed_kgamma_dispatch(")
        scoped = SCOPE + "{" + body(entry, SCOPE) + "}"
        for after in (
            "",
            "self.ssm_h_to_f16_dispatch(seq)?;",
            "self.pre_verify_copy_async(seq)?;",
            "self.kv_cache.lock()",
            "self.embed(",
        ):
            changed = source.replace(scoped, "", 1)
            if after:
                self.assertIn(after, entry)
                changed = changed.replace(after, after + scoped, 1)
            with self.subTest(after=after), self.assertRaises((AssertionError, ValueError)):
                validate_entry(changed)


if __name__ == "__main__":
    unittest.main()
