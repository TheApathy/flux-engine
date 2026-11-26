# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded source/mutation checks only, not Rust or device execution."""

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCHED = ROOT / "crates/spark-server/src/scheduler"


def compact(source):
    return re.sub(r"\s+", "", re.sub(r"//[^\n]*", "", source))


def block(source, start):
    start = source.index("{", start)
    depth = 1
    for end in range(start + 1, len(source)):
        depth += (source[end] == "{") - (source[end] == "}")
        if depth == 0:
            return source[start + 1:end]
    raise AssertionError("unclosed source block")


def body(source, marker):
    source, marker = compact(source), compact(marker)
    return block(source, source.index(marker))


def validate_reconcile(source):
    entry = body(source, "fn reconcile_terminal_verify_state(")
    expected = "k_verify,verifier_accepted,0,consumed_rows,accepted_tree_path,"
    if expected not in entry or "output_tokens" in entry:
        raise AssertionError("terminal state must bind explicit classified rows")
    for guard in (
        "verify_tokens.len()==k_verify&&!verify_tokens.is_empty()",
        "a.seq.tokens.truncate(plan.target_seq_len)",
        "model.commit_verify_state_async(&muta.seq,plan.committed_inputs,k_verify)?;",
    ):
        if guard not in entry:
            raise AssertionError("existing reconciliation guard/effect changed")


def validate_emission(source, batched=False):
    marker = "fn finalize_flat_dflash_verify_commit(" if batched else "fn step_verify_dflash("
    entry = body(source, marker)
    if "output_before_verify" in entry:
        raise AssertionError("visible output cannot establish terminal ownership")
    for token in ("tok", "bonus"):
        exact = f"ifemit_token_for_verify(a,{token},None){{consumed_rows+=1;}}ifa.finished{{"
        if exact not in entry:
            raise AssertionError("each admitted row must count before the terminal check")
    if entry.count("letmutconsumed_rows=0usize;") != 1:
        raise AssertionError("ordinary emission needs one per-frame counter")
    calls = re.findall(r"reconcile_terminal_verify_state\((.*?)\)", entry)
    expected = ["consumed_rows", "consumed_rows"]
    if not batched:
        expected.insert(0, "t.consumed_rows")
    if [call.split(",")[5] for call in calls] != expected:
        raise AssertionError("every terminal call must carry its own classified prefix")


def validate_outcomes(source, expected_count):
    found = 0
    for match in re.finditer(r"\bSpecAcceptOutcome\s*\{", source):
        prefix = source[max(0, match.start() - 16):match.start()].rstrip()
        if prefix.endswith("struct") or prefix.endswith("->"):
            continue
        fields = compact(block(source, match.start()))
        if not re.search(r"consumed_rows(?:,|:0,)", fields):
            raise AssertionError("every outcome must carry prior rows or an explicit zero")
        found += 1
    if found != expected_count:
        raise AssertionError("unexpected outcome constructor inventory")


def validate_walk(source, thinking):
    name = "dflash_thinking_accept_routed" if thinking else "dflash_content_accept_routed"
    entry = body(source, "fn " + name + "(")
    if entry.count("letmutconsumed_rows=0usize;") != 1:
        raise AssertionError("policy walk needs a classified-row accumulator")
    call = "commit_thinking_token(" if thinking else "emit_token_for_verify(a,target,lp)"
    guard = "if!" + call
    cancelled = body(entry, guard)
    if cancelled != "returnSpecAcceptOutcome{num_accepted,bonus:None,consumed_rows,};":
        raise AssertionError("unclassified rows cannot become accepted drafts or bonus")
    counted = entry.index("consumed_rows+=1;")
    if not entry.index(guard) < counted < entry.index("ifboundary||!draft_match" if thinking else "letphase_boundary="):
        raise AssertionError("count only admitted rows, before acceptance/terminal branching")
    if "emit_token(a,target," in entry or "output_tokens.len()-" in entry:
        raise AssertionError("policy count cannot use void emission or visible deltas")


def validate_thinking_commit(source):
    entry = body(source, "fn commit_thinking_token(")
    guard = 'ifretire_cancelled_speculation(a,"spec_emit"){returnfalse;}'
    if guard not in entry or entry.index(guard) > entry.index("ifa.think_end_token==Some(tok)"):
        raise AssertionError("direct EOS requires cancellation admission before effects")
    close = body(entry, "if a.think_end_token == Some(tok)")
    emitted = "if!emit_token_for_verify(a,tok,logprobs){returnfalse;}"
    for reset in ("a.consecutive_confident=0;", "a.in_code_fence=false;", "clear_stale_require_tool_call("):
        if close.index(emitted) >= close.index(reset):
            raise AssertionError("thinking-close effects must follow actual classification")
    eos = body(entry, "if a.eos_tokens.contains(&tok)")
    if "a.thinking_tokens+=1;" not in eos or not eos.endswith("returntrue;"):
        raise AssertionError("hidden thinking EOS is still a classified row")
    if entry.count(emitted) != 2 or not entry.endswith("check_request_timeout(a);true"):
        raise AssertionError("regular and closing tokens must preserve the emitter boolean")


class GenericConsumedRowsSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generic = (SCHED / "verify_dflash_step.rs").read_text()
        cls.batched = (SCHED / "verify_dflash_batched_step.rs").read_text()
        cls.thinking = (SCHED / "think_spec_accept.rs").read_text()
        cls.content = (SCHED / "spec_policy_accept.rs").read_text()

    def test_generic_and_batched_terminal_count_wiring(self):
        validate_reconcile(self.generic)
        validate_emission(self.generic)
        validate_emission(self.batched, True)

    def test_policy_count_wiring_and_every_constructor(self):
        validate_walk(self.thinking, True)
        validate_walk(self.content, False)
        validate_outcomes(self.thinking, 9)
        validate_outcomes(self.content, 30)
        validate_outcomes(self.generic, 1)

    def test_direct_thinking_eos_and_close_are_admitted_before_effects(self):
        validate_thinking_commit(self.thinking)

    def test_output_delta_and_wrong_policy_count_mutations_reject(self):
        for before, after in (
            ("        consumed_rows,\n        accepted_tree_path,", "        a.output_tokens.len(),\n        accepted_tree_path,"),
            ("                t.consumed_rows,", "                t.num_accepted,"),
            ("                    consumed_rows,", "                    a.output_tokens.len(),"),
        ):
            self.assertIn(before, self.generic)
            with self.subTest(mutation=before), self.assertRaises(AssertionError):
                changed = self.generic.replace(before, after, 1)
                validate_reconcile(changed)
                validate_emission(changed)

    def test_void_unconditional_or_missing_emission_count_mutations_reject(self):
        for source, batched in ((self.generic, False), (self.batched, True)):
            for before, after in (
                ("if emit_token_for_verify(a, tok, None)", "if true"),
                ("consumed_rows += 1;", "consumed_rows += 0;"),
                ("if emit_token_for_verify(a, bonus, None)", "if emit_token(a, bonus, None)"),
            ):
                self.assertIn(before, source)
                with self.subTest(batched=batched, mutation=before), self.assertRaises(AssertionError):
                    validate_emission(source.replace(before, after, 1), batched)

    def test_cancelled_policy_row_cannot_become_bonus_or_acceptance(self):
        for source, thinking in ((self.thinking, True), (self.content, False)):
            entry_name = "commit_thinking_token" if thinking else "emit_token_for_verify"
            for before, after in (
                ("if !" + entry_name + "(", "if " + entry_name + "("),
                ("consumed_rows += 1;", "consumed_rows += 0;"),
            ):
                self.assertIn(before, source)
                with self.subTest(thinking=thinking), self.assertRaises((AssertionError, ValueError)):
                    validate_walk(source.replace(before, after, 1), thinking)

    def test_missing_or_invented_outcome_count_mutations_reject(self):
        for source, count in ((self.thinking, 9), (self.content, 30), (self.generic, 1)):
            constructors = []
            for match in re.finditer(r"\bSpecAcceptOutcome\s*\{", source):
                prefix = source[max(0, match.start() - 16):match.start()].rstrip()
                if not (prefix.endswith("struct") or prefix.endswith("->")):
                    constructors.append(block(source, match.start()))
            for before, after in (("consumed_rows,", ""), ("consumed_rows: 0,", "consumed_rows: 1,")):
                fields = next((fields for fields in constructors if before in fields), None)
                if fields is None:
                    continue
                with self.subTest(count=count, mutation=before), self.assertRaises(AssertionError):
                    changed = source.replace(fields, fields.replace(before, after, 1), 1)
                    validate_outcomes(changed, count)

    def test_thinking_eos_cancel_and_hidden_return_mutations_reject(self):
        for before, after in (
            ('if retire_cancelled_speculation(a, "spec_emit")', "if false"),
            ("return true;\n    }\n    if !emit_token_for_verify", "return false;\n    }\n    if !emit_token_for_verify"),
            ("if !emit_token_for_verify(a, tok, logprobs)", "if false"),
        ):
            self.assertIn(before, self.thinking)
            with self.subTest(mutation=before), self.assertRaises((AssertionError, ValueError)):
                validate_thinking_commit(self.thinking.replace(before, after, 1))

    def test_thinking_close_cannot_mutate_state_before_emitter_admission(self):
        source = compact(self.thinking)
        marker = "ifa.think_end_token==Some(tok){"
        check = "if!emit_token_for_verify(a,tok,logprobs){returnfalse;}"
        resets = "a.consecutive_confident=0;a.in_code_fence=false;"
        self.assertIn(check + resets, source)
        changed = source.replace(check + resets, resets + check, 1)
        self.assertIn(marker + resets, changed)
        with self.assertRaises(AssertionError):
            validate_thinking_commit(changed)

    def test_native_regression_definitions_are_registered_not_executed(self):
        generic = (SCHED / "generic_consumed_rows_tests.rs").read_text()
        policy = (SCHED / "think_spec_accept_tests.rs").read_text()
        self.assertIn('#[path = "generic_consumed_rows_tests.rs"]', self.generic)
        for name in (
            "generic_hidden_row_then_length_terminal_keeps_both_classified_inputs",
            "generic_cancellation_after_hidden_row_preserves_only_admitted_prefix",
            "generic_preclassification_cancellation_uses_zero_input_abort",
            "semantic_tree_slot_uses_classified_rows_not_visible_output",
        ):
            self.assertIn("fn " + name + "()", generic)
        for name in (
            "content_hidden_terminal_rows_are_counted_without_visible_output",
            "thinking_hidden_eos_reports_classification_but_cancelled_eos_does_not",
            "thinking_silent_close_is_one_classified_bonus",
            "policy_cancellation_between_sampling_and_emission_has_no_phantom_row",
            "policy_download_failure_preserves_prior_classified_count",
        ):
            self.assertIn("fn " + name + "()", policy)
        self.assertIn("fn batched_hidden_row_then_terminal_keeps_classified_prefix()", self.batched)

    def test_sampled_boolean_still_means_deferred_snapshot_not_consumption(self):
        source = (SCHED / "sampled_dflash.rs").read_text()
        sampled = body(source, "fn emit_sampled_token(")
        self.assertIn("letmutdefer_think_end_snapshot=false;", sampled)
        self.assertIn("&mut|_|defer_think_end_snapshot=true,);defer_think_end_snapshot", sampled)
        self.assertNotIn("consumed_rows", sampled)
        bootstrap = body(self.thinking, "fn bootstrap_thinking_token(")
        self.assertIn("commit_thinking_token(", bootstrap)
        self.assertTrue(bootstrap.endswith("Some(tok)"))


if __name__ == "__main__":
    unittest.main()
