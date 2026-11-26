# SPDX-License-Identifier: AGPL-3.0-only
"""Source/mutation wiring only. Does not execute Rust or qualify a model."""

from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCHED = ROOT / "crates/spark-server/src/scheduler"


def verify_contract(source, k):
    capture = source.index(f"let mut frame = ShortVerifyFrame::capture(a, {k});")
    verify = source.index(f"model.decode_verify_graphed_k{k}(")
    assert capture < verify
    assert source[:capture].count("fail_speculative_step(") == 3
    assert "frame.fail(" not in source[:capture]
    assert source[capture:].count("fail_speculative_step(") == 3 * k
    assert source.count("frame.fail(") == 2 + k
    assert "emit_token(" not in source
    assert source.count("frame.emit(") == k * (k + 1) // 2
    terminals = source.split("if a.finished {")[1:]
    assert len(terminals) == k
    for terminal in terminals:
        before_return = terminal.split("return;", 1)[0]
        assert f"frame.reconcile_terminal(model, a, num_accepted, &tokens_k{k});" in before_return
    outcomes = source.split("    } else ")
    assert len(outcomes) >= k
    tail = source[source.index(f"    if num_accepted == {k - 1} {{"):]
    for branch in tail.split("    } else "):
        emit = branch.index("frame.emit(")
        terminal = branch.index("frame.reconcile_terminal(")
        commit = branch.index("model.commit_verify_state_async(")
        assert emit < terminal < commit
        assert "frame.fail(" in branch[commit:].split("return;", 1)[0]
        for marker in ["save_hidden_for_active_proposer(", "model.trim_proposer_state(", "Err(e) => {"]:
            after = branch[commit:].split(marker, 1)[1].split("return;", 1)[0]
            assert "fail_speculative_step(" in after
            assert "frame.fail(" not in after


def helper_contract(source):
    failure = source.split("pub(super) fn fail(", 1)[1].split("pub(super) fn reconcile_terminal", 1)[0]
    assert failure.index("a.seq.seq_len = self.pre_verify_len;") < failure.index("rollback_failed_verify_frame(")
    assert failure.index("a.seq.tokens.truncate(self.pre_verify_tokens);") < failure.index("rollback_failed_verify_frame(")
    assert failure.index("rollback_failed_verify_frame(") < failure.index("fail_speculative_step(")
    assert 'format!("{error:#}; verify abort rollback failed: {rollback_error:#}")' in failure
    assert "free_sequence(" not in source
    assert source.count("output_tokens.len()") == 2
    assert "output_tokens.len()" not in failure
    assert "if emit_token_for_verify(a, token, logprobs) {" in source
    assert "self.consumed_rows += 1;" in source
    terminal = source.split("pub(super) fn reconcile_terminal", 1)[1]
    assert "output_tokens.len()" not in terminal
    assert "terminal_verify_commit_plan(" in terminal
    assert "self.consumed_rows," in terminal
    assert "a.seq.tokens.truncate(plan.target_seq_len);" in terminal
    assert "a.seq.seq_len = plan.target_seq_len;" in terminal
    assert "if plan.committed_inputs == 0 {" in terminal
    assert "abort_attempted = true;" in terminal
    assert "if abort_attempted {" in terminal
    assert 'fail_speculative_step(a, "short terminal verify abort", error);' in terminal
    assert "model.commit_verify_state_async(&mut a.seq, plan.committed_inputs, self.k)?;" in terminal


class ShortFrameSource(unittest.TestCase):
    def test_actual_fixed_width_wiring(self):
        for k in [3, 4]:
            verify_contract((SCHED / f"verify_k{k}_step.rs").read_text(), k)

    def test_error_and_terminal_mutants_are_rejected(self):
        for k in [3, 4]:
            source = (SCHED / f"verify_k{k}_step.rs").read_text()
            for old, new in [
                ("frame.fail(", "fail_speculative_step("),
                ("frame.emit(", "emit_token("),
                (f"frame.reconcile_terminal(model, a, num_accepted, &tokens_k{k});", ""),
                (f"let mut frame = ShortVerifyFrame::capture(a, {k});", ""),
            ]:
                with self.subTest(k=k, old=old), self.assertRaises((AssertionError, ValueError)):
                    verify_contract(source.replace(old, new, 1), k)

    def test_actual_shared_helper(self):
        helper_contract((SCHED / "short_verify_frame.rs").read_text())

    def test_postcommit_failure_mutants_are_rejected(self):
        for k in [3, 4]:
            source = (SCHED / f"verify_k{k}_step.rs").read_text()
            for stage in ["save proposer hidden", "trim proposer state", "propose next frame"]:
                old = f'fail_speculative_step(a, "K{k} {stage}", e);'
                with self.subTest(k=k, stage=stage), self.assertRaises((AssertionError, ValueError)):
                    verify_contract(source.replace(old, "", 1), k)

    def test_host_restore_and_abort_mutants_are_rejected(self):
        source = (SCHED / "short_verify_frame.rs").read_text()
        for old in [
            "a.seq.seq_len = self.pre_verify_len;",
            "a.seq.tokens.truncate(self.pre_verify_tokens);",
            "rollback_failed_verify_frame(",
            'format!("{error:#}; verify abort rollback failed: {rollback_error:#}")',
        ]:
            with self.subTest(old=old), self.assertRaises((AssertionError, ValueError)):
                helper_contract(source.replace(old, "REMOVED", 1))

    def test_visible_output_counter_and_double_abort_mutants_are_rejected(self):
        source = (SCHED / "short_verify_frame.rs").read_text()
        for old, new in [
            ("self.consumed_rows,", "a.output_tokens.len(),"),
            ("abort_attempted = true;", ""),
            ("if abort_attempted {", "if false {"),
            ("self.consumed_rows += 1;", "self.consumed_rows += 0;"),
        ]:
            with self.subTest(old=old), self.assertRaises((AssertionError, ValueError)):
                helper_contract(source.replace(old, new, 1))

    def test_emission_reports_preclassification_cancellation_only_as_false(self):
        source = (SCHED / "emit_step.rs").read_text()
        body = source.split("fn emit_token_with_policies(", 1)[1].split("fn discard_uncommitted_logprobs(", 1)[0]
        assert ") -> bool {" in body
        assert body.count("return false;") == 1
        assert "if retire_cancelled_speculation(a, \"spec_emit\") {\n        return false;" in body
        assert "return;" not in body
        assert body.count("return true;") >= 10
        assert "apply_speculative_fuzzy_stop(a);\n    }\n    true\n}" in body
        wrapper = source.split("pub(super) fn emit_token_for_verify(", 1)[1].split("\n}\n", 1)[0]
        assert "emit_token_with_policies(" in wrapper
        assert ") -> bool {" in wrapper

    def test_recording_native_cases_are_defined_and_registered(self):
        source = (SCHED / "short_verify_frame_tests.rs").read_text()
        assert source.count("#[test]") == 9
        for name in [
            "shortened_failures_abort_once_after_restoring_host_extent",
            "preentry_errors_do_not_abort_an_unstarted_transaction",
            "every_length_terminal_commits_only_the_classified_input_prefix",
            "eos_at_each_accepted_or_bonus_position_consumes_one_terminal_commit",
            "hidden_rows_before_terminal_content_are_retained_in_the_target_prefix",
            "cancellation_before_first_row_aborts_once_even_when_abort_fails",
            "cancellation_after_a_hidden_row_keeps_that_row_without_inventing_output",
            "terminal_commit_error_notifies_once_and_defers_retirement",
            "postcommit_proposer_errors_terminate_without_aborting_consumed_state",
        ]:
            assert f"fn {name}(" in source
        assert 'mod short_verify_frame;' in (SCHED / "mod.rs").read_text()
        assert '#[path = "short_verify_frame_tests.rs"]' in (SCHED / "short_verify_frame.rs").read_text()


if __name__ == "__main__":
    unittest.main()
