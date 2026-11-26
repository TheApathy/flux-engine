# SPDX-License-Identifier: AGPL-3.0-only
"""K2 transaction wiring/mutation checks; shared helper and Rust are not run."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCHEDULER = ROOT / "crates/spark-server/src/scheduler"


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


def branches(source):
    source = body(source, "fn step_verify_k2(")
    return (body(source, "if accepted { frame.emit("),
            body(source, "else { a.seq.seq_len -= 1;"))


def validate_branch(source, accepted):
    prefix = "accepted" if accepted else "rejected"
    committed = 2 if accepted else 1
    row, token = (1, "v1") if accepted else (0, "v0")
    terminal = f"frame.reconcile_terminal(model,a,{int(accepted)},&tokens_k2);return;"
    if "ifa.finished{" + terminal + "}" not in source:
        raise AssertionError("every terminal exit must reconcile its exact consumed-row prefix")
    commit = f"model.commit_verify_state_async(&muta.seq,{committed},2)"
    save = f"save_hidden_for_active_proposer(model,a,{token},{row})"
    trim = f"model.trim_proposer_state(&muta.seq,{row},0)"
    propose = "proposal_lifecycle::propose_and_install("
    expected = ("frame.emit(", terminal, commit, save, trim, propose)
    positions = [source.index(marker) for marker in expected]
    if positions != sorted(positions):
        raise AssertionError("terminal reconciliation must precede commit, then active hidden/trim/propose")
    for stage in (f"K2 save {prefix} hidden", f"K2 trim {prefix} proposer", f"K2 propose {prefix} prefix"):
        expected = compact(f'fail_speculative_step(a,"{stage}",e);return;')
        if expected not in source:
            raise AssertionError("postcommit failure must notify without aborting consumed journal")
    failed = compact(f'frame.fail(model,a,"K2 commit {prefix} prefix",e);return;')
    if failed not in source:
        raise AssertionError("commit failure must attempt the shared abort path")
    if source.count("frame.fail(") != 1 or source.count("fail_speculative_step(") != 3:
        raise AssertionError("wrong pre/postcommit failure channel census")
    if "model.save_hidden_for_mtp_sequence(" in source or "model.save_hidden_for_dflash(" in source:
        raise AssertionError("inactive proposer hidden state must not be touched")


def validate_entry(source):
    complete = source
    source = body(source, "fn step_verify_k2(")
    capture = "letmutframe=ShortVerifyFrame::capture(a,2);"
    verify = "model.decode_verify_graphed(&tokens_k2,&muta.seq,0)"
    if source.count(capture) != 1 or source.index(capture) >= source.index(verify):
        raise AssertionError("capture exact K2 host extent before verification")
    for stage in ("K2 decode_verify", "K2 EP result"):
        expected = compact(f'frame.fail(model,a,"{stage}",e);return;')
        if expected not in source:
            raise AssertionError("post-entry verifier/EP failure skipped transaction abort")
    for stage in ("K2 sync_secondary", "K2 EP command", "K2 EP token"):
        expected = compact(f'fail_speculative_step(a,"{stage}",e);return;')
        if expected not in source[:source.index(capture)]:
            raise AssertionError("pre-entry failure must notify without inventing a started frame")
    if source.count("frame.emit(") != 3 or "emit_token(" in source:
        raise AssertionError("all classified K2 rows must pass through frame.emit")
    if source.count("frame.reconcile_terminal(") != 2 or source.count("frame.fail(") != 4:
        raise AssertionError("wrong terminal/error frame census")
    left, right = branches(complete)
    validate_branch(left, True)
    validate_branch(right, False)
    if not right.startswith("a.seq.seq_len-=1;a.seq.tokens.pop();frame.emit("):
        raise AssertionError("rejected draft host extent must be trimmed before target emission")


class K2CleanupSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (SCHEDULER / "verify_k2_step.rs").read_text(encoding="utf-8")
        cls.native = (SCHEDULER / "verify_k2_cleanup_tests.rs").read_text(encoding="utf-8")

    def test_entry_capture_abort_channels_and_classified_emit_wiring(self):
        validate_entry(self.source)

    def test_both_terminal_paths_reconcile_before_their_single_ordinary_commit(self):
        accepted, rejected = branches(self.source)
        validate_branch(accepted, True)
        validate_branch(rejected, False)

    def test_no_active_proposer_work_before_successful_ordinary_commit(self):
        for accepted, branch in zip((True, False), branches(self.source)):
            validate_branch(branch, accepted)
            self.assertEqual(branch.count("model.commit_verify_state_async("), 1)
            self.assertEqual(branch.count("save_hidden_for_active_proposer("), 1)
            self.assertEqual(branch.count("proposal_lifecycle::propose_and_install("), 1)

    def test_wrong_capture_width_or_post_entry_capture_is_rejected(self):
        source = compact(self.source)
        capture = "letmutframe=ShortVerifyFrame::capture(a,2);"
        for change in (source.replace(capture, capture.replace("a,2", "a,3"), 1),
                       source.replace(capture, "", 1).replace("a.last_token_time=Instant::now();",
                                                            "a.last_token_time=Instant::now();" + capture, 1)):
            with self.assertRaises(AssertionError):
                validate_entry(change)

    def test_missing_abort_or_swallowed_transaction_errors_are_rejected(self):
        source = compact(self.source)
        for stage in ("K2decode_verify", "K2EPresult", "K2commitacceptedprefix", "K2commitrejectedprefix"):
            before = f'frame.fail(model,a,"{stage}",e);return;'
            self.assertIn(before, source)
            for after in ("return;", f'fail_speculative_step(a,"{stage}",e);return;'):
                with self.subTest(stage=stage), self.assertRaises(AssertionError):
                    validate_entry(source.replace(before, after, 1))

    def test_plain_emit_or_missing_terminal_reconciliation_is_rejected(self):
        source = compact(self.source)
        for index in range(3):
            parts = source.split("frame.emit(")
            parts[index] += "emit_token(" + parts.pop(index + 1)
            with self.subTest(emit=index), self.assertRaises(AssertionError):
                validate_entry("frame.emit(".join(parts))
        for accepted in (0, 1):
            before = f"frame.reconcile_terminal(model,a,{accepted},&tokens_k2);"
            for after in ("", before.replace(f",{accepted},", f",{1 - accepted},")):
                with self.subTest(accepted=accepted), self.assertRaises(AssertionError):
                    validate_entry(source.replace(before, after, 1))

    def test_reject_commit_before_terminal_decision_is_rejected(self):
        accepted, rejected = branches(self.source)
        terminal = "ifa.finished{frame.reconcile_terminal(model,a,0,&tokens_k2);return;}"
        moved = rejected.replace(terminal, "", 1) + terminal
        with self.assertRaises(AssertionError):
            validate_branch(moved, False)
        validate_branch(accepted, True)

    def test_postcommit_errors_cannot_abort_consumed_journal_or_silently_continue(self):
        source = compact(self.source)
        for side in ("accepted", "rejected"):
            for action, ending in (("save", "hidden"), ("trim", "proposer"), ("propose", "prefix")):
                before = f'fail_speculative_step(a,"K2{action}{side}{ending}",e);return;'
                self.assertIn(before, source)
                for after in ("", before.replace("fail_speculative_step(a,", "frame.fail(model,a,"),
                              before.removesuffix("return;")):
                    with self.subTest(side=side, action=action), self.assertRaises(AssertionError):
                        validate_entry(source.replace(before, after, 1))

    def test_inactive_hidden_save_and_precommit_trim_mutations_are_rejected(self):
        source = compact(self.source)
        for row, token in ((1, "v1"), (0, "v0")):
            before = f"save_hidden_for_active_proposer(model,a,{token},{row})"
            after = f"model.save_hidden_for_mtp_sequence({row},&a.seq,0)"
            with self.subTest(row=row), self.assertRaises((AssertionError, ValueError)):
                validate_entry(source.replace(before, after, 1))
        _, rejected = branches(source)
        trim = "model.trim_proposer_state(&muta.seq,0,0)"
        moved = rejected.replace(trim, "Ok::<(),anyhow::Error>(())", 1)
        moved = "let_=" + trim + ";" + moved
        with self.assertRaises((AssertionError, ValueError)):
            validate_branch(moved, False)

    def test_native_recording_tests_are_wired_but_not_executed_here(self):
        self.assertIn('#[path="verify_k2_cleanup_tests.rs"]modtests;', compact(self.source))
        names = re.findall(r"#\[test\]\s*fn\s+([A-Za-z0-9_]+)", self.native)
        self.assertEqual(len(names), 8)
        for name in names:
            native = body(self.native, "fn " + name + "(")
            self.assertIn("step_verify_k2(&model,&muta,&drafts,1)", native)
            self.assertNotIn("set_var(", native)
        native = compact(self.native)
        for proof in ("model.verify_aborts", "model.verify_commits", "model.frees", "model.proposals",
                      "a.cancel_flag", "a.think_end_token", "a.min_tokens", "StreamEvent::Error", "StreamEvent::Done"):
            self.assertIn(proof, native)
        postcommit = body(self.native, "fn postcommit_proposer_failures_notify_without_aborting_consumed_journal(")
        for proof in ('["savehidden","trim","propose"]', "model.verify_aborts.lock().unwrap().is_empty()",
                      "a.pending_drafts.is_empty()"):
            self.assertIn(proof, postcommit)


if __name__ == "__main__":
    unittest.main()
