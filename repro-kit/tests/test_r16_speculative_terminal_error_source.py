# SPDX-License-Identifier: AGPL-3.0-only
"""Static wiring checks only, not Rust execution or engine qualification."""

from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCHEDULER = ROOT / "crates/spark-server/src/scheduler"
GENERIC = (SCHEDULER / "verify_dflash_step.rs").read_text()
BATCHED = (SCHEDULER / "verify_dflash_batched_step.rs").read_text()


class SpeculativeTerminalErrorSourceTests(unittest.TestCase):
    def assert_notified_before_return(self, source, marker, stage):
        branch = re.split(r"return(?:\s+false)?;", source.split(marker, 1)[1], maxsplit=1)[0]
        self.assertIn("fail_speculative_step(", branch)
        self.assertIn(f'"{stage}"', branch)

    def test_generic_mandatory_errors_notify_before_return(self):
        for marker, stage in (
            ('tracing::error!("sync_secondary:', "verify sync_secondary"),
            ("EP broadcast K=gamma verify frame:", "verify EP frame broadcast"),
            ('tracing::error!("decode_verify_dflash:', "decode_verify_dflash"),
            ("K=gamma verifier returned the wrong row count", "verify target row count"),
            ("EP broadcast K=gamma verify result:", "verify EP result broadcast"),
            ("commit_verify_state_async (dflash):", "verify commit"),
            ("compact_verify_kv (dflash tree-fork):", "verify tree KV compaction"),
        ):
            self.assert_notified_before_return(GENERIC, marker, stage)

    def test_terminal_commit_errors_notify_even_after_emission_finished(self):
        for marker, stage in (
            (
                "terminal verify-state commit (policy walk):",
                "terminal policy verify commit",
            ),
            (
                "terminal verify-state commit (accepted draft):",
                "terminal accepted-draft verify commit",
            ),
            ("terminal verify-state commit (bonus):", "terminal bonus verify commit"),
        ):
            self.assert_notified_before_return(GENERIC, marker, stage)

    def test_batched_mandatory_errors_notify_before_return(self):
        for marker, stage in (
            ("sync_secondary (batched dflash):", "batched verify sync_secondary"),
            (
                'tracing::error!("decode_verify_dflash_batched:',
                "decode_verify_dflash_batched",
            ),
            ("batched terminal verify state underflow", "batched verify state extent"),
            ("commit_verify_state_async (batched dflash):", "batched verify commit"),
            (
                "terminal verify-state commit (batched accepted draft):",
                "terminal batched accepted-draft verify commit",
            ),
            (
                "terminal verify-state commit (batched bonus):",
                "terminal batched bonus verify commit",
            ),
        ):
            self.assert_notified_before_return(BATCHED, marker, stage)

    def test_batched_result_shape_is_checked_before_indexing_or_emission(self):
        guard = BATCHED.index("if verified_per_seq.len() != tokens_per_seq.len()")
        first_index = BATCHED.index("let verified = &mut verified_per_seq[batch_pos];")
        self.assertLess(guard, first_index)
        branch = BATCHED[guard:first_index].split("return;", 1)[0]
        self.assertIn("verified.len() != tokens.len()", branch)
        self.assertIn('"batched verify target row count"', branch)
        self.assertIn("fail_speculative_step(&mut active[idx]", branch)

    def test_policy_wrappers_notify_only_the_recorded_download_error(self):
        for filename, name, inner_name, stage in (
            # Both `_at_row` entry points delegate to their `_with_presampled`
            # implementation where the download_error recording guard lives.
            (
                "think_spec_accept.rs",
                "run_dflash_thinking_accept_at_row",
                "run_dflash_thinking_accept_at_row_with_presampled",
                "thinking",
            ),
            (
                "spec_policy_accept.rs",
                "run_dflash_content_accept_at_row",
                "run_dflash_content_accept_at_row_with_presampled",
                "content",
            ),
        ):
            text = (SCHEDULER / filename).read_text()
            outer = text.split(f"fn {name}(", 1)[1].split("\n}\n", 1)[0]
            self.assertIn(f"{inner_name}(", outer)
            wrapper = text.split(f"fn {inner_name}(", 1)[1].split("\n}\n", 1)[0]
            self.assertIn("let mut download_error = None;", wrapper)
            self.assertIn("download_error = Some(e);", wrapper)
            self.assertIn("if let Some(error) = download_error {", wrapper)
            self.assertIn(
                f'fail_speculative_step(a, "batched gumbel {stage} verify", error);',
                wrapper,
            )
            self.assertIn(
                f'fail_speculative_step(a, "{stage} verify logits download", error);',
                wrapper,
            )
            self.assertEqual(wrapper.count("fail_speculative_step("), 2)
            self.assertIn("Some(outcome)", wrapper)

    def test_no_free_failure_core_consumes_sink_and_bootstrap_rejects_cache(self):
        source = (SCHEDULER / "mtp_bootstrap_failure.rs").read_text()
        wrapper, core = source.split(
            "pub(in crate::scheduler) fn fail_speculative_step(", 1
        )
        self.assertIn("*cache_on_finish = false;", wrapper)
        self.assertIn('fail_speculative_step(a, &format!("bootstrap {stage}"), error);', wrapper)
        self.assertIn("a.finished = true;", core)
        self.assertIn("proposal_lifecycle::clear(", core)
        self.assertIn("std::mem::replace(&mut a.sink, ResponseSink::Blocking(None))", core)
        self.assertIn("send_error_to_sink(", core)
        self.assertNotIn("model.free_sequence(", core)
        self.assertNotIn("send_error(model", core)

    def test_fixed_width_and_csk_mandatory_faults_notify_before_exiting(self):
        for width, commits in ((2, 2), (3, 3), (4, 4)):
            source = (SCHEDULER / f"verify_k{width}_step.rs").read_text()
            self.assertEqual(source.count("fail_speculative_step("), 3 + 3 * width)
            self.assertEqual(source.count("frame.fail("), 2 + commits)
            for branch in source.split("if let Err(e) = model.commit_verify_state_async(")[
                1:
            ]:
                self.assertIn("frame.fail(", branch.split("return;", 1)[0])
            for marker in (
                'tracing::error!("sync_secondary:',
                f"EP broadcast verify_k{width} cmd:",
                f"EP broadcast verify_k{width} token:",
                f"EP broadcast verify_k{width} result:",
            ):
                branch = source.split(marker, 1)[1].split("return;", 1)[0]
                self.assertTrue("fail_speculative_step(" in branch or "frame.fail(" in branch)
        for name, commits in (("verify_csk_step_k2.rs", 2), ("verify_csk_step.rs", 3)):
            source = (SCHEDULER / name).read_text()
            self.assertEqual(source.count("fail_speculative_step("), 2 + commits)
            for branch in source.split("if let Err(e) = model.commit_verify_state_async(")[
                1:
            ]:
                self.assertIn("fail_speculative_step(", branch.split("continue;", 1)[0])

    def test_checkpoint_start_failure_uses_notifying_bootstrap_path(self):
        source = (SCHEDULER / "mtp_step.rs").read_text()
        branch = source.split('tracing::error!("bootstrap start_checkpoint_async:', 1)[1]
        branch = branch.split("\n        }", 1)[0]
        self.assertIn(
            'fail_bootstrap(a, &mut cache_on_finish[idx], "start_checkpoint_async", e);',
            branch,
        )
        self.assertNotIn("abort_uncheckpointed_frame", source)
        self.assertEqual(
            source.count(
                "suppress_finished_verify_cache(active, &verify_idxs, cache_on_finish);"
            ),
            4,
        )

    def test_plain_policy_wrappers_delegate_to_row_zero(self):
        for filename, name, call in (
            ("think_spec_accept.rs", "run_dflash_thinking_accept",
             "run_dflash_thinking_accept_at_row(model, a, 0, drafts, verified, ctx)"),
            ("spec_policy_accept.rs", "run_dflash_content_accept",
             "run_dflash_content_accept_at_row(model, a, 0, drafts, verified, ctx)"),
        ):
            wrapper = (SCHEDULER / filename).read_text().split(f"fn {name}(", 1)[1].split("\n}\n", 1)[0]
            self.assertIn(call, wrapper, f"{filename}: {wrapper}")


if __name__ == "__main__":
    unittest.main()
