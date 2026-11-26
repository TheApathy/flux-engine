#!/usr/bin/env python3
"""Static contract checks for the opt-in benchmark intervention ledger."""

from pathlib import Path
from unittest import TestCase

ROOT = Path(__file__).resolve().parents[2]


def source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


class InterventionLedgerContractTests(TestCase):
    def test_ledger_is_opt_in_report_only_and_terminal(self):
        request = source("crates/spark-server/src/openai/chat_request.rs")
        response = source("crates/spark-server/src/openai/chat_response.rs")
        done = source("crates/spark-server/src/api/chat_stream/handle_done.rs")

        self.assertIn("pub report_interventions: bool", request)
        self.assertIn('policy: "report_only"', response)
        self.assertIn("snapshot_shared_interventions", done)

    def test_required_cross_turn_guards_are_recorded(self):
        phases = source("crates/spark-server/src/api/chat_phases.rs")
        thinking = source("crates/spark-server/src/api/chat/thinking.rs")

        for code in (
            '"F7"',
            '"F23"',
            '"F30"',
            '"F31"',
            '"F32"',
            '"F39"',
            '"F49"',
            '"F50"',
            '"F35"',
            '"F52"',
            '"F29"',
        ):
            self.assertIn(f"record_intervention({code}", phases)
        self.assertIn('record_intervention("F28"', thinking)

        # Guards with a possible no-op must gate the ledger on the helper's
        # successful mutation, not merely on detector activation.
        for mutation in (
            "if f32_reposition_failed_tool_result",
            "if prepend_reminder_to_system",
            "if f50_append_original_error",
            "if f29_inject_environment_facts",
        ):
            self.assertIn(mutation, phases)

    def test_other_predispatch_conversation_rewrites_are_recorded(self):
        chat = source("crates/spark-server/src/api/chat/mod.rs")
        build = source("crates/spark-server/src/api/chat/msg_entry.rs")
        loops = source("crates/spark-server/src/api/chat/loop_detect.rs")
        template = source("crates/spark-server/src/api/chat/template.rs")

        for code, action in (
            ("tool_error_hint", "append_recovery_hint"),
            ("tool_args", "replace_invalid_json_with_empty_object"),
            ("cwd_context", "append_environment_hint"),
            ("vacuous_system", "drop_system_message"),
            ("observation_mask", "elide_stale_tool_failure"),
            ("loop_detector", "suppress_tool_call"),
            ("loop_detector", "append_progress_hint"),
            ("task_pin", "append_goal_reminder"),
            ("xml_leak_guard", "strip_historical_tool_markup"),
            ("auto_compact", "rewrite_history"),
            ("context_overflow", "drop_history_message"),
            ("template", "force_thinking"),
        ):
            self.assertIn(f'record_intervention("{code}", "{action}")', chat)

        self.assertIn("tool_error_hint_count", build)
        self.assertIn("invalid_tool_args_count", build)
        self.assertIn("cwd_hint_injected", build)
        self.assertIn("vacuous_system_dropped", build)
        self.assertIn("loop_hint_injected", loops)
        self.assertIn("task_pin_anchor_count", loops)
        self.assertIn("xml_leak_stripped_count", template)
        self.assertIn("overflow_truncated_count", template)
        self.assertIn("template_forced_thinking", template)

    def test_scheduler_thinking_and_content_watchdogs_are_recorded(self):
        combined = "\n".join(
            source(relative)
            for relative in (
                "crates/spark-server/src/scheduler/decode_logits_step.rs",
                "crates/spark-server/src/scheduler/decode_logits_content.rs",
                "crates/spark-server/src/scheduler/emit_step.rs",
                "crates/spark-server/src/scheduler/think_spec_accept.rs",
            )
        )

        for code in (
            "thinking_loop_watchdog",
            "chatml_thinking_cycle_watchdog",
            "repeated_think_end_watchdog",
            "content_loop_watchdog",
            "catastrophic_content_loop_watchdog",
            "fuzzy_repeat_watchdog",
            "inter_tool_prose_watchdog",
        ):
            self.assertIn(f'"{code}"', combined)

    def test_streaming_and_blocking_tool_argument_repairs_are_recorded(self):
        streaming = source("crates/spark-server/src/api/chat_stream/tool_handlers.rs")
        blocking = source("crates/spark-server/src/api/chat_blocking.rs")

        for implementation in (streaming, blocking):
            self.assertIn('"tool_output_repair"', implementation)
            self.assertIn('"backfill_required_params"', implementation)
            self.assertIn('"normalize_paths"', implementation)
        self.assertIn("record_tool_repairs(state, backfilled, normalized)", streaming)
        self.assertIn("intervention_ledger", blocking)

    def test_noncanonical_tool_parser_paths_are_recorded(self):
        blocking = source("crates/spark-server/src/api/chat_blocking.rs")
        stream_state = source("crates/spark-server/src/api/chat_stream/state.rs")
        stream_token = source("crates/spark-server/src/api/chat_stream/handle_token.rs")
        stream_done = source("crates/spark-server/src/api/chat_stream/handle_done.rs")

        self.assertIn("parse_tool_calls_with_audit", blocking)
        self.assertIn('"tool_parser_recovery"', blocking)
        self.assertIn('"recovered_noncanonical_call"', stream_state)
        self.assertIn("take_recovery_count", stream_token)
        self.assertIn("take_recovery_count", stream_done)

    def test_post_dispatch_output_rewrites_are_recorded(self):
        blocking = source("crates/spark-server/src/api/chat_blocking.rs")
        stream = source("crates/spark-server/src/api/chat_stream/handle_token.rs")
        done = source("crates/spark-server/src/api/chat_stream/handle_done.rs")
        dsv = source("crates/spark-server/src/api/chat_stream/dsv41.rs")
        sanitizer = source("crates/spark-server/src/api/sanitizer.rs")
        flush = source("crates/spark-server/src/api/failures/duplicate.rs")

        for action in (
            "fuzzy_tool_name",
            "repair_json_object_prefix",
            "reject_invalid_tool_call",
        ):
            self.assertIn(f'"{action}"', blocking)
        for action in (
            "sanitize_reasoning_delta",
            "drop_partial_reasoning_markup",
            "sanitize_content_delta",
            "sanitize_tool_markup",
        ):
            self.assertIn(f'"{action}"', stream)
        self.assertIn('"drop_partial_tool_markup"', done)
        self.assertIn('"truncate_parallel_calls"', dsv)
        self.assertIn("sanitize_content_chunk_with_report", sanitizer)
        self.assertIn("flush_content_sanitizer_with_report", flush)

    def test_docs_define_complete_scope_and_protocol_exclusions(self):
        flags = source("FLAGS.md")
        normalized_flags = " ".join(flags.split())

        for code in (
            "F32",
            "F39",
            "F49",
            "conditional F50",
            "F35/F52",
            "F29",
            "stale-observation masking",
            "malformed historical tool-argument",
            "context-overflow truncation",
            "schema backfill/type coercion",
            "path normalization",
            "streaming removal of role/thinking",
            "tool_parser_recovery/recovered_noncanonical_call",
        ):
            self.assertIn(code, normalized_flags)
        self.assertIn(
            "Deterministic\nprotocol serialization is intentionally excluded", flags
        )
        self.assertIn("are not activated by a quality/failure verdict", flags)

    def test_quality_consumers_opt_in_and_fail_closed(self):
        bench = source("repro-kit/llm-inference-bench/llm_decode_bench.py")
        qualifier = source("repro-kit/lil_accuracy_qualify.py")
        microgate = source("repro-kit/tool_eval_runaway_microgate.py")
        microgate_docs = source("repro-kit/TOOL_EVAL_RUNAWAY_MICROGATE.md")
        microgate_example = source(
            "repro-kit/examples/tool-eval-runaway-manifest.example.json"
        )

        for consumer in (bench, qualifier, microgate):
            self.assertIn('"atlas.interventions.v1"', consumer)
            self.assertIn('"report_only"', consumer)
            self.assertIn("events must be empty", consumer)
        self.assertIn('"request_overrides": {"report_interventions": True}', bench)
        self.assertIn('{"report_interventions": True}', qualifier)
        self.assertIn('request.get("report_interventions")', microgate)
        self.assertIn("finish-bearing SSE choice", bench)
        self.assertIn("finish-bearing choice", microgate)
        self.assertIn("`report_interventions=true`", microgate_docs)
        self.assertIn('"report_interventions": true', microgate_example)


if __name__ == "__main__":
    import unittest

    unittest.main()
