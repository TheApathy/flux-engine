# SPDX-License-Identifier: AGPL-3.0-only
"""Source wiring only. The adjacent Rust behavioral tests are not executed."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SERVER = ROOT / "crates/spark-server/src"


def read(relative):
    return (SERVER / relative).read_text(encoding="utf-8")


def section(source, start, end):
    return source.split(start, 1)[1].split(end, 1)[0]


def check_role_content(source):
    normalize = section(
        source, "fn normalize_content_delta(", "fn route_content_delta("
    )
    if 'matches!(trimmed, "user" | "assistant" | "tool")' in normalize:
        raise AssertionError("ordinary role words must not be deleted")
    if "after_role_boundary" in normalize or "follows_chatml_role_boundary" in source:
        raise AssertionError("one token must not identify a whole decoded delta")
    if "if state.thinking_done {" not in normalize:
        raise AssertionError("existing thinking-tag policy must remain")
    if 'delta.find("<think>")' not in normalize:
        raise AssertionError("real thinking controls must retain phase handling")


def check_stop_floor(source):
    body = section(
        source, "pub fn only_stop_tokens_remain(", "/// Rollback the grammar"
    )
    for marker in [
        "if !self.matcher.is_grammar_completed()",
        "if !self.fill_bitmask()",
        "let mut has_allowed_stop = false;",
        "if !stop_ids.contains(&(token_id as i32))",
        "has_allowed_stop = true;",
        "\n        has_allowed_stop\n",
    ]:
        if marker not in body:
            raise AssertionError("stop-only test must reject dead or non-stop states")


class SingleToolCloseSourceTests(unittest.TestCase):
    def test_qwen_nonempty_xml_dispatch_is_narrow_and_preserves_other_constraints(self):
        schema = ROOT / "crates/xgrammar/src/schema"
        source = (schema / "gen_scalar.rs").read_text(encoding="utf-8")
        xml = section(source, "if self.at_xml_layer() {", "// JSON-style string.")
        guard = "spec.min_length == 1 && spec.max_length == -1 && self.format == JsonFormat::QwenXml"
        self.assertIn(guard, xml)
        self.assertIn("return Ok(XML_STRING_NONEMPTY.to_string());", xml)
        self.assertLess(xml.index("if let Some(pattern)"), xml.index(guard))
        self.assertLess(xml.index("if let Some(fmt)"), xml.index(guard))
        self.assertLess(xml.index(guard), xml.index("if spec.min_length != 0"))

    def test_qwen_xml_registers_exclusion_and_whitespace_without_other_format_changes(self):
        source = (ROOT / "crates/xgrammar/src/schema/converter.rs").read_text(encoding="utf-8")
        separator = section(source, "pub(super) fn next_separator(", "/// Top-level conversion")
        self.assertIn("self.at_xml_layer() && self.format != JsonFormat::QwenXml", separator)
        self.assertIn("return String::new();", separator)
        self.assertIn("return self.whitespace_pattern();", separator)
        registration = section(source, "fn register_xml_basic_rules(", "/// Create a rule")
        self.assertIn("if self.format == JsonFormat::QwenXml", registration)
        self.assertIn("nonempty_excluding_rules(XML_STRING_NONEMPTY, param_suffix)", registration)

    def test_nonempty_exclusion_automaton_removes_completing_transition(self):
        source = (ROOT / "crates/xgrammar/src/schema/xml_string_rules.rs").read_text(encoding="utf-8")
        self.assertIn("let failure = kmp_failure(&pattern);", source)
        self.assertIn("k = failure[k - 1];", source)
        self.assertIn("if next < len {", source)
        self.assertIn('transitions(0).join(" | ")', source)
        self.assertIn("for s in 0..len {", source)
        self.assertIn("mod xml_string_rules;", (ROOT / "crates/xgrammar/src/schema/mod.rs").read_text())

    def test_qwen_native_regressions_remain_explicitly_unexecuted_by_source_checks(self):
        source = read("grammar/tests/parallel_tool_calls.rs")
        for name in [
            "cached_grammar_keeps_first_call_and_parallel_policies_distinct",
            "qwen3_coder_auto_stop_after_first_is_terminal_after_one_call",
            "qwen3_coder_required_is_terminal_after_one_call",
            "qwen3_coder_multiline_value_with_angle_brackets_closes_the_call",
            "qwen3_coder_value_ends_at_the_parameter_close_tag",
        ]:
            self.assertIn("fn " + name + "()", source)
        self.assertIn("!auto || stop_after_first", source)
        schema = ROOT / "crates/xgrammar/src/schema"
        tests = (schema / "tests_xml_strings.rs").read_text(encoding="utf-8")
        self.assertIn("fn qwen_xml_nonempty_string_keeps_the_close_tag_exclusion()", tests)
        self.assertIn("fn non_qwen_xml_retains_existing_required_string_rules()", tests)
        self.assertIn("mod tests_xml_strings;", (schema / "mod.rs").read_text(encoding="utf-8"))

    def test_role_words_are_not_lexically_filtered_or_dropped_by_token_adjacency(self):
        check_role_content(read("api/chat_stream/handle_token.rs"))

    def test_existing_normal_chatml_control_stop_remains_shared(self):
        emit = read("scheduler/emit_step.rs")
        self.assertIn(
            "im_start == Some(tok) && !a.inside_thinking && !a.ignore_eos", emit
        )
        self.assertIn("publish_hidden_committed_token(a, logprobs", emit)
        self.assertIn("commit_chatml_role_boundary(a, tok", emit)
        self.assertIn(
            "commit_chatml_role_boundary(", read("scheduler/decode_logits_step.rs")
        )

    def test_both_api_sinks_share_request_policy_without_sampling_policy_changes(self):
        setup = read("api/chat/sampling_setup.rs")
        self.assertIn("!use_triggers || parallel_tool_calls == Some(false)", setup)
        self.assertIn(
            "stop_after_first: stop_after_first_for_request(use_triggers, req.parallel_tool_calls)",
            setup,
        )
        chat = read("api/chat/mod.rs")
        self.assertLess(
            chat.index("sampling_setup::build_sampling("), chat.index("if req.stream {")
        )
        self.assertNotIn("ATLAS_TOOL_CALL_START_BIAS", setup)
        self.assertIn("let top_logprobs = req.top_logprobs.map(|n| n.min(20));", setup)

    def test_all_six_grammar_capable_parsers_forward_the_new_policy(self):
        for parser in [
            "hermes",
            "qwen3_coder",
            "glm_xml",
            "gemma4",
            "bare_json",
            "minimax_xml",
        ]:
            with self.subTest(parser=parser):
                source = read(f"tool_parser/{parser}.rs")
                self.assertIn("stop_after_first: bool", source)
                self.assertIn("tools, use_triggers, stop_after_first)", source)
        self.assertIn(
            "(at_least_one, stop_after_first || at_least_one)",
            read("grammar/compile_tools.rs"),
        )

    def test_recognized_close_handler_uses_stop_only_state_without_rearming_thinking(
        self,
    ):
        emit = read("scheduler/emit_step.rs")
        close = section(
            emit,
            "pub(super) fn commit_tool_call_close(",
            "/// Return a fuzzy tail-loop",
        )
        self.assertIn(".is_some_and(|gs| gs.only_stop_tokens_remain())", close)
        self.assertIn("if a.grammar_state.is_none() || grammar_terminal", close)
        after_if = close.split("if a.grammar_state.is_none() || grammar_terminal", 1)[1]
        then_branch, else_rest = after_if.split("} else {", 1)
        else_branch = else_rest.split("\n    }", 1)[0]
        self.assertIn("a.finished = true;", then_branch)
        self.assertNotIn("a.think_ended = false;", then_branch)
        self.assertIn("a.think_ended = false;", else_branch)
        self.assertEqual(close.count("a.output_tokens.push(tok);"), 1)
        self.assertIn("commit_tool_call_close(a, tok, was_inside_thinking)", emit)
        self.assertIn(
            "commit_tool_call_close(a, tok, was_inside_thinking)",
            read("scheduler/decode_logits_step.rs"),
        )

    def test_stop_only_state_requires_a_legal_stop(self):
        check_stop_floor(read("grammar/state.rs"))

    def test_behavioral_rust_tests_are_wired_without_claiming_native_execution(self):
        self.assertIn("mod single_tool_close_tests;", read("scheduler/mod.rs"))
        source = read("scheduler/single_tool_close_tests.rs")
        for name in [
            "serial_and_speculative_blocking_stop_at_first_recognized_close",
            "serial_and_speculative_streaming_publish_close_once_then_tool_calls_done",
            "serial_and_speculative_parallel_grammar_remains_open_for_both_sinks",
        ]:
            self.assertIn("fn " + name + "()", source)
        self.assertIn("process_decode_logits(", source)
        self.assertIn("emit_token(&mut a, CLOSE", source)
        self.assertIn("mod parallel_tool_calls;", read("grammar/tests/mod.rs"))
        self.assertIn(
            "completed_grammar_without_a_legal_stop_is_not_a_stop_only_state",
            read("grammar/tests/parallel_tool_calls.rs"),
        )

    def test_source_checks_reject_role_deletion_and_empty_mask_regressions(self):
        source = read("api/chat_stream/handle_token.rs")
        mutation = source.replace(
            "    // Strip residual think tags",
            '    if matches!(trimmed, "user" | "assistant" | "tool") {}\n    // Strip residual think tags',
            1,
        )
        self.assertNotEqual(source, mutation)
        with self.assertRaises(AssertionError):
            check_role_content(mutation)
        source = read("grammar/state.rs")
        mutation = source.replace("\n        has_allowed_stop\n", "\n        true\n", 1)
        self.assertNotEqual(source, mutation)
        with self.assertRaises(AssertionError):
            check_stop_floor(mutation)


if __name__ == "__main__":
    unittest.main()
