# SPDX-License-Identifier: AGPL-3.0-only
"""Source-contract checks only; the adjacent Rust behavioral tests are not run."""

import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
REQUEST = ROOT / "crates/spark-server/src/openai/chat_request.rs"
POLICY = ROOT / "crates/spark-server/src/api/chat/thinking.rs"


def section(text, start, end):
    return text.split(start, 1)[1].split(end, 1)[0]


def omitted_budget_contract(text):
    resolve = section(
        text, "pub fn resolve_thinking(", "pub(super) fn default_max_tokens("
    )
    anthropic = section(resolve, "// 1. Anthropic:", "// 2. vLLM PR:")
    if "return (true, None);" not in anthropic:
        raise AssertionError("omitted Anthropic budget must defer to model policy")
    if "return (true, Some(budget));" not in anthropic:
        raise AssertionError("explicit Anthropic budget changed")
    kwargs = section(resolve, "// 4. vLLM stable:", "// 5. Atlas legacy:")
    if "return (budget > 0, Some(budget));" not in kwargs:
        raise AssertionError("explicit kwargs budget changed")
    compact = " ".join(kwargs.split())
    if "return if enabled { (true, None) } else { (false, Some(0)) };" not in compact:
        raise AssertionError("kwargs omitted/disabled budget contract changed")
    legacy = section(resolve, "// 5. Atlas legacy:", "// 6. Model default")
    if "if self.enable_thinking {\n            return (true, None);" not in legacy:
        raise AssertionError("legacy omitted budget must defer to model policy")
    if "const UNKNOWN_EFFORT_THINKING_BUDGET: u32 = 256;" not in text:
        raise AssertionError("unknown effort fallback changed")
    if "_ => UNKNOWN_EFFORT_THINKING_BUDGET," not in resolve:
        raise AssertionError("unknown effort must not select the model maximum")


class OmittedThinkingBudgetSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.request = REQUEST.read_text(encoding="utf-8")
        cls.policy = POLICY.read_text(encoding="utf-8")

    def test_only_omitted_enabled_budgets_defer_to_model_policy(self):
        omitted_budget_contract(self.request)

    def test_resolution_priority_and_effort_table_are_preserved(self):
        body = section(
            self.request,
            "pub fn resolve_thinking(",
            "pub(super) fn default_max_tokens(",
        )
        markers = [
            "// 1. Anthropic:",
            "// 2. vLLM PR:",
            "// 3. OpenAI:",
            "// 4. vLLM stable:",
            "// 5. Atlas legacy:",
            "// 6. Model default",
        ]
        positions = [body.index(marker) for marker in markers]
        self.assertEqual(positions, sorted(positions))
        for mapping in [
            '"none" => 0',
            '"minimal" => 64',
            '"low" => 128',
            '"medium" => 256',
            '"high" => 512',
            '"xhigh" | "max" => 1024',
        ]:
            self.assertIn(mapping, body)
        self.assertIn('&& t == "disabled"', body)
        self.assertIn("return (false, Some(0));", body)
        self.assertIn("if model_default {\n            (true, None)", body)

    def test_existing_policy_and_tool_safety_caps_are_not_changed(self):
        cap = section(
            self.policy, "fn cap_thinking_budget(", "/// Generation allowance"
        )
        self.assertIn("if tools_active && thinking_in_tools {\n        7", cap)
        self.assertIn("} else {\n        9", cap)
        self.assertIn(".clamp(1, u32::MAX as u64)", cap)
        # Explicit budgets: respect both the positive model/CLI ceiling and the safety cap.
        self.assertIn(
            "let model_ceiling = if model_max > 0 { model_max } else { u32::MAX };",
            cap,
        )
        self.assertIn("Some(budget) => budget.min(model_ceiling).min(safety_max),", cap)
        # Omitted budgets: model ceiling, safety cap and the MODEL.toml answer
        # reserve (0 = off, the pre-9.15.5 policy).
        self.assertIn("None => model_ceiling\n            .min(safety_max)", cap)
        self.assertIn("omitted_budget_answer_cap(generation_max, answer_reserve)", cap)
        self.assertIn("if answer_reserve == 0 {\n        return u32::MAX;", self.policy)
        self.assertIn("max_tokens.min(tool_max_tokens)", self.policy)

    def test_existing_disable_tool_error_and_model_policy_wiring_remain(self):
        resolver = section(
            self.policy, "pub(super) fn resolve_thinking(", "fn cap_thinking_budget("
        )
        self.assertIn(
            "if state.disable_thinking {\n        return (false, None);", resolver
        )
        # Runtime settings: the model default and effort table come from the
        # request's admission snapshot, startup values unless changed.
        self.assertIn("state.behavior.thinking_default", resolver)
        self.assertIn("req.resolve_thinking_with_effort(model_default,", resolver)
        self.assertIn("&& !req.thinking_explicitly_requested()", resolver)
        self.assertIn("recent_message_is_tool_error(&req.messages)", resolver)
        self.assertIn("let max_budget = defaults.max_thinking_budget;", resolver)
        self.assertIn("generation_max_tokens(", resolver)

    def test_behavioral_rust_tests_are_wired_and_cover_unchanged_controls(self):
        tests = (
            ROOT / "crates/spark-server/src/openai/thinking_budget_tests.rs"
        ).read_text()
        for name in [
            "enabled_without_budget_defers_to_model_policy_on_every_channel",
            "explicit_budgets_disable_and_zero_semantics_are_unchanged",
            "known_and_unknown_effort_budgets_are_unchanged_on_both_spellings",
            "omitted_budget_does_not_fall_through_to_lower_priority_controls",
            "silent_and_legacy_false_still_use_the_model_default",
        ]:
            self.assertIn("fn " + name + "()", tests)
        module = (ROOT / "crates/spark-server/src/openai/mod.rs").read_text()
        self.assertIn("#[cfg(test)]\nmod thinking_budget_tests;", module)
        self.assertIn(
            "fn omitted_budget_resolution_consumes_the_existing_model_and_tool_policy()",
            self.policy,
        )
        self.assertIn(
            "fn explicit_budget_policy_is_unchanged_by_the_omission_fix()", self.policy
        )

    def test_cli_and_completions_measurement_sources_are_unchanged(self):
        for path, expected in [
            (
                "crates/spark-server/src/cli.rs",
                # cli.rs re-pinned after Stage 3B truth-audit fixes;
                # thinking-budget flags and measurement paths unchanged.
                "d6da2e445725c8d45f123e9ffbb17dc1e2b6b401fae849bbcedbe965b5304086",
            ),
            (
                "crates/spark-server/src/api/completions.rs",
                # completions.rs re-pinned after Stage 3B truth-audit fixes (H-41/H-42/H-49);
                # default-path measurement timing unchanged.
                "3c67e270273a587645777dad3032e3ec982d9f8aeaef258fd8e60745c71d2aa4",
            ),
        ]:
            with self.subTest(path=path):
                self.assertEqual(
                    hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), expected
                )

    def test_source_contract_rejects_omission_and_unknown_effort_regressions(self):
        for before, after in [
            ("return (true, None);", "return (true, Some(256));"),
            (
                "return if enabled {\n                    (true, None)",
                "return if enabled {\n                    (true, Some(256))",
            ),
            (
                "if self.enable_thinking {\n            return (true, None);",
                "if self.enable_thinking {\n            return (true, Some(256));",
            ),
            ("_ => UNKNOWN_EFFORT_THINKING_BUDGET,", "_ => return (true, None),"),
        ]:
            with self.subTest(mutation=before):
                self.assertIn(before, self.request)
                with self.assertRaises(AssertionError):
                    omitted_budget_contract(self.request.replace(before, after, 1))


if __name__ == "__main__":
    unittest.main()
