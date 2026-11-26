"""Fail-closed SSE intervention-ledger tests for completion-stat profiles."""

import asyncio
from contextlib import asynccontextmanager
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

MODULE_PATH = Path(__file__).resolve().parents[1] / "llm_decode_bench.py"
SPEC = importlib.util.spec_from_file_location("llm_decode_bench_ledger", MODULE_PATH)
BENCH = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BENCH)


EMPTY_LEDGER = {
    "schema": BENCH.INTERVENTION_LEDGER_SCHEMA,
    "policy": BENCH.INTERVENTION_LEDGER_POLICY,
    "events": [],
}


class FakeStreamingClient:
    def __init__(self, events=(), *, done=True, raw_lines=None):
        self.events = events
        self.done = done
        self.raw_lines = raw_lines

    @asynccontextmanager
    async def stream(self, method, url, *, json, timeout):
        async def lines():
            if self.raw_lines is not None:
                for line in self.raw_lines:
                    yield line
            else:
                for event in self.events:
                    yield "data: " + BENCH.json.dumps(event)
            if self.done:
                yield "data: [DONE]"

        yield SimpleNamespace(status_code=200, aiter_lines=lines)


def base_events():
    return [
        {
            "choices": [
                {"index": 0, "delta": {"content": "answer"}, "finish_reason": None}
            ]
        },
        {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "interventions": copy.deepcopy(EMPTY_LEDGER),
        },
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 8,
                "completion_tokens": 1,
                "total_tokens": 9,
                "prompt_tokens_details": {
                    "cached_tokens": 2,
                    "audio_tokens": 0,
                },
                "completion_tokens_details": {
                    "reasoning_tokens": 0,
                    "audio_tokens": 0,
                    "accepted_prediction_tokens": 0,
                    "rejected_prediction_tokens": 0,
                },
                "time_to_first_token_ms": 12.5,
                "response_token/s": 41.25,
            },
        },
    ]


def run(events, *, done=True):
    return asyncio.run(
        BENCH.stream_completion_stats_request(
            FakeStreamingClient(events, done=done),
            "http://fixture/v1/chat/completions",
            {"stream": True, "report_interventions": True},
            1,
            "profile",
            1,
            "",
            "final_answer",
            False,
        )
    )


def run_raw(*raw_lines, done=True):
    return asyncio.run(
        BENCH.stream_completion_stats_request(
            FakeStreamingClient(done=done, raw_lines=raw_lines),
            "http://fixture/v1/chat/completions",
            {"stream": True, "report_interventions": True},
            1,
            "profile",
            1,
            "",
            "final_answer",
            False,
        )
    )


class InterventionLedgerStreamTests(unittest.TestCase):
    def test_empty_ledger_on_finish_bearing_terminal_event_passes(self):
        result = run(base_events())
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.interventions, EMPTY_LEDGER)

    def test_missing_nonempty_or_nonterminal_ledger_fails_closed(self):
        missing = base_events()
        del missing[1]["interventions"]
        result = run(missing)
        self.assertFalse(result.ok)
        self.assertIn("missing the terminal intervention ledger", result.error)

        nonempty = base_events()
        nonempty[1]["interventions"]["events"] = [
            {"code": "F32", "action": "duplicate", "count": 1}
        ]
        result = run(nonempty)
        self.assertFalse(result.ok)
        self.assertIn("events must be empty", result.error)

        nonterminal = base_events()
        nonterminal[0]["interventions"] = nonterminal[1].pop("interventions")
        result = run(nonterminal)
        self.assertFalse(result.ok)
        self.assertIn("finish-bearing", result.error)

    def test_unknown_schema_or_policy_fails_closed(self):
        for key, value in (("schema", "unknown"), ("policy", "disabled")):
            with self.subTest(key=key):
                events = base_events()
                events[1]["interventions"][key] = value
                result = run(events)
                self.assertFalse(result.ok)
                self.assertIn(key, result.error)

    def test_ledger_does_not_make_a_truncated_or_continued_stream_terminal(self):
        result = run(base_events(), done=False)
        self.assertFalse(result.ok)
        self.assertIn("[DONE]", result.error)

        continued = base_events()
        continued.insert(
            2,
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "too late"},
                        "finish_reason": None,
                    }
                ]
            },
        )
        result = run(continued)
        self.assertFalse(result.ok)
        self.assertIn("choice after the terminal", result.error)

    def test_malformed_duplicate_key_and_nonfinite_events_fail_closed(self):
        for raw, expected in (
            ('data: {"choices":[}', "malformed JSON"),
            ('data:{"choices":[}', "malformed JSON"),
            (
                'data: {"choices":[],"choices":[]}',
                "duplicate key 'choices'",
            ),
            ('data: {"choices":[],"usage":NaN}', "non-finite number 'NaN'"),
            (
                'data: {"choices":[],"usage":{"total_tokens":1e9999}}',
                "non-finite number",
            ),
            (
                'data: {"choices":[],"usage":{"total_tokens":-1e9999}}',
                "non-finite number",
            ),
            ('garbage: {"choices":[]}', "unsupported SSE line"),
        ):
            with self.subTest(raw=raw):
                result = run_raw(raw)
                self.assertFalse(result.ok)
                self.assertIn(expected, result.error)

    def test_duplicate_finish_bearing_choices_fail_closed(self):
        events = base_events()
        events.insert(
            2,
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        )
        result = run(events)
        self.assertFalse(result.ok)
        self.assertIn("choice after the terminal", result.error)

    def test_duplicate_done_or_data_after_done_fails_closed(self):
        serialized = ["data: " + BENCH.json.dumps(event) for event in base_events()]
        result = run_raw(*serialized, "data: [DONE]")
        self.assertFalse(result.ok)
        self.assertIn("more than one [DONE]", result.error)

        result = run_raw(
            *serialized,
            "data: [DONE]",
            'data: {"choices":[]}',
            done=False,
        )
        self.assertFalse(result.ok)
        self.assertIn("data after the [DONE]", result.error)

    def test_terminal_usage_is_exact_once_final_and_uncoerced(self):
        mutations = (
            ("prompt_tokens", "8", "non-negative integer"),
            ("prompt_tokens", 8.0, "non-negative integer"),
            ("prompt_tokens", True, "non-negative integer"),
            ("prompt_tokens", -1, "non-negative integer"),
            ("total_tokens", 10, "inconsistent"),
            ("extra", 0, "keys must be exactly"),
        )
        for key, replacement, expected in mutations:
            with self.subTest(key=key, replacement=replacement):
                events = base_events()
                events[2]["usage"][key] = replacement
                result = run(events)
                self.assertFalse(result.ok)
                self.assertIn(expected, result.error)

        missing = base_events()
        del missing[2]["usage"]
        result = run(missing)
        self.assertFalse(result.ok)
        self.assertIn("exactly one terminal usage", result.error)

        early = base_events()
        usage = early.pop(2)
        early.insert(0, usage)
        result = run(early)
        self.assertFalse(result.ok)
        self.assertIn("must follow", result.error)

        attached = base_events()
        attached[1]["usage"] = attached.pop(2)["usage"]
        result = run(attached)
        self.assertFalse(result.ok)
        self.assertIn("choices-empty", result.error)

        after_usage = base_events()
        after_usage.append({"choices": []})
        result = run(after_usage)
        self.assertFalse(result.ok)
        self.assertIn("after terminal usage", result.error)

    def test_live_atlas_usage_extensions_are_exact_and_hostile_safe(self):
        self.assertTrue(run(base_events()).ok)

        cases = (
            (("prompt_tokens_details", "cached_tokens"), True, "non-negative integer"),
            (("prompt_tokens_details", "cached_tokens"), 9, "exceeds prompt_tokens"),
            (("prompt_tokens_details", "extra"), 0, "keys must be exactly"),
            (
                ("completion_tokens_details", "reasoning_tokens"),
                2,
                "exceeds completion_tokens",
            ),
            (
                ("completion_tokens_details", "rejected_prediction_tokens"),
                1.0,
                "non-negative integer",
            ),
            (("completion_tokens_details", "extra"), 0, "keys must be exactly"),
            (("time_to_first_token_ms",), True, "must be numeric"),
            (("time_to_first_token_ms",), -0.1, "finite and non-negative"),
            (("response_token/s",), "41.25", "must be numeric"),
        )
        for path, replacement, expected in cases:
            with self.subTest(path=path, replacement=replacement):
                events = base_events()
                target = events[2]["usage"]
                for component in path[:-1]:
                    target = target[component]
                target[path[-1]] = replacement
                result = run(events)
                self.assertFalse(result.ok)
                self.assertIn(expected, result.error)


if __name__ == "__main__":
    unittest.main()
