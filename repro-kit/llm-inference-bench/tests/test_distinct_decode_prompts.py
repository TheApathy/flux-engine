"""Distinct per-stream decode prompts and same-length prefill warmup (local patch).

Compiling the whole 23k-line llm_decode_bench.py needs more address space than the
repro-kit unit-test cap allows, so these tests exec only the patch's self-contained
helper block and check the wiring in the source text. The end-to-end behaviour is
covered by test_distinct_decode_prompts_e2e.py, which needs the full module.
"""
import asyncio
from pathlib import Path
import re
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "llm_decode_bench.py"
SOURCE = MODULE_PATH.read_text(encoding="utf-8")


def _block(start: str, end: str) -> str:
    i = SOURCE.index(start)
    return SOURCE[i : SOURCE.index(end, i)]


def _load_helpers() -> dict:
    namespace = {"GENERATION_PROMPT": "GEN", "Optional": None}
    exec(compile(_block("def build_messages(", "def build_user_prompt_messages("), "patch", "exec"), namespace)
    return namespace


H = _load_helpers()


def _run(coro):
    return asyncio.run(coro)


class FakeTokenizer:
    """Uneven fake tokenizer: one token per word plus one per 3 chars of long words."""

    def __init__(self):
        self.calls = 0

    async def __call__(self, text: str) -> int:
        self.calls += 1
        return sum(1 + len(w) // 3 for w in text.split())


FILLER = " ".join(["alpha", "be", "charlie", "d", "echoing", "fox"] * 400)


class StreamTagTests(unittest.TestCase):
    def test_tags_are_unique_per_stream_and_context(self):
        tags = {H["decode_stream_tag"]("runid", ctx, i) for ctx in (8192, 32768) for i in range(1, 9)}
        self.assertEqual(len(tags), 16)
        self.assertEqual(H["decode_stream_tag"]("runid", 8192, 3), "[BENCH_runid_CTX_8192_S3] ")

    def test_measured_prefix_is_not_a_prefix_of_a_stream_tag(self):
        # The stream text must diverge from the measured prompt inside the tag.
        measured = "[BENCH_runid_CTX_8192] "
        self.assertFalse(H["decode_stream_tag"]("runid", 8192, 1).startswith(measured))

    def test_zero_context_tag_is_short(self):
        self.assertEqual(H["decode_stream_tag"]("runid", 0, 2), "[S2] ")

    def test_generation_tag_only_prefixes_the_question(self):
        msgs = H["build_messages"](0, "", generation_tag="[S1] ")
        self.assertEqual(msgs, [{"role": "user", "content": "[S1] GEN"}])
        self.assertEqual(H["build_messages"](0, ""), [{"role": "user", "content": "GEN"}])
        long_msgs = H["build_messages"](10, "doc", generation_tag="[S1] ")
        self.assertIn("doc", long_msgs[0]["content"])
        self.assertEqual(long_msgs[-1]["content"], "[S1] GEN")


class RetargetTests(unittest.TestCase):
    def retarget(self, tag, target, start):
        tok = FakeTokenizer()
        text_for = lambda n: (tag + FILLER)[:n]
        text, count = _run(H["retarget_text_to_tokens"](text_for, tok, target, start))
        return text, count, tok.calls

    def test_exact_match_at_start_costs_one_probe(self):
        base = "[BENCH_r_CTX_100] " + FILLER
        tok = FakeTokenizer()
        n = 600
        target = _run(tok(base[:n]))
        text, count, calls = self.retarget("[BENCH_r_CTX_100] ", target, n)
        self.assertEqual((count, calls), (target, 1))
        self.assertEqual(text, base[:n])

    def test_longer_tag_is_trimmed_to_the_same_token_count(self):
        measured = "[BENCH_r_CTX_100] " + FILLER
        target = _run(FakeTokenizer()(measured[:1500]))
        tag = "[BENCH_r_CTX_100_S7] "
        text, count, calls = self.retarget(tag, target, 1500 + 3)
        self.assertEqual(count, target)
        self.assertTrue(text.startswith(tag))
        self.assertLess(calls, 20)

    def test_gallops_up_and_down_from_a_bad_start(self):
        target = _run(FakeTokenizer()(FILLER[:3000]))
        for start in (10, 9000):
            _, count, calls = self.retarget("", target, start)
            self.assertEqual(count, target, start)
            self.assertLess(calls, 40)

    def test_unreachable_target_returns_closest(self):
        # Every word costs >= 1 token, so a count between words may be impossible;
        # a target of 0 can only be met by the empty text.
        _, count, _ = self.retarget("", 0, 50)
        self.assertEqual(count, 0)
        text, count, _ = self.retarget("xxxxxxxxx ", 10**9, 50)
        self.assertEqual(count, _run(FakeTokenizer()(text)))


class WiringTests(unittest.TestCase):
    def test_every_stream_launch_in_run_one_cell_uses_its_own_payload(self):
        body = _block("async def run_one_cell(\n", "\nasync def run_one_cell(*args")
        launches = re.findall(r"stream_one_request\(\s*\w+,\s*url,\s*([^,]+),\s*i,", body)
        self.assertEqual(launches, ["payload_for(i)"] * 3)
        self.assertIn("stream_messages: Optional[list] = None", body)

    def test_decode_and_burst_cells_pass_stream_messages(self):
        self.assertEqual(SOURCE.count("stream_messages=await _decode_stream_messages("), 2)

    def test_warmup_runs_before_each_measured_prefill_path(self):
        self.assertEqual(SOURCE.count("await _warm_prefill_context(client, ctx, live)"), 3)
        self.assertEqual(SOURCE.count("_attach_prefill_cold(ctx)"), 3)

    def test_flags_and_json_fields(self):
        for needle in (
            '"--shared-decode-prompt"',
            '"--no-prefill-context-warmup"',
            'dest="prefill_context_warmup"',
            '"decode_prompt_mode": getattr(',
            '"decode_prompt_stream_tokens": getattr(',
            '"prefill_context_warmup": getattr(args, "prefill_context_warmup", True),',
            '"cold_ttft_seconds": (',
            '"shared_decode_prompt": getattr(args, "shared_decode_prompt", False),',
        ):
            self.assertIn(needle, SOURCE)

    def test_warmup_prompt_cannot_hit_the_measured_prefix_cache(self):
        self.assertIn('f"[WARMUP_{run_id}_PREFILL_CTX_{ctx}] " + base', SOURCE)


if __name__ == "__main__":
    unittest.main()
