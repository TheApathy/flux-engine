"""End to end: distinct per-stream decode prompts and the same-length prefill warmup.

Runs main() against the FakeStreamingServer of test_mixed_prefill.py. Compiling the
full bench needs more address space than the repro-kit unit-test cap, so the module
skips (rather than errors) when it cannot be loaded. Takes ~25 s: each cell waits a
3 s readiness window and the bench pauses 2 s between cells.
"""
import json
import re
import sys

import pytest

try:
    import test_mixed_prefill as MIXED
except MemoryError:  # the 23k-line bench does not compile under a small RLIMIT_AS
    pytest.skip("llm_decode_bench.py needs more address space than this run allows",
                allow_module_level=True)

BENCH = MIXED.BENCH
STREAM_TAG = re.compile(r"\[(BENCH_[a-z]+_CTX_(\d+)(?:_S(\d+))?|WARMUP_[a-z]+_PREFILL_CTX_(\d+)|S(\d+))\]")


class RecordingServer(MIXED.FakeStreamingServer):
    """Records each chat request's leading tag and its max_tokens, in arrival order."""

    def chat(self, handler, payload):
        text = "".join(str(m.get("content") or "") for m in payload.get("messages") or [])
        match = STREAM_TAG.search(text)
        with self.lock:
            self.__dict__.setdefault("seen", []).append({
                "tag": match.group(1) if match else "",
                "chars": len(text),
                "max_tokens": payload.get("max_tokens"),
            })
        return super().chat(handler, payload)


ARGV = [
    "--concurrency", "2", "--contexts", "0,2k", "--duration", "0.5", "--max-tokens", "24",
    "--decode-warmup-seconds", "0", "--prefill-contexts", "2k", "--temperature", "0",
    "--display-mode", "plain", "--no-hw-monitor", "--no-calibration-cache", "--no-resume",
]


def run(monkeypatch, tmp_path, *extra):
    server = RecordingServer(step_s=0.005, tokens_per_chunk=2)
    try:
        report = MIXED.run_cli(monkeypatch, server, tmp_path, [*ARGV, *extra])
        return report, list(server.seen)
    finally:
        server.close()


def test_default_sends_distinct_prompts_and_warms_each_prefill_length(monkeypatch, tmp_path):
    report, seen = run(monkeypatch, tmp_path)
    meta = report["metadata"]
    assert meta["decode_prompt_mode"] == "distinct_per_stream"
    assert meta["prefill_context_warmup"] is True

    tags = [s["tag"] for s in seen]
    # Context 2k: stream 1 carries its own tag at the measured prompt's char length.
    measured = [s for s in seen if re.fullmatch(r"BENCH_[a-z]+_CTX_2048", s["tag"])]
    stream1 = [s for s in seen if re.fullmatch(r"BENCH_[a-z]+_CTX_2048_S1", s["tag"])]
    assert measured and stream1
    assert {s["chars"] for s in stream1} == {measured[0]["chars"]}
    # Context 0: stream 1 prefixes the generation prompt with a short tag.
    assert "S1" in tags

    # The unmeasured same-length request comes before the measured scout.
    warm = next(i for i, t in enumerate(tags) if t.startswith("WARMUP_") and t.endswith("_PREFILL_CTX_2048"))
    scout = next(i for i, s in enumerate(seen) if s in measured and s["max_tokens"] == 1)
    assert warm < scout
    row = report["prefill"]["2048"]
    assert row["context_warmup"] is True and row["cold_ttft_seconds"] > 0
    assert row["cold_prompt_tokens"] > 0 and row["cold_error"] == ""


def test_opt_outs_restore_shared_prompt_and_cold_prefill(monkeypatch, tmp_path):
    report, seen = run(monkeypatch, tmp_path, "--shared-decode-prompt", "--no-prefill-context-warmup")
    meta = report["metadata"]
    assert meta["decode_prompt_mode"] == "shared"
    assert meta["prefill_context_warmup"] is False
    tags = {s["tag"] for s in seen}
    assert not any(re.search(r"_S\d+$|^S\d+$", t) for t in tags)
    assert not any(t.endswith("_PREFILL_CTX_2048") for t in tags)
    row = report["prefill"]["2048"]
    assert row["context_warmup"] is False and row["cold_ttft_seconds"] is None
