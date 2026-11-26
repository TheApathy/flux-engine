#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Deterministic 2-prompt cold coding decode benchmark for Qwen3.8-Flash-Next.

Measures repeated greedy single-stream decode across two short coding prompts
(`d0a_lru_cache` and `d0b_bplus_tree`), recording:
  1. Repeat-to-repeat greedy output determinism (`stable_output_sha256`).
  2. Client wall-clock decode throughput (`completion_tokens / wall_s`) and,
     when reported by the endpoint, server `usage["response_token/s"]`
     (explicitly tagged via `rate_source`; left `null` when omitted).

For full 8K..128K prefill/TTFT sweeps or paired Plain vs K=5 lossless gates,
use `bench/llm-inference-bench/run-lil-bench.sh` or `repro/qwen38-flash-next/run.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


CODING_PROBES: list[tuple[str, str]] = [
    (
        "d0a_lru_cache",
        "Implement a thread-safe generic LRU cache in Rust using parking_lot and a doubly-linked arena index. Include unit tests and complexity notes.",
    ),
    (
        "d0b_bplus_tree",
        "Write a complete in-memory B+ tree in Python supporting insert, range_scan, and split_leaf, and explain the invariants.",
    ),
]


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def post_json(url: str, payload: dict[str, Any], timeout: int = 1200) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.URLError as exc:
        raise SystemExit(f"endpoint unavailable ({url}): {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen3.8-Flash-Next prefill & K=5 decode benchmark.")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8899/v1/chat/completions")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("/tmp/qwen38_flash_next_probe.json"))
    args = parser.parse_args()

    models_url = args.endpoint.rsplit("/chat/completions", 1)[0] + "/models"
    with urllib.request.urlopen(urllib.request.Request(models_url), timeout=30) as resp:
        model_id = json.load(resp)["data"][0]["id"]

    results: list[dict[str, Any]] = []
    for probe_id, prompt in CODING_PROBES:
        probe_runs: list[dict[str, Any]] = []
        for rep in range(args.repetitions):
            t0 = time.perf_counter()
            resp = post_json(
                args.endpoint,
                {
                    "model": model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": args.max_tokens,
                    "temperature": 0.0,
                },
            )
            wall_s = time.perf_counter() - t0
            choice = resp["choices"][0]
            msg = choice["message"]
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or ""
            usage = resp.get("usage", {})
            comp_tokens = int(usage.get("completion_tokens", 0))
            raw_server_tps = usage.get("response_token/s")
            server_tps = float(raw_server_tps) if raw_server_tps is not None else None
            client_tps = comp_tokens / wall_s if wall_s > 0 else 0.0
            rate_source = "usage.response_token/s" if server_tps is not None else "client_wall_only"
            fingerprint = sha256_hex(
                json.dumps(
                    {
                        "content": content,
                        "reasoning_content": reasoning,
                        "finish_reason": choice.get("finish_reason"),
                    },
                    sort_keys=True,
                ).encode("utf-8")
            )
            row = {
                "probe_id": probe_id,
                "repetition": rep,
                "completion_tokens": comp_tokens,
                "server_response_tokens_per_second": round(server_tps, 2) if server_tps is not None else None,
                "rate_source": rate_source,
                "client_tokens_per_second": round(client_tps, 2),
                "wall_seconds": round(wall_s, 4),
                "stable_output_sha256": fingerprint,
            }
            probe_runs.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

        hashes = {r["stable_output_sha256"] for r in probe_runs}
        srv_rates = [
            r["server_response_tokens_per_second"]
            for r in probe_runs
            if r["server_response_tokens_per_second"] is not None
        ]
        results.append(
            {
                "probe_id": probe_id,
                "deterministic": len(hashes) == 1,
                "stable_output_sha256": next(iter(hashes)) if len(hashes) == 1 else None,
                "median_server_tps": round(statistics.median(srv_rates), 2) if srv_rates else None,
                "median_client_tps": round(
                    statistics.median(r["client_tokens_per_second"] for r in probe_runs), 2
                ),
                "runs": probe_runs,
            }
        )

    all_deterministic = all(p["deterministic"] for p in results)
    probe_srv_medians = [p["median_server_tps"] for p in results if p["median_server_tps"] is not None]
    summary = {
        "model": model_id,
        "deterministic": all_deterministic,
        "median_server_tps": round(statistics.median(probe_srv_medians), 2) if probe_srv_medians else None,
        "median_client_tps": round(statistics.median(p["median_client_tps"] for p in results), 2),
        "probes": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0 if all_deterministic else 1


if __name__ == "__main__":
    raise SystemExit(main())
