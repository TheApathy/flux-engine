#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Deterministic, output-sanitized Qwen3.8-Flash-Next reproduction & evidence verifier."""

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


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_EXPECTED = HERE / "expected.json"
DEFAULT_EVIDENCE_DIR = HERE / "evidence"
PROMPT = (
    "Write a complete MinHeap class in Python with insert, extract_min, and heapify, "
    "then explain the complexity of each method."
)


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def get_json(url: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        None if body is None else json.dumps(body).encode(),
        {} if body is None else {"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            return json.load(response)
    except urllib.error.URLError as error:
        raise SystemExit(f"endpoint unavailable: {url}: {error}") from error


def verify_checked_in_evidence(expected_path: Path, evidence_dir: Path) -> int:
    expected = json.loads(expected_path.read_text())
    estonia = json.loads((evidence_dir / "estonia-v9.json").read_text())
    lil_v9 = json.loads((evidence_dir / "lil-decode-v9.json").read_text())
    lil_w4096 = json.loads((evidence_dir / "lil-decode-v9-w4096.json").read_text())

    errors: list[str] = []
    est_gate = expected["estonia_136k_gate"]
    if estonia.get("pass_count") != est_gate["pass_count"]:
        errors.append(
            f"estonia pass_count {estonia.get('pass_count')} != {est_gate['pass_count']}"
        )
    if estonia.get("prefill_median_tps", 0.0) < est_gate["minimum_prefill_tps"]:
        errors.append("estonia prefill_median_tps below gate")
    if estonia.get("decode_client_median_tps", 0.0) < est_gate["minimum_decode_client_tps"]:
        errors.append("estonia decode_client_median_tps below gate")
    if estonia.get("mem_available_floor_mib", 0) < est_gate["mem_available_floor_mib"]:
        errors.append("estonia mem_available_floor_mib below gate")

    for label, record, gate_key in (
        ("lil-decode-v9", lil_v9, "v9_w16384"),
        ("lil-decode-v9-w4096", lil_w4096, "v9_w4096"),
    ):
        gate = expected["lil_decode_gates"][gate_key]
        if record.get("overall_tps", 0.0) < gate["overall_tps"]:
            errors.append(f"{label} overall_tps below {gate['overall_tps']}")
        if record.get("mem_available_floor_mib", 0) < gate["mem_available_floor_mib"]:
            errors.append(
                f"{label} mem_available_floor_mib below {gate['mem_available_floor_mib']}"
            )
        prefill_gate = expected["prefill_gates"][gate_key]
        if "minimum_16k_tps" in prefill_gate and (
            record["prefill_sweep_tps"]["16k_median"] < prefill_gate["minimum_16k_tps"]
        ):
            errors.append(f"{label} 16k prefill below gate")
        if "minimum_8k_tps" in prefill_gate and (
            record["prefill_sweep_tps"]["8k_median"] < prefill_gate["minimum_8k_tps"]
        ):
            errors.append(f"{label} 8k prefill below gate")
        if record["prefill_sweep_tps"]["128k_median"] < prefill_gate["minimum_128k_tps"]:
            errors.append(f"{label} 128k prefill below gate")

    summary = {
        "estonia_v9_pass": estonia.get("pass_count") == est_gate["pass_count"],
        "errors": errors,
        "lil_v9_overall_tps": lil_v9.get("overall_tps"),
        "lil_v9_w4096_overall_tps": lil_w4096.get("overall_tps"),
        "pass": not errors,
        "schema": "theapathy-qwen38-flash-next-evidence-verification-v1",
    }
    print(json.dumps(summary, sort_keys=True))
    return 0 if not errors else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8899/v1/chat/completions",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs/repro/qwen38-flash-next.json",
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--expected", type=Path, default=DEFAULT_EXPECTED)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--no-expected-gate", action="store_true")
    parser.add_argument(
        "--speculative-k",
        type=int,
        default=5,
        help="Expected speculative verify width K recorded as target contract metadata (not queried from the server)",
    )
    parser.add_argument(
        "--verify-evidence",
        action="store_true",
        help="Verify checked-in sanitized evidence files against expected.json without a live server",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.verify_evidence:
        return verify_checked_in_evidence(args.expected, args.evidence_dir)
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be positive")

    models_url = args.endpoint.rsplit("/chat", 1)[0] + "/models"
    model = get_json(models_url)["data"][0]["id"]
    body: dict[str, Any] = {
        "max_tokens": args.max_tokens,
        "messages": [{"content": PROMPT, "role": "user"}],
        "model": model,
        "reasoning_effort": "none",
        "temperature": 0.0,
    }

    runs: list[dict[str, Any]] = []
    for index in range(args.repetitions):
        started = time.monotonic()
        response = get_json(args.endpoint, body)
        wall_seconds = time.monotonic() - started
        choice = response["choices"][0]
        message = choice["message"]
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or ""
        usage = response.get("usage", {})
        server_rate = usage.get("response_token/s")
        if not isinstance(server_rate, (int, float)):
            raise SystemExit("server response omitted usage['response_token/s']")
        stable = {
            "content": content,
            "finish_reason": choice.get("finish_reason"),
            "reasoning_content": reasoning,
        }
        run = {
            "completion_tokens": usage.get("completion_tokens"),
            "content_bytes": len(content.encode()),
            "content_sha256": sha256(content.encode()),
            "finish_reason": choice.get("finish_reason"),
            "index": index,
            "prompt_tokens": usage.get("prompt_tokens"),
            "reasoning_bytes": len(reasoning.encode()),
            "reasoning_sha256": sha256(reasoning.encode()),
            "server_response_tokens_per_second": server_rate,
            "stable_output_sha256": sha256(canonical_bytes(stable)),
            "wall_seconds": wall_seconds,
        }
        runs.append(run)
        print(json.dumps(run, sort_keys=True), flush=True)

    rates = [run["server_response_tokens_per_second"] for run in runs]
    fingerprints = {
        (
            run["stable_output_sha256"],
            run["completion_tokens"],
            run["finish_reason"],
        )
        for run in runs
    }
    report = {
        "concurrency": 1,
        "deterministic": len(fingerprints) == 1,
        "median_server_response_tokens_per_second": statistics.median(rates),
        "model": model,
        "prompt_sha256": sha256(PROMPT.encode()),
        "protocol": "effort-none",
        "repetitions": args.repetitions,
        "request_body": body,
        "request_body_sha256": sha256(canonical_bytes(body)),
        "runs": runs,
        "schema": "theapathy-qwen38-flash-next-v1",
        "speculative_k": args.speculative_k,
        "speculative_k_source": "cli_expected_contract",
    }

    errors: list[str] = []
    if not report["deterministic"]:
        errors.append("runs were not deterministic")
    if not args.no_expected_gate:
        expected = json.loads(args.expected.read_text())
        if args.repetitions != expected["required_repetitions"]:
            errors.append(
                f"expected {expected['required_repetitions']} repetitions, "
                f"got {args.repetitions}"
            )
        if any(run["completion_tokens"] != expected["completion_tokens"] for run in runs):
            errors.append("completion token count differs from published evidence")
        if any(run["finish_reason"] != expected["finish_reason"] for run in runs):
            errors.append("finish reason differs from published evidence")
        median = report["median_server_response_tokens_per_second"]
        if median < expected["minimum_median_server_tps"]:
            errors.append(
                f"median {median:.3f} below {expected['minimum_median_server_tps']:.3f}"
            )

    report["gate"] = {"errors": errors, "pass": not errors}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bytes(report) + b"\n")
    print(json.dumps({"output": str(args.output), **report}, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
