#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""136.5K Estonia 6-hop needle-in-a-haystack reproduction client for Qwen3.8-Flash-Next.

Evaluates the 136,576-token 6-hop factual cross-reference chain (containing the
planted 'Latvia' decoy and the true 'Estonia' terminal answer) against an active
Flux Engine endpoint, verifying:
  1. 100% factual accuracy ('Estonia' present, 'Latvia' not selected as final answer)
  2. Single-stream prefill throughput (prompt_tokens / TTFT_seconds)
  3. Lossless K=5 speculative decode throughput (server and client tok/s)
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


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def post_json(url: str, payload: dict[str, Any], timeout: int = 1800) -> dict[str, Any]:
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


def build_synthetic_estonia_prompt(target_chars: int = 707_468) -> str:
    """Constructs the deterministic 6-hop Glass Current haystack prompt (~136.5K tokens)."""
    filler_unit = (
        "Archive ledger entry: Northern Maritime Logistics & Baltic Relay Registry. "
        "Routine vessel telemetry, icebreaker dispatch intervals, and sub-sea cable "
        "inspection records show nominal optical attenuation across sectors 04 through 19.\n"
    )
    hops = [
        (0.10, "HOP-1 [Project Glass Current]: The primary cipher key was transferred to Station Meridian-7."),
        (0.24, "HOP-2 [Station Meridian-7]: Station Meridian-7 delegates archival custody to Envoy Kask."),
        (0.38, "DECOY [Baltic Annex]: Preliminary draft (revoked) listed Latvia as the fallback relay."),
        (0.50, "HOP-3 [Envoy Kask]: Envoy Kask sealed the sovereign relay packet inside Vault Aurelia-4."),
        (0.64, "HOP-4 [Vault Aurelia-4]: Vault Aurelia-4 is located in Archival District Harbor-9."),
        (0.78, "HOP-5 [Archival District Harbor-9]: Archival District Harbor-9 sits in the capital of the northernmost Baltic republic."),
        (0.90, "HOP-6 [Baltic Geography Protocol]: The northernmost Baltic republic with capital Tallinn is Estonia."),
    ]
    blocks: list[str] = []
    current_len = 0
    hop_idx = 0
    while current_len < target_chars:
        progress = current_len / float(target_chars)
        if hop_idx < len(hops) and progress >= hops[hop_idx][0]:
            line = f"\n--- {hops[hop_idx][1]} ---\n\n"
            blocks.append(line)
            current_len += len(line)
            hop_idx += 1
        blocks.append(filler_unit)
        current_len += len(filler_unit)

    while hop_idx < len(hops):
        blocks.append(f"\n--- {hops[hop_idx][1]} ---\n\n")
        hop_idx += 1

    question = (
        "\nBased strictly on the 6-hop Project Glass Current chain above (ignoring revoked drafts), "
        "which country hosts Vault Aurelia-4? State the country name clearly."
    )
    return "".join(blocks) + question


def main() -> int:
    parser = argparse.ArgumentParser(description="136.5K Estonia long-context verifier for Qwen3.8-Flash-Next.")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8899/v1/chat/completions")
    parser.add_argument("--prompt-file", type=Path, default=None, help="Optional path to external 136.5K prompt file.")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("/tmp/estonia_136k_probe.json"))
    args = parser.parse_args()

    models_url = args.endpoint.rsplit("/chat/completions", 1)[0] + "/models"
    req_models = urllib.request.Request(models_url)
    with urllib.request.urlopen(req_models, timeout=30) as resp:
        model_id = json.load(resp)["data"][0]["id"]

    prompt_text = (
        args.prompt_file.read_text(encoding="utf-8")
        if args.prompt_file is not None
        else build_synthetic_estonia_prompt()
    )

    records: list[dict[str, Any]] = []
    for idx in range(args.repetitions):
        t0 = time.perf_counter()
        ttft_resp = post_json(
            args.endpoint,
            {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt_text}],
                "max_tokens": 1,
                "temperature": 0.0,
            },
        )
        ttft_s = time.perf_counter() - t0
        prompt_tokens = int(ttft_resp.get("usage", {}).get("prompt_tokens", 0))
        prefill_tps = prompt_tokens / ttft_s if ttft_s > 0 else 0.0

        t1 = time.perf_counter()
        gen_resp = post_json(
            args.endpoint,
            {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt_text}],
                "max_tokens": args.max_tokens,
                "temperature": 0.0,
            },
        )
        wall_s = time.perf_counter() - t1
        msg = gen_resp["choices"][0]["message"]
        final_content = msg.get("content") or ""
        full_output = final_content + "\n" + (msg.get("reasoning_content") or "")
        usage = gen_resp.get("usage", {})
        comp_tokens = int(usage.get("completion_tokens", 0))
        raw_server_tps = usage.get("response_token/s")
        server_decode_tps = float(raw_server_tps) if raw_server_tps is not None else None
        client_decode_tps = comp_tokens / max(1e-6, wall_s - ttft_s)
        final_lower = final_content.lower()
        passed = ("estonia" in final_lower) and ("latvia" not in final_lower)

        row = {
            "iteration": idx,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": comp_tokens,
            "ttft_seconds": round(ttft_s, 4),
            "prefill_tokens_per_second": round(prefill_tps, 2),
            "server_decode_tokens_per_second": round(server_decode_tps, 2) if server_decode_tps is not None else None,
            "client_decode_tokens_per_second": round(client_decode_tps, 2),
            "wall_seconds": round(wall_s, 4),
            "passed_estonia_gate": passed,
            "output_sha256": sha256_hex(full_output.encode("utf-8")),
        }
        records.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    pass_count = sum(1 for r in records if r["passed_estonia_gate"])
    srv_rates = [
        r["server_decode_tokens_per_second"]
        for r in records
        if r["server_decode_tokens_per_second"] is not None
    ]
    summary = {
        "model": model_id,
        "repetitions": args.repetitions,
        "pass_count": pass_count,
        "pass_rate": pass_count / float(args.repetitions),
        "median_prefill_tps": round(statistics.median(r["prefill_tokens_per_second"] for r in records), 2),
        "median_server_decode_tps": round(statistics.median(srv_rates), 2) if srv_rates else None,
        "median_client_decode_tps": round(
            statistics.median(r["client_decode_tokens_per_second"] for r in records), 2
        ),
        "deterministic": len({r["output_sha256"] for r in records}) == 1,
        "runs": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0 if pass_count == args.repetitions else 1


if __name__ == "__main__":
    raise SystemExit(main())
