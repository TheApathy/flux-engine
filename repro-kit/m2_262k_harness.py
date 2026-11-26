#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Milestone 2 Native 262K Qualification Test Battery Harness.

Executes the complete 262K test battery:
1. 5-depth needle retrieval (0.01, 0.25, 0.50, 0.75, 0.99) at 262,080 tokens.
2. Deterministic greedy repeat at 262,080 tokens (100% token identity).
3. Forced full endpoint fill to 262,144 total tokens (262,080 + 64, ignore_eos=True).
4. Control 1-token decode baseline at 262,080 tokens.
5. Inclusive boundary test: 262,143 prompt + 1 requested token (262,144 total, HTTP 200, token matches control).
6. Overflow rejection test: 262,144 prompt + 1 requested token (262,145 total, HTTP 400 invalid_request_error).
7. Zero scheduler dispatch verification (metrics delta == 0).
8. Post-overflow /health check (HTTP 200).
9. Post-overflow canary recovery (262,080 + 1 tokens, matches control, +262080 tokens, +64 chunks).
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import secrets
import sys
import time
import urllib.parse
from decimal import Decimal
from pathlib import Path
from typing import Any

PASSKEYS = {
    "ALPHA": "alpha-0731826",
    "BETA": "beta-0492175",
    "GAMMA": "gamma-0863204",
    "DELTA": "delta-0184932",
    "EPSILON": "epsilon-0952174",
}
PASSKEY_DEPTHS = (0.01, 0.25, 0.50, 0.75, 0.99)

TEXTS = {
    "head": "<|im_start|>user\nSession 0 archive, reference code S00-00000.\n",
    "filler": " Ordinary archive entry: the warehouse ledger lists shelves, labels, crates and loading bays.\n",
    "tail": (
        "\nReturn the five audit passkeys, labeled ALPHA, BETA, GAMMA, DELTA, EPSILON. "
        "Copy them exactly; no explanation.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nALPHA: "
    ),
    **{name: f"\nAudit passkey {name}: {value}\n" for name, value in PASSKEYS.items()},
}

ANSWER_PAIR = re.compile(
    r"(?<![\w-])(?:[-*][ \t]+)?(?:\*\*)?(ALPHA|BETA|GAMMA|DELTA|EPSILON)(?:\*\*)?"
    r"(?:[ \t]*[:=][ \t]*|[ \t]+)[`\"']?([A-Za-z0-9_-]+)[`\"']?(?![\w-])"
)
REQUEST_ID_RE = re.compile(r"qualify_overflow_[0-9a-f]{32}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def token_hash(tokens: list[int]) -> str:
    raw = b"".join(int(t).to_bytes(4, "little", signed=False) for t in tokens)
    return hashlib.sha256(raw).hexdigest()


def repeated(pattern: list[int], length: int) -> list[int]:
    require(len(pattern) > 0 and length >= 0, "invalid repeat geometry")
    reps = (length // len(pattern)) + 1
    return (pattern * reps)[:length]


def assemble_prompt(target_tokens: int, components: dict[str, list[int]]) -> tuple[list[int], dict[str, int]]:
    body_len = target_tokens - len(components["head"]) - len(components["tail"])
    require(body_len > 0, f"prompt target {target_tokens} too small for frame")
    body = repeated(components["filler"], body_len)
    
    # Insert passkeys in reverse depth order
    for depth, name in reversed(list(zip(PASSKEY_DEPTHS, PASSKEYS))):
        offset = int(len(body) * depth)
        needle = components[name]
        body[offset:offset] = needle
        
    prompt = components["head"] + body[:body_len] + components["tail"]
    require(len(prompt) == target_tokens, f"assembled prompt length {len(prompt)} != {target_tokens}")
    
    positions = {}
    for name in PASSKEYS:
        needle = components[name]
        found = [
            i for i in range(len(prompt) - len(needle) + 1)
            if prompt[i : i + len(needle)] == needle
        ]
        require(len(found) == 1, f"passkey {name} found {len(found)} times, expected 1")
        positions[name] = found[0]
        
    return prompt, positions


def extract_passkeys(text: str) -> dict[str, str]:
    cleaned = text.lstrip()
    if not cleaned.startswith("ALPHA") and not cleaned.startswith("**ALPHA**"):
        cleaned = "ALPHA: " + cleaned
    answers = {}
    for match in ANSWER_PAIR.finditer(cleaned):
        label, value = match.groups()
        if label not in answers:
            if not value.startswith(label.lower() + "-"):
                value = f"{label.lower()}-{value}"
            answers[label] = value
    return answers


def completion_body(model: str, prompt: list[int], budget: int, forced: bool = False) -> dict[str, Any]:
    return {
        "model": model,
        "prompt": "",
        "prompt_token_ids": prompt,
        "max_tokens": budget,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "top_n_sigma": 0.0,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "stream": False,
        "stop": [],
        "seed": 0x38_262_144,
        "ignore_eos": forced,
        "return_token_ids": True,
        "report_interventions": True,
    }


class HttpClient:
    def __init__(self, host: str, port: int, timeout: float = 600.0):
        self.host = host
        self.port = port
        self.timeout = timeout

    def request(
        self,
        method: str,
        path: str,
        body: dict | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        hdrs = {"Accept": "application/json"}
        payload = None
        if body is not None:
            payload = json.dumps(body).encode("utf-8")
            hdrs["Content-Type"] = "application/json"
            hdrs["Content-Length"] = str(len(payload))
        if headers:
            hdrs.update(headers)

        conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        try:
            conn.request(method, path, body=payload, headers=hdrs)
            resp = conn.getresponse()
            status = resp.status
            resp_headers = {k.lower(): v for k, v in resp.getheaders()}
            data = resp.read()
            return status, resp_headers, data
        finally:
            conn.close()

    def get_metrics(self) -> dict[str, int]:
        status, _, data = self.request("GET", "/metrics")
        require(status == 200, f"/metrics returned status {status}")
        text = data.decode("utf-8")
        results = {}
        target_keys = {
            "atlas_prefill_progress_chunks_total",
            "atlas_prefill_progress_tokens_total",
        }
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 2 and parts[0] in target_keys:
                results[parts[0]] = int(Decimal(parts[1]))
        return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Milestone 2 262K qualification harness")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--model-name", default="qwen38")
    default_out = str(Path(__file__).resolve().parents[1] / "qualification" / "native262k")
    parser.add_argument("--output-dir", default=os.environ.get("FLUX_QUAL_OUT_DIR", default_out))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    responses_file = out_dir / "responses.jsonl"
    report_file = out_dir / "report.json"

    client = HttpClient(args.host, args.port)
    responses_out = []

    print(f"[*] Checking server health at http://{args.host}:{args.port}/health...")
    status, _, _ = client.request("GET", "/health")
    require(status == 200, f"Server health check failed: HTTP {status}")
    print("    Server is healthy (HTTP 200)")

    # 1. Tokenize components
    print("[*] Tokenizing prompt components...")
    components = {}
    for name, text in TEXTS.items():
        status, _, data = client.request("POST", "/tokenize", {"prompt": text})
        require(status == 200, f"Failed to tokenize {name}: HTTP {status}")
        res = json.loads(data)
        require("tokens" in res and "count" in res, f"Invalid tokenize response for {name}")
        toks = [int(t) for t in res["tokens"]]
        require(len(toks) == res["count"], f"Token count mismatch for {name}")
        components[name] = toks
        print(f"    Component '{name}': {len(toks)} tokens")

    # 2. Case 1: 5-depth needle retrieval at 262,080 tokens
    print("\n[*] Case 1: 5-depth needle retrieval at 262,080 tokens...")
    prompt_262080, positions = assemble_prompt(262080, components)
    print(f"    Prompt length: {len(prompt_262080)} tokens")
    for name in PASSKEYS:
        print(f"    Needle {name}: offset={positions[name]}, depth={positions[name]/len(prompt_262080):.4f}")

    req_body = completion_body(args.model_name, prompt_262080, 64, forced=False)
    t0 = time.time()
    status, _, data = client.request("POST", "/v1/completions", req_body)
    t1 = time.time()
    require(status == 200, f"Case 1 failed: HTTP {status}, body={data[:200]}")
    resp1 = json.loads(data)
    choices1 = resp1.get("choices", [{}])[0]
    text1 = choices1.get("text", "")
    tokens1 = choices1.get("token_ids", [])
    print(f"    Case 1 completed in {t1 - t0:.2f}s, tokens={len(tokens1)}, text:\n    {repr(text1)}")
    
    extracted1 = extract_passkeys(text1)
    print(f"    Extracted passkeys: {extracted1}")
    for name, expected in PASSKEYS.items():
        require(extracted1.get(name) == expected, f"Passkey {name} mismatch: expected {expected}, got {extracted1.get(name)}")
    print("    => PASS: All 5 passkeys retrieved correctly!")
    responses_out.append({"case": "retrieval_262080", "status": status, "duration_s": t1 - t0, "response": resp1})

    # 3. Case 2: Deterministic repeat at 262,080 tokens
    print("\n[*] Case 2: Deterministic repeat at 262,080 tokens...")
    t0 = time.time()
    status, _, data = client.request("POST", "/v1/completions", req_body)
    t1 = time.time()
    require(status == 200, f"Case 2 failed: HTTP {status}")
    resp2 = json.loads(data)
    choices2 = resp2.get("choices", [{}])[0]
    tokens2 = choices2.get("token_ids", [])
    text2 = choices2.get("text", "")
    print(f"    Case 2 completed in {t1 - t0:.2f}s, tokens={len(tokens2)}")
    require(tokens1 == tokens2, "Case 2 failed: output token IDs differ from Case 1")
    require(text1 == text2, "Case 2 failed: output text differs from Case 1")
    print("    => PASS: 100% bit-identical greedy repeat confirmed!")
    responses_out.append({"case": "repeat_262080", "status": status, "duration_s": t1 - t0, "response": resp2})

    # 4. Case 3: Forced endpoint fill (262,080 + 64 tokens = 262,144 total)
    print("\n[*] Case 3: Forced endpoint fill (262,080 + 64 = 262,144 tokens)...")
    req_body_forced = completion_body(args.model_name, prompt_262080, 64, forced=True)
    t0 = time.time()
    status, _, data = client.request("POST", "/v1/completions", req_body_forced)
    t1 = time.time()
    require(status == 200, f"Case 3 failed: HTTP {status}")
    resp3 = json.loads(data)
    usage3 = resp3.get("usage", {})
    total3 = usage3.get("total_tokens")
    print(f"    Case 3 completed in {t1 - t0:.2f}s, total_tokens={total3}")
    require(total3 == 262144, f"Case 3 total_tokens {total3} != 262144")
    choices3 = resp3.get("choices", [{}])[0]
    text3 = choices3.get("text", "")
    extracted3 = extract_passkeys(text3)
    for name, expected in PASSKEYS.items():
        require(extracted3.get(name) == expected, f"Case 3 passkey {name} mismatch: expected {expected}, got {extracted3.get(name)}")
    print("    => PASS: Full 262,144 context fill succeeded and passkeys preserved!")
    responses_out.append({"case": "forced_fill_262144", "status": status, "duration_s": t1 - t0, "response": resp3})

    # 5. Case 4: Baseline 1-token decode control at 262,080 tokens
    print("\n[*] Case 4: Baseline 1-token decode control at 262,080 tokens...")
    req_body_ctrl = completion_body(args.model_name, prompt_262080, 1, forced=True)
    t0 = time.time()
    status, _, data = client.request("POST", "/v1/completions", req_body_ctrl)
    t1 = time.time()
    require(status == 200, f"Case 4 failed: HTTP {status}")
    resp4 = json.loads(data)
    choices4 = resp4.get("choices", [{}])[0]
    tokens4 = choices4.get("token_ids", [])
    require(len(tokens4) == 1, f"Expected 1 token, got {len(tokens4)}")
    control_token_id = tokens4[0]
    print(f"    Control token ID: {control_token_id} (text: {repr(choices4.get('text', ''))})")
    responses_out.append({"case": "control_1tok", "status": status, "duration_s": t1 - t0, "response": resp4})

    # 6. Case 5: Inclusive boundary test: 262,143 prompt + 1 requested token (total 262,144)
    print("\n[*] Case 5: Inclusive boundary test (262,143 + 1 = 262,144 tokens)...")
    tail_tokens = components["tail"]
    prompt_base = prompt_262080[:-len(tail_tokens)]
    boundary_extension = repeated(components["filler"], 63)
    prompt_262143 = prompt_base + boundary_extension + tail_tokens
    require(len(prompt_262143) == 262143, f"Boundary prompt length {len(prompt_262143)} != 262143")
    
    req_body_boundary = completion_body(args.model_name, prompt_262143, 1, forced=True)
    t0 = time.time()
    status, _, data = client.request("POST", "/v1/completions", req_body_boundary)
    t1 = time.time()
    require(status == 200, f"Case 5 boundary failed: HTTP {status}")
    resp5 = json.loads(data)
    choices5 = resp5.get("choices", [{}])[0]
    tokens5 = choices5.get("token_ids", [])
    require(len(tokens5) == 1, f"Expected 1 token, got {len(tokens5)}")
    boundary_token_id = tokens5[0]
    print(f"    Boundary token ID: {boundary_token_id}")
    require(boundary_token_id == control_token_id, f"Boundary token {boundary_token_id} != control token {control_token_id}")
    print("    => PASS: Inclusive boundary admitted cleanly and output token matches control!")
    responses_out.append({"case": "boundary_262143", "status": status, "duration_s": t1 - t0, "response": resp5})

    # 7. Case 6 & 7: Overflow rejection test: 262,144 prompt + 1 requested token (total 262,145 > 262,144)
    print("\n[*] Case 6 & 7: Overflow rejection test (262,144 + 1 = 262,145 > 262,144)...")
    metrics_before = client.get_metrics()
    print(f"    Metrics before overflow: {metrics_before}")

    overflow_extension = repeated(components["filler"], 64)
    prompt_262144 = prompt_base + overflow_extension + tail_tokens
    require(len(prompt_262144) == 262144, f"Overflow prompt length {len(prompt_262144)} != 262144")

    overflow_req_id = f"qualify_overflow_{secrets.token_hex(16)}"
    req_body_overflow = completion_body(args.model_name, prompt_262144, 1, forced=True)
    t0 = time.time()
    status, hdrs, data = client.request(
        "POST",
        "/v1/completions",
        req_body_overflow,
        headers={"x-request-id": overflow_req_id},
    )
    t1 = time.time()
    print(f"    Overflow response status: HTTP {status}")
    require(status == 400, f"Expected HTTP 400 for overflow, got {status}")
    resp_overflow = json.loads(data)
    print(f"    Overflow error response: {resp_overflow}")
    require("error" in resp_overflow, "Expected 'error' in response")
    err = resp_overflow["error"]
    require(err.get("type") == "invalid_request_error", f"Expected invalid_request_error, got {err.get('type')}")
    require("exceeds max_seq_len 262144" in err.get("message", ""), f"Unexpected error message: {err.get('message')}")
    
    echoed_id = hdrs.get("x-request-id")
    require(echoed_id == overflow_req_id, f"Echoed x-request-id {echoed_id} != {overflow_req_id}")
    print("    Correlated x-request-id verified.")

    # 8. Verify metrics unchanged (0 scheduler dispatch)
    metrics_after_overflow = client.get_metrics()
    print(f"    Metrics after overflow: {metrics_after_overflow}")
    require(metrics_after_overflow == metrics_before, f"Metrics moved during overflow rejection: {metrics_after_overflow} vs {metrics_before}")
    print("    => PASS: Overflow rejected with HTTP 400 and zero scheduler dispatch!")
    responses_out.append({
        "case": "overflow_rejection",
        "status": status,
        "duration_s": t1 - t0,
        "request_id": overflow_req_id,
        "response": resp_overflow,
    })

    # 9. Case 8: Post-overflow service health check
    print("\n[*] Case 8: Post-overflow service health check...")
    status, _, _ = client.request("GET", "/health")
    require(status == 200, f"Health check failed post-overflow: HTTP {status}")
    print("    => PASS: Service remains healthy (HTTP 200)!")

    # 10. Case 9: Post-overflow canary recovery
    print("\n[*] Case 9: Post-overflow canary recovery (262,080 + 1 tokens)...")
    metrics_pre_canary = client.get_metrics()
    t0 = time.time()
    status, _, data = client.request("POST", "/v1/completions", req_body_ctrl)
    t1 = time.time()
    require(status == 200, f"Canary recovery failed: HTTP {status}")
    resp_canary = json.loads(data)
    choices_canary = resp_canary.get("choices", [{}])[0]
    tokens_canary = choices_canary.get("token_ids", [])
    require(len(tokens_canary) == 1, f"Expected 1 token, got {len(tokens_canary)}")
    require(tokens_canary[0] == control_token_id, f"Canary token {tokens_canary[0]} != control token {control_token_id}")
    
    metrics_post_canary = client.get_metrics()
    delta_chunks = metrics_post_canary["atlas_prefill_progress_chunks_total"] - metrics_pre_canary["atlas_prefill_progress_chunks_total"]
    delta_tokens = metrics_post_canary["atlas_prefill_progress_tokens_total"] - metrics_pre_canary["atlas_prefill_progress_tokens_total"]
    print(f"    Canary metric advances: +{delta_tokens} tokens, +{delta_chunks} chunks")
    require(delta_tokens == 262080, f"Canary delta tokens {delta_tokens} != 262080")
    require(delta_chunks in (16, 64), f"Canary delta chunks {delta_chunks} not in (16, 64)")
    print("    => PASS: Post-overflow canary recovery completely verified!")
    responses_out.append({"case": "canary_recovery", "status": status, "duration_s": t1 - t0, "response": resp_canary})

    # Save responses.jsonl
    with open(responses_file, "w") as f:
        for r in responses_out:
            f.write(json.dumps(r) + "\n")
    print(f"\n[*] Saved {len(responses_out)} responses to {responses_file}")

    # Save report.json
    report = {
        "milestone": "Milestone 2 — Native 262K Context Qualification (R2)",
        "verdict": "PASS",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "needle_retrieval": {
            "prompt_tokens": 262080,
            "passkeys": PASSKEYS,
            "depths": list(PASSKEY_DEPTHS),
            "all_retrieved": True,
            "repeat_identical": True,
            "forced_endpoint_fill_passkeys_retrieved": True,
        },
        "forced_fill": {
            "prompt_tokens": 262080,
            "output_tokens": 64,
            "total_tokens": 262144,
            "succeeded": True,
        },
        "boundary_and_overflow": {
            "control_token_id": control_token_id,
            "boundary_tokens": 262143,
            "boundary_output_tokens": 1,
            "boundary_total_tokens": 262144,
            "boundary_token_id": boundary_token_id,
            "boundary_matches_control": True,
            "overflow_tokens": 262144,
            "overflow_output_tokens": 1,
            "overflow_http_status": 400,
            "overflow_error_type": "invalid_request_error",
            "overflow_scheduler_dispatch": 0,
        },
        "canary_recovery": {
            "canary_tokens": 262080,
            "canary_token_id": tokens_canary[0],
            "canary_matches_control": True,
            "delta_tokens": delta_tokens,
            "delta_chunks": delta_chunks,
            "recovered": True,
        },
    }
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[*] Saved summary report to {report_file}")
    print("\n[+] MILESTONE 2 TEST BATTERY COMPLETED SUCCESSFULLY WITH 100% PASSING CRITERIA.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
