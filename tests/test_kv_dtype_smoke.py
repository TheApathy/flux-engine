#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
test_kv_dtype_smoke.py — End-to-end smoke for every KvCacheDtype the
runtime advertises in the FromStr table.

Catches dispatch fallthrough regressions across KV cache dtype match arms
(e.g. write_kv_cache.rs and prefill/paged_attn.rs) by starting a fresh
container for each dtype, generating 64 tokens against a short prompt, and
asserting a non-error response with completion_tokens > 0.

Treats explicit model/dtype incompatibility messages as SKIP, and container
crashes or unexpected load timeouts as FAIL.

Usage:
  python3 tests/test_kv_dtype_smoke.py --model-path /path/to/model
  python3 tests/test_kv_dtype_smoke.py --image ghcr.io/theapathy/flux-engine:gb10 \
      --model-path /path/to/models/qwen3.6-35b-fp8 \
      --dtypes turbo2,turbo3,turbo4,bf16k_turbo3v
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request

PORT = 8889
CONTAINER = "flux-kv-smoke"

DEFAULT_DTYPES = [
    # Symmetric — all should pass on any model.
    "bf16", "fp8", "nvfp4", "turbo2", "turbo3", "turbo4", "turbo8",
    # Asymmetric — pass on models whose K-side weights match.
    "bf16k_turbo2v", "bf16k_turbo3v", "bf16k_turbo4v",
    "fp8k_turbo2v", "fp8k_turbo3v", "fp8k_turbo4v",
    "turbo4k_turbo3v", "turbo4k_turbo8v", "turbo3k_turbo8v",
]


def run_docker(args: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def stop_container() -> None:
    run_docker(["stop", CONTAINER], timeout=30)
    run_docker(["rm", "-f", CONTAINER], timeout=30)


def start_container(image: str, model_path: str, kv_dtype: str) -> tuple[bool, str]:
    stop_container()
    proc = run_docker(
        [
            "run", "-d",
            "--name", CONTAINER,
            "--gpus", "all",
            "--ipc=host",
            "-p", f"{PORT}:8888",
            "-v", f"{model_path}:/model:ro",
            "-e", "RUST_LOG=warn",
            image,
            "serve",
            "--model-from-path", "/model",
            "--port", "8888",
            "--bind", "0.0.0.0",
            "--max-seq-len", "8192",
            "--kv-cache-dtype", kv_dtype,
            "--kv-high-precision-layers", "0",
        ]
    )
    out = ((proc.stdout or "") + (proc.stderr or "")).strip()
    return proc.returncode == 0 and bool(proc.stdout.strip()), out[:160]


def container_running() -> bool:
    proc = run_docker(["inspect", "-f", "{{.State.Running}}", CONTAINER], timeout=15)
    return proc.returncode == 0 and proc.stdout.strip().lower() == "true"


def wait_ready(timeout_s: int = 120) -> tuple[bool, str]:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=2) as r:
                json.load(r)
            return True, "ready"
        except Exception:
            if not container_running():
                return False, "exited"
            time.sleep(2)
    return False, "timeout"


def call(prompt: str, max_tok: int = 64, timeout: int = 120) -> dict:
    body = {
        "model": "smoke",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tok,
        "temperature": 0.0,
        "stream": False,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def smoke_one(image: str, model_path: str, dtype: str, prompt: str) -> tuple[str, str]:
    ok, start_out = start_container(image, model_path, dtype)
    if not ok:
        return "FAIL", f"container failed to start: {start_out}"
    ready, reason = wait_ready()
    if not ready:
        log_proc = run_docker(["logs", "--tail", "10", CONTAINER], timeout=15)
        logs = ((log_proc.stdout or "") + (log_proc.stderr or "")).strip()
        lower = logs.lower()
        if "incompatible" in lower or "mismatch" in lower or "unsupported" in lower:
            return "SKIP", "load incompatible with model weights"
        tail_line = logs.splitlines()[-1][:120] if logs else reason
        return "FAIL", f"container {reason}: {tail_line}"
    try:
        d = call(prompt, max_tok=64)
        n = d["usage"]["completion_tokens"]
        if n <= 0:
            return "FAIL", f"empty completion ({n} tokens)"
        return "PASS", f"{n} tok ok"
    except Exception as e:
        return "FAIL", str(e)[:160]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="ghcr.io/theapathy/flux-engine:gb10")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--dtypes", default=",".join(DEFAULT_DTYPES))
    ap.add_argument("--prompt", default="Explain what attention is in one sentence.")
    args = ap.parse_args()

    results = []
    try:
        for dt in args.dtypes.split(","):
            dt = dt.strip()
            if not dt:
                continue
            print(f"\n[{dt}] ", end="", flush=True)
            status, note = smoke_one(args.image, args.model_path, dt, args.prompt)
            print(f"{status:<6} {note}", flush=True)
            results.append((dt, status, note))
    finally:
        stop_container()

    print("\n=== SUMMARY ===")
    fmt = "{:<22} {:<6} {}"
    for dt, st, note in results:
        print(fmt.format(dt, st, note))

    fails = [r for r in results if r[1] == "FAIL"]
    if fails:
        print(f"\n{len(fails)} FAIL", file=sys.stderr)
        sys.exit(1)
    print(
        f"\nAll dispatched dtypes OK ({sum(1 for r in results if r[1] == 'PASS')} pass, "
        f"{sum(1 for r in results if r[1] == 'SKIP')} skip)"
    )


if __name__ == "__main__":
    main()
