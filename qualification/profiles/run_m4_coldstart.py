#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Milestone 4 Cold Start Readiness Qualification Runner.

Measures 3 consecutive cold starts to /health readiness using per-file
posix_fadvise cache eviction on model safetensors and memory recovery (>= 100 GiB).
Verifies median readiness time <= 150.0 s, log anchor sequencing, and 0 fatal patterns.
"""

import glob
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
import urllib.error

WORKTREE = os.environ.get("FLUX_WORKTREE", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
TARGET_DIR = os.environ.get("CARGO_TARGET_DIR", os.path.join(WORKTREE, "target"))
BINARY = os.path.join(TARGET_DIR, "release/flux-server")
MODEL_DIR = os.environ.get("FLUX_MODEL_DIR", os.path.expanduser("~/models/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid"))
QUAL_DIR = os.path.join(WORKTREE, "qualification/profiles")
RECIPE_PATH = os.path.join(WORKTREE, "repro-kit/recipes/flux-v9-w4096-interactive.env")
PORT = 8899
BASE_URL = f"http://127.0.0.1:{PORT}"

FATAL_PATTERNS = re.compile(
    r"\bnvrm\b|\bxid\b|\bcuda[-_: ]*fatal\b|\billegal[-_: ]*memory[-_: ]*access\b|\boom[-_: ]*kill\b|\bpanicked at\b|\bAssertion failed\b",
    re.IGNORECASE,
)

ANCHORS = [
    ("starting", re.compile(r"Atlas Spark starting\.\.\.")),
    ("scheduler", re.compile(r"Scheduler started \(batched mode, max_batch=1")),
    ("listening", re.compile(r"Listening on 0\.0\.0\.0:8899|Listening on 127\.0\.0\.1:8899")),
]


def load_env(env_path: str) -> dict:
    if not os.path.isfile(env_path):
        raise FileNotFoundError(f"Missing required recipe file: {env_path}")
    env = dict(os.environ)
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip()
                if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                    v = v[1:-1]
                env[k] = v
    # Absolute overrides
    env["ATLAS_QWEN4_PREFILL_ATTN_FI"] = "0"
    env["ATLAS_QWEN4_QSA_VERIFY_ROWS"] = "1"
    env["ATLAS_FAST_LOAD"] = "1"
    env["ATLAS_FAST_LOAD_ROUTED_GROUPS"] = "1"
    env["ATLAS_FAST_LOAD_ROUTED_GROUPS_EAGER"] = "1"
    env["FLUX_FAST_LOAD_ROUTED_GROUPS"] = "1"
    env["FLUX_FAST_LOAD_ROUTED_GROUPS_EAGER"] = "1"
    env["ATLAS_FAST_LOAD_DIRECT_IO_TENSOR_CAP"] = "20000"
    env["FLUX_FAST_LOAD_DIRECT_IO_TENSOR_CAP"] = "20000"
    env["CUDA_MODULE_LOADING"] = "LAZY"
    env["RUST_LOG"] = "info,spark::scheduler::spec_accept_log=debug,flux_server::scheduler::spec_accept_log=debug"
    return env


def evict_model_shards(model_dir: str):
    print(f"[*] Evicting clean pages for model shards in {model_dir} via posix_fadvise...")
    evicted_count = 0
    total_bytes = 0
    for pattern in ["*.safetensors", "*.json", "*.jinja", "merges.txt"]:
        for filepath in glob.glob(os.path.join(model_dir, pattern)):
            try:
                size = os.path.getsize(filepath)
                fd = os.open(filepath, os.O_RDONLY)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                os.close(fd)
                evicted_count += 1
                total_bytes += size
            except OSError as e:
                print(f"[!] Warning evicting {filepath}: {e}", file=sys.stderr)
    print(f"[+] Evicted {evicted_count} model files ({total_bytes / (1024**3):.2f} GiB) via posix_fadvise(POSIX_FADV_DONTNEED).")
    return evicted_count


def get_mem_available_gib() -> float:
    with open("/proc/meminfo", "r") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                parts = line.split()
                return int(parts[1]) / (1024 * 1024)
    return 0.0


def wait_for_memory(required_gib: float = 100.0, timeout_s: float = 60.0):
    start = time.time()
    while time.time() - start < timeout_s:
        current = get_mem_available_gib()
        if current >= required_gib:
            return current
        time.sleep(1.0)
    raise RuntimeError(f"Timeout waiting for MemAvailable >= {required_gib} GiB (current: {get_mem_available_gib():.2f} GiB)")


def verify_log(log_path: str) -> dict:
    with open(log_path, "r", errors="replace") as f:
        content = f.read()

    # Fatal pattern check
    fatal_matches = []
    for line in content.splitlines():
        if FATAL_PATTERNS.search(line):
            fatal_matches.append(line.strip())

    # Anchor checks matching repro-kit/r15-gates/startup-readiness:325-326
    starting_m = ANCHORS[0][1].search(content)
    scheduler_m = ANCHORS[1][1].search(content)
    listening_m = ANCHORS[2][1].search(content)
    anchors_found = {
        "starting": starting_m is not None,
        "scheduler": scheduler_m is not None,
        "listening": listening_m is not None,
    }
    in_order = False
    if starting_m and scheduler_m and listening_m:
        in_order = starting_m.start() < scheduler_m.start() < listening_m.start()

    return {
        "fatal_count": len(fatal_matches),
        "fatal_matches": fatal_matches,
        "anchors_verified": in_order and all(anchors_found.values()),
        "anchors": anchors_found,
    }


def run_single_coldstart(run_idx: int) -> dict:
    print(f"\n=======================================================")
    print(f"=== Cold Start Run {run_idx} / 3 ===")
    print(f"=======================================================")

    # 1. Pre-flight memory verification
    current_mem = wait_for_memory(100.0, 60.0)
    print(f"[+] Pre-flight MemAvailable: {current_mem:.2f} GiB (>= 100 GiB safe)")

    # 2. Per-file posix_fadvise cache eviction
    evicted_files_count = evict_model_shards(MODEL_DIR)

    # 3. Environment & Server launch
    env = load_env(RECIPE_PATH)
    logs_dir = os.path.join(QUAL_DIR, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"coldstart_server_run_{run_idx}.log")
    log_file = open(log_path, "w")

    cmd = [
        BINARY, "serve",
        "--no-tui",
        "--bind", "127.0.0.1",
        "--port", str(PORT),
        "--model-from-path", MODEL_DIR,
        "--model-name", "qwen38",
        "--kernel-target", "qwen3.8-flash-next",
        "--gpu-memory-utilization", "0.85",
        "--kv-cache-dtype", "bf16",
        "--block-size", "16",
        "--max-seq-len", "262144",
        "--max-prefill-tokens", "4096",
        "--max-batch-size", "1",
        "--max-num-seqs", "1",
        "--ssm-cache-slots", "0",
        "--swap-space-gb", "0",
        "--qwen4-qsa",
        "--oom-guard-mb", "4096",
        "--request-timeout", "14340",
        "--disable-simhash-watchdog",
        "--enable-prefix-caching", "false",
        "--speculative",
        "--num-drafts", "4",
        "--mtp-vocab", "0",
        "--mtp-quantization", "bf16",
    ]

    print(f"[*] Launching server (PID logged to {log_path})...")
    launch_mono_ns = time.monotonic_ns()
    proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    pid = proc.pid

    # 4. Poll /health until HTTP 200 with status: ready
    ready_mono_ns = None
    elapsed_s = None
    health_payload = None

    try:
        health_url = f"{BASE_URL}/health"
        req = urllib.request.Request(health_url, headers={"User-Agent": "flux-coldstart/1.0"})
        for _ in range(1200):  # up to 300 seconds (0.25s intervals)
            ret = proc.poll()
            if ret is not None:
                raise RuntimeError(f"Server exited prematurely with code {ret}! Check {log_path}")

            try:
                with urllib.request.urlopen(req, timeout=1.0) as resp:
                    if resp.status == 200:
                        body = json.loads(resp.read().decode())
                        if body.get("status") == "ready":
                            ready_mono_ns = time.monotonic_ns()
                            elapsed_s = (ready_mono_ns - launch_mono_ns) / 1e9
                            health_payload = body
                            break
            except (urllib.error.URLError, ConnectionResetError, OSError):
                pass
            time.sleep(0.25)

        if ready_mono_ns is None:
            raise TimeoutError(f"Server did not reach readiness within 300 seconds!")

        print(f"[+] Run {run_idx}: Reached readiness in {elapsed_s:.2f} s! (/health returned 200)")

        # Verify /v1/models
        models_url = f"{BASE_URL}/v1/models"
        with urllib.request.urlopen(urllib.request.Request(models_url), timeout=5.0) as resp:
            models_data = json.loads(resp.read().decode())
        model_ids = [m.get("id") for m in models_data.get("data", []) if isinstance(m, dict)]
        model_identity_matched = "qwen38" in model_ids
        if not model_identity_matched:
            raise RuntimeError(f"Expected model id 'qwen38' in /v1/models, got {model_ids}")
        print(f"[+] Model identity verified: {model_ids}")

    finally:
        # Graceful shutdown
        print(f"[*] Stopping server (PID {pid})...")
        proc.terminate()
        try:
            proc.wait(timeout=30.0)
        except subprocess.TimeoutExpired:
            print(f"[!] Server did not exit in 30s, issuing SIGKILL...")
            proc.kill()
            proc.wait()
        log_file.close()

    # Reclaim memory
    print(f"[*] Settling memory after run {run_idx}...")
    reclaimed_mem = wait_for_memory(100.0, 60.0)
    print(f"[+] Memory settled: {reclaimed_mem:.2f} GiB available.")

    # Audit server log
    log_audit = verify_log(log_path)
    if log_audit["fatal_count"] > 0:
        raise RuntimeError(f"Fatal patterns detected in {log_path}: {log_audit['fatal_matches']}")
    if not log_audit["anchors_verified"]:
        raise RuntimeError(f"Server log anchors failed verification: {log_audit['anchors']}")

    print(f"[+] Run {run_idx} Log Audit: 0 fatal patterns, anchors verified in order.")

    return {
        "run_idx": run_idx,
        "launch_mono_ns": launch_mono_ns,
        "ready_mono_ns": ready_mono_ns,
        "readiness_seconds": round(elapsed_s, 2),
        "health_payload": health_payload,
        "model_identity_matched": model_identity_matched,
        "evicted_files_count": evicted_files_count,
        "log_audit": log_audit,
        "log_path": log_path,
        "pre_mem_gib": round(current_mem, 2),
        "post_mem_gib": round(reclaimed_mem, 2),
    }


def main():
    print(f"=======================================================")
    print(f"=== Milestone 4: Cold Start Qualification Battery ===")
    print(f"=== Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ===")
    print(f"=======================================================")

    results = []
    for r in (1, 2, 3):
        res = run_single_coldstart(r)
        results.append(res)
        time.sleep(2.0)

    times = [r["readiness_seconds"] for r in results]
    med_time = statistics.median(times)
    print(f"\n=======================================================")
    print(f"=== Cold Start Qualification Results ===")
    print(f"=======================================================")
    print(f"Runs (s): {times}")
    print(f"Median Readiness Time: {med_time:.2f} s")
    print(f"Threshold: <= 150.0 s")

    all_models_matched = all(r["model_identity_matched"] for r in results)
    all_anchors_ok = all(r["log_audit"]["anchors_verified"] for r in results)
    total_fatals = sum(r["log_audit"]["fatal_count"] for r in results)
    passed = med_time <= 150.0 and total_fatals == 0 and all_models_matched and all_anchors_ok
    print(f"Verdict: {'PASS' if passed else 'FAIL'}")

    out_file = os.path.join(QUAL_DIR, "coldstart_results.json")
    receipt_data = {
        "schema": "flux-r15-container-gate-receipt/1",
        "milestone": "Milestone 4 — Fast Loading & Cold Start Readiness (R4)",
        "verdict": "PASS" if passed else "FAIL",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target_median_seconds": 150.0,
        "observed_runs_seconds": times,
        "median_seconds": round(med_time, 2),
        "consecutive_passes": sum(
            1
            for r in results
            if r["model_identity_matched"]
            and r["log_audit"]["anchors_verified"]
            and r["log_audit"]["fatal_count"] == 0
        ),
        "total_attempts": len(results),
        "health_status_all_200": all(r["health_payload"] is not None for r in results),
        "model_identity_matched": all_models_matched,
        "anchors_verified": all_anchors_ok,
        "fatal_patterns_count": total_fatals,
        "runs": results,
    }
    with open(out_file, "w") as f:
        json.dump(receipt_data, f, indent=2)
    print(f"[+] Cold start receipt written to {out_file}")

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
