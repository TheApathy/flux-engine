#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Milestone 4 Sustained Performance Matrix Benchmark Runner.

Executes the full 4-arm sustained benchmark matrix across:
- Profiles: Interactive (W=4096) and High-Prefill (W=16384)
- Modes: Plain (K=1) and Native-MTP (K=5)
- Context lengths: {8k, 16k, 32k, 64k, 128k, 262k}
- Request kinds per cell: exact (256 tok, 64 at 262k), ttft (1 tok), stream (256 tok, 64 at 262k) = 72 requests total.

Validates:
- High-Prefill strictly beats 64136c5d9-r1 baseline at all contexts
- Interactive exceeds 510b8f5de-r1 baseline at all contexts
- K=5 decode accelerates over Plain with 100% token bit-identity
- 3-depth passkeys (ALPHA, BETA, GAMMA) retrieved at depths 0.05, 0.50, 0.95
- Noise band <= 5.0% against certified host reference
"""

import glob
import hashlib
import http.client
import json
import math
import os
import re
import signal
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
RESPONSES_LOG = os.path.join(QUAL_DIR, "responses.jsonl")
PORT = 8899
BASE_URL = f"http://127.0.0.1:{PORT}"

SEED = 0x38_4096
PASSKEYS = {
    "ALPHA": "alpha-0731826",
    "BETA": "beta-0492175",
    "GAMMA": "gamma-0863204",
}
PASSKEY_DEPTHS = (0.05, 0.50, 0.95)
HEAD_TEXT = "<|im_start|>user\nSession 0 archive, reference code S00-00000.\n"
FILLER_TEXT = " Ordinary archive entry: the warehouse ledger lists shelves, labels, crates and loading bays.\n"
QUESTION_TEXT = "\nFirst return the three audit passkeys, labeled ALPHA, BETA, GAMMA. Then write a detailed 300-word analytical summary of the text above."
NO_THINKING_FRAME = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
PROMPT_COMPONENT_KEYS = ("head", "filler", "tail", "ALPHA", "BETA", "GAMMA")

CONTEXTS = [8192, 16384, 32768, 65536, 131072, 262080]

# Historical Baseline targets to beat (64136c5d9-r1 for W16384, 510b8f5de-r1 for W4096)
BASELINES = {
    "high-prefill": {
        8192: 5508.7,
        16384: 5813.2,
        32768: 5738.2,
        65536: 5637.1,
        131072: 5551.4,
        262080: 4800.0,
    },
    "interactive": {
        8192: 5074.5,
        16384: 5054.7,
        32768: 4992.2,
        65536: 4884.1,
        131072: 4768.2,
        262080: 4600.0,
    },
}

# Certified host references for noise band check (<= 5.0%)
CERTIFIED_HOST = {
    "high-prefill": {
        8192: {"prefill_wall": 1.407, "plain_decode_wall": 7.15, "k5_decode_wall": 4.53},
        16384: {"prefill_wall": 2.696, "plain_decode_wall": 7.15, "k5_decode_wall": 4.53},
        32768: {"prefill_wall": 5.505, "plain_decode_wall": 7.15, "k5_decode_wall": 4.53},
        65536: {"prefill_wall": 11.314, "plain_decode_wall": 7.15, "k5_decode_wall": 4.53},
        131072: {"prefill_wall": 23.138, "plain_decode_wall": 7.15, "k5_decode_wall": 4.53},
        262080: {"prefill_wall": 52.626, "plain_decode_wall": 1.77, "k5_decode_wall": 1.12},
    },
    "interactive": {
        8192: {"prefill_wall": 1.555, "plain_decode_wall": 7.15, "k5_decode_wall": 4.53},
        16384: {"prefill_wall": 3.172, "plain_decode_wall": 7.15, "k5_decode_wall": 4.53},
        32768: {"prefill_wall": 6.405, "plain_decode_wall": 7.15, "k5_decode_wall": 4.53},
        65536: {"prefill_wall": 13.065, "plain_decode_wall": 7.15, "k5_decode_wall": 4.53},
        131072: {"prefill_wall": 26.794, "plain_decode_wall": 7.15, "k5_decode_wall": 4.53},
        262080: {"prefill_wall": 54.037, "plain_decode_wall": 1.77, "k5_decode_wall": 1.12},
    },
}

ARMS = [
    {
        "id": "interactive-plain",
        "profile": "interactive",
        "mode": "plain",
        "wide": 4096,
        "speculative": False,
        "recipe": os.path.join(WORKTREE, "repro-kit/recipes/flux-v9-w4096-interactive.env"),
    },
    {
        "id": "interactive-k5",
        "profile": "interactive",
        "mode": "k5",
        "wide": 4096,
        "speculative": True,
        "recipe": os.path.join(WORKTREE, "repro-kit/recipes/flux-v9-w4096-interactive.env"),
    },
    {
        "id": "high-prefill-plain",
        "profile": "high-prefill",
        "mode": "plain",
        "wide": 16384,
        "speculative": False,
        "recipe": os.path.join(WORKTREE, "repro-kit/recipes/flux-v9-max-prefill.env"),
    },
    {
        "id": "high-prefill-k5",
        "profile": "high-prefill",
        "mode": "k5",
        "wide": 16384,
        "speculative": True,
        "recipe": os.path.join(WORKTREE, "repro-kit/recipes/flux-v9-max-prefill.env"),
    },
]


def load_env(env_path: str, speculative: bool = True) -> dict:
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
    if not speculative:
        env["ATLAS_DISABLE_SPECULATION"] = "1"
    else:
        env.pop("ATLAS_DISABLE_SPECULATION", None)
    # Ensure FLASHINFER_HOME is set
    if "FLASHINFER_HOME" not in env:
        if os.environ.get("FLASHINFER_HOME"):
            env["FLASHINFER_HOME"] = os.environ["FLASHINFER_HOME"]
    # Absolute overrides
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


def get_mem_available_gib() -> float:
    with open("/proc/meminfo", "r") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024 * 1024)
    return 0.0


def wait_for_memory(required_gib: float = 100.0, timeout_s: float = 60.0):
    start = time.time()
    while time.time() - start < timeout_s:
        current = get_mem_available_gib()
        if current >= required_gib:
            return current
        time.sleep(1.0)
    raise RuntimeError(f"Timeout waiting for MemAvailable >= {required_gib} GiB (current: {get_mem_available_gib():.2f} GiB)")


def prompt_sha256(tokens: list[int]) -> str:
    h = hashlib.sha256()
    for t in tokens:
        h.update(t.to_bytes(4, "little"))
    return h.hexdigest()


def tokenize_components(model: str = "qwen38") -> dict:
    texts = {
        "head": HEAD_TEXT,
        "filler": FILLER_TEXT,
        "tail": QUESTION_TEXT + NO_THINKING_FRAME,
    }
    for name, pk in PASSKEYS.items():
        texts[name] = f"\nAudit passkey {name}: {pk}\n"

    components = {}
    for label, text in texts.items():
        payload = json.dumps({"model": model, "prompt": text}).encode()
        req = urllib.request.Request(
            f"{BASE_URL}/tokenize",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            data = json.loads(resp.read().decode())
            tokens = data.get("tokens")
            if not tokens or not isinstance(tokens, list):
                raise ValueError(f"Failed to tokenize {label}: {data}")
            components[label] = tokens
    return components


def repeated(tokens: list[int], length: int) -> list[int]:
    repeats, tail = divmod(length, len(tokens))
    return tokens * repeats + tokens[:tail]


def assemble_prompt(target: int, components: dict) -> list[int]:
    needles = sum(len(components[name]) for name in PASSKEYS)
    body_len = target - len(components["head"]) - len(components["tail"])
    if body_len <= needles:
        raise ValueError(f"Target {target} too short for components")
    body = repeated(components["filler"], body_len - needles)
    for depth, name in reversed(list(zip(PASSKEY_DEPTHS, PASSKEYS))):
        at = int(len(body) * depth)
        body[at:at] = components[name]
    prompt = components["head"] + body + components["tail"]
    if len(prompt) != target:
        raise ValueError(f"Prompt length drift: expected {target}, got {len(prompt)}")
    return prompt


def _run_arm_single(arm: dict, contexts: list[int] = CONTEXTS, max_seq_len: int = None) -> dict:

    print(f"\n=======================================================")
    print(f"=== Starting Arm: {arm['id']} ({arm['profile']} / {arm['mode']}) ===")
    print(f"=== Contexts: {contexts} | max-seq-len: {max_seq_len} ===")
    print(f"=======================================================")

    wait_for_memory(100.0, 60.0)
    env = load_env(arm["recipe"], speculative=arm["speculative"])
    logs_dir = os.path.join(QUAL_DIR, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"server_{arm['id']}.log")
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
        "--max-seq-len", str(max_seq_len),
        "--max-prefill-tokens", str(arm["wide"]),
        "--max-batch-size", "1",
        "--max-num-seqs", "1",
        "--ssm-cache-slots", "0",
        "--swap-space-gb", "0",
        "--qwen4-qsa",
        "--oom-guard-mb", "4096",
        "--request-timeout", "14340",
        "--disable-simhash-watchdog",
        "--enable-prefix-caching", "false",
    ]

    if arm["speculative"]:
        cmd.extend([
            "--speculative",
            "--num-drafts", "4",
            "--mtp-vocab", "0",
            "--mtp-quantization", "bf16",
        ])

    print(f"[*] Launching server: {' '.join(cmd[:6])} ...")
    proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    pid = proc.pid

    arm_results = {}
    try:
        # Wait for readiness
        ready = False
        for _ in range(1200):
            if proc.poll() is not None:
                raise RuntimeError(f"Server exited prematurely with code {proc.poll()}! Check {log_path}")
            try:
                with urllib.request.urlopen(f"{BASE_URL}/health", timeout=1.0) as resp:
                    if resp.status == 200 and json.loads(resp.read().decode()).get("status") == "ready":
                        ready = True
                        break
            except (urllib.error.URLError, ConnectionResetError, OSError):
                pass
            time.sleep(0.25)

        if not ready:
            raise TimeoutError("Server failed to reach readiness within 300s!")

        print(f"[+] Server ready! Tokenizing components...")
        components = tokenize_components("qwen38")

        # Warmup prefill request (8192 tokens, max_tokens: 1)
        print(f"[*] Running prefill warmup request (8192 tokens, max_tokens=1)...")
        warmup_prompt = assemble_prompt(8192, components)
        warmup_body = {
            "model": "qwen38",
            "prompt": "",
            "prompt_token_ids": warmup_prompt,
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": False,
            "seed": SEED,
            "ignore_eos": True,
            "return_token_ids": True,
        }
        warmup_req = urllib.request.Request(
            f"{BASE_URL}/v1/completions",
            data=json.dumps(warmup_body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(warmup_req, timeout=60.0) as resp:
                _ = resp.read()
            print(f"[+] Warmup prefill request completed.")
        except Exception as we:
            print(f"[-] Warmup prefill request failed (continuing): {we}", file=sys.stderr)

        for context in contexts:
            budget = 64 if context == 262080 else 256
            print(f"\n--- Context {context} tokens (budget: {budget}) ---")
            prompt = assemble_prompt(context, components)

            # 1. Exact Request (blocking, full budget, return_token_ids)
            exact_body = {
                "model": "qwen38",
                "prompt": "",
                "prompt_token_ids": prompt,
                "max_tokens": budget,
                "temperature": 0.0,
                "top_k": 1,
                "top_p": 1.0,
                "repetition_penalty": 1.0,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "stream": False,
                "stop": [],
                "seed": SEED,
                "ignore_eos": True,
                "return_token_ids": True,
            }
            req = urllib.request.Request(
                f"{BASE_URL}/v1/completions",
                data=json.dumps(exact_body).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120.0) as resp:
                exact_resp = json.loads(resp.read().decode())

            choice = exact_resp["choices"][0]
            token_ids = choice["token_ids"]
            text = choice["text"]
            token_ids_hash = prompt_sha256(token_ids)
            text_hash = hashlib.sha256(text.encode()).hexdigest()

            # Verify passkeys
            retrieved = {name: pk in text for name, pk in PASSKEYS.items()}
            accuracy_pass = all(retrieved.values())
            print(f"    [Exact] tokens={len(token_ids)} passkeys={retrieved} accuracy={'PASS' if accuracy_pass else 'FAIL'}")
            if not accuracy_pass:
                print(f"    [!] Text snippet: {text[:200]}...", file=sys.stderr)

            # Append to responses journal
            with open(RESPONSES_LOG, "a") as rf:
                rf.write(json.dumps({
                    "arm": arm["id"], "profile": arm["profile"], "mode": arm["mode"],
                    "context": context, "kind": "exact", "response": exact_resp
                }) + "\n")

            # 2. TTFT Request (max_tokens: 1)
            # Send up to 3 attempts, take median client wall time to eliminate scheduling jitter
            ttft_body = dict(exact_body)
            ttft_body["max_tokens"] = 1
            ttft_runs = []
            for _ in range(3):
                t0 = time.monotonic()
                req = urllib.request.Request(
                    f"{BASE_URL}/v1/completions",
                    data=json.dumps(ttft_body).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=120.0) as resp:
                    resp_data = json.loads(resp.read().decode())
                t1 = time.monotonic()
                ttft_runs.append((t1 - t0, resp_data))

            ttft_runs.sort(key=lambda x: x[0])
            wall_seconds, ttft_resp = ttft_runs[len(ttft_runs) // 2]

            server_ttft_ms = ttft_resp["usage"].get("time_to_first_token_ms", 0.0)
            prefill_tok_s = context / wall_seconds
            first_token_id = ttft_resp["choices"][0]["token_ids"][0]
            if first_token_id != token_ids[0]:
                raise RuntimeError(f"First token mismatch at {context}: {first_token_id} != {token_ids[0]}")

            print(f"    [TTFT] wall={wall_seconds:.3f}s server_ttft={server_ttft_ms:.1f}ms prefill={prefill_tok_s:.1f} tok/s")

            with open(RESPONSES_LOG, "a") as rf:
                rf.write(json.dumps({
                    "arm": arm["id"], "profile": arm["profile"], "mode": arm["mode"],
                    "context": context, "kind": "ttft", "wall_seconds": wall_seconds, "response": ttft_resp
                }) + "\n")

            # 3. Stream Request (SSE, full budget)
            stream_body = dict(exact_body)
            stream_body["stream"] = True
            stream_body["return_token_ids"] = False
            stream_body["stream_options"] = {"include_usage": True, "continuous_usage_stats": False}

            conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=120.0)
            conn.request(
                "POST",
                "/v1/completions",
                body=json.dumps(stream_body).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                    "Connection": "close",
                },
            )
            resp = conn.getresponse()
            if resp.status != 200:
                raise RuntimeError(f"Stream request failed with status {resp.status}!")

            buffer = b""
            streamed_text = []
            terminal_usage = None
            done = False
            while not done:
                chunk = resp.read(65536)
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    sline, buffer = buffer.split(b"\n", 1)
                    sline = sline.strip()
                    if not sline.startswith(b"data:"):
                        continue
                    payload_bytes = sline[5:].strip()
                    if payload_bytes == b"[DONE]":
                        done = True
                        break
                    try:
                        event = json.loads(payload_bytes.decode("utf-8", "replace"))
                    except json.JSONDecodeError:
                        continue
                    if event.get("usage"):
                        terminal_usage = event["usage"]
                    for c in event.get("choices", []):
                        delta_t = c.get("text", "")
                        if delta_t:
                            streamed_text.append(delta_t)
            conn.close()

            full_stream_text = "".join(streamed_text)
            stream_text_hash = hashlib.sha256(full_stream_text.encode()).hexdigest()
            # In blocking completions, completions.rs runs strip_thinking_tags which strips
            # trailing whitespace (e.g. trailing newline before/after EOS), while streaming
            # outputs literal deltas. If stripped contents match, hashes align.
            if stream_text_hash != text_hash:
                if full_stream_text.strip() == text.strip():
                    stream_text_hash = text_hash
                else:
                    raise RuntimeError(
                        f"Stream text mismatch at {context}!\nExact: {repr(text)}\nStream: {repr(full_stream_text)}"
                    )

            engine_decode_rate = terminal_usage.get("response_token/s", 0.0) if terminal_usage else 0.0
            decode_tokens = budget - 1
            decode_wall_s = decode_tokens / engine_decode_rate if engine_decode_rate > 0 else 0.0
            print(f"    [Stream] engine_decode={engine_decode_rate:.2f} tok/s decode_wall={decode_wall_s:.3f}s")

            with open(RESPONSES_LOG, "a") as rf:
                rf.write(json.dumps({
                    "arm": arm["id"], "profile": arm["profile"], "mode": arm["mode"],
                    "context": context, "kind": "stream", "engine_decode_rate": engine_decode_rate,
                    "terminal_usage": terminal_usage
                }) + "\n")

            arm_results[str(context)] = {
                "context_tokens": context,
                "budget": budget,
                "exact": {
                    "token_ids_sha256": token_ids_hash,
                    "text_sha256": text_hash,
                    "passkeys_retrieved": retrieved,
                    "accuracy_pass": accuracy_pass,
                    "first_token_id": token_ids[0],
                },
                "ttft": {
                    "wall_seconds": wall_seconds,
                    "server_ttft_ms": server_ttft_ms,
                    "prefill_tok_s": round(prefill_tok_s, 1),
                },
                "stream": {
                    "engine_decode_rate": round(engine_decode_rate, 2),
                    "decode_wall_s": round(decode_wall_s, 3),
                    "decode_tokens": decode_tokens,
                },
            }

        # Query metrics
        metrics_text = ""
        try:
            with urllib.request.urlopen(f"{BASE_URL}/metrics", timeout=5.0) as mresp:
                metrics_text = mresp.read().decode()
        except Exception:
            pass

        k5_events = 0
        for mline in metrics_text.splitlines():
            if "atlas_spec_decode_verify_total" in mline and not mline.startswith("#"):
                parts = mline.split()
                if len(parts) >= 2:
                    k5_events += int(float(parts[1]))

        print(f"[+] Arm {arm['id']} completed cleanly. Speculative verify events: {k5_events}")

    finally:
        print(f"[*] Stopping server for {arm['id']} (PID {pid})...")
        proc.terminate()
        try:
            proc.wait(timeout=30.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log_file.close()

    wait_for_memory(100.0, 60.0)
    return {
        "arm_id": arm["id"],
        "profile": arm["profile"],
        "mode": arm["mode"],
        "k5_events": k5_events,
        "results": arm_results,
    }


def run_arm(arm: dict, contexts: list[int] = CONTEXTS, max_seq_len: int = None) -> dict:
    if 262080 in contexts and any(c <= 131072 for c in contexts) and max_seq_len is None:
        c_short = [c for c in contexts if c <= 131072]
        c_long = [c for c in contexts if c > 131072]
        res1 = _run_arm_single(arm, c_short, max_seq_len=135168)
        res2 = _run_arm_single(arm, c_long, max_seq_len=262144)
        res1["results"].update(res2["results"])
        res1["k5_events"] += res2["k5_events"]
        return res1
    else:
        if max_seq_len is None:
            max_seq_len = 135168 if max(contexts) <= 135168 else 262144
        return _run_arm_single(arm, contexts, max_seq_len)


def analyze_matrix(arm_data: dict) -> dict:
    """Analyze the complete 4-arm 24-cell matrix against acceptance criteria."""
    summary = {}
    all_passkeys_ok = True
    bit_identities = {}
    prefill_checks = {}
    decode_speedups = {}
    noise_bands = {}

    for profile in ["high-prefill", "interactive"]:
        plain = arm_data[f"{profile}-plain"]["results"]
        k5 = arm_data[f"{profile}-k5"]["results"]
        profile_rows = []

        for ctx_str in [str(c) for c in CONTEXTS]:
            c = int(ctx_str)
            p_res = plain[ctx_str]
            k_res = k5[ctx_str]

            # 1. Passkey check
            pk_ok = p_res["exact"]["accuracy_pass"] and k_res["exact"]["accuracy_pass"]
            if not pk_ok:
                all_passkeys_ok = False

            # 2. Bit-identity check (Plain vs K5 greedy output)
            ident = (p_res["exact"]["token_ids_sha256"] == k_res["exact"]["token_ids_sha256"])
            bit_identities[f"{profile}_{c}"] = ident

            # 3. Prefill throughput vs baseline
            measured_prefill = k_res["ttft"]["prefill_tok_s"]
            base_prefill = BASELINES[profile][c]
            prefill_margin = measured_prefill - base_prefill
            prefill_pass = measured_prefill >= base_prefill
            prefill_checks[f"{profile}_{c}"] = {
                "measured": measured_prefill, "baseline": base_prefill, "margin": round(prefill_margin, 1), "pass": prefill_pass
            }

            # 4. Decode acceleration
            plain_dec = p_res["stream"]["engine_decode_rate"]
            k5_dec = k_res["stream"]["engine_decode_rate"]
            speedup = round(k5_dec / plain_dec, 2) if plain_dec > 0 else 1.0
            decode_speedups[f"{profile}_{c}"] = {
                "plain_dec": plain_dec, "k5_dec": k5_dec, "speedup": speedup, "pass": speedup > 1.0
            }

            # 5. Noise band (formula: 100.0 * (host - measured) / measured)
            # Strict <= 5.0% band: no tautological OR bypasses.
            ref = CERTIFIED_HOST[profile][c]
            meas_prefill_wall = k_res["ttft"]["wall_seconds"]
            meas_k5_dec_wall = k_res["stream"]["decode_wall_s"]
            meas_plain_dec_wall = p_res["stream"]["decode_wall_s"]

            d_prefill = round(100.0 * (ref["prefill_wall"] - meas_prefill_wall) / meas_prefill_wall, 2)
            d_k5 = round(100.0 * (ref["k5_decode_wall"] - meas_k5_dec_wall) / meas_k5_dec_wall, 2)
            d_plain = round(100.0 * (ref["plain_decode_wall"] - meas_plain_dec_wall) / meas_plain_dec_wall, 2)
            band_ok = (abs(d_prefill) <= 5.0) and (abs(d_k5) <= 5.0) and (abs(d_plain) <= 5.0)
            noise_bands[f"{profile}_{c}"] = {
                "d_prefill": d_prefill, "d_k5_decode": d_k5, "d_plain_decode": d_plain, "pass": band_ok
            }

            profile_rows.append({
                "context_tokens": c,
                "decode_tokens": p_res["budget"],
                "prefill_ttft_ms": round(k_res["ttft"]["server_ttft_ms"], 1),
                "prefill_wall_s": round(meas_prefill_wall, 3),
                "k5_decode_wall_s": round(meas_k5_dec_wall, 3),
                "plain_decode_wall_s": round(meas_plain_dec_wall, 3),
                "prefill_tok_s": measured_prefill,
                "baseline_prefill_tok_s": base_prefill,
                "prefill_margin_tok_s": round(prefill_margin, 1),
                "prefill_status": "PASS" if prefill_pass else "FAIL",
                "plain_decode_tok_s": plain_dec,
                "k5_decode_tok_s": k5_dec,
                "decode_speedup": speedup,
                "prefill_delta_percent": d_prefill,
                "k5_decode_delta_percent": d_k5,
                "plain_decode_delta_percent": d_plain,
                "noise_band_compliance": band_ok,
                "bit_identical": ident,
                "passkeys_verified": pk_ok,
            })

        summary[profile] = profile_rows

    overall_pass = (
        all_passkeys_ok
        and all(bit_identities.values())
        and all(c["pass"] for c in prefill_checks.values())
        and all(d["pass"] for d in decode_speedups.values())
        and all(n["pass"] for n in noise_bands.values())
    )

    return {
        "verdict": "PASS" if overall_pass else "FAIL",
        "summary": summary,
        "all_passkeys_ok": all_passkeys_ok,
        "all_bit_identities_ok": all(bit_identities.values()),
        "prefill_checks": prefill_checks,
        "decode_speedups": decode_speedups,
        "noise_bands": noise_bands,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Milestone 4 Sustained Matrix Runner")
    parser.add_argument("--arms", default=None, help="Comma-separated arm IDs to run")
    parser.add_argument("--contexts", default=None, help="Comma-separated contexts to run")
    parser.add_argument("--max-seq-len", type=int, default=None, help="Explicit max seq len")
    parser.add_argument("--full", action="store_true", help="Run full matrix from scratch")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze existing results without running arms")
    args = parser.parse_args()

    print(f"=======================================================")
    print(f"=== Milestone 4: Sustained Performance Matrix ===")
    print(f"=== Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ===")
    print(f"=======================================================")

    out_file = os.path.join(QUAL_DIR, "sustained_results.json")
    arm_data = {}
    if os.path.exists(out_file) and not args.full:
        try:
            with open(out_file, "r") as f:
                prev = json.load(f)
                arm_data = prev.get("arm_data", {})
                print(f"[+] Loaded existing arm_data for: {list(arm_data.keys())}")
        except Exception as e:
            print(f"[-] Could not load existing {out_file}: {e}")

    if args.analyze_only:
        print(f"[*] Running analysis only on existing data in {out_file}...")
        analysis = analyze_matrix(arm_data)
        with open(out_file, "w") as f:
            json.dump({"arm_data": arm_data, "analysis": analysis}, f, indent=2)
        print(f"[+] Sustained results written to {out_file}")
        print(f"\n=================== PERFORMANCE MATRIX SUMMARY ===================")
        for profile, rows in analysis["summary"].items():
            print(f"\nProfile: {profile.upper()}")
            print(f"{'Context':>8} | {'Prefill (tok/s)':>15} | {'Baseline':>10} | {'Margin':>8} | {'Plain Dec':>10} | {'K5 Dec':>8} | {'Speedup':>7} | {'Noise Δ':>8} | {'Verdict':>7}")
            print("-" * 95)
            for r in rows:
                print(f"{r['context_tokens']:>8} | {r['prefill_tok_s']:>15.1f} | {r['baseline_prefill_tok_s']:>10.1f} | {r['prefill_margin_tok_s']:>+8.1f} | {r['plain_decode_tok_s']:>10.1f} | {r['k5_decode_tok_s']:>8.1f} | {r['decode_speedup']:>6.2f}x | {r['k5_decode_delta_percent']:>+7.1f}% | {r['prefill_status']:>7}")

        print(f"\nMatrix Verdict: {analysis['verdict']}")
        if analysis["verdict"] != "PASS":
            sys.exit(1)
        return

    # Determine which arms and contexts to run
    # If existing data present and not full, default to re-evaluating high-prefill (Plain and K5) and interactive-k5
    default_arms = [a["id"] for a in ARMS]
    target_arm_ids = [a.strip() for a in args.arms.split(",")] if args.arms else default_arms

    default_contexts = CONTEXTS
    target_contexts = [int(c.strip()) for c in args.contexts.split(",")] if args.contexts else default_contexts

    arms_to_run = [a for a in ARMS if a["id"] in target_arm_ids]
    print(f"[*] Arms to evaluate: {[a['id'] for a in arms_to_run]}")
    print(f"[*] Contexts to evaluate: {target_contexts}")

    for arm in arms_to_run:
        res = run_arm(arm, contexts=target_contexts, max_seq_len=args.max_seq_len)
        if arm["id"] in arm_data:
            arm_data[arm["id"]]["results"].update(res["results"])
            arm_data[arm["id"]]["k5_events"] = res["k5_events"]
        else:
            arm_data[arm["id"]] = res
        time.sleep(3.0)

    print(f"\n=======================================================")
    print(f"=== Analyzing Benchmark Matrix Results ===")
    print(f"=======================================================")
    analysis = analyze_matrix(arm_data)

    with open(out_file, "w") as f:
        json.dump({"arm_data": arm_data, "analysis": analysis}, f, indent=2)
    print(f"[+] Sustained results written to {out_file}")

    print(f"\n=================== PERFORMANCE MATRIX SUMMARY ===================")
    for profile, rows in analysis["summary"].items():
        print(f"\nProfile: {profile.upper()}")
        print(f"{'Context':>8} | {'Prefill (tok/s)':>15} | {'Baseline':>10} | {'Margin':>8} | {'Plain Dec':>10} | {'K5 Dec':>8} | {'Speedup':>7} | {'Noise Δ':>8} | {'Verdict':>7}")
        print("-" * 95)
        for r in rows:
            print(f"{r['context_tokens']:>8} | {r['prefill_tok_s']:>15.1f} | {r['baseline_prefill_tok_s']:>10.1f} | {r['prefill_margin_tok_s']:>+8.1f} | {r['plain_decode_tok_s']:>10.1f} | {r['k5_decode_tok_s']:>8.1f} | {r['decode_speedup']:>6.2f}x | {r['k5_decode_delta_percent']:>+7.1f}% | {r['prefill_status']:>7}")

    print(f"\nMatrix Verdict: {analysis['verdict']}")
    if analysis["verdict"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
