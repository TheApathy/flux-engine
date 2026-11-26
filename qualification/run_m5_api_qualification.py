#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Milestone 5 Live Container API Qualification Suite.

Executes the 9-assertion API test suite against the live container running
on http://127.0.0.1:8899 under an exclusive GPU lockfile.
"""

from __future__ import annotations
import argparse
import fcntl
import http.client
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List

FATAL_PATTERNS = re.compile(
    r"\bnvrm\b|\bxid\b|\bcuda[-_: ]*fatal\b|\billegal[-_: ]*memory[-_: ]*access\b|\boom[-_: ]*kill\b|\bpanicked at\b|\bAssertion failed\b",
    re.IGNORECASE,
)

TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Return current weather for one city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    }
]


def check_mem_available_gib() -> float:
    with open("/proc/meminfo", "r") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024 * 1024)
    return 0.0


def wait_for_health(base_url: str, timeout_s: float = 360.0) -> bool:
    print(f"[*] Polling {base_url}/health for readiness (timeout={timeout_s}s)...")
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            req = urllib.request.Request(f"{base_url}/health", method="GET")
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    if data.get("status") in ("ok", "ready"):
                        print(f"[+] Server ready after {time.time() - start:.1f}s")
                        return True
        except Exception:
            time.sleep(2.0)
    return False


def test_01_health(base_url: str) -> Dict[str, Any]:
    req = urllib.request.Request(f"{base_url}/health", method="GET")
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        assert resp.status == 200, f"Expected 200, got {resp.status}"
        assert body.get("status") in ("ok", "ready"), f"Expected status ok or ready, got {body}"
        return {"status": "PASS", "http_status": resp.status, "body": body}


def test_02_models(base_url: str) -> Dict[str, Any]:
    req = urllib.request.Request(f"{base_url}/v1/models", method="GET")
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        assert resp.status == 200, f"Expected 200, got {resp.status}"
        models = [m.get("id") for m in body.get("data", [])]
        assert len(models) > 0, "No models returned"
        print(f"    Available models: {models}")
        return {"status": "PASS", "http_status": resp.status, "models": models, "selected_model": models[0]}


def test_03_chat_blocking(base_url: str, model_id: str) -> Dict[str, Any]:
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Reply with the single word: READY"}],
        "max_tokens": 16,
        "temperature": 0.0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            resp_status = resp.status
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", "replace")
        print(f"[-] HTTP {e.code} Error Body: {err_body}")
        raise

    assert resp_status == 200, f"Expected 200, got {resp_status}"
    choice = body["choices"][0]
    content = choice["message"]["content"]
    finish_reason = choice.get("finish_reason")
    usage = body.get("usage", {})
    print(f"    Blocking response: {content!r} (finish_reason={finish_reason})")
    assert "READY" in content.upper(), f"Expected READY in content, got {content!r}"
    return {
        "status": "PASS",
        "http_status": resp_status,
        "content": content,
        "finish_reason": finish_reason,
        "usage": usage,
    }


def test_04_chat_streaming(base_url: str, model_id: str) -> Dict[str, Any]:
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Count from 1 to 5."}],
        "max_tokens": 32,
        "temperature": 0.0,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    chunks_received = 0
    assembled_content = ""
    done_seen = False
    with urllib.request.urlopen(req, timeout=30.0) as resp:
        assert resp.status == 200, f"Expected 200, got {resp.status}"
        assert "text/event-stream" in resp.headers.get("Content-Type", "")
        for line in resp:
            line_str = line.decode("utf-8").strip()
            if not line_str.startswith("data:"):
                continue
            data_part = line_str[5:].strip()
            if data_part == "[DONE]":
                done_seen = True
                break
            try:
                chunk = json.loads(data_part)
                delta = chunk["choices"][0].get("delta", {})
                if "content" in delta and delta["content"]:
                    assembled_content += delta["content"]
                chunks_received += 1
            except Exception:
                pass
    assert done_seen, "Did not observe terminal [DONE]"
    assert chunks_received > 0, "No chunks received"
    print(f"    Streaming response ({chunks_received} chunks): {assembled_content!r}")
    return {
        "status": "PASS",
        "chunks_count": chunks_received,
        "content": assembled_content,
        "done_seen": done_seen,
    }


def test_05_tool_calling_blocking(base_url: str, model_id: str) -> Dict[str, Any]:
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Use get_weather once to look up Paris."}],
        "tools": TOOL_SCHEMA,
        "tool_choice": "required",
        "parallel_tool_calls": False,
        "max_tokens": 64,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30.0) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        assert resp.status == 200, f"Expected 200, got {resp.status}"
        choice = body["choices"][0]
        finish_reason = choice.get("finish_reason")
        tool_calls = choice["message"].get("tool_calls", [])
        assert finish_reason == "tool_calls", f"Expected finish_reason tool_calls, got {finish_reason}"
        assert len(tool_calls) == 1, f"Expected 1 tool call, got {len(tool_calls)}"
        fn = tool_calls[0]["function"]
        assert fn["name"] == "get_weather", f"Expected get_weather, got {fn['name']}"
        args = json.loads(fn["arguments"])
        assert args.get("city") == "Paris", f"Expected Paris, got {args}"
        print(f"    Tool call (blocking): {fn['name']}({args})")
        return {
            "status": "PASS",
            "finish_reason": finish_reason,
            "tool_call": tool_calls[0],
            "parsed_arguments": args,
        }


def test_06_tool_calling_streaming(base_url: str, model_id: str) -> Dict[str, Any]:
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Use get_weather once to look up Paris."}],
        "tools": TOOL_SCHEMA,
        "tool_choice": "required",
        "parallel_tool_calls": False,
        "max_tokens": 64,
        "temperature": 0.0,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    chunks_received = 0
    fn_name = ""
    raw_args = ""
    terminal_finish_reason = None
    with urllib.request.urlopen(req, timeout=30.0) as resp:
        assert resp.status == 200, f"Expected 200, got {resp.status}"
        for line in resp:
            line_str = line.decode("utf-8").strip()
            if not line_str.startswith("data:"):
                continue
            data_part = line_str[5:].strip()
            if data_part == "[DONE]":
                break
            try:
                chunk = json.loads(data_part)
                choice = chunk["choices"][0]
                if choice.get("finish_reason"):
                    terminal_finish_reason = choice["finish_reason"]
                delta = choice.get("delta", {})
                if "tool_calls" in delta and delta["tool_calls"]:
                    tc = delta["tool_calls"][0]
                    if "function" in tc:
                        fn = tc["function"]
                        if "name" in fn and fn["name"]:
                            fn_name = fn["name"]
                        if "arguments" in fn and fn["arguments"]:
                            raw_args += fn["arguments"]
                chunks_received += 1
            except Exception:
                pass
    assert fn_name == "get_weather", f"Expected get_weather, got {fn_name}"
    parsed_args = json.loads(raw_args)
    assert parsed_args.get("city") == "Paris", f"Expected city Paris, got {parsed_args}"
    assert terminal_finish_reason == "tool_calls", f"Expected tool_calls, got {terminal_finish_reason}"
    print(f"    Tool call (streaming assembled): {fn_name}({parsed_args}) (finish_reason={terminal_finish_reason})")
    return {
        "status": "PASS",
        "chunks_count": chunks_received,
        "function_name": fn_name,
        "assembled_arguments": raw_args,
        "parsed_arguments": parsed_args,
        "finish_reason": terminal_finish_reason,
    }


def test_07_client_cancellation(host: str, port: int, model_id: str) -> Dict[str, Any]:
    payload = json.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": "Write a long essay on the history of computation."}],
        "max_tokens": 512,
        "temperature": 0.0,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    conn = http.client.HTTPConnection(host, port, timeout=30.0)
    conn.request("POST", "/v1/chat/completions", body=payload, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    assert resp.status == 200, f"Expected 200, got {resp.status}"

    chunks_read = 0
    abrupt_drop = False
    # Read up to 8 SSE data chunks then drop connection abruptly
    for line in resp:
        line_str = line.decode("utf-8", errors="replace").strip()
        if line_str.startswith("data:") and line_str != "data: [DONE]":
            chunks_read += 1
            if chunks_read >= 8:
                print(f"    Abruptly dropping TCP connection after reading {chunks_read} chunks...")
                conn.sock.close()
                conn.close()
                abrupt_drop = True
                break

    assert chunks_read >= 8 and abrupt_drop, (
        f"Expected >= 8 chunks before abrupt TCP drop, only read {chunks_read} (stream ended early)"
    )
    time.sleep(1.0)
    return {
        "status": "PASS",
        "chunks_before_cancellation": chunks_read,
        "abrupt_drop": abrupt_drop,
    }


def wait_for_memory(required_gib: float = 100.0, timeout_s: float = 90.0) -> float:
    print(f"[*] Waiting for MemAvailable >= {required_gib} GiB (timeout={timeout_s}s)...")
    start = time.time()
    while time.time() - start < timeout_s:
        current = check_mem_available_gib()
        if current >= required_gib:
            return current
        time.sleep(2.0)
    current = check_mem_available_gib()
    if current < required_gib:
        raise RuntimeError(f"Timeout waiting for MemAvailable >= {required_gib} GiB (current: {current:.2f} GiB)")
    return current


def test_08_slot_reuse_and_recovery(base_url: str, model_id: str) -> Dict[str, Any]:
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Post-cancellation probe. Return OK."}],
        "max_tokens": 16,
        "temperature": 0.0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30.0) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        assert resp.status == 200, f"Expected 200, got {resp.status}"
        content = body["choices"][0]["message"]["content"]
        print(f"    Post-cancellation probe response: {content!r}")
        assert len(content.strip()) > 0, "Empty response from post-cancellation probe"

        active_requests: float | None = None
        req_m = urllib.request.Request(f"{base_url}/metrics", method="GET")
        with urllib.request.urlopen(req_m, timeout=5.0) as resp_m:
            assert resp_m.status == 200, f"Expected 200 from /metrics, got {resp_m.status}"
            metrics_text = resp_m.read().decode("utf-8")
            for line in metrics_text.splitlines():
                m = re.match(r"^atlas_requests_active(?:\{[^}]*\})?\s+([0-9eE.+-]+)$", line.strip())
                if m:
                    active_requests = float(m.group(1))
                    break
        assert active_requests is not None, "atlas_requests_active metric missing from /metrics"
        assert active_requests == 0.0, f"Leaked active requests: {active_requests}"
        return {
            "status": "PASS",
            "http_status": resp.status,
            "content": content,
            "atlas_requests_active": active_requests,
        }



def test_09_raw_log_safety(container_name: str) -> Dict[str, Any]:
    logs_proc = subprocess.run(["docker", "logs", container_name], capture_output=True, text=True, check=True)
    logs = logs_proc.stdout + logs_proc.stderr
    fatal_matches: List[str] = []
    for line in logs.splitlines():
        if FATAL_PATTERNS.search(line):
            fatal_matches.append(line.strip())
    print(f"    Fatal log matches found: {len(fatal_matches)}")
    assert len(fatal_matches) == 0, f"Fatal error patterns found: {fatal_matches}"
    return {
        "status": "PASS",
        "total_log_lines": len(logs.splitlines()),
        "fatal_matches_count": 0,
        "fatal_matches": [],
    }


def run_full_api_suite(container_name: str, host: str = "127.0.0.1", port: int = 8899) -> Dict[str, Any]:
    base_url = f"http://{host}:{port}"
    print(f"\n=======================================================")
    print(f"=== Milestone 5 Live Container API Qualification Suite ===")
    print(f"=======================================================")

    results: Dict[str, Any] = {}

    print("\n[1/9] Testing /health readiness...")
    results["health"] = test_01_health(base_url)
    print("      -> PASS")

    print("\n[2/9] Testing /v1/models catalog...")
    models_res = test_02_models(base_url)
    results["models"] = models_res
    model_id = models_res["selected_model"]
    print(f"      -> PASS (Using model: {model_id})")

    print("\n[3/9] Testing /v1/chat/completions (blocking)...")
    results["chat_blocking"] = test_03_chat_blocking(base_url, model_id)
    print("      -> PASS")

    print("\n[4/9] Testing /v1/chat/completions (streaming SSE)...")
    results["chat_streaming_sse"] = test_04_chat_streaming(base_url, model_id)
    print("      -> PASS")

    print("\n[5/9] Testing Tool Calling (blocking)...")
    results["tool_calling_blocking"] = test_05_tool_calling_blocking(base_url, model_id)
    print("      -> PASS")

    print("\n[6/9] Testing Tool Calling (streaming delta assembly)...")
    results["tool_calling_streaming"] = test_06_tool_calling_streaming(base_url, model_id)
    print("      -> PASS")

    print("\n[7/9] Testing Client Cancellation (drop TCP at chunk 8)...")
    results["client_cancellation"] = test_07_client_cancellation(host, port, model_id)
    print("      -> PASS")

    print("\n[8/9] Testing Slot Reuse & Server Recovery...")
    results["slot_reuse_and_recovery"] = test_08_slot_reuse_and_recovery(base_url, model_id)
    print("      -> PASS")

    print("\n[9/9] Testing Container Raw Log Safety Audit...")
    results["raw_log_safety"] = test_09_raw_log_safety(container_name)
    print("      -> PASS")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Milestone 5 API Qualification Suite")
    parser.add_argument("--image", default="flux-engine:r17-candidate", help="Container image")
    parser.add_argument("--container-name", default="flux-m5-api-qualification", help="Container name")
    parser.add_argument("--model-dir", default=os.environ.get("FLUX_MODEL_DIR", os.path.expanduser("~/models/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid")), help="Model directory")
    parser.add_argument("--port", type=int, default=8899, help="Serving port")
    parser.add_argument("--lockfile", default=os.environ.get("FLUX_LOCK_FILE", "/tmp/flux-gpu.lock"), help="GPU lockfile")
    parser.add_argument("--output", default=None, help="Output JSON receipt path")
    args = parser.parse_args()

    print(f"[*] Acquiring exclusive lock on {args.lockfile}...")
    lock_fd = os.open(args.lockfile, os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        print(f"[+] Exclusive lock acquired on {args.lockfile}")

        # Pre-flight memory check
        mem_available = wait_for_memory(100.0, 90.0)
        print(f"[+] Pre-flight MemAvailable: {mem_available:.2f} GiB (>= 100 GiB safe)")


        # Clean existing container if any
        subprocess.run(["docker", "rm", "-f", args.container_name], capture_output=True)

        # Launch container
        cmd = [
            "docker", "run", "-d",
            "--name", args.container_name,
            "--gpus", "all",
            "--ipc=host",
            "--security-opt", "seccomp=unconfined",
            "--ulimit", "memlock=-1",
            "--ulimit", "stack=67108864",
            "--network", "host",
            "-v", f"{args.model_dir}:/models:ro",
            "-e", "FLUX_HOST=127.0.0.1",
            "-e", f"FLUX_PORT={args.port}",
            "-e", "FLUX_MODE=interactive",
            "-e", "FLUX_CONCURRENCY=1",
            "-e", "FLUX_MODEL_NAME=qad5500",
            "-e", "FLUX_THINKING=off",
            "-e", "MODELS_ROOT=/models",
            "-e", "FLUX_PRIVACY_FIRST=1",
            "-e", "FLUX_OPT_OUT=all",
            "-e", "ATLAS_FAST_LOAD_DIRECT_IO_TENSOR_CAP=20000",
            "-e", "ATLAS_FAST_LOAD_ROUTED_GROUPS=1",
            "-e", "ATLAS_FAST_LOAD_ROUTED_GROUPS_EAGER=1",
            args.image,
            "serve",
        ]
        print(f"[*] Starting container with command:\n    {' '.join(cmd)}")
        cid = subprocess.check_output(cmd, text=True).strip()
        print(f"[+] Container started: {cid[:12]}")

        try:
            # Wait for readiness
            base_url = f"http://127.0.0.1:{args.port}"
            t0 = time.time()
            ready = wait_for_health(base_url, timeout_s=360.0)
            readiness_time = round(time.time() - t0, 1)
            if not ready:
                print("[-] Server failed readiness check!", file=sys.stderr)
                log_out = subprocess.run(["docker", "logs", "--tail", "50", args.container_name], capture_output=True, text=True)
                print(f"[-] Last 50 log lines:\n{log_out.stdout}\n{log_out.stderr}")
                return 1

            suite_results = run_full_api_suite(args.container_name, port=args.port)

            receipt_data = {
                "milestone": "Milestone 5 Container API Qualification Suite",
                "verdict": "PASS",
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "endpoint": base_url,
                "lockfile_held": args.lockfile,
                "pre_flight_mem_gib": mem_available,
                "readiness_time_seconds": readiness_time,
                "container_id": cid,
                "cases": suite_results,
            }

            return 0
        finally:
            print(f"[*] Stopping container {args.container_name}...")
            subprocess.run(["docker", "stop", "-t", "10", args.container_name], capture_output=True)
            subprocess.run(["docker", "rm", "-f", args.container_name], capture_output=True)
            print("[+] Container removed.")

            # Post-flight memory check
            post_mem = check_mem_available_gib()
            print(f"[*] Post-flight MemAvailable: {post_mem:.2f} GiB")
            if "receipt_data" in locals():
                receipt_data["post_flight_mem_gib"] = post_mem
                if args.output:
                    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
                    with open(args.output, "w") as f:
                        json.dump(receipt_data, f, indent=2)
                    print(f"[+] API Suite Receipt saved to {args.output}")
            if post_mem < 100.0:
                print(f"[!] WARNING: Post-flight MemAvailable ({post_mem:.2f} GiB) < 100.0 GiB!", file=sys.stderr)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
        print(f"[+] Lock released on {args.lockfile}")


if __name__ == "__main__":
    sys.exit(main())
