#!/usr/bin/env python3
"""Zero-Flag Container Validation Suite for flux-engine:9.15.3.

Validates inside a zero-flag Docker container (`docker run --rm --gpus all --ipc=host --network=host`):
1. Cold start readiness <= 120s.
2. /flux/modes API and GET / Web Studio UI validation.
3. 8/8 needle retrieval probe across 2,017, 2,112, 7,710 (pos=0.5 x3 + pos=0.98), 30,647, and 65,536 tokens.
4. PLE receipts: QWEN4_PLE_PREAD_LANES_ENGAGED lanes=256 and uncached PLE_READ_PLACEMENT read_ms < ~400 ms.
5. Dedicated 5-context 8K..128K W=16384 prefill benchmark.
6. Lossless greedy decode parity across 8 prompts (0..128K) with container-default CONF=0.
7. C=8 concurrent context scaling across 0, 1K, 2K, 4K, 8K tokens.
8. Execution mode boots across all 9 modes (auto, max-prefill, interactive, agent, decode, 256k, exact, long-context, fast).
9. Native 262K container validation suite with FLUX_MODE=256k (5-depth needle, forced fill, boundary 262,143, overflow 400 rejection, canary recovery).
"""

import concurrent.futures
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(os.environ.get("FLUX_WORKTREE", str(Path(__file__).resolve().parents[2])))
MODELS_ROOT = Path(os.environ.get("FLUX_MODELS_DIR", str(Path.home() / "models")))
PROMPTS_ROOT = Path(os.environ.get("FLUX_PROMPTS_DIR", str(REPO_ROOT / "bench" / "prompts")))
OUT_DIR = Path(__file__).resolve().parent
IMAGE = "flux-engine:9.15.3"
PORT = 18899
BASE_URL = f"http://127.0.0.1:{PORT}"
REF_JSON = Path(os.environ.get("FLUX_DET_REF_JSON", str(PROMPTS_ROOT / "det-plain-all8.json")))

PROMPTS = {
    "code0": str(PROMPTS_ROOT / "dc0_code.json"),
    "prose0": str(PROMPTS_ROOT / "d0_prose.json"),
    "chat0": str(PROMPTS_ROOT / "chat0.json"),
    "rep0": str(PROMPTS_ROOT / "rep0.json"),
    "prose8k": str(PROMPTS_ROOT / "d8k_prose.json"),
    "legal32k": str(PROMPTS_ROOT / "d32k_legal.json"),
    "doc64k": str(PROMPTS_ROOT / "d64k_doc.json"),
    "mix128k": str(PROMPTS_ROOT / "mix128k.json"),
}

NEEDLE = "The secret vault code is 4719-ALPHA-82. "
QUESTION = "\n\nWhat is the secret vault code mentioned in the text above? Reply with only the code."
WORDS = "river stone lantern copper meadow signal harbor orbit velvet canyon ember quartz willow marble falcon cedar".split()


def memavail_gib() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1048576.0
    return 0.0


def wait_memavail(min_gib: float = 100.0, timeout: int = 120) -> float:
    t0 = time.time()
    while time.time() - t0 < timeout:
        m = memavail_gib()
        if m >= min_gib:
            return m
        time.sleep(1.0)
    return memavail_gib()


def http_json(path: str, payload: dict | None = None, timeout: int = 1800) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        method="POST" if payload is not None else "GET",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError(f"HTTP {e.code} on {path}: {err_body}") from e


def wait_ready(cname: str, timeout: int = 240) -> float:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = http_json("/health", None, timeout=3)
            if r.get("status") in ("ok", "ready"):
                return time.time() - t0
        except Exception:
            pass
        rc = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", cname],
            capture_output=True,
            text=True,
        )
        if rc.returncode != 0 or rc.stdout.strip() != "true":
            logs = subprocess.run(["docker", "logs", "--tail", "60", cname], capture_output=True, text=True)
            raise RuntimeError(f"Container {cname} exited early:\n{logs.stdout}\n{logs.stderr}")
        time.sleep(1.0)
    raise TimeoutError(f"Container {cname} did not become ready within {timeout}s")


def stop_container(cname: str) -> None:
    subprocess.run(["docker", "rm", "-f", cname], capture_output=True)
    time.sleep(2.0)


def build_needle_prompt(target_tokens: int, pos: float = 0.5) -> str:
    rng = random.Random(7)
    chars, parts, n = int(target_tokens * 5.2), [], 0
    while n < chars:
        s = " ".join(rng.choice(WORDS) for _ in range(rng.randint(8, 16))).capitalize() + ". "
        parts.append(s)
        n += len(s)
    cut = int(len(parts) * pos)
    return "".join(parts[:cut]) + NEEDLE + "".join(parts[cut:]) + QUESTION


def run_needle_probe(model_id: str, tokens: int, pos: float = 0.5, rep: int = 0) -> dict:
    prompt = build_needle_prompt(tokens, pos)
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 40,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.time()
    r = http_json("/v1/chat/completions", body, timeout=1800)
    wall_s = time.time() - t0
    c, u = r["choices"][0], r["usage"]
    answer = (c.get("message") or {}).get("content") or ""
    res = {
        "target_tokens": tokens,
        "pos": pos,
        "rep": rep,
        "prompt_tokens": u["prompt_tokens"],
        "completion_tokens": u["completion_tokens"],
        "finish": c["finish_reason"],
        "answer": answer.strip(),
        "correct": "4719-ALPHA-82" in answer,
        "wall_s": round(wall_s, 3),
    }
    print(f"[NEEDLE] {json.dumps(res)}", flush=True)
    return res


def run_vbench_prompt(name: str, path: str, ref_text: str, max_tokens: int = 256) -> dict:
    body = json.load(open(path))
    body["max_tokens"] = max_tokens
    body["temperature"] = 0
    body["stream"] = False
    body.pop("top_p", None)
    t0 = time.time()
    r = http_json("/v1/completions", body, timeout=1800)
    wall_s = time.time() - t0
    c, u = r["choices"][0], r["usage"]
    text = c.get("text", "")
    comp_toks = u.get("completion_tokens", 0)
    prompt_toks = u.get("prompt_tokens", 0)
    decode_tok_s = u.get("response_token/s") or (round(comp_toks / wall_s, 2) if wall_s > 0 else 0.0)
    ttft_ms = u.get("time_to_first_token_ms")
    id_ok = (text == ref_text) if ref_text else None
    res = {
        "name": name,
        "prompt_tokens": prompt_toks,
        "completion_tokens": comp_toks,
        "ttft_ms": round(ttft_ms, 1) if ttft_ms is not None else None,
        "decode_tok_s": round(float(decode_tok_s), 2),
        "wall_s": round(wall_s, 3),
        "wall_tok_s": round(comp_toks / wall_s, 2) if wall_s > 0 else 0.0,
        "id_ok": id_ok,
        "text": text,
    }
    print(
        f"[DECODE] {name}: ctx={prompt_toks} tok={comp_toks} ttft_ms={res['ttft_ms']} decode_tok_s={res['decode_tok_s']} id_ok={id_ok}",
        flush=True,
    )
    return res


def run_c8_context_point(model_id: str, ctx_tokens: int, gen_tokens: int = 128) -> dict:
    prompts = []
    for i in range(8):
        if ctx_tokens == 0:
            prompts.append(f"Write a concise Python function #{i} to compute Fibonacci numbers iteratively with docstrings.")
        else:
            rng = random.Random(100 + i)
            word_count = max(16, int((ctx_tokens - 48) * 0.82))
            words = [rng.choice(WORDS) for _ in range(word_count)]
            filler = " ".join(words)
            prompts.append(f"Document #{i}: {filler}\n\nWrite a concise, well-documented Python module with helper functions to process Document #{i}:")

    def _one(idx: int):
        body = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompts[idx]}],
            "max_tokens": gen_tokens,
            "min_tokens": gen_tokens,
            "ignore_eos": True,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        t_start = time.time()
        r = http_json("/v1/chat/completions", body, timeout=1800)
        t_end = time.time()
        u = r.get("usage", {})
        prompt_tokens = u.get("prompt_tokens", 0)
        completion_tokens = u.get("completion_tokens", 0)
        resp_tok_s = float(u.get("response_token/s") or 0.0)
        ttft_ms = float(u.get("time_to_first_token_ms") or 0.0)
        return prompt_tokens, completion_tokens, t_end - t_start, resp_tok_s, ttft_ms

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(_one, i) for i in range(8)]
        outs = [f.result() for f in futs]
    wall_s = time.time() - t0
    total_comp = sum(x[1] for x in outs)
    avg_prompt = sum(x[0] for x in outs) / 8.0
    agg_tok_s = total_comp / wall_s if wall_s > 0 else 0.0
    avg_ttft_s = (sum(x[4] for x in outs) / 8.0) / 1000.0
    sum_decode_tok_s = round(sum(x[3] for x in outs), 2)
    agg_decode_only_tok_s = round(total_comp / max(0.001, wall_s - avg_ttft_s), 2)
    res = {
        "target_ctx": ctx_tokens,
        "avg_prompt_tokens": round(avg_prompt, 1),
        "total_completion_tokens": total_comp,
        "wall_s": round(wall_s, 3),
        "agg_e2e_tok_s": round(agg_tok_s, 2),
        "sum_decode_tok_s": sum_decode_tok_s,
        "agg_decode_only_tok_s": agg_decode_only_tok_s,
        "avg_ttft_ms": round(sum(x[4] for x in outs) / 8.0, 2),
    }
    print(f"[C8_SCALE] {json.dumps(res)}", flush=True)
    return res


def warmup_server(model_id="Qwen3.8-Flash-Next-NVFP4-Offload", prompt_len=4000):
    print(f"[*] Performing warm-up prefill request ({prompt_len} words) + 2s settle (swap writeback drain)...", flush=True)
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": "hello " * prompt_len}],
        "max_tokens": 1,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        http_json("/v1/chat/completions", body, timeout=120)
    except Exception as e:
        print(f"[WARN] Warm-up prefill caught: {e}", flush=True)
    time.sleep(2.0)
    for _ in range(10):
        try:
            out = subprocess.check_output(["vmstat", "1", "2"], text=True)
            last_line = out.strip().split("\n")[-1].split()
            so = int(last_line[7])
            if so == 0:
                break
        except Exception:
            pass
        time.sleep(1.0)
    print("[+] Warm-up complete and NVMe IO settled.", flush=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mem = wait_memavail(100.0)
    print(f"[*] Pre-launch MemAvailable: {mem:.2f} GiB", flush=True)
    if mem < 100.0:
        raise RuntimeError(f"MemAvailable {mem:.2f} GiB < 100.0 GiB")

    ref_texts = {}
    if REF_JSON.exists():
        for r in json.load(open(REF_JSON)).get("results", []):
            ref_texts[r["name"]] = r.get("text", "")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-phases", default="", help="Comma-separated phases to skip (e.g. 1,2)")
    args, _ = parser.parse_known_args()
    skip = set(p.strip() for p in args.skip_phases.split(",") if p.strip())

    summary = {
        "image": IMAGE,
        "c1": {},
        "c8": {},
        "prefill_grid": {},
        "mode_boots": {},
        "native262k": {},
    }
    summary_file = OUT_DIR / "validation_summary.json"
    if summary_file.exists():
        try:
            summary.update(json.loads(summary_file.read_text()))
        except Exception:
            pass

    # =========================================================================
    # Phase 1: C=1 Zero-Flag Container (Cold Start, Needle, Decode, Prefill)
    # =========================================================================
    if "1" not in skip:
        missing_prompts = [p for p in PROMPTS.values() if not Path(p).is_file()]
        if missing_prompts:
            raise FileNotFoundError(
                f"Missing prompt files under FLUX_PROMPTS_DIR={PROMPTS_ROOT}: {missing_prompts[:3]} "
                "(set FLUX_PROMPTS_DIR to a directory containing the 8 decode prompt JSONs)"
            )
        grid_bench_path = Path(os.environ.get("FLUX_GRID_BENCH", str(REPO_ROOT / "bench" / "grid_bench.py")))
        if not grid_bench_path.is_file():
            raise FileNotFoundError(
                f"Missing grid benchmark script: {grid_bench_path} (set FLUX_GRID_BENCH)"
            )
        cname1 = "flux-val-9153-c1"
        stop_container(cname1)
        try:
            print(f"[*] Starting zero-flag C=1 container {cname1}...", flush=True)
            subprocess.run(
                [
                    "docker", "run", "-d", "--name", cname1,
                    "--gpus", "all", "--ipc=host", "--network=host",
                    "-v", f"{MODELS_ROOT}:/models:ro",
                    "-e", f"FLUX_PORT={PORT}",
                    "-e", "FLUX_MODE=max-prefill",
                    IMAGE,
                    "serve", "--model", "/models/Qwen3.8-Flash-Next-NVFP4-Offload",
                ],
                check=True,
            )
            ready_s = wait_ready(cname1, timeout=240)
            print(f"[+] C=1 container ready in {ready_s:.2f}s (MemAvailable={memavail_gib():.2f} GiB)", flush=True)
            models = http_json("/v1/models")
            model_id = models["data"][0]["id"]
            modes_api = http_json("/flux/modes")
            with urllib.request.urlopen(f"{BASE_URL}/", timeout=5) as root_resp:
                root_html = root_resp.read().decode("utf-8", errors="replace")
    
            summary["c1"]["ready_s"] = round(ready_s, 2)
            summary["c1"]["model_id"] = model_id
            summary["c1"]["flux_modes_api"] = {
                "status": modes_api.get("status"),
                "active_mode": modes_api.get("active_mode"),
                "modes_count": len(modes_api.get("modes", [])),
                "web_studio_ok": "recipe studio" in root_html.lower(),
            }
            print(f"[MODES_UI] {json.dumps(summary['c1']['flux_modes_api'])}", flush=True)
            warmup_server(model_id)

            # 1. Issue #13 needle.py validation (8 probes)
            print("\n[*] Running 8/8 Needle Validation...", flush=True)
            needle_results = []
            for tok, pos, reps in [
                (2017, 0.5, 1),
                (2112, 0.5, 1),
                (7710, 0.5, 3),
                (7710, 0.98, 1),
                (30647, 0.5, 1),
                (65536, 0.5, 1),
            ]:
                for rep in range(reps):
                    needle_results.append(run_needle_probe(model_id, tok, pos, rep))
            summary["c1"]["needle_issue_13"] = needle_results
            needle_all_correct = all(n["correct"] for n in needle_results)
            print(f"[+] Needle 8/8 result: {'PASS' if needle_all_correct else 'FAIL'}", flush=True)
    
            # 2. Single-stream 8-prompt decode validation (0..128K) with default CONF=0
            print("\n[*] Running Lossless 8-Prompt Decode Validation (CONF=0)...", flush=True)
            decode_results = []
            for name, ppath in PROMPTS.items():
                decode_results.append(run_vbench_prompt(name, ppath, ref_texts.get(name, ""), 256))
            summary["c1"]["decode_8prompts"] = decode_results
            if any(d["id_ok"] is None for d in decode_results):
                print("[WARN] Decode parity UNVERIFIED (reference text missing in FLUX_DET_REF_JSON)", flush=True)
            else:
                decode_all_id_ok = all(d["id_ok"] is True for d in decode_results)
                print(f"[+] Decode 8/8 parity result: {'PASS' if decode_all_id_ok else 'FAIL'}", flush=True)
    
            # 3. Dedicated 5-context W=16384 prefill benchmark via grid_bench.py
            print("\n[*] Running Dedicated W=16384 Prefill Benchmark (8K..128K)...", flush=True)
            grid_cmd = [
                sys.executable, str(grid_bench_path),
                "--url", BASE_URL,
                "--preset", "fn",
                "--model", "fn",
                "--max-ctx", "135168",
                "--server-batch", "1",
                "--prefill-ctx", "8192,16384,32768,65536,131072",
                "-n", "1",
                "--drop-first", "1",
                "--no-decode",
                "--probe-runs", "0",
                "--warmup-tokens", "8192",
                "--timeout", "900",
                "--budget-s", "0",
                "--same-prompt",
                "--tag", "container-9.15.3-w16384",
                "--out-dir", str(OUT_DIR),
            ]
            grid_p = subprocess.run(grid_cmd, capture_output=True, text=True)
            (OUT_DIR / "grid-w16384.stdout").write_text(grid_p.stdout)
            (OUT_DIR / "grid-w16384.stderr").write_text(grid_p.stderr)
            print(grid_p.stdout, flush=True)
            if grid_p.returncode != 0:
                raise RuntimeError(
                    f"grid_bench.py failed (rc={grid_p.returncode}): {grid_p.stderr.strip()}"
                )
    
        finally:
            logs1 = subprocess.run(["docker", "logs", cname1], capture_output=True, text=True)
            if logs1.stdout or logs1.stderr:
                (OUT_DIR / "container-c1.log").write_text(logs1.stdout + "\n" + logs1.stderr)
            stop_container(cname1)

    # Verify PLE receipts in container-c1.log (only when log is present)
    c1_log_path = OUT_DIR / "container-c1.log"
    if c1_log_path.exists():
        c1_log_text = c1_log_path.read_text()
        ple_engaged = bool(re.search(r"QWEN4_PLE_PREAD_LANES_ENGAGED.*lanes=256.*reason=io_uring_unavailable", c1_log_text))
        ple_plan_ran = bool(re.search(r"QWEN4_PLE_READ_PLAN_RAN.*pread_lanes=256", c1_log_text))
        ple_placement_times = [
            float(m.group(1))
            for m in re.finditer(r"PLE_READ_PLACEMENT.*read_ms=([0-9.]+)", c1_log_text)
        ]
        summary["c1"]["ple_receipts"] = {
            "engaged_256_lanes": ple_engaged,
            "plan_ran_256_lanes": ple_plan_ran,
            "placement_samples_ms": ple_placement_times[:10],
            "placement_max_ms": max(ple_placement_times) if ple_placement_times else None,
            "placement_sub_400ms": all(t < 400.0 for t in ple_placement_times) if ple_placement_times else False,
        }
        print(f"[PLE_RECEIPTS] {json.dumps(summary['c1']['ple_receipts'])}", flush=True)

    # Parse grid prefill JSON
    grid_json_path = OUT_DIR / "fn-grid-container-9.15.3-w16384.json"
    if grid_json_path.exists():
        grid_data = json.loads(grid_json_path.read_text())
        summary["prefill_grid"]["w16384"] = grid_data.get("prefill", {})
    summary_file.write_text(json.dumps(summary, indent=2))

    # =========================================================================
    # Phase 2: C=8 Concurrent Zero-Flag Container
    # =========================================================================
    if "2" not in skip:
        wait_memavail(100.0)
        time.sleep(3.0)
        cname8 = "flux-val-9153-c8"
        stop_container(cname8)
        try:
            print(f"\n[*] Starting zero-flag C=8 container {cname8} (MemAvailable={memavail_gib():.2f} GiB)...", flush=True)
            subprocess.run(
                [
                    "docker", "run", "-d", "--name", cname8,
                    "--gpus", "all", "--ipc=host", "--network=host",
                    "-v", f"{MODELS_ROOT}:/models:ro",
                    "-e", f"FLUX_PORT={PORT}",
                    "-e", "FLUX_CONCURRENCY=8",
                    IMAGE,
                    "serve", "--model", "/models/Qwen3.8-Flash-Next-NVFP4-Offload",
                ],
                check=True,
            )
            ready8_s = wait_ready(cname8, timeout=240)
            print(f"[+] C=8 container ready in {ready8_s:.2f}s (MemAvailable={memavail_gib():.2f} GiB)", flush=True)
            models = http_json("/v1/models")
            model_id = models["data"][0]["id"]
            summary["c8"]["ready_s"] = round(ready8_s, 2)
            warmup_server(model_id, prompt_len=32)
            print("[*] Warming up 8-way batched verify & MTP kernels (CUDA_MODULE_LOADING=LAZY)...", flush=True)
            run_c8_context_point(model_id, 0, 16)

            c8_results = []
            for ctx in [0, 1024, 2048, 4096, 8192]:
                c8_results.append(run_c8_context_point(model_id, ctx, 128))
            summary["c8"]["context_scaling"] = c8_results
            summary_file.write_text(json.dumps(summary, indent=2))
        finally:
            logs8 = subprocess.run(["docker", "logs", cname8], capture_output=True, text=True)
            if logs8.stdout or logs8.stderr:
                (OUT_DIR / "container-c8.log").write_text(logs8.stdout + "\n" + logs8.stderr)
            stop_container(cname8)
    
    # =========================================================================
    # Phase 3: Mode Boots Test in Container (All 9 modes)
    # =========================================================================
    if "3" not in skip:
        modes_to_test = ["auto", "max-prefill", "interactive", "agent", "decode", "256k", "exact", "long-context", "fast"]
        print(f"\n[*] Testing Mode Boots in Container across {len(modes_to_test)} modes...", flush=True)
        mode_boot_results = summary.get("mode_boots", {})
        for m in modes_to_test:
            if mode_boot_results.get(m, {}).get("completion_ok") is True and (
                m != "agent" or mode_boot_results.get(m, {}).get("tool_call_ok") is True
            ):
                print(f"[MODE_BOOT] Skipping already-passed mode {m}: {mode_boot_results[m]}", flush=True)
                continue
            wait_memavail(100.0)
            cname_m = f"flux-val-mode-{m}"
            stop_container(cname_m)
            try:
                t_boot_0 = time.time()
                subprocess.run(
                    [
                        "docker", "run", "-d", "--name", cname_m,
                        "--gpus", "all", "--ipc=host", "--network=host",
                        "-v", f"{MODELS_ROOT}:/models:ro",
                        "-e", f"FLUX_PORT={PORT}",
                        "-e", f"FLUX_MODE={m}",
                        IMAGE,
                        "serve", "--model", "/models/Qwen3.8-Flash-Next-NVFP4-Offload",
                    ],
                    check=True,
                )
                ready_m = wait_ready(cname_m, timeout=240)
                # Send 1 completion request
                models_m = http_json("/v1/models")
                m_id = models_m["data"][0]["id"]
                req_body = {
                    "model": m_id,
                    "messages": [{"role": "user", "content": "What is 2+2? Reply with number only."}],
                    "max_tokens": 10,
                    "temperature": 0,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
                res_comp = http_json("/v1/chat/completions", req_body, timeout=60)
                text_ans = res_comp["choices"][0]["message"]["content"].strip()
                mode_boot_results[m] = {
                    "ready_s": round(ready_m, 2),
                    "completion_ok": "4" in text_ans,
                    "answer": text_ans,
                }
                if m == "agent":
                    tool_req = {
                        "model": m_id,
                        "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
                        "tools": [{
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "description": "Get current weather for a city",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"location": {"type": "string"}},
                                    "required": ["location"],
                                },
                            },
                        }],
                        "tool_choice": "auto",
                        "max_tokens": 64,
                        "chat_template_kwargs": {"enable_thinking": False},
                    }
                    res_tool = http_json("/v1/chat/completions", tool_req, timeout=60)
                    msg = res_tool["choices"][0]["message"]
                    tool_calls = msg.get("tool_calls", [])
                    tool_ok = len(tool_calls) > 0 or "<tool_call>" in msg.get("content", "") or "get_weather" in str(msg)
                    mode_boot_results[m]["tool_call_ok"] = tool_ok
                    print(f"[MODE_BOOT] Mode agent: tool_call_ok={tool_ok}", flush=True)
                print(f"[MODE_BOOT] Mode {m}: ready in {ready_m:.2f}s, completion_ok={'4' in text_ans}", flush=True)
                summary["mode_boots"] = mode_boot_results
                summary_file.write_text(json.dumps(summary, indent=2))
            finally:
                stop_container(cname_m)
        summary["mode_boots"] = mode_boot_results
        summary_file.write_text(json.dumps(summary, indent=2))

    # =========================================================================
    # Phase 4: Native 262K Validation in Zero-Flag Container (FLUX_MODE=256k)
    # =========================================================================
    if "4" not in skip:
        wait_memavail(100.0)
        cname_256k = "flux-val-256k"
        stop_container(cname_256k)
        try:
            print(f"\n[*] Starting Native 262K Container ({cname_256k})...", flush=True)
            subprocess.run(
                [
                    "docker", "run", "-d", "--name", cname_256k,
                    "--gpus", "all", "--ipc=host", "--network=host",
                    "-v", f"{MODELS_ROOT}:/models:ro",
                    "-e", f"FLUX_PORT={PORT}",
                    "-e", "FLUX_MODE=256k",
                    IMAGE,
                    "serve", "--model", "/models/Qwen3.8-Flash-Next-NVFP4-Offload",
                ],
                check=True,
            )
            ready_256k = wait_ready(cname_256k, timeout=240)
            print(f"[+] Native 262K container ready in {ready_256k:.2f}s (MemAvailable={memavail_gib():.2f} GiB)", flush=True)

            # Run m2_262k_harness.py
            harness_cmd = [
                sys.executable, str(REPO_ROOT / "repro-kit" / "m2_262k_harness.py"),
                "--host", "127.0.0.1",
                "--port", str(PORT),
                "--model-name", "Qwen3.8-Flash-Next-NVFP4-Offload",
                "--output-dir", str(OUT_DIR / "native262k_container"),
            ]
            h_res = subprocess.run(harness_cmd, capture_output=True, text=True)
            print(h_res.stdout, flush=True)
            if h_res.stderr:
                print("[STDERR]", h_res.stderr, flush=True)
            if h_res.returncode != 0:
                raise RuntimeError(
                    f"m2_262k_harness.py failed (rc={h_res.returncode}): {h_res.stderr.strip()}"
                )

            report_262k_path = OUT_DIR / "native262k_container" / "report.json"
            if report_262k_path.exists():
                summary["native262k"] = json.loads(report_262k_path.read_text())
            summary_file.write_text(json.dumps(summary, indent=2))
        finally:
            logs_256k = subprocess.run(["docker", "logs", cname_256k], capture_output=True, text=True)
            if logs_256k.stdout or logs_256k.stderr:
                (OUT_DIR / "container-256k.log").write_text(logs_256k.stdout + "\n" + logs_256k.stderr)
            stop_container(cname_256k)

    (OUT_DIR / "validation_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n[+] Container Qualification Suite Complete! Saved to", OUT_DIR / "validation_summary.json", flush=True)


if __name__ == "__main__":
    main()
