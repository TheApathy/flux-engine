#!/usr/bin/env python3
"""Host Baseline Qualification Suite for flux-engine:9.15.3.

Executes side-by-side host benchmarks to verify Host-vs-Container Parity:
1. C=1 Host Baseline with flux-v9-max-prefill.env:
   - 8/8 needle probe
   - 8 decode prompts with container-default MTP (greedy outputs saved for diffing against container)
   - 5-context W=16384 prefill benchmark via grid_bench.py
2. W=4096 Interactive Host Baseline with flux-v9-w4096-interactive.env:
   - 8K and 128K prefill benchmark via grid_bench.py
3. C=8 Concurrency Host Baseline with flux-qwen4-native-concurrency-c8.env:
   - 0, 1K, 2K, 4K, 8K context scaling
4. Native 262K Host Suite with flux-qwen4-native-262k-bf16.env:
   - Full m2_262k_harness.py battery (5 passkeys @ 262,080, forced fill, boundary, overflow 400 rejection, canary recovery)
5. Writes `host_validation_summary.json` for side-by-side comparison against `validation_summary.json`.
"""

import concurrent.futures
import json
import os
import random
import re
import signal
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
BIN_PATH = Path(os.environ.get("FLUX_SERVER_BIN", str(REPO_ROOT / "target" / "release" / "flux-server")))
RECIPES_DIR = REPO_ROOT / "repro-kit" / "recipes"
MODEL_OFFLOAD = MODELS_ROOT / "Qwen3.8-Flash-Next-NVFP4-Offload"
MODEL_HYBRID = MODELS_ROOT / "Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid"
PORT = 18899
BASE_URL = f"http://127.0.0.1:{PORT}"

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


def load_env_file(path: Path) -> dict:
    env = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


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


def wait_ready(proc: subprocess.Popen, timeout: int = 240) -> float:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"Server exited prematurely with code {proc.returncode}")
        try:
            r = http_json("/health", None, timeout=3)
            if r.get("status") in ("ok", "ready"):
                return time.time() - t0
        except Exception:
            pass
        time.sleep(1.0)
    raise TimeoutError(f"Server did not become ready within {timeout}s")


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                proc.kill()
            proc.wait()
    time.sleep(3.0)


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
    print(f"[HOST_NEEDLE] {json.dumps(res)}", flush=True)
    return res


def run_decode_prompt(name: str, path: str, max_tokens: int = 256) -> dict:
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
    res = {
        "name": name,
        "prompt_tokens": prompt_toks,
        "completion_tokens": comp_toks,
        "ttft_ms": round(ttft_ms, 1) if ttft_ms is not None else None,
        "decode_tok_s": round(float(decode_tok_s), 2),
        "wall_s": round(wall_s, 3),
        "text": text,
    }
    print(
        f"[HOST_DECODE] {name}: ctx={prompt_toks} tok={comp_toks} ttft_ms={res['ttft_ms']} decode_tok_s={res['decode_tok_s']}",
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
    print(f"[HOST_C8_SCALE] {json.dumps(res)}", flush=True)
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
        print(f"[WARN] Host warm-up prefill caught: {e}", flush=True)
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
    print("[+] Host warm-up complete and NVMe IO settled.", flush=True)


def build_host_launch(mode: str, concurrency: int = 1) -> tuple[list[str], dict[str, str]]:
    autodetect = REPO_ROOT / "repro-kit" / "flux-autodetect"
    env = os.environ.copy()
    env["FLUX_PORT"] = str(PORT)
    env["FLUX_HOST"] = "0.0.0.0"
    env["FLUX_MODE"] = mode
    env["FLUX_CONCURRENCY"] = str(concurrency)
    env["FLUX_SERVER_BIN"] = str(BIN_PATH)
    out = subprocess.check_output(
        [sys.executable, str(autodetect), "serve", "--model", str(MODEL_OFFLOAD), "--dry-run"],
        env=env,
        text=True,
    )
    plan = json.loads(out)["launch_plan"]
    launch_env = os.environ.copy()
    launch_env.update(plan["env"])
    return plan["command"], launch_env


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-phases", default="", help="Comma-separated phases to skip: 1,1b,2,3")
    args = parser.parse_args()
    skip = set(x.strip() for x in args.skip_phases.split(",") if x.strip())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not BIN_PATH.is_file():
        raise FileNotFoundError(
            f"Server binary not found at {BIN_PATH} (build target/release/flux-server or set FLUX_SERVER_BIN)"
        )
    grid_bench_path = Path(os.environ.get("FLUX_GRID_BENCH", str(REPO_ROOT / "bench" / "grid_bench.py")))
    if "1" not in skip:
        missing_prompts = [p for p in PROMPTS.values() if not Path(p).is_file()]
        if missing_prompts:
            raise FileNotFoundError(
                f"Missing prompt files under FLUX_PROMPTS_DIR={PROMPTS_ROOT}: {missing_prompts[:3]} "
                "(set FLUX_PROMPTS_DIR to a directory containing the 8 decode prompt JSONs)"
            )
    if ("1" not in skip or "1b" not in skip) and not grid_bench_path.is_file():
        raise FileNotFoundError(
            f"Missing grid benchmark script: {grid_bench_path} (set FLUX_GRID_BENCH)"
        )
    summary_path = OUT_DIR / "host_validation_summary.json"
    mem = wait_memavail(100.0)
    print(f"[*] Pre-launch MemAvailable on host: {mem:.2f} GiB", flush=True)

    import hashlib
    b_sha = hashlib.sha256(open(BIN_PATH, "rb").read()).hexdigest()
    host_summary = {
        "binary_sha256": b_sha,
        "c1": {},
        "interactive_w4096": {},
        "c8": {},
        "prefill_grid": {},
        "native262k": {},
    }
    if summary_path.exists():
        try:
            existing = json.loads(summary_path.read_text())
            host_summary.update(existing)
            host_summary["binary_sha256"] = b_sha
        except Exception:
            pass

    # =========================================================================
    # Phase 1: Host C=1 Baseline (flux-v9-max-prefill.env)
    # =========================================================================
    if "1" not in skip:
        cmd1, env1 = build_host_launch("auto", concurrency=1)
        log1_file = open(OUT_DIR / "host-c1.log", "w")
        print("\n[*] Starting Host C=1 Server (flux-v9-max-prefill)...", flush=True)
        p1 = subprocess.Popen(cmd1, env=env1, stdout=log1_file, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
        try:
            ready1_s = wait_ready(p1, timeout=240)
            print(f"[+] Host C=1 ready in {ready1_s:.2f}s (MemAvailable={memavail_gib():.2f} GiB)", flush=True)
            models = http_json("/v1/models")
            model_id = models["data"][0]["id"]
            host_summary["c1"]["ready_s"] = round(ready1_s, 2)
            host_summary["c1"]["model_id"] = model_id
            warmup_server(model_id)

            # 1. Needle 8/8
            print("\n[*] Running Host 8/8 Needle Validation...", flush=True)
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
            host_summary["c1"]["needle_issue_13"] = needle_results

            # 2. Decode 8 prompts
            print("\n[*] Running Host Lossless 8-Prompt Decode (CONF=0)...", flush=True)
            decode_results = []
            for name, ppath in PROMPTS.items():
                decode_results.append(run_decode_prompt(name, ppath, 256))
            host_summary["c1"]["decode_8prompts"] = decode_results

            # 3. Grid prefill benchmark W=16384 (8K..128K)
            print("\n[*] Running Host W=16384 Prefill Benchmark (8K..128K)...", flush=True)
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
                "--tag", "host-9.15.3-w16384",
                "--out-dir", str(OUT_DIR),
            ]
            grid_p = subprocess.run(grid_cmd, capture_output=True, text=True)
            (OUT_DIR / "grid-host-w16384.stdout").write_text(grid_p.stdout)
            (OUT_DIR / "grid-host-w16384.stderr").write_text(grid_p.stderr)
            print(grid_p.stdout, flush=True)
            if grid_p.returncode != 0:
                raise RuntimeError(
                    f"Host W=16384 grid_bench.py failed (rc={grid_p.returncode}): {grid_p.stderr.strip()}"
                )

            grid_json = OUT_DIR / "fn-grid-host-9.15.3-w16384.json"
            if grid_json.exists():
                host_summary["prefill_grid"]["w16384"] = json.loads(grid_json.read_text()).get("prefill", {})
            summary_path.write_text(json.dumps(host_summary, indent=2))
        finally:
            stop_server(p1)
            log1_file.close()

    # =========================================================================
    # Phase 1b: Host W=4096 Interactive Prefill (8K, 128K)
    # =========================================================================
    if "1b" not in skip:
        wait_memavail(100.0)
        cmd_inter, env_inter = build_host_launch("interactive", concurrency=1)
        log_inter = open(OUT_DIR / "host-interactive.log", "w")
        print("\n[*] Starting Host W=4096 Interactive Server...", flush=True)
        p_inter = subprocess.Popen(cmd_inter, env=env_inter, stdout=log_inter, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
        try:
            ready_inter_s = wait_ready(p_inter, timeout=240)
            print(f"[+] Host Interactive ready in {ready_inter_s:.2f}s", flush=True)
            warmup_server("Qwen3.8-Flash-Next-NVFP4-Offload")

            grid_inter_cmd = [
                sys.executable, str(grid_bench_path),
                "--url", BASE_URL,
                "--preset", "fn",
                "--model", "fn",
                "--max-ctx", "135168",
                "--server-batch", "1",
                "--prefill-ctx", "8192,131072",
                "-n", "1",
                "--drop-first", "1",
                "--no-decode",
                "--probe-runs", "0",
                "--warmup-tokens", "8192",
                "--timeout", "900",
                "--budget-s", "0",
                "--tag", "host-9.15.3-w4096",
                "--out-dir", str(OUT_DIR),
            ]
            grid_inter_p = subprocess.run(grid_inter_cmd, capture_output=True, text=True)
            (OUT_DIR / "grid-host-w4096.stdout").write_text(grid_inter_p.stdout)
            print(grid_inter_p.stdout, flush=True)
            if grid_inter_p.returncode != 0:
                raise RuntimeError(
                    f"Host W=4096 grid_bench.py failed (rc={grid_inter_p.returncode}): {grid_inter_p.stderr.strip()}"
                )

            grid_inter_json = OUT_DIR / "fn-grid-host-9.15.3-w4096.json"
            if grid_inter_json.exists():
                host_summary["prefill_grid"]["w4096_interactive"] = json.loads(grid_inter_json.read_text()).get("prefill", {})
            summary_path.write_text(json.dumps(host_summary, indent=2))
        finally:
            stop_server(p_inter)
            log_inter.close()

    # =========================================================================
    # Phase 2: Host C=8 Baseline
    # =========================================================================
    if "2" not in skip:
        mem8_pre = wait_memavail(101.5, timeout=30)
        cmd8, env8 = build_host_launch("auto", concurrency=8)
        log8_file = open(OUT_DIR / "host-c8.log", "w")
        print(f"\n[*] Starting Host C=8 Server (MemAvailable={mem8_pre:.2f} GiB)...", flush=True)
        p8 = subprocess.Popen(cmd8, env=env8, stdout=log8_file, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
        try:
            ready8_s = wait_ready(p8, timeout=240)
            mem8_ready = memavail_gib()
            print(f"[+] Host C=8 ready in {ready8_s:.2f}s (MemAvailable={mem8_ready:.2f} GiB)", flush=True)
            models8 = http_json("/v1/models")
            model8_id = models8["data"][0]["id"]
            host_summary["c8"]["ready_s"] = round(ready8_s, 2)
            host_summary["c8"]["ready_memavail_gib"] = round(mem8_ready, 2)
            warmup_server(model8_id, prompt_len=32)
            print("[*] Warming up 8-way batched verify & MTP kernels (CUDA_MODULE_LOADING=LAZY)...", flush=True)
            run_c8_context_point(model8_id, 0, 16)

            c8_results = []
            for ctx in [0, 1024, 2048, 4096, 8192]:
                c8_results.append(run_c8_context_point(model8_id, ctx, 128))
            host_summary["c8"]["context_scaling"] = c8_results
            summary_path.write_text(json.dumps(host_summary, indent=2))
        finally:
            stop_server(p8)
            log8_file.close()

    # =========================================================================
    # Phase 3: Host Native 262K Battery (FLUX_MODE=256k)
    # =========================================================================
    if "3" not in skip:
        mem262_pre = wait_memavail(101.5, timeout=30)
        cmd_256k, env_256k = build_host_launch("256k", concurrency=1)
        log_256k = open(OUT_DIR / "host-256k.log", "w")
        print(f"\n[*] Starting Host Native 262K Server (MemAvailable={mem262_pre:.2f} GiB)...", flush=True)
        p_256k = subprocess.Popen(cmd_256k, env=env_256k, stdout=log_256k, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
        try:
            ready_256k_s = wait_ready(p_256k, timeout=240)
            mem262_ready = memavail_gib()
            print(f"[+] Host 262K ready in {ready_256k_s:.2f}s (MemAvailable={mem262_ready:.2f} GiB)", flush=True)

            harness_cmd = [
                sys.executable, str(REPO_ROOT / "repro-kit" / "m2_262k_harness.py"),
                "--host", "127.0.0.1",
                "--port", str(PORT),
                "--model-name", "Qwen3.8-Flash-Next-NVFP4-Offload",
                "--output-dir", str(OUT_DIR / "native262k_host"),
            ]
            h_res = subprocess.run(harness_cmd, capture_output=True, text=True)
            print(h_res.stdout, flush=True)
            if h_res.returncode != 0:
                raise RuntimeError(
                    f"Host m2_262k_harness.py failed (rc={h_res.returncode}): {h_res.stderr.strip()}"
                )

            rep_262k = OUT_DIR / "native262k_host" / "report.json"
            if rep_262k.exists():
                host_summary["native262k"] = json.loads(rep_262k.read_text())
            summary_path.write_text(json.dumps(host_summary, indent=2))
        finally:
            stop_server(p_256k)
            log_256k.close()

    summary_path.write_text(json.dumps(host_summary, indent=2))
    print("\n[+] Host Baseline Complete! Saved to", summary_path, flush=True)


if __name__ == "__main__":
    main()
