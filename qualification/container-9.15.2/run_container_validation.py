#!/usr/bin/env python3
"""Live Zero-Flag Container Validation for ghcr.io/theapathy/flux-engine:9.15.2.

Validates inside a zero-flag Docker container (`docker run --rm --gpus all --ipc=host --network=host`):
1. Issue #13 exact needle probe across 2,017, 2,112, 7,710 (pos=0.5 x3 + pos=0.98), 30,647, and 65,536 tokens
   (C=1 against `Qwen3.8-Flash-Next-NVFP4-Offload` with `ATLAS_QWEN4_MTP_CONF=0.6`).
2. Single-sequence (C=1) 8-prompt decode (`code0`, `prose0`, `chat0`, `rep0`, `prose8k`, `legal32k`, `doc64k`, `mix128k`)
   with greedy comparison against `FLUX_DET_REF_JSON` (`det-plain-all8.json`, when present; `id_ok=None` when unreferenced).
3. Concurrent C=8 context scaling across 0, 1K, 2K, 4K, 8K tokens (`FLUX_CONCURRENCY=8`, configurable model via `FLUX_C8_MODEL`).
"""
import concurrent.futures
import json
import os
import random
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(os.environ.get("FLUX_WORKTREE", str(Path(__file__).resolve().parents[2])))
MODELS_ROOT = Path(os.environ.get("FLUX_MODELS_DIR", str(Path.home() / "models")))
PROMPTS_ROOT = Path(os.environ.get("FLUX_PROMPTS_DIR", str(REPO_ROOT / "bench" / "prompts")))
OUT_DIR = Path(__file__).resolve().parent
IMAGE = "ghcr.io/theapathy/flux-engine:9.15.2"
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


def wait_ready(cname: str, timeout: int = 420) -> float:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = http_json("/health", None, timeout=5)
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
        time.sleep(1.5)
    raise TimeoutError(f"Container {cname} did not become ready within {timeout}s")


def stop_container(cname: str) -> None:
    subprocess.run(["docker", "rm", "-f", cname], capture_output=True)
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
    }
    print(
        f"[DECODE] {name}: ctx={prompt_toks} tok={comp_toks} ttft_ms={res['ttft_ms']} decode_tok_s={res['decode_tok_s']} id_ok={id_ok}",
        flush=True,
    )
    return res


def run_c8_context_point(model_id: str, ctx_tokens: int, gen_tokens: int = 128) -> dict:
    # Build 8 distinct prompts of ~ctx_tokens tokens
    prompts = []
    for i in range(8):
        if ctx_tokens == 0:
            prompts.append(f"Write a concise Python function #{i} to compute Fibonacci numbers iteratively with docstrings.")
        else:
            rng = random.Random(100 + i)
            words = [rng.choice(WORDS) for _ in range(max(16, ctx_tokens - 24))]
            filler = " ".join(words)
            prompts.append(f"Document #{i}: {filler}\n\nSummarize the key sequence and write a 10-line Python helper for Document #{i}:")

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
        u = r["usage"]
        return u["prompt_tokens"], u["completion_tokens"], t_end - t_start

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(_one, i) for i in range(8)]
        outs = [f.result() for f in futs]
    wall_s = time.time() - t0
    total_comp = sum(x[1] for x in outs)
    avg_prompt = sum(x[0] for x in outs) / 8.0
    agg_tok_s = total_comp / wall_s if wall_s > 0 else 0.0
    res = {
        "target_ctx": ctx_tokens,
        "avg_prompt_tokens": round(avg_prompt, 1),
        "total_completion_tokens": total_comp,
        "wall_s": round(wall_s, 3),
        "agg_e2e_tok_s": round(agg_tok_s, 2),
    }
    print(f"[C8_SCALE] {json.dumps(res)}", flush=True)
    return res


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    missing_prompts = [p for p in PROMPTS.values() if not Path(p).is_file()]
    if missing_prompts:
        raise FileNotFoundError(
            f"Missing prompt files under FLUX_PROMPTS_DIR={PROMPTS_ROOT}: {missing_prompts[:3]} "
            "(set FLUX_PROMPTS_DIR to a directory containing the 8 decode prompt JSONs)"
        )
    mem = memavail_gib()
    print(f"[*] Pre-launch MemAvailable: {mem:.2f} GiB", flush=True)
    if mem < 100.0:
        raise RuntimeError(f"MemAvailable {mem:.2f} GiB < 100.0 GiB")

    ref_texts = {}
    if REF_JSON.exists():
        for r in json.load(open(REF_JSON)).get("results", []):
            ref_texts[r["name"]] = r.get("text", "")

    summary = {"image": IMAGE, "c1": {}, "c8": {}}
    cname1 = "flux-val-9152-c1"
    stop_container(cname1)

    try:
        print(f"[*] Starting zero-flag C=1 container {cname1}...", flush=True)
        subprocess.run(
            [
                "docker", "run", "-d", "--name", cname1,
                "--gpus", "all", "--ipc=host", "--network=host",
                "-v", f"{MODELS_ROOT}:/models:ro",
                "-e", f"FLUX_PORT={PORT}",
                "-e", "ATLAS_QWEN4_MTP_CONF=0.6",
                "-e", "ATLAS_QWEN4_MTP_CONF_EARLY=1",
                "-e", "ATLAS_QWEN4_MTP_CONF_STEPS=0.60,0.65,0.65,0.75",
                "-e", "ATLAS_QWEN4_MTP_CONF_CUM=0.50",
                IMAGE,
                "serve", "--model", "/models/Qwen3.8-Flash-Next-NVFP4-Offload",
            ],
            check=True,
        )
        ready_s = wait_ready(cname1)
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
            "web_studio_ok": "Self-Guiding Recipe Studio" in root_html,
        }
        print(f"[MODES_UI] {json.dumps(summary['c1']['flux_modes_api'])}", flush=True)

        # 1. Issue #13 needle.py validation
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

        # 2. Single-stream 8-prompt decode validation (0..128K)
        decode_results = []
        for name, ppath in PROMPTS.items():
            decode_results.append(run_vbench_prompt(name, ppath, ref_texts.get(name, ""), 256))
        summary["c1"]["decode_8prompts"] = decode_results
    finally:
        logs1 = subprocess.run(["docker", "logs", cname1], capture_output=True, text=True)
        if logs1.stdout or logs1.stderr:
            (OUT_DIR / "container-c1.log").write_text(logs1.stdout + "\n" + logs1.stderr)
        stop_container(cname1)

    # Wait for UMA memory to reclaim above 100 GiB before C=8 bring-up
    for _ in range(60):
        if memavail_gib() >= 100.0:
            break
        time.sleep(2.0)

    cname8 = "flux-val-9152-c8"
    stop_container(cname8)
    try:
        print(f"[*] Starting zero-flag C=8 container {cname8} (MemAvailable={memavail_gib():.2f} GiB)...", flush=True)
        subprocess.run(
            [
                "docker", "run", "-d", "--name", cname8,
                "--gpus", "all", "--ipc=host", "--network=host",
                "-v", f"{MODELS_ROOT}:/models:ro",
                "-e", f"FLUX_PORT={PORT}",
                "-e", "FLUX_CONCURRENCY=8",
                IMAGE,
                "serve", "--model", os.environ.get("FLUX_C8_MODEL", "/models/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid"),
            ],
            check=True,
        )
        ready8_s = wait_ready(cname8)
        print(f"[+] C=8 container ready in {ready8_s:.2f}s (MemAvailable={memavail_gib():.2f} GiB)", flush=True)
        models = http_json("/v1/models")
        model_id = models["data"][0]["id"]
        summary["c8"]["ready_s"] = round(ready8_s, 2)

        c8_results = []
        for ctx in [0, 1024, 2048, 4096, 8192]:
            c8_results.append(run_c8_context_point(model_id, ctx, 128))
        summary["c8"]["context_scaling"] = c8_results
    finally:
        logs8 = subprocess.run(["docker", "logs", cname8], capture_output=True, text=True)
        if logs8.stdout or logs8.stderr:
            (OUT_DIR / "container-c8.log").write_text(logs8.stdout + "\n" + logs8.stderr)
        stop_container(cname8)

    (OUT_DIR / "validation_summary.json").write_text(json.dumps(summary, indent=2))
    print("[+] Validation complete! Saved to", OUT_DIR / "validation_summary.json", flush=True)


if __name__ == "__main__":
    main()
