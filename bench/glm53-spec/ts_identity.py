#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""GLM-5.3 DFlash2 P2: end-to-end target (T) vs speculative (S) identity on flux-server.

Arms run T, S, T, S, T, S, T by default, each with a fresh server. Every arm runs every prompt for
TRIALS trials (greedy, token-ID prompts from req/). The last T arm and the last S arm also run the
perturbed-prompt control.

Each arm's env and argv come from the SHIPPING built-in recipes, not a copied env list:
  T = crates/spark-server/src/recipe/builtin/glm-5.3-flash-exl3-local.yaml
  S = crates/spark-server/src/recipe/builtin/glm-5.3-flash-exl3-dflash2-local.yaml
S arms add ATLAS_GLM53_PHASE_TIMING=1 and, until P3 admits speculation, the explicit escape
ATLAS_GLM53_UNVALIDATED_BRINGUP=1 (omit with --admitted).

Gate (fail closed; status.txt is TS_PASS or TS_FAIL <reasons>; exit code 0 only on TS_PASS):
  * every response parses, finishes with a reason, and has completion tokens > 0;
  * T arms agree with each other (determinism), per prompt and trial;
  * every S trial's text is byte-identical to T's for the same prompt;
  * the perturbed control DIFFERS from the short prompt in both T and S (the gate can fail);
  * every arm's server binary sha256 is the same file.

  ts_identity.py --server /opt/flux/bin/flux-server-glm53 --model /models/GLM-5.3-Flash-exl3-2.05bpw \\
      --draft-model /models/GLM-5.3-Flash-DFlash2-bf582e4 --out /results/glm53-p2
  ts_identity.py ... --dry-run        # print each arm's env and argv, start nothing
  ts_identity.py --score-only DIR     # re-score an existing run directory
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RECIPES = REPO / "crates/spark-server/src/recipe/builtin"
T_RECIPE = RECIPES / "glm-5.3-flash-exl3-local.yaml"
S_RECIPE = RECIPES / "glm-5.3-flash-exl3-dflash2-local.yaml"
PROMPTS = ("short", "long", "prose", "code2", "think")
CONTROL = "ctlperturb"
DEFAULT_ARMS = "T1,S1,T2,S2,T3,S3,T4"
# recipe `defaults` key -> serve flag; booleans become bare flags
FLAG = {
    "max_model_len": "--max-seq-len", "max_prefill_tokens": "--max-prefill-tokens",
    "max_batch_size": "--max-batch-size", "max_num_seqs": "--max-num-seqs", "oom_guard_mb": "--oom-guard-mb",
    "dflash": "--dflash", "draft_model": "--draft-model", "dflash_gamma": "--dflash-gamma",
    "dflash_window_size": "--dflash-window-size", "glm_dflash_max_drafts": "--glm-dflash-max-drafts",
}
SKIP_DEFAULTS = {"port", "host"}


def load_recipe(path: Path) -> dict:
    import yaml
    r = yaml.safe_load(path.read_text())
    unknown = set(r.get("defaults", {})) - set(FLAG) - SKIP_DEFAULTS
    if unknown:
        raise SystemExit(f"ts_identity: recipe {path.name} has defaults this script cannot map: {sorted(unknown)}")
    return r


def arm_plan(kind: str, a: argparse.Namespace, port: int) -> tuple[dict, list[str]]:
    r = load_recipe(T_RECIPE if kind == "T" else S_RECIPE)
    env = {k: str(v) for k, v in (r.get("atlas", {}).get("env") or {}).items()}
    argv = ["serve", "--model-from-path", a.model, "--model-name", "glm53-flash",
            "--bind", "127.0.0.1", "--port", str(port), "--enable-prefix-caching", "false",
            "--no-tui", "--disable-thinking"]
    for k, v in (r.get("defaults") or {}).items():
        if k in SKIP_DEFAULTS:
            continue
        if k == "draft_model":
            v = a.draft_model or v
        if isinstance(v, bool):
            if v:
                argv.append(FLAG[k])
        else:
            argv += [FLAG[k], str(v)]
    if kind == "S":
        if "--dflash" not in argv:
            raise SystemExit("ts_identity: the S recipe no longer enables --dflash")
        env["ATLAS_GLM53_PHASE_TIMING"] = "1"
        if not a.admitted:
            env["ATLAS_GLM53_UNVALIDATED_BRINGUP"] = "1"
    elif "--dflash" in argv:
        raise SystemExit("ts_identity: the T recipe enables --dflash")
    return env, argv


def _rust_manifest() -> tuple[str, ...]:
    """SPEC_SOURCES, parsed from the single declaration in speculative_admission.rs."""
    import re
    src = (REPO / "crates/spark-model/src/model/glm53/speculative_admission.rs").read_text()
    body = src[src.index("spec_sources!(\n"):]
    body = body[:body.index(");")]
    return tuple(re.findall(r'"([^"]+)"', body))


SPEC_SOURCES = _rust_manifest()


def spec_source_fnv1a64(repo: Path = REPO) -> str:
    h = 0xCBF29CE484222325
    for path in SPEC_SOURCES:
        for part in (path.encode(), b"\0", (repo / path).read_bytes(), b"\0"):
            for byte in part:
                h ^= byte
                h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def logged_spec_fnv(log: Path) -> str | None:
    """The source hash the running binary reports when it installs DFlash2."""
    import re
    try:
        m = re.findall(rb"GLM53_SPEC_SOURCES fnv1a64=([0-9a-f]{16})", log.read_bytes())
    except OSError:
        return None
    return m[-1].decode() if m else None


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mem_available_gb() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1024 / 1024
    return 0.0


def http_json(url: str, body: dict | None = None, timeout: float = 600):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data, {"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def run_arm(name: str, a: argparse.Namespace, out: Path, prompts: list[str], plan=None) -> dict:
    kind = name[0]
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    env, argv = plan if plan is not None else arm_plan(kind, a, a.port)
    (d / "env.effective").write_text("".join(f"{k}={v}\n" for k, v in sorted(env.items())))
    (d / "argv.effective").write_text("\n".join(argv) + "\n")
    rec = {"arm": name, "kind": kind, "server_sha256": sha256(a.server), "responses": {}, "errors": []}
    if mem_available_gb() < a.min_mem_gb:
        rec["errors"].append(f"preflight MemAvailable {mem_available_gb():.0f} GB < {a.min_mem_gb}")
        return rec
    full_env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": os.environ.get("HOME", "/root"), **env}
    for k in ("LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", *a.pass_env):
        if k in os.environ:
            full_env[k] = os.environ[k]
    log = open(d / "server.log", "wb")
    srv = subprocess.Popen([a.server, *argv], env=full_env, stdout=log, stderr=subprocess.STDOUT,
                           start_new_session=True)
    base = f"http://127.0.0.1:{a.port}"
    low = 1e9
    try:
        deadline = time.time() + a.load_timeout
        while True:
            low = min(low, mem_available_gb())
            if low < a.watchdog_gb:
                rec["errors"].append(f"memory watchdog: MemAvailable {low:.1f} GB")
                return rec
            if srv.poll() is not None:
                rec["errors"].append(f"server exited rc={srv.returncode} before ready")
                return rec
            try:
                if http_json(base + "/health", timeout=5).get("status") == "ready":
                    break
            except Exception:
                pass
            if time.time() > deadline:
                rec["errors"].append("server not ready before load timeout")
                return rec
            time.sleep(1)
        warm = http_json(base + "/v1/completions", json.loads((a.req_dir / "request-short.json").read_text()))
        # recorded so state traces can be bound to every request, including the warm-up
        rec["warmup"] = {"completion_tokens": (warm.get("usage") or {}).get("completion_tokens"),
                         "finish": ((warm.get("choices") or [{}])[0]).get("finish_reason")}
        jobs = [(t, p) for t in range(1, a.trials + 1) for p in prompts]
        if name in a.control_arms:
            jobs.append((1, CONTROL))
        for t, p in jobs:
            body = json.loads((a.req_dir / f"request-{p}.json").read_text())
            key = f"trial-{t}-{p}"
            t0 = time.perf_counter()
            try:
                r = http_json(base + "/v1/completions", body)
            except Exception as e:
                rec["errors"].append(f"{key}: {type(e).__name__}: {e}")
                continue
            wall = time.perf_counter() - t0
            low = min(low, mem_available_gb())
            (d / f"{key}.response.json").write_text(json.dumps(r))
            ch = (r.get("choices") or [{}])[0]
            u = r.get("usage") or {}
            ct = u.get("completion_tokens") or 0
            ttft = (u.get("time_to_first_token_ms") or 0) / 1000
            rec["responses"][key] = {
                "sha": hashlib.sha256((ch.get("text") or "").encode()).hexdigest(),
                "completion_tokens": ct, "finish": ch.get("finish_reason"),
                "decode_tok_s": (ct - 1) / (wall - ttft) if ct > 1 and wall > ttft else 0.0}
    finally:
        if srv.poll() is None:
            os.killpg(srv.pid, signal.SIGINT)
            try:
                srv.wait(60)
            except subprocess.TimeoutExpired:
                os.killpg(srv.pid, signal.SIGKILL)
                srv.wait()
        log.close()
        rec["mem_available_min_gb"] = round(low, 1)
        rec["spec_source_fnv1a64"] = logged_spec_fnv(d / "server.log") if kind == "S" else None
        (d / "arm.json").write_text(json.dumps(rec, indent=1))
    return rec


def score(recs: list[dict], prompts: list[str]) -> tuple[list[str], dict]:
    fails: list[str] = []
    for r in recs:
        fails += [f"{r['arm']}: {e}" for e in r["errors"]]
        for k, v in r["responses"].items():
            if not v["finish"] or v["completion_tokens"] <= 0:
                fails.append(f"{r['arm']} {k}: empty or unfinished response")
    if len({r["server_sha256"] for r in recs}) > 1:
        fails.append("server binary changed between arms")
    ts = [r for r in recs if r["kind"] == "T"]
    ss = [r for r in recs if r["kind"] == "S"]
    if not ts or not ss:
        fails.append("need at least one T and one S arm")
        return fails, {}
    ref = ts[0]["responses"]
    keys = sorted(k for k in ref if not k.endswith(CONTROL))
    for p in prompts:
        if not any(k.endswith("-" + p) for k in keys):
            fails.append(f"T reference missing prompt {p}")
    for r in recs:
        if r is ts[0]:
            continue
        for k in keys:
            got = r["responses"].get(k)
            if got is None:
                fails.append(f"{r['arm']} {k}: missing")
            elif got["sha"] != ref[k]["sha"]:
                fails.append(f"{r['arm']} {k}: {'S' if r['kind'] == 'S' else 'T'} output differs from T1")
    controls = [(r["arm"], r["responses"].get(f"trial-1-{CONTROL}"), r["responses"].get("trial-1-short"))
                for r in recs if any(k.endswith(CONTROL) for k in r["responses"])]
    if not controls:
        fails.append("no perturbed-control response: the gate cannot be shown able to fail")
    for arm, c, s in controls:
        if c and s and c["sha"] == s["sha"]:
            fails.append(f"{arm}: perturbed control equals short output, gate cannot fail")
    speed = {}
    for kind, group in (("T", ts), ("S", ss)):
        for p in prompts:
            v = [x["decode_tok_s"] for r in group for k, x in r["responses"].items() if k.endswith("-" + p)]
            if v:
                speed.setdefault(kind, {})[p] = round(statistics.median(v), 3)
    return fails, speed


def finish(out: Path, recs: list[dict], prompts: list[str], binding: str) -> int:
    fails, speed = score(recs, prompts)
    (out / "summary.json").write_text(json.dumps({"fails": fails, "median_decode_tok_s": speed,
                                                  "arms": [r["arm"] for r in recs]}, indent=1))
    status = "TS_PASS" if not fails else "TS_FAIL " + "; ".join(fails[:20])
    (out / "status.txt").write_text(status + "\n" + binding + "\n")
    print(status)
    print(json.dumps(speed))
    return 0 if not fails else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--server", default=os.environ.get("FLUX_SERVER", "/opt/flux/bin/flux-server"))
    ap.add_argument("--model", default=os.environ.get("GLM53_MODEL", "/models/GLM-5.3-Flash-exl3-2.05bpw"))
    ap.add_argument("--draft-model", default=os.environ.get("GLM53_DRAFT"))
    ap.add_argument("--out", type=Path, default=Path("runs/p2"))
    ap.add_argument("--arms", default=DEFAULT_ARMS)
    ap.add_argument("--prompts", default=",".join(PROMPTS))
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--port", type=int, default=8893)
    ap.add_argument("--req-dir", type=Path, default=HERE / "req")
    ap.add_argument("--load-timeout", type=int, default=900)
    ap.add_argument("--min-mem-gb", type=float, default=100)
    ap.add_argument("--watchdog-gb", type=float, default=6)
    ap.add_argument("--admitted", action="store_true", help="P3 done: do not set the bring-up escape")
    ap.add_argument("--pass-env", action="append", default=[], metavar="NAME",
                    help="also pass this variable from the caller's env (servers otherwise get only the recipe env)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--score-only", type=Path)
    a = ap.parse_args(argv)
    prompts = [p for p in a.prompts.split(",") if p]
    arms = [x for x in a.arms.split(",") if x]
    if any(x[0] not in "TS" for x in arms):
        raise SystemExit("ts_identity: arm names must start with T or S")
    a.control_arms = {max((x for x in arms if x[0] == k), key=arms.index) for k in "TS" if any(x[0] == k for x in arms)}
    if a.score_only:
        recs = [json.loads((a.score_only / x / "arm.json").read_text()) for x in arms]
        return finish(a.score_only, recs, prompts, "binding rescored")
    if a.dry_run:
        for x in arms:
            env, argv_ = arm_plan(x[0], a, a.port)
            print(f"== {x}\n" + "".join(f"  {k}={v}\n" for k, v in sorted(env.items())) + "  " + " ".join(argv_))
        return 0
    a.out.mkdir(parents=True, exist_ok=True)
    binding = (f"binding server_sha256={sha256(a.server)} t_recipe={sha256(str(T_RECIPE))[:16]} "
               f"s_recipe={sha256(str(S_RECIPE))[:16]} model={a.model} admitted={a.admitted}")
    recs = []
    for x in arms:
        print(f"[{time.strftime('%H:%M:%S')}] arm {x}", flush=True)
        recs.append(run_arm(x, a, a.out, prompts))
    return finish(a.out, recs, prompts, binding)


if __name__ == "__main__":
    sys.exit(main())
