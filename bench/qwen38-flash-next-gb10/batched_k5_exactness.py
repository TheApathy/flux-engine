#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Exactness gate for ATLAS_QWEN4_BATCHED_K5 (cross-sequence native-MTP K5).

For each concurrency C in --concurrency (default 1,2,4,8) this starts one
fresh server through serve.sh (the canonical Flash-Next recipe), sends every
prompt of a fixed mixed-context workload in waves of C concurrent greedy
/v1/completions requests, and records the exact generated token IDs
(`return_token_ids`). Two references are produced the same way:

  plain-c1   RUNTIME_MODE=no-spec, MAX_BATCH_SIZE=1
  k5-c1      RUNTIME_MODE=mtp-k5,  MAX_BATCH_SIZE=1, selector unset
  k5-bC      RUNTIME_MODE=mtp-k5,  MAX_BATCH_SIZE=C, ATLAS_QWEN4_BATCHED_K5=1

PASS requires, for every C and every prompt, token IDs identical to both
references (and k5-c1 identical to plain-c1), every prompt finishing for
the same reason, at least one prompt above the 2048-token sparse-QSA
boundary, and, for C >= 2, server-log proof that the batched frame engaged
(QWEN4_BATCHED_K5_LANES_CONFIGURED lanes=C and QWEN4_BATCHED_K5_ENGAGED)
with no QWEN4_BATCHED_K5_REFUSED fallback unless --allow-refused.

This script launches GPU servers. Ensure exclusive GPU access and export
FLUX_GPU_EXCLUSIVE=1 (or legacy FLUXQ_LOCKS_HELD=1) before running; it
refuses to start otherwise. `--dry-run` prints the plan and launches nothing;
`--self-test` exercises the pure comparison/planning logic only.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVE_SH = HERE / "serve.sh"
QSA_BOUNDARY = 2048
SELECTOR = "ATLAS_QWEN4_BATCHED_K5"
ENGAGED_RE = re.compile(r"QWEN4_BATCHED_K5_ENGAGED")
LANES_RE = re.compile(r"QWEN4_BATCHED_K5_LANES_CONFIGURED lanes=(\d+)")
REFUSED_RE = re.compile(r"QWEN4_BATCHED_K5_REFUSED")
FRAME_RE = re.compile(r"QWEN4_BATCHED_K5 verify frame.*?seqs=(\d+).*?k=(\d+)")
RECEIPT_RE = re.compile(
    r"QWEN4_VERIFY_LOGITS_RECEIPT route=(\w+) prefix_len=(\d+) prefix_fnv=([0-9a-f]+) "
    r"k=(\d+) inputs=([\d,]*) argmax=([\d,]*) logits_fnv=([0-9a-f]+)"
)
RECEIPT_ENV = "ATLAS_QWEN4_VERIFY_LOGITS_RECEIPT"
# Minimum batched-frame logits receipts a C>=2 arm must match against C1.
MIN_MATCHED_RECEIPTS = 10

# Approximate target prompt lengths in tokens: short, around the 2048 QSA
# boundary on both sides, and well past it. Concurrency waves mix them.
DEFAULT_TARGETS = [160, 700, 1900, 2300, 3500, 6000, 12000, 20000]


# ----------------------------------------------------------------------------- workload
def deterministic_text(seed: int, approx_tokens: int) -> str:
    """Varied, deterministic English-like text (~1.3 tokens per word)."""
    words = (
        "river stone lantern harbor silver orchard meadow copper thunder "
        "glacier compass ember willow canyon falcon marble quartz prairie "
        "tide beacon summit hollow cedar violet anchor saffron delta ridge"
    ).split()
    out: list[str] = []
    state = (seed * 2654435761 + 97) & 0xFFFFFFFF
    n_words = max(8, int(approx_tokens / 1.3))
    for i in range(n_words):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(words[state % len(words)])
        if i % 17 == 16:
            out.append(f"({i // 17}).")
    return " ".join(out)


@dataclass(frozen=True)
class Prompt:
    index: int
    target_tokens: int
    text: str

    @property
    def sha(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()[:16]


def build_workload(targets: list[int]) -> list[Prompt]:
    tail = (
        "\n\nContinue the passage above for several paragraphs, describing "
        "each place and object in turn:\n"
    )
    return [
        Prompt(i, t, deterministic_text(i + 1, t) + tail) for i, t in enumerate(targets)
    ]


def waves(n: int, c: int) -> list[list[int]]:
    """Prompt indices grouped into waves of `c`, rotated per wave so every
    wave mixes short and long contexts instead of sorting by length."""
    if c < 1:
        raise ValueError("concurrency must be >= 1")
    order = list(range(n))
    if c > 1:
        # Interleave from both ends: short, long, short, long ...
        lo, hi, mixed = 0, n - 1, []
        while lo <= hi:
            mixed.append(order[lo])
            if lo != hi:
                mixed.append(order[hi])
            lo += 1
            hi -= 1
        order = mixed
    return [order[i : i + c] for i in range(0, n, c)]


# ----------------------------------------------------------------------------- arms
@dataclass(frozen=True)
class Arm:
    name: str
    runtime_mode: str
    batch: int
    batched: bool

    def env(self) -> dict[str, str]:
        env = {
            "RUNTIME_MODE": self.runtime_mode,
            "MAX_BATCH_SIZE": str(self.batch),
            "MAX_NUM_SEQS": str(self.batch),
        }
        if self.runtime_mode == "mtp-k5":
            env[RECEIPT_ENV] = "1"
        if self.batched:
            env[SELECTOR] = "1"
            env["ATLAS_QWEN4_BATCHED_K5_LOG"] = "1"
        return env


def plan_arms(concurrency: list[int]) -> list[Arm]:
    arms = [Arm("plain-c1", "no-spec", 1, False), Arm("k5-c1", "mtp-k5", 1, False)]
    for c in concurrency:
        arms.append(Arm(f"k5-b{c}", "mtp-k5", c, True))
    return arms


# ----------------------------------------------------------------------------- results
@dataclass
class Completion:
    token_ids: list[int]
    finish_reason: str
    prompt_tokens: int


@dataclass
class ArmResult:
    arm: Arm
    completions: dict[int, Completion] = field(default_factory=dict)
    log_path: str = ""
    engaged: bool = False
    lanes: int | None = None
    refused: bool = False
    frame_census: dict[str, int] = field(default_factory=dict)
    # (prefix_len, prefix_fnv, inputs) -> (route, argmax, logits_fnv)
    receipts: dict[tuple[str, str, str], tuple[str, str, str]] = field(default_factory=dict)
    receipt_conflicts: int = 0
    error: str | None = None


def first_divergence(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def compare(candidate: ArmResult, reference: ArmResult) -> list[str]:
    problems = []
    for index, ref in sorted(reference.completions.items()):
        got = candidate.completions.get(index)
        if got is None:
            problems.append(f"{candidate.arm.name} prompt {index}: missing")
            continue
        at = first_divergence(got.token_ids, ref.token_ids)
        if at is not None:
            problems.append(
                f"{candidate.arm.name} vs {reference.arm.name} prompt {index}: "
                f"first divergence at generated token {at} "
                f"(got {got.token_ids[at:at + 4]} want {ref.token_ids[at:at + 4]})"
            )
        if got.finish_reason != ref.finish_reason:
            problems.append(
                f"{candidate.arm.name} vs {reference.arm.name} prompt {index}: "
                f"finish {got.finish_reason!r} != {ref.finish_reason!r}"
            )
        if got.prompt_tokens != ref.prompt_tokens:
            problems.append(
                f"{candidate.arm.name} prompt {index}: prompt_tokens "
                f"{got.prompt_tokens} != {ref.prompt_tokens}"
            )
    return problems


def compare_receipts(candidate: ArmResult, reference: ArmResult) -> tuple[int, list[str]]:
    """Batched-frame logits receipts must equal the C1 receipt of the same
    history and verify window bit for bit. Returns (matched, problems)."""
    matched, problems = 0, []
    for key, (route, argmax, logits) in candidate.receipts.items():
        if route != "batched":
            continue
        ref = reference.receipts.get(key)
        if ref is None:
            continue
        if (ref[1], ref[2]) != (argmax, logits):
            problems.append(
                f"{candidate.arm.name}: logits receipt mismatch at prefix_len={key[0]} "
                f"inputs={key[2]} (argmax {argmax} vs {ref[1]}, logits {logits} vs {ref[2]})"
            )
        else:
            matched += 1
    return matched, problems


def verdict(results: dict[str, ArmResult], allow_refused: bool) -> list[str]:
    problems: list[str] = []
    for result in results.values():
        if result.error:
            problems.append(f"{result.arm.name}: {result.error}")
    plain, k5 = results.get("plain-c1"), results.get("k5-c1")
    if not plain or not k5 or plain.error or k5.error:
        return problems + ["reference arms incomplete"]
    if not any(c.prompt_tokens > QSA_BOUNDARY for c in plain.completions.values()):
        problems.append(f"no prompt exceeded the {QSA_BOUNDARY}-token QSA boundary")
    problems += compare(k5, plain)
    for name, result in results.items():
        if not name.startswith("k5-b") or result.error:
            continue
        problems += compare(result, k5)
        problems += compare(result, plain)
        if result.arm.batch >= 2:
            if result.lanes != result.arm.batch:
                problems.append(f"{name}: lanes={result.lanes}, expected {result.arm.batch}")
            if not result.engaged:
                problems.append(f"{name}: no QWEN4_BATCHED_K5_ENGAGED frame in the server log")
            if result.refused and not allow_refused:
                problems.append(f"{name}: a frame fell back (QWEN4_BATCHED_K5_REFUSED)")
            matched, receipt_problems = compare_receipts(result, k5)
            problems += receipt_problems
            if matched < MIN_MATCHED_RECEIPTS:
                problems.append(
                    f"{name}: only {matched} batched logits receipts matched a C1 frame "
                    f"(need {MIN_MATCHED_RECEIPTS})"
                )
        for arm_result in (result, k5):
            if arm_result.receipt_conflicts:
                problems.append(
                    f"{arm_result.arm.name}: {arm_result.receipt_conflicts} receipts disagree "
                    "with an earlier receipt of the same frame (run-to-run nondeterminism)"
                )
    return problems


# ----------------------------------------------------------------------------- server
def http_json(url: str, body: dict | None, timeout: float) -> dict:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def wait_ready(base: str, proc: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited during load (rc={proc.returncode})")
        try:
            http_json(f"{base}/v1/models", None, 5)
            return
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            time.sleep(5)
    raise RuntimeError("server did not become ready in time")


def stop(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=180)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=60)


def mem_available_gib() -> float:
    with open("/proc/meminfo") as meminfo:
        for line in meminfo:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024 * 1024)
    return 0.0


def complete(base: str, model: str, prompt: Prompt, max_tokens: int, timeout: float) -> Completion:
    body = {
        "model": model,
        "prompt": prompt.text,
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1.0,
        "seed": 7,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "stream": False,
        "return_token_ids": True,
    }
    response = http_json(f"{base}/v1/completions", body, timeout)
    choice = response["choices"][0]
    if "token_ids" not in choice:
        raise RuntimeError("server did not return token_ids (return_token_ids unsupported?)")
    usage = response.get("usage", {})
    return Completion(
        token_ids=[int(t) for t in choice["token_ids"]],
        finish_reason=str(choice.get("finish_reason")),
        prompt_tokens=int(usage.get("prompt_tokens", -1)),
    )


def scan_log(path: Path, result: ArmResult) -> None:
    text = path.read_text(errors="replace") if path.exists() else ""
    result.engaged = bool(ENGAGED_RE.search(text))
    lanes = LANES_RE.findall(text)
    result.lanes = int(lanes[-1]) if lanes else (1 if not result.arm.batched else None)
    result.refused = bool(REFUSED_RE.search(text))
    census: dict[str, int] = {}
    for seqs, k in FRAME_RE.findall(text):
        key = f"C{seqs}xK{k}"
        census[key] = census.get(key, 0) + 1
    result.frame_census = census
    for route, plen, pfnv, _k, inputs, argmax, logits in RECEIPT_RE.findall(text):
        key = (plen, pfnv, inputs)
        value = (route, argmax, logits)
        previous = result.receipts.get(key)
        if previous is not None and previous[1:] != value[1:]:
            result.receipt_conflicts += 1
        if previous is None or route == "batched":
            result.receipts[key] = value


def run_arm(arm: Arm, prompts: list[Prompt], args: argparse.Namespace, out: Path) -> ArmResult:
    result = ArmResult(arm)
    log_path = out / f"{arm.name}.server.log"
    result.log_path = str(log_path)
    env = dict(os.environ)
    env.pop(SELECTOR, None)
    env.pop("ATLAS_QWEN4_BATCHED_K5_LOG", None)
    env.update(arm.env())
    env.update(
        {
            "MODEL_DIR": args.model_dir,
            "PORT": str(args.port),
            "BIN": args.bin,
            # One context ceiling for every arm (references included), sized so
            # C8 x BF16 KV fits; below the native context, so no YaRN change.
            "MAX_SEQ_LEN": str(args.max_seq_len),
        }
    )
    base = f"http://127.0.0.1:{args.port}"
    floor = args.mem_floor_gib
    deadline = time.monotonic() + 600
    while mem_available_gib() < floor and time.monotonic() < deadline:
        time.sleep(10)
    if mem_available_gib() < floor:
        result.error = f"MemAvailable stayed below {floor} GiB before launch"
        return result
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(
            ["bash", str(SERVE_SH)],
            cwd=args.repo,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            wait_ready(base, proc, args.load_timeout)
            for wave in waves(len(prompts), arm.batch):
                with concurrent.futures.ThreadPoolExecutor(len(wave)) as pool:
                    futures = {
                        pool.submit(
                            complete, base, args.model_name, prompts[i], args.max_tokens,
                            args.request_timeout,
                        ): i
                        for i in wave
                    }
                    for future, index in futures.items():
                        result.completions[index] = future.result()
        except Exception as error:  # noqa: BLE001 - recorded in the receipt
            result.error = f"{type(error).__name__}: {error}"
        finally:
            stop(proc)
    scan_log(log_path, result)
    return result


# ----------------------------------------------------------------------------- self-test
def self_test() -> None:
    assert first_divergence([1, 2, 3], [1, 2, 3]) is None
    assert first_divergence([1, 2, 3], [1, 9, 3]) == 1
    assert first_divergence([1, 2], [1, 2, 3]) == 2
    assert waves(8, 1) == [[i] for i in range(8)]
    assert waves(8, 4) == [[0, 7, 1, 6], [2, 5, 3, 4]]
    assert sorted(i for w in waves(8, 8) for i in w) == list(range(8))
    assert sorted(i for w in waves(7, 2) for i in w) == list(range(7))
    arms = plan_arms([1, 2, 4, 8])
    assert [a.name for a in arms] == ["plain-c1", "k5-c1", "k5-b1", "k5-b2", "k5-b4", "k5-b8"]
    assert arms[0].env()["RUNTIME_MODE"] == "no-spec" and SELECTOR not in arms[1].env()
    assert arms[3].env()[SELECTOR] == "1" and arms[3].env()["MAX_BATCH_SIZE"] == "2"
    prompts = build_workload(DEFAULT_TARGETS)
    assert len({p.sha for p in prompts}) == len(prompts)
    assert build_workload(DEFAULT_TARGETS)[3].text == prompts[3].text, "deterministic"

    def result(arm: Arm, ids: dict[int, list[int]], tokens: int = 3000) -> ArmResult:
        r = ArmResult(arm, {i: Completion(t, "length", tokens) for i, t in ids.items()})
        r.lanes = arm.batch if arm.batched else 1
        r.engaged = arm.batched and arm.batch > 1
        return r

    good = {0: [1, 2, 3], 1: [4, 5, 6]}
    results = {a.name: result(a, good) for a in arms}
    assert all("receipts" in p for p in verdict(results, False)), verdict(results, False)
    bad = dict(results)
    bad["k5-b4"] = result(arms[4], {0: [1, 2, 3], 1: [4, 7, 6]})
    problems = verdict(bad, False)
    assert any("prompt 1: first divergence at generated token 1" in p for p in problems)
    unengaged = dict(results)
    unengaged["k5-b2"] = result(arms[3], good)
    unengaged["k5-b2"].engaged = False
    assert any("ENGAGED" in p for p in verdict(unengaged, False))
    short = {a.name: result(a, good, tokens=100) for a in arms}
    assert any("QSA boundary" in p for p in verdict(short, False))

    def with_receipts(r: ArmResult, route: str, logits: str, n: int) -> ArmResult:
        r.receipts = {(str(i), "ab", "1,2"): (route, "3,4", logits) for i in range(n)}
        return r

    rec = {a.name: result(a, good) for a in arms}
    with_receipts(rec["k5-c1"], "c1", "ff", MIN_MATCHED_RECEIPTS)
    for name in ("k5-b2", "k5-b4", "k5-b8"):
        with_receipts(rec[name], "batched", "ff", MIN_MATCHED_RECEIPTS)
    assert verdict(rec, False) == [], verdict(rec, False)
    with_receipts(rec["k5-b4"], "batched", "ee", MIN_MATCHED_RECEIPTS)
    assert any("k5-b4: logits receipt mismatch" in p for p in verdict(rec, False))
    with_receipts(rec["k5-b4"], "batched", "ff", 2)
    assert any("k5-b4: only 2 batched logits receipts" in p for p in verdict(rec, False))
    assert any("batched logits receipts" in p for p in verdict(results, False)), "receipts required"
    print("batched_k5_exactness self-test: PASS")


# ----------------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", default=os.environ.get("MODEL_DIR"))
    parser.add_argument("--repo", default=str(HERE.parents[1]), help="worktree whose serve.sh/binary run")
    parser.add_argument("--bin", default="target/release/spark", help="pinned binary path (relative to --repo)")
    parser.add_argument("--model-name", default="qwen38-native-mtp-frontier-v1")
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--concurrency", default="1,2,4,8")
    parser.add_argument("--targets", default=",".join(map(str, DEFAULT_TARGETS)))
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=32768)
    parser.add_argument("--load-timeout", type=float, default=1800)
    parser.add_argument("--request-timeout", type=float, default=1800)
    parser.add_argument("--mem-floor-gib", type=float, default=97.0)
    parser.add_argument("--out", default="batched-k5-exactness")
    parser.add_argument("--allow-refused", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0
    concurrency = [int(c) for c in args.concurrency.split(",") if c]
    prompts = build_workload([int(t) for t in args.targets.split(",") if t])
    arms = plan_arms(concurrency)
    if args.dry_run:
        for arm in arms:
            print(f"ARM {arm.name} env={arm.env()} waves={waves(len(prompts), arm.batch)}")
        for p in prompts:
            print(f"PROMPT {p.index} target~{p.target_tokens} sha={p.sha} chars={len(p.text)}")
        return 0
    if os.environ.get("FLUX_GPU_EXCLUSIVE") != "1" and os.environ.get("FLUXQ_LOCKS_HELD") != "1":
        print("refusing to launch GPU servers without exclusive GPU lock (set FLUX_GPU_EXCLUSIVE=1)", file=sys.stderr)
        return 2
    if not args.model_dir:
        print("--model-dir (or MODEL_DIR) is required", file=sys.stderr)
        return 2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, ArmResult] = {}
    for arm in arms:
        print(f"== arm {arm.name}", flush=True)
        results[arm.name] = run_arm(arm, prompts, args, out)
    problems = verdict(results, args.allow_refused)
    receipt = {
        "schema": 1,
        "verdict": "PASS" if not problems else "FAIL",
        "problems": problems,
        "prompts": [
            {"index": p.index, "target_tokens": p.target_tokens, "sha": p.sha} for p in prompts
        ],
        "arms": {
            name: {
                "env": r.arm.env(),
                "error": r.error,
                "engaged": r.engaged,
                "lanes": r.lanes,
                "refused": r.refused,
                "frame_census": r.frame_census,
                "log": r.log_path,
                "completions": {
                    str(i): {
                        "prompt_tokens": c.prompt_tokens,
                        "finish_reason": c.finish_reason,
                        "token_ids": c.token_ids,
                    }
                    for i, c in sorted(r.completions.items())
                },
            }
            for name, r in results.items()
        },
    }
    (out / "RECEIPT.json").write_text(json.dumps(receipt, indent=2))
    print(f"VERDICT {receipt['verdict']} ({len(problems)} problems) -> {out / 'RECEIPT.json'}")
    for problem in problems:
        print(f"  {problem}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
