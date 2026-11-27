#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only

"""In-repo OpenAI-API exactness certification harness for Flux Engine.

Certifies end-to-end serving exactness across five live test suites plus a
deterministic artifact comparator (``--compare``):

  1. Exact multi-digit arithmetic & modular checksum probes.
  2. Cold 4K / 8K / 16K / 32K needle passkey recall, including verbatim
     ``FLUX-V12-KV-14336-9472`` passkey recall and verbatim ``code_rust``
     SHA-256 verification.
  3. Cold-vs-warm prefix-cache & GDN recurrent-state replay determinism
     (bit-identical token trajectory and zero logprob drift).
  4. ``B=1`` vs ``B=4/8`` mixed-length concurrent batch invariance with zero
     cross-slot canary leakage.
  5. ``W=1`` vs ``W=8`` vs ``W=16`` speculative verify-width invariance (via
     ``--width-urls`` or multi-artifact ``--compare``).

Zero third-party dependencies (Python stdlib only). Run ``--self-test`` to spin
up an in-process ``ThreadingHTTPServer`` OpenAI-compatible mock server and
verify all 5 suites, ``--compare`` pass/fail gates, ``D_KL < 1e-4``, and
canary-leak fail-closed detection offline.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import re
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

DEFAULT_SEED = 42
DEFAULT_TOP_LOGPROBS = 5
DEFAULT_MAX_KL = 1.0e-4
DEFAULT_MAX_TOP1_FLIP_RATE = 0.0
DEFAULT_WARM_LOGPROB_TOL = 1.0e-5

VERBATIM_KV_PASSKEY = "FLUX-V12-KV-14336-9472"

CANONICAL_CODE_RUST = """pub fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}"""

CANONICAL_CODE_RUST_SHA256 = hashlib.sha256(CANONICAL_CODE_RUST.encode("utf-8")).hexdigest()

ARITHMETIC_CASES: tuple[tuple[str, str, str], ...] = (
    ("arith_mul_1", "Compute 7391 * 4829. Reply with only the exact integer.", "35691139"),
    ("arith_sub_2", "Compute 987654 - 123456. Reply with only the exact integer.", "864198"),
    (
        "arith_combo_3",
        "Compute (419 * 273) + 18827. Reply with only the exact integer.",
        "133214",
    ),
    ("arith_mod_4", "Compute 104729 mod 97. Reply with only the exact integer.", "66"),
)

# Deterministic filler sentence (~16 tokens) used to construct cold haystacks.
_HAYSTACK_LINE_TEMPLATE = (
    "Log entry {idx:06d} [salt={salt}]: deterministic telemetry checksum ok; "
    "no anomalous state transition observed in sector {sector:03d}.\n"
)


@dataclass
class StepLogprob:
    token: str
    logprob: float
    top_logprobs: dict[str, float] = field(default_factory=dict)


@dataclass
class CaseTrace:
    case_id: str
    suite: str
    passed: bool
    reason: str
    text: str
    text_sha256: str
    token_ids: list[int]
    cached_tokens: int
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    logprobs: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_rust_block(raw: str) -> str:
    """Extract fenced Rust block when present and normalize trailing whitespace."""
    stripped = raw.strip()
    fence = re.search(r"```(?:rust|rs)?\s*\n(.*?)\n```", stripped, flags=re.DOTALL)
    body = fence.group(1) if fence else stripped
    lines = [line.rstrip() for line in body.strip().splitlines()]
    return "\n".join(lines)


def build_cold_needle_prompt(
    target_tokens: int,
    passkey: str,
    salt: str,
    depth_fraction: float = 0.5,
) -> str:
    """Construct a deterministic cold haystack prompt of ~``target_tokens`` tokens."""
    # Each line averages ~16 BPE tokens; reserve ~64 tokens for header + needle + question.
    tokens_per_line = 16
    num_lines = max(4, (max(target_tokens, 128) - 64) // tokens_per_line)
    insert_at = max(1, min(num_lines - 1, int(round(num_lines * depth_fraction))))

    parts: list[str] = [
        f"You are verifying long-context exact recall (session_salt={salt}, target_tokens={target_tokens}).\n"
    ]
    for i in range(num_lines):
        if i == insert_at:
            parts.append(
                f"CRITICAL PASSKEY RECORD: The exact verification passkey is {passkey} .\n"
            )
        parts.append(
            _HAYSTACK_LINE_TEMPLATE.format(idx=i, salt=salt, sector=(i * 17 + 3) % 251)
        )
    parts.append(
        "Question: What is the exact verification passkey stated in the CRITICAL PASSKEY RECORD above? "
        "Reply with ONLY the exact passkey token string."
    )
    return "".join(parts)


def build_code_rust_prompt(salt: str) -> str:
    """Construct a deterministic prompt verifying verbatim Rust code reproduction."""
    return (
        f"Exact verbatim reproduction check (salt={salt}). "
        "Reproduce the following Rust function verbatim with no commentary or changes:\n\n"
        f"```rust\n{CANONICAL_CODE_RUST}\n```\n"
    )


def _http_post_json(url: str, payload: dict[str, Any], timeout: float = 120.0) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            if resp.status != HTTPStatus.OK:
                raise RuntimeError(f"POST {url} returned HTTP {resp.status}: {raw[:200]}")
            return json.loads(raw)
    except urllib.error.HTTPError as err:
        body = err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} failed with HTTP {err.code}: {body[:300]}") from err


def _extract_choice_trace(
    resp: dict[str, Any],
    endpoint: str,
) -> tuple[str, list[int], list[dict[str, Any]], int, int, int]:
    choices = resp.get("choices") or []
    if not choices:
        raise RuntimeError(f"response missing choices: {resp}")
    choice = choices[0]
    if endpoint == "/v1/chat/completions":
        msg = choice.get("message") or {}
        text = str(msg.get("content") or "")
    else:
        text = str(choice.get("text") or "")

    raw_ids = choice.get("token_ids") or []
    token_ids = [int(x) for x in raw_ids if isinstance(x, int) and not isinstance(x, bool)]

    parsed_logprobs: list[dict[str, Any]] = []
    lp_container = choice.get("logprobs")
    if isinstance(lp_container, dict) and isinstance(lp_container.get("content"), list):
        for entry in lp_container["content"]:
            if not isinstance(entry, dict):
                continue
            top_map: dict[str, float] = {}
            for item in entry.get("top_logprobs") or []:
                if isinstance(item, dict) and "token" in item and "logprob" in item:
                    val = float(item["logprob"])
                    if math.isfinite(val):
                        top_map[str(item["token"])] = val
            tok_str = str(entry.get("token", ""))
            lp_val = float(entry.get("logprob", 0.0))
            if tok_str and math.isfinite(lp_val) and tok_str not in top_map:
                top_map[tok_str] = lp_val
            parsed_logprobs.append(
                {
                    "token": tok_str,
                    "logprob": lp_val,
                    "top_logprobs": top_map,
                }
            )

    usage = resp.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or len(token_ids))
    details = usage.get("prompt_tokens_details") or {}
    cached_tokens = int(details.get("cached_tokens") or 0)
    return text, token_ids, parsed_logprobs, prompt_tokens, completion_tokens, cached_tokens


def query_model(
    base_url: str,
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int = 128,
    temperature: float = 0.0,
    seed: int = DEFAULT_SEED,
    top_logprobs: int = DEFAULT_TOP_LOGPROBS,
    timeout: float = 120.0,
) -> tuple[str, list[int], list[dict[str, Any]], int, int, int, float]:
    """Query a Flux OpenAI-compatible endpoint deterministically."""
    url = f"{base_url.rstrip('/')}{endpoint}"
    if endpoint == "/v1/chat/completions":
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
            "stream": False,
            "return_token_ids": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if top_logprobs > 0:
            payload["logprobs"] = True
            payload["top_logprobs"] = top_logprobs
    else:
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
            "stream": False,
            "return_token_ids": True,
        }

    t0 = time.perf_counter()
    resp = _http_post_json(url, payload, timeout=timeout)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    text, token_ids, logprobs, p_toks, c_toks, cached = _extract_choice_trace(resp, endpoint)
    return text, token_ids, logprobs, p_toks, c_toks, cached, latency_ms


def _step_kl_divergence(ref_top: dict[str, float], cand_top: dict[str, float]) -> tuple[float, float]:
    """Compute exact non-negative ``D_KL(P_ref || P_cand)`` and ``max |logp_cand - logp_ref|``
    over the union top-K support plus residual tail bucket."""
    if not ref_top or not cand_top:
        return 0.0, 0.0

    union_keys = sorted(set(ref_top.keys()) | set(cand_top.keys()))
    max_abs_diff = 0.0
    for k in set(ref_top.keys()) & set(cand_top.keys()):
        max_abs_diff = max(max_abs_diff, abs(cand_top[k] - ref_top[k]))

    # Convert logprobs to probabilities; allocate residual probability mass to an explicit tail bucket.
    eps = 1.0e-12
    p_ref_raw = {k: math.exp(ref_top[k]) for k in ref_top}
    p_cand_raw = {k: math.exp(cand_top[k]) for k in cand_top}

    sum_ref = sum(p_ref_raw.values())
    sum_cand = sum(p_cand_raw.values())
    # If inputs are truncated top-K probabilities (< 1.0), preserve residual mass in tail bucket;
    # if slightly > 1.0 due to float32 rounding, renormalize cleanly.
    tail_ref = max(0.0, 1.0 - sum_ref)
    tail_cand = max(0.0, 1.0 - sum_cand)
    norm_ref = sum_ref + tail_ref
    norm_cand = sum_cand + tail_cand

    p_ref_vec: list[float] = []
    p_cand_vec: list[float] = []
    for k in union_keys:
        pr = p_ref_raw.get(k, 0.0) / norm_ref
        pc = p_cand_raw.get(k, 0.0) / norm_cand
        p_ref_vec.append(pr)
        p_cand_vec.append(pc)
    if tail_ref > 0.0 or tail_cand > 0.0:
        p_ref_vec.append(tail_ref / norm_ref)
        p_cand_vec.append(tail_cand / norm_cand)

    # Renormalize after epsilon floor on candidate support to keep D_KL finite and >= 0.
    p_ref_total = sum(p_ref_vec)
    p_cand_floored = [max(v, eps) for v in p_cand_vec]
    p_cand_total = sum(p_cand_floored)

    kl = 0.0
    for pr_raw, pc_raw in zip(p_ref_vec, p_cand_floored):
        pr = pr_raw / p_ref_total
        pc = pc_raw / p_cand_total
        if pr > 0.0:
            kl += pr * math.log(pr / pc)
    return max(0.0, kl), max_abs_diff


def compare_two_traces(ref: dict[str, Any], cand: dict[str, Any]) -> dict[str, Any]:
    """Compare one reference trace against one candidate trace."""
    ref_ids = ref.get("token_ids") or []
    cand_ids = cand.get("token_ids") or []
    ref_text = str(ref.get("text") or "")
    cand_text = str(cand.get("text") or "")

    if ref_ids and cand_ids:
        seq_ref: list[Any] = ref_ids
        seq_cand: list[Any] = cand_ids
        unit = "token_id"
    else:
        seq_ref = list(ref_text)
        seq_cand = list(cand_text)
        unit = "char"

    min_len = min(len(seq_ref), len(seq_cand))
    max_len = max(len(seq_ref), len(seq_cand))
    first_div: int | None = None
    mismatches = abs(len(seq_ref) - len(seq_cand))
    for i in range(min_len):
        if seq_ref[i] != seq_cand[i]:
            if first_div is None:
                first_div = i
            mismatches += 1
    if first_div is None and len(seq_ref) != len(seq_cand):
        first_div = min_len

    exact_match = (ref_ids == cand_ids) if (ref_ids and cand_ids) else (ref_text == cand_text)
    if ref_ids and cand_ids:
        exact_match = exact_match and (ref_text == cand_text)

    # Logprob & top-1 flip analysis over overlapping steps.
    ref_lps = ref.get("logprobs") or []
    cand_lps = cand.get("logprobs") or []
    lp_steps = min(len(ref_lps), len(cand_lps))
    top1_flips = 0
    max_abs_lp_diff = 0.0
    step_kls: list[float] = []
    for t in range(lp_steps):
        r_entry = ref_lps[t]
        c_entry = cand_lps[t]
        r_top = r_entry.get("top_logprobs") or {}
        c_top = c_entry.get("top_logprobs") or {}
        r_tok = str(r_entry.get("token", ""))
        c_tok = str(c_entry.get("token", ""))
        if r_tok != c_tok:
            top1_flips += 1
        r_lp = float(r_entry.get("logprob", 0.0))
        c_lp = float(c_entry.get("logprob", 0.0))
        if r_tok == c_tok and math.isfinite(r_lp) and math.isfinite(c_lp):
            max_abs_lp_diff = max(max_abs_lp_diff, abs(c_lp - r_lp))
        kl, step_diff = _step_kl_divergence(r_top, c_top)
        step_kls.append(kl)
        max_abs_lp_diff = max(max_abs_lp_diff, step_diff)

    if lp_steps > 0:
        top1_flip_rate = top1_flips / float(lp_steps)
    elif max_len > 0:
        top1_flip_rate = mismatches / float(max_len)
    else:
        top1_flip_rate = 0.0

    mean_kl = (sum(step_kls) / len(step_kls)) if step_kls else 0.0
    max_kl = max(step_kls) if step_kls else 0.0

    return {
        "case_id": ref.get("case_id", "unknown"),
        "unit": unit,
        "trajectory_exact_match": exact_match,
        "first_divergence_index": first_div,
        "compared_steps": lp_steps if lp_steps > 0 else max_len,
        "top1_flip_rate": round(top1_flip_rate, 8),
        "max_abs_logprob_diff": round(max_abs_lp_diff, 8),
        "mean_kl_divergence": mean_kl,
        "max_kl_divergence": max_kl,
    }


def compare_reports(
    ref_report: dict[str, Any],
    cand_report: dict[str, Any],
    max_kl: float = DEFAULT_MAX_KL,
    max_top1_flip_rate: float = DEFAULT_MAX_TOP1_FLIP_RATE,
    require_exact: bool = True,
) -> dict[str, Any]:
    """Compare a candidate exactness report against a reference report."""
    ref_cases = {c["case_id"]: c for c in ref_report.get("cases", [])}
    cand_cases = {c["case_id"]: c for c in cand_report.get("cases", [])}

    missing_in_cand = sorted(set(ref_cases.keys()) - set(cand_cases.keys()))
    extra_in_cand = sorted(set(cand_cases.keys()) - set(ref_cases.keys()))
    shared_ids = sorted(set(ref_cases.keys()) & set(cand_cases.keys()))

    per_case: list[dict[str, Any]] = []
    for cid in shared_ids:
        per_case.append(compare_two_traces(ref_cases[cid], cand_cases[cid]))

    exact_count = sum(1 for row in per_case if row["trajectory_exact_match"])
    exact_rate = (exact_count / len(per_case)) if per_case else 0.0
    first_div_any = next(
        (
            {"case_id": row["case_id"], "index": row["first_divergence_index"]}
            for row in per_case
            if row["first_divergence_index"] is not None
        ),
        None,
    )
    overall_flip_rate = (
        max((row["top1_flip_rate"] for row in per_case), default=0.0) if per_case else 1.0
    )
    overall_max_lp_diff = max((row["max_abs_logprob_diff"] for row in per_case), default=0.0)
    overall_mean_kl = (
        sum(row["mean_kl_divergence"] for row in per_case) / len(per_case) if per_case else 0.0
    )
    overall_max_kl = max((row["max_kl_divergence"] for row in per_case), default=0.0)

    failures: list[str] = []
    if not shared_ids:
        failures.append("no shared case_ids between reference and candidate reports")
    if missing_in_cand:
        failures.append(f"candidate missing {len(missing_in_cand)} reference cases: {missing_in_cand}")
    if require_exact and exact_count != len(per_case):
        failures.append(
            f"trajectory exactness {exact_count}/{len(per_case)} < 100% (first divergence: {first_div_any})"
        )
    if overall_flip_rate > max_top1_flip_rate:
        failures.append(
            f"max top-1 flip rate {overall_flip_rate:.6f} > gate {max_top1_flip_rate:.6f}"
        )
    if overall_max_kl > max_kl:
        failures.append(f"max D_KL {overall_max_kl:.3e} > gate {max_kl:.3e}")

    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "gates": {
            "max_kl": max_kl,
            "max_top1_flip_rate": max_top1_flip_rate,
            "require_exact": require_exact,
        },
        "summary": {
            "shared_cases": len(shared_ids),
            "missing_in_cand": missing_in_cand,
            "extra_in_cand": extra_in_cand,
            "trajectory_exact_count": exact_count,
            "trajectory_exact_rate": round(exact_rate, 6),
            "first_divergence": first_div_any,
            "max_top1_flip_rate": round(overall_flip_rate, 8),
            "max_abs_logprob_diff": round(overall_max_lp_diff, 8),
            "mean_kl_divergence": overall_mean_kl,
            "max_kl_divergence": overall_max_kl,
        },
        "cases": per_case,
    }


def run_certification_suite(
    base_url: str,
    endpoint: str,
    model: str,
    needle_lengths: list[int],
    batch_sizes: list[int],
    seed: int = DEFAULT_SEED,
    top_logprobs: int = DEFAULT_TOP_LOGPROBS,
    run_salt: str | None = None,
    warm_logprob_tol: float = DEFAULT_WARM_LOGPROB_TOL,
    max_kl: float = DEFAULT_MAX_KL,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Run suites 1–4 against one live OpenAI-compatible Flux server."""
    salt = run_salt or hashlib.sha256(f"flux-v14-cert-{seed}".encode("utf-8")).hexdigest()[:12]
    cases: list[CaseTrace] = []
    suite_failures: list[str] = []

    # ── Suite 1: Exact arithmetic ───────────────────────────────────────────
    for case_id, prompt, expected_ans in ARITHMETIC_CASES:
        text, tids, lps, p_toks, c_toks, cached, lat_ms = query_model(
            base_url=base_url,
            endpoint=endpoint,
            model=model,
            prompt=prompt,
            max_tokens=32,
            temperature=0.0,
            seed=seed,
            top_logprobs=top_logprobs,
            timeout=timeout,
        )
        ok = expected_ans in text.strip()
        reason = "ok" if ok else f"expected `{expected_ans}` in `{text.strip()[:80]}`"
        if not ok:
            suite_failures.append(f"{case_id}: {reason}")
        cases.append(
            CaseTrace(
                case_id=case_id,
                suite="exact_arithmetic",
                passed=ok,
                reason=reason,
                text=text,
                text_sha256=sha256_text(text),
                token_ids=tids,
                cached_tokens=cached,
                prompt_tokens=p_toks,
                completion_tokens=c_toks,
                latency_ms=round(lat_ms, 3),
                logprobs=lps,
                metadata={"expected": expected_ans},
            )
        )

    # ── Suite 2A: Cold needle passkeys (4K/8K/16K/32K + FLUX-V12-KV-14336-9472) ─
    for idx, length in enumerate(needle_lengths):
        # Embed the canonical verbatim FLUX-V12-KV-14336-9472 passkey at the 16K/14K slot
        # (or the middle slot when scaled in --self-test) and length-tagged passkeys elsewhere.
        is_canonical_slot = (length == 16384) or (
            16384 not in needle_lengths and idx == len(needle_lengths) // 2
        )
        passkey = (
            VERBATIM_KV_PASSKEY
            if is_canonical_slot
            else f"FLUX-V14-NEEDLE-{length}-{hashlib.sha256(f'{salt}:{length}'.encode()).hexdigest()[:8].upper()}"
        )
        prompt = build_cold_needle_prompt(
            target_tokens=length,
            passkey=passkey,
            salt=f"{salt}-cold-{length}",
            depth_fraction=0.5,
        )
        text, tids, lps, p_toks, c_toks, cached, lat_ms = query_model(
            base_url=base_url,
            endpoint=endpoint,
            model=model,
            prompt=prompt,
            max_tokens=48,
            temperature=0.0,
            seed=seed,
            top_logprobs=top_logprobs,
            timeout=timeout,
        )
        ok = passkey in text
        reason = "ok" if ok else f"passkey `{passkey}` missing from `{text.strip()[:100]}`"
        case_id = f"needle_cold_{length}"
        if not ok:
            suite_failures.append(f"{case_id}: {reason}")
        cases.append(
            CaseTrace(
                case_id=case_id,
                suite="cold_needle_passkey",
                passed=ok,
                reason=reason,
                text=text,
                text_sha256=sha256_text(text),
                token_ids=tids,
                cached_tokens=cached,
                prompt_tokens=p_toks,
                completion_tokens=c_toks,
                latency_ms=round(lat_ms, 3),
                logprobs=lps,
                metadata={"target_tokens": length, "expected_passkey": passkey},
            )
        )

    # ── Suite 2B: Verbatim code_rust SHA-256 check ──────────────────────────
    rust_prompt = build_code_rust_prompt(salt=f"{salt}-rust")
    text, tids, lps, p_toks, c_toks, cached, lat_ms = query_model(
        base_url=base_url,
        endpoint=endpoint,
        model=model,
        prompt=rust_prompt,
        max_tokens=160,
        temperature=0.0,
        seed=seed,
        top_logprobs=top_logprobs,
        timeout=timeout,
    )
    normalized_rust = normalize_rust_block(text)
    got_rust_sha = sha256_text(normalized_rust)
    rust_ok = got_rust_sha == CANONICAL_CODE_RUST_SHA256
    rust_reason = (
        "ok"
        if rust_ok
        else f"code_rust sha256 {got_rust_sha[:16]} != expected {CANONICAL_CODE_RUST_SHA256[:16]}"
    )
    if not rust_ok:
        suite_failures.append(f"code_rust_sha256: {rust_reason}")
    cases.append(
        CaseTrace(
            case_id="code_rust_sha256",
            suite="code_rust_verbatim",
            passed=rust_ok,
            reason=rust_reason,
            text=text,
            text_sha256=sha256_text(text),
            token_ids=tids,
            cached_tokens=cached,
            prompt_tokens=p_toks,
            completion_tokens=c_toks,
            latency_ms=round(lat_ms, 3),
            logprobs=lps,
            metadata={
                "expected_code_rust_sha256": CANONICAL_CODE_RUST_SHA256,
                "actual_code_rust_sha256": got_rust_sha,
            },
        )
    )

    # ── Suite 3: Cold-vs-warm prefix / GDN replay determinism ───────────────
    replay_len = min(needle_lengths) if needle_lengths else 1024
    replay_passkey = f"FLUX-V14-REPLAY-{salt[:8].upper()}"
    replay_prompt = build_cold_needle_prompt(
        target_tokens=replay_len,
        passkey=replay_passkey,
        salt=f"{salt}-replay-unique-{int(time.time() * 1000) % 100000}",
        depth_fraction=0.45,
    )
    cold_res = query_model(
        base_url=base_url,
        endpoint=endpoint,
        model=model,
        prompt=replay_prompt,
        max_tokens=48,
        temperature=0.0,
        seed=seed,
        top_logprobs=top_logprobs,
        timeout=timeout,
    )
    warm_res = query_model(
        base_url=base_url,
        endpoint=endpoint,
        model=model,
        prompt=replay_prompt,
        max_tokens=48,
        temperature=0.0,
        seed=seed,
        top_logprobs=top_logprobs,
        timeout=timeout,
    )
    cold_trace = CaseTrace(
        case_id="cold_vs_warm_cold",
        suite="cold_vs_warm_replay",
        passed=replay_passkey in cold_res[0],
        reason="ok" if replay_passkey in cold_res[0] else "cold passkey mismatch",
        text=cold_res[0],
        text_sha256=sha256_text(cold_res[0]),
        token_ids=cold_res[1],
        cached_tokens=cold_res[5],
        prompt_tokens=cold_res[3],
        completion_tokens=cold_res[4],
        latency_ms=round(cold_res[6], 3),
        logprobs=cold_res[2],
    )
    warm_trace = CaseTrace(
        case_id="cold_vs_warm_warm",
        suite="cold_vs_warm_replay",
        passed=replay_passkey in warm_res[0],
        reason="ok" if replay_passkey in warm_res[0] else "warm passkey mismatch",
        text=warm_res[0],
        text_sha256=sha256_text(warm_res[0]),
        token_ids=warm_res[1],
        cached_tokens=warm_res[5],
        prompt_tokens=warm_res[3],
        completion_tokens=warm_res[4],
        latency_ms=round(warm_res[6], 3),
        logprobs=warm_res[2],
    )
    cw_diff = compare_two_traces(asdict(cold_trace), asdict(warm_trace))
    cw_ok = (
        cold_trace.passed
        and warm_trace.passed
        and cw_diff["trajectory_exact_match"]
        and cw_diff["max_abs_logprob_diff"] <= warm_logprob_tol
        and cw_diff["max_kl_divergence"] <= max_kl
    )
    if not cw_ok:
        warm_trace.passed = False
        warm_trace.reason = (
            f"cold-vs-warm drift: exact={cw_diff['trajectory_exact_match']}, "
            f"max_lp_diff={cw_diff['max_abs_logprob_diff']}, max_kl={cw_diff['max_kl_divergence']:.3e}"
        )
        suite_failures.append(f"cold_vs_warm_replay: {warm_trace.reason}")
    warm_trace.metadata = {
        "cold_cached_tokens": cold_trace.cached_tokens,
        "warm_cached_tokens": warm_trace.cached_tokens,
        "comparison": cw_diff,
    }
    cases.extend([cold_trace, warm_trace])

    # ── Suite 4: B=1 vs B=4/8 mixed batch invariance + zero canary leaks ────
    b1_prompt = build_cold_needle_prompt(
        target_tokens=max(256, min(needle_lengths) if needle_lengths else 512),
        passkey=VERBATIM_KV_PASSKEY,
        salt=f"{salt}-b1-target",
        depth_fraction=0.5,
    )
    b1_res = query_model(
        base_url=base_url,
        endpoint=endpoint,
        model=model,
        prompt=b1_prompt,
        max_tokens=48,
        temperature=0.0,
        seed=seed,
        top_logprobs=top_logprobs,
        timeout=timeout,
    )
    b1_trace = CaseTrace(
        case_id="batch_b1_reference",
        suite="batch_invariance",
        passed=VERBATIM_KV_PASSKEY in b1_res[0],
        reason="ok" if VERBATIM_KV_PASSKEY in b1_res[0] else "B=1 target passkey missing",
        text=b1_res[0],
        text_sha256=sha256_text(b1_res[0]),
        token_ids=b1_res[1],
        cached_tokens=b1_res[5],
        prompt_tokens=b1_res[3],
        completion_tokens=b1_res[4],
        latency_ms=round(b1_res[6], 3),
        logprobs=b1_res[2],
    )
    if not b1_trace.passed:
        suite_failures.append(f"batch_b1_reference: {b1_trace.reason}")
    cases.append(b1_trace)

    for b_size in batch_sizes:
        if b_size < 2:
            continue
        slot_prompts: list[tuple[int, str, str]] = [(0, VERBATIM_KV_PASSKEY, b1_prompt)]
        for slot in range(1, b_size):
            canary = f"CANARY-ISOLATION-B{b_size}-S{slot}-{hashlib.sha256(f'{salt}:{b_size}:{slot}'.encode()).hexdigest()[:8].upper()}"
            # Deliberately vary distractor lengths to stress ragged mixed-batch scheduling.
            dist_len = 128 + ((slot * 193) % 512)
            dist_prompt = build_cold_needle_prompt(
                target_tokens=dist_len,
                passkey=canary,
                salt=f"{salt}-b{b_size}-slot{slot}",
                depth_fraction=0.35 + 0.1 * (slot % 4),
            )
            slot_prompts.append((slot, canary, dist_prompt))

        def _run_slot(item: tuple[int, str, str]) -> tuple[int, str, tuple[Any, ...]]:
            slot_idx, canary_tok, p_text = item
            res = query_model(
                base_url=base_url,
                endpoint=endpoint,
                model=model,
                prompt=p_text,
                max_tokens=48,
                temperature=0.0,
                seed=seed,
                top_logprobs=top_logprobs,
                timeout=timeout,
            )
            return slot_idx, canary_tok, res

        with concurrent.futures.ThreadPoolExecutor(max_workers=b_size) as pool:
            slot_outputs = list(pool.map(_run_slot, slot_prompts))

        slot_outputs.sort(key=lambda x: x[0])
        target_res = slot_outputs[0][2]
        cand_batch_trace = CaseTrace(
            case_id=f"batch_b{b_size}_vs_b1",
            suite="batch_invariance",
            passed=True,
            reason="ok",
            text=target_res[0],
            text_sha256=sha256_text(target_res[0]),
            token_ids=target_res[1],
            cached_tokens=target_res[5],
            prompt_tokens=target_res[3],
            completion_tokens=target_res[4],
            latency_ms=round(target_res[6], 3),
            logprobs=target_res[2],
        )

        # Check cross-slot canary isolation across ALL pairs (i != j).
        leaks: list[str] = []
        all_canaries = [(s, c) for s, c, _ in slot_outputs]
        for slot_idx, _, res in slot_outputs:
            out_text = res[0]
            for other_slot, other_canary in all_canaries:
                if other_slot != slot_idx and other_canary in out_text:
                    leaks.append(f"slot {slot_idx} leaked canary from slot {other_slot} ({other_canary})")

        b_cmp = compare_two_traces(asdict(b1_trace), asdict(cand_batch_trace))
        b_ok = (
            len(leaks) == 0
            and b_cmp["trajectory_exact_match"]
            and b_cmp["max_kl_divergence"] <= max_kl
        )
        if not b_ok:
            cand_batch_trace.passed = False
            cand_batch_trace.reason = (
                f"batch B={b_size} invariance failed: leaks={leaks}, "
                f"exact={b_cmp['trajectory_exact_match']}, max_kl={b_cmp['max_kl_divergence']:.3e}"
            )
            suite_failures.append(f"batch_b{b_size}_vs_b1: {cand_batch_trace.reason}")
        cand_batch_trace.metadata = {
            "batch_size": b_size,
            "canary_leaks": leaks,
            "comparison_vs_b1": b_cmp,
        }
        cases.append(cand_batch_trace)

    return {
        "schema_version": 1,
        "base_url": base_url,
        "endpoint": endpoint,
        "model": model,
        "seed": seed,
        "salt": salt,
        "passed": len(suite_failures) == 0,
        "failures": suite_failures,
        "summary": {
            "total_cases": len(cases),
            "passed_cases": sum(1 for c in cases if c.passed),
            "failed_cases": sum(1 for c in cases if not c.passed),
        },
        "cases": [asdict(c) for c in cases],
    }


def run_verify_width_invariance(
    width_urls: dict[str, str],
    endpoint: str,
    model: str,
    seed: int = DEFAULT_SEED,
    top_logprobs: int = DEFAULT_TOP_LOGPROBS,
    max_kl: float = DEFAULT_MAX_KL,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Run canonical prompts across ``W=1`` vs ``W=8`` vs ``W=16`` servers and certify exactness."""
    if len(width_urls) < 2:
        raise ValueError("width invariance requires at least two labeled URLs (e.g. w1=...,w8=...,w16=...)")
    salt = hashlib.sha256(f"flux-v14-width-{seed}".encode("utf-8")).hexdigest()[:12]
    reports: dict[str, dict[str, Any]] = {}
    for label, url in width_urls.items():
        reports[label] = run_certification_suite(
            base_url=url,
            endpoint=endpoint,
            model=model,
            needle_lengths=[512],
            batch_sizes=[],
            seed=seed,
            top_logprobs=top_logprobs,
            run_salt=salt,
            max_kl=max_kl,
            timeout=timeout,
        )

    labels = list(width_urls.keys())
    ref_label = "w1" if "w1" in reports else labels[0]
    ref_report = reports[ref_label]
    comparisons: dict[str, Any] = {}
    failures: list[str] = []
    for label in labels:
        if label == ref_label:
            continue
        cmp_res = compare_reports(
            ref_report=ref_report,
            cand_report=reports[label],
            max_kl=max_kl,
            max_top1_flip_rate=0.0,
            require_exact=True,
        )
        comparisons[f"{label}_vs_{ref_label}"] = cmp_res
        if not cmp_res["passed"]:
            failures.extend(f"{label}_vs_{ref_label}: {f}" for f in cmp_res["failures"])

    return {
        "passed": len(failures) == 0 and all(r["passed"] for r in reports.values()),
        "reference": ref_label,
        "failures": failures,
        "comparisons": comparisons,
    }


# ── In-process OpenAI-compatible mock server for --self-test ────────────────


class _MockExactnessState:
    """Deterministic in-memory mock Flux OpenAI server for offline ``--self-test``."""

    def __init__(self, inject_canary_leak: bool = False) -> None:
        self.lock = threading.Lock()
        self.prefix_cache: set[str] = set()
        self.inject_canary_leak = inject_canary_leak
        self.last_canary_seen: str | None = None

    @staticmethod
    def _answer_prompt(prompt: str) -> str:
        # 1. Exact arithmetic
        for _, q_prompt, ans in ARITHMETIC_CASES:
            if prompt.strip() == q_prompt:
                return ans
        # 2. Verbatim code_rust check
        if "pub fn fnv1a_64" in prompt:
            return f"```rust\n{CANONICAL_CODE_RUST}\n```"
        # 3. Embedded CRITICAL PASSKEY RECORD
        match = re.search(r"exact verification passkey is ([A-Za-z0-9_-]+)", prompt)
        if match:
            return match.group(1)
        return "OK"

    def handle_request(self, body: dict[str, Any], is_chat: bool) -> dict[str, Any]:
        if is_chat:
            msgs = body.get("messages") or []
            prompt = str(msgs[-1].get("content") or "") if msgs else ""
        else:
            prompt = str(body.get("prompt") or "")

        prompt_hash = sha256_text(prompt)
        prompt_tokens = max(8, len(prompt.split()))
        with self.lock:
            was_cached = prompt_hash in self.prefix_cache
            self.prefix_cache.add(prompt_hash)
            cached_tokens = (prompt_tokens // 16) * 16 if was_cached else 0

        answer = self._answer_prompt(prompt)
        if self.inject_canary_leak:
            with self.lock:
                if answer.startswith("CANARY-"):
                    self.last_canary_seen = answer
                elif self.last_canary_seen is not None and answer == VERBATIM_KV_PASSKEY:
                    answer = f"{answer} {self.last_canary_seen}"

        tokens = [answer[i : i + 4] for i in range(0, len(answer), 4)] or ["OK"]
        token_ids = [100 + (int(sha256_text(tok)[:4], 16) % 30000) for tok in tokens]
        top_k = int(body.get("top_logprobs") or 0)
        want_lps = bool(body.get("logprobs")) and top_k > 0

        lp_entries: list[dict[str, Any]] = []
        if want_lps:
            # Deterministic peaked distribution (p0 ~= 0.995, p1 ~= 0.004, tail ~= 0.001).
            base_probs = [0.995, 0.004, 0.0006, 0.00025, 0.00015]
            for tok in tokens:
                alts = [
                    {"token": tok, "logprob": math.log(base_probs[0])},
                ]
                for k_idx in range(1, min(top_k, len(base_probs))):
                    alts.append(
                        {"token": f"<alt_{k_idx}>", "logprob": math.log(base_probs[k_idx])}
                    )
                lp_entries.append(
                    {
                        "token": tok,
                        "logprob": alts[0]["logprob"],
                        "top_logprobs": alts,
                    }
                )

        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": len(token_ids),
            "total_tokens": prompt_tokens + len(token_ids),
            "prompt_tokens_details": {"cached_tokens": cached_tokens, "audio_tokens": 0},
            "time_to_first_token_ms": 2.5 if was_cached else 12.0,
            "response_token/s": 120.0,
        }
        model_name = str(body.get("model") or "mock-flux-v14")
        if is_chat:
            choice: dict[str, Any] = {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
                "logprobs": {"content": lp_entries} if want_lps else None,
            }
            if body.get("return_token_ids"):
                choice["token_ids"] = token_ids
            return {
                "id": "chatcmpl-mock-v14",
                "object": "chat.completion",
                "created": 1700000000,
                "model": model_name,
                "choices": [choice],
                "usage": usage,
            }

        choice_c: dict[str, Any] = {
            "index": 0,
            "text": answer,
            "finish_reason": "stop",
        }
        if body.get("return_token_ids"):
            choice_c["token_ids"] = token_ids
        return {
            "id": "cmpl-mock-v14",
            "object": "text_completion",
            "created": 1700000000,
            "model": model_name,
            "choices": [choice_c],
            "usage": usage,
        }


def _start_mock_openai_server(
    state: _MockExactnessState,
) -> tuple[ThreadingHTTPServer, threading.Thread, str]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _send_json(self, code: int, obj: dict[str, Any]) -> None:
            raw = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(raw)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length) if length > 0 else b""
            try:
                body = json.loads(raw_body.decode("utf-8")) if raw_body else {}
            except json.JSONDecodeError as err:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": {"message": str(err)}})
                return
            if self.path == "/v1/chat/completions":
                self._send_json(HTTPStatus.OK, state.handle_request(body, is_chat=True))
            elif self.path == "/v1/completions":
                self._send_json(HTTPStatus.OK, state.handle_request(body, is_chat=False))
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found"}})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, f"http://127.0.0.1:{port}"


def run_self_test() -> int:
    """Verify all 5 exactness suites, ``--compare`` pass/fail gates, and canary leak detection."""
    # 0. Verify full 4K/8K/16K/32K haystack prompt sizing & verbatim passkey embedding.
    for target_len in (4096, 8192, 16384, 32768):
        p = build_cold_needle_prompt(target_len, VERBATIM_KV_PASSKEY, salt="sizecheck")
        assert VERBATIM_KV_PASSKEY in p
        approx_words = len(p.split())
        assert approx_words >= target_len // 3, f"haystack too short for {target_len}: {approx_words}"

    # 1. Start clean mock OpenAI server and run full suite (scaled lengths for <2s CI).
    clean_state = _MockExactnessState(inject_canary_leak=False)
    server, thread, base_url = _start_mock_openai_server(clean_state)
    leaky_state = _MockExactnessState(inject_canary_leak=True)
    leaky_srv, leaky_thr, leaky_url = _start_mock_openai_server(leaky_state)
    try:
        report_a = run_certification_suite(
            base_url=base_url,
            endpoint="/v1/chat/completions",
            model="mock-flux-v14",
            needle_lengths=[256, 512, 1024, 2048],
            batch_sizes=[4, 8],
            seed=42,
            top_logprobs=5,
            run_salt="selftest-salt",
        )
        assert report_a["passed"], f"clean self-test failed: {report_a['failures']}"
        warm_case = next(c for c in report_a["cases"] if c["case_id"] == "cold_vs_warm_warm")
        assert warm_case["cached_tokens"] > 0, "expected warm replay prefix cache hit"

        # 2. Verify W=1 == W=8 == W=16 multi-width invariance.
        width_res = run_verify_width_invariance(
            width_urls={"w1": base_url, "w8": base_url, "w16": base_url},
            endpoint="/v1/chat/completions",
            model="mock-flux-v14",
            seed=42,
            top_logprobs=5,
        )
        assert width_res["passed"], f"width invariance failed: {width_res['failures']}"

        # 3. Verify --compare round-trip via JSON files + negative fail-closed tests.
        with tempfile.TemporaryDirectory(prefix="atlas_exactness_selftest_") as tmpdir:
            tmp = Path(tmpdir)
            ref_path = tmp / "ref.json"
            cand_path = tmp / "cand.json"
            ref_path.write_text(json.dumps(report_a), encoding="utf-8")
            cand_path.write_text(json.dumps(report_a), encoding="utf-8")

            cmp_pass = compare_reports(
                json.loads(ref_path.read_text(encoding="utf-8")),
                json.loads(cand_path.read_text(encoding="utf-8")),
            )
            assert cmp_pass["passed"]
            assert cmp_pass["summary"]["trajectory_exact_rate"] == 1.0
            assert cmp_pass["summary"]["first_divergence"] is None
            assert cmp_pass["summary"]["max_kl_divergence"] == 0.0

            # Negative test A: flip one token -> must detect first_divergence & fail closed.
            bad_tok = json.loads(json.dumps(report_a))
            bad_tok["cases"][0]["token_ids"][0] ^= 1
            cmp_tok_fail = compare_reports(report_a, bad_tok)
            assert not cmp_tok_fail["passed"]
            assert cmp_tok_fail["summary"]["first_divergence"] == {
                "case_id": bad_tok["cases"][0]["case_id"],
                "index": 0,
            }

            # Negative test B: perturb top_logprobs so D_KL > 1e-4 -> must fail KL gate.
            bad_kl = json.loads(json.dumps(report_a))
            step0 = bad_kl["cases"][0]["logprobs"][0]
            tok0 = step0["token"]
            step0["logprob"] = math.log(0.90)
            step0["top_logprobs"][tok0] = math.log(0.90)
            step0["top_logprobs"]["<alt_1>"] = math.log(0.099)
            cmp_kl_fail = compare_reports(report_a, bad_kl, require_exact=False, max_kl=1.0e-4)
            assert not cmp_kl_fail["passed"]
            assert cmp_kl_fail["summary"]["max_kl_divergence"] > 1.0e-4

        # 4. Negative test C: canary leakage in mixed batch MUST fail closed.
        # Seed a canary on leaky_srv then run B=4 batch invariance.
        leaky_report = run_certification_suite(
            base_url=leaky_url,
            endpoint="/v1/chat/completions",
            model="mock-flux-v14",
            needle_lengths=[256],
            batch_sizes=[4],
            seed=42,
            top_logprobs=5,
            run_salt="leakcheck",
        )
        # Run once more so last_canary_seen is guaranteed populated on slot 0.
        if leaky_report["passed"]:
            leaky_report = run_certification_suite(
                base_url=leaky_url,
                endpoint="/v1/chat/completions",
                model="mock-flux-v14",
                needle_lengths=[256],
                batch_sizes=[4],
                seed=42,
                top_logprobs=5,
                run_salt="leakcheck",
            )
        assert not leaky_report["passed"], "canary leak was not caught by batch isolation gate"

        print(
            json.dumps(
                {
                    "self_test": "PASS",
                    "cases_passed": report_a["summary"]["passed_cases"],
                    "total_cases": report_a["summary"]["total_cases"],
                    "code_rust_sha256": CANONICAL_CODE_RUST_SHA256,
                    "width_invariance_passed": width_res["passed"],
                    "negative_gates_verified": [
                        "token_divergence_detected",
                        "kl_gate_1e_4_enforced",
                        "canary_leak_rejected",
                    ],
                },
                indent=2,
            )
        )
    finally:
        for srv, thr in ((server, thread), (leaky_srv, leaky_thr)):
            srv.shutdown()
            srv.server_close()
            thr.join(timeout=2.0)
    return 0


def _parse_width_urls(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"error: --width-urls entry `{item}` must be label=url (e.g. w1=http://...)")
        label, url = item.split("=", 1)
        out[label.strip()] = url.strip()
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="In-repo OpenAI-API exactness certification harness for Flux Engine"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8080", help="Live Flux server base URL")
    parser.add_argument(
        "--endpoint",
        choices=("/v1/chat/completions", "/v1/completions"),
        default="/v1/chat/completions",
        help="OpenAI endpoint to probe",
    )
    parser.add_argument("--model", default="default", help="Served model name")
    parser.add_argument(
        "--needle-lengths",
        default="4096,8192,16384,32768",
        help="Comma-separated cold needle haystack lengths in approximate tokens",
    )
    parser.add_argument(
        "--batch-sizes",
        default="4,8",
        help="Comma-separated mixed-batch concurrency sizes tested against B=1",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Deterministic sampling seed")
    parser.add_argument(
        "--top-logprobs",
        type=int,
        default=DEFAULT_TOP_LOGPROBS,
        help="Top-K logprobs requested per step on /v1/chat/completions",
    )
    parser.add_argument(
        "--width-urls",
        help="Comma-separated label=url list for verify-width invariance (e.g. w1=http://...,w8=http://...,w16=http://...)",
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        type=Path,
        help="Compare reference JSON report against one or more candidate JSON reports",
    )
    parser.add_argument("--max-kl", type=float, default=DEFAULT_MAX_KL, help="Maximum permitted step D_KL")
    parser.add_argument(
        "--max-top1-flip-rate",
        type=float,
        default=DEFAULT_MAX_TOP1_FLIP_RATE,
        help="Maximum permitted top-1 flip rate in --compare",
    )
    parser.add_argument(
        "--allow-non-exact",
        action="store_true",
        help="Do not fail --compare solely on trajectory mismatch (still enforces KL & top-1 flip gates)",
    )
    parser.add_argument("--json-out", type=Path, help="Optional path to write full JSON artifact")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run offline self-test against an in-process mock OpenAI HTTP server",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test()

    if not math.isfinite(args.max_kl) or args.max_kl < 0.0:
        raise SystemExit(f"error: --max-kl must be finite and >= 0, got {args.max_kl}")
    if not math.isfinite(args.max_top1_flip_rate) or not (0.0 <= args.max_top1_flip_rate <= 1.0):
        raise SystemExit(
            f"error: --max-top1-flip-rate must be in [0, 1], got {args.max_top1_flip_rate}"
        )

    if args.compare:
        if len(args.compare) < 2:
            raise SystemExit("error: --compare requires at least 2 JSON paths: <ref.json> <cand.json> [...]")
        ref_report = json.loads(args.compare[0].read_text(encoding="utf-8"))
        all_passed = True
        comparisons: dict[str, Any] = {}
        for cand_path in args.compare[1:]:
            cand_report = json.loads(cand_path.read_text(encoding="utf-8"))
            res = compare_reports(
                ref_report=ref_report,
                cand_report=cand_report,
                max_kl=args.max_kl,
                max_top1_flip_rate=args.max_top1_flip_rate,
                require_exact=not args.allow_non_exact,
            )
            comparisons[cand_path.name] = res
            all_passed = all_passed and res["passed"]
        out = {"passed": all_passed, "reference": str(args.compare[0]), "comparisons": comparisons}
        if args.json_out is not None:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(out, indent=2))
        return 0 if all_passed else 1

    if args.width_urls:
        width_map = _parse_width_urls(args.width_urls)
        res = run_verify_width_invariance(
            width_urls=width_map,
            endpoint=args.endpoint,
            model=args.model,
            seed=args.seed,
            top_logprobs=args.top_logprobs,
            max_kl=args.max_kl,
        )
        if args.json_out is not None:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(res, indent=2))
        return 0 if res["passed"] else 1

    needle_lengths = [int(x.strip()) for x in args.needle_lengths.split(",") if x.strip()]
    if any(n <= 0 for n in needle_lengths):
        raise SystemExit(f"error: --needle-lengths values must be > 0, got {needle_lengths}")
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]
    if any(b < 1 for b in batch_sizes):
        raise SystemExit(f"error: --batch-sizes values must be >= 1, got {batch_sizes}")
    report = run_certification_suite(
        base_url=args.base_url,
        endpoint=args.endpoint,
        model=args.model,
        needle_lengths=needle_lengths,
        batch_sizes=batch_sizes,
        seed=args.seed,
        top_logprobs=args.top_logprobs,
        max_kl=args.max_kl,
    )
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "summary": report["summary"], "failures": report["failures"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
