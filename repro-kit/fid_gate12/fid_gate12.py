#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Fail-closed 12-prompt fidelity & first-divergence diagnostic harness (`fid_gate12`).

Addresses all concrete review findings on legacy `fid.py` (`43862170be189f7e`) and
enforces the four strict-contract invariants:

1. Strict Saved-Panel Schema, Provenance, and Raw-Evidence Binding:
   - Every normalized record in `fid.jsonl` must declare required fields with exact types:
     `schema_version == "fid_gate12.v1"`, `index` (`== manifest_row["index"]`),
     `id` (`== manifest_row["id"]`), `system_sha256` (`== manifest_row["system_sha256"]`),
     `user_sha256` (`== manifest_row["user_sha256"]`), `request_sha256` (`64`-hex hash of the
     canonical request body), `raw_response_sha256` (`64`-hex hash of the canonical raw HTTP
     response), `requested_model`, `response_model`, `response_id`, `max_tokens`,
     `top_logprobs_k`, `seed`, `temperature` (`== 0.0`), `enable_thinking` (`== False`),
     `prompt_tokens`, `completion_tokens` (`== len(tokens)`), and `finish_reason` (`"length"`
     or `"stop"`).
   - Omitting any required field (`schema_version`, `index`, `finish_reason`,
     `completion_tokens`, `user_sha256`, `system_sha256`, `request_sha256`,
     `raw_response_sha256`, etc.) fails closed immediately (`FidGateError`).
   - Equal truncated artifacts (e.g., `finish_reason="length"` with `len(tokens) < max_tokens`,
     or omitted `finish_reason` with short `tokens`) are rejected even if both panels have
     the same truncated length.
   - By default (`require_provenance_files=True`), `load_and_validate_panel_jsonl` also
     requires and cross-verifies the companion `.meta.json` (`status == "completed"`,
     matching `manifest_sha256`, `out_sha256`, `raw_out_sha256`, and request config) and
     `.raw.jsonl` (`12` complete rows whose canonical `request` and `response` hashes match
     each record's `request_sha256` and `raw_response_sha256`), and `compare_panels` verifies
     compatible provenance between `ref` and `alt`.

2. Incremental Raw-Response & Failure/Progress Metadata Persistence:
   - `collect_panel` initializes `.meta.json` (`status="in_progress"`, `completed_count=0`,
     `completed_ids=[]`) and truncates/opens `out_path` and `raw_out_path` before the first
     request.
   - Before each request `i`, `.meta.json` is updated on disk with `current_request_id`,
     `current_request_index`, and `current_request_started_elapsed_s`.
   - Immediately after each HTTP response arrives (even before validation!), a raw evidence
     record is appended and `fsync`ed to `.raw.jsonl` so the exact payload that triggered a
     validation failure is preserved on disk.
   - After validation succeeds, the normalized record is appended and `fsync`ed to `out_path`,
     and `.meta.json` is updated with `completed_count`, `completed_ids`, and partial SHA-256
     digests.
   - On any exception (timeout, HTTP error, validation error, or wall cap), `.meta.json` is
     atomically updated with `status="failed"`, `failed_at_index`, `failed_at_id`,
     `failure_error`, `completed_count`, and the SHA-256 of the retained partial `.jsonl` and
     `.raw.jsonl` files before re-raising `FidGateError`.

3. Accurate Client-Bounded Wall Deadline Contract:
   - A socket timeout argument (`urlopen(..., timeout=...)`) bounds individual socket
     operations, not total multi-read HTTP body transfer or server-side execution, and cannot
     cancel in-flight server work.
   - `default_http_post` enforces an explicit client-side wall-clock deadline across chunked
     reads and runs inside a daemon worker thread so `collect_panel` enforces a bounded
     client wait (`min(request_timeout_s, remaining_wall_s)`).
   - Whenever a request exceeds its client deadline or errors out, `.raw.jsonl` and `.meta.json`
     explicitly record `deadline_exceeded`, `uncertain_server_work=True`, and
     `server_cancellation_guaranteed=False` (never claiming server cancellation).

4. Descriptive Comparisons by Default & Explicit Threshold Opt-In:
   - `compare_panels` defaults to `enforce_gates=False`:
     - Reports `evidence_valid=True` separately from
       `numerical_acceptance="not_evaluated"` and `gate_passed=None` (never emitting
       `gate_passed=True` when threshold checking is disabled).
     - Complete, valid sequences that stop at different valid EOS lengths (`finish_reason="stop"`
       on at least the shorter sequence, while a `finish_reason="length"` sequence has full
       `max_tokens`) are recorded descriptively (`diverge_at = min(len_a, len_b)`,
       `diverge_reason = "eos_length_difference(...)"`) and are never treated as a
       data-integrity failure; when `--enforce-gates` is explicitly enabled, an EOS length
       difference is evaluated using the reference model's in-bounds top1-top2 margin at
       `div_pos` (or `alt`'s margin at `div_pos` when `ref` stopped first with EOS).
   - Keeps explicit `token_ids` vs string-only `top_logprobs` qualifications and fails closed
     on ambiguous decoded-string collisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import socket
import sys
import threading
import time
from typing import Any, Callable
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST_PATH = HERE / "fid_gate12_manifest.jsonl"
DEFAULT_MANIFEST_SHA256 = "34f27ce71072c56a9d3fd34216500b040429197775f9b2beb7148ce0aee4de65"

PANEL_SCHEMA_VERSION = "fid_gate12.v1"
META_SCHEMA_VERSION = "fid_gate12_meta.v1"
COMPARE_SCHEMA_VERSION = "fid_gate12_compare.v1"

EXPECTED_PANEL_SIZE = 12
DEFAULT_MAX_TOKENS = 64
DEFAULT_TOP_LOGPROBS = 5
DEFAULT_SEED = 1234
DEFAULT_REQUEST_TIMEOUT_S = 180.0
DEFAULT_TOTAL_WALL_CAP_S = 600.0

VALID_FINISH_REASONS = frozenset({"length", "stop"})

REQUIRED_PANEL_RECORD_KEYS = (
    "schema_version",
    "id",
    "index",
    "system_sha256",
    "user_sha256",
    "request_sha256",
    "raw_response_sha256",
    "requested_model",
    "response_model",
    "response_id",
    "max_tokens",
    "top_logprobs_k",
    "seed",
    "temperature",
    "enable_thinking",
    "prompt_tokens",
    "completion_tokens",
    "finish_reason",
    "wall_s",
    "token_identity_mode",
    "top_logprobs_identity_mode",
    "token_ids",
    "tokens",
    "logprob",
    "top",
    "content_sha256",
)


class FidGateError(Exception):
    """Raised when any manifest, response, provenance, or comparison integrity check fails."""


class FidThresholdFailure(Exception):
    """Raised when data integrity passes and explicit threshold evaluation (`--enforce-gates`) fails."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def canonical_json_sha256(obj: Any) -> str:
    return sha256_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def is_sha256_hex(val: Any) -> bool:
    return (
        isinstance(val, str)
        and len(val) == 64
        and all(ch in "0123456789abcdef" for ch in val)
    )


def is_finite_logprob(val: Any) -> bool:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return False
    fval = float(val)
    return math.isfinite(fval) and fval <= 1e-5


def _append_jsonl_fsync(path: Path, obj: dict[str, Any]) -> None:
    line = json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def _write_json_atomic(path: Path, obj: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    with tmp_path.open("w", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(path)


def load_and_validate_manifest(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    expected_sha256: str | None = DEFAULT_MANIFEST_SHA256,
    expected_count: int = EXPECTED_PANEL_SIZE,
) -> list[dict[str, Any]]:
    """Load the 12-prompt manifest and verify its SHA-256, schema, and unique IDs."""
    if not manifest_path.is_file():
        raise FidGateError(f"manifest_not_found: {manifest_path}")
    raw = manifest_path.read_bytes()
    actual_sha = sha256_bytes(raw)
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise FidGateError(
            f"manifest_sha256_mismatch: expected {expected_sha256}, got {actual_sha}"
        )

    lines = [ln for ln in raw.decode("utf-8").splitlines() if ln.strip()]
    if len(lines) != expected_count:
        raise FidGateError(
            f"manifest_count_mismatch: expected {expected_count} rows, got {len(lines)}"
        )

    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for idx, line in enumerate(lines):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FidGateError(f"manifest_invalid_json at line {idx + 1}: {exc}") from exc
        if not isinstance(obj, dict):
            raise FidGateError(f"manifest_row_not_object at line {idx + 1}")

        for req_key in (
            "index",
            "id",
            "system",
            "system_sha256",
            "user",
            "user_sha256",
            "expected_prompt_tokens_nothink",
        ):
            if req_key not in obj:
                raise FidGateError(f"manifest_missing_key '{req_key}' at line {idx + 1}")

        if obj["index"] != idx:
            raise FidGateError(
                f"manifest_index_mismatch at line {idx + 1}: expected {idx}, got {obj['index']}"
            )
        pid = obj["id"]
        if not isinstance(pid, str) or not pid.strip():
            raise FidGateError(f"manifest_empty_id at line {idx + 1}")
        if pid in seen_ids:
            raise FidGateError(f"manifest_duplicate_id: '{pid}'")
        seen_ids.add(pid)

        sys_text = obj["system"]
        user_text = obj["user"]
        if not isinstance(sys_text, str) or not sys_text:
            raise FidGateError(f"manifest_invalid_system for '{pid}'")
        if not isinstance(user_text, str) or not user_text:
            raise FidGateError(f"manifest_invalid_user for '{pid}'")
        if sha256_text(sys_text) != obj["system_sha256"]:
            raise FidGateError(f"manifest_system_sha256_mismatch for '{pid}'")
        if sha256_text(user_text) != obj["user_sha256"]:
            raise FidGateError(f"manifest_user_sha256_mismatch for '{pid}'")

        exp_ptok = obj["expected_prompt_tokens_nothink"]
        if isinstance(exp_ptok, bool) or not isinstance(exp_ptok, int) or exp_ptok <= 0:
            raise FidGateError(f"manifest_invalid_expected_prompt_tokens for '{pid}'")

        rows.append(obj)

    return rows


def build_request_body(
    manifest_row: dict[str, Any],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    top_logprobs_k: int = DEFAULT_TOP_LOGPROBS,
    seed: int = DEFAULT_SEED,
    return_token_ids: bool = True,
    model: str = "default",
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": manifest_row["system"]},
            {"role": "user", "content": manifest_row["user"]},
        ],
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "logprobs": True,
        "top_logprobs": top_logprobs_k,
        "seed": seed,
        "stream": False,
    }
    if return_token_ids:
        body["return_token_ids"] = True
    return body


def validate_and_extract_response(
    manifest_row: dict[str, Any],
    raw_resp: dict[str, Any],
    wall_s: float,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    top_logprobs_k: int = DEFAULT_TOP_LOGPROBS,
    seed: int = DEFAULT_SEED,
    require_token_ids: bool = True,
    enforce_prompt_tokens: bool = True,
    request_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a single `/v1/chat/completions` response and extract a provenance-bound record."""
    pid = manifest_row["id"]
    if not isinstance(raw_resp, dict):
        raise FidGateError(f"[{pid}] response_not_json_object")
    if "error" in raw_resp and raw_resp["error"] is not None:
        raise FidGateError(f"[{pid}] server_returned_error: {raw_resp['error']}")

    if request_body is None:
        request_body = build_request_body(
            manifest_row,
            max_tokens=max_tokens,
            top_logprobs_k=top_logprobs_k,
            seed=seed,
            return_token_ids=require_token_ids,
        )

    requested_model = request_body.get("model")
    if not isinstance(requested_model, str) or not requested_model.strip():
        raise FidGateError(f"[{pid}] invalid_requested_model: {requested_model!r}")

    response_model = raw_resp.get("model")
    if not isinstance(response_model, str) or not response_model.strip():
        raise FidGateError(f"[{pid}] missing_or_invalid_response_model: {response_model!r}")
    response_id = raw_resp.get("id")
    if not isinstance(response_id, str) or not response_id.strip():
        raise FidGateError(f"[{pid}] missing_or_invalid_response_id: {response_id!r}")

    choices = raw_resp.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise FidGateError(
            f"[{pid}] invalid_choices_len: expected 1, got {len(choices) if isinstance(choices, list) else type(choices).__name__}"
        )
    choice = choices[0]
    if not isinstance(choice, dict):
        raise FidGateError(f"[{pid}] choice_not_object")

    finish_reason = choice.get("finish_reason")
    if finish_reason not in VALID_FINISH_REASONS:
        raise FidGateError(
            f"[{pid}] invalid_finish_reason: '{finish_reason}' (expected one of {sorted(VALID_FINISH_REASONS)})"
        )
    flux_stop_reason = choice.get("flux_stop_reason")
    if flux_stop_reason is not None:
        raise FidGateError(
            f"[{pid}] unexpected_flux_stop_reason: '{flux_stop_reason}' (indicates guard/timeout intervention)"
        )

    usage = raw_resp.get("usage")
    if not isinstance(usage, dict):
        raise FidGateError(f"[{pid}] missing_usage_object")
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int) or prompt_tokens <= 0:
        raise FidGateError(f"[{pid}] invalid_prompt_tokens: {prompt_tokens}")
    if (
        isinstance(completion_tokens, bool)
        or not isinstance(completion_tokens, int)
        or completion_tokens <= 0
        or completion_tokens > max_tokens
    ):
        raise FidGateError(
            f"[{pid}] invalid_completion_tokens: {completion_tokens} (max_tokens={max_tokens})"
        )

    exp_ptok = manifest_row.get("expected_prompt_tokens_nothink")
    if enforce_prompt_tokens and exp_ptok is not None and prompt_tokens != exp_ptok:
        raise FidGateError(
            f"[{pid}] prompt_tokens_mismatch: expected {exp_ptok}, got {prompt_tokens}"
        )

    lp_wrapper = choice.get("logprobs")
    if not isinstance(lp_wrapper, dict):
        raise FidGateError(f"[{pid}] missing_logprobs_object")
    lp_content = lp_wrapper.get("content")
    if not isinstance(lp_content, list) or len(lp_content) == 0:
        raise FidGateError(f"[{pid}] empty_or_missing_logprobs_content")

    if len(lp_content) != completion_tokens:
        raise FidGateError(
            f"[{pid}] logprobs_len_mismatch: len(logprobs.content)={len(lp_content)} != usage.completion_tokens={completion_tokens}"
        )
    if finish_reason == "length" and len(lp_content) != max_tokens:
        raise FidGateError(
            f"[{pid}] finish_reason_length_but_short_output: len={len(lp_content)} != max_tokens={max_tokens}"
        )

    raw_token_ids = choice.get("token_ids")
    token_ids: list[int] | None = None
    if raw_token_ids is not None and raw_token_ids != []:
        if not isinstance(raw_token_ids, list) or len(raw_token_ids) != len(lp_content):
            raise FidGateError(
                f"[{pid}] token_ids_len_mismatch: len(token_ids)={len(raw_token_ids) if isinstance(raw_token_ids, list) else 'invalid'} != len(logprobs.content)={len(lp_content)}"
            )
        for tidx, tid in enumerate(raw_token_ids):
            if isinstance(tid, bool) or not isinstance(tid, int) or tid < 0:
                raise FidGateError(f"[{pid}] invalid_token_id at pos {tidx}: {tid!r}")
        token_ids = list(raw_token_ids)
    elif require_token_ids:
        raise FidGateError(
            f"[{pid}] missing_token_ids: choice.token_ids is empty or missing while require_token_ids=True"
        )

    tokens: list[str] = []
    logprobs: list[float] = []
    top_list: list[list[list[Any]]] = []

    for pos, entry in enumerate(lp_content):
        if not isinstance(entry, dict):
            raise FidGateError(f"[{pid}] logprobs_entry_not_object at pos {pos}")
        tok_str = entry.get("token")
        lp_val = entry.get("logprob")
        top_k = entry.get("top_logprobs")

        if not isinstance(tok_str, str):
            raise FidGateError(f"[{pid}] token_not_string at pos {pos}: {tok_str!r}")
        if not is_finite_logprob(lp_val):
            raise FidGateError(f"[{pid}] non_finite_or_positive_logprob at pos {pos}: {lp_val!r}")
        if not isinstance(top_k, list) or len(top_k) != top_logprobs_k:
            raise FidGateError(
                f"[{pid}] invalid_top_logprobs_len at pos {pos}: expected {top_logprobs_k}, got {len(top_k) if isinstance(top_k, list) else type(top_k).__name__}"
            )

        pos_top: list[list[Any]] = []
        seen_cand_tokens: set[str] = set()
        prev_lp: float | None = None
        for rank, cand in enumerate(top_k):
            if not isinstance(cand, dict):
                raise FidGateError(f"[{pid}] top_logprobs_cand_not_object at pos {pos} rank {rank}")
            ctok = cand.get("token")
            clp = cand.get("logprob")
            if not isinstance(ctok, str):
                raise FidGateError(
                    f"[{pid}] top_logprobs_token_not_string at pos {pos} rank {rank}: {ctok!r}"
                )
            if not is_finite_logprob(clp):
                raise FidGateError(
                    f"[{pid}] top_logprobs_non_finite at pos {pos} rank {rank}: {clp!r}"
                )
            f_clp = float(clp)
            if prev_lp is not None and f_clp > prev_lp + 1e-6:
                raise FidGateError(
                    f"[{pid}] top_logprobs_not_sorted_descending at pos {pos} rank {rank}: {f_clp} > {prev_lp}"
                )
            if ctok in seen_cand_tokens:
                raise FidGateError(
                    f"[{pid}] ambiguous_duplicate_token_string_in_top_logprobs at pos {pos}: {ctok!r} (chat API omits token IDs in top_logprobs; duplicate decoded strings are ambiguous)"
                )
            seen_cand_tokens.add(ctok)
            prev_lp = f_clp
            pos_top.append([ctok, f_clp])

        if tok_str != pos_top[0][0]:
            raise FidGateError(
                f"[{pid}] greedy_sampled_token_mismatch_top1 at pos {pos}: sampled={tok_str!r} vs top1={pos_top[0][0]!r}"
            )
        if abs(float(lp_val) - pos_top[0][1]) > 1e-5:
            raise FidGateError(
                f"[{pid}] sampled_logprob_mismatch_top1 at pos {pos}: sampled={lp_val} vs top1={pos_top[0][1]}"
            )

        tokens.append(tok_str)
        logprobs.append(float(lp_val))
        top_list.append(pos_top)

    msg = choice.get("message") or {}
    content_text = msg.get("content") if isinstance(msg, dict) else None
    if not isinstance(content_text, str):
        raise FidGateError(f"[{pid}] missing_message_content_string")

    return {
        "schema_version": PANEL_SCHEMA_VERSION,
        "id": pid,
        "index": manifest_row["index"],
        "system_sha256": manifest_row["system_sha256"],
        "user_sha256": manifest_row["user_sha256"],
        "request_sha256": canonical_json_sha256(request_body),
        "raw_response_sha256": canonical_json_sha256(raw_resp),
        "requested_model": requested_model,
        "response_model": response_model,
        "response_id": response_id,
        "max_tokens": max_tokens,
        "top_logprobs_k": top_logprobs_k,
        "seed": seed,
        "temperature": 0.0,
        "enable_thinking": False,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "wall_s": round(float(wall_s), 6),
        "token_identity_mode": "token_ids+decoded_str" if token_ids is not None else "decoded_str_only",
        "top_logprobs_identity_mode": "decoded_str_post_transform_host_path",
        "token_ids": token_ids,
        "tokens": tokens,
        "logprob": logprobs,
        "top": top_list,
        "content_sha256": sha256_text(content_text),
    }


def default_http_post(url: str, body: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    """Perform a single HTTP POST with an explicit client-side wall-clock deadline across reads.

    Note: `socket` timeouts alone only bound individual `recv()` syscalls, not slow trickle
    reads or server-side execution after client disconnect. This function checks a monotonic
    client deadline before and during chunked body reads. It does NOT claim server-side
    request cancellation on timeout.
    """
    if timeout_s <= 0:
        raise FidGateError(
            f"client_wall_deadline_exceeded_before_send: timeout_s={timeout_s:.4f}s "
            "(uncertain_server_work=False, server_cancellation_guaranteed=False)"
        )
    deadline = time.monotonic() + timeout_s
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        rem = deadline - time.monotonic()
        if rem <= 0:
            raise TimeoutError("client_deadline_expired_before_open")
        with urllib.request.urlopen(req, timeout=max(0.001, rem)) as resp:
            status = getattr(resp, "status", 200)
            chunks: list[bytes] = []
            while True:
                rem = deadline - time.monotonic()
                if rem <= 0:
                    raise TimeoutError("client_wall_deadline_exceeded_during_body_read")
                sock_fp = getattr(resp, "fp", None)
                raw_sock = getattr(sock_fp, "raw", None)
                inner_sock = getattr(raw_sock, "_sock", None)
                if inner_sock is not None and hasattr(inner_sock, "settimeout"):
                    inner_sock.settimeout(max(0.001, rem))
                chunk = resp.read(65536)
                if not chunk:
                    break
                chunks.append(chunk)
            raw_bytes = b"".join(chunks)
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        raise FidGateError(
            f"http_status_{exc.code}: {err_body[:500]} "
            "(uncertain_server_work=True, server_cancellation_guaranteed=False)"
        ) from exc
    except (TimeoutError, socket.timeout) as exc:
        raise FidGateError(
            f"client_wall_deadline_exceeded_after_{timeout_s:.2f}s: {exc} "
            "(uncertain_server_work=True, server_cancellation_guaranteed=False)"
        ) from exc
    except urllib.error.URLError as exc:
        if isinstance(getattr(exc, "reason", None), (TimeoutError, socket.timeout)):
            raise FidGateError(
                f"client_wall_deadline_exceeded_after_{timeout_s:.2f}s: {exc.reason} "
                "(uncertain_server_work=True, server_cancellation_guaranteed=False)"
            ) from exc
        raise FidGateError(
            f"http_url_error: {exc} "
            "(uncertain_server_work=True, server_cancellation_guaranteed=False)"
        ) from exc

    if status != 200:
        raise FidGateError(f"http_non_200_status: {status}")
    try:
        return json.loads(raw_bytes.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise FidGateError(f"response_invalid_json: {exc}") from exc


def _invoke_with_client_wall_deadline(
    http_post: Callable[[str, dict[str, Any], float], dict[str, Any]],
    url: str,
    body: dict[str, Any],
    effective_timeout_s: float,
    clock: Callable[[], float],
) -> tuple[dict[str, Any] | None, float, bool, str | None]:
    """Invoke `http_post` with a bounded client wall wait when using the real monotonic clock.

    Returns `(raw_resp, elapsed_wall_s, deadline_exceeded, error_str)`.
    If `clock is time.monotonic`, runs `http_post` in a daemon thread and waits at most
    `effective_timeout_s` on the client side so a hanging transport/mock cannot block the
    caller indefinitely. When a synthetic test clock is injected (`clock is not time.monotonic`),
    calls `http_post` synchronously and checks `clock()` elapsed time.
    """
    req_t0 = clock()
    if clock is time.monotonic:
        q: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

        def _worker() -> None:
            try:
                res = http_post(url, body, effective_timeout_s)
                q.put(("ok", res))
            except Exception as exc:  # noqa: BLE001
                q.put(("err", exc))

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout=max(0.0, effective_timeout_s))
        req_wall = clock() - req_t0
        if t.is_alive():
            return (
                None,
                req_wall,
                True,
                f"client_wall_deadline_exceeded: client wait {req_wall:.3f}s >= cap {effective_timeout_s:.3f}s "
                "(worker thread still blocked; uncertain_server_work=True, server_cancellation_guaranteed=False)",
            )
        try:
            kind, payload = q.get_nowait()
        except queue.Empty:
            return (
                None,
                req_wall,
                True,
                f"client_wall_deadline_exceeded: empty worker queue after {req_wall:.3f}s "
                "(uncertain_server_work=True, server_cancellation_guaranteed=False)",
            )
        if kind == "err":
            err_msg = str(payload)
            is_dl = "deadline_exceeded" in err_msg or "timeout" in err_msg.lower()
            return (None, req_wall, is_dl, err_msg)
        return (payload, req_wall, False, None)
    else:
        try:
            res = http_post(url, body, effective_timeout_s)
        except Exception as exc:  # noqa: BLE001
            req_wall = clock() - req_t0
            err_msg = str(exc)
            is_dl = "deadline_exceeded" in err_msg or "timeout" in err_msg.lower() or req_wall > effective_timeout_s
            return (None, req_wall, is_dl, err_msg)
        req_wall = clock() - req_t0
        return (res, req_wall, False, None)


def collect_panel(
    url: str,
    out_path: Path,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    expected_manifest_sha256: str | None = DEFAULT_MANIFEST_SHA256,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    top_logprobs_k: int = DEFAULT_TOP_LOGPROBS,
    seed: int = DEFAULT_SEED,
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
    total_wall_cap_s: float = DEFAULT_TOTAL_WALL_CAP_S,
    require_token_ids: bool = True,
    enforce_prompt_tokens: bool = True,
    raw_out_path: Path | None = None,
    meta_out_path: Path | None = None,
    http_post: Callable[[str, dict[str, Any], float], dict[str, Any]] = default_http_post,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Execute the pinned 12-prompt panel sequentially with incremental persistence and fail-closed validation."""
    if max_tokens <= 0 or max_tokens > 256:
        raise FidGateError(f"invalid_max_tokens_cap: {max_tokens}")
    if top_logprobs_k <= 0 or top_logprobs_k > 20:
        raise FidGateError(f"invalid_top_logprobs_k: {top_logprobs_k}")
    if request_timeout_s <= 0 or total_wall_cap_s <= 0:
        raise FidGateError("invalid_timeout_caps")

    manifest_rows = load_and_validate_manifest(
        manifest_path=manifest_path,
        expected_sha256=expected_manifest_sha256,
        expected_count=len(
            [
                ln
                for ln in manifest_path.read_bytes().decode("utf-8").splitlines()
                if ln.strip()
            ]
        )
        if expected_manifest_sha256 is None and manifest_path.is_file()
        else EXPECTED_PANEL_SIZE,
    )
    manifest_sha = sha256_bytes(manifest_path.read_bytes())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_out_path is None:
        raw_out_path = out_path.with_suffix(".raw.jsonl")
    if meta_out_path is None:
        meta_out_path = out_path.with_suffix(".meta.json")

    # Initialize empty incremental files on disk before any network activity
    out_path.write_bytes(b"")
    raw_out_path.write_bytes(b"")

    t_start = clock()
    completed_ids: list[str] = []

    deadline_contract = (
        "Client-enforced wall-clock wait cap per request (`min(request_timeout_s, remaining_wall_s)`) "
        "and across the whole panel (`total_wall_cap_s`). Socket timeouts and client disconnects do "
        "NOT guarantee server-side cancellation or server completion; any timeout or transport abort "
        "sets `uncertain_server_work=True` and `server_cancellation_guaranteed=False`."
    )
    api_qualification = (
        "Chat completions top_logprobs forces the host sampling path and exposes "
        "post-transform/suppressed logprobs with decoded token strings (bytes=None); "
        "choice.token_ids provides exact sampled integer token IDs when return_token_ids=True."
    )

    def _build_meta(
        status: str,
        *,
        current_idx: int | None = None,
        current_id: str | None = None,
        current_started_s: float | None = None,
        failed_idx: int | None = None,
        failed_id: str | None = None,
        failure_error: str | None = None,
        deadline_exceeded: bool = False,
        uncertain_server_work: bool = False,
    ) -> dict[str, Any]:
        now_elapsed = round(float(clock() - t_start), 6)
        return {
            "schema_version": META_SCHEMA_VERSION,
            "status": status,
            "url": url,
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha,
            "expected_panel_size": len(manifest_rows),
            "panel_size": len(completed_ids),
            "completed_count": len(completed_ids),
            "completed_ids": list(completed_ids),
            "current_request_index": current_idx,
            "current_request_id": current_id,
            "current_request_started_elapsed_s": (
                None if current_started_s is None else round(float(current_started_s), 6)
            ),
            "failed_at_index": failed_idx,
            "failed_at_id": failed_id,
            "failure_error": failure_error,
            "deadline_exceeded": deadline_exceeded,
            "uncertain_server_work": uncertain_server_work,
            "server_cancellation_guaranteed": False,
            "max_tokens": max_tokens,
            "top_logprobs_k": top_logprobs_k,
            "seed": seed,
            "request_timeout_s": request_timeout_s,
            "total_wall_cap_s": total_wall_cap_s,
            "require_token_ids": require_token_ids,
            "enforce_prompt_tokens": enforce_prompt_tokens,
            "total_wall_s": now_elapsed,
            "out_path": str(out_path),
            "raw_out_path": str(raw_out_path),
            "out_sha256": sha256_bytes(out_path.read_bytes()) if out_path.is_file() else None,
            "raw_out_sha256": sha256_bytes(raw_out_path.read_bytes()) if raw_out_path.is_file() else None,
            "deadline_contract": deadline_contract,
            "api_qualification": api_qualification,
        }

    _write_json_atomic(meta_out_path, _build_meta("in_progress"))

    for row in manifest_rows:
        pid = row["id"]
        idx = row["index"]
        elapsed_before = clock() - t_start
        remaining_wall = total_wall_cap_s - elapsed_before
        if remaining_wall <= 0:
            err_msg = (
                f"total_wall_cap_exceeded before prompt '{pid}': "
                f"elapsed={elapsed_before:.2f}s >= cap={total_wall_cap_s:.2f}s"
            )
            _append_jsonl_fsync(
                raw_out_path,
                {
                    "id": pid,
                    "index": idx,
                    "status": "aborted_before_send",
                    "request": None,
                    "request_sha256": None,
                    "wall_s": 0.0,
                    "effective_timeout_s": 0.0,
                    "deadline_exceeded": True,
                    "uncertain_server_work": False,
                    "server_cancellation_guaranteed": False,
                    "response": None,
                    "raw_response_sha256": None,
                    "error": err_msg,
                },
            )
            _write_json_atomic(
                meta_out_path,
                _build_meta(
                    "failed",
                    failed_idx=idx,
                    failed_id=pid,
                    failure_error=err_msg,
                    deadline_exceeded=True,
                    uncertain_server_work=False,
                ),
            )
            raise FidGateError(err_msg)

        effective_timeout = min(request_timeout_s, remaining_wall)
        body = build_request_body(
            row,
            max_tokens=max_tokens,
            top_logprobs_k=top_logprobs_k,
            seed=seed,
            return_token_ids=require_token_ids,
        )
        req_sha = canonical_json_sha256(body)

        # Persist pre-request progress state before sending over the wire
        _write_json_atomic(
            meta_out_path,
            _build_meta(
                "in_progress",
                current_idx=idx,
                current_id=pid,
                current_started_s=elapsed_before,
            ),
        )

        raw_resp, req_wall, dl_exceeded, transport_err = _invoke_with_client_wall_deadline(
            http_post=http_post,
            url=url,
            body=body,
            effective_timeout_s=effective_timeout,
            clock=clock,
        )

        total_after = clock() - t_start
        over_req_cap = req_wall > request_timeout_s
        over_total_cap = total_after > total_wall_cap_s
        any_deadline_exceeded = dl_exceeded or over_req_cap or over_total_cap

        if transport_err is not None:
            err_msg = f"[{pid}] {transport_err}"
            _append_jsonl_fsync(
                raw_out_path,
                {
                    "id": pid,
                    "index": idx,
                    "status": "transport_or_deadline_error",
                    "request": body,
                    "request_sha256": req_sha,
                    "wall_s": round(float(req_wall), 6),
                    "effective_timeout_s": round(float(effective_timeout), 6),
                    "deadline_exceeded": any_deadline_exceeded,
                    "uncertain_server_work": True,
                    "server_cancellation_guaranteed": False,
                    "response": None,
                    "raw_response_sha256": None,
                    "error": err_msg,
                },
            )
            _write_json_atomic(
                meta_out_path,
                _build_meta(
                    "failed",
                    failed_idx=idx,
                    failed_id=pid,
                    failure_error=err_msg,
                    deadline_exceeded=any_deadline_exceeded,
                    uncertain_server_work=True,
                ),
            )
            raise FidGateError(err_msg)

        raw_resp_sha = (
            canonical_json_sha256(raw_resp) if isinstance(raw_resp, dict) else None
        )

        if over_req_cap or over_total_cap:
            if over_req_cap:
                err_msg = (
                    f"[{pid}] request_wall_exceeded: {req_wall:.2f}s > {request_timeout_s:.2f}s "
                    "(uncertain_server_work=False, server_cancellation_guaranteed=False)"
                )
            else:
                err_msg = (
                    f"[{pid}] total_wall_cap_exceeded after request: {total_after:.2f}s > {total_wall_cap_s:.2f}s "
                    "(uncertain_server_work=False, server_cancellation_guaranteed=False)"
                )
            _append_jsonl_fsync(
                raw_out_path,
                {
                    "id": pid,
                    "index": idx,
                    "status": "deadline_exceeded_after_response",
                    "request": body,
                    "request_sha256": req_sha,
                    "wall_s": round(float(req_wall), 6),
                    "effective_timeout_s": round(float(effective_timeout), 6),
                    "deadline_exceeded": True,
                    "uncertain_server_work": False,
                    "server_cancellation_guaranteed": False,
                    "response": raw_resp,
                    "raw_response_sha256": raw_resp_sha,
                    "error": err_msg,
                },
            )
            _write_json_atomic(
                meta_out_path,
                _build_meta(
                    "failed",
                    failed_idx=idx,
                    failed_id=pid,
                    failure_error=err_msg,
                    deadline_exceeded=True,
                    uncertain_server_work=False,
                ),
            )
            raise FidGateError(err_msg)

        # Validate response; whether validation succeeds or fails, persist the raw response first!
        try:
            assert raw_resp is not None
            rec = validate_and_extract_response(
                manifest_row=row,
                raw_resp=raw_resp,
                wall_s=req_wall,
                max_tokens=max_tokens,
                top_logprobs_k=top_logprobs_k,
                seed=seed,
                require_token_ids=require_token_ids,
                enforce_prompt_tokens=enforce_prompt_tokens,
                request_body=body,
            )
        except Exception as exc:  # noqa: BLE001
            err_msg = str(exc) if isinstance(exc, FidGateError) else f"[{pid}] validation_exception: {exc}"
            _append_jsonl_fsync(
                raw_out_path,
                {
                    "id": pid,
                    "index": idx,
                    "status": "validation_error",
                    "request": body,
                    "request_sha256": req_sha,
                    "wall_s": round(float(req_wall), 6),
                    "effective_timeout_s": round(float(effective_timeout), 6),
                    "deadline_exceeded": False,
                    "uncertain_server_work": False,
                    "server_cancellation_guaranteed": False,
                    "response": raw_resp,
                    "raw_response_sha256": raw_resp_sha,
                    "error": err_msg,
                },
            )
            _write_json_atomic(
                meta_out_path,
                _build_meta(
                    "failed",
                    failed_idx=idx,
                    failed_id=pid,
                    failure_error=err_msg,
                    deadline_exceeded=False,
                    uncertain_server_work=False,
                ),
            )
            if isinstance(exc, FidGateError):
                raise
            raise FidGateError(err_msg) from exc

        # Persist raw response and validated record immediately after each request!
        _append_jsonl_fsync(
            raw_out_path,
            {
                "id": pid,
                "index": idx,
                "status": "ok",
                "request": body,
                "request_sha256": req_sha,
                "wall_s": round(float(req_wall), 6),
                "effective_timeout_s": round(float(effective_timeout), 6),
                "deadline_exceeded": False,
                "uncertain_server_work": False,
                "server_cancellation_guaranteed": False,
                "response": raw_resp,
                "raw_response_sha256": raw_resp_sha,
                "error": None,
            },
        )
        _append_jsonl_fsync(out_path, rec)
        completed_ids.append(pid)

        _write_json_atomic(
            meta_out_path,
            _build_meta("in_progress"),
        )

    if len(completed_ids) != len(manifest_rows):
        err_msg = f"incomplete_panel_collected: {len(completed_ids)}/{len(manifest_rows)}"
        _write_json_atomic(
            meta_out_path,
            _build_meta("failed", failure_error=err_msg),
        )
        raise FidGateError(err_msg)

    final_meta = _build_meta("completed")
    _write_json_atomic(meta_out_path, final_meta)
    return final_meta


def load_and_validate_panel_jsonl(
    jsonl_path: Path,
    manifest_rows: list[dict[str, Any]],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    top_logprobs_k: int = DEFAULT_TOP_LOGPROBS,
    require_token_ids: bool = False,
    enforce_prompt_tokens: bool = True,
    require_provenance_files: bool = True,
    raw_jsonl_path: Path | None = None,
    meta_json_path: Path | None = None,
    manifest_sha256: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Load a saved `fid.jsonl` file and strictly verify schema, completeness, provenance, and raw evidence."""
    if not jsonl_path.is_file():
        raise FidGateError(f"panel_file_not_found: {jsonl_path}")

    raw_panel_bytes = jsonl_path.read_bytes()
    panel_sha256 = sha256_bytes(raw_panel_bytes)
    lines = [ln for ln in raw_panel_bytes.decode("utf-8").splitlines() if ln.strip()]
    if len(lines) == 0:
        raise FidGateError(f"panel_file_empty: {jsonl_path}")

    manifest_by_id = {r["id"]: r for r in manifest_rows}
    expected_ids = [r["id"] for r in manifest_rows]

    by_id: dict[str, dict[str, Any]] = {}
    actual_ids_in_order: list[str] = []

    for lno, line in enumerate(lines, 1):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FidGateError(f"[{jsonl_path.name}:{lno}] invalid_json: {exc}") from exc
        if not isinstance(rec, dict):
            raise FidGateError(f"[{jsonl_path.name}:{lno}] row_not_object")

        for req_key in REQUIRED_PANEL_RECORD_KEYS:
            if req_key not in rec:
                raise FidGateError(
                    f"[{jsonl_path.name}:{lno}] missing_required_field: '{req_key}'"
                )

        if rec["schema_version"] != PANEL_SCHEMA_VERSION:
            raise FidGateError(
                f"[{jsonl_path.name}:{lno}] invalid_schema_version: {rec['schema_version']!r} != {PANEL_SCHEMA_VERSION!r}"
            )

        pid = rec.get("id")
        if not isinstance(pid, str) or not pid:
            raise FidGateError(f"[{jsonl_path.name}:{lno}] missing_or_invalid_id")
        if pid in by_id:
            raise FidGateError(f"[{jsonl_path.name}:{lno}] duplicate_id: '{pid}'")
        if pid not in manifest_by_id:
            raise FidGateError(f"[{jsonl_path.name}:{lno}] extra_unexpected_id: '{pid}'")

        mrow = manifest_by_id[pid]
        rec_index = rec.get("index")
        if isinstance(rec_index, bool) or not isinstance(rec_index, int) or rec_index != mrow["index"]:
            raise FidGateError(
                f"[{jsonl_path.name}:{pid}] index_mismatch: expected {mrow['index']}, got {rec_index!r}"
            )

        if rec.get("system_sha256") != mrow["system_sha256"]:
            raise FidGateError(
                f"[{jsonl_path.name}:{pid}] system_sha256_mismatch: {rec.get('system_sha256')!r} != {mrow['system_sha256']!r}"
            )
        if rec.get("user_sha256") != mrow["user_sha256"]:
            raise FidGateError(
                f"[{jsonl_path.name}:{pid}] user_sha256_mismatch: {rec.get('user_sha256')!r} != {mrow['user_sha256']!r}"
            )

        if not is_sha256_hex(rec.get("request_sha256")):
            raise FidGateError(
                f"[{jsonl_path.name}:{pid}] invalid_request_sha256: {rec.get('request_sha256')!r}"
            )
        if not is_sha256_hex(rec.get("raw_response_sha256")):
            raise FidGateError(
                f"[{jsonl_path.name}:{pid}] invalid_raw_response_sha256: {rec.get('raw_response_sha256')!r}"
            )
        if not is_sha256_hex(rec.get("content_sha256")):
            raise FidGateError(
                f"[{jsonl_path.name}:{pid}] invalid_content_sha256: {rec.get('content_sha256')!r}"
            )

        for str_key in ("requested_model", "response_model", "response_id"):
            sval = rec.get(str_key)
            if not isinstance(sval, str) or not sval.strip():
                raise FidGateError(
                    f"[{jsonl_path.name}:{pid}] invalid_{str_key}: {sval!r}"
                )

        if rec.get("max_tokens") != max_tokens:
            raise FidGateError(
                f"[{jsonl_path.name}:{pid}] max_tokens_mismatch: record={rec.get('max_tokens')!r} != expected={max_tokens}"
            )
        if rec.get("top_logprobs_k") != top_logprobs_k:
            raise FidGateError(
                f"[{jsonl_path.name}:{pid}] top_logprobs_k_mismatch: record={rec.get('top_logprobs_k')!r} != expected={top_logprobs_k}"
            )
        seed_val = rec.get("seed")
        if isinstance(seed_val, bool) or not isinstance(seed_val, int):
            raise FidGateError(f"[{jsonl_path.name}:{pid}] invalid_seed: {seed_val!r}")
        temp_val = rec.get("temperature")
        if isinstance(temp_val, bool) or not isinstance(temp_val, (int, float)) or float(temp_val) != 0.0:
            raise FidGateError(
                f"[{jsonl_path.name}:{pid}] invalid_temperature: {temp_val!r} (expected 0.0)"
            )
        if rec.get("enable_thinking") is not False:
            raise FidGateError(
                f"[{jsonl_path.name}:{pid}] invalid_enable_thinking: {rec.get('enable_thinking')!r} (expected False)"
            )

        expected_req_body = build_request_body(
            mrow,
            max_tokens=max_tokens,
            top_logprobs_k=top_logprobs_k,
            seed=seed_val,
            return_token_ids=(rec.get("token_ids") is not None),
            model=rec["requested_model"],
        )
        expected_req_sha = canonical_json_sha256(expected_req_body)
        if rec["request_sha256"] != expected_req_sha:
            raise FidGateError(
                f"[{jsonl_path.name}:{pid}] request_sha256_mismatch_expected_contract: "
                f"{rec['request_sha256']} != {expected_req_sha}"
            )

        ptok = rec.get("prompt_tokens")
        if isinstance(ptok, bool) or not isinstance(ptok, int) or ptok <= 0:
            raise FidGateError(f"[{jsonl_path.name}:{pid}] invalid_prompt_tokens: {ptok!r}")
        if enforce_prompt_tokens and ptok != mrow["expected_prompt_tokens_nothink"]:
            raise FidGateError(
                f"[{jsonl_path.name}:{pid}] prompt_tokens_mismatch: expected {mrow['expected_prompt_tokens_nothink']}, got {ptok}"
            )

        finish_reason = rec.get("finish_reason")
        if finish_reason not in VALID_FINISH_REASONS:
            raise FidGateError(
                f"[{jsonl_path.name}:{pid}] invalid_or_missing_finish_reason: {finish_reason!r}"
            )

        tokens = rec.get("tokens")
        lps = rec.get("logprob")
        top = rec.get("top")
        if not isinstance(tokens, list) or len(tokens) == 0 or len(tokens) > max_tokens:
            raise FidGateError(
                f"[{jsonl_path.name}:{pid}] invalid_tokens_list_len: {len(tokens) if isinstance(tokens, list) else 'not_list'}"
            )

        comp_tok = rec.get("completion_tokens")
        if (
            isinstance(comp_tok, bool)
            or not isinstance(comp_tok, int)
            or comp_tok != len(tokens)
        ):
            raise FidGateError(
                f"[{jsonl_path.name}:{pid}] completion_tokens_mismatch: completion_tokens={comp_tok!r} != len(tokens)={len(tokens)}"
            )

        if not isinstance(lps, list) or len(lps) != len(tokens):
            raise FidGateError(
                f"[{jsonl_path.name}:{pid}] logprob_len_mismatch: {len(lps) if isinstance(lps, list) else 'not_list'} != {len(tokens)}"
            )
        if not isinstance(top, list) or len(top) != len(tokens):
            raise FidGateError(
                f"[{jsonl_path.name}:{pid}] top_len_mismatch: {len(top) if isinstance(top, list) else 'not_list'} != {len(tokens)}"
            )
        if finish_reason == "length" and len(tokens) != max_tokens:
            raise FidGateError(
                f"[{jsonl_path.name}:{pid}] finish_reason_length_but_tokens_len={len(tokens)} != {max_tokens}"
            )

        tids = rec.get("token_ids")
        if tids is not None:
            if not isinstance(tids, list) or len(tids) != len(tokens):
                raise FidGateError(
                    f"[{jsonl_path.name}:{pid}] token_ids_len_mismatch: {len(tids) if isinstance(tids, list) else 'not_list'} != {len(tokens)}"
                )
            for pos, tid in enumerate(tids):
                if isinstance(tid, bool) or not isinstance(tid, int) or tid < 0:
                    raise FidGateError(
                        f"[{jsonl_path.name}:{pid}] invalid_token_id at pos {pos}: {tid!r}"
                    )
            if rec.get("token_identity_mode") != "token_ids+decoded_str":
                raise FidGateError(
                    f"[{jsonl_path.name}:{pid}] invalid_token_identity_mode: {rec.get('token_identity_mode')!r}"
                )
        else:
            if require_token_ids:
                raise FidGateError(f"[{jsonl_path.name}:{pid}] missing_required_token_ids")
            if rec.get("token_identity_mode") != "decoded_str_only":
                raise FidGateError(
                    f"[{jsonl_path.name}:{pid}] invalid_token_identity_mode: {rec.get('token_identity_mode')!r}"
                )

        for pos, (tok_s, lp_v, top_pos) in enumerate(zip(tokens, lps, top)):
            if not isinstance(tok_s, str):
                raise FidGateError(f"[{jsonl_path.name}:{pid}] token_not_str at pos {pos}")
            if not is_finite_logprob(lp_v):
                raise FidGateError(f"[{jsonl_path.name}:{pid}] non_finite_logprob at pos {pos}: {lp_v!r}")
            if not isinstance(top_pos, list) or len(top_pos) != top_logprobs_k:
                raise FidGateError(
                    f"[{jsonl_path.name}:{pid}] invalid_top_k_at_pos_{pos}: expected {top_logprobs_k}"
                )
            seen_cands: set[str] = set()
            prev_c_lp: float | None = None
            for rank, pair in enumerate(top_pos):
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise FidGateError(
                        f"[{jsonl_path.name}:{pid}] invalid_top_pair at pos {pos} rank {rank}"
                    )
                ctok, clp = pair[0], pair[1]
                if not isinstance(ctok, str) or not is_finite_logprob(clp):
                    raise FidGateError(
                        f"[{jsonl_path.name}:{pid}] invalid_top_entry at pos {pos} rank {rank}: {pair!r}"
                    )
                f_clp = float(clp)
                if prev_c_lp is not None and f_clp > prev_c_lp + 1e-6:
                    raise FidGateError(
                        f"[{jsonl_path.name}:{pid}] top_not_descending at pos {pos} rank {rank}: {f_clp} > {prev_c_lp}"
                    )
                if ctok in seen_cands:
                    raise FidGateError(
                        f"[{jsonl_path.name}:{pid}] duplicate_top_token_str at pos {pos}: {ctok!r}"
                    )
                seen_cands.add(ctok)
                prev_c_lp = f_clp
            if tok_s != top_pos[0][0]:
                raise FidGateError(
                    f"[{jsonl_path.name}:{pid}] sampled_token_mismatch_top1 at pos {pos}: {tok_s!r} != {top_pos[0][0]!r}"
                )
            if abs(float(lp_v) - float(top_pos[0][1])) > 1e-5:
                raise FidGateError(
                    f"[{jsonl_path.name}:{pid}] sampled_logprob_mismatch_top1 at pos {pos}: {lp_v} != {top_pos[0][1]}"
                )

        by_id[pid] = rec
        actual_ids_in_order.append(pid)

    missing_ids = [pid for pid in expected_ids if pid not in by_id]
    if missing_ids:
        raise FidGateError(f"[{jsonl_path.name}] missing_manifest_ids: {missing_ids}")
    if actual_ids_in_order != expected_ids:
        raise FidGateError(
            f"[{jsonl_path.name}] panel_id_order_mismatch: expected {expected_ids}, got {actual_ids_in_order}"
        )

    # Check consistency of run-level request parameters across all 12 rows in the panel
    models_seen = {rec["requested_model"] for rec in by_id.values()}
    if len(models_seen) != 1:
        raise FidGateError(f"[{jsonl_path.name}] inconsistent_requested_models_in_panel: {sorted(models_seen)}")
    resp_models_seen = {rec["response_model"] for rec in by_id.values()}
    if len(resp_models_seen) != 1:
        raise FidGateError(f"[{jsonl_path.name}] inconsistent_response_models_in_panel: {sorted(resp_models_seen)}")
    seeds_seen = {rec["seed"] for rec in by_id.values()}
    if len(seeds_seen) != 1:
        raise FidGateError(f"[{jsonl_path.name}] inconsistent_seeds_in_panel: {sorted(seeds_seen)}")

    if require_provenance_files:
        if raw_jsonl_path is None:
            raw_jsonl_path = jsonl_path.with_suffix(".raw.jsonl")
        if meta_json_path is None:
            meta_json_path = jsonl_path.with_suffix(".meta.json")

        if not meta_json_path.is_file():
            raise FidGateError(f"[{jsonl_path.name}] missing_meta_provenance_file: {meta_json_path}")
        if not raw_jsonl_path.is_file():
            raise FidGateError(f"[{jsonl_path.name}] missing_raw_provenance_file: {raw_jsonl_path}")

        try:
            meta_obj = json.loads(meta_json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise FidGateError(f"[{meta_json_path.name}] invalid_meta_json: {exc}") from exc
        if not isinstance(meta_obj, dict):
            raise FidGateError(f"[{meta_json_path.name}] meta_not_object")
        if meta_obj.get("schema_version") != META_SCHEMA_VERSION:
            raise FidGateError(
                f"[{meta_json_path.name}] invalid_meta_schema_version: {meta_obj.get('schema_version')!r}"
            )
        if meta_obj.get("status") != "completed":
            raise FidGateError(
                f"[{meta_json_path.name}] meta_status_not_completed: status={meta_obj.get('status')!r}"
            )
        if manifest_sha256 is not None and meta_obj.get("manifest_sha256") != manifest_sha256:
            raise FidGateError(
                f"[{meta_json_path.name}] meta_manifest_sha256_mismatch: "
                f"meta={meta_obj.get('manifest_sha256')!r} != expected={manifest_sha256!r}"
            )
        if meta_obj.get("completed_count") != len(manifest_rows) or meta_obj.get("panel_size") != len(manifest_rows):
            raise FidGateError(
                f"[{meta_json_path.name}] meta_panel_size_mismatch: {meta_obj.get('completed_count')}/{len(manifest_rows)}"
            )
        if meta_obj.get("completed_ids") != expected_ids:
            raise FidGateError(
                f"[{meta_json_path.name}] meta_completed_ids_mismatch"
            )
        if meta_obj.get("max_tokens") != max_tokens or meta_obj.get("top_logprobs_k") != top_logprobs_k:
            raise FidGateError(
                f"[{meta_json_path.name}] meta_config_mismatch: max_tokens={meta_obj.get('max_tokens')}, top_logprobs_k={meta_obj.get('top_logprobs_k')}"
            )
        if meta_obj.get("out_sha256") != panel_sha256:
            raise FidGateError(
                f"[{meta_json_path.name}] meta_out_sha256_mismatch: meta={meta_obj.get('out_sha256')} != actual={panel_sha256}"
            )

        raw_bytes = raw_jsonl_path.read_bytes()
        raw_sha256 = sha256_bytes(raw_bytes)
        if meta_obj.get("raw_out_sha256") != raw_sha256:
            raise FidGateError(
                f"[{meta_json_path.name}] meta_raw_out_sha256_mismatch: meta={meta_obj.get('raw_out_sha256')} != actual={raw_sha256}"
            )

        raw_lines = [ln for ln in raw_bytes.decode("utf-8").splitlines() if ln.strip()]
        if len(raw_lines) != len(manifest_rows):
            raise FidGateError(
                f"[{raw_jsonl_path.name}] raw_evidence_count_mismatch: {len(raw_lines)} != {len(manifest_rows)}"
            )

        for r_lno, (r_line, exp_id) in enumerate(zip(raw_lines, expected_ids), 1):
            try:
                raw_rec = json.loads(r_line)
            except json.JSONDecodeError as exc:
                raise FidGateError(f"[{raw_jsonl_path.name}:{r_lno}] invalid_raw_json: {exc}") from exc
            if not isinstance(raw_rec, dict):
                raise FidGateError(f"[{raw_jsonl_path.name}:{r_lno}] raw_row_not_object")
            if raw_rec.get("id") != exp_id or raw_rec.get("index") != manifest_by_id[exp_id]["index"]:
                raise FidGateError(
                    f"[{raw_jsonl_path.name}:{r_lno}] raw_id_or_index_mismatch: got id={raw_rec.get('id')!r} index={raw_rec.get('index')!r}"
                )
            if raw_rec.get("status") != "ok" or raw_rec.get("error") is not None:
                raise FidGateError(
                    f"[{raw_jsonl_path.name}:{exp_id}] raw_record_not_ok: status={raw_rec.get('status')!r} error={raw_rec.get('error')!r}"
                )
            raw_req = raw_rec.get("request")
            raw_resp = raw_rec.get("response")
            if not isinstance(raw_req, dict) or not isinstance(raw_resp, dict):
                raise FidGateError(
                    f"[{raw_jsonl_path.name}:{exp_id}] raw_record_missing_request_or_response"
                )
            actual_req_sha = canonical_json_sha256(raw_req)
            actual_resp_sha = canonical_json_sha256(raw_resp)
            norm_rec = by_id[exp_id]
            if actual_req_sha != norm_rec["request_sha256"] or raw_rec.get("request_sha256") != actual_req_sha:
                raise FidGateError(
                    f"[{raw_jsonl_path.name}:{exp_id}] raw_request_sha256_mismatch_with_panel"
                )
            if actual_resp_sha != norm_rec["raw_response_sha256"] or raw_rec.get("raw_response_sha256") != actual_resp_sha:
                raise FidGateError(
                    f"[{raw_jsonl_path.name}:{exp_id}] raw_response_sha256_mismatch_with_panel"
                )

            # Re-verify that the raw response payload actually validates to the exact panel record fields!
            revalidated = validate_and_extract_response(
                manifest_row=manifest_by_id[exp_id],
                raw_resp=raw_resp,
                wall_s=norm_rec["wall_s"],
                max_tokens=max_tokens,
                top_logprobs_k=top_logprobs_k,
                seed=norm_rec["seed"],
                require_token_ids=(norm_rec["token_ids"] is not None),
                enforce_prompt_tokens=enforce_prompt_tokens,
                request_body=raw_req,
            )
            for bind_key in (
                "system_sha256",
                "user_sha256",
                "request_sha256",
                "raw_response_sha256",
                "requested_model",
                "response_model",
                "response_id",
                "prompt_tokens",
                "completion_tokens",
                "finish_reason",
                "token_ids",
                "tokens",
                "logprob",
                "top",
                "content_sha256",
            ):
                if norm_rec[bind_key] != revalidated[bind_key]:
                    raise FidGateError(
                        f"[{jsonl_path.name}:{exp_id}] panel_record_diverges_from_raw_evidence on field '{bind_key}'"
                    )

    return by_id


def compare_panels(
    ref_path: Path,
    alt_path: Path,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    expected_manifest_sha256: str | None = DEFAULT_MANIFEST_SHA256,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    top_logprobs_k: int = DEFAULT_TOP_LOGPROBS,
    require_token_ids: bool = False,
    enforce_prompt_tokens: bool = True,
    require_provenance_files: bool = True,
    enforce_gates: bool = False,
    min_top1_agree: int = 11,
    max_median_dlp: float = 0.15,
    max_dlp_limit: float = 0.50,
    max_diverge_margin: float = 0.65,
) -> dict[str, Any]:
    """Compare two validated 12-prompt panels.

    By default (`enforce_gates=False`), comparison is strictly fail-closed on data/evidence
    integrity (`evidence_valid=True`) and descriptive on numerical differences
    (`numerical_acceptance="not_evaluated"`, `gate_passed=None`).
    When `enforce_gates=True` (`--enforce-gates`), heuristic threshold checks are evaluated
    and `gate_passed` is set to `True` or `False` (raising `FidThresholdFailure` on `False`).
    """
    manifest_rows = load_and_validate_manifest(
        manifest_path=manifest_path,
        expected_sha256=expected_manifest_sha256,
        expected_count=len(
            [
                ln
                for ln in manifest_path.read_bytes().decode("utf-8").splitlines()
                if ln.strip()
            ]
        )
        if expected_manifest_sha256 is None and manifest_path.is_file()
        else EXPECTED_PANEL_SIZE,
    )
    manifest_sha = sha256_bytes(manifest_path.read_bytes())
    ref_map = load_and_validate_panel_jsonl(
        ref_path,
        manifest_rows,
        max_tokens=max_tokens,
        top_logprobs_k=top_logprobs_k,
        require_token_ids=require_token_ids,
        enforce_prompt_tokens=enforce_prompt_tokens,
        require_provenance_files=require_provenance_files,
        manifest_sha256=manifest_sha,
    )
    alt_map = load_and_validate_panel_jsonl(
        alt_path,
        manifest_rows,
        max_tokens=max_tokens,
        top_logprobs_k=top_logprobs_k,
        require_token_ids=require_token_ids,
        enforce_prompt_tokens=enforce_prompt_tokens,
        require_provenance_files=require_provenance_files,
        manifest_sha256=manifest_sha,
    )

    agree_top1 = 0
    max_dlps: list[float] = []
    per_prompt: list[dict[str, Any]] = []
    threshold_violations: list[str] = []

    for mrow in manifest_rows:
        pid = mrow["id"]
        a = ref_map[pid]
        b = alt_map[pid]

        # Cross-panel request provenance binding
        if a["system_sha256"] != b["system_sha256"] or a["user_sha256"] != b["user_sha256"]:
            raise FidGateError(f"[{pid}] ref_vs_alt_prompt_hash_mismatch")
        if a["prompt_tokens"] != b["prompt_tokens"]:
            raise FidGateError(
                f"[{pid}] ref_vs_alt_prompt_tokens_mismatch: ref={a['prompt_tokens']} vs alt={b['prompt_tokens']}"
            )
        if a["seed"] != b["seed"] or a["max_tokens"] != b["max_tokens"] or a["top_logprobs_k"] != b["top_logprobs_k"]:
            raise FidGateError(
                f"[{pid}] ref_vs_alt_sampling_config_mismatch: "
                f"ref=(seed={a['seed']},max={a['max_tokens']},k={a['top_logprobs_k']}) vs "
                f"alt=(seed={b['seed']},max={b['max_tokens']},k={b['top_logprobs_k']})"
            )
        if a["requested_model"] != b["requested_model"]:
            raise FidGateError(
                f"[{pid}] ref_vs_alt_requested_model_mismatch: ref={a['requested_model']!r} vs alt={b['requested_model']!r}"
            )
        if a["response_model"] != b["response_model"]:
            raise FidGateError(
                f"[{pid}] ref_vs_alt_response_model_mismatch: ref={a['response_model']!r} vs alt={b['response_model']!r}"
            )

        # Prefill last-position top-k comparison (position 0)
        ta = {k: float(v) for k, v in a["top"][0]}
        tb = {k: float(v) for k, v in b["top"][0]}
        common = set(ta.keys()) & set(tb.keys())
        if not common:
            raise FidGateError(
                f"[{pid}] disjoint_top_logprobs_at_pos_0: ref={list(ta.keys())} vs alt={list(tb.keys())}"
            )

        dlp0 = max(abs(ta[t] - tb[t]) for t in common)
        max_dlps.append(dlp0)

        both_have_tids = (a.get("token_ids") is not None) and (b.get("token_ids") is not None)
        if both_have_tids:
            if (a["tokens"][0] == b["tokens"][0]) != (a["token_ids"][0] == b["token_ids"][0]):
                raise FidGateError(
                    f"[{pid}] ambiguous_decoded_string_collision_at_pos_0: "
                    f"ref=({a['token_ids'][0]}, {a['tokens'][0]!r}) vs alt=({b['token_ids'][0]}, {b['tokens'][0]!r})"
                )
            top1_match = a["token_ids"][0] == b["token_ids"][0]
        else:
            top1_match = a["top"][0][0][0] == b["top"][0][0][0]

        if top1_match:
            agree_top1 += 1

        # Continuation comparison across full sequence (never silently truncating on unequal lengths!)
        len_a = len(a["tokens"])
        len_b = len(b["tokens"])
        min_len = min(len_a, len_b)
        div_pos: int | None = None
        div_reason: str | None = None

        for pos in range(min_len):
            str_eq = a["tokens"][pos] == b["tokens"][pos]
            if both_have_tids:
                id_eq = a["token_ids"][pos] == b["token_ids"][pos]
                if str_eq != id_eq:
                    raise FidGateError(
                        f"[{pid}] ambiguous_decoded_string_collision at pos {pos}: "
                        f"ref=({a['token_ids'][pos]}, {a['tokens'][pos]!r}) vs alt=({b['token_ids'][pos]}, {b['tokens'][pos]!r})"
                    )
                same_tok = id_eq
            else:
                same_tok = str_eq

            if not same_tok:
                div_pos = pos
                div_reason = "token_mismatch"
                break

        if div_pos is None and len_a != len_b:
            # Valid sequences with identical prefix of length `min_len`, where the shorter
            # sequence terminated via valid EOS (`finish_reason == "stop"`).
            # Because `load_and_validate_panel_jsonl` already enforced that any sequence with
            # `len < max_tokens` has `finish_reason == "stop"`, this is a valid descriptive
            # EOS-length difference, NOT a data-integrity error.
            div_pos = min_len
            div_reason = (
                f"eos_length_difference(ref={len_a}:{a['finish_reason']},alt={len_b}:{b['finish_reason']})"
            )
        elif div_pos is None and a.get("finish_reason") != b.get("finish_reason"):
            # Both sequences have identical tokens of length `max_tokens`, but one ended with
            # `stop` (EOS on the exact final token) and the other with `length`.
            div_pos = min_len
            div_reason = (
                f"finish_reason_difference_at_end(ref={a['finish_reason']},alt={b['finish_reason']})"
            )

        ref_margin_at_div: float | None = None
        if div_pos is not None:
            if div_pos < len(a["top"]) and len(a["top"][div_pos]) > 1:
                ref_margin_at_div = float(a["top"][div_pos][0][1]) - float(a["top"][div_pos][1][1])
            elif div_pos < len(b["top"]) and len(b["top"][div_pos]) > 1:
                # Reference stopped with EOS at `div_pos - 1` while alt continued; use alt's margin at `div_pos`
                ref_margin_at_div = float(b["top"][div_pos][0][1]) - float(b["top"][div_pos][1][1])
            else:
                ref_margin_at_div = 0.0

            if enforce_gates and ref_margin_at_div > max_diverge_margin:
                threshold_violations.append(
                    f"[{pid}] diverge_at={div_pos} ({div_reason}) has margin={ref_margin_at_div:.4f} > {max_diverge_margin:.4f}"
                )

        ref_margin_pos0 = (
            float(a["top"][0][0][1]) - float(a["top"][0][1][1])
            if len(a["top"][0]) >= 2
            else None
        )
        per_prompt.append(
            {
                "id": pid,
                "prompt_tokens": a["prompt_tokens"],
                "ref_len": len_a,
                "alt_len": len_b,
                "ref_finish_reason": a["finish_reason"],
                "alt_finish_reason": b["finish_reason"],
                "top1_agree": top1_match,
                "pos0_shared_top_k": len(common),
                "pos0_max_abs_dlp": round(dlp0, 6),
                "pos0_ref_margin": None if ref_margin_pos0 is None else round(ref_margin_pos0, 6),
                "diverge_at": div_pos,
                "diverge_reason": div_reason,
                "ref_margin_at_diverge": (
                    None if ref_margin_at_div is None else round(ref_margin_at_div, 6)
                ),
                "comparison_mode": "token_ids+decoded_str" if both_have_tids else "decoded_str_only",
            }
        )

    n = len(per_prompt)
    sorted_dlps = sorted(max_dlps)
    median_dlp = (
        sorted_dlps[n // 2]
        if n % 2 == 1
        else 0.5 * (sorted_dlps[n // 2 - 1] + sorted_dlps[n // 2])
    )
    max_dlp = max(max_dlps)
    identical_continuations = sum(1 for r in per_prompt if r["diverge_at"] is None)

    if enforce_gates:
        if agree_top1 < min_top1_agree:
            threshold_violations.append(
                f"first_token_top1_agree={agree_top1}/{n} < min_top1_agree={min_top1_agree}"
            )
        if median_dlp > max_median_dlp:
            threshold_violations.append(
                f"median_max_dlp={median_dlp:.4f} > max_median_dlp={max_median_dlp:.4f}"
            )
        if max_dlp > max_dlp_limit:
            threshold_violations.append(
                f"max_dlp={max_dlp:.4f} > max_dlp_limit={max_dlp_limit:.4f}"
            )
        numerical_acceptance = "passed" if len(threshold_violations) == 0 else "failed"
        gate_passed: bool | None = len(threshold_violations) == 0
    else:
        numerical_acceptance = "not_evaluated"
        gate_passed = None

    summary = {
        "schema_version": COMPARE_SCHEMA_VERSION,
        "evidence_valid": True,
        "numerical_acceptance": numerical_acceptance,
        "enforce_gates": enforce_gates,
        "gate_passed": gate_passed,
        "ref_path": str(ref_path),
        "ref_sha256": sha256_bytes(ref_path.read_bytes()),
        "alt_path": str(alt_path),
        "alt_sha256": sha256_bytes(alt_path.read_bytes()),
        "panel_size": n,
        "first_token_top1_agree": agree_top1,
        "continuations_identical": identical_continuations,
        "pos0_shared_top5_median_max_abs_dlp": round(median_dlp, 6),
        "pos0_shared_top5_max_abs_dlp": round(max_dlp, 6),
        "threshold_violations": threshold_violations,
        "statistical_qualification": (
            "Descriptive string-level top-5 diagnostic over post-transform host-path logprobs "
            "(plus sampled token_ids when present). By default (`enforce_gates=False`), comparison "
            "reports `evidence_valid=True` and `numerical_acceptance='not_evaluated'` (`gate_passed=None`). "
            "When `--enforce-gates` is explicitly requested, heuristic screeners (.15/.50/.65) are evaluated."
        ),
        "per_prompt": per_prompt,
    }

    if enforce_gates and threshold_violations:
        raise FidThresholdFailure(
            " ; ".join(threshold_violations),
            summary,
        )

    return summary


def format_compare_report(summary: dict[str, Any]) -> str:
    n = summary["panel_size"]
    lines = [
        f"prompts {n}; evidence_valid={summary['evidence_valid']}; "
        f"numerical_acceptance={summary['numerical_acceptance']}; "
        f"first-token top-1 agreement {summary['first_token_top1_agree']}/{n}; "
        f"max|dlogprob| over shared top-5 at prefill last pos: "
        f"median {summary['pos0_shared_top5_median_max_abs_dlp']:.4f}, "
        f"max {summary['pos0_shared_top5_max_abs_dlp']:.4f}",
        f"greedy continuations identical: {summary['continuations_identical']}/{n} "
        f"(enforce_gates={summary['enforce_gates']}, gate_passed={summary['gate_passed']})",
    ]
    for r in summary["per_prompt"]:
        if r["diverge_at"] is not None or r["pos0_max_abs_dlp"] > 0.05 or not r["top1_agree"]:
            m_str = "n/a" if r["ref_margin_at_diverge"] is None else str(r["ref_margin_at_diverge"])
            lines.append(
                f"  {r['id']:16s} prompt={r['prompt_tokens']:5d} len={r['ref_len']:2d}/{r['alt_len']:2d} "
                f"finish={r['ref_finish_reason']}/{r['alt_finish_reason']} "
                f"top1_ok={str(r['top1_agree']):5s} max|dlp|={r['pos0_max_abs_dlp']:.4f} "
                f"diverge_at={str(r['diverge_at']):4s} reason={r['diverge_reason'] or 'none'} "
                f"margin_there={m_str}"
            )
    if summary["threshold_violations"]:
        lines.append("THRESHOLD_VIOLATIONS:")
        for v in summary["threshold_violations"]:
            lines.append(f"  - {v}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fail-closed 12-prompt fidelity harness (fid_gate12)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_col = sub.add_parser("collect", help="Collect 12-prompt fidelity panel from server")
    p_col.add_argument("url", help="Chat completions endpoint URL")
    p_col.add_argument("out", type=Path, help="Output JSONL path")
    p_col.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    p_col.add_argument("--manifest-sha256", default=DEFAULT_MANIFEST_SHA256)
    p_col.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p_col.add_argument("--top-logprobs", type=int, default=DEFAULT_TOP_LOGPROBS)
    p_col.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_col.add_argument("--request-timeout-s", type=float, default=DEFAULT_REQUEST_TIMEOUT_S)
    p_col.add_argument("--total-wall-cap-s", type=float, default=DEFAULT_TOTAL_WALL_CAP_S)
    p_col.add_argument("--allow-missing-token-ids", action="store_true")
    p_col.add_argument("--no-enforce-prompt-tokens", action="store_true")

    p_cmp = sub.add_parser("compare", help="Compare reference and candidate 12-prompt panels")
    p_cmp.add_argument("ref", type=Path, help="Reference panel JSONL")
    p_cmp.add_argument("alt", type=Path, help="Candidate panel JSONL")
    p_cmp.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    p_cmp.add_argument("--manifest-sha256", default=DEFAULT_MANIFEST_SHA256)
    p_cmp.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p_cmp.add_argument("--top-logprobs", type=int, default=DEFAULT_TOP_LOGPROBS)
    p_cmp.add_argument("--require-token-ids", action="store_true")
    p_cmp.add_argument("--no-enforce-prompt-tokens", action="store_true")
    p_cmp.add_argument("--no-require-provenance-files", action="store_true")
    p_cmp.add_argument(
        "--enforce-gates",
        action="store_true",
        help="Explicitly evaluate heuristic numerical thresholds (default: descriptive only, numerical_acceptance=not_evaluated)",
    )
    p_cmp.add_argument(
        "--no-enforce-gates",
        action="store_true",
        help="Explicitly disable heuristic numerical threshold evaluation (default behavior)",
    )
    p_cmp.add_argument("--min-top1-agree", type=int, default=11)
    p_cmp.add_argument("--max-median-dlp", type=float, default=0.15)
    p_cmp.add_argument("--max-dlp-limit", type=float, default=0.50)
    p_cmp.add_argument("--max-diverge-margin", type=float, default=0.65)
    p_cmp.add_argument("--json-out", type=Path, default=None)

    args = parser.parse_args(argv)
    try:
        if args.cmd == "collect":
            meta = collect_panel(
                url=args.url,
                out_path=args.out,
                manifest_path=args.manifest,
                expected_manifest_sha256=args.manifest_sha256 or None,
                max_tokens=args.max_tokens,
                top_logprobs_k=args.top_logprobs,
                seed=args.seed,
                request_timeout_s=args.request_timeout_s,
                total_wall_cap_s=args.total_wall_cap_s,
                require_token_ids=not args.allow_missing_token_ids,
                enforce_prompt_tokens=not args.no_enforce_prompt_tokens,
            )
            print(
                f"COLLECT_OK prompts={meta['panel_size']} wall_s={meta['total_wall_s']:.2f} "
                f"out_sha256={meta['out_sha256']}",
                flush=True,
            )
            return 0
        else:
            enforce_gates_flag = bool(args.enforce_gates and not args.no_enforce_gates)
            summary = compare_panels(
                ref_path=args.ref,
                alt_path=args.alt,
                manifest_path=args.manifest,
                expected_manifest_sha256=args.manifest_sha256 or None,
                max_tokens=args.max_tokens,
                top_logprobs_k=args.top_logprobs,
                require_token_ids=args.require_token_ids,
                enforce_prompt_tokens=not args.no_enforce_prompt_tokens,
                require_provenance_files=not args.no_require_provenance_files,
                enforce_gates=enforce_gates_flag,
                min_top1_agree=args.min_top1_agree,
                max_median_dlp=args.max_median_dlp,
                max_dlp_limit=args.max_dlp_limit,
                max_diverge_margin=args.max_diverge_margin,
            )
            if args.json_out is not None:
                args.json_out.write_text(
                    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
            print(format_compare_report(summary), flush=True)
            return 0
    except FidThresholdFailure as exc:
        if len(exc.args) > 1 and isinstance(exc.args[1], dict):
            summary = exc.args[1]
            if getattr(args, "json_out", None) is not None:
                args.json_out.write_text(
                    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
            print(format_compare_report(summary), flush=True)
        print(f"FID_THRESHOLD_FAILED: {exc.args[0]}", file=sys.stderr, flush=True)
        return 3
    except FidGateError as exc:
        print(f"FID_GATE_ERROR: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
