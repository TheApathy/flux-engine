#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Shared fail-closed endpoint admission evidence for Qwen4 qualifiers."""

from __future__ import annotations

import hashlib
import re
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

METRIC_NAMES = (
    "atlas_prefill_progress_chunks_total",
    "atlas_prefill_progress_tokens_total",
)
REQUEST_ID_RE = re.compile(r"qualify_overflow_[0-9a-f]{32}")
DISPATCH_RE = re.compile(
    r"(?:Chunked prefill start:|Prefilling:) ([0-9]+) prompt tokens,"
)


class RequestLogCapture:
    """Read a stable request-scoped suffix from an append-only server log."""

    def __init__(self, path: Path, settle_seconds: float = 0.02):
        self.path = path
        self.settle_seconds = settle_seconds

    def begin(self) -> int:
        return self.path.stat().st_size

    def finish(self, offset: int) -> bytes:
        deadline = time.monotonic() + 1.0
        stable_since = time.monotonic()
        previous = -1
        while True:
            size = self.path.stat().st_size
            now = time.monotonic()
            if size != previous:
                previous = size
                stable_since = now
            if now - stable_since >= self.settle_seconds or now >= deadline:
                break
            time.sleep(min(0.005, self.settle_seconds or 0.0))
        if type(offset) is not int or offset < 0 or offset > previous:
            raise ValueError("request log offset moved backwards")
        with self.path.open("rb") as handle:
            handle.seek(offset)
            return handle.read(previous - offset)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def validate_context_length_rejection(
    status: int,
    value: object,
    prompt_tokens: int,
    requested_tokens: int,
    max_seq_len: int,
) -> dict[str, Any]:
    for label, number in (
        ("HTTP status", status),
        ("prompt_tokens", prompt_tokens),
        ("requested_tokens", requested_tokens),
        ("max_seq_len", max_seq_len),
    ):
        if type(number) is not int or number < 0:
            raise ValueError(f"overflow {label} is not an exact non-negative integer")
    expected_message = (
        f"Context too long: {prompt_tokens} prompt + {requested_tokens} requested "
        f"output tokens exceeds max_seq_len {max_seq_len}"
    )
    if status != 400 or not isinstance(value, dict) or set(value) != {"error"}:
        raise ValueError("overflow probe did not return the exact HTTP 400 error shape")
    error = value["error"]
    if not isinstance(error, dict) or set(error) != {
        "message",
        "type",
        "param",
        "code",
    }:
        raise ValueError("overflow probe error object differs from the exact contract")
    if error != {
        "message": expected_message,
        "type": "invalid_request_error",
        "param": None,
        "code": None,
    }:
        raise ValueError("overflow probe was rejected for an unexpected reason")
    return {
        "http_status": status,
        "error": error,
        "context_length_rejection": True,
    }


def validate_correlated_request_id(
    headers: list[tuple[str, str]], expected: str
) -> str:
    if type(expected) is not str or REQUEST_ID_RE.fullmatch(expected) is None:
        raise ValueError("overflow probe request ID differs from the exact contract")
    if not isinstance(headers, list):
        raise ValueError("overflow rejection response headers are not a list")
    values = [
        value
        for item in headers
        if isinstance(item, tuple)
        and len(item) == 2
        and type(item[0]) is str
        and type(item[1]) is str
        for name, value in (item,)
        if name.lower() == "x-request-id"
    ]
    if values != [expected]:
        raise ValueError(
            "overflow rejection response does not echo exactly one correlated "
            "x-request-id"
        )
    return expected


def parse_prometheus_integer_metrics(
    payload: bytes, names: tuple[str, ...] = METRIC_NAMES
) -> dict[str, int]:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("/metrics response is not UTF-8") from error
    expected = set(names)
    found: dict[str, int] = {}
    for line in text.splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#") or fields[0] not in expected:
            continue
        name = fields[0]
        if name in found:
            raise ValueError(f"/metrics repeats unlabelled sample {name}")
        if len(fields) != 2:
            raise ValueError(f"/metrics sample {name} has an unexpected shape")
        try:
            value = Decimal(fields[1])
        except InvalidOperation as error:
            raise ValueError(f"/metrics sample {name} is not numeric") from error
        if not value.is_finite() or value < 0 or value != value.to_integral_value():
            raise ValueError(f"/metrics sample {name} is not a non-negative integer")
        found[name] = int(value)
    missing = sorted(expected.difference(found))
    if missing:
        raise ValueError(f"/metrics lacks required samples {missing!r}")
    return {name: found[name] for name in names}


def capture_metrics(client: Any) -> dict[str, Any]:
    status, raw = client.request("GET", "/metrics")
    if type(status) is not int or status != 200:
        raise ValueError(f"admission proof /metrics returned HTTP {status}")
    if not isinstance(raw, bytes):
        raise ValueError("admission proof /metrics body is not bytes")
    return {
        "http_status": status,
        "values": parse_prometheus_integer_metrics(raw),
        "body_sha256": sha256_bytes(raw),
        "body_bytes": len(raw),
    }


def _validated_values(receipt: dict[str, Any], label: str) -> dict[str, int]:
    values = receipt.get("values") if isinstance(receipt, dict) else None
    if not isinstance(values, dict) or set(values) != set(METRIC_NAMES):
        raise ValueError(f"{label} /metrics sample set differs from contract")
    for name in METRIC_NAMES:
        if type(values[name]) is not int or values[name] < 0:
            raise ValueError(
                f"{label} /metrics sample {name} is not an exact non-negative integer"
            )
    return values


def validate_no_scheduler_dispatch(
    logs: str,
    metrics_before: dict[str, Any],
    metrics_after: dict[str, Any],
    http_request_id: str,
) -> dict[str, Any]:
    if (
        type(http_request_id) is not str
        or REQUEST_ID_RE.fullmatch(http_request_id) is None
    ):
        raise ValueError("overflow rejection lacks a correlated HTTP request ID")
    forbidden = (
        "Chunked prefill start:",
        "Prefilling:",
        "Prefilled:",
        "Prefilled (single chunk):",
        "MTP_ACCEPT frame",
        "SPEC_CYCLE_V2 ",
        "SPEC_CYCLE_V2_TERMINAL ",
        "Done:",
    )
    observed = [marker for marker in forbidden if marker in logs]
    if observed:
        raise ValueError(
            f"rejected overflow request reached scheduler dispatch: {observed!r}"
        )
    runtime_request_ids = sorted(set(re.findall(r"\breq=([0-9a-f]{16})\b", logs)))
    if runtime_request_ids:
        raise ValueError(
            "rejected overflow request acquired scheduler request IDs: "
            f"{runtime_request_ids!r}"
        )
    before = _validated_values(metrics_before, "rejection proof before")
    after = _validated_values(metrics_after, "rejection proof after")
    deltas = {name: after[name] - before[name] for name in METRIC_NAMES}
    moved = {name: delta for name, delta in deltas.items() if delta != 0}
    if moved:
        raise ValueError(
            f"rejected overflow request changed scheduler prefill metrics: {moved!r}"
        )
    return {
        "scheduler_dispatch_markers": 0,
        "native_mtp_frames": 0,
        "completed_spec_cycles": 0,
        "runtime_request_ids": [],
        "api_rejection_correlated": True,
        "scheduler_prefill_metrics_unchanged": True,
        "http_request_id": http_request_id,
        "metric_deltas": deltas,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
    }


def validate_post_canary_metrics(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    prompt_tokens: int,
    prefill_chunks: int,
) -> dict[str, Any]:
    for label, number in (
        ("prompt_tokens", prompt_tokens),
        ("prefill_chunks", prefill_chunks),
    ):
        if type(number) is not int or number <= 0:
            raise ValueError(f"post-canary {label} is not an exact positive integer")
    left = _validated_values(before, "post-canary before")
    right = _validated_values(after, "post-canary after")
    deltas = {name: right[name] - left[name] for name in METRIC_NAMES}
    expected = {
        "atlas_prefill_progress_chunks_total": prefill_chunks,
        "atlas_prefill_progress_tokens_total": prompt_tokens,
    }
    if deltas != expected:
        raise ValueError(
            f"post-rejection completion canary metric deltas differ: {deltas!r}"
        )
    return {
        "metrics_before": before,
        "metrics_after": after,
        "metric_deltas": deltas,
        "expected_metric_deltas": expected,
        "completion_canary_prefill_accounted": True,
    }


def dispatch_census(logs: str) -> dict[int, int]:
    counts: dict[int, int] = {}
    for match in DISPATCH_RE.finditer(logs):
        prompt_tokens = int(match.group(1))
        counts[prompt_tokens] = counts.get(prompt_tokens, 0) + 1
    return counts


def validate_final_dispatch_census(
    logs: str,
    expected_counts: dict[int, int],
    *,
    rejected_prompt_tokens: int,
) -> dict[str, Any]:
    if type(rejected_prompt_tokens) is not int or rejected_prompt_tokens <= 0:
        raise ValueError("rejected prompt length is not an exact positive integer")
    if not isinstance(expected_counts, dict) or any(
        type(tokens) is not int or tokens <= 0 or type(count) is not int or count < 0
        for tokens, count in expected_counts.items()
    ):
        raise ValueError("expected dispatch census is not exact integer data")
    observed = dispatch_census(logs)
    if observed.get(rejected_prompt_tokens, 0) != 0:
        raise ValueError(
            "rejected overflow request appears in the final dispatch census"
        )
    selected = {tokens: observed.get(tokens, 0) for tokens in expected_counts}
    if selected != expected_counts:
        raise ValueError(
            f"final scheduler dispatch census differs: expected {expected_counts!r}, "
            f"observed {selected!r}"
        )
    unexpected = {
        tokens: count
        for tokens, count in observed.items()
        if tokens not in expected_counts
    }
    if unexpected:
        raise ValueError(
            f"final scheduler dispatch census has unexpected rows {unexpected!r}"
        )
    return {
        "expected_dispatches": expected_counts,
        "observed_dispatches": observed,
        "rejected_prompt_tokens": rejected_prompt_tokens,
        "rejected_prompt_dispatches": 0,
        "whole_log_census": True,
    }
