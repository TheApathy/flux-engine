#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only

"""Multi-turn agentic tool-call benchmark harness for Flux TripleKV KV prefetch.

Exercises ``POST /v1/kv_cache/prefetch``, ``GET /v1/kv_cache/prefetch``, and
``POST /v1/completions`` (or ``/v1/chat/completions``) across simulated agentic
tool-call latencies (``0.2s - 2.0s``) to quantify repeat-turn TTFT hiding vs a
no-prefetch baseline.

Zero third-party dependencies (Python stdlib only). Run ``--self-test`` to spin
up an in-process ``ThreadingHTTPServer`` mock implementing the exact
``KvPrefetchRequest`` / ``KvPrefetchResponse`` / ``KvPrefetchStatus`` JSON
schemas from ``crates/spark-server/src/api/kv_prefetch.rs`` and verify HTTP
202/200/400 contracts plus strict repeat-turn TTFT reduction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

DEFAULT_NVME_READ_BPS = 3.5e9
DEFAULT_SAFETY_MARGIN_SECS = 0.05
MAX_PREFETCH_TTL_SECS = 3600
MAX_PREFETCH_ISSUE_DELAY_SECS = float(MAX_PREFETCH_TTL_SECS)


@dataclass
class TurnMeasurement:
    session: int
    turn: int
    mode: str
    tool_name: str
    tool_latency_secs: float
    prefetch_http_status: int | None
    prefetch_status: str | None
    issue_delay_secs: float | None
    io_load_secs: float | None
    ttft_ms: float
    total_ms: float


def compute_triplekv_schedule(
    remaining_tool_secs: float,
    queue_wait_secs: float,
    snapshot_bytes: int,
    nvme_bandwidth_bps: float,
    safety_margin_secs: float,
) -> tuple[float, float]:
    """Exact Python mirror of TripleKV Eq. (1) from ``kv_prefetch.rs``:

    ``delta_t_issue = max(0, remaining + queue_wait - snapshot_bytes / B - eps)``.
    Returns ``(io_load_secs, issue_delay_secs)``.
    """
    if not math.isfinite(remaining_tool_secs) or not (0.0 <= remaining_tool_secs <= MAX_PREFETCH_ISSUE_DELAY_SECS):
        raise ValueError(f"remaining_tool_secs={remaining_tool_secs} out of bounds")
    if not math.isfinite(queue_wait_secs) or not (0.0 <= queue_wait_secs <= MAX_PREFETCH_ISSUE_DELAY_SECS):
        raise ValueError(f"queue_wait_secs={queue_wait_secs} out of bounds")
    if not math.isfinite(nvme_bandwidth_bps) or nvme_bandwidth_bps <= 0.0:
        raise ValueError(f"nvme_bandwidth_bps={nvme_bandwidth_bps} must be finite and > 0")
    if not math.isfinite(safety_margin_secs) or not (0.0 <= safety_margin_secs <= MAX_PREFETCH_ISSUE_DELAY_SECS):
        raise ValueError(f"safety_margin_secs={safety_margin_secs} out of bounds")
    io_load_secs = float(snapshot_bytes) / float(nvme_bandwidth_bps)
    raw_delay = remaining_tool_secs + queue_wait_secs - io_load_secs - safety_margin_secs
    issue_delay_secs = max(0.0, min(raw_delay, MAX_PREFETCH_ISSUE_DELAY_SECS))
    return io_load_secs, issue_delay_secs


def _http_json(
    url: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> tuple[int, dict[str, Any], float]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = int(resp.status)
            raw = resp.read().decode("utf-8")
            elapsed = time.perf_counter() - t0
            return status, json.loads(raw) if raw else {}, elapsed
    except urllib.error.HTTPError as err:
        raw = err.read().decode("utf-8", errors="replace")
        elapsed = time.perf_counter() - t0
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            parsed = {"raw_error": raw}
        return int(err.code), parsed, elapsed


def _measure_completion_ttft(
    base_url: str,
    endpoint: str,
    model: str,
    prompt_text: str,
    token_ids: list[int],
    stream: bool,
    timeout: float = 60.0,
) -> tuple[float, float]:
    """Send a completion request and return ``(ttft_ms, total_ms)``."""
    url = f"{base_url.rstrip('/')}{endpoint}"
    if endpoint == "/v1/chat/completions":
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt_text}],
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": stream,
        }
    else:
        body = {
            "model": model,
            "prompt": prompt_text,
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": stream,
            # Carry canonical token_ids hint for mock / compatible endpoints.
            "token_ids": token_ids,
        }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream" if stream else "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    ttft_secs: float | None = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status != HTTPStatus.OK:
            raise RuntimeError(f"completion request failed with HTTP {resp.status}")
        if stream:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                payload_str = line[5:].strip()
                if payload_str == "[DONE]":
                    break
                if ttft_secs is None:
                    ttft_secs = time.perf_counter() - t0
        else:
            _ = resp.read()
            ttft_secs = time.perf_counter() - t0
    total_secs = time.perf_counter() - t0
    if ttft_secs is None:
        ttft_secs = total_secs
    return ttft_secs * 1000.0, total_secs * 1000.0


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * (pct / 100.0)
    lo = int(math.floor(idx))
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _build_session_tokens(session_idx: int, prefix_tokens: int, block_size: int) -> list[int]:
    aligned = max(block_size, (prefix_tokens // block_size) * block_size)
    base = 1000 + session_idx * 4096
    return [base + (i % 2048) for i in range(aligned)]


def run_benchmark(
    base_url: str,
    model: str,
    endpoint: str,
    sessions: int,
    turns: int,
    tool_latencies: list[float],
    prefix_tokens: int,
    block_size: int,
    kv_dtype: str,
    nvme_bandwidth_bps: float,
    safety_margin_secs: float,
    ttl_secs: int,
    stream: bool,
    time_scale: float = 1.0,
) -> dict[str, Any]:
    """Run paired baseline vs TripleKV multi-turn agentic benchmark."""
    status_before_code, status_before, _ = _http_json(f"{base_url.rstrip('/')}/v1/kv_cache/prefetch")
    if status_before_code != HTTPStatus.OK:
        raise RuntimeError(
            f"GET /v1/kv_cache/prefetch returned HTTP {status_before_code}: {status_before}"
        )

    measurements: list[TurnMeasurement] = []
    tool_names = ["search_repo", "run_pytest", "sql_query", "http_fetch"]

    for mode in ("baseline", "triplekv"):
        last_obs_by_tool: dict[str, float] = {}
        for s_idx in range(sessions):
            # Use disjoint prefix token streams between baseline and triplekv so
            # baseline cannot accidentally ride warm L2 entries staged by triplekv.
            stream_id = s_idx if mode == "baseline" else (sessions + s_idx)
            tokens = _build_session_tokens(stream_id, prefix_tokens, block_size)
            prompt = f"[session={mode}:{s_idx}] " + " ".join(f"tok{t}" for t in tokens[:32])

            # Turn 0: initial cold prefill & L3 snapshot population.
            ttft0, total0 = _measure_completion_ttft(
                base_url=base_url,
                endpoint=endpoint,
                model=model,
                prompt_text=prompt,
                token_ids=tokens,
                stream=stream,
            )
            measurements.append(
                TurnMeasurement(
                    session=s_idx,
                    turn=0,
                    mode=mode,
                    tool_name="turn0_init",
                    tool_latency_secs=0.0,
                    prefetch_http_status=None,
                    prefetch_status=None,
                    issue_delay_secs=None,
                    io_load_secs=None,
                    ttft_ms=round(ttft0, 3),
                    total_ms=round(total0, 3),
                )
            )

            for turn in range(1, turns + 1):
                raw_tool_lat = tool_latencies[(s_idx + turn - 1) % len(tool_latencies)]
                tool_lat = raw_tool_lat * time_scale
                tool_name = tool_names[(turn - 1) % len(tool_names)]
                prefetch_code: int | None = None
                prefetch_status: str | None = None
                issue_delay: float | None = None
                io_load: float | None = None

                tool_start = time.perf_counter()
                if mode == "triplekv":
                    req_body: dict[str, Any] = {
                        "token_ids": tokens,
                        "model_digest": model,
                        "kv_dtype": kv_dtype,
                        "block_size": block_size,
                        "ttl_secs": ttl_secs,
                        "tool_name": tool_name,
                        "predicted_tool_remaining_secs": tool_lat,
                        "estimated_queue_wait_secs": 0.0,
                        "nvme_bandwidth_bps": nvme_bandwidth_bps,
                        "safety_margin_secs": safety_margin_secs * time_scale,
                    }
                    if tool_name in last_obs_by_tool:
                        req_body["tool_duration_observation_secs"] = last_obs_by_tool[tool_name]
                    prefetch_code, prefetch_resp, _ = _http_json(
                        f"{base_url.rstrip('/')}/v1/kv_cache/prefetch",
                        method="POST",
                        payload=req_body,
                    )
                    if prefetch_code not in (HTTPStatus.ACCEPTED, HTTPStatus.OK):
                        raise RuntimeError(
                            f"POST /v1/kv_cache/prefetch failed ({prefetch_code}): {prefetch_resp}"
                        )
                    prefetch_status = str(prefetch_resp.get("status", ""))
                    issue_delay = float(prefetch_resp.get("issue_delay_secs", 0.0))
                    io_load = float(prefetch_resp.get("io_load_secs", 0.0))

                # Sleep remaining simulated tool duration (L3->L2 prefetch overlaps here!).
                elapsed_tool = time.perf_counter() - tool_start
                remaining_sleep = tool_lat - elapsed_tool
                if remaining_sleep > 0.0:
                    time.sleep(remaining_sleep)
                last_obs_by_tool[tool_name] = time.perf_counter() - tool_start

                ttft_ms, total_ms = _measure_completion_ttft(
                    base_url=base_url,
                    endpoint=endpoint,
                    model=model,
                    prompt_text=f"{prompt} [turn={turn} tool={tool_name}]",
                    token_ids=tokens,
                    stream=stream,
                )
                measurements.append(
                    TurnMeasurement(
                        session=s_idx,
                        turn=turn,
                        mode=mode,
                        tool_name=tool_name,
                        tool_latency_secs=round(tool_lat, 4),
                        prefetch_http_status=prefetch_code,
                        prefetch_status=prefetch_status,
                        issue_delay_secs=None if issue_delay is None else round(issue_delay, 5),
                        io_load_secs=None if io_load is None else round(io_load, 5),
                        ttft_ms=round(ttft_ms, 3),
                        total_ms=round(total_ms, 3),
                    )
                )

    status_after_code, status_after, _ = _http_json(f"{base_url.rstrip('/')}/v1/kv_cache/prefetch")
    if status_after_code != HTTPStatus.OK:
        raise RuntimeError(f"GET /v1/kv_cache/prefetch after run failed: {status_after_code}")

    base_repeat = [m.ttft_ms for m in measurements if m.mode == "baseline" and m.turn > 0]
    pref_repeat = [m.ttft_ms for m in measurements if m.mode == "triplekv" and m.turn > 0]
    base_mean = statistics.fmean(base_repeat) if base_repeat else 0.0
    pref_mean = statistics.fmean(pref_repeat) if pref_repeat else 0.0
    reduction_ms = base_mean - pref_mean
    reduction_pct = (reduction_ms / base_mean * 100.0) if base_mean > 0.0 else 0.0
    speedup_x = (base_mean / pref_mean) if pref_mean > 0.0 else 0.0

    counter_keys = (
        "scheduled_total",
        "completed_total",
        "hit_hot_total",
        "miss_total",
        "errors_total",
    )
    status_delta = {
        k: int(status_after.get(k, 0)) - int(status_before.get(k, 0)) for k in counter_keys
    }

    return {
        "config": {
            "base_url": base_url,
            "model": model,
            "endpoint": endpoint,
            "sessions": sessions,
            "turns": turns,
            "tool_latencies_secs": tool_latencies,
            "time_scale": time_scale,
            "prefix_tokens": prefix_tokens,
            "block_size": block_size,
            "kv_dtype": kv_dtype,
            "nvme_bandwidth_bps": nvme_bandwidth_bps,
            "safety_margin_secs": safety_margin_secs,
        },
        "summary": {
            "baseline_repeat_ttft_ms": {
                "mean": round(base_mean, 3),
                "p50": round(_percentile(base_repeat, 50.0), 3),
                "p95": round(_percentile(base_repeat, 95.0), 3),
            },
            "triplekv_repeat_ttft_ms": {
                "mean": round(pref_mean, 3),
                "p50": round(_percentile(pref_repeat, 50.0), 3),
                "p95": round(_percentile(pref_repeat, 95.0), 3),
            },
            "ttft_reduction_ms": round(reduction_ms, 3),
            "ttft_reduction_pct": round(reduction_pct, 2),
            "ttft_speedup_x": round(speedup_x, 3),
        },
        "prefetch_status_before": status_before,
        "prefetch_status_after": status_after,
        "prefetch_status_delta": status_delta,
        "turns": [asdict(m) for m in measurements],
    }


# ── In-process TripleKV mock server for --self-test ─────────────────────────


class _MockTripleKvState:
    """Thread-safe mock matching ``crates/spark-server/src/api/kv_prefetch.rs``."""

    def __init__(self, snapshot_bytes: int, warm_prefill_secs: float, cold_io_override_secs: float | None = None) -> None:
        self.lock = threading.Lock()
        self.snapshot_bytes = snapshot_bytes
        self.warm_prefill_secs = warm_prefill_secs
        self.cold_io_override_secs = cold_io_override_secs
        self.l3_store: dict[str, int] = {}
        self.staged: dict[str, float] = {}  # content_id -> expires_at monotonic
        self.tracked_tools: dict[str, list[float]] = {}
        self.scheduled_total = 0
        self.completed_total = 0
        self.hit_hot_total = 0
        self.miss_total = 0
        self.errors_total = 0
        self._timers: list[threading.Timer] = []

    @staticmethod
    def content_id(token_ids: list[int], block_size: int, model_digest: str, kv_dtype: str) -> tuple[str, int]:
        aligned = (len(token_ids) // block_size) * block_size
        h = hashlib.sha256()
        h.update(f"{model_digest}|{kv_dtype}|{block_size}|".encode("utf-8"))
        for tok in token_ids[:aligned]:
            h.update(int(tok).to_bytes(4, "little", signed=False))
        return h.hexdigest()[:32], aligned

    def status_snapshot(self) -> dict[str, Any]:
        with self.lock:
            now = time.monotonic()
            self.staged = {k: exp for k, exp in self.staged.items() if exp > now}
            return {
                "store_enabled": True,
                "radix_tree_attached": True,
                "staged_entries": len(self.staged),
                "staged_bytes": len(self.staged) * self.snapshot_bytes,
                "tracked_tools": len(self.tracked_tools),
                "scheduled_total": self.scheduled_total,
                "completed_total": self.completed_total,
                "hit_hot_total": self.hit_hot_total,
                "miss_total": self.miss_total,
                "errors_total": self.errors_total,
            }

    def schedule_prefetch(self, req: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        token_ids = req.get("token_ids")
        prompt = req.get("prompt")
        if token_ids is not None:
            if not isinstance(token_ids, list) or len(token_ids) == 0 or not all(isinstance(x, int) and x >= 0 for x in token_ids):
                return HTTPStatus.BAD_REQUEST, {"error": {"message": "token_ids must be a non-empty list of u32"}}
        elif isinstance(prompt, str) and prompt:
            token_ids = [ord(c) for c in prompt]
        else:
            return HTTPStatus.BAD_REQUEST, {"error": {"message": "token_ids or non-empty prompt is required"}}

        block_size = int(req.get("block_size") or 16)
        if block_size <= 0:
            return HTTPStatus.BAD_REQUEST, {"error": {"message": "block_size must be > 0"}}
        ttl_secs = min(int(req.get("ttl_secs") or 120), MAX_PREFETCH_TTL_SECS)
        model_digest = str(req.get("model_digest") or "mock-model")
        kv_dtype = str(req.get("kv_dtype") or "bf16")
        tool_name = req.get("tool_name")
        remaining = float(req.get("predicted_tool_remaining_secs", 0.0))
        queue_wait = float(req.get("estimated_queue_wait_secs", 0.0))
        bw = float(req.get("nvme_bandwidth_bps", DEFAULT_NVME_READ_BPS))
        eps = float(req.get("safety_margin_secs", DEFAULT_SAFETY_MARGIN_SECS))
        obs = req.get("tool_duration_observation_secs")

        cid, aligned = self.content_id(token_ids, block_size, model_digest, kv_dtype)
        if aligned == 0:
            return HTTPStatus.BAD_REQUEST, {"error": {"message": "prefix shorter than one block_size"}}

        with self.lock:
            if isinstance(tool_name, str) and tool_name:
                hist = self.tracked_tools.setdefault(tool_name, [])
                if obs is not None and isinstance(obs, (int, float)) and math.isfinite(float(obs)) and float(obs) >= 0.0:
                    hist.append(float(obs))

            now = time.monotonic()
            if cid in self.staged and self.staged[cid] > now:
                self.hit_hot_total += 1
                return HTTPStatus.OK, {
                    "status": "already_hot",
                    "content_id": cid,
                    "aligned_tokens": aligned,
                    "block_size": block_size,
                    "snapshot_bytes": self.l3_store.get(cid),
                    "issue_delay_secs": 0.0,
                    "io_load_secs": 0.0,
                    "ttl_secs": ttl_secs,
                    "radix_nodes_protected": max(1, aligned // block_size),
                }

            snap_bytes = self.l3_store.get(cid)
            if snap_bytes is None:
                self.miss_total += 1
                return HTTPStatus.OK, {
                    "status": "miss_not_in_store",
                    "content_id": cid,
                    "aligned_tokens": aligned,
                    "block_size": block_size,
                    "snapshot_bytes": None,
                    "issue_delay_secs": 0.0,
                    "io_load_secs": 0.0,
                    "ttl_secs": ttl_secs,
                    "radix_nodes_protected": 0,
                }

            try:
                io_load_secs, issue_delay_secs = compute_triplekv_schedule(
                    remaining_tool_secs=remaining,
                    queue_wait_secs=queue_wait,
                    snapshot_bytes=snap_bytes,
                    nvme_bandwidth_bps=bw,
                    safety_margin_secs=eps,
                )
            except ValueError as err:
                return HTTPStatus.BAD_REQUEST, {"error": {"message": str(err)}}

            self.scheduled_total += 1
            effective_io = io_load_secs if self.cold_io_override_secs is None else self.cold_io_override_secs

            def _complete_stage() -> None:
                time.sleep(effective_io)
                with self.lock:
                    self.staged[cid] = time.monotonic() + float(ttl_secs)
                    self.completed_total += 1

            timer = threading.Timer(issue_delay_secs, _complete_stage)
            timer.daemon = True
            self._timers.append(timer)
            timer.start()

            return HTTPStatus.ACCEPTED, {
                "status": "scheduled",
                "content_id": cid,
                "aligned_tokens": aligned,
                "block_size": block_size,
                "snapshot_bytes": snap_bytes,
                "issue_delay_secs": issue_delay_secs,
                "io_load_secs": io_load_secs,
                "ttl_secs": ttl_secs,
                "radix_nodes_protected": max(1, aligned // block_size),
            }

    def serve_completion(self, req: dict[str, Any]) -> tuple[bool, float]:
        token_ids = req.get("token_ids") or [1, 2, 3, 4] * 4
        model_digest = str(req.get("model") or "mock-model")
        cid, _ = self.content_id(token_ids, 16, model_digest, "bf16")
        with self.lock:
            now = time.monotonic()
            was_in_l3 = cid in self.l3_store
            is_hot = cid in self.staged and self.staged[cid] > now
            if is_hot:
                # Consume staged warm L2 entry for this repeat turn (simulating eviction on next idle).
                del self.staged[cid]

        if is_hot:
            # Prefetched warm L2 hit: zero synchronous NVMe stall.
            time.sleep(self.warm_prefill_secs)
        else:
            # Cold L3 miss (or initial turn 0): synchronous L3 load / recompute penalty.
            cold_stall = (
                self.cold_io_override_secs
                if self.cold_io_override_secs is not None
                else (self.snapshot_bytes / DEFAULT_NVME_READ_BPS)
            )
            time.sleep(self.warm_prefill_secs + cold_stall)

        with self.lock:
            if not was_in_l3:
                self.l3_store[cid] = self.snapshot_bytes
        return is_hot, self.warm_prefill_secs

    def shutdown_timers(self) -> None:
        with self.lock:
            timers = list(self._timers)
            self._timers.clear()
        for t in timers:
            t.cancel()


def _start_mock_server(state: _MockTripleKvState) -> tuple[ThreadingHTTPServer, threading.Thread, str]:
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

        def do_GET(self) -> None:
            if self.path == "/v1/kv_cache/prefetch":
                self._send_json(HTTPStatus.OK, state.status_snapshot())
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found"}})

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length) if length > 0 else b""
            try:
                body = json.loads(raw_body.decode("utf-8")) if raw_body else {}
            except json.JSONDecodeError as err:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": {"message": f"invalid JSON: {err}"}})
                return

            if self.path == "/v1/kv_cache/prefetch":
                code, resp = state.schedule_prefetch(body)
                self._send_json(code, resp)
                return

            if self.path in ("/v1/completions", "/v1/chat/completions"):
                was_hot, _ = state.serve_completion(body)
                if body.get("stream"):
                    chunk = json.dumps(
                        {
                            "id": "cmpl-mock",
                            "object": "text_completion",
                            "choices": [{"index": 0, "text": "ok", "finish_reason": "length"}],
                            "triplekv_hot": was_hot,
                        }
                    )
                    payload = f"data: {chunk}\n\ndata: [DONE]\n\n".encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Content-Length", str(len(payload)))
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(payload)
                else:
                    self._send_json(
                        HTTPStatus.OK,
                        {
                            "id": "cmpl-mock",
                            "object": "text_completion",
                            "choices": [{"index": 0, "text": "ok", "finish_reason": "length"}],
                            "triplekv_hot": was_hot,
                        },
                    )
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found"}})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, f"http://127.0.0.1:{port}"


def run_self_test() -> int:
    """Verify exact TripleKV HTTP 202/200/400 contracts, Eq. (1) math, and A/B TTFT hiding."""
    # 1. Exact Eq. (1) schedule verification across real 0.2s..2.0s agentic tool latencies.
    snap_bytes = 350_000_000  # 350 MB KV snapshot => 0.10s at 3.5 GB/s
    for tool_lat, expected_delay in (
        (0.2, 0.05),  # 0.20 - 0.10 - 0.05 = 0.05
        (0.5, 0.35),
        (1.0, 0.85),
        (2.0, 1.85),
    ):
        io_s, delay_s = compute_triplekv_schedule(
            remaining_tool_secs=tool_lat,
            queue_wait_secs=0.0,
            snapshot_bytes=snap_bytes,
            nvme_bandwidth_bps=DEFAULT_NVME_READ_BPS,
            safety_margin_secs=DEFAULT_SAFETY_MARGIN_SECS,
        )
        assert abs(io_s - 0.10) < 1.0e-9
        assert abs(delay_s - expected_delay) < 1.0e-9

    # 2. Spin up in-process stdlib ThreadingHTTPServer mock (scaled ~3.3x faster for fast CI).
    # 105 MB at 3.5 GB/s => 30ms I/O load; scaled tool latencies 60ms..600ms overlap with >=15ms margin.
    mock_state = _MockTripleKvState(
        snapshot_bytes=105_000_000,
        warm_prefill_secs=0.004,
        cold_io_override_secs=0.030,
    )
    server, thread, base_url = _start_mock_server(mock_state)
    try:
        # Contract test A: HTTP 400 on empty token_ids and invalid bandwidth.
        code_bad_empty, _, _ = _http_json(
            f"{base_url}/v1/kv_cache/prefetch", method="POST", payload={"token_ids": []}
        )
        assert code_bad_empty == HTTPStatus.BAD_REQUEST, f"expected 400, got {code_bad_empty}"

        # Contract test B: HTTP 200 miss_not_in_store on unseen prefix.
        unseen_tokens = list(range(100, 164))
        code_miss, resp_miss, _ = _http_json(
            f"{base_url}/v1/kv_cache/prefetch",
            method="POST",
            payload={"token_ids": unseen_tokens, "model_digest": "mock-model", "block_size": 16},
        )
        assert code_miss == HTTPStatus.OK and resp_miss["status"] == "miss_not_in_store"

        # Contract test C: invalid bandwidth after populating L3 -> HTTP 400.
        _measure_completion_ttft(
            base_url=base_url,
            endpoint="/v1/completions",
            model="mock-model",
            prompt_text="seed",
            token_ids=unseen_tokens,
            stream=False,
        )
        code_bad_bw, _, _ = _http_json(
            f"{base_url}/v1/kv_cache/prefetch",
            method="POST",
            payload={
                "token_ids": unseen_tokens,
                "model_digest": "mock-model",
                "block_size": 16,
                "nvme_bandwidth_bps": -1.0,
            },
        )
        assert code_bad_bw == HTTPStatus.BAD_REQUEST, f"expected 400 on bad bw, got {code_bad_bw}"

        # Contract test D: HTTP 202 scheduled -> wait -> HTTP 200 already_hot.
        code_sched, resp_sched, _ = _http_json(
            f"{base_url}/v1/kv_cache/prefetch",
            method="POST",
            payload={
                "token_ids": unseen_tokens,
                "model_digest": "mock-model",
                "block_size": 16,
                "predicted_tool_remaining_secs": 0.03,
                "safety_margin_secs": 0.01,
            },
        )
        assert code_sched == HTTPStatus.ACCEPTED and resp_sched["status"] == "scheduled"
        time.sleep(0.06)
        code_hot, resp_hot, _ = _http_json(
            f"{base_url}/v1/kv_cache/prefetch",
            method="POST",
            payload={"token_ids": unseen_tokens, "model_digest": "mock-model", "block_size": 16},
        )
        assert code_hot == HTTPStatus.OK and resp_hot["status"] == "already_hot"

        # Contract test E: full multi-turn A/B agentic benchmark over 4 tool latencies.
        report = run_benchmark(
            base_url=base_url,
            model="mock-model",
            endpoint="/v1/completions",
            sessions=1,
            turns=4,
            tool_latencies=[0.2, 0.5, 1.0, 2.0],
            prefix_tokens=256,
            block_size=16,
            kv_dtype="bf16",
            nvme_bandwidth_bps=DEFAULT_NVME_READ_BPS,
            safety_margin_secs=0.05,
            ttl_secs=120,
            stream=True,
            time_scale=0.30,  # 60ms..600ms -> fast CI while finishing 30ms I/O ahead of arrival
        )
        base_mean = report["summary"]["baseline_repeat_ttft_ms"]["mean"]
        pref_mean = report["summary"]["triplekv_repeat_ttft_ms"]["mean"]
        assert pref_mean < base_mean * 0.5, (
            f"TripleKV did not strictly reduce repeat-turn TTFT: baseline={base_mean}ms "
            f"vs triplekv={pref_mean}ms"
        )
        assert report["prefetch_status_delta"]["scheduled_total"] == 4
        assert report["prefetch_status_delta"]["completed_total"] >= 4
        assert report["prefetch_status_delta"]["errors_total"] == 0

        print(
            json.dumps(
                {
                    "self_test": "PASS",
                    "contracts_verified": ["400_bad_request", "200_miss", "202_scheduled", "200_already_hot", "get_status"],
                    "summary": report["summary"],
                    "prefetch_status_delta": report["prefetch_status_delta"],
                },
                indent=2,
            )
        )
    finally:
        mock_state.shutdown_timers()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-turn agentic tool-call benchmark harness for Flux TripleKV prefetch"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8080", help="Flux server base URL")
    parser.add_argument("--model", default="default", help="Model name / digest for requests")
    parser.add_argument(
        "--endpoint",
        choices=("/v1/completions", "/v1/chat/completions"),
        default="/v1/completions",
        help="Completion endpoint used to measure TTFT",
    )
    parser.add_argument("--sessions", type=int, default=2, help="Number of agentic sessions per arm")
    parser.add_argument("--turns", type=int, default=4, help="Repeat tool-call turns per session")
    parser.add_argument(
        "--tool-latencies",
        default="0.2,0.5,1.0,2.0",
        help="Comma-separated simulated tool execution latencies in seconds",
    )
    parser.add_argument("--prefix-tokens", type=int, default=2048, help="Shared prefix length in tokens")
    parser.add_argument("--block-size", type=int, default=16, help="KV cache block size")
    parser.add_argument("--kv-dtype", default="bf16", help="KV cache dtype tag")
    parser.add_argument(
        "--nvme-bandwidth-bps",
        type=float,
        default=DEFAULT_NVME_READ_BPS,
        help="Assumed NVMe sequential read bandwidth (bytes/sec)",
    )
    parser.add_argument(
        "--safety-margin-secs",
        type=float,
        default=DEFAULT_SAFETY_MARGIN_SECS,
        help="TripleKV Eq. (1) safety margin epsilon (seconds)",
    )
    parser.add_argument("--ttl-secs", type=int, default=120, help="Radix protection TTL (seconds)")
    parser.add_argument("--no-stream", action="store_true", help="Disable SSE streaming TTFT probe")
    parser.add_argument("--json-out", type=Path, help="Optional output path for full JSON benchmark report")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Spin up an in-process mock server and verify TripleKV HTTP contracts + TTFT hiding",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test()

    tool_latencies = [float(item.strip()) for item in args.tool_latencies.split(",") if item.strip()]
    if not tool_latencies or any(not math.isfinite(x) or x < 0.0 for x in tool_latencies):
        raise SystemExit("error: --tool-latencies must contain non-negative finite floats")

    report = run_benchmark(
        base_url=args.base_url,
        model=args.model,
        endpoint=args.endpoint,
        sessions=args.sessions,
        turns=args.turns,
        tool_latencies=tool_latencies,
        prefix_tokens=args.prefix_tokens,
        block_size=args.block_size,
        kv_dtype=args.kv_dtype,
        nvme_bandwidth_bps=args.nvme_bandwidth_bps,
        safety_margin_secs=args.safety_margin_secs,
        ttl_secs=args.ttl_secs,
        stream=not args.no_stream,
    )
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
