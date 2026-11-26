#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Original, stdlib-only exact-token burst measurements for Flux endpoints.

Only the explicit ``run`` subcommand contacts an endpoint. Public reports are
derived from private, hash-checked artifacts and omit free-form identities,
URLs, paths, case labels, response text, and exception messages.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import threading
import time
from urllib import error, parse, request

CORPUS_SCHEMA = "flux-lens.corpus.v1"
PLAN_SCHEMA = "flux-lens.plan.v1"
RUN_SCHEMA = "flux-lens.run.v1"
REPORT_SCHEMA = "flux-lens.report.v1"
METRIC = "E2E aggregate output tok/s"
MAX_BYTES = 64 * 1024 * 1024
MAX_REQUESTS = 10_000
MAX_NS = (1 << 63) - 1
HEX = re.compile(r"[0-9a-f]{64}\Z")
CONTROLS = {
    "temperature": 0,
    "top_p": 1,
    "top_k": 1,
    "min_p": 0,
    "repetition_penalty": 1,
    "presence_penalty": 0,
    "frequency_penalty": 0,
    "seed": 0,
    "stream": False,
    "n": 1,
    "echo": False,
    "return_token_ids": True,
}


class LensError(ValueError):
    """Stable public code, with details retained only in private artifacts."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def require(condition: bool, code: str) -> None:
    if not condition:
        raise LensError(code)


def integer(value, low: int, high: int, code: str) -> int:
    require(type(value) is int and low <= value <= high, code)
    return value


def string(value, code: str) -> str:
    require(
        type(value) is str and 0 < len(value) <= 1024 and value.strip() == value, code
    )
    require(all(ord(c) >= 32 for c in value), code)
    return value


def keys(value, required: set[str], optional: set[str] = frozenset()) -> dict:
    require(type(value) is dict, "OBJECT_REQUIRED")
    require(required <= value.keys() <= required | optional, "OBJECT_KEYS")
    return value


def token_ids(value, *, empty: bool = False) -> list[int]:
    require(type(value) is list and len(value) <= 1_048_576, "TOKEN_IDS_REQUIRED")
    require(empty or bool(value), "EMPTY_TOKEN_IDS")
    for token in value:
        integer(token, 0, (1 << 32) - 1, "TOKEN_ID_TYPE_OR_RANGE")
    return value


def canonical(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_json(data: bytes):
    require(type(data) is bytes and 0 < len(data) <= MAX_BYTES, "JSON_SIZE")

    def pairs(items):
        obj = {}
        for key, value in items:
            require(key not in obj, "DUPLICATE_JSON_KEY")
            obj[key] = value
        return obj

    def constant(_value):
        raise LensError("NONFINITE_JSON")

    def finite_float(value):
        result = float(value)
        require(math.isfinite(result), "NONFINITE_JSON")
        return result

    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=constant,
            parse_float=finite_float,
        )
    except LensError:
        raise
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise LensError("INVALID_JSON") from exc


def read_bytes(path: Path, *, private: bool = False) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | os.O_NONBLOCK
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        require(
            stat.S_ISREG(info.st_mode) and info.st_size <= MAX_BYTES, "UNSAFE_ARTIFACT"
        )
        if private:
            require(
                info.st_uid == os.getuid() and not info.st_mode & 0o077,
                "ARTIFACT_NOT_PRIVATE",
            )
        with os.fdopen(fd, "rb", closefd=False) as source:
            data = source.read(MAX_BYTES + 1)
        require(len(data) <= MAX_BYTES, "ARTIFACT_TOO_LARGE")
        return data
    finally:
        os.close(fd)


def write_private(path: Path, data: bytes) -> None:
    fd = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600
    )
    with os.fdopen(fd, "wb") as target:
        os.fchmod(target.fileno(), 0o600)
        target.write(data)
        target.flush()
        os.fsync(target.fileno())


def private_directory(path: Path) -> None:
    os.mkdir(path, 0o700)
    os.chmod(path, 0o700)


def check_private_directory(path: Path) -> None:
    info = path.lstat()
    require(
        stat.S_ISDIR(info.st_mode)
        and info.st_uid == os.getuid()
        and not info.st_mode & 0o077,
        "DIRECTORY_NOT_PRIVATE",
    )


def validate_corpus(value) -> dict:
    corpus = keys(value, {"schema", "model", "cases"})
    require(corpus["schema"] == CORPUS_SCHEMA, "CORPUS_SCHEMA")
    model = keys(corpus["model"], {"name", "revision"})
    string(model["name"], "MODEL_NAME")
    string(model["revision"], "MODEL_REVISION_REQUIRED")
    cases = corpus["cases"]
    require(type(cases) is list and 1 <= len(cases) <= MAX_REQUESTS, "CORPUS_CASES")
    seen = set()
    for case in cases:
        keys(case, {"id", "prompt_token_ids", "max_tokens"}, {"expected_token_ids"})
        label = string(case["id"], "CASE_ID")
        require(label not in seen, "DUPLICATE_CASE_ID")
        seen.add(label)
        token_ids(case["prompt_token_ids"])
        maximum = integer(case["max_tokens"], 1, 1_048_576, "MAX_TOKENS")
        if "expected_token_ids" in case:
            require(
                len(token_ids(case["expected_token_ids"])) <= maximum,
                "EXPECTED_TOO_LONG",
            )
    return corpus


def widths(values, *, maximum: int) -> list[int]:
    require(type(values) is list and 1 <= len(values) <= 128, "MATRIX_WIDTHS")
    for value in values:
        integer(value, 1, maximum, "MATRIX_WIDTH")
    require(len(set(values)) == len(values), "DUPLICATE_MATRIX_WIDTH")
    return values


def contract(
    corpus: dict,
    concurrency: list[int],
    bursts: list[int],
    repeats: int,
    allow_length: bool,
) -> dict:
    validate_corpus(corpus)
    widths(concurrency, maximum=128)
    widths(bursts, maximum=MAX_REQUESTS)
    integer(repeats, 1, 100, "REPEATS")
    require(type(allow_length) is bool, "LENGTH_POLICY_TYPE")
    require(min(bursts) >= max(concurrency), "BURST_SMALLER_THAN_CONCURRENCY")
    require(
        sum(bursts) * len(concurrency) * repeats * len(corpus["cases"]) <= MAX_REQUESTS,
        "TOO_MANY_REQUESTS",
    )
    return {
        "corpus": corpus,
        "corpus_sha256": sha(canonical(corpus)),
        "generation": dict(CONTROLS),
        "concurrency": concurrency,
        "burst_sizes": bursts,
        "repeats": repeats,
        "allow_length_finish": allow_length,
        "cache_policy": "require_explicit_reported_zero",
        "assignment": "per_case_cell_worker_strided_v1",
    }


def make_plan(
    corpus: dict,
    concurrency: list[int],
    bursts: list[int],
    repeats: int = 1,
    allow_length: bool = False,
) -> dict:
    workload = contract(corpus, concurrency, bursts, repeats, allow_length)
    return {
        "schema": PLAN_SCHEMA,
        "workload": workload,
        "workload_sha256": sha(canonical(workload)),
        "harness_sha256": sha(Path(__file__).read_bytes()),
    }


def validate_plan(plan) -> dict:
    keys(plan, {"schema", "workload", "workload_sha256", "harness_sha256"})
    require(plan["schema"] == PLAN_SCHEMA, "PLAN_SCHEMA")
    workload = keys(
        plan["workload"],
        {
            "corpus",
            "corpus_sha256",
            "generation",
            "concurrency",
            "burst_sizes",
            "repeats",
            "allow_length_finish",
            "cache_policy",
            "assignment",
        },
    )
    expected = contract(
        workload["corpus"],
        workload["concurrency"],
        workload["burst_sizes"],
        workload["repeats"],
        workload["allow_length_finish"],
    )
    # Canonical bytes distinguish false/0 and true/1 as well as dict contents.
    require(canonical(workload) == canonical(expected), "PLAN_CONTRACT_MISMATCH")
    require(plan["workload_sha256"] == sha(canonical(workload)), "PLAN_HASH_MISMATCH")
    require(
        type(plan["harness_sha256"]) is str and HEX.fullmatch(plan["harness_sha256"]),
        "HARNESS_HASH",
    )
    return plan


def cells(plan: dict) -> list[dict]:
    workload = plan["workload"]
    return [
        {
            "cell_index": index,
            "case_index": case_index,
            "prompt_tokens": len(case["prompt_token_ids"]),
            "max_tokens": case["max_tokens"],
            "concurrency": concurrency,
            "burst_size": burst,
            "repeat": repeat,
        }
        for index, (case_index, case, concurrency, burst, repeat) in enumerate(
            (i, case, c, b, r)
            for i, case in enumerate(workload["corpus"]["cases"])
            for c in workload["concurrency"]
            for b in workload["burst_sizes"]
            for r in range(workload["repeats"])
        )
    ]


def payload_for(plan: dict, case_index: int) -> tuple[dict, dict]:
    corpus = plan["workload"]["corpus"]
    integer(case_index, 0, len(corpus["cases"]) - 1, "CASE_INDEX")
    case = corpus["cases"][case_index]
    payload = {
        **plan["workload"]["generation"],
        "model": corpus["model"]["name"],
        "prompt_token_ids": case["prompt_token_ids"],
        "max_tokens": case["max_tokens"],
    }
    return payload, case


def endpoint_url(value: str) -> str:
    string(value, "ENDPOINT_URL")
    try:
        parsed = parse.urlsplit(value)
        require(
            parsed.scheme in ("http", "https") and bool(parsed.hostname), "ENDPOINT_URL"
        )
        require(
            parsed.username is None and parsed.password is None and parsed.port != 0,
            "ENDPOINT_CREDENTIALS_OR_PORT",
        )
        require(
            parsed.path == "/v1/completions"
            and not parsed.query
            and not parsed.fragment,
            "ENDPOINT_EXACT_COMPLETIONS_PATH",
        )
        require(not any(c.isspace() for c in value), "ENDPOINT_WHITESPACE")
    except ValueError as exc:
        raise LensError("ENDPOINT_URL") from exc
    return value


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


@dataclass(frozen=True)
class HttpReply:
    status: int
    body: bytes


def post(
    endpoint: str, payload: dict, timeout: float, api_key: str | None
) -> HttpReply:
    # A new explicit opener bypasses global openers, environment proxies, and
    # redirect replay. Exactly one POST is attempted; there are no retries.
    opener = request.build_opener(request.ProxyHandler({}), NoRedirect())
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key is not None:
        require(
            type(api_key) is str
            and api_key
            and "\r" not in api_key
            and "\n" not in api_key,
            "API_KEY_FORMAT",
        )
        headers["Authorization"] = "Bearer " + api_key
    outgoing = request.Request(
        endpoint, data=canonical(payload), headers=headers, method="POST"
    )
    try:
        response = opener.open(outgoing, timeout=timeout)
    except error.HTTPError as response:
        with response:
            return HttpReply(response.code, response.read(MAX_BYTES + 1))
    with response:
        return HttpReply(response.status, response.read(MAX_BYTES + 1))


def validate_response(body: bytes, payload: dict, allow_length: bool) -> dict:
    response = parse_json(body)
    require(
        type(response) is dict and "error" not in response, "RESPONSE_ERROR_OR_TYPE"
    )
    require(
        type(response.get("model")) is str and response["model"] == payload["model"],
        "RESPONSE_MODEL_MISMATCH",
    )
    choices = response.get("choices")
    require(
        type(choices) is list and len(choices) == 1 and type(choices[0]) is dict,
        "RESPONSE_CHOICES",
    )
    choice = choices[0]
    integer(choice.get("index"), 0, 0, "CHOICE_INDEX")
    require(
        type(choice.get("text")) is str and bool(choice["text"].strip()),
        "EMPTY_OUTPUT_TEXT",
    )
    ids = token_ids(choice.get("token_ids"))
    require(len(ids) <= payload["max_tokens"], "OUTPUT_OVER_BUDGET")
    finish = choice.get("finish_reason")
    require(
        type(finish) is str
        and (finish == "stop" or (allow_length and finish == "length")),
        "TRUNCATED_OR_UNKNOWN_FINISH",
    )
    require(
        finish != "length" or len(ids) == payload["max_tokens"], "SHORT_LENGTH_FINISH"
    )
    usage = response.get("usage")
    require(type(usage) is dict, "USAGE_REQUIRED")
    prompt_count = integer(
        usage.get("prompt_tokens"),
        len(payload["prompt_token_ids"]),
        len(payload["prompt_token_ids"]),
        "PROMPT_USAGE_MISMATCH",
    )
    count = integer(
        usage.get("completion_tokens"), len(ids), len(ids), "OUTPUT_USAGE_MISMATCH"
    )
    integer(
        usage.get("total_tokens"),
        prompt_count + count,
        prompt_count + count,
        "TOTAL_USAGE_MISMATCH",
    )
    detail = usage.get("prompt_tokens_details")
    require(type(detail) is dict, "CACHE_USAGE_UNKNOWN")
    integer(detail.get("cached_tokens"), 0, 0, "CACHE_USAGE_UNKNOWN_OR_NONZERO")
    return {"token_ids": ids, "finish_reason": finish}


def validate_identity(value) -> dict:
    require(type(value) is dict and len(value) <= 32, "DECLARED_IDENTITY")
    for key, val in value.items():
        string(key, "IDENTITY_KEY")
        string(val, "IDENTITY_VALUE")
    return value


def timestamps(receipt: dict) -> None:
    start = integer(
        receipt.get("start_monotonic_ns"), 1, MAX_NS, "TIMESTAMP_TYPE_OR_RANGE"
    )
    end = integer(receipt.get("end_monotonic_ns"), start + 1, MAX_NS, "TIMESTAMP_ORDER")
    integer(receipt.get("start_utc_ns"), 1, MAX_NS, "UTC_TIMESTAMP")
    integer(receipt.get("end_utc_ns"), 1, MAX_NS, "UTC_TIMESTAMP")
    require(math.isfinite((end - start) / 1_000_000_000), "NONFINITE_DURATION")


def run(
    plan: dict,
    endpoint: str,
    outdir: Path,
    *,
    identity: dict | None = None,
    timeout: float = 300.0,
    api_key: str | None = None,
    transport=post,
) -> dict:
    validate_plan(plan)
    require(
        plan["harness_sha256"] == sha(Path(__file__).read_bytes()),
        "HARNESS_CHANGED_SINCE_PLAN",
    )
    endpoint_url(endpoint)
    require(
        type(timeout) in (int, float)
        and math.isfinite(timeout)
        and 0 < timeout <= 86400,
        "TIMEOUT_TYPE_OR_RANGE",
    )
    identity = validate_identity({} if identity is None else identity)
    private_directory(outdir)
    private_directory(outdir / "raw")
    write_private(outdir / "plan.json", canonical(plan))
    write_private(
        outdir / "identity.json",
        canonical(
            {"endpoint": endpoint, "declared": identity, "timeout_seconds": timeout}
        ),
    )
    attempts = []
    cell_windows = []
    lock = threading.Lock()
    journal_fd = os.open(
        outdir / "journal.jsonl", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    try:
        with os.fdopen(journal_fd, "wb") as journal:
            os.fchmod(journal.fileno(), 0o600)
            offset = 0
            for cell in cells(plan):
                gate_start = []
                barrier = threading.Barrier(
                    cell["concurrency"] + 1,
                    action=lambda: gate_start.append(time.monotonic_ns()),
                )
                base_index = offset

                def worker(worker_index: int):
                    barrier.wait(timeout=min(timeout, 30))
                    for ordinal in range(
                        worker_index, cell["burst_size"], cell["concurrency"]
                    ):
                        index = base_index + ordinal
                        payload, _case = payload_for(plan, cell["case_index"])
                        prefix = f"raw/{index:06d}"
                        write_private(
                            outdir / (prefix + ".request.json"), canonical(payload)
                        )
                        receipt = {
                            "request_index": index,
                            "cell_index": cell["cell_index"],
                            "case_index": cell["case_index"],
                            "ordinal": ordinal,
                            "worker_index": worker_index,
                            "start_utc_ns": time.time_ns(),
                            "start_monotonic_ns": time.monotonic_ns(),
                            "http_status": None,
                            "transport_error": None,
                        }
                        body = b""
                        try:
                            reply = transport(endpoint, payload, timeout, api_key)
                            require(type(reply) is HttpReply, "TRANSPORT_REPLY_TYPE")
                            integer(reply.status, 100, 599, "HTTP_STATUS_TYPE")
                            require(
                                type(reply.body) is bytes
                                and len(reply.body) <= MAX_BYTES,
                                "HTTP_BODY_TYPE_OR_SIZE",
                            )
                            receipt["http_status"] = reply.status
                            body = reply.body
                        except Exception as exc:
                            receipt["transport_error"] = (
                                exc.code
                                if isinstance(exc, LensError)
                                else type(exc).__name__
                            )
                        receipt["end_monotonic_ns"] = time.monotonic_ns()
                        receipt["end_utc_ns"] = time.time_ns()
                        write_private(outdir / (prefix + ".body"), body)
                        with lock:
                            attempts.append(receipt)
                            journal.write(canonical(receipt) + b"\n")
                            journal.flush()

                with ThreadPoolExecutor(max_workers=cell["concurrency"]) as pool:
                    futures = [
                        pool.submit(worker, i) for i in range(cell["concurrency"])
                    ]
                    barrier.wait(timeout=min(timeout, 30))
                    for future in futures:
                        future.result()
                cell_windows.append(
                    {
                        "cell_index": cell["cell_index"],
                        "start_monotonic_ns": gate_start[0],
                    }
                )
                offset += cell["burst_size"]
            journal.flush()
            os.fsync(journal.fileno())
    except BaseException:
        # An interrupted run intentionally has no seal/result and cannot be
        # mistaken for a complete receipt. Its partial journal stays private.
        raise
    write_private(outdir / "windows.json", canonical(cell_windows))
    summary, _tokens = analyze(plan, attempts, cell_windows, outdir)
    write_private(outdir / "result.json", canonical(summary))
    files = artifact_files(outdir)
    manifest = {
        "schema": RUN_SCHEMA,
        "files": {name: sha(read_bytes(outdir / name, private=True)) for name in files},
    }
    write_private(outdir / "manifest.json", canonical(manifest))
    return summary


def artifact_files(directory: Path) -> list[str]:
    check_private_directory(directory)
    check_private_directory(directory / "raw")
    result = []
    for entry in directory.iterdir():
        if entry.name in ("raw", "manifest.json"):
            continue
        require(
            entry.name
            in {
                "plan.json",
                "identity.json",
                "journal.jsonl",
                "windows.json",
                "result.json",
            },
            "UNEXPECTED_ARTIFACT",
        )
        result.append(entry.name)
    for entry in (directory / "raw").iterdir():
        require(
            re.fullmatch(r"[0-9]{6}\.(?:request\.json|body)", entry.name) is not None,
            "UNEXPECTED_RAW_ARTIFACT",
        )
        result.append("raw/" + entry.name)
    return sorted(result)


def analyze(
    plan: dict, attempts: list, windows: list, directory: Path, artifact_reader=None
) -> tuple[dict, dict]:
    validate_plan(plan)
    read_artifact = artifact_reader or (
        lambda name: read_bytes(directory / name, private=True)
    )
    layout = cells(plan)
    total = sum(cell["burst_size"] for cell in layout)
    require(
        type(attempts) is list and len(attempts) == total, "MISSING_OR_EXTRA_ATTEMPTS"
    )
    require(type(windows) is list and len(windows) == len(layout), "CELL_WINDOWS")
    by_index = {}
    for receipt in attempts:
        keys(
            receipt,
            {
                "request_index",
                "cell_index",
                "case_index",
                "ordinal",
                "worker_index",
                "start_utc_ns",
                "start_monotonic_ns",
                "http_status",
                "transport_error",
                "end_monotonic_ns",
                "end_utc_ns",
            },
        )
        index = integer(receipt["request_index"], 0, total - 1, "REQUEST_INDEX")
        require(index not in by_index, "DUPLICATE_REQUEST_INDEX")
        timestamps(receipt)
        by_index[index] = receipt
    successful = {}
    failures = {}
    rows = []
    expected_requests = scored = matched = 0
    offset = 0
    previous_cell_end = 0
    for cell, window in zip(layout, windows):
        keys(window, {"cell_index", "start_monotonic_ns"})
        integer(
            window["cell_index"],
            cell["cell_index"],
            cell["cell_index"],
            "CELL_WINDOW_INDEX",
        )
        begin = integer(window["start_monotonic_ns"], 1, MAX_NS, "CELL_TIMESTAMP")
        require(begin >= previous_cell_end, "OVERLAPPING_CELLS")
        output = 0
        end = begin
        valid = 0
        cell_scored = cell_matched = 0
        latencies = []
        worker_ends = {}
        for ordinal in range(cell["burst_size"]):
            index = offset + ordinal
            receipt = by_index[index]
            integer(
                receipt["cell_index"],
                cell["cell_index"],
                cell["cell_index"],
                "CELL_INDEX",
            )
            integer(
                receipt["case_index"],
                cell["case_index"],
                cell["case_index"],
                "CASE_INDEX",
            )
            integer(receipt["ordinal"], ordinal, ordinal, "REQUEST_ORDINAL")
            worker = ordinal % cell["concurrency"]
            integer(receipt["worker_index"], worker, worker, "WORKER_INDEX")
            require(
                receipt["start_monotonic_ns"]
                >= max(begin, worker_ends.get(worker, begin)),
                "REQUEST_TIMELINE",
            )
            worker_ends[worker] = receipt["end_monotonic_ns"]
            end = max(end, receipt["end_monotonic_ns"])
            payload, case = payload_for(plan, cell["case_index"])
            expected_requests += "expected_token_ids" in case
            prefix = f"raw/{index:06d}"
            require(
                canonical(parse_json(read_artifact(prefix + ".request.json")))
                == canonical(payload),
                "REQUEST_PAYLOAD_MISMATCH",
            )
            body = read_artifact(prefix + ".body")
            try:
                if receipt["transport_error"] is not None:
                    string(receipt["transport_error"], "TRANSPORT_ERROR_TYPE")
                    require(
                        receipt["http_status"] is None, "TRANSPORT_ERROR_WITH_STATUS"
                    )
                    raise LensError("TRANSPORT_ERROR")
                integer(receipt["http_status"], 100, 599, "HTTP_STATUS_TYPE")
                require(receipt["http_status"] == 200, "HTTP_STATUS_NOT_200")
                response = validate_response(
                    body, payload, plan["workload"]["allow_length_finish"]
                )
                ids = response["token_ids"]
                successful[index] = {
                    "token_ids": ids,
                    "finish_reason": response["finish_reason"],
                }
                output += len(ids)
                valid += 1
                latencies.append(
                    (receipt["end_monotonic_ns"] - receipt["start_monotonic_ns"])
                    / 1_000_000_000
                )
                if "expected_token_ids" in case:
                    scored += 1
                    matched += ids == case["expected_token_ids"]
                    cell_scored += 1
                    cell_matched += ids == case["expected_token_ids"]
            except LensError as exc:
                failures[exc.code] = failures.get(exc.code, 0) + 1
        elapsed = (end - begin) / 1_000_000_000
        require(math.isfinite(elapsed) and elapsed > 0, "CELL_DURATION")
        complete = valid == cell["burst_size"]
        cell_expected = cell["burst_size"] if "expected_token_ids" in case else 0
        cell_score = score_status(cell_expected, cell_scored, cell_matched)
        latencies.sort()
        rows.append(
            {
                **cell,
                "valid_requests": valid,
                "output_tokens": output,
                "elapsed_seconds": elapsed,
                "corpus_exact_match": {
                    "status": cell_score,
                    "expected_requests": cell_expected,
                    "scored_requests": cell_scored,
                    "matched_requests": cell_matched,
                },
                "e2e_request_latency_seconds": {
                    "valid_requests": len(latencies),
                    "p50": (
                        latencies[math.ceil(len(latencies) * 0.50) - 1]
                        if latencies
                        else None
                    ),
                    "p95": (
                        latencies[math.ceil(len(latencies) * 0.95) - 1]
                        if latencies
                        else None
                    ),
                    "percentile_method": "nearest_rank",
                },
                "e2e_aggregate_output_tokens_per_second": (
                    output / elapsed if complete else None
                ),
            }
        )
        offset += cell["burst_size"]
        previous_cell_end = end
    complete = len(successful) == total
    score_state = score_status(expected_requests, scored, matched)
    summary = {
        "schema": REPORT_SCHEMA,
        "receipt_status": "complete" if complete else "failed",
        "workload_sha256": plan["workload_sha256"],
        "corpus_sha256": plan["workload"]["corpus_sha256"],
        "harness_sha256": plan["harness_sha256"],
        "metric": METRIC,
        "requests": total,
        "valid_requests": len(successful),
        "failures": failures,
        "corpus_exact_match": {
            "status": score_state,
            "scored_requests": scored,
            "expected_requests": expected_requests,
            "matched_requests": matched,
            "invalid_scored_requests": expected_requests - scored,
            "unscored_requests": total - expected_requests,
        },
        "quality_qualification": "not_established",
        "intervention_attribution": "unverified",
        "identity_status": "declared_not_verified",
        "native_mtp_width": "unknown",
        "engine_active_batch_width": "unknown",
        "cold_process_status": "unverified",
        "cache_policy": "each_valid_response_explicitly_reports_zero_cached_tokens",
        "cells": rows,
    }
    return summary, successful


def score_status(expected: int, scored: int, matched: int) -> str:
    if expected == 0:
        return "unscored"
    if matched != scored:
        return "mismatch"
    return "incomplete" if scored < expected else "all_matched"


def load_run(directory: Path) -> tuple[dict, dict, dict]:
    check_private_directory(directory)
    manifest = keys(
        parse_json(read_bytes(directory / "manifest.json", private=True)),
        {"schema", "files"},
    )
    require(
        manifest["schema"] == RUN_SCHEMA and type(manifest["files"]) is dict,
        "RUN_MANIFEST",
    )
    files = artifact_files(directory)
    require(set(manifest["files"]) == set(files), "MANIFEST_FILE_SET")
    required = {
        "plan.json",
        "identity.json",
        "journal.jsonl",
        "windows.json",
        "result.json",
    }
    require(required <= set(files), "RUN_MISSING_ARTIFACT")
    for name in files:
        expected = manifest["files"][name]
        require(type(expected) is str and HEX.fullmatch(expected), "MANIFEST_DIGEST")

    def verified_read(name):
        data = read_bytes(directory / name, private=True)
        require(sha(data) == manifest["files"][name], "ARTIFACT_HASH_MISMATCH")
        return data

    # Hash the same immutable bytes that each parser consumes, not an earlier
    # filesystem pass followed by an unchecked reopen of mutable artifacts.
    plan = validate_plan(parse_json(verified_read("plan.json")))
    identity = keys(
        parse_json(verified_read("identity.json")),
        {"endpoint", "declared", "timeout_seconds"},
    )
    endpoint_url(identity["endpoint"])
    validate_identity(identity["declared"])
    timeout = identity["timeout_seconds"]
    require(
        type(timeout) in (int, float)
        and math.isfinite(timeout)
        and 0 < timeout <= 86400,
        "TIMEOUT_TYPE_OR_RANGE",
    )
    journal = verified_read("journal.jsonl")
    require(journal.endswith(b"\n"), "TRUNCATED_JOURNAL")
    attempts = [parse_json(line) for line in journal.splitlines()]
    windows = parse_json(verified_read("windows.json"))
    summary, tokens = analyze(
        plan, attempts, windows, directory, artifact_reader=verified_read
    )
    expected_raw = {
        f"raw/{index:06d}.{suffix}"
        for index in range(summary["requests"])
        for suffix in ("body", "request.json")
    }
    require(set(files) == required | expected_raw, "UNEXPECTED_OR_MISSING_RAW")
    require(
        canonical(parse_json(verified_read("result.json"))) == canonical(summary),
        "RESULT_RECOMPUTE_MISMATCH",
    )
    return plan, summary, tokens


def compare(baseline: Path, candidate: Path) -> dict:
    require(not baseline.samefile(candidate), "COMPARISON_SAME_RUN")
    left_plan, left, left_tokens = load_run(baseline)
    right_plan, right, right_tokens = load_run(candidate)
    require(
        canonical(left_plan["workload"]) == canonical(right_plan["workload"]),
        "COMPARISON_WORKLOAD_MISMATCH",
    )
    require(
        left_plan["harness_sha256"] == right_plan["harness_sha256"],
        "COMPARISON_HARNESS_MISMATCH",
    )
    require(
        left["receipt_status"] == right["receipt_status"] == "complete",
        "COMPARISON_INCOMPLETE_RUN",
    )
    require(left_tokens == right_tokens, "COMPARISON_TOKEN_OR_FINISH_MISMATCH")
    require(
        left["corpus_exact_match"]["status"] != "mismatch"
        and right["corpus_exact_match"]["status"] != "mismatch",
        "COMPARISON_EXPECTED_TOKEN_MISMATCH",
    )
    ratios = [
        {
            "cell_index": a["cell_index"],
            "case_index": a["case_index"],
            "prompt_tokens": a["prompt_tokens"],
            "max_tokens": a["max_tokens"],
            "concurrency": a["concurrency"],
            "burst_size": a["burst_size"],
            "repeat": a["repeat"],
            "expected_token_check": a["corpus_exact_match"]["status"],
            "ratio_scope": "observed_e2e_ratio_with_token_parity_not_quality_qualification",
            "baseline": a["e2e_aggregate_output_tokens_per_second"],
            "candidate": b["e2e_aggregate_output_tokens_per_second"],
            "observed_candidate_over_baseline": b[
                "e2e_aggregate_output_tokens_per_second"
            ]
            / a["e2e_aggregate_output_tokens_per_second"],
        }
        for a, b in zip(left["cells"], right["cells"])
    ]
    return {
        "schema": "flux-lens.compare.v1",
        "comparability": "matched",
        "token_and_finish_parity": True,
        "metric": METRIC,
        "workload_sha256": left_plan["workload_sha256"],
        "quality_qualification": "not_established",
        "intervention_attribution": "unverified",
        "identity_status": "declared_not_verified",
        "native_mtp_width": "unknown",
        "cells": ratios,
    }


def number_list(value: str) -> list[int]:
    require(
        re.fullmatch(r"[1-9][0-9]*(?:,[1-9][0-9]*)*", value) is not None,
        "MATRIX_SYNTAX",
    )
    return [int(item) for item in value.split(",")]


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    plan = commands.add_parser(
        "plan", help="validate exact-token corpus and seal an offline workload plan"
    )
    plan.add_argument("--corpus", type=Path, required=True)
    plan.add_argument("--concurrency", default="1")
    plan.add_argument(
        "--burst-sizes",
        help="total requests per finite burst, each >= highest concurrency",
    )
    plan.add_argument("--repeats", type=int, default=1)
    plan.add_argument(
        "--allow-length",
        action="store_true",
        help="explicitly permit a full max_tokens output",
    )
    plan.add_argument("--out", type=Path, required=True)
    execute = commands.add_parser(
        "run", help="explicitly POST the plan to a blocking completion endpoint"
    )
    execute.add_argument("--plan", type=Path, required=True)
    execute.add_argument("--endpoint", required=True)
    execute.add_argument("--outdir", type=Path, required=True)
    execute.add_argument(
        "--identity",
        type=Path,
        help="optional declared metadata JSON, never attestation",
    )
    execute.add_argument("--timeout", type=float, default=300.0)
    execute.add_argument(
        "--api-key-env", help="read bearer key only from this environment variable"
    )
    paired = commands.add_parser(
        "compare", help="fail closed unless workloads and every output token match"
    )
    paired.add_argument("baseline", type=Path)
    paired.add_argument("candidate", type=Path)
    report = commands.add_parser(
        "report", help="verify seal and recompute a path/URL-free reusable report"
    )
    report.add_argument("run_directory", type=Path)
    return cli


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "plan":
            concurrency = number_list(args.concurrency)
            bursts = (
                number_list(args.burst_sizes)
                if args.burst_sizes
                else [max(concurrency)]
            )
            plan = make_plan(
                parse_json(read_bytes(args.corpus)),
                concurrency,
                bursts,
                args.repeats,
                args.allow_length,
            )
            write_private(args.out, canonical(plan))
            output = {
                "schema": PLAN_SCHEMA,
                "status": "planned_offline",
                "workload_sha256": plan["workload_sha256"],
                "cells": cells(plan),
                "network_contacted": False,
            }
        elif args.command == "run":
            key = None
            if args.api_key_env:
                require(
                    re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.api_key_env)
                    is not None,
                    "API_KEY_ENV_NAME",
                )
                key = os.environ.get(args.api_key_env)
                require(key is not None and bool(key), "API_KEY_ENV_UNSET")
            identity = parse_json(read_bytes(args.identity)) if args.identity else {}
            output = run(
                parse_json(read_bytes(args.plan)),
                args.endpoint,
                args.outdir,
                identity=identity,
                timeout=args.timeout,
                api_key=key,
            )
        elif args.command == "compare":
            output = compare(args.baseline, args.candidate)
        else:
            _plan, output, _tokens = load_run(args.run_directory)
        print(canonical(output).decode("utf-8"))
        return int(
            output.get("receipt_status") == "failed"
            or output.get("corpus_exact_match", {}).get("status") == "mismatch"
        )
    except LensError as exc:
        print(
            canonical({"status": "refused", "code": exc.code}).decode("utf-8"),
            file=sys.stderr,
        )
        return 2
    except (OSError, ValueError, TypeError, OverflowError, RecursionError):
        # Never print exceptions here: URLs, private paths, tokens, and caller
        # metadata can all appear in third-party or OS exception messages.
        print('{"status":"refused","code":"INPUT_OR_IO_ERROR"}', file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
