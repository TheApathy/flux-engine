#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Matched plain-versus-native-MTP qualification for Qwen4 long context.

This is the narrow runtime gate for flux-engine issue #13.  It launches three
fresh processes from one binary and checkpoint, constructs exact-token prompts
covering both sides of the 2,048-token QSA boundary, every prompt length
reported on the issue, and the native 262,144-token endpoint.  It sends
byte-for-byte identical greedy requests to the paired arms:

* ``plain`` is target-only with ``FLUX_SPEC_MODE=none``
* ``k5`` is the checkpoint-native MTP proposer with four drafts (K=5 verify)
* ``k5_eager_churn`` forces eager K=5 for boundary, rollback, and slot reuse

A PASS requires known-answer correctness in both arms, scheduler-owned output
token-ID identity, exact completion text, parsed/final answer, finish reason,
and token-count parity for every paired request.  It also requires an observed
native-MTP verify frame and a completed partial-accept cycle between two exact
single-slot canaries.  Length-matched synthetic cases are identified as such;
they do not claim byte identity with a reporter's private prompt.  This script
is not a performance benchmark and a direct-binary PASS is not a container
PASS.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

_REPRO_KIT_DIR = str(Path(__file__).resolve().parent)
if _REPRO_KIT_DIR not in sys.path:
    sys.path.insert(0, _REPRO_KIT_DIR)
import qwen4_endpoint_admission as endpoint_gate  # noqa: E402
import r9_source_to_binary as r9_provenance  # noqa: E402

SCHEMA = "flux-qwen4-mtp-long-context-pair-receipt-v5"
MANIFEST_SCHEMA = "flux-qwen4-mtp-long-context-pair-launch-v5"
REQUEST_SCHEMA = "flux-qwen4-mtp-long-context-request-manifest-v4"
SOURCE_TO_BINARY_SCHEMA = "flux-source-to-binary-receipt/1"
CHECKPOINT_INVENTORY_SCHEMA = "flux-checkpoint-inventory/1"
RELEASE_VERSION = "v0.9.13-r15"

# Each case is a deterministic known-answer synthetic analogue.  The lengths
# prefixed with ``reporter`` bind every row disclosed in issue #13; the edge
# rows isolate dispatch at and around the 2,048-token sparse-QSA boundary.
ISSUE13_CASE_SPECS = (
    ("reporter-control-2017", 2_017, "reporter-length-analogue"),
    ("qsa-edge-2043", 2_043, "qsa-verify-endpoint-boundary"),
    ("qsa-edge-2044", 2_044, "qsa-verify-endpoint-boundary"),
    ("qsa-edge-2045", 2_045, "qsa-boundary"),
    ("qsa-edge-2046", 2_046, "qsa-boundary"),
    ("qsa-edge-2047", 2_047, "qsa-boundary"),
    ("qsa-edge-2048", 2_048, "qsa-boundary"),
    ("qsa-edge-2049", 2_049, "qsa-boundary"),
    ("qsa-edge-2050", 2_050, "qsa-boundary"),
    ("reporter-2112", 2_112, "reporter-length-analogue"),
    ("ladder-4096", 4_096, "qsa-long-context"),
    ("reporter-wiki-7336", 7_336, "reporter-length-analogue"),
    ("reporter-depths-7710", 7_710, "reporter-length-and-depth-analogue"),
    ("ladder-8192", 8_192, "qsa-long-context"),
    ("reporter-30647", 30_647, "reporter-length-analogue"),
    ("ladder-32768", 32_768, "qsa-long-context"),
    ("ladder-65536", 65_536, "qsa-long-context"),
    ("reporter-114755", 114_755, "reporter-length-analogue"),
    ("reporter-122413", 122_413, "reporter-length-analogue"),
    ("ladder-131072", 131_072, "qsa-long-context"),
    ("reporter-238938", 238_938, "reporter-length-analogue"),
)
FULL_ENDPOINT_CASE_SPEC = (
    "native-endpoint-262080-plus-64",
    262_080,
    "native-262144-total-token-fill-identity",
)
ENDPOINT_SEMANTIC_CASE_SPEC = (
    "native-endpoint-262080-semantic",
    262_080,
    "native-endpoint-stopped-retrieval-semantics",
)
ADMISSION_ACCEPT_CASE_SPEC = (
    "native-admission-262143-plus-1",
    262_143,
    "native-262144-total-token-admission",
)
ADMISSION_REJECT_CASE_SPEC = (
    "native-rejection-262144-plus-1",
    262_144,
    "native-262145-total-token-rejection",
)
PAIRED_CASE_SPECS = (
    *ISSUE13_CASE_SPECS,
    ENDPOINT_SEMANTIC_CASE_SPEC,
    FULL_ENDPOINT_CASE_SPEC,
)
CASE_SPECS = (
    *PAIRED_CASE_SPECS,
    ADMISSION_ACCEPT_CASE_SPEC,
    ADMISSION_REJECT_CASE_SPEC,
)
PAIRED_CASE_IDS = tuple(spec[0] for spec in PAIRED_CASE_SPECS)
PROMPT_TOKEN_COUNTS = tuple(dict.fromkeys(spec[1] for spec in CASE_SPECS))
PAIRED_PROMPT_TOKEN_COUNTS = tuple(dict.fromkeys(spec[1] for spec in PAIRED_CASE_SPECS))
MAX_TOKENS = 96
FULL_ENDPOINT_MAX_TOKENS = 64
NATIVE_CONTEXT = FULL_ENDPOINT_CASE_SPEC[1] + FULL_ENDPOINT_MAX_TOKENS
REPETITIONS = 2
FULL_ENDPOINT_REPETITIONS = 1
ADMISSION_PROBE_MAX_TOKENS = 1
SEED = 0x13_20_48_55
COMPONENT_KEYS = ("prefix", "filler", "early", "middle", "late", "query")
# Needles land near 1%, 50%, and 98% of the prompt.  This directly covers the
# middle and 98%-depth retrieval positions reported for the 7,710-token row.
GAP_WEIGHTS = (1, 49, 48, 2)
PRIMARY_ARM_NAMES = ("plain", "k5")
EAGER_CHURN_ARM = "k5_eager_churn"
ARM_NAMES = (*PRIMARY_ARM_NAMES, EAGER_CHURN_ARM)
GRAPH_DISABLED_CASE_IDS = tuple(
    f"qsa-edge-{prompt_tokens}" for prompt_tokens in range(2_043, 2_051)
)
ROLLBACK_SENTINEL_CASE_ID = "reporter-control-2017"
ROLLBACK_TRIGGER_CASE_ID = "qsa-edge-2050"
ROLLBACK_SLOT_REUSE_SEQUENCE = (
    ("sentinel-before", ROLLBACK_SENTINEL_CASE_ID),
    ("rollback-trigger", ROLLBACK_TRIGGER_CASE_ID),
    ("sentinel-after", ROLLBACK_SENTINEL_CASE_ID),
)
ARM_ENVIRONMENTS = {
    "plain": {
        "FLUX_SPEC_MODE": "none",
        "FLUX_GAMMA": "0",
        "ATLAS_DISABLE_SPECULATION": "1",
    },
    "k5": {
        "FLUX_SPEC_MODE": "native-mtp",
        "FLUX_GAMMA": "4",
        "ATLAS_DISABLE_SPECULATION": "0",
    },
    EAGER_CHURN_ARM: {
        "FLUX_SPEC_MODE": "native-mtp",
        "FLUX_GAMMA": "4",
        "ATLAS_DISABLE_SPECULATION": "0",
        "ATLAS_QWEN4_VERIFY_GRAPH": "0",
        "ATLAS_DFLASH_DEBUG_NO_GRAPH": "1",
        "ATLAS_QWEN4_QSA_ROUTE_TELEMETRY": "1",
    },
}
SPECULATION_ENV = frozenset(
    {"FLUX_SPEC_MODE", "FLUX_GAMMA", "ATLAS_DISABLE_SPECULATION"}
)

PLAIN_SCHEDULER_MARKER = (
    "Scheduler started (batched mode, max_batch=1, mtp=false, ngram=false, "
    "num_drafts=0"
)
K5_SCHEDULER_MARKER = (
    "Scheduler started (batched mode, max_batch=1, mtp=true, ngram=false, "
    "num_drafts=4"
)
K5_CONSTRUCTION_MARKER = "Qwen4 native MTP head constructed"
K5_VERIFY_MARKER = "MTP_ACCEPT frame"
# Native-K5 draft policy the log parsers enforce. Fixed K5 (the default) drafts
# every chain to gamma=4 and verifies 5 rows. Confidence-gated K5 stops a chain
# at its first low-confidence draft, so a frame carries 1..4 drafts and verifies
# 2..5 rows; chains shorter than 4 run the K2..K4 verify steps, which emit an
# MTP_ACCEPT frame but no SPEC_CYCLE_V2 record. Callers opt in through
# configure_k5_draft_policy; verify still decides every committed token.
K5_FIXED_POLICY = "fixed-k5"
K5_GATED_POLICY = "confidence-gated-k5"
K5_NUM_DRAFTS = 4
K5_DRAFT_POLICY = K5_FIXED_POLICY
SPEC_CYCLE_MARKER = "SPEC_CYCLE_V2 "
K5_QSA_VERIFY_ROWS_MARKER = (
    "QSA_VERIFY_ROWS_ENGAGED selector=ATLAS_QWEN4_QSA_VERIFY_ROWS"
)
K5_VERIFY_GRAPH_CAPTURE_MARKER = "Captured CUDA graph for K=γ verify"
QSA_ROUTE_TELEMETRY_ENV = "ATLAS_QWEN4_QSA_ROUTE_TELEMETRY"
QSA_ROUTE_TELEMETRY_ENGAGED_MARKER = (
    f"QSA_VERIFY_ROUTE_TELEMETRY_ENGAGED selector={QSA_ROUTE_TELEMETRY_ENV}"
)
QSA_ROUTE_RECEIPT_MARKER = "QSA_VERIFY_ROUTE_RECEIPT "
QSA_FULL_ATTENTION_LAYER_IDS = tuple(range(12))
FORBIDDEN_LOG_MARKERS = (
    "CUDA out of memory",
    "KV cache exhausted",
    "NVRM: Xid",
    "illegal memory access",
    "context overflow truncated",
    "panicked at",
    "thread 'main' panicked",
)
REJECTION_METRIC_NAMES = endpoint_gate.METRIC_NAMES
ANSWER_RE = re.compile(
    r"EARLY=([0-9A-F]{16}) MIDDLE=([0-9A-F]{16}) LATE=([0-9A-F]{16})"
)
ANSWER_LABEL_RE = re.compile(r"(?i)\b(EARLY|MIDDLE|LATE)\s*=")


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def strict_json_loads(payload: bytes | str, source: str) -> Any:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{source} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"{source} contains non-finite JSON number {value!r}")

    try:
        value = json.loads(
            payload,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"{source} is not strict JSON") from error
    validate_finite_json_numbers(value, source)
    return value


def validate_finite_json_numbers(value: Any, source: str) -> None:
    """Reject exponent-overflow floats that ``parse_constant`` cannot see."""

    if type(value) is float and not math.isfinite(value):
        raise ValueError(f"{source} contains a non-finite JSON number")
    if isinstance(value, list):
        for item in value:
            validate_finite_json_numbers(item, source)
    elif isinstance(value, dict):
        for item in value.values():
            validate_finite_json_numbers(item, source)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_hex(value: object, width: int) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(rf"[0-9a-f]{{{width}}}", value))


def write_all(fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(fd, payload[offset:])
        if written <= 0:
            raise OSError("receipt write made no progress")
        offset += written


def token_u32le_sha256(tokens: list[int]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(token.to_bytes(4, "little"))
    return digest.hexdigest()


prompt_sha256 = token_u32le_sha256


def validate_token_vector(value: object, label: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} token vector is empty")
    if any(type(token) is not int or not 0 <= token <= 0xFFFF_FFFF for token in value):
        raise ValueError(f"{label} contains a non-u32 token")
    return value


def code_for(case_id: str, prompt_tokens: int, label: str) -> str:
    material = f"flux-issue-13-v2:{case_id}:{prompt_tokens}:{label}".encode("ascii")
    return hashlib.sha256(material).hexdigest()[:16].upper()


def component_texts(
    case_id: str, prompt_tokens: int
) -> tuple[dict[str, str], dict[str, str]]:
    if prompt_tokens not in PROMPT_TOKEN_COUNTS:
        raise ValueError("prompt length is outside the issue-13 workload")
    codes = {
        name: code_for(case_id, prompt_tokens, name)
        for name in ("early", "middle", "late")
    }
    texts = {
        "prefix": (
            "Archive accuracy audit. Read every record. Ignore neutral filler. "
            "At the final query, return only the three requested codes in the exact format.\n"
        ),
        "filler": (
            "Neutral archive record: amber basalt cedar delta harbor ivory juniper. "
            "This record contains no audit code.\n"
        ),
        "early": f"\n[EARLY] audit code {codes['early']} [/EARLY]\n",
        "middle": f"\n[MIDDLE] audit code {codes['middle']} [/MIDDLE]\n",
        "late": f"\n[LATE] audit code {codes['late']} [/LATE]\n",
        "query": (
            "\nReturn one line only: EARLY=<early-code> MIDDLE=<middle-code> "
            "LATE=<late-code>\nAnswer:"
        ),
    }
    return texts, codes


def expected_answer(codes: dict[str, str]) -> str:
    if set(codes) != {"early", "middle", "late"}:
        raise ValueError("needle code set differs")
    return f"EARLY={codes['early']} MIDDLE={codes['middle']} LATE={codes['late']}"


def repeated(tokens: list[int], length: int) -> list[int]:
    repeats, tail = divmod(length, len(tokens))
    return tokens * repeats + tokens[:tail]


def assemble_prompt(
    prompt_tokens: int, components: dict[str, list[int]]
) -> tuple[list[int], dict[str, int]]:
    if prompt_tokens not in PROMPT_TOKEN_COUNTS:
        raise ValueError("prompt length is outside the issue-13 workload")
    if tuple(components) != COMPONENT_KEYS:
        raise ValueError("component token map has wrong keys or order")
    for label, tokens in components.items():
        validate_token_vector(tokens, label)
    fixed = sum(len(components[name]) for name in COMPONENT_KEYS if name != "filler")
    if fixed >= prompt_tokens:
        raise ValueError("fixed components leave no filler capacity")
    remaining = prompt_tokens - fixed
    gaps = [remaining * weight // sum(GAP_WEIGHTS) for weight in GAP_WEIGHTS[:-1]]
    gaps.append(remaining - sum(gaps))
    prompt = list(components["prefix"])
    positions: dict[str, int] = {}
    for index, label in enumerate(("early", "middle", "late")):
        prompt.extend(repeated(components["filler"], gaps[index]))
        positions[label] = len(prompt)
        prompt.extend(components[label])
    prompt.extend(repeated(components["filler"], gaps[-1]))
    prompt.extend(components["query"])
    if len(prompt) != prompt_tokens:
        raise AssertionError("exact prompt assembly drifted")
    return prompt, positions


def request_contract(case_id: str) -> dict[str, Any]:
    if case_id == ENDPOINT_SEMANTIC_CASE_SPEC[0]:
        return {
            "max_tokens": FULL_ENDPOINT_MAX_TOKENS,
            "repetitions": FULL_ENDPOINT_REPETITIONS,
            "ignore_eos": False,
            "stop": ["\n"],
            "validation": "exact-known-answer",
        }
    if case_id == FULL_ENDPOINT_CASE_SPEC[0]:
        return {
            "max_tokens": FULL_ENDPOINT_MAX_TOKENS,
            "repetitions": FULL_ENDPOINT_REPETITIONS,
            "ignore_eos": True,
            "stop": [],
            "validation": "native-endpoint-exact-token-identity",
        }
    if case_id == ADMISSION_ACCEPT_CASE_SPEC[0]:
        return {
            "max_tokens": ADMISSION_PROBE_MAX_TOKENS,
            "repetitions": 1,
            "ignore_eos": True,
            "stop": [],
            "validation": "native-endpoint-admission",
        }
    if case_id == ADMISSION_REJECT_CASE_SPEC[0]:
        return {
            "max_tokens": ADMISSION_PROBE_MAX_TOKENS,
            "repetitions": 1,
            "ignore_eos": True,
            "stop": [],
            "validation": "context-length-rejection",
        }
    if case_id not in {spec[0] for spec in ISSUE13_CASE_SPECS}:
        raise ValueError(f"unknown issue-13 case {case_id}")
    return {
        "max_tokens": MAX_TOKENS,
        "repetitions": REPETITIONS,
        "ignore_eos": False,
        "stop": ["\n"],
        "validation": "exact-known-answer",
    }


def launch_request_contract() -> dict[str, Any]:
    return {
        "prompt_token_counts": list(PROMPT_TOKEN_COUNTS),
        "paired_prompt_token_counts": list(PAIRED_PROMPT_TOKEN_COUNTS),
        "max_tokens": MAX_TOKENS,
        "repetitions": REPETITIONS,
        "seed": SEED,
        "sampling": "exact-greedy",
        "endpoint_semantic": {
            "case_id": ENDPOINT_SEMANTIC_CASE_SPEC[0],
            "prompt_tokens": ENDPOINT_SEMANTIC_CASE_SPEC[1],
            "max_tokens": FULL_ENDPOINT_MAX_TOKENS,
            "maximum_total_tokens": NATIVE_CONTEXT,
            "repetitions": FULL_ENDPOINT_REPETITIONS,
            "ignore_eos": False,
            "stop": ["\n"],
            "validation": "exact-single-structured-answer",
        },
        "full_endpoint": {
            "case_id": FULL_ENDPOINT_CASE_SPEC[0],
            "prompt_tokens": FULL_ENDPOINT_CASE_SPEC[1],
            "max_tokens": FULL_ENDPOINT_MAX_TOKENS,
            "total_tokens": NATIVE_CONTEXT,
            "repetitions": FULL_ENDPOINT_REPETITIONS,
            "ignore_eos": True,
            "stop": [],
            "return_token_ids": True,
        },
        "admission_boundary": {
            "execution_arm": "plain",
            "accepted": {
                "case_id": ADMISSION_ACCEPT_CASE_SPEC[0],
                "prompt_tokens": ADMISSION_ACCEPT_CASE_SPEC[1],
                "max_tokens": ADMISSION_PROBE_MAX_TOKENS,
                "total_tokens": NATIVE_CONTEXT,
                "ignore_eos": True,
                "stop": [],
                "return_token_ids": True,
            },
            "rejected": {
                "case_id": ADMISSION_REJECT_CASE_SPEC[0],
                "prompt_tokens": ADMISSION_REJECT_CASE_SPEC[1],
                "max_tokens": ADMISSION_PROBE_MAX_TOKENS,
                "total_tokens": NATIVE_CONTEXT + 1,
                "required_http_status": 400,
                "must_not_dispatch": True,
                "post_rejection_health_canary": ROLLBACK_SENTINEL_CASE_ID,
            },
        },
        "graph_disabled_boundary_case_ids": list(GRAPH_DISABLED_CASE_IDS),
        "rollback_slot_reuse_sequence": [
            {"phase": phase, "case_id": case_id}
            for phase, case_id in ROLLBACK_SLOT_REUSE_SEQUENCE
        ],
    }


def completion_body(
    model: str,
    prompt: list[int],
    *,
    max_tokens: int = MAX_TOKENS,
    ignore_eos: bool = False,
    stop: list[str] | None = None,
) -> dict[str, Any]:
    validate_token_vector(prompt, "prompt")
    if not model or type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("model and positive max_tokens are required")
    if stop is None:
        stop = ["\n"]
    if not isinstance(stop, list) or any(not isinstance(value, str) for value in stop):
        raise ValueError("stop must be a string array")
    return {
        "model": model,
        "prompt": "",
        "prompt_token_ids": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "top_n_sigma": 0.0,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "stream": False,
        "return_token_ids": True,
        "stop": stop,
        "seed": SEED,
        "ignore_eos": ignore_eos,
    }


def parse_final_answer(text: str) -> dict[str, str]:
    matches = ANSWER_RE.findall(text)
    if len(matches) != 1:
        raise ValueError("completion must contain exactly one canonical final answer")
    early, middle, late = matches[0]
    labels = [label.upper() for label in ANSWER_LABEL_RE.findall(text)]
    if labels != ["EARLY", "MIDDLE", "LATE"]:
        raise ValueError(
            "completion contains another or malformed structured answer assignment"
        )
    return {"early": early, "middle": middle, "late": late}


def canonical_answer(parsed: dict[str, str]) -> str:
    return f"EARLY={parsed['early']} MIDDLE={parsed['middle']} LATE={parsed['late']}"


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            raise ValueError(f"{path}:{number}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", key):
            raise ValueError(f"{path}:{number}: invalid environment key")
        if key in values:
            raise ValueError(f"{path}:{number}: duplicate environment key {key}")
        values[key] = value.strip().strip('"').strip("'")
    return values


def validate_common_environment(values: dict[str, str]) -> None:
    required = {
        "FLUX_CONCURRENCY": "1",
        "FLUX_KV_CACHE_DTYPE": "bf16",
        "FLUX_CTX_OVERFLOW_TRUNCATE": "0",
        "ATLAS_CTX_OVERFLOW_TRUNCATE": "0",
        # Keep the optimized verifier enabled so the 2,043/2,044 rows straddle
        # its K=5 endpoint and exercise the source-pinned safety fallback.
        "ATLAS_QWEN4_QSA_VERIFY_ROWS": "1",
        # The primary K5 arm uses the release graph path. The auxiliary K5
        # arm overrides this exact selector to zero in a fresh process.
        "ATLAS_QWEN4_VERIFY_GRAPH": "1",
        "ATLAS_QWEN4_QSA_GRAPH_SAFE": "1",
        "ATLAS_MTP_ACCEPT_LOG": "1",
        # A complete schema-2 record is emitted only after commit, proposer
        # state, and finalize phases.  It upgrades the pre-commit acceptance
        # trace into explicit evidence that a partial-accept rollback finished.
        "ATLAS_DFLASH_ASYNC": "0",
        "ATLAS_DFLASH_SPEC_CYCLE_V2": "1",
    }
    for key, expected in required.items():
        if values.get(key) != expected:
            raise ValueError(f"common environment requires {key}={expected}")
    if any(key in values for key in SPECULATION_ENV):
        raise ValueError("speculation controls must appear only in the arm environment")
    arm_only_environment = {
        "ATLAS_DFLASH_DEBUG_NO_GRAPH",
        QSA_ROUTE_TELEMETRY_ENV,
    }
    arm_only_present = sorted(arm_only_environment.intersection(values))
    if arm_only_present:
        raise ValueError(
            "forced-eager controls must appear only in the forced-eager arm: "
            f"{arm_only_present!r}"
        )
    forbidden_topology_environment = {
        "FLUX_WORLD_SIZE",
        "FLUX_TP_SIZE",
        "FLUX_EP_SIZE",
        "FLUX_HIGH_SPEED_SWAP",
    }
    present_topology_environment = sorted(
        forbidden_topology_environment.intersection(values)
    )
    if present_topology_environment:
        raise ValueError(
            "topology and HSS must be owned by the exact server argv, not the "
            f"common environment: {present_topology_environment!r}"
        )
    if "spec_accept_log=debug" not in values.get("RUST_LOG", ""):
        raise ValueError(
            "RUST_LOG must enable spec_accept_log=debug engagement receipts"
        )


def argv_value(argv: list[str], flag: str) -> str | None:
    try:
        index = argv.index(flag)
    except ValueError:
        return None
    return argv[index + 1] if index + 1 < len(argv) else None


def require_argv_value(argv: list[str], flag: str, expected: str) -> None:
    if argv.count(flag) != 1 or argv_value(argv, flag) != expected:
        raise ValueError(f"{flag} must occur once with value {expected}")


def validate_base_url(value: object, argv: list[str]) -> tuple[str, int]:
    if not isinstance(value, str):
        raise ValueError("base_url must be a string")
    parsed = urllib.parse.urlsplit(value)
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("base_url has an invalid port") from error
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or port is None
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("base_url must be an unadorned loopback HTTP origin")
    if argv_value(argv, "--bind") != parsed.hostname or argv_value(
        argv, "--port"
    ) != str(port):
        raise ValueError("base_url must match child --bind and --port")
    return parsed.hostname, port


def strip_k5_only_argv(argv: list[str]) -> list[str]:
    value_flags = {"--num-drafts", "--mtp-vocab", "--mtp-quantization"}
    result: list[str] = []
    index = 0
    while index < len(argv):
        value = argv[index]
        if value == "--speculative":
            index += 1
            continue
        if value in value_flags:
            if index + 1 >= len(argv):
                raise ValueError(f"{value} is missing its value")
            index += 2
            continue
        result.append(value)
        index += 1
    return result


def validate_arm_argv(name: str, argv: object) -> list[str]:
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(x, str) or not x for x in argv)
    ):
        raise ValueError(f"{name} argv must be a non-empty string array")
    if len(argv) < 2 or argv[1] != "serve":
        raise ValueError(f"{name} argv must select the serve command")
    required = {
        "--kv-cache-dtype": "bf16",
        "--max-batch-size": "1",
        "--max-num-seqs": "1",
        "--world-size": "1",
        "--tp-size": "1",
        "--ep-size": "1",
        "--block-size": "16",
        "--enable-prefix-caching": "false",
        "--ssm-cache-slots": "0",
        "--swap-space-gb": "0",
        "--bind": "127.0.0.1",
        "--kernel-target": "qwen3.8-flash-next",
    }
    for flag, expected in required.items():
        require_argv_value(argv, flag, expected)
    for flag in ("--model-from-path", "--model-name", "--max-prefill-tokens"):
        if argv.count(flag) != 1 or not argv_value(argv, flag):
            raise ValueError(f"{name} {flag} must occur exactly once with a value")
    if argv.count("--max-seq-len") != 1:
        raise ValueError(f"{name} --max-seq-len must occur exactly once")
    max_seq_raw = argv_value(argv, "--max-seq-len") or ""
    if not max_seq_raw.isascii() or not max_seq_raw.isdecimal():
        raise ValueError(f"{name} --max-seq-len must be decimal")
    if int(max_seq_raw) != NATIVE_CONTEXT:
        raise ValueError(
            f"{name} --max-seq-len must equal the native {NATIVE_CONTEXT}-token endpoint"
        )
    if argv.count("--port") != 1:
        raise ValueError(f"{name} --port must occur exactly once")
    port = argv_value(argv, "--port") or ""
    if not port.isascii() or not port.isdecimal() or not 1 <= int(port) <= 65_535:
        raise ValueError(f"{name} --port must be a decimal TCP port")
    if argv.count("--qwen4-qsa") != 1:
        raise ValueError(f"{name} requires --qwen4-qsa")
    forbidden = (
        "--dflash",
        "--self-speculative",
        "--ngram-speculative",
        "--mtp-from-path",
        "--rope-yarn-factor",
        "--rope-theta-override",
    )
    if any(flag in argv for flag in forbidden):
        raise ValueError(
            f"{name} contains a forbidden alternate or context-extension route"
        )
    if any(flag.startswith("--high-speed-swap") for flag in argv):
        raise ValueError(f"{name} must keep high-speed swap disabled")
    if name == "plain":
        if any(
            flag in argv
            for flag in (
                "--speculative",
                "--num-drafts",
                "--mtp-vocab",
                "--mtp-quantization",
            )
        ):
            raise ValueError("plain arm must be target-only")
    elif name in ("k5", EAGER_CHURN_ARM):
        if argv.count("--speculative") != 1:
            raise ValueError("k5 arm requires exactly one --speculative flag")
        require_argv_value(argv, "--num-drafts", "4")
        require_argv_value(argv, "--mtp-vocab", "0")
        require_argv_value(argv, "--mtp-quantization", "bf16")
    else:
        raise ValueError(f"unknown arm {name}")
    return argv


def validate_source_to_binary_receipt(
    receipt_path: Path, provenance: dict[str, Any], verify_files: bool
) -> None:
    if not receipt_path.is_file():
        if verify_files:
            raise ValueError("source-to-binary receipt is not a file")
        return
    if (
        Path(provenance["source_to_binary_receipt_path"]).resolve()
        != receipt_path.resolve()
    ):
        raise ValueError("source-to-binary receipt paths differ")
    receipt = strict_json_loads(
        receipt_path.read_bytes(), f"source-to-binary receipt {receipt_path}"
    )
    if not isinstance(receipt, dict):
        raise ValueError("source-to-binary receipt must be a JSON object")
    r9_provenance.release_identity(provenance.get("release_version"))

    original_release = r9_provenance.RELEASE_VERSION
    try:
        r9_provenance.configure_release(provenance["release_version"])
        r9_provenance.validate_qualification_provenance(provenance)
    except (OSError, r9_provenance.ProvenanceError) as exc:
        raise ValueError(f"source-to-binary provenance failed: {exc}") from exc
    finally:
        r9_provenance.configure_release(original_release)


def validate_checkpoint_inventory(
    inventory_path: Path,
    checkpoint_root: Path,
    provenance: dict[str, Any],
    verify_files: bool,
) -> None:
    if not inventory_path.is_file():
        if verify_files:
            raise ValueError("checkpoint inventory is not a file")
        return
    if sha256_file(inventory_path) != provenance["checkpoint_manifest_sha256"]:
        raise ValueError("checkpoint inventory SHA-256 mismatch")
    inventory = strict_json_loads(
        inventory_path.read_bytes(), f"checkpoint inventory {inventory_path}"
    )
    required = {
        "schema",
        "checkpoint_id",
        "checkpoint_revision",
        "file_count",
        "total_bytes",
        "files",
    }
    if not isinstance(inventory, dict) or set(inventory) != required:
        raise ValueError("checkpoint inventory keys differ")
    if (
        inventory["schema"] != CHECKPOINT_INVENTORY_SCHEMA
        or inventory["checkpoint_id"] != provenance["checkpoint_id"]
        or inventory["checkpoint_revision"] != provenance["checkpoint_revision"]
    ):
        raise ValueError("checkpoint inventory identity differs")
    files = inventory["files"]
    if not isinstance(files, list) or not files:
        raise ValueError("checkpoint inventory files must be a non-empty array")
    normalized: dict[str, tuple[int, str]] = {}
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != {"path", "bytes", "sha256"}:
            raise ValueError("checkpoint inventory file entry keys differ")
        relative = entry["path"]
        pure = PurePosixPath(relative) if isinstance(relative, str) else None
        if (
            pure is None
            or pure.is_absolute()
            or ".." in pure.parts
            or str(pure) != relative
            or relative in normalized
        ):
            raise ValueError(
                "checkpoint inventory contains an invalid or duplicate path"
            )
        size = entry["bytes"]
        digest = entry["sha256"]
        if type(size) is not int or size < 0 or not is_hex(digest, 64):
            raise ValueError("checkpoint inventory contains invalid file metadata")
        normalized[relative] = (size, digest)
    if inventory["file_count"] != len(normalized):
        raise ValueError("checkpoint inventory file_count differs")
    if inventory["total_bytes"] != sum(size for size, _ in normalized.values()):
        raise ValueError("checkpoint inventory total_bytes differs")
    if not verify_files:
        return
    if not checkpoint_root.is_dir():
        raise ValueError("checkpoint root is not a directory")
    try:
        inventory_path.resolve().relative_to(checkpoint_root.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("checkpoint inventory must be outside the checkpoint root")
    actual: list[str] = []
    for path in checkpoint_root.rglob("*"):
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"checkpoint contains a non-file entry: {path}")
        actual.append(path.relative_to(checkpoint_root).as_posix())
    actual.sort()
    if actual != sorted(normalized):
        raise ValueError(
            "checkpoint inventory file set differs from selected checkpoint"
        )
    for relative in actual:
        file_path = checkpoint_root / relative
        expected_size, expected_digest = normalized[relative]
        if (
            file_path.stat().st_size != expected_size
            or sha256_file(file_path) != expected_digest
        ):
            raise ValueError(f"checkpoint inventory mismatch for {relative}")


def validate_provenance(
    provenance: object, plain_argv: list[str], manifest_path: Path, verify_files: bool
) -> dict[str, Any]:
    keys = {
        "release_version",
        "source_revision",
        "source_tree",
        "source_root_path",
        "source_bundle_manifest_path",
        "source_bundle_manifest_sha256",
        "source_bundle_path",
        "source_bundle_sha256",
        "build_receipt_path",
        "build_receipt_sha256",
        "source_to_binary_receipt_path",
        "source_to_binary_receipt_sha256",
        "binary_path",
        "binary_sha256",
        "checkpoint_id",
        "checkpoint_revision",
        "checkpoint_path",
        "checkpoint_manifest_path",
        "checkpoint_manifest_sha256",
    }
    if not isinstance(provenance, dict) or set(provenance) != keys:
        raise ValueError("provenance keys differ")
    r9_provenance.release_identity(provenance["release_version"])
    if not is_hex(provenance["source_revision"], 40):
        raise ValueError("source_revision must be a lowercase Git SHA-1")
    if not is_hex(provenance["source_tree"], 40):
        raise ValueError("source_tree must be a lowercase Git tree SHA-1")
    for key in (
        "source_bundle_manifest_sha256",
        "source_bundle_sha256",
        "build_receipt_sha256",
        "source_to_binary_receipt_sha256",
        "binary_sha256",
        "checkpoint_manifest_sha256",
    ):
        if not is_hex(provenance[key], 64):
            raise ValueError(f"{key} must be lowercase SHA-256")
    if not (
        is_hex(provenance["checkpoint_revision"], 40)
        or is_hex(provenance["checkpoint_revision"], 64)
    ):
        raise ValueError("checkpoint_revision must be an immutable hash")
    if (
        not isinstance(provenance["checkpoint_id"], str)
        or not provenance["checkpoint_id"]
    ):
        raise ValueError("checkpoint_id must be non-empty")

    path_keys = (
        "source_root_path",
        "source_bundle_manifest_path",
        "source_bundle_path",
        "build_receipt_path",
        "source_to_binary_receipt_path",
        "binary_path",
        "checkpoint_path",
        "checkpoint_manifest_path",
    )
    for key in path_keys:
        value = provenance[key]
        if not isinstance(value, str) or not value or not Path(value).is_absolute():
            raise ValueError(f"{key} must be a non-empty absolute path")

    binary = Path(provenance["binary_path"])
    receipt = Path(provenance["source_to_binary_receipt_path"])
    checkpoint_root = Path(provenance["checkpoint_path"])
    checkpoint_manifest = Path(provenance["checkpoint_manifest_path"])
    if Path(plain_argv[0]).resolve() != binary.resolve():
        raise ValueError("argv[0] and provenance binary_path differ")
    if (
        Path(argv_value(plain_argv, "--model-from-path") or "").resolve()
        != checkpoint_root.resolve()
    ):
        raise ValueError("argv checkpoint and provenance checkpoint_path differ")
    if verify_files and (
        not binary.is_file() or sha256_file(binary) != provenance["binary_sha256"]
    ):
        raise ValueError("binary SHA-256 mismatch")
    validate_source_to_binary_receipt(receipt, provenance, verify_files)
    validate_checkpoint_inventory(
        checkpoint_manifest, checkpoint_root, provenance, verify_files
    )
    return provenance


def load_manifest(path: Path, verify_files: bool = True) -> dict[str, Any]:
    manifest_bytes = path.read_bytes()
    value = strict_json_loads(manifest_bytes, f"launch manifest {path}")
    required = {
        "schema",
        "model",
        "common_env_file",
        "common_env_sha256",
        "request_contract",
        "provenance",
        "arms",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("launch manifest keys differ")
    if value["schema"] != MANIFEST_SCHEMA:
        raise ValueError("wrong launch manifest schema")
    if not isinstance(value["model"], str) or not value["model"]:
        raise ValueError("model must be non-empty")
    request_contract = value["request_contract"]
    expected_contract = launch_request_contract()
    if request_contract != expected_contract:
        raise ValueError("request_contract differs from the fixed issue-13 workload")
    common_env_path = Path(value["common_env_file"])
    if not common_env_path.is_absolute():
        common_env_path = (path.parent / common_env_path).resolve()
    if not common_env_path.is_file():
        raise ValueError("common_env_file is not a file")
    if not is_hex(value["common_env_sha256"], 64):
        raise ValueError("common_env_sha256 must be lowercase SHA-256")
    if sha256_file(common_env_path) != value["common_env_sha256"]:
        raise ValueError("common environment SHA-256 mismatch")
    common_env = parse_env_file(common_env_path)
    validate_common_environment(common_env)
    arms = value["arms"]
    if not isinstance(arms, dict) or tuple(arms) != ARM_NAMES:
        raise ValueError("arms must be ordered as plain, k5, then k5_eager_churn")
    for name in ARM_NAMES:
        arm = arms[name]
        if not isinstance(arm, dict) or set(arm) != {
            "argv",
            "cwd",
            "base_url",
            "environment",
        }:
            raise ValueError(f"{name} arm keys differ")
        arm["argv"] = validate_arm_argv(name, arm["argv"])
        if arm["environment"] != ARM_ENVIRONMENTS[name]:
            raise ValueError(f"{name} arm environment differs")
        if not isinstance(arm["cwd"], str) or not Path(arm["cwd"]).is_absolute():
            raise ValueError(f"{name} cwd must be an absolute path")
        if verify_files and not Path(arm["cwd"]).is_dir():
            raise ValueError(f"{name} cwd is not a directory")
        validate_base_url(arm["base_url"], arm["argv"])
        if argv_value(arm["argv"], "--model-name") != value["model"]:
            raise ValueError(f"{name} --model-name differs from request model")
    plain = arms["plain"]
    for name in ("k5", EAGER_CHURN_ARM):
        arm = arms[name]
        if strip_k5_only_argv(arm["argv"]) != plain["argv"]:
            raise ValueError(
                f"plain and {name} argv differ outside native-MTP controls"
            )
        if plain["cwd"] != arm["cwd"] or plain["base_url"] != arm["base_url"]:
            raise ValueError(
                f"plain and {name} must use the same cwd and sequential endpoint"
            )
    value["provenance"] = validate_provenance(
        value["provenance"], plain["argv"], path, verify_files
    )
    value["_common_env"] = common_env
    value["_common_env_path"] = str(common_env_path)
    value["_manifest_source_sha256"] = sha256_bytes(manifest_bytes)
    value["_manifest_canonical_sha256"] = sha256_bytes(
        canonical_bytes(
            {key: item for key, item in value.items() if not key.startswith("_")}
        )
    )
    return value


def public_manifest(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if not key.startswith("_")}


def revalidate_manifest_and_artifacts(
    manifest: dict[str, Any], manifest_path: Path, phase: str
) -> None:
    fresh = load_manifest(manifest_path, verify_files=True)
    if fresh["_manifest_source_sha256"] != manifest["_manifest_source_sha256"]:
        raise ValueError(
            f"{phase}: launch manifest bytes changed after initial snapshot"
        )
    if fresh["_manifest_canonical_sha256"] != manifest["_manifest_canonical_sha256"]:
        raise ValueError(
            f"{phase}: canonical launch manifest changed after initial snapshot"
        )
    if canonical_bytes(public_manifest(fresh)) != canonical_bytes(
        public_manifest(manifest)
    ):
        raise ValueError(
            f"{phase}: parsed launch manifest changed after initial snapshot"
        )
    if fresh["_common_env"] != manifest["_common_env"]:
        raise ValueError(f"{phase}: parsed common environment changed")


class HttpClient:
    def __init__(self, base_url: str, timeout: float):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def request(
        self, method: str, route: str, body: object | None = None
    ) -> tuple[int, bytes]:
        status, raw, _ = self.request_observed(method, route, body)
        return status, raw

    def request_observed(
        self,
        method: str,
        route: str,
        body: object | None = None,
        request_headers: dict[str, str] | None = None,
    ) -> tuple[int, bytes, list[tuple[str, str]]]:
        payload = None if body is None else canonical_bytes(body)
        headers = {"Content-Type": "application/json"}
        if request_headers is not None:
            headers.update(request_headers)
        request = urllib.request.Request(
            self.base_url + route,
            data=payload,
            method=method,
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.status, response.read(), list(response.headers.items())
        except urllib.error.HTTPError as error:
            headers = [] if error.headers is None else list(error.headers.items())
            return error.code, error.read(), headers

    def json(
        self, method: str, route: str, body: object | None = None
    ) -> tuple[int, Any, bytes]:
        status, raw = self.request(method, route, body)
        value = strict_json_loads(raw, f"{route} HTTP {status} response")
        return status, value, raw

    def correlated_json(
        self,
        method: str,
        route: str,
        body: object,
        request_id: str,
    ) -> tuple[int, Any, bytes, list[tuple[str, str]]]:
        status, raw, headers = self.request_observed(
            method,
            route,
            body,
            {"X-Request-Id": request_id},
        )
        value = strict_json_loads(raw, f"{route} HTTP {status} response")
        return status, value, raw, headers


class RequestLogCapture:
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
        if offset < 0 or offset > previous:
            raise ValueError("request log offset moved backwards")
        with self.path.open("rb") as handle:
            handle.seek(offset)
            return handle.read(previous - offset)


def capture_request_log_evidence(
    capture: RequestLogCapture | None,
    offset: int | None,
    prompt_tokens: int,
    require_native_mtp: bool,
    require_qsa_route_telemetry: bool = False,
) -> dict[str, Any] | None:
    if capture is None:
        return None
    if offset is None:
        raise AssertionError("request log capture lacks its starting offset")
    delta = capture.finish(offset)
    evidence = validate_request_log_delta(
        delta.decode("utf-8", errors="replace"),
        prompt_tokens,
        require_native_mtp,
        require_qsa_route_telemetry,
    )
    evidence["log_delta_bytes"] = len(delta)
    evidence["log_delta_sha256"] = sha256_bytes(delta)
    return evidence


def tokenize_components(
    client: HttpClient, case_id: str, prompt_tokens: int
) -> tuple[dict[str, list[int]], dict[str, str]]:
    texts, codes = component_texts(case_id, prompt_tokens)
    result: dict[str, list[int]] = {}
    for label in COMPONENT_KEYS:
        status, value, _ = client.json("POST", "/tokenize", {"prompt": texts[label]})
        if (
            status != 200
            or not isinstance(value, dict)
            or set(value) != {"tokens", "count"}
        ):
            raise ValueError(
                f"/tokenize failed for {prompt_tokens}/{label}: HTTP {status}"
            )
        tokens = validate_token_vector(value["tokens"], label)
        if type(value["count"]) is not int or value["count"] != len(tokens):
            raise ValueError(f"/tokenize count differs for {prompt_tokens}/{label}")
        result[label] = tokens
    return result, codes


def build_request_bundle(client: HttpClient, model: str) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    bodies: list[dict[str, Any]] = []
    components_by_case: dict[str, dict[str, list[int]]] = {}
    for case_id, prompt_tokens, coverage in CASE_SPECS:
        components, codes = tokenize_components(client, case_id, prompt_tokens)
        prompt, positions = assemble_prompt(prompt_tokens, components)
        contract = request_contract(case_id)
        body = completion_body(
            model,
            prompt,
            max_tokens=contract["max_tokens"],
            ignore_eos=contract["ignore_eos"],
            stop=contract["stop"],
        )
        body_sha = sha256_bytes(canonical_bytes(body))
        cases.append(
            {
                "case_id": case_id,
                "prompt_tokens": prompt_tokens,
                "coverage": coverage,
                "requested_tokens": contract["max_tokens"],
                "repetitions": contract["repetitions"],
                "ignore_eos": contract["ignore_eos"],
                "stop": contract["stop"],
                "validation": contract["validation"],
                "prompt_origin": "deterministic-synthetic-known-answer",
                "prompt_u32le_sha256": prompt_sha256(prompt),
                "request_sha256": body_sha,
                "needle_positions": positions,
                "needle_target_depths": {
                    "early": 0.01,
                    "middle": 0.50,
                    "late": 0.98,
                },
                "expected_answer": expected_answer(codes),
            }
        )
        bodies.append(body)
        components_by_case[case_id] = components
    request_manifest = {
        "schema": REQUEST_SCHEMA,
        "model": model,
        "sampling": {
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "top_n_sigma": 0.0,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "seed": SEED,
            "stop": ["\n"],
        },
        "max_tokens": MAX_TOKENS,
        "repetitions": REPETITIONS,
        "endpoint_semantic": launch_request_contract()["endpoint_semantic"],
        "full_endpoint": {
            "case_id": FULL_ENDPOINT_CASE_SPEC[0],
            "prompt_tokens": FULL_ENDPOINT_CASE_SPEC[1],
            "max_tokens": FULL_ENDPOINT_MAX_TOKENS,
            "total_tokens": NATIVE_CONTEXT,
            "repetitions": FULL_ENDPOINT_REPETITIONS,
            "ignore_eos": True,
            "stop": [],
            "return_token_ids": True,
        },
        "admission_boundary": launch_request_contract()["admission_boundary"],
        "reporter_prompt_bytes_available": False,
        "reporter_length_matrix_complete": True,
        "qsa_boundary_matrix_complete": True,
        "cases": cases,
    }
    return {
        "manifest": request_manifest,
        "manifest_sha256": sha256_bytes(canonical_bytes(request_manifest)),
        "bodies": bodies,
        "components": components_by_case,
    }


def validate_tokenizer_parity(client: HttpClient, bundle: dict[str, Any]) -> None:
    for case in bundle["manifest"]["cases"]:
        prompt_tokens = case["prompt_tokens"]
        observed, _ = tokenize_components(client, case["case_id"], prompt_tokens)
        if observed != bundle["components"][case["case_id"]]:
            raise ValueError(f"tokenizer drift between arms at {prompt_tokens} tokens")


def validate_completion_response(
    value: object,
    prompt_tokens: int,
    expected: str,
    request_sha256: str,
    expected_model: str,
    *,
    requested_tokens: int = MAX_TOKENS,
    validation: str = "exact-known-answer",
) -> dict[str, Any]:
    if validation not in {
        "exact-known-answer",
        "native-endpoint-exact-token-identity",
        "native-endpoint-admission",
    }:
        raise ValueError(f"unknown response validation {validation}")
    if not isinstance(value, dict) or "error" in value:
        raise ValueError("completion response is an error or not an object")
    if value.get("model") != expected_model:
        raise ValueError("completion response model differs from the request")
    usage = value.get("usage")
    choices = value.get("choices")
    if (
        not isinstance(usage, dict)
        or not isinstance(choices, list)
        or len(choices) != 1
    ):
        raise ValueError("completion response shape differs")
    observed_prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if (
        type(observed_prompt_tokens) is not int
        or observed_prompt_tokens != prompt_tokens
    ):
        raise ValueError("server prompt token accounting differs from exact request")
    if (
        type(completion_tokens) is not int
        or not 1 <= completion_tokens <= requested_tokens
    ):
        raise ValueError("completion token count is out of range")
    total_tokens = usage.get("total_tokens")
    if (
        type(total_tokens) is not int
        or total_tokens != prompt_tokens + completion_tokens
    ):
        raise ValueError("total token accounting differs")
    details = usage.get("prompt_tokens_details")
    cached_tokens = details.get("cached_tokens") if isinstance(details, dict) else None
    if type(cached_tokens) is not int or cached_tokens != 0:
        raise ValueError(
            "qualification request used or failed to report cached prompt tokens"
        )
    choice = choices[0]
    choice_index = choice.get("index") if isinstance(choice, dict) else None
    if (
        not isinstance(choice, dict)
        or type(choice_index) is not int
        or choice_index != 0
        or not isinstance(choice.get("text"), str)
    ):
        raise ValueError("completion choice differs")
    completion_token_ids = validate_token_vector(choice.get("token_ids"), "completion")
    if len(completion_token_ids) != completion_tokens:
        raise ValueError(
            "scheduler-owned completion token IDs differ from completion_tokens"
        )
    finish_reason = choice.get("finish_reason")
    if validation == "exact-known-answer":
        if finish_reason != "stop":
            raise ValueError(
                f"completion is truncated or abnormal: finish_reason={finish_reason!r}"
            )
        parsed = parse_final_answer(choice["text"])
        final = canonical_answer(parsed)
        correct = final == expected
        if not correct:
            raise ValueError(
                f"known-answer mismatch: expected {expected!r}, got {final!r}"
            )
        if choice["text"].strip() != expected:
            raise ValueError("completion contains text outside the exact final answer")
        reached_native_endpoint = False
        retrieval_oracle = "exact-single-structured-answer"
        endpoint_fill_oracle = "not-applicable"
    elif validation == "native-endpoint-exact-token-identity":
        if (
            prompt_tokens != FULL_ENDPOINT_CASE_SPEC[1]
            or requested_tokens != FULL_ENDPOINT_MAX_TOKENS
            or completion_tokens != FULL_ENDPOINT_MAX_TOKENS
            or usage.get("total_tokens") != NATIVE_CONTEXT
        ):
            raise ValueError(
                "ignore_eos endpoint request did not reach exactly 262144 total tokens"
            )
        if finish_reason != "length":
            raise ValueError(
                "ignore_eos endpoint request must finish by the exact length budget"
            )
        parsed = None
        final = None
        correct = None
        reached_native_endpoint = True
        retrieval_oracle = f"separate-stopped-case:{ENDPOINT_SEMANTIC_CASE_SPEC[0]}"
        endpoint_fill_oracle = "paired-plain-k5-exact-text-and-token-id-identity"
    else:
        if (
            prompt_tokens != ADMISSION_ACCEPT_CASE_SPEC[1]
            or requested_tokens != ADMISSION_PROBE_MAX_TOKENS
            or completion_tokens != ADMISSION_PROBE_MAX_TOKENS
            or usage.get("total_tokens") != NATIVE_CONTEXT
        ):
            raise ValueError(
                "admission probe did not reach exactly 262144 total tokens"
            )
        if finish_reason != "length":
            raise ValueError("admission probe must finish by its one-token budget")
        parsed = None
        final = None
        correct = True
        reached_native_endpoint = True
        retrieval_oracle = "not-applicable-admission-only"
        endpoint_fill_oracle = "one-token-boundary-admission"
    semantic = {
        "request_sha256": request_sha256,
        "model": expected_model,
        "validation": validation,
        "prompt_tokens": prompt_tokens,
        "requested_tokens": requested_tokens,
        "completion_tokens": completion_tokens,
        "completion_token_ids": completion_token_ids,
        "completion_u32le_sha256": token_u32le_sha256(completion_token_ids),
        "total_tokens": prompt_tokens + completion_tokens,
        "finish_reason": finish_reason,
        "text": choice["text"],
        "parsed_answer": parsed,
        "final_answer": final,
        "correct": correct,
        "reached_native_endpoint": reached_native_endpoint,
        "retrieval_oracle": retrieval_oracle,
        "endpoint_fill_oracle": endpoint_fill_oracle,
    }
    semantic["semantic_sha256"] = sha256_bytes(canonical_bytes(semantic))
    return semantic


def execute_requests(
    client: HttpClient,
    bundle: dict[str, Any],
    case_ids: tuple[str, ...] | None = None,
    log_capture: RequestLogCapture | None = None,
    require_native_mtp: bool = False,
    require_qsa_route_telemetry: bool = False,
) -> list[dict[str, Any]]:
    if case_ids is None:
        case_ids = PAIRED_CASE_IDS
    selected = set(case_ids)
    results: list[dict[str, Any]] = []
    for case, body in zip(bundle["manifest"]["cases"], bundle["bodies"], strict=True):
        if selected is not None and case["case_id"] not in selected:
            continue
        observations: list[dict[str, Any]] = []
        request_log_evidence: list[dict[str, Any]] = []
        for repetition in range(case["repetitions"]):
            log_offset = log_capture.begin() if log_capture is not None else None
            status, value, raw = client.json("POST", "/v1/completions", body)
            if status != 200:
                raise ValueError(
                    f"{case['case_id']} repetition {repetition} returned HTTP {status}: {raw[:300]!r}"
                )
            row = validate_completion_response(
                value,
                case["prompt_tokens"],
                case["expected_answer"],
                case["request_sha256"],
                body["model"],
                requested_tokens=case["requested_tokens"],
                validation=case["validation"],
            )
            row["repetition"] = repetition
            observations.append(row)
            evidence = capture_request_log_evidence(
                log_capture,
                log_offset,
                case["prompt_tokens"],
                require_native_mtp,
                require_qsa_route_telemetry,
            )
            if evidence is not None:
                evidence["repetition"] = repetition
                request_log_evidence.append(evidence)
        baseline = {
            key: value for key, value in observations[0].items() if key != "repetition"
        }
        for row in observations[1:]:
            current = {key: value for key, value in row.items() if key != "repetition"}
            if current != baseline:
                raise ValueError(
                    f"{case['case_id']} is nondeterministic within one arm"
                )
        result = {"case_id": case["case_id"], "observations": observations}
        if log_capture is not None:
            if len(request_log_evidence) != len(observations):
                raise AssertionError("request-scoped log evidence count drifted")
            result["request_log_evidence"] = request_log_evidence
        results.append(result)
    if {row["case_id"] for row in results} != selected:
        raise ValueError("requested case subset differs from the canonical bundle")
    return results


def bundle_case(
    bundle: dict[str, Any], case_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    matches = [
        (case, body)
        for case, body in zip(
            bundle["manifest"]["cases"], bundle["bodies"], strict=True
        )
        if case["case_id"] == case_id
    ]
    if len(matches) != 1:
        raise ValueError(f"canonical bundle has {len(matches)} rows for {case_id}")
    return matches[0]


def execute_one_case(
    client: HttpClient,
    case: dict[str, Any],
    body: dict[str, Any],
    phase: str,
    log_capture: RequestLogCapture | None = None,
    require_native_mtp: bool = False,
    require_qsa_route_telemetry: bool = False,
) -> dict[str, Any]:
    log_offset = log_capture.begin() if log_capture is not None else None
    status, value, raw = client.json("POST", "/v1/completions", body)
    if status != 200:
        raise ValueError(
            f"{phase}/{case['case_id']} returned HTTP {status}: {raw[:300]!r}"
        )
    observation = validate_completion_response(
        value,
        case["prompt_tokens"],
        case["expected_answer"],
        case["request_sha256"],
        body["model"],
        requested_tokens=case["requested_tokens"],
        validation=case["validation"],
    )
    result = {
        "phase": phase,
        "case_id": case["case_id"],
        "observation": observation,
    }
    evidence = capture_request_log_evidence(
        log_capture,
        log_offset,
        case["prompt_tokens"],
        require_native_mtp,
        require_qsa_route_telemetry,
    )
    if evidence is not None:
        result["request_log_evidence"] = evidence
    return result


def execute_rollback_slot_reuse(
    client: HttpClient,
    bundle: dict[str, Any],
    log_capture: RequestLogCapture | None = None,
    require_qsa_route_telemetry: bool = False,
) -> dict[str, Any]:
    sequence: list[dict[str, Any]] = []
    for phase, case_id in ROLLBACK_SLOT_REUSE_SEQUENCE:
        case, body = bundle_case(bundle, case_id)
        sequence.append(
            execute_one_case(
                client,
                case,
                body,
                phase,
                log_capture=log_capture,
                require_native_mtp=True,
                require_qsa_route_telemetry=require_qsa_route_telemetry,
            )
        )
    before = sequence[0]["observation"]
    after = sequence[-1]["observation"]
    fields = (
        "request_sha256",
        "model",
        "validation",
        "prompt_tokens",
        "requested_tokens",
        "completion_tokens",
        "completion_token_ids",
        "completion_u32le_sha256",
        "total_tokens",
        "finish_reason",
        "text",
        "parsed_answer",
        "final_answer",
        "correct",
        "reached_native_endpoint",
        "retrieval_oracle",
        "endpoint_fill_oracle",
        "semantic_sha256",
    )
    mismatched = [field for field in fields if before.get(field) != after.get(field)]
    if mismatched:
        raise ValueError(
            f"single-slot sentinel changed after rollback exercise in {mismatched}"
        )
    return {
        "sequence": sequence,
        "single_slot_geometry": {
            "max_batch_size": 1,
            "max_num_seqs": 1,
            "ssm_cache_slots": 0,
            "basis": "each completed request releases the only model-owned slot before the next request",
        },
        "sentinel_exact_match_after_reuse": True,
        "rollback_log_evidence": None,
    }


def validate_context_length_rejection(
    status: int,
    value: object,
    prompt_tokens: int,
    requested_tokens: int,
) -> dict[str, Any]:
    return endpoint_gate.validate_context_length_rejection(
        status, value, prompt_tokens, requested_tokens, NATIVE_CONTEXT
    )


def execute_endpoint_admission_probes(
    client: HttpClient,
    bundle: dict[str, Any],
    log_capture: RequestLogCapture,
    max_prefill_tokens: int = 16_384,
) -> dict[str, Any]:
    if type(max_prefill_tokens) is not int or max_prefill_tokens <= 0:
        raise ValueError("max_prefill_tokens must be an exact positive integer")
    accepted_case, accepted_body = bundle_case(bundle, ADMISSION_ACCEPT_CASE_SPEC[0])
    accepted = execute_one_case(
        client,
        accepted_case,
        accepted_body,
        "native-endpoint-admission",
        log_capture=log_capture,
        require_native_mtp=False,
    )

    rejected_case, rejected_body = bundle_case(bundle, ADMISSION_REJECT_CASE_SPEC[0])
    metrics_before = capture_rejection_metrics(client)
    offset = log_capture.begin()
    http_request_id = f"qualify_overflow_{uuid.uuid4().hex}"
    status, value, raw, response_headers = client.correlated_json(
        "POST", "/v1/completions", rejected_body, http_request_id
    )
    delta = log_capture.finish(offset)
    metrics_after = capture_rejection_metrics(client)
    rejection = validate_context_length_rejection(
        status,
        value,
        rejected_case["prompt_tokens"],
        rejected_case["requested_tokens"],
    )
    rejection["request_sha256"] = rejected_case["request_sha256"]
    rejection["response_sha256"] = sha256_bytes(raw)
    rejection["response_bytes"] = len(raw)
    rejection["http_request_id"] = validate_correlated_request_id(
        response_headers, http_request_id
    )
    rejection["log_delta_sha256"] = sha256_bytes(delta)
    rejection["log_delta_bytes"] = len(delta)
    rejection["scheduler_evidence"] = validate_no_scheduler_dispatch(
        delta.decode("utf-8", errors="replace"),
        metrics_before,
        metrics_after,
        http_request_id,
    )

    health_status, health_raw = client.request("GET", "/health")
    if health_status != 200:
        raise ValueError(
            f"post-rejection health endpoint returned HTTP {health_status}"
        )
    canary_case, canary_body = bundle_case(bundle, ROLLBACK_SENTINEL_CASE_ID)
    completion_canary = execute_one_case(
        client,
        canary_case,
        canary_body,
        "post-rejection-completion-canary",
        log_capture=log_capture,
        require_native_mtp=False,
    )
    metrics_post_canary = capture_rejection_metrics(client)
    canary_observation = completion_canary["observation"]
    post_canary_metrics = endpoint_gate.validate_post_canary_metrics(
        metrics_after,
        metrics_post_canary,
        prompt_tokens=canary_observation["prompt_tokens"],
        prefill_chunks=(canary_observation["prompt_tokens"] + max_prefill_tokens - 1)
        // max_prefill_tokens,
    )
    return {
        "accepted_262143_plus_1": accepted,
        "rejected_262144_plus_1": rejection,
        "post_rejection_health": {
            "http_status": health_status,
            "body_sha256": sha256_bytes(health_raw),
            "body_bytes": len(health_raw),
        },
        "post_rejection_completion_canary": completion_canary,
        "post_rejection_completion_canary_metrics": post_canary_metrics,
    }


def compare_arms(
    plain: list[dict[str, Any]],
    k5: list[dict[str, Any]],
    expected_case_ids: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    if expected_case_ids is None:
        expected_case_ids = PAIRED_CASE_IDS
    if len(plain) != len(k5) or len(plain) != len(expected_case_ids):
        raise ValueError("arm result counts differ")
    compared = 0
    semantic_correct = 0
    identity_only = 0
    endpoint_matches = 0
    for expected_case_id, plain_case, k5_case in zip(
        expected_case_ids, plain, k5, strict=True
    ):
        if (
            plain_case["case_id"] != expected_case_id
            or k5_case["case_id"] != expected_case_id
        ):
            raise ValueError("arm case ordering differs")
        plain_rows = plain_case["observations"]
        k5_rows = k5_case["observations"]
        expected_repetitions = request_contract(expected_case_id)["repetitions"]
        if (
            len(plain_rows) != expected_repetitions
            or len(k5_rows) != expected_repetitions
        ):
            raise ValueError("arm repetition count differs")
        for plain_row, k5_row in zip(plain_rows, k5_rows, strict=True):
            fields = (
                "request_sha256",
                "model",
                "validation",
                "prompt_tokens",
                "requested_tokens",
                "completion_tokens",
                "completion_token_ids",
                "completion_u32le_sha256",
                "total_tokens",
                "finish_reason",
                "text",
                "parsed_answer",
                "final_answer",
                "correct",
                "reached_native_endpoint",
                "retrieval_oracle",
                "endpoint_fill_oracle",
                "semantic_sha256",
            )
            mismatched = [
                field for field in fields if plain_row.get(field) != k5_row.get(field)
            ]
            if mismatched:
                raise ValueError(
                    f"{plain_case['case_id']} repetition {plain_row['repetition']} "
                    f"plain/K5 mismatch in {mismatched}"
                )
            validation = request_contract(expected_case_id)["validation"]
            if (
                plain_row.get("validation") != validation
                or k5_row.get("validation") != validation
            ):
                raise ValueError(
                    f"{plain_case['case_id']} validation differs from its case contract"
                )
            if validation == "exact-known-answer":
                if (
                    plain_row.get("correct") is not True
                    or k5_row.get("correct") is not True
                ):
                    raise ValueError(
                        f"{plain_case['case_id']} is not semantically correct in both arms"
                    )
                semantic_correct += 1
            elif validation == "native-endpoint-exact-token-identity":
                required_none_fields = ("correct", "parsed_answer", "final_answer")
                if (
                    any(field not in plain_row for field in required_none_fields)
                    or any(field not in k5_row for field in required_none_fields)
                    or plain_row.get("correct") is not None
                    or k5_row.get("correct") is not None
                    or plain_row.get("parsed_answer") is not None
                    or k5_row.get("parsed_answer") is not None
                    or plain_row.get("final_answer") is not None
                    or k5_row.get("final_answer") is not None
                    or plain_row.get("reached_native_endpoint") is not True
                    or k5_row.get("reached_native_endpoint") is not True
                    or plain_row.get("prompt_tokens") != FULL_ENDPOINT_CASE_SPEC[1]
                    or k5_row.get("prompt_tokens") != FULL_ENDPOINT_CASE_SPEC[1]
                    or plain_row.get("requested_tokens") != FULL_ENDPOINT_MAX_TOKENS
                    or k5_row.get("requested_tokens") != FULL_ENDPOINT_MAX_TOKENS
                    or plain_row.get("completion_tokens") != FULL_ENDPOINT_MAX_TOKENS
                    or k5_row.get("completion_tokens") != FULL_ENDPOINT_MAX_TOKENS
                    or plain_row.get("total_tokens") != NATIVE_CONTEXT
                    or k5_row.get("total_tokens") != NATIVE_CONTEXT
                    or plain_row.get("finish_reason") != "length"
                    or k5_row.get("finish_reason") != "length"
                ):
                    raise ValueError(
                        f"{plain_case['case_id']} is not a pure endpoint-fill identity row"
                    )
                identity_only += 1
            else:
                raise ValueError(
                    f"paired case {plain_case['case_id']} has unsupported validation {validation}"
                )
            if plain_row["reached_native_endpoint"]:
                endpoint_matches += 1
            compared += 1
    return {
        "cases": len(expected_case_ids),
        "requests_per_arm": compared,
        "correct_per_arm": semantic_correct,
        "semantic_correct_per_arm": semantic_correct,
        "identity_only_per_arm": identity_only,
        "plain_k5_exact_matches": compared,
        "plain_k5_exact_token_id_matches": compared,
        "native_262144_endpoint_exact_token_id_matches": endpoint_matches,
        "errors": 0,
        "truncations": 0,
    }


def configure_k5_draft_policy(policy: str) -> None:
    """Select the native-K5 draft policy for every later log parse."""
    global K5_DRAFT_POLICY
    if policy not in (K5_FIXED_POLICY, K5_GATED_POLICY):
        raise ValueError(f"unknown native K5 draft policy {policy!r}")
    K5_DRAFT_POLICY = policy


def k5_gated() -> bool:
    return K5_DRAFT_POLICY == K5_GATED_POLICY


def parse_mtp_accept_frames(logs: str) -> list[dict[str, int | str]]:
    frames: list[dict[str, int | str]] = []
    for line in logs.splitlines():
        if K5_VERIFY_MARKER not in line:
            continue
        fields: dict[str, str] = {}
        for key in ("proposer", "seq_len", "num_drafts", "accepted", "pattern"):
            match = re.search(rf"(?:^|\s){key}=(\"[^\"]*\"|\S+)", line)
            if match is None:
                raise ValueError(f"MTP_ACCEPT frame lacks {key}: {line[:300]!r}")
            fields[key] = match.group(1).strip('"')
        try:
            numeric = {
                key: int(fields[key]) for key in ("seq_len", "num_drafts", "accepted")
            }
        except ValueError as error:
            raise ValueError(
                f"MTP_ACCEPT frame has non-decimal fields: {line[:300]!r}"
            ) from error
        # num_drafts is the requested width; the pattern has one letter per
        # draft actually verified, fewer than requested only under the gate.
        drafted = len(fields["pattern"])
        if (
            fields["proposer"] != "mtp"
            or numeric["num_drafts"] != K5_NUM_DRAFTS
            or not 1 <= drafted <= K5_NUM_DRAFTS
            or (drafted != K5_NUM_DRAFTS and not k5_gated())
            or not 0 <= numeric["accepted"] <= drafted
            or set(fields["pattern"]) - {"Y", "N"}
        ):
            raise ValueError(
                f"MTP_ACCEPT frame violates native K5 shape: {line[:300]!r}"
            )
        frames.append(
            {**numeric, "proposer": fields["proposer"], "pattern": fields["pattern"]}
        )
    return frames


SPEC_CYCLE_TIMING_FIELDS = (
    "secondary_wait_enqueue_host_ns",
    "setup_host_ns",
    "verify_complete_host_ns",
    "accept_host_ns",
    "commit_enqueue_host_ns",
    "post_commit_enqueue_host_ns",
    "proposer_state_host_ns",
    "propose_complete_host_ns",
    "finalize_host_ns",
)


def parse_completed_spec_cycles(logs: str) -> list[dict[str, int | str]]:
    records: list[dict[str, int | str]] = []
    numeric_fields = (
        "schema",
        "pre",
        "k",
        "gamma",
        "accepted",
        "emitted",
        "c",
        "async_requested",
        "async_engaged",
        *SPEC_CYCLE_TIMING_FIELDS,
        "total_host_ns",
        "syncs_added",
    )
    for line_number, line in enumerate(logs.splitlines(), 1):
        if SPEC_CYCLE_MARKER not in line:
            continue
        fields: dict[str, str] = {}
        for key in ("req", *numeric_fields):
            match = re.search(rf"(?:^|\s){key}=(\S+)", line)
            if match is None:
                raise ValueError(
                    f"completed SPEC_CYCLE_V2 record lacks {key}: {line[:300]!r}"
                )
            fields[key] = match.group(1)
        if not re.fullmatch(r"[0-9a-f]{16}", fields["req"]):
            raise ValueError(
                f"completed SPEC_CYCLE_V2 record has invalid req: {line[:300]!r}"
            )
        try:
            numeric = {key: int(fields[key]) for key in numeric_fields}
        except ValueError as error:
            raise ValueError(
                f"completed SPEC_CYCLE_V2 record has non-decimal fields: {line[:300]!r}"
            ) from error
        if any(value < 0 for value in numeric.values()):
            raise ValueError(
                f"completed SPEC_CYCLE_V2 record has negative fields: {line[:300]!r}"
            )
        if (
            numeric["schema"] != 2
            or numeric["k"] != numeric["gamma"] + 1
            or not 1 <= numeric["gamma"] <= K5_NUM_DRAFTS
            or (numeric["gamma"] != K5_NUM_DRAFTS and not k5_gated())
            or not 0 <= numeric["accepted"] <= numeric["gamma"]
            or numeric["emitted"] != numeric["accepted"] + 1
            or numeric["c"] != 1
            or numeric["async_requested"] != 0
            or numeric["async_engaged"] != 0
            or numeric["syncs_added"] != 0
            or numeric["total_host_ns"] == 0
            or numeric["total_host_ns"]
            != sum(numeric[key] for key in SPEC_CYCLE_TIMING_FIELDS)
        ):
            raise ValueError(
                f"completed SPEC_CYCLE_V2 record violates frozen K5 C=1 shape: {line[:300]!r}"
            )
        records.append(
            {
                **numeric,
                "req": fields["req"],
                "line_number": line_number,
            }
        )
    return records


def validate_rollback_slot_log_sequence(
    records: list[dict[str, int | str]],
) -> dict[str, Any]:
    sentinel_starts: list[dict[str, int | str]] = []
    seen: set[str] = set()
    sentinel_first_spec_position = (
        next(spec[1] for spec in CASE_SPECS if spec[0] == ROLLBACK_SENTINEL_CASE_ID) + 1
    )
    trigger_first_spec_position = (
        next(spec[1] for spec in CASE_SPECS if spec[0] == ROLLBACK_TRIGGER_CASE_ID) + 1
    )
    for record in records:
        req = str(record["req"])
        if record["pre"] == sentinel_first_spec_position and req not in seen:
            seen.add(req)
            sentinel_starts.append(record)
    if len(sentinel_starts) < 2:
        raise ValueError(
            "K5 churn log lacks two distinct completed 2017-token canary requests"
        )
    before, after = sentinel_starts[-2:]
    between = [
        record
        for record in records
        if int(before["line_number"])
        < int(record["line_number"])
        < int(after["line_number"])
    ]
    trigger_ids = []
    for record in between:
        req = str(record["req"])
        if (
            record["pre"] == trigger_first_spec_position
            and req not in (before["req"], after["req"])
            and req not in trigger_ids
        ):
            trigger_ids.append(req)
    if not trigger_ids:
        raise ValueError(
            "K5 churn log lacks a completed 2050-token trigger request between canaries"
        )
    partial = [
        record
        for record in between
        if record["req"] in trigger_ids and record["accepted"] < record["gamma"]
    ]
    if not partial:
        raise ValueError(
            "K5 churn trigger has no completed partial-accept rollback between canaries"
        )
    return {
        "sentinel_before_req": before["req"],
        "trigger_req": trigger_ids[0],
        "sentinel_after_req": after["req"],
        "distinct_request_ids": len(
            {str(before["req"]), *trigger_ids, str(after["req"])}
        ),
        "completed_partial_accept_rollbacks": len(partial),
        "first_completed_partial_accept_rollback": partial[0],
    }


QSA_ROUTE_COUNT_FIELDS = (
    "dense_paged_rows",
    "sparse_qsa_rows",
    "high_speed_swap_rows",
    "mla_rows",
)
QSA_ROUTE_TOTAL_FIELDS = (
    "row_serial_frame_total",
    "dense_paged_row_total",
    "sparse_qsa_row_total",
    "high_speed_swap_row_total",
    "mla_row_total",
    "sparse_rows_legacy_frame_total",
    "sparse_rows_v2_frame_total",
)
QSA_ROUTE_LABELS = frozenset({"row-serial", "sparse-rows-legacy", "sparse-rows-v2"})


def _one_log_field(line: str, key: str) -> str:
    matches = re.findall(rf"(?:^|\s){re.escape(key)}=(\S+)", line)
    if len(matches) != 1:
        raise ValueError(
            f"QSA route receipt must contain exactly one {key}: {line[:300]!r}"
        )
    return matches[0]


def parse_qsa_route_receipts(logs: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    integer_fields = (
        "frame_id",
        "rows",
        "seq_len",
        "layer",
        *QSA_ROUTE_COUNT_FIELDS,
        *QSA_ROUTE_TOTAL_FIELDS,
    )
    for line_number, line in enumerate(logs.splitlines(), 1):
        if QSA_ROUTE_RECEIPT_MARKER not in line:
            continue
        if line.count(QSA_ROUTE_RECEIPT_MARKER) != 1:
            raise ValueError("QSA route receipt marker is duplicated on one line")
        route = _one_log_field(line, "route")
        if route not in QSA_ROUTE_LABELS:
            raise ValueError(f"QSA route receipt has unknown route {route!r}")
        raw_fields = {key: _one_log_field(line, key) for key in integer_fields}
        if any(
            not value.isascii() or not value.isdecimal()
            for value in raw_fields.values()
        ):
            raise ValueError(
                f"QSA route receipt has a non-decimal field: {line[:300]!r}"
            )
        numeric = {key: int(value) for key, value in raw_fields.items()}
        if numeric["frame_id"] <= 0 or numeric["seq_len"] <= 0:
            raise ValueError("QSA route receipt frame_id and seq_len must be positive")
        if not (
            numeric["rows"] == K5_NUM_DRAFTS + 1
            or (k5_gated() and 2 <= numeric["rows"] <= K5_NUM_DRAFTS + 1)
        ):
            raise ValueError("QSA route receipt is not an exact native K=5 frame")
        if numeric["layer"] not in QSA_FULL_ATTENTION_LAYER_IDS:
            raise ValueError(
                "QSA route receipt layer is outside the exact 12-layer set"
            )
        row_total = sum(numeric[key] for key in QSA_ROUTE_COUNT_FIELDS)
        if route == "row-serial":
            if row_total != numeric["rows"]:
                raise ValueError(
                    "row-serial QSA receipt does not account for every row"
                )
        elif row_total != 0:
            raise ValueError("sparse multi-row QSA receipt contains row-serial counts")
        records.append({"route": route, **numeric, "line_number": line_number})
    return records


def validate_qsa_route_receipts(
    logs: str,
    mtp_frames: list[dict[str, int | str]],
    *,
    require_zero_baseline: bool = False,
) -> dict[str, Any]:
    records = parse_qsa_route_receipts(logs)
    if not records:
        raise ValueError("request-scoped eager K5 log lacks QSA route receipts")

    groups: list[list[dict[str, Any]]] = []
    for record in records:
        if not groups or record["frame_id"] != groups[-1][-1]["frame_id"]:
            groups.append([])
        groups[-1].append(record)

    frame_ids = [int(group[0]["frame_id"]) for group in groups]
    if frame_ids != list(range(frame_ids[0], frame_ids[0] + len(frame_ids))):
        raise ValueError(
            "request-scoped QSA route frame IDs are not strictly contiguous"
        )

    previous: dict[str, Any] | None = None
    sealed_frames: list[dict[str, Any]] = []
    for group in groups:
        layers = tuple(int(record["layer"]) for record in group)
        if layers != QSA_FULL_ATTENTION_LAYER_IDS:
            raise ValueError(
                "QSA route frame lacks the exact ordered 12-layer receipt set"
            )
        rows = {int(record["rows"]) for record in group}
        seq_lens = {int(record["seq_len"]) for record in group}
        if len(rows) != 1 or len(seq_lens) != 1:
            raise ValueError("QSA route frame has inconsistent rows or sequence length")
        for record in group:
            contribution = {
                "row_serial_frame_total": int(record["route"] == "row-serial"),
                "dense_paged_row_total": int(record["dense_paged_rows"]),
                "sparse_qsa_row_total": int(record["sparse_qsa_rows"]),
                "high_speed_swap_row_total": int(record["high_speed_swap_rows"]),
                "mla_row_total": int(record["mla_rows"]),
                "sparse_rows_legacy_frame_total": int(
                    record["route"] == "sparse-rows-legacy"
                ),
                "sparse_rows_v2_frame_total": int(record["route"] == "sparse-rows-v2"),
            }
            if previous is None and require_zero_baseline:
                for key in QSA_ROUTE_TOTAL_FIELDS:
                    if int(record[key]) != contribution[key]:
                        raise ValueError(
                            f"QSA route cumulative total {key} lacks a zero baseline"
                        )
            elif previous is not None:
                for key in QSA_ROUTE_TOTAL_FIELDS:
                    if int(record[key]) != int(previous[key]) + contribution[key]:
                        raise ValueError(
                            f"QSA route cumulative total {key} is not monotone-exact"
                        )
            previous = record
        route_counts = {
            route: sum(record["route"] == route for record in group)
            for route in sorted(QSA_ROUTE_LABELS)
        }
        sealed_frames.append(
            {
                "frame_id": int(group[0]["frame_id"]),
                "rows": next(iter(rows)),
                "seq_len": next(iter(seq_lens)),
                "layer_ids": list(layers),
                "route_counts": route_counts,
                "receipts": len(group),
            }
        )

    mtp_seq_lens = [int(frame["seq_len"]) for frame in mtp_frames]
    qsa_seq_lens = [frame["seq_len"] for frame in sealed_frames]
    qsa_rows = [frame["rows"] for frame in sealed_frames]
    mtp_rows = [len(str(frame["pattern"])) + 1 for frame in mtp_frames]
    if qsa_seq_lens != mtp_seq_lens or qsa_rows != mtp_rows:
        raise ValueError(
            "QSA route frames do not match the request's MTP frame geometry"
        )
    return {
        "frames": len(sealed_frames),
        "receipts": len(records),
        "first_frame_id": frame_ids[0],
        "last_frame_id": frame_ids[-1],
        "exact_layer_ids": list(QSA_FULL_ATTENTION_LAYER_IDS),
        "frame_receipts": sealed_frames,
    }


def validate_request_log_delta(
    logs: str,
    prompt_tokens: int,
    require_native_mtp: bool,
    require_qsa_route_telemetry: bool = False,
) -> dict[str, Any]:
    dispatch_markers = (
        f"Chunked prefill start: {prompt_tokens} prompt tokens,",
        f"Prefilling: {prompt_tokens} prompt tokens,",
    )
    dispatch_count = sum(logs.count(marker) for marker in dispatch_markers)
    if dispatch_count != 1:
        raise ValueError(
            f"request-scoped log has {dispatch_count} scheduler dispatches "
            f"for {prompt_tokens} prompt tokens"
        )
    frames = parse_mtp_accept_frames(logs)
    records = parse_completed_spec_cycles(logs)
    if require_qsa_route_telemetry and not require_native_mtp:
        raise ValueError("QSA route telemetry requires native MTP")
    if require_native_mtp:
        if "control=loaded-serial" in logs:
            raise ValueError("request-scoped K5 log shows target-only demotion")
        if not frames:
            raise ValueError("request-scoped K5 log lacks an MTP verify frame")
        full_frames = [
            frame for frame in frames if len(str(frame["pattern"])) == K5_NUM_DRAFTS
        ]
        # Fixed K5 verifies every chain on the instrumented generic route; the
        # gate's short chains (K2..K4 steps) emit no SPEC_CYCLE_V2 record.
        if not records and (full_frames or not k5_gated()):
            raise ValueError(
                "request-scoped K5 log lacks a completed SPEC_CYCLE_V2 record"
            )
        first_spec_position = prompt_tokens + 1
        matching_frames = [
            frame for frame in frames if frame["seq_len"] == first_spec_position
        ]
        if not matching_frames:
            raise ValueError(
                "request-scoped K5 log lacks an MTP verify frame at the exact "
                "first-speculative geometry"
            )
        matching_records = [
            record for record in records if record["pre"] == first_spec_position
        ]
        first_frame_full = len(str(matching_frames[0]["pattern"])) == K5_NUM_DRAFTS
        if not matching_records and (first_frame_full or not k5_gated()):
            raise ValueError(
                "request-scoped K5 log lacks a completed spec cycle at the "
                "exact first-speculative geometry"
            )
        request_ids = sorted({str(record["req"]) for record in records})
        if len(request_ids) != (1 if records else 0):
            raise ValueError(
                "request-scoped K5 log does not bind exactly one runtime request ID"
            )
    else:
        if frames or records:
            raise ValueError("target-only request unexpectedly used native MTP")
        request_ids = []
        matching_frames = []
        matching_records = []
    qsa_route_evidence = None
    if require_qsa_route_telemetry:
        qsa_route_evidence = validate_qsa_route_receipts(logs, frames)
    elif QSA_ROUTE_RECEIPT_MARKER in logs:
        raise ValueError(
            "QSA route receipts escaped the forced-eager qualification arm"
        )
    return {
        "prompt_tokens": prompt_tokens,
        "scheduler_dispatches": dispatch_count,
        "native_mtp_required": require_native_mtp,
        "native_mtp_frames": len(frames),
        "completed_spec_cycles": len(records),
        "matching_geometry_native_mtp_frames": len(matching_frames),
        "matching_geometry_completed_spec_cycles": len(matching_records),
        "runtime_request_ids": request_ids,
        "first_native_mtp_frame": frames[0] if frames else None,
        "first_completed_spec_cycle": records[0] if records else None,
        "verify_graph_capture_markers": logs.count(K5_VERIFY_GRAPH_CAPTURE_MARKER),
        "qsa_route_telemetry_required": require_qsa_route_telemetry,
        "qsa_route_telemetry": qsa_route_evidence,
    }


def parse_prometheus_integer_metrics(
    payload: bytes, names: tuple[str, ...] = REJECTION_METRIC_NAMES
) -> dict[str, int]:
    return endpoint_gate.parse_prometheus_integer_metrics(payload, names)


def capture_rejection_metrics(client: HttpClient) -> dict[str, Any]:
    return endpoint_gate.capture_metrics(client)


def validate_correlated_request_id(
    headers: list[tuple[str, str]], expected: str
) -> str:
    return endpoint_gate.validate_correlated_request_id(headers, expected)


def validate_no_scheduler_dispatch(
    logs: str,
    metrics_before: dict[str, Any],
    metrics_after: dict[str, Any],
    http_request_id: str,
) -> dict[str, Any]:
    return endpoint_gate.validate_no_scheduler_dispatch(
        logs, metrics_before, metrics_after, http_request_id
    )


def validate_log_contract(name: str, logs: str) -> dict[str, Any]:
    bad = [marker for marker in FORBIDDEN_LOG_MARKERS if marker in logs]
    if bad:
        raise ValueError(f"{name} server log contains fatal markers: {bad!r}")
    if name == "plain":
        if PLAIN_SCHEDULER_MARKER not in logs:
            raise ValueError("plain log lacks target-only scheduler receipt")
        if K5_VERIFY_MARKER in logs:
            raise ValueError("plain log unexpectedly contains an MTP verify frame")
        if SPEC_CYCLE_MARKER in logs:
            raise ValueError("plain log unexpectedly contains a completed spec cycle")
        if (
            QSA_ROUTE_TELEMETRY_ENGAGED_MARKER in logs
            or QSA_ROUTE_RECEIPT_MARKER in logs
        ):
            raise ValueError("plain log unexpectedly enabled QSA route telemetry")
        return {
            "native_mtp_frames": 0,
            "precommit_partial_accept_frames": 0,
            "completed_spec_cycles": 0,
            "completed_partial_accept_rollbacks": 0,
            "verify_graph_capture_markers": logs.count(K5_VERIFY_GRAPH_CAPTURE_MARKER),
        }
    elif name in ("k5", EAGER_CHURN_ARM):
        for marker in (
            K5_SCHEDULER_MARKER,
            K5_CONSTRUCTION_MARKER,
            K5_QSA_VERIFY_ROWS_MARKER,
            K5_VERIFY_MARKER,
        ):
            if marker not in logs:
                raise ValueError(f"k5 log lacks engagement marker: {marker}")
        if "control=loaded-serial" in logs:
            raise ValueError("k5 log shows native MTP demoted to serial target-only")
        frames = parse_mtp_accept_frames(logs)
        if not frames:
            raise ValueError("k5 log contains no parseable native MTP frames")
        precommit_partial = [
            frame for frame in frames if frame["accepted"] < len(str(frame["pattern"]))
        ]
        records = parse_completed_spec_cycles(logs)
        if not records and (
            not k5_gated()
            or any(len(str(frame["pattern"])) == K5_NUM_DRAFTS for frame in frames)
        ):
            raise ValueError("k5 log contains no completed SPEC_CYCLE_V2 records")
        completed_partial = [
            record for record in records if record["accepted"] < record["gamma"]
        ]
        captures = logs.count(K5_VERIFY_GRAPH_CAPTURE_MARKER)
        churn_sequence = None
        qsa_route_evidence = None
        if name == EAGER_CHURN_ARM:
            if captures:
                raise ValueError(
                    "graph-disabled K5 churn arm captured a verify CUDA graph"
                )
            churn_sequence = validate_rollback_slot_log_sequence(records)
            engaged = logs.count(QSA_ROUTE_TELEMETRY_ENGAGED_MARKER)
            if engaged != 1:
                raise ValueError(
                    "graph-disabled K5 churn arm must engage QSA route telemetry exactly once"
                )
            qsa_route_evidence = validate_qsa_route_receipts(
                logs, frames, require_zero_baseline=True
            )
        elif captures < 1:
            raise ValueError(
                "primary K5 log lacks a successful verify CUDA-graph capture/use marker"
            )
        elif (
            QSA_ROUTE_TELEMETRY_ENGAGED_MARKER in logs
            or QSA_ROUTE_RECEIPT_MARKER in logs
        ):
            raise ValueError(
                "primary graph-on K5 arm unexpectedly enabled QSA telemetry"
            )
        return {
            "native_mtp_frames": len(frames),
            "precommit_partial_accept_frames": len(precommit_partial),
            "completed_spec_cycles": len(records),
            "completed_partial_accept_rollbacks": len(completed_partial),
            "first_completed_partial_accept_rollback": (
                completed_partial[0] if completed_partial else None
            ),
            "verify_graph_capture_markers": captures,
            "verify_graph_route_observed": name == "k5" and captures > 0,
            "rollback_slot_log_sequence": churn_sequence,
            "qsa_route_telemetry": qsa_route_evidence,
        }
    else:
        raise ValueError(f"unknown arm {name}")


def require_unoccupied_endpoint(host: str, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.25)
        result = probe.connect_ex((host, port))
    if result == 0:
        raise RuntimeError("qualification endpoint is occupied before child launch")
    if result != errno.ECONNREFUSED:
        raise OSError(result, "could not prove qualification endpoint is unused")


def stop_fresh_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def run_arm(
    name: str,
    arm: dict[str, Any],
    common_env: dict[str, str],
    receipt_path: Path,
    timeout: float,
    bundle: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    log_path = receipt_path.with_suffix(receipt_path.suffix + f".{name}.server.log")
    if log_path.exists():
        raise FileExistsError(f"{name} server log already exists")
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("ATLAS_", "FLUX_"))
    }
    environment.update(common_env)
    environment.update(arm["environment"])
    host, port = validate_base_url(arm["base_url"], arm["argv"])
    require_unoccupied_endpoint(host, port)
    process: subprocess.Popen[bytes] | None = None
    started = time.time()
    log_handle = log_path.open("xb")
    try:
        process = subprocess.Popen(
            arm["argv"],
            cwd=arm["cwd"],
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        client = HttpClient(arm["base_url"], timeout)
        deadline = time.monotonic() + timeout
        while True:
            if process.poll() is not None:
                raise RuntimeError(
                    f"{name} fresh server exited rc={process.returncode}"
                )
            try:
                status, _ = client.request("GET", "/health")
                if status == 200:
                    break
            except (OSError, urllib.error.URLError):
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError(f"{name} fresh server did not become healthy")
            time.sleep(1.0)
        if bundle is None:
            bundle = build_request_bundle(
                client, argv_value(arm["argv"], "--model-name") or ""
            )
        else:
            validate_tokenizer_parity(client, bundle)
        log_capture = RequestLogCapture(log_path)
        rollback_slot_reuse = None
        endpoint_admission = None
        if name == EAGER_CHURN_ARM:
            results = execute_requests(
                client,
                bundle,
                GRAPH_DISABLED_CASE_IDS,
                log_capture=log_capture,
                require_native_mtp=True,
                require_qsa_route_telemetry=True,
            )
            rollback_slot_reuse = execute_rollback_slot_reuse(
                client,
                bundle,
                log_capture,
                require_qsa_route_telemetry=True,
            )
        else:
            results = execute_requests(
                client,
                bundle,
                log_capture=log_capture,
                require_native_mtp=name == "k5",
            )
            if name == "plain":
                max_prefill_raw = argv_value(arm["argv"], "--max-prefill-tokens")
                if (
                    max_prefill_raw is None
                    or not max_prefill_raw.isascii()
                    or not max_prefill_raw.isdecimal()
                    or int(max_prefill_raw) <= 0
                ):
                    raise ValueError("plain --max-prefill-tokens must be positive")
                endpoint_admission = execute_endpoint_admission_probes(
                    client, bundle, log_capture, int(max_prefill_raw)
                )
        if process.poll() is not None:
            raise RuntimeError(
                f"{name} server exited before receipt sealing rc={process.returncode}"
            )
        stop_fresh_process(process)
        log_handle.flush()
        os.fsync(log_handle.fileno())
        log_handle.close()
        logs = log_path.read_text(encoding="utf-8", errors="replace")
        log_evidence = validate_log_contract(name, logs)
        if endpoint_admission is not None:
            expected_dispatches: dict[int, int] = {}
            for result in results:
                observations = result["observations"]
                prompt_tokens = observations[0]["prompt_tokens"]
                expected_dispatches[prompt_tokens] = expected_dispatches.get(
                    prompt_tokens, 0
                ) + len(observations)
            for admission_row in (
                endpoint_admission["accepted_262143_plus_1"],
                endpoint_admission["post_rejection_completion_canary"],
            ):
                observation = admission_row["observation"]
                prompt_tokens = observation["prompt_tokens"]
                expected_dispatches[prompt_tokens] = (
                    expected_dispatches.get(prompt_tokens, 0) + 1
                )
            endpoint_admission["final_dispatch_census"] = (
                endpoint_gate.validate_final_dispatch_census(
                    logs,
                    expected_dispatches,
                    rejected_prompt_tokens=ADMISSION_REJECT_CASE_SPEC[1],
                )
            )
        if rollback_slot_reuse is not None:
            rollback_slot_reuse["rollback_log_evidence"] = {
                "completed_partial_accept_rollbacks": log_evidence[
                    "completed_partial_accept_rollbacks"
                ],
                "first_completed_partial_accept_rollback": log_evidence[
                    "first_completed_partial_accept_rollback"
                ],
                "request_sequence": log_evidence["rollback_slot_log_sequence"],
            }
        arm_receipt = {
            "fresh_process": True,
            "process_pid": process.pid,
            "started_unix": started,
            "elapsed_seconds": time.time() - started,
            "argv_sha256": sha256_bytes(canonical_bytes(arm["argv"])),
            "environment_sha256": sha256_bytes(
                canonical_bytes({**common_env, **arm["environment"]})
            ),
            "server_log_sha256": sha256_file(log_path),
            "log_evidence": log_evidence,
            "results": results,
        }
        if rollback_slot_reuse is not None:
            arm_receipt["rollback_slot_reuse"] = rollback_slot_reuse
        if endpoint_admission is not None:
            arm_receipt["endpoint_admission"] = endpoint_admission
        return arm_receipt, bundle
    finally:
        if process is not None:
            stop_fresh_process(process)
        log_handle.close()


def run_live(
    manifest: dict[str, Any], manifest_path: Path, receipt_path: Path, timeout: float
) -> dict[str, Any]:
    if receipt_path.exists():
        raise FileExistsError("receipt already exists")
    revalidate_manifest_and_artifacts(manifest, manifest_path, "before plain arm")
    plain_receipt, bundle = run_arm(
        "plain",
        manifest["arms"]["plain"],
        manifest["_common_env"],
        receipt_path,
        timeout,
        None,
    )
    revalidate_manifest_and_artifacts(manifest, manifest_path, "before k5 arm")
    k5_receipt, returned_bundle = run_arm(
        "k5",
        manifest["arms"]["k5"],
        manifest["_common_env"],
        receipt_path,
        timeout,
        bundle,
    )
    if returned_bundle is not bundle:
        raise AssertionError("request bundle identity drifted between arms")
    revalidate_manifest_and_artifacts(
        manifest, manifest_path, "before forced-eager arm"
    )
    eager_receipt, eager_bundle = run_arm(
        EAGER_CHURN_ARM,
        manifest["arms"][EAGER_CHURN_ARM],
        manifest["_common_env"],
        receipt_path,
        timeout,
        bundle,
    )
    if eager_bundle is not bundle:
        raise AssertionError("request bundle identity drifted in eager K5 arm")
    comparison = compare_arms(plain_receipt["results"], k5_receipt["results"])
    plain_by_id = {row["case_id"]: row for row in plain_receipt["results"]}
    k5_by_id = {row["case_id"]: row for row in k5_receipt["results"]}
    graph_disabled_plain = [plain_by_id[case_id] for case_id in GRAPH_DISABLED_CASE_IDS]
    graph_disabled_k5 = [k5_by_id[case_id] for case_id in GRAPH_DISABLED_CASE_IDS]
    plain_vs_forced_eager = compare_arms(
        graph_disabled_plain,
        eager_receipt["results"],
        GRAPH_DISABLED_CASE_IDS,
    )
    normal_k5_vs_forced_eager = compare_arms(
        graph_disabled_k5,
        eager_receipt["results"],
        GRAPH_DISABLED_CASE_IDS,
    )
    revalidate_manifest_and_artifacts(manifest, manifest_path, "before final seal")
    receipt = {
        "schema": SCHEMA,
        "status": "PASS",
        "execution_scope": "direct-binary-not-container",
        "run_nonce": uuid.uuid4().hex,
        "manifest_sha256": manifest["_manifest_source_sha256"],
        "manifest_canonical_sha256": manifest["_manifest_canonical_sha256"],
        "request_manifest": bundle["manifest"],
        "request_manifest_sha256": bundle["manifest_sha256"],
        "provenance": manifest["provenance"],
        "runtime_contract": {
            "concurrency": 1,
            "world_size": 1,
            "tensor_parallel_size": 1,
            "expert_parallel_size": 1,
            "high_speed_swap": False,
            "kv_cache_dtype": "bf16",
            "qwen4_qsa": True,
            "prefix_caching": False,
            "plain_spec_mode": "none",
            "k5_spec_mode": "native-mtp",
            "k5_num_drafts": 4,
            "k5_verify_rows": 5,
            "native_context_total_tokens": NATIVE_CONTEXT,
            "endpoint_semantic_case_id": ENDPOINT_SEMANTIC_CASE_SPEC[0],
            "endpoint_semantic_prompt_tokens": ENDPOINT_SEMANTIC_CASE_SPEC[1],
            "endpoint_semantic_stop": ["\n"],
            "full_endpoint_prompt_tokens": FULL_ENDPOINT_CASE_SPEC[1],
            "full_endpoint_requested_tokens": FULL_ENDPOINT_MAX_TOKENS,
            "full_endpoint_ignore_eos": True,
            "accepted_admission_geometry": "262143+1=262144",
            "rejected_admission_geometry": "262144+1=262145",
            "rejection_scheduler_dispatches": 0,
            "rejection_final_whole_log_census": True,
            "post_rejection_canary_prefill_metrics_exact": True,
            "k5_verify_graph_capture_use_required": True,
            "qsa_sparse_multi_row_selector": True,
            "qsa_sparse_multi_row_dense_endpoint": 2_048,
            "qsa_long_context_fallback": "row-by-row-exact",
            "graph_disabled_boundary_prompt_tokens": list(range(2_043, 2_051)),
            "graph_disabled_selectors": [
                "ATLAS_QWEN4_VERIFY_GRAPH=0",
                "ATLAS_DFLASH_DEBUG_NO_GRAPH=1",
            ],
            "rollback_required": (
                "completed SPEC_CYCLE_V2 accepted<gamma on the 2050-token "
                "request between distinct 2017-token canaries"
            ),
            "slot_reuse_geometry": "single model-owned slot",
        },
        "arms": {
            "plain": plain_receipt,
            "k5": k5_receipt,
            EAGER_CHURN_ARM: eager_receipt,
        },
        "comparison": comparison,
        "endpoint_admission": plain_receipt["endpoint_admission"],
        "graph_disabled_boundary_comparisons": {
            "plain_vs_forced_eager": plain_vs_forced_eager,
            "normal_k5_vs_forced_eager": normal_k5_vs_forced_eager,
        },
        "rollback_slot_reuse": eager_receipt["rollback_slot_reuse"],
    }
    payload = canonical_bytes(receipt)
    fd = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        write_all(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    return receipt


def dry_run_plan(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": SCHEMA + "/dry-run",
        "execution_scope": "direct-binary-not-container",
        "fresh_processes": list(ARM_NAMES),
        "request_identity": "same in-memory canonical body objects for both arms",
        "prompt_token_counts": list(PROMPT_TOKEN_COUNTS),
        "max_tokens": MAX_TOKENS,
        "repetitions": REPETITIONS,
        "endpoint_semantic": launch_request_contract()["endpoint_semantic"],
        "full_endpoint": {
            "prompt_tokens": FULL_ENDPOINT_CASE_SPEC[1],
            "max_tokens": FULL_ENDPOINT_MAX_TOKENS,
            "total_tokens": NATIVE_CONTEXT,
            "repetitions": FULL_ENDPOINT_REPETITIONS,
            "ignore_eos": True,
            "stop": [],
        },
        "admission_boundary": launch_request_contract()["admission_boundary"],
        "graph_disabled_boundary_case_ids": list(GRAPH_DISABLED_CASE_IDS),
        "rollback_slot_reuse_sequence": [
            {"phase": phase, "case_id": case_id}
            for phase, case_id in ROLLBACK_SLOT_REUSE_SEQUENCE
        ],
        "sampling": "exact-greedy",
        "required_pair_fields": [
            "request_sha256",
            "completion_tokens",
            "completion_token_ids",
            "completion_u32le_sha256",
            "finish_reason",
            "text",
            "parsed_answer",
            "final_answer",
            "correct",
        ],
        "required_k5_engagement_marker": K5_VERIFY_MARKER,
        "required_k5_graph_engagement_marker": K5_VERIFY_GRAPH_CAPTURE_MARKER,
        "required_k5_request_scope": (
            "one scheduler dispatch, at least one MTP_ACCEPT frame, and exactly "
            "one completed SPEC_CYCLE_V2 request ID per required K5 request"
        ),
        "required_rollback_evidence": (
            "completed SPEC_CYCLE_V2 accepted<gamma on the 2050-token request "
            "between distinct 2017-token canaries"
        ),
        "eager_churn_telemetry_hook": {
            "arm": EAGER_CHURN_ARM,
            "environment": dict(ARM_ENVIRONMENTS[EAGER_CHURN_ARM]),
            "world_size": 1,
            "tensor_parallel_size": 1,
            "expert_parallel_size": 1,
            "concurrency": 1,
            "high_speed_swap": False,
            "source_telemetry_overlay_included": True,
            "engagement_marker": QSA_ROUTE_TELEMETRY_ENGAGED_MARKER,
            "receipt_marker": QSA_ROUTE_RECEIPT_MARKER.strip(),
            "exact_layer_ids": list(QSA_FULL_ATTENTION_LAYER_IDS),
            "request_scoped": True,
        },
        "provenance": manifest["provenance"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--receipt", type=Path, default=Path("qwen4-mtp-long-context-pair-receipt.json")
    )
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--dry-run", action="store_true")
    parsed = parser.parse_args(argv)
    manifest = load_manifest(parsed.manifest, verify_files=not parsed.dry_run)
    if parsed.dry_run:
        print(json.dumps(dry_run_plan(manifest), indent=2, sort_keys=True))
        return 0
    receipt = run_live(manifest, parsed.manifest, parsed.receipt, parsed.timeout)
    print(
        f"PASS receipt={parsed.receipt} sha256={sha256_bytes(canonical_bytes(receipt))}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
