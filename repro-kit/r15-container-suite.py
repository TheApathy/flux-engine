#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Standalone, sealable R15 immutable-container qualification orchestrator.

The outer exact-image wrapper executes this file from a sealed memfd and passes
the suite plan through a second sealed memfd.  This program intentionally uses
only the Python standard library, imports no sibling module, and reads no
resource relative to ``__file__``.  Every gate driver is likewise opened from
the proven source root, copied to a sealed memfd, and executed without a shell.

This program orchestrates and records runtime work.  It does not manufacture a
PASS: every required gate must execute successfully, write an exact bound gate
receipt to its exclusive inherited descriptor, leave no process-group
descendant, and satisfy the outer gate-specific semantic validator.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import date
import ctypes
import fcntl
import hashlib
import json
import math
import mmap
import os
from pathlib import Path, PurePosixPath
import re
import select
import shlex
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
from typing import Any, Iterator, Mapping, Sequence
import urllib.parse

PLAN_SCHEMA = "flux-r15-container-suite-plan/2"
# Plan /3 carries profile contract /2 (per-profile geometry, both recipes and
# the full launch argv per mode); every other plan field is unchanged.
PLAN_SCHEMA_V3 = "flux-r15-container-suite-plan/3"
PLAN_SCHEMA_V4 = "flux-r15-container-suite-plan/4"
PLAN_PROFILE_CONTRACTS = {
    PLAN_SCHEMA: "flux-performance-profile-contract/1",
    PLAN_SCHEMA_V3: "flux-performance-profile-contract/2",
    PLAN_SCHEMA_V4: "flux-performance-profile-contract/2",
}
LIVE_REQUEST_V2 = "flux-r16-live-process-request/2"
LIVE_PROOF_V2 = "flux-r16-live-process-proof/2"
# Reviewed profile geometry /2: WIDE is per-chunk row geometry, so it equals
# the scheduler chunk and --max-prefill-tokens. Capacity is the native window.
PROFILE_GEOMETRY_V2 = {
    "interactive": {
        "recipe_relpath": "repro-kit/recipes/flux-v9-w4096-interactive.env",
        "flux_mode": "interactive-w4096",
        "chunk_tokens": 4096,
    },
    "high-prefill": {
        "recipe_relpath": "repro-kit/recipes/flux-v9-max-prefill.env",
        "flux_mode": "max-prefill",
        "chunk_tokens": 16384,
    },
}
PROFILE_CAPACITY_TOKENS = 262144
PROFILE_BLOCK_SIZE = 16
PROFILE_LAUNCH_MODEL = "@PROFILE_MODEL@"
PROFILE_LAUNCH_PORT = "@CONTAINER_PORT@"
# Host recipe argv engine selectors (recipes/flux-serve*.argv). Without them
# CLI defaults apply in the container only (gpu-memory 0.90, ssm-cache-slots
# 16, mtp-vocab 100000), so host and container would run different engines.
PROFILE_ENGINE_OVERLAY_V2 = (
    "--kernel-target",
    "qwen3.8-flash-next",
    "--gpu-memory-utilization",
    "0.85",
    "--ssm-cache-slots",
    "0",
    "--oom-guard-mb",
    "4096",
    "--request-timeout",
    "300",
    "--disable-simhash-watchdog",
    "--mtp-vocab",
    "0",
    "--mtp-quantization",
    "bf16",
)


def reviewed_profile_overlay_v2(chunk: int) -> list[str]:
    """The one sealed contract /2 overlay; profiles differ only in chunk."""
    return [
        "--max-prefill-tokens",
        str(chunk),
        "--max-seq-len",
        str(PROFILE_CAPACITY_TOKENS),
        "--block-size",
        str(PROFILE_BLOCK_SIZE),
        "--qwen4-qsa",
        "--scheduling-policy",
        "fifo",
        "--tbt-deadline-ms",
        "100",
        *PROFILE_ENGINE_OVERLAY_V2,
    ]
SUITE_SCHEMA = "flux-r15-container-suite-receipt/2"
GATE_SCHEMA = "flux-r15-container-gate-receipt/1"
# Compatibility default only; live identity comes from the sealed parent plan.
RELEASE_VERSION = "v0.9.13-r15"
RELEASE_VERSION_PATTERN = re.compile(
    r"v[0-9]+\.[0-9]+\.[0-9]+-(?:(r[1-9][0-9]*)|rc\.(r[1-9][0-9]*)\.([0-9]{8}))"
)
MAX_PLAN_BYTES = 4 * 1024 * 1024
MAX_GATE_RECEIPT_BYTES = 256 * 1024 * 1024
MAX_PROGRAM_BYTES = 16 * 1024 * 1024
MAX_PROGRAM_OUTPUT_BYTES = 64 * 1024 * 1024
MAX_RAW_LOG_BYTES = 8 * 1024 * 1024 * 1024
TERMINATE_GRACE_SECONDS = 5.0

REQUIRED_GATES = (
    "startup-readiness",
    "tool-api-streaming-blocking",
    "mmlu-pro",
    "gsm8k",
    "plain-k5-long-context",
    "native-262k",
    "native-262k-w4096-owner16",
    "native-concurrency-c2-c8",
    "prefill-decode-profiles",
    "memory-floor",
    "cancellation-slot-reuse",
    "raw-log-safety",
)
NATIVE_CONTEXT_SCHEMA = "flux-native-context-recipe-contract/1"
NATIVE_CONTEXT_REQUEST = "flux-r16-native-context-process-request/1"
NATIVE_CONTEXT_PROOF = "flux-r16-native-context-process-proof/1"
NATIVE_CONTEXT_RECIPES = {
    "native-262k": {
        "recipe_relpath": "repro-kit/recipes/flux-qwen4-native-262k-bf16.env",
        "recipe_sha256": "aa02f5d73b18ccb277743df034dd15c2105342a923283bac1265fe6ed12fbfe8",
        "chunk_tokens": 16384,
    },
    "native-262k-w4096-owner16": {
        "recipe_relpath": "repro-kit/recipes/flux-qwen4-native-262k-w4096-bf16-owner16.env",
        "recipe_sha256": "28c21eb3d42f7b643e0a556129a23cd0a5d6d7c6f61ab78ce318cbc0252d4718",
        "chunk_tokens": 4096,
    },
}


def validate_native_context_contract(value):
    if type(value) is not dict or set(value) != {"schema", "gates"}:
        raise ValueError("native context contract keys differ")
    if value["schema"] != NATIVE_CONTEXT_SCHEMA or type(value["gates"]) is not dict:
        raise ValueError("native context contract schema differs")
    if set(value["gates"]) != set(NATIVE_CONTEXT_RECIPES):
        raise ValueError("both native context recipes are required")
    for gate, expected in NATIVE_CONTEXT_RECIPES.items():
        row = value["gates"][gate]
        if type(row) is not dict or set(row) != set(expected):
            raise ValueError("native context recipe keys differ")
        if any(type(row[key]) is not type(wanted) or row[key] != wanted
               for key, wanted in expected.items()):
            raise ValueError("native context recipe authority differs")
    return value


def native_context_recipe_plan(launch, image, gate):
    if launch.get("schema") != "flux-r15-container-suite-plan/4":
        raise ValueError("native context requires suite plan/4")
    contract = validate_native_context_contract(launch.get("native_context_contract"))
    if type(gate) is not str or gate not in NATIVE_CONTEXT_RECIPES:
        raise ValueError("native context gate is not registered")
    for key in ("in_container_binary_path", "in_container_binary_sha256",
                "in_container_recipe_path", "recipe_sha256"):
        if image.get(key) != launch.get(key):
            raise ValueError("native context image differs from sealed launch")
    row = contract["gates"][gate]
    return {
        "in_container_binary_path": launch["in_container_binary_path"],
        "in_container_binary_sha256": launch["in_container_binary_sha256"],
        "in_container_recipe_path": "/opt/flux/" + row["recipe_relpath"],
        "recipe_sha256": row["recipe_sha256"],
        "native_gate": gate,
        "chunk_tokens": row["chunk_tokens"],
        "native_context_contract": contract,
    }


def validate_native_context_request(request, plan):
    if type(request) is not dict or set(request) != {"schema", "gate", "mode", "host_pid"}:
        raise ValueError("native context process request keys differ")
    gate = request["gate"]
    contract = validate_native_context_contract(plan.get("native_context_contract"))
    if (request["schema"] != NATIVE_CONTEXT_REQUEST or request["mode"] != "none"
            or type(gate) is not str or gate not in contract["gates"]
            or plan.get("native_gate") != gate):
        raise ValueError("native context process request identity differs")
    row = contract["gates"][gate]
    if (plan.get("in_container_recipe_path") != "/opt/flux/" + row["recipe_relpath"]
            or plan.get("recipe_sha256") != row["recipe_sha256"]
            or type(plan.get("chunk_tokens")) is not int
            or plan["chunk_tokens"] != row["chunk_tokens"]):
        raise ValueError("native context process recipe differs")
    if type(request["host_pid"]) is not int or not 0 < request["host_pid"] < (1 << 31):
        raise ValueError("native context process PID differs")
    return gate


def validate_native_context_live_geometry(plan, argv, environment):
    chunk = str(plan["chunk_tokens"]).encode()
    for key in (b"ATLAS_QWEN4_PREFILL_WIDE", b"FLUX_QWEN4_PREFILL_WIDE"):
        if environment.get(key) != chunk:
            raise ValueError("native context live WIDE differs")
    if _live_cli_values(argv, b"--max-prefill-tokens") != [chunk]:
        raise ValueError("native context live chunk differs")
    if (_live_cli_values(argv, b"--max-seq-len") != [b"262144"]
            or _live_cli_values(argv, b"--kv-cache-dtype") != [b"bf16"]
            or list(argv).count(b"--qwen4-qsa") != 1):
        raise ValueError("native context live capacity or BF16 QSA differs")
    if any(key in environment for key in (b"ATLAS_MAX_BATCH_TOKENS", b"FLUX_MAX_BATCH_TOKENS")):
        raise ValueError("native context unbound arena override")


RAW_LOG_PATTERNS = (
    "CUDA fatal",
    "CUDA out of memory",
    "illegal memory access",
    "NVRM",
    "Xid",
)
# Hostile log taxonomies are ever-changing: the census must not depend on the
# printed case ("CUDA fatal"/"cuda Fatal"/"nvrm").  Every label above maps to
# one case-insensitive spelled-out regex that is matched against the complete
# raw log.  Each label must still survive verbatim so the receipt contract and
# the outer wrapper agree on the exact public census labels.
_RAW_LOG_REGEX_LABELS = {
    "CUDA fatal": r"cuda[-_: ]*fatal",
    "CUDA out of memory": r"cuda[-_: ]*out[-_: ]*of[-_: ]*memory",
    "illegal memory access": r"illegal[-_: ]*memory[-_: ]*access",
    "NVRM": r"\bNVRM\b",
    "Xid": r"\bXid\b",
}
_RAW_LOG_COMPILED = tuple(
    (
        label,
        re.compile(regex.encode("ascii"), re.IGNORECASE | re.ASCII),
    )
    for label, regex in _RAW_LOG_REGEX_LABELS.items()
)
assert tuple(label for label, _ in _RAW_LOG_COMPILED) == RAW_LOG_PATTERNS
MAX_PATTERN_MATCH_COUNT = 100_000
PROCESS_PROOF_TOKEN = "@PROCESS_PROOF_FD@"
PROC_ROOT = Path("/proc")
IDENTITY_PLACEHOLDERS = (
    "@CONTAINER_ID@",
    "@IMAGE_ID@",
    "@IMAGE_DIGEST@",
    "@BASE_URL@",
    "@RUN_ID@",
    "@GATE_RECEIPT_FD@",
    "@RAW_LOG_FD@",
)
HEX_40 = re.compile(r"[0-9a-f]{40}")
HEX_64 = re.compile(r"[0-9a-f]{64}")
CID_RE = re.compile(r"[0-9a-f]{64}")
SAFE_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}")
SAFE_ARTIFACT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,255}")
SAFE_MODEL_ID = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}/[A-Za-z0-9][A-Za-z0-9_.-]{0,191}"
)


def reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def parse_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number {value!r} is forbidden")
    return parsed


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def strict_json_loads(raw: bytes | str, label: str) -> Any:
    try:
        return json.loads(
            raw,
            parse_constant=reject_constant,
            parse_float=parse_float,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is not strict JSON: {error}") from error


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def directory_identity_sha256(path: Path, label: str) -> str:
    descriptor = open_directory_chain(path, label)
    try:
        info = os.fstat(descriptor)
        return sha256_bytes(canonical_bytes([info.st_dev, info.st_ino]))
    finally:
        os.close(descriptor)


def exact_object(value: object, keys: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{label} keys differ")
    return value


def exact_int(value: object, label: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{label} must be an exact integer in [{minimum}, {maximum}]")
    return value


def exact_hex(value: object, width: int, label: str, *, prefixed: bool = False) -> str:
    if type(value) is not str:
        raise ValueError(f"{label} must be a string")
    raw = value
    if prefixed:
        if not raw.startswith("sha256:"):
            raise ValueError(f"{label} must have a sha256: prefix")
        raw = raw[7:]
    pattern = HEX_40 if width == 40 else HEX_64
    if pattern.fullmatch(raw) is None or set(raw) == {"0"}:
        raise ValueError(f"{label} must be a nonzero lowercase hexadecimal value")
    return value


def safe_artifact(value: object, label: str) -> str:
    if type(value) is not str or SAFE_ARTIFACT.fullmatch(value) is None:
        raise ValueError(f"{label} must be one safe runtime-relative locator")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"{label} must stay beneath runtime_dir")
    if path.as_posix() != value:
        raise ValueError(f"{label} must be normalized")
    return value


def normalized_absolute(value: object, label: str) -> Path:
    if type(value) is not str or not value or "\x00" in value:
        raise ValueError(f"{label} must be a non-empty NUL-free path")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError(f"{label} must be one normalized absolute path")
    return path


def open_directory_chain(path: Path, label: str) -> int:
    required = ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW")
    if any(not hasattr(os, name) for name in required):
        raise RuntimeError("no-follow directory traversal is unavailable")
    if (
        not path.is_absolute()
        or ".." in path.parts
        or str(Path(str(path))) != str(path)
    ):
        raise ValueError(f"{label} must be one normalized absolute path")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open("/", flags)
    try:
        for component in path.parts[1:]:
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError as error:
                raise ValueError(
                    f"{label} contains an unavailable or symlinked component"
                ) from error
            os.close(descriptor)
            descriptor = child
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ValueError(f"{label} is not a directory")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def open_absolute_file(path: Path, label: str) -> int:
    if not path.name:
        raise ValueError(f"{label} must identify a file")
    parent = open_directory_chain(path.parent, f"{label} parent")
    try:
        try:
            return os.open(
                path.name,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=parent,
            )
        except OSError as error:
            raise ValueError(f"{label} is unavailable") from error
    finally:
        os.close(parent)


def open_beneath(
    root: Path, locator: object, label: str, flags: int, mode: int = 0o777
) -> int:
    relative = PurePosixPath(safe_artifact(locator, label))
    parent = open_directory_chain(root, "runtime_dir")
    try:
        directory_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
        for component in relative.parts[:-1]:
            try:
                child = os.open(component, directory_flags, dir_fd=parent)
            except OSError as error:
                raise ValueError(
                    f"{label} contains an unavailable or symlinked parent"
                ) from error
            os.close(parent)
            parent = child
        try:
            return os.open(relative.parts[-1], flags, mode, dir_fd=parent)
        except FileExistsError:
            raise FileExistsError(f"refusing pre-existing {label}") from None
        except OSError as error:
            raise ValueError(f"cannot open {label}") from error
    finally:
        os.close(parent)


def create_beneath(root: Path, locator: object, label: str, mode: int) -> int:
    return open_beneath(
        root,
        locator,
        label,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        mode,
    )


def read_fd(descriptor: int, label: str, maximum: int, *, nlink: int | None) -> bytes:
    before = os.fstat(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or (nlink is not None and before.st_nlink != nlink)
        or before.st_size > maximum
    ):
        raise ValueError(f"{label} must be one bounded regular file")
    chunks: list[bytes] = []
    offset = 0
    remaining = before.st_size
    while remaining:
        chunk = os.pread(descriptor, min(1024 * 1024, remaining), offset)
        if not chunk:
            raise ValueError(f"{label} ended before its declared size")
        chunks.append(chunk)
        offset += len(chunk)
        remaining -= len(chunk)
    after = os.fstat(descriptor)

    def identity(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
        return (
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
            info.st_nlink,
        )

    if identity(before) != identity(after):
        raise ValueError(f"{label} changed while it was read")
    return b"".join(chunks)


def write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("short write")
        offset += written


def anonymous_writable_file(label: str) -> int:
    descriptor = os.memfd_create(label, os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    os.fchmod(descriptor, 0o600)
    return descriptor


def seal_written_file(descriptor: int, label: str, maximum: int) -> bytes:
    os.fsync(descriptor)
    seals = (
        fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE | fcntl.F_SEAL_SEAL
    )
    fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, seals)
    if fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) != seals:
        raise RuntimeError(f"{label} did not retain every memfd seal")
    return read_fd(descriptor, label, maximum, nlink=None)


def persist_reserved_file(descriptor: int, payload: bytes) -> None:
    if os.fstat(descriptor).st_size != 0:
        raise ValueError("reserved gate artifact was not empty")
    write_all(descriptor, payload)
    os.fsync(descriptor)
    os.fchmod(descriptor, 0o400)


def readonly_duplicate(descriptor: int, label: str) -> int:
    duplicate = os.open(f"/proc/self/fd/{descriptor}", os.O_RDONLY | os.O_CLOEXEC)
    original_info = os.fstat(descriptor)
    duplicate_info = os.fstat(duplicate)
    if (original_info.st_dev, original_info.st_ino) != (
        duplicate_info.st_dev,
        duplicate_info.st_ino,
    ):
        os.close(duplicate)
        raise ValueError(f"{label} descriptor identity changed")
    return duplicate


def assert_path_identity(root: Path, locator: str, descriptor: int, label: str) -> None:
    reopened = open_beneath(
        root, locator, label, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    )
    try:
        current = os.fstat(reopened)
        original = os.fstat(descriptor)
        if (
            not stat.S_ISREG(current.st_mode)
            or current.st_nlink != 1
            or (current.st_dev, current.st_ino) != (original.st_dev, original.st_ino)
        ):
            raise ValueError(f"{label} pathname identity changed")
    finally:
        os.close(reopened)


@contextmanager
def sealed_program(path: Path, expected_sha256: str, label: str) -> Iterator[int]:
    source = open_absolute_file(path, label)
    try:
        payload = read_fd(source, label, MAX_PROGRAM_BYTES, nlink=1)
        if sha256_bytes(payload) != expected_sha256:
            raise ValueError(f"{label} SHA-256 mismatch")
        descriptor = os.memfd_create(label, os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
        try:
            write_all(descriptor, payload)
            os.fchmod(descriptor, 0o500)
            seals = (
                fcntl.F_SEAL_SHRINK
                | fcntl.F_SEAL_GROW
                | fcntl.F_SEAL_WRITE
                | fcntl.F_SEAL_SEAL
            )
            fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, seals)
            if fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) != seals:
                raise RuntimeError(f"{label} did not retain every memfd seal")
            yield descriptor
        finally:
            os.close(descriptor)
    finally:
        os.close(source)


def validate_base_url(value: object) -> str:
    if type(value) is not str:
        raise ValueError("base URL must be a string")
    parsed = urllib.parse.urlsplit(value)
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("base URL port is invalid") from error
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or port is None
        or not 1 <= port <= 65535
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("base URL must be one unadorned loopback HTTP origin")
    return f"http://127.0.0.1:{port}"


def release_identity(value: object) -> dict[str, str]:
    """Derive public labels and the source-schema lane from one exact version."""
    if type(value) is not str or not 0 < len(value) <= 96:
        raise ValueError("release version must be one bounded exact string")
    match = RELEASE_VERSION_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError("release version must be vX.Y.Z-rN or vX.Y.Z-rc.rN.YYYYMMDD")
    if match.group(3) is not None:
        stamp = match.group(3)
        date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:]))
    return {"release_version": value, "oci_version": value[1:],
            "source_lane": match.group(1) or match.group(2)}


def load_plan(descriptor: int) -> dict[str, Any]:
    seals = (
        fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE | fcntl.F_SEAL_SEAL
    )
    if fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) != seals:
        raise ValueError("suite plan descriptor is not fully sealed")
    raw = read_fd(descriptor, "suite plan", MAX_PLAN_BYTES, nlink=None)
    value = exact_object(
        strict_json_loads(raw, "suite plan"),
        {
            "schema",
            "release_version",
            "runtime_dir",
            "source_root_path",
            "source_revision",
            "source_tree",
            "source_bundle_manifest_sha256",
            "in_container_binary_sha256",
            "in_container_binary_path",
            "in_container_recipe_path",
            "recipe_sha256",
            "checkpoint_id",
            "checkpoint_revision",
            "checkpoint_manifest_sha256",
            "performance_profile_contract",
            "native_context_contract",
            "gate_jobs",
            "raw_log_relpath",
        },
        "suite plan",
    )
    if value["schema"] != PLAN_SCHEMA_V4:
        raise ValueError("suite plan schema differs")
    release_identity(value["release_version"])
    runtime = normalized_absolute(value["runtime_dir"], "suite plan runtime_dir")
    source_root = normalized_absolute(
        value["source_root_path"], "suite plan source_root_path"
    )
    runtime_identity_sha256 = directory_identity_sha256(
        runtime, "suite plan runtime_dir"
    )
    open_source = open_directory_chain(source_root, "suite plan source_root_path")
    os.close(open_source)
    exact_hex(value["source_revision"], 40, "suite plan source_revision")
    exact_hex(value["source_tree"], 40, "suite plan source_tree")
    for key, prefix in (
        ("in_container_binary_path", "/opt/flux/bin/"),
        ("in_container_recipe_path", "/opt/flux/repro-kit/recipes/"),
    ):
        item = value[key]
        if (
            type(item) is not str
            or not item.startswith(prefix)
            or str(PurePosixPath(item)) != item
            or ".." in PurePosixPath(item).parts
        ):
            raise ValueError(f"suite plan {key} is not an immutable release path")
    for key in (
        "source_bundle_manifest_sha256",
        "in_container_binary_sha256",
        "recipe_sha256",
        "checkpoint_manifest_sha256",
    ):
        exact_hex(value[key], 64, f"suite plan {key}")
    if (
        type(value["checkpoint_id"]) is not str
        or SAFE_MODEL_ID.fullmatch(value["checkpoint_id"]) is None
    ):
        raise ValueError("suite plan checkpoint_id must be one safe public model ID")
    revision = value["checkpoint_revision"]
    if type(revision) is not str or not (
        HEX_40.fullmatch(revision) or HEX_64.fullmatch(revision)
    ):
        raise ValueError("suite plan checkpoint_revision must be immutable")
    validate_performance_profile_contract(
        value["performance_profile_contract"],
        value["schema"],
        value["in_container_binary_path"],
    )
    validate_native_context_contract(value["native_context_contract"])
    raw_log = safe_artifact(value["raw_log_relpath"], "suite plan raw_log_relpath")
    jobs = exact_object(value["gate_jobs"], set(REQUIRED_GATES), "suite plan gate_jobs")
    seen_artifacts = {raw_log}
    for gate in REQUIRED_GATES:
        label = f"suite plan gate {gate}"
        job = exact_object(
            jobs[gate],
            {
                "program_path",
                "program_sha256",
                "argv",
                "timeout_seconds",
                "receipt_relpath",
                "stdout_relpath",
                "stderr_relpath",
            },
            label,
        )
        program = normalized_absolute(job["program_path"], f"{label} program_path")
        try:
            program.relative_to(source_root)
        except ValueError as error:
            raise ValueError(
                f"{label} program must stay in source_root_path"
            ) from error
        exact_hex(job["program_sha256"], 64, f"{label} program_sha256")
        argv = job["argv"]
        if type(argv) is not list or not argv or argv[0] != str(program):
            raise ValueError(f"{label} argv must start with its exact program path")
        if any(type(item) is not str or not item or "\x00" in item for item in argv):
            raise ValueError(f"{label} argv entries must be non-empty NUL-free strings")
        if argv.count(PROCESS_PROOF_TOKEN) != (1 if gate == "startup-readiness" else 0):
            raise ValueError(f"{label} startup process-proof token differs")
        counts = {item: argv.count(item) for item in IDENTITY_PLACEHOLDERS}
        if any(count != 1 for count in counts.values()):
            raise ValueError(f"{label} argv must contain every identity token once")
        if any(
            placeholder in item and item != placeholder
            for item in argv
            for placeholder in IDENTITY_PLACEHOLDERS
        ):
            raise ValueError(f"{label} identity tokens must not be embedded")
        exact_int(job["timeout_seconds"], f"{label} timeout_seconds", 1, 86_400)
        for key in ("receipt_relpath", "stdout_relpath", "stderr_relpath"):
            locator = safe_artifact(job[key], f"{label} {key}")
            if locator in seen_artifacts:
                raise ValueError("suite plan artifact locators must be globally unique")
            seen_artifacts.add(locator)
    value["_raw_sha256"] = sha256_bytes(raw)
    value["_runtime_dir_identity_sha256"] = runtime_identity_sha256
    return value


def profile_launch_argv(
    binary_path: str, model: str, port: str, overlay: Sequence[str], mode: str
) -> list[str]:
    """The exact profile server argv; plan /3 binds its template by hash."""
    argv = [
        binary_path,
        "serve",
        "--no-tui",
        "--model-from-path",
        "/models",
        "--model-name",
        model,
        "--bind",
        "0.0.0.0",
        "--port",
        port,
        "--max-batch-size",
        "1",
        "--max-num-seqs",
        "1",
        "--kv-cache-dtype",
        "bf16",
        "--enable-prefix-caching",
        "false",
        *overlay,
    ]
    if mode == "native-mtp-k5":
        argv += ["--speculative", "--num-drafts", "4"]
    return argv


def profile_launch_template_sha256(
    binary_path: str, overlay: Sequence[str], mode: str
) -> str:
    return sha256_bytes(
        canonical_bytes(
            profile_launch_argv(
                binary_path, PROFILE_LAUNCH_MODEL, PROFILE_LAUNCH_PORT, overlay, mode
            )
        )
    )


def profile_overlay_v2(argv: object, chunk: int) -> dict[str, Any]:
    """Contract /2 overlay: every selector explicit once, --qwen4-qsa present."""
    if (
        type(argv) is not list
        or len(argv) != 11 + len(PROFILE_ENGINE_OVERLAY_V2)
        or any(type(item) is not str or not item or "\x00" in item for item in argv)
    ):
        raise ValueError("performance profile /2 overlay is malformed")
    if argv[11:] != list(PROFILE_ENGINE_OVERLAY_V2):
        raise ValueError("performance profile /2 overlay differs from the host recipe engine argv")
    if type(chunk) is not int or argv != reviewed_profile_overlay_v2(chunk):
        raise ValueError("performance profile /2 overlay differs from the sealed profile recipe")
    argv = argv[:11]
    if argv.count("--qwen4-qsa") != 1:
        raise ValueError("performance profile /2 overlay must enable --qwen4-qsa")
    pairs = [item for item in argv if item != "--qwen4-qsa"]
    values: dict[str, Any] = {}
    for key, item in zip(pairs[::2], pairs[1::2]):
        if (
            key
            not in {
                "--max-prefill-tokens",
                "--max-seq-len",
                "--block-size",
                "--scheduling-policy",
                "--tbt-deadline-ms",
            }
            or key in values
        ):
            raise ValueError("performance profile selector is unsupported or repeated")
        values[key] = item
    if len(values) != 5:
        raise ValueError("performance profile /2 overlay omits a pinned selector")
    for key, expected in (
        ("--max-prefill-tokens", chunk),
        ("--max-seq-len", PROFILE_CAPACITY_TOKENS),
        ("--block-size", PROFILE_BLOCK_SIZE),
    ):
        if re.fullmatch(r"[0-9]+", values[key]) is None or int(values[key]) != expected:
            raise ValueError("performance profile chunk/context/block differs")
        values[key] = expected
    if (
        values["--scheduling-policy"] not in ("fifo", "slai")
        or re.fullmatch(r"[0-9]+", values["--tbt-deadline-ms"]) is None
        or int(values["--tbt-deadline-ms"]) > (1 << 64) - 1
    ):
        raise ValueError("performance profile scheduler differs")
    values["--tbt-deadline-ms"] = int(values["--tbt-deadline-ms"])
    values["--qwen4-qsa"] = True
    return values


def _validate_profile_contract_v2(contract: dict[str, Any], binary_path: str) -> None:
    profiles = exact_object(
        contract["profiles"], set(PROFILE_GEOMETRY_V2), "performance profiles"
    )
    hashes, without_chunk = set(), []
    for name, geometry in PROFILE_GEOMETRY_V2.items():
        row = exact_object(
            profiles[name],
            {
                "recipe_relpath",
                "recipe_sha256",
                "in_container_recipe_path",
                "chunk_tokens",
                "argv_overlay",
                "launch_argv_sha256",
            },
            "performance profile /2 binding",
        )
        if (
            row["recipe_relpath"] != geometry["recipe_relpath"]
            or row["in_container_recipe_path"] != "/opt/flux/" + geometry["recipe_relpath"]
        ):
            raise ValueError("performance profile canonical recipe differs")
        exact_hex(row["recipe_sha256"], 64, "performance profile recipe hash")
        hashes.add(row["recipe_sha256"])
        if type(row["chunk_tokens"]) is not int or row["chunk_tokens"] != geometry["chunk_tokens"]:
            raise ValueError("performance profile chunk differs from reviewed geometry")
        values = profile_overlay_v2(row["argv_overlay"], geometry["chunk_tokens"])
        del values["--max-prefill-tokens"]
        without_chunk.append(values)
        templates = exact_object(
            row["launch_argv_sha256"],
            {"none", "native-mtp-k5"},
            "performance profile launch argv hashes",
        )
        for mode, observed in templates.items():
            if observed != profile_launch_template_sha256(
                binary_path, row["argv_overlay"], mode
            ):
                raise ValueError("performance profile full launch argv hash differs")
    if len(hashes) != 2:
        raise ValueError("performance profiles must bind two distinct recipes")
    if without_chunk[0] != without_chunk[1]:
        raise ValueError("performance profiles differ outside the scheduler chunk")


def profile_contract_chunk(contract: Mapping[str, Any], profile: str) -> int:
    """Version dispatch for the scheduler chunk of one profile."""
    if contract["schema"] == PLAN_PROFILE_CONTRACTS[PLAN_SCHEMA]:
        return 4096
    return contract["profiles"][profile]["chunk_tokens"]


def validate_performance_profile_contract(
    value: object, plan_schema: str = PLAN_SCHEMA, binary_path: str | None = None
) -> None:
    contract = exact_object(
        value, {"schema", "profiles"}, "performance profile contract"
    )
    if contract["schema"] != PLAN_PROFILE_CONTRACTS.get(plan_schema):
        raise ValueError("performance profile contract schema differs")
    if plan_schema in (PLAN_SCHEMA_V3, PLAN_SCHEMA_V4):
        if type(binary_path) is not str:
            raise ValueError("performance profile /2 needs the release binary path")
        _validate_profile_contract_v2(contract, binary_path)
        return
    recipes = {
        "interactive": "repro-kit/recipes/flux-v9-w4096-interactive.env",
        "high-prefill": "repro-kit/recipes/flux-v9-max-prefill.env",
    }
    profiles = exact_object(contract["profiles"], set(recipes), "performance profiles")
    effective = []
    for name, path in recipes.items():
        row = exact_object(
            profiles[name],
            {"recipe_relpath", "recipe_sha256", "argv_overlay"},
            "performance profile binding",
        )
        if row["recipe_relpath"] != path:
            raise ValueError("performance profile canonical recipe differs")
        exact_hex(row["recipe_sha256"], 64, "performance profile recipe hash")
        if row["recipe_sha256"] == "0" * 64:
            raise ValueError("performance profile recipe hash is unset")
        argv = row["argv_overlay"]
        if (
            type(argv) is not list
            or len(argv) % 2
            or len(argv) > 8
            or any(type(item) is not str or not item or "\x00" in item for item in argv)
        ):
            raise ValueError("performance profile overlay is malformed")
        values = {}
        for key, item in zip(argv[::2], argv[1::2]):
            if (
                key
                not in {
                    "--max-prefill-tokens",
                    "--max-seq-len",
                    "--scheduling-policy",
                    "--tbt-deadline-ms",
                }
                or key in values
            ):
                raise ValueError(
                    "performance profile selector is unsupported or repeated"
                )
            values[key] = item
        for key, expected in (
            ("--max-prefill-tokens", 4096),
            ("--max-seq-len", 262144),
        ):
            item = values.get(key)
            if (
                type(item) is not str
                or re.fullmatch(r"[0-9]+", item) is None
                or int(item) != expected
            ):
                raise ValueError("performance profile chunk/context differs")
            values[key] = expected
        policy = values.setdefault("--scheduling-policy", "fifo")
        deadline = values.get("--tbt-deadline-ms", "100")
        if (
            policy not in ("fifo", "slai")
            or re.fullmatch(r"[0-9]+", deadline) is None
            or int(deadline) > (1 << 64) - 1
        ):
            raise ValueError("performance profile scheduler differs")
        values["--tbt-deadline-ms"] = int(deadline) if policy == "slai" else None
        effective.append(values)
    if effective[0] == effective[1]:
        raise ValueError("performance profiles share one effective argv")


def substitute_argv(
    argv: Sequence[str],
    identities: Mapping[str, str],
    receipt_fd: int,
    raw_log_fd: int,
) -> list[str]:
    replacements = {
        "@CONTAINER_ID@": identities["container_id"],
        "@IMAGE_ID@": identities["image_id"],
        "@IMAGE_DIGEST@": identities["image_digest"],
        "@BASE_URL@": identities["base_url"],
        "@RUN_ID@": identities["run_id"],
        "@GATE_RECEIPT_FD@": str(receipt_fd),
        "@RAW_LOG_FD@": str(raw_log_fd),
    }
    return [replacements.get(item, item) for item in argv]


def run_gate_program(
    argv: Sequence[str],
    *,
    executable_fd: int,
    receipt_fd: int,
    raw_log_fd: int,
    stdout_fd: int,
    stderr_fd: int,
    timeout_seconds: int,
    process_proof: tuple[dict[str, Any], Mapping[str, str]] | None = None,
) -> list[dict[str, Any]]:
    with owned_gate_children():
        return _run_gate_program_owned(
            argv,
            executable_fd=executable_fd,
            receipt_fd=receipt_fd,
            raw_log_fd=raw_log_fd,
            stdout_fd=stdout_fd,
            stderr_fd=stderr_fd,
            timeout_seconds=timeout_seconds,
            process_proof=process_proof,
        )


def _direct_children() -> list[int]:
    try:
        raw = Path(f"/proc/self/task/{os.getpid()}/children").read_text("ascii")
        return [int(value) for value in raw.split()]
    except (OSError, ValueError) as error:
        raise RuntimeError("kernel child census is unavailable") from error


@contextmanager
def owned_gate_children() -> Iterator[None]:
    """Adopt orphaned descendants before launch, including double-fork/setsid.

    This standalone single-thread supervisor must start with no children. The
    kernel then reparents every orphan in this launch subtree to us, rather
    than losing ownership when a driver changes its process group/session.
    """
    if threading.active_count() != 1 or len(os.listdir("/proc/self/task")) != 1:
        raise RuntimeError("gate containment requires a single-thread supervisor")
    if _direct_children():
        raise RuntimeError("gate containment requires an empty child namespace")
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise RuntimeError("gate containment requires Linux pidfds")
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(previous), 0, 0, 0) != 0:
        raise RuntimeError("cannot query child subreaper support")
    if libc.prctl(36, 1, 0, 0, 0) != 0:
        raise RuntimeError("cannot enable child subreaper containment")
    try:
        yield
    finally:
        leaked = bool(_direct_children())
        deadline = time.monotonic() + TERMINATE_GRACE_SECONDS
        while True:
            children = _direct_children()
            if not children:
                break
            for pid in children:
                try:
                    descriptor = os.pidfd_open(pid)
                except ProcessLookupError:
                    continue
                try:
                    signal.pidfd_send_signal(descriptor, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                finally:
                    os.close(descriptor)
            while True:
                try:
                    reaped, _status = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    break
                if not reaped:
                    break
            if time.monotonic() >= deadline:
                raise RuntimeError("gate descendants could not be drained")
            time.sleep(0.01)
        if libc.prctl(36, previous.value, 0, 0, 0) != 0:
            raise RuntimeError("cannot restore child subreaper state")
        if leaked:
            raise RuntimeError("gate driver left a process descendant")


def _run_gate_program_owned(
    argv: Sequence[str],
    *,
    executable_fd: int,
    receipt_fd: int,
    raw_log_fd: int,
    stdout_fd: int,
    stderr_fd: int,
    timeout_seconds: int,
    process_proof: tuple[dict[str, Any], Mapping[str, str]] | None = None,
) -> list[dict[str, Any]]:
    supervisor = driver = None
    pass_fds = (executable_fd, receipt_fd, raw_log_fd)
    command = list(argv)
    if process_proof is not None:
        supervisor, driver = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        command = [
            str(driver.fileno()) if item == PROCESS_PROOF_TOKEN else item
            for item in command
        ]
        pass_fds += (driver.fileno(),)
    try:
        process = subprocess.Popen(
            command,
            executable=f"/proc/self/fd/{executable_fd}",
            stdin=subprocess.DEVNULL,
            stdout=stdout_fd,
            stderr=stderr_fd,
            env={
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            pass_fds=pass_fds,
            start_new_session=True,
        )
    except BaseException:
        for channel in (supervisor, driver):
            if channel is not None:
                channel.close()
        raise
    proofs: list[dict[str, Any]] = []
    if driver is not None:
        driver.close()
    try:
        if supervisor is None:
            process.wait(timeout=timeout_seconds)
        else:
            deadline = time.monotonic() + timeout_seconds
            while process.poll() is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout_seconds)
                readable, _, _ = select.select(
                    [supervisor], [], [], min(0.05, remaining)
                )
                if not readable:
                    continue
                packet, _ancillary, flags, _address = supervisor.recvmsg(16 * 1024)
                if flags & socket.MSG_TRUNC:
                    raise ValueError("startup process-proof packet is oversized")
                if not packet:
                    if len(proofs) != 2:
                        raise ValueError("startup driver omitted live process proofs")
                    process.wait(timeout=max(0.01, deadline - time.monotonic()))
                    break
                if len(proofs) >= 2:
                    raise ValueError("startup driver sent extra process proofs")
                plan, identities = process_proof
                proof = verify_live_server(
                    strict_json_loads(packet, "live process request"), plan, identities
                )
                if proof["mode"] != ("none", "native-mtp-k5")[len(proofs)]:
                    raise ValueError("startup process proofs are out of order")
                if any(
                    (row["host_pid"], row["start_time_ticks"])
                    == (proof["host_pid"], proof["start_time_ticks"])
                    for row in proofs
                ):
                    raise ValueError("startup modes reused the same process")
                proofs.append(proof)
                supervisor.sendall(canonical_bytes(proof))
    except BaseException:
        process.kill()
        try:
            process.wait(timeout=TERMINATE_GRACE_SECONDS)
        except BaseException:
            pass
        raise
    finally:
        if supervisor is not None:
            supervisor.close()
    if process.returncode != 0:
        raise RuntimeError(f"gate driver exited rc={process.returncode}")
    if process_proof is not None and len(proofs) != 2:
        raise ValueError("startup driver omitted live process proofs")
    return proofs


def _hash_regular_fd(
    descriptor: int, label: str, limit: int
) -> tuple[str, bytes | None]:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= limit:
        raise ValueError(f"{label} is not a bounded regular file")
    digest = hashlib.sha256()
    parts = [] if limit <= MAX_PROGRAM_BYTES else None
    offset = 0
    while offset < before.st_size:
        block = os.pread(descriptor, min(1024 * 1024, before.st_size - offset), offset)
        if not block:
            raise ValueError(f"{label} was truncated")
        digest.update(block)
        if parts is not None:
            parts.append(block)
        offset += len(block)
    after = os.fstat(descriptor)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ValueError(f"{label} changed during verification")
    return digest.hexdigest(), None if parts is None else b"".join(parts)


def _open_process_file(root_fd: int, path: str) -> int:
    parts = PurePosixPath(path).parts[1:]
    parent = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            child = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=parent,
            )
            os.close(parent)
            parent = child
        return os.open(
            parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent
        )
    finally:
        os.close(parent)


def _process_start_ticks(pid: int) -> int:
    raw = (PROC_ROOT / str(pid) / "stat").read_bytes()
    return int(raw[raw.rfind(b")") + 2 :].split()[19])


def _recipe_environment(payload: bytes) -> dict[bytes, bytes]:
    values: dict[bytes, bytes] = {}
    for line in payload.decode("utf-8").splitlines():
        tokens = shlex.split(line, comments=True, posix=True)
        if not tokens:
            continue
        if tokens[0] == "export":
            tokens = tokens[1:]
        if (
            len(tokens) != 1
            or "=" not in tokens[0]
            or any(character in tokens[0] for character in "$`")
        ):
            raise ValueError("release recipe must use literal environment assignments")
        key, value = tokens[0].split("=", 1)
        if (
            re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) is None
            or key.encode() in values
        ):
            raise ValueError("release recipe assignment is ambiguous")
        values[key.encode()] = value.encode()
    if not values:
        raise ValueError("release recipe is empty")
    return values


def _require_readonly_paths(mountinfo: str, paths: Sequence[str]) -> None:
    mounts = []
    for line in mountinfo.splitlines():
        fields = line.split()
        if len(fields) < 10:
            raise ValueError("server mount topology is malformed")
        point = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), fields[4])
        mounts.append((PurePosixPath(point), fields[5].split(",")))
    for path in paths:
        target = PurePosixPath(path)
        covered = [
            (point, options)
            for point, options in mounts
            if point == target or point in target.parents
        ]
        if not covered:
            raise ValueError("server release path has no mount mapping")
        longest = max(len(point.parts) for point, _ in covered)
        selected = [
            options for point, options in covered if len(point.parts) == longest
        ]
        if len(selected) != 1 or "ro" not in selected[0]:
            raise ValueError("server executable or recipe is on a writable mount")


def _read_process_bytes(path: Path, label: str, maximum: int) -> bytes:
    with path.open("rb") as stream:
        payload = stream.read(maximum + 1)
    if len(payload) > maximum:
        raise ValueError(f"live {label} is oversized")
    return payload


def _live_cli_values(argv: Sequence[bytes], option: bytes) -> list[bytes]:
    values = []
    for index, item in enumerate(argv):
        if item == b"--":
            break
        if item.startswith(option + b"="):
            values.append(item[len(option) + 1 :])
        elif item == option:
            if index + 1 >= len(argv) or argv[index + 1].startswith(b"-"):
                raise ValueError("live server CLI selector has no value")
            values.append(argv[index + 1])
    return values


def _verify_live_mode(
    argv: Sequence[bytes], environment: Mapping[bytes, bytes], mode: str
) -> None:
    # main::sync_flux_env_aliases fills absent ATLAS keys once at startup.
    # /proc/environ may still show only the original FLUX assignment.
    def effective(key: bytes) -> bytes | None:
        return environment.get(key, environment.get(b"FLUX_" + key[6:]))

    disabled = effective(b"ATLAS_DISABLE_SPECULATION")
    if disabled not in (None, b"0", b"1"):
        raise ValueError("live speculation kill switch is malformed")
    option_argv = list(argv[: argv.index(b"--")]) if b"--" in argv else list(argv)
    options = {item.split(b"=", 1)[0] for item in option_argv}
    if option_argv.count(b"--no-tui") != 1:
        raise ValueError(
            "live release process requires --no-tui to prohibit model swaps"
        )
    foreign = {
        b"--mtp-from-path",
        b"--draft-model",
        b"--dflash-donor-model",
        b"--dflash",
        b"--self-speculative",
        b"--ngram-speculative",
        b"--mtp-gate",
    }
    if (
        options & foreign
        or any(environment.get(key) for key in (b"FLUX_DRAFT_MODEL", b"FLUX_MTP_PATH"))
        or effective(b"ATLAS_REST_STORE") is not None
        or effective(b"ATLAS_SELF_CONTEXT_DRAFT") == b"1"
    ):
        raise ValueError("live process is not native K5 or target-only plain")
    if mode == "none":
        if disabled != b"1":
            raise ValueError(
                "live plain process requires effective ATLAS_DISABLE_SPECULATION=1"
            )
    elif (
        disabled == b"1"
        or option_argv.count(b"--speculative") != 1
        or _live_cli_values(argv, b"--num-drafts") != [b"4"]
        or environment.get(b"FLUX_SPEC_MODE") == b"dflash"
    ):
        raise ValueError("live process is not native K5")


def _checkpoint_mount_source(document: Mapping[str, Any], mountinfo: str) -> Path:
    mounts = document.get("Mounts")
    if type(mounts) is not list:
        raise ValueError("live checkpoint mount inventory is missing")
    selected = [
        item
        for item in mounts
        if type(item) is dict and item.get("Destination") == "/models"
    ]
    if (
        len(selected) != 1
        or selected[0].get("Type") != "bind"
        or selected[0].get("RW") is not False
    ):
        raise ValueError("live checkpoint requires the read-only /models bind")
    points = [
        PurePosixPath(
            re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), line.split()[4])
        )
        for line in mountinfo.splitlines()
    ]
    if points.count(PurePosixPath("/models")) != 1 or any(
        PurePosixPath("/models") in point.parents for point in points
    ):
        raise ValueError("live checkpoint mount is missing or has nested overrides")
    return normalized_absolute(selected[0].get("Source"), "live checkpoint source")


def _live_listener_identity(
    proc: Path,
    init_proc: Path,
    document: Mapping[str, Any],
    identities: Mapping[str, str],
    argv: Sequence[bytes],
) -> tuple[int, int, int, str]:
    namespace = (proc / "ns/net").stat().st_ino
    if namespace != (init_proc / "ns/net").stat().st_ino:
        raise ValueError("live server network namespace differs from container")
    port_values = _live_cli_values(argv, b"--port")
    if len(port_values) > 1 or (port_values and not port_values[0].isdigit()):
        raise ValueError("live server port selector is ambiguous")
    port = int(port_values[0]) if port_values else 8888
    if not 1 <= port <= 65535:
        raise ValueError("live server port is out of range")
    host_port = urllib.parse.urlsplit(validate_base_url(identities["base_url"])).port
    host = document.get("HostConfig")
    if type(host) is not dict or type(host.get("NetworkMode")) is not str:
        raise ValueError("live container network mode is missing")
    host_namespace = (PROC_ROOT / "self/ns/net").stat().st_ino
    host_network = host["NetworkMode"] == "host"
    if host_network:
        if namespace != host_namespace or port != host_port:
            raise ValueError("live host-network endpoint differs from the release URL")
    else:
        if namespace == host_namespace:
            raise ValueError(
                "live published endpoint unexpectedly uses host networking"
            )
        settings = document.get("NetworkSettings")
        mappings = settings.get("Ports") if type(settings) is dict else None
        if type(mappings) is not dict:
            raise ValueError("live published-port mapping is missing")
        selected = []
        for key, bindings in mappings.items():
            if not key.endswith("/tcp"):
                continue
            if bindings is None:
                continue
            if type(bindings) is not list:
                raise ValueError("live published-port mapping is malformed")
            for binding in bindings:
                if type(binding) is not dict:
                    raise ValueError("live published-port binding is malformed")
                if binding.get("HostIp") in ("127.0.0.1", "0.0.0.0") and binding.get(
                    "HostPort"
                ) == str(host_port):
                    selected.append(key)
        if selected != [f"{port}/tcp"]:
            raise ValueError("live published endpoint differs from the release URL")
    owned_sockets = set()
    with os.scandir(proc / "fd") as entries:
        for count, entry in enumerate(entries, 1):
            if count > 65536:
                raise ValueError("live descriptor inventory is oversized")
            try:
                target = os.readlink(entry.path)
            except FileNotFoundError:
                continue
            match = re.fullmatch(r"socket:\[([0-9]+)\]", target)
            if match:
                owned_sockets.add(int(match[1]))
    listeners = []
    for table, width in (("tcp", 8), ("tcp6", 32)):
        raw = _read_process_bytes(
            proc / "net" / table, "TCP listener table", 8 * 1024 * 1024
        )
        lines = raw.decode("ascii").splitlines()
        if not lines or "local_address" not in lines[0] or "inode" not in lines[0]:
            raise ValueError("live TCP listener table is malformed")
        for line in lines[1:]:
            fields = line.split()
            if len(fields) < 10:
                raise ValueError("live TCP listener row is malformed")
            local = re.fullmatch(
                rf"([0-9A-Fa-f]{{{width}}}):([0-9A-Fa-f]{{4}})", fields[1]
            )
            if local is None:
                raise ValueError("live TCP listener address is malformed")
            if fields[3] != "0A" or int(local[2], 16) != port:
                continue
            address = local[1].upper()
            if table == "tcp6":
                if address == "0" * 32 or address.endswith("FFFF00000100007F"):
                    raise ValueError(
                        "live IPv4 endpoint has an ambiguous IPv6 listener"
                    )
                continue
            allowed = {"00000000", "0100007F"} if host_network else {"00000000"}
            if address in allowed:
                listeners.append((int(fields[9]), address))
    if len(listeners) != 1 or listeners[0][0] not in owned_sockets:
        raise ValueError("release endpoint is not uniquely owned by the live server")
    return namespace, port, listeners[0][0], listeners[0][1]


def load_native_context_recipe(args, image, gate):
    parent = os.getppid()
    argv = Path(f"/proc/{parent}/cmdline").read_bytes().rstrip(b"\0").split(b"\0")
    if argv.count(b"--plan-fd") != 1:
        raise ValueError("native context sealed parent plan is missing")
    index = argv.index(b"--plan-fd") + 1
    if index >= len(argv) or re.fullmatch(rb"[0-9]+", argv[index]) is None:
        raise ValueError("native context parent plan descriptor differs")
    descriptor = os.open(f"/proc/{parent}/fd/{int(argv[index])}", os.O_RDONLY | os.O_CLOEXEC)
    try:
        launch = load_plan(descriptor)
        if os.getppid() != parent:
            raise ValueError("native context parent changed")
    finally:
        os.close(descriptor)
    selected = native_context_recipe_plan(launch, image, gate)
    document = strict_json_loads(args.lifecycle.docker([
        "container", "inspect", "--format", "{{json .}}", args.container_id
    ]), "native context container")
    if (type(document) is not dict or document.get("Id") != args.container_id
            or document.get("Image") != args.image_id
            or document.get("State", {}).get("Running") is not True
            or document.get("HostConfig", {}).get("ReadonlyRootfs") is not True):
        raise ValueError("native context immutable container identity differs")
    init_pid = exact_int(document["State"].get("Pid"), "container init PID", 1, 1 << 31)
    root = os.open(f"/proc/{init_pid}/root", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        recipe_fd = _open_process_file(root, selected["in_container_recipe_path"])
        try:
            observed, payload = _hash_regular_fd(recipe_fd, "native context recipe", MAX_PROGRAM_BYTES)
            if observed != selected["recipe_sha256"]:
                raise ValueError("native context image recipe bytes differ")
            environment = {key.decode(): value.decode()
                           for key, value in _recipe_environment(payload).items()}
        finally:
            os.close(recipe_fd)
    finally:
        os.close(root)
    return selected, environment


def native_context_request(plan, gate, pid):
    request = {"schema": NATIVE_CONTEXT_REQUEST, "gate": gate, "mode": "none", "host_pid": pid}
    validate_native_context_request(request, plan)
    return request


def verify_live_server(
    request: object, plan: dict[str, Any], identities: Mapping[str, str]
) -> dict[str, Any]:
    profile = None
    native_gate = None
    if type(request) is dict and request.get("schema") == NATIVE_CONTEXT_REQUEST:
        native_gate = validate_native_context_request(request, plan)
    elif type(request) is dict and request.get("schema") == LIVE_REQUEST_V2:
        # Broker /2: exactly one of the two reviewed profile recipes, by sha.
        request = exact_object(
            request, {"schema", "profile", "mode", "host_pid"}, "live process request"
        )
        profile = request["profile"]
        geometry = PROFILE_GEOMETRY_V2.get(profile) if type(profile) is str else None
        if (
            geometry is None
            or plan.get("profile") != profile
            or plan.get("chunk_tokens") != geometry["chunk_tokens"]
            or plan.get("in_container_recipe_path")
            != "/opt/flux/" + geometry["recipe_relpath"]
        ):
            raise ValueError("live process request profile differs from its recipe")
    else:
        request = exact_object(
            request, {"schema", "mode", "host_pid"}, "live process request"
        )
        if request["schema"] != "flux-r15-live-process-request/1":
            raise ValueError("live process request schema or mode differs")
    if request["mode"] not in ("none", "native-mtp-k5"):
        raise ValueError("live process request schema or mode differs")
    pid = exact_int(request["host_pid"], "live server host PID", 1, 1 << 31)
    result = subprocess.run(
        [
            "docker",
            "container",
            "inspect",
            "--format",
            "{{json .}}",
            identities["container_id"],
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        raise ValueError("live container inspection failed")
    document = strict_json_loads(result.stdout, "live container inspection")
    state = document.get("State") if type(document) is dict else None
    if (
        not isinstance(state, dict)
        or document.get("Id") != identities["container_id"]
        or document.get("Image") != identities["image_id"]
        or state.get("Running") is not True
    ):
        raise ValueError("live server container identity differs")
    init_pid = exact_int(state.get("Pid"), "container init host PID", 1, 1 << 31)
    pid_fd = os.pidfd_open(pid)
    root_fd = exe_fd = recipe_fd = binary_fd = models_fd = source_fd = None
    try:
        proc, init_proc = PROC_ROOT / str(pid), PROC_ROOT / str(init_pid)
        start = _process_start_ticks(pid)
        namespace = (proc / "ns/mnt").stat().st_ino
        if namespace != (init_proc / "ns/mnt").stat().st_ino:
            raise ValueError("live server mount namespace differs from container")
        cgroup = (proc / "cgroup").read_bytes()
        if (
            cgroup != (init_proc / "cgroup").read_bytes()
            or re.search(
                rb"(?<![0-9a-f])"
                + identities["container_id"].encode()
                + rb"(?![0-9a-f])",
                cgroup,
            )
            is None
        ):
            raise ValueError("live server cgroup does not identify the exact container")
        # /proc's root/exe are trusted kernel magic links. Every component
        # inside the container root must be opened without following links.
        root_fd = os.open(proc / "root", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        root_info = os.fstat(root_fd)
        binary_fd = _open_process_file(root_fd, plan["in_container_binary_path"])
        recipe_fd = _open_process_file(root_fd, plan["in_container_recipe_path"])
        exe_fd = os.open(proc / "exe", os.O_RDONLY | os.O_CLOEXEC)
        exe_info, binary_info = os.fstat(exe_fd), os.fstat(binary_fd)
        if (exe_info.st_dev, exe_info.st_ino) != (
            binary_info.st_dev,
            binary_info.st_ino,
        ):
            raise ValueError("live server is not executing the release binary")
        executable_sha, _ = _hash_regular_fd(
            exe_fd, "live executable", 2 * 1024 * 1024 * 1024
        )
        recipe_sha, recipe = _hash_regular_fd(
            recipe_fd, "live recipe", MAX_PROGRAM_BYTES
        )
        if (
            executable_sha != plan["in_container_binary_sha256"]
            or recipe_sha != plan["recipe_sha256"]
        ):
            raise ValueError("live executable or recipe hash differs")
        mountinfo = (proc / "mountinfo").read_text("utf-8")
        _require_readonly_paths(
            mountinfo,
            [
                plan["in_container_binary_path"],
                plan["in_container_recipe_path"],
                "/models",
            ],
        )
        model_source = _checkpoint_mount_source(document, mountinfo)
        source_fd = open_directory_chain(model_source, "live checkpoint source")
        models_fd = _open_process_file(root_fd, "/models")
        source_info, models_info = os.fstat(source_fd), os.fstat(models_fd)
        model_identity = (models_info.st_dev, models_info.st_ino)
        if not stat.S_ISDIR(models_info.st_mode) or model_identity != (
            source_info.st_dev,
            source_info.st_ino,
        ):
            raise ValueError(
                "live checkpoint directory differs from the pinned /models bind"
            )
        environ_raw = _read_process_bytes(
            proc / "environ", "environment", 4 * 1024 * 1024
        )
        environment = {}
        for item in environ_raw.split(b"\0"):
            if not item:
                continue
            key, value = item.split(b"=", 1)
            if key in environment:
                raise ValueError("live environment contains duplicate keys")
            environment[key] = value
        expected = _recipe_environment(recipe)
        mode_keys = {
            b"FLUX_SPEC_MODE",
            b"FLUX_DISABLE_SPECULATION",
            b"ATLAS_DISABLE_SPECULATION",
        }
        for key, value in expected.items():
            if request["mode"] == "none" and key in mode_keys:
                continue
            if environment.get(key) != value:
                raise ValueError("live environment differs from the release recipe")
        argv_raw = _read_process_bytes(
            proc / "cmdline", "command line", 4 * 1024 * 1024
        )
        argv = argv_raw.rstrip(b"\0").split(b"\0")
        if _live_cli_values(argv, b"--model-from-path") != [b"/models"]:
            raise ValueError(
                "live server must select the pinned --model-from-path /models"
            )
        _verify_live_mode(argv, environment, request["mode"])
        if native_gate is not None:
            validate_native_context_live_geometry(plan, argv, environment)
        if profile is not None:
            chunk = str(plan["chunk_tokens"]).encode()
            for key in (b"ATLAS_QWEN4_PREFILL_WIDE", b"FLUX_QWEN4_PREFILL_WIDE"):
                if environment.get(key) != chunk:
                    raise ValueError("live WIDE differs from the profile geometry")
            if _live_cli_values(argv, b"--max-prefill-tokens") != [chunk]:
                raise ValueError("live scheduler chunk differs from the profile geometry")
            # Arena capacity is not part of either reviewed recipe.
            if any(
                key in environment
                for key in (b"ATLAS_MAX_BATCH_TOKENS", b"FLUX_MAX_BATCH_TOKENS")
            ):
                raise ValueError("live arena capacity is not bound by the profile")
        listener = _live_listener_identity(proc, init_proc, document, identities, argv)
        final_exe = (proc / "exe").stat()
        final_root = (proc / "root").stat()
        if (final_exe.st_dev, final_exe.st_ino) != (
            exe_info.st_dev,
            exe_info.st_ino,
        ) or (final_root.st_dev, final_root.st_ino) != (
            root_info.st_dev,
            root_info.st_ino,
        ):
            raise ValueError(
                "live server executable or root changed during verification"
            )
        if (
            _process_start_ticks(pid) != start
            or select.select([pid_fd], [], [], 0)[0]
            or (proc / "mountinfo").read_text("utf-8") != mountinfo
            or (proc / "ns/mnt").stat().st_ino != namespace
            or (proc / "cgroup").read_bytes() != cgroup
            or _read_process_bytes(proc / "cmdline", "command line", 4 * 1024 * 1024)
            != argv_raw
            or _read_process_bytes(proc / "environ", "environment", 4 * 1024 * 1024)
            != environ_raw
            or _live_listener_identity(proc, init_proc, document, identities, argv)
            != listener
        ):
            raise ValueError("live server identity changed during verification")
        for checkpoint_path in (None, model_source):
            descriptor = (
                _open_process_file(root_fd, "/models")
                if checkpoint_path is None
                else open_directory_chain(checkpoint_path, "live checkpoint source")
            )
            try:
                final_model = os.fstat(descriptor)
                if (final_model.st_dev, final_model.st_ino) != model_identity:
                    raise ValueError(
                        "live checkpoint directory changed during verification"
                    )
            finally:
                os.close(descriptor)
        proof = {
            "schema": "flux-r15-live-process-proof/1",
            "status": "PASS",
            "mode": request["mode"],
            "container_id": identities["container_id"],
            "run_id": identities["run_id"],
            "host_pid": pid,
            "start_time_ticks": start,
            "executable_sha256": executable_sha,
            "recipe_sha256": recipe_sha,
            "recipe_environment_verified": True,
            "mount_namespace_match": True,
            "cgroup_match": True,
            "paths_read_only": True,
        }
        if profile is not None:
            proof["schema"] = LIVE_PROOF_V2
            proof["profile"] = profile
        if native_gate is not None:
            proof.update(schema=NATIVE_CONTEXT_PROOF, gate=native_gate,
                         chunk_tokens=plan["chunk_tokens"],
                         native_context_contract_sha256=sha256_bytes(canonical_bytes(plan["native_context_contract"])))
        return proof
    finally:
        for descriptor in (
            source_fd,
            models_fd,
            binary_fd,
            recipe_fd,
            exe_fd,
            root_fd,
            pid_fd,
        ):
            if descriptor is not None:
                os.close(descriptor)


def validate_gate_receipt(raw: bytes, gate: str, identities: Mapping[str, str]) -> None:
    receipt = exact_object(
        strict_json_loads(raw, f"gate {gate} receipt"),
        {
            "schema",
            "gate",
            "status",
            "release_qualified",
            "execution_scope",
            "image_id",
            "image_digest",
            "container_id",
            "run_id",
            "evidence",
        },
        f"gate {gate} receipt",
    )
    expected = {
        "schema": GATE_SCHEMA,
        "gate": gate,
        "status": "PASS",
        "release_qualified": True,
        "execution_scope": "immutable-container",
        "image_id": identities["image_id"],
        "image_digest": identities["image_digest"],
        "container_id": identities["container_id"],
        "run_id": identities["run_id"],
    }
    for key, wanted in expected.items():
        if receipt[key] != wanted or type(receipt[key]) is not type(wanted):
            raise ValueError(f"gate {gate} {key} differs from launch identity")
    if type(receipt["evidence"]) is not dict or not receipt["evidence"]:
        raise ValueError(f"gate {gate} evidence is empty")


def append_log_frame(raw_log_fd: int, gate: str, stream: str, payload: bytes) -> None:
    header = canonical_bytes({"gate": gate, "stream": stream, "bytes": len(payload)})
    write_all(raw_log_fd, b"\nR15_LOG_FRAME " + header + b"\n")
    write_all(raw_log_fd, payload)


def scan_raw_log(descriptor: int) -> dict[str, Any]:
    os.fsync(descriptor)
    before = os.fstat(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size <= 0
        or before.st_size > MAX_RAW_LOG_BYTES
    ):
        raise ValueError("complete raw log must be one non-empty bounded file")
    digest = hashlib.sha256()
    per_label = {item: 0 for item in RAW_LOG_PATTERNS}
    matches = 0
    # One mapping preserves arbitrarily long separators and word boundaries
    # across every read boundary without copying the complete log into RAM.
    with mmap.mmap(descriptor, 0, access=mmap.ACCESS_READ) as window:
        digest.update(window)
        for label, pattern in _RAW_LOG_COMPILED:
            for _match in pattern.finditer(window):
                if matches >= MAX_PATTERN_MATCH_COUNT:
                    raise ValueError("complete raw log census exceeded its bound")
                per_label[label] += 1
                matches += 1
    after = os.fstat(descriptor)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
        before.st_nlink,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
        after.st_nlink,
    ):
        raise ValueError("complete raw log changed while it was scanned")
    return {
        "scan_scope": "complete-uncapped",
        "patterns": list(RAW_LOG_PATTERNS),
        "match_count": matches,
        "per_pattern_match_counts": per_label,
        "raw_log_bytes": after.st_size,
        "raw_log_sha256": digest.hexdigest(),
    }


def execute_suite(
    plan: dict[str, Any], identities: Mapping[str, str]
) -> dict[str, Any]:
    runtime = Path(plan["runtime_dir"])
    if (
        directory_identity_sha256(runtime, "suite runtime_dir before execution")
        != plan["_runtime_dir_identity_sha256"]
    ):
        raise ValueError("suite runtime_dir identity changed before execution")
    raw_locator = plan["raw_log_relpath"]
    raw_log_fd = create_beneath(runtime, raw_locator, "complete raw log", 0o400)
    raw_log_read_fd = readonly_duplicate(raw_log_fd, "complete raw log")
    gate_results: dict[str, dict[str, Any]] = {}
    process_bindings: list[dict[str, Any]] = []
    try:
        for gate in REQUIRED_GATES:
            job = plan["gate_jobs"][gate]
            receipt_artifact_fd = create_beneath(
                runtime, job["receipt_relpath"], f"gate {gate} receipt", 0o400
            )
            stdout_artifact_fd: int | None = None
            stderr_artifact_fd: int | None = None
            receipt_fd: int | None = None
            stdout_fd: int | None = None
            stderr_fd: int | None = None
            try:
                stdout_artifact_fd = create_beneath(
                    runtime, job["stdout_relpath"], f"gate {gate} stdout", 0o400
                )
                stderr_artifact_fd = create_beneath(
                    runtime, job["stderr_relpath"], f"gate {gate} stderr", 0o400
                )
                receipt_fd = anonymous_writable_file(f"gate-{gate}-receipt")
                stdout_fd = anonymous_writable_file(f"gate-{gate}-stdout")
                stderr_fd = anonymous_writable_file(f"gate-{gate}-stderr")
                if gate == "raw-log-safety":
                    os.fsync(raw_log_fd)
                with sealed_program(
                    Path(job["program_path"]),
                    job["program_sha256"],
                    f"gate-{gate}",
                ) as program_fd:
                    argv = substitute_argv(
                        job["argv"], identities, receipt_fd, raw_log_read_fd
                    )
                    proofs = run_gate_program(
                        argv,
                        executable_fd=program_fd,
                        receipt_fd=receipt_fd,
                        raw_log_fd=raw_log_read_fd,
                        stdout_fd=stdout_fd,
                        stderr_fd=stderr_fd,
                        timeout_seconds=job["timeout_seconds"],
                        process_proof=(
                            (plan, identities) if gate == "startup-readiness" else None
                        ),
                    )
                    process_bindings.extend(proofs)
                receipt_raw = seal_written_file(
                    receipt_fd, f"gate {gate} receipt", MAX_GATE_RECEIPT_BYTES
                )
                stdout_raw = seal_written_file(
                    stdout_fd, f"gate {gate} stdout", MAX_PROGRAM_OUTPUT_BYTES
                )
                stderr_raw = seal_written_file(
                    stderr_fd, f"gate {gate} stderr", MAX_PROGRAM_OUTPUT_BYTES
                )
                persist_reserved_file(receipt_artifact_fd, receipt_raw)
                persist_reserved_file(stdout_artifact_fd, stdout_raw)
                persist_reserved_file(stderr_artifact_fd, stderr_raw)
                assert_path_identity(
                    runtime,
                    job["receipt_relpath"],
                    receipt_artifact_fd,
                    f"gate {gate} receipt",
                )
                assert_path_identity(
                    runtime,
                    job["stdout_relpath"],
                    stdout_artifact_fd,
                    f"gate {gate} stdout",
                )
                assert_path_identity(
                    runtime,
                    job["stderr_relpath"],
                    stderr_artifact_fd,
                    f"gate {gate} stderr",
                )
                validate_gate_receipt(receipt_raw, gate, identities)
                if gate == "raw-log-safety":
                    if stdout_raw or stderr_raw:
                        raise ValueError(
                            "raw-log-safety driver must be silent after its census"
                        )
                else:
                    append_log_frame(raw_log_fd, gate, "stdout", stdout_raw)
                    append_log_frame(raw_log_fd, gate, "stderr", stderr_raw)
                gate_results[gate] = {
                    "status": "PASS",
                    "receipt_relpath": job["receipt_relpath"],
                    "receipt_sha256": sha256_bytes(receipt_raw),
                }
            finally:
                if stderr_fd is not None:
                    os.close(stderr_fd)
                if stdout_fd is not None:
                    os.close(stdout_fd)
                if receipt_fd is not None:
                    os.close(receipt_fd)
                if stderr_artifact_fd is not None:
                    os.close(stderr_artifact_fd)
                if stdout_artifact_fd is not None:
                    os.close(stdout_artifact_fd)
                os.close(receipt_artifact_fd)
        os.fsync(raw_log_fd)
        os.fchmod(raw_log_fd, 0o400)
        assert_path_identity(runtime, raw_locator, raw_log_fd, "complete raw log")
        safety = scan_raw_log(raw_log_fd)
        if safety["match_count"] != 0:
            raise ValueError("complete raw log contains a forbidden marker")
        if (
            directory_identity_sha256(runtime, "suite runtime_dir after execution")
            != plan["_runtime_dir_identity_sha256"]
        ):
            raise ValueError("suite runtime_dir identity changed during execution")
        raw_gate_path = plan["gate_jobs"]["raw-log-safety"]["receipt_relpath"]
        raw_gate_fd = open_beneath(
            runtime,
            raw_gate_path,
            "raw-log-safety receipt",
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        try:
            raw_gate = strict_json_loads(
                read_fd(
                    raw_gate_fd,
                    "raw-log-safety receipt",
                    MAX_GATE_RECEIPT_BYTES,
                    nlink=1,
                ),
                "raw-log-safety receipt",
            )
        finally:
            os.close(raw_gate_fd)
        evidence = raw_gate["evidence"]
        for key in (
            "scan_scope",
            "patterns",
            "match_count",
            "raw_log_bytes",
            "raw_log_sha256",
        ):
            if evidence.get(key) != safety[key] or type(evidence.get(key)) is not type(
                safety[key]
            ):
                raise ValueError("raw-log-safety receipt differs from the final log")
        return {
            "schema": SUITE_SCHEMA,
            "status": "PASS",
            "release_qualified": True,
            "release_version": plan["release_version"],
            "execution_scope": "immutable-container",
            "image_id": identities["image_id"],
            "image_digest": identities["image_digest"],
            "container_id": identities["container_id"],
            "run_id": identities["run_id"],
            "source_revision": plan["source_revision"],
            "source_tree": plan["source_tree"],
            "source_bundle_manifest_sha256": plan["source_bundle_manifest_sha256"],
            "in_container_binary_sha256": plan["in_container_binary_sha256"],
            "recipe_sha256": plan["recipe_sha256"],
            "checkpoint_id": plan["checkpoint_id"],
            "checkpoint_revision": plan["checkpoint_revision"],
            "checkpoint_manifest_sha256": plan["checkpoint_manifest_sha256"],
            "suite_plan_sha256": plan["_raw_sha256"],
            "runtime_dir_identity_sha256": plan["_runtime_dir_identity_sha256"],
            "gates": gate_results,
            "process_bindings": process_bindings,
            "raw_log_safety": {
                **safety,
                "raw_log_relpath": raw_locator,
            },
        }
    finally:
        os.close(raw_log_read_fd)
        os.close(raw_log_fd)


def validate_inherited_receipt_fd(descriptor: int) -> None:
    info = os.fstat(descriptor)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_nlink not in (0, 1)
        or info.st_size != 0
    ):
        raise ValueError(
            "suite receipt descriptor must be one empty regular file or memfd"
        )
    if fcntl.fcntl(descriptor, fcntl.F_GETFD) & fcntl.FD_CLOEXEC:
        raise ValueError("suite receipt descriptor was not inherited correctly")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container-id", required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--plan-fd", required=True, type=int)
    parser.add_argument("--receipt-fd", required=True, type=int)
    args = parser.parse_args(argv)

    if CID_RE.fullmatch(args.container_id) is None:
        raise ValueError("container ID must be full lowercase hexadecimal")
    exact_hex(args.image_id, 64, "image ID", prefixed=True)
    exact_hex(args.image_digest, 64, "image digest", prefixed=True)
    if SAFE_RUN_ID.fullmatch(args.run_id) is None:
        raise ValueError("run ID is unsafe")
    identities = {
        "container_id": args.container_id,
        "image_id": args.image_id,
        "image_digest": args.image_digest,
        "base_url": validate_base_url(args.base_url),
        "run_id": args.run_id,
    }
    if args.plan_fd == args.receipt_fd or min(args.plan_fd, args.receipt_fd) < 3:
        raise ValueError("plan and receipt descriptors must be distinct inherited FDs")
    validate_inherited_receipt_fd(args.receipt_fd)
    plan = load_plan(args.plan_fd)
    receipt = execute_suite(plan, identities)
    payload = canonical_bytes(receipt)
    write_all(args.receipt_fd, payload)
    os.fsync(args.receipt_fd)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as error:
        print(f"R15 suite HOLD: {type(error).__name__}: {error}", file=sys.stderr)
        raise
