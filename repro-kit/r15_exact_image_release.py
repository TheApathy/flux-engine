#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""One-command, fail-closed R15 immutable-container qualification wrapper.

This wrapper deliberately does not score model output itself.  It composes the
reviewed lifecycle-v3 Docker adapter with one source-pinned qualification
program, validates that program's complete release-suite receipt, and seals an
outer receipt only after exact-container cleanup has proved the release flock
safe to unlock.

The live path requires Docker and a GPU-capable qualification program.  Import,
manifest validation, and ``--dry-run`` do not contact Docker, launch a server,
or use a GPU.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import secrets
import shlex
import stat
import subprocess
import sys
import tempfile
import types
from typing import Any, Callable, Mapping, Protocol, Sequence
import urllib.parse

_KIT_DIR = str(Path(__file__).resolve().parent)
if _KIT_DIR not in sys.path:
    sys.path.insert(0, _KIT_DIR)

import container_lifecycle_v3 as lifecycle  # noqa: E402
import qwen4_native_262k_qualify as native262k_gate  # noqa: E402
import r12_container_release_runner as r12  # noqa: E402
import r9_source_to_binary as provenance_gate  # noqa: E402

MANIFEST_SCHEMA = "flux-r15-exact-image-launch/2"
SUITE_SCHEMA = "flux-r15-container-suite-receipt/2"
GATE_SCHEMA = "flux-r15-container-gate-receipt/1"
RECEIPT_SCHEMA = "flux-r15-exact-image-release-receipt/1"
# Compatibility defaults only; live identity is bound by the launch manifest.
RELEASE_VERSION = "v0.9.13-r15"
OCI_VERSION = RELEASE_VERSION.removeprefix("v")
MAX_SUITE_RECEIPT_BYTES = 32 * 1024 * 1024
MAX_SUITE_PLAN_BYTES = 4 * 1024 * 1024
MAX_PROGRAM_OUTPUT_BYTES = 64 * 1024 * 1024
MAX_GATE_RECEIPT_BYTES = 256 * 1024 * 1024
MAX_RAW_LOG_BYTES = 8 * 1024 * 1024 * 1024
MAX_IMAGE_BINARY_BYTES = 2 * 1024 * 1024 * 1024
MAX_IMAGE_RECIPE_BYTES = 16 * 1024 * 1024
PERFORMANCE_RECIPE_PATHS = {
    "interactive": "/opt/flux/repro-kit/recipes/flux-v9-w4096-interactive.env",
    "high-prefill": "/opt/flux/repro-kit/recipes/flux-v9-max-prefill.env",
}

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
# Mirror of the suite census: the outer recensus must agree with the suite and
# hostile logs do not own a fixed case ("cuda fatal", "NVRM", "nvrm").
_RAW_LOG_REGEXES = (
    ("CUDA fatal", re.compile(rb"cuda[-_: ]*fatal", re.IGNORECASE | re.ASCII)),
    (
        "CUDA out of memory",
        re.compile(rb"cuda[-_: ]*out[-_: ]*of[-_: ]*memory", re.IGNORECASE | re.ASCII),
    ),
    (
        "illegal memory access",
        re.compile(rb"illegal[-_: ]*memory[-_: ]*access", re.IGNORECASE | re.ASCII),
    ),
    ("NVRM", re.compile(rb"\bNVRM\b", re.IGNORECASE | re.ASCII)),
    ("Xid", re.compile(rb"\bXid\b", re.IGNORECASE | re.ASCII)),
)
assert tuple(label for label, _ in _RAW_LOG_REGEXES) == RAW_LOG_PATTERNS
PLACEHOLDERS = (
    "@CONTAINER_ID@",
    "@IMAGE_ID@",
    "@IMAGE_DIGEST@",
    "@BASE_URL@",
    "@RUN_ID@",
    "@PLAN_FD@",
    "@RECEIPT_FD@",
)

ALLOWED_MOUNT_DESTINATIONS = frozenset({"/models", "/results"})
FORBIDDEN_EXACT_IMAGE_RUN_OPTIONS = frozenset({"--device", "--env", "-e"})
BROAD_MOUNT_SOURCES = frozenset(
    {"/", "/home", "/mnt", "/models", "/results", "/srv", "/tmp", "/var"}
)

HEX_40 = re.compile(r"[0-9a-f]{40}")
HEX_64 = re.compile(r"[0-9a-f]{64}")
SAFE_NAME = re.compile(r"[a-z0-9][a-z0-9_.-]{0,62}")
SAFE_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}")
SAFE_ARTIFACT_LOCATOR = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,255}")
SAFE_MODEL_ID = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}/[A-Za-z0-9][A-Za-z0-9_.-]{0,191}"
)


class SuiteBackend(Protocol):
    def run(
        self,
        argv: Sequence[str],
        *,
        timeout_s: int,
        stdout_fd: int,
        stderr_fd: int,
        env: Mapping[str, str],
        executable_fd: int,
        plan_fd: int,
        receipt_fd: int,
    ) -> int: ...


class SubprocessSuiteBackend:
    def run(
        self,
        argv: Sequence[str],
        *,
        timeout_s: int,
        stdout_fd: int,
        stderr_fd: int,
        env: Mapping[str, str],
        executable_fd: int,
        plan_fd: int,
        receipt_fd: int,
    ) -> int:
        with r12.owned_child_processes():
            completed = subprocess.run(
                list(argv),
                executable=f"/proc/self/fd/{executable_fd}",
                stdin=subprocess.DEVNULL,
                stdout=stdout_fd,
                stderr=stderr_fd,
                env=dict(env),
                pass_fds=(executable_fd, plan_fd, receipt_fd),
                timeout=timeout_s,
                check=False,
            )
        return completed.returncode


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _parse_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number {value!r} is forbidden")
    return parsed


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def strict_json_loads(raw: bytes | str, label: str) -> Any:
    try:
        return json.loads(
            raw,
            parse_constant=_reject_constant,
            parse_float=_parse_float,
            object_pairs_hook=_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is not strict JSON: {error}") from error


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("short write")
        offset += written


def _read_fd_bytes(
    descriptor: int,
    label: str,
    max_bytes: int,
    *,
    required_nlink: int | None = 1,
) -> bytes:
    info = os.fstat(descriptor)
    if (
        not stat.S_ISREG(info.st_mode)
        or (required_nlink is not None and info.st_nlink != required_nlink)
        or info.st_size > max_bytes
    ):
        raise ValueError(f"{label} must be one bounded regular file")
    chunks: list[bytes] = []
    offset = 0
    remaining = info.st_size
    while remaining:
        chunk = os.pread(descriptor, min(1024 * 1024, remaining), offset)
        if not chunk:
            raise ValueError(f"{label} ended before its declared size")
        chunks.append(chunk)
        offset += len(chunk)
        remaining -= len(chunk)
    after = os.fstat(descriptor)
    if (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
        after.st_nlink,
    ) != (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
        info.st_nlink,
    ):
        raise ValueError(f"{label} changed while it was read")
    return b"".join(chunks)


def _open_stable(path: Path, label: str, max_bytes: int) -> tuple[int, bytes]:
    if (
        not path.is_absolute()
        or ".." in path.parts
        or str(Path(str(path))) != str(path)
        or not path.name
    ):
        raise ValueError(f"{label} must use one normalized absolute path")
    parent_fd = _open_directory_chain(path.parent, f"{label} parent")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        descriptor = os.open(path.name, flags, dir_fd=parent_fd)
    except OSError as error:
        os.close(parent_fd)
        raise ValueError(f"{label} is unavailable") from error
    try:
        payload = _read_fd_bytes(descriptor, label, max_bytes)
        path_info = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        fd_info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(path_info.st_mode)
            or path_info.st_nlink != 1
            or (path_info.st_dev, path_info.st_ino) != (fd_info.st_dev, fd_info.st_ino)
        ):
            raise ValueError(f"{label} pathname identity changed")
        return descriptor, payload
    except BaseException:
        os.close(descriptor)
        raise
    finally:
        os.close(parent_fd)


@contextmanager
def _sealed_executable(path: Path, expected_sha256: str):
    source_fd, payload = _open_stable(path, "suite program", 16 * 1024 * 1024)
    try:
        if sha256_bytes(payload) != expected_sha256:
            raise ValueError("suite program SHA-256 mismatch")
        flags = os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING
        executable_fd = os.memfd_create("flux-r15-suite", flags)
        try:
            _write_all(executable_fd, payload)
            os.fchmod(executable_fd, 0o500)
            seals = (
                fcntl.F_SEAL_SHRINK
                | fcntl.F_SEAL_GROW
                | fcntl.F_SEAL_WRITE
                | fcntl.F_SEAL_SEAL
            )
            fcntl.fcntl(executable_fd, fcntl.F_ADD_SEALS, seals)
            if fcntl.fcntl(executable_fd, fcntl.F_GET_SEALS) != seals:
                raise RuntimeError("suite program memfd did not retain every seal")
            if (
                sha256_bytes(
                    _read_fd_bytes(
                        executable_fd,
                        "sealed suite program",
                        16 * 1024 * 1024,
                        required_nlink=None,
                    )
                )
                != expected_sha256
            ):
                raise ValueError("sealed suite program SHA-256 mismatch")
            yield executable_fd
        finally:
            os.close(executable_fd)
    finally:
        os.close(source_fd)


@contextmanager
def _sealed_data_file(path: Path, expected_sha256: str, label: str, max_bytes: int):
    source_fd, payload = _open_stable(path, label, max_bytes)
    try:
        if sha256_bytes(payload) != expected_sha256:
            raise ValueError(f"{label} SHA-256 mismatch")
        descriptor = os.memfd_create(label, os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
        try:
            _write_all(descriptor, payload)
            os.fchmod(descriptor, 0o400)
            seals = (
                fcntl.F_SEAL_SHRINK
                | fcntl.F_SEAL_GROW
                | fcntl.F_SEAL_WRITE
                | fcntl.F_SEAL_SEAL
            )
            fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, seals)
            if fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) != seals:
                raise RuntimeError(f"{label} memfd did not retain every seal")
            if (
                sha256_bytes(
                    _read_fd_bytes(descriptor, label, max_bytes, required_nlink=None)
                )
                != expected_sha256
            ):
                raise ValueError(f"sealed {label} SHA-256 mismatch")
            yield descriptor
        finally:
            os.close(descriptor)
    finally:
        os.close(source_fd)


def _anonymous_writable_file(label: str) -> int:
    descriptor = os.memfd_create(label, os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    os.fchmod(descriptor, 0o600)
    return descriptor


def _seal_written_file(descriptor: int, label: str, max_bytes: int) -> bytes:
    os.fsync(descriptor)
    seals = (
        fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE | fcntl.F_SEAL_SEAL
    )
    fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, seals)
    if fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) != seals:
        raise RuntimeError(f"{label} did not retain every memfd seal")
    return _read_fd_bytes(descriptor, label, max_bytes, required_nlink=None)


def _persist_reserved_file(descriptor: int, payload: bytes) -> None:
    if os.fstat(descriptor).st_size != 0:
        raise ValueError("reserved release artifact was not empty")
    _write_all(descriptor, payload)
    os.fsync(descriptor)
    os.fchmod(descriptor, 0o400)


def _exact_object(value: object, keys: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{label} keys differ")
    return value


def _hex(value: object, length: int, label: str, *, prefixed: bool = False) -> str:
    if type(value) is not str:
        raise ValueError(f"{label} must be a string")
    raw = value
    if prefixed:
        if not raw.startswith("sha256:"):
            raise ValueError(f"{label} must have a sha256: prefix")
        raw = raw[7:]
    pattern = HEX_40 if length == 40 else HEX_64
    if pattern.fullmatch(raw) is None:
        raise ValueError(f"{label} must be lowercase hexadecimal length {length}")
    return value


def _plain_int(value: object, label: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{label} must be an exact integer in [{minimum}, {maximum}]")
    return value


def _normalized_absolute_path(value: object, label: str) -> Path:
    if type(value) is not str or not value or "\x00" in value:
        raise ValueError(f"{label} must be a non-empty NUL-free path string")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError(f"{label} must be one normalized absolute path")
    return path


def _open_directory_chain(path: Path, label: str) -> int:
    """Open every absolute path component without following a symlink.

    Holding each parent descriptor while opening its child removes the usual
    ``resolve()`` then ``open()`` race.  Release qualification is Linux-only;
    fail closed if the required descriptor flags are unavailable.
    """

    if (
        not path.is_absolute()
        or ".." in path.parts
        or str(Path(str(path))) != str(path)
    ):
        raise ValueError(f"{label} must be one normalized absolute path")
    required = ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW")
    if any(not hasattr(os, name) for name in required):
        raise RuntimeError("no-follow directory traversal is unavailable")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        descriptor = os.open("/", flags)
    except OSError as error:  # pragma: no cover - unusable Linux runtime
        raise ValueError(f"{label} root is unavailable") from error
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
        info = os.fstat(descriptor)
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"{label} must resolve to a directory")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _directory_identity(path: Path, label: str) -> tuple[int, int]:
    descriptor = _open_directory_chain(path, label)
    try:
        info = os.fstat(descriptor)
        return info.st_dev, info.st_ino
    finally:
        os.close(descriptor)


def _directory_identity_sha256(path: Path, label: str) -> str:
    return sha256_bytes(canonical_bytes(list(_directory_identity(path, label))))


def _safe_artifact_locator(value: object, label: str) -> str:
    if type(value) is not str or SAFE_ARTIFACT_LOCATOR.fullmatch(value) is None:
        raise ValueError(f"{label} must be one safe runtime-relative locator")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"{label} must stay beneath runtime_dir")
    if path.as_posix() != value:
        raise ValueError(f"{label} must be normalized")
    return value


def _runtime_relative(root: Path, path: Path, label: str) -> str:
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"{label} must stay beneath runtime_dir") from error
    return _safe_artifact_locator(relative, label)


def _open_beneath(root: Path, locator: object, label: str) -> int:
    """Open a runtime artifact through held, no-follow directory descriptors."""

    relative = PurePosixPath(_safe_artifact_locator(locator, label))
    parent_fd = _open_directory_chain(root, "runtime_dir")
    try:
        directory_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
        for component in relative.parts[:-1]:
            try:
                child = os.open(component, directory_flags, dir_fd=parent_fd)
            except OSError as error:
                raise ValueError(
                    f"{label} contains an unavailable or symlinked parent"
                ) from error
            os.close(parent_fd)
            parent_fd = child
        try:
            descriptor = os.open(
                relative.parts[-1],
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
        except OSError as error:
            raise ValueError(f"{label} is unavailable") from error
        return descriptor
    finally:
        os.close(parent_fd)


def _stable_beneath_bytes(
    root: Path, locator: object, label: str, max_bytes: int
) -> bytes:
    descriptor = _open_beneath(root, locator, label)
    try:
        return _read_fd_bytes(descriptor, label, max_bytes)
    finally:
        os.close(descriptor)


def _create_beneath(root: Path, locator: object, label: str, mode: int) -> int:
    relative = PurePosixPath(_safe_artifact_locator(locator, label))
    parent_fd = _open_directory_chain(root, "runtime_dir")
    try:
        directory_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
        for component in relative.parts[:-1]:
            try:
                child = os.open(component, directory_flags, dir_fd=parent_fd)
            except OSError as error:
                raise ValueError(
                    f"{label} contains an unavailable or symlinked parent"
                ) from error
            os.close(parent_fd)
            parent_fd = child
        try:
            return os.open(
                relative.parts[-1],
                os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                mode,
                dir_fd=parent_fd,
            )
        except FileExistsError:
            raise FileExistsError(f"refusing pre-existing {label}") from None
        except OSError as error:
            raise ValueError(f"cannot reserve {label}") from error
    finally:
        os.close(parent_fd)


def _assert_beneath_identity(
    root: Path, locator: str, descriptor: int, label: str
) -> None:
    reopened = _open_beneath(root, locator, label)
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


def _validate_mount_source(value: str, label: str, *, verify: bool) -> Path:
    source = _normalized_absolute_path(value, label)
    if str(source) in BROAD_MOUNT_SOURCES:
        raise ValueError(f"{label} is too broad for release qualification")
    if verify:
        _directory_identity(source, label)
    return source


def _within(root: Path, value: object, label: str) -> Path:
    path = _normalized_absolute_path(value, label)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} must stay inside runtime_dir") from error
    if path == root:
        raise ValueError(f"{label} must not equal runtime_dir")
    return path


def _require_disjoint_directories(
    first: Path, second: Path, first_label: str, second_label: str, *, verify: bool
) -> None:
    """Reject equal or nested release directories in either direction.

    Two release trees that share an ancestor relation can corrupt each other by
    one writing beneath the other's immutable namespace (the checkpoint must
    never be rewritable, and the runtime tree must never live on top of it).
    Each component is already opened without symlinks; with normalized paths
    and verified identities, a prefix relation is an exact containment.
    """

    if (
        first == second
        or _exactly_ancestor(first, second)
        or _exactly_ancestor(second, first)
    ):
        raise ValueError(
            f"{first_label} and {second_label} must be disjoint release trees"
        )
    if verify:
        if _directory_identity(first, first_label) == _directory_identity(
            second, second_label
        ):
            raise ValueError(f"{first_label} and {second_label} are the same directory")
        first_device, first_backing = _mount_backing_path(first)
        second_device, second_backing = _mount_backing_path(second)
        if first_device == second_device and (
            first_backing == second_backing
            or _exactly_ancestor(first_backing, second_backing)
            or _exactly_ancestor(second_backing, first_backing)
        ):
            raise ValueError("checkpoint/runtime bind aliases overlap")


def _mount_backing_path(path: Path) -> tuple[str, Path]:
    """Resolve bind aliases using mountinfo's filesystem root, not path text."""
    try:
        lines = Path("/proc/self/mountinfo").read_text("utf-8").splitlines()
    except OSError as error:
        raise ValueError("mount topology is unavailable") from error
    entries: list[tuple[Path, str, Path]] = []
    for line in lines:
        fields = line.split()
        if len(fields) < 10 or "-" not in fields[6:]:
            raise ValueError("mount topology is malformed")

        def decode(value: str) -> Path:
            value = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), value)
            return _normalized_absolute_path(value, "mount topology path")

        mountpoint = decode(fields[4])
        if _exactly_ancestor(path, mountpoint):
            raise ValueError("release directories must not contain nested mounts")
        if mountpoint == path or _exactly_ancestor(mountpoint, path):
            entries.append((mountpoint, fields[2], decode(fields[3])))
    if not entries:
        raise ValueError("release path has no mount topology mapping")
    length = max(len(mountpoint.parts) for mountpoint, _, _ in entries)
    selected = [entry for entry in entries if len(entry[0].parts) == length]
    if len(selected) != 1:
        raise ValueError("stacked mount aliases are unsupported")
    mountpoint, device, backing = selected[0]
    info = path.stat()
    if device != f"{os.major(info.st_dev)}:{os.minor(info.st_dev)}":
        raise ValueError("mount topology does not match directory device")
    return device, backing / path.relative_to(mountpoint)


def _exactly_ancestor(outer: Path, inner: Path) -> bool:
    if outer == inner:
        return False
    try:
        inner.relative_to(outer)
    except ValueError:
        return False
    return True


def _launch_relpath(
    manifest: dict[str, Any], value: str, label: str
) -> tuple[str, ...]:
    runtime_dir = manifest["_runtime_dir"]
    path = _within(Path(runtime_dir), value, label)
    relative = path.relative_to(runtime_dir)
    return relative.parts


def _open_nofollow_below(
    runtime_dir: Path,
    parts: Sequence[str],
    label: str,
    *,
    create: bool,
) -> int:
    """Open ``runtime_dir/parts...`` creating every trailing missing directory.

    Each component is stat'ed and opened with ``O_NOFOLLOW`` through a
    directory descriptor chain, so a hostile symlink swap at any depth fails
    closed instead of redirecting the release namespace.
    """

    for part in parts:
        if not part or part in (".", ".."):
            raise ValueError(f"{label} must contain only plain path components")
    if not parts:
        raise ValueError(f"{label} must have at least one path component")
    descriptor = _open_directory_chain(runtime_dir, "runtime_dir")
    try:
        for index, part in enumerate(parts):
            try:
                child_stat = os.stat(part, dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                if not create:
                    raise ValueError(f"{label} component '{part}' is missing") from None
                os.mkdir(part, dir_fd=descriptor)
                child_stat = os.stat(part, dir_fd=descriptor, follow_symlinks=False)
            if not stat.S_ISDIR(child_stat.st_mode):
                raise ValueError(f"{label} component '{part}' is not a directory")
            child = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _assert_target_not_symlink(parent_fd: int, name: str, label: str) -> None:
    try:
        target_stat = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(target_stat.st_mode):
        raise ValueError(f"{label} must not be a symlink")


def _assert_file_target_free(parent_fd: int, name: str, label: str) -> None:
    try:
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise ValueError(f"{label} must not pre-exist")


def _prepare_release_launch_paths(
    manifest: dict[str, Any], *, expect_cidfile_free: bool = True
) -> None:
    """Verify every launch namespace below runtime_dir without following links.

    The diagnostics directory and the lockfile parent chain are created (any
    missing component) through a no-follow descriptor walk; the cidfile target
    must be absent and all three targets must not be symlinks.  This runs
    before the runner touches docker, so a prepared namespace can never be
    redirected by a planted symlink gap deeper than the final component.
    The seal-time re-verification re-runs with ``expect_cidfile_free=False``
    because the stopped container legitimately keeps its cidfile on disk.
    """

    launch = manifest["launch"]
    runtime_dir = Path(manifest["_runtime_dir"])
    diagnostics_parts = _launch_relpath(
        manifest, launch["diagnostics_dir"], "launch diagnostics_dir"
    )
    diagnostics_fd = _open_nofollow_below(
        runtime_dir, diagnostics_parts, "launch diagnostics_dir", create=True
    )
    os.close(diagnostics_fd)
    for key, label, must_be_free, create_parents in (
        ("cidfile", "launch cidfile", expect_cidfile_free, False),
        ("lockfile", "launch lockfile", False, True),
    ):
        path = _within(Path(runtime_dir), launch[key], label)
        relative = path.relative_to(runtime_dir)
        parts = relative.parts
        if len(parts) < 1:
            raise ValueError(f"{label} must not equal runtime_dir")
        parent_parts, name = parts[:-1], parts[-1]
        parent_fd = (
            _open_nofollow_below(
                runtime_dir, parent_parts, f"{label} parent", create=create_parents
            )
            if parent_parts
            else os.open(
                runtime_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
            )
        )
        try:
            _assert_target_not_symlink(parent_fd, name, label)
            if must_be_free:
                _assert_file_target_free(parent_fd, name, label)
        finally:
            os.close(parent_fd)


def _release_diagnostic_capture(
    manifest: dict[str, Any], docker: r12.CommandBackend
) -> Callable[[lifecycle.ContainerRecord], None]:
    """Diagnostics writer that never follows a pre-planted symlink filename.

    The default r12 capture writes via ``Path.write_text`` after a
    ``mkdir(parents=True)``, which follows symlinks at every depth.  The exact
    image release owns capture: every component below runtime_dir is verified
    no-follow, each capture file is created O_EXCL|O_NOFOLLOW through the
    verified parent chain, and nothing writes outside that namespace.
    """

    runtime_dir = Path(manifest["_runtime_dir"])
    diagnostics_parts = list(
        _launch_relpath(
            manifest, manifest["launch"]["diagnostics_dir"], "launch diagnostics_dir"
        )
    )
    suffixes = (
        (".inspect.json", ("docker", "container", "inspect")),
        (".server.log", ("docker", "container", "logs", "--timestamps")),
    )
    held_parent_fd = _open_nofollow_below(
        runtime_dir, diagnostics_parts, "launch diagnostics_dir", create=False
    )

    def close() -> None:
        nonlocal held_parent_fd
        if held_parent_fd is not None:
            os.close(held_parent_fd)
            held_parent_fd = None

    def capture(record: lifecycle.ContainerRecord) -> None:
        if held_parent_fd is None:
            raise RuntimeError("diagnostic namespace has been closed")
        for suffix, argv in suffixes:
            result = docker.run(
                (*argv, record.container_id),
                timeout_s=r12.DEFAULT_COMMAND_TIMEOUT_S,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"diagnostic capture {suffix} rc={result.returncode}: "
                    f"{result.stderr!r}"
                )
            parent_fd = os.dup(held_parent_fd)
            try:
                name = f"{record.container_id}{suffix}"
                descriptor = os.open(
                    name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                    0o400,
                    dir_fd=parent_fd,
                )
                payload = result.stdout.encode("utf-8")
                try:
                    stat_result = os.fstat(descriptor)
                    if payload and not stat.S_ISREG(stat_result.st_mode):
                        raise ValueError("diagnostic capture target is not a file")
                    offset = 0
                    descriptor_size = 0
                    while offset < len(payload):
                        written = os.write(descriptor, payload[offset:])
                        if written <= 0:
                            raise OSError("short write during diagnostic capture")
                        offset += written
                        descriptor_size += written
                    os.fsync(descriptor)
                    os.fchmod(descriptor, 0o400)
                finally:
                    os.close(descriptor)
                reopened = os.open(
                    name,
                    os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                    dir_fd=parent_fd,
                )
                os.close(reopened)
                final_stat = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                if final_stat.st_nlink != 1 or final_stat.st_size != descriptor_size:
                    raise ValueError("diagnostic capture file changed identity")
            finally:
                os.close(parent_fd)

    capture.close = close
    return capture


def validate_base_url(value: object) -> str:
    if type(value) is not str:
        raise ValueError("launch.base_url must be a string")
    parsed = urllib.parse.urlsplit(value)
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("launch.base_url has an invalid port") from error
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
        raise ValueError("launch.base_url must be an unadorned loopback HTTP origin")
    return f"http://127.0.0.1:{port}"


def _container_path(value: str, label: str) -> str:
    path = PurePosixPath(value)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or str(path) != value
        or ":" in value
    ):
        raise ValueError(f"{label} must be one normalized absolute container path")
    return value


def _iter_run_options(values: Sequence[str]):
    index = 0
    while index < len(values):
        token = values[index]
        if token in r12.ALLOWED_FLAG_RUN_OPTIONS:
            yield token, None
            index += 1
            continue
        if "=" in token:
            option, option_value = token.split("=", 1)
            yield option, option_value
            index += 1
        else:
            yield token, values[index + 1]
            index += 2


def _validate_volume(
    value: str, label: str, *, verify_sources: bool
) -> tuple[str, Path]:
    parts = value.split(":")
    if len(parts) not in (2, 3) or not parts[0] or not parts[1]:
        raise ValueError(f"{label} must be an exact host:container[:options] bind")
    source = _validate_mount_source(parts[0], f"{label} source", verify=verify_sources)
    destination = _container_path(parts[1], f"{label} destination")
    if destination not in ALLOWED_MOUNT_DESTINATIONS:
        raise ValueError(f"{label} destination is not release-allowlisted")
    options = set(parts[2].split(",")) if len(parts) == 3 else set()
    expected_options = {"ro"} if destination == "/models" else {"rw"}
    if options != expected_options:
        raise ValueError(f"{label} does not have its exact release mount mode")
    return destination, source


def _validate_mount(
    value: str, label: str, *, verify_sources: bool
) -> tuple[str, Path]:
    fields: dict[str, str | None] = {}
    for item in value.split(","):
        if "=" in item:
            key, field_value = item.split("=", 1)
            if not key or not field_value or key in fields:
                raise ValueError(f"{label} contains invalid duplicate fields")
            fields[key] = field_value
        else:
            if not item or item in fields:
                raise ValueError(f"{label} contains invalid duplicate flags")
            fields[item] = None
    allowed = {
        "type",
        "src",
        "source",
        "dst",
        "target",
        "destination",
        "readonly",
        "ro",
    }
    if not set(fields) <= allowed or fields.get("type") != "bind":
        raise ValueError(f"{label} must be one exact bind mount")
    if any(fields.get(key) is not None for key in ("readonly", "ro") if key in fields):
        raise ValueError(f"{label} read-only markers must be value-less flags")
    sources = [fields[key] for key in ("src", "source") if key in fields]
    destinations = [
        fields[key] for key in ("dst", "target", "destination") if key in fields
    ]
    if len(sources) != 1 or len(destinations) != 1:
        raise ValueError(f"{label} must have one source and one destination")
    source = _validate_mount_source(
        str(sources[0]), f"{label} source", verify=verify_sources
    )
    destination = _container_path(str(destinations[0]), f"{label} destination")
    if destination not in ALLOWED_MOUNT_DESTINATIONS:
        raise ValueError(f"{label} destination is not release-allowlisted")
    read_only = "readonly" in fields or "ro" in fields
    if destination == "/models" and not read_only:
        raise ValueError("the /models release mount must be read-only")
    if destination == "/results" and read_only:
        raise ValueError("the /results release mount must be writable")
    return destination, source


def validate_exact_image_run_args(
    values: Sequence[str],
    *,
    checkpoint_path: Path | None = None,
    runtime_dir: Path | None = None,
    verify_sources: bool = True,
) -> list[str]:
    normalized = list(r12.DockerLifecycleAdapter._validate_extra_run_args(values))
    bindings: list[tuple[str, Path]] = []
    for option, option_value in _iter_run_options(normalized):
        if option in FORBIDDEN_EXACT_IMAGE_RUN_OPTIONS:
            raise ValueError(
                f"Docker run option is forbidden for exact images: {option}"
            )
        if option in {"--volume", "-v"}:
            assert option_value is not None
            bindings.append(
                _validate_volume(option_value, option, verify_sources=verify_sources)
            )
        elif option == "--mount":
            assert option_value is not None
            bindings.append(
                _validate_mount(option_value, option, verify_sources=verify_sources)
            )
    destinations = [destination for destination, _source in bindings]
    if len(destinations) != len(set(destinations)):
        raise ValueError("exact-image mount destinations must be unique")
    if set(destinations) != ALLOWED_MOUNT_DESTINATIONS:
        raise ValueError("exact-image launch requires exact /models and /results binds")
    if "--read-only" not in normalized:
        raise ValueError("exact-image launch requires a read-only container rootfs")
    if checkpoint_path is not None:
        models_source = dict(bindings)["/models"]
        checkpoint = _normalized_absolute_path(
            str(checkpoint_path), "provenance.checkpoint_path"
        )
        identity_differs = verify_sources and (
            _directory_identity(models_source, "/models bind source")
            != _directory_identity(checkpoint, "provenance.checkpoint_path")
        )
        if models_source != checkpoint or identity_differs:
            raise ValueError(
                "the /models bind source must equal the proven checkpoint path"
            )
    if runtime_dir is not None:
        results_source = dict(bindings)["/results"]
        runtime = _normalized_absolute_path(str(runtime_dir), "runtime_dir")
        identity_differs = verify_sources and (
            _directory_identity(results_source, "/results bind source")
            != _directory_identity(runtime, "runtime_dir")
        )
        if results_source != runtime or identity_differs:
            raise ValueError(
                "the /results bind source must equal the dedicated runtime_dir"
            )
    return normalized


def _validate_provenance_shape(value: object) -> dict[str, Any]:
    required = {
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
        "recipe_path",
        "recipe_sha256",
    }
    provenance = _exact_object(value, required, "provenance")
    provenance_gate.release_identity(provenance["release_version"])
    _hex(provenance["source_revision"], 40, "provenance.source_revision")
    _hex(provenance["source_tree"], 40, "provenance.source_tree")
    if (
        type(provenance["checkpoint_id"]) is not str
        or SAFE_MODEL_ID.fullmatch(provenance["checkpoint_id"]) is None
    ):
        raise ValueError("provenance.checkpoint_id must be one safe public model ID")
    revision = provenance["checkpoint_revision"]
    if type(revision) is not str or not (
        HEX_40.fullmatch(revision) or HEX_64.fullmatch(revision)
    ):
        raise ValueError("provenance.checkpoint_revision must be an immutable hash")
    for key in (
        "source_bundle_manifest_sha256",
        "source_bundle_sha256",
        "build_receipt_sha256",
        "source_to_binary_receipt_sha256",
        "binary_sha256",
        "checkpoint_manifest_sha256",
        "recipe_sha256",
    ):
        _hex(provenance[key], 64, f"provenance.{key}")
    for key in required:
        if key.endswith("_path") or key == "source_root_path":
            raw = provenance[key]
            if type(raw) is not str or not Path(raw).is_absolute():
                raise ValueError(f"provenance.{key} must be an absolute path")
    return provenance


def _image_performance_recipes(value: object) -> dict[str, Any]:
    bindings = _exact_object(
        value, set(PERFORMANCE_RECIPE_PATHS), "image.performance_recipes"
    )
    hashes = set()
    for profile, path in PERFORMANCE_RECIPE_PATHS.items():
        binding = _exact_object(
            bindings[profile], {"path", "sha256"}, f"image {profile} recipe"
        )
        if type(binding["path"]) is not str or binding["path"] != path:
            raise ValueError(f"image {profile} recipe path is not canonical")
        digest = _nonzero_hex(binding["sha256"], 64, f"image {profile} recipe hash")
        if digest in hashes:
            raise ValueError("image performance recipe bindings must not alias")
        hashes.add(digest)
    return bindings


def load_manifest(path: Path, *, verify_files: bool) -> dict[str, Any]:
    absolute_manifest = path if path.is_absolute() else Path.cwd() / path
    manifest_fd, raw = _open_stable(
        absolute_manifest, "launch manifest", 4 * 1024 * 1024
    )
    os.close(manifest_fd)
    value = strict_json_loads(raw, "launch manifest")
    raw_canonical_sha256 = sha256_bytes(canonical_bytes(value))
    manifest = _exact_object(
        value,
        {
            "schema",
            "release_version",
            "runtime_dir",
            "image",
            "provenance",
            "launch",
            "suite",
        },
        "launch manifest",
    )
    if manifest["schema"] != MANIFEST_SCHEMA:
        raise ValueError(f"manifest schema must equal {MANIFEST_SCHEMA}")
    provenance_gate.release_identity(manifest["release_version"])

    runtime_dir = _normalized_absolute_path(manifest["runtime_dir"], "runtime_dir")
    if str(runtime_dir) in BROAD_MOUNT_SOURCES:
        raise ValueError("runtime_dir is too broad for release qualification")
    runtime_identity: tuple[int, int] | None = None
    if verify_files:
        runtime_identity = _directory_identity(runtime_dir, "runtime_dir")

    image = _exact_object(
        manifest["image"],
        {
            "id",
            "digest",
            "rootfs_layers_sha256",
            "recipe_sha256",
            "in_container_binary_sha256",
            "in_container_binary_path",
            "in_container_recipe_path",
            "performance_recipes",
        },
        "image",
    )
    for key in ("id", "digest"):
        _hex(image[key], 64, f"image.{key}", prefixed=True)
    for key in (
        "rootfs_layers_sha256",
        "recipe_sha256",
        "in_container_binary_sha256",
    ):
        _hex(image[key], 64, f"image.{key}")
    performance_recipes = _image_performance_recipes(image["performance_recipes"])

    provenance = _validate_provenance_shape(manifest["provenance"])
    if provenance["release_version"] != manifest["release_version"]:
        raise ValueError("provenance.release_version differs from the launch manifest")
    # The proven checkpoint and the writable runtime tree must be disjoint
    # release trees: neither may live beneath the other, and both identities
    # must differ.  Flat lexical checks also run in dry-run mode.
    _require_disjoint_directories(
        _normalized_absolute_path(
            provenance["checkpoint_path"], "provenance.checkpoint_path"
        ),
        runtime_dir,
        "provenance.checkpoint_path",
        "runtime_dir",
        verify=verify_files,
    )
    if image["in_container_binary_sha256"] != provenance["binary_sha256"]:
        raise ValueError("image binary hash differs from source-to-binary provenance")
    if image["recipe_sha256"] != provenance["recipe_sha256"]:
        raise ValueError("image recipe hash differs from release provenance")
    binary_path = _container_path(
        image["in_container_binary_path"], "image.in_container_binary_path"
    )
    recipe_path = _container_path(
        image["in_container_recipe_path"], "image.in_container_recipe_path"
    )
    if not binary_path.startswith("/opt/flux/bin/"):
        raise ValueError("in-container binary must stay under /opt/flux/bin")
    if not recipe_path.startswith("/opt/flux/repro-kit/recipes/"):
        raise ValueError(
            "in-container recipe must stay under /opt/flux/repro-kit/recipes"
        )
    if verify_files:
        provenance_gate.configure_release(manifest["release_version"])
        provenance_gate.validate_qualification_provenance(provenance)
        recipe = Path(provenance["recipe_path"])
        recipe_fd, recipe_raw = _open_stable(
            recipe, "release recipe", MAX_IMAGE_RECIPE_BYTES
        )
        os.close(recipe_fd)
        if sha256_bytes(recipe_raw) != provenance["recipe_sha256"]:
            raise ValueError("release recipe SHA-256 mismatch")
        native262k_gate.validate_checkpoint_inventory(
            Path(provenance["checkpoint_manifest_path"]),
            Path(provenance["checkpoint_path"]),
            provenance,
            True,
        )
        checkpoint_identity = _directory_identity(
            Path(provenance["checkpoint_path"]), "provenance.checkpoint_path"
        )
    else:
        checkpoint_identity = None

    launch = _exact_object(
        manifest["launch"],
        {
            "name",
            "run_id",
            "cidfile",
            "lockfile",
            "diagnostics_dir",
            "base_url",
            "extra_run_args",
            "entrypoint",
            "container_command",
        },
        "launch",
    )
    if type(launch["name"]) is not str or SAFE_NAME.fullmatch(launch["name"]) is None:
        raise ValueError("launch.name is not an exact safe container name")
    if (
        type(launch["run_id"]) is not str
        or SAFE_RUN_ID.fullmatch(launch["run_id"]) is None
    ):
        raise ValueError("launch.run_id is not an exact safe run ID")
    launch["cidfile"] = str(_within(runtime_dir, launch["cidfile"], "launch.cidfile"))
    launch["lockfile"] = str(
        _within(runtime_dir, launch["lockfile"], "launch.lockfile")
    )
    launch["diagnostics_dir"] = str(
        _within(runtime_dir, launch["diagnostics_dir"], "launch.diagnostics_dir")
    )
    launch["base_url"] = validate_base_url(launch["base_url"])
    if type(launch["extra_run_args"]) is not list:
        raise ValueError("launch.extra_run_args must be an array")
    launch["extra_run_args"] = validate_exact_image_run_args(
        launch["extra_run_args"],
        checkpoint_path=Path(provenance["checkpoint_path"]),
        runtime_dir=runtime_dir,
        verify_sources=verify_files,
    )
    if launch["entrypoint"] != "/usr/bin/sleep":
        raise ValueError("launch.entrypoint must equal /usr/bin/sleep")
    command = launch["container_command"]
    if (
        type(command) is not list
        or not command
        or any(type(item) is not str or not item for item in command)
    ):
        raise ValueError("launch.container_command must be a non-empty string array")

    suite = _exact_object(
        manifest["suite"],
        {
            "program_path",
            "program_sha256",
            "plan_path",
            "plan_sha256",
            "argv",
            "timeout_seconds",
            "receipt_path",
            "stdout_path",
            "stderr_path",
        },
        "suite",
    )
    program = _normalized_absolute_path(suite["program_path"], "suite.program_path")
    source_root = _normalized_absolute_path(
        provenance["source_root_path"], "provenance.source_root_path"
    )
    try:
        program.relative_to(source_root)
    except ValueError as error:
        raise ValueError(
            "suite program must be inside the proven clean source tree"
        ) from error
    _hex(suite["program_sha256"], 64, "suite.program_sha256")
    plan = _normalized_absolute_path(suite["plan_path"], "suite.plan_path")
    _hex(suite["plan_sha256"], 64, "suite.plan_sha256")
    if verify_files:
        source_root_identity = _directory_identity(
            source_root, "provenance.source_root_path"
        )
        program_fd, program_bytes = _open_stable(
            program, "suite program", 16 * 1024 * 1024
        )
        os.close(program_fd)
        if sha256_bytes(program_bytes) != suite["program_sha256"]:
            raise ValueError("suite program SHA-256 mismatch")
        plan_fd, plan_bytes = _open_stable(plan, "suite plan", MAX_SUITE_PLAN_BYTES)
        os.close(plan_fd)
        if sha256_bytes(plan_bytes) != suite["plan_sha256"]:
            raise ValueError("suite plan SHA-256 mismatch")
        # The suite plan is part of the release provenance chain: its source
        # bundle hash must exactly equal the launcher's proven bundle manifest,
        # otherwise a plan authored against a different source closure could be
        # replayed behind a matching plan hash.
        try:
            plan_document = strict_json_loads(plan_bytes, "suite plan")
        except ValueError as error:
            raise ValueError(f"suite plan is not strict JSON: {error}") from error
        plan_bundle = plan_document.get("source_bundle_manifest_sha256")
        if (
            type(plan_bundle) is not str
            or plan_bundle != provenance["source_bundle_manifest_sha256"]
            or set(plan_bundle) == {"0"}
        ):
            raise ValueError(
                "suite plan does not bind the proven source bundle manifest"
            )
        _validate_program_closure(provenance, suite, plan_document, source_root)
        # The unchanged plan's profile contract has just been checked against
        # the proven source inventory and reopened canonical source bytes.
        profile_contract = _profile_contract(
            plan_document["performance_profile_contract"], plan_document
        )
        for profile, binding in performance_recipes.items():
            if (
                binding["sha256"]
                != profile_contract["profiles"][profile]["recipe_sha256"]
            ):
                raise ValueError(f"image {profile} recipe differs from proven source")
        for key in ("in_container_binary_path", "in_container_recipe_path"):
            if plan_document.get(key) != image[key]:
                raise ValueError(f"suite plan {key} differs from image")
    else:
        source_root_identity = None
    argv = suite["argv"]
    if type(argv) is not list or not argv or argv[0] != str(program):
        raise ValueError("suite.argv must start with the exact suite program path")
    if any(type(item) is not str or not item or "\x00" in item for item in argv):
        raise ValueError("suite.argv entries must be non-empty NUL-free strings")
    counts = {placeholder: argv.count(placeholder) for placeholder in PLACEHOLDERS}
    if any(count != 1 for count in counts.values()):
        raise ValueError(
            "suite.argv must contain every identity placeholder as an exact token once"
        )
    if any(
        placeholder in item and item != placeholder
        for item in argv
        for placeholder in PLACEHOLDERS
    ):
        raise ValueError(
            "suite identity placeholders must not be embedded in arguments"
        )
    suite["timeout_seconds"] = _plain_int(
        suite["timeout_seconds"], "suite.timeout_seconds", 1, 86_400
    )
    for key in ("receipt_path", "stdout_path", "stderr_path"):
        suite[key] = str(_within(runtime_dir, suite[key], f"suite.{key}"))
    if len({suite["receipt_path"], suite["stdout_path"], suite["stderr_path"]}) != 3:
        raise ValueError("suite output paths must be distinct")
    suite["_receipt_relpath"] = _runtime_relative(
        runtime_dir, Path(suite["receipt_path"]), "suite.receipt_path"
    )
    suite["_stdout_relpath"] = _runtime_relative(
        runtime_dir, Path(suite["stdout_path"]), "suite.stdout_path"
    )
    suite["_stderr_relpath"] = _runtime_relative(
        runtime_dir, Path(suite["stderr_path"]), "suite.stderr_path"
    )

    manifest["_raw_sha256"] = sha256_bytes(raw)
    manifest["_canonical_sha256"] = raw_canonical_sha256
    manifest["_runtime_dir"] = str(runtime_dir)
    if verify_files:
        manifest["_runtime_identity"] = runtime_identity
        manifest["_checkpoint_identity"] = checkpoint_identity
        manifest["_source_root_identity"] = source_root_identity
    return manifest


def _validate_program_closure(
    provenance: dict[str, Any],
    suite: dict[str, Any],
    plan: dict[str, Any],
    source_root: Path,
) -> None:
    if (
        plan.get("schema") != "flux-r15-container-suite-plan/4"
        or plan.get("release_version") != provenance["release_version"]
    ):
        raise ValueError("suite plan schema or release version differs")
    plan_root = _normalized_absolute_path(
        plan.get("source_root_path"), "suite plan source root"
    )
    if plan_root != source_root or _directory_identity(
        plan_root, "suite plan source root"
    ) != _directory_identity(source_root, "proven source root"):
        raise ValueError("suite plan source root differs from proven source root")
    for key in (
        "source_revision",
        "source_tree",
        "source_bundle_manifest_sha256",
        "checkpoint_id",
        "checkpoint_revision",
        "checkpoint_manifest_sha256",
        "recipe_sha256",
    ):
        if plan.get(key) != provenance[key]:
            raise ValueError(f"suite plan {key} differs from provenance")
    if plan.get("in_container_binary_sha256") != provenance["binary_sha256"]:
        raise ValueError("suite plan binary differs from provenance")
    descriptor, raw = _open_stable(
        Path(provenance["source_bundle_manifest_path"]),
        "proven source inventory",
        64 * 1024 * 1024,
    )
    os.close(descriptor)
    if sha256_bytes(raw) != provenance["source_bundle_manifest_sha256"]:
        raise ValueError("proven source inventory hash differs")
    inventory = strict_json_loads(raw, "proven source inventory")
    if type(inventory) is not dict or type(inventory.get("files")) is not list:
        raise ValueError("proven source inventory has no file list")
    entries: dict[str, dict[str, Any]] = {}
    for entry in inventory["files"]:
        if (
            type(entry) is not dict
            or type(entry.get("path")) is not str
            or entry["path"] in entries
        ):
            raise ValueError("proven source inventory entry differs")
        entries[entry["path"]] = entry
    validate_native_context_contract(plan.get("native_context_contract"))
    jobs = _exact_object(plan.get("gate_jobs"), set(REQUIRED_GATES), "suite gate jobs")
    programs = [(suite["program_path"], suite["program_sha256"])]
    for name in REQUIRED_GATES:
        job = jobs[name]
        if type(job) is not dict:
            raise ValueError("suite gate job is not an object")
        programs.append((job.get("program_path"), job.get("program_sha256")))
    for value, digest in programs:
        program = _normalized_absolute_path(value, "qualification program")
        try:
            relative = program.relative_to(source_root).as_posix()
        except ValueError as error:
            raise ValueError(
                "qualification program is outside proven source"
            ) from error
        entry = entries.get(f"{provenance_gate.ENGINE_PREFIX}/{relative}")
        if (
            not entry
            or entry.get("role") != "engine-source"
            or entry.get("mode") != 0o755
            or entry.get("sha256") != digest
        ):
            raise ValueError("qualification program is not a proven tracked executable")
        descriptor, payload = _open_stable(
            program, "qualification program", 16 * 1024 * 1024
        )
        os.close(descriptor)
        if len(payload) != entry.get("bytes") or sha256_bytes(payload) != digest:
            raise ValueError("qualification program differs from proven inventory")
    if (
        Path(suite["program_path"]).relative_to(source_root).as_posix()
        != "repro-kit/r15-container-suite.py"
    ):
        raise ValueError("suite program must be the tracked release orchestrator")
    _validate_profile_source_closure(provenance, plan, source_root, entries)
    _validate_native_context_source_closure(provenance, suite, plan, source_root, entries)


def _validate_profile_source_closure(
    provenance: dict[str, Any],
    plan: dict[str, Any],
    source_root: Path,
    entries: dict[str, dict[str, Any]],
) -> None:
    contract = _profile_contract(plan.get("performance_profile_contract"), plan)
    files = {
        "crates/spark-server/src/api/completions.rs": PROFILE_MEASUREMENT_CONTRACT[
            "completions_source_sha256"
        ],
        "crates/spark-server/src/cli.rs": PROFILE_MEASUREMENT_CONTRACT[
            "cli_source_sha256"
        ],
        **{
            item["recipe_relpath"]: item["recipe_sha256"]
            for item in contract["profiles"].values()
        },
    }
    for relative, expected in files.items():
        entry = entries.get(f"{provenance_gate.ENGINE_PREFIX}/{relative}")
        if (
            not entry
            or entry.get("role") != "engine-source"
            or type(entry.get("mode")) is not int
            or entry["mode"] not in (0o644, 0o755)
            or entry.get("sha256") != expected
        ):
            raise ValueError(
                "profile measurement/policy source is not in proven inventory"
            )
        descriptor, raw = _open_stable(
            source_root / relative, "profile policy source", MAX_IMAGE_RECIPE_BYTES
        )
        os.close(descriptor)
        if len(raw) != entry.get("bytes") or sha256_bytes(raw) != expected:
            raise ValueError(
                "profile measurement/policy source differs from proven inventory"
            )
        for name, (path, wide, mode) in PROFILE_RECIPES.items():
            if relative != path:
                continue
            environment = {}
            for line in raw.decode("utf-8").splitlines():
                tokens = shlex.split(line, comments=True, posix=True)
                if tokens and tokens[0] == "export":
                    tokens = tokens[1:]
                if not tokens:
                    continue
                if (
                    len(tokens) != 1
                    or "=" not in tokens[0]
                    or any(c in tokens[0] for c in "$`")
                ):
                    raise ValueError("profile recipe must use literal assignments")
                key, value = tokens[0].split("=", 1)
                if (
                    re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) is None
                    or key in environment
                ):
                    raise ValueError("profile recipe assignment is ambiguous")
                environment[key] = value
            if environment.get("FLUX_MODE") != mode or any(
                environment.get(key) != wide
                for key in ("FLUX_QWEN4_PREFILL_WIDE", "ATLAS_QWEN4_PREFILL_WIDE")
            ):
                raise ValueError(f"{name} source-defined recipe semantics differ")


def revalidate_manifest(original: dict[str, Any], path: Path, phase: str) -> None:
    fresh = load_manifest(path, verify_files=True)
    if (
        fresh["_raw_sha256"] != original["_raw_sha256"]
        or fresh["_canonical_sha256"] != original["_canonical_sha256"]
    ):
        raise ValueError(f"{phase}: launch manifest changed after validation")
    for key, label in (
        ("_runtime_identity", "runtime_dir"),
        ("_checkpoint_identity", "checkpoint directory"),
        ("_source_root_identity", "source root"),
    ):
        if fresh[key] != original[key]:
            raise ValueError(f"{phase}: {label} identity changed after validation")


def _inspect_document(raw: str) -> dict[str, Any]:
    value = strict_json_loads(raw, "docker image inspect")
    if type(value) is not list or len(value) != 1 or type(value[0]) is not dict:
        raise ValueError("docker image inspect must return exactly one object")
    return value[0]


def inspect_exact_image(
    manifest: dict[str, Any], backend: r12.CommandBackend
) -> dict[str, Any]:
    image = manifest["image"]
    result = backend.run(
        ("docker", "image", "inspect", image["id"]),
        timeout_s=r12.DEFAULT_COMMAND_TIMEOUT_S,
    )
    if result.returncode != 0:
        raise RuntimeError("docker image inspect failed for the immutable image ID")
    document = _inspect_document(result.stdout)
    if document.get("Id") != image["id"]:
        raise ValueError("docker image inspect returned a different image ID")
    digests = document.get("RepoDigests")
    matching_digests = (
        []
        if type(digests) is not list
        else [
            item
            for item in digests
            if type(item) is str
            and (item == image["digest"] or item.endswith("@" + image["digest"]))
        ]
    )
    if len(matching_digests) != 1:
        raise ValueError("immutable image digest is absent from RepoDigests")
    rootfs = document.get("RootFS")
    layers = rootfs.get("Layers") if type(rootfs) is dict else None
    if type(layers) is not list or any(
        type(layer) is not str or lifecycle.IMAGE_ID_RE.fullmatch(layer) is None
        for layer in layers
    ):
        raise ValueError("docker image inspect has an invalid rootfs layer list")
    if sha256_bytes(canonical_bytes(layers)) != image["rootfs_layers_sha256"]:
        raise ValueError("rootfs layer inventory SHA-256 mismatch")
    config = document.get("Config")
    labels = config.get("Labels") if type(config) is dict else None
    if type(labels) is not dict:
        raise ValueError("docker image inspect lacks OCI labels")
    expected_labels = {
        "org.opencontainers.image.version": provenance_gate.release_identity(
            manifest["release_version"]
        )["oci_version"],
        "org.opencontainers.image.revision": manifest["provenance"]["source_revision"],
        "com.theapathy.flux.source-tree": manifest["provenance"]["source_tree"],
        "com.theapathy.flux.binary-sha256": image["in_container_binary_sha256"],
        "com.theapathy.flux.recipe-sha256": image["recipe_sha256"],
        "com.theapathy.flux.performance.interactive.recipe-path": image[
            "performance_recipes"
        ]["interactive"]["path"],
        "com.theapathy.flux.performance.interactive.recipe-sha256": image[
            "performance_recipes"
        ]["interactive"]["sha256"],
        "com.theapathy.flux.performance.high-prefill.recipe-path": image[
            "performance_recipes"
        ]["high-prefill"]["path"],
        "com.theapathy.flux.performance.high-prefill.recipe-sha256": image[
            "performance_recipes"
        ]["high-prefill"]["sha256"],
    }
    unexpected = {
        key
        for key in labels
        if type(key) is str
        and key.startswith("com.theapathy.flux.performance.")
        and key not in expected_labels
    }
    if unexpected:
        raise ValueError(
            f"immutable image has unexpected performance labels: {sorted(unexpected)}"
        )
    mismatched = {
        key: {"expected": expected, "observed": labels.get(key)}
        for key, expected in expected_labels.items()
        if labels.get(key) != expected
    }
    if mismatched:
        raise ValueError(f"immutable image OCI labels differ: {sorted(mismatched)}")
    return {
        "id": image["id"],
        "digest": image["digest"],
        "rootfs_layers_sha256": image["rootfs_layers_sha256"],
        "rootfs_layer_count": len(layers),
        "labels_sha256": sha256_bytes(canonical_bytes(expected_labels)),
        "performance_recipes": image["performance_recipes"],
    }


def inspect_running_container(
    manifest: dict[str, Any],
    container_id: str,
    backend: r12.CommandBackend,
) -> dict[str, Any]:
    if lifecycle.CID_RE.fullmatch(container_id) is None:
        raise ValueError("running-container inspection requires one full ID")
    result = backend.run(
        (
            "docker",
            "container",
            "inspect",
            "--format",
            "{{json .}}",
            container_id,
        ),
        timeout_s=r12.DEFAULT_COMMAND_TIMEOUT_S,
    )
    if result.returncode != 0:
        raise RuntimeError("exact running-container inspection failed")
    document = strict_json_loads(result.stdout, "docker container inspect")
    if type(document) is not dict:
        raise ValueError("docker container inspect must return one object")
    state = document.get("State")
    host = document.get("HostConfig")
    mounts = document.get("Mounts")
    if (
        document.get("Id") != container_id
        or document.get("Image") != manifest["image"]["id"]
        or document.get("Path") != "/usr/bin/sleep"
        or document.get("Args") != ["infinity"]
        or type(state) is not dict
        or state.get("Running") is not True
        or type(host) is not dict
        or host.get("ReadonlyRootfs") is not True
        or type(mounts) is not list
        or len(mounts) != 2
    ):
        raise ValueError("running container differs from the exact launch contract")
    expected = {
        "/models": (manifest["provenance"]["checkpoint_path"], False),
        "/results": (manifest["_runtime_dir"], True),
    }
    normalized: dict[str, dict[str, Any]] = {}
    for index, mount_value in enumerate(mounts):
        if type(mount_value) is not dict:
            raise ValueError(f"running container mount {index} is not an object")
        destination = mount_value.get("Destination")
        if destination not in expected or destination in normalized:
            raise ValueError("running container has an unexpected or duplicate mount")
        source, writable = expected[destination]
        if (
            mount_value.get("Type") != "bind"
            or mount_value.get("Source") != source
            or mount_value.get("RW") is not writable
        ):
            raise ValueError(f"running container mount {destination} differs")
        normalized[destination] = {
            "type": "bind",
            "source_identity_match": True,
            "read_only": not writable,
        }
    if set(normalized) != set(expected):
        raise ValueError("running container is missing an exact release mount")
    return {
        "container_id": container_id,
        "image_id": manifest["image"]["id"],
        "entrypoint": "/usr/bin/sleep",
        "arguments": ["infinity"],
        "rootfs_read_only": True,
        "mounts": normalized,
    }


_STANDOFF_PREFIX = "flux-r15-standoff-"
_STANDOFF_ENTRYPOINT = "/bin/false"


def _unique_standoff_name() -> str:
    return _STANDOFF_PREFIX + secrets.token_hex(12)


def _create_standoff_container(backend: r12.CommandBackend, image_id: str) -> str:
    """Create (never start) the standoff container for one unmounted copy.

    A created container owns the exact immutable image filesystem without
    entering any running container's live mount namespace.  Copying release
    artifacts from the standoff means the bytes cannot drift between the
    before/after snapshots: the image layers are immutable by construction and
    the copy source is not the (scheduler-visible) running rootfs.
    """

    name = _unique_standoff_name()
    result = backend.run(
        (
            "docker",
            "create",
            "--entrypoint",
            _STANDOFF_ENTRYPOINT,
            "--name",
            name,
            image_id,
        ),
        timeout_s=r12.DEFAULT_COMMAND_TIMEOUT_S,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"failed to create the standoff image-bundle container: {result.stderr!r}"
        )
    expected = result.stdout.strip()
    if lifecycle.CID_RE.fullmatch(expected) is None:
        try:
            backend.run(
                ("docker", "container", "rm", "--force", name),
                timeout_s=r12.DEFAULT_COMMAND_TIMEOUT_S,
            )
        finally:
            pass
        raise RuntimeError("standoff container did not return one full container ID")
    return expected


def _remove_standoff_container(backend: r12.CommandBackend, container_id: str) -> None:
    result = backend.run(
        ("docker", "rm", "--force", container_id),
        timeout_s=r12.DEFAULT_COMMAND_TIMEOUT_S,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"failed to remove the standoff image-bundle container: {result.stderr!r}"
        )


def _copy_container_artifact(
    backend: r12.CommandBackend,
    standoff_container: str,
    container_path: str,
    destination: Path,
    label: str,
    expected_sha256: str,
    max_bytes: int,
) -> dict[str, Any]:
    """Copy one release artifact from the unmounted standoff bundle.

    ``docker cp`` without ``--follow-link`` copies the source path verbatim,
    including a symlink at the source path, which then fails the regular-file
    identity check below instead of silently hashing the link's target.
    """

    result = backend.run(
        (
            "docker",
            "cp",
            f"{standoff_container}:{container_path}",
            str(destination),
        ),
        timeout_s=r12.DEFAULT_COMMAND_TIMEOUT_S,
    )
    if result.returncode != 0:
        raise RuntimeError(f"failed to copy exact in-container {label}")
    descriptor, payload = _open_stable(
        destination, f"copied in-container {label}", max_bytes
    )
    os.close(descriptor)
    size = len(payload)
    if size <= 0:
        raise ValueError(f"in-container {label} must not be empty")
    observed = sha256_bytes(payload)
    if observed != expected_sha256:
        raise ValueError(f"in-container {label} SHA-256 mismatch")
    return {
        "container_path": container_path,
        "sha256": observed,
        "bytes": size,
        "source": "standoff-image-bundle",
    }


def measure_container_artifacts(
    manifest: dict[str, Any],
    container_id: str,
    backend: r12.CommandBackend,
) -> dict[str, Any]:
    if lifecycle.CID_RE.fullmatch(container_id) is None:
        raise ValueError("container artifact measurement requires a full container ID")
    image = manifest["image"]
    runtime_dir = Path(manifest["_runtime_dir"])
    standoff = _create_standoff_container(backend, image["id"])
    try:
        with tempfile.TemporaryDirectory(
            prefix="r15-image-artifacts-", dir=runtime_dir
        ) as temp:
            root = Path(temp)
            return {
                "binary": _copy_container_artifact(
                    backend,
                    standoff,
                    image["in_container_binary_path"],
                    root / "binary",
                    "binary",
                    image["in_container_binary_sha256"],
                    MAX_IMAGE_BINARY_BYTES,
                ),
                "recipe": _copy_container_artifact(
                    backend,
                    standoff,
                    image["in_container_recipe_path"],
                    root / "recipe",
                    "recipe",
                    image["recipe_sha256"],
                    MAX_IMAGE_RECIPE_BYTES,
                ),
                "performance_recipes": {
                    profile: _copy_container_artifact(
                        backend,
                        standoff,
                        binding["path"],
                        root / f"recipe-{profile}",
                        f"{profile} performance recipe",
                        binding["sha256"],
                        MAX_IMAGE_RECIPE_BYTES,
                    )
                    for profile, binding in image["performance_recipes"].items()
                },
            }
    finally:
        _remove_standoff_container(backend, standoff)


def _substitute_suite_argv(
    manifest: dict[str, Any], container_id: str, plan_fd: int, receipt_fd: int
) -> list[str]:
    suite = manifest["suite"]
    replacements = {
        "@CONTAINER_ID@": container_id,
        "@IMAGE_ID@": manifest["image"]["id"],
        "@IMAGE_DIGEST@": manifest["image"]["digest"],
        "@BASE_URL@": manifest["launch"]["base_url"],
        "@RUN_ID@": manifest["launch"]["run_id"],
        "@PLAN_FD@": str(plan_fd),
        "@RECEIPT_FD@": str(receipt_fd),
    }
    return [
        item if item not in replacements else replacements[item]
        for item in suite["argv"]
    ]


def _nonzero_hex(value: object, length: int, label: str) -> str:
    result = _hex(value, length, label)
    if set(result) == {"0"}:
        raise ValueError(f"{label} must not be an all-zero placeholder")
    return result


def _stable_path_bytes(path: Path, label: str, max_bytes: int) -> bytes:
    descriptor, payload = _open_stable(path, label, max_bytes)
    os.close(descriptor)
    return payload


MMLU_PRO_DATASET_SHA256 = (
    "a229ed37f28bea012bf08e81d8ae3358dda17ef7370dca360f632952e3bdcaca"
)
GSM8K_DATASET_SHA256 = (
    "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
)
PROFILE_CONTEXTS = (8_192, 16_384, 32_768, 65_536, 131_072, 262_080)
PROFILE_NAMES = ("interactive", "high-prefill")
PROFILE_MEASUREMENT_CONTRACT = {
    "decode_clock": "scheduler-decode-time:completion-tokens-minus-one/v1",
    "prefill_clock": "client-monotonic-blocking-max-tokens-one/v1",
    # completions.rs re-pinned (fix/test-triage): only its prompt_token_tests module
    # changed (checkpoint path via test_paths::resolve); serving code is identical.
    # completions.rs re-pinned (fix/request-contract): the InferenceRequest
    # literals gained decode_timeout_at: None; completions timing is unchanged.
    # re-pinned after OSS cleanup (SPDX headers, rustfmt, doc/comment scrub);
    # completions timing and CLI measurement flags are unchanged.
    "completions_source_sha256": "3c67e270273a587645777dad3032e3ec982d9f8aeaef258fd8e60745c71d2aa4",
    "cli_source_sha256": "d6da2e445725c8d45f123e9ffbb17dc1e2b6b401fae849bbcedbe965b5304086",
}
PROFILE_PROVENANCE_KEYS = (
    "source_revision",
    "source_tree",
    "source_bundle_manifest_sha256",
    "checkpoint_id",
    "checkpoint_revision",
    "checkpoint_manifest_sha256",
)
PROFILE_RECIPES = {
    "interactive": (
        "repro-kit/recipes/flux-v9-w4096-interactive.env",
        "4096",
        "interactive-w4096",
    ),
    "high-prefill": (
        "repro-kit/recipes/flux-v9-max-prefill.env",
        "16384",
        "max-prefill",
    ),
}
# Plan /3 carries profile contract /2: per-profile geometry, both recipes and
# the full launch argv per mode. It pairs with host reference /4, metrics /4
# and live proof /2; every cross pair is refused.
PROFILE_PLAN_CONTRACTS = {
    "flux-r15-container-suite-plan/2": "flux-performance-profile-contract/1",
    "flux-r15-container-suite-plan/3": "flux-performance-profile-contract/2",
    "flux-r15-container-suite-plan/4": "flux-performance-profile-contract/2",
}
PROFILE_CONTRACT_V2 = "flux-performance-profile-contract/2"
PROFILE_PAIRINGS = {
    "flux-performance-profile-contract/1": {
        "host_reference": "flux-prefill-decode-host-reference/3",
        "metrics": "flux-profile-metrics/3",
        "proof": "flux-r15-live-process-proof/1",
        "chunk_key": "chunk_tokens",
    },
    PROFILE_CONTRACT_V2: {
        "host_reference": "flux-prefill-decode-host-reference/4",
        "metrics": "flux-profile-metrics/4",
        "proof": "flux-r16-live-process-proof/2",
        "chunk_key": "profile_chunk_tokens",
    },
}
PROFILE_CHUNKS_V2 = {"interactive": 4096, "high-prefill": 16384}
PROFILE_LAUNCH_MODEL = "@PROFILE_MODEL@"
PROFILE_LAUNCH_PORT = "@CONTAINER_PORT@"
# Independently restated host recipe engine selectors (recipes/flux-serve*.argv):
# the container must not fall back to CLI defaults the host never ran.
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
LONG_CONTEXT_PROMPT_TOKENS = (
    2_017,
    *range(2_043, 2_051),
    2_112,
    7_336,
    7_710,
    8_192,
    30_647,
    32_768,
    65_536,
    114_755,
    122_413,
    131_072,
    238_938,
    262_080,
)


def _exact_true(value: object, label: str) -> None:
    if type(value) is not bool or value is not True:
        raise ValueError(f"{label} must be exact true")


def _exact_zero(value: object, label: str) -> None:
    if type(value) is not int or value != 0:
        raise ValueError(f"{label} must be exact integer zero")


def _finite_number(
    value: object,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be one finite JSON number")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{label} must be <= {maximum}")
    return result


GATE_VALIDATOR_SCHEMA = "flux-r15-gate-validator/1"


def _validator_document(
    evidence: dict[str, Any],
    gate: str,
    manifest: dict[str, Any],
    container_id: str,
) -> dict[str, Any]:
    label = f"{gate} evidence"
    expected = _nonzero_hex(
        evidence["validator_receipt_sha256"],
        64,
        f"{label}.validator_receipt_sha256",
    )
    locator = _safe_artifact_locator(
        evidence["validator_artifact"], f"{label}.validator_artifact"
    )
    raw = _stable_beneath_bytes(
        Path(manifest["_runtime_dir"]),
        locator,
        f"{label} validator artifact",
        MAX_GATE_RECEIPT_BYTES,
    )
    if not raw or sha256_bytes(raw) != expected:
        raise ValueError(f"{label} validator artifact SHA-256 mismatch")
    document = _exact_object(
        strict_json_loads(raw, f"{label} validator artifact"),
        {
            "schema",
            "gate",
            "status",
            "release_qualified",
            "container_id",
            "run_id",
            "raw",
        },
        f"{label} validator document",
    )
    if (
        document["schema"] != GATE_VALIDATOR_SCHEMA
        or document["gate"] != gate
        or document["container_id"] != container_id
        or document["run_id"] != manifest["launch"]["run_id"]
        or type(document["gate"]) is not str
        or type(document["container_id"]) is not str
        or type(document["run_id"]) is not str
    ):
        raise ValueError(f"{label} validator document identity differs")
    if document["status"] != "PASS":
        raise ValueError(f"{label} validator document is not a PASS")
    _exact_true(document["release_qualified"], f"{label} validator release_qualified")
    if type(document["raw"]) is not dict:
        raise ValueError(f"{label} validator raw is not an object")
    if document["gate"] != _evidence_gate_name(label):
        raise ValueError(f"{label} validator document gate differs")
    return document["raw"]


def _evidence_gate_name(label: str) -> str:
    return label.rpartition(" evidence")[0]


def _raw_object(raw: dict[str, Any], key: str, label: str) -> dict[str, Any]:
    if type(raw.get(key)) is not dict:
        raise ValueError(f"{label} validator raw `{key}` is not an object")
    return raw[key]


def _raw_bool(raw: dict[str, Any], key: str, label: str) -> bool:
    if type(raw.get(key)) is not bool:
        raise ValueError(f"{label} validator raw `{key}` is not a boolean")
    return raw[key]


def _raw_str(raw: dict[str, Any], key: str, label: str, wanted: str) -> str:
    value = raw.get(key)
    if type(value) is not str or value != wanted:
        raise ValueError(f"{label} validator raw `{key}` differs")
    return value


def _raw_int(raw: dict[str, Any], key: str, label: str, low: int, high: int) -> int:
    return _plain_int(raw.get(key), f"{label} validator raw `{key}`", low, high)


def _raw_hex(raw: dict[str, Any], key: str, label: str) -> str:
    return _nonzero_hex(raw.get(key), 64, f"{label} validator raw `{key}`")


def _raw_list(raw: dict[str, Any], key: str, label: str, max_items: int) -> list[Any]:
    value = raw.get(key)
    if type(value) is not list or len(value) > max_items:
        raise ValueError(f"{label} validator raw `{key}` is not a bounded list")
    return value


def _derive_all_bool(raw: dict[str, Any], keys: Sequence[str], label: str) -> bool:
    return all(_raw_bool(raw, key, label) for key in keys)


def _recompute_startup_evidence(
    evidence: dict[str, Any], raw: dict[str, Any], label: str
) -> None:
    attempts = _raw_list(raw, "attempts", label, 2)
    if len(attempts) != 2:
        raise ValueError(f"{label} validator raw attempts are not exactly two")
    modes: list[str] = []
    statuses: list[int] = []
    for index, attempt_value in enumerate(attempts):
        attempt = _exact_object(
            attempt_value,
            {
                "fresh_process_mode",
                "ready_http_status",
                "model_identity_match",
                "checkpoint_identity_match",
                "plain_scheduler_ready",
                "k5_scheduler_ready",
            },
            f"{label} validator raw attempts[{index}]",
        )
        mode = attempt.get("fresh_process_mode")
        if type(mode) is not str:
            raise ValueError(f"{label} validator raw fresh_process_mode is not a str")
        modes.append(mode)
        statuses.append(_raw_int(attempt, "ready_http_status", label, 100, 999))
        for key in (
            "model_identity_match",
            "checkpoint_identity_match",
            "plain_scheduler_ready",
            "k5_scheduler_ready",
        ):
            if not _raw_bool(attempt, key, label):
                raise ValueError(f"{label} validator raw {key} differs")
    if modes[0] != "none" or modes[1] != "native-mtp-k5":
        raise ValueError(f"{label} validator raw process modes differ")
    if (
        modes != evidence["fresh_process_modes"]
        or statuses != evidence["ready_http_statuses"]
        or not statuses == [200, 200]
        or _raw_bool(raw, "rootfs_read_only", label) is not True
        or _raw_hex(raw, "process_executable_sha256", label)
        != evidence["process_executable_sha256"]
    ):
        raise ValueError(f"{label} process identity differs from the validator raw")


_TOOL_FINISH_REASONS = frozenset(("tool_calls", "stop"))


def _recompute_tool_evidence(
    evidence: dict[str, Any], raw: dict[str, Any], label: str
) -> None:
    cases = _raw_list(raw, "cases", label, 4_096)
    if not cases:
        raise ValueError(f"{label} validator raw cases are empty")
    blocking = 0
    streaming = 0
    calls_expected = 0
    calls_observed = 0
    aggregate = {
        "exact_arguments": True,
        "finish_reasons_valid": True,
        "no_blank_line_runaway": True,
        "no_all_tools_runaway": True,
        "thinking_spacing_valid": True,
    }
    for index, case_value in enumerate(cases):
        case = _exact_object(
            case_value,
            {
                "phase",
                "tool_calls_observed",
                "tool_calls_expected",
                "exact_arguments",
                "finish_reason",
                "no_blank_line_runaway",
                "no_all_tools_runaway",
                "thinking_spacing_valid",
            },
            f"{label} validator raw cases[{index}]",
        )
        phase = case.get("phase")
        if type(phase) is not str or phase not in ("blocking", "streaming"):
            raise ValueError(f"{label} validator raw phase differs")
        calls_expected += _raw_int(case, "tool_calls_expected", label, 0, 100_000)
        calls_observed += _raw_int(case, "tool_calls_observed", label, 0, 100_000)
        reason = case.get("finish_reason")
        if type(reason) is not str or reason not in _TOOL_FINISH_REASONS:
            raise ValueError(f"{label} validator raw finish_reason differs")
        if phase == "blocking":
            blocking += 1
        else:
            streaming += 1
        if reason != "tool_calls":
            aggregate["finish_reasons_valid"] = False
        for key in (
            "exact_arguments",
            "no_blank_line_runaway",
            "no_all_tools_runaway",
            "thinking_spacing_valid",
        ):
            if not _raw_bool(case, key, label):
                aggregate[key] = False
    if (
        blocking < 1
        or streaming < 1
        or blocking != evidence["blocking_cases"]
        or streaming != evidence["streaming_cases"]
        or calls_expected != evidence["tool_calls_expected"]
        or calls_observed != evidence["tool_calls_observed"]
        or calls_expected != calls_observed
    ):
        raise ValueError(f"{label} census differs from the validator raw")
    for key, derived in aggregate.items():
        if derived is not evidence[key]:
            raise ValueError(f"{label} aggregate {key} differs from the validator raw")


_ACCURACY_RAW_KEYS = (
    "dataset_sha256",
    "items",
)


def _recompute_accuracy_evidence(
    evidence: dict[str, Any],
    raw: dict[str, Any],
    label: str,
    dataset_sha256: str,
    items: int,
    floor: float,
) -> None:
    del floor
    if _raw_hex(raw, "dataset_sha256", label) != dataset_sha256:
        raise ValueError(f"{label} dataset differs from the validator raw")
    rows = _raw_list(raw, "items", label, 4_096)
    if len(rows) != items:
        raise ValueError(f"{label} validator raw items census differs")
    no_spec = 0
    k5 = 0
    for index, row_value in enumerate(rows):
        row = _exact_object(
            row_value,
            {
                "plain_correct",
                "k5_correct",
                "answer_identity",
                "correctness_identity",
                "token_identity",
            },
            f"{label} validator raw items[{index}]",
        )
        if _raw_bool(row, "plain_correct", label):
            no_spec += 1
        if _raw_bool(row, "k5_correct", label):
            k5 += 1
        if row["plain_correct"] is not row["k5_correct"]:
            raise ValueError(f"{label} per-item correctness differs")
        for key in (
            "answer_identity",
            "correctness_identity",
            "token_identity",
        ):
            if not _raw_bool(row, key, label):
                raise ValueError(f"{label} validator raw {key} differs")
    if no_spec != evidence["no_spec_correct"] or k5 != evidence["k5_correct"]:
        raise ValueError(f"{label} correct counts differ from the validator raw")


def _recompute_long_context_evidence(
    evidence: dict[str, Any], raw: dict[str, Any], label: str
) -> None:
    cases = _raw_list(raw, "cases", label, 64)
    if len(cases) != evidence["case_count"]:
        raise ValueError(f"{label} validator raw case census differs")
    prompt_counts: list[int] = []
    max_prompt = 0
    max_total = 0
    for index, case_value in enumerate(cases):
        case = _exact_object(
            case_value,
            {
                "prompt_tokens",
                "plain_total_tokens",
                "k5_total_tokens",
                "token_identity",
                "text_identity",
                "finish_reason_identity",
                "usage_identity",
                "native_k5_engaged",
                "partial_accept_cycle",
                "qsa_verify_rows_engaged",
            },
            f"{label} validator raw cases[{index}]",
        )
        prompt = _raw_int(case, "prompt_tokens", label, 1, 1_000_000)
        prompt_counts.append(prompt)
        max_prompt = max(max_prompt, prompt)
        plain_total = _raw_int(case, "plain_total_tokens", label, 1, 1_000_000)
        k5_total = _raw_int(case, "k5_total_tokens", label, 1, 1_000_000)
        max_total = max(max_total, plain_total, k5_total)
        for key in (
            "token_identity",
            "text_identity",
            "finish_reason_identity",
            "usage_identity",
            "native_k5_engaged",
            "partial_accept_cycle",
            "qsa_verify_rows_engaged",
        ):
            if not _raw_bool(case, key, label):
                raise ValueError(f"{label} validator raw {key} differs")
    if (
        tuple(prompt_counts) != tuple(evidence["prompt_token_counts"])
        or max_prompt != evidence["max_prompt_tokens"]
        or max_total != evidence["max_total_tokens"]
    ):
        raise ValueError(f"{label} token census differs from the validator raw")


def _recompute_native_262k_evidence(
    evidence: dict[str, Any], raw: dict[str, Any], label: str
) -> None:
    if (
        _raw_str(raw, "kv_dtype", label, evidence["kv_dtype"]) != evidence["kv_dtype"]
        or _raw_int(raw, "max_seq_len", label, 1, 1_000_000) != evidence["max_seq_len"]
        or _raw_int(raw, "passkeys_per_prompt", label, 1, 1_000_000)
        != evidence["passkeys_per_prompt"]
    ):
        raise ValueError(f"{label} validator raw identity differs")
    retrievals = _raw_list(raw, "retrievals", label, 16)
    prompt_tokens_returned = []
    answers_all_exact = True
    for index, item_value in enumerate(retrievals):
        item = _exact_object(
            item_value,
            {"prompt_tokens", "answers", "expected"},
            f"{label} validator raw retrievals[{index}]",
        )
        prompt_tokens_returned.append(
            _raw_int(item, "prompt_tokens", label, 1, 1_000_000)
        )
        answers = item.get("answers")
        expected_answers = item.get("expected")
        for value in (answers, expected_answers):
            if type(value) is not list or any(type(text) is not str for text in value):
                raise ValueError(f"{label} validator raw retrieval answers differ")
            if (
                len(value) != evidence["passkeys_per_prompt"]
                or any(not text.strip() for text in value)
                or len(set(value)) != len(value)
            ):
                raise ValueError(
                    f"{label} retrieval requires distinct nonempty passkeys"
                )
        if answers != expected_answers:
            answers_all_exact = False
    if (
        tuple(prompt_tokens_returned) != tuple(evidence["retrieval_prompt_tokens"])
        or answers_all_exact is not evidence["all_retrieval_answers_exact"]
        or answers_all_exact is not True
    ):
        raise ValueError(f"{label} retrieval census differs from the validator raw")
    boundary = _raw_object(raw, "boundary", label)
    overflow = _raw_object(raw, "overflow", label)
    boundary_pass = (
        _raw_int(boundary, "prompt_tokens", label, 1, 1_000_000)
        == evidence["boundary_prompt_tokens"]
        and _raw_bool(boundary, "admitted", label) is True
    )
    overflow_pass = (
        _raw_int(overflow, "prompt_tokens", label, 1, 1_000_000)
        == evidence["overflow_prompt_tokens"]
        and _raw_bool(overflow, "admitted", label) is False
    )
    if (
        boundary_pass is not evidence["boundary_admission_pass"]
        or overflow_pass is not evidence["overflow_rejection_pass"]
        or boundary_pass is not True
        or overflow_pass is not True
        or _raw_bool(raw, "qsa_terminal_chunks_exact", label) is not True
    ):
        raise ValueError(f"{label} boundary census differs from the validator raw")


_CONCURRENCY_GRID = tuple(
    (level, mode) for level in (2, 4, 8) for mode in ("none", "native-mtp-k5")
)


def _recompute_concurrency_evidence(
    evidence: dict[str, Any], raw: dict[str, Any], label: str
) -> None:
    arms = _raw_list(raw, "arms", label, 64)
    cells: list[tuple[int, str]] = []
    fresh_processes = 0
    full_frames: list[int] = []
    for index, arm_value in enumerate(arms):
        arm = _exact_object(
            arm_value,
            {
                "level",
                "mode",
                "fresh_process",
                "full_frames",
                "forward_reverse_order_all_frames",
                "per_frame_logit_identity",
                "per_frame_state_identity",
            },
            f"{label} validator raw arms[{index}]",
        )
        level = _raw_int(arm, "level", label, 0, 10_000)
        mode_value = arm.get("mode")
        if type(mode_value) is not str:
            raise ValueError(f"{label} validator raw arm mode is not a str")
        cells.append((level, mode_value))
        if _raw_bool(arm, "fresh_process", label):
            fresh_processes += 1
        full_frames.append(_raw_int(arm, "full_frames", label, 0, 1_000_000))
        for key in (
            "forward_reverse_order_all_frames",
            "per_frame_logit_identity",
            "per_frame_state_identity",
        ):
            if not _raw_bool(arm, key, label):
                raise ValueError(f"{label} validator raw {key} differs")
    grid = set(cells)
    if (
        grid != set(_CONCURRENCY_GRID)
        or len(arms) != 3 * len(_CONCURRENCY_GRID)
        or any(cells.count(cell) != 3 for cell in _CONCURRENCY_GRID)
    ):
        raise ValueError(f"{label} validator raw process grid differs")
    samples = _raw_list(raw, "available_mib_samples", label, 10_000)
    if not samples or any(type(value) is not int or value < 0 for value in samples):
        raise ValueError(f"{label} validator raw available-memory samples differ")
    minimum_frames = min(full_frames) if full_frames else 0
    if (
        fresh_processes != evidence["fresh_processes"]
        or minimum_frames != evidence["minimum_full_frames"]
        or min(samples) != evidence["minimum_available_mib"]
        or _raw_bool(raw, "cancellation_replacement_pass", label) is not True
        or _raw_bool(raw, "final_highest_slot_reused", label) is not True
        or _raw_bool(raw, "post_drain_clean", label) is not True
    ):
        raise ValueError(f"{label} concurrency census differs from the validator raw")


def _profile_equal(value: object, expected: object, label: str) -> None:
    if canonical_bytes(value) != canonical_bytes(expected):
        raise ValueError(f"profile {label} differs from independently derived operands")


def _profile_overlay(value: object) -> dict[str, Any]:
    if (
        type(value) is not list
        or len(value) % 2
        or len(value) > 8
        or any(type(item) is not str or not item or "\x00" in item for item in value)
    ):
        raise ValueError("profile overlay is malformed")
    result: dict[str, Any] = {}
    for key, item in zip(value[::2], value[1::2]):
        if (
            key
            not in {
                "--max-prefill-tokens",
                "--max-seq-len",
                "--scheduling-policy",
                "--tbt-deadline-ms",
            }
            or key in result
        ):
            raise ValueError("profile selector is unsupported or repeated")
        result[key] = item
    for key, expected in (("--max-prefill-tokens", 4096), ("--max-seq-len", 262144)):
        item = result.get(key)
        if (
            type(item) is not str
            or re.fullmatch(r"[0-9]+", item) is None
            or int(item) != expected
        ):
            raise ValueError("profile chunk/context selector differs")
        result[key] = expected
    policy = result.setdefault("--scheduling-policy", "fifo")
    deadline = result.get("--tbt-deadline-ms", "100")
    if (
        policy not in ("fifo", "slai")
        or re.fullmatch(r"[0-9]+", deadline) is None
        or int(deadline) > (1 << 64) - 1
    ):
        raise ValueError("profile scheduler selector differs")
    result["--tbt-deadline-ms"] = int(deadline) if policy == "slai" else None
    return result


def _profile_launch_argv(
    binary_path: str, model: str, port: str, overlay: list[str], mode: str
) -> list[str]:
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


def _profile_overlay_v2(value: object, chunk: int) -> dict[str, Any]:
    # Contract /2: chunk, context, block, policy and deadline each explicit
    # once, plus --qwen4-qsa (capacity exceeds 2048). Nothing else.
    if (
        type(value) is not list
        or len(value) != 11 + len(PROFILE_ENGINE_OVERLAY_V2)
        or any(type(item) is not str or not item or "\x00" in item for item in value)
        or value.count("--qwen4-qsa") != 1
    ):
        raise ValueError("profile /2 overlay is malformed")
    if value[11:] != list(PROFILE_ENGINE_OVERLAY_V2):
        raise ValueError("profile /2 overlay differs from the host recipe engine argv")
    if value[:11] != [
        "--max-prefill-tokens",
        str(chunk),
        "--max-seq-len",
        "262144",
        "--block-size",
        "16",
        "--qwen4-qsa",
        "--scheduling-policy",
        "fifo",
        "--tbt-deadline-ms",
        "100",
    ]:
        raise ValueError("profile /2 overlay differs from the sealed profile recipe")
    value = value[:11]
    pairs = [item for item in value if item != "--qwen4-qsa"]
    result: dict[str, Any] = {}
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
            or key in result
        ):
            raise ValueError("profile selector is unsupported or repeated")
        result[key] = item
    if len(result) != 5:
        raise ValueError("profile /2 overlay omits a pinned selector")
    for key, expected in (
        ("--max-prefill-tokens", chunk),
        ("--max-seq-len", 262144),
        ("--block-size", 16),
    ):
        if re.fullmatch(r"[0-9]+", result[key]) is None or int(result[key]) != expected:
            raise ValueError("profile chunk/context/block selector differs")
        result[key] = expected
    deadline = result["--tbt-deadline-ms"]
    if (
        result["--scheduling-policy"] not in ("fifo", "slai")
        or re.fullmatch(r"[0-9]+", deadline) is None
        or int(deadline) > (1 << 64) - 1
    ):
        raise ValueError("profile scheduler selector differs")
    result["--tbt-deadline-ms"] = int(deadline)
    result["--qwen4-qsa"] = True
    return result


def _profile_contract_v2(result: dict[str, Any], binary_path: str) -> dict[str, Any]:
    profiles = _exact_object(
        result["profiles"], set(PROFILE_NAMES), "performance profiles"
    )
    hashes, shared = set(), []
    for name, (path, wide, _mode) in PROFILE_RECIPES.items():
        row = _exact_object(
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
        if row["recipe_relpath"] != path or row["in_container_recipe_path"] != (
            PERFORMANCE_RECIPE_PATHS[name]
        ):
            raise ValueError("performance profile canonical recipe differs")
        hashes.add(_nonzero_hex(row["recipe_sha256"], 64, "performance profile recipe hash"))
        if (
            type(row["chunk_tokens"]) is not int
            or row["chunk_tokens"] != PROFILE_CHUNKS_V2[name]
            or str(row["chunk_tokens"]) != wide
        ):
            raise ValueError("performance profile chunk differs from reviewed geometry")
        values = _profile_overlay_v2(row["argv_overlay"], row["chunk_tokens"])
        del values["--max-prefill-tokens"]
        shared.append(values)
        templates = _exact_object(
            row["launch_argv_sha256"],
            {"none", "native-mtp-k5"},
            "performance profile launch argv hashes",
        )
        for mode, observed in templates.items():
            expected = sha256_bytes(
                canonical_bytes(
                    _profile_launch_argv(
                        binary_path,
                        PROFILE_LAUNCH_MODEL,
                        PROFILE_LAUNCH_PORT,
                        row["argv_overlay"],
                        mode,
                    )
                )
            )
            if observed != expected:
                raise ValueError("performance profile full launch argv hash differs")
    if len(hashes) != 2:
        raise ValueError("performance profiles must bind two distinct recipes")
    if shared[0] != shared[1]:
        raise ValueError("performance profiles differ outside the scheduler chunk")
    return result


def _profile_chunk(contract: dict[str, Any], name: str) -> int:
    if contract["schema"] == PROFILE_CONTRACT_V2:
        return contract["profiles"][name]["chunk_tokens"]
    return 4096


def _profile_contract(
    value: object, plan: dict[str, Any] | None = None
) -> dict[str, Any]:
    result = _exact_object(
        value, {"schema", "profiles"}, "performance profile contract"
    )
    if plan is not None and PROFILE_PLAN_CONTRACTS.get(plan.get("schema")) != result[
        "schema"
    ]:
        raise ValueError("performance profile contract and plan are a cross pair")
    if result["schema"] == PROFILE_CONTRACT_V2:
        binary_path = plan.get("in_container_binary_path") if plan is not None else None
        if type(binary_path) is not str:
            raise ValueError("performance profile /2 needs its sealed plan")
        return _profile_contract_v2(result, binary_path)
    if result["schema"] != "flux-performance-profile-contract/1":
        raise ValueError("performance profile contract schema differs")
    profiles = _exact_object(
        result["profiles"], set(PROFILE_NAMES), "performance profiles"
    )
    effective = []
    for name, (path, _wide, _mode) in PROFILE_RECIPES.items():
        row = _exact_object(
            profiles[name],
            {"recipe_relpath", "recipe_sha256", "argv_overlay"},
            "performance profile binding",
        )
        if row["recipe_relpath"] != path:
            raise ValueError("performance profile canonical recipe differs")
        _nonzero_hex(row["recipe_sha256"], 64, "performance profile recipe hash")
        effective.append(_profile_overlay(row["argv_overlay"]))
    if effective[0] == effective[1]:
        raise ValueError("performance profiles share one effective argv")
    return result


def _profile_clock(value: object) -> float:
    clock = _exact_object(value, {"start", "finish"}, "profile client monotonic proof")
    if any(
        type(item) is not float or not math.isfinite(item) for item in clock.values()
    ):
        raise ValueError("profile client monotonic proof is not finite floats")
    wall = clock["finish"] - clock["start"]
    if not math.isfinite(wall) or wall <= 0.0:
        raise ValueError("profile client request wall is not positive")
    return wall


def _profile_usage(
    value: object, context: int, budget: int, *, decode: bool
) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError("profile terminal usage is missing")
    for key, expected in (
        ("prompt_tokens", context),
        ("completion_tokens", budget),
        ("total_tokens", context + budget),
    ):
        if type(value.get(key)) is not int or value[key] != expected:
            raise ValueError("profile terminal usage workload census differs")
    details = value.get("prompt_tokens_details")
    if (
        type(details) is not dict
        or type(details.get("cached_tokens")) is not int
        or details["cached_tokens"] != 0
    ):
        raise ValueError("profile terminal usage reuses the prefix cache")
    if not decode:
        return value
    rate = _raw_positive_number(value, "response_token/s", "profile terminal usage")
    wall = (budget - 1) / rate
    if not math.isfinite(wall) or wall <= 0.0:
        raise ValueError("profile source-derived decode wall is not positive")
    return {
        "decode_tokens": budget - 1,
        "decode_wall_seconds": wall,
        "decode_clock": PROFILE_MEASUREMENT_CONTRACT["decode_clock"],
        "engine_decode_tokens_per_second": rate,
        "terminal_usage": value,
    }


def _profile_runtime_recipe_binding(
    contract: dict[str, Any], image: dict[str, Any]
) -> None:
    if contract["schema"] == PROFILE_CONTRACT_V2:
        # Broker /2 serves each profile from its own image recipe. Both must be
        # the image's labelled performance recipes, byte-hash for byte-hash.
        recipes = _image_performance_recipes(image["performance_recipes"])
        for name, row in contract["profiles"].items():
            if (
                recipes[name]["path"] != row["in_container_recipe_path"]
                or recipes[name]["sha256"] != row["recipe_sha256"]
            ):
                raise ValueError("image does not carry this profile's recipe")
        return
    # The current immutable-image broker admits one recipe only. Two recipe
    # labels cannot relax its environment proof or manufacture profile parity.
    if any(
        row["recipe_sha256"] != image["recipe_sha256"]
        for row in contract["profiles"].values()
    ):
        raise ValueError(
            "HOLD: immutable image/broker cannot bind both profile recipes"
        )


def _profile_plan(manifest: dict[str, Any]) -> dict[str, Any]:
    suite = manifest["suite"]
    raw = _stable_path_bytes(
        Path(suite["plan_path"]), "profile sealed-plan source", MAX_SUITE_PLAN_BYTES
    )
    if sha256_bytes(raw) != suite["plan_sha256"]:
        raise ValueError("profile suite plan changed after capture")
    plan = strict_json_loads(raw, "profile suite plan")
    if type(plan) is not dict:
        raise ValueError("profile suite plan is not an object")
    _validate_program_closure(
        manifest["provenance"],
        suite,
        plan,
        Path(manifest["provenance"]["source_root_path"]),
    )
    return plan


def _profile_option(
    plan: dict[str, Any], option: str, default: object = None
) -> object:
    argv = plan["gate_jobs"]["prefill-decode-profiles"].get("argv")
    if type(argv) is not list or any(type(item) is not str for item in argv):
        raise ValueError("profile pinned driver argv is missing")
    positions = [i for i, item in enumerate(argv) if item == option]
    if not positions and default is not None:
        return default
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise ValueError(f"profile pinned {option} is missing or repeated")
    return argv[positions[0] + 1]


def _profile_processes(
    raw: dict[str, Any],
    plan: dict[str, Any],
    manifest: dict[str, Any],
    container_id: str,
    contract: dict[str, Any] | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    v2 = contract is not None and contract["schema"] == PROFILE_CONTRACT_V2
    bindings = _raw_list(raw, "process_bindings", "profile", 4)
    metrics = _raw_list(raw, "verification_metrics", "profile", 4)
    expected_keys = {
        (name, mode) for name in PROFILE_NAMES for mode in ("none", "native-mtp-k5")
    }
    if len(bindings) != 4 or len(metrics) != 4:
        raise ValueError("profile process/metrics census differs")
    by_key = {}
    identities = set()
    port = str(_profile_option(plan, "--container-port", "8888"))
    if re.fullmatch(r"[0-9]+", port) is None or not 1 <= int(port) <= 65535:
        raise ValueError("profile pinned container port differs")
    for item in bindings:
        row = _exact_object(
            item,
            {"profile", "mode", "proof", "launch_argv", "launch_argv_sha256"},
            "profile process binding",
        )
        key = (row["profile"], row["mode"])
        if (
            any(type(part) is not str for part in key)
            or key not in expected_keys
            or key in by_key
        ):
            raise ValueError("profile process binding identity differs")
        argv = row["launch_argv"]
        if (
            type(argv) is not list
            or len(argv) < 20
            or any(type(item) is not str for item in argv)
        ):
            raise ValueError("profile observed launch argv is malformed")
        model = argv[6]
        if (
            re.fullmatch(r"flux-profile-prefill-decode-profiles-[0-9a-f]{32}", model)
            is None
        ):
            raise ValueError("profile launch model alias differs")
        expected = [
            plan["in_container_binary_path"],
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
            *plan["performance_profile_contract"]["profiles"][key[0]]["argv_overlay"],
        ]
        if key[1] == "native-mtp-k5":
            expected += ["--speculative", "--num-drafts", "4"]
        _profile_equal(argv, expected, "observed launch argv")
        if v2:
            # The observed argv, with its per-run alias and port as
            # placeholders, is exactly the sealed full-argv template.
            template = list(argv)
            template[6], template[10] = PROFILE_LAUNCH_MODEL, PROFILE_LAUNCH_PORT
            if (
                sha256_bytes(canonical_bytes(template))
                != contract["profiles"][key[0]]["launch_argv_sha256"][key[1]]
            ):
                raise ValueError("profile observed argv differs from its full-argv binding")
        if row["launch_argv_sha256"] != sha256_bytes(canonical_bytes(argv)):
            raise ValueError("profile observed launch argv hash differs")
        proof = row["proof"]
        if type(proof) is not dict or proof.get("mode") != key[1]:
            raise ValueError("profile process proof mode differs")
        identity = (
            _plain_int(proof.get("host_pid"), "profile process PID", 1, 1 << 31),
            _plain_int(
                proof.get("start_time_ticks"), "profile process start ticks", 1, 1 << 63
            ),
        )
        if identity in identities:
            raise ValueError("profile arms reused a process identity")
        identities.add(identity)
        by_key[key] = row
    for name in PROFILE_NAMES:
        _validate_process_bindings(
            [by_key[(name, mode)]["proof"] for mode in ("none", "native-mtp-k5")],
            manifest,
            container_id,
            (name, contract["profiles"][name]["recipe_sha256"]) if v2 else None,
        )
    seen = set()
    for item in metrics:
        row = _exact_object(
            item,
            {"profile", "mode", "k5_events", "metrics_sha256", "metrics"},
            "profile verifier metrics",
        )
        key = (row["profile"], row["mode"])
        if (
            any(type(part) is not str for part in key)
            or key not in expected_keys
            or key in seen
        ):
            raise ValueError("profile verifier metrics identity differs")
        seen.add(key)
        if (
            type(row["metrics"]) is not str
            or sha256_bytes(row["metrics"].encode()) != row["metrics_sha256"]
        ):
            raise ValueError("profile verifier metrics hash differs")
        total = sum(
            float(value)
            for value in re.findall(
                r'^atlas_spec_decode_verify_total\{[^\n}]*\bk="5"[^\n}]*\}\s+([0-9.eE+-]+)\s*$',
                row["metrics"],
                re.MULTILINE,
            )
        )
        if (
            not math.isfinite(total)
            or not 0 <= total <= 1_000_000_000_000
            or not total.is_integer()
            or type(row["k5_events"]) is not int
            or row["k5_events"] != int(total)
            or (total > 0) != (key[1] == "native-mtp-k5")
        ):
            raise ValueError("profile native K5 verifier engagement differs")
    return by_key


def _profile_journal(
    raw: dict[str, Any],
    manifest: dict[str, Any],
    processes: dict[tuple[str, str], dict[str, Any]],
    host_rows: dict,
    contract: dict[str, Any] | None = None,
) -> dict:
    v2 = contract is not None and contract["schema"] == PROFILE_CONTRACT_V2
    journal = _exact_object(
        raw["response_journal"],
        {"artifact", "sha256", "requests"},
        "profile response journal",
    )
    if type(journal["requests"]) is not int or journal["requests"] != 72:
        raise ValueError("profile response journal request census differs")
    payload = _stable_beneath_bytes(
        Path(manifest["_runtime_dir"]),
        _safe_artifact_locator(journal["artifact"], "profile journal artifact"),
        "profile response journal",
        MAX_GATE_RECEIPT_BYTES,
    )
    if sha256_bytes(payload) != _nonzero_hex(
        journal["sha256"], 64, "profile journal hash"
    ):
        raise ValueError("profile response journal hash differs")
    lines = payload.splitlines(keepends=True)
    if len(lines) != 72:
        raise ValueError("profile response journal record census differs")
    arms = {}
    request_ids = set()
    seen = set()
    for line in lines:
        record = strict_json_loads(line, "profile response journal record")
        if canonical_bytes(record) + b"\n" != line or type(record) is not dict:
            raise ValueError("profile response journal record is not canonical")
        role = (
            record.get("profile"),
            record.get("mode"),
            record.get("context_tokens"),
            record.get("kind"),
        )
        name, mode, context, kind = role
        if (
            type(name) is not str
            or type(mode) is not str
            or type(context) is not int
            or type(kind) is not str
            or (name, mode) not in processes
            or context not in PROFILE_CONTEXTS
            or kind not in ("exact", "ttft", "stream")
            or role in seen
        ):
            raise ValueError("profile response journal role differs or is duplicated")
        seen.add(role)
        _exact_object(
            record,
            {
                "profile",
                "mode",
                "context_tokens",
                "kind",
                "prompt_sha256",
                "request",
                "client_wall_seconds",
                "client_monotonic",
                "events" if kind == "stream" else "response",
            }
            | ({"chunk_tokens", "recipe_sha256"} if v2 else set()),
            "profile response journal record",
        )
        if v2 and (
            type(record["chunk_tokens"]) is not int
            or record["chunk_tokens"] != _profile_chunk(contract, name)
            or record["recipe_sha256"] != contract["profiles"][name]["recipe_sha256"]
        ):
            # A record cannot be relabelled onto another profile's geometry.
            raise ValueError("profile journal record chunk/recipe binding differs")
        if record["prompt_sha256"] != host_rows[(name, context)]["prompt_sha256"]:
            raise ValueError("profile journal prompt differs from pinned host workload")
        budget = min(256, 262144 - context)
        requested = 1 if kind == "ttft" else budget
        model = processes[(name, mode)]["launch_argv"][6]
        request = {
            "model": model,
            "prompt": "",
            "max_tokens": requested,
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "stream": kind == "stream",
            "stop": [],
            "seed": 0x38_4096,
            "ignore_eos": True,
        }
        if kind == "stream":
            request["stream_options"] = {
                "include_usage": True,
                "continuous_usage_stats": False,
            }
        else:
            request["return_token_ids"] = True
        _profile_equal(record["request"], request, "journal request workload")
        wall = _profile_clock(record["client_monotonic"])
        _profile_equal(record["client_wall_seconds"], wall, "journal client wall")
        arm = arms.setdefault((name, context, mode), {})
        if kind == "stream":
            events = record["events"]
            if type(events) is not list or not 2 <= len(events) <= 100000:
                raise ValueError("profile stream event census differs")
            pieces = []
            stamps = []
            usage = identity = None
            previous = 0.0
            for event in events:
                if (
                    type(event) is not list
                    or len(event) != 2
                    or type(event[0]) is not float
                    or not math.isfinite(event[0])
                    or not previous <= event[0] <= wall
                    or type(event[1]) is not str
                    or usage is not None
                ):
                    raise ValueError("profile stream event ordering/clock differs")
                stamp, data = event
                previous = stamp
                value = strict_json_loads(data, "profile stream event")
                if type(value) is not dict or "error" in value:
                    raise ValueError("profile stream event is not a completion")
                response_id = value.get("id")
                if (
                    type(response_id) is not str
                    or not response_id
                    or (identity is not None and identity != response_id)
                ):
                    raise ValueError("profile stream response identity differs")
                identity = response_id
                choices = value.get("choices")
                if (
                    type(choices) is not list
                    or len(choices) != 1
                    or type(choices[0]) is not dict
                ):
                    raise ValueError("profile stream choices differ")
                choice = choices[0]
                if (
                    type(choice.get("index")) is not int
                    or choice["index"] != 0
                    or type(choice.get("text")) is not str
                ):
                    raise ValueError("profile stream choice geometry differs")
                terminal = type(value.get("usage")) is dict
                if terminal:
                    if choice.get("finish_reason") != "length" or choice["text"] != "":
                        raise ValueError("profile stream terminal geometry differs")
                    usage = value["usage"]
                elif choice.get("finish_reason") is not None:
                    raise ValueError("profile stream finished without terminal usage")
                if choice["text"]:
                    pieces.append(choice["text"])
                    stamps.append(stamp)
            if not pieces or len(pieces) > budget or type(usage) is not dict:
                raise ValueError("profile stream text/terminal census differs")
            arm[kind] = {
                **_profile_usage(usage, context, budget, decode=True),
                "streamed_text_sha256": sha256_bytes("".join(pieces).encode()),
                "stream_text_events": len(pieces),
                "observed_text_event_wall_seconds": stamps[-1] - stamps[0],
                "finish_reason": "length",
            }
        else:
            response = record["response"]
            if type(response) is not dict or "error" in response:
                raise ValueError("profile blocking response is not a completion")
            identity = response.get("id")
            usage = _profile_usage(
                response.get("usage"), context, requested, decode=False
            )
            choices = response.get("choices")
            if (
                type(choices) is not list
                or len(choices) != 1
                or type(choices[0]) is not dict
            ):
                raise ValueError("profile blocking choice census differs")
            choice = choices[0]
            if type(choice.get("index")) is not int or choice["index"] != 0:
                raise ValueError("profile blocking choice index differs")
            if choice.get("finish_reason") != "length":
                raise ValueError(
                    "profile blocking finish reason differs from ignore_eos budget"
                )
            ids, text = choice.get("token_ids"), choice.get("text")
            if (
                type(ids) is not list
                or len(ids) != requested
                or type(text) is not str
                or any(
                    type(token) is not int or not 0 <= token <= 0xFFFFFFFF
                    for token in ids
                )
            ):
                raise ValueError("profile scheduler-owned token IDs differ")
            if kind == "exact":
                if choice.get("finish_reason") != "length":
                    raise ValueError("profile blocking finish reason differs")
                retrieved = {
                    name: passkey in text
                    for name, passkey in {
                        "ALPHA": "alpha-0731826",
                        "BETA": "beta-0492175",
                        "GAMMA": "gamma-0863204",
                    }.items()
                }
                arm[kind] = {
                    "completion_tokens": requested,
                    "finish_reason": "length",
                    "text_sha256": sha256_bytes(text.encode()),
                    "token_ids_sha256": sha256_bytes(
                        b"".join(token.to_bytes(4, "little") for token in ids)
                    ),
                    "passkeys_retrieved": retrieved,
                    "accuracy_pass": all(retrieved.values()),
                    "time_to_first_token_ms": usage.get("time_to_first_token_ms"),
                }
                arm["first_token_id_exact"] = ids[0]
            else:
                server_ms = _raw_positive_number(
                    usage, "time_to_first_token_ms", "profile TTFT"
                )
                if server_ms > wall * 1000.0 + 1.0:
                    raise ValueError("profile engine TTFT exceeds client request wall")
                arm[kind] = {
                    "prefill_prompt_tokens": context,
                    "prefill_ttft_seconds": wall,
                    "client_request_wall_prompt_tokens_per_second": context / wall,
                    "prefill_clock": PROFILE_MEASUREMENT_CONTRACT["prefill_clock"],
                    "client_monotonic": record["client_monotonic"],
                    "server_time_to_first_token_ms": server_ms,
                    "first_token_id": ids[0],
                }
        if type(identity) is not str or not identity or identity in request_ids:
            raise ValueError("profile response ID is missing or reused")
        request_ids.add(identity)
    return arms


def _validate_profile_measurements(
    raw: dict[str, Any], manifest: dict[str, Any], container_id: str
) -> None:
    plan = _profile_plan(manifest)
    pairing = PROFILE_PAIRINGS[
        PROFILE_PLAN_CONTRACTS.get(plan.get("schema"), "flux-performance-profile-contract/1")
    ]
    chunk_key = pairing["chunk_key"]
    _exact_object(
        raw,
        {
            "measurement_schema",
            "measurement_contract",
            "profile_contract",
            chunk_key,
            "rows",
            "all_host_tolerances_met",
            "all_accuracy_checks_pass",
            "host_reference_sha256",
            "host_reference",
            "profile_identity",
            "host_bands",
            "observations",
            "response_journal",
            "process_bindings",
            "verification_metrics",
        },
        "profile raw measurements",
    )
    if raw["measurement_schema"] != pairing["metrics"]:
        raise ValueError("profile raw measurement schema differs")
    _profile_equal(
        raw["measurement_contract"],
        PROFILE_MEASUREMENT_CONTRACT,
        "measurement clock/source contract",
    )
    contract = _profile_contract(raw["profile_contract"], plan)
    v2 = contract["schema"] == PROFILE_CONTRACT_V2
    chunks = {name: _profile_chunk(contract, name) for name in PROFILE_NAMES}
    _profile_equal(
        raw[chunk_key], chunks if v2 else 4096, "validator raw chunk census"
    )
    _profile_equal(
        contract, plan.get("performance_profile_contract"), "sealed profile contract"
    )
    _profile_runtime_recipe_binding(contract, manifest["image"])
    host = _exact_object(
        raw["host_reference"],
        {
            "schema",
            chunk_key,
            "binary_sha256",
            "decode_budget_tokens",
            "profiles",
            "measurement_contract",
            "provenance",
        },
        "profile host reference",
    )
    if host["schema"] != pairing["host_reference"]:
        raise ValueError("profile host reference schema differs")
    _profile_equal(
        host["measurement_contract"],
        PROFILE_MEASUREMENT_CONTRACT,
        "host clock/source contract",
    )
    _profile_equal(
        host["provenance"],
        {key: manifest["provenance"][key] for key in PROFILE_PROVENANCE_KEYS},
        "host source/checkpoint provenance",
    )
    _profile_equal(host[chunk_key], chunks if v2 else 4096, "host chunk census")
    _profile_equal(host["decode_budget_tokens"], 256, "host output budget")
    if host["binary_sha256"] != manifest["image"]["in_container_binary_sha256"]:
        raise ValueError("profile host binary differs")
    expected_hash = _nonzero_hex(
        _profile_option(plan, "--host-reference-sha256"),
        64,
        "pinned host reference hash",
    )
    if (
        raw["host_reference_sha256"] != expected_hash
        or sha256_bytes(canonical_bytes(host)) != expected_hash
    ):
        raise ValueError("profile host reference bytes/hash differ from pinned plan")
    profiles = _exact_object(host["profiles"], set(PROFILE_NAMES), "host profiles")
    host_rows = {}
    for name in PROFILE_NAMES:
        profile = _exact_object(
            profiles[name],
            {"argv_overlay", "host_recipe_sha256", "rows"}
            | ({"launch_argv_sha256"} if v2 else set()),
            "host profile",
        )
        binding = contract["profiles"][name]
        if v2:
            # The host reference measured the same full launch argv per mode.
            _profile_equal(
                profile["launch_argv_sha256"],
                binding["launch_argv_sha256"],
                "host full launch argv",
            )
        _profile_equal(
            profile["argv_overlay"], binding["argv_overlay"], "host profile overlay"
        )
        if profile["host_recipe_sha256"] != binding["recipe_sha256"]:
            raise ValueError("profile host recipe differs from source policy")
        rows = profile["rows"]
        if type(rows) is not list or len(rows) != 6:
            raise ValueError("profile host row census differs")
        for item in rows:
            row = _exact_object(
                item,
                {
                    "context_tokens",
                    "decode_tokens",
                    "k5_decode_wall_seconds",
                    "plain_decode_wall_seconds",
                    "prefill_wall_seconds",
                    "prompt_sha256",
                    "k5_usage",
                    "plain_usage",
                    "prefill_client_monotonic",
                },
                "profile host row",
            )
            context = row["context_tokens"]
            if (
                type(context) is not int
                or context not in PROFILE_CONTEXTS
                or (name, context) in host_rows
            ):
                raise ValueError("profile host context census differs")
            budget = min(256, 262144 - context)
            _profile_equal(
                row["decode_tokens"], budget - 1, "host timed decode token budget"
            )
            _nonzero_hex(row["prompt_sha256"], 64, "host workload prompt hash")
            for mode in ("plain", "k5"):
                measured = _profile_usage(
                    row[mode + "_usage"], context, budget, decode=True
                )
                _profile_equal(
                    row[mode + "_decode_wall_seconds"],
                    measured["decode_wall_seconds"],
                    "host source-derived decode wall",
                )
            _profile_equal(
                row["prefill_wall_seconds"],
                _profile_clock(row["prefill_client_monotonic"]),
                "host request wall",
            )
            host_rows[(name, context)] = row
    processes = _profile_processes(raw, plan, manifest, container_id, contract)
    identity = _exact_object(
        raw["profile_identity"],
        {"image_recipe_sha256", "profiles", "profiles_share_argv"},
        "profile identity",
    )
    _profile_equal(
        identity["image_recipe_sha256"],
        manifest["image"]["recipe_sha256"],
        "image recipe identity",
    )
    _profile_equal(identity["profiles_share_argv"], False, "distinct profile argv")
    identities = _exact_object(
        identity["profiles"], set(PROFILE_NAMES), "profile identity census"
    )
    for name in PROFILE_NAMES:
        binding = contract["profiles"][name]
        expected = {
            **binding,
            "host_recipe_sha256": binding["recipe_sha256"],
            **{
                key: manifest["provenance"][key]
                for key in (
                    "source_revision",
                    "source_tree",
                    "source_bundle_manifest_sha256",
                )
            },
            "effective_overlay": (
                _profile_overlay_v2(binding["argv_overlay"], chunks[name])
                if v2
                else _profile_overlay(binding["argv_overlay"])
            ),
            "process_proof_sha256": {
                mode: sha256_bytes(canonical_bytes(processes[(name, mode)]["proof"]))
                for mode in ("none", "native-mtp-k5")
            },
        }
        _profile_equal(
            identities[name], expected, "source/recipe/argv/process identity"
        )
    arms = _profile_journal(raw, manifest, processes, host_rows, contract)
    observations = _raw_list(raw, "observations", "profile", 12)
    bands = _raw_list(raw, "host_bands", "profile", 12)
    rows = _raw_list(raw, "rows", "profile", 12)
    if len(observations) != 12 or len(bands) != 12 or len(rows) != 12:
        raise ValueError("profile observation/band/raw row census differs")
    expected_observations, expected_bands, expected_rows = [], [], []
    for name in PROFILE_NAMES:
        for context in PROFILE_CONTEXTS:
            host_row = host_rows[(name, context)]
            plain, native = (
                arms[(name, context, mode)] for mode in ("none", "native-mtp-k5")
            )
            for arm in (plain, native):
                if (
                    arm["exact"]["text_sha256"] != arm["stream"]["streamed_text_sha256"]
                    or arm["ttft"]["first_token_id"] != arm["first_token_id_exact"]
                    or arm["exact"]["accuracy_pass"] is not True
                ):
                    raise ValueError(
                        "profile exact/stream/first-token/accuracy identity differs"
                    )
            for key in ("token_ids_sha256", "text_sha256", "finish_reason"):
                if plain["exact"][key] != native["exact"][key]:
                    raise ValueError("profile plain/native K5 output is not lossless")
            if plain["first_token_id_exact"] != native["first_token_id_exact"]:
                raise ValueError("profile plain/native first token differs")
            expected_observations.append(
                {
                    "profile": name,
                    "context_tokens": context,
                    "prompt_sha256": host_row["prompt_sha256"],
                    "plain": plain,
                    "native_k5": native,
                }
            )
            values = {
                "profile": name,
                "context_tokens": context,
                "prefill_prompt_tokens": context,
                "prefill_ttft_seconds": native["ttft"]["prefill_ttft_seconds"],
                "plain_decode_tokens": plain["stream"]["decode_tokens"],
                "plain_decode_wall_seconds": plain["stream"]["decode_wall_seconds"],
                "k5_decode_tokens": native["stream"]["decode_tokens"],
                "k5_decode_wall_seconds": native["stream"]["decode_wall_seconds"],
                "host_decode_wall_seconds": host_row["k5_decode_wall_seconds"],
                "profile_decode_wall_seconds": native["stream"]["decode_wall_seconds"],
                "k5_lossless": True,
                "accuracy_pass": True,
            }
            if v2:
                values["chunk_tokens"] = chunks[name]
                values["recipe_sha256"] = contract["profiles"][name]["recipe_sha256"]
            expected_rows.append(values)
            k5_wall, plain_wall, prefill_wall = (
                values[key]
                for key in (
                    "k5_decode_wall_seconds",
                    "plain_decode_wall_seconds",
                    "prefill_ttft_seconds",
                )
            )
            deltas = [
                100.0 * (host_row[key] - wall) / wall
                for key, wall in (
                    ("k5_decode_wall_seconds", k5_wall),
                    ("plain_decode_wall_seconds", plain_wall),
                    ("prefill_wall_seconds", prefill_wall),
                )
            ]
            if any(not math.isfinite(delta) or abs(delta) > 5.0 for delta in deltas):
                raise ValueError("profile host decode/plain/prefill tolerance exceeded")
            expected_bands.append(
                {
                    "profile": name,
                    "context_tokens": context,
                    "decode_clock": PROFILE_MEASUREMENT_CONTRACT["decode_clock"],
                    "prefill_clock": PROFILE_MEASUREMENT_CONTRACT["prefill_clock"],
                    "k5_decode_delta_percent": deltas[0],
                    "plain_decode_delta_percent": deltas[1],
                    "prefill_delta_percent": deltas[2],
                    "host_k5_decode_wall_seconds": host_row["k5_decode_wall_seconds"],
                    "host_plain_decode_wall_seconds": host_row[
                        "plain_decode_wall_seconds"
                    ],
                    "host_client_prefill_wall_seconds": host_row[
                        "prefill_wall_seconds"
                    ],
                    "measured_k5_decode_wall_seconds": k5_wall,
                    "measured_plain_decode_wall_seconds": plain_wall,
                    "measured_client_prefill_wall_seconds": prefill_wall,
                }
            )
    _profile_equal(observations, expected_observations, "journal-derived observations")
    _profile_equal(rows, expected_rows, "journal-derived raw rows")
    _profile_equal(bands, expected_bands, "independently derived host bands")


def _recompute_profile_evidence(
    evidence: dict[str, Any],
    raw: dict[str, Any],
    label: str,
    manifest: dict[str, Any],
    container_id: str,
) -> None:
    _validate_profile_measurements(raw, manifest, container_id)
    v2 = "profile_chunk_tokens" in evidence
    if v2:
        if raw.get("profile_chunk_tokens") != evidence["profile_chunk_tokens"]:
            raise ValueError(f"{label} chunk census differs from the validator raw")
    elif _raw_int(raw, "chunk_tokens", label, 1, 1_000_000) != evidence["chunk_tokens"]:
        raise ValueError(f"{label} chunk census differs from the validator raw")
    rows_raw = _raw_list(raw, "rows", label, 64)
    rows_evidence = evidence["rows"]
    if len(rows_raw) != len(rows_evidence):
        raise ValueError(f"{label} row census differs from the validator raw")
    for index, (row_value, row_evidence) in enumerate(zip(rows_raw, rows_evidence)):
        row = _exact_object(
            row_value,
            {
                "profile",
                "context_tokens",
                "prefill_prompt_tokens",
                "prefill_ttft_seconds",
                "plain_decode_tokens",
                "plain_decode_wall_seconds",
                "k5_decode_tokens",
                "k5_decode_wall_seconds",
                "host_decode_wall_seconds",
                "profile_decode_wall_seconds",
                "k5_lossless",
                "accuracy_pass",
            }
            | ({"chunk_tokens", "recipe_sha256"} if v2 else set()),
            f"{label} validator raw rows[{index}]",
        )
        if (
            _raw_str(row, "profile", label, row_evidence["profile"])
            != row_evidence["profile"]
            or _raw_int(row, "context_tokens", label, 1, 1_000_000)
            != row_evidence["context_tokens"]
        ):
            raise ValueError(f"{label} row identity differs from the validator raw")
        prefill_tokens = _raw_int(row, "prefill_prompt_tokens", label, 1, 1_000_000)
        prefill_wall = _raw_positive_number(row, "prefill_ttft_seconds", label)
        plain_tokens = _raw_int(row, "plain_decode_tokens", label, 1, 100_000_000)
        plain_wall = _raw_positive_number(row, "plain_decode_wall_seconds", label)
        k5_tokens = _raw_int(row, "k5_decode_tokens", label, 1, 100_000_000)
        k5_wall = _raw_positive_number(row, "k5_decode_wall_seconds", label)
        host_wall = _raw_positive_number(row, "host_decode_wall_seconds", label)
        profile_wall = _raw_positive_number(row, "profile_decode_wall_seconds", label)
        context = row["context_tokens"]
        if (
            prefill_tokens != context
            or plain_tokens != min(256, 262144 - context) - 1
            or k5_tokens != plain_tokens
            or profile_wall != k5_wall
        ):
            raise ValueError(f"{label} workload budget or duplicate wall differs")
        derived = {
            "prefill_tokens_per_second": prefill_tokens / prefill_wall,
            "plain_decode_tokens_per_second": plain_tokens / plain_wall,
            "k5_decode_tokens_per_second": k5_tokens / k5_wall,
            "host_delta_percent": 100.0 * (host_wall - profile_wall) / profile_wall,
        }
        for key, expected_value in derived.items():
            observed_value = row_evidence[key]
            if (
                type(observed_value) is not float
                or type(expected_value) is not float
                or observed_value != expected_value
            ):
                raise ValueError(
                    f"{label} derived {key} differs from the validator raw"
                )
        if (
            _raw_bool(row, "k5_lossless", label) is not True
            or _raw_bool(row, "accuracy_pass", label) is not True
        ):
            raise ValueError(f"{label} row checks differ from the validator raw")
    if (
        _raw_bool(raw, "all_host_tolerances_met", label) is not True
        or _raw_bool(raw, "all_accuracy_checks_pass", label) is not True
    ):
        raise ValueError(f"{label} aggregates differ from the validator raw")


def _raw_positive_number(raw: dict[str, Any], key: str, label: str) -> float:
    value = raw.get(key)
    if type(value) is not float and type(value) is not int:
        raise ValueError(f"{label} validator raw `{key}` is not a number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} validator raw `{key}` is not a positive number")
    return number


def _recompute_memory_evidence(
    evidence: dict[str, Any], raw: dict[str, Any], label: str
) -> None:
    samples = _raw_list(raw, "samples", label, 10_000)
    if not samples:
        raise ValueError(f"{label} validator raw samples are empty")
    available: list[int] = []
    used: list[int] = []
    census = {
        "allocator_warning_count": 0,
        "cuda_oom_count": 0,
        "nvrm_count": 0,
        "xid_count": 0,
    }
    for index, sample_value in enumerate(samples):
        sample = _exact_object(
            sample_value,
            {
                "available_mib",
                "used_mib",
                "allocator_warnings",
                "cuda_oom_count",
                "nvrm_count",
                "xid_count",
            },
            f"{label} validator raw samples[{index}]",
        )
        available.append(_raw_int(sample, "available_mib", label, 0, 10_000_000))
        used.append(_raw_int(sample, "used_mib", label, 0, 10_000_000))
        census["allocator_warning_count"] += _raw_int(
            sample, "allocator_warnings", label, 0, 1_000_000
        )
        census["cuda_oom_count"] += _raw_int(
            sample, "cuda_oom_count", label, 0, 1_000_000
        )
        census["nvrm_count"] += _raw_int(sample, "nvrm_count", label, 0, 1_000_000)
        census["xid_count"] += _raw_int(sample, "xid_count", label, 0, 1_000_000)
    observed = {
        "sample_count": len(samples),
        "minimum_available_mib": min(available),
        "peak_used_mib": max(used),
        **census,
    }
    for key, value in observed.items():
        if evidence[key] != value:
            raise ValueError(f"{label} {key} differs from the validator raw")


def _recompute_cancellation_evidence(
    evidence: dict[str, Any], raw: dict[str, Any], label: str
) -> None:
    concurrency = _raw_int(raw, "concurrency", label, 1, 256)
    slots = _raw_list(raw, "slots", label, 256)
    if len(slots) != concurrency or concurrency != evidence["concurrency"]:
        raise ValueError(f"{label} concurrency census differs from the validator raw")
    cancelled = 0
    survivors = 0
    for index, slot_value in enumerate(slots):
        slot = _exact_object(
            slot_value,
            {"cancelled", "survivor"},
            f"{label} validator raw slots[{index}]",
        )
        cancelled_flag = _raw_bool(slot, "cancelled", label)
        survivor_flag = _raw_bool(slot, "survivor", label)
        if cancelled_flag == survivor_flag:
            raise ValueError(f"{label} validator raw slot outcome is ambiguous")
        if cancelled_flag:
            cancelled += 1
        else:
            survivors += 1
    replacements_raw = _raw_list(raw, "replacement_requests", label, 256)
    if (
        len(replacements_raw) != 1
        or any(
            type(value) is not bool or value is not True for value in replacements_raw
        )
        or _raw_bool(raw, "replacement_survivor", label) is not True
    ):
        raise ValueError(f"{label} replacement census differs from the validator raw")
    if (
        cancelled != evidence["cancelled_requests"]
        or survivors != evidence["survivors"]
        or len(replacements_raw) != evidence["replacement_requests"]
        or len(slots) - cancelled != survivors
        or _raw_bool(raw, "plain_and_k5", label) is not True
        or _raw_bool(raw, "exact_survivor_identity", label) is not True
        or _raw_bool(raw, "final_highest_slot_reused", label) is not True
        or _raw_bool(raw, "post_drain_clean", label) is not True
    ):
        raise ValueError(f"{label} census differs from the validator raw")


def _recompute_raw_log_evidence(
    evidence: dict[str, Any], raw: dict[str, Any], label: str
) -> None:
    counts = _raw_object(raw, "per_pattern_match_counts", label)
    if set(counts.keys()) != set(RAW_LOG_PATTERNS):
        raise ValueError(f"{label} validator raw pattern census keys differ")
    for pattern_label, count in counts.items():
        if type(count) is not int or count != 0:
            raise ValueError(f"{label} validator raw pattern census differs")
    if (
        _raw_bool(raw, "all_processes_included", label) is not True
        or _raw_str(raw, "scan_scope", label, evidence["scan_scope"])
        != evidence["scan_scope"]
        or _raw_int(raw, "raw_log_bytes", label, 1, MAX_RAW_LOG_BYTES)
        != evidence["raw_log_bytes"]
    ):
        raise ValueError(f"{label} census differs from the validator raw")


def _validate_startup_evidence(
    value: object, manifest: dict[str, Any], container_id: str
) -> dict[str, Any]:
    label = "startup-readiness evidence"
    evidence = _exact_object(
        value,
        {
            "fresh_process_modes",
            "ready_http_statuses",
            "model_identity_match",
            "checkpoint_identity_match",
            "rootfs_read_only",
            "plain_scheduler_ready",
            "k5_scheduler_ready",
            "process_executable_sha256",
            "validator_artifact",
            "validator_receipt_sha256",
        },
        label,
    )
    if evidence["fresh_process_modes"] != ["none", "native-mtp-k5"]:
        raise ValueError(f"{label}.fresh_process_modes differs")
    if evidence["ready_http_statuses"] != [200, 200] or any(
        type(item) is not int for item in evidence["ready_http_statuses"]
    ):
        raise ValueError(f"{label}.ready_http_statuses differs")
    for key in (
        "model_identity_match",
        "checkpoint_identity_match",
        "rootfs_read_only",
        "plain_scheduler_ready",
        "k5_scheduler_ready",
    ):
        _exact_true(evidence[key], f"{label}.{key}")
    if (
        evidence["process_executable_sha256"]
        != manifest["image"]["in_container_binary_sha256"]
    ):
        raise ValueError(f"{label}.process_executable_sha256 differs")
    raw = _validator_document(evidence, "startup-readiness", manifest, container_id)
    _recompute_startup_evidence(evidence, raw, label)
    return evidence


def _validate_tool_evidence(
    value: object, _manifest: dict[str, Any], container_id: str
) -> dict[str, Any]:
    label = "tool-api-streaming-blocking evidence"
    evidence = _exact_object(
        value,
        {
            "blocking_cases",
            "streaming_cases",
            "tool_calls_expected",
            "tool_calls_observed",
            "exact_arguments",
            "finish_reasons_valid",
            "no_blank_line_runaway",
            "no_all_tools_runaway",
            "thinking_spacing_valid",
            "validator_artifact",
            "validator_receipt_sha256",
        },
        label,
    )
    for key in (
        "blocking_cases",
        "streaming_cases",
        "tool_calls_expected",
        "tool_calls_observed",
    ):
        _plain_int(evidence[key], f"{label}.{key}", 1, 1_000_000)
    if evidence["tool_calls_observed"] != evidence["tool_calls_expected"]:
        raise ValueError(f"{label} tool-call census differs")
    for key in (
        "exact_arguments",
        "finish_reasons_valid",
        "no_blank_line_runaway",
        "no_all_tools_runaway",
        "thinking_spacing_valid",
    ):
        _exact_true(evidence[key], f"{label}.{key}")
    raw = _validator_document(
        evidence, "tool-api-streaming-blocking", _manifest, container_id
    )
    _recompute_tool_evidence(evidence, raw, label)
    return evidence


def _validate_accuracy_evidence(
    value: object,
    *,
    manifest: dict[str, Any],
    container_id: str,
    gate: str,
    dataset_sha256: str,
    items: int,
    floor: float,
) -> dict[str, Any]:
    label = f"{gate} evidence"
    evidence = _exact_object(
        value,
        {
            "dataset_sha256",
            "items",
            "no_spec_correct",
            "k5_correct",
            "score",
            "provisional_floor",
            "per_item_answer_identity",
            "per_item_correctness_identity",
            "per_item_token_identity",
            "validator_artifact",
            "validator_receipt_sha256",
        },
        label,
    )
    if evidence["dataset_sha256"] != dataset_sha256:
        raise ValueError(f"{label}.dataset_sha256 differs")
    _plain_int(evidence["items"], f"{label}.items", items, items)
    no_spec = _plain_int(
        evidence["no_spec_correct"], f"{label}.no_spec_correct", 0, items
    )
    k5 = _plain_int(evidence["k5_correct"], f"{label}.k5_correct", 0, items)
    if no_spec != k5:
        raise ValueError(f"{label} plain and K5 correct counts differ")
    score = _finite_number(evidence["score"], f"{label}.score", minimum=floor)
    observed_floor = _finite_number(
        evidence["provisional_floor"], f"{label}.provisional_floor"
    )
    if not math.isclose(observed_floor, floor, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label}.provisional_floor differs")
    if not math.isclose(score, no_spec / items, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label}.score differs from the exact item census")
    for key in (
        "per_item_answer_identity",
        "per_item_correctness_identity",
        "per_item_token_identity",
    ):
        _exact_true(evidence[key], f"{label}.{key}")
    raw = _validator_document(evidence, gate, manifest, container_id)
    _recompute_accuracy_evidence(evidence, raw, label, dataset_sha256, items, floor)
    return evidence


def _validate_long_context_evidence(
    value: object, _manifest: dict[str, Any], container_id: str
) -> dict[str, Any]:
    label = "plain-k5-long-context evidence"
    evidence = _exact_object(
        value,
        {
            "case_count",
            "max_prompt_tokens",
            "max_total_tokens",
            "prompt_token_counts",
            "plain_k5_token_identity",
            "plain_k5_text_identity",
            "plain_k5_finish_reason_identity",
            "plain_k5_usage_identity",
            "native_k5_engaged",
            "partial_accept_cycle",
            "qsa_verify_rows_engaged",
            "validator_artifact",
            "validator_receipt_sha256",
        },
        label,
    )
    _plain_int(
        evidence["case_count"],
        f"{label}.case_count",
        len(LONG_CONTEXT_PROMPT_TOKENS),
        len(LONG_CONTEXT_PROMPT_TOKENS),
    )
    if (
        evidence["max_prompt_tokens"] != 262_080
        or type(evidence["max_prompt_tokens"]) is not int
    ):
        raise ValueError(f"{label}.max_prompt_tokens differs")
    if (
        evidence["max_total_tokens"] != 262_144
        or type(evidence["max_total_tokens"]) is not int
    ):
        raise ValueError(f"{label}.max_total_tokens differs")
    if evidence["prompt_token_counts"] != list(LONG_CONTEXT_PROMPT_TOKENS) or any(
        type(item) is not int for item in evidence["prompt_token_counts"]
    ):
        raise ValueError(f"{label}.prompt_token_counts differs")
    for key in (
        "plain_k5_token_identity",
        "plain_k5_text_identity",
        "plain_k5_finish_reason_identity",
        "plain_k5_usage_identity",
        "native_k5_engaged",
        "partial_accept_cycle",
        "qsa_verify_rows_engaged",
    ):
        _exact_true(evidence[key], f"{label}.{key}")
    raw = _validator_document(
        evidence, "plain-k5-long-context", _manifest, container_id
    )
    _recompute_long_context_evidence(evidence, raw, label)
    return evidence


def _validate_native_262k_evidence(
    value: object, _manifest: dict[str, Any], container_id: str
) -> dict[str, Any]:
    label = "native-262k evidence"
    evidence = _exact_object(
        value,
        {
            "kv_dtype",
            "max_seq_len",
            "retrieval_prompt_tokens",
            "boundary_prompt_tokens",
            "overflow_prompt_tokens",
            "passkeys_per_prompt",
            "all_retrieval_answers_exact",
            "boundary_admission_pass",
            "overflow_rejection_pass",
            "qsa_terminal_chunks_exact",
            "validator_artifact",
            "validator_receipt_sha256",
        },
        label,
    )
    expected = {
        "kv_dtype": "bf16",
        "max_seq_len": 262_144,
        "retrieval_prompt_tokens": [258_000, 261_888, 262_080],
        "boundary_prompt_tokens": 262_143,
        "overflow_prompt_tokens": 262_144,
        "passkeys_per_prompt": 3,
    }
    for key, wanted in expected.items():
        if evidence[key] != wanted or type(evidence[key]) is not type(wanted):
            raise ValueError(f"{label}.{key} differs")
    if any(type(item) is not int for item in evidence["retrieval_prompt_tokens"]):
        raise ValueError(f"{label}.retrieval_prompt_tokens contains non-integers")
    for key in (
        "all_retrieval_answers_exact",
        "boundary_admission_pass",
        "overflow_rejection_pass",
        "qsa_terminal_chunks_exact",
    ):
        _exact_true(evidence[key], f"{label}.{key}")
    raw = _validator_document(evidence, "native-262k", _manifest, container_id)
    _recompute_native_262k_evidence(evidence, raw, label)
    _validate_native_context_image_binding(_manifest, "native-262k", NATIVE_CONTEXT_RECIPES["native-262k"]["recipe_sha256"])
    _native_context_required_module().validate_native_context_process_bindings(raw, _manifest, container_id, "native-262k")
    return evidence


def _validate_concurrency_evidence(
    value: object, _manifest: dict[str, Any], container_id: str
) -> dict[str, Any]:
    label = "native-concurrency-c2-c8 evidence"
    evidence = _exact_object(
        value,
        {
            "levels",
            "modes",
            "fresh_processes",
            "minimum_full_frames",
            "forward_reverse_order_all_frames",
            "per_frame_logit_identity",
            "per_frame_state_identity",
            "cancellation_replacement_pass",
            "final_highest_slot_reused",
            "post_drain_clean",
            "minimum_available_mib",
            "validator_artifact",
            "validator_receipt_sha256",
        },
        label,
    )
    if evidence["levels"] != [2, 4, 8] or any(
        type(item) is not int for item in evidence["levels"]
    ):
        raise ValueError(f"{label}.levels differs")
    if evidence["modes"] != ["none", "native-mtp-k5"]:
        raise ValueError(f"{label}.modes differs")
    _plain_int(evidence["fresh_processes"], f"{label}.fresh_processes", 18, 256)
    _plain_int(
        evidence["minimum_full_frames"],
        f"{label}.minimum_full_frames",
        8,
        1_000_000,
    )
    for key in (
        "forward_reverse_order_all_frames",
        "per_frame_logit_identity",
        "per_frame_state_identity",
        "cancellation_replacement_pass",
        "final_highest_slot_reused",
        "post_drain_clean",
    ):
        _exact_true(evidence[key], f"{label}.{key}")
    _plain_int(
        evidence["minimum_available_mib"],
        f"{label}.minimum_available_mib",
        1,
        10_000_000,
    )
    raw = _validator_document(
        evidence, "native-concurrency-c2-c8", _manifest, container_id
    )
    _recompute_concurrency_evidence(evidence, raw, label)
    return evidence


def _validate_profile_evidence(
    value: object, _manifest: dict[str, Any], container_id: str
) -> dict[str, Any]:
    label = "prefill-decode-profiles evidence"
    chunk_key = (
        "profile_chunk_tokens"
        if type(value) is dict and "profile_chunk_tokens" in value
        else "chunk_tokens"
    )
    evidence = _exact_object(
        value,
        {
            chunk_key,
            "contexts",
            "profiles",
            "rows",
            "all_host_tolerances_met",
            "all_accuracy_checks_pass",
            "validator_artifact",
            "validator_receipt_sha256",
        },
        label,
    )
    if chunk_key == "profile_chunk_tokens":
        chunks = evidence[chunk_key]
        if (
            type(chunks) is not dict
            or chunks != PROFILE_CHUNKS_V2
            or any(type(item) is not int for item in chunks.values())
        ):
            raise ValueError(f"{label}.profile_chunk_tokens differs")
    elif evidence["chunk_tokens"] != 4_096 or type(evidence["chunk_tokens"]) is not int:
        raise ValueError(f"{label}.chunk_tokens differs")
    if evidence["contexts"] != list(PROFILE_CONTEXTS) or any(
        type(item) is not int for item in evidence["contexts"]
    ):
        raise ValueError(f"{label}.contexts differs")
    if evidence["profiles"] != list(PROFILE_NAMES):
        raise ValueError(f"{label}.profiles differs")
    rows = evidence["rows"]
    if type(rows) is not list or len(rows) != len(PROFILE_CONTEXTS) * len(
        PROFILE_NAMES
    ):
        raise ValueError(f"{label}.rows has the wrong cardinality")
    seen: set[tuple[str, int]] = set()
    for index, row_value in enumerate(rows):
        row = _exact_object(
            row_value,
            {
                "profile",
                "context_tokens",
                "prefill_tokens_per_second",
                "plain_decode_tokens_per_second",
                "k5_decode_tokens_per_second",
                "k5_lossless",
                "host_delta_percent",
            },
            f"{label}.rows[{index}]",
        )
        key = (row["profile"], row["context_tokens"])
        if (
            key
            not in {
                (profile, context)
                for profile in PROFILE_NAMES
                for context in PROFILE_CONTEXTS
            }
            or key in seen
        ):
            raise ValueError(f"{label}.rows contains an unexpected or duplicate key")
        seen.add(key)
        for metric in (
            "prefill_tokens_per_second",
            "plain_decode_tokens_per_second",
            "k5_decode_tokens_per_second",
        ):
            _finite_number(row[metric], f"{label}.rows[{index}].{metric}", minimum=0.01)
        _finite_number(
            row["host_delta_percent"],
            f"{label}.rows[{index}].host_delta_percent",
            minimum=-5.0,
            maximum=5.0,
        )
        _exact_true(row["k5_lossless"], f"{label}.rows[{index}].k5_lossless")
    for key in ("all_host_tolerances_met", "all_accuracy_checks_pass"):
        _exact_true(evidence[key], f"{label}.{key}")
    raw = _validator_document(
        evidence, "prefill-decode-profiles", _manifest, container_id
    )
    _recompute_profile_evidence(evidence, raw, label, _manifest, container_id)
    return evidence


def _validate_memory_evidence(
    value: object, _manifest: dict[str, Any], container_id: str
) -> dict[str, Any]:
    label = "memory-floor evidence"
    evidence = _exact_object(
        value,
        {
            "sample_count",
            "minimum_available_mib",
            "peak_used_mib",
            "allocator_warning_count",
            "cuda_oom_count",
            "nvrm_count",
            "xid_count",
            "validator_artifact",
            "validator_receipt_sha256",
        },
        label,
    )
    _plain_int(evidence["sample_count"], f"{label}.sample_count", 3, 10_000_000)
    _plain_int(
        evidence["minimum_available_mib"],
        f"{label}.minimum_available_mib",
        1,
        10_000_000,
    )
    _plain_int(evidence["peak_used_mib"], f"{label}.peak_used_mib", 1, 10_000_000)
    for key in (
        "allocator_warning_count",
        "cuda_oom_count",
        "nvrm_count",
        "xid_count",
    ):
        _exact_zero(evidence[key], f"{label}.{key}")
    raw = _validator_document(evidence, "memory-floor", _manifest, container_id)
    _recompute_memory_evidence(evidence, raw, label)
    return evidence


def _validate_cancellation_evidence(
    value: object, _manifest: dict[str, Any], container_id: str
) -> dict[str, Any]:
    label = "cancellation-slot-reuse evidence"
    evidence = _exact_object(
        value,
        {
            "concurrency",
            "cancelled_requests",
            "survivors",
            "replacement_requests",
            "plain_and_k5",
            "exact_survivor_identity",
            "final_highest_slot_reused",
            "post_drain_clean",
            "validator_artifact",
            "validator_receipt_sha256",
        },
        label,
    )
    exact_integers = {
        "concurrency": 8,
        "cancelled_requests": 1,
        "survivors": 7,
        "replacement_requests": 1,
    }
    for key, wanted in exact_integers.items():
        if evidence[key] != wanted or type(evidence[key]) is not int:
            raise ValueError(f"{label}.{key} differs")
    for key in (
        "plain_and_k5",
        "exact_survivor_identity",
        "final_highest_slot_reused",
        "post_drain_clean",
    ):
        _exact_true(evidence[key], f"{label}.{key}")
    raw = _validator_document(
        evidence, "cancellation-slot-reuse", _manifest, container_id
    )
    _recompute_cancellation_evidence(evidence, raw, label)
    return evidence


def _validate_raw_log_evidence(
    value: object, _manifest: dict[str, Any], container_id: str
) -> dict[str, Any]:
    label = "raw-log-safety evidence"
    evidence = _exact_object(
        value,
        {
            "scan_scope",
            "patterns",
            "match_count",
            "raw_log_bytes",
            "raw_log_sha256",
            "all_processes_included",
            "validator_artifact",
            "validator_receipt_sha256",
        },
        label,
    )
    if evidence["scan_scope"] != "complete-uncapped":
        raise ValueError(f"{label}.scan_scope differs")
    if evidence["patterns"] != list(RAW_LOG_PATTERNS):
        raise ValueError(f"{label}.patterns differs")
    _exact_zero(evidence["match_count"], f"{label}.match_count")
    _plain_int(
        evidence["raw_log_bytes"], f"{label}.raw_log_bytes", 1, MAX_RAW_LOG_BYTES
    )
    _nonzero_hex(evidence["raw_log_sha256"], 64, f"{label}.raw_log_sha256")
    _exact_true(evidence["all_processes_included"], f"{label}.all_processes_included")
    raw = _validator_document(evidence, "raw-log-safety", _manifest, container_id)
    _recompute_raw_log_evidence(evidence, raw, label)
    return evidence


def _validate_gate_evidence(
    name: str, value: object, manifest: dict[str, Any], container_id: str
) -> dict[str, Any]:
    if name == "startup-readiness":
        return _validate_startup_evidence(value, manifest, container_id)
    if name == "tool-api-streaming-blocking":
        return _validate_tool_evidence(value, manifest, container_id)
    if name == "mmlu-pro":
        return _validate_accuracy_evidence(
            value,
            manifest=manifest,
            container_id=container_id,
            gate=name,
            dataset_sha256=MMLU_PRO_DATASET_SHA256,
            items=1_000,
            floor=0.65,
        )
    if name == "gsm8k":
        return _validate_accuracy_evidence(
            value,
            manifest=manifest,
            container_id=container_id,
            gate=name,
            dataset_sha256=GSM8K_DATASET_SHA256,
            items=1_319,
            floor=0.95,
        )
    if name == "plain-k5-long-context":
        return _validate_long_context_evidence(value, manifest, container_id)
    if name == "native-262k":
        return _validate_native_262k_evidence(value, manifest, container_id)
    if name == "native-262k-w4096-owner16":
        return _native_context_required_module().validate_required_gate_evidence(name, value, manifest, container_id)
    if name == "native-concurrency-c2-c8":
        return _validate_concurrency_evidence(value, manifest, container_id)
    if name == "prefill-decode-profiles":
        return _validate_profile_evidence(value, manifest, container_id)
    if name == "memory-floor":
        return _validate_memory_evidence(value, manifest, container_id)
    if name == "cancellation-slot-reuse":
        return _validate_cancellation_evidence(value, manifest, container_id)
    if name == "raw-log-safety":
        return _validate_raw_log_evidence(value, manifest, container_id)
    raise ValueError(f"unsupported release gate {name!r}")


def _validate_gate_artifact(
    name: str,
    locator: str,
    expected_sha256: str,
    manifest: dict[str, Any],
    container_id: str,
) -> dict[str, Any]:
    raw = _stable_beneath_bytes(
        Path(manifest["_runtime_dir"]),
        locator,
        f"gate {name} receipt",
        MAX_GATE_RECEIPT_BYTES,
    )
    observed_sha256 = sha256_bytes(raw)
    if observed_sha256 != expected_sha256:
        raise ValueError(f"gate {name} receipt SHA-256 mismatch")
    document = _exact_object(
        strict_json_loads(raw, f"gate {name} receipt"),
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
        f"gate {name} receipt",
    )
    expected = {
        "schema": GATE_SCHEMA,
        "gate": name,
        "status": "PASS",
        "release_qualified": True,
        "execution_scope": "immutable-container",
        "image_id": manifest["image"]["id"],
        "image_digest": manifest["image"]["digest"],
        "container_id": container_id,
        "run_id": manifest["launch"]["run_id"],
    }
    for key, wanted in expected.items():
        if document[key] != wanted or type(document[key]) is not type(wanted):
            raise ValueError(f"gate {name} {key} differs from exact launch identity")
    evidence = _validate_gate_evidence(
        name, document["evidence"], manifest, container_id
    )
    result = {
        "status": "PASS",
        "artifact": locator,
        "receipt_sha256": observed_sha256,
        "receipt_bytes": len(raw),
        "evidence_sha256": sha256_bytes(canonical_bytes(evidence)),
        "evidence": evidence,
    }
    if name == "prefill-decode-profiles":
        validator = _validator_document(evidence, name, manifest, container_id)
        journal = validator["response_journal"]
        payload = _stable_beneath_bytes(
            Path(manifest["_runtime_dir"]),
            journal["artifact"],
            "profile dependent response journal",
            MAX_GATE_RECEIPT_BYTES,
        )
        if sha256_bytes(payload) != journal["sha256"]:
            raise ValueError(
                "profile dependent response journal changed during capture"
            )
        result["dependent_artifacts"] = [
            {
                "artifact": journal["artifact"],
                "sha256": journal["sha256"],
                "bytes": len(payload),
            }
        ]
    return result


def _scan_stable_raw_log(root: Path, locator: str) -> dict[str, Any]:
    descriptor = _open_beneath(root, locator, "complete raw log")
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > MAX_RAW_LOG_BYTES
        ):
            raise ValueError("complete raw log must be one non-empty bounded file")
        digest = hashlib.sha256()
        matches = 0
        per_label = {label: 0 for label, _ in _RAW_LOG_REGEXES}
        with mmap.mmap(descriptor, 0, access=mmap.ACCESS_READ) as window:
            digest.update(window)
            for label, pattern in _RAW_LOG_REGEXES:
                for _match in pattern.finditer(window):
                    matches += 1
                    per_label[label] += 1
        after = os.fstat(descriptor)
        if (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_nlink,
        ) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
            before.st_nlink,
        ):
            raise ValueError("complete raw log changed while it was scanned")
        return {
            "raw_log_sha256": digest.hexdigest(),
            "raw_log_bytes": after.st_size,
            "match_count": matches,
            "per_pattern_match_counts": per_label,
        }
    finally:
        os.close(descriptor)


def _validate_suite_receipt_bytes(
    raw: bytes, manifest: dict[str, Any], container_id: str
) -> dict[str, Any]:
    value = strict_json_loads(raw, "suite receipt")
    receipt = _exact_object(
        value,
        {
            "schema",
            "status",
            "release_qualified",
            "release_version",
            "execution_scope",
            "image_id",
            "image_digest",
            "container_id",
            "run_id",
            "source_revision",
            "source_tree",
            "source_bundle_manifest_sha256",
            "in_container_binary_sha256",
            "recipe_sha256",
            "checkpoint_id",
            "checkpoint_revision",
            "checkpoint_manifest_sha256",
            "suite_plan_sha256",
            "runtime_dir_identity_sha256",
            "gates",
            "process_bindings",
            "raw_log_safety",
        },
        "suite receipt",
    )
    expected = {
        "schema": SUITE_SCHEMA,
        "status": "PASS",
        "release_qualified": True,
        "release_version": manifest["release_version"],
        "execution_scope": "immutable-container",
        "image_id": manifest["image"]["id"],
        "image_digest": manifest["image"]["digest"],
        "container_id": container_id,
        "run_id": manifest["launch"]["run_id"],
        "source_revision": manifest["provenance"]["source_revision"],
        "source_tree": manifest["provenance"]["source_tree"],
        "source_bundle_manifest_sha256": manifest["provenance"][
            "source_bundle_manifest_sha256"
        ],
        "in_container_binary_sha256": manifest["image"]["in_container_binary_sha256"],
        "recipe_sha256": manifest["image"]["recipe_sha256"],
        "checkpoint_id": manifest["provenance"]["checkpoint_id"],
        "checkpoint_revision": manifest["provenance"]["checkpoint_revision"],
        "checkpoint_manifest_sha256": manifest["provenance"][
            "checkpoint_manifest_sha256"
        ],
        "suite_plan_sha256": manifest["suite"]["plan_sha256"],
        "runtime_dir_identity_sha256": _directory_identity_sha256(
            Path(manifest["_runtime_dir"]), "suite receipt runtime_dir"
        ),
    }
    for key, wanted in expected.items():
        if receipt[key] != wanted or type(receipt[key]) is not type(wanted):
            raise ValueError(f"suite receipt {key} differs from exact launch identity")
    _validate_process_bindings(receipt["process_bindings"], manifest, container_id)
    gates = _exact_object(receipt["gates"], set(REQUIRED_GATES), "suite receipt gates")
    normalized_gates: dict[str, dict[str, Any]] = {}
    artifact_paths: set[str] = set()
    for name in REQUIRED_GATES:
        gate = _exact_object(
            gates[name],
            {"status", "receipt_relpath", "receipt_sha256"},
            f"gate {name}",
        )
        if gate["status"] != "PASS" or type(gate["status"]) is not str:
            raise ValueError(f"gate {name} did not pass")
        expected_sha256 = _nonzero_hex(
            gate["receipt_sha256"], 64, f"gate {name} receipt_sha256"
        )
        locator = _safe_artifact_locator(
            gate["receipt_relpath"], f"gate {name} receipt_relpath"
        )
        if locator in artifact_paths or locator in {
            manifest["suite"]["_receipt_relpath"],
            manifest["suite"]["_stdout_relpath"],
            manifest["suite"]["_stderr_relpath"],
        }:
            raise ValueError("gate receipt paths must be unique release artifacts")
        artifact_paths.add(locator)
        normalized = _validate_gate_artifact(
            name, locator, expected_sha256, manifest, container_id
        )
        validator_locator = normalized["evidence"]["validator_artifact"]
        if validator_locator in artifact_paths or validator_locator in {
            manifest["suite"]["_receipt_relpath"],
            manifest["suite"]["_stdout_relpath"],
            manifest["suite"]["_stderr_relpath"],
        }:
            raise ValueError("validator artifacts must be globally unique")
        artifact_paths.add(validator_locator)
        for dependency in normalized.get("dependent_artifacts", []):
            locator = dependency["artifact"]
            if locator in artifact_paths or locator in {
                manifest["suite"]["_receipt_relpath"],
                manifest["suite"]["_stdout_relpath"],
                manifest["suite"]["_stderr_relpath"],
            }:
                raise ValueError("dependent artifacts must be globally unique")
            artifact_paths.add(locator)
        normalized_gates[name] = normalized
    safety = _exact_object(
        receipt["raw_log_safety"],
        {
            "scan_scope",
            "patterns",
            "match_count",
            "per_pattern_match_counts",
            "raw_log_relpath",
            "raw_log_bytes",
            "raw_log_sha256",
        },
        "raw_log_safety",
    )
    if (
        safety["scan_scope"] != "complete-uncapped"
        or safety["patterns"] != list(RAW_LOG_PATTERNS)
        or type(safety["match_count"]) is not int
        or safety["match_count"] != 0
    ):
        raise ValueError("raw log safety is not a complete zero-match census")
    per_label = _exact_object(
        safety["per_pattern_match_counts"], set(RAW_LOG_PATTERNS), "raw_log_safety"
    )
    for label, count in per_label.items():
        if type(count) is not int or count != 0:
            raise ValueError(
                f"raw log safety per-pattern census for {label} is not zero"
            )
    expected_raw_sha256 = _nonzero_hex(
        safety["raw_log_sha256"], 64, "raw_log_safety.raw_log_sha256"
    )
    expected_raw_bytes = _plain_int(
        safety["raw_log_bytes"], "raw_log_safety.raw_log_bytes", 1, MAX_RAW_LOG_BYTES
    )
    raw_log_locator = _safe_artifact_locator(
        safety["raw_log_relpath"], "raw_log_safety.raw_log_relpath"
    )
    if raw_log_locator in artifact_paths or raw_log_locator in {
        manifest["suite"]["_receipt_relpath"],
        manifest["suite"]["_stdout_relpath"],
        manifest["suite"]["_stderr_relpath"],
    }:
        raise ValueError("complete raw log must be a distinct release artifact")
    observed_safety = _scan_stable_raw_log(
        Path(manifest["_runtime_dir"]), raw_log_locator
    )
    if (
        observed_safety["raw_log_sha256"] != expected_raw_sha256
        or observed_safety["raw_log_bytes"] != expected_raw_bytes
        or observed_safety["match_count"] != 0
    ):
        raise ValueError("complete raw log safety census differs from the artifact")
    raw_gate_evidence = normalized_gates["raw-log-safety"]["evidence"]
    if any(
        raw_gate_evidence[key] != observed_safety[key]
        for key in ("raw_log_sha256", "raw_log_bytes", "match_count")
    ):
        raise ValueError("raw-log-safety gate differs from the complete log artifact")
    receipt["gates"] = normalized_gates
    receipt["raw_log_safety"] = {
        "scan_scope": "complete-uncapped",
        "patterns": list(RAW_LOG_PATTERNS),
        "raw_log_relpath": raw_log_locator,
        **observed_safety,
    }
    return receipt


def _validate_process_bindings(
    value: object,
    manifest: dict[str, Any],
    container_id: str,
    profile: tuple[str, str] | None = None,
) -> None:
    # `profile` is (name, recipe sha) for broker /2 proofs of one profile.
    if type(value) is not list or len(value) != 2:
        raise ValueError("suite requires two live process bindings")
    identities = set()
    for mode, item in zip(("none", "native-mtp-k5"), value):
        proof = _exact_object(
            item,
            {
                "schema",
                "status",
                "mode",
                "container_id",
                "run_id",
                "host_pid",
                "start_time_ticks",
                "executable_sha256",
                "recipe_sha256",
                "recipe_environment_verified",
                "mount_namespace_match",
                "cgroup_match",
                "paths_read_only",
            }
            | ({"profile"} if profile is not None else set()),
            "live process proof",
        )
        expected = {
            "schema": (
                "flux-r15-live-process-proof/1"
                if profile is None
                else PROFILE_PAIRINGS[PROFILE_CONTRACT_V2]["proof"]
            ),
            "status": "PASS",
            "mode": mode,
            "container_id": container_id,
            "run_id": manifest["launch"]["run_id"],
            "executable_sha256": manifest["image"]["in_container_binary_sha256"],
            "recipe_sha256": (
                manifest["image"]["recipe_sha256"] if profile is None else profile[1]
            ),
            "recipe_environment_verified": True,
            "mount_namespace_match": True,
            "cgroup_match": True,
            "paths_read_only": True,
        }
        if profile is not None:
            expected["profile"] = profile[0]
        for key, wanted in expected.items():
            if type(proof[key]) is not type(wanted) or proof[key] != wanted:
                raise ValueError("live process proof differs from release identity")
        pid = _plain_int(proof["host_pid"], "live process PID", 1, 1 << 31)
        start = _plain_int(
            proof["start_time_ticks"], "live process start ticks", 1, 1 << 63
        )
        if (pid, start) in identities:
            raise ValueError("startup modes did not use distinct processes")
        identities.add((pid, start))


def validate_suite_receipt(
    path: Path, manifest: dict[str, Any], container_id: str
) -> dict[str, Any]:
    raw = _stable_path_bytes(path, "suite receipt", MAX_SUITE_RECEIPT_BYTES)
    return _validate_suite_receipt_bytes(raw, manifest, container_id)


def _run_suite(
    manifest: dict[str, Any],
    container_id: str,
    backend: SuiteBackend,
) -> dict[str, Any]:
    if lifecycle.CID_RE.fullmatch(container_id) is None:
        raise ValueError("suite container ID must be full lowercase hexadecimal")
    suite = manifest["suite"]
    runtime = Path(manifest["_runtime_dir"])
    receipt_artifact_fd = _create_beneath(
        runtime, suite["_receipt_relpath"], "suite receipt", 0o400
    )
    stdout_artifact_fd: int | None = None
    stderr_artifact_fd: int | None = None
    receipt_fd: int | None = None
    stdout_fd: int | None = None
    stderr_fd: int | None = None
    try:
        stdout_artifact_fd = _create_beneath(
            runtime, suite["_stdout_relpath"], "suite stdout", 0o400
        )
        stderr_artifact_fd = _create_beneath(
            runtime, suite["_stderr_relpath"], "suite stderr", 0o400
        )
        receipt_fd = _anonymous_writable_file("r15-suite-receipt")
        stdout_fd = _anonymous_writable_file("r15-suite-stdout")
        stderr_fd = _anonymous_writable_file("r15-suite-stderr")
        with (
            _sealed_executable(
                Path(suite["program_path"]), suite["program_sha256"]
            ) as executable_fd,
            _sealed_data_file(
                Path(suite["plan_path"]),
                suite["plan_sha256"],
                "suite-plan",
                MAX_SUITE_PLAN_BYTES,
            ) as plan_fd,
        ):
            argv = _substitute_suite_argv(manifest, container_id, plan_fd, receipt_fd)
            rc = backend.run(
                argv,
                timeout_s=suite["timeout_seconds"],
                stdout_fd=stdout_fd,
                stderr_fd=stderr_fd,
                env={
                    "PATH": "/usr/local/bin:/usr/bin:/bin",
                    "LANG": "C.UTF-8",
                    "LC_ALL": "C.UTF-8",
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
                executable_fd=executable_fd,
                plan_fd=plan_fd,
                receipt_fd=receipt_fd,
            )
        receipt_raw = _seal_written_file(
            receipt_fd, "suite receipt", MAX_SUITE_RECEIPT_BYTES
        )
        stdout_raw = _seal_written_file(
            stdout_fd, "suite stdout", MAX_PROGRAM_OUTPUT_BYTES
        )
        stderr_raw = _seal_written_file(
            stderr_fd, "suite stderr", MAX_PROGRAM_OUTPUT_BYTES
        )
        _persist_reserved_file(receipt_artifact_fd, receipt_raw)
        _persist_reserved_file(stdout_artifact_fd, stdout_raw)
        _persist_reserved_file(stderr_artifact_fd, stderr_raw)
        if type(rc) is not int or rc != 0:
            raise RuntimeError(f"container qualification suite exited rc={rc!r}")
        _assert_beneath_identity(
            runtime,
            suite["_receipt_relpath"],
            receipt_artifact_fd,
            "reserved suite receipt",
        )
        _assert_beneath_identity(
            runtime,
            suite["_stdout_relpath"],
            stdout_artifact_fd,
            "reserved suite stdout",
        )
        _assert_beneath_identity(
            runtime,
            suite["_stderr_relpath"],
            stderr_artifact_fd,
            "reserved suite stderr",
        )
        receipt = _validate_suite_receipt_bytes(receipt_raw, manifest, container_id)
        return {
            "receipt": receipt,
            "receipt_sha256": sha256_bytes(receipt_raw),
            "receipt_artifact": suite["_receipt_relpath"],
            "stdout_sha256": sha256_bytes(stdout_raw),
            "stdout_artifact": suite["_stdout_relpath"],
            "stderr_sha256": sha256_bytes(stderr_raw),
            "stderr_artifact": suite["_stderr_relpath"],
            "argv_sha256": sha256_bytes(canonical_bytes(argv)),
            "program_sha256": suite["program_sha256"],
            "plan_sha256": suite["plan_sha256"],
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


def _seal(path: Path, value: dict[str, Any]) -> None:
    absolute = path if path.is_absolute() else Path.cwd() / path
    if (
        ".." in absolute.parts
        or str(Path(str(absolute))) != str(absolute)
        or not absolute.name
    ):
        raise ValueError("outer release receipt path must be normalized")
    payload = canonical_bytes(value) + b"\n"
    parent_fd = _open_directory_chain(absolute.parent, "outer receipt parent")
    descriptor: int | None = None
    try:
        try:
            descriptor = os.open(
                absolute.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o400,
                dir_fd=parent_fd,
            )
        except FileExistsError:
            raise FileExistsError("outer release receipt already exists") from None
        except OSError as error:
            raise ValueError("cannot create outer release receipt") from error
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short write while sealing outer release receipt")
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        reopened = os.open(
            absolute.name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        try:
            original = os.fstat(descriptor)
            current = os.fstat(reopened)
            if (
                not stat.S_ISREG(current.st_mode)
                or current.st_nlink != 1
                or (original.st_dev, original.st_ino)
                != (current.st_dev, current.st_ino)
            ):
                raise ValueError("outer release receipt pathname identity changed")
        finally:
            os.close(reopened)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)


def _revalidate_after_cleanup(
    manifest: dict[str, Any], suite_result: dict[str, Any]
) -> None:
    """Re-read every release artifact only after container descendants are gone."""

    runtime = Path(manifest["_runtime_dir"])
    for name, gate in suite_result["receipt"]["gates"].items():
        raw = _stable_beneath_bytes(
            runtime,
            gate["artifact"],
            f"post-cleanup gate {name} receipt",
            MAX_GATE_RECEIPT_BYTES,
        )
        if (
            len(raw) != gate["receipt_bytes"]
            or sha256_bytes(raw) != gate["receipt_sha256"]
        ):
            raise ValueError(f"gate {name} receipt changed before final seal")
        validator = gate["evidence"]
        validator_raw = _stable_beneath_bytes(
            runtime,
            validator["validator_artifact"],
            f"post-cleanup gate {name} validator artifact",
            MAX_GATE_RECEIPT_BYTES,
        )
        if sha256_bytes(validator_raw) != validator["validator_receipt_sha256"]:
            raise ValueError(
                f"gate {name} validator artifact changed before final seal"
            )
        for dependency in gate.get("dependent_artifacts", []):
            payload = _stable_beneath_bytes(
                runtime,
                dependency["artifact"],
                f"post-cleanup gate {name} dependent artifact",
                MAX_GATE_RECEIPT_BYTES,
            )
            if (
                len(payload) != dependency["bytes"]
                or sha256_bytes(payload) != dependency["sha256"]
            ):
                raise ValueError(
                    f"gate {name} dependent artifact changed before final seal"
                )
    safety = suite_result["receipt"]["raw_log_safety"]
    observed = _scan_stable_raw_log(runtime, safety["raw_log_relpath"])
    if any(observed[key] != safety[key] for key in observed):
        raise ValueError("complete raw log changed before final seal")
    for name in ("receipt", "stdout", "stderr"):
        raw = _stable_beneath_bytes(
            runtime,
            suite_result[f"{name}_artifact"],
            f"post-cleanup suite {name}",
            MAX_SUITE_RECEIPT_BYTES if name == "receipt" else MAX_PROGRAM_OUTPUT_BYTES,
        )
        if sha256_bytes(raw) != suite_result[f"{name}_sha256"]:
            raise ValueError(f"suite {name} changed before final seal")


def run_live(
    manifest: dict[str, Any],
    manifest_path: Path,
    receipt_path: Path,
    *,
    docker_backend: r12.CommandBackend | None = None,
    suite_backend: SuiteBackend | None = None,
    runner_factory: Callable[..., Any] = r12.R12ContainerReleaseRunner,
) -> dict[str, Any]:
    if receipt_path.exists() or receipt_path.is_symlink():
        raise FileExistsError("outer release receipt already exists")
    docker = docker_backend or r12.SubprocessBackend()
    suite = suite_backend or SubprocessSuiteBackend()
    revalidate_manifest(manifest, manifest_path, "before image inspection")
    image_before = inspect_exact_image(manifest, docker)

    launch = manifest["launch"]
    # Blocker-5 namespace hardening: verify no-follow visibility for
    # cidfile/lockfile/diagnostics and create missing parents before the
    # runner (or docker) ever touches them.
    _prepare_release_launch_paths(manifest)
    capture = _release_diagnostic_capture(manifest, docker)
    try:
        runner = runner_factory(
            image_id=manifest["image"]["id"],
            name=launch["name"],
            run_id=launch["run_id"],
            cidfile=Path(launch["cidfile"]),
            lockfile=Path(launch["lockfile"]),
            diagnostics_dir=Path(launch["diagnostics_dir"]),
            extra_run_args=launch["extra_run_args"],
            entrypoint=launch["entrypoint"],
            container_command=launch["container_command"],
            backend=docker,
            diagnostic_capture=capture,
        )
    except BaseException:
        capture.close()
        raise

    def qualify(
        identity: lifecycle.LaunchIdentity, container_id: str
    ) -> dict[str, Any]:
        if (
            identity.image_id != manifest["image"]["id"]
            or identity.run_id != launch["run_id"]
            or identity.name != launch["name"]
        ):
            raise ValueError("lifecycle identity differs from the validated manifest")
        revalidate_manifest(manifest, manifest_path, "before qualification suite")
        container_before = inspect_running_container(manifest, container_id, docker)
        artifacts_before = measure_container_artifacts(manifest, container_id, docker)
        result = _run_suite(manifest, container_id, suite)
        revalidate_manifest(manifest, manifest_path, "after qualification suite")
        container_after = inspect_running_container(manifest, container_id, docker)
        if container_after != container_before:
            raise ValueError("running container changed during qualification")
        artifacts_after = measure_container_artifacts(manifest, container_id, docker)
        if artifacts_after != artifacts_before:
            raise ValueError(
                "in-container release artifacts changed during qualification"
            )
        result["runtime_container"] = container_after
        result["image_artifacts"] = artifacts_after
        return result

    try:
        suite_result, cleanup = runner.run(qualify)
    finally:
        capture.close()
    _revalidate_after_cleanup(manifest, suite_result)
    revalidate_manifest(manifest, manifest_path, "before final seal")
    image_after = inspect_exact_image(manifest, docker)
    if image_after != image_before:
        raise ValueError("immutable image inspection changed during qualification")
    if (
        cleanup.lock_safe is not True
        or cleanup.lock_released is not True
        or cleanup.launch_client_settled is not True
        or cleanup.errors
        or not (cleanup.stable_absence_sweeps or cleanup.stable_stopped_sweeps)
    ):
        raise RuntimeError("container cleanup did not produce an exact unlock proof")
    cleanup_value = asdict(cleanup)
    # Second stable re-read pass immediately before the seal step: any
    # post-cleanup tampering with receipts, raw log, or launch namespace must
    # fail closed rather than seal a stale proof.
    _revalidate_after_cleanup(manifest, suite_result)
    _prepare_release_launch_paths(manifest, expect_cidfile_free=False)
    revalidate_manifest(manifest, manifest_path, "before final seal")
    result = {
        "schema": RECEIPT_SCHEMA,
        "status": "PASS",
        "release_qualified": True,
        "release_version": manifest["release_version"],
        "execution_scope": "immutable-container",
        "manifest_sha256": manifest["_raw_sha256"],
        "image": image_after,
        "source_revision": manifest["provenance"]["source_revision"],
        "source_tree": manifest["provenance"]["source_tree"],
        "source_to_binary_receipt_sha256": manifest["provenance"][
            "source_to_binary_receipt_sha256"
        ],
        "launch": {
            "name": launch["name"],
            "run_id": launch["run_id"],
            "container_id": suite_result["receipt"]["container_id"],
            "base_url": launch["base_url"],
        },
        "suite": {
            key: suite_result[key]
            for key in (
                "receipt_sha256",
                "receipt_artifact",
                "stdout_sha256",
                "stdout_artifact",
                "stderr_sha256",
                "stderr_artifact",
                "argv_sha256",
                "program_sha256",
                "plan_sha256",
            )
        },
        "runtime_container": suite_result["runtime_container"],
        "image_artifacts": suite_result["image_artifacts"],
        "process_bindings": suite_result["receipt"]["process_bindings"],
        "gates": suite_result["receipt"]["gates"],
        "raw_log_safety": suite_result["receipt"]["raw_log_safety"],
        "cleanup": cleanup_value,
    }
    _seal(receipt_path, result)
    return result


def dry_run_plan(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": RECEIPT_SCHEMA + "/dry-run",
        "release_qualified": False,
        "release_version": manifest["release_version"],
        "execution_scope": "plan-only-no-docker-gpu-server-or-network",
        "image_id": manifest["image"]["id"],
        "image_digest": manifest["image"]["digest"],
        "required_gates": list(REQUIRED_GATES),
        "complete_raw_log_patterns": list(RAW_LOG_PATTERNS),
        "lifecycle": r12.R12_CLEANUP_VERSION,
        "suite_program_sha256": manifest["suite"]["program_sha256"],
        "suite_plan_sha256": manifest["suite"]["plan_sha256"],
        "manifest_sha256": manifest["_raw_sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--receipt", type=Path, default=Path("r15-exact-image-release-receipt.json")
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest, verify_files=not args.dry_run)
    if args.dry_run:
        print(json.dumps(dry_run_plan(manifest), indent=2, sort_keys=True))
        return 0
    result = run_live(manifest, args.manifest, args.receipt)
    print(f"PASS receipt={args.receipt} sha256={sha256_bytes(canonical_bytes(result))}")
    return 0


NATIVE_CONTEXT_PROGRAMS = {
    "native-262k": ("repro-kit/r15-gates/native-262k", "f974ac95e95b3871ade9f38e011bef8c5e6b2401f702ccfa7b828dca03091aa3"),
    "native-262k-w4096-owner16": (
        "repro-kit/r15-gates/native-262k-w4096-owner16", "dc6ab71e3ee4aaaface2d52bb27f2b73356b51a2fee1614c5b73d613085f26f8"),
}
NATIVE_CONTEXT_REQUIRED_MODULE = "repro-kit/r16_native_context_required.py"
NATIVE_CONTEXT_REQUIRED_MODULE_SHA256 = "be2ad639d3d624c34cd0871226165d3ef50e5c9de7539ac439745e1ec06e7e86"


def _native_context_required_module():
    descriptor, payload = _open_stable(
        Path(_KIT_DIR) / "r16_native_context_required.py", "native context required validator", 1 << 20)
    os.close(descriptor)
    if sha256_bytes(payload) != NATIVE_CONTEXT_REQUIRED_MODULE_SHA256:
        raise ValueError("native context independent validator source seal differs")
    module = types.ModuleType("_sealed_native_context_required")
    module.__file__ = str(Path(_KIT_DIR) / "r16_native_context_required.py")
    module.release = sys.modules[__name__]
    exec(compile(payload, module.__file__, "exec", dont_inherit=True), module.__dict__)
    return module


def _validate_native_context_image_binding(manifest, gate, recipe_sha256):
    plan = _profile_plan(manifest)
    contract = validate_native_context_contract(plan.get("native_context_contract"))
    if (plan.get("schema") != "flux-r15-container-suite-plan/4"
            or contract["gates"][gate]["recipe_sha256"] != recipe_sha256):
        raise ValueError("native context sealed recipe binding differs")
    return contract["gates"][gate]


def _validate_native_context_source_closure(provenance, suite, plan, source_root, entries):
    relative, digest = NATIVE_CONTEXT_REQUIRED_MODULE, NATIVE_CONTEXT_REQUIRED_MODULE_SHA256
    entry = entries.get(f"{provenance_gate.ENGINE_PREFIX}/{relative}")
    if (type(entry) is not dict or entry.get("role") != "engine-source"
            or type(entry.get("mode")) is not int or entry["mode"] != 0o644
            or entry.get("sha256") != digest):
        raise ValueError("native context validator is not a proven tracked source")
    descriptor, payload = _open_stable(source_root / relative, "native context validator source", 1 << 20)
    os.close(descriptor)
    if len(payload) != entry.get("bytes") or sha256_bytes(payload) != digest:
        raise ValueError("native context validator differs from proven inventory")
    required = _native_context_required_module()
    for gate in NATIVE_CONTEXT_PROGRAMS:
        required.validate_required_program_closure(
            gate, plan["gate_jobs"][gate], provenance, suite, plan, source_root)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"HOLD: {error}", file=sys.stderr)
        raise SystemExit(1)
