#!/usr/bin/env python3
"""Fail-closed direct-binary qualifier for native Qwen4 concurrency.

This client deliberately does not start, stop, or build the server. A resource
owner launches one fresh process for exactly one active width and selector
profile, then passes its append-only log here. The probe runs isolated C1
controls, forward and reverse concurrent waves, and (at C8) cancellation,
replacement, drain, and slot-reuse checks. It seals exact token IDs, complete
logits-row SHA-256 values, and terminal KV/recurrent-state SHA-256 values.

`ATLAS_LOGITS_FNV=1`, `ATLAS_QWEN4_NATIVE_BATCH_RECEIPT=1`, and
`ATLAS_QWEN4_TARGET_STATE_RECEIPT=1` must be armed for the full tier. These are
qualification-only diagnostics. They are not performance measurements.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat as stat_module
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

if "r9_provenance" not in globals():
    _REPRO_KIT_DIR = str(Path(__file__).resolve().parent)
    if _REPRO_KIT_DIR not in sys.path:
        sys.path.insert(0, _REPRO_KIT_DIR)
    import r9_source_to_binary as r9_provenance  # noqa: E402

SCHEMA = "flux-qwen4-native-concurrency-probe-v3"
LAUNCH_SCHEMA = "flux-qwen4-native-concurrency-launch-v3"
RELEASE_VERSION = "v0.9.13-r15"
PUBLIC_MODEL_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
PUBLIC_MODEL_ID_RE = re.compile(
    r"hf://[A-Za-z0-9][A-Za-z0-9._-]{0,95}/" r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}"
)
ENGAGED_RE = re.compile(
    r"QWEN4_NATIVE_BATCH_ENGAGED rows=([0-9]+) "
    r"active_only=true target_only=true qsa=false$",
    re.MULTILINE,
)
LOGITS_RE = re.compile(
    r"LOGITS_FNV session=([0-9a-f]{16}) idx=([0-9]+) "
    r"fnv=([0-9a-f]{16}) sha256=([0-9a-f]{64})$"
)
FRAME_RE = re.compile(
    r"QWEN4_(NATIVE|SERIAL)_BATCH_FRAME rows=([0-9]+) "
    r"sessions=([0-9a-f]{16}(?:,[0-9a-f]{16})*) "
    r"slots=([0-9]+(?:,[0-9]+)*) pre_step_seq_lens=([0-9]+(?:,[0-9]+)*) "
    r"(?:(?:active_only=true target_only=true qsa=false)|"
    r"(?:selector_on=(true|false) qsa_required=(true|false))) graph=false$"
)
STATE_RE = re.compile(
    r"QWEN4_NATIVE_MTP_REQUEST_RECEIPT schema=([0-9]+) "
    r"mode=([^ ]+) scheduler_path=([^ ]+) proposer_loaded=(true|false) "
    r"session_hash=([0-9a-f]{16}) seq_len=([0-9]+) token_count=([0-9]+) "
    r"output_tokens=([0-9]+) gdn_layers=([0-9]+) attention_layers=([0-9]+) "
    r"logical_kv_blocks=([0-9]+) kv_token_positions=([0-9]+) "
    r"compared_buffers=([0-9]+) compared_bytes=([0-9]+) "
    r"state_scope=live-gdn-ple-plus-logical-bf16-kv checkpoints_excluded=true "
    r"fed_tokens_sha256=([0-9a-f]{64}) target_state_sha256=([0-9a-f]{64})$"
)
RECEIPT_MARKERS = (
    "QWEN4_NATIVE_BATCH_FRAME",
    "QWEN4_SERIAL_BATCH_FRAME",
    "LOGITS_FNV",
    "QWEN4_NATIVE_MTP_REQUEST_RECEIPT",
)
FORBIDDEN_LOG_MARKERS = (
    "CUDA error",
    "illegal memory access",
    "out of memory",
    "decode_batch error",
    "aliases another request's recurrent state",
    "alias persistent slot",
    "independent rows alias a physical KV block",
    "does not exclusively own its KV block",
    "QWEN4_NATIVE_MTP_REQUEST_RECEIPT_FAILED",
    "NVRM: Xid",
    "illegal instruction",
    "segmentation fault",
    "stack backtrace:",
    "thread 'main' panicked",
)
EXACT_BUDGET_FINISH_REASONS = frozenset(("length", "stop"))
USAGE_IDENTITY_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "prompt_tokens_details",
    "completion_tokens_details",
)
WAVE_ADMISSION_STAGGER_S = 0.005
OPTIMIZATION_SELECTORS = (
    "ATLAS_ATTN_QKV_EXACT_STRIDED",
    "ATLAS_ATTN_QK_NORM_BATCHED",
    "ATLAS_ATTN_GATE_BATCHED",
    "ATLAS_ATTN_KV_WRITE_BATCHED",
    "ATLAS_QWEN4_NATIVE_SSM_PTR_CACHE",
)
OPTIMIZATION_ENGAGEMENT_MARKERS = {
    "ATLAS_ATTN_QKV_EXACT_STRIDED": "ENGAGED ATLAS_ATTN_QKV_EXACT_STRIDED:",
    "ATLAS_ATTN_QK_NORM_BATCHED": "ENGAGED ATLAS_ATTN_QK_NORM_BATCHED:",
    "ATLAS_ATTN_GATE_BATCHED": "ENGAGED ATLAS_ATTN_GATE_BATCHED:",
    "ATLAS_ATTN_KV_WRITE_BATCHED": "ENGAGED ATLAS_ATTN_KV_WRITE_BATCHED:",
    "ATLAS_QWEN4_NATIVE_SSM_PTR_CACHE": "QWEN4_NATIVE_SSM_PTR_CACHE_ENGAGED ",
}
OPTIMIZATION_ENGAGEMENT_RES = {
    "ATLAS_ATTN_QKV_EXACT_STRIDED": re.compile(
        r"ENGAGED ATLAS_ATTN_QKV_EXACT_STRIDED: multi_seq M=[0-9]+$"
    ),
    "ATLAS_ATTN_QK_NORM_BATCHED": re.compile(
        r"ENGAGED ATLAS_ATTN_QK_NORM_BATCHED: multi_seq M=[0-9]+$"
    ),
    "ATLAS_ATTN_GATE_BATCHED": re.compile(
        r"ENGAGED ATLAS_ATTN_GATE_BATCHED: multi_seq M=[0-9]+ "
        r"dim=[0-9]+ stride=[0-9]+$"
    ),
    "ATLAS_ATTN_KV_WRITE_BATCHED": re.compile(
        r"ENGAGED ATLAS_ATTN_KV_WRITE_BATCHED: multi_seq M=[0-9]+$"
    ),
    "ATLAS_QWEN4_NATIVE_SSM_PTR_CACHE": re.compile(
        r"QWEN4_NATIVE_SSM_PTR_CACHE_ENGAGED rows=[0-9]+ " r"table=stable h2d=elided$"
    ),
}
SELECTOR_PROFILES = {
    "native-reference": {
        "ATLAS_QWEN4_NATIVE_BATCH": "1",
        **{selector: "0" for selector in OPTIMIZATION_SELECTORS},
    },
    "native-optimized": {
        "ATLAS_QWEN4_NATIVE_BATCH": "1",
        **{selector: "1" for selector in OPTIMIZATION_SELECTORS},
    },
    "serial-control": {
        "ATLAS_QWEN4_NATIVE_BATCH": "0",
        **{selector: "0" for selector in OPTIMIZATION_SELECTORS},
    },
}
SOURCE_CLOSURE_PATH_KEYS = (
    "source_root_path",
    "source_bundle_manifest_path",
    "source_bundle_path",
    "build_receipt_path",
    "source_to_binary_receipt_path",
    "binary_path",
)
SOURCE_CLOSURE_DIGEST_KEYS = (
    "source_bundle_manifest_sha256",
    "source_bundle_sha256",
    "build_receipt_sha256",
    "source_to_binary_receipt_sha256",
    "binary_sha256",
)
PROVENANCE_KEYS = {
    "release_version",
    "source_revision",
    "source_tree",
    *SOURCE_CLOSURE_PATH_KEYS,
    *SOURCE_CLOSURE_DIGEST_KEYS,
    "checkpoint_id",
    "checkpoint_revision",
    "checkpoint_path",
    "checkpoint_manifest_path",
    "checkpoint_manifest_sha256",
}


class ProbeError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class Case:
    label: str
    ordinal: int
    content: str

    @property
    def prompt_sha256(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()


@dataclasses.dataclass
class RequestResult:
    label: str
    token_ids: list[int]
    usage: dict[str, Any]
    finish_reason: str
    started_unix: float
    finished_unix: float


@dataclasses.dataclass
class CancelResult:
    label: str
    content_chunks: int
    started_unix: float
    closed_unix: float


@dataclasses.dataclass
class LogitRow:
    session: str
    index: int
    fnv: str
    sha256: str


@dataclasses.dataclass
class Frame:
    route: str
    rows: int
    sessions: list[str]
    slots: list[int]
    pre_step_seq_lens: list[int]
    selector_on: bool | None
    qsa_required: bool | None
    logits: list[LogitRow]
    line_number: int


@dataclasses.dataclass
class StateReceipt:
    session: str
    schema: int
    mode: str
    scheduler_path: str
    proposer_loaded: bool
    seq_len: int
    token_count: int
    output_tokens: int
    gdn_layers: int
    attention_layers: int
    logical_kv_blocks: int
    kv_token_positions: int
    compared_buffers: int
    compared_bytes: int
    fed_tokens_sha256: str
    target_state_sha256: str
    line_number: int


@dataclasses.dataclass
class ParsedLog:
    rows_by_session: dict[str, dict[int, LogitRow]]
    frames: list[Frame]
    state_by_session: dict[str, StateReceipt]
    text_sha256: str


class LogCursor:
    def __init__(self, path: Path):
        self.path = path
        self._identity = stable_identity(path, "server log")[:2]

    def mark(self) -> int:
        identity = stable_identity(self.path, "server log")
        if identity[:2] != self._identity:
            raise ProbeError("server log inode changed during qualification")
        return identity[2]

    def read_bytes_from(
        self,
        offset: int,
        settle_seconds: float = 0.15,
        end_offset: int | None = None,
    ) -> bytes:
        time.sleep(settle_seconds)
        identity = stable_identity(self.path, "server log")
        if identity[:2] != self._identity:
            raise ProbeError("server log inode changed during qualification")
        if identity[2] < offset:
            raise ProbeError("server log was truncated during qualification")
        if end_offset is None:
            end_offset = identity[2]
        if end_offset < offset or identity[2] < end_offset:
            raise ProbeError("server log range is invalid or was truncated")
        with self.path.open("rb") as handle:
            handle.seek(offset)
            return handle.read(end_offset - offset)

    def read_from(
        self,
        offset: int,
        settle_seconds: float = 0.15,
        end_offset: int | None = None,
    ) -> str:
        raw = self.read_bytes_from(offset, settle_seconds, end_offset)
        try:
            return raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise ProbeError("server log contains invalid UTF-8") from error


class MemorySampler:
    def __init__(self, pid: int | None):
        self.pid = pid
        self.samples: list[dict[str, int | float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _fields(path: Path) -> dict[str, int]:
        values: dict[str, int] = {}
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            return values
        for line in lines:
            match = re.fullmatch(r"([A-Za-z_]+):\s+([0-9]+)\s+kB", line)
            if match:
                values[match.group(1)] = int(match.group(2))
        return values

    def _sample(self) -> None:
        mem = self._fields(Path("/proc/meminfo"))
        row: dict[str, int | float] = {
            "monotonic_s": round(time.monotonic(), 6),
            "mem_available_kib": mem.get("MemAvailable", 0),
        }
        if self.pid is not None:
            status = self._fields(Path(f"/proc/{self.pid}/status"))
            row["server_vm_rss_kib"] = status.get("VmRSS", 0)
            row["server_vm_hwm_kib"] = status.get("VmHWM", 0)
        self.samples.append(row)

    def start(self) -> None:
        self._sample()

        def loop() -> None:
            while not self._stop.wait(0.25):
                self._sample()

        self._thread = threading.Thread(target=loop, name="memory-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._sample()

    def summary(self) -> dict[str, Any]:
        available = [int(row["mem_available_kib"]) for row in self.samples]
        rss = [int(row.get("server_vm_rss_kib", 0)) for row in self.samples]
        hwm = [int(row.get("server_vm_hwm_kib", 0)) for row in self.samples]
        return {
            "samples": len(self.samples),
            "mem_available_floor_kib": min(available, default=0),
            "server_vm_rss_peak_kib": max(rss, default=0),
            "server_vm_hwm_peak_kib": max(hwm, default=0),
        }


def json_canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def json_loads_strict(payload: str | bytes, label: str) -> Any:
    """Parse JSON while rejecting duplicate keys and non-finite numbers."""

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ProbeError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def constant(value: str) -> Any:
        raise ProbeError(f"{label} contains non-finite number {value}")

    def floating(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ProbeError(f"{label} contains non-finite number {value}")
        return parsed

    try:
        return json.loads(
            payload,
            object_pairs_hook=object_pairs,
            parse_constant=constant,
            parse_float=floating,
        )
    except UnicodeDecodeError as error:
        raise ProbeError(f"{label} is not UTF-8") from error
    except json.JSONDecodeError as error:
        raise ProbeError(f"{label} is not strict JSON: {error}") from error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_identity(path: Path, label: str) -> tuple[int, int, int, int, int]:
    try:
        stat = path.stat()
    except OSError as error:
        raise ProbeError(f"cannot stat {label}: {error}") from error
    if not path.is_file():
        raise ProbeError(f"{label} must be a regular file")
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_mode)


def sha256_stable_file(
    path: Path, label: str
) -> tuple[str, tuple[int, int, int, int, int]]:
    before = stable_identity(path, label)
    digest = sha256_file(path)
    after = stable_identity(path, label)
    if before != after:
        raise ProbeError(f"{label} changed while hashing")
    return digest, after


def read_stable_file_bytes(
    path: Path, label: str
) -> tuple[bytes, str, tuple[int, int, int, int, int]]:
    """Capture bytes once so callers hash and parse the identical file image."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ProbeError(f"cannot open {label}") from error
    try:
        before_stat = os.fstat(descriptor)
        if not stat_module.S_ISREG(before_stat.st_mode):
            raise ProbeError(f"{label} must be a regular file")
        before = (
            before_stat.st_dev,
            before_stat.st_ino,
            before_stat.st_size,
            before_stat.st_mtime_ns,
            before_stat.st_mode,
        )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after_stat = os.fstat(descriptor)
        after = (
            after_stat.st_dev,
            after_stat.st_ino,
            after_stat.st_size,
            after_stat.st_mtime_ns,
            after_stat.st_mode,
        )
    finally:
        os.close(descriptor)
    if before != after:
        raise ProbeError(f"{label} changed while reading")
    if stable_identity(path, label) != after:
        raise ProbeError(f"{label} path identity changed while reading")
    payload = b"".join(chunks)
    if len(payload) != after[2]:
        raise ProbeError(f"{label} byte count changed while reading")
    return payload, hashlib.sha256(payload).hexdigest(), after


def public_model_name(value: Any, label: str) -> str:
    if type(value) is not str or PUBLIC_MODEL_NAME_RE.fullmatch(value) is None:
        raise ProbeError(f"{label} must be a path-free public API model alias")
    return value


def public_checkpoint_id(value: Any, label: str) -> str:
    if type(value) is not str or PUBLIC_MODEL_ID_RE.fullmatch(value) is None:
        raise ProbeError(f"{label} must use the public hf://owner/model schema")
    return value


def canonical_sha256(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or not re.fullmatch(r"[0-9a-f]{64}", value)
        or value == "0" * 64
    ):
        raise ProbeError(f"{label} must be a nonzero lowercase SHA-256")
    return value


def exact_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        suffix = "" if minimum is None else f" >= {minimum}"
        raise ProbeError(f"{label} must be an exact built-in integer{suffix}")
    return value


def validate_usage(
    usage: Any,
    token_count: int,
    label: str,
    *,
    expected_prompt_tokens: int | None = None,
) -> dict[str, Any]:
    if type(usage) is not dict:
        raise ProbeError(f"{label}: missing terminal usage")
    prompt_tokens = exact_int(
        usage.get("prompt_tokens"), f"{label}.prompt_tokens", minimum=1
    )
    completion_tokens = exact_int(
        usage.get("completion_tokens"), f"{label}.completion_tokens", minimum=0
    )
    total_tokens = exact_int(
        usage.get("total_tokens"), f"{label}.total_tokens", minimum=1
    )
    if expected_prompt_tokens is not None and prompt_tokens != expected_prompt_tokens:
        raise ProbeError(f"{label}: prompt-token count differs")
    if completion_tokens != token_count:
        raise ProbeError(f"{label}: token-ID cardinality differs from usage")
    if total_tokens != prompt_tokens + completion_tokens:
        raise ProbeError(f"{label}: total-token arithmetic differs")
    for field in ("prompt_tokens_details", "completion_tokens_details"):
        details = usage.get(field)
        if type(details) is not dict:
            raise ProbeError(f"{label}.{field} must be an object")
        for key, value in details.items():
            if type(key) is not str or not key:
                raise ProbeError(f"{label}.{field} has a malformed key")
            exact_int(value, f"{label}.{field}.{key}", minimum=0)
    return usage


def validate_r15_source_closure(provenance: dict[str, Any]) -> None:
    original_release = r9_provenance.RELEASE_VERSION
    try:
        r9_provenance.configure_release(provenance.get("release_version"))
        r9_provenance.validate_qualification_provenance(provenance)
    except (OSError, r9_provenance.ProvenanceError) as error:
        raise ProbeError(f"R15 source-to-binary provenance failed: {error}") from error
    finally:
        r9_provenance.configure_release(original_release)


def argv_value(argv: list[Any], flag: str) -> str | None:
    matches = [index for index, value in enumerate(argv) if value == flag]
    if len(matches) > 1:
        raise ProbeError(f"manifest argv repeats {flag}")
    if not matches:
        return None
    index = matches[0]
    if index + 1 >= len(argv) or not isinstance(argv[index + 1], str):
        raise ProbeError(f"manifest argv has no value for {flag}")
    return argv[index + 1]


def process_start_ticks(pid: int) -> int:
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    close = raw.rfind(")")
    if close < 0:
        raise ProbeError("live /proc/PID/stat is malformed")
    fields = raw[close + 2 :].split()
    if len(fields) <= 19:
        raise ProbeError("live /proc/PID/stat omits process start time")
    try:
        return int(fields[19])
    except ValueError as error:
        raise ProbeError("live /proc/PID/stat start time is malformed") from error


def validate_log_binding(server_log: Path, pid: int) -> tuple[int, int]:
    identity = stable_identity(server_log, "server log")
    device_inode = identity[:2]
    bound = []
    for fd in (1, 2):
        try:
            stat = Path(f"/proc/{pid}/fd/{fd}").stat()
        except OSError:
            continue
        if (stat.st_dev, stat.st_ino) == device_inode:
            bound.append(fd)
    if bound != [1, 2]:
        raise ProbeError("server log must be both live process stdout and stderr")
    return device_inode


def validate_checkpoint_inventory(
    root: Path,
    inventory_path: Path,
    expected_sha256: str,
    checkpoint_id: str,
    checkpoint_revision: str,
    verify_bytes: bool,
) -> tuple[
    dict[str, Any],
    tuple[int, int, int, int, int],
    dict[str, tuple[int, int, int, int, int]],
]:
    inventory_bytes, observed, manifest_identity = read_stable_file_bytes(
        inventory_path, "checkpoint inventory"
    )
    if observed != expected_sha256:
        raise ProbeError("checkpoint inventory SHA-256 differs")
    inventory = json_loads_strict(inventory_bytes, "checkpoint inventory")
    if type(inventory) is not dict or set(inventory) != {
        "schema",
        "checkpoint_id",
        "checkpoint_revision",
        "file_count",
        "total_bytes",
        "files",
    }:
        raise ProbeError("checkpoint inventory keys differ")
    if (
        inventory["schema"] != "flux-checkpoint-inventory/1"
        or inventory["checkpoint_id"] != checkpoint_id
        or inventory["checkpoint_revision"] != checkpoint_revision
    ):
        raise ProbeError("checkpoint inventory identity differs")
    records = inventory["files"]
    if type(records) is not list or not records:
        raise ProbeError("checkpoint inventory has no file records")
    if type(inventory["file_count"]) is not int or inventory["file_count"] != len(
        records
    ):
        raise ProbeError("checkpoint inventory file count differs")
    root = root.resolve()
    if not root.is_dir():
        raise ProbeError("checkpoint_path is not a directory")
    identities: dict[str, tuple[int, int, int, int, int]] = {}
    paths: list[str] = []
    total_bytes = 0
    for record in records:
        if type(record) is not dict or set(record) != {"path", "bytes", "sha256"}:
            raise ProbeError("checkpoint inventory file record keys differ")
        relative = record["path"]
        size = record["bytes"]
        digest = record["sha256"]
        if (
            type(relative) is not str
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or Path(relative).as_posix() != relative
            or "\\" in relative
            or type(size) is not int
            or size < 0
            or type(digest) is not str
        ):
            raise ProbeError("checkpoint inventory file record is malformed")
        canonical_sha256(digest, f"checkpoint inventory {relative}")
        path = root / relative
        identity = stable_identity(path, f"checkpoint file {relative}")
        if identity[2] != size:
            raise ProbeError(f"checkpoint file size differs: {relative}")
        if verify_bytes:
            observed_digest, identity = sha256_stable_file(
                path, f"checkpoint file {relative}"
            )
            if observed_digest != digest:
                raise ProbeError(f"checkpoint file SHA-256 differs: {relative}")
        identities[relative] = identity
        paths.append(relative)
        total_bytes += size
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ProbeError("checkpoint inventory paths are not unique and sorted")
    actual_paths = sorted(
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    )
    if actual_paths != paths:
        raise ProbeError("checkpoint directory file set differs from inventory")
    if (
        type(inventory["total_bytes"]) is not int
        or inventory["total_bytes"] != total_bytes
    ):
        raise ProbeError("checkpoint inventory total bytes differ")
    return inventory, manifest_identity, identities


def validate_provenance(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    binary_sha256 = canonical_sha256(args.binary_sha256, "--binary-sha256")
    manifest_sha256 = canonical_sha256(args.manifest_sha256, "--manifest-sha256")
    observed_binary_sha256, binary_identity = sha256_stable_file(
        args.binary, "sealed server executable"
    )
    if observed_binary_sha256 != binary_sha256:
        raise ProbeError("sealed binary SHA-256 differs from --binary-sha256")
    manifest_bytes, observed_manifest_sha256, manifest_identity = (
        read_stable_file_bytes(args.manifest, "launch manifest")
    )
    if observed_manifest_sha256 != manifest_sha256:
        raise ProbeError("launch manifest SHA-256 differs from --manifest-sha256")
    manifest = json_loads_strict(manifest_bytes, "launch manifest")
    if type(manifest) is not dict or set(manifest) != {
        "schema",
        "argv",
        "cwd",
        "base_url",
        "model",
        "environment",
        "provenance",
    }:
        raise ProbeError("launch manifest keys differ")
    if manifest.get("schema") != LAUNCH_SCHEMA:
        raise ProbeError("launch manifest schema differs")
    if manifest.get("base_url") != args.base_url or manifest.get("model") != args.model:
        raise ProbeError("launch manifest endpoint/model differ from probe arguments")
    provenance = manifest.get("provenance")
    if type(provenance) is not dict or set(provenance) != PROVENANCE_KEYS:
        raise ProbeError("launch manifest R15 provenance keys differ")
    try:
        r9_provenance.release_identity(provenance.get("release_version"))
    except ValueError as error:
        raise ProbeError(f"launch manifest release_version is invalid: {error}") from error
    if provenance.get("binary_sha256") != binary_sha256:
        raise ProbeError("launch manifest does not bind the exact binary SHA-256")
    for key in ("source_revision", "source_tree"):
        value = provenance.get(key)
        if (
            not isinstance(value, str)
            or not re.fullmatch(r"[0-9a-f]{40}", value)
            or value == "0" * 40
        ):
            raise ProbeError(
                f"launch manifest provenance.{key} must be a nonzero Git object"
            )
    for key in (*SOURCE_CLOSURE_DIGEST_KEYS, "checkpoint_manifest_sha256"):
        value = provenance.get(key)
        canonical_sha256(value, f"manifest provenance.{key}")
    for key in (
        *SOURCE_CLOSURE_PATH_KEYS,
        "checkpoint_path",
        "checkpoint_manifest_path",
    ):
        value = provenance.get(key)
        if type(value) is not str or not value or not Path(value).is_absolute():
            raise ProbeError(
                f"launch manifest provenance.{key} must be an absolute path"
            )
    checkpoint_id = provenance.get("checkpoint_id")
    checkpoint_revision = provenance.get("checkpoint_revision")
    if (
        type(checkpoint_revision) is not str
        or not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", checkpoint_revision)
        or set(checkpoint_revision) == {"0"}
    ):
        raise ProbeError(
            "launch manifest must bind a public checkpoint id and immutable revision"
        )
    checkpoint_id = public_checkpoint_id(
        checkpoint_id, "launch manifest provenance.checkpoint_id"
    )

    argv = manifest.get("argv")
    environment = manifest.get("environment")
    if type(argv) is not list or not all(type(value) is str for value in argv):
        raise ProbeError("launch manifest argv must be an array of strings")
    if type(environment) is not dict or not all(
        type(key) is str and type(value) is str for key, value in environment.items()
    ):
        raise ProbeError("launch manifest environment must be a string map")
    required_flags = {
        "--bind": "127.0.0.1",
        "--port": str(urllib.parse.urlsplit(args.base_url).port),
        "--model-name": args.model,
        "--kernel-target": "qwen3.8-flash-next",
        "--max-seq-len": "2048",
        "--max-prefill-tokens": "1024",
        "--kv-cache-dtype": "bf16",
        "--block-size": "16",
        "--ssm-cache-slots": "0",
        "--swap-space-gb": "0",
        "--enable-prefix-caching": "false",
        "--world-size": "1",
        "--tp-size": "1",
        "--ep-size": "1",
        "--request-timeout": "900",
    }
    for flag, expected in required_flags.items():
        if argv_value(argv, flag) != expected:
            raise ProbeError(f"launch manifest must set {flag} {expected}")
    raw_max_batch_size = argv_value(argv, "--max-batch-size") or ""
    raw_max_num_seqs = argv_value(argv, "--max-num-seqs") or ""
    if not re.fullmatch(r"[1-9][0-9]*", raw_max_batch_size) or not re.fullmatch(
        r"[1-9][0-9]*", raw_max_num_seqs
    ):
        raise ProbeError("manifest concurrency flags must be canonical integers")
    max_batch_size = int(raw_max_batch_size)
    max_num_seqs = int(raw_max_num_seqs)
    if not (args.width <= max_batch_size <= 128) or not (
        args.width <= max_num_seqs <= 128
    ):
        raise ProbeError(
            "launch manifest concurrency capacity is below the probed width"
        )
    if (
        argv.count("--qwen4-qsa") != 1
        or argv.count("--no-tui") != 1
        or argv.count("serve") != 1
    ):
        raise ProbeError("launch manifest must select serve, --no-tui, and --qwen4-qsa")
    for forbidden in (
        "--speculative",
        "--dflash",
        "--self-speculative",
        "--ngram-speculative",
        "--profile",
        "--high-speed-swap",
    ):
        if forbidden in argv:
            raise ProbeError(f"launch manifest must omit {forbidden}")
    expected_selectors = SELECTOR_PROFILES[args.selector_profile]
    for selector, expected in expected_selectors.items():
        if environment.get(selector) != expected:
            raise ProbeError(f"{args.selector_profile} requires {selector}={expected}")
    if args.evidence_tier == "full":
        if environment.get("ATLAS_QWEN4_NATIVE_BATCH_RECEIPT") != "1":
            raise ProbeError("full tier requires ATLAS_QWEN4_NATIVE_BATCH_RECEIPT=1")
        if environment.get("ATLAS_LOGITS_FNV") != "1":
            raise ProbeError("full tier requires ATLAS_LOGITS_FNV=1")
        if environment.get("ATLAS_QWEN4_TARGET_STATE_RECEIPT") != "1":
            raise ProbeError("full tier requires ATLAS_QWEN4_TARGET_STATE_RECEIPT=1")
        if "ATLAS_LOGITS_FNV_REQUEST_LIMIT" in environment:
            raise ProbeError("full tier requires ATLAS_LOGITS_FNV_REQUEST_LIMIT unset")
        if "ATLAS_QWEN4_TARGET_STATE_RECEIPT_LIMIT" in environment:
            raise ProbeError(
                "full tier requires ATLAS_QWEN4_TARGET_STATE_RECEIPT_LIMIT unset"
            )
    expected_environment = {
        **expected_selectors,
        "RUST_LOG": "info,spark_server::scheduler=debug",
    }
    if args.evidence_tier == "full":
        expected_environment.update(
            {
                "ATLAS_QWEN4_NATIVE_BATCH_RECEIPT": "1",
                "ATLAS_LOGITS_FNV": "1",
                "ATLAS_QWEN4_TARGET_STATE_RECEIPT": "1",
            }
        )
    if environment != expected_environment:
        raise ProbeError(
            "launch manifest environment differs from the exact selector/diagnostic contract"
        )

    if type(args.server_pid) is not int or args.server_pid <= 0:
        raise ProbeError("--server-pid must be a positive exact integer")
    proc_exe = Path(f"/proc/{args.server_pid}/exe")
    if not proc_exe.exists():
        raise ProbeError("--server-pid is not live")
    if sha256_file(proc_exe) != binary_sha256:
        raise ProbeError("live /proc/PID/exe differs from the sealed binary")
    proc_cmdline = Path(f"/proc/{args.server_pid}/cmdline").read_bytes().split(b"\0")
    if proc_cmdline and proc_cmdline[-1] == b"":
        proc_cmdline.pop()
    try:
        live_argv = [value.decode("utf-8", errors="strict") for value in proc_cmdline]
    except UnicodeDecodeError as error:
        raise ProbeError("live server argv is not UTF-8") from error
    if live_argv != argv:
        raise ProbeError("live /proc/PID/cmdline differs from manifest argv")
    binary_path = provenance.get("binary_path")
    if (
        type(binary_path) is not str
        or not Path(binary_path).is_absolute()
        or not Path(argv[0]).is_absolute()
        or Path(binary_path).resolve() != args.binary.resolve()
        or Path(argv[0]).resolve() != args.binary.resolve()
    ):
        raise ProbeError("manifest argv[0] differs from the sealed binary path")
    validate_r15_source_closure(provenance)
    manifest_cwd = manifest.get("cwd")
    if (
        type(manifest_cwd) is not str
        or not Path(manifest_cwd).is_absolute()
        or Path(f"/proc/{args.server_pid}/cwd").resolve()
        != Path(manifest_cwd).resolve()
    ):
        raise ProbeError("live process cwd differs from manifest cwd")
    selector_prefixes = ("ATLAS_", "FLUX_")
    live_environment: dict[str, str] = {}
    for entry in Path(f"/proc/{args.server_pid}/environ").read_bytes().split(b"\0"):
        if not entry or b"=" not in entry:
            continue
        raw_key, raw_value = entry.split(b"=", 1)
        key = raw_key.decode("utf-8", errors="strict")
        if key.startswith(selector_prefixes) or key == "RUST_LOG":
            live_environment[key] = raw_value.decode("utf-8", errors="strict")
    if live_environment != environment:
        raise ProbeError(
            "live ATLAS_/FLUX_/RUST_LOG environment differs from manifest environment"
        )
    checkpoint_path = provenance["checkpoint_path"]
    checkpoint_manifest_path = provenance["checkpoint_manifest_path"]
    checkpoint_root = Path(checkpoint_path)
    if argv_value(argv, "--model-from-path") != str(checkpoint_root):
        raise ProbeError("manifest model path differs from provenance checkpoint_path")
    checkpoint_manifest = Path(checkpoint_manifest_path)
    checkpoint_inventory, checkpoint_manifest_identity, checkpoint_identities = (
        validate_checkpoint_inventory(
            checkpoint_root,
            checkpoint_manifest,
            provenance["checkpoint_manifest_sha256"],
            checkpoint_id,
            checkpoint_revision,
            verify_bytes=args.evidence_tier == "full",
        )
    )
    source_receipt_path = Path(provenance["source_to_binary_receipt_path"])
    source_receipt_identity = stable_identity(
        source_receipt_path, "source-to-binary receipt"
    )
    pid_start_ticks = process_start_ticks(args.server_pid)
    boot_id = (
        Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    )
    if not re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        boot_id,
    ) or set(boot_id) <= {"0", "-"}:
        raise ProbeError("kernel boot id is malformed")
    log_device_inode = validate_log_binding(args.server_log, args.server_pid)
    public = {
        "manifest_schema": manifest["schema"],
        "manifest_name": args.manifest.name,
        "manifest_sha256": manifest_sha256,
        "binary_name": args.binary.name,
        "binary_sha256": binary_sha256,
        "release_version": provenance["release_version"],
        "source_revision": provenance["source_revision"],
        "source_tree": provenance["source_tree"],
        "source_bundle_manifest_name": Path(
            provenance["source_bundle_manifest_path"]
        ).name,
        "source_bundle_manifest_sha256": provenance["source_bundle_manifest_sha256"],
        "source_bundle_name": Path(provenance["source_bundle_path"]).name,
        "source_bundle_sha256": provenance["source_bundle_sha256"],
        "build_receipt_name": Path(provenance["build_receipt_path"]).name,
        "build_receipt_sha256": provenance["build_receipt_sha256"],
        "source_to_binary_receipt_name": source_receipt_path.name,
        "source_to_binary_receipt_sha256": provenance[
            "source_to_binary_receipt_sha256"
        ],
        "checkpoint_id": provenance["checkpoint_id"],
        "checkpoint_revision": provenance["checkpoint_revision"],
        "checkpoint_manifest_sha256": provenance["checkpoint_manifest_sha256"],
        "checkpoint_files_verified": len(checkpoint_inventory["files"]),
        "checkpoint_bytes_verified": checkpoint_inventory["total_bytes"],
        "selector_profile": args.selector_profile,
        "configured_max_batch_size": max_batch_size,
        "configured_max_num_seqs": max_num_seqs,
        "server_process": {
            "pid": args.server_pid,
            "boot_id": boot_id,
            "start_ticks": pid_start_ticks,
            "log_device": log_device_inode[0],
            "log_inode": log_device_inode[1],
        },
    }
    seals = {
        "binary_identity": binary_identity,
        "manifest_identity": manifest_identity,
        "source_receipt_path": source_receipt_path,
        "source_receipt_identity": source_receipt_identity,
        "checkpoint_manifest": checkpoint_manifest,
        "checkpoint_manifest_identity": checkpoint_manifest_identity,
        "checkpoint_root": checkpoint_root.resolve(),
        "checkpoint_identities": checkpoint_identities,
        "pid_start_ticks": pid_start_ticks,
        "boot_id": boot_id,
        "log_device_inode": log_device_inode,
        "source_closure_provenance": dict(provenance),
    }
    return public, seals


def revalidate_provenance(args: argparse.Namespace, seals: dict[str, Any]) -> None:
    validate_r15_source_closure(seals["source_closure_provenance"])
    if (
        stable_identity(args.binary, "sealed server executable")
        != seals["binary_identity"]
    ):
        raise ProbeError("sealed binary identity changed during qualification")
    if sha256_file(args.binary) != args.binary_sha256:
        raise ProbeError("sealed binary bytes changed during qualification")
    if stable_identity(args.manifest, "launch manifest") != seals["manifest_identity"]:
        raise ProbeError("launch manifest identity changed during qualification")
    if sha256_file(args.manifest) != args.manifest_sha256:
        raise ProbeError("launch manifest bytes changed during qualification")
    if (
        stable_identity(seals["source_receipt_path"], "source-to-binary receipt")
        != seals["source_receipt_identity"]
    ):
        raise ProbeError("source-to-binary receipt changed during qualification")
    if (
        stable_identity(seals["checkpoint_manifest"], "checkpoint inventory")
        != seals["checkpoint_manifest_identity"]
    ):
        raise ProbeError("checkpoint inventory changed during qualification")
    for relative, expected in seals["checkpoint_identities"].items():
        observed = stable_identity(
            seals["checkpoint_root"] / relative, f"checkpoint file {relative}"
        )
        if observed != expected:
            raise ProbeError(
                f"checkpoint file changed during qualification: {relative}"
            )
    if process_start_ticks(args.server_pid) != seals["pid_start_ticks"]:
        raise ProbeError("server PID was reused during qualification")
    if (
        Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
        != seals["boot_id"]
    ):
        raise ProbeError("host rebooted during qualification")
    if (
        validate_log_binding(args.server_log, args.server_pid)
        != seals["log_device_inode"]
    ):
        raise ProbeError("server log binding changed during qualification")
    if sha256_file(Path(f"/proc/{args.server_pid}/exe")) != args.binary_sha256:
        raise ProbeError("live executable changed during qualification")


def make_cases(width: int) -> list[Case]:
    # Heterogeneous prompts perturb prefill completion order and sequence
    # positions while remaining far below the immutable 2,048-token QSA gate.
    repeats = (0, 3, 7, 15, 31, 47, 63, 95)
    cases = []
    for ordinal in range(width):
        filler = " ledger" * repeats[ordinal % len(repeats)]
        content = (
            f"R9 concurrency row {ordinal}. Keep extending a deterministic plain-text "
            f"answer about integer {1009 + ordinal}; do not stop early.{filler}"
        )
        cases.append(Case(f"row-{ordinal:02d}", ordinal, content))
    return cases


def validate_origin(base_url: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.port is None
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ProbeError("--base-url must be an unadorned 127.0.0.1 HTTP origin")
    return base_url.rstrip("/")


def request_json(
    base_url: str, path: str, body: dict[str, Any] | None, timeout: float
) -> tuple[int, bytes]:
    data = None if body is None else json_canonical(body)
    request = urllib.request.Request(
        base_url + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if body is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def healthcheck(base_url: str, timeout: float) -> None:
    status, raw = request_json(base_url, "/health", None, timeout)
    if status != 200:
        raise ProbeError(f"health returned HTTP {status}: {raw[:200]!r}")


def request_body(model: str, case: Case, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": case.content}],
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,
        "ignore_eos": True,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "top_n_sigma": 0.0,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "enable_thinking": False,
        "stream": True,
        "return_token_ids": True,
        "stream_options": {"include_usage": True},
    }


def stream_request(
    base_url: str, model: str, case: Case, max_tokens: int, timeout: float
) -> RequestResult:
    started = time.time()
    request = urllib.request.Request(
        base_url + "/v1/chat/completions",
        data=json_canonical(request_body(model, case, max_tokens)),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    token_ids: list[int] | None = None
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    done_count = 0
    event_count = 0
    usage_event: int | None = None
    finish_event: int | None = None
    token_ids_event: int | None = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise ProbeError(f"{case.label}: HTTP {response.status}")
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="strict").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    done_count += 1
                    if done_count != 1:
                        raise ProbeError(f"{case.label}: duplicate [DONE] sentinel")
                    continue
                if done_count:
                    raise ProbeError(f"{case.label}: SSE data appeared after [DONE]")
                if not payload:
                    raise ProbeError(f"{case.label}: empty SSE data event")
                value = json_loads_strict(payload, f"{case.label} SSE event")
                if type(value) is not dict:
                    raise ProbeError(f"{case.label}: SSE event must be an object")
                if "error" in value:
                    raise ProbeError(f"{case.label}: stream error {value['error']}")
                if value.get("usage") is not None:
                    if usage_event is not None:
                        raise ProbeError(f"{case.label}: terminal usage appeared twice")
                    usage = value["usage"]
                    usage_event = event_count
                choices = value.get("choices")
                if type(choices) is not list:
                    raise ProbeError(f"{case.label}: SSE choices must be an array")
                if value.get("usage") is not None and choices:
                    raise ProbeError(
                        f"{case.label}: terminal usage event must have no choices"
                    )
                for choice in choices:
                    if type(choice) is not dict:
                        raise ProbeError(f"{case.label}: SSE choice must be an object")
                    if choice.get("finish_reason") is not None:
                        if (
                            len(choices) != 1
                            or type(choice.get("index")) is not int
                            or choice["index"] != 0
                        ):
                            raise ProbeError(
                                f"{case.label}: finish event must contain exact choice index 0"
                            )
                        if finish_event is not None:
                            raise ProbeError(
                                f"{case.label}: finish_reason appeared twice"
                            )
                        finish_reason = choice["finish_reason"]
                        finish_event = event_count
                    if "token_ids" in choice:
                        if (
                            len(choices) != 1
                            or type(choice.get("index")) is not int
                            or choice["index"] != 0
                        ):
                            raise ProbeError(
                                f"{case.label}: token-ID event must contain exact choice index 0"
                            )
                        if token_ids is not None:
                            raise ProbeError(f"{case.label}: token_ids appeared twice")
                        token_ids = choice["token_ids"]
                        token_ids_event = event_count
                event_count += 1
    except urllib.error.HTTPError as error:
        raise ProbeError(
            f"{case.label}: HTTP {error.code}: {error.read()[:300]!r}"
        ) from error
    except urllib.error.URLError as error:
        raise ProbeError(f"{case.label}: request failed: {error}") from error

    if done_count != 1:
        raise ProbeError(f"{case.label}: expected exactly one [DONE] sentinel")
    if type(token_ids) is not list or not all(
        type(token) is int and 0 <= token <= 0xFFFF_FFFF for token in token_ids
    ):
        raise ProbeError(f"{case.label}: missing or malformed terminal token_ids")
    if (
        usage_event != event_count - 1
        or finish_event is None
        or token_ids_event is None
        or finish_event != token_ids_event
        or finish_event != usage_event - 1
    ):
        raise ProbeError(
            f"{case.label}: terminal finish/token-ID/usage event order differs"
        )
    validate_usage(usage, len(token_ids), f"{case.label}.usage")
    if len(token_ids) != max_tokens:
        raise ProbeError(
            f"{case.label}: expected {max_tokens} tokens, got {len(token_ids)}"
        )
    # Atlas records a sampled ChatML <|im_start|> hard boundary in the terminal
    # token IDs and reports `stop`, even when that token lands exactly at the
    # max-token boundary under ignore_eos. Exact cardinality is the budget
    # invariant. The matched C1 comparison below separately requires the same
    # finish reason for isolated and concurrent execution.
    if finish_reason not in EXACT_BUDGET_FINISH_REASONS:
        raise ProbeError(
            f"{case.label}: expected exact-budget length/stop finish, got {finish_reason!r}"
        )
    prompt_tokens = usage["prompt_tokens"]
    if prompt_tokens + max_tokens >= 2_048:
        raise ProbeError(
            f"{case.label}: prompt/output budget is not native-short ({prompt_tokens}+{max_tokens})"
        )
    return RequestResult(
        label=case.label,
        token_ids=token_ids,
        usage=usage,
        finish_reason=finish_reason,
        started_unix=started,
        finished_unix=time.time(),
    )


def stream_then_cancel(
    base_url: str,
    model: str,
    case: Case,
    max_tokens: int,
    timeout: float,
    cancel_after_chunks: int,
    cancelled: threading.Event,
) -> CancelResult:
    started = time.time()
    request = urllib.request.Request(
        base_url + "/v1/chat/completions",
        data=json_canonical(request_body(model, case, max_tokens)),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    chunks = 0
    reached_cancel_boundary = False
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise ProbeError(f"{case.label}: HTTP {response.status}")
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="strict").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    raise ProbeError(f"{case.label}: completed before cancellation")
                value = json_loads_strict(payload, f"{case.label} SSE event")
                if type(value) is not dict:
                    raise ProbeError(f"{case.label}: SSE event must be an object")
                if "error" in value:
                    raise ProbeError(f"{case.label}: stream error {value['error']}")
                if value.get("usage") is not None:
                    raise ProbeError(
                        f"{case.label}: terminal usage arrived before cancellation"
                    )
                choices = value.get("choices")
                if type(choices) is not list:
                    raise ProbeError(f"{case.label}: SSE choices must be an array")
                for choice in choices:
                    if type(choice) is not dict:
                        raise ProbeError(f"{case.label}: SSE choice must be an object")
                    if choice.get("finish_reason") is not None or "token_ids" in choice:
                        raise ProbeError(
                            f"{case.label}: terminal event arrived before cancellation"
                        )
                    delta = choice.get("delta") or {}
                    if type(delta) is not dict:
                        raise ProbeError(f"{case.label}: SSE delta must be an object")
                    if delta.get("content") or delta.get("reasoning_content"):
                        chunks += 1
                if chunks > cancel_after_chunks:
                    raise ProbeError(
                        f"{case.label}: cancellation boundary overshot "
                        f"({chunks} > {cancel_after_chunks} content chunks)"
                    )
                if chunks == cancel_after_chunks:
                    reached_cancel_boundary = True
                    break
    except urllib.error.HTTPError as error:
        raise ProbeError(
            f"{case.label}: HTTP {error.code}: {error.read()[:300]!r}"
        ) from error
    except urllib.error.URLError as error:
        raise ProbeError(f"{case.label}: request failed: {error}") from error
    if not reached_cancel_boundary:
        raise ProbeError(f"{case.label}: stream ended before cancellation")

    # urllib's response context has exited here, so the server-facing stream is
    # closed before the replacement is allowed to leave its wait. Stamping and
    # signalling inside the context creates a race where both requests can own
    # capacity concurrently and does not prove post-cancellation reuse.
    closed = time.time()
    cancelled.set()
    return CancelResult(case.label, chunks, started, closed)


def validate_evidence_line_syntax(text: str) -> None:
    patterns: dict[str, re.Pattern[str]] = {
        RECEIPT_MARKERS[0]: FRAME_RE,
        RECEIPT_MARKERS[1]: FRAME_RE,
        RECEIPT_MARKERS[2]: LOGITS_RE,
        RECEIPT_MARKERS[3]: STATE_RE,
        "QWEN4_NATIVE_BATCH_ENGAGED": ENGAGED_RE,
    }
    patterns.update(
        {
            OPTIMIZATION_ENGAGEMENT_MARKERS[selector].rstrip(": "): pattern
            for selector, pattern in OPTIMIZATION_ENGAGEMENT_RES.items()
        }
    )
    for line_number, line in enumerate(text.splitlines(), start=1):
        present = [marker for marker in patterns if marker in line]
        if not present:
            continue
        if len(present) != 1 or line.count(present[0]) != 1:
            raise ProbeError(
                f"line {line_number}: multiple qualification evidence markers"
            )
        marker = present[0]
        if patterns[marker].search(line) is None:
            raise ProbeError(
                f"line {line_number}: malformed {marker.rstrip()} qualification evidence"
            )


def parse_log(text: str) -> ParsedLog:
    validate_evidence_line_syntax(text)
    rows_by_session: dict[str, dict[int, LogitRow]] = {}
    frames: list[Frame] = []
    state_by_session: dict[str, StateReceipt] = {}
    pending: Frame | None = None
    for line_number, line in enumerate(text.splitlines(), start=1):
        frame_match = FRAME_RE.search(line)
        if frame_match:
            if pending is not None and len(pending.logits) != pending.rows:
                raise ProbeError(
                    "a batch frame did not emit exactly one logits receipt per row"
                )
            route, rows, sessions, slots, seq_lens, selector_on, qsa = (
                frame_match.groups()
            )
            pending = Frame(
                route=route.lower(),
                rows=int(rows),
                sessions=sessions.split(","),
                slots=[int(value) for value in slots.split(",")],
                pre_step_seq_lens=[int(value) for value in seq_lens.split(",")],
                selector_on=None if selector_on is None else selector_on == "true",
                qsa_required=None if qsa is None else qsa == "true",
                logits=[],
                line_number=line_number,
            )
            if (
                len(pending.sessions) != pending.rows
                or len(pending.slots) != pending.rows
                or len(pending.pre_step_seq_lens) != pending.rows
            ):
                raise ProbeError("batch frame metadata cardinality differs from rows")
            if len(set(pending.sessions)) != pending.rows:
                raise ProbeError("batch frame aliases a logical session")
            if len(set(pending.slots)) != pending.rows:
                raise ProbeError("batch frame aliases a physical slot")
            frames.append(pending)
            continue
        state_match = STATE_RE.search(line)
        if state_match:
            (
                schema,
                mode,
                scheduler_path,
                proposer_loaded,
                session,
                seq_len,
                token_count,
                output_tokens,
                gdn_layers,
                attention_layers,
                logical_kv_blocks,
                kv_token_positions,
                compared_buffers,
                compared_bytes,
                fed_tokens_sha256,
                target_state_sha256,
            ) = state_match.groups()
            if session in state_by_session:
                raise ProbeError(
                    f"duplicate target-state receipt session={session} within one phase"
                )
            state_by_session[session] = StateReceipt(
                session=session,
                schema=int(schema),
                mode=mode,
                scheduler_path=scheduler_path,
                proposer_loaded=proposer_loaded == "true",
                seq_len=int(seq_len),
                token_count=int(token_count),
                output_tokens=int(output_tokens),
                gdn_layers=int(gdn_layers),
                attention_layers=int(attention_layers),
                logical_kv_blocks=int(logical_kv_blocks),
                kv_token_positions=int(kv_token_positions),
                compared_buffers=int(compared_buffers),
                compared_bytes=int(compared_bytes),
                fed_tokens_sha256=fed_tokens_sha256,
                target_state_sha256=target_state_sha256,
                line_number=line_number,
            )
            continue
        logits_match = LOGITS_RE.search(line)
        if not logits_match:
            continue
        session, index, fnv, sha256 = logits_match.groups()
        row = LogitRow(session, int(index), fnv, sha256)
        session_rows = rows_by_session.setdefault(session, {})
        if row.index in session_rows:
            raise ProbeError(
                f"duplicate logits receipt session={session} idx={row.index} within one phase"
            )
        session_rows[row.index] = row
        if pending is not None:
            pending.logits.append(row)
            if len(pending.logits) > pending.rows:
                raise ProbeError("batch frame emitted too many logits receipts")
            if len(pending.logits) == pending.rows:
                if [row.session for row in pending.logits] != pending.sessions:
                    raise ProbeError(
                        "batch frame session order differs from its logits rows"
                    )
                pending = None
            continue
    if pending is not None:
        raise ProbeError("final batch frame is incomplete")
    return ParsedLog(
        rows_by_session,
        frames,
        state_by_session,
        hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def sole_session(parsed: ParsedLog, label: str) -> str:
    if len(parsed.rows_by_session) != 1:
        raise ProbeError(
            f"{label}: isolated C1 log has {len(parsed.rows_by_session)} sessions"
        )
    return next(iter(parsed.rows_by_session))


def row_digest_map(rows: dict[int, LogitRow]) -> dict[int, str]:
    return {index: row.sha256 for index, row in sorted(rows.items())}


def serialize_logit_rows(rows: dict[int, LogitRow]) -> list[dict[str, Any]]:
    return [
        {"index": row.index, "fnv": row.fnv, "sha256": row.sha256}
        for _, row in sorted(rows.items())
    ]


def serialize_frame(frame: Frame) -> dict[str, Any]:
    return {
        "line_number": frame.line_number,
        "route": frame.route,
        "rows": frame.rows,
        "sessions": frame.sessions,
        "slots": frame.slots,
        "pre_step_seq_lens": frame.pre_step_seq_lens,
        "logits": [
            {
                "session": row.session,
                "index": row.index,
                "fnv": row.fnv,
                "sha256": row.sha256,
            }
            for row in frame.logits
        ],
    }


def validate_state_receipt(
    scope: str,
    receipt: StateReceipt,
    expected_output_tokens: int,
    expected_prompt_tokens: int,
) -> None:
    if (
        receipt.schema != 1
        or receipt.mode != "unloaded-target-only"
        or receipt.scheduler_path != "ordinary-target-only"
        or receipt.proposer_loaded
    ):
        raise ProbeError(f"{scope}: target-state receipt is not target-only")
    if (
        receipt.seq_len != receipt.token_count
        or receipt.output_tokens != expected_output_tokens
        or receipt.seq_len != expected_prompt_tokens + expected_output_tokens - 1
    ):
        raise ProbeError(f"{scope}: target-state receipt token census differs")
    if receipt.gdn_layers != 36 or receipt.attention_layers != 12:
        raise ProbeError(f"{scope}: target-state receipt layer census differs")
    if (
        receipt.compared_buffers <= 0
        or receipt.compared_bytes <= 0
        or receipt.logical_kv_blocks <= 0
        or receipt.kv_token_positions != receipt.seq_len * receipt.attention_layers
    ):
        raise ProbeError(f"{scope}: target-state receipt coverage is incomplete")


def cancellation_geometry(
    scope: str, output_tokens: int, row_indices: list[int]
) -> str:
    if output_tokens < 1:
        raise ProbeError(f"{scope}: cancelled output count is invalid")
    if row_indices == list(range(1, output_tokens)):
        return "pending-output"
    if row_indices == list(range(1, output_tokens + 1)):
        return "computed-unemitted"
    raise ProbeError(f"{scope}: cancelled request logits prefix is incomplete")


def validate_cancelled_state_receipt(
    scope: str,
    receipt: StateReceipt,
    expected_prompt_tokens: int,
    rows: dict[int, LogitRow],
) -> str:
    if any(
        row.session != receipt.session or row.index != index
        for index, row in rows.items()
    ):
        raise ProbeError(f"{scope}: cancelled logits row identity differs")
    geometry = cancellation_geometry(scope, receipt.output_tokens, sorted(rows))
    # Every raw decode row consumed one input. The final row may have been
    # computed before cancellation suppressed publication of its sampled token.
    shadow = dataclasses.replace(receipt, output_tokens=len(rows) + 1)
    validate_state_receipt(scope, shadow, len(rows) + 1, expected_prompt_tokens)
    return geometry


def validate_cancelled_frame_extent(
    scope: str,
    frames: list[Frame],
    receipt: StateReceipt,
    rows: dict[int, LogitRow],
) -> None:
    target_frames = [frame for frame in frames if receipt.session in frame.sessions]
    if not target_frames or not rows:
        raise ProbeError(f"{scope}: cancelled final logits row has no batch frame")
    last_frame = target_frames[-1]
    position = last_frame.sessions.index(receipt.session)
    if (
        last_frame.logits[position] != rows[max(rows)]
        or receipt.seq_len != last_frame.pre_step_seq_lens[position] + 1
    ):
        raise ProbeError(
            f"{scope}: cancelled final frame does not bind target-fed extent"
        )


def state_signature(receipt: StateReceipt) -> dict[str, Any]:
    return {
        "schema": receipt.schema,
        "mode": receipt.mode,
        "scheduler_path": receipt.scheduler_path,
        "proposer_loaded": receipt.proposer_loaded,
        "session": receipt.session,
        "seq_len": receipt.seq_len,
        "token_count": receipt.token_count,
        "output_tokens": receipt.output_tokens,
        "gdn_layers": receipt.gdn_layers,
        "attention_layers": receipt.attention_layers,
        "logical_kv_blocks": receipt.logical_kv_blocks,
        "kv_token_positions": receipt.kv_token_positions,
        "compared_buffers": receipt.compared_buffers,
        "compared_bytes": receipt.compared_bytes,
        "fed_tokens_sha256": receipt.fed_tokens_sha256,
        "target_state_sha256": receipt.target_state_sha256,
    }


def serialize_state_receipt(receipt: StateReceipt) -> dict[str, Any]:
    return {**state_signature(receipt), "line_number": receipt.line_number}


def run_wave(
    base_url: str,
    model: str,
    cases: list[Case],
    max_tokens: int,
    timeout: float,
) -> list[RequestResult]:
    barrier = threading.Barrier(len(cases))

    order = {case.label: position for position, case in enumerate(cases)}

    def worker(case: Case) -> RequestResult:
        barrier.wait(timeout=timeout)
        # Preserve the requested adversarial admission order even when the
        # client's thread scheduler wakes barrier waiters in another order.
        # Keep the whole C8 launch window well below one observed short-prompt
        # prefill so the once-per-process engagement marker cannot be consumed
        # by an accidental C2-C7 frame before all eight requests are admitted.
        time.sleep(order[case.label] * WAVE_ADMISSION_STAGGER_S)
        return stream_request(base_url, model, case, max_tokens, timeout)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(cases)) as pool:
        futures = [pool.submit(worker, case) for case in cases]
        results = [future.result(timeout=timeout) for future in futures]
    by_label = {result.label: result for result in results}
    return [by_label[case.label] for case in cases]


def validate_cancel_handoff(
    cancelled: CancelResult,
    replacement: RequestResult,
    expected_chunks: int,
) -> None:
    if cancelled.content_chunks != expected_chunks:
        raise ProbeError(
            f"churn/cancelled: expected exactly {expected_chunks} content chunks, "
            f"got {cancelled.content_chunks}"
        )
    if replacement.started_unix < cancelled.closed_unix:
        raise ProbeError(
            "churn/replacement: request started before the cancelled response closed"
        )


def run_api_churn(
    base_url: str,
    model: str,
    cases: list[Case],
    replacement: Case,
    max_tokens: int,
    timeout: float,
    baseline_results: dict[str, RequestResult],
) -> dict[str, Any]:
    barrier = threading.Barrier(len(cases))
    cancelled = threading.Event()
    # Cancel the last ordered admission. With capacity C8 this is the cleanest
    # deterministic exercise of retiring and reusing the highest live slot.
    target = cases[-1]
    order = {case.label: position for position, case in enumerate(cases)}

    def original(case: Case) -> RequestResult | CancelResult:
        barrier.wait(timeout=timeout)
        time.sleep(order[case.label] * WAVE_ADMISSION_STAGGER_S)
        if case == target:
            return stream_then_cancel(
                base_url,
                model,
                case,
                max_tokens,
                timeout,
                cancel_after_chunks=8,
                cancelled=cancelled,
            )
        return stream_request(base_url, model, case, max_tokens, timeout)

    def replacement_request() -> RequestResult:
        if not cancelled.wait(timeout=timeout):
            raise ProbeError("replacement timed out waiting for cancellation")
        return stream_request(base_url, model, replacement, max_tokens, timeout)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(cases) + 1) as pool:
        originals = {case.label: pool.submit(original, case) for case in cases}
        replacement_future = pool.submit(replacement_request)
        original_results = {
            label: future.result(timeout=timeout) for label, future in originals.items()
        }
        replacement_result = replacement_future.result(timeout=timeout)

    cancel_result = original_results[target.label]
    if not isinstance(cancel_result, CancelResult):
        raise ProbeError("churn target was not cancelled")
    validate_cancel_handoff(cancel_result, replacement_result, expected_chunks=8)
    survivors: list[RequestResult] = []
    for case in cases:
        if case == target:
            continue
        result = original_results[case.label]
        if not isinstance(result, RequestResult):
            raise ProbeError(f"{case.label}: survivor did not finish")
        validate_request_identity(
            f"churn/{case.label}", result, baseline_results[case.label]
        )
        survivors.append(result)
    validate_request_identity(
        "churn/replacement", replacement_result, baseline_results[replacement.label]
    )

    return {
        "target": target,
        "cancel_result": cancel_result,
        "survivor_results": survivors,
        "replacement_result": replacement_result,
    }


def serialize_result(result: RequestResult) -> dict[str, Any]:
    return {
        "label": result.label,
        "prompt_tokens": result.usage["prompt_tokens"],
        "completion_tokens": result.usage["completion_tokens"],
        "usage_identity": {
            field: result.usage.get(field) for field in USAGE_IDENTITY_FIELDS
        },
        "token_ids_sha256": hashlib.sha256(
            json_canonical(result.token_ids)
        ).hexdigest(),
        "token_ids": result.token_ids,
        "finish_reason": result.finish_reason,
        "elapsed_s": round(result.finished_unix - result.started_unix, 6),
    }


def validate_request_identity(
    scope: str,
    observed: RequestResult,
    baseline: RequestResult,
) -> None:
    if observed.token_ids != baseline.token_ids:
        raise ProbeError(f"{scope}: exact output token IDs differ from C1")
    if observed.finish_reason != baseline.finish_reason:
        raise ProbeError(
            f"{scope}: finish reason differs from C1 "
            f"({observed.finish_reason!r} != {baseline.finish_reason!r})"
        )
    for field in USAGE_IDENTITY_FIELDS:
        if observed.usage.get(field) != baseline.usage.get(field):
            raise ProbeError(f"{scope}: usage.{field} differs from C1")


def validate_token_wave(
    name: str,
    cases: list[Case],
    results: list[RequestResult],
    baseline_results: dict[str, RequestResult],
) -> dict[str, Any]:
    expected_order = [case.label for case in cases]
    if [result.label for result in results] != expected_order:
        raise ProbeError(f"{name}: request result order differs from admission plan")
    result_by_label = {result.label: result for result in results}
    for case in cases:
        validate_request_identity(
            f"{name}/{case.label}",
            result_by_label[case.label],
            baseline_results[case.label],
        )
    return {
        "name": name,
        "request_order": expected_order,
        "requests": [serialize_result(result) for result in results],
    }


def validate_frame_ownership(
    scope: str,
    frames: list[Frame],
    state_by_session: dict[str, StateReceipt],
    *,
    allow_retired_slot_reuse: bool,
) -> dict[str, int]:
    session_slots: dict[str, int] = {}
    live_slot_owners: dict[int, str] = {}
    previous_seq_len: dict[str, int] = {}
    for frame in frames:
        for session, slot, seq_len in zip(
            frame.sessions, frame.slots, frame.pre_step_seq_lens
        ):
            retired = state_by_session.get(session)
            if retired is not None and retired.line_number <= frame.line_number:
                raise ProbeError(
                    f"{scope}: retired session {session} re-entered a frame"
                )
            prior_slot = session_slots.setdefault(session, slot)
            if prior_slot != slot:
                raise ProbeError(
                    f"{scope}: logical session {session} moved physical slots"
                )
            prior_owner = live_slot_owners.get(slot)
            if prior_owner is not None and prior_owner != session:
                owner_retirement = state_by_session.get(prior_owner)
                if (
                    not allow_retired_slot_reuse
                    or owner_retirement is None
                    or owner_retirement.line_number >= frame.line_number
                ):
                    raise ProbeError(
                        f"{scope}: physical slot {slot} changed live owner"
                    )
            live_slot_owners[slot] = session
            prior_seq_len = previous_seq_len.get(session)
            if prior_seq_len is not None and seq_len != prior_seq_len + 1:
                raise ProbeError(
                    f"{scope}: session {session} sequence length is not stepwise"
                )
            previous_seq_len[session] = seq_len
    return session_slots


def validate_wave(
    name: str,
    expected_route: str,
    width: int,
    cases: list[Case],
    results: list[RequestResult],
    parsed: ParsedLog,
    baseline_results: dict[str, RequestResult],
    baseline_sessions: dict[str, str],
    baseline_logits: dict[str, dict[int, str]],
    baseline_states: dict[str, StateReceipt],
    minimum_full_frames: int,
) -> dict[str, Any]:
    expected_order = [case.label for case in cases]
    if [result.label for result in results] != expected_order:
        raise ProbeError(f"{name}: request result order differs from admission plan")
    case_by_session = {session: label for label, session in baseline_sessions.items()}
    result_by_label = {result.label: result for result in results}
    for case in cases:
        validate_request_identity(
            f"{name}/{case.label}",
            result_by_label[case.label],
            baseline_results[case.label],
        )
        session = baseline_sessions[case.label]
        observed = parsed.rows_by_session.get(session)
        if observed is None:
            raise ProbeError(
                f"{name}/{case.label}: no full-logits receipts for session {session}"
            )
        if row_digest_map(observed) != baseline_logits[case.label]:
            raise ProbeError(
                f"{name}/{case.label}: full-logits SHA-256 differs from C1"
            )
        state = parsed.state_by_session.get(session)
        if state is None:
            raise ProbeError(f"{name}/{case.label}: target-state receipt is missing")
        validate_state_receipt(
            name + "/" + case.label,
            state,
            len(result_by_label[case.label].token_ids),
            result_by_label[case.label].usage["prompt_tokens"],
        )
        if state_signature(state) != state_signature(baseline_states[case.label]):
            raise ProbeError(
                f"{name}/{case.label}: terminal KV/recurrent state differs from C1"
            )

    expected_sessions = {baseline_sessions[case.label] for case in cases}
    if set(parsed.rows_by_session) != expected_sessions:
        raise ProbeError(f"{name}: logits session set differs from the request wave")
    if set(parsed.state_by_session) != expected_sessions:
        raise ProbeError(
            f"{name}: target-state session set differs from the request wave"
        )

    full = [
        frame
        for frame in parsed.frames
        if frame.rows == width and frame.route == expected_route
    ]
    if len(full) < minimum_full_frames:
        raise ProbeError(
            f"{name}: saw {len(full)} complete {expected_route} C{width} frames; "
            f"need {minimum_full_frames}"
        )
    wrong = [
        frame
        for frame in parsed.frames
        if 2 <= frame.rows <= 8 and frame.route != expected_route
    ]
    if wrong:
        raise ProbeError(f"{name}: a C2-C8 frame used the wrong route")
    session_slots = validate_frame_ownership(
        name, parsed.frames, parsed.state_by_session, allow_retired_slot_reuse=False
    )
    for frame in parsed.frames:
        if not 2 <= frame.rows <= 8:
            raise ProbeError(f"{name}: unexpected frame width C{frame.rows}")
        if expected_route == "native":
            if frame.selector_on is not None or frame.qsa_required is not None:
                raise ProbeError(f"{name}: native frame metadata is malformed")
        elif frame.selector_on is not False or frame.qsa_required is not False:
            raise ProbeError(
                f"{name}: serialized control did not prove selector-off short context"
            )
        if (
            len(frame.logits) != frame.rows
            or len({row.session for row in frame.logits}) != frame.rows
        ):
            raise ProbeError(
                f"{name}: C{frame.rows} frame did not contain independent sessions"
            )
        unknown = [
            row.session for row in frame.logits if row.session not in case_by_session
        ]
        if unknown:
            raise ProbeError(f"{name}: unknown sessions in frame: {unknown}")
        for session, seq_len, row in zip(
            frame.sessions, frame.pre_step_seq_lens, frame.logits
        ):
            label = case_by_session[row.session]
            expected_seq_len = (
                result_by_label[label].usage["prompt_tokens"] + row.index - 1
            )
            if seq_len != expected_seq_len:
                raise ProbeError(
                    f"{name}/{label}: frame sequence length differs from output index"
                )
            expected = baseline_logits[label].get(row.index)
            if expected != row.sha256:
                raise ProbeError(
                    f"{name}/{label}: frame logits differ at output index {row.index}"
                )
    first_order = [case_by_session[row.session] for row in full[0].logits]
    if first_order != expected_order:
        raise ProbeError(
            f"{name}: first full-frame order differs from exact request order"
        )
    expected_slots = full[0].slots
    for frame_index, frame in enumerate(full):
        observed_order = [case_by_session[session] for session in frame.sessions]
        if observed_order != expected_order:
            raise ProbeError(
                f"{name}: full frame {frame_index} differs from exact request order"
            )
        if frame.slots != expected_slots:
            raise ProbeError(
                f"{name}: full frame {frame_index} differs from stable slot vector"
            )
    return {
        "name": name,
        "route": expected_route,
        "request_order": expected_order,
        "full_width_frames": len(full),
        "first_full_frame_order": first_order,
        "slot_sets": [frame.slots for frame in full],
        "session_slots": session_slots,
        "logits": {
            case.label: serialize_logit_rows(
                parsed.rows_by_session[baseline_sessions[case.label]]
            )
            for case in cases
        },
        "terminal_states": {
            case.label: serialize_state_receipt(
                parsed.state_by_session[baseline_sessions[case.label]]
            )
            for case in cases
        },
        "frames": [serialize_frame(frame) for frame in parsed.frames],
        "state_sha256": {
            case.label: parsed.state_by_session[
                baseline_sessions[case.label]
            ].target_state_sha256
            for case in cases
        },
        "log_sha256": parsed.text_sha256,
        "requests": [serialize_result(result) for result in results],
    }


def validate_reuse_phase(
    parsed: ParsedLog,
    case: Case,
    result: RequestResult,
    baseline_result: RequestResult,
    baseline_session: str,
    baseline_logits: dict[int, str],
    baseline_state: StateReceipt,
) -> dict[str, Any]:
    validate_request_identity("churn/post-drain-reuse", result, baseline_result)
    if parsed.frames:
        raise ProbeError(
            "churn/post-drain-reuse unexpectedly overlapped another request"
        )
    if set(parsed.rows_by_session) != {baseline_session}:
        raise ProbeError("churn/post-drain-reuse logits session differs")
    if row_digest_map(parsed.rows_by_session[baseline_session]) != baseline_logits:
        raise ProbeError("churn/post-drain-reuse full logits differ from C1")
    state = parsed.state_by_session.get(baseline_session)
    if state is None or set(parsed.state_by_session) != {baseline_session}:
        raise ProbeError("churn/post-drain-reuse target-state receipt differs")
    validate_state_receipt(
        "churn/post-drain-reuse",
        state,
        len(result.token_ids),
        result.usage["prompt_tokens"],
    )
    if state_signature(state) != state_signature(baseline_state):
        raise ProbeError("churn/post-drain-reuse terminal state differs from C1")
    logits = serialize_logit_rows(parsed.rows_by_session[baseline_session])
    return {
        "request": serialize_result(result),
        "session": baseline_session,
        "logits": logits,
        "logits_sha256_chain": hashlib.sha256(json_canonical(logits)).hexdigest(),
        "target_state": serialize_state_receipt(state),
        "frames": [],
        "log_sha256": parsed.text_sha256,
    }


def validate_full_churn(
    expected_route: str,
    width: int,
    cases: list[Case],
    replacement: Case,
    churn: dict[str, Any],
    parsed: ParsedLog,
    baseline_results: dict[str, RequestResult],
    baseline_sessions: dict[str, str],
    baseline_logits: dict[str, dict[int, str]],
    baseline_states: dict[str, StateReceipt],
) -> dict[str, Any]:
    target: Case = churn["target"]
    cancel_result: CancelResult = churn["cancel_result"]
    replacement_result: RequestResult = churn["replacement_result"]
    survivors: list[RequestResult] = churn["survivor_results"]
    target_session = baseline_sessions[target.label]
    replacement_session = baseline_sessions[replacement.label]
    known_sessions = {baseline_sessions[case.label] for case in cases}
    known_sessions.add(replacement_session)
    if set(parsed.rows_by_session) != known_sessions:
        raise ProbeError(
            "churn: logits session set differs from originals plus replacement"
        )
    if set(parsed.state_by_session) != known_sessions:
        raise ProbeError(
            "churn: target-state session set differs from originals plus replacement"
        )
    for result in [*survivors, replacement_result]:
        label = result.label
        session = baseline_sessions[label]
        if row_digest_map(parsed.rows_by_session[session]) != baseline_logits[label]:
            raise ProbeError(f"churn/{label}: full logits differ from C1")
        state = parsed.state_by_session[session]
        validate_state_receipt(
            f"churn/{label}",
            state,
            len(result.token_ids),
            result.usage["prompt_tokens"],
        )
        if state_signature(state) != state_signature(baseline_states[label]):
            raise ProbeError(f"churn/{label}: terminal state differs from C1")

    cancelled_state = parsed.state_by_session[target_session]
    if (
        not 0
        < cancelled_state.output_tokens
        < baseline_results[target.label].usage["completion_tokens"]
    ):
        raise ProbeError("churn: cancelled request target-state token count is invalid")
    cancelled_geometry = validate_cancelled_state_receipt(
        "churn/cancelled",
        cancelled_state,
        baseline_results[target.label].usage["prompt_tokens"],
        parsed.rows_by_session[target_session],
    )
    cancelled_rows = row_digest_map(parsed.rows_by_session[target_session])
    for index, digest in cancelled_rows.items():
        if baseline_logits[target.label].get(index) != digest:
            raise ProbeError(
                f"churn/{target.label}: cancelled logits differ at output index {index}"
            )
    wrong = [
        frame
        for frame in parsed.frames
        if 2 <= frame.rows <= 8 and frame.route != expected_route
    ]
    if wrong:
        raise ProbeError("churn: a C2-C8 frame used the wrong route")
    for frame in parsed.frames:
        if expected_route == "native":
            if frame.selector_on is not None or frame.qsa_required is not None:
                raise ProbeError("churn: native frame metadata is malformed")
        elif frame.selector_on is not False or frame.qsa_required is not False:
            raise ProbeError(
                "churn: serialized control did not prove selector-off short context"
            )
    session_slots = validate_frame_ownership(
        "churn",
        parsed.frames,
        parsed.state_by_session,
        allow_retired_slot_reuse=True,
    )
    labels_by_session = {
        baseline_sessions[case.label]: case.label for case in [*cases, replacement]
    }
    for frame in parsed.frames:
        for session, seq_len, row in zip(
            frame.sessions, frame.pre_step_seq_lens, frame.logits
        ):
            label = labels_by_session[session]
            expected_seq_len = (
                baseline_results[label].usage["prompt_tokens"] + row.index - 1
            )
            if seq_len != expected_seq_len:
                raise ProbeError(
                    f"churn/{label}: frame sequence length differs from output index"
                )
    target_frames = [
        frame for frame in parsed.frames if target_session in frame.sessions
    ]
    replacement_frames = [
        frame for frame in parsed.frames if replacement_session in frame.sessions
    ]
    if not target_frames or not replacement_frames:
        raise ProbeError("churn: target or replacement never entered a batch frame")
    validate_cancelled_frame_extent(
        "churn/cancelled",
        parsed.frames,
        cancelled_state,
        parsed.rows_by_session[target_session],
    )
    original_sessions = [baseline_sessions[case.label] for case in cases]
    replacement_sessions = [
        *[baseline_sessions[result.label] for result in survivors],
        replacement_session,
    ]
    full_frames = [frame for frame in parsed.frames if frame.rows == width]
    original_full_frames = [
        frame for frame in full_frames if target_session in frame.sessions
    ]
    replacement_full_frames = [
        frame for frame in full_frames if replacement_session in frame.sessions
    ]
    if not original_full_frames or not replacement_full_frames:
        raise ProbeError("churn: original or replacement phase lacks a full C8 frame")
    stable_slots = original_full_frames[0].slots
    for frame_index, frame in enumerate(full_frames):
        if target_session in frame.sessions:
            expected_sessions = original_sessions
        elif replacement_session in frame.sessions:
            expected_sessions = replacement_sessions
        else:
            raise ProbeError("churn: full C8 frame belongs to neither phase")
        if frame.sessions != expected_sessions:
            raise ProbeError(
                f"churn: full frame {frame_index} differs from exact phase order"
            )
        if frame.slots != stable_slots:
            raise ProbeError(
                f"churn: full frame {frame_index} differs from stable slot vector"
            )
    last_target = target_frames[-1]
    first_replacement = replacement_frames[0]
    target_slot = last_target.slots[last_target.sessions.index(target_session)]
    replacement_slot = first_replacement.slots[
        first_replacement.sessions.index(replacement_session)
    ]
    if not (
        last_target.line_number
        < cancelled_state.line_number
        < first_replacement.line_number
    ):
        raise ProbeError("churn: replacement entered before cancelled state retirement")
    if replacement_slot != target_slot:
        raise ProbeError(
            f"churn: replacement used slot {replacement_slot}, not retired slot {target_slot}"
        )
    if (
        original_full_frames[-1].sessions[-1] != target_session
        or original_full_frames[-1].slots[-1] != target_slot
        or target_slot != max(original_full_frames[-1].slots)
    ):
        raise ProbeError(
            "churn: cancelled final admission was not the highest live slot"
        )
    return {
        "route": expected_route,
        "cancelled": {
            "label": cancel_result.label,
            "content_chunks_before_close": cancel_result.content_chunks,
            "closed_unix": cancel_result.closed_unix,
            "session": target_session,
            "state_output_tokens": cancelled_state.output_tokens,
            "terminal_geometry": cancelled_geometry,
            "retired_slot": target_slot,
            "logits": serialize_logit_rows(parsed.rows_by_session[target_session]),
            "target_state": serialize_state_receipt(cancelled_state),
        },
        "replacement": serialize_result(replacement_result),
        "replacement_started_unix": replacement_result.started_unix,
        "replacement_reused_retired_slot": True,
        "survivors": [serialize_result(result) for result in survivors],
        "session_slots": session_slots,
        "logits": {
            case.label: serialize_logit_rows(
                parsed.rows_by_session[baseline_sessions[case.label]]
            )
            for case in [*cases, replacement]
        },
        "terminal_states": {
            case.label: serialize_state_receipt(
                parsed.state_by_session[baseline_sessions[case.label]]
            )
            for case in [*cases, replacement]
        },
        "frames": [serialize_frame(frame) for frame in parsed.frames],
        "log_sha256": parsed.text_sha256,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument(
        "--selector-profile",
        choices=tuple(SELECTOR_PROFILES),
        required=True,
        help=(
            "native-reference uses the native route with five optimization selectors off; "
            "native-optimized enables them; serial-control disables native batching"
        ),
    )
    parser.add_argument(
        "--evidence-tier",
        choices=("token-only-smoke", "full"),
        required=True,
        help=(
            "token-only-smoke proves terminal token parity plus one engagement marker; "
            "full requires per-frame logits and terminal target-state receipts"
        ),
    )
    parser.add_argument("--server-log", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--server-pid", type=int)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--reference-receipt", type=Path)
    parser.add_argument("--reference-receipt-sha256")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--minimum-full-frames", type=int, default=8)
    parser.add_argument("--memory-floor-mib", type=int, default=6656)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if PUBLIC_MODEL_NAME_RE.fullmatch(args.model) is None:
        parser.error("--model must be a path-free public API model alias")
    if args.width not in (2, 4, 8):
        parser.error("--width must be one of 2, 4, or 8")
    if args.max_tokens < 16 or args.max_tokens > 256:
        parser.error("--max-tokens must be in 16..256")
    if args.minimum_full_frames < 8:
        parser.error("--minimum-full-frames must be at least 8")
    if args.memory_floor_mib < 1:
        parser.error("--memory-floor-mib must be positive")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be positive and finite")
    args.expected_route = (
        "serial" if args.selector_profile == "serial-control" else "native"
    )
    if (args.reference_receipt is None) != (args.reference_receipt_sha256 is None):
        parser.error("--reference-receipt and --reference-receipt-sha256 are a pair")
    if args.evidence_tier == "full" and args.selector_profile != "native-reference":
        if args.reference_receipt is None:
            parser.error(
                "full native-optimized and serial-control runs require a native-reference receipt"
            )
    elif (
        args.selector_profile == "native-reference"
        and args.reference_receipt is not None
    ):
        parser.error("native-reference must not consume another reference receipt")
    return args


def baseline_semantics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "label",
        "prompt_tokens",
        "completion_tokens",
        "usage_identity",
        "token_ids_sha256",
        "token_ids",
        "finish_reason",
        "session",
        "logits_rows",
        "logits",
        "logits_sha256_chain",
        "target_state",
    )
    normalized = []
    for row in rows:
        if type(row) is not dict or any(key not in row for key in keys):
            raise ProbeError("reference receipt baseline C1 row is incomplete")
        label = row["label"]
        prompt_tokens = row["prompt_tokens"]
        completion_tokens = row["completion_tokens"]
        token_ids = row["token_ids"]
        usage = row["usage_identity"]
        state = row["target_state"]
        if (
            type(label) is not str
            or not label
            or type(prompt_tokens) is not int
            or prompt_tokens <= 0
            or type(completion_tokens) is not int
            or completion_tokens <= 0
            or type(token_ids) is not list
            or len(token_ids) != completion_tokens
            or any(
                type(token) is not int or not 0 <= token <= 0xFFFF_FFFF
                for token in token_ids
            )
            or row["finish_reason"] not in EXACT_BUDGET_FINISH_REASONS
            or type(row["session"]) is not str
            or not re.fullmatch(r"[0-9a-f]{16}", row["session"])
            or type(row["logits_rows"]) is not int
            or row["logits_rows"] != completion_tokens - 1
        ):
            raise ProbeError("reference receipt baseline C1 row is malformed")
        if hashlib.sha256(json_canonical(token_ids)).hexdigest() != canonical_sha256(
            row["token_ids_sha256"], "baseline token_ids_sha256"
        ):
            raise ProbeError("reference receipt baseline token-ID SHA-256 differs")
        logits = row["logits"]
        if type(logits) is not list or len(logits) != completion_tokens - 1:
            raise ProbeError("reference receipt baseline logits are incomplete")
        for expected_index, logit in enumerate(logits, start=1):
            if (
                type(logit) is not dict
                or set(logit) != {"index", "fnv", "sha256"}
                or type(logit["index"]) is not int
                or logit["index"] != expected_index
                or type(logit["fnv"]) is not str
                or not re.fullmatch(r"[0-9a-f]{16}", logit["fnv"])
            ):
                raise ProbeError("reference receipt baseline logit row is malformed")
            canonical_sha256(logit["sha256"], "baseline logit row")
        if hashlib.sha256(json_canonical(logits)).hexdigest() != canonical_sha256(
            row["logits_sha256_chain"], "baseline logits chain"
        ):
            raise ProbeError("reference receipt baseline logits chain differs")
        if type(usage) is not dict or set(usage) != set(USAGE_IDENTITY_FIELDS):
            raise ProbeError("reference receipt baseline usage identity is malformed")
        validate_usage(
            usage,
            completion_tokens,
            "reference receipt baseline usage",
            expected_prompt_tokens=prompt_tokens,
        )
        state_keys = {
            "schema",
            "mode",
            "scheduler_path",
            "proposer_loaded",
            "session",
            "seq_len",
            "token_count",
            "output_tokens",
            "gdn_layers",
            "attention_layers",
            "logical_kv_blocks",
            "kv_token_positions",
            "compared_buffers",
            "compared_bytes",
            "fed_tokens_sha256",
            "target_state_sha256",
        }
        if (
            type(state) is not dict
            or set(state) != state_keys
            or type(state["schema"]) is not int
            or state["schema"] != 1
            or state["mode"] != "unloaded-target-only"
            or state["scheduler_path"] != "ordinary-target-only"
            or type(state["proposer_loaded"]) is not bool
            or state["proposer_loaded"]
            or state["session"] != row["session"]
            or type(state["seq_len"]) is not int
            or state["seq_len"] != prompt_tokens + completion_tokens - 1
            or type(state["token_count"]) is not int
            or state["token_count"] != state["seq_len"]
            or type(state["output_tokens"]) is not int
            or state["output_tokens"] != completion_tokens
            or type(state["gdn_layers"]) is not int
            or state["gdn_layers"] != 36
            or type(state["attention_layers"]) is not int
            or state["attention_layers"] != 12
            or type(state["logical_kv_blocks"]) is not int
            or state["logical_kv_blocks"] <= 0
            or type(state["kv_token_positions"]) is not int
            or state["kv_token_positions"] != state["seq_len"] * 12
            or type(state["compared_buffers"]) is not int
            or state["compared_buffers"] <= 0
            or type(state["compared_bytes"]) is not int
            or state["compared_bytes"] <= 0
        ):
            raise ProbeError("reference receipt baseline target state is malformed")
        canonical_sha256(state["fed_tokens_sha256"], "baseline fed tokens")
        canonical_sha256(state["target_state_sha256"], "baseline target state")
        canonical_sha256(row.get("log_sha256"), "baseline log")
        normalized.append({key: row[key] for key in keys})
    return normalized


def validate_reference_receipt(
    args: argparse.Namespace,
    plan: dict[str, Any],
    provenance: dict[str, Any],
    baseline_receipts: list[dict[str, Any]],
) -> dict[str, str] | None:
    if args.reference_receipt is None:
        return None
    expected = canonical_sha256(
        args.reference_receipt_sha256, "--reference-receipt-sha256"
    )
    reference_bytes, observed, _ = read_stable_file_bytes(
        args.reference_receipt, "reference receipt"
    )
    if observed != expected:
        raise ProbeError("reference receipt SHA-256 differs")
    reference = json_loads_strict(reference_bytes, "reference receipt")
    if type(reference) is not dict:
        raise ProbeError("reference receipt must be an object")
    if (
        reference.get("schema") != SCHEMA
        or reference.get("result") != "PASS"
        or reference.get("arm_qualified") is not True
        or reference.get("release_qualified") is not False
        or reference.get("evidence_tier") != "full"
        or reference.get("selector_profile") != "native-reference"
        or reference.get("expected_route") != "native"
        or reference.get("width") != args.width
        or reference.get("model") != args.model
        or reference.get("max_tokens") != args.max_tokens
        or reference.get("memory_floor_mib") != args.memory_floor_mib
        or reference.get("cases") != plan["cases"]
        or reference.get("probe_sha256") != plan["probe_sha256"]
    ):
        raise ProbeError("reference receipt qualification contract differs")
    reference_engagement = reference.get("optimization_selector_engagement_counts")
    if (
        type(reference_engagement) is not dict
        or set(reference_engagement) != set(OPTIMIZATION_SELECTORS)
        or any(
            type(count) is not int or count != 0
            for count in reference_engagement.values()
        )
    ):
        raise ProbeError("reference receipt selector engagement differs")
    reference_provenance = reference.get("provenance")
    if type(reference_provenance) is not dict:
        raise ProbeError("reference receipt provenance is missing")
    for key in (
        "release_version",
        "binary_sha256",
        "source_revision",
        "source_tree",
        "source_bundle_manifest_sha256",
        "source_bundle_sha256",
        "build_receipt_sha256",
        "source_to_binary_receipt_sha256",
        "checkpoint_id",
        "checkpoint_revision",
        "checkpoint_manifest_sha256",
    ):
        if reference_provenance.get(key) != provenance.get(key):
            raise ProbeError(f"reference receipt provenance.{key} differs")
    if baseline_semantics(reference.get("baseline_c1", [])) != baseline_semantics(
        baseline_receipts
    ):
        raise ProbeError("C1 semantic receipts differ from native-reference")
    return {"name": args.reference_receipt.name, "sha256": expected}


def main() -> int:
    args = parse_args()
    probe_path = Path(__file__).resolve()
    probe_sha256, probe_identity = sha256_stable_file(probe_path, "qualifier source")
    base_url = validate_origin(args.base_url)
    wave_cases = make_cases(args.width)
    churn_replacement = make_cases(args.width + 1)[-1] if args.width == 8 else None
    baseline_cases = wave_cases + ([churn_replacement] if churn_replacement else [])
    plan = {
        "schema": SCHEMA,
        "evidence_tier": args.evidence_tier,
        "selector_profile": args.selector_profile,
        "width": args.width,
        "expected_route": args.expected_route,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "minimum_full_frames": args.minimum_full_frames,
        "memory_floor_mib": args.memory_floor_mib,
        "probe_name": probe_path.name,
        "probe_sha256": probe_sha256,
        "expected_binary_sha256": args.binary_sha256,
        "expected_manifest_sha256": args.manifest_sha256,
        "cases": [
            {
                "label": case.label,
                "ordinal": case.ordinal,
                "prompt_sha256": case.prompt_sha256,
            }
            for case in baseline_cases
        ],
    }
    plan["plan_sha256"] = hashlib.sha256(json_canonical(plan)).hexdigest()
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if args.receipt.exists():
        raise FileExistsError(f"refusing to overwrite {args.receipt}")
    if not args.server_log.is_file():
        raise ProbeError("--server-log must be the fresh process's existing log file")
    provenance, provenance_seals = validate_provenance(args)
    healthcheck(base_url, min(args.timeout, 30.0))
    cursor = LogCursor(args.server_log)
    initial_log_offset = cursor.mark()
    preexisting_log = cursor.read_from(
        0, settle_seconds=0.0, end_offset=initial_log_offset
    )
    validate_evidence_line_syntax(preexisting_log)
    preexisting_receipts = (
        ENGAGED_RE.search(preexisting_log),
        FRAME_RE.search(preexisting_log),
        LOGITS_RE.search(preexisting_log),
        STATE_RE.search(preexisting_log),
    )
    if any(preexisting_receipts):
        raise ProbeError("server log contains inference receipts; use a fresh process")

    memory = MemorySampler(args.server_pid)
    memory.start()
    started = time.time()
    reference_receipt: dict[str, str] | None = None
    churn_receipt: dict[str, Any] | None = None
    reuse_receipt: dict[str, Any] | None = None
    try:
        baseline_results: dict[str, RequestResult] = {}
        baseline_sessions: dict[str, str] = {}
        baseline_logits: dict[str, dict[int, str]] = {}
        baseline_states: dict[str, StateReceipt] = {}
        baseline_receipts: list[dict[str, Any]] = []
        for case in baseline_cases:
            offset = cursor.mark() if args.evidence_tier == "full" else 0
            result = stream_request(
                base_url, args.model, case, args.max_tokens, args.timeout
            )
            baseline_results[case.label] = result
            if args.evidence_tier == "full":
                parsed = parse_log(cursor.read_from(offset))
                session = sole_session(parsed, case.label)
                rows = row_digest_map(parsed.rows_by_session[session])
                expected_rows = args.max_tokens - 1
                if list(rows) != list(range(1, args.max_tokens)):
                    raise ProbeError(
                        f"{case.label}: C1 logits indices differ from "
                        f"1..{args.max_tokens - 1} ({len(rows)} rows, "
                        f"expected {expected_rows})"
                    )
                if parsed.frames:
                    raise ProbeError(
                        f"{case.label}: isolated C1 unexpectedly emitted batch frames"
                    )
                if set(parsed.state_by_session) != {session}:
                    raise ProbeError(
                        f"{case.label}: isolated C1 target-state session differs"
                    )
                state = parsed.state_by_session[session]
                validate_state_receipt(
                    case.label,
                    state,
                    len(result.token_ids),
                    result.usage["prompt_tokens"],
                )
                baseline_sessions[case.label] = session
                baseline_logits[case.label] = rows
                baseline_states[case.label] = state
                baseline_receipts.append(
                    {
                        **serialize_result(result),
                        "session": session,
                        "logits_rows": len(rows),
                        "logits": serialize_logit_rows(parsed.rows_by_session[session]),
                        "logits_sha256_chain": hashlib.sha256(
                            json_canonical(
                                serialize_logit_rows(parsed.rows_by_session[session])
                            )
                        ).hexdigest(),
                        "target_state": state_signature(state),
                        "log_sha256": parsed.text_sha256,
                    }
                )
            else:
                baseline_receipts.append(serialize_result(result))

        if args.evidence_tier == "full" and len(set(baseline_sessions.values())) != len(
            baseline_sessions
        ):
            raise ProbeError("distinct C1 prompts collided on the session hash")

        if args.evidence_tier == "full":
            reference_receipt = validate_reference_receipt(
                args, plan, provenance, baseline_receipts
            )

        forward_offset = cursor.mark()
        forward_results = run_wave(
            base_url, args.model, wave_cases, args.max_tokens, args.timeout
        )
        if args.evidence_tier == "full":
            forward = validate_wave(
                "forward",
                args.expected_route,
                args.width,
                wave_cases,
                forward_results,
                parse_log(cursor.read_from(forward_offset)),
                baseline_results,
                baseline_sessions,
                baseline_logits,
                baseline_states,
                args.minimum_full_frames,
            )
        else:
            forward = validate_token_wave(
                "forward", wave_cases, forward_results, baseline_results
            )

        reverse_cases = list(reversed(wave_cases))
        reverse_offset = cursor.mark()
        reverse_results = run_wave(
            base_url, args.model, reverse_cases, args.max_tokens, args.timeout
        )
        if args.evidence_tier == "full":
            reverse = validate_wave(
                "reverse",
                args.expected_route,
                args.width,
                reverse_cases,
                reverse_results,
                parse_log(cursor.read_from(reverse_offset)),
                baseline_results,
                baseline_sessions,
                baseline_logits,
                baseline_states,
                args.minimum_full_frames,
            )
        else:
            reverse = validate_token_wave(
                "reverse", reverse_cases, reverse_results, baseline_results
            )
        if churn_replacement is not None:
            churn_offset = cursor.mark()
            churn = run_api_churn(
                base_url,
                args.model,
                wave_cases,
                churn_replacement,
                args.max_tokens,
                args.timeout,
                baseline_results,
            )
            if args.evidence_tier == "full":
                churn_receipt = validate_full_churn(
                    args.expected_route,
                    args.width,
                    wave_cases,
                    churn_replacement,
                    churn,
                    parse_log(cursor.read_from(churn_offset)),
                    baseline_results,
                    baseline_sessions,
                    baseline_logits,
                    baseline_states,
                )
            else:
                cancelled: CancelResult = churn["cancel_result"]
                churn_receipt = {
                    "cancelled": {
                        "label": cancelled.label,
                        "content_chunks_before_close": cancelled.content_chunks,
                    },
                    "replacement": serialize_result(churn["replacement_result"]),
                    "survivors": [
                        serialize_result(result) for result in churn["survivor_results"]
                    ],
                }

            # Every churn future has completed here, so this isolated request
            # proves the scheduler can drain fully and admit work again.
            reuse_offset = cursor.mark() if args.evidence_tier == "full" else 0
            reuse_result = stream_request(
                base_url,
                args.model,
                churn_replacement,
                args.max_tokens,
                args.timeout,
            )
            if args.evidence_tier == "full":
                reuse_receipt = validate_reuse_phase(
                    parse_log(cursor.read_from(reuse_offset)),
                    churn_replacement,
                    reuse_result,
                    baseline_results[churn_replacement.label],
                    baseline_sessions[churn_replacement.label],
                    baseline_logits[churn_replacement.label],
                    baseline_states[churn_replacement.label],
                )
            else:
                validate_request_identity(
                    "churn/post-drain-reuse",
                    reuse_result,
                    baseline_results[churn_replacement.label],
                )
                reuse_receipt = {"request": serialize_result(reuse_result)}

        healthcheck(base_url, min(args.timeout, 30.0))
    finally:
        memory.stop()

    qualification_log_end = cursor.mark()
    complete_log_bytes = cursor.read_bytes_from(
        0, settle_seconds=0.0, end_offset=qualification_log_end
    )
    try:
        complete_log = complete_log_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ProbeError("server log contains invalid UTF-8") from error
    validate_evidence_line_syntax(complete_log)
    lowered_log = complete_log.lower()
    safety_log_marker_counts = {
        marker: lowered_log.count(marker.lower()) for marker in FORBIDDEN_LOG_MARKERS
    }
    for marker, count in safety_log_marker_counts.items():
        if count:
            raise ProbeError(f"server log contains forbidden marker: {marker}")
    engaged_widths = [int(value) for value in ENGAGED_RE.findall(complete_log)]
    if args.expected_route == "native":
        if engaged_widths != [args.width]:
            raise ProbeError(
                f"expected one C{args.width} native engagement marker, got {engaged_widths}"
            )
    elif engaged_widths:
        raise ProbeError(
            f"serial control unexpectedly engaged native route: {engaged_widths}"
        )
    selector_engagement_counts = {
        selector: complete_log.count(marker)
        for selector, marker in OPTIMIZATION_ENGAGEMENT_MARKERS.items()
    }
    expected_selector_engagement = (
        1 if args.selector_profile == "native-optimized" else 0
    )
    for selector, count in selector_engagement_counts.items():
        if count != expected_selector_engagement:
            raise ProbeError(
                f"{selector}: expected {expected_selector_engagement} engagement "
                f"markers, got {count}"
            )
    memory_summary = memory.summary()
    if memory_summary["mem_available_floor_kib"] < args.memory_floor_mib * 1024:
        raise ProbeError(
            f"MemAvailable floor {memory_summary['mem_available_floor_kib']} KiB is below "
            f"{args.memory_floor_mib} MiB"
        )
    revalidate_provenance(args, provenance_seals)
    if stable_identity(probe_path, "qualifier source") != probe_identity:
        raise ProbeError("qualifier source changed during qualification")
    observed_probe_sha256, _ = sha256_stable_file(probe_path, "qualifier source")
    if observed_probe_sha256 != probe_sha256:
        raise ProbeError("qualifier source bytes changed during qualification")
    arm_qualified = args.evidence_tier == "full"
    receipt = {
        **plan,
        "provenance": provenance,
        "result": "PASS" if args.evidence_tier == "full" else "SMOKE_ONLY",
        "arm_qualified": arm_qualified,
        # No single process proves the selector-on/off matrix. A separate
        # matrix seal must consume all nine immutable arm receipts.
        "release_qualified": False,
        "started_unix": started,
        "finished_unix": time.time(),
        "server_pid": args.server_pid,
        "server_log_name": args.server_log.name,
        "server_log_qualified_start_offset": 0,
        "server_log_qualified_end_offset": qualification_log_end,
        "server_log_sha256": hashlib.sha256(complete_log_bytes).hexdigest(),
        "safety_log_marker_counts": safety_log_marker_counts,
        "memory": memory_summary,
        "baseline_c1": baseline_receipts,
        "waves": [forward, reverse],
        "reference_receipt": reference_receipt,
        "api_churn": churn_receipt,
        "post_drain_reuse": reuse_receipt,
        "native_engagement_widths": engaged_widths,
        "optimization_selector_engagement_counts": selector_engagement_counts,
        "limitations": (
            [
                "token-only smoke: exact terminal token IDs and one native engagement only",
                "token-only smoke does not bind each logits row or each frame to its route",
                "serial route is inferred from absence of native engagement only",
                "direct-binary endpoint evidence, not container evidence",
                "correctness receipt, not a throughput receipt",
                "cancellation/replacement/drain/reuse is exercised only by C8",
            ]
            if args.evidence_tier == "token-only-smoke"
            else [
                "direct-binary endpoint evidence, not container evidence",
                "correctness receipt, not a throughput receipt",
                "terminal state covers all live GDN/PLE state and logical BF16 KV prefixes",
                "per-step evidence covers complete logits plus logical-session/physical-slot/sequence-length routing",
                "cancellation/replacement/drain/reuse is exercised only by C8",
                "release requires a separate nine-arm selector matrix seal",
            ]
        ),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    if args.evidence_tier == "full":
        print(f"PASS C{args.width} {args.expected_route}: {args.receipt}")
    else:
        print(
            f"SMOKE-ONLY C{args.width} {args.expected_route}: {args.receipt}; "
            "NOT A FULL-LOGIT OR PER-FRAME RELEASE PASS"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ProbeError, ValueError, json.JSONDecodeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
