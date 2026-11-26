#!/usr/bin/env python3
"""Seal the fail-closed 3-profile by 3-width native-concurrency matrix.

This is an offline receipt consumer. It independently replays the complete
probe-v3 evidence carried by each arm receipt; it does not trust the probe's
PASS bit or infer evidence that is absent from the receipt. It does not contact
a server, build a binary, or inspect host-local qualification paths.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat as stat_module
import subprocess
import sys
import types
from typing import Any

SCHEMA = "flux-qwen4-native-concurrency-matrix-v2"
PROFILES = ("native-reference", "native-optimized", "serial-control")
WIDTHS = (2, 4, 8)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MATRIX_RELATIVE_PATH = Path("repro-kit/qwen4_native_concurrency_matrix.py")
PROBE_RELATIVE_PATH = Path("repro-kit/qwen4_native_concurrency_probe.py")
R9_HELPER_RELATIVE_PATH = Path("repro-kit/r9_source_to_binary.py")
SOURCE_RELATIVE_PATHS = (
    MATRIX_RELATIVE_PATH,
    PROBE_RELATIVE_PATH,
    R9_HELPER_RELATIVE_PATH,
)
GIT_OID_RE = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")


class SourcePreflightError(RuntimeError):
    pass


@dataclass(frozen=True)
class CapturedGitSource:
    path: Path
    blob: str
    sha256: str
    payload: bytes
    identity: tuple[int, int, int, int, int]


@dataclass(frozen=True)
class CapturedSourceClosure:
    commit: str
    tree: str
    sources: Mapping[Path, CapturedGitSource]


def sanitized_git_environment() -> dict[str, str]:
    """Return the complete, intentionally minimal environment for Git reads."""

    return {
        "PATH": os.defpath,
        "LC_ALL": "C",
        "LANG": "C",
        "GIT_NO_LAZY_FETCH": "1",
    }


def preflight_git_output(
    arguments: list[str],
    label: str,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> bytes:
    try:
        completed = subprocess.run(
            [
                "git",
                "--no-replace-objects",
                "-C",
                str(repository_root),
                *arguments,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            env=sanitized_git_environment(),
        )
    except OSError as error:
        raise SourcePreflightError(f"cannot execute Git for {label}") from error
    if completed.returncode != 0:
        raise SourcePreflightError(f"cannot resolve {label} from Git")
    return completed.stdout


def preflight_git_oid(
    arguments: list[str],
    label: str,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> str:
    try:
        value = (
            preflight_git_output(arguments, label, repository_root=repository_root)
            .decode("ascii")
            .strip()
        )
    except UnicodeDecodeError as error:
        raise SourcePreflightError(f"Git returned malformed {label}") from error
    if GIT_OID_RE.fullmatch(value) is None or set(value) == {"0"}:
        raise SourcePreflightError(f"Git returned malformed {label}")
    return value


def read_regular_file_bytes(
    path: Path, label: str
) -> tuple[bytes, tuple[int, int, int, int, int]]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SourcePreflightError(f"cannot open {label}") from error
    try:
        before_stat = os.fstat(descriptor)
        if not stat_module.S_ISREG(before_stat.st_mode):
            raise SourcePreflightError(f"{label} is not a regular file")
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
        raise SourcePreflightError(f"{label} changed while reading")
    payload = b"".join(chunks)
    if len(payload) != after[2]:
        raise SourcePreflightError(f"{label} byte count changed while reading")
    try:
        current_stat = path.stat(follow_symlinks=False)
    except OSError as error:
        raise SourcePreflightError(f"cannot restat {label}") from error
    if not stat_module.S_ISREG(current_stat.st_mode):
        raise SourcePreflightError(f"{label} path is not a regular file")
    current = (
        current_stat.st_dev,
        current_stat.st_ino,
        current_stat.st_size,
        current_stat.st_mtime_ns,
        current_stat.st_mode,
    )
    if current != after:
        raise SourcePreflightError(f"{label} path identity changed while reading")
    return payload, after


def capture_source_closure(
    *,
    repository_root: Path = REPOSITORY_ROOT,
    relative_paths: tuple[Path, ...] = SOURCE_RELATIVE_PATHS,
) -> CapturedSourceClosure:
    observed_root = preflight_git_output(
        ["rev-parse", "--show-toplevel"],
        "repository root",
        repository_root=repository_root,
    )
    try:
        resolved_root = Path(observed_root.decode("utf-8").strip()).resolve()
    except (UnicodeDecodeError, OSError) as error:
        raise SourcePreflightError("Git repository root is malformed") from error
    expected_root = repository_root.resolve()
    if resolved_root != expected_root:
        raise SourcePreflightError("qualifier is not in the expected Git worktree")
    commit = preflight_git_oid(
        ["rev-parse", "--verify", "HEAD^{commit}"],
        "HEAD commit",
        repository_root=repository_root,
    )
    tree = preflight_git_oid(
        ["rev-parse", "--verify", f"{commit}^{{tree}}"],
        "captured commit tree",
        repository_root=repository_root,
    )
    sources: dict[Path, CapturedGitSource] = {}
    for relative_path in relative_paths:
        relative = Path(relative_path)
        label = relative.name
        blob = preflight_git_oid(
            [
                "rev-parse",
                "--verify",
                f"{commit}:{relative.as_posix()}",
            ],
            f"reviewed {label} blob",
            repository_root=repository_root,
        )
        if (
            preflight_git_output(
                ["cat-file", "-t", blob],
                f"reviewed {label} type",
                repository_root=repository_root,
            )
            != b"blob\n"
        ):
            raise SourcePreflightError(f"reviewed {label} Git object is not a blob")
        tracked_bytes = preflight_git_output(
            ["cat-file", "blob", blob],
            f"reviewed {label} bytes",
            repository_root=repository_root,
        )
        path = expected_root / relative
        payload, identity = read_regular_file_bytes(path, f"reviewed {label} source")
        if payload != tracked_bytes:
            raise SourcePreflightError(
                f"reviewed {label} source differs from its captured Git blob"
            )
        sources[relative] = CapturedGitSource(
            path=path,
            blob=blob,
            sha256=hashlib.sha256(tracked_bytes).hexdigest(),
            payload=tracked_bytes,
            identity=identity,
        )
    final_commit = preflight_git_oid(
        ["rev-parse", "--verify", "HEAD^{commit}"],
        "final HEAD commit",
        repository_root=repository_root,
    )
    if final_commit != commit:
        raise SourcePreflightError("Git HEAD changed during source preflight")
    return CapturedSourceClosure(
        commit=commit,
        tree=tree,
        sources=types.MappingProxyType(sources),
    )


def preflight_probe_import() -> tuple[tuple[str, str, str, str], bytes]:
    """Compatibility wrapper returning the probe seal from one source capture."""

    closure = capture_source_closure()
    source = closure.sources[PROBE_RELATIVE_PATH]
    return (
        (closure.commit, closure.tree, source.blob, source.sha256),
        source.payload,
    )


def execute_private_module(
    name: str,
    source: CapturedGitSource,
    *,
    injected: dict[str, Any] | None = None,
) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__file__ = str(source.path)
    module.__package__ = ""
    if injected is not None:
        module.__dict__.update(injected)
    sys.modules[name] = module
    try:
        exec(compile(source.payload, module.__file__, "exec"), module.__dict__)
    except Exception as error:
        sys.modules.pop(name, None)
        raise SourcePreflightError(
            "cannot import the reviewed source closure"
        ) from error
    return module


try:
    PREIMPORT_SOURCE_CLOSURE = capture_source_closure()
    _r9_source = PREIMPORT_SOURCE_CLOSURE.sources[R9_HELPER_RELATIVE_PATH]
    R9_MODULE_NAME = f"_atlas_reviewed_r9_{_r9_source.blob[:16]}"
    r9_provenance = execute_private_module(R9_MODULE_NAME, _r9_source)
    _probe_source = PREIMPORT_SOURCE_CLOSURE.sources[PROBE_RELATIVE_PATH]
    PROBE_MODULE_NAME = f"_atlas_reviewed_qwen4_probe_{_probe_source.blob[:16]}"
    probe = execute_private_module(
        PROBE_MODULE_NAME,
        _probe_source,
        injected={"r9_provenance": r9_provenance},
    )
except SourcePreflightError as error:
    if __name__ == "__main__":
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1) from None
    raise

PREIMPORT_PROBE_SEAL = (
    PREIMPORT_SOURCE_CLOSURE.commit,
    PREIMPORT_SOURCE_CLOSURE.tree,
    _probe_source.blob,
    _probe_source.sha256,
)
PREIMPORT_PROBE_BYTES = _probe_source.payload


@dataclass(frozen=True)
class TrackedSourceSeal:
    path: Path
    sha256: str
    identity: tuple[int, int, int, int, int]
    commit: str
    tree: str
    blob: str


PLAN_KEYS = (
    "schema",
    "evidence_tier",
    "selector_profile",
    "width",
    "expected_route",
    "model",
    "max_tokens",
    "minimum_full_frames",
    "memory_floor_mib",
    "probe_name",
    "probe_sha256",
    "expected_binary_sha256",
    "expected_manifest_sha256",
    "cases",
)
REQUEST_KEYS = {
    "label",
    "prompt_tokens",
    "completion_tokens",
    "usage_identity",
    "token_ids_sha256",
    "token_ids",
    "finish_reason",
    "elapsed_s",
}
USAGE_KEYS = set(probe.USAGE_IDENTITY_FIELDS)
LOGIT_KEYS = {"index", "fnv", "sha256"}
FRAME_LOGIT_KEYS = {"session", *LOGIT_KEYS}
STATE_KEYS = {
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
RUNTIME_STATE_KEYS = {*STATE_KEYS, "line_number"}
FRAME_KEYS = {
    "line_number",
    "route",
    "rows",
    "sessions",
    "slots",
    "pre_step_seq_lens",
    "logits",
}
WAVE_KEYS = {
    "name",
    "route",
    "request_order",
    "full_width_frames",
    "first_full_frame_order",
    "slot_sets",
    "session_slots",
    "logits",
    "terminal_states",
    "frames",
    "state_sha256",
    "log_sha256",
    "requests",
}
PROVENANCE_KEYS = {
    "manifest_schema",
    "manifest_name",
    "manifest_sha256",
    "binary_name",
    "binary_sha256",
    "release_version",
    "source_revision",
    "source_tree",
    "source_bundle_manifest_name",
    "source_bundle_manifest_sha256",
    "source_bundle_name",
    "source_bundle_sha256",
    "build_receipt_name",
    "build_receipt_sha256",
    "source_to_binary_receipt_name",
    "source_to_binary_receipt_sha256",
    "checkpoint_id",
    "checkpoint_revision",
    "checkpoint_manifest_sha256",
    "checkpoint_files_verified",
    "checkpoint_bytes_verified",
    "selector_profile",
    "configured_max_batch_size",
    "configured_max_num_seqs",
    "server_process",
}
COMMON_PROVENANCE_KEYS = (
    "manifest_schema",
    "binary_name",
    "binary_sha256",
    "release_version",
    "source_revision",
    "source_tree",
    "source_bundle_manifest_name",
    "source_bundle_manifest_sha256",
    "source_bundle_name",
    "source_bundle_sha256",
    "build_receipt_name",
    "build_receipt_sha256",
    "source_to_binary_receipt_name",
    "source_to_binary_receipt_sha256",
    "checkpoint_id",
    "checkpoint_revision",
    "checkpoint_manifest_sha256",
    "checkpoint_files_verified",
    "checkpoint_bytes_verified",
)
ARM_KEYS = {
    *PLAN_KEYS,
    "plan_sha256",
    "provenance",
    "result",
    "arm_qualified",
    "release_qualified",
    "started_unix",
    "finished_unix",
    "server_pid",
    "server_log_name",
    "server_log_qualified_start_offset",
    "server_log_qualified_end_offset",
    "server_log_sha256",
    "safety_log_marker_counts",
    "memory",
    "baseline_c1",
    "waves",
    "reference_receipt",
    "api_churn",
    "post_drain_reuse",
    "native_engagement_widths",
    "optimization_selector_engagement_counts",
    "limitations",
}
FULL_LIMITATIONS = [
    "direct-binary endpoint evidence, not container evidence",
    "correctness receipt, not a throughput receipt",
    "terminal state covers all live GDN/PLE state and logical BF16 KV prefixes",
    "per-step evidence covers complete logits plus logical-session/physical-slot/sequence-length routing",
    "cancellation/replacement/drain/reuse is exercised only by C8",
    "release requires a separate nine-arm selector matrix seal",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        nargs=2,
        metavar=("RECEIPT", "SHA256"),
        required=True,
        help="repeat exactly nine times, once per full-tier profile/width arm",
    )
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    if len(args.arm) != len(PROFILES) * len(WIDTHS):
        parser.error("--arm must appear exactly nine times")
    return args


def fail(label: str, message: str) -> None:
    raise probe.ProbeError(f"{label}: {message}")


def exact_object(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        fail(label, "object keys differ")
    return value


def exact_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        suffix = "" if minimum is None else f" >= {minimum}"
        fail(label, f"must be an exact built-in integer{suffix}")
    return value


def exact_bool(value: Any, expected: bool, label: str) -> None:
    if type(value) is not bool or value is not expected:
        fail(label, f"must be exactly {expected}")


def finite_number(value: Any, label: str, *, minimum: float | None = None) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        fail(label, "must be a finite built-in number")
    if minimum is not None and value < minimum:
        fail(label, f"must be >= {minimum}")
    return float(value)


def nonempty_string(value: Any, label: str) -> str:
    if type(value) is not str or not value:
        fail(label, "must be a nonempty string")
    return value


def basename(value: Any, label: str) -> str:
    result = nonempty_string(value, label)
    if (
        result in (".", "..")
        or Path(result).name != result
        or "/" in result
        or "\\" in result
    ):
        fail(label, "must be a basename without host path disclosure")
    return result


def canonical_sha256(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or re.fullmatch(r"[0-9a-f]{64}", value) is None
        or value == "0" * 64
    ):
        fail(label, "must be a nonzero lowercase SHA-256")
    return value


def public_model_id(value: Any, label: str) -> str:
    return probe.public_checkpoint_id(value, label)


def hex16(value: Any, label: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{16}", value) is None:
        fail(label, "must be 16 lowercase hexadecimal digits")
    return value


def reviewed_source(
    relative_path: Path,
    label: str,
    *,
    module: types.ModuleType | None = None,
) -> TrackedSourceSeal:
    captured = PREIMPORT_SOURCE_CLOSURE.sources[relative_path]
    path = REPOSITORY_ROOT / relative_path
    if module is not None and Path(module.__file__).resolve() != path:
        fail("matrix", f"imported {label} source is not the reviewed sibling")
    try:
        payload, identity = read_regular_file_bytes(path, f"reviewed {label} source")
    except SourcePreflightError as error:
        fail("matrix", f"cannot capture the reviewed {label} source")
        raise AssertionError from error
    if payload != captured.payload:
        fail("matrix", f"reviewed {label} source differs from its captured Git blob")
    if identity != captured.identity:
        fail("matrix", f"reviewed {label} source identity changed after preflight")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != captured.sha256:
        fail("matrix", f"reviewed {label} source digest changed after preflight")
    return TrackedSourceSeal(
        path,
        digest,
        identity,
        PREIMPORT_SOURCE_CLOSURE.commit,
        PREIMPORT_SOURCE_CLOSURE.tree,
        captured.blob,
    )


def reviewed_source_closure() -> dict[Path, TrackedSourceSeal]:
    return {
        MATRIX_RELATIVE_PATH: reviewed_source(MATRIX_RELATIVE_PATH, "matrix sealer"),
        PROBE_RELATIVE_PATH: reviewed_source(
            PROBE_RELATIVE_PATH, "qualifier", module=probe
        ),
        R9_HELPER_RELATIVE_PATH: reviewed_source(
            R9_HELPER_RELATIVE_PATH,
            "R9 provenance helper",
            module=r9_provenance,
        ),
    }


def reviewed_probe_source() -> TrackedSourceSeal:
    return reviewed_source(PROBE_RELATIVE_PATH, "qualifier", module=probe)


def revalidate_reviewed_source_closure(
    seals: dict[Path, TrackedSourceSeal],
) -> None:
    if set(seals) != set(SOURCE_RELATIVE_PATHS):
        fail("matrix", "reviewed source closure is incomplete")
    current = reviewed_source_closure()
    if current != seals:
        fail("matrix", "reviewed source closure changed during sealing")


def revalidate_reviewed_probe_source(
    seal: TrackedSourceSeal,
) -> None:
    try:
        _, identity = read_regular_file_bytes(seal.path, "reviewed qualifier source")
    except SourcePreflightError as error:
        fail("matrix", "cannot revalidate the reviewed qualifier source")
        raise AssertionError from error
    if identity != seal.identity:
        fail("matrix", "reviewed qualifier source identity changed during sealing")
    current = reviewed_probe_source()
    if current != seal:
        fail("matrix", "reviewed qualifier source bytes changed during sealing")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(probe.json_canonical(value)).hexdigest()


def validate_cases(value: Any, width: int, label: str) -> list[dict[str, Any]]:
    count = width + (1 if width == 8 else 0)
    if type(value) is not list or len(value) != count:
        fail(label, f"must contain exactly {count} immutable requests")
    cases: list[dict[str, Any]] = []
    for ordinal, case in enumerate(value):
        case_label = f"{label}[{ordinal}]"
        exact_object(case, {"label", "ordinal", "prompt_sha256"}, case_label)
        expected_name = f"row-{ordinal:02d}"
        if case["label"] != expected_name:
            fail(case_label, f"label must be {expected_name}")
        if exact_int(case["ordinal"], case_label + ".ordinal", minimum=0) != ordinal:
            fail(case_label, "ordinal differs from position")
        canonical_sha256(case["prompt_sha256"], case_label + ".prompt_sha256")
        cases.append(case)
    if len({case["prompt_sha256"] for case in cases}) != count:
        fail(label, "prompt SHA-256 values are not distinct")
    return cases


def validate_usage(
    value: Any, prompt_tokens: int, completion_tokens: int, label: str
) -> dict[str, Any]:
    usage = exact_object(value, USAGE_KEYS, label)
    prompt = exact_int(usage["prompt_tokens"], label + ".prompt_tokens", minimum=1)
    completion = exact_int(
        usage["completion_tokens"], label + ".completion_tokens", minimum=1
    )
    total = exact_int(usage["total_tokens"], label + ".total_tokens", minimum=2)
    if prompt != prompt_tokens or completion != completion_tokens:
        fail(label, "top-level and terminal usage counts differ")
    if total != prompt + completion:
        fail(label, "total-token arithmetic differs")
    for field in ("prompt_tokens_details", "completion_tokens_details"):
        details = usage[field]
        if type(details) is not dict:
            fail(label + "." + field, "must be an object")
        for key, item in details.items():
            if type(key) is not str or not key:
                fail(label + "." + field, "contains a malformed key")
            exact_int(item, f"{label}.{field}.{key}", minimum=0)
    return usage


def validate_request(value: Any, label: str, max_tokens: int) -> dict[str, Any]:
    request = exact_object(value, REQUEST_KEYS, label)
    request_label = nonempty_string(request["label"], label + ".label")
    prompt_tokens = exact_int(
        request["prompt_tokens"], label + ".prompt_tokens", minimum=1
    )
    completion_tokens = exact_int(
        request["completion_tokens"], label + ".completion_tokens", minimum=1
    )
    if completion_tokens != max_tokens:
        fail(label, "completion count differs from the exact token budget")
    token_ids = request["token_ids"]
    if type(token_ids) is not list or len(token_ids) != completion_tokens:
        fail(label, "token-ID cardinality differs")
    for index, token in enumerate(token_ids):
        token = exact_int(token, f"{label}.token_ids[{index}]", minimum=0)
        if token > 0xFFFF_FFFF:
            fail(label, "token ID exceeds uint32")
    digest = canonical_sha256(request["token_ids_sha256"], label + ".token_ids_sha256")
    if canonical_hash(token_ids) != digest:
        fail(label, "token-ID SHA-256 differs")
    if request["finish_reason"] not in probe.EXACT_BUDGET_FINISH_REASONS:
        fail(label, "finish reason is not an exact-budget terminal reason")
    validate_usage(
        request["usage_identity"], prompt_tokens, completion_tokens, label + ".usage"
    )
    finite_number(request["elapsed_s"], label + ".elapsed_s", minimum=0.0)
    return {
        "label": request_label,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "usage_identity": request["usage_identity"],
        "token_ids_sha256": digest,
        "token_ids": token_ids,
        "finish_reason": request["finish_reason"],
    }


def validate_logits(
    value: Any,
    output_tokens: int,
    label: str,
    *,
    expected: list[dict[str, Any]] | None = None,
    terminal_geometry: str = "pending-output",
) -> list[dict[str, Any]]:
    if terminal_geometry not in ("pending-output", "computed-unemitted"):
        fail(label, "terminal geometry differs")
    expected_rows = output_tokens - 1 + (terminal_geometry == "computed-unemitted")
    if type(value) is not list or len(value) != expected_rows:
        fail(label, "logits list is incomplete")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(value, start=1):
        row_label = f"{label}[{index - 1}]"
        exact_object(row, LOGIT_KEYS, row_label)
        if exact_int(row["index"], row_label + ".index", minimum=1) != index:
            fail(row_label, "logit index sequence differs")
        hex16(row["fnv"], row_label + ".fnv")
        canonical_sha256(row["sha256"], row_label + ".sha256")
        rows.append(row)
    if expected is not None and rows != expected:
        fail(label, "full logits differ from the C1 baseline")
    return rows


def validate_state(
    value: Any,
    label: str,
    *,
    prompt_tokens: int,
    output_tokens: int,
    session: str,
    runtime: bool,
    cancelled_logits: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], int | None]:
    state = exact_object(value, RUNTIME_STATE_KEYS if runtime else STATE_KEYS, label)
    if exact_int(state["schema"], label + ".schema", minimum=1) != 1:
        fail(label, "target-state schema differs")
    if state["mode"] != "unloaded-target-only":
        fail(label, "target-state mode differs")
    if state["scheduler_path"] != "ordinary-target-only":
        fail(label, "target-state scheduler path differs")
    exact_bool(state["proposer_loaded"], False, label + ".proposer_loaded")
    if hex16(state["session"], label + ".session") != session:
        fail(label, "target-state session differs")
    seq_len = exact_int(state["seq_len"], label + ".seq_len", minimum=2)
    token_count = exact_int(state["token_count"], label + ".token_count", minimum=2)
    observed_output = exact_int(
        state["output_tokens"], label + ".output_tokens", minimum=1
    )
    expected_seq_len = prompt_tokens + output_tokens - 1
    if cancelled_logits is not None:
        probe.cancellation_geometry(
            label, output_tokens, [row["index"] for row in cancelled_logits]
        )
        expected_seq_len = prompt_tokens + len(cancelled_logits)
    if (
        observed_output != output_tokens
        or seq_len != expected_seq_len
        or token_count != seq_len
    ):
        fail(label, "target-state token census differs")
    if exact_int(state["gdn_layers"], label + ".gdn_layers", minimum=1) != 36:
        fail(label, "target-state GDN layer census differs")
    attention = exact_int(
        state["attention_layers"], label + ".attention_layers", minimum=1
    )
    if attention != 12:
        fail(label, "target-state attention layer census differs")
    exact_int(state["logical_kv_blocks"], label + ".logical_kv_blocks", minimum=1)
    if (
        exact_int(state["kv_token_positions"], label + ".kv_token_positions", minimum=1)
        != seq_len * attention
    ):
        fail(label, "target-state KV position census differs")
    exact_int(state["compared_buffers"], label + ".compared_buffers", minimum=1)
    exact_int(state["compared_bytes"], label + ".compared_bytes", minimum=1)
    canonical_sha256(state["fed_tokens_sha256"], label + ".fed_tokens_sha256")
    canonical_sha256(state["target_state_sha256"], label + ".target_state_sha256")
    line_number = None
    if runtime:
        line_number = exact_int(state["line_number"], label + ".line_number", minimum=1)
    base = {key: state[key] for key in STATE_KEYS}
    return base, line_number


def validate_baselines(
    value: Any,
    cases: list[dict[str, Any]],
    max_tokens: int,
    label: str,
) -> dict[str, dict[str, Any]]:
    if type(value) is not list or len(value) != len(cases):
        fail(label, "C1 controls differ from the immutable request plan")
    expected_keys = {
        *REQUEST_KEYS,
        "session",
        "logits_rows",
        "logits",
        "logits_sha256_chain",
        "target_state",
        "log_sha256",
    }
    result: dict[str, dict[str, Any]] = {}
    sessions: set[str] = set()
    for index, (row, case) in enumerate(zip(value, cases)):
        row_label = f"{label}[{index}]"
        exact_object(row, expected_keys, row_label)
        request = validate_request(
            {key: row[key] for key in REQUEST_KEYS}, row_label + ".request", max_tokens
        )
        if request["label"] != case["label"]:
            fail(row_label, "C1 request label differs from the plan")
        session = hex16(row["session"], row_label + ".session")
        if session in sessions:
            fail(label, "distinct C1 requests collided on a session")
        sessions.add(session)
        if (
            exact_int(row["logits_rows"], row_label + ".logits_rows", minimum=1)
            != max_tokens - 1
        ):
            fail(row_label, "logits_rows differs from the exact token budget")
        logits = validate_logits(row["logits"], max_tokens, row_label + ".logits")
        chain = canonical_sha256(
            row["logits_sha256_chain"], row_label + ".logits_sha256_chain"
        )
        if canonical_hash(logits) != chain:
            fail(row_label, "logits SHA-256 chain differs")
        state, _ = validate_state(
            row["target_state"],
            row_label + ".target_state",
            prompt_tokens=request["prompt_tokens"],
            output_tokens=max_tokens,
            session=session,
            runtime=False,
        )
        canonical_sha256(row["log_sha256"], row_label + ".log_sha256")
        result[case["label"]] = {
            "request": request,
            "session": session,
            "logits": logits,
            "state": state,
        }
    return result


def validate_request_matches(
    value: Any, baseline: dict[str, Any], label: str, max_tokens: int
) -> None:
    observed = validate_request(value, label, max_tokens)
    if observed != baseline["request"]:
        fail(label, "terminal request semantics differ from C1")


def validate_frames(
    value: Any,
    *,
    label: str,
    route: str,
    width: int,
    baseline: dict[str, dict[str, Any]],
    phase_logits: dict[str, list[dict[str, Any]]],
    terminal_states: dict[str, tuple[dict[str, Any], int]],
    allow_retired_slot_reuse: bool,
) -> dict[str, Any]:
    if type(value) is not list or not value:
        fail(label, "frame evidence is missing")
    labels_by_session = {
        row["session"]: request_label for request_label, row in baseline.items()
    }
    logits_by_session = {
        baseline[request_label]["session"]: {row["index"]: row for row in rows}
        for request_label, rows in phase_logits.items()
    }
    state_line_by_session = {
        baseline[request_label]["session"]: state[1]
        for request_label, state in terminal_states.items()
    }
    session_slots: dict[str, int] = {}
    live_slot_owners: dict[int, str] = {}
    prior_seq: dict[str, int] = {}
    last_frame_by_session: dict[str, int] = {}
    frames: list[dict[str, Any]] = []
    previous_line = 0
    full: list[dict[str, Any]] = []
    for frame_index, frame in enumerate(value):
        frame_label = f"{label}[{frame_index}]"
        exact_object(frame, FRAME_KEYS, frame_label)
        line = exact_int(frame["line_number"], frame_label + ".line_number", minimum=1)
        if line <= previous_line:
            fail(frame_label, "frame line numbers are not strictly increasing")
        previous_line = line
        if frame["route"] != route:
            fail(frame_label, "frame route differs")
        rows = exact_int(frame["rows"], frame_label + ".rows", minimum=2)
        if rows > width:
            fail(frame_label, "frame width exceeds the probed width")
        sessions = frame["sessions"]
        slots = frame["slots"]
        seq_lens = frame["pre_step_seq_lens"]
        frame_logits = frame["logits"]
        if not all(
            type(item) is list and len(item) == rows
            for item in (sessions, slots, seq_lens, frame_logits)
        ):
            fail(frame_label, "frame row cardinality differs")
        if len(set(sessions)) != rows:
            fail(frame_label, "frame aliases a logical session")
        for session_index, session in enumerate(sessions):
            hex16(session, f"{frame_label}.sessions[{session_index}]")
            if session not in labels_by_session:
                fail(frame_label, "frame contains an unknown session")
        normalized_slots = [
            exact_int(slot, f"{frame_label}.slots[{index}]", minimum=0)
            for index, slot in enumerate(slots)
        ]
        if len(set(normalized_slots)) != rows:
            fail(frame_label, "frame aliases a physical slot")
        for row_index, (session, slot, seq_len, logit) in enumerate(
            zip(sessions, normalized_slots, seq_lens, frame_logits)
        ):
            item_label = f"{frame_label}.row[{row_index}]"
            seq_len = exact_int(seq_len, item_label + ".pre_step_seq_len", minimum=2)
            exact_object(logit, FRAME_LOGIT_KEYS, item_label + ".logit")
            if hex16(logit["session"], item_label + ".logit.session") != session:
                fail(item_label, "frame session order differs from its logits")
            index = exact_int(logit["index"], item_label + ".logit.index", minimum=1)
            hex16(logit["fnv"], item_label + ".logit.fnv")
            canonical_sha256(logit["sha256"], item_label + ".logit.sha256")
            expected_logit = logits_by_session[session].get(index)
            if (
                expected_logit is None
                or {key: logit[key] for key in LOGIT_KEYS} != expected_logit
            ):
                fail(item_label, "frame logits do not bind the full phase evidence")
            request_label = labels_by_session[session]
            expected_seq = (
                baseline[request_label]["request"]["prompt_tokens"] + index - 1
            )
            if seq_len != expected_seq:
                fail(item_label, "frame sequence length differs from logits index")
            prior = prior_seq.get(session)
            if prior is not None and seq_len != prior + 1:
                fail(item_label, "session sequence length is not stepwise")
            prior_seq[session] = seq_len
            retired_line = state_line_by_session[session]
            if retired_line <= line:
                fail(item_label, "retired session re-entered a frame")
            prior_slot = session_slots.setdefault(session, slot)
            if prior_slot != slot:
                fail(item_label, "logical session moved physical slots")
            prior_owner = live_slot_owners.get(slot)
            if prior_owner is not None and prior_owner != session:
                owner_retired = state_line_by_session.get(prior_owner)
                if (
                    not allow_retired_slot_reuse
                    or owner_retired is None
                    or owner_retired >= line
                ):
                    fail(item_label, "physical slot changed live owner")
            live_slot_owners[slot] = session
            last_frame_by_session[session] = line
        frames.append(frame)
        if rows == width:
            full.append(frame)
    if set(last_frame_by_session) != set(labels_by_session):
        fail(label, "not every request entered a batch frame")
    for session, last_line in last_frame_by_session.items():
        if state_line_by_session[session] <= last_line:
            fail(label, "terminal state does not follow the session's final frame")
    state_lines = list(state_line_by_session.values())
    if len(state_lines) != len(set(state_lines)):
        fail(label, "terminal state line numbers are not distinct")
    return {
        "frames": frames,
        "full": full,
        "session_slots": session_slots,
        "labels_by_session": labels_by_session,
    }


def validate_wave(
    value: Any,
    *,
    name: str,
    route: str,
    width: int,
    cases: list[dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    max_tokens: int,
    minimum_full_frames: int,
    label: str,
) -> None:
    wave = exact_object(value, WAVE_KEYS, label)
    if wave["name"] != name or wave["route"] != route:
        fail(label, "wave identity or route differs")
    ascending = [case["label"] for case in cases[:width]]
    expected_order = ascending if name == "forward" else list(reversed(ascending))
    if wave["request_order"] != expected_order:
        fail(label, "request order is not the exact required order")
    requests = wave["requests"]
    if type(requests) is not list or len(requests) != width:
        fail(label, "request evidence cardinality differs")
    for index, request_label in enumerate(expected_order):
        if (
            type(requests[index]) is not dict
            or requests[index].get("label") != request_label
        ):
            fail(label, "serialized request order differs")
        validate_request_matches(
            requests[index],
            baseline[request_label],
            f"{label}.requests[{index}]",
            max_tokens,
        )
    logits_value = exact_object(wave["logits"], set(expected_order), label + ".logits")
    states_value = exact_object(
        wave["terminal_states"], set(expected_order), label + ".terminal_states"
    )
    state_sha = exact_object(
        wave["state_sha256"], set(expected_order), label + ".state_sha256"
    )
    phase_logits: dict[str, list[dict[str, Any]]] = {}
    terminal_states: dict[str, tuple[dict[str, Any], int]] = {}
    for request_label in expected_order:
        base = baseline[request_label]
        phase_logits[request_label] = validate_logits(
            logits_value[request_label],
            max_tokens,
            f"{label}.logits.{request_label}",
            expected=base["logits"],
        )
        state, line = validate_state(
            states_value[request_label],
            f"{label}.terminal_states.{request_label}",
            prompt_tokens=base["request"]["prompt_tokens"],
            output_tokens=max_tokens,
            session=base["session"],
            runtime=True,
        )
        if state != base["state"]:
            fail(label, f"{request_label} terminal state differs from C1")
        if (
            canonical_sha256(
                state_sha[request_label],
                f"{label}.state_sha256.{request_label}",
            )
            != state["target_state_sha256"]
        ):
            fail(label, f"{request_label} state SHA-256 summary differs")
        assert line is not None
        terminal_states[request_label] = (state, line)
    replay = validate_frames(
        wave["frames"],
        label=label + ".frames",
        route=route,
        width=width,
        baseline={
            request_label: baseline[request_label] for request_label in expected_order
        },
        phase_logits=phase_logits,
        terminal_states=terminal_states,
        allow_retired_slot_reuse=False,
    )
    full = replay["full"]
    observed_full = exact_int(
        wave["full_width_frames"], label + ".full_width_frames", minimum=0
    )
    if observed_full != len(full) or observed_full < minimum_full_frames:
        fail(label, "full-width frame count differs or is below the contract")
    if not full:
        fail(label, "no full-width frame exists")
    first_order = [
        replay["labels_by_session"][session] for session in full[0]["sessions"]
    ]
    if (
        wave["first_full_frame_order"] != expected_order
        or first_order != expected_order
    ):
        fail(label, "first full frame does not preserve exact request order")
    expected_slots = full[0]["slots"]
    for frame_index, frame in enumerate(full):
        observed_order = [
            replay["labels_by_session"][session] for session in frame["sessions"]
        ]
        if observed_order != expected_order:
            fail(
                f"{label}.frames.full[{frame_index}]",
                "full-width frame does not preserve exact request order",
            )
        if frame["slots"] != expected_slots:
            fail(
                f"{label}.frames.full[{frame_index}]",
                "full-width frame does not preserve the stable slot vector",
            )
    slot_sets = wave["slot_sets"]
    if type(slot_sets) is not list or len(slot_sets) != len(full):
        fail(label, "slot_sets cardinality differs")
    for frame_index, slots in enumerate(slot_sets):
        if type(slots) is not list or len(slots) != width:
            fail(label, "slot_sets row cardinality differs")
        for slot_index, slot in enumerate(slots):
            exact_int(
                slot,
                f"{label}.slot_sets[{frame_index}][{slot_index}]",
                minimum=0,
            )
    if slot_sets != [frame["slots"] for frame in full]:
        fail(label, "slot_sets does not replay from full-width frames")
    if any(slots != expected_slots for slots in slot_sets):
        fail(label, "slot_sets does not preserve the stable full-width slot vector")
    session_slots = exact_object(
        wave["session_slots"], set(replay["session_slots"]), label + ".session_slots"
    )
    for session, slot in session_slots.items():
        exact_int(slot, f"{label}.session_slots.{session}", minimum=0)
    if session_slots != replay["session_slots"]:
        fail(label, "session_slots does not replay from frames")
    canonical_sha256(wave["log_sha256"], label + ".log_sha256")


def validate_post_drain(
    value: Any, baseline: dict[str, Any], *, max_tokens: int, label: str
) -> None:
    reuse = exact_object(
        value,
        {
            "request",
            "session",
            "logits",
            "logits_sha256_chain",
            "target_state",
            "frames",
            "log_sha256",
        },
        label,
    )
    validate_request_matches(reuse["request"], baseline, label + ".request", max_tokens)
    if hex16(reuse["session"], label + ".session") != baseline["session"]:
        fail(label, "post-drain session differs from C1")
    logits = validate_logits(
        reuse["logits"],
        max_tokens,
        label + ".logits",
        expected=baseline["logits"],
    )
    chain = canonical_sha256(
        reuse["logits_sha256_chain"], label + ".logits_sha256_chain"
    )
    if canonical_hash(logits) != chain:
        fail(label, "post-drain logits chain differs")
    state, _ = validate_state(
        reuse["target_state"],
        label + ".target_state",
        prompt_tokens=baseline["request"]["prompt_tokens"],
        output_tokens=max_tokens,
        session=baseline["session"],
        runtime=True,
    )
    if state != baseline["state"]:
        fail(label, "post-drain terminal state differs from C1")
    if reuse["frames"] != []:
        fail(label, "post-drain isolated request contains batch frames")
    canonical_sha256(reuse["log_sha256"], label + ".log_sha256")


def validate_churn(
    value: Any,
    post_drain: Any,
    *,
    route: str,
    cases: list[dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    max_tokens: int,
    label: str,
) -> None:
    churn = exact_object(
        value,
        {
            "route",
            "cancelled",
            "replacement",
            "replacement_started_unix",
            "replacement_reused_retired_slot",
            "survivors",
            "session_slots",
            "logits",
            "terminal_states",
            "frames",
            "log_sha256",
        },
        label,
    )
    if churn["route"] != route:
        fail(label, "churn route differs")
    originals = [case["label"] for case in cases[:8]]
    target_label = originals[-1]
    replacement_label = cases[8]["label"]
    all_labels = [*originals, replacement_label]
    cancelled = exact_object(
        churn["cancelled"],
        {
            "label",
            "content_chunks_before_close",
            "closed_unix",
            "session",
            "state_output_tokens",
            "terminal_geometry",
            "retired_slot",
            "logits",
            "target_state",
        },
        label + ".cancelled",
    )
    if cancelled["label"] != target_label:
        fail(label, "the deterministic final admission was not cancelled")
    if (
        exact_int(
            cancelled["content_chunks_before_close"],
            label + ".cancelled.content_chunks_before_close",
            minimum=1,
        )
        != 8
    ):
        fail(label, "cancellation did not occur at the exact eight-chunk boundary")
    closed = finite_number(cancelled["closed_unix"], label + ".cancelled.closed_unix")
    target = baseline[target_label]
    if hex16(cancelled["session"], label + ".cancelled.session") != target["session"]:
        fail(label, "cancelled session differs from C1")
    output_tokens = exact_int(
        cancelled["state_output_tokens"],
        label + ".cancelled.state_output_tokens",
        minimum=1,
    )
    if output_tokens >= max_tokens:
        fail(label, "cancelled request reached its terminal token budget")
    cancelled_logits = validate_logits(
        cancelled["logits"],
        output_tokens,
        label + ".cancelled.logits",
        terminal_geometry=cancelled["terminal_geometry"],
    )
    if cancelled["terminal_geometry"] != probe.cancellation_geometry(
        label + ".cancelled", output_tokens, [row["index"] for row in cancelled_logits]
    ):
        fail(label, "cancelled terminal geometry does not replay")
    if cancelled_logits != target["logits"][: len(cancelled_logits)]:
        fail(label, "cancelled logits are not an exact C1 prefix")
    cancelled_state, cancelled_line = validate_state(
        cancelled["target_state"],
        label + ".cancelled.target_state",
        prompt_tokens=target["request"]["prompt_tokens"],
        output_tokens=output_tokens,
        session=target["session"],
        runtime=True,
        cancelled_logits=cancelled_logits,
    )
    assert cancelled_line is not None
    retired_slot = exact_int(
        cancelled["retired_slot"], label + ".cancelled.retired_slot", minimum=0
    )
    replacement_started = finite_number(
        churn["replacement_started_unix"], label + ".replacement_started_unix"
    )
    if replacement_started < closed:
        fail(label, "replacement started before cancellation closed")
    exact_bool(
        churn["replacement_reused_retired_slot"],
        True,
        label + ".replacement_reused_retired_slot",
    )
    validate_request_matches(
        churn["replacement"],
        baseline[replacement_label],
        label + ".replacement",
        max_tokens,
    )
    survivors = churn["survivors"]
    survivor_labels = originals[:-1]
    if type(survivors) is not list or len(survivors) != len(survivor_labels):
        fail(label, "survivor request evidence differs")
    for index, request_label in enumerate(survivor_labels):
        if (
            type(survivors[index]) is not dict
            or survivors[index].get("label") != request_label
        ):
            fail(label, "survivor order differs from the original admission order")
        validate_request_matches(
            survivors[index],
            baseline[request_label],
            f"{label}.survivors[{index}]",
            max_tokens,
        )
    logits_value = exact_object(churn["logits"], set(all_labels), label + ".logits")
    states_value = exact_object(
        churn["terminal_states"], set(all_labels), label + ".terminal_states"
    )
    phase_logits: dict[str, list[dict[str, Any]]] = {}
    terminal_states: dict[str, tuple[dict[str, Any], int]] = {}
    for request_label in all_labels:
        base = baseline[request_label]
        if request_label == target_label:
            phase_logits[request_label] = validate_logits(
                logits_value[request_label],
                output_tokens,
                f"{label}.logits.{request_label}",
                terminal_geometry=cancelled["terminal_geometry"],
            )
            if phase_logits[request_label] != cancelled_logits:
                fail(label, "cancelled logits summary differs")
            state, line = validate_state(
                states_value[request_label],
                f"{label}.terminal_states.{request_label}",
                prompt_tokens=base["request"]["prompt_tokens"],
                output_tokens=output_tokens,
                session=base["session"],
                runtime=True,
                cancelled_logits=cancelled_logits,
            )
            if states_value[request_label] != cancelled["target_state"]:
                fail(label, "cancelled target-state summaries differ")
            if state != cancelled_state or line != cancelled_line:
                fail(label, "cancelled target state does not replay")
        else:
            phase_logits[request_label] = validate_logits(
                logits_value[request_label],
                max_tokens,
                f"{label}.logits.{request_label}",
                expected=base["logits"],
            )
            state, line = validate_state(
                states_value[request_label],
                f"{label}.terminal_states.{request_label}",
                prompt_tokens=base["request"]["prompt_tokens"],
                output_tokens=max_tokens,
                session=base["session"],
                runtime=True,
            )
            if state != base["state"]:
                fail(label, f"{request_label} terminal state differs from C1")
        assert line is not None
        terminal_states[request_label] = (state, line)
    replay = validate_frames(
        churn["frames"],
        label=label + ".frames",
        route=route,
        width=8,
        baseline={
            request_label: baseline[request_label] for request_label in all_labels
        },
        phase_logits=phase_logits,
        terminal_states=terminal_states,
        allow_retired_slot_reuse=True,
    )
    session_slots = exact_object(
        churn["session_slots"], set(replay["session_slots"]), label + ".session_slots"
    )
    for session, slot in session_slots.items():
        exact_int(slot, f"{label}.session_slots.{session}", minimum=0)
    if session_slots != replay["session_slots"]:
        fail(label, "churn session_slots does not replay from frames")
    target_session = target["session"]
    replacement_session = baseline[replacement_label]["session"]
    target_frames = [
        frame for frame in replay["frames"] if target_session in frame["sessions"]
    ]
    replacement_frames = [
        frame for frame in replay["frames"] if replacement_session in frame["sessions"]
    ]
    if not target_frames or not replacement_frames:
        fail(label, "target or replacement never entered a frame")
    original_sessions = [baseline[item]["session"] for item in originals]
    replacement_sessions = [
        *[baseline[item]["session"] for item in survivor_labels],
        replacement_session,
    ]
    full_frames = [frame for frame in replay["frames"] if frame["rows"] == 8]
    original_full_frames = [
        frame for frame in full_frames if target_session in frame["sessions"]
    ]
    replacement_full_frames = [
        frame for frame in full_frames if replacement_session in frame["sessions"]
    ]
    if not original_full_frames or not replacement_full_frames:
        fail(label, "original or replacement phase lacks a full C8 frame")
    stable_slots = original_full_frames[0]["slots"]
    for frame_index, frame in enumerate(full_frames):
        if target_session in frame["sessions"]:
            expected_sessions = original_sessions
        elif replacement_session in frame["sessions"]:
            expected_sessions = replacement_sessions
        else:
            fail(label, "full C8 frame belongs to neither churn phase")
        if frame["sessions"] != expected_sessions:
            fail(
                f"{label}.frames.full[{frame_index}]",
                "full C8 frame does not preserve exact phase order",
            )
        if frame["slots"] != stable_slots:
            fail(
                f"{label}.frames.full[{frame_index}]",
                "full C8 frame does not preserve the stable slot vector",
            )
    last_target = target_frames[-1]
    first_replacement = replacement_frames[0]
    final_position = last_target["sessions"].index(target_session)
    if (
        last_target["logits"][final_position]["index"] != len(cancelled_logits)
        or cancelled_state["seq_len"]
        != last_target["pre_step_seq_lens"][final_position] + 1
    ):
        fail(label, "cancelled final frame does not bind target-fed extent")
    target_slot = last_target["slots"][last_target["sessions"].index(target_session)]
    replacement_slot = first_replacement["slots"][
        first_replacement["sessions"].index(replacement_session)
    ]
    if not (
        last_target["line_number"] < cancelled_line < first_replacement["line_number"]
    ):
        fail(label, "replacement entered before cancelled-state retirement")
    if target_slot != retired_slot or replacement_slot != retired_slot:
        fail(label, "replacement did not reuse the exact retired slot")
    if (
        original_full_frames[-1]["sessions"][-1] != target_session
        or original_full_frames[-1]["slots"][-1] != retired_slot
        or retired_slot != max(original_full_frames[-1]["slots"])
    ):
        fail(label, "cancelled final admission was not the highest live slot")
    canonical_sha256(churn["log_sha256"], label + ".log_sha256")
    validate_post_drain(
        post_drain,
        baseline[replacement_label],
        max_tokens=max_tokens,
        label=label + ".post_drain",
    )


def validate_memory(value: Any, floor_mib: int, label: str) -> None:
    memory = exact_object(
        value,
        {
            "samples",
            "mem_available_floor_kib",
            "server_vm_rss_peak_kib",
            "server_vm_hwm_peak_kib",
        },
        label,
    )
    exact_int(memory["samples"], label + ".samples", minimum=2)
    floor = exact_int(
        memory["mem_available_floor_kib"],
        label + ".mem_available_floor_kib",
        minimum=1,
    )
    rss = exact_int(
        memory["server_vm_rss_peak_kib"],
        label + ".server_vm_rss_peak_kib",
        minimum=1,
    )
    hwm = exact_int(
        memory["server_vm_hwm_peak_kib"],
        label + ".server_vm_hwm_peak_kib",
        minimum=1,
    )
    if floor < floor_mib * 1024:
        fail(label, "MemAvailable floor is below the arm contract")
    if hwm < rss:
        fail(label, "VmHWM peak is below VmRSS peak")


def validate_provenance(
    value: Any, profile: str, width: int, server_pid: int, label: str
) -> tuple[tuple[str, int, int], tuple[int, int]]:
    provenance = exact_object(value, PROVENANCE_KEYS, label)
    if provenance["manifest_schema"] != probe.LAUNCH_SCHEMA:
        fail(label, "launch manifest schema differs")
    for field in (
        "manifest_name",
        "binary_name",
        "source_bundle_manifest_name",
        "source_bundle_name",
        "build_receipt_name",
        "source_to_binary_receipt_name",
    ):
        basename(provenance[field], f"{label}.{field}")
    for field in (
        "manifest_sha256",
        "binary_sha256",
        "source_bundle_manifest_sha256",
        "source_bundle_sha256",
        "build_receipt_sha256",
        "source_to_binary_receipt_sha256",
        "checkpoint_manifest_sha256",
    ):
        canonical_sha256(provenance[field], f"{label}.{field}")
    try:
        probe.r9_provenance.release_identity(provenance["release_version"])
    except ValueError:
        fail(label, "release version is invalid")
    for field in ("source_revision", "source_tree"):
        item = provenance[field]
        if (
            type(item) is not str
            or re.fullmatch(r"[0-9a-f]{40}", item) is None
            or item == "0" * 40
        ):
            fail(label, f"{field} is not a nonzero Git object")
    public_model_id(provenance["checkpoint_id"], label + ".checkpoint_id")
    revision = provenance["checkpoint_revision"]
    if (
        type(revision) is not str
        or re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", revision) is None
        or set(revision) == {"0"}
    ):
        fail(label, "checkpoint revision is not immutable")
    exact_int(
        provenance["checkpoint_files_verified"],
        label + ".checkpoint_files_verified",
        minimum=1,
    )
    exact_int(
        provenance["checkpoint_bytes_verified"],
        label + ".checkpoint_bytes_verified",
        minimum=1,
    )
    if provenance["selector_profile"] != profile:
        fail(label, "selector profile differs")
    for field in ("configured_max_batch_size", "configured_max_num_seqs"):
        capacity = exact_int(provenance[field], f"{label}.{field}", minimum=width)
        if capacity > 128:
            fail(label, f"{field} exceeds the qualified ceiling")
    process = exact_object(
        provenance["server_process"],
        {"pid", "boot_id", "start_ticks", "log_device", "log_inode"},
        label + ".server_process",
    )
    pid = exact_int(process["pid"], label + ".server_process.pid", minimum=1)
    if pid != server_pid:
        fail(label, "server PID differs from the live-process binding")
    boot_id = process["boot_id"]
    if (
        type(boot_id) is not str
        or re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            boot_id,
        )
        is None
        or set(boot_id) <= {"0", "-"}
    ):
        fail(label, "boot ID is malformed")
    start_ticks = exact_int(
        process["start_ticks"], label + ".server_process.start_ticks", minimum=1
    )
    log_device = exact_int(
        process["log_device"], label + ".server_process.log_device", minimum=0
    )
    log_inode = exact_int(
        process["log_inode"], label + ".server_process.log_inode", minimum=1
    )
    return (boot_id, pid, start_ticks), (log_device, log_inode)


def validate_arm(
    receipt: dict[str, Any],
    label: str,
    *,
    expected_probe_sha256: str | None = None,
) -> tuple[str, int]:
    exact_object(receipt, ARM_KEYS, label)
    profile = receipt["selector_profile"]
    width = receipt["width"]
    if type(profile) is not str or profile not in PROFILES:
        fail(label, "unexpected selector profile")
    width = exact_int(width, label + ".width", minimum=1)
    if width not in WIDTHS:
        fail(label, "unexpected concurrency width")
    route = "serial" if profile == "serial-control" else "native"
    if (
        receipt["schema"] != probe.SCHEMA
        or receipt["evidence_tier"] != "full"
        or receipt["result"] != "PASS"
        or receipt["expected_route"] != route
    ):
        fail(label, "not a full passing probe-v3 arm")
    exact_bool(receipt["arm_qualified"], True, label + ".arm_qualified")
    exact_bool(receipt["release_qualified"], False, label + ".release_qualified")
    probe.public_model_name(receipt["model"], label + ".model")
    max_tokens = exact_int(receipt["max_tokens"], label + ".max_tokens", minimum=16)
    if max_tokens > 256:
        fail(label, "max_tokens exceeds the qualifier contract")
    minimum_full_frames = exact_int(
        receipt["minimum_full_frames"], label + ".minimum_full_frames", minimum=8
    )
    memory_floor_mib = exact_int(
        receipt["memory_floor_mib"], label + ".memory_floor_mib", minimum=1
    )
    if receipt["probe_name"] != "qwen4_native_concurrency_probe.py":
        fail(label, "probe source basename differs")
    arm_probe_sha256 = canonical_sha256(
        receipt["probe_sha256"], label + ".probe_sha256"
    )
    probe_source_seal: TrackedSourceSeal | None = None
    if expected_probe_sha256 is None:
        probe_source_seal = reviewed_probe_source()
        expected_probe_sha256 = probe_source_seal.sha256
    if arm_probe_sha256 != expected_probe_sha256:
        fail(label, "probe SHA-256 differs from the reviewed qualifier source")
    canonical_sha256(
        receipt["expected_binary_sha256"], label + ".expected_binary_sha256"
    )
    canonical_sha256(
        receipt["expected_manifest_sha256"], label + ".expected_manifest_sha256"
    )
    cases = validate_cases(receipt["cases"], width, label + ".cases")
    plan = {key: receipt[key] for key in PLAN_KEYS}
    if canonical_sha256(
        receipt["plan_sha256"], label + ".plan_sha256"
    ) != canonical_hash(plan):
        fail(label, "plan SHA-256 differs")
    started = finite_number(receipt["started_unix"], label + ".started_unix")
    finished = finite_number(receipt["finished_unix"], label + ".finished_unix")
    if finished < started:
        fail(label, "qualification finished before it started")
    server_pid = exact_int(receipt["server_pid"], label + ".server_pid", minimum=1)
    validate_provenance(
        receipt["provenance"], profile, width, server_pid, label + ".provenance"
    )
    provenance = receipt["provenance"]
    if (
        receipt["expected_binary_sha256"] != provenance["binary_sha256"]
        or receipt["expected_manifest_sha256"] != provenance["manifest_sha256"]
    ):
        fail(label, "expected provenance digests differ")
    basename(receipt["server_log_name"], label + ".server_log_name")
    if (
        exact_int(
            receipt["server_log_qualified_start_offset"],
            label + ".server_log_qualified_start_offset",
            minimum=0,
        )
        != 0
    ):
        fail(label, "qualified log must begin at byte zero")
    exact_int(
        receipt["server_log_qualified_end_offset"],
        label + ".server_log_qualified_end_offset",
        minimum=1,
    )
    canonical_sha256(receipt["server_log_sha256"], label + ".server_log_sha256")
    safety = exact_object(
        receipt["safety_log_marker_counts"],
        set(probe.FORBIDDEN_LOG_MARKERS),
        label + ".safety_log_marker_counts",
    )
    for marker, count in safety.items():
        if (
            exact_int(count, f"{label}.safety_log_marker_counts.{marker}", minimum=0)
            != 0
        ):
            fail(label, "server safety marker count is nonzero")
    validate_memory(receipt["memory"], memory_floor_mib, label + ".memory")
    baseline = validate_baselines(
        receipt["baseline_c1"], cases, max_tokens, label + ".baseline_c1"
    )
    waves = receipt["waves"]
    if type(waves) is not list or len(waves) != 2:
        fail(label, "forward and reverse wave evidence is missing")
    for index, name in enumerate(("forward", "reverse")):
        validate_wave(
            waves[index],
            name=name,
            route=route,
            width=width,
            cases=cases,
            baseline=baseline,
            max_tokens=max_tokens,
            minimum_full_frames=minimum_full_frames,
            label=f"{label}.waves[{index}]",
        )
    expected_engagement = [] if profile == "serial-control" else [width]
    engagement = receipt["native_engagement_widths"]
    if type(engagement) is not list:
        fail(label, "native engagement width receipt must be an array")
    for index, engaged_width in enumerate(engagement):
        exact_int(
            engaged_width,
            f"{label}.native_engagement_widths[{index}]",
            minimum=1,
        )
    if engagement != expected_engagement:
        fail(label, "native engagement width receipt differs")
    selector_counts = exact_object(
        receipt["optimization_selector_engagement_counts"],
        set(probe.OPTIMIZATION_SELECTORS),
        label + ".optimization_selector_engagement_counts",
    )
    expected_count = 1 if profile == "native-optimized" else 0
    for selector, count in selector_counts.items():
        if exact_int(count, f"{label}.{selector}", minimum=0) != expected_count:
            fail(label, "optimization selector engagement count differs")
    if width == 8:
        validate_churn(
            receipt["api_churn"],
            receipt["post_drain_reuse"],
            route=route,
            cases=cases,
            baseline=baseline,
            max_tokens=max_tokens,
            label=label + ".api_churn",
        )
    elif receipt["api_churn"] is not None or receipt["post_drain_reuse"] is not None:
        fail(label, "non-C8 arm unexpectedly claims churn/reuse evidence")
    reference = receipt["reference_receipt"]
    if profile == "native-reference":
        if reference is not None:
            fail(label, "native-reference arm must not consume a reference")
    else:
        exact_object(reference, {"name", "sha256"}, label + ".reference_receipt")
        basename(reference["name"], label + ".reference_receipt.name")
        canonical_sha256(reference["sha256"], label + ".reference_receipt.sha256")
    if receipt["limitations"] != FULL_LIMITATIONS:
        fail(label, "full-tier limitations differ")
    if probe_source_seal is not None:
        revalidate_reviewed_probe_source(probe_source_seal)
    return profile, width


def load_arms(
    specs: list[list[str]],
) -> tuple[
    dict[tuple[str, int], dict[str, Any]],
    dict[tuple[str, int], dict[str, str]],
    dict[Path, tuple[tuple[int, int, int, int, int], str]],
]:
    arms: dict[tuple[str, int], dict[str, Any]] = {}
    public: dict[tuple[str, int], dict[str, str]] = {}
    identities: dict[Path, tuple[tuple[int, int, int, int, int], str]] = {}
    probe_source_seal = reviewed_probe_source()
    for raw_path, raw_digest in specs:
        path = Path(raw_path)
        expected = canonical_sha256(raw_digest, f"{path.name} SHA-256")
        payload, observed, identity = probe.read_stable_file_bytes(
            path, f"arm {path.name}"
        )
        if observed != expected:
            fail(path.name, "arm receipt SHA-256 differs")
        value = probe.json_loads_strict(payload, f"arm {path.name}")
        if type(value) is not dict:
            fail(path.name, "arm receipt must be an object")
        key = validate_arm(
            value,
            path.name,
            expected_probe_sha256=probe_source_seal.sha256,
        )
        if key in arms:
            fail(path.name, f"duplicate arm {key[0]} C{key[1]}")
        arms[key] = value
        public[key] = {"name": path.name, "sha256": expected}
        identities[path] = (identity, expected)
    expected_keys = {(profile, width) for profile in PROFILES for width in WIDTHS}
    if set(arms) != expected_keys:
        fail("matrix", "arm set differs from the required 3x3 matrix")
    revalidate_reviewed_probe_source(probe_source_seal)
    return arms, public, identities


def baseline_semantics(receipt: dict[str, Any], label: str) -> list[dict[str, Any]]:
    cases = validate_cases(receipt["cases"], receipt["width"], label + ".cases")
    rows = validate_baselines(
        receipt["baseline_c1"],
        cases,
        receipt["max_tokens"],
        label + ".baseline_c1",
    )
    return [rows[case["label"]] for case in cases]


def validate_public(
    public: Any, expected_keys: set[tuple[str, int]]
) -> dict[tuple[str, int], dict[str, str]]:
    if type(public) is not dict or set(public) != expected_keys:
        fail("matrix.public", "public arm set differs")
    digests: set[str] = set()
    for key, row in public.items():
        exact_object(row, {"name", "sha256"}, f"matrix.public.{key}")
        basename(row["name"], f"matrix.public.{key}.name")
        digest = canonical_sha256(row["sha256"], f"matrix.public.{key}.sha256")
        if digest in digests:
            fail("matrix.public", "arm receipt SHA-256 values are not distinct")
        digests.add(digest)
    return public


def validate_matrix(
    arms: dict[tuple[str, int], dict[str, Any]],
    public: dict[tuple[str, int], dict[str, str]],
) -> dict[str, Any]:
    probe_source_seal = reviewed_probe_source()
    expected_keys = {(profile, width) for profile in PROFILES for width in WIDTHS}
    if type(arms) is not dict or set(arms) != expected_keys:
        fail("matrix", "arm set differs from the required 3x3 matrix")
    validate_public(public, expected_keys)
    for key, arm in arms.items():
        if (
            type(arm) is not dict
            or validate_arm(
                arm,
                f"{key[0]} C{key[1]}",
                expected_probe_sha256=probe_source_seal.sha256,
            )
            != key
        ):
            fail("matrix", f"arm identity differs for {key}")

    first = arms[("native-reference", 2)]
    common_provenance = {
        key: first["provenance"][key] for key in COMMON_PROVENANCE_KEYS
    }
    common_contract = {
        "model": first["model"],
        "max_tokens": first["max_tokens"],
        "minimum_full_frames": first["minimum_full_frames"],
        "memory_floor_mib": first["memory_floor_mib"],
        "probe_name": first["probe_name"],
        "probe_sha256": first["probe_sha256"],
    }
    process_identities: set[tuple[str, int, int]] = set()
    log_identities: set[tuple[int, int]] = set()
    for key, arm in arms.items():
        label = f"{key[0]} C{key[1]}"
        observed_provenance = {
            field: arm["provenance"][field] for field in COMMON_PROVENANCE_KEYS
        }
        if observed_provenance != common_provenance:
            fail(label, "immutable R15 provenance differs")
        observed_contract = {field: arm[field] for field in common_contract}
        if observed_contract != common_contract:
            fail(label, "model/probe/qualification contract differs")
        process = arm["provenance"]["server_process"]
        process_identity = (
            process["boot_id"],
            process["pid"],
            process["start_ticks"],
        )
        log_identity = (process["log_device"], process["log_inode"])
        if process_identity in process_identities:
            fail(label, "server process identity is not fresh")
        if log_identity in log_identities:
            fail(label, "server log device/inode identity is not fresh")
        process_identities.add(process_identity)
        log_identities.add(log_identity)

    for width in WIDTHS:
        reference = arms[("native-reference", width)]
        reference_public = public[("native-reference", width)]
        reference_semantics = baseline_semantics(
            reference, f"native-reference C{width}"
        )
        for profile in ("native-optimized", "serial-control"):
            candidate = arms[(profile, width)]
            if candidate["reference_receipt"] != reference_public:
                fail(f"{profile} C{width}", "native-reference link differs")
            if candidate["cases"] != reference["cases"]:
                fail(f"{profile} C{width}", "immutable request plan differs")
            if (
                baseline_semantics(candidate, f"{profile} C{width}")
                != reference_semantics
            ):
                fail(f"{profile} C{width}", "C1 semantic evidence differs")

    arm_rows = []
    for profile in PROFILES:
        for width in WIDTHS:
            arm = arms[(profile, width)]
            arm_rows.append(
                {
                    "selector_profile": profile,
                    "width": width,
                    **public[(profile, width)],
                    "plan_sha256": arm["plan_sha256"],
                    "manifest_sha256": arm["provenance"]["manifest_sha256"],
                    "server_log_sha256": arm["server_log_sha256"],
                    "server_process": arm["provenance"]["server_process"],
                    "memory": arm["memory"],
                }
            )
    result = {
        "schema": SCHEMA,
        "result": "PASS",
        "release_qualified": True,
        "claim_scope": "direct-binary target-only greedy native concurrency C2/C4/C8",
        "profiles": list(PROFILES),
        "widths": list(WIDTHS),
        **common_contract,
        "provenance": common_provenance,
        "arms": arm_rows,
        "limitations": [
            "direct-binary endpoint evidence, not container evidence",
            "correctness receipt, not a throughput receipt",
            "native model batching is qualified only at C2, C4, and C8",
            "logical scheduling above C8 and native MTP concurrency are out of scope",
        ],
    }
    result["qualifier_source_commit"] = probe_source_seal.commit
    result["qualifier_source_tree"] = probe_source_seal.tree
    result["probe_git_blob"] = probe_source_seal.blob
    revalidate_reviewed_probe_source(probe_source_seal)
    return result


def main() -> int:
    args = parse_args()
    source_seals = reviewed_source_closure()
    sealer_source_seal = source_seals[MATRIX_RELATIVE_PATH]
    probe_source_seal = source_seals[PROBE_RELATIVE_PATH]
    r9_source_seal = source_seals[R9_HELPER_RELATIVE_PATH]
    if args.receipt.exists():
        raise FileExistsError(f"refusing to overwrite {args.receipt.name}")
    arms, public, identities = load_arms(args.arm)
    result = validate_matrix(arms, public)
    for path, (identity, expected_sha256) in identities.items():
        if probe.stable_identity(path, f"arm {path.name}") != identity:
            fail(path.name, "arm receipt changed during sealing")
        _, observed_sha256, _ = probe.read_stable_file_bytes(path, f"arm {path.name}")
        if observed_sha256 != expected_sha256:
            fail(path.name, "arm receipt bytes changed during sealing")
    revalidate_reviewed_source_closure(source_seals)
    if result["probe_sha256"] != probe_source_seal.sha256:
        fail("matrix", "sealed probe SHA-256 differs from reviewed source")
    result["sealer_name"] = sealer_source_seal.path.name
    result["sealer_sha256"] = sealer_source_seal.sha256
    result["sealer_git_blob"] = sealer_source_seal.blob
    result["r9_helper_name"] = r9_source_seal.path.name
    result["r9_helper_sha256"] = r9_source_seal.sha256
    result["r9_helper_git_blob"] = r9_source_seal.blob
    result["matrix_sha256"] = canonical_hash(result)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(f"PASS direct-binary C2/C4/C8 selector matrix: {args.receipt.name}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, probe.ProbeError, ValueError, json.JSONDecodeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
