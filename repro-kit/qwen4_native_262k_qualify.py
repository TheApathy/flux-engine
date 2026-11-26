#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Fresh-process native-262K qualification for Qwen3.8 Flash-Next.

This is a correctness receipt, not a performance benchmark.  It uses the
QAD5500 Hybrid manual no-thinking chat frame and proves the native boundary
without silently extending the checkpoint context:

* semantic three-passkey retrieval at 258,000, 261,888, and 262,080 tokens;
* deterministic repetition at 262,080 and a forced 64-token decode that
  reaches 262,144 total tokens;
* a 262,143-token semantic extension whose one boundary token must match a
  control from the 262,080-token prompt (262,144 total);
* a 262,144-token prompt plus one requested token is rejected, not truncated.

The runner launches the server itself, uses that server's ``/tokenize``
endpoint with a pinned tokenizer to construct exact token-ID prompts, and
requires a direct C1/BF16/QSA server argv.  Exact answer formatting is recorded
but is not a correctness gate.  Every observed HTTP response is retained
losslessly in the PASS receipt or a sealed failure artifact.
"""

from __future__ import annotations

import argparse
import base64
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
import r9_source_to_binary as r9_provenance  # noqa: E402
import qwen4_endpoint_admission as endpoint_gate  # noqa: E402

SCHEMA = "flux-qwen4-native-262k-binary-qualification-v4"
FAILURE_SCHEMA = "flux-qwen4-native-262k-binary-qualification-failure-v3"
MANIFEST_SCHEMA = "flux-qwen4-native-262k-binary-launch-v2"
SOURCE_TO_BINARY_SCHEMA = "flux-source-to-binary-receipt/1"
CHECKPOINT_INVENTORY_SCHEMA = "flux-checkpoint-inventory/1"
NATIVE_CONTEXT = 262_144
RETRIEVAL_PROMPT_TOKENS = NATIVE_CONTEXT - 64
CONTROL_PROMPT_TOKENS = 258_000
BRIDGE_PROMPT_TOKENS = 261_888
RETRIEVAL_TARGETS = (
    CONTROL_PROMPT_TOKENS,
    BRIDGE_PROMPT_TOKENS,
    RETRIEVAL_PROMPT_TOKENS,
)
BOUNDARY_PROMPT_TOKENS = NATIVE_CONTEXT - 1
OVERFLOW_PROMPT_TOKENS = NATIVE_CONTEXT
RETRIEVAL_OUTPUT_TOKENS = 64
BOUNDARY_EXTENSION_TOKENS = BOUNDARY_PROMPT_TOKENS - RETRIEVAL_PROMPT_TOKENS
SEED = 0x38_262_144
TOKENIZER_SHA256 = "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
EXPECTED_PROMPT_SHA256 = {
    CONTROL_PROMPT_TOKENS: "05d689a3149f158b2af5047421a152f03cc93c78f5377712801bf0d0f50ca2ad",
    BRIDGE_PROMPT_TOKENS: "9692378175dbb8126e65d7e4ee022d7150ffe8a06608fb9c936159c05ba4e806",
    RETRIEVAL_PROMPT_TOKENS: "a664d90f8924cc85fd3862c7617b9f9894a4c22b031757c93fd22d8aac00c3c0",
}
PASSKEYS = {
    "ALPHA": "alpha-0731826",
    "BETA": "beta-0492175",
    "GAMMA": "gamma-0863204",
}
PASSKEY_DEPTHS = (0.05, 0.50, 0.95)
HEAD_TEXT = "<|im_start|>user\nSession 0 archive, reference code S00-00000.\n"
FILLER_TEXT = (
    " Ordinary archive entry: the warehouse ledger lists shelves, labels, "
    "crates and loading bays.\n"
)
QUESTION_TEXT = (
    "\nReturn the three audit passkeys, labeled ALPHA, BETA, GAMMA. "
    "Copy them exactly; no explanation."
)
NO_THINKING_FRAME = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
PROMPT_COMPONENT_KEYS = ("head", "filler", "tail", "ALPHA", "BETA", "GAMMA")
QSA_MARKER = "Qwen4 QSA physical side cache initialized"
QSA_TARGET_BF16_RECEIPTS = {
    258_000: (
        "QWEN4_PREFILL_QSA_CHUNK_RECEIPT selector=ATLAS_QWEN4_PREFILL_QSA "
        "start=245760 rows=12240 end=258000"
    ),
    261_888: (
        "QWEN4_PREFILL_QSA_CHUNK_RECEIPT selector=ATLAS_QWEN4_PREFILL_QSA "
        "start=245760 rows=16128 end=261888"
    ),
    262_080: (
        "QWEN4_PREFILL_QSA_CHUNK_RECEIPT selector=ATLAS_QWEN4_PREFILL_QSA "
        "start=245760 rows=16320 end=262080"
    ),
}
QSA_BOUNDARY_BF16_RECEIPT = (
    "QWEN4_PREFILL_QSA_CHUNK_RECEIPT selector=ATLAS_QWEN4_PREFILL_QSA "
    "start=245760 rows=16383 end=262143"
)
QSA_FP8_MARKER = "core=qsa_sparse_e4m3_"
FORBIDDEN_LOG_MARKERS = (
    "CUDA out of memory",
    "KV cache exhausted",
    "NVRM: Xid",
    "illegal memory access",
    "context overflow truncated",
)


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


def workload_texts() -> dict[str, str]:
    texts = {
        "head": HEAD_TEXT,
        "filler": FILLER_TEXT,
        "tail": QUESTION_TEXT + NO_THINKING_FRAME,
    }
    texts.update(
        {
            name: f"\nAudit passkey {name}: {passkey}\n"
            for name, passkey in PASSKEYS.items()
        }
    )
    return texts


def canonical_passkey_answer() -> str:
    return "\n".join(f"{name}: {passkey}" for name, passkey in PASSKEYS.items())


def validate_token_vector(value: object, label: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} token vector is empty")
    if any(type(token) is not int or not 0 <= token <= 0xFFFF_FFFF for token in value):
        raise ValueError(f"{label} contains a non-u32 token")
    return value


def repeated(tokens: list[int], length: int) -> list[int]:
    repeats, tail = divmod(length, len(tokens))
    return tokens * repeats + tokens[:tail]


def assemble_prompt(
    target: int, components: dict[str, list[int]]
) -> tuple[list[int], dict[str, int]]:
    """Build the calibrated manual no-thinking chat prompt at an exact length."""
    if target not in RETRIEVAL_TARGETS:
        raise ValueError("target is outside the native-262K retrieval ladder")
    if tuple(components) != PROMPT_COMPONENT_KEYS:
        raise ValueError("component token map has wrong keys or order")
    for label, tokens in components.items():
        validate_token_vector(tokens, label)
    body_len = target - len(components["head"]) - len(components["tail"])
    if body_len <= 0:
        raise ValueError("target is too short for the framed prompt")
    body = repeated(components["filler"], body_len)
    for depth, name in reversed(list(zip(PASSKEY_DEPTHS, PASSKEYS))):
        at = int(len(body) * depth)
        body[at:at] = components[name]
    prompt = components["head"] + body[:body_len] + components["tail"]
    if len(prompt) != target:
        raise AssertionError("exact prompt assembly drifted")
    positions: dict[str, int] = {}
    for name in PASSKEYS:
        needle = components[name]
        limit = len(prompt) - len(needle) + 1
        matches = [
            index
            for index in range(limit)
            if prompt[index : index + len(needle)] == needle
        ]
        if len(matches) != 1:
            raise AssertionError(f"{name} needle occurrence count differs")
        positions[name] = matches[0]
    return prompt, positions


def extend_retrieval_to_boundary(
    retrieval_prompt: list[int], components: dict[str, list[int]]
) -> tuple[list[int], dict[str, int]]:
    """Insert the exact 63-token boundary extension immediately before the query.

    Keeping the needles and query byte-for-byte identical makes the one-token
    control at 262,080 a deterministic oracle for the 262,143-token request.
    """
    validate_token_vector(retrieval_prompt, "retrieval prompt")
    if len(retrieval_prompt) != RETRIEVAL_PROMPT_TOKENS:
        raise ValueError("boundary extension requires the exact retrieval prompt")
    if tuple(components) != PROMPT_COMPONENT_KEYS:
        raise ValueError("component token map has wrong keys or order")
    tail = validate_token_vector(components["tail"], "tail")
    filler = validate_token_vector(components["filler"], "filler")
    if len(tail) >= len(retrieval_prompt) or retrieval_prompt[-len(tail) :] != tail:
        raise ValueError("retrieval prompt does not end with the qualified chat frame")
    extension = repeated(filler, BOUNDARY_EXTENSION_TOKENS)
    start = len(retrieval_prompt) - len(tail)
    boundary_prompt = retrieval_prompt[:start] + extension + retrieval_prompt[start:]
    if len(boundary_prompt) != BOUNDARY_PROMPT_TOKENS:
        raise AssertionError("native-boundary prompt assembly drifted")
    return boundary_prompt, {
        "start": start,
        "end": start + len(extension),
        "tokens": len(extension),
    }


def prompt_sha256(tokens: list[int]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(token.to_bytes(4, "little"))
    return digest.hexdigest()


def completion_body(
    model: str,
    prompt: list[int],
    max_tokens: int,
    *,
    ignore_eos: bool = False,
    return_token_ids: bool = False,
) -> dict[str, Any]:
    validate_token_vector(prompt, "prompt")
    if not model or type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("model and positive max_tokens are required")
    body = {
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
        "stop": [],
        "seed": SEED,
        "ignore_eos": ignore_eos,
    }
    if return_token_ids:
        body["return_token_ids"] = True
    return body


def argv_value(argv: list[str], flag: str) -> str | None:
    try:
        index = argv.index(flag)
    except ValueError:
        return None
    return argv[index + 1] if index + 1 < len(argv) else None


def validate_server_argv(argv: object) -> list[str]:
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(x, str) or not x for x in argv)
    ):
        raise ValueError("argv must be a non-empty string array")
    required = {
        "--max-seq-len": str(NATIVE_CONTEXT),
        "--max-prefill-tokens": "16384",
        "--kv-cache-dtype": "bf16",
        "--max-batch-size": "1",
        "--max-num-seqs": "1",
        "--block-size": "16",
        "--enable-prefix-caching": "false",
        "--ssm-cache-slots": "0",
        "--swap-space-gb": "0",
        "--bind": "127.0.0.1",
    }
    for flag, expected in required.items():
        if argv.count(flag) != 1:
            raise ValueError(f"{flag} must occur exactly once")
        observed = argv_value(argv, flag)
        if observed != expected:
            raise ValueError(f"{flag} must equal {expected}, got {observed!r}")
    if argv.count("--qwen4-qsa") != 1:
        raise ValueError("native 262K qualification requires --qwen4-qsa")
    if "--rope-yarn-factor" in argv or "--rope-theta-override" in argv:
        raise ValueError("native 262K must not use context-extension RoPE flags")
    if any(flag in argv for flag in ("--speculative", "--dflash")):
        raise ValueError("native 262K capacity qualification is target-only")
    if argv.count("--port") != 1:
        raise ValueError("--port must occur exactly once")
    port = argv_value(argv, "--port")
    if (
        port is None
        or not port.isascii()
        or not port.isdecimal()
        or not 1 <= int(port) <= 65535
    ):
        raise ValueError("--port must be a decimal TCP port")
    if len(argv) < 2 or argv[1] != "serve":
        raise ValueError("binary argv must select the serve command")
    for flag in ("--model-from-path", "--model-name", "--kernel-target"):
        if argv.count(flag) != 1 or not argv_value(argv, flag):
            raise ValueError(f"{flag} must occur exactly once with a value")
    if argv_value(argv, "--kernel-target") != "qwen3.8-flash-next":
        raise ValueError(
            "native 262K qualification requires the qwen3.8-flash-next target"
        )
    return argv


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
        raise ValueError("base_url must match the child --bind and --port")
    return parsed.hostname, port


def require_unoccupied_endpoint(host: str, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.25)
        result = probe.connect_ex((host, port))
    if result == 0:
        raise RuntimeError(
            "qualification endpoint is already occupied before child launch"
        )
    if result != errno.ECONNREFUSED:
        raise OSError(result, "could not prove qualification endpoint is unused")


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
        values[key] = value.strip().strip('"').strip("'")
    required = {
        "FLUX_MODE": "256k",
        "FLUX_MAX_SEQ_LEN": str(NATIVE_CONTEXT),
        "FLUX_MAX_PREFILL_TOKENS": "16384",
        "FLUX_CONCURRENCY": "1",
        "FLUX_KV_CACHE_DTYPE": "bf16",
        "FLUX_SPEC_MODE": "none",
        "FLUX_CTX_OVERFLOW_TRUNCATE": "0",
        "ATLAS_CTX_OVERFLOW_TRUNCATE": "0",
        "ATLAS_QWEN4_QSA_VERIFY_ROWS": "0",
        "ATLAS_QWEN4_PREFILL_QSA_FP8": "0",
    }
    for key, expected in required.items():
        if values.get(key) != expected:
            raise ValueError(f"recipe requires {key}={expected}")
    return values


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
    release_version = provenance.get("release_version")
    r9_provenance.release_identity(release_version)

    # The full source-closure verifier predates R10, but its validation is not
    # intrinsically R9-specific. Bind the verifier to the selected immutable
    # release through its validated public configuration entry point while
    # preserving every clean-worktree, archive, build, embedded-commit, and
    # binary-hash check. Restoring the prior identity keeps repeated in-process
    # checks deterministic.
    original_release = r9_provenance.RELEASE_VERSION
    try:
        r9_provenance.configure_release(release_version)
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
        raise ValueError("checkpoint inventory must be outside the inventoried root")
    actual = []
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


def validate_tokenizer_pin(checkpoint_root: Path, verify_file: bool) -> None:
    """Require the exact tokenizer used to calibrate the prompt receipts."""
    if not verify_file:
        return
    tokenizer = checkpoint_root / "tokenizer.json"
    if not tokenizer.is_file():
        raise ValueError("checkpoint tokenizer.json is not a file")
    if sha256_file(tokenizer) != TOKENIZER_SHA256:
        raise ValueError(
            "checkpoint tokenizer.json SHA-256 differs from the pinned oracle"
        )


def load_manifest(path: Path, verify_files: bool = True) -> dict[str, Any]:
    manifest_bytes = path.read_bytes()
    value = strict_json_loads(manifest_bytes, f"launch manifest {path}")
    required = {"schema", "argv", "cwd", "env_file", "base_url", "model", "provenance"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("launch manifest keys differ")
    if value["schema"] != MANIFEST_SCHEMA:
        raise ValueError("wrong launch manifest schema")
    argv = validate_server_argv(value["argv"])
    base_url = value["base_url"]
    validate_base_url(base_url, argv)
    if not isinstance(value["model"], str) or not value["model"]:
        raise ValueError("model must be non-empty")
    cwd = Path(value["cwd"])
    env_path = Path(value["env_file"])
    if not env_path.is_absolute():
        env_path = (path.parent / env_path).resolve()
    recipe = parse_env_file(env_path)
    provenance = value["provenance"]
    pkeys = {
        "release_version",
        "source_revision",
        "source_tree",
        "source_root_path",
        "source_bundle_manifest_sha256",
        "source_bundle_manifest_path",
        "source_bundle_sha256",
        "source_bundle_path",
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
        "recipe_sha256",
    }
    if not isinstance(provenance, dict) or set(provenance) != pkeys:
        raise ValueError("provenance keys differ")
    r9_provenance.release_identity(provenance["release_version"])
    if not is_hex(provenance["source_revision"], 40):
        raise ValueError("source_revision must be a lowercase Git SHA-1")
    if not is_hex(provenance["source_tree"], 40):
        raise ValueError("source_tree must be a lowercase Git tree SHA-1")
    if not is_hex(provenance["binary_sha256"], 64):
        raise ValueError("binary_sha256 must be lowercase SHA-256")
    if (
        not isinstance(provenance["checkpoint_id"], str)
        or not provenance["checkpoint_id"]
    ):
        raise ValueError("checkpoint_id must be non-empty")
    if not (
        is_hex(provenance["checkpoint_revision"], 40)
        or is_hex(provenance["checkpoint_revision"], 64)
    ):
        raise ValueError("checkpoint_revision must be an immutable hash")
    for key in (
        "source_bundle_manifest_sha256",
        "source_bundle_sha256",
        "build_receipt_sha256",
        "source_to_binary_receipt_sha256",
        "checkpoint_manifest_sha256",
        "recipe_sha256",
    ):
        if not is_hex(provenance[key], 64):
            raise ValueError(f"{key} must be lowercase SHA-256")
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
        if (
            not isinstance(provenance[key], str)
            or not provenance[key]
            or not Path(provenance[key]).is_absolute()
        ):
            raise ValueError(f"{key} must be a non-empty absolute path")
    binary = Path(provenance["binary_path"])
    source_to_binary_receipt = Path(provenance["source_to_binary_receipt_path"])
    checkpoint_root = Path(provenance["checkpoint_path"])
    checkpoint_manifest = Path(provenance["checkpoint_manifest_path"])
    if Path(argv[0]).resolve() != binary.resolve():
        raise ValueError("argv[0] and provenance binary_path differ")
    if (
        Path(argv_value(argv, "--model-from-path") or "").resolve()
        != checkpoint_root.resolve()
    ):
        raise ValueError("argv checkpoint and provenance checkpoint_path differ")
    if argv_value(argv, "--model-name") != value["model"]:
        raise ValueError("argv --model-name and request model differ")
    if verify_files:
        if not cwd.is_dir():
            raise ValueError("launch cwd is not a directory")
        checks = (
            (binary, provenance["binary_sha256"], "binary"),
            (env_path, provenance["recipe_sha256"], "recipe"),
        )
        for file_path, expected, label in checks:
            if not file_path.is_file() or sha256_file(file_path) != expected:
                raise ValueError(f"{label} SHA-256 mismatch")
    validate_source_to_binary_receipt(
        source_to_binary_receipt, provenance, verify_files
    )
    validate_checkpoint_inventory(
        checkpoint_manifest, checkpoint_root, provenance, verify_files
    )
    validate_tokenizer_pin(checkpoint_root, verify_files)
    value["_env_path"] = str(env_path)
    value["_recipe"] = recipe
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
    if fresh["_recipe"] != manifest["_recipe"]:
        raise ValueError(f"{phase}: parsed recipe environment changed")


def public_provenance(provenance: dict[str, Any]) -> dict[str, Any]:
    """Keep immutable identities and hashes while omitting machine-local paths."""

    path_keys = {
        "source_root_path",
        "source_bundle_manifest_path",
        "source_bundle_path",
        "build_receipt_path",
        "source_to_binary_receipt_path",
        "binary_path",
        "checkpoint_path",
        "checkpoint_manifest_path",
    }
    return {key: value for key, value in provenance.items() if key not in path_keys}


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


def observed_json_request(
    client: HttpClient,
    method: str,
    route: str,
    body: object | None,
    label: str,
    observations: list[dict[str, Any]],
) -> tuple[int, Any, bytes, int]:
    """Issue one request while retaining the exact response bytes for audit."""
    status, raw = client.request(method, route, body)
    record: dict[str, Any] = {
        "label": label,
        "route": route,
        "http_status": status,
        "response_body_base64": base64.b64encode(raw).decode("ascii"),
        "response_sha256": sha256_bytes(raw),
    }
    observations.append(record)
    index = len(observations) - 1
    try:
        value = strict_json_loads(raw, f"{label} HTTP {status} response")
    except ValueError as error:
        record["json_decoded"] = False
        raise ValueError(
            f"{label} returned non-JSON or duplicate-key HTTP {status}"
        ) from error
    record["json_decoded"] = True
    record["response"] = value
    return status, value, raw, index


def tokenize_components(
    client: HttpClient, observations: list[dict[str, Any]] | None = None
) -> dict[str, list[int]]:
    texts = workload_texts()
    result: dict[str, list[int]] = {}
    sink = [] if observations is None else observations
    for label in PROMPT_COMPONENT_KEYS:
        status, value, _, _ = observed_json_request(
            client,
            "POST",
            "/tokenize",
            {"prompt": texts[label]},
            f"tokenize.{label}",
            sink,
        )
        if (
            status != 200
            or not isinstance(value, dict)
            or set(value) != {"tokens", "count"}
        ):
            raise ValueError(f"/tokenize failed for {label}: HTTP {status}")
        tokens = validate_token_vector(value["tokens"], label)
        if type(value["count"]) is not int or value["count"] != len(tokens):
            raise ValueError(f"/tokenize count differs for {label}")
        result[label] = tokens
    return result


def score_passkeys(text: str) -> dict[str, Any]:
    """Score retrieval semantically; retain exact formatting as advisory data."""
    if not isinstance(text, str):
        raise ValueError("passkey response text must be a string")
    retrieved = {name: passkey in text for name, passkey in PASSKEYS.items()}
    canonical = canonical_passkey_answer()
    return {
        "semantic_retrieval": {
            "retrieved": retrieved,
            "all_retrieved": all(retrieved.values()),
        },
        "exact_format": {
            "expected": canonical,
            "matched": text.strip() == canonical,
            "required_for_pass": False,
        },
    }


def validate_passkey_response(
    value: object,
    prompt_tokens: int,
    requested: int,
    *,
    require_token_ids: bool = False,
) -> dict[str, Any]:
    row = validate_usage(
        value,
        prompt_tokens,
        requested,
        require_token_ids=require_token_ids,
    )
    row.update(score_passkeys(row["text"]))
    return row


def validate_usage(
    value: object,
    prompt_tokens: int,
    requested: int,
    *,
    require_token_ids: bool = False,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("completion response must be an object")
    usage = value.get("usage")
    choices = value.get("choices")
    if (
        not isinstance(usage, dict)
        or not isinstance(choices, list)
        or len(choices) != 1
    ):
        raise ValueError("completion response shape differs")
    observed_prompt_tokens = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if (
        type(observed_prompt_tokens) is not int
        or observed_prompt_tokens != prompt_tokens
    ):
        raise ValueError("server prompt token accounting differs from exact request")
    if type(completion) is not int or not 1 <= completion <= requested:
        raise ValueError("post-prefill decode token count is out of range")
    total_tokens = usage.get("total_tokens")
    if type(total_tokens) is not int or total_tokens != prompt_tokens + completion:
        raise ValueError("total token accounting differs")
    details = usage.get("prompt_tokens_details")
    cached_tokens = details.get("cached_tokens") if isinstance(details, dict) else None
    if type(cached_tokens) is not int or cached_tokens != 0:
        raise ValueError("qualification request unexpectedly used cached prompt tokens")
    choice = choices[0]
    choice_index = choice.get("index") if isinstance(choice, dict) else None
    if (
        not isinstance(choice, dict)
        or type(choice_index) is not int
        or choice_index != 0
        or not isinstance(choice.get("text"), str)
    ):
        raise ValueError("completion choice differs")
    token_ids: list[int] | None = None
    if require_token_ids or "token_ids" in choice:
        token_ids = validate_token_vector(
            choice.get("token_ids"), "scheduler-owned completion"
        )
        if len(token_ids) != completion:
            raise ValueError(
                "scheduler-owned completion token IDs differ from completion_tokens"
            )
    row = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion,
        "total_tokens": prompt_tokens + completion,
        "finish_reason": choice.get("finish_reason"),
        "text": choice["text"],
        "response_sha256": sha256_bytes(canonical_bytes(value)),
        "time_to_first_token_ms": usage.get("time_to_first_token_ms"),
    }
    if token_ids is not None:
        row["completion_token_ids"] = token_ids
        row["completion_u32le_sha256"] = prompt_sha256(token_ids)
    return row


def validate_boundary_oracle(control: dict[str, Any], boundary: dict[str, Any]) -> int:
    """Require the 262143+1 scheduler token ID to equal the 262080+1 control."""
    if control.get("completion_tokens") != 1 or boundary.get("completion_tokens") != 1:
        raise ValueError(
            "boundary oracle requires one-token control and boundary outputs"
        )
    control_ids = validate_token_vector(
        control.get("completion_token_ids"), "boundary control completion"
    )
    boundary_ids = validate_token_vector(
        boundary.get("completion_token_ids"), "native boundary completion"
    )
    if len(control_ids) != 1 or len(boundary_ids) != 1:
        raise ValueError(
            "boundary oracle requires one scheduler-owned token ID per row"
        )
    if boundary_ids != control_ids:
        raise ValueError("262143 boundary token ID differs from the 262080 control")
    return control_ids[0]


def validate_qsa_log_contract(logs: str) -> None:
    if QSA_MARKER not in logs:
        raise ValueError("server log lacks QSA side-cache engagement marker")
    receipts = tuple(QSA_TARGET_BF16_RECEIPTS.items()) + (
        (BOUNDARY_PROMPT_TOKENS, QSA_BOUNDARY_BF16_RECEIPT),
    )
    for endpoint, marker in receipts:
        if marker not in logs:
            raise ValueError(
                f"server log lacks BF16 sparse-QSA endpoint receipt for {endpoint} tokens"
            )
    if QSA_FP8_MARKER in logs:
        raise ValueError(
            "server log shows quantized QSA prefill operands in the BF16 gate"
        )
    found_bad = [marker for marker in FORBIDDEN_LOG_MARKERS if marker in logs]
    if found_bad:
        raise ValueError(f"server log contains fatal markers: {found_bad!r}")


def write_all(fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(fd, payload[offset:])
        if written <= 0:
            raise OSError("receipt write made no progress")
        offset += written


def seal_json_artifact(path: Path, value: object) -> None:
    payload = canonical_bytes(value)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        write_all(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)


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


def run_live(
    manifest: dict[str, Any], manifest_path: Path, receipt_path: Path, timeout: float
) -> dict[str, Any]:
    run_nonce = uuid.uuid4().hex
    log_path = receipt_path.with_suffix(receipt_path.suffix + ".server.log")
    failure_path = receipt_path.with_suffix(receipt_path.suffix + ".failure.json")
    if receipt_path.exists() or log_path.exists() or failure_path.exists():
        raise FileExistsError("receipt, failure artifact, or server log already exists")
    started = time.time()
    evidence: dict[str, Any] = {
        "schema": FAILURE_SCHEMA,
        "status": "RUNNING",
        "execution_scope": "direct-binary-not-container",
        "run_nonce": run_nonce,
        "fresh_process": True,
        "started_unix": started,
        "manifest_sha256": manifest["_manifest_source_sha256"],
        "manifest_canonical_sha256": manifest["_manifest_canonical_sha256"],
        "provenance": public_provenance(manifest["provenance"]),
        "prompt_contract": {
            "frame": "qad5500-hybrid-manual-no-thinking-chat",
            "tokenizer_json_sha256": TOKENIZER_SHA256,
            "expected_prompt_u32le_sha256": {
                str(target): digest for target, digest in EXPECTED_PROMPT_SHA256.items()
            },
            "semantic_passkey_retrieval_is_gate": True,
            "exact_format_is_gate": False,
        },
        "cases": [],
        "http_observations": [],
    }
    try:
        return _run_live_impl(
            manifest,
            manifest_path,
            receipt_path,
            log_path,
            timeout,
            started,
            run_nonce,
            evidence,
        )
    except BaseException as error:
        evidence["status"] = "FAIL"
        evidence["elapsed_seconds"] = time.time() - started
        evidence["error_type"] = type(error).__name__
        evidence["error"] = str(error)
        if log_path.is_file():
            try:
                evidence["server_log_sha256"] = sha256_file(log_path)
            except Exception as hash_error:
                evidence["server_log_hash_error"] = str(hash_error)
        # Every response body is already present losslessly as base64 in
        # http_observations, even when JSON decoding or a later gate fails.
        seal_json_artifact(failure_path, evidence)
        raise


def _run_live_impl(
    manifest: dict[str, Any],
    manifest_path: Path,
    receipt_path: Path,
    log_path: Path,
    timeout: float,
    started: float,
    run_nonce: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    revalidate_manifest_and_artifacts(manifest, manifest_path, "before native-262K arm")
    # A qualification receipt must be determined by its hashed recipe. Drop
    # unrecorded inherited engine selectors before applying that recipe.
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("ATLAS_", "FLUX_"))
    }
    environment.update(manifest["_recipe"])
    host, port = validate_base_url(manifest["base_url"], manifest["argv"])
    require_unoccupied_endpoint(host, port)
    process: subprocess.Popen[bytes] | None = None
    log_handle = log_path.open("xb")
    try:
        process = subprocess.Popen(
            manifest["argv"],
            cwd=manifest["cwd"],
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        client = HttpClient(manifest["base_url"], timeout)
        deadline = time.monotonic() + timeout
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"fresh server exited rc={process.returncode}")
            try:
                status, _ = client.request("GET", "/health")
                if status == 200:
                    break
            except (OSError, urllib.error.URLError):
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError("fresh server did not become healthy")
            time.sleep(1.0)

        log_capture = endpoint_gate.RequestLogCapture(log_path)
        components = tokenize_components(client, evidence["http_observations"])
        prompts: dict[int, list[int]] = {}
        retrieval_ladder: list[dict[str, Any]] = []
        for target in RETRIEVAL_TARGETS:
            prompt, positions = assemble_prompt(target, components)
            prompts[target] = prompt
            prompt_hash = prompt_sha256(prompt)
            case: dict[str, Any] = {
                "case": f"semantic-retrieval-{target}",
                "target_prompt_tokens": target,
                "requested_tokens": 64 if target == RETRIEVAL_PROMPT_TOKENS else 96,
                "prompt_u32le_sha256": prompt_hash,
                "expected_prompt_u32le_sha256": EXPECTED_PROMPT_SHA256[target],
                "needle_positions": positions,
            }
            evidence["cases"].append(case)
            retrieval_ladder.append(case)
            if prompt_hash != EXPECTED_PROMPT_SHA256[target]:
                raise ValueError(f"{target} calibrated prompt SHA-256 differs")
            body = completion_body(manifest["model"], prompt, case["requested_tokens"])
            status, value, _, observation_index = observed_json_request(
                client,
                "POST",
                "/v1/completions",
                body,
                case["case"],
                evidence["http_observations"],
            )
            case["http_observation_index"] = observation_index
            if status != 200:
                raise ValueError(f"{target} retrieval returned HTTP {status}")
            row = validate_passkey_response(value, target, case["requested_tokens"])
            case["observation"] = row
            if row["completion_tokens"] < 2:
                raise ValueError(f"{target} retrieval did not exercise decode")
            if not row["semantic_retrieval"]["all_retrieved"]:
                raise ValueError(f"{target} retrieval lost one or more passkeys")

        retrieval_prompt = prompts[RETRIEVAL_PROMPT_TOKENS]
        target_observation = retrieval_ladder[-1]["observation"]
        repeat_case: dict[str, Any] = {
            "case": "semantic-retrieval-262080-repeat",
            "target_prompt_tokens": RETRIEVAL_PROMPT_TOKENS,
            "requested_tokens": RETRIEVAL_OUTPUT_TOKENS,
            "prompt_u32le_sha256": EXPECTED_PROMPT_SHA256[RETRIEVAL_PROMPT_TOKENS],
        }
        evidence["cases"].append(repeat_case)
        repeat_body = completion_body(
            manifest["model"], retrieval_prompt, RETRIEVAL_OUTPUT_TOKENS
        )
        status, value, _, observation_index = observed_json_request(
            client,
            "POST",
            "/v1/completions",
            repeat_body,
            repeat_case["case"],
            evidence["http_observations"],
        )
        repeat_case["http_observation_index"] = observation_index
        if status != 200:
            raise ValueError(f"262080 repeat returned HTTP {status}")
        repeat = validate_passkey_response(
            value, RETRIEVAL_PROMPT_TOKENS, RETRIEVAL_OUTPUT_TOKENS
        )
        repeat_case["observation"] = repeat
        repeat_case["determinism"] = {
            "text_identical": repeat["text"] == target_observation["text"],
            "completion_tokens_identical": repeat["completion_tokens"]
            == target_observation["completion_tokens"],
            "finish_reason_identical": repeat["finish_reason"]
            == target_observation["finish_reason"],
        }
        if not repeat["semantic_retrieval"]["all_retrieved"] or not all(
            repeat_case["determinism"].values()
        ):
            raise ValueError(
                "262080 semantic retrieval is incorrect or nondeterministic"
            )

        boundary_decode_body = completion_body(
            manifest["model"],
            retrieval_prompt,
            RETRIEVAL_OUTPUT_TOKENS,
            ignore_eos=True,
            return_token_ids=True,
        )
        forced_case: dict[str, Any] = {
            "case": "semantic-retrieval-262080-forced-plus-64",
            "target_prompt_tokens": RETRIEVAL_PROMPT_TOKENS,
            "requested_tokens": RETRIEVAL_OUTPUT_TOKENS,
            "ignore_eos": True,
        }
        evidence["cases"].append(forced_case)
        status, value, _, observation_index = observed_json_request(
            client,
            "POST",
            "/v1/completions",
            boundary_decode_body,
            forced_case["case"],
            evidence["http_observations"],
        )
        forced_case["http_observation_index"] = observation_index
        if status != 200:
            raise ValueError(f"262080+64 boundary decode returned HTTP {status}")
        boundary_decode = validate_passkey_response(
            value,
            RETRIEVAL_PROMPT_TOKENS,
            RETRIEVAL_OUTPUT_TOKENS,
            require_token_ids=True,
        )
        forced_case["observation"] = boundary_decode
        if (
            boundary_decode["completion_tokens"] != RETRIEVAL_OUTPUT_TOKENS
            or boundary_decode["total_tokens"] != NATIVE_CONTEXT
        ):
            raise ValueError(
                "ignore_eos boundary decode did not reach 262144 total tokens"
            )
        if not boundary_decode["semantic_retrieval"]["all_retrieved"]:
            raise ValueError("ignore_eos boundary decode lost one or more passkeys")

        control_case: dict[str, Any] = {
            "case": "262080-plus-1-control",
            "target_prompt_tokens": RETRIEVAL_PROMPT_TOKENS,
            "requested_tokens": 1,
        }
        evidence["cases"].append(control_case)
        control_body = completion_body(
            manifest["model"], retrieval_prompt, 1, return_token_ids=True
        )
        status, value, _, observation_index = observed_json_request(
            client,
            "POST",
            "/v1/completions",
            control_body,
            control_case["case"],
            evidence["http_observations"],
        )
        control_case["http_observation_index"] = observation_index
        if status != 200:
            raise ValueError(f"262080 one-token control returned HTTP {status}")
        boundary_control = validate_usage(
            value, RETRIEVAL_PROMPT_TOKENS, 1, require_token_ids=True
        )
        control_case["observation"] = boundary_control

        boundary_prompt, boundary_extension = extend_retrieval_to_boundary(
            retrieval_prompt, components
        )
        boundary_case: dict[str, Any] = {
            "case": "262143-plus-1-boundary-identity",
            "target_prompt_tokens": BOUNDARY_PROMPT_TOKENS,
            "requested_tokens": 1,
            "prompt_u32le_sha256": prompt_sha256(boundary_prompt),
            "extension_span": boundary_extension,
        }
        evidence["cases"].append(boundary_case)
        boundary_body = completion_body(
            manifest["model"], boundary_prompt, 1, return_token_ids=True
        )
        status, value, _, observation_index = observed_json_request(
            client,
            "POST",
            "/v1/completions",
            boundary_body,
            boundary_case["case"],
            evidence["http_observations"],
        )
        boundary_case["http_observation_index"] = observation_index
        if status != 200:
            raise ValueError(f"262143+1 boundary returned HTTP {status}")
        boundary = validate_usage(
            value, BOUNDARY_PROMPT_TOKENS, 1, require_token_ids=True
        )
        boundary_case["observation"] = boundary
        if (
            boundary["completion_tokens"] != 1
            or boundary["total_tokens"] != NATIVE_CONTEXT
        ):
            raise ValueError(
                "native-context boundary did not produce exactly one token"
            )
        boundary_oracle_token_id = validate_boundary_oracle(boundary_control, boundary)
        boundary_case["identity"] = {
            "matched": True,
            "basis": "exact scheduler-owned completion token ID",
            "control_token_id": boundary_oracle_token_id,
            "control_text_advisory": boundary_control["text"],
            "boundary_text_advisory": boundary["text"],
        }

        overflow_prompt = boundary_prompt + [components["filler"][0]]
        if len(overflow_prompt) != OVERFLOW_PROMPT_TOKENS:
            raise AssertionError("overflow prompt length drifted")
        overflow_body = completion_body(manifest["model"], overflow_prompt, 1)
        overflow_case: dict[str, Any] = {
            "case": "262144-plus-1-overflow-rejection",
            "target_prompt_tokens": OVERFLOW_PROMPT_TOKENS,
            "requested_tokens": 1,
            "prompt_u32le_sha256": prompt_sha256(overflow_prompt),
        }
        evidence["cases"].append(overflow_case)
        metrics_before = endpoint_gate.capture_metrics(client)
        log_offset = log_capture.begin()
        http_request_id = f"qualify_overflow_{uuid.uuid4().hex}"
        status, overflow_value, raw, response_headers = client.correlated_json(
            "POST", "/v1/completions", overflow_body, http_request_id
        )
        observation = {
            "label": overflow_case["case"],
            "route": "/v1/completions",
            "http_status": status,
            "response_body_base64": base64.b64encode(raw).decode("ascii"),
            "response_sha256": sha256_bytes(raw),
            "json_decoded": True,
            "response": overflow_value,
        }
        evidence["http_observations"].append(observation)
        observation_index = len(evidence["http_observations"]) - 1
        overflow_case["http_observation_index"] = observation_index
        log_delta = log_capture.finish(log_offset)
        metrics_after = endpoint_gate.capture_metrics(client)
        rejection = endpoint_gate.validate_context_length_rejection(
            status,
            overflow_value,
            OVERFLOW_PROMPT_TOKENS,
            1,
            NATIVE_CONTEXT,
        )
        overflow_case["rejected_as_required"] = True
        overflow_case["rejection"] = rejection
        overflow_case["http_request_id"] = endpoint_gate.validate_correlated_request_id(
            response_headers, http_request_id
        )
        overflow_case["log_delta_bytes"] = len(log_delta)
        overflow_case["log_delta_sha256"] = sha256_bytes(log_delta)
        overflow_case["scheduler_evidence"] = (
            endpoint_gate.validate_no_scheduler_dispatch(
                log_delta.decode("utf-8", errors="replace"),
                metrics_before,
                metrics_after,
                http_request_id,
            )
        )

        health_status, health_raw = client.request("GET", "/health")
        if type(health_status) is not int or health_status != 200:
            raise ValueError(
                f"post-rejection health endpoint returned HTTP {health_status}"
            )
        post_canary_case: dict[str, Any] = {
            "case": "post-overflow-262080-plus-1-canary",
            "target_prompt_tokens": RETRIEVAL_PROMPT_TOKENS,
            "requested_tokens": 1,
        }
        evidence["cases"].append(post_canary_case)
        post_canary_body = completion_body(
            manifest["model"], retrieval_prompt, 1, return_token_ids=True
        )
        status, value, _, observation_index = observed_json_request(
            client,
            "POST",
            "/v1/completions",
            post_canary_body,
            post_canary_case["case"],
            evidence["http_observations"],
        )
        post_canary_case["http_observation_index"] = observation_index
        if type(status) is not int or status != 200:
            raise ValueError(f"post-overflow canary returned HTTP {status}")
        post_canary = validate_usage(
            value, RETRIEVAL_PROMPT_TOKENS, 1, require_token_ids=True
        )
        post_canary_case["observation"] = post_canary
        metrics_post_canary = endpoint_gate.capture_metrics(client)
        post_canary_case["metrics"] = endpoint_gate.validate_post_canary_metrics(
            metrics_after,
            metrics_post_canary,
            prompt_tokens=RETRIEVAL_PROMPT_TOKENS,
            prefill_chunks=(RETRIEVAL_PROMPT_TOKENS + 16_384 - 1) // 16_384,
        )
        overflow_case["post_rejection_health"] = {
            "http_status": health_status,
            "body_sha256": sha256_bytes(health_raw),
            "body_bytes": len(health_raw),
        }
        overflow_case["post_rejection_completion_canary"] = post_canary_case

        if process.poll() is not None:
            raise RuntimeError(
                f"fresh server exited before receipt sealing rc={process.returncode}"
            )
        # Stop the process before sealing the log hash. Otherwise shutdown lines
        # could mutate the file after the receipt records its digest.
        stop_fresh_process(process)
        log_handle.flush()
        os.fsync(log_handle.fileno())
        log_handle.close()
        logs = log_path.read_text(encoding="utf-8", errors="replace")
        validate_qsa_log_contract(logs)
        overflow_case["final_dispatch_census"] = (
            endpoint_gate.validate_final_dispatch_census(
                logs,
                {
                    CONTROL_PROMPT_TOKENS: 1,
                    BRIDGE_PROMPT_TOKENS: 1,
                    RETRIEVAL_PROMPT_TOKENS: 5,
                    BOUNDARY_PROMPT_TOKENS: 1,
                },
                rejected_prompt_tokens=OVERFLOW_PROMPT_TOKENS,
            )
        )
        revalidate_manifest_and_artifacts(manifest, manifest_path, "before final seal")

        receipt = {
            "schema": SCHEMA,
            "status": "PASS",
            "execution_scope": "direct-binary-not-container",
            "run_nonce": run_nonce,
            "fresh_process": True,
            "process_pid": process.pid,
            "started_unix": started,
            "elapsed_seconds": time.time() - started,
            "manifest_sha256": manifest["_manifest_source_sha256"],
            "manifest_canonical_sha256": manifest["_manifest_canonical_sha256"],
            "provenance": public_provenance(manifest["provenance"]),
            "runtime_contract": {
                "max_seq_len": NATIVE_CONTEXT,
                "kv_cache_dtype": "bf16",
                "qwen4_qsa": True,
                "concurrency": 1,
                "speculative": False,
                "rope_extension": False,
                "overflow_truncation": False,
            },
            "prompt_contract": evidence["prompt_contract"],
            "retrieval_ladder": retrieval_ladder,
            "repeat_262080": repeat_case,
            "forced_to_262144": forced_case,
            "boundary_control": control_case,
            "native_boundary": boundary_case,
            "overflow": overflow_case,
            "http_observations": evidence["http_observations"],
            "qsa_log_marker": QSA_MARKER,
            "qsa_endpoint_bf16_log_markers": [
                *QSA_TARGET_BF16_RECEIPTS.values(),
                QSA_BOUNDARY_BF16_RECEIPT,
            ],
            "server_log_sha256": sha256_file(log_path),
        }
        seal_json_artifact(receipt_path, receipt)
        return receipt
    finally:
        cleanup_errors = []
        if process is not None:
            try:
                stop_fresh_process(process)
            except Exception as error:
                cleanup_errors.append(f"process cleanup failed: {error}")
        if not log_handle.closed:
            try:
                log_handle.flush()
                os.fsync(log_handle.fileno())
            except Exception as error:
                cleanup_errors.append(f"server log finalization failed: {error}")
            finally:
                log_handle.close()
        if cleanup_errors:
            evidence["cleanup_errors"] = cleanup_errors
            if sys.exc_info()[0] is None:
                raise RuntimeError("; ".join(cleanup_errors))


def dry_run_plan(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": SCHEMA + "/dry-run",
        "fresh_process": True,
        "execution_scope": "direct-binary-not-container",
        "argv_contract": {
            "max_seq_len": NATIVE_CONTEXT,
            "kv_cache_dtype": "bf16",
            "qwen4_qsa": True,
            "concurrency": 1,
            "target_only": True,
            "rope_extension": False,
        },
        "prompt_contract": {
            "frame": "qad5500-hybrid-manual-no-thinking-chat",
            "tokenizer_json_sha256": TOKENIZER_SHA256,
            "expected_prompt_u32le_sha256": {
                str(target): digest for target, digest in EXPECTED_PROMPT_SHA256.items()
            },
            "semantic_passkey_retrieval_is_gate": True,
            "exact_format_is_gate": False,
        },
        "cases": [
            {
                "prompt_tokens": CONTROL_PROMPT_TOKENS,
                "max_tokens": 96,
                "semantic_passkeys": True,
                "post_prefill_decode": True,
                "prior_comparable_control": True,
            },
            {
                "prompt_tokens": BRIDGE_PROMPT_TOKENS,
                "max_tokens": 96,
                "semantic_passkeys": True,
                "post_prefill_decode": True,
                "bridge": True,
            },
            {
                "prompt_tokens": RETRIEVAL_PROMPT_TOKENS,
                "max_tokens": 64,
                "semantic_passkeys": True,
                "post_prefill_decode": True,
                "native_target": True,
            },
            {
                "prompt_tokens": RETRIEVAL_PROMPT_TOKENS,
                "max_tokens": 64,
                "semantic_passkeys": True,
                "repeat_determinism": True,
            },
            {
                "prompt_tokens": RETRIEVAL_PROMPT_TOKENS,
                "max_tokens": 64,
                "ignore_eos": True,
                "exact_output_token_ids": True,
                "required_total_tokens": NATIVE_CONTEXT,
            },
            {
                "prompt_tokens": RETRIEVAL_PROMPT_TOKENS,
                "max_tokens": 1,
                "boundary_oracle_control": True,
                "exact_output_token_ids": True,
            },
            {
                "prompt_tokens": BOUNDARY_PROMPT_TOKENS,
                "max_tokens": 1,
                "boundary_generation": True,
                "exact_output_token_ids": True,
                "matches_control_token_id": True,
            },
            {
                "prompt_tokens": OVERFLOW_PROMPT_TOKENS,
                "max_tokens": 1,
                "expected_http": 400,
                "no_truncation": True,
            },
        ],
        "provenance": public_provenance(manifest["provenance"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--receipt", type=Path, default=Path("qwen4-native-262k-receipt.json")
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
