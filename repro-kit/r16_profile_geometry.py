# SPDX-License-Identifier: AGPL-3.0-only
"""Versioned profile geometry (/2) and the dual-recipe live broker contract.

Pure validation only: no process, container, model or device access. The
existing single-recipe broker (`r15-container-suite.py`, request /1) and the
three hard-coded 4096 sites are unchanged; R16_PROFILE_GEOMETRY_V2.md maps
each of them to the explicit interface below.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shlex
from typing import Any, Mapping

GEOMETRY_V1 = "flux-profile-geometry/1"
GEOMETRY_V2 = "flux-profile-geometry/2"
BROKER_V1 = "flux-r15-live-process-request/1"
BROKER_V2 = "flux-r16-live-process-broker/2"
REQUEST_V2 = "flux-r16-live-process-request/2"
ROW_V2 = "flux-profile-geometry-row/2"

V1_CHUNK_TOKENS = 4096
PROFILE_NAMES = ("interactive", "high-prefill")
MODES = ("none", "native-mtp-k5")
PROFILE_CONTEXTS = (8_192, 16_384, 32_768, 65_536, 131_072, 262_080)
# Blocking exact token IDs, the one-token request wall, and streaming usage.
REQUEST_KINDS = ("exact", "ttft", "stream")
CENSUS_RECORDS = len(PROFILE_NAMES) * len(MODES) * len(PROFILE_CONTEXTS) * len(REQUEST_KINDS)
DECODE_BUDGET = 256
FINAL_DECODE_BUDGET = 64
# Clocks and host identity are carried unchanged from host reference /3.
DECODE_CLOCK = "scheduler-decode-time:completion-tokens-minus-one/v1"
PREFILL_CLOCK = "client-monotonic-blocking-max-tokens-one/v1"
HOST_IDENTITY_KEYS = (
    "source_revision", "source_tree", "source_bundle_manifest_sha256",
    "checkpoint_id", "checkpoint_revision", "checkpoint_manifest_sha256",
    "binary_sha256",
)

# Reviewed per-profile geometry. WIDE is per-chunk row geometry, so it must
# equal the scheduler chunk and --max-prefill-tokens; capacity is the native
# window in 16-token blocks.
REVIEWED_GEOMETRY = {
    "interactive": {
        "recipe_relpath": "repro-kit/recipes/flux-v9-w4096-interactive.env",
        "flux_mode": "interactive-w4096",
        "wide": 4096,
        "chunk_tokens": 4096,
        "max_prefill_tokens": 4096,
        "block_size": 16,
        "capacity_tokens": 262_144,
    },
    "high-prefill": {
        "recipe_relpath": "repro-kit/recipes/flux-v9-max-prefill.env",
        "flux_mode": "max-prefill",
        "wide": 16384,
        "chunk_tokens": 16384,
        "max_prefill_tokens": 16384,
        "block_size": 16,
        "capacity_tokens": 262_144,
    },
}
GEOMETRY_KEYS = tuple(REVIEWED_GEOMETRY["interactive"])
# Each profile is bound to its OWN reviewed recipe bytes: a consistent swap of
# the two hashes is refused. A reviewed recipe edit must refresh these pins.
# Re-pinned for 9.15.6. Shipped changes since the last pins:
# interactive: 295a6aaa1, 753bc22b9, 0240e05ce, f5f230a87; high-prefill: the
# same plus e563a8138 and 8cd57fde6.
REVIEWED_RECIPE_SHA256 = {
    "interactive": "318b36a75b485a4b73cc20b4b13285d21a1c66327f1de5a95d574c6a03039663",
    "high-prefill": "2942f6ae474c84a175bb3969f2d6eedad43c60bd917162f76ad443f706420889",
}
IN_CONTAINER_ROOT = "/opt/flux/"

# Every artifact version moves together; any cross pair is refused.
VERSION_PAIRINGS = {
    GEOMETRY_V1: {
        "plan": "flux-r15-container-suite-plan/2",
        "contract": "flux-performance-profile-contract/1",
        "host_reference": "flux-prefill-decode-host-reference/3",
        "metrics": "flux-profile-metrics/3",
        "proof": "flux-r15-live-process-proof/1",
    },
    GEOMETRY_V2: {
        "plan": "flux-r15-container-suite-plan/3",
        "contract": "flux-performance-profile-contract/2",
        "host_reference": "flux-prefill-decode-host-reference/4",
        "metrics": "flux-profile-metrics/4",
        "proof": "flux-r16-live-process-proof/2",
    },
}

# Pinned in every effective argv: omitting one would select a CLI default.
PINNED_FLAGS = ("--max-prefill-tokens", "--max-seq-len", "--block-size",
                "--scheduling-policy", "--tbt-deadline-ms")
# One scheduling policy for both profiles and modes. FIFO ignores the TBT
# deadline (cli.rs:689-695), which is still pinned at its CLI default.
REVIEWED_SCHEDULING = ("fifo", "100")
# Engine selectors of the host recipe argv (recipes/flux-serve.argv and
# flux-serve-w4096.argv). The container launch must carry the same values, or
# CLI defaults apply in the container only: --gpu-memory-utilization 0.90,
# --ssm-cache-slots 16 and --mtp-vocab 100000 (cli.rs:404, :766, :660), and no
# --kernel-target or simhash watchdog switch. The host and container argv then
# differ only in executable path, model path, model alias, bind and port.
ENGINE_FLAGS = (
    ("--kernel-target", "qwen3.8-flash-next"),
    ("--gpu-memory-utilization", "0.85"),
    ("--ssm-cache-slots", "0"),
    ("--oom-guard-mb", "4096"),
    ("--request-timeout", "300"),
    ("--disable-simhash-watchdog", None),
    ("--mtp-vocab", "0"),
    ("--mtp-quantization", "bf16"),
)
# Mode is derived from the argv itself: native K5 is exactly --speculative
# with --num-drafts 4; Plain carries no speculation selector at all.
SPECULATION_FLAGS = ("--speculative", "--num-drafts", "--mtp-from-path", "--mtp-gate",
                     "--draft-model", "--dflash", "--dflash-donor-model",
                     "--self-speculative", "--ngram-speculative")
K5_NUM_DRAFTS = "4"
DISABLE_SPECULATION = "ATLAS_DISABLE_SPECULATION"
DECIMAL = re.compile(r"0|[1-9][0-9]*")
MAX_ARENA_TOKENS = 1 << 32
SCHEDULING_POLICIES = ("fifo", "slai")
WIDE_SELECTORS = ("FLUX_QWEN4_PREFILL_WIDE", "ATLAS_QWEN4_PREFILL_WIDE")
MAX_BATCH_TOKENS = "ATLAS_MAX_BATCH_TOKENS"

# Independent release authority. These pins do not come from the document
# being checked, so a consistently resealed geometry and broker pair still
# cannot admit another executable, model, base argv or KV dtype.
REVIEWED_EXECUTABLE = {
    "path": "/opt/flux/bin/flux-server-real",
    "sha256": "3460c766856d736dc7bae854d4cb486d4ab6fde79c36ca15015a7fa5619027d3",
    "size": 92_082_232,
}
MODEL_PATH = "/models"
KV_CACHE_DTYPE = "bf16"
MAX_BATCH_SIZE = 1
MODEL_PLACEHOLDER = "@PROFILE_MODEL@"
PORT_PLACEHOLDER = "@CONTAINER_PORT@"
MODEL_ALIAS = re.compile(r"flux-profile-prefill-decode-profiles-[0-9a-f]{32}")
MODEL_INDEX, PORT_INDEX = 6, 10
# kv_cache.rs at e82 (resolve_prefill_budget_with_override): spec_tokens =
# drafts + 2 when speculative, else 1 (:131). The default and minimum arena is
# max(prefill_budget + max_batch_size, spec_tokens,
# padded_decode_rows(max_batch_size)) (:223-230). An override must be >= that
# minimum (:108-121) and <= max(max_seq_len * max_batch_size, minimum)
# (:236-246). padded_decode_rows uses this ladder (spark-runtime
# buffers/decode_meta.rs:14-26).
PADDED_DECODE_LADDER = (2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128)


def reviewed_overlay(profile: str) -> list[str]:
    """The sealed per-profile overlay: geometry, QSA, scheduling, engine flags.

    The suite's and the wrapper's contract /2 overlay must equal this list
    exactly; only the --max-prefill-tokens value differs between profiles.
    """
    if profile not in PROFILE_NAMES:
        raise GeometryError("unknown profile")
    overlay = [
        "--max-prefill-tokens", str(REVIEWED_GEOMETRY[profile]["chunk_tokens"]),
        "--max-seq-len", str(REVIEWED_GEOMETRY[profile]["capacity_tokens"]),
        "--block-size", str(REVIEWED_GEOMETRY[profile]["block_size"]),
        "--qwen4-qsa", "--scheduling-policy", REVIEWED_SCHEDULING[0],
        "--tbt-deadline-ms", REVIEWED_SCHEDULING[1],
    ]
    for flag, value in ENGINE_FLAGS:
        overlay += [flag] if value is None else [flag, value]
    return overlay


def reviewed_effective_argv(profile: str, mode: str) -> list[str]:
    """The exact reviewed launch argv; alias and port are placeholders."""
    if profile not in PROFILE_NAMES or mode not in MODES:
        raise GeometryError("unknown profile or mode")
    argv = [
        REVIEWED_EXECUTABLE["path"], "serve", "--no-tui",
        "--model-from-path", MODEL_PATH, "--model-name", MODEL_PLACEHOLDER,
        "--bind", "0.0.0.0", "--port", PORT_PLACEHOLDER,
        "--max-batch-size", str(MAX_BATCH_SIZE), "--max-num-seqs", "1",
        "--kv-cache-dtype", KV_CACHE_DTYPE, "--enable-prefix-caching", "false",
        *reviewed_overlay(profile),
    ]
    if mode == "native-mtp-k5":
        argv += ["--speculative", "--num-drafts", K5_NUM_DRAFTS]
    return argv


def reviewed_geometry_v2() -> dict[str, Any]:
    """The one trusted /2 geometry; its sha is REVIEWED_GEOMETRY_SHA256."""
    return {
        "schema": GEOMETRY_V2,
        "profiles": {
            name: dict(
                reviewed,
                recipe_sha256=REVIEWED_RECIPE_SHA256[name],
                max_batch_tokens=None,
                effective_argv={mode: reviewed_effective_argv(name, mode) for mode in MODES},
            )
            for name, reviewed in REVIEWED_GEOMETRY.items()
        },
    }


# Follows REVIEWED_RECIPE_SHA256 (the geometry binds each profile's recipe).
REVIEWED_GEOMETRY_SHA256 = "85e926066028abb997005074f201d208141d1214e8913097552d2dd574ed60de"


def padded_decode_rows(rows: int) -> int:
    return next((rung for rung in PADDED_DECODE_LADDER if rung >= rows), rows)


def arena_bounds(chunk: int, capacity: int, max_batch_size: int = MAX_BATCH_SIZE,
                 num_drafts: int = int(K5_NUM_DRAFTS)) -> tuple[int, int]:
    """(minimum, maximum) ATLAS_MAX_BATCH_TOKENS for a profile serving both
    modes: chunk plus decode slots, the K5 verify rows and the padded decode
    width, mirroring the engine's resolved minimum."""
    minimum = max(chunk + max_batch_size, num_drafts + 2, 1, padded_decode_rows(max_batch_size))
    return minimum, max(capacity * max_batch_size, minimum)


class GeometryError(ValueError):
    """A geometry, broker, row or census input is outside the reviewed contract."""


def _exact(value: object, keys: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise GeometryError(f"{label} keys differ")
    return value


def _sha(value: object, label: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None or value == "0" * 64:
        raise GeometryError(f"{label} is not a nonzero sha256")
    return value


def _int(value: object, label: str, minimum: int = 0, maximum: int = 1 << 63) -> int:
    """Exact JSON integer in range; bool and float are refused."""
    if type(value) is not int or not minimum <= value <= maximum:
        raise GeometryError(f"{label} is not an exact integer in [{minimum}, {maximum}]")
    return value


def strict_decimal(value: object, label: str, minimum: int = 0, maximum: int = 1 << 63) -> int:
    """Strict ASCII decimal text: no sign, float, exponent, whitespace, empty or leading zero."""
    if type(value) is not str or DECIMAL.fullmatch(value) is None:
        raise GeometryError(f"{label} is not a strict ASCII decimal integer")
    return _int(int(value), label, minimum, maximum)


def argv_sha256(argv: list[str]) -> str:
    """sha256 of the NUL-joined argv, the /proc/<pid>/cmdline form."""
    return hashlib.sha256(b"\0".join(item.encode() for item in argv) + b"\0").hexdigest()


def recipe_environment(payload: bytes) -> dict[str, str]:
    """Literal assignments of a recipe, as the live broker reads them."""
    values: dict[str, str] = {}
    for line in payload.decode("utf-8").splitlines():
        tokens = shlex.split(line, comments=True, posix=True)
        if not tokens:
            continue
        if tokens[0] == "export":
            tokens = tokens[1:]
        if len(tokens) != 1 or "=" not in tokens[0] or any(c in tokens[0] for c in "$`"):
            raise GeometryError("recipe must use literal environment assignments")
        key, value = tokens[0].split("=", 1)
        if key in values:
            raise GeometryError("recipe assignment is ambiguous")
        values[key] = value
    return values


def _option_name(item: str) -> str | None:
    return item.split("=", 1)[0] if item.startswith("--") else None


def _reject_duplicate_options(argv: list[str], label: str) -> None:
    """Any option repeated in any spelling (`--x v`, `--x=v`, trailing) is refused."""
    names = [name for name in map(_option_name, argv) if name is not None]
    if len(names) != len(set(names)):
        raise GeometryError(f"{label} repeats an option")
    if any("=" in item for item in argv if item.startswith("--")):
        raise GeometryError(f"{label} uses an --option=value spelling")


def _flag_values(argv: list[str], flag: str) -> list[str]:
    positions = [index for index, item in enumerate(argv) if _option_name(item) == flag]
    if any(index + 1 >= len(argv) or argv[index + 1].startswith("--") for index in positions):
        raise GeometryError(f"{flag} has no value")
    return [argv[index + 1] for index in positions]


def argv_mode(argv: list[str]) -> str:
    """The mode the argv itself selects; anything else is refused."""
    present = [flag for flag in SPECULATION_FLAGS if any(_option_name(item) == flag for item in argv)]
    if not present:
        return "none"
    if set(present) == {"--speculative", "--num-drafts"} and _flag_values(argv, "--num-drafts") == [K5_NUM_DRAFTS]:
        return "native-mtp-k5"
    raise GeometryError("argv selects an unreviewed speculation mode")


def _validate_base_argv(argv: list[str], label: str) -> None:
    """Executable, serve, model and KV dtype are pinned outside the document."""
    if argv[0] != REVIEWED_EXECUTABLE["path"]:
        raise GeometryError(f"{label} executable is not the reviewed release binary")
    if len(argv) < 2 or argv[1] != "serve":
        raise GeometryError(f"{label} does not run 'serve'")
    if _flag_values(argv, "--model-from-path") != [MODEL_PATH] or any(
        _option_name(item) == "--model" for item in argv
    ):
        raise GeometryError(f"{label} model is not the pinned {MODEL_PATH}")
    if _flag_values(argv, "--kv-cache-dtype") != [KV_CACHE_DTYPE]:
        raise GeometryError(f"{label} KV cache dtype is not {KV_CACHE_DTYPE}")


def _validate_effective_argv(argv: object, geometry: Mapping[str, Any], label: str,
                             mode: str) -> list[str]:
    if type(argv) is not list or not argv or any(type(item) is not str or not item or "\0" in item for item in argv):
        raise GeometryError(f"{label} effective argv is malformed")
    _reject_duplicate_options(argv, label)
    _validate_base_argv(argv, label)
    if argv_mode(argv) != mode:
        raise GeometryError(f"{label} argv selects another mode")
    for flag in PINNED_FLAGS:
        if len(_flag_values(argv, flag)) != 1:
            raise GeometryError(f"{label} must pin {flag} exactly once")
    expected = {
        "--max-prefill-tokens": str(geometry["max_prefill_tokens"]),
        "--max-seq-len": str(geometry["capacity_tokens"]),
        "--block-size": str(geometry["block_size"]),
    }
    for flag, value in expected.items():
        if _flag_values(argv, flag) != [value]:
            raise GeometryError(f"{label} {flag} differs from the reviewed geometry")
    if _flag_values(argv, "--scheduling-policy")[0] not in SCHEDULING_POLICIES:
        raise GeometryError(f"{label} scheduling policy is not reviewed")
    strict_decimal(_flag_values(argv, "--tbt-deadline-ms")[0], f"{label} TBT deadline", 0, (1 << 64) - 1)
    if (_flag_values(argv, "--scheduling-policy") + _flag_values(argv, "--tbt-deadline-ms")
            != list(REVIEWED_SCHEDULING)):
        raise GeometryError(f"{label} scheduling differs from the one reviewed policy")
    for flag, value in ENGINE_FLAGS:
        observed = argv.count(flag) if value is None else _flag_values(argv, flag)
        if observed != (1 if value is None else [value]):
            raise GeometryError(f"{label} {flag} differs from the host recipe argv")
    if geometry["capacity_tokens"] > 2048 and argv.count("--qwen4-qsa") != 1:
        raise GeometryError(f"{label} needs --qwen4-qsa for a context above 2048")
    return argv


def verify_recipe_bytes(profile: str, payload: bytes) -> dict[str, str]:
    """The profile's own reviewed recipe bytes, whose WIDE is the profile chunk."""
    if profile not in PROFILE_NAMES:
        raise GeometryError("unknown profile")
    if hashlib.sha256(payload).hexdigest() != REVIEWED_RECIPE_SHA256[profile]:
        raise GeometryError(f"{profile} recipe bytes are not its reviewed recipe")
    environment = recipe_environment(payload)
    reviewed = REVIEWED_GEOMETRY[profile]
    if environment.get("FLUX_MODE") != reviewed["flux_mode"] or any(
        environment.get(selector) != str(reviewed["wide"]) for selector in WIDE_SELECTORS
    ):
        raise GeometryError(f"{profile} recipe WIDE or mode differs from its chunk")
    if MAX_BATCH_TOKENS in environment or "FLUX_MAX_BATCH_TOKENS" in environment:
        raise GeometryError(f"{profile} recipe sets an arena outside the geometry")
    return environment


def _without_chunk(argv: list[str]) -> list[str]:
    index = argv.index("--max-prefill-tokens")
    return argv[:index + 1] + argv[index + 2:]


def validate_geometry_v2(document: object) -> dict[str, Any]:
    """Validate a /2 geometry document and return it unchanged."""
    document = _exact(document, {"schema", "profiles"}, "profile geometry")
    if document["schema"] != GEOMETRY_V2:
        raise GeometryError("profile geometry is not /2")
    profiles = _exact(document["profiles"], set(PROFILE_NAMES), "profile geometry profiles")
    for name in PROFILE_NAMES:
        row = _exact(
            profiles[name],
            set(GEOMETRY_KEYS) | {"recipe_sha256", "max_batch_tokens", "effective_argv"},
            f"{name} geometry",
        )
        for key, expected in REVIEWED_GEOMETRY[name].items():
            if type(row[key]) is not type(expected) or row[key] != expected:
                raise GeometryError(f"{name} {key} differs from the reviewed geometry")
        if not (row["wide"] == row["chunk_tokens"] == row["max_prefill_tokens"]):
            raise GeometryError(f"{name} WIDE, chunk and max-prefill differ")
        if row["capacity_tokens"] % row["chunk_tokens"] or row["capacity_tokens"] % row["block_size"]:
            raise GeometryError(f"{name} capacity is not chunk and block aligned")
        if _sha(row["recipe_sha256"], f"{name} recipe hash") != REVIEWED_RECIPE_SHA256[name]:
            raise GeometryError(f"{name} recipe hash is not its own reviewed recipe")
        if row["max_batch_tokens"] is not None:
            # null means ATLAS_MAX_BATCH_TOKENS must be absent; otherwise an
            # exact arena within the engine's resolved bounds.
            _int(row["max_batch_tokens"], f"{name} max batch tokens",
                 *arena_bounds(row["chunk_tokens"], row["capacity_tokens"]))
        argv = _exact(row["effective_argv"], set(MODES), f"{name} effective argv")
        for mode in MODES:
            _validate_effective_argv(argv[mode], row, f"{name} {mode}", mode)
            if argv[mode] != reviewed_effective_argv(name, mode):
                raise GeometryError(f"{name} {mode} argv differs from the reviewed launch argv")
    first, second = (profiles[name] for name in PROFILE_NAMES)
    if first["recipe_sha256"] == second["recipe_sha256"]:
        raise GeometryError("both profiles bind one recipe")
    for mode in MODES:
        # The base argv, policy and deadline are shared; only the chunk differs.
        if _without_chunk(first["effective_argv"][mode]) != _without_chunk(second["effective_argv"][mode]):
            raise GeometryError(f"{mode} profiles differ outside the scheduler chunk")
    if _document_sha256(document) != REVIEWED_GEOMETRY_SHA256:
        raise GeometryError("profile geometry is not the trusted reviewed geometry")
    return document


def read_geometry(document: object) -> dict[str, Any]:
    """Read /1 (one universal 4096 chunk) or /2; anything else is refused."""
    if type(document) is dict and document.get("schema") == GEOMETRY_V1:
        document = _exact(document, {"schema", "chunk_tokens"}, "profile geometry /1")
        if _int(document["chunk_tokens"], "/1 chunk") != V1_CHUNK_TOKENS:
            raise GeometryError("profile geometry /1 chunk is not 4096")
        return document
    return validate_geometry_v2(document)


def require_dual_profile(document: object) -> dict[str, Any]:
    """Only /2 can describe interactive 4096 beside high-prefill 16384."""
    document = read_geometry(document)
    if document["schema"] != GEOMETRY_V2:
        raise GeometryError("profile geometry /1 assumes a universal 4096 chunk and cannot admit dual profiles")
    return document


def chunk_tokens(document: object, profile: str) -> int:
    """Version dispatch for every site that hard-codes CHUNK_TOKENS=4096."""
    document = read_geometry(document)
    if profile not in PROFILE_NAMES:
        raise GeometryError("unknown profile")
    if document["schema"] == GEOMETRY_V1:
        return V1_CHUNK_TOKENS
    return document["profiles"][profile]["chunk_tokens"]


def check_version_pairing(versions: object) -> str:
    """Return the geometry version all artifact versions agree on."""
    versions = _exact(versions, {"geometry", *VERSION_PAIRINGS[GEOMETRY_V2]}, "artifact versions")
    pairing = VERSION_PAIRINGS.get(versions["geometry"])
    if pairing is None or any(versions[key] != value for key, value in pairing.items()):
        raise GeometryError("artifact versions are a cross pair")
    return versions["geometry"]


def output_budget(context: int) -> int:
    return FINAL_DECODE_BUDGET if context == PROFILE_CONTEXTS[-1] else DECODE_BUDGET


def role_geometry(document: object, profile: str, context: int) -> dict[str, int]:
    """Derived chunk count and partial last chunk of one prompt context."""
    document = require_dual_profile(document)
    if profile not in PROFILE_NAMES:
        raise GeometryError("unknown profile")
    geometry = document["profiles"][profile]
    # `8192.0 in PROFILE_CONTEXTS` is True in Python; require an exact int.
    if type(context) is not int or context not in PROFILE_CONTEXTS:
        raise GeometryError("context is not a reviewed integer role")
    chunk = geometry["chunk_tokens"]
    chunks = math.ceil(context / chunk)
    total = context + output_budget(context)
    if total > geometry["capacity_tokens"]:
        raise GeometryError("role exceeds the profile capacity")
    return {
        "chunk_tokens": chunk,
        "chunks": chunks,
        "last_chunk_rows": context - (chunks - 1) * chunk,
        "capacity_headroom": geometry["capacity_tokens"] - total,
    }


def bind_row(document: object, profile: str, mode: str, context: int, kind: str) -> dict[str, Any]:
    """The binding every v2 host-reference and measured row carries."""
    document = require_dual_profile(document)
    if mode not in MODES or kind not in REQUEST_KINDS:
        raise GeometryError("row mode or kind is not reviewed")
    return {
        "schema": ROW_V2,
        "profile": profile,
        "mode": mode,
        "context_tokens": context,
        "kind": kind,
        "recipe_sha256": document["profiles"][profile]["recipe_sha256"],
        "effective_argv_sha256": argv_sha256(document["profiles"][profile]["effective_argv"][mode]),
        **role_geometry(document, profile, context),
    }


def require_row(document: object, row: object, profile: str) -> dict[str, Any]:
    """Refuse a row that does not carry exactly the required profile's binding.

    A 4096 row can never satisfy a 16384 requirement or the reverse, and a row
    cannot be relabelled: its profile, chunk, recipe and argv all re-derive.
    """
    row = _exact(row, set(bind_row(document, profile, MODES[0], PROFILE_CONTEXTS[0], REQUEST_KINDS[0])), "geometry row")
    if row["profile"] != profile:
        raise GeometryError("row belongs to another profile")
    expected = bind_row(document, profile, row["mode"], row["context_tokens"], row["kind"])
    if json.dumps(row, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise GeometryError("row binding differs from the required profile geometry")
    return row


def recensus(document: object, journal: bytes) -> set[tuple[str, str, int, str]]:
    """Re-census the raw canonical journal: exactly 72 unique bound roles."""
    document = require_dual_profile(document)
    lines = journal.splitlines(keepends=True)
    if len(lines) != CENSUS_RECORDS:
        raise GeometryError("profile journal record census differs")
    seen = set()
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise GeometryError("profile journal record is not JSON") from error
        if type(record) is not dict or json.dumps(record, sort_keys=True, separators=(",", ":")).encode() + b"\n" != line:
            raise GeometryError("profile journal record is not canonical")
        geometry = record.get("geometry")
        role = (record.get("profile"), record.get("mode"), record.get("context_tokens"), record.get("kind"))
        if role[0] not in PROFILE_NAMES or type(role[2]) is not int or role in seen:
            raise GeometryError("profile journal role differs or is duplicated")
        require_row(document, geometry, role[0])
        if (geometry["mode"], geometry["context_tokens"], geometry["kind"]) != role[1:]:
            raise GeometryError("profile journal record is relabelled")
        seen.add(role)
    expected = {(p, m, c, k) for p in PROFILE_NAMES for m in MODES for c in PROFILE_CONTEXTS for k in REQUEST_KINDS}
    if seen != expected:
        raise GeometryError("profile journal does not cover every reviewed role")
    return seen


def validate_host_reference_header(document: object, header: object) -> dict[str, Any]:
    """Host reference /4 keeps /3 clocks and host identity, adding geometry."""
    document = require_dual_profile(document)
    header = _exact(header, {"schema", "decode_clock", "prefill_clock", "geometry_sha256", *HOST_IDENTITY_KEYS}, "host reference header")
    if header["schema"] != VERSION_PAIRINGS[GEOMETRY_V2]["host_reference"]:
        raise GeometryError("host reference is not /4")
    if (header["decode_clock"], header["prefill_clock"]) != (DECODE_CLOCK, PREFILL_CLOCK):
        raise GeometryError("host reference timer differs")
    for key in HOST_IDENTITY_KEYS:
        if type(header[key]) is not str or not header[key]:
            raise GeometryError(f"host identity {key} is missing")
    if header["geometry_sha256"] != geometry_sha256(document):
        raise GeometryError("host reference geometry differs")
    return header


def _document_sha256(document: object) -> str:
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def geometry_sha256(document: object) -> str:
    """The sha of a validated geometry; it is always REVIEWED_GEOMETRY_SHA256."""
    return _document_sha256(require_dual_profile(document))


def broker_contract(document: object) -> dict[str, Any]:
    """The dual-recipe broker /2: exactly the two reviewed recipes, by sha."""
    document = require_dual_profile(document)
    return {
        "schema": BROKER_V2,
        "geometry_sha256": geometry_sha256(document),
        "executable": dict(REVIEWED_EXECUTABLE),
        "model_path": MODEL_PATH,
        "recipes": {
            name: {
                "in_container_recipe_path": IN_CONTAINER_ROOT + row["recipe_relpath"],
                "recipe_sha256": row["recipe_sha256"],
                "wide": row["wide"],
                "max_batch_tokens": row["max_batch_tokens"],
                "effective_argv_sha256": {mode: argv_sha256(row["effective_argv"][mode]) for mode in MODES},
            }
            for name, row in document["profiles"].items()
        },
    }


def admit_live_recipe(document: object, broker: object, request: object, observed: object,
                      *, trusted_model_identity: object) -> dict[str, Any]:
    """Admit one live server against exactly one of the two reviewed recipes.

    NOT standalone engine admission. The caller must observe through /proc
    and pass in: the opened executable's path, sha256 and size; the /models
    directory identity; the recipe path and sha; the exact argv; and the
    environment. `trusted_model_identity` is REQUIRED: the caller's pinned
    checkpoint identity, `{"config_sha256": hex}` or `{"dev": int, "ino": int}`.
    Executable, base argv, KV dtype and geometry are pinned here.
    """
    model_keys = _trusted_model_keys(trusted_model_identity)
    expected_broker = broker_contract(document)
    if type(broker) is not dict or broker.get("schema") != BROKER_V2:
        raise GeometryError("a single-recipe broker cannot admit dual profiles")
    if json.dumps(broker, sort_keys=True) != json.dumps(expected_broker, sort_keys=True):
        raise GeometryError("broker recipes differ from the reviewed pair")
    request = _exact(request, {"schema", "profile", "mode", "host_pid"}, "live process request")
    if request["schema"] != REQUEST_V2 or request["profile"] not in PROFILE_NAMES or request["mode"] not in MODES:
        raise GeometryError("live process request schema, profile or mode differs")
    _int(request["host_pid"], "live host PID", 1, 1 << 31)
    observed = _exact(observed, {"executable", "model", "recipe_path", "recipe_sha256", "argv", "environment"},
                      "live observation")
    if observed["executable"] != REVIEWED_EXECUTABLE or type(observed["executable"]["size"]) is not int:
        raise GeometryError("live executable is not the reviewed release binary")
    model = _exact(observed["model"], {"path", *model_keys}, "live model identity")
    if model != {"path": MODEL_PATH, **trusted_model_identity} or any(
        type(model[key]) is not type(trusted_model_identity[key]) for key in model_keys
    ):
        raise GeometryError("live model is not the trusted checkpoint")
    if type(observed["argv"]) is not list or any(type(item) is not str for item in observed["argv"]):
        raise GeometryError("live argv is malformed")
    _reject_duplicate_options(observed["argv"], "live argv")
    _validate_base_argv(observed["argv"], "live argv")
    if len(observed["argv"]) <= PORT_INDEX or MODEL_ALIAS.fullmatch(observed["argv"][MODEL_INDEX]) is None:
        raise GeometryError("live model alias is malformed")
    port = strict_decimal(observed["argv"][PORT_INDEX], "live port", 1, 65535)
    template = list(observed["argv"])
    template[MODEL_INDEX], template[PORT_INDEX] = MODEL_PLACEHOLDER, PORT_PLACEHOLDER
    if argv_mode(observed["argv"]) != request["mode"]:
        raise GeometryError("live argv selects another mode than the request")
    recipe = broker["recipes"][request["profile"]]
    known = {row["recipe_sha256"]: name for name, row in broker["recipes"].items()}
    if observed["recipe_sha256"] not in known:
        raise GeometryError("live recipe is not one of the two reviewed recipes")
    if known[observed["recipe_sha256"]] != request["profile"] or observed["recipe_path"] != recipe["in_container_recipe_path"]:
        raise GeometryError("live recipe belongs to another profile")
    if argv_sha256(template) != recipe["effective_argv_sha256"][request["mode"]]:
        raise GeometryError("live argv differs from the bound effective argv")
    environment = observed["environment"]
    if type(environment) is not dict:
        raise GeometryError("live environment is malformed")
    for selector in WIDE_SELECTORS:
        if environment.get(selector) != str(recipe["wide"]):
            raise GeometryError("live WIDE differs from the profile geometry")
    if any(type(key) is not str or type(value) is not str for key, value in environment.items()):
        raise GeometryError("live environment is malformed")
    # The engine fills an absent ATLAS_ key from its FLUX_ alias at startup.
    disabled = environment.get(DISABLE_SPECULATION, environment.get("FLUX_DISABLE_SPECULATION"))
    if (request["mode"] == "none") != (disabled == "1") or disabled not in (None, "0", "1"):
        raise GeometryError("live speculation kill switch differs from the mode")
    bound = recipe["max_batch_tokens"]
    for key in (MAX_BATCH_TOKENS, "FLUX_MAX_BATCH_TOKENS"):
        if key not in environment:
            continue
        # Present means bound: an empty, signed, float or other value is refused.
        if bound is None or key != MAX_BATCH_TOKENS:
            raise GeometryError("live arena capacity is set but not bound")
        if strict_decimal(environment[key], "live arena capacity", 1, MAX_ARENA_TOKENS) != bound:
            raise GeometryError("live arena capacity is not the bound value")
    if bound is not None and MAX_BATCH_TOKENS not in environment:
        raise GeometryError("live arena capacity is bound but absent")
    return {"profile": request["profile"], "mode": request["mode"], "recipe_sha256": observed["recipe_sha256"],
            "model_alias": observed["argv"][MODEL_INDEX], "port": port}


def _trusted_model_keys(identity: object) -> set[str]:
    """The caller's pinned checkpoint identity; there is no default."""
    if type(identity) is dict and set(identity) == {"config_sha256"}:
        _sha(identity["config_sha256"], "trusted model config hash")
        return {"config_sha256"}
    if type(identity) is dict and set(identity) == {"dev", "ino"}:
        _int(identity["dev"], "trusted model device", 1)
        _int(identity["ino"], "trusted model inode", 1)
        return {"dev", "ino"}
    raise GeometryError("a trusted model identity is required")


# --------------------------------------------------------------------------- packet v4
# Sealed per-profile recipes, live startup proofs, the K5 tail capacity proof
# and the host argv binding. Pure functions; the callers observe and pass in.

RECIPE_V1 = "flux-profile-recipe/1"
SEALED_RECIPE_RELPATHS = {
    "interactive": "repro-kit/recipes/flux-profile-interactive-w4096.json",
    "high-prefill": "repro-kit/recipes/flux-profile-high-prefill-w16384.json",
}
# Image ENV (repro-kit/Dockerfile ENV block, FLUX_ENGINE_VERSION at its ARG
# default). docker exec inherits it, and main.rs mirrors each FLUX_ key to an
# absent ATLAS_ alias, so the host producer must start from the same values.
IMAGE_ENVIRONMENT = {
    "DEBIAN_FRONTEND": "noninteractive",
    "PYTHONUNBUFFERED": "1",
    # 753bc22b9 bundles FlashInfer into the image.
    "FLASHINFER_HOME": "/opt/flux/fi_home",
    "FLUX_ENGINE_VERSION": "1.0.3",
    "FLUX_TARGET": "qwen3.8-flash-next",
    "FLUX_MODE": "auto",
    "FLUX_HOST": "0.0.0.0",
    "FLUX_PORT": "8899",
    "FLUX_CONCURRENCY": "1",
    "FLUX_MAX_SEQ_LEN": "131072",
    "FLUX_PRIVACY_FIRST": "1",
    "FLUX_OPT_OUT": "all",
    "DO_NOT_TRACK": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "CUDA_MODULE_LOADING": "LAZY",
    "NCCL_P2P_DISABLE": "0",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "LIL_BENCH_DIR": "/opt/flux/llm-inference-bench",
    "LLM_BENCH_NO_UPDATE_CHECK": "1",
}
# Selectors whose host and container values must agree; transport-only keys
# such as PATH, HOME and LD_LIBRARY_PATH may differ by host layout.
ENGINE_ENV_PREFIXES = ("ATLAS_", "FLUX_", "CUDA_", "NCCL_", "RUST_LOG", "PYTORCH_")
# preflight.rs:283-287: ssm_prefill_chunk = min(max_seq_len, 8192) on Qwen4.
SSM_PREFILL_CHUNK_CAP = 8192
PREFILL_CONFIG = re.compile(
    r"Prefill config: ssm_prefill_chunk=(0|[1-9][0-9]*), args\.max_prefill_tokens=(0|[1-9][0-9]*), "
    r"prefill_budget=(0|[1-9][0-9]*), max_batch_tokens=(0|[1-9][0-9]*)")
KV_POOL = re.compile(
    r"KV cache \(post-construction\): [0-9]+\.[0-9] GB free, [0-9]+\.[0-9] GB allocatable, "
    r"(0|[1-9][0-9]*) blocks × (0|[1-9][0-9]*) tok/block = (0|[1-9][0-9]*) max tokens")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
HOST_BIND = "127.0.0.1"
EXECUTABLE_INDEX, MODEL_PATH_INDEX, BIND_INDEX = 0, 4, 8


def spec_rows(mode: str) -> int:
    """kv_cache.rs:131: drafts + 2 rows for speculation, else one."""
    return int(K5_NUM_DRAFTS) + 2 if mode == "native-mtp-k5" else 1


def expected_prefill_config(profile: str, mode: str) -> dict[str, int]:
    """The one 'Prefill config:' line this profile's engine must log.

    kv_cache.rs:132-134 keeps a non-default --max-prefill-tokens as the budget
    (no prefix caching or high-speed swap here), and :223-232 resolves the
    arena as max(budget + max_batch_size, spec rows, padded decode rows) when
    ATLAS_MAX_BATCH_TOKENS is absent, which the recipe and broker require.
    """
    if profile not in PROFILE_NAMES or mode not in MODES:
        raise GeometryError("unknown profile or mode")
    geometry = REVIEWED_GEOMETRY[profile]
    chunk = geometry["chunk_tokens"]
    return {
        "ssm_prefill_chunk": min(geometry["capacity_tokens"], SSM_PREFILL_CHUNK_CAP),
        "args.max_prefill_tokens": chunk,
        "prefill_budget": chunk,
        "max_batch_tokens": max(chunk + MAX_BATCH_SIZE, spec_rows(mode), padded_decode_rows(MAX_BATCH_SIZE)),
    }


def expected_kv_pool(profile: str) -> dict[str, int]:
    """build.rs:1020-1024: the pool is capped at batch x (blocks + 1 spare) + 1
    dummy; a smaller budget-derived pool still starts (:1063-1080) but loses
    the spare block the K5 tail needs, so the live census requires the cap."""
    geometry = REVIEWED_GEOMETRY[profile]
    blocks = MAX_BATCH_SIZE * (geometry["capacity_tokens"] // geometry["block_size"] + 1) + 1
    return {"blocks": blocks, "block_size": geometry["block_size"],
            "max_tokens": blocks * geometry["block_size"]}


def _log_lines(log: object, label: str) -> list[str]:
    if type(log) is not bytes:
        raise GeometryError(f"{label} must be raw bytes")
    try:
        text = log.decode("utf-8")
    except UnicodeDecodeError as error:
        raise GeometryError(f"{label} is not UTF-8") from error
    return [ANSI.sub("", line) for line in text.splitlines()]


def startup_census(log: object, arms: object) -> list[dict[str, Any]]:
    """Bind one live startup per (profile, mode) arm, in launch order.

    The engine log must contain exactly one 'Prefill config:' line and one
    'KV cache (post-construction):' line per arm, each equal to the values the
    sealed recipe derives; a missing, extra, reordered or drifted line fails.
    """
    if type(arms) is not list or not arms:
        raise GeometryError("startup census needs the launched arms")
    lines = _log_lines(log, "engine log")
    prefill = [PREFILL_CONFIG.search(line) for line in lines]
    prefill = [match for match in prefill if match is not None]
    pools = [KV_POOL.search(line) for line in lines]
    pools = [match for match in pools if match is not None]
    if sum("Prefill config:" in line for line in lines) != len(prefill):
        raise GeometryError("a Prefill config line is malformed")
    if sum("KV cache (post-construction):" in line for line in lines) != len(pools):
        raise GeometryError("a KV pool line is malformed")
    if len(prefill) != len(arms) or len(pools) != len(arms):
        raise GeometryError("engine startup line census differs from the launched arms")
    census = []
    for arm, config, pool in zip(arms, prefill, pools):
        if type(arm) is not tuple or len(arm) != 2:
            raise GeometryError("startup arm is not a (profile, mode) pair")
        profile, mode = arm
        expected = expected_prefill_config(profile, mode)
        observed = dict(zip(expected, (int(value) for value in config.groups())))
        if observed != expected:
            raise GeometryError(f"{profile} {mode} Prefill config differs from the sealed recipe")
        kv = expected_kv_pool(profile)
        observed_pool = dict(zip(kv, (int(value) for value in pool.groups())))
        if observed_pool != kv:
            raise GeometryError(f"{profile} {mode} KV pool is not the addressable cap with its spare block")
        census.append({"profile": profile, "mode": mode, "prefill_config": observed, "kv_pool": observed_pool})
    return census


def k5_tail_proof(document: object, profile: str, context: int) -> dict[str, int]:
    """Capacity proof for the final K5 verify frames of one role.

    Admission is inclusive (api/context_budget.rs:10-12), so 262080 + 64 =
    262144 is admitted, and the scheduler does not shorten a frame for
    capacity. A frame starts while fewer than `budget` tokens were emitted and
    writes the last token plus the drafts, inside an arena of drafts + 2 rows.
    Every row is therefore below context + budget + spec rows, which must fit
    in the addressable rows: the block stride is max_seq_len / bs + 1
    (impl_a1.rs:574), the pool's dummy takes one block (impl_a1.rs:582) of the
    capped pool (build.rs:1020-1024), and the MTP cache has the same
    blocks + 1 (qwen4_mtp.rs:294). The live pool census must show that cap.
    """
    role = role_geometry(document, profile, context)
    geometry = require_dual_profile(document)["profiles"][profile]
    pool = expected_kv_pool(profile)
    addressable = (pool["blocks"] - 1) * geometry["block_size"]
    stride = (geometry["capacity_tokens"] // geometry["block_size"] + 1) * geometry["block_size"]
    needed = context + output_budget(context) + spec_rows("native-mtp-k5")
    headroom = min(addressable, stride) - needed
    if headroom < 0:
        raise GeometryError(f"{profile} {context} K5 tail exceeds the addressable KV rows")
    return {**role, "k5_rows_needed": needed, "addressable_rows": min(addressable, stride),
            "k5_tail_headroom_rows": headroom}


def canonical_json(value: object) -> bytes:
    """The release tools' canonical JSON (r15-container-suite.canonical_bytes)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def launch_argv_sha256(argv: list[str]) -> str:
    """Contract /2's launch_argv_sha256: sha of the canonical JSON template."""
    return hashlib.sha256(canonical_json(argv)).hexdigest()


def profile_environment(profile: str, mode: str, recipe: bytes) -> dict[str, str]:
    """The exact engine environment of one arm: image ENV, recipe, kill switch.

    The gate starts from the recipe and sets the kill switch per mode
    (prefill-decode-profiles server()); docker exec adds the image ENV.
    """
    environment = dict(IMAGE_ENVIRONMENT)
    environment.update(verify_recipe_bytes(profile, recipe))
    if mode not in MODES:
        raise GeometryError("unknown mode")
    environment[DISABLE_SPECULATION] = "1" if mode == "none" else "0"
    return environment


def engine_environment(environment: Mapping[str, str]) -> dict[str, str]:
    return {key: value for key, value in environment.items() if key.startswith(ENGINE_ENV_PREFIXES)}


def profile_recipe_document(profile: str) -> dict[str, Any]:
    """The sealed recipe of one profile: everything a launch must reproduce."""
    document = reviewed_geometry_v2()
    geometry = document["profiles"][profile]
    argv = {mode: reviewed_effective_argv(profile, mode) for mode in MODES}
    return {
        "schema": RECIPE_V1,
        "profile": profile,
        "geometry_schema": GEOMETRY_V2,
        "geometry_sha256": REVIEWED_GEOMETRY_SHA256,
        "versions": dict(VERSION_PAIRINGS[GEOMETRY_V2], geometry=GEOMETRY_V2),
        "environment_recipe": {
            "relpath": geometry["recipe_relpath"],
            "in_container_path": IN_CONTAINER_ROOT + geometry["recipe_relpath"],
            "sha256": geometry["recipe_sha256"],
            "flux_mode": geometry["flux_mode"],
            "wide": geometry["wide"],
        },
        "geometry": {key: geometry[key] for key in
                     ("chunk_tokens", "max_prefill_tokens", "block_size", "capacity_tokens")},
        "scheduling": {"policy": REVIEWED_SCHEDULING[0], "tbt_deadline_ms": int(REVIEWED_SCHEDULING[1])},
        "argv_overlay": reviewed_overlay(profile),
        "effective_argv": argv,
        "launch_argv_sha256": {mode: launch_argv_sha256(argv[mode]) for mode in MODES},
        "cmdline_sha256": {mode: argv_sha256(argv[mode]) for mode in MODES},
        "host_substitutions": {
            "executable": {"index": EXECUTABLE_INDEX, "container": REVIEWED_EXECUTABLE["path"]},
            "model_path": {"index": MODEL_PATH_INDEX, "container": MODEL_PATH},
            "model_alias": {"index": MODEL_INDEX, "template": MODEL_PLACEHOLDER},
            "bind": {"index": BIND_INDEX, "container": "0.0.0.0", "host": HOST_BIND},
            "port": {"index": PORT_INDEX, "template": PORT_PLACEHOLDER},
        },
        "environment": {
            "image": dict(IMAGE_ENVIRONMENT),
            "absent": [MAX_BATCH_TOKENS, "FLUX_MAX_BATCH_TOKENS", "RUST_LOG"],
            "kill_switch": {DISABLE_SPECULATION: {"none": "1", "native-mtp-k5": "0"}},
        },
        "live_proof": {
            "prefill_config": {mode: expected_prefill_config(profile, mode) for mode in MODES},
            "kv_pool": expected_kv_pool(profile),
        },
        "roles": [k5_tail_proof(document, profile, context) for context in PROFILE_CONTEXTS],
        "output_budget": {str(context): output_budget(context) for context in PROFILE_CONTEXTS},
    }


def sealed_recipe_bytes(profile: str) -> bytes:
    return canonical_json(profile_recipe_document(profile)) + b"\n"


def normalize_host_argv(observed: object, profile: str, mode: str) -> list[str]:
    """Map a host server's /proc cmdline onto the container template.

    Only the executable path, model path, model alias, bind (127.0.0.1) and
    port may differ; everything else must already equal the sealed argv.
    """
    expected = reviewed_effective_argv(profile, mode)
    if (type(observed) is not list or len(observed) != len(expected)
            or any(type(item) is not str or not item or "\0" in item for item in observed)):
        raise GeometryError("host argv is malformed")
    if MODEL_ALIAS.fullmatch(observed[MODEL_INDEX]) is None or observed[BIND_INDEX] != HOST_BIND:
        raise GeometryError("host model alias or bind differs from the host substitution")
    strict_decimal(observed[PORT_INDEX], "host port", 1, 65535)
    if not observed[EXECUTABLE_INDEX].startswith("/") or not observed[MODEL_PATH_INDEX].startswith("/"):
        raise GeometryError("host executable and model must be absolute paths")
    template = list(observed)
    template[EXECUTABLE_INDEX] = REVIEWED_EXECUTABLE["path"]
    template[MODEL_PATH_INDEX] = MODEL_PATH
    template[MODEL_INDEX], template[PORT_INDEX] = MODEL_PLACEHOLDER, PORT_PLACEHOLDER
    template[BIND_INDEX] = "0.0.0.0"
    if template != expected:
        raise GeometryError("host argv differs from the sealed recipe outside the host substitutions")
    return template
