# SPDX-License-Identifier: AGPL-3.0-only
"""Pinned independent authority for both required native-context recipes.

Loaded from hash-checked source by the outer wrapper. Source preparation does
not award runtime, context, accuracy, or release credit.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import sys
from types import MappingProxyType
from typing import Any, Mapping

_KIT_DIR = str(Path(__file__).resolve().parent)
if _KIT_DIR not in sys.path:
    sys.path.insert(0, _KIT_DIR)

# The sealed wrapper injects its own release module before execution.

NATIVE262K_W4096_OWNER16_GATE = "native-262k-w4096-owner16"
NATIVE_CONTEXT_REQUIRED_GATES = ("native-262k", NATIVE262K_W4096_OWNER16_GATE)
NATIVE_CONTEXT_REQUIRED_GATE_AUTHORITY = MappingProxyType(
    {
        NATIVE262K_W4096_OWNER16_GATE: MappingProxyType(
            {
                "authority_schema": "flux-r16-required-gate-authority/1",
                "job_schema": "flux-r16-native262k-w4096-owner16-job/1",
                "evidence_schema": "flux-r16-native262k-w4096-owner16-evidence/1",
                "registration": "required",
                "control_gate": "native-262k",
                "upstream_packet_manifest_sha256": (
                    "2828a1a6ba6ca6d70337ed9ce63e0089e8ae938677a48ec8b2e08a99022ad1bd"
                ),
                "program": (
                    "repro-kit/r15-gates/native-262k-w4096-owner16",
                    "dc6ab71e3ee4aaaface2d52bb27f2b73356b51a2fee1614c5b73d613085f26f8",
                ),
                "recipe": (
                    "repro-kit/recipes/flux-qwen4-native-262k-w4096-bf16-owner16.env",
                    "28c21eb3d42f7b643e0a556129a23cd0a5d6d7c6f61ab78ce318cbc0252d4718",
                ),
                "helpers": (
                    (
                        "--lifecycle-program",
                        "repro-kit/r15-gates/mmlu-pro",
                        "dfae597a00c668ee5e3b2f02042c39df6035fa48cbb887cea53a3cb50a41b032",
                    ),
                    (
                        "--endpoint-program",
                        "repro-kit/qwen4_endpoint_admission.py",
                        "81d336c6bc477b6efafeb100e8e72b78c8e8b7cffc54c8880a77227c3fa84832",
                    ),
                    (
                        "--engagement-program",
                        "repro-kit/qwen4_owner16_engagement.py",
                        "31145e6a6eed07030e42ab349cdd0bf81d236fcce5071eccc60020547d754403",
                    ),
                ),
                "tokenizer_sha256": (
                    "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
                ),
            }
        )
    }
)
assert set(NATIVE_CONTEXT_REQUIRED_GATES) <= set(release.REQUIRED_GATES)
# Both authorities are constructed below before validators run.
assert set(NATIVE_CONTEXT_REQUIRED_GATES) <= set(release.REQUIRED_GATES)

JOB_KEYS = frozenset(
    {
        "program_path",
        "program_sha256",
        "argv",
        "timeout_seconds",
        "receipt_relpath",
        "stdout_relpath",
        "stderr_relpath",
    }
)
# Identity options of the frozen driver take exactly these suite placeholders.
IDENTITY_OPTIONS = (
    ("--container-id", "@CONTAINER_ID@"),
    ("--image-id", "@IMAGE_ID@"),
    ("--image-digest", "@IMAGE_DIGEST@"),
    ("--base-url", "@BASE_URL@"),
    ("--run-id", "@RUN_ID@"),
    ("--receipt-fd", "@GATE_RECEIPT_FD@"),
    ("--raw-log-fd", "@RAW_LOG_FD@"),
)
SUITE_RELPATH = "repro-kit/r15-container-suite.py"
# Exact successor live controls (both aliases must agree in the driver) and
# exact owner16 census output of the pinned public helper.
NATIVE262K_W4096_OWNER16_CONTROLS = MappingProxyType(
    {
        "QWEN4_PREFILL_WIDE": "4096",
        "QWEN4_PREFILL_MULTI_CHUNK": "1",
        "QWEN4_PREFILL_QSA": "1",
        "QWEN4_PREFILL_QSA_FP8": "0",
        "CTX_OVERFLOW_TRUNCATE": "0",
        "DISABLE_SPECULATION": "1",
        "FAST_LOAD_ROUTED_GROUPS": "1",
        "FAST_LOAD_ROUTED_GROUPS_EAGER": "1",
        "QWEN4_BOUNDED_TRANSPOSE_OWNERS": "1",
        "QWEN4_BOUNDED_TRANSPOSE_OWNER_CAP_MIB": "16",
    }
)
NATIVE262K_W4096_OWNER16_CENSUS = MappingProxyType(
    {
        "schema": "r16.bounded-transpose.engagement.v1",
        "status": "PASS",
        "target_receipts": 48,
        "destination_owners": 4176,
        "destination_payload_bytes": 67_947_724_800,
        "destination_requested_bytes": 67_948_793_856,
        "receipt_fields": MappingProxyType(
            {
                "schema": "atlas.moe-bounded-transpose-owners.v1",
                "selector": "ATLAS_QWEN4_BOUNDED_TRANSPOSE_OWNERS",
                "owner_count": "87",
                "owner_cap_bytes": "16777216",
                "payload_bytes": "1415577600",
                "requested_bytes": "1415599872",
                "guard_and_alignment_bytes": "22272",
                "experts": "512",
                "projections": "3",
                "managed": "false",
                "base_unexposed": "true",
                "down_allocation_after_phase_b": "true",
            }
        ),
        "legacy_two_slab_receipts": 0,
        "phases": (
            "group-plan",
            "eager-source-owners",
            "bounded-host-copy",
            "unified-pass-start",
            "48-after-Phase-D-target-receipts",
            "unified-pass-done",
            "pass-memory-receipt",
        ),
        "phase_proof_scope": (
            "source-reviewed after-Phase-D receipts, not per-call driver proof"
        ),
        "runtime_qualified": False,
        "release_qualified": False,
    }
)
NATIVE262K_W4096_OWNER16_EVIDENCE_KEYS = frozenset(
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
        "prefill_chunk_tokens",
        "destination_owner_cap_mib",
        "source_group_cap_mib",
        "owner16_engagement_exact",
        "engagement_helper_sha256",
        "validator_artifact",
        "validator_receipt_sha256",
    }
)
NATIVE262K_W4096_OWNER16_CANARY = MappingProxyType(
    {
        "atlas_prefill_progress_chunks_total": 64,
        "atlas_prefill_progress_tokens_total": 262_080,
    }
)


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def required_authority(name: object) -> Mapping[str, Any]:
    if type(name) is not str or name not in NATIVE_CONTEXT_REQUIRED_GATES:
        raise ValueError(f"{name!r} is not a registered required gate")
    return NATIVE_CONTEXT_REQUIRED_GATE_AUTHORITY[name]


def _argv_options(argv: object, label: str) -> dict[str, str]:
    """Parse ``program --option value ...`` with no repeats or bare flags."""
    if type(argv) is not list or not argv or len(argv) % 2 == 0:
        raise ValueError(f"{label} argv must be one program plus option pairs")
    options: dict[str, str] = {}
    for key, item in zip(argv[1::2], argv[2::2]):
        if (
            type(key) is not str
            or re.fullmatch(r"--[a-z][a-z0-9-]*", key) is None
            or type(item) is not str
            or not item
            or "\x00" in item
            or item.startswith("--")
            or key in options
        ):
            raise ValueError(f"{label} argv option is malformed or repeated")
        options[key] = item
    return options


def validate_required_gate_job(
    name: str, job: object, *, source_root: Path, runtime_dir: str
) -> dict[str, Any]:
    """Validate one required job definition against its own authority only."""
    authority = required_authority(name)
    label = f"required gate {name}"
    root = release._normalized_absolute_path(str(source_root), f"{label} source root")
    runtime = release._normalized_absolute_path(runtime_dir, f"{label} runtime_dir")
    value = release._exact_object(job, set(JOB_KEYS), label)
    program_relpath, program_sha256 = authority["program"]
    program = str(root / program_relpath)
    if value["program_path"] != program or value["program_sha256"] != program_sha256:
        raise ValueError(f"{label} program differs from its pinned authority")
    argv = value["argv"]
    options = _argv_options(argv, label)
    if argv[0] != program:
        raise ValueError(f"{label} argv must start with its exact program path")
    expected = dict(IDENTITY_OPTIONS)
    expected["--runtime-dir"] = str(runtime)
    expected["--suite-program"] = str(root / SUITE_RELPATH)
    for option, relpath, _digest in authority["helpers"]:
        expected[option] = str(root / relpath)
    required = {*expected, "--suite-sha256"}
    if not required <= set(options) <= required | {"--container-port"}:
        raise ValueError(f"{label} argv options differ from the pinned driver")
    for option, wanted in expected.items():
        if options[option] != wanted:
            raise ValueError(f"{label} {option} differs from its pinned authority")
    release._nonzero_hex(options["--suite-sha256"], 64, f"{label} --suite-sha256")
    port = options.get("--container-port", "8888")
    if re.fullmatch(r"[1-9][0-9]{0,4}", port) is None or int(port) > 65535:
        raise ValueError(f"{label} --container-port differs")
    release._plain_int(value["timeout_seconds"], f"{label}.timeout_seconds", 1, 86_400)
    locators = {
        release._safe_artifact_locator(value[key], f"{label}.{key}")
        for key in ("receipt_relpath", "stdout_relpath", "stderr_relpath")
    }
    if len(locators) != 3:
        raise ValueError(f"{label} artifact locators must be unique")
    return {**value, "argv": list(argv)}


def _proven_entries(provenance: dict[str, Any]) -> dict[str, dict[str, Any]]:
    descriptor, raw = release._open_stable(
        Path(provenance["source_bundle_manifest_path"]),
        "proven source inventory",
        64 * 1024 * 1024,
    )
    os.close(descriptor)
    if release.sha256_bytes(raw) != provenance["source_bundle_manifest_sha256"]:
        raise ValueError("proven source inventory hash differs")
    inventory = release.strict_json_loads(raw, "proven source inventory")
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
    return entries


def validate_required_program_closure(
    name: str,
    job: object,
    provenance: dict[str, Any],
    suite: dict[str, Any],
    plan: dict[str, Any],
    source_root: Path,
) -> dict[str, Any]:
    """Bind one required job to its pinned program, helpers, recipe and suite."""
    authority = required_authority(name)
    label = f"required gate {name}"
    normalized = validate_required_gate_job(
        name, job, source_root=source_root, runtime_dir=plan.get("runtime_dir")
    )
    options = _argv_options(normalized["argv"], label)
    if (
        options["--suite-program"] != suite["program_path"]
        or options["--suite-sha256"] != suite["program_sha256"]
    ):
        raise ValueError(f"{label} parent suite differs from the sealed orchestrator")
    _recipe_relpath, recipe_sha256 = authority["recipe"]
    # The frozen driver requires the image recipe to be its own complete recipe.
    contract = release.validate_native_context_contract(plan.get("native_context_contract"))
    if (plan.get("schema") != "flux-r15-container-suite-plan/4"
            or contract["gates"][name]["recipe_sha256"] != recipe_sha256):
        raise ValueError(f"{label} requires its own sealed native-context recipe")
    root = release._normalized_absolute_path(str(source_root), f"{label} source root")
    entries = _proven_entries(provenance)
    closure = [(*authority["program"], (0o755,))]
    closure.extend(
        (relpath, digest, (0o644, 0o755))
        for _option, relpath, digest in authority["helpers"]
    )
    closure.append((*authority["recipe"], (0o644,)))
    bound = []
    for relative, digest, modes in closure:
        entry = entries.get(f"{release.provenance_gate.ENGINE_PREFIX}/{relative}")
        if (
            not entry
            or entry.get("role") != "engine-source"
            or entry.get("mode") not in modes
            or entry.get("sha256") != digest
        ):
            raise ValueError(f"{label} {relative} is not a proven tracked source")
        descriptor, payload = release._open_stable(
            root / relative, f"{label} source", release.MAX_IMAGE_RECIPE_BYTES
        )
        os.close(descriptor)
        if len(payload) != entry.get("bytes") or release.sha256_bytes(payload) != digest:
            raise ValueError(f"{label} {relative} differs from proven inventory")
        bound.append({"relpath": relative, "sha256": digest, "bytes": len(payload)})
    return {
        "gate": name,
        "registration": authority["registration"],
        "authority_schema": authority["authority_schema"],
        "job_schema": authority["job_schema"],
        "upstream_packet_manifest_sha256": authority["upstream_packet_manifest_sha256"],
        "closure": bound,
    }


def _owner16_census(value: object, label: str) -> dict[str, Any]:
    expected = _plain(NATIVE262K_W4096_OWNER16_CENSUS)
    census = release._exact_object(
        value, {*expected, "raw_bytes", "raw_sha256"}, f"{label} owner16 census"
    )
    observed = {key: census[key] for key in expected}
    if release.canonical_bytes(observed) != release.canonical_bytes(expected):
        raise ValueError(f"{label} owner16 geometry differs")
    release._plain_int(census["raw_bytes"], f"{label}.raw_bytes", 1, 32 * 1024 * 1024)
    release._nonzero_hex(census["raw_sha256"], 64, f"{label}.raw_sha256")
    return census


def _recompute_native_262k_w4096_owner16_evidence(
    evidence: dict[str, Any],
    raw: dict[str, Any],
    manifest: dict[str, Any],
    container_id: str,
    label: str,
) -> None:
    authority = NATIVE_CONTEXT_REQUIRED_GATE_AUTHORITY[NATIVE262K_W4096_OWNER16_GATE]
    helpers = {option: digest for option, _relpath, digest in authority["helpers"]}
    if (
        release._raw_int(raw, "prefill_chunk_tokens", label, 1, 1_000_000)
        != evidence["prefill_chunk_tokens"]
        or raw.get("lifecycle_sha256") != helpers["--lifecycle-program"]
        or raw.get("endpoint_helper_sha256") != helpers["--endpoint-program"]
    ):
        raise ValueError(f"{label} validator raw authority differs")
    engagement = release._exact_object(
        raw.get("owner16_engagement"),
        {"helper_sha256", "before_generation", "after_cleanup"},
        f"{label} validator raw owner16_engagement",
    )
    if engagement["helper_sha256"] != evidence["engagement_helper_sha256"]:
        raise ValueError(f"{label} engagement helper differs from the validator raw")
    before = _owner16_census(engagement["before_generation"], f"{label} before")
    after = _owner16_census(engagement["after_cleanup"], f"{label} after")
    if after["raw_bytes"] < before["raw_bytes"]:
        raise ValueError(f"{label} owner16 census streams shrank")
    controls = release._exact_object(
        raw.get("live_controls"),
        {"tokenizer_sha256", "controls", "argv_sha256", "environment_sha256"},
        f"{label} validator raw live_controls",
    )
    if (
        release.canonical_bytes(controls["controls"])
        != release.canonical_bytes(_plain(NATIVE262K_W4096_OWNER16_CONTROLS))
        or controls["tokenizer_sha256"] != authority["tokenizer_sha256"]
    ):
        raise ValueError(f"{label} live W4096/owner16 controls differ")
    for key in ("argv_sha256", "environment_sha256"):
        release._nonzero_hex(controls[key], 64, f"{label} live_controls.{key}")
    proofs = release._raw_list(raw, "process_bindings", label, 16)
    if not proofs or any(
        type(proof) is not dict
        or proof.get("recipe_sha256") != authority["recipe"][1]
        or proof.get("container_id") != container_id
        or proof.get("run_id") != manifest["launch"]["run_id"]
        or proof.get("recipe_environment_verified") is not True
        for proof in proofs
    ):
        raise ValueError(f"{label} process bindings differ from the successor recipe")
    recovered = release._raw_object(raw, "post_rejection_metrics", label)
    canary = _plain(NATIVE262K_W4096_OWNER16_CANARY)
    if (
        recovered.get("expected_metric_deltas") != canary
        or recovered.get("metric_deltas") != canary
        or recovered.get("completion_canary_prefill_accounted") is not True
    ):
        raise ValueError(f"{label} post-rejection canary is not 64 W4096 chunks")


def _validate_native_262k_w4096_owner16_evidence(
    value: object, manifest: dict[str, Any], container_id: str
) -> dict[str, Any]:
    name = NATIVE262K_W4096_OWNER16_GATE
    authority = NATIVE_CONTEXT_REQUIRED_GATE_AUTHORITY[name]
    label = f"{name} evidence"
    evidence = release._exact_object(
        value, set(NATIVE262K_W4096_OWNER16_EVIDENCE_KEYS), label
    )
    helpers = {option: digest for option, _relpath, digest in authority["helpers"]}
    expected = {
        "kv_dtype": "bf16",
        "max_seq_len": 262_144,
        "retrieval_prompt_tokens": [258_000, 261_888, 262_080],
        "boundary_prompt_tokens": 262_143,
        "overflow_prompt_tokens": 262_144,
        "passkeys_per_prompt": 3,
        "prefill_chunk_tokens": 4096,
        "destination_owner_cap_mib": 16,
        "source_group_cap_mib": 64,
        "engagement_helper_sha256": helpers["--engagement-program"],
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
        "owner16_engagement_exact",
    ):
        release._exact_true(evidence[key], f"{label}.{key}")
    release._validate_native_context_image_binding(manifest, name, authority["recipe"][1])
    raw = release._validator_document(evidence, name, manifest, container_id)
    release._recompute_native_262k_evidence(evidence, raw, label)
    validate_native_context_process_bindings(raw, manifest, container_id, name)
    _recompute_native_262k_w4096_owner16_evidence(
        evidence, raw, manifest, container_id, label
    )
    return evidence


def validate_required_gate_evidence(
    name: str, value: object, manifest: dict[str, Any], container_id: str
) -> dict[str, Any]:
    required_authority(name)
    if name == NATIVE262K_W4096_OWNER16_GATE:
        return _validate_native_262k_w4096_owner16_evidence(
            value, manifest, container_id
        )
    raise ValueError(f"unsupported required gate {name!r}")


def validate_required_gate_artifact(
    name: str,
    locator: str,
    expected_sha256: str,
    manifest: dict[str, Any],
    container_id: str,
) -> dict[str, Any]:
    """Validate one required gate receipt; full release still requires every gate."""
    authority = required_authority(name)
    label = f"required gate {name} receipt"
    raw = release._stable_beneath_bytes(
        Path(manifest["_runtime_dir"]),
        release._safe_artifact_locator(locator, label),
        label,
        release.MAX_GATE_RECEIPT_BYTES,
    )
    observed_sha256 = release.sha256_bytes(raw)
    if observed_sha256 != release._nonzero_hex(expected_sha256, 64, label):
        raise ValueError(f"{label} SHA-256 mismatch")
    document = release._exact_object(
        release.strict_json_loads(raw, label),
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
        label,
    )
    expected = {
        "schema": release.GATE_SCHEMA,
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
            raise ValueError(f"{label} {key} differs")
    evidence = validate_required_gate_evidence(
        name, document["evidence"], manifest, container_id
    )
    return {
        "status": "PASS",
        "registration": authority["registration"],
        "artifact": locator,
        "receipt_sha256": observed_sha256,
        "receipt_bytes": len(raw),
        "evidence_sha256": release.sha256_bytes(release.canonical_bytes(evidence)),
        "evidence": evidence,
    }


_control = dict(NATIVE_CONTEXT_REQUIRED_GATE_AUTHORITY[NATIVE262K_W4096_OWNER16_GATE])
_control.update(
    program=release.NATIVE_CONTEXT_PROGRAMS["native-262k"],
    recipe=(release.NATIVE_CONTEXT_RECIPES["native-262k"]["recipe_relpath"],
            release.NATIVE_CONTEXT_RECIPES["native-262k"]["recipe_sha256"]),
    helpers=tuple(row for row in _control["helpers"] if row[0] != "--engagement-program"),
    job_schema="flux-r16-native262k-w16384-job/1",
    evidence_schema="flux-r16-native262k-w16384-evidence/1",
    control_gate="native-262k-w4096-owner16",
)
NATIVE_CONTEXT_REQUIRED_GATE_AUTHORITY = MappingProxyType({
    "native-262k": MappingProxyType(_control),
    NATIVE262K_W4096_OWNER16_GATE: NATIVE_CONTEXT_REQUIRED_GATE_AUTHORITY[NATIVE262K_W4096_OWNER16_GATE],
})
assert tuple(NATIVE_CONTEXT_REQUIRED_GATE_AUTHORITY) == NATIVE_CONTEXT_REQUIRED_GATES


def validate_native_context_process_bindings(raw, manifest, container_id, gate):
    plan = release._profile_plan(manifest)
    contract = release.validate_native_context_contract(plan.get("native_context_contract"))
    binding = contract["gates"][gate]
    proofs = release._raw_list(raw, "process_bindings", gate + " evidence", 16)
    if len(proofs) != 1:
        raise ValueError("native context requires one owned plain process proof")
    expected = {
        "schema": release.NATIVE_CONTEXT_PROOF,
        "status": "PASS", "mode": "none", "gate": gate,
        "container_id": container_id, "run_id": manifest["launch"]["run_id"],
        "executable_sha256": manifest["image"]["in_container_binary_sha256"],
        "recipe_sha256": binding["recipe_sha256"],
        "chunk_tokens": binding["chunk_tokens"],
        "native_context_contract_sha256": release.sha256_bytes(release.canonical_bytes(contract)),
        "recipe_environment_verified": True, "mount_namespace_match": True,
        "cgroup_match": True, "paths_read_only": True,
    }
    proof = release._exact_object(proofs[0], {*expected, "host_pid", "start_time_ticks"},
                                  "native context process proof")
    if any(type(proof[key]) is not type(wanted) or proof[key] != wanted
           for key, wanted in expected.items()):
        raise ValueError("native context process proof differs from recipe or release identity")
    release._plain_int(proof["host_pid"], "native context process PID", 1, 1 << 31)
    release._plain_int(proof["start_time_ticks"], "native context process start", 1, 1 << 63)
    return proof
