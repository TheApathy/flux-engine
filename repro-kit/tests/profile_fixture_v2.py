# SPDX-License-Identifier: AGPL-3.0-only
"""Synthetic plan /4 (profile contract /2) fixtures derived from profile_fixture.

Nothing here is a measured host baseline. The transformation keeps every v1
value and adds only the per-profile geometry, recipes and full-argv bindings.
"""

import copy
import json
from pathlib import Path

PLAN_V3 = "flux-r15-container-suite-plan/3"
PLAN_V4 = "flux-r15-container-suite-plan/4"
CHUNKS = {"interactive": 4096, "high-prefill": 16384}


def overlay(chunk):
    return [
        "--max-prefill-tokens", str(chunk),
        "--max-seq-len", "262144",
        "--block-size", "16",
        "--qwen4-qsa",
        "--scheduling-policy", "fifo",
        "--tbt-deadline-ms", "100",
        # Host recipe engine selectors (recipes/flux-serve*.argv).
        "--kernel-target", "qwen3.8-flash-next",
        "--gpu-memory-utilization", "0.85",
        "--ssm-cache-slots", "0",
        "--oom-guard-mb", "4096",
        "--request-timeout", "300",
        "--disable-simhash-watchdog",
        "--mtp-vocab", "0",
        "--mtp-quantization", "bf16",
    ]


def template_sha256(q, binary_path, argv_overlay, mode):
    return q.sha256_bytes(q.canonical_bytes(q._profile_launch_argv(
        binary_path, q.PROFILE_LAUNCH_MODEL, q.PROFILE_LAUNCH_PORT, argv_overlay, mode)))


def contract(q, v1, binary_path):
    profiles = {}
    for name, binding in v1["profiles"].items():
        argv_overlay = overlay(CHUNKS[name])
        profiles[name] = {
            "recipe_relpath": binding["recipe_relpath"],
            "recipe_sha256": binding["recipe_sha256"],
            "in_container_recipe_path": q.PERFORMANCE_RECIPE_PATHS[name],
            "chunk_tokens": CHUNKS[name],
            "argv_overlay": argv_overlay,
            "launch_argv_sha256": {
                mode: template_sha256(q, binary_path, argv_overlay, mode)
                for mode in ("none", "native-mtp-k5")
            },
        }
    return {"schema": "flux-performance-profile-contract/2", "profiles": profiles}


def host_reference(q, manifest, policy):
    import profile_fixture
    host = profile_fixture.host_reference(q, manifest, policy)
    host["schema"] = "flux-prefill-decode-host-reference/4"
    del host["chunk_tokens"]
    host["profile_chunk_tokens"] = dict(CHUNKS)
    for name, profile in host["profiles"].items():
        profile["argv_overlay"] = policy["profiles"][name]["argv_overlay"]
        profile["launch_argv_sha256"] = policy["profiles"][name]["launch_argv_sha256"]
    return host


def upgrade(q, manifest, raw, evidence):
    """Upgrade profile evidence while preserving the required plan /4 closure."""
    path = Path(manifest["suite"]["plan_path"])
    plan = q.strict_json_loads(path.read_bytes(), "synthetic plan")
    policy = contract(q, plan["performance_profile_contract"], plan["in_container_binary_path"])
    assert plan["schema"] == PLAN_V4
    q.validate_native_context_contract(plan["native_context_contract"])
    assert set(q.NATIVE_CONTEXT_PROGRAMS) <= set(plan["gate_jobs"])
    plan["performance_profile_contract"] = policy
    raw = copy.deepcopy(raw)
    evidence = copy.deepcopy(evidence)
    host = raw["host_reference"]
    host["schema"] = "flux-prefill-decode-host-reference/4"
    del host["chunk_tokens"]
    host["profile_chunk_tokens"] = dict(CHUNKS)
    for name, profile in host["profiles"].items():
        profile["argv_overlay"] = policy["profiles"][name]["argv_overlay"]
        profile["launch_argv_sha256"] = policy["profiles"][name]["launch_argv_sha256"]
    digest = q.sha256_bytes(q.canonical_bytes(host))
    raw["host_reference_sha256"] = digest
    plan["gate_jobs"]["prefill-decode-profiles"]["argv"][-1] = digest
    path.write_bytes(q.canonical_bytes(plan))
    manifest["suite"]["plan_sha256"] = q.sha256_file(path)
    raw["measurement_schema"] = "flux-profile-metrics/4"
    raw["profile_contract"] = policy
    del raw["chunk_tokens"]
    raw["profile_chunk_tokens"] = dict(CHUNKS)
    proof_hashes = {}
    for binding in raw["process_bindings"]:
        name, mode = binding["profile"], binding["mode"]
        argv = binding["launch_argv"]
        suffix = argv[-3:] if mode == "native-mtp-k5" else []
        argv[:] = argv[:19] + policy["profiles"][name]["argv_overlay"] + suffix
        binding["launch_argv_sha256"] = q.sha256_bytes(q.canonical_bytes(argv))
        proof = binding["proof"]
        proof["schema"] = "flux-r16-live-process-proof/2"
        proof["profile"] = name
        proof["recipe_sha256"] = policy["profiles"][name]["recipe_sha256"]
        proof_hashes[(name, mode)] = q.sha256_bytes(q.canonical_bytes(proof))
    for row in raw["rows"]:
        row["chunk_tokens"] = CHUNKS[row["profile"]]
        row["recipe_sha256"] = policy["profiles"][row["profile"]]["recipe_sha256"]
    for name, identity in raw["profile_identity"]["profiles"].items():
        binding = policy["profiles"][name]
        identity.update(binding)
        identity["host_recipe_sha256"] = binding["recipe_sha256"]
        identity["effective_overlay"] = {
            "--max-prefill-tokens": CHUNKS[name], "--max-seq-len": 262144, "--block-size": 16,
            "--scheduling-policy": "fifo", "--tbt-deadline-ms": 100, "--qwen4-qsa": True,
        }
        identity["process_proof_sha256"] = {
            mode: proof_hashes[(name, mode)] for mode in ("none", "native-mtp-k5")
        }
    journal = Path(manifest["runtime_dir"]) / raw["response_journal"]["artifact"]
    records = [json.loads(line) for line in journal.read_bytes().splitlines()]
    for record in records:
        record["chunk_tokens"] = CHUNKS[record["profile"]]
        record["recipe_sha256"] = policy["profiles"][record["profile"]]["recipe_sha256"]
    journal.write_bytes(b"".join(q.canonical_bytes(record) + b"\n" for record in records))
    raw["response_journal"]["sha256"] = q.sha256_file(journal)
    del evidence["chunk_tokens"]
    evidence["profile_chunk_tokens"] = dict(CHUNKS)
    return plan, raw, evidence
