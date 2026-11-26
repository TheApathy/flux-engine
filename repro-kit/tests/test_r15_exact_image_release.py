# SPDX-License-Identifier: AGPL-3.0-only

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import profile_fixture
import profile_fixture_v2

MODULE_PATH = Path(__file__).resolve().parents[1] / "r15_exact_image_release.py"
KIT_DIR = MODULE_PATH.parent
SPEC = importlib.util.spec_from_file_location("r15_exact_image_release", MODULE_PATH)
assert SPEC and SPEC.loader
Q = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = Q
SPEC.loader.exec_module(Q)

IMAGE_ID = "sha256:" + "a" * 64
IMAGE_DIGEST = "sha256:" + "b" * 64
CONTAINER_ID = "c" * 64
CONTAINER_STANDOFF = "d" * 64
SOURCE_REVISION = "d" * 40
SOURCE_TREE = "e" * 40
BINARY_BYTES = b"fixture flux server binary\n"
RECIPE_BYTES = b"FLUX_MODE=interactive\nFLUX_SPEC_MODE=native-mtp\n"
BINARY_SHA = Q.sha256_bytes(BINARY_BYTES)
RECIPE_SHA = Q.sha256_bytes(RECIPE_BYTES)
CHECKPOINT_ID = "JMNI-Labs/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid"
CHECKPOINT_REVISION = "8" * 40
LAYERS = ["sha256:" + "2" * 64, "sha256:" + "3" * 64]
_NATIVE_W4096_TEMPLATE = None


def native_process_proof(gate, manifest):
    plan = Q.strict_json_loads(Path(manifest["suite"]["plan_path"]).read_bytes(), "fixture plan")
    recipe = plan["native_context_contract"]["gates"][gate]
    return {"schema": Q.NATIVE_CONTEXT_PROOF, "status": "PASS", "mode": "none", "gate": gate,
        "container_id": CONTAINER_ID, "run_id": manifest["launch"]["run_id"],
        "host_pid": 31415, "start_time_ticks": 814,
        "executable_sha256": manifest["image"]["in_container_binary_sha256"],
        "recipe_sha256": recipe["recipe_sha256"], "chunk_tokens": recipe["chunk_tokens"],
        "native_context_contract_sha256": Q.sha256_bytes(Q.canonical_bytes(plan["native_context_contract"])),
        **{key: True for key in ("recipe_environment_verified", "mount_namespace_match", "cgroup_match", "paths_read_only")}}


def native_w4096_fixture(manifest):
    """Reuse the explicitly mocked nine-request matrix, never a live endpoint."""
    global _NATIVE_W4096_TEMPLATE
    if _NATIVE_W4096_TEMPLATE is None:
        import test_native262k_w4096_owner16 as native
        case = native.MatrixTests('test_real_orchestration_preserves_all_nine_cases_and_64_chunk_canary')
        case.setUp()
        try:
            receipt, state = case.run_matrix()
        finally:
            case.doCleanups()
        documents = [json.loads(bytes(raw)) for raw in state['storage'].values()
                     if raw.startswith(b'{"container_id"')]
        validator, = [document for document in documents if document.get('raw')]
        _NATIVE_W4096_TEMPLATE = validator['raw'], receipt['evidence']
    raw, evidence = copy.deepcopy(_NATIVE_W4096_TEMPLATE)
    required = Q._native_context_required_module()
    raw['process_bindings'] = [native_process_proof('native-262k-w4096-owner16', manifest)]
    raw['live_controls'] = {
        'tokenizer_sha256': required.required_authority('native-262k-w4096-owner16')['tokenizer_sha256'],
        'controls': required._plain(required.NATIVE262K_W4096_OWNER16_CONTROLS),
        'argv_sha256': '3' * 64, 'environment_sha256': '4' * 64}
    return raw, evidence


def provenance(root, source_root, checkpoint_root, checkpoint_manifest, recipe):
    return {
        "release_version": Q.RELEASE_VERSION,
        "source_revision": SOURCE_REVISION,
        "source_tree": SOURCE_TREE,
        "source_root_path": str(source_root),
        "source_bundle_manifest_path": str(root / "source-manifest.json"),
        "source_bundle_manifest_sha256": "4" * 64,
        "source_bundle_path": str(root / "source.tar.gz"),
        "source_bundle_sha256": "5" * 64,
        "build_receipt_path": str(root / "build.json"),
        "build_receipt_sha256": "6" * 64,
        "source_to_binary_receipt_path": str(root / "source-to-binary.json"),
        "source_to_binary_receipt_sha256": "7" * 64,
        "binary_path": str(root / "flux-server"),
        "binary_sha256": BINARY_SHA,
        "checkpoint_id": CHECKPOINT_ID,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_path": str(checkpoint_root),
        "checkpoint_manifest_path": str(checkpoint_manifest),
        "checkpoint_manifest_sha256": Q.sha256_file(checkpoint_manifest),
        "recipe_path": str(recipe),
        "recipe_sha256": RECIPE_SHA,
    }


def make_fixture(root):
    root = Path(root)
    runtime = root / "runtime"
    runtime.mkdir()
    diagnostics = runtime / "diagnostics"
    diagnostics.mkdir()
    source_root = root / "source"
    program = source_root / "repro-kit" / "r15-container-suite.py"
    program.parent.mkdir(parents=True)
    program.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    program.chmod(0o755)
    plan = root / "suite-plan.json"
    plan.write_bytes(Q.canonical_bytes({"source_bundle_manifest_sha256": "4" * 64}))
    recipe = root / "release.env"
    recipe.write_bytes(RECIPE_BYTES)
    checkpoint_root = root / "checkpoint"
    checkpoint_root.mkdir()
    checkpoint_file = checkpoint_root / "weights.bin"
    checkpoint_file.write_bytes(b"fixture checkpoint")
    checkpoint_inventory = {
        "schema": Q.native262k_gate.CHECKPOINT_INVENTORY_SCHEMA,
        "checkpoint_id": CHECKPOINT_ID,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "file_count": 1,
        "total_bytes": checkpoint_file.stat().st_size,
        "files": [
            {
                "path": "weights.bin",
                "bytes": checkpoint_file.stat().st_size,
                "sha256": Q.sha256_file(checkpoint_file),
            }
        ],
    }
    checkpoint_manifest = root / "checkpoint-inventory.json"
    checkpoint_manifest.write_bytes(Q.canonical_bytes(checkpoint_inventory))
    manifest = {
        "schema": Q.MANIFEST_SCHEMA,
        "release_version": Q.RELEASE_VERSION,
        "runtime_dir": str(runtime),
        "image": {
            "id": IMAGE_ID,
            "digest": IMAGE_DIGEST,
            "rootfs_layers_sha256": Q.sha256_bytes(Q.canonical_bytes(LAYERS)),
            "recipe_sha256": RECIPE_SHA,
            "in_container_binary_sha256": BINARY_SHA,
            "in_container_binary_path": "/opt/flux/bin/flux-server-real",
            "in_container_recipe_path": (
                "/opt/flux/repro-kit/recipes/flux-r15-release.env"
            ),
            "performance_recipes": {
                name: {
                    "path": path,
                    "sha256": Q.sha256_file(
                        KIT_DIR.parent / path.removeprefix("/opt/flux/")
                    ),
                }
                for name, path in Q.PERFORMANCE_RECIPE_PATHS.items()
            },
        },
        "provenance": provenance(
            root, source_root, checkpoint_root, checkpoint_manifest, recipe
        ),
        "launch": {
            "name": "flux-r15-qual",
            "run_id": "r15-test-001",
            "cidfile": str(runtime / "container.cid"),
            "lockfile": str(runtime / "release.lock"),
            "diagnostics_dir": str(diagnostics),
            "base_url": "http://127.0.0.1:18898",
            "extra_run_args": [
                "--read-only",
                "--network",
                "host",
                "--gpus",
                "all",
                "--volume",
                f"{checkpoint_root}:/models:ro",
                "--volume",
                f"{runtime}:/results:rw",
            ],
            "entrypoint": "/usr/bin/sleep",
            "container_command": ["infinity"],
        },
        "suite": {
            "program_path": str(program),
            "program_sha256": Q.sha256_file(program),
            "plan_path": str(plan),
            "plan_sha256": Q.sha256_file(plan),
            "argv": [str(program), *Q.PLACEHOLDERS],
            "timeout_seconds": 600,
            "receipt_path": str(runtime / "suite-receipt.json"),
            "stdout_path": str(runtime / "suite.stdout"),
            "stderr_path": str(runtime / "suite.stderr"),
        },
    }
    required = Q._native_context_required_module()
    native_files = {Q.NATIVE_CONTEXT_REQUIRED_MODULE: (Q.NATIVE_CONTEXT_REQUIRED_MODULE_SHA256, 0o644)}
    for gate in required.NATIVE_CONTEXT_REQUIRED_GATES:
        authority = required.required_authority(gate)
        native_files[authority["program"][0]] = (authority["program"][1], 0o755)
        native_files[authority["recipe"][0]] = (authority["recipe"][1], 0o644)
        native_files.update({path: (digest, 0o644) for _flag, path, digest in authority["helpers"]})
    native_entries = []
    for relative, (digest, mode) in native_files.items():
        target = source_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = (KIT_DIR.parent / relative).read_bytes()
        assert Q.sha256_bytes(payload) == digest
        target.write_bytes(payload)
        target.chmod(mode)
        native_entries.append({"path": "flux-engine/" + relative, "role": "engine-source",
            "mode": mode, "bytes": len(payload), "sha256": digest})
    inventory_path = Path(manifest["provenance"]["source_bundle_manifest_path"])
    inventory_path.write_bytes(
        Q.canonical_bytes(
            {
                "files": [
                    {
                        "path": "flux-engine/repro-kit/r15-container-suite.py",
                        "role": "engine-source",
                        "mode": 0o755,
                        "bytes": program.stat().st_size,
                        "sha256": Q.sha256_file(program),
                    },
                    *profile_fixture.source_entries(Q, source_root),
                    *native_entries,
                ]
            }
        )
    )
    manifest["provenance"]["source_bundle_manifest_sha256"] = Q.sha256_file(
        inventory_path
    )
    plan_document = {
        key: manifest["provenance"][key]
        for key in (
            "source_root_path",
            "source_revision",
            "source_tree",
            "source_bundle_manifest_sha256",
            "checkpoint_id",
            "checkpoint_revision",
            "checkpoint_manifest_sha256",
            "recipe_sha256",
        )
    }
    plan_document["in_container_binary_sha256"] = BINARY_SHA
    plan_document["schema"] = "flux-r15-container-suite-plan/4"
    plan_document["runtime_dir"] = str(runtime)
    plan_document["native_context_contract"] = {
        "schema": Q.NATIVE_CONTEXT_SCHEMA, "gates": copy.deepcopy(Q.NATIVE_CONTEXT_RECIPES)}
    plan_document["release_version"] = Q.RELEASE_VERSION
    policy = profile_fixture_v2.contract(Q, profile_fixture.contract(Q),
                                         manifest["image"]["in_container_binary_path"])
    plan_document["performance_profile_contract"] = policy
    for key in ("in_container_binary_path", "in_container_recipe_path"):
        plan_document[key] = manifest["image"][key]
    plan_document["gate_jobs"] = {
        gate: {"program_path": str(program), "program_sha256": Q.sha256_file(program)}
        for gate in Q.REQUIRED_GATES
    }
    for gate in required.NATIVE_CONTEXT_REQUIRED_GATES:
        authority = required.required_authority(gate)
        driver = str(source_root / authority["program"][0])
        options = [*required.IDENTITY_OPTIONS, ("--runtime-dir", str(runtime)),
            ("--suite-program", str(program)), ("--suite-sha256", Q.sha256_file(program)),
            *[(flag, str(source_root / relative)) for flag, relative, _digest in authority["helpers"]]]
        plan_document["gate_jobs"][gate] = {
            "program_path": driver, "program_sha256": authority["program"][1],
            "argv": [driver, *[part for pair in options for part in pair]], "timeout_seconds": 86400,
            "receipt_relpath": "gates/" + gate + ".json", "stdout_relpath": "stdout/" + gate + ".log",
            "stderr_relpath": "stderr/" + gate + ".log"}
    plan_document["gate_jobs"]["prefill-decode-profiles"]["argv"] = [
        str(program),
        "--host-reference-sha256",
        Q.sha256_bytes(
            Q.canonical_bytes(profile_fixture_v2.host_reference(Q, manifest, policy))
        ),
    ]
    plan.write_bytes(Q.canonical_bytes(plan_document))
    manifest["suite"]["plan_sha256"] = Q.sha256_file(plan)
    manifest_path = root / "launch.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, manifest


def gate_evidence(name, manifest, raw_log):
    runtime = Path(manifest["runtime_dir"])
    validators = runtime / "validators"
    validators.mkdir(exist_ok=True)
    validator_path = validators / f"{name}.json"

    def document(raw):
        return {
            "schema": Q.GATE_VALIDATOR_SCHEMA,
            "gate": name,
            "status": "PASS",
            "release_qualified": True,
            "container_id": CONTAINER_ID,
            "run_id": manifest["launch"]["run_id"],
            "raw": raw,
        }

    if name == "startup-readiness":
        raw = {
            "attempts": [
                {
                    "fresh_process_mode": "none",
                    "ready_http_status": 200,
                    "model_identity_match": True,
                    "checkpoint_identity_match": True,
                    "plain_scheduler_ready": True,
                    "k5_scheduler_ready": True,
                },
                {
                    "fresh_process_mode": "native-mtp-k5",
                    "ready_http_status": 200,
                    "model_identity_match": True,
                    "checkpoint_identity_match": True,
                    "plain_scheduler_ready": True,
                    "k5_scheduler_ready": True,
                },
            ],
            "rootfs_read_only": True,
            "process_executable_sha256": BINARY_SHA,
        }
        validator_path.write_bytes(Q.canonical_bytes(document(raw)))
        return {
            "fresh_process_modes": ["none", "native-mtp-k5"],
            "ready_http_statuses": [200, 200],
            "model_identity_match": True,
            "checkpoint_identity_match": True,
            "rootfs_read_only": True,
            "plain_scheduler_ready": True,
            "k5_scheduler_ready": True,
            "process_executable_sha256": BINARY_SHA,
            "validator_artifact": validator_path.relative_to(runtime).as_posix(),
            "validator_receipt_sha256": Q.sha256_file(validator_path),
        }
    if name == "tool-api-streaming-blocking":
        cases = []
        for index in range(24):
            cases.append(
                {
                    "phase": "blocking" if index < 12 else "streaming",
                    "tool_calls_observed": 1,
                    "tool_calls_expected": 1,
                    "exact_arguments": True,
                    "finish_reason": "tool_calls",
                    "no_blank_line_runaway": True,
                    "no_all_tools_runaway": True,
                    "thinking_spacing_valid": True,
                }
            )
        raw = {"cases": cases}
        validator_path.write_bytes(Q.canonical_bytes(document(raw)))
        return {
            "blocking_cases": 12,
            "streaming_cases": 12,
            "tool_calls_expected": 24,
            "tool_calls_observed": 24,
            "exact_arguments": True,
            "finish_reasons_valid": True,
            "no_blank_line_runaway": True,
            "no_all_tools_runaway": True,
            "thinking_spacing_valid": True,
            "validator_artifact": validator_path.relative_to(runtime).as_posix(),
            "validator_receipt_sha256": Q.sha256_file(validator_path),
        }
    if name in ("mmlu-pro", "gsm8k"):
        items = 1_000 if name == "mmlu-pro" else 1_319
        correct = 650 if name == "mmlu-pro" else 1_254
        raw = {
            "dataset_sha256": (
                Q.MMLU_PRO_DATASET_SHA256
                if name == "mmlu-pro"
                else Q.GSM8K_DATASET_SHA256
            ),
            "items": [
                {
                    "plain_correct": index < correct,
                    "k5_correct": index < correct,
                    "answer_identity": True,
                    "correctness_identity": True,
                    "token_identity": True,
                }
                for index in range(items)
            ],
        }
        validator_path.write_bytes(Q.canonical_bytes(document(raw)))
        return {
            "dataset_sha256": raw["dataset_sha256"],
            "items": items,
            "no_spec_correct": correct,
            "k5_correct": correct,
            "score": correct / items,
            "provisional_floor": 0.65 if name == "mmlu-pro" else 0.95,
            "per_item_answer_identity": True,
            "per_item_correctness_identity": True,
            "per_item_token_identity": True,
            "validator_artifact": validator_path.relative_to(runtime).as_posix(),
            "validator_receipt_sha256": Q.sha256_file(validator_path),
        }
    if name == "plain-k5-long-context":
        cases = []
        for prompt_tokens in Q.LONG_CONTEXT_PROMPT_TOKENS:
            total = min(prompt_tokens + 64, 262_144)
            cases.append(
                {
                    "prompt_tokens": prompt_tokens,
                    "plain_total_tokens": total,
                    "k5_total_tokens": total,
                    "token_identity": True,
                    "text_identity": True,
                    "finish_reason_identity": True,
                    "usage_identity": True,
                    "native_k5_engaged": True,
                    "partial_accept_cycle": True,
                    "qsa_verify_rows_engaged": True,
                }
            )
        raw = {"cases": cases}
        validator_path.write_bytes(Q.canonical_bytes(document(raw)))
        return {
            "case_count": len(Q.LONG_CONTEXT_PROMPT_TOKENS),
            "max_prompt_tokens": max(Q.LONG_CONTEXT_PROMPT_TOKENS),
            "max_total_tokens": max(
                min(prompt_tokens + 64, 262_144)
                for prompt_tokens in Q.LONG_CONTEXT_PROMPT_TOKENS
            ),
            "prompt_token_counts": list(Q.LONG_CONTEXT_PROMPT_TOKENS),
            "plain_k5_token_identity": True,
            "plain_k5_text_identity": True,
            "plain_k5_finish_reason_identity": True,
            "plain_k5_usage_identity": True,
            "native_k5_engaged": True,
            "partial_accept_cycle": True,
            "qsa_verify_rows_engaged": True,
            "validator_artifact": validator_path.relative_to(runtime).as_posix(),
            "validator_receipt_sha256": Q.sha256_file(validator_path),
        }
    if name == "native-262k":
        answers = ["passkey", "armored", "vault"]
        raw = {
            "kv_dtype": "bf16",
            "max_seq_len": 262_144,
            "passkeys_per_prompt": 3,
            "retrievals": [
                {"prompt_tokens": tokens, "answers": answers, "expected": answers}
                for tokens in (258_000, 261_888, 262_080)
            ],
            "boundary": {"prompt_tokens": 262_143, "admitted": True},
            "overflow": {"prompt_tokens": 262_144, "admitted": False},
            "qsa_terminal_chunks_exact": True,
            "process_bindings": [native_process_proof(name, manifest)],
        }
        validator_path.write_bytes(Q.canonical_bytes(document(raw)))
        return {
            "kv_dtype": "bf16",
            "max_seq_len": 262_144,
            "retrieval_prompt_tokens": [258_000, 261_888, 262_080],
            "boundary_prompt_tokens": 262_143,
            "overflow_prompt_tokens": 262_144,
            "passkeys_per_prompt": 3,
            "all_retrieval_answers_exact": True,
            "boundary_admission_pass": True,
            "overflow_rejection_pass": True,
            "qsa_terminal_chunks_exact": True,
            "validator_artifact": validator_path.relative_to(runtime).as_posix(),
            "validator_receipt_sha256": Q.sha256_file(validator_path),
        }
    if name == "native-262k-w4096-owner16":
        raw, evidence = native_w4096_fixture(manifest)
        validator_path.write_bytes(Q.canonical_bytes(document(raw)))
        return {**evidence, "validator_artifact": validator_path.relative_to(runtime).as_posix(),
                "validator_receipt_sha256": Q.sha256_file(validator_path)}
    if name == "native-concurrency-c2-c8":
        arms = []
        for level in (2, 4, 8):
            for mode in ("none", "native-mtp-k5"):
                for _replica in range(3):
                    arms.append(
                        {
                            "level": level,
                            "mode": mode,
                            "fresh_process": True,
                            "full_frames": 8,
                            "forward_reverse_order_all_frames": True,
                            "per_frame_logit_identity": True,
                            "per_frame_state_identity": True,
                        }
                    )
        raw = {
            "arms": arms,
            "available_mib_samples": [20_000, 21_000, 22_000],
            "cancellation_replacement_pass": True,
            "final_highest_slot_reused": True,
            "post_drain_clean": True,
        }
        validator_path.write_bytes(Q.canonical_bytes(document(raw)))
        return {
            "levels": [2, 4, 8],
            "modes": ["none", "native-mtp-k5"],
            "fresh_processes": 18,
            "minimum_full_frames": 8,
            "forward_reverse_order_all_frames": True,
            "per_frame_logit_identity": True,
            "per_frame_state_identity": True,
            "cancellation_replacement_pass": True,
            "final_highest_slot_reused": True,
            "post_drain_clean": True,
            "minimum_available_mib": 20_000,
            "validator_artifact": validator_path.relative_to(runtime).as_posix(),
            "validator_receipt_sha256": Q.sha256_file(validator_path),
        }
    if name == "prefill-decode-profiles":
        raw, evidence = profile_fixture.profile_raw(Q, manifest, CONTAINER_ID)
        _plan, raw, evidence = profile_fixture_v2.upgrade(Q, manifest, raw, evidence)
        validator_path.write_bytes(Q.canonical_bytes(document(raw)))
        return {
            **evidence,
            "validator_artifact": validator_path.relative_to(runtime).as_posix(),
            "validator_receipt_sha256": Q.sha256_file(validator_path),
        }
    if name == "memory-floor":
        samples = []
        for index in range(100):
            samples.append(
                {
                    "available_mib": 20_000 if index == 0 else 21_000,
                    "used_mib": 80_000 if index == 0 else 79_000,
                    "allocator_warnings": 0,
                    "cuda_oom_count": 0,
                    "nvrm_count": 0,
                    "xid_count": 0,
                }
            )
        raw = {"samples": samples}
        validator_path.write_bytes(Q.canonical_bytes(document(raw)))
        return {
            "sample_count": 100,
            "minimum_available_mib": 20_000,
            "peak_used_mib": 80_000,
            "allocator_warning_count": 0,
            "cuda_oom_count": 0,
            "nvrm_count": 0,
            "xid_count": 0,
            "validator_artifact": validator_path.relative_to(runtime).as_posix(),
            "validator_receipt_sha256": Q.sha256_file(validator_path),
        }
    if name == "cancellation-slot-reuse":
        slots = [
            {"cancelled": index == 0, "survivor": index != 0} for index in range(8)
        ]
        raw = {
            "concurrency": 8,
            "slots": slots,
            "replacement_requests": [True],
            "replacement_survivor": True,
            "plain_and_k5": True,
            "exact_survivor_identity": True,
            "final_highest_slot_reused": True,
            "post_drain_clean": True,
        }
        validator_path.write_bytes(Q.canonical_bytes(document(raw)))
        return {
            "concurrency": 8,
            "cancelled_requests": 1,
            "survivors": 7,
            "replacement_requests": 1,
            "plain_and_k5": True,
            "exact_survivor_identity": True,
            "final_highest_slot_reused": True,
            "post_drain_clean": True,
            "validator_artifact": validator_path.relative_to(runtime).as_posix(),
            "validator_receipt_sha256": Q.sha256_file(validator_path),
        }
    if name == "raw-log-safety":
        raw = {
            "scan_scope": "complete-uncapped",
            "per_pattern_match_counts": {label: 0 for label in Q.RAW_LOG_PATTERNS},
            "raw_log_bytes": raw_log.stat().st_size,
            "all_processes_included": True,
        }
        validator_path.write_bytes(Q.canonical_bytes(document(raw)))
        return {
            "scan_scope": "complete-uncapped",
            "patterns": list(Q.RAW_LOG_PATTERNS),
            "match_count": 0,
            "raw_log_bytes": raw_log.stat().st_size,
            "raw_log_sha256": Q.sha256_file(raw_log),
            "all_processes_included": True,
            "validator_artifact": validator_path.relative_to(runtime).as_posix(),
            "validator_receipt_sha256": Q.sha256_file(validator_path),
        }
    raise AssertionError(name)


def suite_receipt(manifest):
    runtime = Path(manifest["runtime_dir"])
    gates_dir = runtime / "gates"
    gates_dir.mkdir(exist_ok=True)
    raw_log = runtime / "complete-raw.log"
    raw_log.write_bytes(b"complete fixture server log without fatal markers\n")
    gates = {}
    for index, name in enumerate(Q.REQUIRED_GATES):
        path = gates_dir / f"{index:02d}-{name}.json"
        document = {
            "schema": Q.GATE_SCHEMA,
            "gate": name,
            "status": "PASS",
            "release_qualified": True,
            "execution_scope": "immutable-container",
            "image_id": IMAGE_ID,
            "image_digest": IMAGE_DIGEST,
            "container_id": CONTAINER_ID,
            "run_id": manifest["launch"]["run_id"],
            "evidence": gate_evidence(name, manifest, raw_log),
        }
        path.write_bytes(Q.canonical_bytes(document))
        gates[name] = {
            "status": "PASS",
            "receipt_relpath": path.relative_to(runtime).as_posix(),
            "receipt_sha256": Q.sha256_file(path),
        }
    return {
        "schema": Q.SUITE_SCHEMA,
        "status": "PASS",
        "release_qualified": True,
        "release_version": Q.RELEASE_VERSION,
        "execution_scope": "immutable-container",
        "image_id": IMAGE_ID,
        "image_digest": IMAGE_DIGEST,
        "container_id": CONTAINER_ID,
        "run_id": manifest["launch"]["run_id"],
        "source_revision": SOURCE_REVISION,
        "source_tree": SOURCE_TREE,
        "source_bundle_manifest_sha256": manifest["provenance"][
            "source_bundle_manifest_sha256"
        ],
        "in_container_binary_sha256": BINARY_SHA,
        "recipe_sha256": RECIPE_SHA,
        "checkpoint_id": CHECKPOINT_ID,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_manifest_sha256": manifest["provenance"][
            "checkpoint_manifest_sha256"
        ],
        "suite_plan_sha256": manifest["suite"]["plan_sha256"],
        "runtime_dir_identity_sha256": Q._directory_identity_sha256(
            runtime, "fixture runtime_dir"
        ),
        "gates": gates,
        "process_bindings": [
            {
                "schema": "flux-r15-live-process-proof/1",
                "status": "PASS",
                "mode": mode,
                "container_id": CONTAINER_ID,
                "run_id": manifest["launch"]["run_id"],
                "host_pid": 100 + index,
                "start_time_ticks": 200 + index,
                "executable_sha256": BINARY_SHA,
                "recipe_sha256": RECIPE_SHA,
                "recipe_environment_verified": True,
                "mount_namespace_match": True,
                "cgroup_match": True,
                "paths_read_only": True,
            }
            for index, mode in enumerate(("none", "native-mtp-k5"))
        ],
        "raw_log_safety": {
            "scan_scope": "complete-uncapped",
            "patterns": list(Q.RAW_LOG_PATTERNS),
            "match_count": 0,
            "per_pattern_match_counts": {label: 0 for label in Q.RAW_LOG_PATTERNS},
            "raw_log_relpath": raw_log.relative_to(runtime).as_posix(),
            "raw_log_bytes": raw_log.stat().st_size,
            "raw_log_sha256": Q.sha256_file(raw_log),
        },
    }


def image_document(manifest):
    labels = {
        "org.opencontainers.image.version": Q.OCI_VERSION,
        "org.opencontainers.image.revision": SOURCE_REVISION,
        "com.theapathy.flux.source-tree": SOURCE_TREE,
        "com.theapathy.flux.binary-sha256": BINARY_SHA,
        "com.theapathy.flux.recipe-sha256": RECIPE_SHA,
    }
    for profile, binding in manifest["image"]["performance_recipes"].items():
        prefix = f"com.theapathy.flux.performance.{profile}.recipe"
        labels[f"{prefix}-path"] = binding["path"]
        labels[f"{prefix}-sha256"] = binding["sha256"]
    return {
        "Id": IMAGE_ID,
        "RepoDigests": ["ghcr.io/example/flux@" + IMAGE_DIGEST],
        "RootFS": {"Layers": LAYERS},
        "Config": {"Labels": labels},
    }


def container_document(manifest):
    return {
        "Id": CONTAINER_ID,
        "Image": IMAGE_ID,
        "Path": "/usr/bin/sleep",
        "Args": ["infinity"],
        "State": {"Running": True},
        "HostConfig": {"ReadonlyRootfs": True},
        "Mounts": [
            {
                "Type": "bind",
                "Source": manifest["provenance"]["checkpoint_path"],
                "Destination": "/models",
                "RW": False,
            },
            {
                "Type": "bind",
                "Source": manifest["runtime_dir"],
                "Destination": "/results",
                "RW": True,
            },
        ],
    }


class DockerBackend:
    def __init__(
        self,
        manifest,
        mutate=None,
        *,
        container_mutate=None,
        binary_bytes=BINARY_BYTES,
        recipe_bytes=RECIPE_BYTES,
        performance_recipe_bytes=None,
    ):
        self.manifest = manifest
        self.mutate = mutate
        self.container_mutate = container_mutate
        self.binary_bytes = binary_bytes
        self.recipe_bytes = recipe_bytes
        self.performance_recipe_bytes = performance_recipe_bytes or {
            name: (KIT_DIR.parent / path.removeprefix("/opt/flux/")).read_bytes()
            for name, path in Q.PERFORMANCE_RECIPE_PATHS.items()
        }
        self.calls = []

    def run(self, argv, *, timeout_s):
        self.calls.append(tuple(argv))
        if tuple(argv[:3]) == ("docker", "image", "inspect"):
            document = image_document(self.manifest)
            if self.mutate is not None:
                self.mutate(document)
            return Q.r12.CommandResult(0, json.dumps([document]), "")
        if tuple(argv[:3]) == ("docker", "container", "inspect"):
            document = container_document(self.manifest)
            if self.container_mutate is not None:
                self.container_mutate(document)
            return Q.r12.CommandResult(0, json.dumps(document), "")
        if tuple(argv[:3]) == ("docker", "create", "--entrypoint"):
            source = argv[-1]
            if source != self.manifest["image"]["id"]:
                return Q.r12.CommandResult(1, "", "unexpected standoff image")
            return Q.r12.CommandResult(0, CONTAINER_STANDOFF + "\n", "")
        if tuple(argv[:2]) == ("docker", "rm"):
            if argv[-1] != CONTAINER_STANDOFF:
                return Q.r12.CommandResult(1, "", "unexpected standoff removal")
            return Q.r12.CommandResult(0, CONTAINER_STANDOFF + "\n", "")
        if tuple(argv[:2]) == ("docker", "cp"):
            if "--follow-link" in argv or len(argv) != 4:
                return Q.r12.CommandResult(1, "", "unexpected docker cp invocation")
            source = argv[-2].split(":", 1)[1]
            destination = Path(argv[-1])
            if source == self.manifest["image"]["in_container_binary_path"]:
                destination.write_bytes(self.binary_bytes)
            elif source == self.manifest["image"]["in_container_recipe_path"]:
                destination.write_bytes(self.recipe_bytes)
            else:
                for name, binding in self.manifest["image"][
                    "performance_recipes"
                ].items():
                    if source == binding["path"]:
                        destination.write_bytes(self.performance_recipe_bytes[name])
                        break
                else:
                    return Q.r12.CommandResult(1, "", "unexpected fixture path")
            return Q.r12.CommandResult(0, "", "")
        return Q.r12.CommandResult(1, "", "unexpected Docker command")

    def popen(self, argv):  # pragma: no cover - the fake runner owns launch
        raise AssertionError("fake runner must not call Docker popen")


class _CaptureBackend:
    """Minimal docker stand-in for wrapper-owned diagnostics capture."""

    def __init__(self):
        self.calls = []

    def run(self, argv, *, timeout_s):
        self.calls.append(tuple(argv))
        prefix = tuple(argv[:3])
        if prefix in (
            ("docker", "container", "inspect"),
            ("docker", "container", "logs"),
        ):
            return Q.r12.CommandResult(0, "captured\n", "")
        return Q.r12.CommandResult(1, "", "unexpected diagnostics command")

    def popen(self, argv):  # pragma: no cover - capture never launches
        raise AssertionError("diagnostics capture must not popen")


class SuiteBackend:
    def __init__(self, receipt, *, rc=0, hook=None, before_capture=None):
        self.receipt = receipt
        self.rc = rc
        self.hook = hook
        self.before_capture = before_capture
        self.calls = []

    def run(
        self,
        argv,
        *,
        timeout_s,
        stdout_fd,
        stderr_fd,
        env,
        executable_fd,
        plan_fd,
        receipt_fd,
    ):
        if self.before_capture is not None:
            self.before_capture()
        executable_sha256 = Q.sha256_bytes(
            Q._read_fd_bytes(
                executable_fd,
                "fixture executable",
                16 * 1024 * 1024,
                required_nlink=None,
            )
        )
        plan_sha256 = Q.sha256_bytes(
            Q._read_fd_bytes(
                plan_fd,
                "fixture plan",
                Q.MAX_SUITE_PLAN_BYTES,
                required_nlink=None,
            )
        )
        self.calls.append(
            (
                list(argv),
                timeout_s,
                dict(env),
                executable_sha256,
                plan_sha256,
                receipt_fd,
            )
        )
        Q._write_all(stdout_fd, b"complete suite stdout\n")
        Q._write_all(stderr_fd, b"")
        if self.hook is not None:
            self.hook()
        Q._write_all(receipt_fd, Q.canonical_bytes(self.receipt))
        return self.rc


class Runner:
    def __init__(self, kwargs, cleanup=None, after_qualify=None):
        self.kwargs = kwargs
        self.cleanup = cleanup
        self.after_qualify = after_qualify

    def run(self, qualify):
        identity = Q.lifecycle.LaunchIdentity(
            name=self.kwargs["name"],
            image_id=self.kwargs["image_id"],
            run_id=self.kwargs["run_id"],
            owner_token="9" * 64,
            cleanup_version=Q.r12.R12_CLEANUP_VERSION,
            cidfile=str(self.kwargs["cidfile"]),
        )
        result = qualify(identity, CONTAINER_ID)
        if self.after_qualify is not None:
            self.after_qualify(result)
        cleanup = self.cleanup or Q.lifecycle.CleanupReceipt(
            lock_safe=True,
            lock_released=True,
            launch_client_settled=True,
            stable_stopped_sweeps=2,
            cycles=2,
            stopped_ids=[CONTAINER_ID],
            diagnostics_attempted_ids=[CONTAINER_ID],
        )
        return result, cleanup


class R15ExactImageTests(unittest.TestCase):
    def validate_synthetic_suite(self, *args, **kwargs):
        # Explicitly scoped synthetic orchestration, not runtime qualification.
        with mock.patch.object(Q, "_profile_runtime_recipe_binding"):
            return Q.validate_suite_receipt(*args, **kwargs)

    def run_synthetic_live(self, *args, **kwargs):
        with mock.patch.object(Q, "_profile_runtime_recipe_binding"):
            return Q.run_live(*args, **kwargs)

    def load(self, path, verify_files=True):
        with (
            mock.patch.object(Q.provenance_gate, "configure_release") as configure,
            mock.patch.object(
                Q.provenance_gate, "validate_qualification_provenance"
            ) as validate,
        ):
            value = Q.load_manifest(path, verify_files=verify_files)
        if verify_files:
            configure.assert_called_once_with(Q.RELEASE_VERSION)
            validate.assert_called_once()
        return value

    def test_dry_run_is_explicitly_non_qualified(self):
        with tempfile.TemporaryDirectory() as temp:
            path, _ = make_fixture(temp)
            manifest = self.load(path, verify_files=False)
            plan = Q.dry_run_plan(manifest)
        self.assertFalse(plan["release_qualified"])
        self.assertEqual(plan["required_gates"], list(Q.REQUIRED_GATES))
        self.assertIn("no-docker", plan["execution_scope"])

    def test_plan4_fixture_preserves_required_native_recipes_and_source_closure(self):
        with tempfile.TemporaryDirectory() as temp:
            path, source = make_fixture(temp)
            self.load(path)
            plan_path = Path(source["suite"]["plan_path"])
            original = Q.strict_json_loads(plan_path.read_bytes(), "fixture plan")
            self.assertEqual(original["schema"], "flux-r15-container-suite-plan/4")
            self.assertEqual(original["performance_profile_contract"]["schema"],
                             "flux-performance-profile-contract/2")
            self.assertEqual(original["native_context_contract"]["gates"], Q.NATIVE_CONTEXT_RECIPES)
            self.assertNotEqual(*[value["recipe_sha256"]
                                 for value in Q.NATIVE_CONTEXT_RECIPES.values()])
            for missing in Q.NATIVE_CONTEXT_PROGRAMS:
                changed = copy.deepcopy(original)
                changed["gate_jobs"].pop(missing)
                plan_path.write_bytes(Q.canonical_bytes(changed))
                source["suite"]["plan_sha256"] = Q.sha256_file(plan_path)
                path.write_bytes(Q.canonical_bytes(source))
                with self.subTest(missing=missing), self.assertRaises(ValueError):
                    self.load(path)
            changed = copy.deepcopy(original)
            changed["schema"] = "flux-r15-container-suite-plan/3"
            plan_path.write_bytes(Q.canonical_bytes(changed))
            source["suite"]["plan_sha256"] = Q.sha256_file(plan_path)
            path.write_bytes(Q.canonical_bytes(source))
            with self.assertRaises(ValueError):
                self.load(path)
            plan_path.write_bytes(Q.canonical_bytes(original))
            source["suite"]["plan_sha256"] = Q.sha256_file(plan_path)
            path.write_bytes(Q.canonical_bytes(source))
            target = Path(source["provenance"]["source_root_path"]) / Q.NATIVE_CONTEXT_REQUIRED_MODULE
            target.write_bytes(target.read_bytes() + b'\n# fixture source drift\n')
            with self.assertRaisesRegex(ValueError, "validator differs from proven inventory"):
                self.load(path)

    def test_checked_in_example_matches_the_dry_run_schema(self):
        manifest = self.load(
            KIT_DIR / "examples" / "r15-exact-image-launch.example.json",
            verify_files=False,
        )
        plan = Q.dry_run_plan(manifest)
        self.assertEqual(manifest["schema"], Q.MANIFEST_SCHEMA)
        self.assertEqual(manifest["release_version"], manifest["provenance"]["release_version"])
        self.assertEqual(plan["release_version"], manifest["release_version"])
        self.assertEqual(plan["required_gates"], list(Q.REQUIRED_GATES))
        self.assertFalse(plan["release_qualified"])

    def test_strict_json_rejects_duplicates_and_exponent_overflow(self):
        for raw in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":1e9999}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                Q.strict_json_loads(raw, "hostile")

    def test_manifest_rejects_bool_timeout_embedded_placeholder_and_escape(self):
        mutations = []
        with tempfile.TemporaryDirectory() as temp:
            path, source = make_fixture(temp)
            bad = copy.deepcopy(source)
            bad["suite"]["timeout_seconds"] = True
            mutations.append(bad)
            bad = copy.deepcopy(source)
            bad["suite"]["argv"][1] = "--container=@CONTAINER_ID@"
            mutations.append(bad)
            bad = copy.deepcopy(source)
            bad["suite"]["receipt_path"] = str(Path(temp) / "outside.json")
            mutations.append(bad)
            bad = copy.deepcopy(source)
            bad["provenance"]["checkpoint_id"] = "home/private/checkpoint/model"
            mutations.append(bad)
            bad = copy.deepcopy(source)
            bad["runtime_dir"] = str(Path(temp) / "runtime" / ".." / "runtime")
            mutations.append(bad)
            for index, value in enumerate(mutations):
                with self.subTest(index=index):
                    path.write_text(json.dumps(value), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        self.load(path, verify_files=False)

    def test_manifest_rejects_mount_overlay_injection_and_checkpoint_drift(self):
        with tempfile.TemporaryDirectory() as temp:
            path, source = make_fixture(temp)
            checkpoint = source["provenance"]["checkpoint_path"]
            runtime = source["runtime_dir"]
            candidates = []

            missing_results = copy.deepcopy(source)
            missing_results["launch"]["extra_run_args"] = missing_results["launch"][
                "extra_run_args"
            ][:-2]
            candidates.append(missing_results)

            duplicate_models = copy.deepcopy(source)
            duplicate_models["launch"]["extra_run_args"][-1] = (
                f"{checkpoint}:/models:ro"
            )
            candidates.append(duplicate_models)

            overlay_binary = copy.deepcopy(source)
            overlay_binary["launch"]["extra_run_args"][6] = (
                f"{checkpoint}:/opt/flux/bin:ro"
            )
            candidates.append(overlay_binary)

            wrong_checkpoint = copy.deepcopy(source)
            other_model = Path(temp) / "other-model"
            other_model.mkdir()
            wrong_checkpoint["launch"]["extra_run_args"][6] = (
                f"{other_model}:/models:ro"
            )
            candidates.append(wrong_checkpoint)

            writable_models = copy.deepcopy(source)
            writable_models["launch"]["extra_run_args"][6] = f"{checkpoint}:/models:rw"
            candidates.append(writable_models)

            relabel_models = copy.deepcopy(source)
            relabel_models["launch"]["extra_run_args"][6] = f"{checkpoint}:/models:ro,z"
            candidates.append(relabel_models)

            false_readonly = copy.deepcopy(source)
            false_readonly["launch"]["extra_run_args"][5:7] = [
                "--mount",
                f"type=bind,src={checkpoint},dst=/models,readonly=false",
            ]
            candidates.append(false_readonly)

            readonly_results = copy.deepcopy(source)
            readonly_results["launch"]["extra_run_args"][-1] = f"{runtime}:/results:ro"
            candidates.append(readonly_results)

            injected_env = copy.deepcopy(source)
            injected_env["launch"]["extra_run_args"].extend(["--env", "PATH=/attacker"])
            candidates.append(injected_env)

            injected_device = copy.deepcopy(source)
            injected_device["launch"]["extra_run_args"].extend(
                ["--device", "/dev/fake"]
            )
            candidates.append(injected_device)

            wrong_entrypoint = copy.deepcopy(source)
            wrong_entrypoint["launch"]["entrypoint"] = "/opt/flux/bin/flux-server-real"
            candidates.append(wrong_entrypoint)

            for index, value in enumerate(candidates):
                with self.subTest(index=index):
                    path.write_text(json.dumps(value), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        self.load(path, verify_files=False)

    def test_mounts_reject_broad_sources_runtime_drift_and_symlink_components(self):
        with tempfile.TemporaryDirectory() as temp:
            path, source = make_fixture(temp)
            checkpoint = source["provenance"]["checkpoint_path"]
            candidates = []

            broad_results = copy.deepcopy(source)
            broad_results["launch"]["extra_run_args"][-1] = "/:/results:rw"
            candidates.append(broad_results)

            wrong_runtime = Path(temp) / "other-runtime"
            wrong_runtime.mkdir()
            mismatched_results = copy.deepcopy(source)
            mismatched_results["launch"]["extra_run_args"][-1] = (
                f"{wrong_runtime}:/results:rw"
            )
            candidates.append(mismatched_results)

            no_read_only = copy.deepcopy(source)
            no_read_only["launch"]["extra_run_args"].remove("--read-only")
            candidates.append(no_read_only)

            for index, value in enumerate(candidates):
                with self.subTest(index=index):
                    path.write_text(json.dumps(value), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        self.load(path, verify_files=False)

            checkpoint_link = Path(temp) / "checkpoint-link"
            checkpoint_link.symlink_to(checkpoint, target_is_directory=True)
            args = copy.deepcopy(source["launch"]["extra_run_args"])
            args[6] = f"{checkpoint_link}:/models:ro"
            with self.assertRaisesRegex(ValueError, "symlinked component"):
                Q.validate_exact_image_run_args(
                    args,
                    checkpoint_path=checkpoint_link,
                    runtime_dir=Path(source["runtime_dir"]),
                    verify_sources=True,
                )

    def test_image_inspection_binds_id_digest_layers_and_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            path, source = make_fixture(temp)
            manifest = self.load(path, verify_files=False)
            good = Q.inspect_exact_image(manifest, DockerBackend(source))
            self.assertEqual(good["rootfs_layer_count"], 2)
            for mutation in (
                lambda doc: doc.__setitem__("Id", "sha256:" + "0" * 64),
                lambda doc: doc.__setitem__("RepoDigests", []),
                lambda doc: doc["RootFS"].__setitem__("Layers", LAYERS[:1]),
                lambda doc: doc["Config"]["Labels"].__setitem__(
                    "org.opencontainers.image.revision", "0" * 40
                ),
            ):
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    Q.inspect_exact_image(manifest, DockerBackend(source, mutation))

    def test_suite_receipt_requires_all_gates_and_exact_types(self):
        with tempfile.TemporaryDirectory() as temp:
            path, source = make_fixture(temp)
            manifest = self.load(path, verify_files=False)
            receipt_path = Path(manifest["suite"]["receipt_path"])
            candidates = []
            missing = suite_receipt(source)
            missing["gates"].pop(Q.REQUIRED_GATES[-1])
            candidates.append(missing)
            boolean_count = suite_receipt(source)
            boolean_count["raw_log_safety"]["match_count"] = False
            candidates.append(boolean_count)
            direct = suite_receipt(source)
            direct["execution_scope"] = "direct-binary-not-container"
            candidates.append(direct)
            wrong_plan = suite_receipt(source)
            wrong_plan["suite_plan_sha256"] = "f" * 64
            candidates.append(wrong_plan)
            wrong_runtime = suite_receipt(source)
            wrong_runtime["runtime_dir_identity_sha256"] = "f" * 64
            candidates.append(wrong_runtime)
            for index, value in enumerate(candidates):
                with self.subTest(index=index):
                    receipt_path.write_text(json.dumps(value), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        self.validate_synthetic_suite(
                            receipt_path, manifest, CONTAINER_ID
                        )
                    receipt_path.unlink()

    def test_suite_receipt_replays_gate_and_raw_log_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            path, source = make_fixture(temp)
            manifest = self.load(path, verify_files=False)
            receipt_path = Path(manifest["suite"]["receipt_path"])

            zero_gate = suite_receipt(source)
            zero_gate["gates"][Q.REQUIRED_GATES[0]]["receipt_sha256"] = "0" * 64
            receipt_path.write_bytes(Q.canonical_bytes(zero_gate))
            with self.assertRaisesRegex(ValueError, "all-zero"):
                self.validate_synthetic_suite(receipt_path, manifest, CONTAINER_ID)
            receipt_path.unlink()

            zero_raw = suite_receipt(source)
            zero_raw["raw_log_safety"]["raw_log_sha256"] = "0" * 64
            receipt_path.write_bytes(Q.canonical_bytes(zero_raw))
            with self.assertRaisesRegex(ValueError, "all-zero"):
                self.validate_synthetic_suite(receipt_path, manifest, CONTAINER_ID)
            receipt_path.unlink()

            missing_gate = suite_receipt(source)
            missing_path = (
                Path(source["runtime_dir"])
                / missing_gate["gates"][Q.REQUIRED_GATES[0]]["receipt_relpath"]
            )
            missing_path.unlink()
            receipt_path.write_bytes(Q.canonical_bytes(missing_gate))
            with self.assertRaisesRegex(ValueError, "unavailable"):
                self.validate_synthetic_suite(receipt_path, manifest, CONTAINER_ID)
            receipt_path.unlink()

            wrong_hash = suite_receipt(source)
            wrong_hash["gates"][Q.REQUIRED_GATES[0]]["receipt_sha256"] = "f" * 64
            receipt_path.write_bytes(Q.canonical_bytes(wrong_hash))
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                self.validate_synthetic_suite(receipt_path, manifest, CONTAINER_ID)
            receipt_path.unlink()

            wrong_identity = suite_receipt(source)
            gate = wrong_identity["gates"][Q.REQUIRED_GATES[0]]
            gate_path = Path(source["runtime_dir"]) / gate["receipt_relpath"]
            gate_document = json.loads(gate_path.read_text(encoding="utf-8"))
            gate_document["container_id"] = "f" * 64
            gate_path.write_bytes(Q.canonical_bytes(gate_document))
            gate["receipt_sha256"] = Q.sha256_file(gate_path)
            receipt_path.write_bytes(Q.canonical_bytes(wrong_identity))
            with self.assertRaisesRegex(ValueError, "exact launch identity"):
                self.validate_synthetic_suite(receipt_path, manifest, CONTAINER_ID)
            receipt_path.unlink()

            hidden_fatal = suite_receipt(source)
            raw_path = (
                Path(source["runtime_dir"])
                / hidden_fatal["raw_log_safety"]["raw_log_relpath"]
            )
            raw_path.write_bytes(b"fixture preface\nNVRM: injected fatal marker\n")
            hidden_fatal["raw_log_safety"]["raw_log_bytes"] = raw_path.stat().st_size
            hidden_fatal["raw_log_safety"]["raw_log_sha256"] = Q.sha256_file(raw_path)
            receipt_path.write_bytes(Q.canonical_bytes(hidden_fatal))
            with self.assertRaisesRegex(ValueError, "census differs"):
                self.validate_synthetic_suite(receipt_path, manifest, CONTAINER_ID)
            receipt_path.unlink()

    def test_every_gate_rejects_generic_or_incomplete_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            path, source = make_fixture(temp)
            manifest = self.load(path, verify_files=False)
            receipt_path = Path(manifest["suite"]["receipt_path"])
            runtime = Path(source["runtime_dir"])
            for name in Q.REQUIRED_GATES:
                with self.subTest(gate=name):
                    receipt = suite_receipt(source)
                    gate = receipt["gates"][name]
                    gate_path = runtime / gate["receipt_relpath"]
                    document = json.loads(gate_path.read_text(encoding="utf-8"))
                    document["evidence"] = {"fixture": name}
                    gate_path.write_bytes(Q.canonical_bytes(document))
                    gate["receipt_sha256"] = Q.sha256_file(gate_path)
                    receipt_path.write_bytes(Q.canonical_bytes(receipt))
                    with self.assertRaisesRegex(ValueError, "evidence.*keys differ"):
                        self.validate_synthetic_suite(
                            receipt_path, manifest, CONTAINER_ID
                        )
                    receipt_path.unlink()

    def test_gate_semantics_reject_accuracy_context_concurrency_and_profile_bypasses(
        self,
    ):
        mutations = (
            ("startup-readiness", "process_executable_sha256", "f" * 64),
            ("tool-api-streaming-blocking", "no_all_tools_runaway", False),
            ("mmlu-pro", "k5_correct", 649),
            ("gsm8k", "score", 0.5),
            ("plain-k5-long-context", "native_k5_engaged", False),
            (
                "plain-k5-long-context",
                "prompt_token_counts",
                list(Q.LONG_CONTEXT_PROMPT_TOKENS[:-1]),
            ),
            ("native-262k", "kv_dtype", "nvfp4"),
            ("native-concurrency-c2-c8", "modes", ["none"]),
            ("memory-floor", "cuda_oom_count", 1),
            ("cancellation-slot-reuse", "final_highest_slot_reused", False),
            ("raw-log-safety", "all_processes_included", False),
        )
        with tempfile.TemporaryDirectory() as temp:
            path, source = make_fixture(temp)
            manifest = self.load(path, verify_files=False)
            runtime = Path(source["runtime_dir"])
            receipt_path = Path(manifest["suite"]["receipt_path"])
            for gate_name, key, hostile in mutations:
                with self.subTest(gate=gate_name, key=key):
                    receipt = suite_receipt(source)
                    gate = receipt["gates"][gate_name]
                    gate_path = runtime / gate["receipt_relpath"]
                    document = json.loads(gate_path.read_text(encoding="utf-8"))
                    document["evidence"][key] = hostile
                    gate_path.write_bytes(Q.canonical_bytes(document))
                    gate["receipt_sha256"] = Q.sha256_file(gate_path)
                    receipt_path.write_bytes(Q.canonical_bytes(receipt))
                    with self.assertRaises(ValueError):
                        self.validate_synthetic_suite(
                            receipt_path, manifest, CONTAINER_ID
                        )
                    receipt_path.unlink()

            receipt = suite_receipt(source)
            gate = receipt["gates"]["prefill-decode-profiles"]
            gate_path = runtime / gate["receipt_relpath"]
            document = json.loads(gate_path.read_text(encoding="utf-8"))
            document["evidence"]["rows"][0]["host_delta_percent"] = 5.01
            gate_path.write_bytes(Q.canonical_bytes(document))
            gate["receipt_sha256"] = Q.sha256_file(gate_path)
            receipt_path.write_bytes(Q.canonical_bytes(receipt))
            with self.assertRaisesRegex(ValueError, "must be <= 5.0"):
                self.validate_synthetic_suite(receipt_path, manifest, CONTAINER_ID)
            receipt_path.unlink()

    def test_artifact_locator_traversal_and_symlink_parent_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            path, source = make_fixture(temp)
            manifest = self.load(path, verify_files=False)
            receipt_path = Path(manifest["suite"]["receipt_path"])
            traversal = suite_receipt(source)
            traversal["gates"][Q.REQUIRED_GATES[0]]["receipt_relpath"] = (
                "../outside.json"
            )
            receipt_path.write_bytes(Q.canonical_bytes(traversal))
            with self.assertRaisesRegex(ValueError, "runtime-relative"):
                self.validate_synthetic_suite(receipt_path, manifest, CONTAINER_ID)
            receipt_path.unlink()

            hostile = suite_receipt(source)
            gates = Path(source["runtime_dir"]) / "gates"
            moved = Path(source["runtime_dir"]) / "real-gates"
            gates.rename(moved)
            gates.symlink_to(moved, target_is_directory=True)
            receipt_path.write_bytes(Q.canonical_bytes(hostile))
            with self.assertRaisesRegex(ValueError, "symlinked parent"):
                self.validate_synthetic_suite(receipt_path, manifest, CONTAINER_ID)
            receipt_path.unlink()

    def test_complete_live_composition_seals_only_after_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            path, source = make_fixture(temp)
            manifest = self.load(path)
            docker = DockerBackend(source)
            suite = SuiteBackend(suite_receipt(source))
            runner_kwargs = []

            def factory(**kwargs):
                runner_kwargs.append(kwargs)
                return Runner(kwargs)

            outer = Path(temp) / "outer.json"
            with (
                mock.patch.object(Q.provenance_gate, "configure_release"),
                mock.patch.object(
                    Q.provenance_gate, "validate_qualification_provenance"
                ),
            ):
                result = self.run_synthetic_live(
                    manifest,
                    path,
                    outer,
                    docker_backend=docker,
                    suite_backend=suite,
                    runner_factory=factory,
                )
            self.assertTrue(result["release_qualified"])
            self.assertTrue(result["cleanup"]["lock_safe"])
            self.assertTrue(result["cleanup"]["lock_released"])
            self.assertEqual(len(docker.calls), 16)
            self.assertEqual(
                sum(
                    call[:3] == ("docker", "image", "inspect") for call in docker.calls
                ),
                2,
            )
            self.assertEqual(
                sum(call[:2] == ("docker", "create") for call in docker.calls),
                2,
            )
            self.assertEqual(
                sum(call[:2] == ("docker", "rm") for call in docker.calls),
                2,
            )
            # Every artifact copy comes from the unmounted standoff, never
            # --follow-link against the running rootfs.
            self.assertEqual(
                sum(
                    call[:2] == ("docker", "cp")
                    and "--follow-link" not in call
                    and call[-2].split(":", 1)[0] == "d" * 64
                    for call in docker.calls
                ),
                8,
            )
            self.assertEqual(
                sum(
                    call[:3] == ("docker", "container", "inspect")
                    for call in docker.calls
                ),
                2,
            )
            self.assertEqual(runner_kwargs[0]["image_id"], IMAGE_ID)
            self.assertEqual(runner_kwargs[0]["entrypoint"], "/usr/bin/sleep")
            argv, timeout, env, executable_sha256, plan_sha256, receipt_fd = (
                suite.calls[0]
            )
            self.assertEqual(timeout, 600)
            self.assertEqual(argv[1], CONTAINER_ID)
            self.assertEqual(argv[-1], str(receipt_fd))
            self.assertEqual(executable_sha256, source["suite"]["program_sha256"])
            self.assertEqual(plan_sha256, source["suite"]["plan_sha256"])
            self.assertNotIn("HOME", env)
            self.assertEqual(result["image_artifacts"]["binary"]["sha256"], BINARY_SHA)
            self.assertEqual(result["image_artifacts"]["recipe"]["sha256"], RECIPE_SHA)
            for profile, binding in source["image"]["performance_recipes"].items():
                measured = result["image_artifacts"]["performance_recipes"][profile]
                self.assertEqual(measured["sha256"], binding["sha256"])
                self.assertEqual(measured["container_path"], binding["path"])
                self.assertEqual(measured["source"], "standoff-image-bundle")
                self.assertGreater(measured["bytes"], 0)
            self.assertTrue(result["runtime_container"]["rootfs_read_only"])
            self.assertNotIn(
                source["runtime_dir"], json.dumps(result["runtime_container"])
            )
            self.assertEqual(json.loads(outer.read_text()), result)

    def test_running_container_contract_failures_are_rejected(self):
        mutations = (
            lambda document: document["HostConfig"].update(ReadonlyRootfs=False),
            lambda document: document["Mounts"][0].update(RW=True),
            lambda document: document["Mounts"][1].update(Source="/tmp/results"),
            lambda document: document.update(Path="/bin/sh"),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as temp:
                path, source = make_fixture(temp)
                manifest = self.load(path)
                backend = DockerBackend(source, container_mutate=mutate)
                with self.assertRaisesRegex(ValueError, "running container"):
                    Q.inspect_running_container(manifest, CONTAINER_ID, backend)

    def test_runtime_directory_path_replacement_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path, _source = make_fixture(temp)
            manifest = self.load(path)
            runtime = Path(manifest["_runtime_dir"])
            runtime.rename(Path(temp) / "original-runtime")
            runtime.mkdir()
            (runtime / "diagnostics").mkdir()
            with (
                mock.patch.object(Q.provenance_gate, "configure_release"),
                mock.patch.object(
                    Q.provenance_gate, "validate_qualification_provenance"
                ),
                self.assertRaisesRegex(ValueError, "runtime_dir identity changed"),
            ):
                Q.revalidate_manifest(manifest, path, "hostile replacement")

    def test_outer_receipt_rejects_symlinked_parent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            real = root / "real"
            real.mkdir()
            linked = root / "linked"
            linked.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlinked component"):
                Q._seal(linked / "receipt.json", {"status": "PASS"})
            self.assertFalse((real / "receipt.json").exists())

    def test_suite_failure_never_seals_outer(self):
        with tempfile.TemporaryDirectory() as temp:
            path, source = make_fixture(temp)
            manifest = self.load(path)
            outer = Path(temp) / "outer.json"
            docker = DockerBackend(source)

            def factory(**kwargs):
                return Runner(kwargs)

            with (
                mock.patch.object(Q.provenance_gate, "configure_release"),
                mock.patch.object(
                    Q.provenance_gate, "validate_qualification_provenance"
                ),
                self.assertRaises(RuntimeError),
            ):
                self.run_synthetic_live(
                    manifest,
                    path,
                    outer,
                    docker_backend=docker,
                    suite_backend=SuiteBackend(suite_receipt(source), rc=7),
                    runner_factory=factory,
                )
            self.assertFalse(outer.exists())

    def test_receipt_replacement_program_swap_and_image_drift_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            path, source = make_fixture(temp)
            manifest = self.load(path)
            outer = Path(temp) / "outer.json"
            reserved = Path(manifest["suite"]["receipt_path"])

            def replace_reserved_path():
                reserved.unlink()
                reserved.write_bytes(b"replacement pathname\n")

            with (
                mock.patch.object(Q.provenance_gate, "configure_release"),
                mock.patch.object(
                    Q.provenance_gate, "validate_qualification_provenance"
                ),
                self.assertRaises(ValueError),
            ):
                self.run_synthetic_live(
                    manifest,
                    path,
                    outer,
                    docker_backend=DockerBackend(source),
                    suite_backend=SuiteBackend(
                        suite_receipt(source), hook=replace_reserved_path
                    ),
                    runner_factory=lambda **kwargs: Runner(kwargs),
                )
            self.assertFalse(outer.exists())

        with tempfile.TemporaryDirectory() as temp:
            path, source = make_fixture(temp)
            manifest = self.load(path)
            outer = Path(temp) / "outer.json"
            program = Path(source["suite"]["program_path"])
            original = program.read_bytes()

            def swap_and_restore_program():
                program.write_bytes(b"#!/bin/sh\nexit 99\n")
                program.write_bytes(original)

            backend = SuiteBackend(
                suite_receipt(source), before_capture=swap_and_restore_program
            )
            with (
                mock.patch.object(Q.provenance_gate, "configure_release"),
                mock.patch.object(
                    Q.provenance_gate, "validate_qualification_provenance"
                ),
            ):
                result = self.run_synthetic_live(
                    manifest,
                    path,
                    outer,
                    docker_backend=DockerBackend(source),
                    suite_backend=backend,
                    runner_factory=lambda **kwargs: Runner(kwargs),
                )
            self.assertTrue(result["release_qualified"])
            self.assertEqual(backend.calls[0][3], source["suite"]["program_sha256"])

        with tempfile.TemporaryDirectory() as temp:
            path, source = make_fixture(temp)
            manifest = self.load(path)
            outer = Path(temp) / "outer.json"
            with (
                mock.patch.object(Q.provenance_gate, "configure_release"),
                mock.patch.object(
                    Q.provenance_gate, "validate_qualification_provenance"
                ),
                self.assertRaisesRegex(ValueError, "binary SHA-256 mismatch"),
            ):
                self.run_synthetic_live(
                    manifest,
                    path,
                    outer,
                    docker_backend=DockerBackend(source, binary_bytes=b"tampered"),
                    suite_backend=SuiteBackend(suite_receipt(source)),
                    runner_factory=lambda **kwargs: Runner(kwargs),
                )
            self.assertFalse(outer.exists())

    def test_post_cleanup_evidence_mutation_never_seals_outer(self):
        with tempfile.TemporaryDirectory() as temp:
            path, source = make_fixture(temp)
            manifest = self.load(path)
            receipt = suite_receipt(source)
            first = receipt["gates"][Q.REQUIRED_GATES[0]]
            gate_path = Path(source["runtime_dir"]) / first["receipt_relpath"]
            outer = Path(temp) / "outer.json"

            def mutate_after_qualification(_result):
                gate_path.write_bytes(b"mutated after qualification\n")

            with (
                mock.patch.object(Q.provenance_gate, "configure_release"),
                mock.patch.object(
                    Q.provenance_gate, "validate_qualification_provenance"
                ),
                self.assertRaisesRegex(ValueError, "changed before final seal"),
            ):
                self.run_synthetic_live(
                    manifest,
                    path,
                    outer,
                    docker_backend=DockerBackend(source),
                    suite_backend=SuiteBackend(receipt),
                    runner_factory=lambda **kwargs: Runner(
                        kwargs, after_qualify=mutate_after_qualification
                    ),
                )
            self.assertFalse(outer.exists())

    def test_post_cleanup_validator_mutation_never_seals_outer(self):
        with tempfile.TemporaryDirectory() as temp:
            path, source = make_fixture(temp)
            manifest = self.load(path)
            receipt = suite_receipt(source)
            first = receipt["gates"][Q.REQUIRED_GATES[0]]
            gate_path = Path(source["runtime_dir"]) / first["receipt_relpath"]
            gate_receipt = json.loads(gate_path.read_text(encoding="utf-8"))
            validator_path = (
                Path(source["runtime_dir"])
                / gate_receipt["evidence"]["validator_artifact"]
            )
            outer = Path(temp) / "outer.json"

            def mutate_after_qualification(_result):
                validator_path.write_bytes(b"mutated validator evidence\n")

            with (
                mock.patch.object(Q.provenance_gate, "configure_release"),
                mock.patch.object(
                    Q.provenance_gate, "validate_qualification_provenance"
                ),
                self.assertRaisesRegex(ValueError, "validator artifact changed"),
            ):
                self.run_synthetic_live(
                    manifest,
                    path,
                    outer,
                    docker_backend=DockerBackend(source),
                    suite_backend=SuiteBackend(receipt),
                    runner_factory=lambda **kwargs: Runner(
                        kwargs, after_qualify=mutate_after_qualification
                    ),
                )
            self.assertFalse(outer.exists())

    def test_dependent_journal_mutation_after_cleanup_or_at_preseal_never_seals(self):
        for phase in ("after-cleanup", "immediate-preseal"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as temp:
                path, source = make_fixture(temp)
                manifest = self.load(path)
                receipt = suite_receipt(source)
                journal = (
                    Path(source["runtime_dir"]) / "validators/profile-responses.jsonl"
                )
                outer = Path(temp) / "outer.json"

                def mutate(_result=None):
                    journal.write_bytes(journal.read_bytes() + b"tampered\n")

                class MutatingBackend(DockerBackend):
                    image_reads = 0

                    def run(self, argv, *, timeout_s):
                        if tuple(argv[:3]) == ("docker", "image", "inspect"):
                            self.image_reads += 1
                            if self.image_reads == 2 and phase == "immediate-preseal":
                                mutate()
                        return super().run(argv, timeout_s=timeout_s)

                with (
                    mock.patch.object(Q.provenance_gate, "configure_release"),
                    mock.patch.object(
                        Q.provenance_gate, "validate_qualification_provenance"
                    ),
                    self.assertRaisesRegex(
                        ValueError, "dependent artifact changed before final seal"
                    ),
                ):
                    self.run_synthetic_live(
                        manifest,
                        path,
                        outer,
                        docker_backend=MutatingBackend(source),
                        suite_backend=SuiteBackend(receipt),
                        runner_factory=lambda **kwargs: Runner(
                            kwargs,
                            after_qualify=mutate if phase == "after-cleanup" else None,
                        ),
                    )
                self.assertFalse(outer.exists())

    def test_preexisting_output_unsafe_cleanup_and_toctou_never_seal_outer(self):
        with tempfile.TemporaryDirectory() as temp:
            path, source = make_fixture(temp)
            manifest = self.load(path)
            outer = Path(temp) / "outer.json"
            Path(manifest["suite"]["stdout_path"]).write_text("stale", encoding="utf-8")

            with (
                mock.patch.object(Q.provenance_gate, "configure_release"),
                mock.patch.object(
                    Q.provenance_gate, "validate_qualification_provenance"
                ),
                self.assertRaises(FileExistsError),
            ):
                self.run_synthetic_live(
                    manifest,
                    path,
                    outer,
                    docker_backend=DockerBackend(source),
                    suite_backend=SuiteBackend(suite_receipt(source)),
                    runner_factory=lambda **kwargs: Runner(kwargs),
                )
            self.assertFalse(outer.exists())

        with tempfile.TemporaryDirectory() as temp:
            path, source = make_fixture(temp)
            manifest = self.load(path)
            outer = Path(temp) / "outer.json"
            unsafe = Q.lifecycle.CleanupReceipt(
                lock_safe=False,
                lock_released=False,
                launch_client_settled=True,
                stable_stopped_sweeps=2,
                errors=["fixture cleanup failure"],
            )

            with (
                mock.patch.object(Q.provenance_gate, "configure_release"),
                mock.patch.object(
                    Q.provenance_gate, "validate_qualification_provenance"
                ),
                self.assertRaisesRegex(RuntimeError, "cleanup"),
            ):
                self.run_synthetic_live(
                    manifest,
                    path,
                    outer,
                    docker_backend=DockerBackend(source),
                    suite_backend=SuiteBackend(suite_receipt(source)),
                    runner_factory=lambda **kwargs: Runner(kwargs, cleanup=unsafe),
                )
            self.assertFalse(outer.exists())

        with tempfile.TemporaryDirectory() as temp:
            path, source = make_fixture(temp)
            manifest = self.load(path)
            outer = Path(temp) / "outer.json"

            def mutate_manifest():
                changed = copy.deepcopy(source)
                changed["launch"]["run_id"] = "r15-mutated"
                path.write_text(json.dumps(changed), encoding="utf-8")

            with (
                mock.patch.object(Q.provenance_gate, "configure_release"),
                mock.patch.object(
                    Q.provenance_gate, "validate_qualification_provenance"
                ),
                self.assertRaisesRegex(ValueError, "changed after validation"),
            ):
                self.run_synthetic_live(
                    manifest,
                    path,
                    outer,
                    docker_backend=DockerBackend(source),
                    suite_backend=SuiteBackend(
                        suite_receipt(source), hook=mutate_manifest
                    ),
                    runner_factory=lambda **kwargs: Runner(kwargs),
                )
            self.assertFalse(outer.exists())

    def test_suite_plan_must_bind_the_proven_source_bundle(self):
        payloads = {
            "malformed json": b"not-json\n",
            "missing binding": b'{"release_version": "x"}\n',
            "all-zero binding": Q.canonical_bytes(
                {"source_bundle_manifest_sha256": "0" * 64}
            ),
            "mismatched binding": Q.canonical_bytes(
                {"source_bundle_manifest_sha256": "a" * 64}
            ),
        }
        with tempfile.TemporaryDirectory() as outer_temp:
            _, source = make_fixture(outer_temp)
            for label, payload in payloads.items():
                with self.subTest(label=label), tempfile.TemporaryDirectory() as inner:
                    plan_path = Path(inner) / "plan.json"
                    plan_path.write_bytes(payload)
                    fixture = copy.deepcopy(source)
                    fixture["suite"]["plan_path"] = str(plan_path)
                    fixture["suite"]["plan_sha256"] = Q.sha256_file(plan_path)
                    manifest_path = Path(inner) / "launch.json"
                    manifest_path.write_text(json.dumps(fixture), encoding="utf-8")
                    with (
                        mock.patch.object(Q.provenance_gate, "configure_release"),
                        mock.patch.object(
                            Q.provenance_gate, "validate_qualification_provenance"
                        ),
                        self.assertRaisesRegex(ValueError, "source bundle|strict JSON"),
                    ):
                        Q.load_manifest(manifest_path, verify_files=True)

    def test_checkpoint_and_runtime_must_be_disjoint_release_trees(self):
        with tempfile.TemporaryDirectory() as temp:
            _, source = make_fixture(temp)
            checkpoint = Path(source["provenance"]["checkpoint_path"])
            nested_runtime = checkpoint / "nested-runtime"

            def rebuild(runtime_dir, temp_root):
                fixture = copy.deepcopy(source)
                runtime = Path(runtime_dir)
                runtime.mkdir(parents=True, exist_ok=True)
                (runtime / "diagnostics").mkdir(exist_ok=True)
                fixture["runtime_dir"] = str(runtime)
                fixture["launch"]["cidfile"] = str(runtime / "container.cid")
                fixture["launch"]["lockfile"] = str(runtime / "release.lock")
                fixture["launch"]["diagnostics_dir"] = str(runtime / "diagnostics")
                fixture["suite"]["receipt_path"] = str(runtime / "suite-receipt.json")
                fixture["suite"]["stdout_path"] = str(runtime / "suite.stdout")
                fixture["suite"]["stderr_path"] = str(runtime / "suite.stderr")
                fixture_path = Path(temp_root) / "launch.json"
                fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
                return fixture_path

            variants = (
                ("nested under checkpoint", nested_runtime, True),
                ("dry-run nested under checkpoint", nested_runtime, False),
                ("runtime equals checkpoint", checkpoint, False),
            )
            for label, runtime_dir, verify in variants:
                with self.subTest(label=label), tempfile.TemporaryDirectory() as inner:
                    variant = rebuild(runtime_dir, inner)
                    with (
                        mock.patch.object(Q.provenance_gate, "configure_release"),
                        mock.patch.object(
                            Q.provenance_gate, "validate_qualification_provenance"
                        ),
                        self.assertRaisesRegex(ValueError, "disjoint"),
                    ):
                        Q.load_manifest(variant, verify_files=verify)

    def test_launch_path_replacement_failures_are_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            path, source = make_fixture(temp)
            lockfile = Path(source["launch"]["lockfile"])
            lockfile.parent.mkdir(parents=True, exist_ok=True)
            outside = Path(temp) / "poison"
            outside.write_text("poison", encoding="utf-8")
            outside.chmod(0o400)
            lockfile.symlink_to(outside)
            manifest = self.load(path)
            with self.assertRaisesRegex(ValueError, "lockfile must not be a symlink"):
                Q._prepare_release_launch_paths(manifest)

        with tempfile.TemporaryDirectory() as temp:
            path, source = make_fixture(temp)
            cidfile = Path(source["launch"]["cidfile"])
            cidfile.parent.mkdir(parents=True, exist_ok=True)
            cidfile.write_text("stale container id", encoding="utf-8")
            manifest = self.load(path)
            with self.assertRaisesRegex(ValueError, "cidfile must not pre-exist"):
                Q._prepare_release_launch_paths(manifest)

        with tempfile.TemporaryDirectory() as temp:
            path, source = make_fixture(temp)
            manifest = self.load(path)
            Q._prepare_release_launch_paths(manifest)
            # A stopped container legitimately keeps its cidfile on disk, so
            # the seal-time re-verification does not demand the target free.
            (Path(source["launch"]["cidfile"])).write_text(
                "running-chain\n", encoding="utf-8"
            )
            Q._prepare_release_launch_paths(manifest, expect_cidfile_free=False)

    def test_diagnostic_capture_never_follows_planted_symlinks(self):
        with tempfile.TemporaryDirectory() as temp:
            path, source = make_fixture(temp)
            manifest = self.load(path)
            Q._prepare_release_launch_paths(manifest)
            backend = _CaptureBackend()
            capture = Q._release_diagnostic_capture(manifest, backend)
            self.addCleanup(capture.close)
            record = Q.lifecycle.ContainerRecord(
                container_id=CONTAINER_ID,
                name="exact-image-fixture",
                image_id=IMAGE_ID,
                labels={},
                running=False,
            )
            capture(record)
            diagnostics = Path(source["launch"]["diagnostics_dir"])
            for suffix in (".inspect.json", ".server.log"):
                artifact = diagnostics / (CONTAINER_ID + suffix)
                self.assertEqual(artifact.read_bytes(), b"captured\n")
                self.assertEqual(oct(artifact.stat().st_mode & 0o777), "0o400")

        with tempfile.TemporaryDirectory() as temp:
            path, source = make_fixture(temp)
            manifest = self.load(path)
            Q._prepare_release_launch_paths(manifest)
            diagnostics = Path(source["launch"]["diagnostics_dir"])
            escape = Path(temp) / "escape-target.json"
            escape.write_bytes(b"poison\n")
            (diagnostics / (CONTAINER_ID + ".inspect.json")).symlink_to(escape)
            backend = _CaptureBackend()
            capture = Q._release_diagnostic_capture(manifest, backend)
            self.addCleanup(capture.close)
            record = Q.lifecycle.ContainerRecord(
                container_id=CONTAINER_ID,
                name="exact-image-fixture",
                image_id=IMAGE_ID,
                labels={},
                running=False,
            )
            with self.assertRaises(OSError):
                capture(record)
            self.assertEqual(escape.read_bytes(), b"poison\n")


if __name__ == "__main__":
    unittest.main()
