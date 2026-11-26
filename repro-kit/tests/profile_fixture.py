# SPDX-License-Identifier: AGPL-3.0-only
"""Synthetic CPU arithmetic fixtures. Nothing here is a measured host baseline.

These builders emit the legacy contract `/1` shape (which triggers the `/1`
single-recipe HOLD on dual-recipe inputs, so legacy orchestration/arithmetic
tests mock only that policy). Plan `/4` (contract `/2`) tests upgrade via
`profile_fixture_v2` and validate the real dual-recipe binding without mocking.
"""

import copy
from pathlib import Path


def contract(q):
    kit = Path(q.__file__).parent
    return {
        "schema": "flux-performance-profile-contract/1",
        "profiles": {
            name: {
                "recipe_relpath": path,
                "recipe_sha256": q.sha256_file(kit.parent / path),
                "argv_overlay": [
                    "--max-prefill-tokens",
                    "4096",
                    "--max-seq-len",
                    "262144",
                    "--scheduling-policy",
                    "fifo" if name == "interactive" else "slai",
                ],
            }
            for name, (path, _wide, _mode) in q.PROFILE_RECIPES.items()
        },
    }


def source_entries(q, source_root):
    kit = Path(q.__file__).parent
    paths = [
        "crates/spark-server/src/api/completions.rs",
        "crates/spark-server/src/cli.rs",
        *(path for path, _wide, _mode in q.PROFILE_RECIPES.values()),
    ]
    entries = []
    for relative in paths:
        target = source_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((kit.parent / relative).read_bytes())
        target.chmod(0o644)
        entries.append(
            {
                "path": "flux-engine/" + relative,
                "role": "engine-source",
                "mode": 0o644,
                "bytes": target.stat().st_size,
                "sha256": q.sha256_file(target),
            }
        )
    return entries


def usage(context, budget, rate):
    return {
        "prompt_tokens": context,
        "completion_tokens": budget,
        "total_tokens": context + budget,
        "prompt_tokens_details": {"cached_tokens": 0},
        "response_token/s": rate,
        "time_to_first_token_ms": 1000.0,
    }


def host_reference(q, manifest, policy):
    profiles = {}
    for name in q.PROFILE_NAMES:
        rows = []
        for context in q.PROFILE_CONTEXTS:
            budget = min(256, 262144 - context)
            rows.append(
                {
                    "context_tokens": context,
                    "decode_tokens": budget - 1,
                    "plain_decode_wall_seconds": (budget - 1) / 40.0,
                    "k5_decode_wall_seconds": (budget - 1) / 50.0,
                    "prefill_wall_seconds": 2.0,
                    "prompt_sha256": q.sha256_bytes(
                        f"synthetic-prompt-{context}".encode()
                    ),
                    "plain_usage": usage(context, budget, 40.0),
                    "k5_usage": usage(context, budget, 50.0),
                    "prefill_client_monotonic": {"start": 100.0, "finish": 102.0},
                }
            )
        profiles[name] = {
            "argv_overlay": policy["profiles"][name]["argv_overlay"],
            "host_recipe_sha256": policy["profiles"][name]["recipe_sha256"],
            "rows": rows,
        }
    return {
        "schema": "flux-prefill-decode-host-reference/3",
        "chunk_tokens": 4096,
        "binary_sha256": manifest["image"]["in_container_binary_sha256"],
        "decode_budget_tokens": 256,
        "profiles": profiles,
        "measurement_contract": copy.deepcopy(q.PROFILE_MEASUREMENT_CONTRACT),
        "provenance": {
            key: manifest["provenance"][key] for key in q.PROFILE_PROVENANCE_KEYS
        },
    }


def profile_raw(q, manifest, container_id):
    plan = q.strict_json_loads(
        Path(manifest["suite"]["plan_path"]).read_bytes(), "synthetic plan"
    )
    policy = plan["performance_profile_contract"]
    host = host_reference(q, manifest, policy)
    proofs, metrics, records, arms, identities = [], [], [], {}, {}
    text = "ALPHA alpha-0731826 BETA beta-0492175 GAMMA gamma-0863204 " + "." * 256
    for index, (name, mode) in enumerate(
        (name, mode) for name in q.PROFILE_NAMES for mode in ("none", "native-mtp-k5")
    ):
        model = "flux-profile-prefill-decode-profiles-" + f"{index:032x}"
        argv = [
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
            "8888",
            "--max-batch-size",
            "1",
            "--max-num-seqs",
            "1",
            "--kv-cache-dtype",
            "bf16",
            "--enable-prefix-caching",
            "false",
            *policy["profiles"][name]["argv_overlay"],
        ]
        if mode == "native-mtp-k5":
            argv += ["--speculative", "--num-drafts", "4"]
        proof = {
            "schema": "flux-r15-live-process-proof/1",
            "status": "PASS",
            "mode": mode,
            "container_id": container_id,
            "run_id": manifest["launch"]["run_id"],
            "host_pid": 1000 + index,
            "start_time_ticks": 2000 + index,
            "executable_sha256": manifest["image"]["in_container_binary_sha256"],
            "recipe_sha256": manifest["image"]["recipe_sha256"],
            "recipe_environment_verified": True,
            "mount_namespace_match": True,
            "cgroup_match": True,
            "paths_read_only": True,
        }
        proofs.append(
            {
                "profile": name,
                "mode": mode,
                "proof": proof,
                "launch_argv": argv,
                "launch_argv_sha256": q.sha256_bytes(q.canonical_bytes(argv)),
            }
        )
        identities[(name, mode)] = q.sha256_bytes(q.canonical_bytes(proof))
        metric_text = (
            'atlas_spec_decode_verify_total{k="5"} '
            + ("0" if mode == "none" else "7")
            + "\n"
        )
        metrics.append(
            {
                "profile": name,
                "mode": mode,
                "k5_events": 0 if mode == "none" else 7,
                "metrics": metric_text,
                "metrics_sha256": q.sha256_bytes(metric_text.encode()),
            }
        )
        for context in q.PROFILE_CONTEXTS:
            budget = min(256, 262144 - context)
            prompt_sha = q.sha256_bytes(f"synthetic-prompt-{context}".encode())
            rate = 40.0 if mode == "none" else 50.0
            arm = {"first_token_id_exact": 42}
            for kind in ("exact", "ttft", "stream"):
                requested = 1 if kind == "ttft" else budget
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
                clock = {"start": 100.0, "finish": 102.0 if kind == "ttft" else 116.0}
                record = {
                    "profile": name,
                    "mode": mode,
                    "context_tokens": context,
                    "kind": kind,
                    "prompt_sha256": prompt_sha,
                    "request": request,
                    "client_monotonic": clock,
                    "client_wall_seconds": clock["finish"] - clock["start"],
                }
                response_id = f"synthetic-{index}-{context}-{kind}"
                proof_usage = usage(context, requested, rate)
                if kind == "stream":
                    events = [
                        [
                            0.25,
                            q.canonical_bytes(
                                {
                                    "id": response_id,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "text": text,
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                            ).decode(),
                        ],
                        [
                            0.5,
                            q.canonical_bytes(
                                {
                                    "id": response_id,
                                    "usage": proof_usage,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "text": "",
                                            "finish_reason": "length",
                                        }
                                    ],
                                }
                            ).decode(),
                        ],
                    ]
                    record["events"] = events
                    arm[kind] = {
                        "decode_tokens": budget - 1,
                        "decode_wall_seconds": (budget - 1) / rate,
                        "decode_clock": q.PROFILE_MEASUREMENT_CONTRACT["decode_clock"],
                        "engine_decode_tokens_per_second": rate,
                        "terminal_usage": proof_usage,
                        "streamed_text_sha256": q.sha256_bytes(text.encode()),
                        "stream_text_events": 1,
                        "observed_text_event_wall_seconds": 0.0,
                        "finish_reason": "length",
                    }
                else:
                    response = {
                        "id": response_id,
                        "usage": proof_usage,
                        "choices": [
                            {
                                "index": 0,
                                "text": text,
                                "token_ids": [42] * requested,
                                "finish_reason": "length",
                            }
                        ],
                    }
                    record["response"] = response
                    if kind == "exact":
                        arm[kind] = {
                            "completion_tokens": requested,
                            "finish_reason": "length",
                            "text_sha256": q.sha256_bytes(text.encode()),
                            "token_ids_sha256": q.sha256_bytes(
                                (42).to_bytes(4, "little") * requested
                            ),
                            "passkeys_retrieved": {
                                key: True for key in ("ALPHA", "BETA", "GAMMA")
                            },
                            "accuracy_pass": True,
                            "time_to_first_token_ms": 1000.0,
                        }
                    else:
                        arm[kind] = {
                            "prefill_prompt_tokens": context,
                            "prefill_ttft_seconds": 2.0,
                            "client_request_wall_prompt_tokens_per_second": context
                            / 2.0,
                            "prefill_clock": q.PROFILE_MEASUREMENT_CONTRACT[
                                "prefill_clock"
                            ],
                            "client_monotonic": clock,
                            "server_time_to_first_token_ms": 1000.0,
                            "first_token_id": 42,
                        }
                records.append(record)
            arms[(name, context, mode)] = arm
    rows, evidence_rows, observations, bands = [], [], [], []
    for name in q.PROFILE_NAMES:
        for context in q.PROFILE_CONTEXTS:
            budget = min(256, 262144 - context)
            k5_wall, plain_wall = (budget - 1) / 50.0, (budget - 1) / 40.0
            rows.append(
                {
                    "profile": name,
                    "context_tokens": context,
                    "prefill_prompt_tokens": context,
                    "prefill_ttft_seconds": 2.0,
                    "plain_decode_tokens": budget - 1,
                    "plain_decode_wall_seconds": plain_wall,
                    "k5_decode_tokens": budget - 1,
                    "k5_decode_wall_seconds": k5_wall,
                    "host_decode_wall_seconds": k5_wall,
                    "profile_decode_wall_seconds": k5_wall,
                    "k5_lossless": True,
                    "accuracy_pass": True,
                }
            )
            evidence_rows.append(
                {
                    "profile": name,
                    "context_tokens": context,
                    "prefill_tokens_per_second": context / 2.0,
                    "plain_decode_tokens_per_second": (budget - 1) / plain_wall,
                    "k5_decode_tokens_per_second": (budget - 1) / k5_wall,
                    "k5_lossless": True,
                    "host_delta_percent": 0.0,
                }
            )
            observations.append(
                {
                    "profile": name,
                    "context_tokens": context,
                    "prompt_sha256": q.sha256_bytes(
                        f"synthetic-prompt-{context}".encode()
                    ),
                    "plain": arms[(name, context, "none")],
                    "native_k5": arms[(name, context, "native-mtp-k5")],
                }
            )
            bands.append(
                {
                    "profile": name,
                    "context_tokens": context,
                    "decode_clock": q.PROFILE_MEASUREMENT_CONTRACT["decode_clock"],
                    "prefill_clock": q.PROFILE_MEASUREMENT_CONTRACT["prefill_clock"],
                    "k5_decode_delta_percent": 0.0,
                    "plain_decode_delta_percent": 0.0,
                    "prefill_delta_percent": 0.0,
                    "host_k5_decode_wall_seconds": k5_wall,
                    "host_plain_decode_wall_seconds": plain_wall,
                    "host_client_prefill_wall_seconds": 2.0,
                    "measured_k5_decode_wall_seconds": k5_wall,
                    "measured_plain_decode_wall_seconds": plain_wall,
                    "measured_client_prefill_wall_seconds": 2.0,
                }
            )
    journal = b"".join(q.canonical_bytes(record) + b"\n" for record in records)
    journal_path = Path(manifest["runtime_dir"]) / "validators/profile-responses.jsonl"
    journal_path.write_bytes(journal)
    identity_profiles = {}
    for name in q.PROFILE_NAMES:
        binding = policy["profiles"][name]
        identity_profiles[name] = {
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
            "effective_overlay": {
                "--max-prefill-tokens": 4096,
                "--max-seq-len": 262144,
                "--scheduling-policy": "fifo" if name == "interactive" else "slai",
                "--tbt-deadline-ms": None if name == "interactive" else 100,
            },
            "process_proof_sha256": {
                mode: identities[(name, mode)] for mode in ("none", "native-mtp-k5")
            },
        }
    raw = {
        "measurement_schema": "flux-profile-metrics/3",
        "measurement_contract": copy.deepcopy(q.PROFILE_MEASUREMENT_CONTRACT),
        "profile_contract": policy,
        "chunk_tokens": 4096,
        "rows": rows,
        "all_host_tolerances_met": True,
        "all_accuracy_checks_pass": True,
        "host_reference_sha256": q.sha256_bytes(q.canonical_bytes(host)),
        "host_reference": host,
        "profile_identity": {
            "image_recipe_sha256": manifest["image"]["recipe_sha256"],
            "profiles": identity_profiles,
            "profiles_share_argv": False,
        },
        "host_bands": bands,
        "observations": observations,
        "response_journal": {
            "artifact": "validators/profile-responses.jsonl",
            "sha256": q.sha256_bytes(journal),
            "requests": 72,
        },
        "process_bindings": proofs,
        "verification_metrics": metrics,
    }
    evidence = {
        "chunk_tokens": 4096,
        "contexts": list(q.PROFILE_CONTEXTS),
        "profiles": list(q.PROFILE_NAMES),
        "rows": evidence_rows,
        "all_host_tolerances_met": True,
        "all_accuracy_checks_pass": True,
    }
    return raw, evidence
