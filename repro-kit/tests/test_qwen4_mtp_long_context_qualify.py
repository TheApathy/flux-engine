# SPDX-License-Identifier: AGPL-3.0-only

import copy
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

MODULE_PATH = Path(__file__).resolve().parents[1] / "qwen4_mtp_long_context_qualify.py"
SPEC = importlib.util.spec_from_file_location(
    "qwen4_mtp_long_context_qualify", MODULE_PATH
)
assert SPEC and SPEC.loader
Q = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(Q)


def fake_components(target):
    return {
        "prefix": [1, 2, 3],
        "filler": [10, 11, 12, 13, 14],
        "early": [20, target % 997, 21],
        "middle": [30, target % 991, 31],
        "late": [40, target % 983, 41],
        "query": [50, 51, 52, 53],
    }


def good_argv():
    return [
        "/bin/true",
        "serve",
        "--model-from-path",
        "/models/qad5500",
        "--model-name",
        "qad5500",
        "--kernel-target",
        "qwen3.8-flash-next",
        "--max-seq-len",
        "262144",
        "--max-prefill-tokens",
        "16384",
        "--kv-cache-dtype",
        "bf16",
        "--max-batch-size",
        "1",
        "--max-num-seqs",
        "1",
        "--world-size",
        "1",
        "--tp-size",
        "1",
        "--ep-size",
        "1",
        "--block-size",
        "16",
        "--qwen4-qsa",
        "--enable-prefix-caching",
        "false",
        "--ssm-cache-slots",
        "0",
        "--swap-space-gb",
        "0",
        "--bind",
        "127.0.0.1",
        "--port",
        "18991",
    ]


def common_env_text():
    return "\n".join(
        [
            "FLUX_CONCURRENCY=1",
            "FLUX_KV_CACHE_DTYPE=bf16",
            "FLUX_CTX_OVERFLOW_TRUNCATE=0",
            "ATLAS_CTX_OVERFLOW_TRUNCATE=0",
            "ATLAS_QWEN4_QSA_VERIFY_ROWS=1",
            "ATLAS_QWEN4_VERIFY_GRAPH=1",
            "ATLAS_QWEN4_QSA_GRAPH_SAFE=1",
            "ATLAS_MTP_ACCEPT_LOG=1",
            "ATLAS_DFLASH_ASYNC=0",
            "ATLAS_DFLASH_SPEC_CYCLE_V2=1",
            "RUST_LOG=info,spark::scheduler::spec_accept_log=debug",
            "",
        ]
    )


def provenance():
    return {
        "release_version": "v0.9.13-r15",
        "source_revision": "a" * 40,
        "source_tree": "2" * 40,
        "source_root_path": "/source",
        "source_bundle_manifest_path": "/receipts/source-manifest.json",
        "source_bundle_manifest_sha256": "b" * 64,
        "source_bundle_path": "/receipts/source.tar.gz",
        "source_bundle_sha256": "c" * 64,
        "build_receipt_path": "/receipts/build.json",
        "build_receipt_sha256": "3" * 64,
        "source_to_binary_receipt_path": "/receipts/source.json",
        "source_to_binary_receipt_sha256": "d" * 64,
        "binary_path": "/bin/true",
        "binary_sha256": "e" * 64,
        "checkpoint_id": "JMNI-Labs/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid",
        "checkpoint_revision": "f" * 40,
        "checkpoint_path": "/models/qad5500",
        "checkpoint_manifest_path": "/receipts/checkpoint.json",
        "checkpoint_manifest_sha256": "1" * 64,
    }


def full_source_provenance(root):
    root = Path(root)
    source = root / "source"
    flashinfer = root / "flashinfer"
    source.mkdir()
    flashinfer.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(
        ["git", "-C", str(source), "config", "user.email", "fixture@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(source), "config", "user.name", "Fixture"],
        check=True,
    )

    (source / "src/nested").mkdir(parents=True)
    (source / "src/nested/lib.rs").write_text("pub fn fixture() {}\n")
    profile_root = source / "crates/spark-server/src/recipe/builtin"
    profile_root.mkdir(parents=True)
    profiles = []
    for name in (
        "qwen3.8-flash-next-prefill4k-champion.yaml",
        "qwen3.8-flash-next-native-mtp-safe-local.yaml",
        "qwen3.8-27b-dflash2-v15-safe-local.yaml",
    ):
        path = profile_root / name
        path.write_text(f"profile: {name}\n")
        profiles.append(path)

    provenance_module = Q.r9_provenance
    header = flashinfer / "include/flashinfer/example.cuh"
    header.parent.mkdir(parents=True)
    header.write_bytes(b"// pinned fixture\n")
    flashinfer_manifest = source / provenance_module.FLASHINFER_MANIFEST_RELATIVE
    flashinfer_manifest.parent.mkdir(parents=True)
    flashinfer_manifest.write_text(
        "include/flashinfer/example.cuh\t"
        f"{provenance_module.sha256_file(header)}\t{header.stat().st_size}\n"
    )
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(source), "commit", "-q", "-m", "fixture"],
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    tree = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD^{tree}"], text=True
    ).strip()

    original_release = provenance_module.RELEASE_VERSION
    try:
        provenance_module.configure_release(Q.RELEASE_VERSION)
        bundle, manifest = provenance_module.build_source_bundle(
            source,
            commit,
            tree,
            flashinfer,
            root / "bundle",
        )
        binary = root / "spark-r15"
        binary.write_bytes(b"\x7fELFfixture\0" + commit.encode("ascii"))
        binary.chmod(0o555)
        log = root / "build.log"
        log.write_text(
            "compiled 1 kernels for target 0\ncompiled 1 kernels for target 1\n"
        )
        build_receipt = root / "build-receipt.json"
        track = "dense-qwen-dflash-v15"
        build = {
            "schema": provenance_module.BUILD_RECEIPT_SCHEMA,
            "source": {
                "path": str(source.resolve()),
                "commit": commit,
                "tree": tree,
                "status": "clean",
            },
            "build": {
                "track": track,
                "command": (
                    "ATLAS_TARGET_MODEL=qwen3.8-27b,qwen3.8-flash-next "
                    f"ATLAS_RELEASE_PROFILE_TRACK={track} "
                    "cargo build --locked --offline --release -j2 "
                    "-p spark-server --bin spark"
                ),
                "started_utc": "2026-10-18T00:00:00Z",
                "finished_utc": "2026-10-18T00:01:00Z",
                "log": str(log),
                "log_sha256": provenance_module.sha256_file(log),
                "flashinfer_input_manifest_sha256": provenance_module.sha256_file(
                    flashinfer_manifest
                ),
                "flashinfer_input_count": 1,
            },
            "kernel_targets": list(provenance_module.EXPECTED_KERNEL_TARGETS),
            "profiles": [
                {
                    "path": str(path),
                    "sha256": provenance_module.sha256_file(path),
                    "profile_kind": "fixture",
                    "environment_sha256": "0" * 64,
                }
                for path in profiles
            ],
            "binary": {
                "path": str(binary),
                "sha256": provenance_module.sha256_file(binary),
                "build_id": "fixture",
                "bytes": binary.stat().st_size,
                "mode": "0555",
                "nlink": 1,
            },
        }
        build_receipt.write_bytes(provenance_module.canonical_bytes(build))
        source_receipt = root / provenance_module.SOURCE_TO_BINARY_NAME
        receipt = provenance_module.make_source_to_binary_receipt(
            source,
            commit,
            tree,
            manifest,
            bundle,
            build_receipt,
            binary,
        )
        source_receipt.write_bytes(provenance_module.canonical_bytes(receipt))
        values = {
            "release_version": Q.RELEASE_VERSION,
            "source_revision": commit,
            "source_tree": tree,
            "source_root_path": str(source),
            "source_bundle_manifest_path": str(manifest),
            "source_bundle_manifest_sha256": provenance_module.sha256_file(manifest),
            "source_bundle_path": str(bundle),
            "source_bundle_sha256": provenance_module.sha256_file(bundle),
            "build_receipt_path": str(build_receipt),
            "build_receipt_sha256": provenance_module.sha256_file(build_receipt),
            "source_to_binary_receipt_path": str(source_receipt),
            "source_to_binary_receipt_sha256": provenance_module.sha256_file(
                source_receipt
            ),
            "binary_path": str(binary),
            "binary_sha256": provenance_module.sha256_file(binary),
        }
        return values, source, source_receipt
    finally:
        provenance_module.configure_release(original_release)


def response(
    prompt_tokens,
    answer,
    completion_tokens=31,
    finish_reason="stop",
    cached=0,
    token_ids=None,
):
    if token_ids is None:
        token_ids = list(range(1, completion_tokens + 1))
    return {
        "model": "qad5500",
        "choices": [
            {
                "index": 0,
                "text": " " + answer,
                "finish_reason": finish_reason,
                "token_ids": token_ids,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": cached},
        },
    }


def mtp_frame(seq_len=2_050, accepted=2, pattern="YYNN"):
    return (
        'MTP_ACCEPT frame proposer="mtp" '
        f'seq_len={seq_len} num_drafts=4 accepted={accepted} pattern="{pattern}"'
    )


def spec_cycle(req, pre, accepted=2, *, schema=2, k=5, gamma=4):
    emitted = accepted + 1
    timings = " ".join(
        f"{key}={index}" for index, key in enumerate(Q.SPEC_CYCLE_TIMING_FIELDS, 1)
    )
    return (
        f"SPEC_CYCLE_V2 schema={schema} req={req:016x} pre={pre} k={k} "
        f"gamma={gamma} accepted={accepted} emitted={emitted} c=1 "
        f"async_requested=0 async_engaged=0 {timings} total_host_ns=45 "
        "syncs_added=0"
    )


def qsa_route_frame(frame_id, seq_len, *, route="row-serial", rows=5):
    lines = []
    for layer in Q.QSA_FULL_ATTENTION_LAYER_IDS:
        ordinal = (frame_id - 1) * len(Q.QSA_FULL_ATTENTION_LAYER_IDS) + layer + 1
        dense_rows = rows if route == "row-serial" else 0
        row_serial_total = ordinal if route == "row-serial" else 0
        sparse_legacy_total = ordinal if route == "sparse-rows-legacy" else 0
        sparse_v2_total = ordinal if route == "sparse-rows-v2" else 0
        lines.append(
            "QSA_VERIFY_ROUTE_RECEIPT "
            f"route={route} frame_id={frame_id} rows={rows} seq_len={seq_len} "
            f"layer={layer} dense_paged_rows={dense_rows} sparse_qsa_rows=0 "
            "high_speed_swap_rows=0 mla_rows=0 "
            f"row_serial_frame_total={row_serial_total} "
            f"dense_paged_row_total={ordinal * dense_rows} "
            "sparse_qsa_row_total=0 high_speed_swap_row_total=0 mla_row_total=0 "
            f"sparse_rows_legacy_frame_total={sparse_legacy_total} "
            f"sparse_rows_v2_frame_total={sparse_v2_total}"
        )
    return lines


def request_log(prompt_tokens, req=1, accepted=2, *, qsa_telemetry=False):
    dispatch = (
        f"Chunked prefill start: {prompt_tokens} prompt tokens, chunk_size=16384, "
        "max_tokens=96"
        if prompt_tokens > 16_384
        else f"Prefilling: {prompt_tokens} prompt tokens, max_tokens=96"
    )
    first_spec_position = prompt_tokens + 1
    qsa_lines = qsa_route_frame(1, first_spec_position) if qsa_telemetry else []
    return "\n".join(
        [
            dispatch,
            *qsa_lines,
            mtp_frame(first_spec_position, accepted),
            spec_cycle(req, first_spec_position, accepted),
        ]
    )


def rejection_metric_values():
    return {
        "atlas_prefill_progress_chunks_total": 104,
        "atlas_prefill_progress_tokens_total": 105,
    }


def metrics_payload(values):
    return "".join(f"{name} {value}\n" for name, value in values.items()).encode()


def metric_receipt(values):
    raw = metrics_payload(values)
    return {
        "http_status": 200,
        "values": values,
        "body_sha256": Q.sha256_bytes(raw),
        "body_bytes": len(raw),
    }


class TokenizeClient:
    def __init__(self, drift=False):
        self.drift = drift
        self.calls = 0

    def json(self, method, route, body):
        assert method == "POST"
        assert route == "/tokenize"
        self.calls += 1
        tokens = [(ord(char) % 251) + 1 for char in body["prompt"]]
        if self.drift and self.calls == 1:
            tokens = tokens + [99]
        value = {"tokens": tokens, "count": len(tokens)}
        return 200, value, Q.canonical_bytes(value)


class LongContextPromptTests(unittest.TestCase):
    def test_workload_binds_qsa_edge_and_all_reported_lengths(self):
        self.assertEqual(
            Q.PROMPT_TOKEN_COUNTS[:9],
            (2_017, 2_043, 2_044, 2_045, 2_046, 2_047, 2_048, 2_049, 2_050),
        )
        for reported in (2_017, 2_112, 7_336, 7_710, 30_647, 114_755, 122_413, 238_938):
            self.assertIn(reported, Q.PROMPT_TOKEN_COUNTS)
        self.assertEqual(Q.PAIRED_PROMPT_TOKEN_COUNTS[-2:], (238_938, 262_080))
        self.assertEqual(Q.PROMPT_TOKEN_COUNTS[-2:], (262_143, 262_144))
        self.assertEqual(Q.REPETITIONS, 2)
        self.assertEqual(Q.FULL_ENDPOINT_REPETITIONS, 1)
        self.assertEqual(Q.NATIVE_CONTEXT, 262_144)

    def test_exact_prompt_lengths_and_spread_needles(self):
        for target in Q.PROMPT_TOKEN_COUNTS:
            components = fake_components(target)
            prompt, positions = Q.assemble_prompt(target, components)
            self.assertEqual(len(prompt), target)
            self.assertLess(positions["early"], target // 8)
            self.assertTrue(target * 2 // 5 < positions["middle"] < target * 3 // 5)
            self.assertGreater(positions["late"], target * 7 // 8)
            for label, start in positions.items():
                self.assertEqual(
                    prompt[start : start + len(components[label])], components[label]
                )

    def test_request_is_exact_greedy_and_deterministic(self):
        body = Q.completion_body("qad5500", [1, 2, 3])
        self.assertEqual(body["prompt_token_ids"], [1, 2, 3])
        self.assertEqual(body["temperature"], 0.0)
        self.assertEqual(body["top_k"], 1)
        self.assertEqual(body["top_p"], 1.0)
        self.assertEqual(body["seed"], Q.SEED)
        self.assertTrue(body["return_token_ids"])
        self.assertEqual(body["stop"], ["\n"])
        self.assertFalse(body["stream"])
        self.assertFalse(body["ignore_eos"])
        endpoint = Q.completion_body(
            "qad5500",
            [1, 2, 3],
            max_tokens=Q.FULL_ENDPOINT_MAX_TOKENS,
            ignore_eos=True,
            stop=[],
        )
        self.assertEqual(endpoint["max_tokens"], 64)
        self.assertTrue(endpoint["ignore_eos"])
        self.assertEqual(endpoint["stop"], [])

    def test_request_bundle_has_exact_lengths_and_hashes(self):
        client = TokenizeClient()
        bundle = Q.build_request_bundle(client, "qad5500")
        self.assertEqual(client.calls, len(Q.CASE_SPECS) * len(Q.COMPONENT_KEYS))
        self.assertEqual(
            [row["prompt_tokens"] for row in bundle["manifest"]["cases"]],
            [spec[1] for spec in Q.CASE_SPECS],
        )
        for case, body in zip(
            bundle["manifest"]["cases"], bundle["bodies"], strict=True
        ):
            self.assertEqual(len(body["prompt_token_ids"]), case["prompt_tokens"])
            self.assertEqual(
                Q.sha256_bytes(Q.canonical_bytes(body)), case["request_sha256"]
            )
        semantic_case = bundle["manifest"]["cases"][-4]
        semantic_body = bundle["bodies"][-4]
        self.assertEqual(semantic_case["case_id"], Q.ENDPOINT_SEMANTIC_CASE_SPEC[0])
        self.assertEqual(semantic_case["validation"], "exact-known-answer")
        self.assertEqual(semantic_body["max_tokens"], 64)
        self.assertFalse(semantic_body["ignore_eos"])
        self.assertEqual(semantic_body["stop"], ["\n"])
        endpoint_case = bundle["manifest"]["cases"][-3]
        endpoint_body = bundle["bodies"][-3]
        self.assertEqual(
            endpoint_case["validation"], "native-endpoint-exact-token-identity"
        )
        self.assertEqual(endpoint_case["repetitions"], 1)
        self.assertEqual(endpoint_body["max_tokens"], 64)
        self.assertTrue(endpoint_body["ignore_eos"])
        self.assertEqual(endpoint_body["stop"], [])
        accepted_case = bundle["manifest"]["cases"][-2]
        accepted_body = bundle["bodies"][-2]
        self.assertEqual(accepted_case["validation"], "native-endpoint-admission")
        self.assertEqual(accepted_case["prompt_tokens"], 262_143)
        self.assertEqual(accepted_body["max_tokens"], 1)
        rejected_case = bundle["manifest"]["cases"][-1]
        self.assertEqual(rejected_case["validation"], "context-length-rejection")
        self.assertEqual(rejected_case["prompt_tokens"], 262_144)
        self.assertEqual(
            Q.sha256_bytes(Q.canonical_bytes(bundle["manifest"])),
            bundle["manifest_sha256"],
        )

    def test_second_arm_tokenizer_must_match_plain_arm(self):
        bundle = Q.build_request_bundle(TokenizeClient(), "qad5500")
        Q.validate_tokenizer_parity(TokenizeClient(), bundle)
        with self.assertRaisesRegex(ValueError, "tokenizer drift"):
            Q.validate_tokenizer_parity(TokenizeClient(drift=True), bundle)


class AnswerAndPairTests(unittest.TestCase):
    def setUp(self):
        self.codes = {"early": "A" * 16, "middle": "B" * 16, "late": "C" * 16}
        self.answer = Q.expected_answer(self.codes)

    def row(self):
        result = Q.validate_completion_response(
            response(4_096, self.answer),
            4_096,
            self.answer,
            "2" * 64,
            "qad5500",
        )
        result["repetition"] = 0
        return result

    def test_parser_requires_one_canonical_answer(self):
        self.assertEqual(Q.parse_final_answer(" " + self.answer), self.codes)
        for value in (
            "no answer",
            self.answer + " and " + self.answer,
            self.answer.lower(),
        ):
            with self.assertRaisesRegex(ValueError, "exactly one"):
                Q.parse_final_answer(value)

    def test_response_seals_correctness_and_exact_usage(self):
        row = self.row()
        self.assertTrue(row["correct"])
        self.assertEqual(row["parsed_answer"], self.codes)
        self.assertEqual(row["final_answer"], self.answer)
        self.assertEqual(row["completion_tokens"], 31)
        self.assertEqual(row["completion_token_ids"], list(range(1, 32)))
        self.assertRegex(row["completion_u32le_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(row["finish_reason"], "stop")
        self.assertRegex(row["semantic_sha256"], r"^[0-9a-f]{64}$")

    def test_response_rejects_wrong_answer_truncation_cache_and_usage_drift(self):
        bad_answer = self.answer.replace("A" * 16, "D" * 16)
        cases = [
            response(4_096, bad_answer),
            response(4_096, self.answer, finish_reason="length"),
            response(4_096, self.answer, cached=1),
            response(4_095, self.answer),
        ]
        extra_text = response(4_096, self.answer)
        extra_text["choices"][0]["text"] += " trailing prose"
        cases.append(extra_text)
        wrong_model = response(4_096, self.answer)
        wrong_model["model"] = "another-model"
        cases.append(wrong_model)
        wrong_index = response(4_096, self.answer)
        wrong_index["choices"][0]["index"] = 1
        cases.append(wrong_index)
        bool_index = response(4_096, self.answer)
        bool_index["choices"][0]["index"] = False
        cases.append(bool_index)
        bool_cached = response(4_096, self.answer)
        bool_cached["usage"]["prompt_tokens_details"]["cached_tokens"] = False
        cases.append(bool_cached)
        bool_prompt_tokens = response(4_096, self.answer)
        bool_prompt_tokens["usage"]["prompt_tokens"] = True
        cases.append(bool_prompt_tokens)
        missing_ids = response(4_096, self.answer)
        del missing_ids["choices"][0]["token_ids"]
        cases.append(missing_ids)
        wrong_id_count = response(4_096, self.answer, token_ids=[1])
        cases.append(wrong_id_count)
        for value in cases:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    Q.validate_completion_response(
                        value, 4_096, self.answer, "2" * 64, "qad5500"
                    )

    def test_full_endpoint_requires_exact_262144_geometry_and_scheduler_ids(self):
        value = response(
            262_080,
            self.answer + " trailing forced decode",
            completion_tokens=64,
            finish_reason="length",
            token_ids=list(range(100, 164)),
        )
        row = Q.validate_completion_response(
            value,
            262_080,
            self.answer,
            "3" * 64,
            "qad5500",
            requested_tokens=64,
            validation="native-endpoint-exact-token-identity",
        )
        self.assertTrue(row["reached_native_endpoint"])
        self.assertEqual(row["total_tokens"], 262_144)
        self.assertEqual(row["completion_token_ids"], list(range(100, 164)))
        self.assertIsNone(row["parsed_answer"])
        self.assertIsNone(row["correct"])
        for mutation in (
            "short",
            "wrong-finish",
            "wrong-total",
            "cached",
            "missing-ids",
        ):
            bad = copy.deepcopy(value)
            if mutation == "short":
                bad["usage"]["completion_tokens"] = 63
                bad["usage"]["total_tokens"] = 262_143
                bad["choices"][0]["token_ids"] = list(range(100, 163))
            elif mutation == "wrong-finish":
                bad["choices"][0]["finish_reason"] = "stop"
            elif mutation == "wrong-total":
                bad["usage"]["total_tokens"] = 262_143
            elif mutation == "cached":
                bad["usage"]["prompt_tokens_details"]["cached_tokens"] = 1
            elif mutation == "missing-ids":
                del bad["choices"][0]["token_ids"]
            with self.subTest(mutation=mutation):
                with self.assertRaises(ValueError):
                    Q.validate_completion_response(
                        bad,
                        262_080,
                        self.answer,
                        "3" * 64,
                        "qad5500",
                        requested_tokens=64,
                        validation="native-endpoint-exact-token-identity",
                    )

    def test_stopped_endpoint_semantic_oracle_rejects_contradictions(self):
        value = response(262_080, self.answer)
        row = Q.validate_completion_response(
            value,
            262_080,
            self.answer,
            "5" * 64,
            "qad5500",
            requested_tokens=64,
            validation="exact-known-answer",
        )
        self.assertTrue(row["correct"])
        for text in (
            self.answer.replace("A" * 16, "D" * 16),
            self.answer + " " + self.answer,
            self.answer + " early=DEADBEEFDEADBEEF",
            "no canonical answer",
        ):
            bad = response(262_080, text)
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    Q.validate_completion_response(
                        bad,
                        262_080,
                        self.answer,
                        "5" * 64,
                        "qad5500",
                        requested_tokens=64,
                        validation="exact-known-answer",
                    )

    def test_one_token_native_endpoint_admission_is_exact_geometry(self):
        value = response(
            262_143,
            "boundary token",
            completion_tokens=1,
            finish_reason="length",
            token_ids=[777],
        )
        row = Q.validate_completion_response(
            value,
            262_143,
            self.answer,
            "4" * 64,
            "qad5500",
            requested_tokens=1,
            validation="native-endpoint-admission",
        )
        self.assertTrue(row["reached_native_endpoint"])
        self.assertEqual(row["completion_token_ids"], [777])
        self.assertIsNone(row["parsed_answer"])
        for mutation in ("short-prompt", "wrong-total", "wrong-finish"):
            bad = copy.deepcopy(value)
            prompt_tokens = 262_143
            if mutation == "short-prompt":
                prompt_tokens -= 1
                bad["usage"]["prompt_tokens"] -= 1
                bad["usage"]["total_tokens"] -= 1
            elif mutation == "wrong-total":
                bad["usage"]["total_tokens"] -= 1
            else:
                bad["choices"][0]["finish_reason"] = "stop"
            with self.subTest(mutation=mutation):
                with self.assertRaises(ValueError):
                    Q.validate_completion_response(
                        bad,
                        prompt_tokens,
                        self.answer,
                        "4" * 64,
                        "qad5500",
                        requested_tokens=1,
                        validation="native-endpoint-admission",
                    )

    def test_pair_comparison_accepts_only_exact_semantics(self):
        row = self.row()
        plain = [
            {
                "case_id": "ladder-4096",
                "observations": [row, {**row, "repetition": 1}],
            }
        ]
        k5 = copy.deepcopy(plain)
        summary = Q.compare_arms(plain, k5, ("ladder-4096",))
        self.assertEqual(summary["plain_k5_exact_matches"], 2)
        self.assertEqual(summary["semantic_correct_per_arm"], 2)
        self.assertEqual(summary["identity_only_per_arm"], 0)
        self.assertEqual(summary["errors"], 0)
        self.assertEqual(summary["truncations"], 0)
        incorrect_plain = copy.deepcopy(plain)
        incorrect_k5 = copy.deepcopy(k5)
        for pair in (incorrect_plain, incorrect_k5):
            for observation in pair[0]["observations"]:
                observation["correct"] = False
                observation["semantic_sha256"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "semantically correct"):
            Q.compare_arms(incorrect_plain, incorrect_k5, ("ladder-4096",))

    def test_forced_fill_pair_requires_identity_not_semantic_truthiness(self):
        value = response(
            262_080,
            "arbitrary forced fill that is not a semantic answer",
            completion_tokens=64,
            finish_reason="length",
            token_ids=list(range(100, 164)),
        )
        row = Q.validate_completion_response(
            value,
            262_080,
            self.answer,
            "3" * 64,
            "qad5500",
            requested_tokens=64,
            validation="native-endpoint-exact-token-identity",
        )
        row["repetition"] = 0
        plain = [{"case_id": Q.FULL_ENDPOINT_CASE_SPEC[0], "observations": [row]}]
        k5 = copy.deepcopy(plain)
        summary = Q.compare_arms(plain, k5, (Q.FULL_ENDPOINT_CASE_SPEC[0],))
        self.assertEqual(summary["correct_per_arm"], 0)
        self.assertEqual(summary["semantic_correct_per_arm"], 0)
        self.assertEqual(summary["identity_only_per_arm"], 1)
        self.assertEqual(summary["plain_k5_exact_token_id_matches"], 1)
        self.assertEqual(summary["native_262144_endpoint_exact_token_id_matches"], 1)

        for field, mutation in (
            ("text", "different fill"),
            ("completion_token_ids", [999] * 64),
            ("completion_u32le_sha256", "f" * 64),
        ):
            changed = copy.deepcopy(k5)
            changed[0]["observations"][0][field] = mutation
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "plain/K5 mismatch"):
                    Q.compare_arms(plain, changed, (Q.FULL_ENDPOINT_CASE_SPEC[0],))

        falsely_semantic_plain = copy.deepcopy(plain)
        falsely_semantic_k5 = copy.deepcopy(k5)
        falsely_semantic_plain[0]["observations"][0]["correct"] = True
        falsely_semantic_k5[0]["observations"][0]["correct"] = True
        falsely_semantic_plain[0]["observations"][0]["semantic_sha256"] = "e" * 64
        falsely_semantic_k5[0]["observations"][0]["semantic_sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "pure endpoint-fill identity"):
            Q.compare_arms(
                falsely_semantic_plain,
                falsely_semantic_k5,
                (Q.FULL_ENDPOINT_CASE_SPEC[0],),
            )
        for mutation in ("missing-correct", "wrong-geometry"):
            bad_plain = copy.deepcopy(plain)
            bad_k5 = copy.deepcopy(k5)
            for pair in (bad_plain, bad_k5):
                observation = pair[0]["observations"][0]
                if mutation == "missing-correct":
                    del observation["correct"]
                else:
                    observation["total_tokens"] = 262_143
                observation["semantic_sha256"] = "c" * 64
            with self.subTest(mutation=mutation):
                with self.assertRaisesRegex(ValueError, "pure endpoint-fill identity"):
                    Q.compare_arms(
                        bad_plain,
                        bad_k5,
                        (Q.FULL_ENDPOINT_CASE_SPEC[0],),
                    )

    def test_pair_comparison_rejects_every_release_field_drift(self):
        for field, changed in (
            ("request_sha256", "9" * 64),
            ("completion_tokens", 30),
            ("completion_token_ids", [99] * 31),
            ("completion_u32le_sha256", "8" * 64),
            ("finish_reason", "length"),
            ("text", "different"),
            ("final_answer", "different"),
            ("correct", False),
        ):
            row = self.row()
            plain = [
                {
                    "case_id": "ladder-4096",
                    "observations": [row, {**row, "repetition": 1}],
                }
            ]
            k5 = copy.deepcopy(plain)
            k5[0]["observations"][0][field] = changed
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "plain/K5 mismatch"):
                    Q.compare_arms(plain, k5, ("ladder-4096",))

    def test_rollback_sequence_reuses_exact_2017_canary(self):
        answers = {
            2_017: self.answer,
            2_050: self.answer.replace("A" * 16, "D" * 16),
        }
        cases = []
        bodies = []
        for case_id, prompt_tokens in (
            (Q.ROLLBACK_SENTINEL_CASE_ID, 2_017),
            (Q.ROLLBACK_TRIGGER_CASE_ID, 2_050),
        ):
            body = Q.completion_body("qad5500", [1] * prompt_tokens)
            cases.append(
                {
                    "case_id": case_id,
                    "prompt_tokens": prompt_tokens,
                    "expected_answer": answers[prompt_tokens],
                    "request_sha256": Q.sha256_bytes(Q.canonical_bytes(body)),
                    "requested_tokens": Q.MAX_TOKENS,
                    "validation": "exact-known-answer",
                    "repetitions": Q.REPETITIONS,
                }
            )
            bodies.append(body)
        bundle = {"manifest": {"cases": cases}, "bodies": bodies}

        class Client:
            def __init__(self, drift=False):
                self.calls = 0
                self.drift = drift

            def json(self, method, route, body):
                self.calls += 1
                prompt_tokens = len(body["prompt_token_ids"])
                token_ids = list(range(1, 32))
                if self.drift and self.calls == 3:
                    token_ids[-1] = 99
                value = response(
                    prompt_tokens,
                    answers[prompt_tokens],
                    token_ids=token_ids,
                )
                return 200, value, Q.canonical_bytes(value)

        receipt = Q.execute_rollback_slot_reuse(Client(), bundle)
        self.assertTrue(receipt["sentinel_exact_match_after_reuse"])
        self.assertEqual(
            [row["case_id"] for row in receipt["sequence"]],
            [
                Q.ROLLBACK_SENTINEL_CASE_ID,
                Q.ROLLBACK_TRIGGER_CASE_ID,
                Q.ROLLBACK_SENTINEL_CASE_ID,
            ],
        )
        with self.assertRaisesRegex(ValueError, "sentinel changed"):
            Q.execute_rollback_slot_reuse(Client(drift=True), bundle)


class LaunchContractTests(unittest.TestCase):
    def make_manifest(self, root):
        root = Path(root)
        env_path = root / "common.env"
        env_path.write_text(common_env_text(), encoding="utf-8")
        plain = good_argv()
        k5 = plain + [
            "--speculative",
            "--num-drafts",
            "4",
            "--mtp-vocab",
            "0",
            "--mtp-quantization",
            "bf16",
        ]
        value = {
            "schema": Q.MANIFEST_SCHEMA,
            "model": "qad5500",
            "common_env_file": str(env_path),
            "common_env_sha256": Q.sha256_file(env_path),
            "request_contract": Q.launch_request_contract(),
            "provenance": provenance(),
            "arms": {
                "plain": {
                    "argv": plain,
                    "cwd": str(root),
                    "base_url": "http://127.0.0.1:18991",
                    "environment": Q.ARM_ENVIRONMENTS["plain"],
                },
                "k5": {
                    "argv": k5,
                    "cwd": str(root),
                    "base_url": "http://127.0.0.1:18991",
                    "environment": Q.ARM_ENVIRONMENTS["k5"],
                },
                Q.EAGER_CHURN_ARM: {
                    "argv": list(k5),
                    "cwd": str(root),
                    "base_url": "http://127.0.0.1:18991",
                    "environment": Q.ARM_ENVIRONMENTS[Q.EAGER_CHURN_ARM],
                },
            },
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(value), encoding="utf-8")
        return manifest_path, value

    def test_manifest_requires_same_artifacts_and_only_speculation_differences(self):
        with tempfile.TemporaryDirectory() as temp:
            path, _ = self.make_manifest(temp)
            value = Q.load_manifest(path, verify_files=False)
            self.assertEqual(value["arms"]["plain"]["argv"][0], "/bin/true")
            self.assertEqual(value["arms"]["k5"]["environment"]["FLUX_GAMMA"], "4")
            self.assertEqual(value["_common_env"]["ATLAS_QWEN4_QSA_VERIFY_ROWS"], "1")
            self.assertEqual(
                value["arms"][Q.EAGER_CHURN_ARM]["environment"][
                    "ATLAS_QWEN4_VERIFY_GRAPH"
                ],
                "0",
            )
            self.assertEqual(
                value["arms"][Q.EAGER_CHURN_ARM]["environment"][
                    "ATLAS_DFLASH_DEBUG_NO_GRAPH"
                ],
                "1",
            )

    def test_manifest_rejects_k5_width_profile_and_target_drift(self):
        with tempfile.TemporaryDirectory() as temp:
            path, value = self.make_manifest(temp)
            mutations = []
            bad_width = copy.deepcopy(value)
            index = bad_width["arms"]["k5"]["argv"].index("--num-drafts") + 1
            bad_width["arms"]["k5"]["argv"][index] = "3"
            mutations.append(bad_width)
            bad_profile = copy.deepcopy(value)
            bad_profile["arms"]["plain"]["environment"]["FLUX_SPEC_MODE"] = "auto"
            mutations.append(bad_profile)
            bad_target = copy.deepcopy(value)
            index = bad_target["arms"]["k5"]["argv"].index("--max-prefill-tokens") + 1
            bad_target["arms"]["k5"]["argv"][index] = "8192"
            mutations.append(bad_target)
            bad_max_seq = copy.deepcopy(value)
            index = bad_max_seq["arms"]["plain"]["argv"].index("--max-seq-len") + 1
            bad_max_seq["arms"]["plain"]["argv"][index] = "262145"
            mutations.append(bad_max_seq)
            bad_eager = copy.deepcopy(value)
            del bad_eager["arms"][Q.EAGER_CHURN_ARM]["environment"][
                "ATLAS_DFLASH_DEBUG_NO_GRAPH"
            ]
            mutations.append(bad_eager)
            bad_tp = copy.deepcopy(value)
            index = bad_tp["arms"][Q.EAGER_CHURN_ARM]["argv"].index("--tp-size") + 1
            bad_tp["arms"][Q.EAGER_CHURN_ARM]["argv"][index] = "2"
            mutations.append(bad_tp)
            bad_hss = copy.deepcopy(value)
            bad_hss["arms"][Q.EAGER_CHURN_ARM]["argv"].append("--high-speed-swap")
            mutations.append(bad_hss)
            for number, mutation in enumerate(mutations):
                with self.subTest(number=number):
                    path.write_text(json.dumps(mutation), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        Q.load_manifest(path, verify_files=False)

    def test_manifest_rejects_weak_or_changed_workload(self):
        with tempfile.TemporaryDirectory() as temp:
            path, value = self.make_manifest(temp)
            value["request_contract"]["prompt_token_counts"] = [2_049]
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fixed issue-13 workload"):
                Q.load_manifest(path, verify_files=False)

    def test_strict_json_rejects_duplicate_keys_in_manifest_api_and_inventory(self):
        with self.assertRaisesRegex(ValueError, "duplicate JSON key 'nested'"):
            Q.strict_json_loads('{"outer":{"nested":1,"nested":2}}', "fixture")
        for value in ("NaN", "Infinity", "-Infinity", "1e9999", "-1e9999"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "non-finite JSON number"):
                    Q.strict_json_loads(f'{{"value":{value}}}', "fixture")

        class DuplicateApiClient(Q.HttpClient):
            def request(self, method, route, body=None):
                return 200, b'{"model":"a","model":"b"}'

        with self.assertRaisesRegex(ValueError, "duplicate JSON key 'model'"):
            DuplicateApiClient("http://127.0.0.1:1", 1).json("GET", "/fixture")

        class NonFiniteApiClient(Q.HttpClient):
            def request(self, method, route, body=None):
                return 200, b'{"value":NaN}'

        with self.assertRaisesRegex(ValueError, "non-finite JSON number"):
            NonFiniteApiClient("http://127.0.0.1:1", 1).json("GET", "/fixture")

        with tempfile.TemporaryDirectory() as temp:
            manifest_path, _ = self.make_manifest(temp)
            manifest_path.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON key 'schema'"):
                Q.load_manifest(manifest_path, verify_files=False)
            manifest_path.write_text('{"schema":NaN}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-finite JSON number"):
                Q.load_manifest(manifest_path, verify_files=False)

            inventory_path = Path(temp) / "inventory.json"
            inventory_path.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
            inventory_bytes = inventory_path.read_bytes()
            inventory_provenance = {
                "checkpoint_manifest_sha256": Q.sha256_bytes(inventory_bytes),
                "checkpoint_id": "checkpoint",
                "checkpoint_revision": "a" * 40,
            }
            with self.assertRaisesRegex(ValueError, "duplicate JSON key 'schema'"):
                Q.validate_checkpoint_inventory(
                    inventory_path,
                    Path(temp) / "checkpoint",
                    inventory_provenance,
                    verify_files=False,
                )
            receipt_path = Path(temp) / "source-receipt.json"
            receipt_path.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON key 'schema'"):
                Q.validate_source_to_binary_receipt(
                    receipt_path,
                    {
                        "source_to_binary_receipt_path": str(receipt_path),
                        "source_to_binary_receipt_sha256": Q.sha256_file(receipt_path),
                        "release_version": "v0.9.13-r15",
                        "source_revision": "a" * 40,
                        "source_bundle_manifest_sha256": "b" * 64,
                        "source_bundle_sha256": "c" * 64,
                        "binary_sha256": "d" * 64,
                    },
                    verify_files=False,
                )

    def test_full_source_closure_rejects_handwritten_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            values, _, receipt_path = full_source_provenance(temp)
            Q.validate_source_to_binary_receipt(receipt_path, values, True)
            forged = {
                "schema": Q.SOURCE_TO_BINARY_SCHEMA,
                "verdict": "PASS",
                "release_version": Q.RELEASE_VERSION,
                "release_commit": values["source_revision"],
                "source_bundle_manifest_sha256": values[
                    "source_bundle_manifest_sha256"
                ],
                "source_bundle_sha256": values["source_bundle_sha256"],
                "binary": {"sha256": values["binary_sha256"]},
            }
            receipt_path.write_bytes(Q.canonical_bytes(forged))
            values["source_to_binary_receipt_sha256"] = Q.sha256_file(receipt_path)
            with self.assertRaisesRegex(ValueError, "differs from verified inputs"):
                Q.validate_source_to_binary_receipt(receipt_path, values, True)

    def test_full_source_closure_rejects_missing_or_dirty_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            values, source, receipt_path = full_source_provenance(temp)
            for missing in (
                "source_tree",
                "source_root_path",
                "source_bundle_manifest_path",
                "source_bundle_manifest_sha256",
                "source_bundle_path",
                "source_bundle_sha256",
                "build_receipt_path",
                "build_receipt_sha256",
            ):
                incomplete = dict(values)
                del incomplete[missing]
                with self.subTest(missing=missing):
                    with self.assertRaisesRegex(ValueError, "provenance is incomplete"):
                        Q.validate_source_to_binary_receipt(
                            receipt_path, incomplete, True
                        )

            (source / "untracked-release-input").write_text("dirty\n")
            with self.assertRaisesRegex(ValueError, "dirty"):
                Q.validate_source_to_binary_receipt(receipt_path, values, True)

    def test_manifest_rejects_malformed_release_identities(self):
        with tempfile.TemporaryDirectory() as temp:
            path, value = self.make_manifest(temp)
            for release in ("v0.9.13-r0", "v0.9.13-rc.r17.20260230", "v0.9.13-r16\n", ""):
                mutation = copy.deepcopy(value)
                mutation["provenance"]["release_version"] = release
                path.write_text(json.dumps(mutation), encoding="utf-8")
                with self.subTest(release=release):
                    with self.assertRaises(ValueError):
                        Q.load_manifest(path, verify_files=False)

    def test_manifest_snapshot_revalidation_rejects_toctou(self):
        manifest = {
            "schema": Q.MANIFEST_SCHEMA,
            "_manifest_source_sha256": "a" * 64,
            "_manifest_canonical_sha256": "b" * 64,
            "_common_env": {"PIN": "1"},
        }
        changed = copy.deepcopy(manifest)
        changed["_manifest_source_sha256"] = "c" * 64
        with mock.patch.object(Q, "load_manifest", return_value=changed):
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                Q.revalidate_manifest_and_artifacts(
                    manifest, Path("/manifest.json"), "before k5 arm"
                )
        changed = copy.deepcopy(manifest)
        changed["_common_env"] = {"PIN": "0"}
        with mock.patch.object(Q, "load_manifest", return_value=changed):
            with self.assertRaisesRegex(ValueError, "common environment changed"):
                Q.revalidate_manifest_and_artifacts(
                    manifest, Path("/manifest.json"), "before final seal"
                )

    def test_live_runner_revalidates_before_each_arm_and_final_seal(self):
        manifest = {
            "arms": {name: {} for name in Q.ARM_NAMES},
            "_common_env": {},
            "_manifest_source_sha256": "a" * 64,
            "_manifest_canonical_sha256": "b" * 64,
            "provenance": {},
        }
        bundle = {"manifest": {}, "manifest_sha256": "c" * 64}
        paired = [{"case_id": case_id} for case_id in Q.PAIRED_CASE_IDS]
        eager = [{"case_id": case_id} for case_id in Q.GRAPH_DISABLED_CASE_IDS]
        arms = [
            ({"results": paired, "endpoint_admission": {"passed": True}}, bundle),
            ({"results": paired}, bundle),
            (
                {
                    "results": eager,
                    "rollback_slot_reuse": {"passed": True},
                },
                bundle,
            ),
        ]
        phases = []

        def record_revalidation(_manifest, _path, phase):
            phases.append(phase)

        with tempfile.TemporaryDirectory() as temp:
            receipt_path = Path(temp) / "receipt.json"
            with (
                mock.patch.object(Q, "run_arm", side_effect=arms),
                mock.patch.object(
                    Q, "compare_arms", return_value={"exact_matches": True}
                ),
                mock.patch.object(
                    Q,
                    "revalidate_manifest_and_artifacts",
                    side_effect=record_revalidation,
                ),
            ):
                receipt = Q.run_live(
                    manifest,
                    Path("/manifest.json"),
                    receipt_path,
                    timeout=1,
                )
        self.assertEqual(
            phases,
            [
                "before plain arm",
                "before k5 arm",
                "before forced-eager arm",
                "before final seal",
            ],
        )
        self.assertEqual(receipt["endpoint_admission"], {"passed": True})

    def test_manifest_rejects_common_spec_controls_and_missing_receipt_logging(self):
        for line in (
            "FLUX_SPEC_MODE=none\n",
            "FLUX_TP_SIZE=2\n",
            "ATLAS_DFLASH_DEBUG_NO_GRAPH=1\n",
            "ATLAS_QWEN4_QSA_ROUTE_TELEMETRY=1\n",
            "RUST_LOG=info\n",
        ):
            with self.subTest(line=line):
                with tempfile.TemporaryDirectory() as temp:
                    path, value = self.make_manifest(temp)
                    env_path = Path(value["common_env_file"])
                    if line.startswith(("FLUX", "ATLAS")):
                        env_path.write_text(common_env_text() + line, encoding="utf-8")
                    else:
                        text = common_env_text().replace(
                            "RUST_LOG=info,spark::scheduler::spec_accept_log=debug\n",
                            line,
                        )
                        env_path.write_text(text, encoding="utf-8")
                    value["common_env_sha256"] = Q.sha256_file(env_path)
                    path.write_text(json.dumps(value), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        Q.load_manifest(path, verify_files=False)

    def test_log_contract_proves_route_engagement(self):
        Q.validate_log_contract("plain", Q.PLAIN_SCHEDULER_MARKER)
        evidence = Q.validate_log_contract(
            "k5",
            "\n".join(
                [
                    Q.K5_SCHEDULER_MARKER,
                    Q.K5_CONSTRUCTION_MARKER,
                    Q.K5_QSA_VERIFY_ROWS_MARKER,
                    Q.K5_VERIFY_GRAPH_CAPTURE_MARKER,
                    mtp_frame(),
                    spec_cycle(1, 2_050),
                ]
            ),
        )
        self.assertEqual(evidence["native_mtp_frames"], 1)
        self.assertEqual(evidence["completed_spec_cycles"], 1)
        self.assertTrue(evidence["verify_graph_route_observed"])
        without_graph = "\n".join(
            [
                Q.K5_SCHEDULER_MARKER,
                Q.K5_CONSTRUCTION_MARKER,
                Q.K5_QSA_VERIFY_ROWS_MARKER,
                mtp_frame(),
                spec_cycle(1, 2_050),
            ]
        )
        with self.assertRaisesRegex(ValueError, "capture/use marker"):
            Q.validate_log_contract("k5", without_graph)
        with self.assertRaisesRegex(ValueError, "MTP_ACCEPT frame"):
            Q.validate_log_contract(
                "k5",
                "\n".join(
                    [
                        Q.K5_SCHEDULER_MARKER,
                        Q.K5_CONSTRUCTION_MARKER,
                        Q.K5_QSA_VERIFY_ROWS_MARKER,
                    ]
                ),
            )
        with self.assertRaisesRegex(ValueError, "unexpectedly"):
            Q.validate_log_contract(
                "plain", Q.PLAIN_SCHEDULER_MARKER + Q.K5_VERIFY_MARKER
            )

    def test_completed_spec_cycle_parser_is_fail_closed(self):
        records = Q.parse_completed_spec_cycles(spec_cycle(0xAB, 2_050, accepted=3))
        self.assertEqual(records[0]["req"], "00000000000000ab")
        self.assertEqual(records[0]["emitted"], 4)
        for mutation in (
            spec_cycle(1, 2_050, schema=1),
            spec_cycle(1, 2_050, k=4),
            spec_cycle(1, 2_050, gamma=3),
            spec_cycle(1, 2_050).replace("finalize_host_ns=9", "finalize_host_ns=x"),
        ):
            with self.subTest(mutation=mutation):
                with self.assertRaises(ValueError):
                    Q.parse_completed_spec_cycles(mutation)

    def test_request_scoped_mtp_receipt_is_required_for_every_k5_request(self):
        for index, prompt_tokens in enumerate(
            (*range(2_043, 2_051), Q.FULL_ENDPOINT_CASE_SPEC[1]), 1
        ):
            with self.subTest(prompt_tokens=prompt_tokens):
                evidence = Q.validate_request_log_delta(
                    request_log(prompt_tokens, req=index),
                    prompt_tokens,
                    require_native_mtp=True,
                )
                self.assertEqual(evidence["scheduler_dispatches"], 1)
                self.assertEqual(evidence["native_mtp_frames"], 1)
                self.assertEqual(evidence["runtime_request_ids"], [f"{index:016x}"])
        dispatch_only = "Prefilling: 2043 prompt tokens, max_tokens=96"
        with self.assertRaisesRegex(ValueError, "lacks an MTP verify frame"):
            Q.validate_request_log_delta(dispatch_only, 2_043, True)
        with self.assertRaisesRegex(ValueError, "exactly one runtime request ID"):
            Q.validate_request_log_delta(
                request_log(2_043, req=1) + "\n" + spec_cycle(2, 2_046),
                2_043,
                True,
            )
        with self.assertRaisesRegex(ValueError, "first-speculative geometry"):
            Q.validate_request_log_delta(
                "\n".join(
                    (
                        "Prefilling: 2043 prompt tokens, max_tokens=96",
                        mtp_frame(2_045),
                        spec_cycle(1, 2_045),
                    )
                ),
                2_043,
                True,
            )

    def test_eager_request_consumes_exact_qsa_route_frames(self):
        logs = request_log(2_043, qsa_telemetry=True)
        evidence = Q.validate_request_log_delta(
            logs,
            2_043,
            require_native_mtp=True,
            require_qsa_route_telemetry=True,
        )
        route = evidence["qsa_route_telemetry"]
        self.assertEqual(route["frames"], 1)
        self.assertEqual(route["receipts"], 12)
        self.assertEqual(route["exact_layer_ids"], list(range(12)))
        self.assertEqual(route["frame_receipts"][0]["seq_len"], 2_044)
        self.assertEqual(route["frame_receipts"][0]["route_counts"]["row-serial"], 12)

        mutations = {
            "missing-layer": "\n".join(logs.splitlines()[:-3] + logs.splitlines()[-2:]),
            "wrong-seq": logs.replace("seq_len=2044 layer=11", "seq_len=2045 layer=11"),
            "bad-row-accounting": logs.replace(
                "dense_paged_rows=5", "dense_paged_rows=4", 1
            ),
            "bad-total": logs.replace(
                "row_serial_frame_total=2", "row_serial_frame_total=3", 1
            ),
            "duplicate-layer": logs.replace("layer=11", "layer=10", 1),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    Q.validate_request_log_delta(
                        mutation,
                        2_043,
                        require_native_mtp=True,
                        require_qsa_route_telemetry=True,
                    )

        with self.assertRaisesRegex(ValueError, "escaped"):
            Q.validate_request_log_delta(logs, 2_043, require_native_mtp=True)
        with self.assertRaisesRegex(ValueError, "lacks QSA route receipts"):
            Q.validate_request_log_delta(
                request_log(2_043),
                2_043,
                require_native_mtp=True,
                require_qsa_route_telemetry=True,
            )

    def test_k5_draft_policy_defaults_to_fixed_and_keeps_strict_shapes(self):
        self.assertEqual(Q.K5_DRAFT_POLICY, Q.K5_FIXED_POLICY)
        self.assertEqual(Q.parse_completed_spec_cycles(spec_cycle(1, 2_050))[0]["gamma"], 4)
        self.assertEqual(Q.parse_mtp_accept_frames(mtp_frame())[0]["pattern"], "YYNN")
        for gamma in (1, 2, 3):
            with self.subTest(gamma=gamma), self.assertRaisesRegex(ValueError, "frozen K5"):
                Q.parse_completed_spec_cycles(
                    spec_cycle(1, 2_050, accepted=0, k=gamma + 1, gamma=gamma)
                )
        for pattern in ("Y", "YN", "YNN"):
            with self.subTest(pattern=pattern), self.assertRaisesRegex(ValueError, "native K5"):
                Q.parse_mtp_accept_frames(mtp_frame(accepted=0, pattern=pattern))
        for rows in (2, 3, 4):
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, "native K=5"):
                Q.parse_qsa_route_receipts("\n".join(qsa_route_frame(1, 2_050, rows=rows)))
        # A fixed-K5 request without a completed cycle still fails closed.
        with self.assertRaisesRegex(ValueError, "SPEC_CYCLE_V2"):
            Q.validate_request_log_delta(
                "Prefilling: 2043 prompt tokens, max_tokens=96\n" + mtp_frame(2_044),
                2_043,
                True,
            )


    def test_overflow_rejection_requires_exact_error_and_zero_dispatch(self):
        value = {
            "error": {
                "message": (
                    "Context too long: 262144 prompt + 1 requested output tokens "
                    "exceeds max_seq_len 262144"
                ),
                "type": "invalid_request_error",
                "param": None,
                "code": None,
            }
        }
        result = Q.validate_context_length_rejection(400, value, 262_144, 1)
        self.assertTrue(result["context_length_rejection"])
        before_values = rejection_metric_values()
        after_values = dict(before_values)
        http_request_id = "qualify_overflow_" + "a" * 32
        self.assertEqual(
            Q.validate_no_scheduler_dispatch(
                "HTTP request rejected",
                metric_receipt(before_values),
                metric_receipt(after_values),
                http_request_id,
            )["scheduler_dispatch_markers"],
            0,
        )
        for status, mutation in (
            (200, value),
            (400, {"error": {**value["error"], "code": "another_error"}}),
        ):
            with self.subTest(status=status, mutation=mutation):
                with self.assertRaises(ValueError):
                    Q.validate_context_length_rejection(status, mutation, 262_144, 1)
        for marker in (
            "Chunked prefill start: 262144 prompt tokens,",
            mtp_frame(262_144),
            spec_cycle(1, 262_144),
        ):
            with self.subTest(marker=marker):
                with self.assertRaisesRegex(ValueError, "scheduler dispatch"):
                    Q.validate_no_scheduler_dispatch(
                        marker,
                        metric_receipt(before_values),
                        metric_receipt(after_values),
                        http_request_id,
                    )
        scheduler_moved = dict(after_values)
        scheduler_moved["atlas_prefill_progress_chunks_total"] += 1
        with self.assertRaisesRegex(ValueError, "scheduler prefill metrics"):
            Q.validate_no_scheduler_dispatch(
                "HTTP request rejected",
                metric_receipt(before_values),
                metric_receipt(scheduler_moved),
                http_request_id,
            )
        with self.assertRaisesRegex(ValueError, "scheduler request IDs"):
            Q.validate_no_scheduler_dispatch(
                "rejected req=0000000000000001",
                metric_receipt(before_values),
                metric_receipt(after_values),
                http_request_id,
            )
        with self.assertRaisesRegex(ValueError, "correlated HTTP request ID"):
            Q.validate_no_scheduler_dispatch(
                "HTTP request rejected",
                metric_receipt(before_values),
                metric_receipt(after_values),
                "bad",
            )

        self.assertEqual(
            Q.validate_correlated_request_id(
                [("X-Request-Id", http_request_id)], http_request_id
            ),
            http_request_id,
        )
        for headers in (
            [],
            [("x-request-id", "wrong")],
            [
                ("x-request-id", http_request_id),
                ("x-request-id", http_request_id),
            ],
        ):
            with self.subTest(headers=headers):
                with self.assertRaisesRegex(ValueError, "echo exactly one"):
                    Q.validate_correlated_request_id(headers, http_request_id)

    def test_rejection_metrics_parser_is_exact_and_complete(self):
        values = rejection_metric_values()
        self.assertEqual(
            Q.REJECTION_METRIC_NAMES,
            (
                "atlas_prefill_progress_chunks_total",
                "atlas_prefill_progress_tokens_total",
            ),
        )
        self.assertEqual(
            Q.parse_prometheus_integer_metrics(metrics_payload(values)), values
        )
        metrics_source = (
            Path(__file__).resolve().parents[2]
            / "crates"
            / "spark-server"
            / "src"
            / "metrics.rs"
        ).read_text(encoding="utf-8")
        self.assertIn("pub fn record_prefill_progress", metrics_source)
        for name in Q.REJECTION_METRIC_NAMES:
            self.assertIn(f'"{name}"', metrics_source)
        with self.assertRaisesRegex(ValueError, "repeats unlabelled sample"):
            Q.parse_prometheus_integer_metrics(
                metrics_payload(values) + b"atlas_prefill_progress_chunks_total 99\n"
            )
        with self.assertRaisesRegex(ValueError, "lacks required samples"):
            Q.parse_prometheus_integer_metrics(b"atlas_requests_total 1\n")
        for invalid in (b"NaN", b"+Inf", b"1.5", b"-1"):
            mutated = dict(values)
            mutated["atlas_prefill_progress_chunks_total"] = invalid.decode()
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "non-negative integer"):
                    Q.parse_prometheus_integer_metrics(metrics_payload(mutated))

    def test_post_canary_metrics_and_final_census_reject_false_passes(self):
        before_values = rejection_metric_values()
        after_values = dict(before_values)
        after_values["atlas_prefill_progress_chunks_total"] += 1
        after_values["atlas_prefill_progress_tokens_total"] += 2_017
        evidence = Q.endpoint_gate.validate_post_canary_metrics(
            metric_receipt(before_values),
            metric_receipt(after_values),
            prompt_tokens=2_017,
            prefill_chunks=1,
        )
        self.assertTrue(evidence["completion_canary_prefill_accounted"])
        drift = dict(after_values)
        drift["atlas_prefill_progress_tokens_total"] -= 1
        with self.assertRaisesRegex(ValueError, "metric deltas differ"):
            Q.endpoint_gate.validate_post_canary_metrics(
                metric_receipt(before_values),
                metric_receipt(drift),
                prompt_tokens=2_017,
                prefill_chunks=1,
            )

        logs = "\n".join(
            [
                "Prefilling: 2017 prompt tokens, max_tokens=96",
                "Chunked prefill start: 262143 prompt tokens, chunk_size=16384, max_tokens=1",
            ]
        )
        census = Q.endpoint_gate.validate_final_dispatch_census(
            logs, {2_017: 1, 262_143: 1}, rejected_prompt_tokens=262_144
        )
        self.assertTrue(census["whole_log_census"])
        late_dispatch = (
            logs + "\nChunked prefill start: 262144 prompt tokens, "
            "chunk_size=16384, max_tokens=1"
        )
        with self.assertRaisesRegex(ValueError, "final dispatch census"):
            Q.endpoint_gate.validate_final_dispatch_census(
                late_dispatch,
                {2_017: 1, 262_143: 1},
                rejected_prompt_tokens=262_144,
            )

    def test_live_admission_sequence_requires_post_rejection_health_canary(self):
        bundle = Q.build_request_bundle(TokenizeClient(), "qad5500")
        cases = {case["prompt_tokens"]: case for case in bundle["manifest"]["cases"]}

        with tempfile.TemporaryDirectory() as temp:
            log_path = Path(temp) / "server.log"
            log_path.write_bytes(b"")

            class Client:
                def __init__(
                    self,
                    inject_rejected_dispatch=False,
                    inject_metric_only=False,
                    omit_request_id_echo=False,
                ):
                    self.inject_rejected_dispatch = inject_rejected_dispatch
                    self.inject_metric_only = inject_metric_only
                    self.omit_request_id_echo = omit_request_id_echo
                    self.health_calls = 0
                    self.metric_calls = 0
                    self.metrics = rejection_metric_values()

                def request(self, method, route, body=None):
                    if route == "/metrics":
                        self.metric_calls += 1
                        return 200, metrics_payload(self.metrics)
                    self.assert_health_request(method, route)
                    self.health_calls += 1
                    return 200, b"healthy"

                def assert_health_request(self, method, route):
                    if method != "GET" or route != "/health":
                        raise AssertionError(f"unexpected request {method} {route}")

                def json(self, method, route, body):
                    prompt_tokens = len(body["prompt_token_ids"])
                    if prompt_tokens == 262_144:
                        if self.inject_rejected_dispatch:
                            with log_path.open("ab") as handle:
                                handle.write(
                                    b"Prefilling: 262144 prompt tokens, max_tokens=1\n"
                                )
                        if self.inject_metric_only:
                            self.metrics["atlas_prefill_progress_chunks_total"] += 1
                        value = {
                            "error": {
                                "message": (
                                    "Context too long: 262144 prompt + 1 requested "
                                    "output tokens exceeds max_seq_len 262144"
                                ),
                                "type": "invalid_request_error",
                                "param": None,
                                "code": None,
                            }
                        }
                        return 400, value, Q.canonical_bytes(value)
                    with log_path.open("ab") as handle:
                        handle.write(
                            f"Prefilling: {prompt_tokens} prompt tokens, max_tokens={body['max_tokens']}\n".encode()
                        )
                    case = cases[prompt_tokens]
                    if prompt_tokens == 262_143:
                        value = response(
                            prompt_tokens,
                            "boundary token",
                            completion_tokens=1,
                            finish_reason="length",
                            token_ids=[777],
                        )
                    else:
                        value = response(prompt_tokens, case["expected_answer"])
                    self.metrics["atlas_prefill_progress_chunks_total"] += (
                        prompt_tokens + 16_384 - 1
                    ) // 16_384
                    self.metrics["atlas_prefill_progress_tokens_total"] += prompt_tokens
                    return 200, value, Q.canonical_bytes(value)

                def correlated_json(self, method, route, body, request_id):
                    status, value, raw = self.json(method, route, body)
                    headers = []
                    if not self.omit_request_id_echo:
                        headers.append(("x-request-id", request_id))
                    return status, value, raw, headers

            client = Client()
            receipt = Q.execute_endpoint_admission_probes(
                client, bundle, Q.RequestLogCapture(log_path, settle_seconds=0)
            )
            self.assertEqual(client.health_calls, 1)
            self.assertEqual(client.metric_calls, 3)
            self.assertEqual(
                receipt["rejected_262144_plus_1"]["scheduler_evidence"][
                    "scheduler_dispatch_markers"
                ],
                0,
            )
            self.assertEqual(
                receipt["post_rejection_completion_canary"]["case_id"],
                Q.ROLLBACK_SENTINEL_CASE_ID,
            )

            log_path.write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "scheduler dispatch"):
                Q.execute_endpoint_admission_probes(
                    Client(inject_rejected_dispatch=True),
                    bundle,
                    Q.RequestLogCapture(log_path, settle_seconds=0),
                )

            log_path.write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "scheduler prefill metrics"):
                Q.execute_endpoint_admission_probes(
                    Client(inject_metric_only=True),
                    bundle,
                    Q.RequestLogCapture(log_path, settle_seconds=0),
                )

            log_path.write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "echo exactly one"):
                Q.execute_endpoint_admission_probes(
                    Client(omit_request_id_echo=True),
                    bundle,
                    Q.RequestLogCapture(log_path, settle_seconds=0),
                )

    def test_eager_churn_requires_completed_partial_between_distinct_canaries(self):
        prefix = [
            Q.K5_SCHEDULER_MARKER,
            Q.K5_CONSTRUCTION_MARKER,
            Q.K5_QSA_VERIFY_ROWS_MARKER,
            Q.QSA_ROUTE_TELEMETRY_ENGAGED_MARKER,
            *qsa_route_frame(1, 2_050),
            mtp_frame(),
        ]
        cycles = [
            spec_cycle(1, 2_018, accepted=4),
            spec_cycle(1, 2_023, accepted=4),
            spec_cycle(1, 2_051, accepted=2),
            spec_cycle(2, 2_051, accepted=2),
            spec_cycle(3, 2_018, accepted=4),
        ]
        evidence = Q.validate_log_contract(
            Q.EAGER_CHURN_ARM, "\n".join([*prefix, *cycles])
        )
        sequence = evidence["rollback_slot_log_sequence"]
        self.assertEqual(sequence["sentinel_before_req"], "0000000000000001")
        self.assertEqual(sequence["trigger_req"], "0000000000000002")
        self.assertEqual(sequence["sentinel_after_req"], "0000000000000003")
        self.assertEqual(sequence["distinct_request_ids"], 3)
        self.assertEqual(sequence["completed_partial_accept_rollbacks"], 1)
        with self.assertRaisesRegex(ValueError, "zero baseline"):
            Q.validate_log_contract(
                Q.EAGER_CHURN_ARM,
                "\n".join([*prefix, *cycles]).replace(
                    "row_serial_frame_total=1", "row_serial_frame_total=7", 1
                ),
            )
        for bad_cycles in (
            [
                spec_cycle(1, 2_018, accepted=4),
                spec_cycle(2, 2_051, accepted=4),
                spec_cycle(3, 2_018, accepted=4),
            ],
            [spec_cycle(1, 2_018), spec_cycle(2, 2_051)],
        ):
            with self.subTest(bad_cycles=bad_cycles):
                with self.assertRaises(ValueError):
                    Q.validate_log_contract(
                        Q.EAGER_CHURN_ARM, "\n".join([*prefix, *bad_cycles])
                    )
        with self.assertRaisesRegex(ValueError, "captured"):
            Q.validate_log_contract(
                Q.EAGER_CHURN_ARM,
                "\n".join([*prefix, *cycles, Q.K5_VERIFY_GRAPH_CAPTURE_MARKER]),
            )

    def test_dry_run_names_pair_fields_but_claims_no_runtime_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            path, _ = self.make_manifest(temp)
            manifest = Q.load_manifest(path, verify_files=False)
            plan = Q.dry_run_plan(manifest)
            self.assertEqual(plan["fresh_processes"], list(Q.ARM_NAMES))
            self.assertEqual(plan["prompt_token_counts"][0], 2_017)
            self.assertEqual(plan["full_endpoint"]["total_tokens"], 262_144)
            self.assertEqual(
                plan["graph_disabled_boundary_case_ids"],
                list(Q.GRAPH_DISABLED_CASE_IDS),
            )
            self.assertIn("completion_tokens", plan["required_pair_fields"])
            self.assertIn("completion_token_ids", plan["required_pair_fields"])
            self.assertEqual(
                plan["eager_churn_telemetry_hook"],
                {
                    "arm": Q.EAGER_CHURN_ARM,
                    "environment": Q.ARM_ENVIRONMENTS[Q.EAGER_CHURN_ARM],
                    "world_size": 1,
                    "tensor_parallel_size": 1,
                    "expert_parallel_size": 1,
                    "concurrency": 1,
                    "high_speed_swap": False,
                    "source_telemetry_overlay_included": True,
                    "engagement_marker": Q.QSA_ROUTE_TELEMETRY_ENGAGED_MARKER,
                    "receipt_marker": Q.QSA_ROUTE_RECEIPT_MARKER.strip(),
                    "exact_layer_ids": list(Q.QSA_FULL_ATTENTION_LAYER_IDS),
                    "request_scoped": True,
                },
            )
            self.assertNotIn("status", plan)

    def test_checked_in_example_names_the_fixed_pair_contract(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "examples"
            / "qwen4-mtp-long-context-pair-launch.example.json"
        )
        value = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(value["schema"], Q.MANIFEST_SCHEMA)
        self.assertEqual(set(value["provenance"]), set(provenance()))
        self.assertEqual(value["provenance"]["release_version"], "v0.9.13-r15")
        self.assertEqual(value["model"], "qwen38-qad5500-r15")
        self.assertEqual(value["request_contract"], Q.launch_request_contract())
        self.assertEqual(
            value["arms"]["plain"]["environment"], Q.ARM_ENVIRONMENTS["plain"]
        )
        self.assertEqual(value["arms"]["k5"]["environment"], Q.ARM_ENVIRONMENTS["k5"])
        self.assertEqual(
            value["arms"][Q.EAGER_CHURN_ARM]["environment"],
            Q.ARM_ENVIRONMENTS[Q.EAGER_CHURN_ARM],
        )
        self.assertEqual(
            Q.strip_k5_only_argv(value["arms"]["k5"]["argv"]),
            value["arms"]["plain"]["argv"],
        )
        self.assertEqual(
            Q.strip_k5_only_argv(value["arms"][Q.EAGER_CHURN_ARM]["argv"]),
            value["arms"]["plain"]["argv"],
        )


class GatedK5PolicyTests(unittest.TestCase):
    def setUp(self):
        Q.configure_k5_draft_policy(Q.K5_GATED_POLICY)
        self.addCleanup(Q.configure_k5_draft_policy, Q.K5_FIXED_POLICY)

    def test_gated_fixture_parses_every_short_width(self):
        for gamma in (1, 2, 3, 4):
            with self.subTest(gamma=gamma):
                record = Q.parse_completed_spec_cycles(
                    spec_cycle(1, 2_050, accepted=gamma - 1, k=gamma + 1, gamma=gamma)
                )[0]
                self.assertEqual((record["k"], record["gamma"]), (gamma + 1, gamma))
        for rows in (2, 3, 4, 5):
            pattern = "Y" * (rows - 2) + "N"
            first = 2_044
            lines = [
                "Prefilling: 2043 prompt tokens, max_tokens=96",
                *qsa_route_frame(1, first, rows=rows),
                mtp_frame(first, rows - 2, pattern),
            ]
            # Only the full-width generic route emits SPEC_CYCLE_V2.
            if rows == 5:
                lines.append(spec_cycle(7, first, accepted=3))
            with self.subTest(rows=rows):
                evidence = Q.validate_request_log_delta(
                    "\n".join(lines), 2_043, True, require_qsa_route_telemetry=True
                )
                frame = evidence["qsa_route_telemetry"]["frame_receipts"][0]
                self.assertEqual(frame["rows"], rows)
                self.assertEqual(evidence["runtime_request_ids"],
                                 ["0000000000000007"] if rows == 5 else [])

    def test_gated_parsers_refuse_malformed_shapes(self):
        for line in (
            spec_cycle(1, 2_050, accepted=0, k=1, gamma=0),
            spec_cycle(1, 2_050, accepted=0, k=6, gamma=5),
            spec_cycle(1, 2_050, accepted=0, k=5, gamma=3),
            spec_cycle(1, 2_050, accepted=3, k=3, gamma=2),
        ):
            with self.subTest(line=line), self.assertRaises(ValueError):
                Q.parse_completed_spec_cycles(line)
        for line in (
            mtp_frame(accepted=0, pattern=""),
            mtp_frame(accepted=0, pattern="YYYYY"),
            mtp_frame(accepted=3, pattern="YN"),
            mtp_frame(accepted=0, pattern="YX"),
            mtp_frame(accepted=0, pattern="YN").replace("num_drafts=4", "num_drafts=2"),
        ):
            with self.subTest(line=line), self.assertRaises(ValueError):
                Q.parse_mtp_accept_frames(line)
        for rows in (0, 1, 6):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                Q.parse_qsa_route_receipts("\n".join(qsa_route_frame(1, 2_050, rows=rows)))
        # Route rows must equal the paired frame's verified drafts plus one.
        mismatch = "\n".join([
            "Prefilling: 2043 prompt tokens, max_tokens=96",
            *qsa_route_frame(1, 2_044, rows=4),
            mtp_frame(2_044, 0, "YN"),
        ])
        with self.assertRaisesRegex(ValueError, "MTP frame geometry"):
            Q.validate_request_log_delta(mismatch, 2_043, True, require_qsa_route_telemetry=True)
        # A full-width first frame must still bind its completed cycle.
        with self.assertRaisesRegex(ValueError, "SPEC_CYCLE_V2"):
            Q.validate_request_log_delta(
                "Prefilling: 2043 prompt tokens, max_tokens=96\n" + mtp_frame(2_044),
                2_043,
                True,
            )
        for policy in ("", "gated", "native-mtp-k5", None):
            with self.subTest(policy=policy), self.assertRaisesRegex(ValueError, "draft policy"):
                Q.configure_k5_draft_policy(policy)
        self.assertEqual(Q.K5_DRAFT_POLICY, Q.K5_GATED_POLICY)


if __name__ == "__main__":
    unittest.main()
