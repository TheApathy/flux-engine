# SPDX-License-Identifier: AGPL-3.0-only

import base64
import copy
import importlib.util
import json
import socket
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

MODULE_PATH = Path(__file__).resolve().parents[1] / "qwen4_native_262k_qualify.py"
SPEC = importlib.util.spec_from_file_location("qwen4_native_262k_qualify", MODULE_PATH)
assert SPEC and SPEC.loader
Q = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(Q)


def calibrated_components():
    return {
        "head": [
            248045,
            846,
            198,
            5115,
            220,
            15,
            17593,
            11,
            5604,
            1970,
            326,
            15,
            15,
            12,
            15,
            15,
            15,
            15,
            15,
            13,
            198,
        ],
        "filler": [
            94533,
            17593,
            4202,
            25,
            279,
            29128,
            45366,
            11140,
            34044,
            11,
            8928,
            11,
            78902,
            321,
            8019,
            292,
            912,
            13,
            198,
        ],
        "tail": [
            198,
            5423,
            279,
            2250,
            23513,
            1440,
            10250,
            11,
            28853,
            79940,
            11,
            417,
            20069,
            11,
            78990,
            4680,
            13,
            14122,
            1070,
            6681,
            26,
            874,
            15673,
            13,
            248046,
            198,
            248045,
            74455,
            198,
            248068,
            271,
            248069,
            271,
        ],
        "ALPHA": [
            198,
            71981,
            1440,
            767,
            79940,
            25,
            8029,
            12,
            15,
            22,
            18,
            16,
            23,
            17,
            21,
            198,
        ],
        "BETA": [
            198,
            71981,
            1440,
            767,
            417,
            20069,
            25,
            13053,
            12,
            15,
            19,
            24,
            17,
            16,
            22,
            20,
            198,
        ],
        "GAMMA": [
            198,
            71981,
            1440,
            767,
            78990,
            4680,
            25,
            20956,
            12,
            15,
            23,
            21,
            18,
            17,
            15,
            19,
            198,
        ],
    }


class Native262KContractTests(unittest.TestCase):
    def test_checked_in_example_is_pinned_to_r15_release(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "examples"
            / "qwen4-native-262k-binary-launch.example.json"
        )
        value = json.loads(path.read_text(encoding="utf-8"))
        provenance = value["provenance"]
        self.assertEqual(provenance["release_version"], "v0.9.13-r15")
        self.assertEqual(value["model"], "qwen38-qad5500-r15")
        self.assertIn("v0.9.13-r15", provenance["source_bundle_path"])

    def test_r15_runbook_pins_both_provenance_commands(self):
        path = Path(__file__).resolve().parents[1] / "QWEN4_R9_QUALIFICATION.md"
        runbook = path.read_text(encoding="utf-8")
        self.assertEqual(runbook.count("--release-version v0.9.13-r15"), 2)
        self.assertNotIn("v0.9.13-r9", runbook)
        self.assertIn(
            "/results/r15-source/flux-engine-source-v0.9.13-r15.tar.gz", runbook
        )

    def test_exact_native_boundary_ladder(self):
        self.assertEqual(Q.NATIVE_CONTEXT, 262_144)
        self.assertEqual(Q.RETRIEVAL_TARGETS, (258_000, 261_888, 262_080))
        self.assertEqual(Q.RETRIEVAL_PROMPT_TOKENS + Q.RETRIEVAL_OUTPUT_TOKENS, 262_144)
        self.assertEqual(Q.BOUNDARY_PROMPT_TOKENS + 1, 262_144)
        self.assertEqual(Q.OVERFLOW_PROMPT_TOKENS + 1, 262_145)
        self.assertEqual(Q.BOUNDARY_EXTENSION_TOKENS, 63)

    def test_calibrated_prompt_hashes_lengths_and_spread_needles(self):
        components = calibrated_components()
        expected_positions = {
            258_000: {"ALPHA": 12_920, "BETA": 129_018, "GAMMA": 245_102},
            261_888: {"ALPHA": 13_114, "BETA": 130_962, "GAMMA": 248_796},
            262_080: {"ALPHA": 13_124, "BETA": 131_058, "GAMMA": 248_978},
        }
        for target in Q.RETRIEVAL_TARGETS:
            prompt, positions = Q.assemble_prompt(target, components)
            self.assertEqual(len(prompt), target)
            self.assertEqual(positions, expected_positions[target])
            self.assertEqual(Q.prompt_sha256(prompt), Q.EXPECTED_PROMPT_SHA256[target])
            self.assertLess(positions["ALPHA"], target // 8)
            self.assertTrue(target * 2 // 5 < positions["BETA"] < target * 3 // 5)
            self.assertGreater(positions["GAMMA"], target * 7 // 8)
            for label, start in positions.items():
                self.assertEqual(
                    prompt[start : start + len(components[label])], components[label]
                )

    def test_boundary_prompt_is_exact_63_token_extension_before_same_query(self):
        components = calibrated_components()
        retrieval, positions = Q.assemble_prompt(Q.RETRIEVAL_PROMPT_TOKENS, components)
        boundary, extension = Q.extend_retrieval_to_boundary(retrieval, components)
        tail_start = len(retrieval) - len(components["tail"])

        self.assertEqual(len(boundary), Q.BOUNDARY_PROMPT_TOKENS)
        self.assertEqual(extension["tokens"], 63)
        self.assertEqual(extension["start"], tail_start)
        self.assertEqual(extension["end"], tail_start + 63)
        self.assertEqual(boundary[:tail_start], retrieval[:tail_start])
        self.assertEqual(
            boundary[tail_start : tail_start + 63],
            Q.repeated(components["filler"], 63),
        )
        self.assertEqual(boundary[tail_start + 63 :], components["tail"])
        for label, start in positions.items():
            self.assertEqual(
                boundary[start : start + len(components[label])], components[label]
            )

    def test_manual_frame_and_pinned_tokenizer_are_explicit(self):
        texts = Q.workload_texts()
        self.assertEqual(tuple(texts), Q.PROMPT_COMPONENT_KEYS)
        self.assertTrue(texts["head"].startswith("<|im_start|>user\n"))
        self.assertTrue(
            texts["tail"].endswith(
                "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            )
        )
        self.assertEqual(
            Q.TOKENIZER_SHA256,
            "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
        )

    def test_prompt_hash_is_u32_endian_stable(self):
        self.assertEqual(
            Q.prompt_sha256([1, 256]),
            Q.sha256_bytes(b"\x01\x00\x00\x00\x00\x01\x00\x00"),
        )

    def test_completion_request_is_exact_token_greedy(self):
        body = Q.completion_body("qwen38", [1, 2, 3], 1)
        self.assertEqual(body["prompt_token_ids"], [1, 2, 3])
        self.assertEqual(body["prompt"], "")
        self.assertEqual(body["temperature"], 0.0)
        self.assertEqual(body["top_k"], 1)
        self.assertEqual(body["seed"], Q.SEED)
        self.assertFalse(body["ignore_eos"])
        self.assertNotIn("return_token_ids", body)
        boundary = Q.completion_body(
            "qwen38",
            [1, 2, 3],
            64,
            ignore_eos=True,
            return_token_ids=True,
        )
        self.assertTrue(boundary["ignore_eos"])
        self.assertTrue(boundary["return_token_ids"])

    def test_server_argv_is_fail_closed(self):
        good = [
            "/bin/flux-server-real",
            "serve",
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
            "18899",
            "--model-from-path",
            "/models/qad5500",
            "--model-name",
            "qad5500",
            "--kernel-target",
            "qwen3.8-flash-next",
        ]
        self.assertEqual(Q.validate_server_argv(good), good)
        for mutation in (
            good + ["--rope-yarn-factor", "2"],
            good + ["--speculative"],
            ["nvfp4" if item == "bf16" else item for item in good],
            [item for item in good if item != "--qwen4-qsa"],
            [item for item in good if item not in ("--max-prefill-tokens", "16384")],
            good + ["--max-seq-len", "262144"],
            good + ["--port", "18899"],
            ["wrong-target" if item == "qwen3.8-flash-next" else item for item in good],
        ):
            with self.assertRaises(ValueError):
                Q.validate_server_argv(mutation)

    def test_base_url_must_match_exact_child_endpoint(self):
        argv = ["server", "--bind", "127.0.0.1", "--port", "18899"]
        self.assertEqual(
            Q.validate_base_url("http://127.0.0.1:18899", argv), ("127.0.0.1", 18899)
        )
        for url in (
            "http://127.0.0.1:18900",
            "http://localhost:18899",
            "http://127.0.0.1:18899/health",
            "http://user@127.0.0.1:18899",
        ):
            with self.assertRaises(ValueError):
                Q.validate_base_url(url, argv)

    def test_prelaunch_endpoint_must_be_unused(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            host, port = listener.getsockname()
            with self.assertRaisesRegex(RuntimeError, "already occupied"):
                Q.require_unoccupied_endpoint(host, port)

    def test_recipe_requires_bf16_c1_qsa_and_no_truncation(self):
        recipe = (
            Path(__file__).resolve().parents[1]
            / "recipes"
            / "flux-qwen4-native-262k-bf16.env"
        )
        values = Q.parse_env_file(recipe)
        self.assertEqual(values["FLUX_KV_CACHE_DTYPE"], "bf16")
        self.assertEqual(values["FLUX_CONCURRENCY"], "1")
        self.assertEqual(values["FLUX_MAX_PREFILL_TOKENS"], "16384")
        self.assertEqual(values["FLUX_CTX_OVERFLOW_TRUNCATE"], "0")
        self.assertEqual(values["FLUX_QWEN4_PREFILL_QSA_FP8"], "0")
        self.assertEqual(values["ATLAS_QWEN4_PREFILL_QSA"], "1")
        self.assertEqual(values["ATLAS_QWEN4_PREFILL_QSA_FP8"], "0")
        self.assertEqual(values["ATLAS_QWEN4_PREFILL_MULTI_CHUNK"], "1")
        self.assertEqual(values["ATLAS_QWEN4_PREFILL_WIDE"], "16384")
        self.assertEqual(values["ATLAS_QWEN4_PREFILL_MOE_BATCH"], "1")
        self.assertEqual(values["ATLAS_QWEN4_PREFILL_ATTN_CORE16"], "1")
        self.assertEqual(values["ATLAS_QWEN4_PREFILL_HC_EXACT"], "1")
        self.assertEqual(values["ATLAS_QWEN4_PREFILL_ATTN_QKV16"], "1")
        self.assertEqual(values["ATLAS_QWEN4_PREFILL_ATTN_O16"], "1")
        self.assertEqual(values["ATLAS_QWEN4_K16_EXACT"], "1")
        self.assertEqual(values["ATLAS_ATTN_QKV_EXACT_STRIDED"], "1")
        self.assertEqual(values["ATLAS_ATTN_QK_NORM_BATCHED"], "1")
        self.assertEqual(values["ATLAS_ATTN_GATE_BATCHED"], "1")

    def test_usage_requires_exact_prompt_and_post_prefill_decode(self):
        value = {
            "choices": [{"index": 0, "text": "ok", "finish_reason": "length"}],
            "usage": {
                "prompt_tokens": 262143,
                "completion_tokens": 1,
                "total_tokens": 262144,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        }
        row = Q.validate_usage(value, 262143, 1)
        self.assertEqual(row["total_tokens"], 262144)
        for field, replacement in (("index", False),):
            bad = copy.deepcopy(value)
            bad["choices"][0][field] = replacement
            with self.assertRaisesRegex(ValueError, "choice differs"):
                Q.validate_usage(bad, 262143, 1)
        bad = copy.deepcopy(value)
        bad["usage"]["prompt_tokens_details"]["cached_tokens"] = False
        with self.assertRaisesRegex(ValueError, "cached prompt tokens"):
            Q.validate_usage(bad, 262143, 1)
        value["usage"]["prompt_tokens"] = 262142
        with self.assertRaisesRegex(ValueError, "accounting"):
            Q.validate_usage(value, 262143, 1)

    def test_exact_token_id_capture_is_opt_in_and_fail_closed(self):
        value = {
            "choices": [
                {
                    "index": 0,
                    "text": "ok",
                    "finish_reason": "length",
                    "token_ids": [248045],
                }
            ],
            "usage": {
                "prompt_tokens": 262143,
                "completion_tokens": 1,
                "total_tokens": 262144,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        }
        row = Q.validate_usage(value, 262143, 1, require_token_ids=True)
        self.assertEqual(row["completion_token_ids"], [248045])
        self.assertRegex(row["completion_u32le_sha256"], r"^[0-9a-f]{64}$")
        del value["choices"][0]["token_ids"]
        with self.assertRaisesRegex(ValueError, "token vector is empty"):
            Q.validate_usage(value, 262143, 1, require_token_ids=True)

    def test_semantic_passkeys_gate_is_independent_of_optional_exact_format(self):
        unlabeled = "alpha-0731826, beta-0492175, gamma-0863204"
        score = Q.score_passkeys(unlabeled)
        self.assertTrue(score["semantic_retrieval"]["all_retrieved"])
        self.assertFalse(score["exact_format"]["matched"])
        self.assertFalse(score["exact_format"]["required_for_pass"])

        exact = Q.score_passkeys(Q.canonical_passkey_answer())
        self.assertTrue(exact["semantic_retrieval"]["all_retrieved"])
        self.assertTrue(exact["exact_format"]["matched"])

        missing = Q.score_passkeys("alpha-0731826 beta-0492175")
        self.assertFalse(missing["semantic_retrieval"]["all_retrieved"])
        self.assertFalse(missing["semantic_retrieval"]["retrieved"]["GAMMA"])

    def test_observed_http_response_is_preserved_losslessly_before_json_gate(self):
        class Client:
            def __init__(self, raw):
                self.raw = raw

            def request(self, method, route, body):
                return 200, self.raw

        raw = b'{"choices":[{"text":"alpha-0731826"}],"opaque":"full"}'
        observations = []
        _, value, returned, index = Q.observed_json_request(
            Client(raw), "POST", "/v1/completions", {}, "case", observations
        )
        self.assertEqual(index, 0)
        self.assertEqual(returned, raw)
        self.assertEqual(value["opaque"], "full")
        self.assertEqual(base64.b64decode(observations[0]["response_body_base64"]), raw)
        self.assertEqual(observations[0]["response"], json.loads(raw))

        invalid = []
        with self.assertRaisesRegex(ValueError, "non-JSON"):
            Q.observed_json_request(
                Client(b"not-json"),
                "POST",
                "/v1/completions",
                {},
                "invalid",
                invalid,
            )
        self.assertEqual(
            base64.b64decode(invalid[0]["response_body_base64"]), b"not-json"
        )
        self.assertFalse(invalid[0]["json_decoded"])

        duplicate = []
        with self.assertRaisesRegex(ValueError, "duplicate-key"):
            Q.observed_json_request(
                Client(b'{"model":"a","model":"b"}'),
                "POST",
                "/v1/completions",
                {},
                "duplicate",
                duplicate,
            )
        self.assertEqual(
            base64.b64decode(duplicate[0]["response_body_base64"]),
            b'{"model":"a","model":"b"}',
        )
        self.assertFalse(duplicate[0]["json_decoded"])

        nonfinite = []
        with self.assertRaisesRegex(ValueError, "non-JSON"):
            Q.observed_json_request(
                Client(b'{"value":Infinity}'),
                "POST",
                "/v1/completions",
                {},
                "nonfinite",
                nonfinite,
            )
        self.assertFalse(nonfinite[0]["json_decoded"])

    def test_live_failure_seals_responses_collected_before_the_gate_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "launch.json"
            manifest_path.write_text("{}", encoding="utf-8")
            receipt_path = root / "receipt.json"
            raw = b'{"choices":[{"text":"complete observed answer"}]}'

            def fail_after_response(*args):
                evidence = args[-1]
                evidence["http_observations"].append(
                    {
                        "label": "retrieval",
                        "response_body_base64": base64.b64encode(raw).decode("ascii"),
                        "response": json.loads(raw),
                    }
                )
                raise ValueError("semantic gate failed")

            manifest = {
                "provenance": {
                    "source_revision": "a" * 40,
                    "binary_path": "/private/flux-server",
                },
                "_manifest_source_sha256": Q.sha256_file(manifest_path),
                "_manifest_canonical_sha256": Q.sha256_bytes(b"{}"),
            }
            with mock.patch.object(
                Q, "_run_live_impl", side_effect=fail_after_response
            ):
                with self.assertRaisesRegex(ValueError, "semantic gate failed"):
                    Q.run_live(manifest, manifest_path, receipt_path, 1.0)

            failure_path = root / "receipt.json.failure.json"
            failure = json.loads(failure_path.read_text(encoding="ascii"))
            self.assertEqual(failure["status"], "FAIL")
            self.assertEqual(failure["error_type"], "ValueError")
            self.assertEqual(
                base64.b64decode(
                    failure["http_observations"][0]["response_body_base64"]
                ),
                raw,
            )
            self.assertEqual(stat.S_IMODE(failure_path.stat().st_mode), 0o444)
            self.assertFalse(receipt_path.exists())

    def test_boundary_oracle_rejects_usage_only_false_pass(self):
        control = {
            "completion_tokens": 1,
            "completion_token_ids": [123],
            "text": " alpha",
        }
        boundary = {
            "completion_tokens": 1,
            "completion_token_ids": [123],
            "text": " alpha",
        }
        self.assertEqual(
            Q.validate_boundary_oracle(control, boundary),
            123,
        )

        for bad_boundary in (
            {"completion_tokens": 1, "completion_token_ids": [124], "text": " alpha"},
            {"completion_tokens": 1, "text": " alpha"},
            {"completion_tokens": 0, "completion_token_ids": [123], "text": " alpha"},
            {
                "completion_tokens": 1,
                "completion_token_ids": [123, 124],
                "text": " alpha",
            },
        ):
            with self.assertRaises(ValueError):
                Q.validate_boundary_oracle(control, bad_boundary)

    def test_qsa_log_contract_requires_every_bf16_endpoint(self):
        complete = "\n".join(
            (
                Q.QSA_MARKER,
                *Q.QSA_TARGET_BF16_RECEIPTS.values(),
                Q.QSA_BOUNDARY_BF16_RECEIPT,
            )
        )
        Q.validate_qsa_log_contract(complete)
        with self.assertRaisesRegex(ValueError, "261888"):
            Q.validate_qsa_log_contract(
                complete.replace(Q.QSA_TARGET_BF16_RECEIPTS[261_888], "")
            )
        with self.assertRaisesRegex(ValueError, "quantized"):
            Q.validate_qsa_log_contract(complete + "\n" + Q.QSA_FP8_MARKER)

    def test_endpoint_receipt_uses_prefill_chunk_size_not_qsa_micro_tile(self):
        source = (
            Path(__file__).resolve().parents[2]
            / "crates/spark-model/src/layers/qwen4_qsa/prefill_sparse_run.rs"
        ).read_text(encoding="utf-8")
        self.assertIn("qwen4_prefill_moe::max_chunk_rows()", source)
        self.assertIn("if c.rows < max_chunk_rows", source)
        self.assertNotIn("if c.rows < TILE_ROWS", source)

    def test_write_all_retries_short_writes(self):
        calls = []
        original = Q.os.write
        try:

            def short_write(fd, payload):
                calls.append(bytes(payload))
                return min(2, len(payload))

            Q.os.write = short_write
            Q.write_all(7, b"abcdef")
        finally:
            Q.os.write = original
        self.assertEqual([len(item) for item in calls], [6, 4, 2])

    def test_hand_written_source_receipt_is_not_sufficient(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source-to-binary.json"
            provenance = {
                "release_version": "v0.9.13-r9",
                "source_revision": "a" * 40,
                "source_tree": "b" * 40,
                "source_root_path": str(Path(directory) / "source"),
                "source_bundle_manifest_path": str(Path(directory) / "source.json"),
                "source_bundle_manifest_sha256": "b" * 64,
                "source_bundle_path": str(Path(directory) / "source.tar.gz"),
                "source_bundle_sha256": "c" * 64,
                "build_receipt_path": str(Path(directory) / "build.json"),
                "build_receipt_sha256": "e" * 64,
                "source_to_binary_receipt_path": str(path),
                "binary_sha256": "d" * 64,
                "binary_path": str(Path(directory) / "binary"),
            }
            receipt = {
                "schema": Q.SOURCE_TO_BINARY_SCHEMA,
                "verdict": "PASS",
                "release_version": provenance["release_version"],
                "release_commit": provenance["source_revision"],
                "source_bundle_manifest_sha256": provenance[
                    "source_bundle_manifest_sha256"
                ],
                "source_bundle_sha256": provenance["source_bundle_sha256"],
                "binary": {"sha256": provenance["binary_sha256"]},
            }
            path.write_bytes(Q.canonical_bytes(receipt))
            provenance["source_to_binary_receipt_sha256"] = Q.sha256_file(path)
            with self.assertRaisesRegex(ValueError, "source manifest must be"):
                Q.validate_source_to_binary_receipt(path, provenance, True)

    def test_release_provenance_contract_generalizes_without_skipping_verifier(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source-to-binary.json"
            path.write_text("{}", encoding="ascii")
            provenance = {
                "release_version": "v0.9.13-r11",
                "source_to_binary_receipt_path": str(path),
            }
            original_release = Q.r9_provenance.RELEASE_VERSION

            def inspect(selected):
                self.assertIs(selected, provenance)
                self.assertEqual(Q.r9_provenance.RELEASE_VERSION, "v0.9.13-r11")
                self.assertEqual(
                    Q.r9_provenance.SOURCE_MANIFEST_SCHEMA,
                    "flux-r11-source-bundle-manifest/2",
                )
                self.assertEqual(
                    Q.r9_provenance.SOURCE_BUNDLE_NAME,
                    "flux-engine-source-v0.9.13-r11.tar.gz",
                )

            with mock.patch.object(
                Q.r9_provenance,
                "validate_qualification_provenance",
                side_effect=inspect,
            ) as verifier:
                Q.validate_source_to_binary_receipt(path, provenance, True)
            verifier.assert_called_once_with(provenance)
            self.assertEqual(Q.r9_provenance.RELEASE_VERSION, original_release)
            self.assertEqual(
                Q.r9_provenance.SOURCE_MANIFEST_SCHEMA,
                "flux-r9-source-bundle-manifest/2",
            )
            self.assertEqual(
                Q.r9_provenance.SOURCE_BUNDLE_NAME,
                "flux-engine-source-v0.9.13-r9.tar.gz",
            )

    def test_checkpoint_inventory_binds_selected_file_set_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "checkpoint"
            root.mkdir()
            (root / "config.json").write_text("{}", encoding="utf-8")
            (root / "weights.bin").write_bytes(b"weights")
            provenance = {
                "checkpoint_id": "JMNI-Labs/QAD5500",
                "checkpoint_revision": "f" * 64,
            }
            entries = []
            for file_path in sorted(root.iterdir()):
                entries.append(
                    {
                        "path": file_path.name,
                        "bytes": file_path.stat().st_size,
                        "sha256": Q.sha256_file(file_path),
                    }
                )
            inventory = {
                "schema": Q.CHECKPOINT_INVENTORY_SCHEMA,
                "checkpoint_id": provenance["checkpoint_id"],
                "checkpoint_revision": provenance["checkpoint_revision"],
                "file_count": len(entries),
                "total_bytes": sum(entry["bytes"] for entry in entries),
                "files": entries,
            }
            inventory_path = base / "checkpoint-inventory.json"
            inventory_path.write_bytes(Q.canonical_bytes(inventory))
            provenance["checkpoint_manifest_sha256"] = Q.sha256_file(inventory_path)
            Q.validate_checkpoint_inventory(inventory_path, root, provenance, True)
            (root / "weights.bin").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "weights.bin"):
                Q.validate_checkpoint_inventory(inventory_path, root, provenance, True)

    def test_tokenizer_file_is_pinned_to_calibrated_sha256(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tokenizer.json").write_bytes(b"fixture")
            with mock.patch.object(Q, "sha256_file", return_value=Q.TOKENIZER_SHA256):
                Q.validate_tokenizer_pin(root, True)
            with mock.patch.object(Q, "sha256_file", return_value="0" * 64):
                with self.assertRaisesRegex(ValueError, "pinned oracle"):
                    Q.validate_tokenizer_pin(root, True)
            (root / "tokenizer.json").unlink()
            with self.assertRaisesRegex(ValueError, "not a file"):
                Q.validate_tokenizer_pin(root, True)
            Q.validate_tokenizer_pin(root, False)

    def test_strict_json_and_manifest_snapshot_reject_false_passes(self):
        with self.assertRaisesRegex(ValueError, "duplicate JSON key 'schema'"):
            Q.strict_json_loads('{"schema":"a","schema":"b"}', "fixture")
        for value in ("NaN", "Infinity", "-Infinity", "1e9999", "-1e9999"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "non-finite JSON number"):
                    Q.strict_json_loads(f'{{"value":{value}}}', "fixture")
        manifest = {
            "schema": Q.MANIFEST_SCHEMA,
            "_manifest_source_sha256": "a" * 64,
            "_manifest_canonical_sha256": "b" * 64,
            "_recipe": {"PIN": "1"},
        }
        changed = dict(manifest)
        changed["_manifest_source_sha256"] = "c" * 64
        with mock.patch.object(Q, "load_manifest", return_value=changed):
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                Q.revalidate_manifest_and_artifacts(
                    manifest, Path("/manifest.json"), "before native-262K arm"
                )
        changed = dict(manifest)
        changed["_recipe"] = {"PIN": "0"}
        with mock.patch.object(Q, "load_manifest", return_value=changed):
            with self.assertRaisesRegex(ValueError, "recipe environment changed"):
                Q.revalidate_manifest_and_artifacts(
                    manifest, Path("/manifest.json"), "before final seal"
                )

    def test_legacy_gate_uses_shared_strict_overflow_evidence(self):
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
        rejection = Q.endpoint_gate.validate_context_length_rejection(
            400, value, 262_144, 1, 262_144
        )
        self.assertTrue(rejection["context_length_rejection"])
        with self.assertRaisesRegex(ValueError, "exact non-negative integer"):
            Q.endpoint_gate.validate_context_length_rejection(
                False, value, 262_144, 1, 262_144
            )
        request_id = "qualify_overflow_" + "a" * 32
        self.assertEqual(
            Q.endpoint_gate.validate_correlated_request_id(
                [("X-Request-Id", request_id)], request_id
            ),
            request_id,
        )

    def test_dry_run_plan_names_every_required_gate(self):
        manifest = {
            "provenance": {
                "source_revision": "a" * 40,
                "binary_path": "/private/build/flux-server",
            }
        }
        plan = Q.dry_run_plan(manifest)
        self.assertTrue(plan["fresh_process"])
        self.assertEqual(plan["execution_scope"], "direct-binary-not-container")
        self.assertEqual(
            [case["prompt_tokens"] for case in plan["cases"]],
            [258000, 261888, 262080, 262080, 262080, 262080, 262143, 262144],
        )
        self.assertTrue(plan["cases"][0]["semantic_passkeys"])
        self.assertTrue(plan["cases"][0]["post_prefill_decode"])
        self.assertTrue(plan["cases"][1]["bridge"])
        self.assertTrue(plan["cases"][2]["native_target"])
        self.assertTrue(plan["cases"][3]["repeat_determinism"])
        self.assertTrue(plan["cases"][4]["ignore_eos"])
        self.assertTrue(plan["cases"][4]["exact_output_token_ids"])
        self.assertEqual(plan["cases"][4]["required_total_tokens"], 262144)
        self.assertTrue(plan["cases"][5]["boundary_oracle_control"])
        self.assertTrue(plan["cases"][5]["exact_output_token_ids"])
        self.assertTrue(plan["cases"][6]["boundary_generation"])
        self.assertTrue(plan["cases"][6]["matches_control_token_id"])
        self.assertTrue(plan["cases"][7]["no_truncation"])
        self.assertEqual(
            plan["prompt_contract"]["tokenizer_json_sha256"],
            Q.TOKENIZER_SHA256,
        )
        self.assertTrue(plan["prompt_contract"]["semantic_passkey_retrieval_is_gate"])
        self.assertFalse(plan["prompt_contract"]["exact_format_is_gate"])
        self.assertEqual(plan["provenance"], {"source_revision": "a" * 40})


if __name__ == "__main__":
    unittest.main()
