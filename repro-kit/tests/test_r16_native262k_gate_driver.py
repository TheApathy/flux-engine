# SPDX-License-Identifier: AGPL-3.0-only
"""Offline-only native-262K gate tests. No Docker, HTTP, GPU, or server runs."""

import array
import base64
import copy
from contextlib import contextmanager
import hashlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import types
import unittest
from unittest import mock

KIT = Path(__file__).resolve().parents[1]


def module(path, name):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    loader.exec_module(value)
    return value


N = module(KIT / "r15-gates/native-262k", "native262_driver_fixture")
S = module(KIT / "r15-container-suite.py", "native262_suite_fixture")
Q = module(KIT / "r15_exact_image_release.py", "native262_outer_fixture")
OLD = module(KIT / "qwen4_native_262k_qualify.py", "native262_prior_fixture")
C = N.load_checked(
    KIT / "r15-gates/mmlu-pro", N.LIFECYCLE_SHA256, "native262_lifecycle_fixture"
)
E = N.load_checked(
    KIT / "qwen4_endpoint_admission.py", N.ENDPOINT_SHA256, "native262_endpoint_fixture"
)
# N.token_hash joins one 4-byte object per token: ~12 MB transient at 262K,
# which with a ~74 MB import baseline exceeds the 90 MB address-space cap the
# suites run under. lean_token_hash validates through the driver's own
# N.tokens and hashes the identical little-endian u32 bytes in one buffer;
# test_lean_token_hash_matches_the_driver_hash pins it to N.token_hash, and
# the release-size replays use it in N.token_hash's place.
assert array.array("I").itemsize == 4 and sys.byteorder == "little"


def lean_token_hash(value):
    return N.digest(array.array("I", N.tokens(value)).tobytes())


COMPONENTS = {
    "head": [1],
    "filler": [2, 3],
    "tail": [4, 5],
    "ALPHA": [6],
    "BETA": [7],
    "GAMMA": [8],
}


def labeled_answer(values=None):
    values = N.PASSKEYS if values is None else values
    return "\n".join(f"{label}: {value}" for label, value in values.items())


def chunk_receipt(target):
    return (
        "QWEN4_PREFILL_QSA_CHUNK_RECEIPT selector=ATLAS_QWEN4_PREFILL_QSA "
        f"start=245760 rows={target - 245760} end={target} "
        "chunk_rows=16384 tile_rows=10 core=qsa_sparse_bf16_non_bit_exact"
    )



IMAGE_PLAN = {
    "in_container_binary_path": "/opt/flux/bin/fixture",
    "in_container_binary_sha256": "1" * 64,
    "in_container_recipe_path": "/opt/flux/repro-kit/recipes/release.env",
    "recipe_sha256": "2" * 64,
}


def native_plan(image, gate=N.GATE):
    """The plan the driver gets from a sealed parent plan /4 (295a6aaa1)."""
    launch = {
        **image,
        "schema": S.PLAN_SCHEMA_V4,
        "native_context_contract": {
            "schema": S.NATIVE_CONTEXT_SCHEMA,
            "gates": S.NATIVE_CONTEXT_RECIPES,
        },
    }
    return S.native_context_recipe_plan(launch, image, gate)

class Native262DriverTests(unittest.TestCase):
    def setUp(self):
        self.guard = mock.patch.object(
            socket, "socket", side_effect=AssertionError("network forbidden")
        )
        self.guard.start()
        self.addCleanup(self.guard.stop)
        self.process_guard = mock.patch.object(
            N.subprocess, "Popen", side_effect=AssertionError("server forbidden")
        )
        self.process_guard.start()
        self.addCleanup(self.process_guard.stop)
        self.docker_guard = mock.patch.object(
            C, "docker", side_effect=AssertionError("Docker forbidden")
        )
        self.docker_guard.start()
        self.addCleanup(self.docker_guard.stop)

    def response(self, prompt=None, budget=64, forced=False, text=None):
        prompt = [1, 2, 3] if prompt is None else prompt
        ids = list(range(100, 100 + (budget if forced else 2)))
        return {
            "id": "fixture-response",
            "model": "fixture-model",
            "choices": [
                {
                    "index": 0,
                    "text": "fixture answer" if text is None else text,
                    "token_ids": ids,
                    "finish_reason": "length" if forced else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt),
                "completion_tokens": len(ids),
                "total_tokens": len(prompt) + len(ids),
                "prompt_tokens_details": {"cached_tokens": 0},
            },
            "interventions": copy.deepcopy(N.LEDGER),
        }

    def test_source_pins_and_same_bytes_loader_refuse_substitution(self):
        self.assertEqual(
            hashlib.sha256((KIT / "r15-gates/mmlu-pro").read_bytes()).hexdigest(),
            N.LIFECYCLE_SHA256,
        )
        self.assertEqual(
            hashlib.sha256(
                (KIT / "qwen4_endpoint_admission.py").read_bytes()
            ).hexdigest(),
            N.ENDPOINT_SHA256,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.py"
            path.write_text("raise AssertionError('unverified code executed')\n")
            with self.assertRaisesRegex(ValueError, "hash"):
                N.load_checked(path, "1" * 64, "native262_untrusted_fixture")
            path.write_text("value = 123\n")
            raw = path.read_bytes()
            real_pread = os.pread

            def consume_then_replace(fd, count, offset):
                result = real_pread(fd, count, offset)
                path.unlink()
                path.write_text("raise AssertionError('pathname reopened')\n")
                return result

            with mock.patch.object(N.os, "pread", side_effect=consume_then_replace):
                with self.assertRaisesRegex(ValueError, "identity"):
                    N.load_checked(
                        path, N.digest(raw), "native262_replaced_during_read"
                    )
            path.write_bytes(raw)
            real_compile = compile

            def replace_after_verification(payload, filename, mode):
                path.unlink()
                path.write_text("raise AssertionError('pathname reopened')\n")
                return real_compile(payload, filename, mode)

            with mock.patch("builtins.compile", side_effect=replace_after_verification):
                loaded = N.load_checked(
                    path, N.digest(raw), "native262_exact_bytes_fixture"
                )
            self.assertEqual(loaded.value, 123)

    def test_bootstrap_binds_source_helper_to_actual_sealed_parent_suite(self):
        argv = [
            "--container-id",
            "a" * 64,
            "--image-id",
            "sha256:" + "b" * 64,
            "--image-digest",
            "sha256:" + "c" * 64,
            "--base-url",
            "http://127.0.0.1:8888",
            "--run-id",
            "fixture",
            "--receipt-fd",
            "4",
            "--raw-log-fd",
            "5",
            "--runtime-dir",
            "/private/results",
            "--suite-program",
            "/source/r15-container-suite.py",
            "--suite-sha256",
            "d" * 64,
            "--lifecycle-program",
            "/source/mmlu-pro",
            "--endpoint-program",
            "/source/qwen4_endpoint_admission.py",
        ]
        with (
            mock.patch.object(N, "load_checked", side_effect=[C, E]) as checked,
            mock.patch.object(C, "verify_parent_suite") as parent,
            mock.patch.object(C, "load_module", return_value=S) as suite,
        ):
            args = N.parse_args(argv)
        parent.assert_called_once_with("d" * 64)
        suite.assert_called_once_with(
            "/source/r15-container-suite.py", "d" * 64, "_r16_native262_suite"
        )
        self.assertEqual(checked.call_args_list[0].args[1], N.LIFECYCLE_SHA256)
        self.assertEqual(checked.call_args_list[1].args[1], N.ENDPOINT_SHA256)
        self.assertIs(args.suite, S)

    def test_fresh_server_launch_has_explicit_qsa_contract_and_owned_cleanup(self):
        args = types.SimpleNamespace(
            lifecycle=C,
            suite=S,
            container_id="a" * 64,
            container_port=8888,
            identities={},
        )
        process = mock.Mock()
        process.poll.side_effect = [None, 0]
        proof = {"host_pid": 123, "status": "PASS"}
        with (
            mock.patch.object(N.subprocess, "Popen", return_value=process) as popen,
            mock.patch.object(C, "find_pid", return_value=123),
            mock.patch.object(C, "claim_pid", return_value=700),
            mock.patch.object(C, "request", return_value=b"healthy"),
            mock.patch.object(S, "verify_live_server", return_value=proof) as verify,
            mock.patch.object(
                N.select, "select", side_effect=[([], [], []), ([700], [], [])]
            ),
            mock.patch.object(N.signal, "pidfd_send_signal") as kill,
            mock.patch.object(N.os, "close") as close,
        ):
            with N.server(
                args,
                native_plan(IMAGE_PLAN),
                {"ATLAS_QWEN4_PREFILL_WIDE": "16384"},
            ):
                pass
        command = popen.call_args.args[0]
        self.assertEqual(
            command[:5],
            ["docker", "exec", "a" * 64, "/usr/bin/env", "ATLAS_DISABLE_SPECULATION=1"],
        )
        for selector in (
            "--qwen4-qsa",
            "--no-tui",
            "--max-prefill-tokens",
            "--block-size",
            "--enable-prefix-caching",
            "--ssm-cache-slots",
            "--swap-space-gb",
        ):
            self.assertIn(selector, command)
        self.assertNotIn("stdout", popen.call_args.kwargs)
        self.assertNotIn("stderr", popen.call_args.kwargs)
        self.assertEqual(verify.call_count, 2)
        kill.assert_called_once_with(700, N.signal.SIGTERM)
        close.assert_called_once_with(700)

    def test_lean_token_hash_matches_the_driver_hash(self):
        vectors = (
            [0],
            [1, 2, 3],
            [0, 1, 255, 256, 65535, 65536, (1 << 31) - 1, 1 << 31, (1 << 32) - 1],
            [(index * 2654435761) % (1 << 32) for index in range(4096)],
            N.assemble(258000, COMPONENTS)[0][:8192],
        )
        for value in vectors:
            self.assertEqual(lean_token_hash(value), N.token_hash(value))
            self.assertEqual(lean_token_hash(value), OLD.prompt_sha256(value))
        for invalid in ([], [-1], [1 << 32], [1.0], [True], (1, 2), [0] * (N.CONTEXT + 1)):
            with self.assertRaisesRegex(ValueError, "u32 token vector"):
                N.token_hash(invalid)
            with self.assertRaisesRegex(ValueError, "u32 token vector"):
                lean_token_hash(invalid)

    def test_token_builders_match_existing_calibrated_qualifier(self):
        self.assertEqual(N.TEXTS, OLD.workload_texts())
        self.assertEqual(N.EXPECTED_PROMPT_SHA256, OLD.EXPECTED_PROMPT_SHA256)
        self.assertEqual(N.TOKENIZER_SHA256, OLD.TOKENIZER_SHA256)
        for target in N.TARGETS:
            prompt, positions = N.assemble(target, COMPONENTS)
            expected, expected_positions = OLD.assemble_prompt(target, COMPONENTS)
            self.assertEqual(prompt, expected)
            self.assertEqual(positions, expected_positions)
            del expected
            # Release-size hash via the pinned lean equivalent (see
            # lean_token_hash); OLD.prompt_sha256 streams token by token.
            self.assertEqual(lean_token_hash(prompt), OLD.prompt_sha256(prompt))
            del prompt
        prompt = N.assemble(262080, COMPONENTS)[0]
        self.assertEqual(
            N.boundary_prompt(prompt, COMPONENTS),
            OLD.extend_retrieval_to_boundary(prompt, COMPONENTS)[0],
        )
        self.assertEqual(len(N.boundary_prompt(prompt, COMPONENTS)), 262143)

    def test_request_controls_are_exact_and_report_only(self):
        body = N.completion_body("fixture-model", [1, 2], 64, True)
        self.assertEqual(body["prompt_token_ids"], [1, 2])
        self.assertIs(body["ignore_eos"], True)
        self.assertIs(body["return_token_ids"], True)
        self.assertIs(body["report_interventions"], True)
        self.assertEqual(body["stop"], [])
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["max_tokens"], 64)

    def test_transport_failure_preserves_attempt_before_dispatch(self):
        connection = mock.Mock()
        connection.request.side_effect = TimeoutError("private endpoint details")
        journal = mock.Mock()
        args = types.SimpleNamespace(base_url="http://127.0.0.1:8888", suite=S)
        body = N.completion_body("fixture", [1, 2, 3], 1, True)
        with mock.patch.object(
            N.http.client, "HTTPConnection", return_value=connection
        ):
            with self.assertRaises(TimeoutError):
                N.observed(
                    args, journal, "interrupted", "POST", "/v1/completions", body
                )
        journal.append.assert_called_once()
        self.assertEqual(journal.append.call_args.args[0]["event"], "request")
        self.assertEqual(journal.append.call_args.args[0]["request"], body)
        connection.close.assert_called_once()

    def test_measurement_requires_nonempty_answer_exact_ids_cache_ledger_and_geometry(
        self,
    ):
        args = types.SimpleNamespace(suite=S)
        valid = self.response()
        self.assertEqual(
            N.measurement(args, valid, "fixture-model", [1, 2, 3], 64)["token_ids"],
            [100, 101],
        )
        mutations = [
            lambda value: value["choices"][0].update(text=""),
            lambda value: value["choices"][0].update(text=" \n"),
            lambda value: value["choices"][0].update(index=True),
            lambda value: value["choices"][0].pop("token_ids"),
            lambda value: value["choices"][0].update(token_ids=[100, True]),
            lambda value: value["choices"][0].update(token_ids=[100]),
            lambda value: value["choices"][0].update(finish_reason="length"),
            lambda value: value["usage"].update(prompt_tokens=True),
            lambda value: value["usage"].update(total_tokens=6),
            lambda value: value["usage"]["prompt_tokens_details"].update(
                cached_tokens=True
            ),
            lambda value: value["usage"]["prompt_tokens_details"].update(
                cached_tokens=1
            ),
            lambda value: value["usage"].pop("prompt_tokens_details"),
            lambda value: value.pop("interventions"),
            lambda value: value["interventions"].update(events=[{"code": "modified"}]),
            lambda value: value.update(model="foreign-model"),
            lambda value: value.update(id=""),
        ]
        for mutate in mutations:
            value = copy.deepcopy(valid)
            mutate(value)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                N.measurement(args, value, "fixture-model", [1, 2, 3], 64)
        forced = self.response(budget=64, forced=True)
        self.assertEqual(
            N.measurement(args, forced, "fixture-model", [1, 2, 3], 64, True)[
                "completion_tokens"
            ],
            64,
        )
        forced["choices"][0]["token_ids"].pop()
        forced["usage"].update(completion_tokens=63, total_tokens=66)
        with self.assertRaises(ValueError):
            N.measurement(args, forced, "fixture-model", [1, 2, 3], 64, True)

    def test_live_process_validates_actual_controls_and_mounted_tokenizer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "root/models").mkdir(parents=True)
            payload = b"fixture tokenizer"
            (root / "root/models/tokenizer.json").write_bytes(payload)
            environment = {
                "ATLAS_QWEN4_PREFILL_WIDE": "16384",
                "ATLAS_QWEN4_PREFILL_MULTI_CHUNK": "1",
                "ATLAS_QWEN4_PREFILL_QSA": "1",
                "ATLAS_QWEN4_PREFILL_QSA_FP8": "0",
                "ATLAS_CTX_OVERFLOW_TRUNCATE": "0",
                "ATLAS_DISABLE_SPECULATION": "1",
            }
            argv = [
                "server",
                "serve",
                "--max-seq-len",
                "262144",
                "--kv-cache-dtype",
                "bf16",
                "--max-num-seqs",
                "1",
                "--max-batch-size",
                "1",
                "--max-prefill-tokens",
                "16384",
                "--block-size",
                "16",
                "--enable-prefix-caching",
                "false",
                "--ssm-cache-slots",
                "0",
                "--swap-space-gb",
                "0",
                "--kernel-target",
                "qwen3.8-flash-next",
                "--qwen4-qsa",
            ]

            def files():
                (root / "environ").write_bytes(
                    b"\0".join(
                        (key + "=" + value).encode()
                        for key, value in environment.items()
                    )
                    + b"\0"
                )
                (root / "cmdline").write_bytes(
                    b"\0".join(value.encode() for value in argv) + b"\0"
                )

            files()
            with (
                mock.patch.object(N, "Path", return_value=root),
                mock.patch.object(N, "TOKENIZER_SHA256", N.digest(payload)),
            ):
                self.assertEqual(
                    N.native_process(types.SimpleNamespace(suite=S), {"host_pid": 123})[
                        "tokenizer_sha256"
                    ],
                    N.digest(payload),
                )
                environment["ATLAS_QWEN4_PREFILL_QSA_FP8"] = "qkpv"
                files()
                with self.assertRaisesRegex(ValueError, "environment"):
                    N.native_process(types.SimpleNamespace(suite=S), {"host_pid": 123})
                environment["ATLAS_QWEN4_PREFILL_QSA_FP8"] = "0"
                argv.remove("--qwen4-qsa")
                files()
                with self.assertRaisesRegex(ValueError, "QSA"):
                    N.native_process(types.SimpleNamespace(suite=S), {"host_pid": 123})

    def test_full_mocked_gate_and_existing_outer_validator(self):
        self.full_run()

    def test_labeled_answers_allow_only_exact_unique_keyed_values(self):
        expected = list(N.PASSKEYS.values())
        for text in (
            labeled_answer(),
            "ALPHA alpha-0731826 BETA beta-0492175 GAMMA gamma-0863204",
            "- **ALPHA**: `alpha-0731826`\n- **BETA**: `beta-0492175`\n- **GAMMA**: `gamma-0863204`",
            labeled_answer(dict(reversed(list(N.PASSKEYS.items())))),
        ):
            with self.subTest(text=text):
                self.assertEqual(N.passkey_answers(text), expected)
        valid = labeled_answer()
        for text in (
            " ".join(expected),
            valid.replace("alpha-0731826", "alpha-07318269"),
            valid.replace("alpha-0731826", "wrong-alpha-0731826"),
            valid.replace("alpha-0731826", "alpha-0731826_wrong"),
            valid.replace("alpha-0731826", "alpha-0731826.9"),
            valid.replace("alpha-0731826", "alpha-0731826\N{GREEK SMALL LETTER ALPHA}"),
            valid.replace("ALPHA:", "BETA:"),
            valid.replace("ALPHA: alpha-0731826", "ALPHA: beta-0492175"),
            valid + "\nALPHA: alpha-0731826",
            valid + "\nALPHA: wrong",
            valid + "\nThese answers are not correct",
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                N.passkey_answers(text)

    def test_extended_wrong_retrieval_values_cannot_reach_outer_pass(self):
        extended = labeled_answer(
            {label: value + "9" for label, value in N.PASSKEYS.items()}
        )
        self.full_run(failure="exact-retrieval", retrieval_text=extended)

    def test_duplicate_or_mislabeled_retrieval_cannot_reach_outer_pass(self):
        valid = labeled_answer()
        for text in (
            valid + "\nALPHA: alpha-0731826",
            valid.replace("ALPHA:", "BETA:"),
            valid.replace("ALPHA: alpha-0731826", "ALPHA: beta-0492175"),
        ):
            with self.subTest(text=text):
                self.full_run(failure="exact-retrieval", retrieval_text=text)

    def test_forced_fill_also_requires_exact_labeled_passkeys(self):
        extended = labeled_answer(
            {label: value + "9" for label, value in N.PASSKEYS.items()}
        )
        self.full_run(failure="exact-fill", fill_text=extended)

    def test_late_fatal_spellings_cannot_reach_outer_pass(self):
        for marker in (
            b"CUDA_ERROR_OUT_OF_MEMORY",
            b"cudaErrorMemoryAllocation",
            b"CUDA_ERROR_ILLEGAL_ADDRESS",
            b"CUDA-out-of-memory",
            b"Out of memory: Killed process 123",
        ):
            with self.subTest(marker=marker):
                self.full_run(failure="fatal-log", log_suffix=marker)

    def test_extended_terminal_receipt_cannot_reach_outer_pass(self):
        self.full_run(failure="chunk-receipt", corrupt_receipt=True)

    def test_kernel_command_requires_complete_current_boot_journal(self):
        with tempfile.TemporaryFile() as target:
            for code, error in ((1, b""), (0, b"permission denied")):
                with (
                    mock.patch.object(
                        N.subprocess,
                        "run",
                        return_value=types.SimpleNamespace(returncode=code, stderr=error),
                    ) as command,
                    self.assertRaisesRegex(ValueError, "kernel journal is unavailable"),
                ):
                    N.kernel_command(["--lines=1"], target.fileno())
                argv = command.call_args.args[0]
                self.assertIn("--dmesg", argv)
                self.assertIn("--boot", argv)
                self.assertIn("--output=json", argv)
            with (
                mock.patch.object(N.subprocess, "run", side_effect=TimeoutError),
                self.assertRaises(TimeoutError),
            ):
                N.kernel_command(["--lines=1"], target.fileno())

    def test_current_boot_identity_requires_real_uuid_shape(self):
        identity = mock.Mock()
        with mock.patch.object(N, "Path", return_value=identity):
            identity.read_text.return_value = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa\n"
            self.assertEqual(N.current_boot(), "a" * 32)
            for invalid in ("", "a" * 32, "arbitrary boot identity"):
                identity.read_text.return_value = invalid
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    N.current_boot()

    def test_kernel_only_fatal_events_cannot_reach_outer_pass(self):
        for marker in (
            "NVRM: Xid 79, GPU has fallen off the bus",
            "CUDA_ERROR_ILLEGAL_ADDRESS",
            "Out of memory: Killed process 123 (engine)",
            "oom-kill:constraint=CONSTRAINT_MEMCG",
        ):
            with self.subTest(marker=marker):
                self.full_run(failure="kernel-fatal", kernel_message=marker)

    def test_lost_cursor_changed_boot_or_incomplete_kernel_capture_cannot_pass(self):
        for problem in (
            "lost-cursor",
            "changed-record-boot",
            "changed-live-boot",
            "duplicate-cursor",
            "journal-unavailable",
            "malformed-journal",
            "baseline-boot",
            "mutated-baseline",
        ):
            with self.subTest(problem=problem):
                self.full_run(failure="kernel-capture", kernel_problem=problem)

    def test_unchanged_kernel_cursor_with_no_new_events_can_qualify(self):
        self.full_run(kernel_problem="no-new-events")

    def test_native_cleanup_failure_still_captures_kernel_and_never_passes(self):
        self.full_run(failure="cleanup")

    def test_kernel_artifacts_cannot_change_after_consumption_before_hashing(self):
        for name in ("kernel-start.jsonl", "kernel-full.jsonl", "kernel-measured.log"):
            with self.subTest(name=name):
                self.full_run(failure="kernel-artifact", mutate_kernel_artifact=name)

    def test_wrong_retrieval_never_yields_capacity_only_pass(self):
        self.full_run(failure="retrieval")

    def test_missing_intervention_ledger_never_qualifies(self):
        self.full_run(failure="ledger")

    def test_boundary_token_mismatch_never_qualifies(self):
        self.full_run(failure="boundary")

    def test_overflow_counter_mutation_never_qualifies(self):
        self.full_run(failure="overflow")

    def test_late_overflow_dispatch_never_qualifies(self):
        self.full_run(failure="late-dispatch")

    def full_run(
        self,
        failure=None,
        retrieval_text=None,
        fill_text=None,
        log_suffix=None,
        corrupt_receipt=False,
        kernel_message="clean measured kernel fixture",
        kernel_problem=None,
        mutate_kernel_artifact=None,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            runtime.chmod(0o700)
            descriptors = [
                os.open(runtime / name, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
                for name in ("receipt", "stdout", "stderr")
            ]
            receipt_fd, stdout_fd, stderr_fd = descriptors
            os.set_inheritable(receipt_fd, True)
            raw_path = runtime / "raw"
            raw_path.write_bytes(b"canonical raw remains read only")
            raw_fd = os.open(raw_path, os.O_RDONLY)
            descriptors.append(raw_fd)
            args = types.SimpleNamespace(
                suite=S,
                lifecycle=C,
                endpoint=E,
                runtime_dir=str(runtime),
                receipt_fd=receipt_fd,
                raw_log_fd=raw_fd,
                container_id="a" * 64,
                image_id="sha256:" + "b" * 64,
                image_digest="sha256:" + "c" * 64,
                run_id="mock-native262",
                base_url="http://127.0.0.1:8888",
                container_port=8888,
            )
            plan = dict(IMAGE_PLAN)
            row = S.NATIVE_CONTEXT_RECIPES[N.GATE]
            contract = {"schema": S.NATIVE_CONTEXT_SCHEMA, "gates": S.NATIVE_CONTEXT_RECIPES}
            # The owned plain process proof the suite issues for native context
            # gates (295a6aaa1); the outer validator checks every field.
            proof = {
                "schema": S.NATIVE_CONTEXT_PROOF, "status": "PASS", "mode": "none",
                "gate": N.GATE, "container_id": args.container_id, "run_id": args.run_id,
                "executable_sha256": plan["in_container_binary_sha256"],
                "recipe_sha256": row["recipe_sha256"], "chunk_tokens": row["chunk_tokens"],
                "native_context_contract_sha256": S.sha256_bytes(S.canonical_bytes(contract)),
                "recipe_environment_verified": True, "mount_namespace_match": True,
                "cgroup_match": True, "paths_read_only": True,
                "host_pid": 123, "start_time_ticks": 500,
            }
            controls = {"tokenizer_sha256": N.TOKENIZER_SHA256}
            count = 0
            progress = {
                "atlas_prefill_progress_chunks_total": 0,
                "atlas_prefill_progress_tokens_total": 0,
            }
            real_streams = N.Streams
            real_assert_identity = S.assert_path_identity
            cleaned = []
            kernel_calls = []
            mutated_artifacts = []

            def assert_identity(root, locator, descriptor, label):
                real_assert_identity(root, locator, descriptor, label)
                if (
                    mutate_kernel_artifact is not None
                    and locator == "native262k/" + mutate_kernel_artifact
                    and not mutated_artifacts
                ):
                    raw = os.pread(descriptor, os.fstat(descriptor).st_size, 0)
                    if mutate_kernel_artifact.endswith(".jsonl"):
                        rows = [json.loads(line) for line in raw.splitlines()]
                        rows[-1]["MESSAGE"] = "NVRM Xid 79"
                        raw = b"".join(S.canonical_bytes(row) + b"\n" for row in rows)
                    else:
                        raw = b"NVRM Xid 79\n"
                    os.ftruncate(descriptor, 0)
                    os.pwrite(descriptor, raw, 0)
                    mutated_artifacts.append(locator)

            def fake_kernel(arguments, target):
                kernel_calls.append(arguments)
                initial = {
                    "__CURSOR": "native-start-cursor",
                    "_BOOT_ID": "a" * 32,
                    "MESSAGE": "clean kernel baseline",
                }
                if arguments == ["--lines=1"]:
                    self.assertFalse(cleaned)
                    self.assertEqual(count, 0)
                    if kernel_problem == "baseline-boot":
                        initial["_BOOT_ID"] = "b" * 32
                    rows = [initial]
                else:
                    self.assertEqual(arguments, ["--cursor=native-start-cursor"])
                    self.assertEqual(cleaned, [True])
                    if kernel_problem == "journal-unavailable":
                        raise ValueError("complete kernel journal is unavailable")
                    if kernel_problem == "lost-cursor":
                        initial["__CURSOR"] = "lost-cursor"
                    if kernel_problem == "mutated-baseline":
                        initial["MESSAGE"] = "baseline payload changed"
                    if kernel_problem == "malformed-journal":
                        os.write(target, b"malformed journal event\n")
                        return
                    rows = [initial]
                    if kernel_problem != "no-new-events":
                        rows.append(
                            {
                                "__CURSOR": "native-end-cursor",
                                "_BOOT_ID": "a" * 32,
                                "MESSAGE": kernel_message,
                            }
                        )
                    if kernel_problem == "changed-record-boot":
                        rows[-1]["_BOOT_ID"] = "b" * 32
                    if kernel_problem == "duplicate-cursor":
                        rows[-1]["__CURSOR"] = "native-start-cursor"
                for row in rows:
                    os.write(target, S.canonical_bytes(row) + b"\n")

            @contextmanager
            def fake_server(_args, _plan, _recipe):
                os.write(stderr_fd, b"Qwen4 QSA physical side cache initialized\n")
                try:
                    yield "fixture-model", proof, 999
                finally:
                    cleaned.append(True)
                    if log_suffix is not None:
                        os.write(stderr_fd, b"\n" + log_suffix + b"\n")
                    if failure == "late-dispatch":
                        os.write(
                            stderr_fd,
                            b"Chunked prefill start: 262144 prompt tokens, 1 requested\n",
                        )
                    if failure == "cleanup":
                        raise ValueError("native cleanup failed")

            class Connection:
                def __init__(connection, *_args, **_kwargs):
                    pass

                def request(connection, method, route, body=None, headers=None):
                    (
                        connection.method,
                        connection.route,
                        connection.body,
                        connection.headers,
                    ) = method, route, body, headers

                def close(connection):
                    pass

                def getresponse(connection):
                    nonlocal count
                    status, headers = 200, []
                    body = (
                        None if connection.body is None else json.loads(connection.body)
                    )
                    if connection.route == "/tokenize":
                        name = next(
                            key
                            for key, text in N.TEXTS.items()
                            if text == body["prompt"]
                        )
                        value = {
                            "tokens": COMPONENTS[name],
                            "count": len(COMPONENTS[name]),
                        }
                    elif connection.route == "/metrics":
                        raw = "".join(
                            f"{key} {value}\n" for key, value in progress.items()
                        ).encode()
                        return Reply(200, raw, [])
                    elif connection.route == "/health":
                        value = {"status": "ok"}
                    else:
                        count += 1
                        prompt, budget = body["prompt_token_ids"], body["max_tokens"]
                        if len(prompt) == N.CONTEXT:
                            status = 400
                            headers = [
                                ("x-request-id", connection.headers["x-request-id"])
                            ]
                            value = {
                                "error": {
                                    "message": "Context too long: 262144 prompt + 1 requested output tokens exceeds max_seq_len 262144",
                                    "type": "invalid_request_error",
                                    "param": None,
                                    "code": None,
                                }
                            }
                            if failure == "overflow":
                                progress["atlas_prefill_progress_tokens_total"] += 1
                        else:
                            progress["atlas_prefill_progress_chunks_total"] += (
                                len(prompt) + 16383
                            ) // 16384
                            progress["atlas_prefill_progress_tokens_total"] += len(
                                prompt
                            )
                            os.write(
                                stderr_fd,
                                f"Chunked prefill start: {len(prompt)} prompt tokens, request\n".encode(),
                            )
                            if len(prompt) in (*N.TARGETS, 262143):
                                receipt = chunk_receipt(len(prompt))
                                if corrupt_receipt:
                                    receipt = (
                                        receipt.replace(
                                            f"end={len(prompt)} ",
                                            f"end={len(prompt)}0 ",
                                        )
                                        .replace("chunk_rows=16384", "chunk_rows=1")
                                        .replace("tile_rows=10", "tile_rows=1")
                                        .replace(
                                            "core=qsa_sparse_bf16_non_bit_exact",
                                            "core=unrelated",
                                        )
                                    )
                                os.write(
                                    stderr_fd,
                                    (receipt + "\n").encode(),
                                )
                            value = self.response(
                                prompt,
                                budget,
                                body["ignore_eos"],
                                labeled_answer(),
                            )
                            value["id"] = "fixture-" + str(count)
                            if failure == "retrieval":
                                value["choices"][0]["text"] = "wrong answer"
                            if retrieval_text is not None and not body["ignore_eos"]:
                                value["choices"][0]["text"] = retrieval_text
                            if fill_text is not None and body["ignore_eos"] and budget == 64:
                                value["choices"][0]["text"] = fill_text
                            if failure == "ledger":
                                value.pop("interventions")
                            if failure == "boundary" and len(prompt) == 262143:
                                value["choices"][0]["token_ids"] = [999]
                    return Reply(status, S.canonical_bytes(value), headers)

            class Reply:
                def __init__(reply, status, raw, headers):
                    reply.status, reply.raw, reply.headers = status, raw, headers

                def read(reply, maximum):
                    return reply.raw[:maximum]

                def getheaders(reply):
                    return reply.headers

            hashes = {
                target: lean_token_hash(N.assemble(target, COMPONENTS)[0])
                for target in N.TARGETS
            }
            try:
                with (
                    mock.patch.object(N, "token_hash", lean_token_hash),
                    mock.patch.object(C, "inspect_container", return_value=(plan, {})),
                    # 295a6aaa1: the driver takes its recipe from the sealed
                    # parent plan /4's native-context contract.
                    mock.patch.object(
                        S,
                        "load_native_context_recipe",
                        side_effect=lambda _args, image, gate: (native_plan(image, gate), {}),
                    ),
                    mock.patch.object(N, "server", side_effect=fake_server),
                    mock.patch.object(N, "native_process", return_value=controls),
                    mock.patch.object(N.select, "select", return_value=([], [], [])),
                    mock.patch.object(N.http.client, "HTTPConnection", Connection),
                    mock.patch.object(
                        N,
                        "Streams",
                        side_effect=lambda: real_streams((stdout_fd, stderr_fd)),
                    ),
                    mock.patch.dict(N.EXPECTED_PROMPT_SHA256, hashes),
                    mock.patch.object(N.time, "sleep"),
                    mock.patch.object(S, "assert_path_identity", side_effect=assert_identity),
                    mock.patch.object(N, "kernel_command", side_effect=fake_kernel),
                    mock.patch.object(
                        N,
                        "current_boot",
                        side_effect=[
                            "a" * 32,
                            ("b" if kernel_problem == "changed-live-boot" else "a") * 32,
                        ],
                    ),
                ):
                    if failure:
                        with self.assertRaises(ValueError):
                            N.run(args)
                        self.assertEqual(os.fstat(receipt_fd).st_size, 0)
                        self.assertFalse(
                            (runtime / "native262k/validator.json").exists()
                        )
                    else:
                        receipt = N.run(args)
                        self.assertIs(receipt["release_qualified"], True)
                        self.assertEqual(
                            receipt["execution_scope"], "immutable-container"
                        )
                        # The outer validator binds to the sealed suite plan /4
                        # (295a6aaa1). Its program closure is test_r15_exact_image_
                        # release's job; here the plan is the checked-in example.
                        example = json.loads(
                            (KIT / "examples/r15-container-suite-plan.example.json").read_bytes()
                        )
                        with mock.patch.object(Q, "_profile_plan", return_value=example):
                            Q._validate_native_262k_evidence(
                                receipt["evidence"],
                                {
                                    "_runtime_dir": str(runtime),
                                    "launch": {"run_id": args.run_id},
                                    "image": {"in_container_binary_sha256": plan["in_container_binary_sha256"]},
                                },
                                args.container_id,
                            )
                        validator = json.loads(
                            (runtime / "native262k/validator.json").read_bytes()
                        )
                        raw = validator["raw"]
                        self.assertEqual(len(raw["cases"]), 8)
                        self.assertEqual(
                            raw["final_dispatch_census"]["rejected_prompt_dispatches"],
                            0,
                        )
                        # Streamed: every request entry carries its full
                        # 262K-token prompt, so parsing all 38 at once does
                        # not fit under the suite's address-space cap.
                        journal_sha, entries = hashlib.sha256(), 0
                        with (runtime / "native262k/responses.jsonl").open("rb") as journal:
                            for line in journal:
                                journal_sha.update(line)
                                entry = json.loads(line)
                                entries += 1
                                if entry["event"] == "response":
                                    self.assertEqual(
                                        N.digest(base64.b64decode(entry["body_base64"])),
                                        entry["body_sha256"],
                                    )
                                del entry
                        self.assertEqual(
                            journal_sha.hexdigest(), raw["response_journal"]["sha256"]
                        )
                        self.assertEqual(entries, 38)
                        self.assertEqual(
                            os.stat(runtime / "native262k/responses.jsonl").st_mode
                            & 0o777,
                            0o400,
                        )
                        for stream in raw["server_streams"]:
                            self.assertEqual(
                                N.digest((runtime / stream["artifact"]).read_bytes()),
                                stream["sha256"],
                            )
                        kernel = raw["kernel_window"]
                        self.assertEqual(kernel["source"], "current-boot-kernel-journal")
                        self.assertEqual(kernel["boot_id"], "a" * 32)
                        self.assertEqual(kernel["fatal_events"], 0)
                        self.assertEqual(
                            kernel["measured_events"],
                            0 if kernel_problem == "no-new-events" else 1,
                        )
                        self.assertEqual(len(kernel["artifacts"]), 3)
                        for artifact in kernel["artifacts"]:
                            payload = (runtime / artifact["artifact"]).read_bytes()
                            self.assertEqual(N.digest(payload), artifact["sha256"])
                            self.assertEqual(len(payload), artifact["bytes"])
                        self.assertNotIn(
                            b"clean measured kernel fixture",
                            (runtime / "native262k/server-stream-1.log").read_bytes(),
                        )
                baseline_failed = kernel_problem == "baseline-boot"
                self.assertEqual(cleaned, [] if baseline_failed else [True])
                self.assertEqual(len(kernel_calls), 1 if baseline_failed else 2)
                if mutate_kernel_artifact is not None:
                    self.assertEqual(
                        mutated_artifacts, ["native262k/" + mutate_kernel_artifact]
                    )
                self.assertEqual(
                    raw_path.read_bytes(), b"canonical raw remains read only"
                )
                if failure:
                    self.assertTrue((runtime / "native262k/responses.jsonl").exists())
                    self.assertTrue(
                        (runtime / "native262k/server-stream-1.log").exists()
                    )
            finally:
                for descriptor in descriptors:
                    os.close(descriptor)

    def test_log_comparator_requires_bf16_terminal_chunks_and_full_dispatch_census(
        self,
    ):
        logs = ["Qwen4 QSA physical side cache initialized"]
        expected = {target: 1 for target in (*N.TARGETS, 262143)}
        for target in expected:
            logs += [
                f"Chunked prefill start: {target} prompt tokens, request",
                chunk_receipt(target),
            ]
        raw = "\n".join(logs).encode()
        N.validate_logs(E, raw, expected)
        for addition in (
            b"\ncore=qsa_sparse_e4m3_fp8",
            b"\nCUDA out of memory",
            b"\nCUDA_ERROR_OUT_OF_MEMORY",
            b"\ncudaErrorMemoryAllocation",
            b"\nCUDA_ERROR_ILLEGAL_ADDRESS",
            b"\ncudaErrorIllegalAddress",
            b"\nCUDA-out-of-memory",
            b"\nOut of memory: Killed process 123",
            b"\noom-kill:constraint=CONSTRAINT_MEMCG",
            b"\nAllocator warning: allocation failed",
            b"\nNVRM: GPU has fallen off the bus",
            b"\nXid 43",
            b"\nChunked prefill start: 262144 prompt tokens, overflow",
        ):
            with self.assertRaises(ValueError):
                N.validate_logs(E, raw + addition, expected)
        with self.assertRaises(ValueError):
            N.validate_logs(
                E,
                raw.replace(b"rows=16383 end=262143", b"rows=16382 end=262143"),
                expected,
            )
        for field, wrong in (
            (b"end=262143", b"end=2621430"),
            (b"chunk_rows=16384", b"chunk_rows=1"),
            (b"tile_rows=10", b"tile_rows=1"),
            (b"core=qsa_sparse_bf16_non_bit_exact", b"core=unrelated"),
            (
                b"core=qsa_sparse_bf16_non_bit_exact",
                b"core=qsa_sparse_bf16_non_bit_exact_wrong",
            ),
            (b"end=262143 chunk_rows", b"end=262143_wrong chunk_rows"),
        ):
            with self.subTest(field=field, wrong=wrong), self.assertRaises(ValueError):
                N.validate_logs(E, raw.replace(field, wrong), expected)


if __name__ == "__main__":
    unittest.main()
