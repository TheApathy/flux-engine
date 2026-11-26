# SPDX-License-Identifier: AGPL-3.0-only
"""Offline-only tests. All server, Docker, and HTTP execution is mocked."""

import copy
from contextlib import contextmanager
import hashlib
import importlib.machinery
import importlib.util
import json
import marshal
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def module(path, name):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    loader.exec_module(value)
    return value


S = module(ROOT / "r15-container-suite.py", "r16_accuracy_suite_fixture")
Q = module(ROOT / "r15_exact_image_release.py", "r16_accuracy_outer_fixture")
DRIVERS = [
    module(ROOT / "r15-gates" / gate, "r16_accuracy_" + gate.replace("-", "_"))
    for gate in ("mmlu-pro", "gsm8k")
]


LIL_PATH = ROOT / "llm-inference-bench/llm_decode_bench.py"
# Compiles the pinned file in a fresh interpreter and returns its code object.
COMPILE_CHILD = (
    "import marshal, sys; p = sys.argv[1]; "
    "sys.stdout.buffer.write(marshal.dumps(compile(open(p, 'rb').read(), p, 'exec')))"
)


def load_pinned_lil(driver):
    # Compiling the 23k-line lil bench costs ~40 MB of address space, which does
    # not fit beside pytest under the 90 MB suite cap. Compile it in a fresh
    # interpreter, then run the driver's own load_module (read, pin check,
    # exec) with its compile() returning that code for the verified bytes.
    child = subprocess.run(
        [sys.executable, "-I", "-B", "-c", COMPILE_CHILD, str(LIL_PATH)],
        check=True, capture_output=True,
    )
    code = marshal.loads(child.stdout)

    def compile_pinned(payload, filename, mode):
        if (hashlib.sha256(payload).hexdigest(), filename, mode) != (
            driver.LIL_SHA256, str(LIL_PATH), "exec"
        ):
            raise AssertionError("precompiled lil does not match the verified payload")
        return code

    with mock.patch.object(driver, "compile", compile_pinned, create=True):
        return driver.load_module(LIL_PATH, driver.LIL_SHA256, "r16_accuracy_actual_lil")


class RealAccuracyDrivers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bench = load_pinned_lil(DRIVERS[0])

    def response(self, answer="A", model="fixture-model", tokens=None):
        return {
            "id": "fixture-id",
            "model": model,
            "choices": [
                {
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                    "token_ids": [1, 2, 3] if tokens is None else tokens,
                }
            ],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 3,
                "total_tokens": 53,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
            "interventions": {
                "schema": "atlas.interventions.v1",
                "policy": "report_only",
                "events": [],
            },
            "rendered_prompt_evidence": {
                "schema": "atlas.rendered_prompt.v1",
                "nonce": "a" * 32,
                "response_id": "fixture-id",
                "prompt_sha256": "b" * 64,
            },
        }

    def test_pinned_dataset_apis_cover_every_real_item_without_downloads(self):
        for driver in DRIVERS:
            name, filename, digest, count, _floor = driver.CONTRACTS[driver.GATE]
            path = ROOT / "llm-inference-bench/data" / filename
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            with mock.patch.object(
                self.bench,
                "resolve_benchmark_dataset",
                return_value=(str(path), digest),
            ):
                items, _metadata = self.bench.load_benchmark_dataset_items(
                    driver.GATE, self.bench.BUILTIN_TEST_PROFILES[driver.GATE]
                )
            self.assertEqual(len(items), count)
            self.assertEqual(len({item["item_id"] for item in items}), count)
            self.assertTrue(
                all(item["prompt"] and item["expected_answer"] for item in items)
            )

    def test_source_dependency_hash_mismatch_fails_before_module_execution(self):
        for driver in DRIVERS:
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "untrusted.py"
                path.write_text("raise AssertionError('must not execute')\n")
                with self.assertRaisesRegex(ValueError, "dependency hash"):
                    driver.load_module(path, "1" * 64, "untrusted_accuracy_dependency")

    def test_real_lil_scoring_and_scheduler_token_hashes(self):
        for driver in DRIVERS:
            profile = self.bench.BUILTIN_TEST_PROFILES[driver.GATE]
            item = (
                {
                    "item_id": "fixture",
                    "expected_answer": "A",
                    "expected_letter": "A",
                    "num_options": 4,
                }
                if driver.GATE == "mmlu-pro"
                else {
                    "item_id": "fixture",
                    "expected_answer": "42",
                    "expected_number": 42.0,
                }
            )
            response = self.response(item["expected_answer"])
            measured = driver.response_measurement(
                types.SimpleNamespace(suite=S),
                self.bench,
                profile,
                item,
                "fixture-model",
                response,
                "a" * 32,
            )
            self.assertIs(measured["correct"], True)
            self.assertEqual(
                measured["token_ids_sha256"],
                hashlib.sha256(S.canonical_bytes([1, 2, 3])).hexdigest(),
            )
            response["choices"][0]["message"]["content"] = (
                "B" if driver.GATE == "mmlu-pro" else "43"
            )
            wrong = driver.response_measurement(
                types.SimpleNamespace(suite=S),
                self.bench,
                profile,
                item,
                "fixture-model",
                response,
                "a" * 32,
            )
            self.assertIs(wrong["correct"], False)

    def test_missing_tokens_incomplete_output_and_interventions_fail_closed(self):
        driver = DRIVERS[0]
        item = {
            "item_id": "fixture",
            "expected_answer": "A",
            "expected_letter": "A",
            "num_options": 4,
        }
        changes = (
            lambda row: row["choices"][0].pop("token_ids"),
            lambda row: row["choices"][0].update(token_ids=[1, True, 3]),
            lambda row: row["choices"][0].update(token_ids=[1, 2]),
            lambda row: row["choices"][0].update(finish_reason="length"),
            lambda row: row["choices"][0]["message"].update(content=""),
            lambda row: row["choices"][0]["message"].update(tool_calls=[{}]),
            lambda row: row.pop("interventions"),
            lambda row: row["interventions"].update(
                events=[{"code": "watchdog", "action": "stop", "count": 1}]
            ),
            lambda row: row["usage"].update(completion_tokens=True),
            lambda row: row["usage"].update(total_tokens=54),
            lambda row: row["usage"]["prompt_tokens_details"].update(cached_tokens=1),
            lambda row: row["usage"].pop("prompt_tokens_details"),
            lambda row: row.pop("rendered_prompt_evidence"),
            lambda row: row["rendered_prompt_evidence"].update(nonce="wrong"),
            lambda row: row["rendered_prompt_evidence"].update(response_id="wrong"),
            lambda row: row.update(model="other-model"),
        )
        for change in changes:
            value = self.response()
            change(value)
            with self.subTest(change=change), self.assertRaises(ValueError):
                driver.response_measurement(
                    types.SimpleNamespace(suite=S),
                    self.bench,
                    self.bench.BUILTIN_TEST_PROFILES[driver.GATE],
                    item,
                    "fixture-model",
                    value,
                    "a" * 32,
                )

    def test_pair_rejects_equal_count_different_ids_and_swapped_correctness(self):
        driver = DRIVERS[0]
        first = {
            "correct": True,
            "parsed_answer": "A",
            "answer_sha256": "a",
            "reasoning_sha256": "b",
            "rendered_prompt_sha256": "d",
            "finish_reason": "stop",
            "token_ids_sha256": "c",
            "completion_tokens": 3,
            "prompt_tokens": 50,
        }
        plain = {"one": first, "two": dict(first, correct=False)}
        for changed in (
            {
                "one": dict(first, token_ids_sha256="d"),
                "two": dict(first, correct=False),
            },
            {"one": dict(first, correct=False), "two": dict(first, correct=True)},
            {"one": first},
        ):
            with self.assertRaises(ValueError):
                driver.compare_pairs(plain, changed, 2, 0.0)
        rows, _observations, correct = driver.compare_pairs(
            plain, copy.deepcopy(plain), 2, 0.5
        )
        self.assertEqual(correct, 1)
        self.assertEqual(len(rows), 2)
        with self.assertRaisesRegex(ValueError, "sanity floor"):
            driver.compare_pairs(plain, copy.deepcopy(plain), 2, 0.65)

    def exercise_mocked_run(self, driver, mutation=None, *, missing_k5=False):
        """Explicit two-item evaluator fixture, never production qualification."""
        dataset_name, filename, digest, _count, floor = driver.CONTRACTS[driver.GATE]
        path = ROOT / "llm-inference-bench/data" / filename
        with mock.patch.object(
            self.bench, "resolve_benchmark_dataset", return_value=(str(path), digest)
        ):
            items, _metadata = self.bench.load_benchmark_dataset_items(
                driver.GATE, self.bench.BUILTIN_TEST_PROFILES[driver.GATE]
            )
        fixture_items = items[:2]
        mode = [None]
        read_fd, write_fd = os.pipe()
        try:
            with tempfile.TemporaryDirectory() as temp:
                runtime = Path(temp)
                receipt = runtime / "receipt.json"
                receipt_fd = os.open(
                    receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
                )
                args = types.SimpleNamespace(
                    suite=S,
                    lil_program=ROOT / "llm-inference-bench/llm_decode_bench.py",
                    runtime_dir=str(runtime),
                    receipt_fd=receipt_fd,
                    container_id="1" * 64,
                    image_id="sha256:" + "2" * 64,
                    image_digest="sha256:" + "3" * 64,
                    run_id="offline-fixture",
                )

                @contextmanager
                def fixture_server(_args, arm_mode, _plan, _recipe):
                    mode[0] = arm_mode
                    yield "fixture-model-" + arm_mode, {
                        "mode": arm_mode,
                        "offline_test_only": True,
                    }, read_fd

                def fixture_request(_args, method, route, body=None):
                    if route == "/metrics":
                        if mode[0] == "none" or missing_k5:
                            return b"# No actual K5 verification event\n"
                        return (
                            b'atlas_spec_decode_verify_total{k="5",outcome="accepted"} '
                            + (b"1\n" if mode[0] == "native-mtp-k5" else b"0\n")
                        )
                    index = next(
                        index
                        for index, item in enumerate(fixture_items)
                        if item["prompt"] == body["messages"][0]["content"]
                    )
                    row = self.response(
                        fixture_items[index]["expected_answer"], body["model"]
                    )
                    row["id"] = mode[0] + "-" + str(index)
                    row["rendered_prompt_evidence"]["nonce"] = body[
                        "rendered_prompt_evidence_nonce"
                    ]
                    row["rendered_prompt_evidence"]["response_id"] = row["id"]
                    if mutation:
                        mutation(row, mode[0], index)
                    return S.canonical_bytes(row)

                patches = (
                    mock.patch.dict(
                        driver.CONTRACTS,
                        {driver.GATE: (dataset_name, filename, digest, 2, floor)},
                    ),
                    mock.patch.object(driver, "load_module", return_value=self.bench),
                    mock.patch.object(
                        self.bench,
                        "load_benchmark_dataset_items",
                        return_value=(fixture_items, {}),
                    ),
                    mock.patch.object(
                        driver, "inspect_container", return_value=({}, {})
                    ),
                    mock.patch.object(driver, "server", side_effect=fixture_server),
                    mock.patch.object(driver, "request", side_effect=fixture_request),
                )
                from contextlib import ExitStack

                try:
                    with ExitStack() as stack:
                        for patch in patches:
                            stack.enter_context(patch)
                        if mutation or missing_k5:
                            with self.assertRaises(ValueError):
                                driver.run(args)
                            self.assertEqual(receipt.read_bytes(), b"")
                            self.assertFalse(
                                (
                                    runtime
                                    / "accuracy"
                                    / driver.GATE
                                    / "validator.json"
                                ).exists()
                            )
                        else:
                            result = driver.run(args)
                            evidence = result["evidence"]
                            Q._validate_accuracy_evidence(
                                evidence,
                                manifest={
                                    "_runtime_dir": str(runtime),
                                    "launch": {"run_id": args.run_id},
                                },
                                container_id=args.container_id,
                                gate=driver.GATE,
                                dataset_sha256=digest,
                                items=2,
                                floor=floor,
                            )
                            raw = (
                                runtime / evidence["validator_artifact"]
                            ).read_bytes()
                            self.assertEqual(
                                hashlib.sha256(raw).hexdigest(),
                                evidence["validator_receipt_sha256"],
                            )
                            document = json.loads(raw)
                            journal = document["raw"]["response_journal"]
                            journal_raw = (runtime / journal["artifact"]).read_bytes()
                            self.assertEqual(
                                hashlib.sha256(journal_raw).hexdigest(),
                                journal["sha256"],
                            )
                            self.assertEqual(len(journal_raw.splitlines()), 4)
                            self.assertTrue(
                                all(
                                    "token_ids"
                                    in json.loads(line)["response"]["choices"][0]
                                    for line in journal_raw.splitlines()
                                )
                            )
                            self.assertEqual(json.loads(receipt.read_bytes()), result)
                finally:
                    os.close(receipt_fd)
        finally:
            os.close(read_fd)
            os.close(write_fd)

    def test_mocked_end_to_end_collects_real_scorer_outputs_and_journals(self):
        for driver in DRIVERS:
            with self.subTest(gate=driver.GATE):
                self.exercise_mocked_run(driver)

    def test_mocked_end_to_end_refuses_token_mismatch_without_pass_receipt(self):
        def different_tokens(row, mode, index):
            if mode == "native-mtp-k5" and index == 0:
                row["choices"][0]["token_ids"] = [1, 2, 4]

        for driver in DRIVERS:
            self.exercise_mocked_run(driver, different_tokens)

    def test_mocked_end_to_end_refuses_reused_response_ids(self):
        self.exercise_mocked_run(
            DRIVERS[0], lambda row, _mode, _index: row.update(id="reused")
        )

    def test_mocked_end_to_end_refuses_missing_native_k5_verifier_events(self):
        self.exercise_mocked_run(DRIVERS[0], missing_k5=True)

    def test_mocked_end_to_end_refuses_rendered_prompt_drift(self):
        def different_prompt(row, mode, _index):
            if mode == "native-mtp-k5":
                row["rendered_prompt_evidence"]["prompt_sha256"] = "c" * 64

        self.exercise_mocked_run(DRIVERS[0], different_prompt)

    def test_symlink_runtime_artifact_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "outside").mkdir()
            (root / "accuracy").symlink_to(root / "outside", target_is_directory=True)
            with self.assertRaises(OSError):
                DRIVERS[0].make_directory(S, root, ["accuracy", "mmlu-pro"])

    def test_parent_verifier_pin_requires_sealed_actual_launch_bytes(self):
        driver = DRIVERS[0]
        descriptor = S.anonymous_writable_file("offline-parent-suite")
        S.write_all(descriptor, b"offline sealed suite fixture")
        expected = hashlib.sha256(b"offline sealed suite fixture").hexdigest()
        try:
            with mock.patch.object(
                driver.os, "getppid", return_value=123
            ), mock.patch.object(
                driver.Path, "read_bytes", return_value=b"python3\0/proc/self/fd/99\0"
            ), mock.patch.object(
                driver.os,
                "open",
                side_effect=lambda *_args, **_kwargs: os.dup(descriptor),
            ):
                with self.assertRaisesRegex(ValueError, "not sealed"):
                    driver.verify_parent_suite(expected)
                S.seal_written_file(descriptor, "offline parent", 1000)
                driver.verify_parent_suite(expected)
                with self.assertRaisesRegex(ValueError, "differs"):
                    driver.verify_parent_suite("1" * 64)
        finally:
            os.close(descriptor)

    def exercise_mocked_server(
        self, driver, mode, *, force_cleanup=False, verifier_failure=False
    ):
        read_fd, write_fd = os.pipe()
        dead = [False]
        process = mock.Mock()
        process.poll.return_value = None
        process.wait.side_effect = lambda **_kwargs: setattr(
            process.poll, "return_value", 0
        )
        proof = {"offline_test_only": True, "mode": mode}
        verifier = mock.Mock(
            side_effect=ValueError("live binding failed") if verifier_failure else None,
            return_value=proof,
        )
        args = types.SimpleNamespace(
            container_id="1" * 64,
            container_port=8888,
            suite=types.SimpleNamespace(verify_live_server=verifier),
            identities={"base_url": "http://127.0.0.1:8888"},
        )

        def sent(_descriptor, signum):
            if signum == signal.SIGKILL or not force_cleanup:
                dead[0] = True

        try:
            with mock.patch.object(
                driver.subprocess, "Popen", return_value=process
            ) as launch, mock.patch.object(
                driver, "find_pid", return_value=123
            ), mock.patch.object(
                driver, "claim_pid", side_effect=lambda *_args: os.dup(read_fd)
            ), mock.patch.object(
                driver, "request", return_value=b"{}"
            ) as endpoint, mock.patch.object(
                driver.select,
                "select",
                side_effect=lambda descriptors, _write, _error, _timeout: (
                    descriptors if dead[0] else [],
                    [],
                    [],
                ),
            ), mock.patch.object(
                driver.signal, "pidfd_send_signal", side_effect=sent
            ) as kill:
                if force_cleanup or verifier_failure:
                    with self.assertRaisesRegex(
                        ValueError,
                        "forced cleanup" if force_cleanup else "live binding",
                    ):
                        with driver.server(
                            args,
                            mode,
                            {
                                "in_container_binary_path": "/opt/flux/bin/flux-server-real"
                            },
                            {},
                        ):
                            pass
                else:
                    with driver.server(
                        args,
                        mode,
                        {"in_container_binary_path": "/opt/flux/bin/flux-server-real"},
                        {},
                    ) as (_model, actual_proof, _pidfd):
                        self.assertEqual(actual_proof, proof)
                    self.assertEqual(verifier.call_count, 2)
                self.assertEqual(endpoint.call_count, 2)
                command = launch.call_args.args[0]
                self.assertIn(
                    "ATLAS_DISABLE_SPECULATION=" + ("1" if mode == "none" else "0"),
                    command,
                )
                self.assertIn("--no-tui", command)
                self.assertIn("--qwen4-qsa", command)
                self.assertEqual(command[command.index("--max-seq-len") + 1], "262144")
                self.assertEqual(command[command.index("--kv-cache-dtype") + 1], "bf16")
                self.assertEqual(command.count("--request-timeout"), 1)
                server_timeout = int(command[command.index("--request-timeout") + 1])
                self.assertEqual(server_timeout, 14340)
                self.assertEqual(server_timeout, driver.SERVER_REQUEST_TIMEOUT)
                self.assertLess(server_timeout, driver.REQUEST_TIMEOUT)
                self.assertEqual(
                    command[command.index("--model-from-path") + 1], "/models"
                )
                self.assertNotIn("--mtp-gate", command)
                self.assertEqual("--speculative" in command, mode == "native-mtp-k5")
                self.assertEqual(
                    [call.args[1] for call in kill.call_args_list],
                    (
                        [signal.SIGTERM, signal.SIGKILL]
                        if force_cleanup
                        else [signal.SIGTERM]
                    ),
                )
                process.wait.assert_called()
        finally:
            os.close(read_fd)
            os.close(write_fd)

    def test_mocked_lifecycle_uses_canonical_argv_double_health_and_exact_cleanup(self):
        for driver in DRIVERS:
            for mode in ("none", "native-mtp-k5"):
                self.exercise_mocked_server(driver, mode)

    def test_mocked_forced_cleanup_fails_the_gate(self):
        self.exercise_mocked_server(DRIVERS[0], "none", force_cleanup=True)

    def test_mocked_binding_failure_still_terminates_the_claimed_server(self):
        self.exercise_mocked_server(DRIVERS[0], "none", verifier_failure=True)

    def test_http_deadline_remains_later_than_explicit_finite_server_deadline(self):
        for driver in DRIVERS:
            with self.subTest(gate=driver.GATE):
                connection = mock.Mock()
                connection.getresponse.return_value.status = 200
                connection.getresponse.return_value.read.return_value = b"{}"
                args = types.SimpleNamespace(base_url="http://127.0.0.1:8888", suite=S)
                with mock.patch.object(
                    driver.http.client, "HTTPConnection", return_value=connection
                ) as connect:
                    driver.request(args, "POST", "/v1/chat/completions", {})
                    self.assertEqual(connect.call_args.kwargs["timeout"], 14400)
                    self.assertGreater(
                        connect.call_args.kwargs["timeout"], driver.SERVER_REQUEST_TIMEOUT
                    )
                    driver.request(args, "GET", "/health", timeout=2)
                    self.assertEqual(connect.call_args.kwargs["timeout"], 2)
                self.assertEqual(connection.close.call_count, 2)

    def test_standalone_drivers_only_differ_by_fixed_gate_identity(self):
        first = (
            (ROOT / "r15-gates/mmlu-pro")
            .read_text()
            .replace('GATE = "mmlu-pro"', 'GATE = "GATE"')
        )
        second = (
            (ROOT / "r15-gates/gsm8k")
            .read_text()
            .replace('GATE = "gsm8k"', 'GATE = "GATE"')
        )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
