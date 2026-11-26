#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Original Flux Lens contract tests. All HTTP is mocked, never a live socket."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import threading
import unittest
from unittest import mock
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import flux_lens as lens  # noqa: E402

RELEASE_MAX_BYTES = lens.MAX_BYTES
# Bounded reads (read(MAX_BYTES + 1)) allocate the full bound up front, so the
# 64 MiB release bound exceeds the 90 MB address-space cap these suites run
# under. Every fixture is far smaller; the bound checks themselves do not
# depend on its size, and test_release_artifact_bound_is_64_mib pins it.
EXERCISE_MAX_BYTES = 1024 * 1024


def corpus(*, expected=True):
    case = {"id": "private-case", "prompt_token_ids": [10, 11, 12], "max_tokens": 3}
    if expected:
        case["expected_token_ids"] = [30, 31]
    return {
        "schema": lens.CORPUS_SCHEMA,
        "model": {"name": "declared-model", "revision": "declared-checkpoint-revision"},
        "cases": [case],
    }


def response(payload, *, ids=None, finish="stop"):
    ids = [30, 31] if ids is None else ids
    prompt = len(payload["prompt_token_ids"])
    return {
        "model": payload["model"],
        "choices": [
            {
                "index": 0,
                "text": "visible answer",
                "token_ids": ids,
                "finish_reason": finish,
            }
        ],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": len(ids),
            "total_tokens": prompt + len(ids),
            "prompt_tokens_details": {"cached_tokens": 0},
        },
    }


def good_transport(_endpoint, payload, _timeout, _key):
    return lens.HttpReply(200, lens.canonical(response(payload)))


class LensCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="flux-lens-mock-")
        self.root = Path(self.temp.name)
        # Fail the entire suite if any code accidentally attempts a real socket.
        self.network_guard = mock.patch(
            "socket.socket", side_effect=AssertionError("network forbidden")
        )
        self.network_guard.start()
        self.max_bytes = mock.patch.object(lens, "MAX_BYTES", EXERCISE_MAX_BYTES)
        self.max_bytes.start()
        self.addCleanup(self.max_bytes.stop)

    def tearDown(self):
        self.network_guard.stop()
        self.temp.cleanup()

    def error(self, code, function, *args, **kwargs):
        with self.assertRaises(lens.LensError) as caught:
            function(*args, **kwargs)
        self.assertEqual(caught.exception.code, code)

    def make_run(self, name="run", *, plan=None, transport=good_transport, **kwargs):
        directory = self.root / name
        result = lens.run(
            plan or lens.make_plan(corpus(), [1], [1]),
            "http://127.0.0.1:12345/v1/completions",
            directory,
            transport=transport,
            **kwargs,
        )
        return directory, result

    def reseal(self, directory):
        manifest = {
            "schema": lens.RUN_SCHEMA,
            "files": {
                name: lens.sha(lens.read_bytes(directory / name, private=True))
                for name in lens.artifact_files(directory)
            },
        }
        (directory / "manifest.json").write_bytes(lens.canonical(manifest))

    def rewrite_json(self, path, change):
        value = lens.parse_json(path.read_bytes())
        change(value)
        path.write_bytes(lens.canonical(value))


class CorpusAndPlanTests(LensCase):
    def test_valid_corpus_and_generation_are_explicit(self):
        plan = lens.make_plan(corpus(), [1, 2, 4], [4, 8], repeats=2)
        self.assertEqual(lens.validate_plan(plan), plan)
        self.assertEqual(len(lens.cells(plan)), 12)
        payload, case = lens.payload_for(plan, 0)
        self.assertEqual(payload["prompt_token_ids"], case["prompt_token_ids"])
        self.assertNotIn("prompt", payload)
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["presence_penalty"], 0)
        self.assertEqual(payload["repetition_penalty"], 1)
        self.assertIs(payload["return_token_ids"], True)
        self.assertIs(payload["stream"], False)

    def test_every_case_gets_its_own_context_concurrency_cell(self):
        value = corpus()
        second = copy.deepcopy(value["cases"][0])
        second.update(
            id="longer-private-case", prompt_token_ids=list(range(100)), max_tokens=4
        )
        value["cases"].append(second)
        plan = lens.make_plan(value, [1, 2], [2])
        layout = lens.cells(plan)
        self.assertEqual(
            [(c["case_index"], c["prompt_tokens"], c["concurrency"]) for c in layout],
            [(0, 3, 1), (0, 3, 2), (1, 100, 1), (1, 100, 2)],
        )
        self.assertEqual(
            lens.payload_for(plan, 1)[0]["prompt_token_ids"], list(range(100))
        )

    def test_integer_fields_reject_bool_float_and_negative(self):
        for bad in [True, False, 1.0, -1, "1", None, 1 << 32]:
            with self.subTest(bad=bad):
                value = corpus()
                value["cases"][0]["prompt_token_ids"][0] = bad
                self.error("TOKEN_ID_TYPE_OR_RANGE", lens.validate_corpus, value)
        for bad in [True, 1.0, 0, -1, "3"]:
            value = corpus()
            value["cases"][0]["max_tokens"] = bad
            self.error("MAX_TOKENS", lens.validate_corpus, value)

    def test_empty_tokens_and_empty_expected_fail(self):
        for field in ["prompt_token_ids", "expected_token_ids"]:
            value = corpus()
            value["cases"][0][field] = []
            self.error("EMPTY_TOKEN_IDS", lens.validate_corpus, value)

    def test_model_revision_and_unique_case_ids_required(self):
        value = corpus()
        value["model"]["revision"] = ""
        self.error("MODEL_REVISION_REQUIRED", lens.validate_corpus, value)
        value = corpus()
        value["cases"] *= 2
        self.error("DUPLICATE_CASE_ID", lens.validate_corpus, value)

    def test_unknown_corpus_fields_fail(self):
        value = corpus()
        value["arm"] = "native-k5"
        self.error("OBJECT_KEYS", lens.validate_corpus, value)

    def test_json_duplicate_nonfinite_empty_and_truncated_fail(self):
        for data, code in [
            (b'{"a":1,"a":2}', "DUPLICATE_JSON_KEY"),
            (b'{"x":NaN}', "NONFINITE_JSON"),
            (b'{"x":Infinity}', "NONFINITE_JSON"),
            (b'{"x":1e999}', "NONFINITE_JSON"),
            (b"", "JSON_SIZE"),
            (b'{"x":', "INVALID_JSON"),
            (b"\xff", "INVALID_JSON"),
        ]:
            self.error(code, lens.parse_json, data)

    def test_matrix_rejects_duplicates_bad_types_and_undersized_bursts(self):
        self.error("DUPLICATE_MATRIX_WIDTH", lens.make_plan, corpus(), [1, 1], [1])
        self.error("MATRIX_WIDTH", lens.make_plan, corpus(), [True], [1])
        self.error("MATRIX_WIDTH", lens.make_plan, corpus(), [1.0], [1])
        self.error(
            "BURST_SMALLER_THAN_CONCURRENCY", lens.make_plan, corpus(), [1, 4], [2]
        )
        self.error("TOO_MANY_REQUESTS", lens.make_plan, corpus(), [1, 2], [10_000])
        self.error("REPEATS", lens.make_plan, corpus(), [1], [1], True)
        self.error("MATRIX_WIDTH", lens.make_plan, corpus(), [129], [129])

    def test_plan_hash_and_fixed_generation_are_checked(self):
        plan = lens.make_plan(corpus(), [1], [1])
        plan["workload_sha256"] = "0" * 64
        self.error("PLAN_HASH_MISMATCH", lens.validate_plan, plan)
        for field, bad in [
            ("temperature", False),
            ("n", True),
            ("stream", 0),
            ("top_k", 5),
        ]:
            plan = lens.make_plan(corpus(), [1], [1])
            plan["workload"]["generation"][field] = bad
            self.error("PLAN_CONTRACT_MISMATCH", lens.validate_plan, plan)

    def test_url_requires_exact_path_no_credentials_query_fragment_or_whitespace(self):
        for bad in [
            "ftp://x/v1/completions",
            "http://x/v1/chat/completions",
            "http://x",
            "http://name:password@x/v1/completions",
            "http://x/v1/completions?q=1",
            "http://x/v1/completions#part",
            "http://x:0/v1/completions",
            "http://x:invalid/v1/completions",
            "http://x /v1/completions",
        ]:
            with self.subTest(bad=bad), self.assertRaises(lens.LensError):
                lens.endpoint_url(bad)

    def test_plan_cli_is_offline_private_and_does_not_overwrite(self):
        source = self.root / "private-corpus.json"
        source.write_bytes(lens.canonical(corpus()))
        target = self.root / "plan.json"
        args = [
            "plan",
            "--corpus",
            str(source),
            "--concurrency",
            "1,2",
            "--burst-sizes",
            "2,4",
            "--out",
            str(target),
        ]
        stdout = io.StringIO()
        with redirect_stdout(stdout), mock.patch.object(
            lens, "post", side_effect=AssertionError
        ):
            self.assertEqual(lens.main(args), 0)
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o600)
        self.assertFalse(json.loads(stdout.getvalue())["network_contacted"])
        before = target.read_bytes()
        with redirect_stderr(io.StringIO()):
            self.assertEqual(lens.main(args), 2)
        self.assertEqual(target.read_bytes(), before)


class ResponseTests(LensCase):
    def setUp(self):
        super().setUp()
        self.plan = lens.make_plan(corpus(), [1], [1])
        self.payload, _ = lens.payload_for(self.plan, 0)

    def check(self, value, code):
        self.error(
            code, lens.validate_response, lens.canonical(value), self.payload, False
        )

    def test_valid_full_ids_and_usage(self):
        result = lens.validate_response(
            lens.canonical(response(self.payload)), self.payload, False
        )
        self.assertEqual(result, {"token_ids": [30, 31], "finish_reason": "stop"})

    def test_error_empty_choices_missing_ids_and_empty_output_fail(self):
        value = response(self.payload)
        value["error"] = None
        self.check(value, "RESPONSE_ERROR_OR_TYPE")
        value = response(self.payload)
        value["choices"] = []
        self.check(value, "RESPONSE_CHOICES")
        value = response(self.payload)
        value["choices"][0].pop("token_ids")
        self.check(value, "TOKEN_IDS_REQUIRED")
        self.check(response(self.payload, ids=[]), "EMPTY_TOKEN_IDS")
        value = response(self.payload)
        value["choices"][0]["text"] = " \n"
        self.check(value, "EMPTY_OUTPUT_TEXT")

    def test_prompt_usage_and_completion_usage_exact_integer_only(self):
        for field, bad, code in [
            ("prompt_tokens", 4, "PROMPT_USAGE_MISMATCH"),
            ("prompt_tokens", 3.0, "PROMPT_USAGE_MISMATCH"),
            ("completion_tokens", True, "OUTPUT_USAGE_MISMATCH"),
            ("completion_tokens", 1, "OUTPUT_USAGE_MISMATCH"),
            ("total_tokens", 6, "TOTAL_USAGE_MISMATCH"),
        ]:
            value = response(self.payload)
            value["usage"][field] = bad
            self.check(value, code)

    def test_missing_cache_details_are_unknown_not_zero(self):
        value = response(self.payload)
        value["usage"].pop("prompt_tokens_details")
        self.check(value, "CACHE_USAGE_UNKNOWN")
        for bad in [None, False, 0.0, 1, -1]:
            value = response(self.payload)
            value["usage"]["prompt_tokens_details"]["cached_tokens"] = bad
            self.check(value, "CACHE_USAGE_UNKNOWN_OR_NONZERO")

    def test_length_finish_is_explicit_and_requires_full_budget(self):
        value = response(self.payload, ids=[30, 31, 32], finish="length")
        self.check(value, "TRUNCATED_OR_UNKNOWN_FINISH")
        self.assertEqual(
            lens.validate_response(lens.canonical(value), self.payload, True)[
                "token_ids"
            ],
            [30, 31, 32],
        )
        self.error(
            "SHORT_LENGTH_FINISH",
            lens.validate_response,
            lens.canonical(response(self.payload, finish="length")),
            self.payload,
            True,
        )
        for finish in [None, "error", "cancelled", "tool_calls"]:
            self.check(
                response(self.payload, finish=finish), "TRUNCATED_OR_UNKNOWN_FINISH"
            )

    def test_model_and_index_match_without_inferring_actual_identity(self):
        value = response(self.payload)
        value["model"] = "different-model"
        self.check(value, "RESPONSE_MODEL_MISMATCH")
        value = response(self.payload)
        value["choices"][0]["index"] = False
        self.check(value, "CHOICE_INDEX")


class TransportTests(LensCase):
    class Reply:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, size):
            if size != lens.MAX_BYTES + 1:
                raise AssertionError("bounded read required")
            return b"body"

    def test_one_post_no_proxy_no_redirect_and_no_global_urlopen(self):
        opener = mock.Mock()
        opener.open.return_value = self.Reply()
        with mock.patch.dict(
            os.environ,
            {
                "HTTP_PROXY": "http://private.invalid:1",
                "HTTPS_PROXY": "http://private.invalid:2",
            },
        ), mock.patch.object(
            lens.request, "build_opener", return_value=opener
        ) as build, mock.patch.object(
            lens.request, "urlopen", side_effect=AssertionError
        ):
            result = lens.post(
                "http://example.invalid/v1/completions", {"a": 1}, 5, "mock-secret"
            )
        self.assertEqual(result, lens.HttpReply(200, b"body"))
        self.assertEqual(opener.open.call_count, 1)
        proxy, redirect = build.call_args.args
        self.assertEqual(proxy.proxies, {})
        self.assertIsInstance(redirect, lens.NoRedirect)
        outgoing = opener.open.call_args.args[0]
        self.assertEqual(outgoing.get_method(), "POST")
        self.assertEqual(outgoing.get_header("Authorization"), "Bearer mock-secret")
        self.assertIsNone(
            redirect.redirect_request(None, None, 307, "redirect", {}, "http://other")
        )

    def test_http_errors_are_recorded_not_retried(self):
        opener = mock.Mock()
        opener.open.side_effect = HTTPError(
            "http://example.invalid", 503, "busy", {}, io.BytesIO(b"busy")
        )
        with mock.patch.object(lens.request, "build_opener", return_value=opener):
            self.assertEqual(
                lens.post("http://example.invalid/v1/completions", {}, 5, None),
                lens.HttpReply(503, b"busy"),
            )
        self.assertEqual(opener.open.call_count, 1)

    def test_bearer_header_injection_fails_before_open(self):
        with mock.patch.object(lens.request, "build_opener") as build:
            self.error(
                "API_KEY_FORMAT",
                lens.post,
                "http://example.invalid/v1/completions",
                {},
                5,
                "key\r\nOther: value",
            )
        build.return_value.open.assert_not_called()


class RunTests(LensCase):
    def test_cell_clock_is_recorded_in_final_arrival_action_before_gate_release(self):
        real_barrier = threading.Barrier
        inside_release = threading.Event()
        gate_reads = []
        clock_lock = threading.Lock()
        ticks = 100_000_000

        def clock():
            nonlocal ticks
            with clock_lock:
                ticks += 1_000_000
                if inside_release.is_set():
                    gate_reads.append(ticks)
                return ticks

        def barrier_factory(parties, *, action):
            def release():
                inside_release.set()
                try:
                    action()
                finally:
                    inside_release.clear()

            return real_barrier(parties, action=release)

        with mock.patch.object(
            lens.threading, "Barrier", side_effect=barrier_factory
        ), mock.patch.object(lens.time, "monotonic_ns", side_effect=clock):
            directory, _ = self.make_run(plan=lens.make_plan(corpus(), [2], [2]))
        windows = lens.parse_json((directory / "windows.json").read_bytes())
        self.assertEqual(len(gate_reads), 1)
        self.assertEqual(windows[0]["start_monotonic_ns"], gate_reads[0])
        for line in (directory / "journal.jsonl").read_bytes().splitlines():
            self.assertGreater(
                lens.parse_json(line)["start_monotonic_ns"], gate_reads[0]
            )
        self.assertEqual(lens.load_run(directory)[1]["receipt_status"], "complete")

    def test_matrix_seal_and_private_modes(self):
        plan = lens.make_plan(corpus(), [1, 2], [2, 3], 2)
        directory, result = self.make_run(plan=plan)
        self.assertEqual(result["requests"], 20)
        self.assertEqual(result["receipt_status"], "complete")
        self.assertEqual(result["metric"], lens.METRIC)
        self.assertEqual(result["corpus_exact_match"]["matched_requests"], 20)
        for path in [directory, directory / "raw"]:
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o700)
        for path in directory.rglob("*"):
            if path.is_file():
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        loaded_plan, replay, tokens = lens.load_run(directory)
        self.assertEqual(loaded_plan, plan)
        self.assertEqual(replay, result)
        self.assertEqual(len(tokens), 20)
        self.assertTrue(
            all(
                c["e2e_aggregate_output_tokens_per_second"] > 0 for c in result["cells"]
            )
        )

    def test_workers_are_real_client_concurrency_not_serial_request_loop(self):
        # Four default 8 MiB thread stacks do not fit beside the import
        # baseline under the 90 MB address-space cap; the workers need little.
        previous = threading.stack_size(512 * 1024)
        self.addCleanup(threading.stack_size, previous)
        rendezvous = threading.Barrier(4)
        active = 0
        maximum = 0
        lock = threading.Lock()

        def transport(*args):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            rendezvous.wait(timeout=5)
            with lock:
                active -= 1
            return good_transport(*args)

        _, result = self.make_run(
            plan=lens.make_plan(corpus(), [4], [4]), transport=transport
        )
        self.assertEqual(maximum, 4)
        self.assertEqual(result["engine_active_batch_width"], "unknown")

    def test_default_single_burst_still_measures_every_corpus_case(self):
        value = corpus()
        second = copy.deepcopy(value["cases"][0])
        second.update(id="second-case", prompt_token_ids=[50, 51, 52, 53, 54])
        value["cases"].append(second)
        observed = []

        def observe(endpoint, payload, timeout, key):
            observed.append(payload["prompt_token_ids"])
            return good_transport(endpoint, payload, timeout, key)

        directory, result = self.make_run(
            plan=lens.make_plan(value, [1], [1]), transport=observe
        )
        self.assertEqual(observed, [[10, 11, 12], [50, 51, 52, 53, 54]])
        self.assertEqual(result["requests"], 2)
        self.assertEqual([cell["case_index"] for cell in result["cells"]], [0, 1])
        self.assertEqual(lens.load_run(directory)[1], result)
        for cell in result["cells"]:
            latency = cell["e2e_request_latency_seconds"]
            self.assertEqual(latency["valid_requests"], 1)
            self.assertGreater(latency["p50"], 0)
            self.assertEqual(latency["p50"], latency["p95"])
            self.assertEqual(cell["corpus_exact_match"]["status"], "all_matched")

    def test_endpoint_and_declared_engine_labels_cannot_prove_native_or_identity(self):
        directory, result = self.make_run(
            identity={
                "arm": "native-k5",
                "image": "/private/path",
                "engine": "http://secret.invalid",
            }
        )
        self.assertEqual(result["identity_status"], "declared_not_verified")
        self.assertEqual(result["native_mtp_width"], "unknown")
        self.assertEqual(result["intervention_attribution"], "unverified")
        self.assertEqual(result["cold_process_status"], "unverified")
        self.assertEqual(result["quality_qualification"], "not_established")
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(lens.main(["report", str(directory)]), 0)
        for private in [
            "/private/path",
            "secret.invalid",
            str(directory),
            "127.0.0.1",
            "native-k5",
        ]:
            self.assertNotIn(private, stdout.getvalue())

    def test_unscored_corpus_never_gets_quality_pass_headline(self):
        _, result = self.make_run(plan=lens.make_plan(corpus(expected=False), [1], [1]))
        self.assertEqual(result["corpus_exact_match"]["status"], "unscored")
        self.assertEqual(result["corpus_exact_match"]["scored_requests"], 0)
        self.assertEqual(result["quality_qualification"], "not_established")
        self.assertNotIn("quality_pass", result)

    def test_wrong_expected_ids_are_scored_not_silently_accepted(self):
        value = corpus()
        value["cases"][0]["expected_token_ids"] = [99]
        directory, result = self.make_run(plan=lens.make_plan(value, [1], [1]))
        self.assertEqual(result["corpus_exact_match"]["status"], "mismatch")
        self.assertEqual(result["cells"][0]["corpus_exact_match"]["status"], "mismatch")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(lens.main(["report", str(directory)]), 1)

    def test_failed_scored_request_is_incomplete_not_all_matched(self):
        directory, result = self.make_run(
            transport=lambda *_: lens.HttpReply(500, b"error")
        )
        self.assertEqual(result["receipt_status"], "failed")
        self.assertEqual(result["corpus_exact_match"]["status"], "incomplete")
        self.assertEqual(result["corpus_exact_match"]["invalid_scored_requests"], 1)
        self.assertIsNone(result["cells"][0]["e2e_aggregate_output_tokens_per_second"])
        self.assertEqual(lens.load_run(directory)[1], result)

    def test_errors_redirects_truncation_and_oversize_are_not_retried(self):
        for label, reply in [
            ("redirect", lens.HttpReply(307, b"redirect")),
            ("server-error", lens.HttpReply(500, b"error")),
            ("empty", lens.HttpReply(200, b"")),
            ("truncated", lens.HttpReply(200, b'{"model":')),
            ("status-bool", lens.HttpReply(True, b"{}")),
            ("wrong-body-type", lens.HttpReply(200, "not bytes")),
        ]:
            calls = mock.Mock(return_value=reply)
            _, result = self.make_run(label, transport=calls)
            self.assertEqual(calls.call_count, 1)
            self.assertEqual(result["receipt_status"], "failed")
            self.assertIsNone(
                result["cells"][0]["e2e_aggregate_output_tokens_per_second"]
            )

    def test_release_artifact_bound_is_64_mib(self):
        self.assertEqual(RELEASE_MAX_BYTES, 64 * 1024 * 1024)

    def test_oversize_body_fails_without_becoming_a_valid_partial_response(self):
        calls = mock.Mock(return_value=lens.HttpReply(200, b"x" * 4097))
        with mock.patch.object(lens, "MAX_BYTES", 4096):
            directory, result = self.make_run(transport=calls)
            self.assertEqual(lens.load_run(directory)[1], result)
        self.assertEqual(calls.call_count, 1)
        self.assertEqual(result["receipt_status"], "failed")
        self.assertEqual(result["failures"], {"TRANSPORT_ERROR": 1})

    def test_transport_exception_is_private_and_fail_closed(self):
        def broken(*_args):
            raise OSError("http://private.example /secret/path secret-key")

        directory, result = self.make_run(transport=broken)
        self.assertEqual(result["failures"], {"TRANSPORT_ERROR": 1})
        self.assertEqual(lens.load_run(directory)[1], result)
        public = lens.canonical(result).decode()
        self.assertNotIn("secret", public)
        self.assertNotIn("private.example", public)

    def test_nonfinite_timeouts_rejected_without_creating_output(self):
        for bad in [float("nan"), float("inf"), True, 0, -1]:
            target = self.root / "not-created"
            self.error(
                "TIMEOUT_TYPE_OR_RANGE",
                lens.run,
                lens.make_plan(corpus(), [1], [1]),
                "http://x/v1/completions",
                target,
                timeout=bad,
                transport=good_transport,
            )
            self.assertFalse(target.exists())

    def test_run_refuses_stale_harness_and_existing_output(self):
        plan = lens.make_plan(corpus(), [1], [1])
        plan["harness_sha256"] = "0" * 64
        self.error("HARNESS_CHANGED_SINCE_PLAN", self.make_run, plan=plan)
        directory, _ = self.make_run()
        before = (directory / "manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.make_run()
        self.assertEqual((directory / "manifest.json").read_bytes(), before)


class ReplayAndCompareTests(LensCase):
    def test_paired_comparison_requires_full_token_ids_and_bound_workload(self):
        left, _ = self.make_run("baseline")
        right, _ = self.make_run("candidate", identity={"arm": "alleged MTP"})
        result = lens.compare(left, right)
        self.assertEqual(result["comparability"], "matched")
        self.assertTrue(result["token_and_finish_parity"])
        self.assertGreater(result["cells"][0]["observed_candidate_over_baseline"], 0)
        self.assertEqual(result["native_mtp_width"], "unknown")
        self.assertEqual(result["quality_qualification"], "not_established")
        self.error("COMPARISON_SAME_RUN", lens.compare, left, left)

    def test_same_token_counts_but_different_ids_fail_compare(self):
        plan = lens.make_plan(corpus(expected=False), [1], [1])
        left, _ = self.make_run("left", plan=plan)

        def different(_endpoint, payload, *_args):
            return lens.HttpReply(200, lens.canonical(response(payload, ids=[30, 99])))

        right, _ = self.make_run("right", plan=plan, transport=different)
        self.error("COMPARISON_TOKEN_OR_FINISH_MISMATCH", lens.compare, left, right)

    def test_same_ids_with_different_finish_reasons_fail_compare(self):
        plan = lens.make_plan(corpus(expected=False), [1], [1], allow_length=True)

        def reply(finish):
            return lambda _endpoint, payload, *_args: lens.HttpReply(
                200, lens.canonical(response(payload, ids=[30, 31, 32], finish=finish))
            )

        left, _ = self.make_run("left", plan=plan, transport=reply("stop"))
        right, _ = self.make_run("right", plan=plan, transport=reply("length"))
        self.error("COMPARISON_TOKEN_OR_FINISH_MISMATCH", lens.compare, left, right)

    def test_model_revision_corpus_generation_finish_policy_matrix_pairing(self):
        left, _ = self.make_run("left")
        variants = []
        value = corpus()
        value["model"]["revision"] = "different-revision"
        variants.append(lens.make_plan(value, [1], [1]))
        value = corpus()
        value["cases"][0]["prompt_token_ids"] = [13, 14, 15]
        variants.append(lens.make_plan(value, [1], [1]))
        value = corpus()
        value["cases"][0]["max_tokens"] = 4
        variants.append(lens.make_plan(value, [1], [1]))
        variants.append(lens.make_plan(corpus(), [1], [2]))
        variants.append(lens.make_plan(corpus(), [1], [1], allow_length=True))
        for index, plan in enumerate(variants):
            right, _ = self.make_run(f"right-{index}", plan=plan)
            self.error("COMPARISON_WORKLOAD_MISMATCH", lens.compare, left, right)

    def test_missing_or_modified_artifacts_and_permissions_fail(self):
        directory, _ = self.make_run()
        raw = directory / "raw/000000.body"
        raw.write_bytes(raw.read_bytes() + b" ")
        self.error("ARTIFACT_HASH_MISMATCH", lens.load_run, directory)
        self.reseal(directory)
        raw.chmod(0o644)
        self.error("ARTIFACT_NOT_PRIVATE", lens.load_run, directory)

    def test_symlink_raw_or_root_is_refused(self):
        directory, _ = self.make_run()
        link = self.root / "link"
        link.symlink_to(directory, target_is_directory=True)
        self.error("DIRECTORY_NOT_PRIVATE", lens.load_run, link)
        raw = directory / "raw/000000.body"
        hidden = self.root / "hidden"
        raw.rename(hidden)
        raw.symlink_to(hidden)
        with self.assertRaises(OSError):
            lens.load_run(directory)

    def test_missing_request_is_not_compared_as_empty_output(self):
        left, _ = self.make_run("left")
        right, _ = self.make_run(
            "right", transport=lambda *_: lens.HttpReply(200, b"{}")
        )
        self.error("COMPARISON_INCOMPLETE_RUN", lens.compare, left, right)

    def test_resealed_summary_tampering_fails_recomputation(self):
        directory, _ = self.make_run()
        self.rewrite_json(
            directory / "result.json", lambda value: value.update(requests=999)
        )
        self.reseal(directory)
        self.error("RESULT_RECOMPUTE_MISMATCH", lens.load_run, directory)

    def test_resealed_request_tampering_fails_payload_contract(self):
        directory, _ = self.make_run()
        self.rewrite_json(
            directory / "raw/000000.request.json",
            lambda value: value.update(temperature=1),
        )
        self.reseal(directory)
        self.error("REQUEST_PAYLOAD_MISMATCH", lens.load_run, directory)

    def test_replay_hashes_the_same_bytes_it_consumes_after_concurrent_mutation(self):
        directory, _ = self.make_run(
            plan=lens.make_plan(corpus(expected=False), [1], [1])
        )
        original_read = lens.read_bytes
        mutated = False

        def mutate_after_windows(path, **kwargs):
            nonlocal mutated
            data = original_read(path, **kwargs)
            if path == directory / "windows.json" and not mutated:
                body_path = directory / "raw/000000.body"
                body = lens.parse_json(body_path.read_bytes())
                body["choices"][0]["token_ids"] = [30, 99]
                body_path.write_bytes(lens.canonical(body))
                mutated = True
            return data

        with mock.patch.object(lens, "read_bytes", side_effect=mutate_after_windows):
            self.error("ARTIFACT_HASH_MISMATCH", lens.load_run, directory)
        self.assertTrue(mutated)

    def test_resealed_case_owner_and_overlapping_cell_timeline_fail(self):
        directory, _ = self.make_run("owner")
        journal = directory / "journal.jsonl"
        value = lens.parse_json(journal.read_bytes().strip())
        value["case_index"] = 1
        journal.write_bytes(lens.canonical(value) + b"\n")
        self.reseal(directory)
        self.error("CASE_INDEX", lens.load_run, directory)
        directory, _ = self.make_run(
            "timeline", plan=lens.make_plan(corpus(), [1], [1], repeats=2)
        )
        self.rewrite_json(
            directory / "windows.json",
            lambda value: value[1].update(start_monotonic_ns=1),
        )
        self.reseal(directory)
        self.error("OVERLAPPING_CELLS", lens.load_run, directory)

    def test_resealed_malformed_timestamps_fail_even_if_result_is_unchanged(self):
        for index, bad in enumerate([True, 1.0, float("inf"), float("nan"), -1]):
            directory, _ = self.make_run(f"bad-{index}")
            path = directory / "journal.jsonl"
            entry = json.loads(path.read_text())
            entry["start_monotonic_ns"] = bad
            path.write_text(json.dumps(entry) + "\n")
            self.reseal(directory)
            with self.assertRaises(lens.LensError):
                lens.load_run(directory)

    def test_truncated_journal_and_duplicate_attempt_fail(self):
        directory, _ = self.make_run("truncated")
        journal = directory / "journal.jsonl"
        journal.write_bytes(journal.read_bytes().rstrip(b"\n"))
        self.reseal(directory)
        self.error("TRUNCATED_JOURNAL", lens.load_run, directory)
        directory, _ = self.make_run(
            "duplicate", plan=lens.make_plan(corpus(), [1], [2])
        )
        journal = directory / "journal.jsonl"
        entries = journal.read_bytes().splitlines()
        journal.write_bytes(entries[0] + b"\n" + entries[0] + b"\n")
        self.reseal(directory)
        self.error("DUPLICATE_REQUEST_INDEX", lens.load_run, directory)

    def test_unexpected_artifacts_and_unsealed_run_fail(self):
        directory, _ = self.make_run()
        lens.write_private(directory / "extra.json", b"{}")
        self.error("UNEXPECTED_ARTIFACT", lens.load_run, directory)
        target = self.root / "unsealed"
        lens.private_directory(target)
        with self.assertRaises(FileNotFoundError):
            lens.load_run(target)

    def test_public_cli_error_does_not_echo_private_path_or_endpoint(self):
        output = io.StringIO()
        private = "/private/account-name/secret-run"
        with redirect_stderr(output):
            self.assertEqual(lens.main(["report", private]), 2)
        self.assertNotIn(private, output.getvalue())
        self.assertNotIn("account-name", output.getvalue())
        self.assertEqual(json.loads(output.getvalue())["code"], "INPUT_OR_IO_ERROR")


if __name__ == "__main__":
    unittest.main()
