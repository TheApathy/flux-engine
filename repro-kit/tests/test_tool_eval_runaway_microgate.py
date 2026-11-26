# SPDX-License-Identifier: AGPL-3.0-only

import copy
import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

MODULE_PATH = Path(__file__).resolve().parents[1] / "tool_eval_runaway_microgate.py"
FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "tool_eval_runaway_micro.json"
)
PROVENANCE_EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "tool-eval-runaway-live-provenance.example.json"
)
DOCUMENTATION_PATH = (
    Path(__file__).resolve().parents[1] / "TOOL_EVAL_RUNAWAY_MICROGATE.md"
)
SPEC = importlib.util.spec_from_file_location(
    "tool_eval_runaway_microgate", MODULE_PATH
)
assert SPEC and SPEC.loader
G = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = G
SPEC.loader.exec_module(G)


def fixture():
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def captures(value):
    return G.fixture_captures(value)


def evaluate(value):
    manifest, stream, blocking, elapsed, request_pair = captures(value)
    return G.evaluate_captures(
        manifest,
        stream,
        blocking,
        pair_elapsed_ms=elapsed,
        request_pair=request_pair,
    )


def parse_sse(raw):
    return G.parse_sse(raw, expected_nonce="1" * 32)


def parse_blocking(raw):
    return G.parse_blocking(raw, expected_nonce="2" * 32)


def decode_events(raw):
    events = []
    for block in raw.strip().split("\n\n"):
        assert block.startswith("data: ")
        payload = block[6:]
        if payload != "[DONE]":
            events.append(json.loads(payload))
    return events


def encode_events(events, *, done=True):
    blocks = ["data: " + json.dumps(event, separators=(",", ":")) for event in events]
    if done:
        blocks.append("data: [DONE]")
    return "\n\n".join(blocks) + "\n\n"


def blocking(value):
    return json.loads(value["blocking_http"]["raw_body_utf8"])


def set_blocking(value, body):
    value["blocking_http"]["raw_body_utf8"] = json.dumps(body, separators=(",", ":"))


def set_stream_tool_arguments(value, arguments):
    events = decode_events(value["stream_http"]["raw_sse_utf8"])
    events[3]["choices"][0]["delta"]["tool_calls"][0]["function"][
        "arguments"
    ] = arguments
    events[4]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] = ""
    events[5]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] = ""
    value["stream_http"]["raw_sse_utf8"] = encode_events(events)


def live_provenance(local_identity=None, *, execution_mode="host-binary"):
    local_identity = local_identity or {
        "git_tree": "a" * 40,
        "git_blob": "b" * 40,
        "sha256": "c" * 64,
    }
    return {
        "schema": G.LIVE_PROVENANCE_SCHEMA,
        "runner": copy.deepcopy(local_identity),
        "source": {"git_tree": local_identity["git_tree"]},
        "execution": {
            "mode": execution_mode,
            "binary_sha256": "d" * 64,
            "image_digest": (
                "sha256:" + "e" * 64 if execution_mode == "container-image" else None
            ),
        },
        "checkpoint": {
            "id": "JMNI-Labs/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid",
            "revision": "f" * 40,
            "served_model": "qad5500",
        },
    }


def scalar_leaf_paths(value, path=()):
    if type(value) is dict:
        for key, child in value.items():
            yield from scalar_leaf_paths(child, path + (key,))
    elif type(value) is list:
        for index, child in enumerate(value):
            yield from scalar_leaf_paths(child, path + (index,))
    else:
        yield path


def replace_path(value, path, replacement):
    target = value
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = replacement


class ManifestTests(unittest.TestCase):
    def test_fixture_manifest_pins_fail_closed_profile(self):
        validated = G.validate_manifest(fixture()["manifest"])
        self.assertEqual(validated["launch_contract"]["profile"], "interactive-w4096")
        self.assertEqual(validated["launch_contract"]["concurrency"], 1)
        self.assertEqual(validated["launch_contract"]["prefill_chunk_tokens"], 4096)
        self.assertEqual(validated["launch_contract"]["kv_cache_dtype"], "bf16")
        self.assertEqual(validated["launch_contract"]["speculation"]["mode"], "none")
        self.assertEqual(
            validated["launch_contract"]["tool_call_parser"], "qwen3_coder"
        )
        self.assertFalse(validated["request"]["parallel_tool_calls"])
        self.assertTrue(validated["request"]["return_token_ids"])
        self.assertTrue(validated["request"]["report_interventions"])
        self.assertFalse(validated["prompt_audit"]["observed"])
        self.assertEqual(
            validated["prompt_audit"]["source"],
            "server-terminal-response-digest",
        )

    def test_profile_parser_template_and_speculation_drift_fail(self):
        cases = (
            (("profile",), "max-prefill"),
            (("concurrency",), 2),
            (("max_batch_size",), 2),
            (("max_num_seqs",), 2),
            (("prefill_chunk_tokens",), 16384),
            (("kv_cache_dtype",), "fp8"),
            (("speculation", "mode"), "native-mtp"),
            (("speculation", "num_drafts"), 4),
            (("tool_call_parser",), "hermes"),
            (("chat_template", "source"), "server-override"),
            (("chat_template", "tool_format"), "hermes-json"),
            (("chat_template", "native_tool_render_count"), 2),
            (("chat_template", "hermes_prompt_injection_count"), 1),
        )
        for path, replacement in cases:
            with self.subTest(path=path):
                value = fixture()["manifest"]
                target = value["launch_contract"]
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = replacement
                with self.assertRaises(G.GateError):
                    G.validate_manifest(value)

    def test_request_must_be_greedy_bounded_single_tool_mode(self):
        cases = {
            "max_tokens": 65,
            "temperature": 0.1,
            "top_k": 2,
            "top_p": 0.9,
            "parallel_tool_calls": True,
            "return_token_ids": False,
            "report_interventions": False,
            "n": 2,
        }
        for key, replacement in cases.items():
            with self.subTest(key=key):
                value = fixture()["manifest"]
                value["request"][key] = replacement
                with self.assertRaises(G.GateError):
                    G.validate_manifest(value)
        value = fixture()["manifest"]
        del value["request"]["report_interventions"]
        with self.assertRaisesRegex(G.GateError, "report_interventions"):
            G.validate_manifest(value)
        value = fixture()["manifest"]
        value["timeout_seconds"] = 45.001
        with self.assertRaisesRegex(G.GateError, "timeout_seconds"):
            G.validate_manifest(value)
        value = fixture()["manifest"]
        value["token_cap"] = G.HARD_TOKEN_CAP + 1
        value["request"]["max_tokens"] = G.HARD_TOKEN_CAP + 1
        with self.assertRaisesRegex(G.GateError, "hard microgate cap"):
            G.validate_manifest(value)

    def test_runner_owns_stream_fields(self):
        for key, replacement in (
            ("stream", False),
            ("stream_options", {}),
            ("rendered_prompt_evidence_nonce", "1" * 32),
        ):
            with self.subTest(key=key):
                value = fixture()["manifest"]
                value["request"][key] = replacement
                with self.assertRaisesRegex(G.GateError, "runner owns"):
                    G.validate_manifest(value)

    def test_manifest_pins_checkpoint_thinking_off_explicitly(self):
        value = fixture()["manifest"]
        del value["request"]["chat_template_kwargs"]
        with self.assertRaisesRegex(G.GateError, "chat_template_kwargs"):
            G.validate_manifest(value)

        value = fixture()["manifest"]
        value["request"]["chat_template_kwargs"]["enable_thinking"] = True
        with self.assertRaisesRegex(G.GateError, "enable_thinking"):
            G.validate_manifest(value)

        value = fixture()["manifest"]
        value["request"]["chat_template_kwargs"]["unexpected"] = False
        with self.assertRaisesRegex(G.GateError, "keys differ"):
            G.validate_manifest(value)

    def test_manifest_pins_the_exact_city_required_tool_schema(self):
        for path, replacement in (
            (("required",), []),
            (("additionalProperties",), True),
            (("properties", "city", "type"), "object"),
        ):
            with self.subTest(path=path):
                value = fixture()["manifest"]
                schema = value["request"]["tools"][0]["function"]["parameters"]
                target = schema
                for component in path[:-1]:
                    target = target[component]
                target[path[-1]] = replacement
                with self.assertRaisesRegex(G.GateError, "sealed city-required"):
                    G.validate_manifest(value)

        value = fixture()["manifest"]
        value["expected_tool_names"] = ["get_weather", "delete_all"]
        with self.assertRaisesRegex(G.GateError, "sealed tool-name"):
            G.validate_manifest(value)

        value = fixture()["manifest"]
        value["request"]["messages"][0]["content"] = "Use get_weather for Rome."
        with self.assertRaisesRegex(G.GateError, "sealed Paris test prompt"):
            G.validate_manifest(value)

    def test_rendered_prompt_audit_accepts_native_once_and_rejects_duplicates(self):
        value = fixture()["manifest"]
        request = value["request"]
        native = G.expected_rendered_prompt(request)
        self.assertTrue(native.endswith("<think>\n\n</think>\n\n"))
        audit = G.validate_rendered_prompt(native, request)
        self.assertTrue(audit["observed"])
        self.assertEqual(audit["native_qwen_header_count"], 1)

        with self.assertRaisesRegex(G.GateError, "exactly once"):
            G.validate_rendered_prompt(native + "\n" + native, request)
        with self.assertRaisesRegex(G.GateError, "Hermes"):
            G.validate_rendered_prompt(native + "\n" + G.HERMES_MARKERS[0], request)

    def test_operator_prompt_claim_is_forbidden_and_expected_prompt_is_bound(self):
        value = fixture()["manifest"]
        value["launch_contract"]["rendered_prompt_utf8"] = "operator claim"
        with self.assertRaisesRegex(G.GateError, "keys differ"):
            G.validate_manifest(value)

        value = fixture()["manifest"]
        value["request"]["messages"][0]["content"] = "Use get_weather for Rome."
        prompt = G.expected_rendered_prompt(fixture()["manifest"]["request"])
        with self.assertRaisesRegex(G.GateError, "exact native Qwen"):
            G.validate_rendered_prompt(prompt, value["request"])

        value = fixture()["manifest"]
        value["request"]["tools"][0]["function"]["parameters"]["required"] = []
        with self.assertRaisesRegex(G.GateError, "requested tool schema"):
            G.validate_rendered_prompt(prompt, value["request"])

        value = fixture()["manifest"]
        assistant = "<|im_start|>assistant\n"
        injected = assistant + prompt.replace(assistant, "")
        with self.assertRaisesRegex(G.GateError, "exactly one|exact native Qwen"):
            G.validate_rendered_prompt(injected, value["request"])

    def test_manifest_file_rejects_duplicate_keys_and_nonfinite_numbers(self):
        for raw, expected in (
            ('{"schema":"first","schema":"second"}', "duplicate key 'schema'"),
            ('{"timeout_seconds":NaN}', "non-finite number 'NaN'"),
            ('{"timeout_seconds":1e9999}', "non-finite number"),
            ('{"timeout_seconds":-1e9999}', "non-finite number"),
        ):
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "manifest.json"
                path.write_text(raw, encoding="utf-8")
                with self.assertRaisesRegex(G.GateError, expected):
                    G._json_file(path, "manifest")


class ProvenanceTests(unittest.TestCase):
    def test_documented_live_provenance_schema_is_exact_and_non_authoritative(self):
        example = json.loads(PROVENANCE_EXAMPLE_PATH.read_text(encoding="utf-8"))
        local = example["runner"]
        with mock.patch.object(G, "_local_runner_identity", return_value=local):
            normalized = G._validate_live_provenance(example)
        self.assertEqual(normalized, example)

        documentation = DOCUMENTATION_PATH.read_text(encoding="utf-8")
        self.assertIn("--live-provenance /tmp/tool-live-provenance.json", documentation)
        self.assertIn("Do not replace its prompt, tool", documentation)
        self.assertIn("authoritative=false", documentation)
        self.assertIn("does not replace a fresh-process full", documentation)

    def test_git_identity_disables_replace_refs_and_inherited_git_environment(self):
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="a" * 40 + "\n", stderr=""
        )
        with mock.patch.dict(
            os.environ,
            {
                "PATH": "/private/attacker/bin",
                "LD_PRELOAD": "/private/attacker/library.so",
                "GIT_DIR": "/private/redirect",
                "GIT_WORK_TREE": "/private/tree",
                "GIT_REPLACE_REF_BASE": "refs/private",
            },
        ), mock.patch.object(G.subprocess, "run", return_value=completed) as run:
            self.assertEqual(G._git_output(["rev-parse", "HEAD^{tree}"]), "a" * 40)
        command = run.call_args.args[0]
        environment = run.call_args.kwargs["env"]
        self.assertEqual(command[:2], [G.GIT_EXECUTABLE, "--no-replace-objects"])
        self.assertEqual(
            environment,
            {
                "PATH": os.defpath,
                "LC_ALL": "C",
                "LANG": "C",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_NO_LAZY_FETCH": "1",
                "GIT_TERMINAL_PROMPT": "0",
            },
        )
        self.assertNotIn("LD_PRELOAD", environment)

    def test_peer_controlled_http_exception_text_is_not_reflected(self):
        for stage, failure in (
            ("response", G.http.client.BadStatusLine("PRIVATE_STATUS_LINE_MARKER")),
            ("request", ValueError("PRIVATE_HEADER_VALUE_MARKER")),
        ):
            with self.subTest(failure=type(failure).__name__):
                connection = mock.Mock()
                connection.sock = None
                if stage == "request":
                    connection.request.side_effect = failure
                else:
                    connection.getresponse.side_effect = failure
                with mock.patch.object(
                    G,
                    "_connection_for",
                    return_value=(connection, "/v1/chat/completions"),
                ), self.assertRaisesRegex(
                    G.GateError, r"^HTTP request failed$"
                ) as raised:
                    G.http_post(
                        "http://127.0.0.1:8899/v1/chat/completions",
                        {},
                        None,
                        G.time.monotonic() + 1.0,
                        stream=False,
                    )
                self.assertNotIn("PRIVATE_", str(raised.exception))
                connection.close.assert_called_once_with()

    def test_offline_binding_is_explicitly_non_authoritative(self):
        binding = G._capture_binding("offline-fixture")
        self.assertEqual(binding["mode"], "offline-fixture")
        self.assertFalse(binding["authoritative"])
        self.assertIsNone(binding["endpoint"])
        self.assertIsNone(binding["identity"])
        with self.assertRaisesRegex(G.GateError, "cannot claim live"):
            G._capture_binding(
                "offline-fixture",
                endpoint="http://127.0.0.1/v1/chat/completions",
            )

    def test_live_binding_requires_all_immutable_identities(self):
        local = {
            "git_tree": "a" * 40,
            "git_blob": "b" * 40,
            "sha256": "c" * 64,
        }
        provenance = live_provenance(local)
        with mock.patch.object(G, "_local_runner_identity", return_value=local):
            binding = G._capture_binding(
                "live-http",
                endpoint="HTTP://LOCALHOST/v1/chat/completions",
                live_provenance=provenance,
            )
        self.assertFalse(binding["authoritative"])
        self.assertEqual(
            binding["authority_scope"], "live-parser-only-unverified-runtime"
        )
        self.assertEqual(binding["endpoint"], "http://localhost:80/v1/chat/completions")
        self.assertIsNone(binding["identity"])
        self.assertIsNone(binding["identity_sha256"])
        self.assertEqual(binding["declared_identity"], provenance)
        self.assertRegex(binding["declared_identity_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(
            binding["runtime_binding"],
            {
                "listener_executable_observed": False,
                "container_image_observed": False,
                "checkpoint_inventory_observed": False,
            },
        )
        G._bind_capture_to_manifest(binding, G.validate_manifest(fixture()["manifest"]))
        wrong_model = copy.deepcopy(binding)
        wrong_model["declared_identity"]["checkpoint"]["served_model"] = "other-model"
        with self.assertRaisesRegex(G.GateError, "served_model"):
            G._bind_capture_to_manifest(
                wrong_model, G.validate_manifest(fixture()["manifest"])
            )

        for section, key in (
            ("runner", "git_tree"),
            ("runner", "git_blob"),
            ("runner", "sha256"),
            ("source", "git_tree"),
            ("execution", "binary_sha256"),
            ("checkpoint", "id"),
            ("checkpoint", "revision"),
            ("checkpoint", "served_model"),
        ):
            with self.subTest(section=section, key=key):
                tampered = copy.deepcopy(provenance)
                del tampered[section][key]
                with mock.patch.object(
                    G, "_local_runner_identity", return_value=local
                ), self.assertRaises(G.GateError):
                    G._capture_binding(
                        "live-http",
                        endpoint="http://localhost/v1/chat/completions",
                        live_provenance=tampered,
                    )

        container = live_provenance(local, execution_mode="container-image")
        container["execution"]["image_digest"] = None
        with mock.patch.object(
            G, "_local_runner_identity", return_value=local
        ), self.assertRaises(G.GateError):
            G._capture_binding(
                "live-http",
                endpoint="http://localhost/v1/chat/completions",
                live_provenance=container,
            )

        mismatch = live_provenance(local)
        mismatch["runner"]["git_blob"] = "0" * 40
        with mock.patch.object(
            G, "_local_runner_identity", return_value=local
        ), self.assertRaisesRegex(G.GateError, "committed runner"):
            G._capture_binding(
                "live-http",
                endpoint="http://localhost/v1/chat/completions",
                live_provenance=mismatch,
            )


class ParserAndGateTests(unittest.TestCase):
    def test_good_fixture_passes_and_preserves_space_bytes(self):
        receipt = evaluate(fixture())
        self.assertEqual(receipt["result"], "PASS")
        self.assertEqual(receipt["errors"], [])
        self.assertEqual(receipt["capture"]["mode"], "offline-fixture")
        self.assertFalse(receipt["capture"]["authoritative"])
        self.assertIsNone(receipt["capture"]["identity"])
        self.assertEqual(receipt["stream_capture"]["content_type"], "text/event-stream")
        self.assertEqual(
            receipt["blocking_capture"]["content_type"], "application/json"
        )
        self.assertEqual(receipt["stream"]["content"]["utf8"], "I will check.")
        self.assertEqual(
            receipt["stream"]["content"]["utf8_hex"],
            "492077696c6c20636865636b2e",
        )
        self.assertEqual(
            receipt["stream"]["tool_fragments"][0]["arguments"]["utf8"], ""
        )
        self.assertEqual(
            receipt["stream"]["terminal_token_ids"],
            [101, 102, 103, 104, 105, 106, 107, 108],
        )
        expected_ledger = {
            "schema": G.INTERVENTION_LEDGER_SCHEMA,
            "policy": G.INTERVENTION_LEDGER_POLICY,
            "events": [],
        }
        self.assertEqual(receipt["stream"]["interventions"], expected_ledger)
        self.assertEqual(receipt["blocking"]["interventions"], expected_ledger)

    def test_request_pair_must_match_the_exact_nonced_manifest(self):
        manifest, stream, blocking, elapsed, pair = captures(fixture())
        tampered_stream = copy.deepcopy(pair.stream)
        tampered_stream["messages"][0]["content"] += " injected"
        tampered = G.RequestPair(
            tampered_stream,
            pair.blocking,
            pair.stream_nonce,
            pair.blocking_nonce,
        )
        with self.assertRaisesRegex(G.GateError, "exact nonced manifest request"):
            G.evaluate_captures(
                manifest,
                stream,
                blocking,
                pair_elapsed_ms=elapsed,
                request_pair=tampered,
            )

    def test_intervention_ledgers_are_terminal_exact_and_empty(self):
        value = fixture()
        events = decode_events(value["stream_http"]["raw_sse_utf8"])
        del events[6]["interventions"]
        with self.assertRaisesRegex(G.GateError, "missing the terminal"):
            parse_sse(encode_events(events).encode())

        value = fixture()
        events = decode_events(value["stream_http"]["raw_sse_utf8"])
        ledger = events[6].pop("interventions")
        events[2]["interventions"] = ledger
        with self.assertRaisesRegex(G.GateError, "finish-bearing"):
            parse_sse(encode_events(events).encode())

        value = fixture()
        events = decode_events(value["stream_http"]["raw_sse_utf8"])
        events.insert(
            7,
            {
                "id": "chatcmpl-123e4567-e89b-42d3-a456-426614174000",
                "system_fingerprint": G.ATLAS_SYSTEM_FINGERPRINT,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "too late"},
                        "finish_reason": None,
                    }
                ],
            },
        )
        with self.assertRaisesRegex(G.GateError, "choice after the terminal"):
            parse_sse(encode_events(events).encode())

        for key, replacement, error in (
            ("schema", "unknown", "schema"),
            ("policy", "disabled", "policy"),
            (
                "events",
                [{"code": "F32", "action": "duplicate", "count": 1}],
                "events must be empty",
            ),
        ):
            with self.subTest(key=key):
                value = fixture()
                events = decode_events(value["stream_http"]["raw_sse_utf8"])
                events[6]["interventions"][key] = replacement
                with self.assertRaisesRegex(G.GateError, error):
                    parse_sse(encode_events(events).encode())

        value = fixture()
        body = blocking(value)
        del body["interventions"]
        with self.assertRaisesRegex(G.GateError, "must be an object"):
            parse_blocking(json.dumps(body).encode())

        repair_event = {
            "code": "tool_output_repair",
            "action": "backfill_required_params",
            "count": 1,
        }
        value = fixture()
        events = decode_events(value["stream_http"]["raw_sse_utf8"])
        events[6]["interventions"]["events"] = [repair_event]
        with self.assertRaisesRegex(G.GateError, "events must be empty"):
            parse_sse(encode_events(events).encode())

        value = fixture()
        body = blocking(value)
        body["interventions"]["events"] = [repair_event]
        with self.assertRaisesRegex(G.GateError, "events must be empty"):
            parse_blocking(json.dumps(body).encode())

    def test_four_blank_lines_fail_even_when_stream_and_blocking_match(self):
        value = fixture()
        events = decode_events(value["stream_http"]["raw_sse_utf8"])
        events[2]["choices"][0]["delta"]["content"] += "\n\n\n\n\n"
        value["stream_http"]["raw_sse_utf8"] = encode_events(events)
        body = blocking(value)
        body["choices"][0]["message"]["content"] += "\n\n\n\n\n"
        set_blocking(value, body)
        receipt = evaluate(value)
        self.assertEqual(receipt["result"], "FAIL")
        self.assertTrue(
            any("consecutive blank lines" in error for error in receipt["errors"])
        )

    def test_unexpected_tool_fails_both_arms(self):
        value = fixture()
        events = decode_events(value["stream_http"]["raw_sse_utf8"])
        events[3]["choices"][0]["delta"]["tool_calls"][0]["function"][
            "name"
        ] = "delete_all"
        value["stream_http"]["raw_sse_utf8"] = encode_events(events)
        body = blocking(value)
        body["choices"][0]["message"]["tool_calls"][0]["function"][
            "name"
        ] = "delete_all"
        set_blocking(value, body)
        with self.assertRaises(G.RawPromptLeakError):
            evaluate(value)

    def test_second_tool_fails_when_parallel_false(self):
        value = fixture()
        events = decode_events(value["stream_http"]["raw_sse_utf8"])
        second = copy.deepcopy(events[3])
        fragment = second["choices"][0]["delta"]["tool_calls"][0]
        fragment["index"] = 1
        fragment["id"] = "call_0000000000000003"
        fragment["function"]["arguments"] = '{"city":"Rome"}'
        events.insert(6, second)
        value["stream_http"]["raw_sse_utf8"] = encode_events(events)
        body = blocking(value)
        second_blocking = copy.deepcopy(body["choices"][0]["message"]["tool_calls"][0])
        second_blocking["id"] = "call_0000000000000004"
        second_blocking["function"]["arguments"] = '{"city":"Rome"}'
        body["choices"][0]["message"]["tool_calls"].append(second_blocking)
        set_blocking(value, body)
        with self.assertRaises(G.RawPromptLeakError):
            evaluate(value)

    def test_length_finish_reason_fails(self):
        value = fixture()
        events = decode_events(value["stream_http"]["raw_sse_utf8"])
        events[6]["choices"][0]["finish_reason"] = "length"
        value["stream_http"]["raw_sse_utf8"] = encode_events(events)
        body = blocking(value)
        body["choices"][0]["finish_reason"] = "length"
        set_blocking(value, body)
        receipt = evaluate(value)
        self.assertTrue(
            any("finish_reason=length" in error for error in receipt["errors"])
        )

    def test_missing_done_is_malformed(self):
        value = fixture()
        events = decode_events(value["stream_http"]["raw_sse_utf8"])
        value["stream_http"]["raw_sse_utf8"] = encode_events(events, done=False)
        with self.assertRaises(G.RawPromptLeakError):
            evaluate(value)

    def test_malformed_sse_fails_closed(self):
        value = fixture()
        value["stream_http"]["raw_sse_utf8"] = "data: {not-json}\n\ndata: [DONE]\n\n"
        with self.assertRaises(G.RawPromptLeakError):
            evaluate(value)

    def test_sse_rejects_every_non_data_channel(self):
        valid = fixture()["stream_http"]["raw_sse_utf8"].encode()
        for channel in (
            b": SECRET_PROMPT\n",
            b"event: SECRET_PROMPT\n",
            b"id: SECRET_PROMPT\n",
            b"retry: 1000\n",
        ):
            with self.subTest(channel=channel):
                with self.assertRaises(G.RawPromptLeakError):
                    parse_sse(channel + valid)

    def test_wire_json_rejects_duplicate_keys_and_nonfinite_numbers(self):
        for raw, expected in (
            (b'{"choices":[],"choices":[]}', "duplicate key 'choices'"),
            (b'{"choices":[],"usage":NaN}', "non-finite number 'NaN'"),
            (b'{"choices":[],"usage":{"x":1e9999}}', "non-finite number"),
            (b'{"choices":[],"usage":{"x":-1e9999}}', "non-finite number"),
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(G.RawPromptLeakError) as caught:
                    parse_blocking(raw)
                inner = caught.exception.__cause__ or caught.exception.__context__
                self.assertIsNotNone(inner)
                self.assertIn(expected, str(inner))

        for raw, expected in (
            (
                b'data: {"choices":[],"choices":[]}\n\ndata: [DONE]\n\n',
                "duplicate key 'choices'",
            ),
            (
                b'data: {"choices":[],"usage":Infinity}\n\ndata: [DONE]\n\n',
                "non-finite number 'Infinity'",
            ),
            (
                b'data: {"choices":[],"usage":{"x":1e9999}}\n\ndata: [DONE]\n\n',
                "non-finite number",
            ),
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(G.RawPromptLeakError) as caught:
                    parse_sse(raw)
                inner = caught.exception.__cause__ or caught.exception.__context__
                self.assertIsNotNone(inner)
                self.assertIn(expected, str(inner))

    def test_tool_argument_json_rejects_duplicate_keys(self):
        value = fixture()
        events = decode_events(value["stream_http"]["raw_sse_utf8"])
        events[4]["choices"][0]["delta"]["tool_calls"][0]["function"][
            "arguments"
        ] = '{"city":"Paris","city":"Rome"}'
        events[5]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] = ""
        with self.assertRaises(G.RawPromptLeakError):
            parse_sse(encode_events(events).encode())

        body = blocking(value)
        body["choices"][0]["message"]["tool_calls"][0]["function"][
            "arguments"
        ] = '{"city":"Paris","city":"Rome"}'
        with self.assertRaises(G.RawPromptLeakError):
            parse_blocking(json.dumps(body).encode())

    def test_tool_argument_json_rejects_exponent_overflow(self):
        for raw_number in ("1e9999", "-1e9999"):
            with self.subTest(arm="stream", raw_number=raw_number):
                value = fixture()
                events = decode_events(value["stream_http"]["raw_sse_utf8"])
                events[4]["choices"][0]["delta"]["tool_calls"][0]["function"][
                    "arguments"
                ] = ('{"city":' + raw_number + "}")
                events[5]["choices"][0]["delta"]["tool_calls"][0]["function"][
                    "arguments"
                ] = ""
                with self.assertRaises(G.RawPromptLeakError):
                    parse_sse(encode_events(events).encode())

            with self.subTest(arm="blocking", raw_number=raw_number):
                value = fixture()
                body = blocking(value)
                body["choices"][0]["message"]["tool_calls"][0]["function"][
                    "arguments"
                ] = ('{"city":' + raw_number + "}")
                with self.assertRaises(G.RawPromptLeakError):
                    parse_blocking(json.dumps(body).encode())

    def test_decoded_tool_arguments_obey_sealed_schema_recursively(self):
        hostile_arguments = (
            "{}",
            '{"city":"Paris","extra":"SECRET_PROMPT"}',
            '{"city":{"value":"Paris"}}',
            '{"city":"Paris","nested":{"raw_prompt":"SECRET_PROMPT"}}',
        )
        for arm in ("stream", "blocking"):
            for arguments in hostile_arguments:
                with self.subTest(arm=arm, arguments=arguments):
                    value = fixture()
                    if arm == "stream":
                        set_stream_tool_arguments(value, arguments)
                    else:
                        body = blocking(value)
                        body["choices"][0]["message"]["tool_calls"][0]["function"][
                            "arguments"
                        ] = arguments
                        set_blocking(value, body)
                    with self.assertRaises(G.RawPromptLeakError):
                        evaluate(value)

    def test_content_types_are_exact_and_normalized(self):
        value = fixture()
        value["stream_http"]["content_type"] = "TEXT/EVENT-STREAM"
        value["blocking_http"]["content_type"] = "APPLICATION/JSON"
        receipt = evaluate(value)
        self.assertEqual(receipt["stream_capture"]["content_type"], "text/event-stream")
        self.assertEqual(
            receipt["blocking_capture"]["content_type"], "application/json"
        )

        for arm, content_type in (
            ("stream_http", "text/event-stream; secret=SECRET_PROMPT"),
            ("stream_http", "text/event-stream; charset=utf-8"),
            ("blocking_http", "application/json; charset=utf-8"),
            ("blocking_http", "application/json,SECRET_PROMPT"),
        ):
            with self.subTest(arm=arm, content_type=content_type):
                value = fixture()
                value[arm]["content_type"] = content_type
                with self.assertRaises(G.RawPromptLeakError):
                    evaluate(value)

    def test_safe_metadata_is_exact_or_absent(self):
        stream_mutations = (
            lambda event: event.__setitem__("object", "SECRET_PROMPT"),
            lambda event: event.__setitem__("system_fingerprint", "SECRET_PROMPT"),
            lambda event: event.__setitem__("created", -1),
            lambda event: event.__setitem__("id", "SECRET PROMPT"),
            lambda event: event.__setitem__("model", "SECRET_PROMPT"),
            lambda event: event["choices"][0]["delta"].__setitem__(
                "refusal", "SECRET_PROMPT"
            ),
        )
        for mutation in stream_mutations:
            with self.subTest(arm="stream", mutation=repr(mutation)):
                value = fixture()
                events = decode_events(value["stream_http"]["raw_sse_utf8"])
                mutation(events[0])
                value["stream_http"]["raw_sse_utf8"] = encode_events(events)
                with self.assertRaises(G.RawPromptLeakError):
                    evaluate(value)

        blocking_mutations = (
            lambda body: body.__setitem__("object", "SECRET_PROMPT"),
            lambda body: body.__setitem__("system_fingerprint", "SECRET_PROMPT"),
            lambda body: body.__setitem__("service_tier", "SECRET_PROMPT"),
            lambda body: body.__setitem__("created", -1),
            lambda body: body.__setitem__("id", "SECRET PROMPT"),
            lambda body: body.__setitem__("model", "SECRET_PROMPT"),
            lambda body: body["choices"][0]["message"].__setitem__(
                "refusal", "SECRET_PROMPT"
            ),
        )
        for mutation in blocking_mutations:
            with self.subTest(arm="blocking", mutation=repr(mutation)):
                value = fixture()
                body = blocking(value)
                mutation(body)
                set_blocking(value, body)
                with self.assertRaises(G.RawPromptLeakError):
                    evaluate(value)

    def test_server_prompt_evidence_is_nonce_response_and_hash_bound(self):
        for mutation, expected in (
            (("nonce", "f" * 32), "not bound to this live request"),
            (("response_id", "chatcmpl-other"), "not bound to the response"),
            (("prompt_sha256", "x" * 64), "64 lowercase hex"),
        ):
            with self.subTest(field=mutation[0]):
                value = fixture()
                events = decode_events(value["stream_http"]["raw_sse_utf8"])
                events[6]["rendered_prompt_evidence"][mutation[0]] = mutation[1]
                with self.assertRaisesRegex(G.GateError, expected):
                    parse_sse(encode_events(events).encode())

        value = fixture()
        body = blocking(value)
        body["rendered_prompt_evidence"]["nonce"] = "f" * 32
        with self.assertRaisesRegex(G.GateError, "not bound to this live request"):
            parse_blocking(json.dumps(body).encode())

    def test_raw_prompt_fields_are_rejected_anywhere_in_either_response(self):
        value = fixture()
        events = decode_events(value["stream_http"]["raw_sse_utf8"])
        events[0]["choices"][0]["delta"]["raw_prompt_bytes"] = [65, 66]
        with self.assertRaisesRegex(G.GateError, "forbidden raw-prompt field"):
            parse_sse(encode_events(events).encode())

        value = fixture()
        events = decode_events(value["stream_http"]["raw_sse_utf8"])
        events[6]["rendered_prompt_utf8"] = "secret prompt"
        value["stream_http"]["raw_sse_utf8"] = encode_events(events)
        with self.assertRaisesRegex(G.GateError, "forbidden raw-prompt field"):
            parse_sse(encode_events(events).encode())
        with self.assertRaises(G.RawPromptLeakError):
            evaluate(value)

        value = fixture()
        body = blocking(value)
        body["rendered_prompt_utf8"] = "secret prompt"
        with self.assertRaisesRegex(G.GateError, "forbidden raw-prompt field"):
            parse_blocking(json.dumps(body).encode())

        value = fixture()
        body = blocking(value)
        body["choices"][0]["message"]["prompt_text"] = "secret prompt"
        with self.assertRaisesRegex(G.GateError, "forbidden raw-prompt field"):
            parse_blocking(json.dumps(body).encode())

        value = fixture()
        body = blocking(value)
        body["rendered_prompt_utf8"] = "secret prompt"
        set_blocking(value, body)
        with tempfile.TemporaryDirectory() as directory:
            fixture_path = Path(directory) / "leaky-fixture.json"
            fixture_path.write_text(json.dumps(value), encoding="utf-8")
            output = Path(directory) / "must-not-exist"
            self.assertEqual(
                G.main(["--fixture", str(fixture_path), "--out-dir", str(output)]),
                2,
            )
            self.assertFalse(output.exists())

    def test_neutral_nested_response_fields_abort_before_raw_artifacts(self):
        value = fixture()
        events = decode_events(value["stream_http"]["raw_sse_utf8"])
        events[6]["rendered_prompt_evidence"]["utf8"] = "SECRET_RENDERED_PROMPT_BYTES"
        value["stream_http"]["raw_sse_utf8"] = encode_events(events)
        with self.assertRaisesRegex(
            G.RawPromptLeakError, "unapproved pre-artifact response fields"
        ):
            evaluate(value)

        value = fixture()
        body = blocking(value)
        body["choices"][0]["message"]["telemetry"] = {
            "utf8": "SECRET_RENDERED_PROMPT_BYTES"
        }
        set_blocking(value, body)
        with self.assertRaisesRegex(
            G.RawPromptLeakError, "unapproved pre-artifact response fields"
        ):
            evaluate(value)

        value = fixture()
        body = blocking(value)
        body["rendered_prompt_evidence"]["bytes"] = [83, 69, 67, 82, 69, 84]
        set_blocking(value, body)
        with tempfile.TemporaryDirectory() as directory:
            fixture_path = Path(directory) / "neutral-leak-fixture.json"
            fixture_path.write_text(json.dumps(value), encoding="utf-8")
            output = Path(directory) / "must-not-exist"
            self.assertEqual(
                G.main(["--fixture", str(fixture_path), "--out-dir", str(output)]),
                2,
            )
            self.assertFalse(output.exists())

    def test_every_response_schema_leaf_rejects_nested_containers(self):
        stream_events = decode_events(fixture()["stream_http"]["raw_sse_utf8"])
        stream_events[0]["created"] = 0
        stream_events[0]["system_fingerprint"] = G.ATLAS_SYSTEM_FINGERPRINT
        stream_events[0]["choices"][0]["logprobs"] = None
        stream_events[0]["choices"][0]["delta"]["refusal"] = None
        for path in tuple(scalar_leaf_paths(stream_events)):
            for replacement in ({"secret": "payload"}, ["secret payload"]):
                with self.subTest(arm="stream", path=path, replacement=replacement):
                    mutated = copy.deepcopy(stream_events)
                    replace_path(mutated, path, replacement)
                    with self.assertRaises(G.RawPromptLeakError):
                        parse_sse(encode_events(mutated).encode())

        blocking_body = blocking(fixture())
        blocking_body["created"] = 0
        blocking_body["system_fingerprint"] = G.ATLAS_SYSTEM_FINGERPRINT
        blocking_body["service_tier"] = "default"
        blocking_body["metadata"] = None
        blocking_body["choices"][0]["logprobs"] = None
        blocking_body["choices"][0]["message"]["annotations"] = None
        blocking_body["choices"][0]["message"]["refusal"] = None
        for path in tuple(scalar_leaf_paths(blocking_body)):
            for replacement in ({"secret": "payload"}, ["secret payload"]):
                with self.subTest(arm="blocking", path=path, replacement=replacement):
                    mutated = copy.deepcopy(blocking_body)
                    replace_path(mutated, path, replacement)
                    with self.assertRaises(G.RawPromptLeakError):
                        parse_blocking(json.dumps(mutated).encode())

    def test_all_reported_leaf_bypasses_abort_before_output_directory(self):
        hostile_fixtures = []

        value = fixture()
        events = decode_events(value["stream_http"]["raw_sse_utf8"])
        events[0]["created"] = {"secret": "payload"}
        value["stream_http"]["raw_sse_utf8"] = encode_events(events)
        hostile_fixtures.append(("stream-created", value))

        value = fixture()
        events = decode_events(value["stream_http"]["raw_sse_utf8"])
        events[6]["choices"][0]["token_ids"] = [{"secret": "payload"}]
        value["stream_http"]["raw_sse_utf8"] = encode_events(events)
        hostile_fixtures.append(("stream-token-ids", value))

        value = fixture()
        events = decode_events(value["stream_http"]["raw_sse_utf8"])
        events[1]["choices"][0]["delta"]["content"] = {"secret": "payload"}
        value["stream_http"]["raw_sse_utf8"] = encode_events(events)
        hostile_fixtures.append(("stream-content", value))

        value = fixture()
        events = decode_events(value["stream_http"]["raw_sse_utf8"])
        events[7]["usage"]["prompt_tokens"] = {"secret": "payload"}
        value["stream_http"]["raw_sse_utf8"] = encode_events(events)
        hostile_fixtures.append(("stream-prompt-tokens", value))

        value = fixture()
        body = blocking(value)
        body["choices"][0]["message"]["content"] = {"secret": "payload"}
        set_blocking(value, body)
        hostile_fixtures.append(("blocking-content", value))

        value = fixture()
        body = blocking(value)
        body["choices"][0]["token_ids"] = [{"secret": "payload"}]
        set_blocking(value, body)
        hostile_fixtures.append(("blocking-token-ids", value))

        value = fixture()
        body = blocking(value)
        body["service_tier"] = {"secret": "payload"}
        set_blocking(value, body)
        hostile_fixtures.append(("blocking-service-tier", value))

        value = fixture()
        body = blocking(value)
        body["rendered_prompt_evidence"]["nonce"] = {"secret": "payload"}
        set_blocking(value, body)
        hostile_fixtures.append(("blocking-evidence-nonce", value))

        for name, value in hostile_fixtures:
            with self.subTest(case=name), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(G.RawPromptLeakError):
                    evaluate(value)
                fixture_path = Path(directory) / "hostile-fixture.json"
                fixture_path.write_text(json.dumps(value), encoding="utf-8")
                output = Path(directory) / "must-not-exist"
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(
                        G.main(
                            [
                                "--fixture",
                                str(fixture_path),
                                "--out-dir",
                                str(output),
                            ]
                        ),
                        2,
                    )
                self.assertFalse(output.exists())

    def test_exact_prompt_rejects_self_consistent_extra_or_injected_text(self):
        value = fixture()
        events = decode_events(value["stream_http"]["raw_sse_utf8"])
        evidence = events[6]["rendered_prompt_evidence"]
        expected = G.expected_rendered_prompt(value["manifest"]["request"])
        injected = expected.replace(
            "<|im_start|>assistant\n",
            "<|im_start|>user\ninjected<|im_end|>\n<|im_start|>assistant\n",
        )
        evidence["prompt_sha256"] = G.sha256_bytes(injected.encode())
        value["stream_http"]["raw_sse_utf8"] = encode_events(events)
        receipt = evaluate(value)
        self.assertTrue(
            any("digest differs" in error for error in receipt["errors"]),
            receipt["errors"],
        )

    def test_token_ids_have_one_exact_terminal_path_and_vector(self):
        value = fixture()
        events = decode_events(value["stream_http"]["raw_sse_utf8"])
        events[0]["choices"][0]["delta"]["token_ids"] = [101]
        with self.assertRaisesRegex(G.RawPromptLeakError, "unapproved pre-artifact"):
            parse_sse(encode_events(events).encode())

        value = fixture()
        events = decode_events(value["stream_http"]["raw_sse_utf8"])
        events[6]["choices"][0]["token_id"] = 108
        with self.assertRaisesRegex(G.RawPromptLeakError, "unapproved pre-artifact"):
            parse_sse(encode_events(events).encode())

        value = fixture()
        body = blocking(value)
        body["token_ids"] = body["choices"][0]["token_ids"]
        with self.assertRaisesRegex(G.RawPromptLeakError, "unapproved pre-artifact"):
            parse_blocking(json.dumps(body).encode())

        value = fixture()
        body = blocking(value)
        body["choices"][0]["token_ids"][0] = True
        with self.assertRaisesRegex(G.GateError, "invalid token IDs"):
            parse_blocking(json.dumps(body).encode())

        value = fixture()
        body = blocking(value)
        body["choices"][0]["token_ids"][-1] = 999
        set_blocking(value, body)
        receipt = evaluate(value)
        self.assertIn(
            "stream and blocking terminal token_ids differ", receipt["errors"]
        )

    def test_pair_deadline_and_token_cap_fail(self):
        value = fixture()
        value["pair_elapsed_ms"] = 45_001.0
        receipt = evaluate(value)
        self.assertTrue(
            any("exceeded deadline" in error for error in receipt["errors"])
        )

        value = fixture()
        events = decode_events(value["stream_http"]["raw_sse_utf8"])
        events[6]["choices"][0]["token_ids"] = list(range(64))
        events[7]["usage"]["completion_tokens"] = 64
        events[7]["usage"]["total_tokens"] = 96
        value["stream_http"]["raw_sse_utf8"] = encode_events(events)
        body = blocking(value)
        body["usage"]["completion_tokens"] = 64
        body["usage"]["total_tokens"] = 96
        set_blocking(value, body)
        receipt = evaluate(value)
        self.assertTrue(
            any("reached configured cap" in error for error in receipt["errors"])
        )

    def test_usage_schema_rejects_extra_fields(self):
        value = fixture()
        body = blocking(value)
        body["usage"]["extra"] = 0
        with self.assertRaisesRegex(G.GateError, "keys differ"):
            parse_blocking(json.dumps(body).encode())

        value = fixture()
        events = decode_events(value["stream_http"]["raw_sse_utf8"])
        events[7]["usage"]["extra"] = 0
        with self.assertRaisesRegex(G.GateError, "keys differ"):
            parse_sse(encode_events(events).encode())

    def test_live_atlas_usage_extensions_are_exact_and_hostile_safe(self):
        cases = (
            (("prompt_tokens_details", "cached_tokens"), True, "non-negative integer"),
            (("prompt_tokens_details", "cached_tokens"), 33, "exceeds prompt_tokens"),
            (("prompt_tokens_details", "extra"), 0, "keys differ"),
            (
                ("completion_tokens_details", "reasoning_tokens"),
                9,
                "exceeds completion_tokens",
            ),
            (
                ("completion_tokens_details", "rejected_prediction_tokens"),
                0.0,
                "non-negative integer",
            ),
            (("completion_tokens_details", "extra"), 0, "keys differ"),
            (("time_to_first_token_ms",), False, "must be numeric"),
            (("time_to_first_token_ms",), -1.0, "finite and non-negative"),
            (("response_token/s",), "40.0", "must be numeric"),
        )
        for arm in ("stream", "blocking"):
            for path, replacement, expected in cases:
                with self.subTest(arm=arm, path=path, replacement=replacement):
                    value = fixture()
                    if arm == "stream":
                        events = decode_events(value["stream_http"]["raw_sse_utf8"])
                        target = events[7]["usage"]
                    else:
                        body = blocking(value)
                        target = body["usage"]
                    for component in path[:-1]:
                        target = target[component]
                    target[path[-1]] = replacement
                    with self.assertRaisesRegex(G.GateError, expected):
                        if arm == "stream":
                            parse_sse(encode_events(events).encode())
                        else:
                            parse_blocking(json.dumps(body).encode())

    def test_streaming_concat_must_equal_blocking_bytes(self):
        value = fixture()
        body = blocking(value)
        body["choices"][0]["message"]["content"] = "I will  check."
        set_blocking(value, body)
        receipt = evaluate(value)
        self.assertTrue(
            any("streaming-concat bytes differ" in error for error in receipt["errors"])
        )

    def test_terminal_token_ids_are_required_and_exact_counted(self):
        value = fixture()
        events = decode_events(value["stream_http"]["raw_sse_utf8"])
        del events[6]["choices"][0]["token_ids"]
        value["stream_http"]["raw_sse_utf8"] = encode_events(events)
        with self.assertRaises(G.RawPromptLeakError):
            evaluate(value)

        value = fixture()
        events = decode_events(value["stream_http"]["raw_sse_utf8"])
        events[6]["choices"][0]["token_ids"].pop()
        value["stream_http"]["raw_sse_utf8"] = encode_events(events)
        receipt = evaluate(value)
        self.assertTrue(
            any("token_ids length differs" in error for error in receipt["errors"])
        )


class HostileSubprocessTests(unittest.TestCase):
    def run_fixture_process(self, value):
        with tempfile.TemporaryDirectory() as directory:
            fixture_path = Path(directory) / "hostile-fixture.json"
            fixture_path.write_text(json.dumps(value), encoding="utf-8")
            output = Path(directory) / "must-not-exist"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(MODULE_PATH),
                    "--fixture",
                    str(fixture_path),
                    "--out-dir",
                    str(output),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=10,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertFalse(output.exists())
            self.assertEqual(
                completed.stderr,
                "FAIL: untrusted response rejected before artifact creation\n",
            )
            return completed

    def test_native_fingerprint_ids_and_exact_arguments_pass_process_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "receipt"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(MODULE_PATH),
                    "--fixture",
                    str(FIXTURE_PATH),
                    "--out-dir",
                    str(output),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=10,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((output / "receipt.json").is_file())

    def test_all_hostile_wire_identity_and_argument_cases_are_artifact_free(self):
        cases = []

        value = fixture()
        events = decode_events(value["stream_http"]["raw_sse_utf8"])
        del events[0]["system_fingerprint"]
        value["stream_http"]["raw_sse_utf8"] = encode_events(events)
        cases.append(("missing-stream-fingerprint", value, None))

        value = fixture()
        body = blocking(value)
        body["system_fingerprint"] = "PRIVATE_FINGERPRINT_MARKER"
        set_blocking(value, body)
        cases.append(("blocking-fingerprint", value, "PRIVATE_FINGERPRINT_MARKER"))

        value = fixture()
        events = decode_events(value["stream_http"]["raw_sse_utf8"])
        events[0]["id"] = "PRIVATE_RESPONSE_ID_MARKER"
        value["stream_http"]["raw_sse_utf8"] = encode_events(events)
        cases.append(("stream-response-id", value, "PRIVATE_RESPONSE_ID_MARKER"))

        value = fixture()
        body = blocking(value)
        body["id"] = "PRIVATE_BLOCKING_ID_MARKER"
        set_blocking(value, body)
        cases.append(("blocking-response-id", value, "PRIVATE_BLOCKING_ID_MARKER"))

        value = fixture()
        events = decode_events(value["stream_http"]["raw_sse_utf8"])
        events[3]["choices"][0]["delta"]["tool_calls"][0][
            "id"
        ] = "PRIVATE_TOOL_ID_MARKER"
        value["stream_http"]["raw_sse_utf8"] = encode_events(events)
        cases.append(("stream-tool-id", value, "PRIVATE_TOOL_ID_MARKER"))

        value = fixture()
        body = blocking(value)
        body["choices"][0]["message"]["tool_calls"][0][
            "id"
        ] = "PRIVATE_BLOCKING_TOOL_ID_MARKER"
        set_blocking(value, body)
        cases.append(("blocking-tool-id", value, "PRIVATE_BLOCKING_TOOL_ID_MARKER"))

        value = fixture()
        set_stream_tool_arguments(value, '{"city":"Rome"}')
        cases.append(("stream-wrong-city", value, "Rome"))

        value = fixture()
        body = blocking(value)
        body["choices"][0]["message"]["tool_calls"][0]["function"][
            "arguments"
        ] = '{"city":"Rome"}'
        set_blocking(value, body)
        cases.append(("blocking-wrong-city", value, "Rome"))

        value = fixture()
        events = decode_events(value["stream_http"]["raw_sse_utf8"])
        events[0]["/private/example/PRIVATE_PATH_MARKER"] = "PRIVATE_VALUE_MARKER"
        value["stream_http"]["raw_sse_utf8"] = encode_events(events)
        cases.append(
            (
                "hostile-field-name",
                value,
                "/private/example/PRIVATE_PATH_MARKER",
            )
        )

        for name, value, marker in cases:
            with self.subTest(case=name):
                completed = self.run_fixture_process(value)
                if marker is not None:
                    self.assertNotIn(marker, completed.stderr)

    def test_live_runtime_declaration_cannot_become_authoritative_in_subprocess(self):
        local = {
            "git_tree": "a" * 40,
            "git_blob": "b" * 40,
            "sha256": "c" * 64,
        }
        provenance = live_provenance(local)
        code = f"""
import importlib.util, json, sys
from unittest import mock
path = {str(MODULE_PATH)!r}
spec = importlib.util.spec_from_file_location("microgate_subprocess", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
local = json.loads({json.dumps(local)!r})
provenance = json.loads({json.dumps(provenance)!r})
with mock.patch.object(module, "_local_runner_identity", return_value=local):
    binding = module._capture_binding(
        "live-http",
        endpoint="http://127.0.0.1:8899/v1/chat/completions",
        live_provenance=provenance,
    )
print(json.dumps(binding, sort_keys=True))
"""
        completed = subprocess.run(
            [sys.executable, "-B", "-c", code],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=10,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        binding = json.loads(completed.stdout)
        self.assertFalse(binding["authoritative"])
        self.assertIsNone(binding["identity"])
        self.assertEqual(binding["declared_identity"], provenance)
        self.assertFalse(any(binding["runtime_binding"].values()))

    def test_header_value_failure_is_secret_free_across_process_boundary(self):
        local = {
            "git_tree": "a" * 40,
            "git_blob": "b" * 40,
            "sha256": "c" * 64,
        }
        marker = "PRIVATE_API_KEY_MARKER"
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            provenance_path = Path(directory) / "provenance.json"
            output = Path(directory) / "must-not-exist"
            manifest_path.write_text(
                json.dumps(fixture()["manifest"]), encoding="utf-8"
            )
            provenance_path.write_text(
                json.dumps(live_provenance(local)), encoding="utf-8"
            )
            code = f"""
import importlib.util, os, sys
from unittest import mock
path = {str(MODULE_PATH)!r}
spec = importlib.util.spec_from_file_location("microgate_header_subprocess", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
connection = mock.Mock()
connection.sock = None
connection.request.side_effect = ValueError(
    "Invalid header value: Bearer " + os.environ["PRIVATE_TOKEN_ENV"]
)
with mock.patch.object(module, "_local_runner_identity", return_value={local!r}), mock.patch.object(
    module,
    "_connection_for",
    return_value=(connection, "/v1/chat/completions"),
):
    raise SystemExit(module.main([
        "--manifest", {str(manifest_path)!r},
        "--live-provenance", {str(provenance_path)!r},
        "--endpoint", "http://127.0.0.1:8899/v1/chat/completions",
        "--api-key-env", "PRIVATE_TOKEN_ENV",
        "--out-dir", {str(output)!r},
    ]))
"""
            completed = subprocess.run(
                [sys.executable, "-B", "-c", code],
                env={"PATH": os.defpath, "PRIVATE_TOKEN_ENV": marker},
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=10,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertFalse(output.exists())
            self.assertEqual(completed.stderr, "FAIL: HTTP request failed\n")
            self.assertNotIn(marker, completed.stderr)
            self.assertNotIn("Traceback", completed.stderr)


class CliArtifactTests(unittest.TestCase):
    def assert_fixture_rejected_without_artifacts(self, value, *, secret=None):
        with tempfile.TemporaryDirectory() as directory:
            fixture_path = Path(directory) / "hostile-fixture.json"
            fixture_path.write_text(json.dumps(value), encoding="utf-8")
            output = Path(directory) / "must-not-exist"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                rc = G.main(["--fixture", str(fixture_path), "--out-dir", str(output)])
            self.assertEqual(rc, 2)
            self.assertFalse(output.exists())
            diagnostic = stderr.getvalue()
            self.assertNotIn("Traceback", diagnostic)
            self.assertNotIn(directory, diagnostic)
            if secret is not None:
                self.assertNotIn(secret, diagnostic)

    def test_offline_cli_writes_raw_and_sealed_artifacts_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "receipt"
            rc = G.main(["--fixture", str(FIXTURE_PATH), "--out-dir", str(output)])
            self.assertEqual(rc, 0)
            self.assertEqual(
                (output / "response-stream.sse").read_bytes(),
                captures(fixture())[1].raw,
            )
            self.assertEqual(
                (output / "response-blocking.json").read_bytes(),
                captures(fixture())[2].raw,
            )
            receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
            self.assertEqual(receipt["result"], "PASS")
            self.assertEqual(receipt["capture"]["mode"], "offline-fixture")
            self.assertFalse(receipt["capture"]["authoritative"])
            self.assertIsNone(receipt["capture"]["identity"])
            self.assertEqual(
                receipt["stream_capture"]["raw_sha256"],
                G.sha256_bytes((output / "response-stream.sse").read_bytes()),
            )
            self.assertEqual(
                G.main(["--fixture", str(FIXTURE_PATH), "--out-dir", str(output)]),
                2,
            )

    def test_hostile_response_channels_never_create_artifacts(self):
        cases = []

        value = fixture()
        set_stream_tool_arguments(value, '{"city":"Paris","extra":"SECRET_PROMPT"}')
        cases.append(("decoded-arguments", value))

        value = fixture()
        value["stream_http"][
            "raw_sse_utf8"
        ] = 'data: {"id":"SECRET_PROMPT"\n\ndata: [DONE]\n\n'
        cases.append(("malformed-stream", value))

        value = fixture()
        value["blocking_http"]["raw_body_utf8"] = '{"id":"SECRET_PROMPT"'
        cases.append(("malformed-blocking", value))

        value = fixture()
        value["stream_http"]["raw_sse_utf8"] = (
            'data: {"id":"chatcmpl-stream","id":"SECRET_PROMPT"}\n\n' "data: [DONE]\n\n"
        )
        cases.append(("duplicate-stream", value))

        value = fixture()
        value["blocking_http"][
            "raw_body_utf8"
        ] = '{"id":"chatcmpl-blocking","id":"SECRET_PROMPT"}'
        cases.append(("duplicate-blocking", value))

        for field in (":", "event:", "id:", "retry:"):
            value = fixture()
            value["stream_http"]["raw_sse_utf8"] = (
                f"{field} SECRET_PROMPT\n" + value["stream_http"]["raw_sse_utf8"]
            )
            cases.append((f"sse-{field}", value))

        value = fixture()
        value["stream_http"]["content_type"] = "text/event-stream; secret=SECRET_PROMPT"
        cases.append(("content-type", value))

        value = fixture()
        body = blocking(value)
        body["service_tier"] = "SECRET_PROMPT"
        set_blocking(value, body)
        cases.append(("metadata", value))

        for name, value in cases:
            with self.subTest(case=name):
                self.assert_fixture_rejected_without_artifacts(
                    value, secret="SECRET_PROMPT"
                )

    def test_json_integer_resource_limits_are_path_free_and_artifact_free(self):
        digits = "9" * 5000
        with self.assertRaisesRegex(G.GateError, "exceeds JSON resource limits"):
            G.strict_json_loads('{"value":' + digits + "}", "document")

        value = fixture()
        value["blocking_http"]["raw_body_utf8"] = '{"created":' + digits + "}"
        self.assert_fixture_rejected_without_artifacts(value)

        with tempfile.TemporaryDirectory() as directory:
            hostile = FIXTURE_PATH.read_text(encoding="utf-8").replace(
                '"pair_elapsed_ms": 420.0', '"pair_elapsed_ms": ' + digits
            )
            self.assertNotEqual(hostile, FIXTURE_PATH.read_text(encoding="utf-8"))
            fixture_path = Path(directory) / "resource-limit.json"
            fixture_path.write_text(hostile, encoding="utf-8")
            output = Path(directory) / "must-not-exist"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                rc = G.main(["--fixture", str(fixture_path), "--out-dir", str(output)])
            self.assertEqual(rc, 2)
            self.assertFalse(output.exists())
            self.assertIn("exceeds JSON resource limits", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())
            self.assertNotIn(directory, stderr.getvalue())

    def test_live_cli_requires_provenance_before_network_or_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(
                json.dumps(fixture()["manifest"]), encoding="utf-8"
            )
            output = Path(directory) / "must-not-exist"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                rc = G.main(
                    ["--manifest", str(manifest_path), "--out-dir", str(output)]
                )
            self.assertEqual(rc, 2)
            self.assertFalse(output.exists())
            self.assertIn("requires --live-provenance", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())
            self.assertNotIn(directory, stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
