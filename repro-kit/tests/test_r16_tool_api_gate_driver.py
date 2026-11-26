# SPDX-License-Identifier: AGPL-3.0-only
"""Offline tests for r15-gates/tool-api-streaming-blocking.

A loopback mock answers real HTTP requests with the microgate's committed real-shaped Atlas captures. It
substitutes each request's rendered-prompt nonce, so the driver's real request path is exercised: SSE and
blocking requests, the microgate's strict parsers and comparison, and the gate's case derivation. The produced
evidence is then fed to the outer wrapper's own `_validate_tool_evidence`. These tests do not qualify an image,
a server or a model: no Docker, GPU or engine is involved.
"""

import copy
import hashlib
import http.server
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest

KIT = Path(__file__).resolve().parents[1]
DRIVER_PATH = KIT / "r15-gates" / "tool-api-streaming-blocking"
_loader = importlib.machinery.SourceFileLoader("_r16_tool_api_driver_under_test", str(DRIVER_PATH))
_spec = importlib.util.spec_from_loader(_loader.name, _loader)
D = importlib.util.module_from_spec(_spec)
sys.modules[_loader.name] = D
_loader.exec_module(D)
MG = D.load_checked(str(KIT / "tool_eval_runaway_microgate.py"), D.MICROGATE_SHA256, "_r16_tool_api_test_mg")
MANIFEST_PATH = KIT / "examples" / "tool-eval-runaway-manifest.example.json"
MANIFEST_VALUE, VALIDATED = D.load_sealed_manifest(str(MANIFEST_PATH), MG)
FIXTURE = json.loads((KIT / "tests" / "fixtures" / "tool_eval_runaway_micro.json").read_text("utf-8"))
_, FX_STREAM, FX_BLOCKING, _, FX_PAIR = MG.fixture_captures(FIXTURE)
_wrapper_spec = importlib.util.spec_from_file_location("_r16_tool_api_wrapper", KIT / "r15_exact_image_release.py")
sys.path.insert(0, str(KIT))
R = importlib.util.module_from_spec(_wrapper_spec)
sys.modules[_wrapper_spec.name] = R
_wrapper_spec.loader.exec_module(R)

CID = "c" * 64
RUN_ID = "r15-tool-api-test"
IMAGE = "sha256:" + "d" * 64


class _BoundedStackTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous = threading.stack_size(512 * 1024)
        cls.addClassCleanup(threading.stack_size, previous)


class MockAtlas:
    """One-thread loopback server replaying real-shaped Atlas captures with the request's nonce."""

    def __init__(self, stream_raw=FX_STREAM.raw, blocking_raw=FX_BLOCKING.raw):
        self.stream_raw, self.blocking_raw = stream_raw, blocking_raw
        self.requests = []
        mock = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                mock.requests.append(body)
                nonce = body["rendered_prompt_evidence_nonce"].encode()
                if body["stream"]:
                    raw = mock.stream_raw.replace(FX_PAIR.stream_nonce.encode(), nonce)
                    self.send_response(200)
                    self.send_header("Content-Type", FX_STREAM.content_type)
                    self.end_headers()
                    for line in raw.splitlines(keepends=True):
                        self.wfile.write(line)
                        self.wfile.flush()
                else:
                    raw = mock.blocking_raw.replace(FX_PAIR.blocking_nonce.encode(), nonce)
                    self.send_response(200)
                    self.send_header("Content-Type", FX_BLOCKING.content_type)
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)

        self.server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        self.endpoint = f"http://127.0.0.1:{self.server.server_address[1]}/v1/chat/completions"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_exc):
        self.server.shutdown()
        self.server.server_close()


def live_round(mock):
    pair, stream, blocking, elapsed = D.exchange(MG, VALIDATED, mock.endpoint)
    return D.qualify(MG, MANIFEST_VALUE, VALIDATED, pair, stream, blocking, elapsed)


class WrapperRoundTrip(_BoundedStackTestCase):
    """Real-shaped output must pass the outer wrapper; every tampered piece must not."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.runtime = Path(self.tmp.name)
        with MockAtlas() as mock:
            self.cases, self.core = live_round(mock)
            self.requests = mock.requests

    def tearDown(self):
        self.tmp.cleanup()

    def seal(self, cases=None, core=None, *, status="PASS", container_id=CID, name="validator.json"):
        validator_raw, locator, evidence, receipt = D.documents(
            cases or self.cases, core or self.core, {}, {"process_bindings": []},
            CID, RUN_ID, IMAGE, IMAGE, MG.canonical_bytes,
        )
        document = json.loads(validator_raw)
        document["status"], document["container_id"] = status, container_id
        if cases is not None:
            document["raw"]["cases"] = cases
        validator_raw = MG.canonical_bytes(document)
        evidence["validator_receipt_sha256"] = hashlib.sha256(validator_raw).hexdigest()
        evidence["validator_artifact"] = D.BASE + "/" + name
        (self.runtime / D.BASE).mkdir(exist_ok=True)
        (self.runtime / D.BASE / name).write_bytes(validator_raw)
        return evidence, receipt

    def validate(self, evidence):
        manifest = {"_runtime_dir": str(self.runtime), "launch": {"run_id": RUN_ID}}
        return R._validate_tool_evidence(evidence, manifest, CID)

    def test_real_requests_were_streaming_then_blocking(self):
        self.assertEqual([body["stream"] for body in self.requests], [True, False])
        self.assertEqual(self.requests[0]["tools"], self.requests[1]["tools"])
        self.assertNotEqual(self.requests[0]["rendered_prompt_evidence_nonce"],
                            self.requests[1]["rendered_prompt_evidence_nonce"])
        self.assertEqual(self.requests[0]["model"], "qad5500")

    def test_real_shaped_output_is_accepted(self):
        evidence, receipt = self.seal()
        self.assertEqual(self.validate(evidence), evidence)
        self.assertEqual(
            {k: evidence[k] for k in ("blocking_cases", "streaming_cases", "tool_calls_expected",
                                       "tool_calls_observed")},
            {"blocking_cases": 1, "streaming_cases": 1, "tool_calls_expected": 2, "tool_calls_observed": 2},
        )
        self.assertEqual(set(evidence), {
            "blocking_cases", "streaming_cases", "tool_calls_expected", "tool_calls_observed",
            "exact_arguments", "finish_reasons_valid", "no_blank_line_runaway", "no_all_tools_runaway",
            "thinking_spacing_valid", "validator_artifact", "validator_receipt_sha256"})
        self.assertEqual((receipt["gate"], receipt["execution_scope"], receipt["status"]),
                         (D.GATE, "immutable-container", "PASS"))
        for case in self.cases:
            self.assertEqual(len(case), 8)
        self.assertTrue(all(self.core["consistency"].values()))
        self.assertEqual(self.core["tool_calls"]["stream"], self.core["tool_calls"]["blocking"])

    def test_tampered_evidence_census_is_rejected(self):
        evidence, _ = self.seal()
        for key, value in (("tool_calls_observed", 3), ("tool_calls_expected", 3),
                           ("streaming_cases", 2), ("blocking_cases", 2)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(dict(evidence, **{key: value}))
        for key in ("exact_arguments", "finish_reasons_valid", "no_blank_line_runaway",
                    "no_all_tools_runaway", "thinking_spacing_valid"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(dict(evidence, **{key: False}))

    def test_tampered_or_missing_artifact_is_rejected(self):
        evidence, _ = self.seal()
        path = self.runtime / D.BASE / "validator.json"
        raw = path.read_bytes()
        path.write_bytes(raw.replace(b'"PASS"', b'"PASs"', 1))
        with self.assertRaises(ValueError):
            self.validate(evidence)
        path.unlink()
        with self.assertRaises(ValueError):
            self.validate(evidence)

    def test_raw_cases_must_match_the_evidence(self):
        bad = copy.deepcopy(self.cases)
        bad[1]["finish_reason"] = "stop"
        evidence, _ = self.seal(cases=bad, name="v-stop.json")
        with self.assertRaises(ValueError):
            self.validate(evidence)
        evidence, _ = self.seal(cases=[self.cases[0]], name="v-one.json")
        with self.assertRaises(ValueError):
            self.validate(evidence)
        extra = copy.deepcopy(self.cases)
        extra[0]["request_sha256"] = "0" * 64
        evidence, _ = self.seal(cases=extra, name="v-extra.json")
        with self.assertRaises(ValueError):
            self.validate(evidence)

    def test_identity_and_status_bind(self):
        evidence, _ = self.seal(container_id="e" * 64, name="v-cid.json")
        with self.assertRaises(ValueError):
            self.validate(evidence)
        evidence, _ = self.seal(status="FAIL", name="v-fail.json")
        with self.assertRaises(ValueError):
            self.validate(evidence)


class LiveContractFailures(_BoundedStackTestCase):
    """The driver refuses to produce a PASS when the live wire breaks the sealed contract."""

    def assert_no_pass(self, **mock):
        with MockAtlas(**mock) as server:
            with self.assertRaises(ValueError):
                live_round(server)

    def test_streamed_arguments_differ(self):
        self.assertEqual(FX_STREAM.raw.count(b"Paris"), 1)
        self.assert_no_pass(stream_raw=FX_STREAM.raw.replace(b"Paris", b"Lyons"))

    def test_missing_done_marker(self):
        self.assert_no_pass(stream_raw=FX_STREAM.raw.replace(b"data: [DONE]", b""))

    def test_blank_line_runaway_in_blocking_content(self):
        self.assert_no_pass(blocking_raw=FX_BLOCKING.raw.replace(
            b'"I will check."', b'"I will check.\\n\\n\\n\\n\\nok"', 1))

    def test_gate_failure_carries_check_list(self):
        with MockAtlas(blocking_raw=FX_BLOCKING.raw.replace(
                b'"I will check."', b'"I will check!"', 1)) as server:
            pair, stream, blocking, elapsed = D.exchange(MG, VALIDATED, server.endpoint)
            with self.assertRaises(D.GateFailure) as ctx:
                D.qualify(MG, MANIFEST_VALUE, VALIDATED, pair, stream, blocking, elapsed)
        self.assertTrue(any("content_bytes_identical" in item or "differ" in item for item in ctx.exception.args[1]))


class CaseSemantics(unittest.TestCase):
    def setUp(self):
        self.summary = MG.parse_blocking(FX_BLOCKING.raw, expected_nonce=FX_PAIR.blocking_nonce)

    def case(self, summary=None, rendered=True):
        return D.gate_case("blocking", summary or self.summary, VALIDATED, rendered, MG.SEALED_TOOL_ARGUMENTS)

    def test_clean_case(self):
        case = self.case()
        self.assertEqual(case, {"phase": "blocking", "tool_calls_expected": 1, "tool_calls_observed": 1,
                                "exact_arguments": True, "finish_reason": "tool_calls",
                                "no_blank_line_runaway": True, "no_all_tools_runaway": True,
                                "thinking_spacing_valid": True})

    def test_each_rule_fails_on_its_own_hazard(self):
        s = copy.deepcopy(self.summary)
        s["finish_reason"] = "length"
        self.assertFalse(self.case(s)["no_blank_line_runaway"])
        s = copy.deepcopy(self.summary)
        s["max_consecutive_blank_lines"] = 4
        self.assertFalse(self.case(s)["no_blank_line_runaway"])
        s = copy.deepcopy(self.summary)
        s["usage"]["completion_tokens"] = VALIDATED["token_cap"]
        self.assertFalse(self.case(s)["no_blank_line_runaway"])
        s = copy.deepcopy(self.summary)
        s["tool_calls"] = s["tool_calls"] * 2
        self.assertFalse(self.case(s)["no_all_tools_runaway"])
        self.assertFalse(self.case(s)["exact_arguments"])
        s = copy.deepcopy(self.summary)
        s["tool_calls"][0]["parsed_arguments"] = {"city": "Paris", "unit": "c"}
        self.assertFalse(self.case(s)["exact_arguments"])
        s = copy.deepcopy(self.summary)
        s["reasoning_content"]["bytes"] = 3
        self.assertFalse(self.case(s)["thinking_spacing_valid"])
        s = copy.deepcopy(self.summary)
        s["content"]["utf8"] = "</think>I will check."
        self.assertFalse(self.case(s)["thinking_spacing_valid"])
        self.assertFalse(self.case(rendered=False)["thinking_spacing_valid"])


class LaunchContract(unittest.TestCase):
    ENV = (b"ATLAS_DISABLE_SPECULATION=1\0FLUX_MODE=interactive-w4096\0"
           b"ATLAS_QWEN4_PREFILL_WIDE=4096\0FLUX_QWEN4_PREFILL_WIDE=4096\0")

    def argv(self, model="qad5500"):
        return [item.encode() for item in D.server_argv("/opt/flux/bin/flux-server-real", model, 8888)]

    def test_canonical_argv_satisfies_the_live_contract(self):
        controls = D.check_live_controls(self.argv(), self.ENV, "qad5500")
        self.assertEqual(controls["contract"]["--tool-call-parser"], "qwen3_coder")

    def test_contract_breaks_are_refused(self):
        env = self.ENV
        argv = self.argv()
        index = argv.index(b"--tool-call-parser")
        variants = {
            "no parser": argv[:index] + argv[index + 2:],
            "speculative": argv + [b"--speculative"],
            "template override": argv + [b"--default-chat-template-kwargs", b'{"enable_thinking":true}'],
            "fp8 kv": [b"fp8" if item == b"bf16" else item for item in argv],
            "wider batch": argv + [b"--max-batch-size=2"],
        }
        for label, candidate in variants.items():
            with self.subTest(label), self.assertRaises(ValueError):
                D.check_live_controls(candidate, env, "qad5500")
        for environ in (b"ATLAS_DISABLE_SPECULATION=0\0", b"", b"A=1\0A=2\0"):
            with self.assertRaises(ValueError):
                D.check_live_controls(argv, environ, "qad5500")
        with self.assertRaises(ValueError):
            D.check_live_controls(argv, env, "other-model")


class Pins(unittest.TestCase):
    def test_helper_pins_are_the_tracked_bytes(self):
        for path, pinned in (("r15-gates/mmlu-pro", D.LIFECYCLE_SHA256),
                             ("tool_eval_runaway_microgate.py", D.MICROGATE_SHA256),
                             ("examples/tool-eval-runaway-manifest.example.json", D.MANIFEST_SHA256)):
            self.assertEqual(hashlib.sha256((KIT / path).read_bytes()).hexdigest(), pinned, path)

    def test_wrapper_gate_name_and_driver_path(self):
        self.assertIn(D.GATE, R.REQUIRED_GATES)
        self.assertTrue(os.access(DRIVER_PATH, os.X_OK))

    def test_identity_checks_fail_closed(self):
        class A:
            container_id, image_id, image_digest = CID, IMAGE, IMAGE
            run_id, suite_sha256, container_port = RUN_ID, "a" * 64, 8888
            base_url = "http://127.0.0.1:8899"
        D.check_identities(A)
        for key, value in (("container_id", "x"), ("image_id", "sha256:zz"), ("run_id", "../x"),
                           ("suite_sha256", "0" * 64), ("base_url", "http://example.com:1"),
                           ("container_port", 0)):
            bad = type("B", (A,), {key: value})
            with self.subTest(key), self.assertRaises(ValueError):
                D.check_identities(bad)


if __name__ == "__main__":
    unittest.main()
