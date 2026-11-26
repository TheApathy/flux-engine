# SPDX-License-Identifier: AGPL-3.0-only
"""Offline tests for the live cancellation-slot-reuse gate driver.

Every Docker, server and process binding is mocked; nothing here qualifies an
image. The native-concurrency fake engine supplies the probe's receipt grammar
(frames, logits rows, state receipts, cancellation, slot reuse), the real probe
validators run unchanged, and the outer wrapper's strict validator must accept
the real-shaped output and reject every tampered or missing piece.
"""

from contextlib import ExitStack, contextmanager
import copy
import hashlib
import http.server
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def module(path, name):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    loader.exec_module(value)
    return value


NT = module(ROOT / "tests" / "test_r16_native_concurrency_gate_driver.py", "r16_cancel_nc_fixture")
S, Q, P, NC = NT.S, NT.Q, NT.P, NT.D
C = module(ROOT / "r15-gates" / "cancellation-slot-reuse", "r16_cancellation_driver")
DRIVER = ROOT / "r15-gates" / "cancellation-slot-reuse"
NC_PATH = ROOT / "r15-gates" / "native-concurrency" / "native-concurrency-c2-c8"


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


class CancellationSlotReuseDriverTests(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        # Worker threads must fit the 90 MiB address-space cap of the source suite.
        previous = threading.stack_size(256 * 1024)
        cls.addClassCleanup(threading.stack_size, previous)

    def setUp(self):
        self.fixtures = NT.NativeConcurrencyDriverTests("test_arm_argv_is_the_canonical_short_context_contract")
        patcher = mock.patch.object(C, "IDLE_TIMEOUT_SECONDS", 0.1)
        patcher.start()
        self.addCleanup(patcher.stop)

    # --- end to end with the fake engine ---------------------------------

    def exercise(self, knobs=None, *, modes=C.MODES, expect_pass=True, active=lambda mode, n: 0):
        knobs = knobs or {}
        engines, counters, polls = [], {"ticks": 0}, {"n": 0}
        temp = tempfile.mkdtemp()
        type(self).addClassCleanup(shutil.rmtree, temp, True)
        runtime = Path(temp)
        receipt = runtime / "receipt.json"
        receipt_fd = os.open(receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        raw_log_fd = os.open(runtime / "raw.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        type(self).addClassCleanup(os.close, receipt_fd)
        type(self).addClassCleanup(os.close, raw_log_fd)
        probe_sha = sha(NT.PROBE_PATH.read_bytes())
        _payload, probe_identity = NC.read_pinned_bytes(NT.PROBE_PATH, probe_sha)
        args = types.SimpleNamespace(
            nc=NC, suite=S, probe=P, probe_identity=probe_identity,
            probe_program=str(NT.PROBE_PATH), probe_sha256=probe_sha,
            runtime_dir=str(runtime), receipt_fd=receipt_fd, raw_log_fd=raw_log_fd,
            container_id="1" * 64, image_id="sha256:" + "2" * 64, image_digest="sha256:" + "3" * 64,
            run_id="offline-cancel-fixture", base_url="http://127.0.0.1:8888", container_port=8888,
            container_cwd="/", container_results_path="/results", max_tokens=NT.MAX_TOKENS,
            minimum_full_frames=8, memory_floor_mib=1, timeout=30.0, cancel_token=NT.CANCEL_TOKEN,
            modes=modes, ladder=False, identities={"base_url": "http://127.0.0.1:8888"}, model_name=None,
        )
        mounts = {
            "/models": {"Source": "/ckpt", "RW": False, "Type": "bind"},
            "/results": {"Source": str(runtime), "RW": True, "Type": "bind"},
        }

        def fake_urlopen(request_object, timeout=None):
            if request_object.data is None:
                return NT.HealthResponse()
            body = json.loads(request_object.data)
            return NT.FakeResponse(engines[-1].stream(body))

        def fake_request(_args, _method, route, body=None, **_kwargs):
            if route != "/metrics":
                return b"{}"
            engine = engines[-1]
            polls["n"] += 1
            k5 = b'atlas_spec_decode_verify_total{k="5",outcome="accepted"} 12\n' if engine.mode == "native-mtp-k5" else b""
            return k5 + f"atlas_requests_active {active(engine.mode, polls['n'])}\n".encode()

        @contextmanager
        def server(_args, context, mode, profile, width, stem):
            args.model_name = context.model
            with self.fixtures.fixture_server(runtime, knobs, engines, counters, args, context, mode, profile, width, stem) as live:
                yield live

        original_churn = P.run_api_churn
        original_read = P.LogCursor.read_bytes_from

        def ordered_churn(base_url, model, cases, replacement, max_tokens, timeout, baseline):
            engines[-1].plan_admissions(cases, replacement)
            return original_churn(base_url, model, cases, replacement, max_tokens, timeout, baseline)

        patches = (
            mock.patch.object(NC, "inspect_container", return_value=(self.fixtures.plan(), self.fixtures.recipe(), {}, mounts)),
            mock.patch.object(NC, "verify_runtime_bind"),
            mock.patch.object(NC, "preflight_provenance", return_value=NT.provenance_fixture()),
            mock.patch.object(NC, "recheck_provenance"),
            mock.patch.object(NC, "server", side_effect=server),
            # Production loads the pinned driver with its frame label rebound.
            mock.patch.object(NC, "GATE", C.GATE),
            mock.patch.object(NC, "request", side_effect=fake_request),
            mock.patch.object(urllib.request, "urlopen", fake_urlopen),
            mock.patch.object(P, "run_api_churn", side_effect=ordered_churn),
            mock.patch.object(P, "WAVE_ADMISSION_STAGGER_S", 0.002),
            mock.patch.object(
                P.LogCursor, "read_bytes_from",
                lambda self, offset, settle_seconds=0.15, end_offset=None: original_read(self, offset, 0.0, end_offset),
            ),
        )
        try:
            with ExitStack() as stack:
                for patch in patches:
                    stack.enter_context(patch)
                if not expect_pass:
                    with self.assertRaises(Exception) as caught:
                        C.run(args)
                    self.assertEqual(receipt.read_bytes(), b"")
                    base = runtime / "cancellation" / C.GATE
                    self.assertFalse((base / "validator.json").exists())
                    self.assertEqual(len(list(base.glob("failure-*.json"))), 1)
                    return caught.exception
                result = C.run(args)
        finally:
            P.urllib = urllib
        self.assertEqual(json.loads(receipt.read_bytes()), result)
        S.validate_gate_receipt(
            receipt.read_bytes(), C.GATE,
            {"image_id": args.image_id, "image_digest": args.image_digest, "container_id": args.container_id, "run_id": args.run_id},
        )
        manifest = {"_runtime_dir": str(runtime), "launch": {"run_id": args.run_id}}
        Q._validate_cancellation_evidence(result["evidence"], manifest, args.container_id)
        return result["evidence"], runtime, manifest, args

    _shared = None

    def passing_run(self):
        """One real-shaped PASS run shared by the read-only and tamper tests."""
        if CancellationSlotReuseDriverTests._shared is None:
            CancellationSlotReuseDriverTests._shared = self.exercise()
        return CancellationSlotReuseDriverTests._shared

    def test_mocked_live_run_passes_the_strict_wrapper_validator(self):
        evidence, runtime, _manifest, _args = self.passing_run()
        self.assertEqual(
            {k: evidence[k] for k in ("concurrency", "cancelled_requests", "survivors", "replacement_requests")},
            {"concurrency": 8, "cancelled_requests": 1, "survivors": 7, "replacement_requests": 1},
        )
        raw = json.loads((runtime / evidence["validator_artifact"]).read_bytes())["raw"]
        self.assertEqual([o["mode"] for o in raw["mode_observations"]], list(C.MODES))
        self.assertEqual(raw["slots"][-1], {"cancelled": True, "survivor": False})
        for observation in raw["mode_observations"]:
            self.assertEqual(observation["replacement"]["slot"], observation["cancelled"]["retired_slot"])
            self.assertEqual(
                observation["replacement"]["token_ids_sha256"],
                observation["replacement"]["fresh_slot_reference_token_ids_sha256"],
            )
            for survivor in observation["survivors"]:
                self.assertEqual(survivor["token_ids_sha256"], survivor["fresh_slot_reference_token_ids_sha256"])
            self.assertEqual([i["phase"] for i in observation["requests_active"]], ["before-churn", "after-drain", "after-reuse"])
            churn = observation["exchanges"]["churn"]
            self.assertEqual(len(churn), 9)
            cancelled = [row for row in churn if row["label"] == observation["cancelled"]["label"]][0]
            self.assertFalse(cancelled["completed_stream"])
            self.assertTrue(all(row["completed_stream"] for row in churn if row is not cancelled))
            self.assertTrue(all(len(row["request_sha256"]) == 64 and len(row["response_sha256"]) == 64 for row in churn))
            arm = (runtime / observation["arm_receipt"]["artifact"]).read_bytes()
            self.assertEqual(sha(arm), observation["arm_receipt"]["sha256"])
            self.assertEqual(sha((runtime / observation["server_log"]["artifact"]).read_bytes()), observation["server_log"]["sha256"])
        self.assertEqual((runtime / "raw.log").read_bytes().count(b"R15_LOG_FRAME"), 2)
        self.assertIn(b'"gate":"cancellation-slot-reuse"', (runtime / "raw.log").read_bytes())

    def test_replacement_in_a_fresh_slot_fails_closed(self):
        error = self.exercise({"replacement_wrong_slot": True}, expect_pass=False)
        self.assertIn("slot", str(error))

    def test_missing_cancelled_state_receipt_fails_closed(self):
        error = self.exercise({"missing_state": True}, expect_pass=False)
        self.assertIn("state", str(error))

    def test_slot_never_released_on_metrics_fails_closed(self):
        error = self.exercise(expect_pass=False, active=lambda mode, n: 1 if n > 1 else 0)
        self.assertIn("never drained", str(error))

    def test_plain_only_cannot_produce_a_pass(self):
        error = self.exercise(modes=("none",), expect_pass=False)
        self.assertIn("both plain and native-K5", str(error))

    # --- wrapper rejects tampered or missing pieces -------------------------

    def rewrite_validator(self, runtime, evidence, mutate):
        path = runtime / evidence["validator_artifact"]
        document = json.loads(path.read_bytes())
        mutate(document)
        os.chmod(path, 0o600)
        path.write_bytes(Q.canonical_bytes(document))
        return {**evidence, "validator_receipt_sha256": sha(path.read_bytes())}

    def test_wrapper_rejects_tampered_validator_raw_and_evidence(self):
        evidence, runtime, manifest, args = self.passing_run()
        pristine = (runtime / evidence["validator_artifact"]).read_bytes()
        raw_mutations = {
            "two cancelled": lambda d: d["raw"]["slots"][0].update(cancelled=True, survivor=False),
            "seven slots": lambda d: d["raw"]["slots"].pop(),
            "ambiguous slot": lambda d: d["raw"]["slots"][1].update(cancelled=True),
            "no replacement": lambda d: d["raw"].update(replacement_requests=[]),
            "replacement false": lambda d: d["raw"].update(replacement_requests=[False]),
            "replacement died": lambda d: d["raw"].update(replacement_survivor=False),
            "plain only": lambda d: d["raw"].update(plain_and_k5=False),
            "survivor drift": lambda d: d["raw"].update(exact_survivor_identity=False),
            "fresh slot": lambda d: d["raw"].update(final_highest_slot_reused=False),
            "dirty drain": lambda d: d["raw"].update(post_drain_clean=False),
            "missing census": lambda d: d["raw"].pop("slots"),
            "not pass": lambda d: d.update(status="FAIL"),
            "other run": lambda d: d.update(run_id="other"),
            "other gate": lambda d: d.update(gate="memory-floor"),
        }
        try:
            for name, mutate in raw_mutations.items():
                with self.subTest(raw=name):
                    (runtime / evidence["validator_artifact"]).chmod(0o600)
                    (runtime / evidence["validator_artifact"]).write_bytes(pristine)
                    tampered = self.rewrite_validator(runtime, evidence, mutate)
                    with self.assertRaises(ValueError):
                        Q._validate_cancellation_evidence(tampered, manifest, args.container_id)
        finally:
            (runtime / evidence["validator_artifact"]).chmod(0o600)
            (runtime / evidence["validator_artifact"]).write_bytes(pristine)
        Q._validate_cancellation_evidence(evidence, manifest, args.container_id)
        evidence_mutations = {
            "survivors": {**evidence, "survivors": 6},
            "cancelled": {**evidence, "cancelled_requests": 2},
            "flag as int": {**evidence, "post_drain_clean": 1},
            "sha": {**evidence, "validator_receipt_sha256": "f" * 64},
            "missing key": {k: v for k, v in evidence.items() if k != "plain_and_k5"},
            "extra key": {**evidence, "pass": True},
        }
        for name, tampered in evidence_mutations.items():
            with self.subTest(evidence=name), self.assertRaises(ValueError):
                Q._validate_cancellation_evidence(tampered, manifest, args.container_id)
        with self.assertRaises(ValueError):
            Q._validate_cancellation_evidence(evidence, manifest, "9" * 64)

    # --- driver recomputation rejects tampered observations -----------------

    def arm_inputs(self):
        evidence, runtime, _manifest, _args = self.passing_run()
        raw = json.loads((runtime / evidence["validator_artifact"]).read_bytes())["raw"]
        observation = raw["mode_observations"][0]
        arm = json.loads((runtime / observation["arm_receipt"]["artifact"]).read_bytes())
        labels = [row["label"] for row in observation["exchanges"]["churn"] if row["label"] != observation["replacement"]["label"]]
        order = [case.label for case in P.make_cases(8)]
        self.assertEqual(sorted(labels), sorted(order))
        return (
            order, arm["baseline_c1"], arm["api_churn"], arm["post_drain_reuse"],
            observation["exchanges"]["churn"], observation["exchanges"]["post_drain_reuse"],
            observation["requests_active"],
        )

    def test_derive_observation_recomputes_and_rejects_every_tampered_row(self):
        inputs = self.arm_inputs()
        accepted = C.derive_observation("none", *copy.deepcopy(inputs))
        self.assertTrue(accepted["exact_survivor_identity"] and accepted["post_drain_clean"])

        def tamper(index, edit):
            values = list(copy.deepcopy(inputs))
            edit(values[index])
            return values

        def bump_token(rows):
            rows[0]["token_ids"] = [rows[0]["token_ids"][0] + 1, *rows[0]["token_ids"][1:]]

        def wrong_slot(churn):
            session = [r for r in inputs[1] if r["label"] == churn["replacement"]["label"]][0]["session"]
            for frame in churn["frames"]:
                if session in frame["sessions"]:
                    i = frame["sessions"].index(session)
                    frame["slots"][i] = frame["slots"][i] + 100
                    break

        cases = {
            "survivor token drift": (2, lambda churn: bump_token(churn["survivors"])),
            "survivor sha drift": (2, lambda churn: churn["survivors"][1].update(token_ids_sha256="0" * 64)),
            "replacement drift": (2, lambda churn: churn["replacement"].update(finish_reason="stop-other")),
            "replacement fresh slot": (2, wrong_slot),
            "cancel not final admission": (2, lambda churn: churn["cancelled"].update(label="row-00")),
            "cancel boundary": (2, lambda churn: churn["cancelled"].update(content_chunks_before_close=7)),
            "replacement before close": (2, lambda churn: churn.update(replacement_started_unix=churn["cancelled"]["closed_unix"] - 1)),
            "missing survivor": (2, lambda churn: churn["survivors"].pop()),
            "reuse overlapped": (3, lambda reuse: reuse["frames"].append({"sessions": [], "slots": []})),
            "reuse drift": (3, lambda reuse: reuse["request"].update(token_ids_sha256="0" * 64)),
            "cancel completed": (4, lambda rows: [r.update(completed_stream=True) for r in rows]),
            "extra request": (4, lambda rows: rows.append(dict(rows[0]))),
            "missing reuse": (5, lambda rows: rows.clear()),
            "slot still active": (6, lambda idle: idle[1].update(final=1)),
            "missing idle phase": (6, lambda idle: idle.pop()),
        }
        for name, (index, edit) in cases.items():
            with self.subTest(case=name), self.assertRaises(C.GateFailure):
                C.derive_observation("none", *tamper(index, edit))

    def test_census_requires_both_modes_in_order_and_identical_slots(self):
        base = {
            "slots": [{"cancelled": i == 7, "survivor": i != 7} for i in range(8)],
            "exact_survivor_identity": True, "final_highest_slot_reused": True,
            "post_drain_clean": True, "replacement_survivor": True,
        }
        good = [{**base, "mode": "none"}, {**base, "mode": "native-mtp-k5"}]
        raw = C.census(copy.deepcopy(good))
        evidence = C.evidence_from(raw, "x/validator.json", "a" * 64)
        self.assertEqual((evidence["cancelled_requests"], evidence["survivors"], evidence["replacement_requests"]), (1, 7, 1))
        for name, observations in {
            "reversed": list(reversed(good)),
            "plain only": good[:1],
            "slot mismatch": [good[0], {**good[1], "slots": base["slots"][::-1]}],
            "k5 drift": [good[0], {**good[1], "exact_survivor_identity": False}],
            "k5 replacement died": [good[0], {**good[1], "replacement_survivor": False}],
        }.items():
            with self.subTest(case=name), self.assertRaises(C.GateFailure):
                C.census(copy.deepcopy(observations))

    # --- request/response hashing against a real local mock server ----------

    def test_exchange_hook_hashes_real_bytes_and_marks_client_disconnect(self):
        lines = [b'data: {"choices":[{"delta":{"content":"t%d"}}]}\n\n' % i for i in range(12)]
        final = b'data: {"choices":[{"delta":{},"finish_reason":"length","token_ids":[1,2]}],"usage":{"prompt_tokens":3,"completion_tokens":12}}\n\ndata: [DONE]\n\n'

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                try:
                    for line in [*lines, final]:
                        self.wfile.write(line)
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *_args):
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = f"http://127.0.0.1:{server.server_address[1]}"
        registry = C.ExchangeRegistry()
        C.install_exchange_hook(P, registry)
        self.addCleanup(setattr, P, "urllib", urllib)
        case = P.make_cases(1)[0]
        cancelled = P.stream_then_cancel(base, "m", case, 12, 5.0, cancel_after_chunks=4, cancelled=threading.Event())
        self.assertEqual(cancelled.content_chunks, 4)
        rows = C.exchange_rows(registry.between(0, registry.mark()), C.request_labels(P, "m", [case], 12))
        self.assertEqual(len(rows), 1)
        body = P.json_canonical(P.request_body("m", case, 12))
        self.assertEqual(rows[0]["request_sha256"], sha(body))
        # Iteration stops right after the fourth content line, before its blank separator.
        self.assertEqual(rows[0]["response_sha256"], sha(b"".join(lines[:3]) + lines[3][:-1]))
        self.assertFalse(rows[0]["completed_stream"])
        self.assertGreater(rows[0]["response_bytes"], 0)
        self.assertLess(rows[0]["response_bytes"], len(b"".join([*lines, final])))
        with self.assertRaises(C.GateFailure):
            C.exchange_rows(registry.between(0, registry.mark()), {})

    # --- argv contract -------------------------------------------------------

    def test_own_flags_are_required_once_and_the_pin_is_enforced(self):
        argv = ["--container-id", "x", "--concurrency-driver-program", str(NC_PATH), "--concurrency-driver-sha256", "a" * 64]
        own, rest = C.split_own_flags(argv)
        self.assertEqual(rest, ["--container-id", "x"])
        self.assertEqual(own["--concurrency-driver-sha256"], "a" * 64)
        for bad in (argv[:2], argv + argv[2:4], ["--concurrency-driver-program=" + str(NC_PATH), *argv[2:]]):
            with self.subTest(argv=bad), self.assertRaises(C.GateFailure):
                C.split_own_flags(bad)
        with self.assertRaises(C.GateFailure):
            C.load_concurrency_driver(NC_PATH, "b" * 64)
        loaded = C.load_concurrency_driver(NC_PATH, sha(NC_PATH.read_bytes()))
        self.assertEqual(loaded.GATE, C.GATE)
        self.assertEqual(NC.GATE, "native-concurrency-c2-c8")

    def test_example_plan_entry_matches_the_suite_contract_and_driver_flags(self):
        example = json.loads((ROOT / "examples" / "r15-container-suite-plan.example.json").read_bytes())
        job = example["gate_jobs"][C.GATE]
        self.assertTrue(job["program_path"].endswith("repro-kit/r15-gates/cancellation-slot-reuse"))
        self.assertEqual(job["argv"][0], job["program_path"])
        for token in S.IDENTITY_PLACEHOLDERS:
            self.assertEqual(job["argv"].count(token), 1)
        own, rest = C.split_own_flags(job["argv"][1:])
        self.assertTrue(own["--concurrency-driver-program"].endswith("native-concurrency/native-concurrency-c2-c8"))
        native = example["gate_jobs"]["native-concurrency-c2-c8"]["argv"][1:]
        self.assertEqual(rest, native)
        self.assertTrue(os.access(DRIVER, os.X_OK))


if __name__ == "__main__":
    unittest.main()
