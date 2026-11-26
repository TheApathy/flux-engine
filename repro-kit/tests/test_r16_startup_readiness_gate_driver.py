# SPDX-License-Identifier: AGPL-3.0-only
"""Offline startup-readiness gate tests. Docker, HTTP, /proc and servers are mocked."""

from contextlib import contextmanager
import hashlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import socket
import tempfile
import types
import unittest
from unittest import mock

KIT = Path(__file__).resolve().parents[1]


def module(path, name):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    value = importlib.util.module_from_spec(spec)
    loader.exec_module(value)
    return value


D = module(KIT / "r15-gates/startup-readiness", "startup_readiness_driver_fixture")
S = module(KIT / "r15-container-suite.py", "startup_readiness_suite_fixture")
Q = module(KIT / "r15_exact_image_release.py", "startup_readiness_outer_fixture")
H = D.load_checked(KIT / "r15-gates/mmlu-pro", D.LIFECYCLE_SHA256, "startup_lifecycle_fixture")

BINARY = "a" * 64
CONTAINER = "c" * 64
PORT = 8888


def live_proof(mode, run_id, *, reused=False, executable=None, ticks=500):
    return {
        "schema": "flux-r15-live-process-proof/1", "status": "PASS", "mode": mode,
        "container_id": CONTAINER, "run_id": run_id,
        "host_pid": 42 if mode == "none" or reused else 43,
        "start_time_ticks": ticks, "executable_sha256": executable or BINARY,
        "recipe_sha256": "b" * 64, "recipe_environment_verified": True,
        "mount_namespace_match": True, "cgroup_match": True, "paths_read_only": True,
    }


def startup_log(mode, port=PORT, mtp=None, drafts=None, admission=None, extra=b""):
    k5 = mode == "native-mtp-k5"
    mtp = ("true" if k5 else "false") if mtp is None else mtp
    drafts = (4 if k5 else 0) if drafts is None else drafts
    admission = k5 if admission is None else admission
    lines = [
        b"INFO spark_server: Atlas Spark starting...\n",
        b"INFO spark_server: Prefill config: prefill_budget=4096\n",
    ]
    if k5:
        lines.extend((
            b"INFO spark_model: Qwen4 native MTP schema admitted for direct proposer construction\n",
            b"INFO spark_model: Qwen4 native MTP head constructed hidden=2560 residual=10240 experts=512 vocab=248077\n",
        ))
    if admission:
        lines.append(b"INFO spark_server: QWEN4_SPECULATION_ADMISSION_ENGAGED mode=native-only\n")
    lines.append(
        f"INFO spark_server::scheduler: Scheduler started (batched mode, max_batch=1, mtp={mtp}, "
        f"ngram=false, num_drafts={drafts}, policy=fifo, chunked_prefill=true, "
        f"max_prefill_tokens=4096)\n".encode()
    )
    lines.append(f"INFO spark_server: Listening on 0.0.0.0:{port}\n".encode())
    return b"".join(lines) + extra


class StartupDriver(unittest.TestCase):
    def setUp(self):
        self.network = mock.patch.object(
            socket.socket, "connect", side_effect=AssertionError("real network is forbidden"))
        self.network.start()
        self.addCleanup(self.network.stop)
        for name in ("run", "Popen"):
            guard = mock.patch.object(
                D.subprocess, name, side_effect=AssertionError("real subprocess is forbidden"))
            guard.start()
            self.addCleanup(guard.stop)

    def exercise(self, *, logs=None, health=None, models=None, reused_identity=False,
                 checkpoint_drift=False, checkpoint_error=False, rootfs_error=False,
                 executable=None, exited=False, lifecycle_probes=True, broker_drift=False):
        logs = logs or {}
        current = {}
        forwarded = bytearray()
        original_write = S.write_all

        @contextmanager
        def server(args, mode, plan, recipe):
            current["mode"] = mode
            model = "flux-startup-" + mode
            current["model"] = model
            original_write(args.lifecycle.subprocess.descriptor,
                           logs.get(mode, startup_log(mode)))
            if lifecycle_probes:
                for _ in range(2):
                    try:
                        args.lifecycle.request(args, "GET", "/health", timeout=2)
                    except ConnectionError:
                        pass
            proof = live_proof(mode, args.run_id, reused=reused_identity, executable=executable)
            yield model, proof, 90

        calls = {"n": 0}

        def lifecycle_request(args, method, route, body=None, **kwargs):
            calls["n"] += 1
            if calls["n"] % 2:
                raise ConnectionError("server is not ready")
            return S.canonical_bytes({"status": "ready", "model": current["model"]})

        def http_get(args, route):
            model = current["model"]
            if route == "/health":
                value = health(current["mode"], model) if health else (
                    200, {"status": "ready", "model": model})
            else:
                value = models(current["mode"], model) if models else (
                    200, {"object": "list", "data": [
                        {"id": model, "object": "model", "created": 1, "owned_by": "atlas-spark"}]})
            status, body = value
            return status, S.canonical_bytes(body)

        def checkpoint(args, pid):
            if checkpoint_error:
                raise ValueError("startup checkpoint live /models differs from the bind source")
            config = "d" * 64 if not (checkpoint_drift and pid == 43) else "e" * 64
            return {"models_dev": 1, "models_ino": 2, "config_sha256": config, "source": "/ckpt"}

        def rootfs(args, pid):
            if rootfs_error:
                raise ValueError("server executable or recipe is on a writable mount")
            return True

        with tempfile.TemporaryDirectory() as temp:
            runtime = Path(temp)
            descriptor = os.open(runtime / "receipt.json", os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
            raw_path = runtime / "raw.log"
            raw_path.write_bytes(b"previous gate fixture\n")
            raw_fd = os.open(raw_path, os.O_RDONLY)
            broker, driver_end = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
            for mode in D.MODES:
                broker.sendall(S.canonical_bytes(live_proof(
                    mode, "offline-startup-fixture", reused=reused_identity,
                    executable=executable,
                    ticks=501 if broker_drift and mode == "native-mtp-k5" else 500)))
            helper = types.SimpleNamespace(
                make_directory=H.make_directory, inspect_container=lambda *_: (self.plan, {}),
                server=server, request=lifecycle_request, subprocess=D.subprocess,
                docker=lambda *_: b"{}")
            args = types.SimpleNamespace(
                suite=S, lifecycle=helper, runtime_dir=str(runtime), receipt_fd=descriptor,
                raw_log_fd=raw_fd, container_id=CONTAINER, image_id="sha256:" + "a" * 64,
                image_digest="sha256:" + "b" * 64, run_id="offline-startup-fixture",
                container_port=PORT, base_url="http://127.0.0.1:18888",
                proof_channel=D.ProofChannel(driver_end.detach()))

            def write(target, raw):
                if target == 2:
                    forwarded.extend(raw)
                elif target == raw_fd:
                    raise AssertionError("attempted writing read-only raw log")
                else:
                    original_write(target, raw)

            self.plan = {
                "in_container_binary_path": "/opt/flux/bin/server",
                "in_container_recipe_path": "/opt/flux/repro-kit/recipes/fixture.env",
                "in_container_binary_sha256": BINARY, "recipe_sha256": "b" * 64,
                "checkpoint_id": "JMNI-Labs/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid",
                "checkpoint_revision": "d" * 40,
            }
            ready = [([90], [], [])] if exited else [([], [], [])]
            error = result = None
            try:
                with mock.patch.object(D, "parent_plan", return_value=(self.plan, "e" * 64)), \
                        mock.patch.object(D, "http_get", side_effect=http_get), \
                        mock.patch.object(D, "checkpoint_identity", side_effect=checkpoint), \
                        mock.patch.object(D, "rootfs_read_only", side_effect=rootfs), \
                        mock.patch.object(D.select, "select", side_effect=lambda *a: ready[0]), \
                        mock.patch.object(S, "write_all", side_effect=write):
                    try:
                        result = D.run(args)
                    except ValueError as caught:
                        error = str(caught)
                receipt_bytes = os.pread(descriptor, 1 << 20, 0)
                broker.settimeout(1)
                requests = []
                while True:
                    try:
                        packet = broker.recv(16 * 1024)
                    except ConnectionResetError:
                        # Pre-queued broker replies the driver never consumed (early faults).
                        self.assertIsNone(result)
                        break
                    if not packet:
                        break
                    requests.append(json.loads(packet))
                documents = {path.name: path.read_bytes() for path in (runtime / "startup").iterdir()}
                for path in (runtime / "startup").iterdir():
                    self.assertEqual(path.stat().st_mode & 0o777, 0o400)
                manifest = {"_runtime_dir": str(runtime), "launch": {"run_id": args.run_id},
                            "image": {"in_container_binary_sha256": BINARY}}
                checked = None
                if result is not None:
                    checked = Q._validate_startup_evidence(result["evidence"], manifest, CONTAINER)
                    self.tamper(result["evidence"], manifest, runtime)
                return types.SimpleNamespace(result=result, error=error, receipt=receipt_bytes,
                                             requests=requests,
                                             documents=documents, forwarded=bytes(forwarded),
                                             checked=checked)
            finally:
                os.close(descriptor)
                os.close(raw_fd)
                broker.close()

    def tamper(self, evidence, manifest, runtime):
        """The outer validator rejects every tampered or missing piece of a real output."""
        validator = runtime / evidence["validator_artifact"]
        original = validator.read_bytes()
        document = json.loads(original)

        def reject(changed_evidence=None, changed_document=None, *, container=CONTAINER):
            if changed_document is not None:
                validator.chmod(0o600)
                payload = S.canonical_bytes(changed_document)
                validator.write_bytes(payload)
                changed_evidence = dict(changed_evidence or evidence,
                                        validator_receipt_sha256=hashlib.sha256(payload).hexdigest())
            try:
                with self.assertRaises(ValueError):
                    Q._validate_startup_evidence(changed_evidence or evidence, manifest, container)
            finally:
                validator.chmod(0o600)
                validator.write_bytes(original)

        for key, value in (("fresh_process_modes", ["native-mtp-k5", "none"]),
                           ("ready_http_statuses", [200, 503]),
                           ("model_identity_match", False), ("checkpoint_identity_match", False),
                           ("rootfs_read_only", False), ("plain_scheduler_ready", False),
                           ("k5_scheduler_ready", False), ("process_executable_sha256", "f" * 64),
                           ("validator_receipt_sha256", "0" * 63 + "1")):
            reject(dict(evidence, **{key: value}))
        reject({k: v for k, v in evidence.items() if k != "k5_scheduler_ready"})
        reject(container="d" * 64)
        raw = document["raw"]
        attempts = raw["attempts"]
        for mutated in (
            dict(raw, attempts=attempts[:1]),
            dict(raw, attempts=[attempts[0], dict(attempts[1], ready_http_status=503)]),
            dict(raw, attempts=[dict(attempts[0], k5_scheduler_ready=False), attempts[1]]),
            dict(raw, attempts=[attempts[0], dict(attempts[1], extra=True)]),
            dict(raw, attempts=[attempts[1], attempts[0]]),
            dict(raw, rootfs_read_only=False),
            dict(raw, process_executable_sha256="f" * 64),
        ):
            reject(changed_document=dict(document, raw=mutated))
        reject(changed_document=dict(document, status="FAIL"))
        reject(changed_document=dict(document, run_id="other-run"))
        validator.chmod(0o600)
        validator.write_bytes(original + b" ")
        try:
            with self.assertRaises(ValueError):
                Q._validate_startup_evidence(evidence, manifest, CONTAINER)
        finally:
            validator.write_bytes(original)
            validator.chmod(0o400)

    def test_lifecycle_pin_matches_tracked_source(self):
        self.assertEqual(hashlib.sha256((KIT / "r15-gates/mmlu-pro").read_bytes()).hexdigest(),
                         D.LIFECYCLE_SHA256)

    def test_real_shaped_output_is_accepted_and_tampering_rejected(self):
        run = self.exercise()
        self.assertIsNone(run.error)
        self.assertEqual(json.loads(run.receipt), run.result)
        self.assertEqual(run.checked, run.result["evidence"])
        evidence = run.result["evidence"]
        self.assertEqual(evidence["fresh_process_modes"], ["none", "native-mtp-k5"])
        self.assertEqual(evidence["ready_http_statuses"], [200, 200])
        self.assertEqual(set(run.documents), {"engine.log", "responses.jsonl", "validator.json"})
        raw = json.loads(run.documents["validator.json"])["raw"]
        for observation in raw["observations"]:
            outcomes = [row["outcome"] for row in observation["readiness_probes"]]
            self.assertEqual(outcomes, ["ConnectionError", "ok"])
            self.assertLess(observation["launch_monotonic_ns"], observation["ready_monotonic_ns"])
            self.assertLessEqual(observation["ready_monotonic_ns"], observation["health_monotonic_ns"])
        self.assertEqual(raw["observations"][1]["log_census"]["num_drafts"], 4)
        self.assertEqual(raw["artifacts"]["engine.log"]["sha256"],
                         hashlib.sha256(run.documents["engine.log"]).hexdigest())
        self.assertEqual(raw["artifacts"]["responses.jsonl"]["sha256"],
                         hashlib.sha256(run.documents["responses.jsonl"]).hexdigest())
        rows = [json.loads(line) for line in run.documents["responses.jsonl"].splitlines()]
        self.assertEqual([(row["mode"], row["route"]) for row in rows],
                         [("none", "/health"), ("none", "/v1/models"),
                          ("native-mtp-k5", "/health"), ("native-mtp-k5", "/v1/models")])
        self.assertIn(b"Listening on 0.0.0.0:8888", run.forwarded)
        self.assertEqual(run.requests, [
            {"schema": "flux-r15-live-process-request/1", "mode": "none", "host_pid": 42},
            {"schema": "flux-r15-live-process-request/1", "mode": "native-mtp-k5", "host_pid": 43}])
        self.assertEqual([row["process_binding"] for row in raw["observations"]],
                         [live_proof(mode, "offline-startup-fixture") for mode in D.MODES])

    def test_faults_never_emit_a_pass_receipt(self):
        def log(mode, **kwargs):
            return {mode: startup_log(mode, **kwargs)}
        cases = [
            {"health": lambda mode, model: (503, {"status": "loading"})},
            {"health": lambda mode, model: (200, {"status": "ready", "model": "other"})},
            {"models": lambda mode, model: (200, {"object": "list", "data": []})},
            {"models": lambda mode, model: (200, {"object": "list", "data": [
                {"id": model, "object": "model", "created": 1, "owned_by": "atlas-spark"},
                {"id": "x", "object": "model", "created": 1, "owned_by": "atlas-spark"}]})},
            {"logs": log("native-mtp-k5", admission=False)},
            {"logs": log("none", admission=True)},
            {"logs": log("native-mtp-k5", mtp="false")},
            {"logs": log("native-mtp-k5", drafts=2)},
            {"logs": log("none", port=9999)},
            {"logs": {"none": b"INFO Atlas Spark starting...\n"}},
            {"logs": log("none", extra=b"NVRM: Xid (PCI:000f:01:00): 31, pid=1\n")},
            {"logs": log("native-mtp-k5", extra=b"thread 'main' panicked at src/x.rs:1:1\n")},
            {"logs": {"none": startup_log("none") + startup_log("none")}},
            {"reused_identity": True},
            {"checkpoint_drift": True},
            {"checkpoint_error": True},
            {"rootfs_error": True},
            {"executable": "f" * 64},
            {"exited": True},
            {"broker_drift": True},
        ]
        for case in cases:
            with self.subTest(case=sorted(case)):
                run = self.exercise(**case)
                self.assertIsNone(run.result)
                self.assertIsNotNone(run.error)
                self.assertFalse(run.receipt)
                self.assertNotIn("validator.json", run.documents)

    def test_proof_channel_requires_an_ordered_sequenced_packet_socket(self):
        broker, driver_end = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        try:
            channel = D.ProofChannel(driver_end.detach())
            with self.assertRaisesRegex(ValueError, "out of order"):
                channel.request(S, "native-mtp-k5", 43)
            broker.sendall(S.canonical_bytes({"status": "PASS"}))
            self.assertEqual(channel.request(S, "none", 42), {"status": "PASS"})
            channel.close()
            self.assertEqual(json.loads(broker.recv(1024))["host_pid"], 42)
            self.assertEqual(broker.recv(1024), b"")
        finally:
            broker.close()
        stream, other = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            with self.assertRaisesRegex(ValueError, "sequenced-packet"):
                D.ProofChannel(os.dup(stream.fileno()))
        finally:
            stream.close()
            other.close()
        with tempfile.TemporaryFile() as handle:
            with self.assertRaisesRegex(ValueError, "not a socket"):
                D.ProofChannel(handle.fileno())

    def test_anchor_census_is_mode_exact(self):
        none = D.anchor_census(startup_log("none"), "none", PORT)
        k5 = D.anchor_census(startup_log("native-mtp-k5"), "native-mtp-k5", PORT)
        self.assertEqual((none["mtp"], k5["mtp"], k5["num_drafts"]), ("false", "true", 4))
        self.assertTrue(none["plain_scheduler_ready"] and k5["k5_scheduler_ready"])
        reordered = (b"INFO Listening on 0.0.0.0:8888\n" + b"INFO Atlas Spark starting...\n"
                     + startup_log("none").split(b"starting...\n", 1)[1].replace(
                         b"INFO spark_server: Listening on 0.0.0.0:8888\n", b""))
        with self.assertRaisesRegex(ValueError, "out of order"):
            D.anchor_census(reordered, "none", PORT)
        for text in (b"Xid 79", b"NVRM: nvAssertOkFailedNoLog", b"CUDA_ERROR_ILLEGAL_ADDRESS",
                     b"Assertion failed: x", b"invoked oom-killer"):
            with self.assertRaisesRegex(ValueError, "fatal"):
                D.anchor_census(startup_log("none", extra=text + b"\n"), "none", PORT)

    def test_model_identity_requires_exact_listing(self):
        good = S.canonical_bytes({"object": "list", "data": [
            {"id": "m", "object": "model", "created": 3, "owned_by": "atlas-spark"}]})
        health = S.canonical_bytes({"status": "ready", "model": "m"})
        self.assertTrue(D.model_identity(S, "m", 200, health, 200, good))
        for args in ((503, health, 200, good), (200, health, 500, good),
                     (200, S.canonical_bytes({"status": "loading"}), 200, good),
                     (200, health, 200, good.replace(b'"created":3', b'"created":"3"'))):
            with self.assertRaises(ValueError):
                D.model_identity(S, "m", *args)

    def test_checkpoint_comparison_uses_directory_identity_and_config_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            first, second = Path(temp, "a"), Path(temp, "b")
            for path, body in ((first, b'{"a":1}'), (second, b'{"a":1}')):
                path.mkdir()
                (path / "config.json").write_bytes(body)
            fds = [os.open(path, os.O_RDONLY | os.O_DIRECTORY) for path in (first, first, second)]
            try:
                same = D.compare_checkpoint(S, fds[0], fds[1], "fixture")
                self.assertEqual(same["config_sha256"], hashlib.sha256(b'{"a":1}').hexdigest())
                with self.assertRaisesRegex(ValueError, "bind source"):
                    D.compare_checkpoint(S, fds[0], fds[2], "fixture")
            finally:
                for fd in fds:
                    os.close(fd)

    def test_rootfs_check_reads_the_live_root_mount(self):
        rw = "1 0 0:1 / / rw,relatime - overlay overlay rw\n"
        ro = "1 0 0:1 / / ro,relatime - overlay overlay ro\n"
        args = types.SimpleNamespace(suite=S)
        with mock.patch.object(D.Path, "read_text", return_value=ro):
            self.assertTrue(D.rootfs_read_only(args, 7))
        with mock.patch.object(D.Path, "read_text", return_value=rw):
            with self.assertRaises(ValueError):
                D.rootfs_read_only(args, 7)

    def test_http_get_returns_the_exact_status_and_body(self):
        class Response:
            status = 503

            def read(self, limit):
                return b'{"status":"loading"}'

        class Connection:
            def __init__(self, host, port, timeout):
                self.target = (host, port, timeout)

            def request(self, method, route, headers):
                self.sent = (method, route)

            def getresponse(self):
                return Response()

            def close(self):
                pass

        args = types.SimpleNamespace(base_url="http://127.0.0.1:18888")
        with mock.patch.object(D.http.client, "HTTPConnection", Connection):
            self.assertEqual(D.http_get(args, "/health"), (503, b'{"status":"loading"}'))

    def test_engine_log_capture_contract(self):
        args, plan, recipe = object(), object(), object()
        proxy = D.EngineLog(71, args, plan, recipe)
        proxy.mode = "none"
        with mock.patch.object(D.subprocess, "Popen", return_value="fixture") as launch, \
                mock.patch.object(D, "startup_launch", return_value=["validated"]) as transform:
            self.assertEqual(proxy.Popen(["docker", "exec"]), "fixture")
            transform.assert_called_once_with(["docker", "exec"], args, plan, recipe, "none")
            self.assertEqual(launch.call_args.args, (["validated"],))
            self.assertEqual(launch.call_args.kwargs["stdout"], 71)
            with self.assertRaisesRegex(ValueError, "logging contract"):
                proxy.Popen([], stdout=1)

    def test_example_plan_entry_carries_every_driver_input_and_the_proof_token(self):
        example = json.loads((KIT / "examples/r15-container-suite-plan.example.json").read_bytes())
        argv = example["gate_jobs"][D.GATE]["argv"]
        self.assertEqual(argv[0], example["gate_jobs"][D.GATE]["program_path"])
        self.assertTrue(argv[0].endswith("repro-kit/r15-gates/startup-readiness"))
        self.assertEqual(argv.count(S.PROCESS_PROOF_TOKEN), 1)
        for token in S.IDENTITY_PLACEHOLDERS:
            self.assertEqual(argv.count(token), 1)
        for flag in ("--container-id", "--image-id", "--image-digest", "--base-url", "--run-id",
                     "--receipt-fd", "--raw-log-fd", "--runtime-dir", "--suite-program",
                     "--suite-sha256", "--lifecycle-program", "--process-proof-fd"):
            self.assertEqual(argv.count(flag), 1, flag)
        self.assertTrue(argv[argv.index("--lifecycle-program") + 1].endswith("r15-gates/mmlu-pro"))

    def test_validator_document_keys_match_the_outer_contract(self):
        self.assertEqual(D.GATE, "startup-readiness")
        self.assertIn(D.GATE, Q.REQUIRED_GATES)
        source = (KIT / "r15-gates/startup-readiness").read_text()
        for key in ("fresh_process_modes", "ready_http_statuses", "model_identity_match",
                    "checkpoint_identity_match", "rootfs_read_only", "plain_scheduler_ready",
                    "k5_scheduler_ready", "process_executable_sha256", "validator_artifact",
                    "validator_receipt_sha256"):
            self.assertIn('"' + key + '"', source)


if __name__ == "__main__":
    unittest.main()
