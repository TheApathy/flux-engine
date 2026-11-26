# SPDX-License-Identifier: AGPL-3.0-only
"""Offline memory gate tests. Docker, HTTP, kernel/proc and servers are mocked."""

from contextlib import contextmanager
import fcntl
import hashlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import types
import unittest
from unittest import mock

KIT = Path(__file__).resolve().parents[1]
# Fault and artifact-binding cases replay the whole run a dozen times; at the
# release contexts each replay parses ~28 MB of token lists, which exceeds the
# 90 MB address-space cap these suites run under. The rules they check do not
# depend on prompt length; the release CONTEXTS stay covered by the fixture
# test, which builds and validates every release-size request.
EXERCISE_CONTEXTS = (192, 320, 448)


def module(path, name):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    value = importlib.util.module_from_spec(spec)
    loader.exec_module(value)
    return value


D = module(KIT / "r15-gates/memory-floor", "memory_floor_driver_fixture")
S = module(KIT / "r15-container-suite.py", "memory_floor_suite_fixture")
Q = module(KIT / "r15_exact_image_release.py", "memory_floor_outer_fixture")
H = D.load_checked(
    KIT / "r15-gates/mmlu-pro", D.LIFECYCLE_SHA256, "memory_lifecycle_fixture"
)


class MemoryDriver(unittest.TestCase):
    def setUp(self):
        self.network = mock.patch.object(
            socket.socket,
            "connect",
            side_effect=AssertionError("real network is forbidden"),
        )
        self.network.start()
        self.addCleanup(self.network.stop)
        for name in ("run", "Popen"):
            guard = mock.patch.object(
                D.subprocess,
                name,
                side_effect=AssertionError("real subprocess is forbidden"),
            )
            guard.start()
            self.addCleanup(guard.stop)

    def response(self, body, index=0):
        return {
            "id": "memory-response-" + str(index),
            "model": body["model"],
            "choices": [
                {
                    "finish_reason": "length",
                    "token_ids": list(range(D.OUTPUT_TOKENS)),
                    "text": "fixture output",
                }
            ],
            "usage": {
                "prompt_tokens": len(body["prompt_token_ids"]),
                "completion_tokens": D.OUTPUT_TOKENS,
                "total_tokens": len(body["prompt_token_ids"]) + D.OUTPUT_TOKENS,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        }

    def test_lifecycle_literal_pin_matches_tracked_source(self):
        self.assertEqual(
            hashlib.sha256((KIT / "r15-gates/mmlu-pro").read_bytes()).hexdigest(),
            D.LIFECYCLE_SHA256,
        )

    def test_wrong_helper_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "helper.py"
            path.write_text("raise AssertionError('untrusted helper executed')\n")
            with self.assertRaisesRegex(ValueError, "source pin"):
                D.load_checked(path, D.LIFECYCLE_SHA256, "untrusted_memory_helper")

    def test_helper_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "helper.py"
            path.symlink_to(KIT / "r15-gates/mmlu-pro")
            with self.assertRaises(OSError):
                D.load_checked(path, D.LIFECYCLE_SHA256, "linked_memory_helper")

    def test_fixture_is_exact_source_pinned_and_avoids_shared_prefixes(self):
        prefixes = []
        for context in D.CONTEXTS:
            body = D.fixture(context, "test-model")
            self.assertEqual(len(body["prompt_token_ids"]), context - 64)
            self.assertTrue(body["ignore_eos"])
            self.assertEqual(body["max_tokens"], 64)
            self.assertEqual(body, D.fixture(context, "test-model"))
            prefixes.append(body["prompt_token_ids"][0])
            D.validate_response(types.SimpleNamespace(), body, self.response(body))
        self.assertEqual(len(set(prefixes)), len(D.CONTEXTS))

    def test_measured_geometry_cache_and_scheduler_token_failures_reject(self):
        body = D.fixture(8192, "test-model")
        mutations = [
            lambda row: row.update(model="foreign-model"),
            lambda row: row.update(id=""),
            lambda row: row["choices"][0].update(finish_reason="stop"),
            lambda row: row["choices"][0].pop("token_ids"),
            lambda row: row["choices"][0].update(token_ids=[True] * 64),
            lambda row: row["choices"][0].update(token_ids=list(range(63))),
            lambda row: row["usage"].update(prompt_tokens=8192),
            lambda row: row["usage"].update(completion_tokens=True),
            lambda row: row["usage"].update(total_tokens=8191),
            lambda row: row["usage"].pop("prompt_tokens_details"),
            lambda row: row["usage"]["prompt_tokens_details"].update(cached_tokens=1),
            lambda row: row["usage"]["prompt_tokens_details"].update(
                cached_tokens=False
            ),
        ]
        for mutation in mutations:
            value = self.response(body)
            mutation(value)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                D.validate_response(types.SimpleNamespace(), body, value)

    def test_metric_engagement_requires_numeric_integer_counter(self):
        self.assertEqual(
            D.verify_counter(
                'atlas_spec_decode_verify_total{k="5",route="native"} 3\n'
            ),
            3,
        )
        self.assertEqual(
            D.verify_counter('atlas_spec_decode_verify_total{k="dflash"} 50\n'), 0
        )
        for number in ("1e999", "1.2", "-1", "NaN", "broken"):
            with self.subTest(number=number), self.assertRaises(ValueError):
                D.verify_counter(
                    'atlas_spec_decode_verify_total{k="5"} ' + number + "\n"
                )

    def test_full_log_census_has_no_chunk_boundary_blind_spots(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "engine.log"
            with path.open("w+b") as handle:
                # Written in pieces: concatenating would hold a second
                # 4 MiB copy under the address-space cap.
                handle.write(b"\0" * (4 * D.MIB - 2))
                handle.write(b"NVRM\nXid 79\nallocator warning: pressure\n")
                handle.flush()
                counts = D.scan_log(handle.fileno())
            self.assertEqual(
                counts,
                {
                    "allocator_warning_count": 1,
                    "cuda_oom_count": 0,
                    "nvrm_count": 1,
                    "xid_count": 1,
                },
            )
            with path.open("w+b") as handle:
                handle.write(b"CUDA")
                handle.write(b"-" * (4 * D.MIB + 1))
                handle.write(b"out of memory\n")
                handle.flush()
                with self.assertRaisesRegex(ValueError, "fatal engine"):
                    D.scan_log(handle.fileno())

    def test_log_fatal_and_safety_false_positive_boundaries(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "engine.log"
            for raw in (
                b"cuda fatal",
                b"illegal memory access",
                b"CUDA_ERROR_OUT_OF_MEMORY",
                b"DriverError(CUDA_ERROR_OUT_OF_MEMORY)",
                b"cudaErrorMemoryAllocation",
                b"CUDA_ERROR_ILLEGAL_ADDRESS",
                b"cudaErrorIllegalAddress",
            ):
                with path.open("w+b") as handle:
                    handle.write(raw)
                    handle.flush()
                    with self.assertRaisesRegex(ValueError, "fatal engine"):
                        D.scan_log(handle.fileno())
            with path.open("w+b") as handle:
                handle.write(b"XidSuffix NVRMish allocation complete\n")
                handle.flush()
                self.assertFalse(any(D.scan_log(handle.fileno()).values()))

    def test_kernel_command_requires_complete_readable_journal(self):
        with tempfile.TemporaryFile() as target:
            for code, error in ((1, b""), (0, b"permission denied")):
                with mock.patch.object(
                    D.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess([], code, b"", error),
                ), self.assertRaisesRegex(ValueError, "journal is unavailable"):
                    D.kernel_command(None, ["--lines=1"], target.fileno())

    def test_engine_logs_are_captured_and_raw_fd_is_not_used(self):
        with mock.patch.object(D.subprocess, "Popen", return_value="fixture") as launch:
            self.assertEqual(
                D.EngineLog(71).Popen(["docker", "exec"], stdin=subprocess.DEVNULL),
                "fixture",
            )
            self.assertEqual(launch.call_args.kwargs["stdout"], 71)
            self.assertEqual(launch.call_args.kwargs["stderr"], 71)
            with self.assertRaisesRegex(ValueError, "logging contract"):
                D.EngineLog(71).Popen([], stdout=subprocess.DEVNULL)

    def exercise(
        self,
        *,
        mutation=None,
        kernel_message="clean kernel fixture",
        cleanup_error=False,
        missing_samples=False,
        baseline_lost=False,
        low_memory=False,
        reused_identity=False,
        artifact_rewrite=None,
        contexts=EXERCISE_CONTEXTS,
    ):
        current = [None]
        mode_now = [None]
        sequence = [0]
        original_write = S.write_all
        original_assert = S.assert_path_identity
        forwarded = bytearray()
        rewrites = []

        class Source:
            def snapshot(self):
                return {
                    "available_mib": 6000 if low_memory else 10000,
                    "used_mib": 80000,
                }, {
                    "host": {"MemTotal": 90000 * D.MIB, "MemAvailable": 10000 * D.MIB},
                    "cgroup": {
                        "memory.current": 50000 * D.MIB,
                        "memory.max": None,
                        "memory.peak": 51000 * D.MIB,
                    },
                    "events": {"oom": 0, "oom_kill": 0},
                }

            def close(self):
                pass

        class OfflineSampler(D.Sampler):
            def start(self):
                current[0] = self
                self.sample()

            def stop(self):
                self.check()

        @contextmanager
        def server(args, mode, plan, recipe):
            mode_now[0] = mode
            current[0].sample()
            original_write(
                args.lifecycle.subprocess.descriptor, b"complete clean engine fixture\n"
            )
            proof = {
                "host_pid": 42 if mode == "none" or reused_identity else 43,
                "start_time_ticks": 500,
                "mode": mode,
            }
            yield "fixture-" + mode, proof, 90
            if cleanup_error:
                raise ValueError("server required forced cleanup")

        def request(args, method, route, body=None):
            if route == "/metrics":
                number = sequence[0] if mode_now[0] == "native-mtp-k5" else 0
                return (
                    'atlas_spec_decode_verify_total{k="5"} ' + str(number) + "\n"
                ).encode()
            if not missing_samples:
                current[0].sample()
            sequence[0] += 1
            value = self.response(body, sequence[0])
            if mutation:
                mutation(value)
            return S.canonical_bytes(value)

        def kernel(args, arguments, target):
            initial = {
                "__CURSOR": "start-cursor",
                "_BOOT_ID": "boot-id",
                "MESSAGE": "prior Xid must be excluded",
            }
            if arguments == ["--lines=1"]:
                rows = [initial]
            else:
                rows = [
                    dict(initial, __CURSOR="lost" if baseline_lost else "start-cursor"),
                    {
                        "__CURSOR": "end-cursor",
                        "_BOOT_ID": "boot-id",
                        "MESSAGE": kernel_message,
                    },
                ]
            original_write(
                target, b"".join(S.canonical_bytes(row) + b"\n" for row in rows)
            )

        with tempfile.TemporaryDirectory() as temp:
            runtime = Path(temp)
            receipt_path = runtime / "receipt.json"
            descriptor = os.open(
                receipt_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600
            )
            raw_path = runtime / "raw.log"
            raw_path.write_bytes(b"previous gate fixture\n")
            raw_fd = os.open(raw_path, os.O_RDONLY)
            helper = types.SimpleNamespace(
                make_directory=H.make_directory,
                inspect_container=lambda *_: (self.plan, {}),
                server=server,
                request=request,
                subprocess=subprocess,
            )
            args = types.SimpleNamespace(
                suite=S,
                lifecycle=helper,
                runtime_dir=str(runtime),
                receipt_fd=descriptor,
                raw_log_fd=raw_fd,
                container_id="c" * 64,
                image_id="sha256:" + "a" * 64,
                image_digest="sha256:" + "b" * 64,
                run_id="offline-memory-fixture",
            )

            def write(target, raw):
                if target == 2:
                    forwarded.extend(raw)
                elif target == raw_fd:
                    raise AssertionError("attempted writing read-only raw log")
                else:
                    original_write(target, raw)

            def assert_identity(root, locator, target, label):
                if locator == "memory/" + str(artifact_rewrite):
                    raw = os.pread(target, os.fstat(target).st_size, 0)
                    if locator.endswith(".jsonl"):
                        rows = [json.loads(line) for line in raw.splitlines()]
                        if "kernel-" in locator:
                            rows[-1]["MESSAGE"] = "NVRM Xid 79"
                        elif locator.endswith("samples.jsonl"):
                            rows[-1]["available_mib"] = 1
                        else:
                            rows[-1]["kind"] = "changed-after-validation"
                        raw = b"".join(S.canonical_bytes(row) + b"\n" for row in rows)
                    else:
                        raw += b"NVRM Xid 79\n"
                    os.ftruncate(target, 0)
                    os.pwrite(target, raw, 0)
                    rewrites.append(locator)
                original_assert(root, locator, target, label)

            self.plan = {
                "in_container_binary_path": "/opt/flux/bin/server",
                "in_container_recipe_path": "/opt/flux/repro-kit/recipes/fixture.env",
                "in_container_binary_sha256": "a" * 64,
                "recipe_sha256": "b" * 64,
                "checkpoint_id": "JMNI-Labs/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid",
                "checkpoint_revision": "d" * 40,
            }
            error = None
            try:
                with mock.patch.object(
                    D, "CONTEXTS", contexts
                ), mock.patch.object(
                    D, "parent_plan", return_value=(self.plan, "e" * 64)
                ), mock.patch.object(
                    D, "MemorySource", return_value=Source()
                ), mock.patch.object(
                    D, "Sampler", OfflineSampler
                ), mock.patch.object(
                    D, "kernel_command", side_effect=kernel
                ), mock.patch.object(
                    D, "tokenizer_pin"
                ), mock.patch.object(
                    D.select, "select", return_value=([], [], [])
                ), mock.patch.object(
                    S, "write_all", side_effect=write
                ), mock.patch.object(
                    S, "assert_path_identity", side_effect=assert_identity
                ):
                    try:
                        result = D.run(args)
                    except ValueError as caught:
                        error = str(caught)
                        result = None
                receipt_bytes = os.pread(descriptor, 1 << 20, 0)
                documents = {}
                for path in (runtime / "memory").iterdir():
                    self.assertEqual(path.stat().st_mode & 0o777, 0o400)
                    if path.name.endswith(".json"):
                        documents[path.name] = json.loads(path.read_bytes())
                if result is not None:
                    manifest = {
                        "_runtime_dir": str(runtime),
                        "launch": {"run_id": args.run_id},
                    }
                    self.assertEqual(
                        Q._validate_memory_evidence(
                            result["evidence"], manifest, args.container_id
                        ),
                        result["evidence"],
                    )
                if artifact_rewrite is not None:
                    self.assertEqual(rewrites, ["memory/" + artifact_rewrite])
                return result, error, receipt_bytes, documents, bytes(forwarded)
            finally:
                os.close(descriptor)
                os.close(raw_fd)

    def test_complete_mocked_run_replays_exact_outer_contract(self):
        receipt, error, raw, documents, forwarded = self.exercise()
        self.assertIsNone(error)
        self.assertEqual(json.loads(raw), receipt)
        self.assertEqual(receipt["gate"], "memory-floor")
        self.assertGreaterEqual(receipt["evidence"]["sample_count"], 3)
        self.assertEqual(receipt["evidence"]["minimum_available_mib"], 10000)
        validator = documents["validator.json"]
        self.assertEqual(len(validator["raw"]["workloads"]), 6)
        # The run iterates CONTEXTS (scaled here), once per mode.
        self.assertEqual(
            [row["context_tokens"] for row in validator["raw"]["workloads"]],
            list(EXERCISE_CONTEXTS) * 2,
        )
        self.assertEqual(validator["raw"]["safety_floor_mib"], 6656)
        self.assertEqual(
            set(validator["raw"]["samples"][0]),
            {
                "available_mib",
                "used_mib",
                "allocator_warnings",
                "cuda_oom_count",
                "nvrm_count",
                "xid_count",
            },
        )
        self.assertIn(b"complete clean engine fixture", forwarded)
        self.assertIn(b"clean kernel fixture", forwarded)
        self.assertNotIn(b"prior Xid", forwarded)

    def test_faults_never_emit_pass_receipt(self):
        cases = [
            {"kernel_message": "NVRM: Xid 79"},
            {"kernel_message": "allocator warning: failed allocation"},
            {"kernel_message": "CUDA out of memory"},
            {"kernel_message": "Out of memory: Killed process 4321 (worker)"},
            {
                "kernel_message": "oom-kill:constraint=CONSTRAINT_NONE,task=worker,pid=4321"
            },
            {"cleanup_error": True},
            {"missing_samples": True},
            {"baseline_lost": True},
            {"low_memory": True},
            {"reused_identity": True},
            {
                "mutation": lambda row: row["usage"]["prompt_tokens_details"].update(
                    cached_tokens=1
                )
            },
        ]
        # The scaled contexts pass when nothing is injected, so every case
        # below fails for its own fault, not for the shorter workload.
        receipt, error, _raw, documents, _forwarded = self.exercise()
        self.assertIsNone(error)
        self.assertIsNotNone(receipt)
        self.assertIn("validator.json", documents)
        for case in cases:
            with self.subTest(case=case):
                receipt, error, raw, documents, _forwarded = self.exercise(**case)
                self.assertIsNone(receipt)
                self.assertIsNotNone(error)
                self.assertFalse(raw)
                self.assertNotIn("validator.json", documents)

    def test_artifact_hashes_cannot_bind_bytes_other_than_the_validated_workload(self):
        for artifact in (
            "kernel-start.jsonl",
            "kernel-full.jsonl",
            "kernel-measured.log",
            "engine.log",
            "samples.jsonl",
            "requests.jsonl",
        ):
            with self.subTest(artifact=artifact):
                receipt, error, raw, documents, _ = self.exercise(
                    artifact_rewrite=artifact
                )
                self.assertIsNone(receipt)
                self.assertIn("bytes consumed by validation", error)
                self.assertFalse(raw)
                self.assertNotIn("validator.json", documents)

    def test_full_log_snapshot_binds_the_exact_scanned_bytes(self):
        with tempfile.NamedTemporaryFile() as target:
            raw = b"complete clean log\n"
            target.write(raw)
            target.flush()
            counts, snapshot = D.scan_log_snapshot(target.fileno())
            self.assertFalse(any(counts.values()))
            self.assertEqual(snapshot, {"bytes": len(raw), "sha256": D.digest(raw)})

    def source(self, changes=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        descriptor = os.open(temporary.name, os.O_RDONLY | os.O_DIRECTORY)
        cid = "c" * 64
        counters = {
            "/proc/42/cgroup": "0::/system.slice/docker-" + cid + ".scope\n",
            "/proc/meminfo": "MemTotal: 100000000 kB\nMemAvailable: 20000000 kB\n",
            "memory.current": str(50 * 1024**3),
            "memory.peak": str(51 * 1024**3),
            "memory.max": "max",
            "memory.events": "low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n",
        }
        args = types.SimpleNamespace(
            container_id=cid,
            image_id="sha256:" + "a" * 64,
            lifecycle=types.SimpleNamespace(
                docker=lambda _: S.canonical_bytes(
                    {
                        "Id": cid,
                        "Image": "sha256:" + "a" * 64,
                        "State": {"Running": True, "Pid": 42},
                    }
                )
            ),
            suite=types.SimpleNamespace(
                strict_json_loads=S.strict_json_loads,
                exact_int=S.exact_int,
                normalized_absolute=S.normalized_absolute,
                open_directory_chain=lambda *_: os.dup(descriptor),
                _process_start_ticks=lambda _: 90,
            ),
        )
        patcher = mock.patch.object(
            D, "read_small", side_effect=lambda path, **kwargs: counters[str(path)]
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        source = D.MemorySource(args)
        self.addCleanup(source.close)
        self.addCleanup(os.close, descriptor)
        if changes:
            counters.update(changes)
        return source, counters, os.fstat(descriptor)

    def test_host_and_container_memory_are_actually_sampled(self):
        source, _counters, info = self.source()
        with mock.patch.object(D.Path, "stat", return_value=info):
            measured, detail = source.snapshot()
        self.assertEqual(measured["available_mib"], 20000000 // 1024)
        self.assertEqual(detail["cgroup"]["memory.current"], 50 * 1024**3)

    def test_finite_cgroup_remaining_headroom_is_enforced(self):
        source, _counters, info = self.source({"memory.max": str(54 * 1024**3)})
        with mock.patch.object(D.Path, "stat", return_value=info):
            measured, _detail = source.snapshot()
        self.assertEqual(measured["available_mib"], 4096)

    def test_unlimited_leaf_cannot_hide_constrained_ancestor(self):
        source, counters, info = self.source()
        self.assertGreater(len(source.groups), 1)
        parent = source.groups[-1]["descriptor"]

        def read(path, *, dir_fd=None):
            if dir_fd == parent and path == "memory.max":
                return str(54 * 1024**3)
            return counters[str(path)]

        with mock.patch.object(D.Path, "stat", return_value=info), mock.patch.object(
            D, "read_small", side_effect=read
        ):
            measured, detail = source.snapshot()
        self.assertEqual(detail["cgroup"]["memory.max"], None)
        self.assertEqual(measured["available_mib"], 4096)

    def test_background_sampler_errors_are_propagated(self):
        source = types.SimpleNamespace(
            snapshot=mock.Mock(
                side_effect=[
                    ({"available_mib": 10000, "used_mib": 80000}, {}),
                    OSError("counter unavailable"),
                ]
            )
        )
        with tempfile.TemporaryFile() as journal, mock.patch.object(
            D, "INTERVAL_SECONDS", 0.001
        ):
            sampler = D.Sampler(
                types.SimpleNamespace(suite=S), source, journal.fileno()
            )
            sampler.start()
            self.assertTrue(sampler.stop_event.wait(1))
            with self.assertRaisesRegex(OSError, "counter unavailable"):
                sampler.stop()

    def test_sample_count_is_bounded_without_discarding_earlier_samples(self):
        source = types.SimpleNamespace(
            snapshot=lambda: ({"available_mib": 10000, "used_mib": 80000}, {})
        )
        with tempfile.TemporaryFile() as journal, mock.patch.object(
            D, "MAX_SAMPLES", 2
        ):
            sampler = D.Sampler(
                types.SimpleNamespace(suite=S), source, journal.fileno()
            )
            sampler.sample()
            sampler.sample()
            with self.assertRaisesRegex(ValueError, "bound exceeded"):
                sampler.sample()
            self.assertEqual(len(sampler.samples), 2)

    def test_oom_events_malformed_counters_and_identity_drift_fail_closed(self):
        changes = [
            {"memory.events": "low 0\nhigh 0\nmax 0\noom 1\noom_kill 0\n"},
            {"memory.current": "invalid"},
            {"memory.peak": "1"},
            {"/proc/meminfo": "MemTotal: 1 kB\nMemAvailable: 2 kB\n"},
            {"/proc/42/cgroup": "0::/foreign-container\n"},
        ]
        for change in changes:
            with self.subTest(change=change):
                source, _counters, info = self.source(change)
                with mock.patch.object(
                    D.Path, "stat", return_value=info
                ), self.assertRaises(ValueError):
                    source.snapshot()

    def test_actual_sealed_parent_plan_binds_runtime_and_identities(self):
        args = types.SimpleNamespace(
            suite=S,
            runtime_dir="/tmp/offline-memory",
            container_id="c" * 64,
            image_id="sha256:" + "a" * 64,
            image_digest="sha256:" + "b" * 64,
            base_url="http://127.0.0.1:8888",
            run_id="human-memory-run",
        )
        plan = {
            "schema": S.PLAN_SCHEMA,
            "runtime_dir": args.runtime_dir,
            "checkpoint_id": "JMNI-Labs/fixture",
            "checkpoint_revision": "d" * 40,
        }
        raw = S.canonical_bytes(plan)
        descriptor = os.memfd_create("offline-memory-plan", os.MFD_ALLOW_SEALING)
        self.addCleanup(os.close, descriptor)
        os.write(descriptor, raw)
        argv = [b"python", b"/proc/self/fd/19", b"--plan-fd", b"22"]
        for name in ("container-id", "image-id", "image-digest", "base-url", "run-id"):
            argv.extend(
                (("--" + name).encode(), getattr(args, name.replace("-", "_")).encode())
            )
        command_line = b"\0".join(argv) + b"\0"
        real_open = os.open

        def open_fixture(path, flags, **kwargs):
            self.assertEqual(path, "/proc/777/fd/22")
            return os.dup(descriptor)

        with mock.patch.object(D.os, "getppid", return_value=777), mock.patch.object(
            D.Path, "read_bytes", return_value=command_line
        ), mock.patch.object(D.os, "open", side_effect=open_fixture):
            with self.assertRaisesRegex(ValueError, "not sealed"):
                D.parent_plan(args)
            seals = (
                fcntl.F_SEAL_SEAL
                | fcntl.F_SEAL_SHRINK
                | fcntl.F_SEAL_GROW
                | fcntl.F_SEAL_WRITE
            )
            fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, seals)
            self.assertEqual(
                D.parent_plan(args), (plan, hashlib.sha256(raw).hexdigest())
            )
            args.run_id = "foreign-run"
            with self.assertRaisesRegex(ValueError, "launch identity"):
                D.parent_plan(args)
            args.run_id = "human-memory-run"
            args.runtime_dir = "/tmp/foreign-runtime"
            with self.assertRaisesRegex(ValueError, "runtime plan"):
                D.parent_plan(args)
        self.assertIs(os.open, real_open)

    def test_parser_binds_helper_to_parent_and_rejects_writable_raw_fd(self):
        with tempfile.TemporaryDirectory() as temp:
            receipt = os.open(
                Path(temp) / "receipt", os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600
            )
            raw_path = Path(temp) / "raw"
            raw_path.write_bytes(b"raw fixture\n")
            raw = os.open(raw_path, os.O_RDONLY)
            writable = os.open(raw_path, os.O_RDWR)
            self.addCleanup(os.close, receipt)
            self.addCleanup(os.close, raw)
            self.addCleanup(os.close, writable)
            os.set_inheritable(receipt, True)
            values = {
                "container-id": "c" * 64,
                "image-id": "sha256:" + "a" * 64,
                "image-digest": "sha256:" + "b" * 64,
                "base-url": "http://127.0.0.1:8888",
                "run-id": "human-memory-run",
                "receipt-fd": str(receipt),
                "raw-log-fd": str(raw),
                "runtime-dir": temp,
                "suite-program": str(KIT / "r15-container-suite.py"),
                "suite-sha256": "d" * 64,
                "lifecycle-program": str(KIT / "r15-gates/mmlu-pro"),
            }
            parent_check = mock.Mock()
            helper = types.SimpleNamespace(
                verify_parent_suite=parent_check, load_module=lambda *_: S
            )

            def argv():
                return [
                    item
                    for name, value in values.items()
                    for item in ("--" + name, value)
                ]

            with mock.patch.object(D, "load_checked", return_value=helper) as load:
                self.assertEqual(D.parse_args(argv()).run_id, "human-memory-run")
                self.assertEqual(load.call_args.args[1], D.LIFECYCLE_SHA256)
                parent_check.assert_called_with("d" * 64)
                values["raw-log-fd"] = str(writable)
                with self.assertRaisesRegex(ValueError, "read-only"):
                    D.parse_args(argv())
                values["raw-log-fd"] = str(raw)
                values["run-id"] = "bad/unsafe"
                with self.assertRaisesRegex(ValueError, "unsafe"):
                    D.parse_args(argv())


if __name__ == "__main__":
    unittest.main()
