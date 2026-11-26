# SPDX-License-Identifier: AGPL-3.0-only
"""Offline-only tests for the live native-concurrency C2-C8 gate driver.

Every server, Docker, HTTP, and process binding is mocked. A fake engine
reproduces the probe's receipt grammar (frames, logits rows, state receipts,
cancellation, slot reuse) with hostile knobs for each failure class. The real
probe validators and the real matrix sealer run against the driver's output.
"""

from contextlib import ExitStack, contextmanager
import concurrent.futures
import hashlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent


def module(path, name):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    loader.exec_module(value)
    return value


S = module(ROOT / "r15-container-suite.py", "r16_concurrency_suite_fixture")
Q = module(ROOT / "r15_exact_image_release.py", "r16_concurrency_outer_fixture")
D = module(
    ROOT / "r15-gates" / "native-concurrency" / "native-concurrency-c2-c8",
    "r16_native_concurrency_driver",
)
PROBE_PATH = ROOT / "qwen4_native_concurrency_probe.py"
MATRIX_PATH = ROOT / "qwen4_native_concurrency_matrix.py"
R9_PATH = ROOT / "r9_source_to_binary.py"
TARGET_STATE_PATH = Path("crates/spark-model/src/model/qwen4_target_state_receipt.rs")
LEGACY_TARGET_STATE_EMITTER_SNIPPET = """
fn classify_legacy(proposer: Option<()>, force_serial: bool) -> Result<()> {
    match (proposer, force_serial) {
        (None, true) => bail!("target-state receipt rejects a disabled control without a proposer"),
        _ => Ok(()),
    }
}
"""
sys.path.insert(0, str(ROOT))
P = module(PROBE_PATH, "qwen4_native_concurrency_probe")


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


ADMIT_GRACE_S = 0.04
MAX_TOKENS = 16
CANCEL_TOKEN = 5


class FakeEngine:
    """Lockstep decode simulator emitting the engine's qualification receipts."""

    def __init__(self, log_path, *, profile, mode, width, argv, environment, knobs):
        self.log = open(log_path, "ab", buffering=0)
        self.profile = profile
        self.mode = mode
        self.width = width
        self.argv = argv
        self.environment = environment
        self.knobs = knobs
        self.route = "serial" if profile == "serial-control" else "native"
        self.condition = threading.Condition()
        self.live = []
        self.free_slots = set()
        self.next_slot = 0
        self.engaged = False
        self.last_admit = time.monotonic()
        self.requests = 0
        self.pending_admissions = []
        self.replacement_prompt = None
        self.waiting_for_replacement = False

    def close(self):
        self.log.close()

    def write(self, line):
        self.log.write((line + "\n").encode())

    @staticmethod
    def prompt_tokens(prompt):
        return 20 + len(prompt) % 7

    @staticmethod
    def token(prompt, index):
        return (int(sha(prompt)[:8], 16) + index) % 50_000

    def plan_admissions(self, cases, replacement=None):
        with self.condition:
            if self.live or self.pending_admissions or self.replacement_prompt:
                raise AssertionError("fixture admission plan overlaps a live wave")
            self.pending_admissions = [case.content for case in cases]
            self.replacement_prompt = None if replacement is None else replacement.content

    def admit(self, prompt):
        with self.condition:
            # Relative sleeps in the real clients do not order Python threads.
            # Model their requested admission order before assigning live slots.
            if not self.condition.wait_for(
                lambda: not self.pending_admissions or prompt == self.pending_admissions[0],
                timeout=5.0,
            ):
                raise AssertionError("fixture timed out waiting for planned admission")
            if self.pending_admissions:
                self.pending_admissions.pop(0)
            if prompt == self.replacement_prompt:
                self.replacement_prompt = None
                self.waiting_for_replacement = False
            if self.free_slots and not self.knobs.get("replacement_wrong_slot"):
                slot = min(self.free_slots)
                self.free_slots.discard(slot)
            else:
                slot = self.next_slot
                self.next_slot += 1
            if self.knobs.get("duplicate_slot") and self.live:
                slot = self.live[0]["slot"]
            session = {
                "id": self.requests,
                "hash": sha(prompt)[:16],
                "prompt": prompt,
                "slot": slot,
                "produced": 1,
                "fed_tokens": 0,
                "cancelled": False,
                "waiting": False,
                "done": False,
                "concurrent": bool(self.live),
            }
            self.requests += 1
            self.live.append(session)
            self.last_admit = time.monotonic()
            self.condition.notify_all()
            return session

    def state_triple(self):
        if self.knobs.get("target_state_emitter_source"):
            source = self.knobs["target_state_emitter_source"]
            loaded = "--speculative" in self.argv
            disabled = self.environment.get("ATLAS_DISABLE_SPECULATION") == "1"
            if not loaded and disabled:
                if '(None, _) => ("unloaded-target-only", "ordinary-target-only", false)' in source:
                    return D.PLAIN_STATE
                if '(None, true) => bail!("target-state receipt rejects a disabled control without a proposer")' in source:
                    return None
                raise AssertionError("unrecognized source emitter policy in offline fixture")
            if loaded and disabled:
                return ("loaded-serial", "ordinary-target-only", True)
            return D.K5_STATE if loaded else D.PLAIN_STATE
        if self.knobs.get("k5_state_plain"):
            return D.PLAIN_STATE
        return D.K5_STATE if self.mode == "native-mtp-k5" else D.PLAIN_STATE

    def retire(self, session):
        with self.condition:
            if session["done"]:
                return
            session["done"] = True
            self.live.remove(session)
            self.free_slots.add(session["slot"])
            if session["cancelled"] and self.replacement_prompt is not None:
                self.waiting_for_replacement = True
            produced = session["produced"]
            seq_len = self.prompt_tokens(session["prompt"]) + session["fed_tokens"]
            if not session["cancelled"] and self.knobs.get("normal_terminal_position_offset"):
                seq_len += 1
            triple = self.state_triple()
            if triple is None:
                self.write(
                    "QWEN4_NATIVE_MTP_REQUEST_RECEIPT_FAILED "
                    "detail=target-state receipt rejects a disabled control without a proposer"
                )
            elif not self.knobs.get("missing_state"):
                mode, path, loaded = triple
                line = (
                    "QWEN4_NATIVE_MTP_REQUEST_RECEIPT schema=1 "
                    f"mode={mode} scheduler_path={path} "
                    f"proposer_loaded={'true' if loaded else 'false'} "
                    f"session_hash={session['hash']} seq_len={seq_len} "
                    f"token_count={seq_len} output_tokens={produced} "
                    "gdn_layers=36 attention_layers=12 logical_kv_blocks=12 "
                    f"kv_token_positions={seq_len * 12} compared_buffers=49 "
                    "compared_bytes=123456 "
                    "state_scope=live-gdn-ple-plus-logical-bf16-kv "
                    "checkpoints_excluded=true "
                    f"fed_tokens_sha256={sha('fed:' + session['prompt'] + str(session['fed_tokens']))} "
                    f"target_state_sha256={sha('state:' + session['prompt'] + str(session['fed_tokens']))}"
                )
                if (
                    self.knobs.get("mid_cancel_state_mutation")
                    and self.knobs.get("mid_cancel_mutation_mode", self.mode) == self.mode
                    and self.width == 8
                    and session["id"] >= 5 * self.width + 3
                    and produced < session["max_tokens"]
                ):
                    for key, value in self.knobs["mid_cancel_state_mutation"].items():
                        line = re.sub(r"(?<!\S)" + re.escape(key) + r"=[^ ]+", f"{key}={value}", line)
                self.write(line)
            self.condition.notify_all()

    def logits_line(self, session, index):
        if self.knobs.get("wave_logits_drift") and len(self.live) >= 2 and index == 3:
            payload = "drift"
        else:
            payload = f"logit:{session['prompt']}:{index}"
        return (
            f"LOGITS_FNV session={session['hash']} idx={index} "
            f"fnv={sha('fnv' + payload)[:16]} sha256={sha(payload)}"
        )

    def step(self):
        rows = list(self.live)
        if len(rows) >= 2:
            route = self.route
            if self.knobs.get("collapse_to_serial"):
                route = "serial"
            sessions = ",".join(s["hash"] for s in rows)
            slots = ",".join(str(s["slot"]) for s in rows)
            lengths = ",".join(
                str(
                    self.prompt_tokens(s["prompt"]) + s["fed_tokens"]
                    + (1 if self.width >= 9 and self.knobs.get("ladder_position_offset") else 0)
                )
                for s in rows
            )
            if route == "native":
                self.write(
                    f"QWEN4_NATIVE_BATCH_FRAME rows={len(rows)} sessions={sessions} "
                    f"slots={slots} pre_step_seq_lens={lengths} "
                    "active_only=true target_only=true qsa=false graph=false"
                )
                if not self.engaged:
                    self.engaged = True
                    engaged_rows = self.knobs.get("engaged_rows_override", len(rows))
                    self.write(
                        f"QWEN4_NATIVE_BATCH_ENGAGED rows={engaged_rows} "
                        "active_only=true target_only=true qsa=false"
                    )
                    if self.profile == "native-optimized":
                        for line in (
                            f"ENGAGED ATLAS_ATTN_QKV_EXACT_STRIDED: multi_seq M={len(rows)}",
                            f"ENGAGED ATLAS_ATTN_QK_NORM_BATCHED: multi_seq M={len(rows)}",
                            f"ENGAGED ATLAS_ATTN_GATE_BATCHED: multi_seq M={len(rows)} dim=2560 stride=10240",
                            f"ENGAGED ATLAS_ATTN_KV_WRITE_BATCHED: multi_seq M={len(rows)}",
                            f"QWEN4_NATIVE_SSM_PTR_CACHE_ENGAGED rows={len(rows)} table=stable h2d=elided",
                        ):
                            self.write(line)
            else:
                selector = "true" if self.profile != "serial-control" else "false"
                self.write(
                    f"QWEN4_SERIAL_BATCH_FRAME rows={len(rows)} sessions={sessions} "
                    f"slots={slots} pre_step_seq_lens={lengths} "
                    f"selector_on={selector} qsa_required=false graph=false"
                )
        for session in rows:
            self.write(self.logits_line(session, session["produced"]))
        for session in rows:
            session["fed_tokens"] += 1
            if session["cancelled"]:
                self.retire(session)
            else:
                session["produced"] += 1
                session["waiting"] = False

    def step_ready(self):
        return (
            not self.pending_admissions
            and not self.waiting_for_replacement
            and all(session["waiting"] for session in self.live)
            and time.monotonic() - self.last_admit >= ADMIT_GRACE_S
        )

    def cancel(self, session):
        with self.condition:
            if session["done"]:
                return
            # The scheduler computes/logs the next target row before checking
            # cancellation, but never emits that row to the cancelled stream.
            session["cancelled"] = True
            session["waiting"] = True
            while not session["done"]:
                if self.step_ready():
                    self.step()
                    self.condition.notify_all()
                else:
                    self.condition.wait(timeout=0.005)

    def next_token(self, session):
        with self.condition:
            target = session["produced"]
            session["waiting"] = True
            while session["produced"] == target:
                if self.step_ready():
                    self.step()
                    self.condition.notify_all()
                else:
                    self.condition.wait(timeout=0.005)
            return self.token(session["prompt"], target)

    def stream(self, body):
        prompt = body["messages"][0]["content"]
        max_tokens = body["max_tokens"]
        session = self.admit(prompt)
        session["max_tokens"] = max_tokens
        token_ids = [self.token(prompt, 0)]
        try:
            yield self.event({"choices": [{"index": 0, "delta": {"content": "t"}}]})
            while len(token_ids) < max_tokens:
                token_ids.append(self.next_token(session))
                yield self.event({"choices": [{"index": 0, "delta": {"content": "t"}}]})
            self.retire(session)
            if self.knobs.get("wave_token_drift") and session["concurrent"]:
                token_ids[4] ^= 1
            prompt_tokens = self.prompt_tokens(prompt)
            yield self.event(
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "length",
                            "token_ids": token_ids,
                        }
                    ]
                }
            )
            yield self.event(
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": len(token_ids),
                        "total_tokens": prompt_tokens + len(token_ids),
                        "prompt_tokens_details": {"cached_tokens": 0},
                        "completion_tokens_details": {"reasoning_tokens": 0},
                    },
                }
            )
            yield b"data: [DONE]\n"
        finally:
            self.cancel(session)

    @staticmethod
    def event(value):
        return b"data: " + json.dumps(value).encode() + b"\n"


class FakeResponse:
    status = 200

    def __init__(self, generator):
        self.generator = generator

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        if hasattr(self.generator, "close"):
            self.generator.close()
        return None

    def __iter__(self):
        return self.generator

    def read(self, *_args):
        return b"{}"


class HealthResponse(FakeResponse):
    def __init__(self):
        super().__init__(iter(()))


def provenance_fixture():
    public = {
        "release_version": P.RELEASE_VERSION,
        "source_revision": "1" * 40,
        "source_tree": "2" * 40,
        "source_bundle_manifest_name": "SOURCE_BUNDLE_MANIFEST.json",
        "source_bundle_manifest_sha256": "3" * 64,
        "source_bundle_name": "flux-engine-source-v0.9.13-r15.tar.gz",
        "source_bundle_sha256": "4" * 64,
        "build_receipt_name": "BUILD_RECEIPT.json",
        "build_receipt_sha256": "5" * 64,
        "source_to_binary_receipt_name": "SOURCE_TO_BINARY_RECEIPT.json",
        "source_to_binary_receipt_sha256": "6" * 64,
        "checkpoint_id": "hf://JMNI-Labs/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid",
        "checkpoint_revision": "7" * 40,
        "checkpoint_manifest_sha256": "8" * 64,
        "checkpoint_files_verified": 3,
        "checkpoint_bytes_verified": 300,
        "binary_name": "flux-server-real",
        "binary_sha256": "9" * 64,
    }
    return D.Provenance(public, Path("/nonexistent"), {}, Path("/x"), (), Path("/y"), (), {})


# Waves run one client thread per request (up to C8) beside the fake engine's
# threads. At the default 8 MiB stack each thread reserves, they do not fit
# under the 90 MB address-space cap these suites run under; the fixture
# workers need far less.
FIXTURE_THREAD_STACK_BYTES = 1024 * 1024


class NativeConcurrencyDriverTests(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        previous = threading.stack_size(FIXTURE_THREAD_STACK_BYTES)
        self.addCleanup(threading.stack_size, previous)

    def test_fixture_admission_plan_orders_reversed_worker_arrivals(self):
        cases = list(reversed(P.make_cases(4)))
        with tempfile.TemporaryDirectory() as temp:
            engine = FakeEngine(
                Path(temp) / "server.log", profile="native-reference", mode="none",
                width=len(cases), argv=[], environment={}, knobs={},
            )
            try:
                engine.plan_admissions(cases)
                entered = {case.label: threading.Event() for case in cases}

                def worker(case):
                    entered[case.label].set()
                    return engine.admit(case.content)

                with concurrent.futures.ThreadPoolExecutor(max_workers=len(cases)) as pool:
                    futures = []
                    for case in reversed(cases):
                        futures.append(pool.submit(worker, case))
                        self.assertTrue(entered[case.label].wait(timeout=5.0))
                    for future in futures:
                        future.result(timeout=5.0)
                self.assertEqual([row["prompt"] for row in engine.live], [case.content for case in cases])
                self.assertEqual([row["slot"] for row in engine.live], list(range(len(cases))))
                self.assertEqual(engine.pending_admissions, [])
            finally:
                engine.close()

    def test_fixture_records_target_fed_normal_and_cancelled_geometry(self):
        cases = P.make_cases(2)
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp) / "server.log"
            engine = FakeEngine(
                log, profile="native-reference", mode="none", width=2,
                argv=[], environment={}, knobs={},
            )
            try:
                engine.plan_admissions(cases)

                def worker(case):
                    with FakeResponse(engine.stream(P.request_body("fixture", case, 8))) as response:
                        content_chunks = 0
                        for line in response:
                            if b'"content": "t"' in line:
                                content_chunks += 1
                            if case == cases[-1] and content_chunks == 3:
                                break

                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    futures = [pool.submit(worker, case) for case in cases]
                    for future in futures:
                        future.result(timeout=5.0)
                parsed = P.parse_log(log.read_text())
                normal = parsed.state_by_session[sha(cases[0].content)[:16]]
                cancelled = parsed.state_by_session[sha(cases[1].content)[:16]]
                self.assertEqual(normal.output_tokens, 8)
                self.assertEqual(normal.seq_len, engine.prompt_tokens(cases[0].content) + 7)
                self.assertEqual(cancelled.output_tokens, 3)
                self.assertEqual(cancelled.seq_len, engine.prompt_tokens(cases[1].content) + 3)
                self.assertEqual(list(parsed.rows_by_session[normal.session]), list(range(1, 8)))
                self.assertEqual(list(parsed.rows_by_session[cancelled.session]), [1, 2, 3])
                prompts = {sha(case.content)[:16]: engine.prompt_tokens(case.content) for case in cases}
                for frame in parsed.frames:
                    for session, seq_len, row in zip(frame.sessions, frame.pre_step_seq_lens, frame.logits):
                        self.assertEqual(seq_len, prompts[session] + row.index - 1)
                last_target = [frame for frame in parsed.frames if cancelled.session in frame.sessions][-1]
                row = last_target.sessions.index(cancelled.session)
                self.assertEqual(last_target.logits[row].index, 3)
                self.assertEqual(cancelled.seq_len, last_target.pre_step_seq_lens[row] + 1)
            finally:
                engine.close()

    def plan(self):
        return {
            "in_container_binary_path": "/opt/flux/bin/flux-server-real",
            "in_container_recipe_path": "/opt/flux/repro-kit/recipes/flux-r15.env",
            "in_container_binary_sha256": "9" * 64,
            "recipe_sha256": "a" * 64,
        }

    def recipe(self):
        return {"RUST_LOG": "info,spark::scheduler::spec_accept_log=debug", "ATLAS_X": "1"}

    # --- unit: launch contract ---------------------------------------------

    def test_arm_environment_overlays_recipe_and_rejects_pinned_selectors(self):
        overrides, live = D.arm_environment(
            P, self.recipe(), {"FLUX_PORT": "8899", "PATH": "/bin"}, "none", "native-optimized"
        )
        self.assertEqual(overrides["ATLAS_QWEN4_NATIVE_BATCH"], "1")
        self.assertEqual(overrides["ATLAS_ATTN_GATE_BATCHED"], "1")
        self.assertEqual(overrides["ATLAS_DISABLE_SPECULATION"], "1")
        self.assertEqual(overrides["ATLAS_LOGITS_FNV"], "1")
        self.assertEqual(live["FLUX_PORT"], "8899")
        self.assertNotIn("PATH", live)
        self.assertEqual(live["RUST_LOG"], self.recipe()["RUST_LOG"])
        _overrides, k5 = D.arm_environment(P, self.recipe(), {}, "native-mtp-k5", "serial-control")
        self.assertEqual(k5["ATLAS_DISABLE_SPECULATION"], "0")
        self.assertEqual(k5["ATLAS_QWEN4_NATIVE_BATCH"], "0")
        with self.assertRaisesRegex(ValueError, "pins a qualification selector"):
            D.arm_environment(
                P, {**self.recipe(), "ATLAS_QWEN4_NATIVE_BATCH": "0"}, {}, "none", "native-reference"
            )
        with self.assertRaisesRegex(ValueError, "pins a qualification selector"):
            D.arm_environment(
                P, self.recipe(), {"ATLAS_LOGITS_FNV_REQUEST_LIMIT": "1"}, "none", "native-reference"
            )
        with self.assertRaisesRegex(ValueError, "RUST_LOG"):
            D.arm_environment(P, {"RUST_LOG": "warn"}, {}, "none", "native-reference")
        with self.assertRaisesRegex(ValueError, "RUST_LOG"):
            D.arm_environment(P, {"ATLAS_X": "1"}, {}, "none", "native-reference")

    def test_arm_argv_is_the_canonical_short_context_contract(self):
        args = types.SimpleNamespace(container_port=8888)
        argv = D.arm_argv(self.plan(), args, "flux-concurrency-abc", 8, "none")
        self.assertEqual(argv[0], "/opt/flux/bin/flux-server-real")
        for flag, value in (
            ("--model-from-path", "/models"),
            ("--max-batch-size", "8"),
            ("--max-num-seqs", "8"),
            ("--max-seq-len", "2048"),
            ("--max-prefill-tokens", "1024"),
            ("--kv-cache-dtype", "bf16"),
            ("--enable-prefix-caching", "false"),
            ("--kernel-target", "qwen3.8-flash-next"),
            ("--bind", "0.0.0.0"),
            ("--port", "8888"),
        ):
            self.assertEqual(P.argv_value(argv, flag), value)
        self.assertEqual(argv.count("--no-tui"), 1)
        self.assertEqual(argv.count("--qwen4-qsa"), 1)
        self.assertNotIn("--speculative", argv)
        k5 = D.arm_argv(self.plan(), args, "flux-concurrency-abc", 2, "native-mtp-k5")
        self.assertEqual(k5.count("--speculative"), 1)
        self.assertEqual(P.argv_value(k5, "--num-drafts"), "4")
        self.assertNotIn("--mtp-gate", k5)

    def test_launch_command_binds_stdout_and_stderr_to_the_results_log(self):
        args = types.SimpleNamespace(container_cwd="/", container_id="1" * 64)
        command = D.launch_command(
            args, self.plan(), {"B": "2", "A": "x y"}, ["/opt/flux/bin/flux-server-real", "serve"],
            "/results/concurrency/x.server.log",
        )
        self.assertEqual(command[:4], ["docker", "exec", "-w", "/"])
        self.assertEqual(command[5:7], ["/bin/sh", "-c"])
        self.assertTrue(command[7].startswith("exec /usr/bin/env 'A=x y' B=2 /opt/flux/bin/flux-server-real serve"))
        self.assertTrue(command[7].endswith('>>"$0" 2>&1'))
        self.assertEqual(command[8], "/results/concurrency/x.server.log")

    def test_find_pid_ignores_the_shell_wrapper_and_rejects_ambiguity(self):
        args = types.SimpleNamespace(container_id="1" * 64)
        rows = (
            "PID ARGS\n"
            "10 /bin/sh -c exec /usr/bin/env A=1 /opt/flux/bin/flux-server-real serve --model-name m1\n"
            "11 /opt/flux/bin/flux-server-real serve --model-name m1\n"
        )
        with mock.patch.object(D, "docker", return_value=rows.encode()):
            self.assertEqual(D.find_pid(args, self.plan(), "m1"), 11)
            self.assertIsNone(D.find_pid(args, self.plan(), "m2"))
        twice = rows + "12 /opt/flux/bin/flux-server-real serve --model-name m1\n"
        with mock.patch.object(D, "docker", return_value=twice.encode()):
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                D.find_pid(args, self.plan(), "m1")

    def test_bind_arm_process_requires_manifest_argv_cwd_environment_and_log(self):
        pid = os.getpid()
        manifest = {
            "argv": ["python3"],
            "cwd": os.readlink(f"/proc/{pid}/cwd"),
            "environment": D.live_environment_subset(dict(os.environ)),
        }
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp) / "server.log"
            log.write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "cmdline differs"):
                D.bind_arm_process(P, pid, manifest, log)
            real_argv = [
                item.decode() for item in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if item
            ]
            with self.assertRaisesRegex(Exception, "stdout and stderr"):
                D.bind_arm_process(P, pid, {**manifest, "argv": real_argv}, log)
            with self.assertRaisesRegex(ValueError, "cwd differs"):
                D.bind_arm_process(P, pid, {**manifest, "argv": real_argv, "cwd": "/nope"}, log)
            with self.assertRaisesRegex(ValueError, "environment differs"):
                D.bind_arm_process(
                    P, pid, {**manifest, "argv": real_argv, "environment": {"ATLAS_NOPE": "1"}}, log
                )

    def test_frame_census_detects_serial_collapse_and_native_leaks(self):
        text = "\n".join(
            [
                "x QWEN4_NATIVE_BATCH_FRAME rows=8 sessions=a slots=1 pre_step_seq_lens=1 active_only=true target_only=true qsa=false graph=false",
                "x QWEN4_NATIVE_BATCH_FRAME rows=7 sessions=a slots=1 pre_step_seq_lens=1 active_only=true target_only=true qsa=false graph=false",
                "x QWEN4_SERIAL_BATCH_FRAME rows=8 sessions=a slots=1 pre_step_seq_lens=1 selector_on=true qsa_required=false graph=false",
            ]
        )
        census = D.frame_census(text)
        self.assertEqual(census["native_width_executed"], 8)
        self.assertEqual(census["native_frames_by_rows"], {"7": 1, "8": 1})
        self.assertEqual(census["serial_frames_by_rows"], {"8": 1})
        with self.assertRaisesRegex(ValueError, "collapse to serial"):
            D.require_executed_width(census, "native", 8, "forward")
        with self.assertRaisesRegex(ValueError, "unexpectedly executed the native route"):
            D.require_executed_width(census, "serial", 8, "forward")
        native_only = D.frame_census(text.splitlines()[0])
        D.require_executed_width(native_only, "native", 8, "forward")
        with self.assertRaisesRegex(ValueError, "collapse to serial"):
            D.require_executed_width(native_only, "native", 4, "forward")
        D.require_executed_width(D.frame_census(""), "serial", 8, "forward")

    def test_token_divergence_reports_first_index_and_length_mismatch(self):
        self.assertIsNone(D.token_divergence([1, 2, 3], [1, 2, 3]))
        self.assertEqual(D.token_divergence([1, 2, 4], [1, 2, 3]), 2)
        self.assertEqual(D.token_divergence([1, 2], [1, 2, 3]), 2)

    def test_state_contract_enforces_engaged_native_k5_and_restores_probe(self):
        receipt = P.StateReceipt(
            session="a" * 16,
            schema=1,
            mode="engaged-native-mtp",
            scheduler_path="native-mtp",
            proposer_loaded=True,
            seq_len=119,
            token_count=119,
            output_tokens=20,
            gdn_layers=36,
            attention_layers=12,
            logical_kv_blocks=8,
            kv_token_positions=1428,
            compared_buffers=49,
            compared_bytes=1,
            fed_tokens_sha256="c" * 64,
            target_state_sha256="d" * 64,
            line_number=3,
        )
        original = P.validate_state_receipt
        with self.assertRaisesRegex(Exception, "not target-only"):
            P.validate_state_receipt("scope", receipt, 20, 100)
        with D.state_contract(P, "native-mtp-k5"):
            self.assertIsNot(P.validate_state_receipt, original)
            P.validate_state_receipt("scope", receipt, 20, 100)
            plain = P.StateReceipt(**{**receipt.__dict__, "mode": "unloaded-target-only"})
            with self.assertRaisesRegex(ValueError, "not an engaged native-K5"):
                P.validate_state_receipt("scope", plain, 20, 100)
            with self.assertRaisesRegex(Exception, "token census"):
                P.validate_state_receipt("scope", receipt, 21, 100)
            with self.assertRaisesRegex(ValueError, "not native-K5"):
                P.baseline_semantics([{"target_state": {"mode": "unloaded-target-only"}}])
        self.assertIs(P.validate_state_receipt, original)
        with D.state_contract(P, "none"):
            self.assertIs(P.validate_state_receipt, original)

    def test_timed_response_records_first_content_event(self):
        lines = [
            b'data: {"choices":[{"index":0,"delta":{"role":"assistant"}}]}\n',
            b'data: {"choices":[{"index":0,"delta":{"content":"a"}}]}\n',
            b"data: [DONE]\n",
        ]
        record = {"first_content_unix": None, "finished_unix": None}
        with D.TimedResponse(FakeResponse(iter(lines)), record) as response:
            seen = list(response)
        self.assertEqual(seen, lines)
        self.assertIsNotNone(record["first_content_unix"])
        self.assertIsNotNone(record["finished_unix"])

    def test_arm_limitations_match_the_matrix_sealer_contract_exactly(self):
        matrix_source = MATRIX_PATH.read_text()
        for item in D.arm_limitations(P, "none"):
            self.assertIn(json.dumps(item), matrix_source)
        self.assertEqual(len(D.arm_limitations(P, "none")), 6)
        self.assertEqual(len(D.arm_limitations(P, "native-mtp-k5")), 7)

    def test_load_launch_manifest_requires_exact_provenance_keys_and_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "launch.json"
            payload = S.canonical_bytes({"provenance": {"release_version": "x"}})
            path.write_bytes(payload)
            args = types.SimpleNamespace(launch_manifest=str(path), launch_manifest_sha256="1" * 64)
            with self.assertRaisesRegex(ValueError, "hash differs"):
                D.load_launch_manifest(args, S)
            args.launch_manifest_sha256 = sha(payload)
            with self.assertRaisesRegex(ValueError, "release provenance"):
                D.load_launch_manifest(args, S)

    def test_parse_widths_and_modes_validation(self):
        self.assertEqual(D.parse_widths("9-12"), (9, 10, 11, 12))
        for bad in ("8-12", "9-33", "12-9", "abc"):
            with self.assertRaises(ValueError):
                D.parse_widths(bad)

    def base_argv(self, temp):
        return [
            "--container-id", "1" * 64,
            "--image-id", "sha256:" + "2" * 64,
            "--image-digest", "sha256:" + "3" * 64,
            "--base-url", "http://127.0.0.1:8888",
            "--run-id", "offline",
            "--receipt-fd", "3",
            "--raw-log-fd", "4",
            "--runtime-dir", temp,
            "--suite-program", str(ROOT / "r15-container-suite.py"),
            "--suite-sha256", sha((ROOT / "r15-container-suite.py").read_bytes()),
            "--probe-program", str(PROBE_PATH),
            "--probe-sha256", sha(PROBE_PATH.read_bytes()),
            "--r9-program", str(R9_PATH),
            "--r9-sha256", sha(R9_PATH.read_bytes()),
            "--matrix-program", str(MATRIX_PATH),
            "--matrix-sha256", sha(MATRIX_PATH.read_bytes()),
            "--launch-manifest", str(Path(temp) / "launch.json"),
            "--launch-manifest-sha256", "5" * 64,
        ]

    def test_parse_args_pins_helpers_and_rejects_weakened_contracts(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(
            D, "verify_parent_suite"
        ) as parent:
            args = D.parse_args(self.base_argv(temp))
            parent.assert_called_once()
            self.assertEqual(args.modes, ("none", "native-mtp-k5"))
            self.assertEqual(args.probe.r9_provenance.__name__, "_r16_r9_native_concurrency")
            self.assertTrue(callable(args.suite.verify_live_server))
            self.assertEqual(args.ladder_widths, tuple(range(9, 33)))
            for extra, message in (
                (["--modes", "weird"], "modes"),
                (["--cancel-token", "64"], "cancel-token"),
                (["--max-tokens", "8"], "max-tokens"),
                (["--minimum-full-frames", "7"], "minimum-full-frames"),
                (["--ladder-profiles", "serial-control"], "ladder-profiles"),
                (["--container-cwd", "relative"], "absolute"),
            ):
                with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, message):
                    D.parse_args(self.base_argv(temp) + extra)
            wrong = self.base_argv(temp)
            wrong[wrong.index("--probe-sha256") + 1] = "7" * 64
            with self.assertRaisesRegex(ValueError, "dependency hash"):
                D.parse_args(wrong)
            renamed = self.base_argv(temp)
            renamed[renamed.index("--probe-program") + 1] = str(MATRIX_PATH)
            with self.assertRaisesRegex(ValueError, "basename"):
                D.parse_args(renamed)

    # --- unit: server lifecycle ---------------------------------------------

    def exercise_mocked_server(
        self, mode, profile, *, force_cleanup=False, bind_failure=False, replace_log=False
    ):
        read_fd, write_fd = os.pipe()
        dead = [False]
        process = mock.Mock()
        process.poll.return_value = None
        process.wait.side_effect = lambda **_kwargs: setattr(process.poll, "return_value", 0)
        proof = {"offline_test_only": True, "mode": mode}
        verifier = mock.Mock(return_value=proof)
        bound = {"pid": 123, "boot_id": "b", "start_ticks": 1, "log_device": 1, "log_inode": 2}

        def sent(_descriptor, signum):
            if signum == signal.SIGKILL or not force_cleanup:
                dead[0] = True

        with tempfile.TemporaryDirectory() as temp:
            runtime = Path(temp)
            D.make_directory(S, runtime, ["concurrency", D.GATE, "arms", mode])
            args = types.SimpleNamespace(
                container_id="1" * 64,
                container_port=8888,
                container_cwd="/",
                container_results_path="/results",
                base_url="http://127.0.0.1:8888",
                suite=types.SimpleNamespace(
                    verify_live_server=verifier,
                    canonical_bytes=S.canonical_bytes,
                    create_beneath=S.create_beneath,
                    write_all=S.write_all,
                    assert_path_identity=S.assert_path_identity,
                    MAX_GATE_RECEIPT_BYTES=S.MAX_GATE_RECEIPT_BYTES,
                    read_fd=S.read_fd,
                    append_log_frame=S.append_log_frame,
                    MAX_RAW_LOG_BYTES=S.MAX_RAW_LOG_BYTES,
                ),
                raw_log_fd=None,
                identities={"base_url": "http://127.0.0.1:8888"},
            )
            context = types.SimpleNamespace(
                probe=P,
                plan=self.plan(),
                recipe=self.recipe(),
                container_env={"FLUX_PORT": "8899"},
                model="flux-concurrency-abc",
                runtime=runtime,
                provenance=provenance_fixture(),
            )
            stem = f"concurrency/{D.GATE}/arms/{mode}/{profile}-c8"
            raw_fd = os.open(runtime / "raw.log", os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
            args.raw_log_fd = raw_fd
            try:
                with ExitStack() as stack:
                    launch = stack.enter_context(mock.patch.object(D.subprocess, "Popen", return_value=process))
                    stack.enter_context(mock.patch.object(D, "find_pid", return_value=123))
                    stack.enter_context(mock.patch.object(D, "claim_pid", side_effect=lambda *_a: os.dup(read_fd)))
                    endpoint = stack.enter_context(mock.patch.object(D, "request", return_value=b"{}"))
                    stack.enter_context(
                        mock.patch.object(
                            D,
                            "bind_arm_process",
                            side_effect=ValueError("live binding failed") if bind_failure else None,
                            return_value=bound,
                        )
                    )
                    stack.enter_context(mock.patch.object(D, "recheck_arm_process"))
                    stack.enter_context(
                        mock.patch.object(
                            D.select,
                            "select",
                            side_effect=lambda descriptors, _w, _e, _t: (descriptors if dead[0] else [], [], []),
                        )
                    )
                    kill = stack.enter_context(mock.patch.object(D.signal, "pidfd_send_signal", side_effect=sent))
                    if force_cleanup:
                        with self.assertRaises(D.GateFailure) as caught:
                            with D.server(args, context, mode, profile, 8, stem):
                                pass
                        self.assertEqual(caught.exception.failure_class, "VOID")
                        self.assertIn("forced cleanup", str(caught.exception))
                    elif bind_failure:
                        with self.assertRaisesRegex(ValueError, "live binding"):
                            with D.server(args, context, mode, profile, 8, stem):
                                pass
                    elif replace_log:
                        with self.assertRaisesRegex(ValueError, "pathname identity changed"):
                            with D.server(args, context, mode, profile, 8, stem) as live:
                                os.write(live.log_fd, b"owned original log\n")
                                live.host_log.rename(str(live.host_log) + ".moved")
                                replacement = S.create_beneath(
                                    runtime, live.log_locator, "replacement log", 0o600
                                )
                                os.close(replacement)
                        os.lseek(raw_fd, 0, os.SEEK_SET)
                        self.assertIn(b"owned original log\n", os.read(raw_fd, 4096))
                        with self.assertRaises(OSError):
                            os.fstat(live.log_fd)
                    else:
                        with D.server(args, context, mode, profile, 8, stem) as live:
                            self.assertEqual(live.proof, proof)
                            self.assertEqual(live.process, bound)
                            self.assertEqual(live.manifest["environment"]["FLUX_PORT"], "8899")
                            self.assertTrue((runtime / live.log_locator).exists())
                            self.assertEqual(live.manifest_name, f"{profile}-c8.manifest.json")
                        self.assertEqual(verifier.call_count, 2)
                        self.assertEqual(stat_mode(runtime / live.log_locator), 0o400)
                    self.assertEqual(endpoint.call_count, 2)
                    command = launch.call_args.args[0]
                    script = command[command.index("-c") + 1]
                    self.assertEqual(command[:2], ["docker", "exec"])
                    self.assertIn("ATLAS_DISABLE_SPECULATION=" + ("1" if mode == "none" else "0"), script)
                    self.assertIn(
                        "ATLAS_QWEN4_NATIVE_BATCH=" + ("0" if profile == "serial-control" else "1"), script
                    )
                    self.assertIn("ATLAS_LOGITS_FNV=1", script)
                    self.assertIn("--no-tui", script)
                    self.assertIn("--model-from-path /models", script)
                    self.assertEqual("--speculative" in script, mode == "native-mtp-k5")
                    self.assertTrue(command[-1].startswith("/results/concurrency/"))
                    self.assertEqual(
                        [call.args[1] for call in kill.call_args_list],
                        [signal.SIGTERM, signal.SIGKILL] if force_cleanup else [signal.SIGTERM],
                    )
                    process.wait.assert_called()
            finally:
                os.close(raw_fd)
                os.close(read_fd)
                os.close(write_fd)

    def test_mocked_lifecycle_uses_canonical_launch_and_exact_cleanup(self):
        for mode in D.MODES:
            for profile in D.PROFILES:
                with self.subTest(mode=mode, profile=profile):
                    self.exercise_mocked_server(mode, profile)

    def test_mocked_forced_cleanup_is_a_void_verdict(self):
        self.exercise_mocked_server("none", "native-reference", force_cleanup=True)

    def test_mocked_binding_failure_still_terminates_the_claimed_server(self):
        self.exercise_mocked_server("none", "native-reference", bind_failure=True)

    def test_replaced_log_fails_after_publishing_the_owned_log(self):
        self.exercise_mocked_server("none", "native-reference", replace_log=True)

    def test_popen_failure_closes_log_and_publishes_owned_empty_log(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime = Path(temp)
            D.make_directory(S, runtime, ["concurrency", D.GATE, "arms", "none"])
            stem = f"concurrency/{D.GATE}/arms/none/launch-failure"
            context = types.SimpleNamespace(
                probe=P, plan=self.plan(), recipe=self.recipe(), container_env={},
                model="fixture-launch-failure", runtime=runtime,
                provenance=provenance_fixture(),
            )
            created = []
            original_create = S.create_beneath

            def capture_log(*arguments, **keywords):
                descriptor = original_create(*arguments, **keywords)
                if arguments[2] == "concurrency server log":
                    created.append(descriptor)
                return descriptor

            with (runtime / "raw.log").open("w+b") as raw_log:
                args = types.SimpleNamespace(
                    suite=S, raw_log_fd=raw_log.fileno(), container_id="1" * 64,
                    container_port=8888, container_cwd="/", container_results_path="/results",
                    base_url="http://127.0.0.1:8888",
                )
                with mock.patch.object(S, "create_beneath", side_effect=capture_log), mock.patch.object(
                    D.subprocess, "Popen", side_effect=OSError("launcher unavailable")
                ), mock.patch.object(D, "terminate_server") as terminate:
                    with self.assertRaisesRegex(OSError, "launcher unavailable"):
                        with D.server(args, context, "none", "native-reference", 2, stem):
                            self.fail("failed launch entered the arm")
                terminate.assert_not_called()
                self.assertEqual(len(created), 1)
                with self.assertRaises(OSError):
                    os.fstat(created[0])
                raw_log.seek(0)
                self.assertEqual(raw_log.read().count(b"R15_LOG_FRAME"), 1)
                self.assertEqual(stat_mode(runtime / (stem + ".server.log")), 0o400)

    # --- end to end with the fake engine ---------------------------------------

    @contextmanager
    def fixture_server(self, runtime, knobs, engines, counters, args, context, mode, profile, width, stem):
        _overrides, environment = D.arm_environment(P, context.recipe, context.container_env, mode, profile)
        manifest = {
            "argv": D.arm_argv(context.plan, args, context.model, width, mode),
            "cwd": "/",
            "environment": environment,
        }
        manifest_raw = S.canonical_bytes(manifest)
        manifest_locator = stem + ".manifest.json"
        D.write_artifact(S, runtime, manifest_locator, manifest_raw)
        log_locator = stem + ".server.log"
        log_fd = S.create_beneath(runtime, log_locator, "log", 0o600)
        host_log = runtime / log_locator
        if knobs.get("nvrm_marker"):
            os.write(log_fd, b"kernel: NVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus\n")
        engine = FakeEngine(
            host_log, profile=profile, mode=mode, width=width,
            argv=manifest["argv"], environment=environment, knobs=knobs,
        )
        engines.append(engine)
        counters["ticks"] += 1
        info = os.stat(host_log)
        live = D.LiveArm(
            mode,
            profile,
            width,
            context.model,
            os.getpid(),
            None,
            {"schema": "flux-r15-live-process-proof/1", "mode": mode, "host_pid": os.getpid()},
            {
                "pid": os.getpid(),
                "boot_id": "0123abcd-0000-4000-8000-00000000abcd",
                "start_ticks": counters["ticks"],
                "log_device": info.st_dev,
                "log_inode": info.st_ino,
            },
            manifest,
            Path(manifest_locator).name,
            sha(manifest_raw),
            host_log,
            log_locator,
            log_fd,
        )
        try:
            yield live
        finally:
            engines.remove(engine)
            if knobs.get("shutdown_tail"):
                engine.write("server graceful shutdown complete")
            engine.close()
            os.fchmod(log_fd, 0o400)
            try:
                live.complete_log_sha256 = D.publish_arm_log(args, S, log_locator, log_fd)
                S.assert_path_identity(runtime, log_locator, log_fd, "complete concurrency server log")
            finally:
                os.close(log_fd)

    def exercise_run(self, knobs=None, *, modes=D.MODES, expect_pass=True, ladder=False):
        knobs = knobs or {}
        engines = []
        counters = {"ticks": 0}

        def fake_urlopen(request_object, timeout=None):
            if request_object.data is None:
                return HealthResponse()
            body = json.loads(request_object.data)
            self.assertEqual(body["model"], args.model_name)
            return FakeResponse(engines[-1].stream(body))

        def fake_request(_args, _method, route, body=None, **_kwargs):
            if route == "/metrics":
                engine = engines[-1]
                positive = engine.mode == "native-mtp-k5"
                if knobs.get("invert_k5_metric"):
                    positive = not positive
                return (
                    b'atlas_spec_decode_verify_total{k="5",outcome="accepted"} 12\n'
                    if positive
                    else b"# no K5 events\n"
                )
            return b"{}"

        read_fd, write_fd = os.pipe()
        try:
            temp = tempfile.mkdtemp()
            self.addCleanup(shutil.rmtree, temp, True)
            runtime = Path(temp)
            receipt = runtime / "receipt.json"
            raw_log = runtime / "raw.log"
            receipt_fd = os.open(receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            raw_log_fd = os.open(raw_log, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            snap = tempfile.mkdtemp(prefix="flux-nc-src-")
            self.addCleanup(shutil.rmtree, snap, True)
            snap_repo = Path(snap)
            for rel, src in (
                (Path("repro-kit/qwen4_native_concurrency_probe.py"), PROBE_PATH),
                (Path("repro-kit/qwen4_native_concurrency_matrix.py"), MATRIX_PATH),
                (Path("repro-kit/r9_source_to_binary.py"), R9_PATH),
            ):
                dest = snap_repo / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(src.read_bytes())
            git_env = {
                **os.environ,
                "GIT_AUTHOR_NAME": "test",
                "GIT_AUTHOR_EMAIL": "test@example.invalid",
                "GIT_COMMITTER_NAME": "test",
                "GIT_COMMITTER_EMAIL": "test@example.invalid",
            }
            for cmd in (
                ["git", "-C", str(snap_repo), "init", "-q"],
                ["git", "-C", str(snap_repo), "add", "repro-kit"],
                ["git", "-C", str(snap_repo), "commit", "-q", "-m", "snapshot"],
            ):
                subprocess.run(cmd, env=git_env, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            snap_probe = snap_repo / "repro-kit/qwen4_native_concurrency_probe.py"
            snap_matrix = snap_repo / "repro-kit/qwen4_native_concurrency_matrix.py"
            snap_r9 = snap_repo / "repro-kit/r9_source_to_binary.py"
            _payload, probe_identity = D.read_pinned_bytes(snap_probe, sha(snap_probe.read_bytes()))
            args = types.SimpleNamespace(
                suite=S,
                probe=P,
                probe_identity=probe_identity,
                probe_program=str(snap_probe),
                probe_sha256=sha(snap_probe.read_bytes()),
                r9_sha256=sha(snap_r9.read_bytes()),
                matrix_program=str(snap_matrix),
                matrix_sha256=sha(snap_matrix.read_bytes()),
                runtime_dir=str(runtime),
                receipt_fd=receipt_fd,
                raw_log_fd=raw_log_fd,
                container_id="1" * 64,
                image_id="sha256:" + "2" * 64,
                image_digest="sha256:" + "3" * 64,
                run_id="offline-fixture",
                base_url="http://127.0.0.1:8888",
                container_port=8888,
                container_cwd="/",
                container_results_path="/results",
                max_tokens=MAX_TOKENS,
                minimum_full_frames=8,
                memory_floor_mib=1,
                timeout=30.0,
                cancel_token=CANCEL_TOKEN,
                modes=modes,
                scope="full",
                gate=D.GATE,
                ladder=ladder,
                ladder_widths=(9, 10),
                ladder_profiles=("native-reference",),
                identities={"base_url": "http://127.0.0.1:8888"},
                model_name=None,
            )
            mounts = {
                "/models": {"Source": "/ckpt", "RW": False, "Type": "bind"},
                "/results": {"Source": str(runtime), "RW": True, "Type": "bind"},
            }

            @contextmanager
            def server(_args, context, mode, profile, width, stem):
                args.model_name = context.model
                with self.fixture_server(
                    runtime, knobs, engines, counters, args, context, mode, profile, width, stem
                ) as live:
                    yield live

            original_read = P.LogCursor.read_bytes_from
            original_wave = P.run_wave
            original_churn = P.run_api_churn
            original_mid_churn = D.run_mid_cancel_churn

            def ordered_wave(base_url, model, cases, max_tokens, timeout):
                engines[-1].plan_admissions(cases)
                return original_wave(base_url, model, cases, max_tokens, timeout)

            def ordered_churn(base_url, model, cases, replacement, max_tokens, timeout, baseline):
                engines[-1].plan_admissions(cases, replacement)
                return original_churn(base_url, model, cases, replacement, max_tokens, timeout, baseline)

            def ordered_mid_churn(args, probe, context, live, plan, base, cursor):
                engines[-1].plan_admissions(plan.wave_cases, plan.churn_replacement)
                return original_mid_churn(args, probe, context, live, plan, base, cursor)

            patches = (
                mock.patch.object(
                    D, "inspect_container", return_value=(self.plan(), self.recipe(), {}, mounts)
                ),
                mock.patch.object(D, "verify_runtime_bind"),
                mock.patch.object(D, "preflight_provenance", return_value=provenance_fixture()),
                mock.patch.object(D, "recheck_provenance"),
                mock.patch.object(D, "server", side_effect=server),
                mock.patch.object(D, "request", side_effect=fake_request),
                mock.patch.object(urllib.request, "urlopen", fake_urlopen),
                mock.patch.object(P, "run_wave", side_effect=ordered_wave),
                mock.patch.object(P, "run_api_churn", side_effect=ordered_churn),
                mock.patch.object(D, "run_mid_cancel_churn", side_effect=ordered_mid_churn),
                mock.patch.object(P, "WAVE_ADMISSION_STAGGER_S", 0.002),
                mock.patch.object(
                    P.LogCursor,
                    "read_bytes_from",
                    lambda self, offset, settle_seconds=0.15, end_offset=None: original_read(
                        self, offset, 0.0, end_offset
                    ),
                ),
            )
            try:
                with ExitStack() as stack:
                    for patch in patches:
                        stack.enter_context(patch)
                    if expect_pass:
                        result = D.run(args)
                    else:
                        with self.assertRaises(Exception) as caught:
                            D.run(args)
                        self.assertEqual(receipt.read_bytes(), b"")
                        self.assertFalse(
                            (runtime / "concurrency" / D.GATE / "validator.json").exists()
                        )
                        failures = list((runtime / "concurrency" / D.GATE).glob("failure-*.json"))
                        self.assertEqual(len(failures), 1)
                        return caught.exception, json.loads(failures[0].read_bytes()), runtime
                evidence = result["evidence"]
                Q._validate_concurrency_evidence(
                    evidence,
                    {"_runtime_dir": str(runtime), "launch": {"run_id": args.run_id}},
                    args.container_id,
                )
                raw = (runtime / evidence["validator_artifact"]).read_bytes()
                self.assertEqual(sha(raw), evidence["validator_receipt_sha256"])
                document = json.loads(raw)
                self.assertEqual(json.loads(receipt.read_bytes()), result)
                S.validate_gate_receipt(
                    receipt.read_bytes(),
                    D.GATE,
                    {
                        "image_id": args.image_id,
                        "image_digest": args.image_digest,
                        "container_id": args.container_id,
                        "run_id": args.run_id,
                    },
                )
                self.assertIn(b"R15_LOG_FRAME", raw_log.read_bytes())
                self.assertEqual(raw_log.read_bytes().count(b"R15_LOG_FRAME"), 18 + (2 if ladder else 0))
                return document, runtime, evidence
            finally:
                P.urllib = urllib
                os.close(receipt_fd)
                os.close(raw_log_fd)
        finally:
            os.close(read_fd)
            os.close(write_fd)

    def test_mocked_end_to_end_seals_eighteen_arms_with_the_real_sealer(self):
        document, runtime, evidence = self.exercise_run()
        raw = document["raw"]
        self.assertEqual(len(raw["arms"]), 18)
        self.assertEqual(evidence["fresh_processes"], 18)
        self.assertGreaterEqual(evidence["minimum_full_frames"], 8)
        matrix = raw["plain_matrix_seal"]
        matrix_raw = (runtime / matrix["artifact"]).read_bytes()
        self.assertEqual(sha(matrix_raw), matrix["sha256"])
        sealed = json.loads(matrix_raw)
        self.assertEqual(sealed["schema"], "flux-qwen4-native-concurrency-matrix-v2")
        self.assertIs(sealed["release_qualified"], True)
        self.assertEqual(len(sealed["arms"]), 9)
        self.assertEqual(sealed["sealer_sha256"], sha(MATRIX_PATH.read_bytes()))
        observations = raw["arm_observations"]
        self.assertEqual(
            sorted((o["mode"], o["profile"], o["width"]) for o in observations),
            sorted((m, p, w) for m in D.MODES for p in D.PROFILES for w in D.LEVELS),
        )
        for observation in observations:
            self.assertTrue(observation["forward_reverse_token_identity"])
            self.assertEqual(observation["same_prompt_wave"]["rows"], observation["width"])
            expected = observation["width"] if observation["profile"] != "serial-control" else 0
            self.assertEqual(
                observation["same_prompt_wave"]["census"]["native_width_executed"], expected
            )
            self.assertEqual(observation["k5_events"] > 0, observation["mode"] == "native-mtp-k5")
            if observation["width"] == 8:
                churn = observation["mid_cancel_churn"]
                self.assertEqual(churn["cancelled_label"], "row-03")
                self.assertEqual(churn["content_chunks_before_close"], CANCEL_TOKEN)
                self.assertEqual(churn["cancelled_terminal_geometry"], "computed-unemitted")
                self.assertEqual(len(churn["survivors"]), 7)
            else:
                self.assertIsNone(observation["mid_cancel_churn"])
            arm = json.loads((runtime / observation["arm_receipt"]["artifact"]).read_bytes())
            self.assertEqual(arm["result"], "PASS")
            self.assertEqual(arm["native_engagement_widths"], [] if observation["profile"] == "serial-control" else [observation["width"]])
            self.assertEqual(arm["provenance"]["selector_profile"], observation["profile"])
            self.assertEqual(stat_mode(runtime / observation["server_log"]["artifact"]), 0o400)
            self.assertEqual(sha((runtime / observation["server_log"]["artifact"]).read_bytes()), observation["server_log"]["sha256"])
        timing = raw["timing_rows"]
        self.assertIn("none/native-reference-c2/forward", timing)
        self.assertEqual(len(timing["none/native-reference-c2/forward"]), 2)
        self.assertTrue(all(row["tokens_per_second"] for row in timing["none/native-reference-c2/forward"]))
        self.assertEqual(raw["diagnostics"]["none/native-reference-c8/forward"]["census"]["native_width_executed"], 8)
        self.assertEqual(len(raw["available_mib_samples"]), 54)
        self.assertEqual(evidence["minimum_available_mib"], min(raw["available_mib_samples"]))

    def test_mocked_ladder_rungs_are_fresh_native_processes(self):
        document, _runtime, _evidence = self.exercise_run(ladder=True)
        ladder = document["raw"]["ladder"]
        self.assertEqual([row["width"] for row in ladder], [9, 10])
        for row in ladder:
            self.assertEqual(row["native_engagement_widths"], [row["width"]])
            self.assertEqual([wave["name"] for wave in row["waves"]], ["forward", "reverse"])
            self.assertGreaterEqual(min(wave["full_width_frames"] for wave in row["waves"]), 8)
        self.assertEqual(len(document["raw"]["available_mib_samples"]), 60)

    def test_mocked_serial_collapse_fails_closed_with_executed_width_diagnosis(self):
        error, failure, _runtime = self.exercise_run({"collapse_to_serial": True}, expect_pass=False)
        self.assertEqual(failure["status"], "FAIL")
        census = failure["diagnostics"]["none/native-reference-c2/forward"]["census"]
        self.assertEqual(census["native_width_executed"], 0)
        self.assertIn("2", census["serial_frames_by_rows"])
        self.assertIn("native C2 frames", str(error))

    def test_mocked_token_drift_records_first_divergence_index(self):
        error, failure, _runtime = self.exercise_run({"wave_token_drift": True}, expect_pass=False)
        self.assertIn("token IDs differ", str(error))
        divergence = failure["diagnostics"]["none/native-reference-c2/forward"]["divergence"]
        # Either worker can win admission. The injected fault is on the row
        # admitted while another row is live, not on a fixed public label.
        self.assertCountEqual([row["first_divergence_index"] for row in divergence], [None, 4])

    def test_mocked_logits_drift_in_a_frame_fails_closed(self):
        error, _failure, _runtime = self.exercise_run({"wave_logits_drift": True}, expect_pass=False)
        self.assertIn("logits", str(error))

    def test_mocked_engaged_width_mismatch_fails_closed(self):
        error, _failure, _runtime = self.exercise_run({"engaged_rows_override": 1}, expect_pass=False)
        self.assertIn("engagement marker", str(error))

    def test_mocked_replacement_in_a_fresh_slot_fails_the_c8_churn(self):
        error, _failure, _runtime = self.exercise_run({"replacement_wrong_slot": True}, expect_pass=False)
        self.assertIn("slot", str(error))

    def test_mocked_duplicate_physical_slot_fails_closed(self):
        error, _failure, _runtime = self.exercise_run({"duplicate_slot": True}, expect_pass=False)
        self.assertIn("slot", str(error))

    def test_mocked_missing_state_receipt_fails_closed(self):
        error, _failure, _runtime = self.exercise_run({"missing_state": True}, expect_pass=False)
        self.assertIn("state", str(error))

    def test_mocked_normal_receipt_cannot_claim_a_final_unfed_token(self):
        error, _failure, _runtime = self.exercise_run(
            {"normal_terminal_position_offset": True}, expect_pass=False
        )
        self.assertIn("token census", str(error))

    def test_original_plain_receipt_contract_is_unsatisfiable_without_emitter_fix(self):
        argv = D.arm_argv(self.plan(), types.SimpleNamespace(container_port=8888), "fixture", 2, "none")
        _overrides, environment = D.arm_environment(P, self.recipe(), {}, "none", "native-reference")
        self.assertNotIn("--speculative", argv)
        encoded_argv = [item.encode() for item in argv]
        encoded_environment = {key.encode(): value.encode() for key, value in environment.items()}
        S._verify_live_mode(encoded_argv, encoded_environment, "none")
        with self.assertRaisesRegex(ValueError, "ATLAS_DISABLE_SPECULATION=1"):
            S._verify_live_mode(
                encoded_argv, {**encoded_environment, b"ATLAS_DISABLE_SPECULATION": b"0"}, "none"
            )
        # The legacy emitter source rejects this launch. Never synthesize its absent receipt.
        error, _failure, runtime = self.exercise_run(
            {"target_state_emitter_source": LEGACY_TARGET_STATE_EMITTER_SNIPPET},
            expect_pass=False,
        )
        self.assertIn("qualification evidence", str(error))
        self.assertIn(
            b"rejects a disabled control without a proposer", (runtime / "raw.log").read_bytes()
        )
        self.assertFalse((runtime / "concurrency" / D.GATE / "matrix" / "none.json").exists())

    def test_fixed_source_classifier_accepts_truthful_plain_receipt_contract(self):
        source = (REPO / TARGET_STATE_PATH).read_text()
        self.assertIn("fn classify_capture_mode(", source, "emitter classifier repair is a source dependency")
        self.assertIn("classify_capture_mode(proposer_kind, force_serial)?;", source)
        self.assertIn(
            '(None, _) => ("unloaded-target-only", "ordinary-target-only", false)', source
        )
        document, runtime, evidence = self.exercise_run({"target_state_emitter_source": source})
        self.assertEqual(evidence["fresh_processes"], 18)
        self.assertEqual(document["status"], "PASS")
        first = document["raw"]["arm_observations"][0]
        raw = (runtime / first["server_log"]["artifact"]).read_bytes()
        self.assertIn(b"mode=unloaded-target-only scheduler_path=ordinary-target-only proposer_loaded=false", raw)
        self.assertNotIn(b"REQUEST_RECEIPT_FAILED", raw)

    def test_mid_cancelled_receipt_metadata_and_token_bounds_fail_closed(self):
        for mutation, expected in (
            ({"schema": 0, "mode": "wrong-mode", "compared_bytes": 0}, "target-only"),
            ({"gdn_layers": 0}, "layer census"),
            ({"output_tokens": CANCEL_TOKEN - 1}, "token count"),
            ({"output_tokens": MAX_TOKENS}, "token count"),
        ):
            with self.subTest(mutation=mutation):
                error, _failure, _runtime = self.exercise_run(
                    {"mid_cancel_state_mutation": mutation}, expect_pass=False
                )
                self.assertIn(expected, str(error))

    def test_mid_cancelled_k5_receipt_cannot_claim_plain_mode(self):
        error, _failure, runtime = self.exercise_run(
            {
                "mid_cancel_mutation_mode": "native-mtp-k5",
                "mid_cancel_state_mutation": {
                    "mode": "unloaded-target-only",
                    "scheduler_path": "ordinary-target-only",
                    "proposer_loaded": "false",
                },
            },
            expect_pass=False,
        )
        self.assertIn("native-K5", str(error))
        self.assertTrue((runtime / "concurrency" / D.GATE / "matrix" / "none.json").exists())

    def test_ladder_offset_positions_fail_closed(self):
        error, _failure, _runtime = self.exercise_run(
            {"ladder_position_offset": True}, expect_pass=False, ladder=True
        )
        self.assertIn("frame sequence length differs from output index", str(error))

    def test_complete_log_artifact_hash_includes_shutdown_tail(self):
        document, runtime, _evidence = self.exercise_run({"shutdown_tail": True})
        for observation in document["raw"]["arm_observations"]:
            log = (runtime / observation["server_log"]["artifact"]).read_bytes()
            arm = json.loads((runtime / observation["arm_receipt"]["artifact"]).read_bytes())
            self.assertTrue(log.endswith(b"server graceful shutdown complete\n"))
            self.assertEqual(sha(log), observation["server_log"]["sha256"])
            qualified = log[: arm["server_log_qualified_end_offset"]]
            self.assertEqual(sha(qualified), arm["server_log_sha256"])

    def test_mocked_nvrm_marker_is_a_void_verdict(self):
        error, failure, runtime = self.exercise_run({"nvrm_marker": True}, expect_pass=False)
        self.assertEqual(failure["status"], "VOID")
        self.assertIn("forbidden marker", str(error))
        self.assertIn(b"NVRM: Xid", (runtime / "raw.log").read_bytes())
        self.assertEqual((runtime / "raw.log").read_bytes().count(b"R15_LOG_FRAME"), 1)

    def test_mocked_k5_arm_with_plain_receipts_fails_and_keeps_the_plain_seal(self):
        error, failure, runtime = self.exercise_run({"k5_state_plain": True}, expect_pass=False)
        self.assertIn("native-K5", str(error))
        self.assertTrue((runtime / "concurrency" / D.GATE / "matrix" / "none.json").exists())
        self.assertEqual(len(failure["process_bindings"]), 10)

    def test_mocked_k5_counter_mismatch_fails_closed(self):
        error, _failure, _runtime = self.exercise_run({"invert_k5_metric": True}, expect_pass=False)
        self.assertIn("K5 verifier engagement", str(error))

    def test_plain_only_modes_cannot_produce_a_pass_receipt(self):
        error, _failure, runtime = self.exercise_run(modes=("none",), expect_pass=False)
        self.assertIn("incomplete", str(error))
        self.assertTrue((runtime / "concurrency" / D.GATE / "matrix" / "none.json").exists())

    def test_example_plan_entry_matches_the_suite_plan_contract(self):
        example = json.loads((ROOT / "examples" / "r15-container-suite-plan.example.json").read_bytes())
        job = example["gate_jobs"][D.GATE]
        self.assertTrue(job["program_path"].endswith("repro-kit/r15-gates/native-concurrency/native-concurrency-c2-c8"))
        self.assertEqual(job["argv"][0], job["program_path"])
        for token in S.IDENTITY_PLACEHOLDERS:
            self.assertEqual(job["argv"].count(token), 1)
        for flag in ("--probe-program", "--matrix-program", "--r9-program", "--launch-manifest", "--suite-program"):
            self.assertIn(flag, job["argv"])
        entry = json.loads(
            (ROOT / "r15-gates" / "native-concurrency" / "plan-entry.example.json").read_bytes()
        )
        self.assertEqual(entry[D.GATE]["argv"], job["argv"])


def stat_mode(path):
    return os.stat(path).st_mode & 0o777


if __name__ == "__main__":
    unittest.main()
