# SPDX-License-Identifier: AGPL-3.0-only
"""Offline-only tests. All server, Docker, and HTTP execution is mocked."""

import array
import contextlib
import base64
import gc
from contextlib import ExitStack, contextmanager
import io
import hashlib
import importlib.machinery
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import struct
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_qwen4_mtp_long_context_qualify as fixtures  # noqa: E402


def module(path, name):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    loader.exec_module(value)
    return value


S = module(ROOT / "r15-container-suite.py", "r16_long_context_suite_fixture")
Q = module(ROOT / "r15_exact_image_release.py", "r16_long_context_outer_fixture")
D = module(ROOT / "r15-gates" / "plain-k5-long-context", "r16_long_context_driver")


def load_qualifier_isolated():
    # load_qualifier re-executes the pinned modules under their real names, as
    # the driver does in its own process. Restore sys.modules afterwards so
    # later test files keep one shared copy of r9_source_to_binary and friends.
    names = [name for name, _ in D.QUALIFIER_PINS]
    saved = {name: sys.modules[name] for name in names if name in sys.modules}
    try:
        return D.load_qualifier(str(ROOT))
    finally:
        for name in names:
            if name in saved:
                sys.modules[name] = saved[name]
            else:
                sys.modules.pop(name, None)


L = load_qualifier_isolated()
TOKEN_BASE = 1000
CHUNK = 8192
# At the release geometry a mocked run holds ~35 MB of prompt lists (the
# qualifier's 1.83M-token bundle plus each arm's continuation bodies), which
# with a ~70 MB import baseline exceeds the 90 MB address-space cap the suites
# run under. Fault replays run at a scaled geometry instead: the same case ids,
# contracts and endpoint structure (fill to the native endpoint, accept
# endpoint-1 + 1, reject endpoint + 1), with every release rung below the
# scaled endpoint kept. The rules they check do not depend on prompt length,
# and each fault replay first proves a clean scaled run passes, so a fault
# fails for its own reason. The release geometry and the outer wrapper stay
# covered by test_mocked_end_to_end_passes_the_outer_wrapper_and_seals_artifacts.
SCALED_CONTEXT = 16_384
RELEASE_RUN_ADDRESS_SPACE = 192 * 1024 * 1024
PLAN_EXAMPLE = ROOT / "examples" / "r15-container-suite-plan.example.json"
TARGET_STATE_SOURCE = ROOT.parent / "crates/spark-model/src/model/qwen4_target_state_receipt.rs"


def startup_lines(arm):
    tail = ", policy=fcfs, chunked_prefill=true, max_prefill_tokens=8192)"
    if arm == "plain":
        return [L.PLAIN_SCHEDULER_MARKER + tail]
    lines = [
        L.K5_SCHEDULER_MARKER + tail,
        L.K5_CONSTRUCTION_MARKER,
        L.K5_QSA_VERIFY_ROWS_MARKER,
    ]
    if arm == "k5":
        lines.append(L.K5_VERIFY_GRAPH_CAPTURE_MARKER + " (slot=0 K=5)")
    else:
        lines.append(L.QSA_ROUTE_TELEMETRY_ENGAGED_MARKER)
    return lines


def decode(tokens):
    return bytes(token - TOKEN_BASE for token in tokens).decode("utf-8", "replace")


def word_tokens(text):
    return [
        int.from_bytes(hashlib.sha256(word.encode()).digest()[:3], "little")
        for word in text.split()
    ]


assert array.array("I").itemsize == 4 and sys.byteorder == "little"


def fixture_fed_digest(tokens):
    fields = (
        (b"domain", b"atlas-qwen4-fed-tokens-v1"),
        (b"count", struct.pack("<Q", len(tokens))),
    )
    framed = b"".join(
        struct.pack("<Q", len(label)) + label + struct.pack("<Q", len(value)) + value
        for label, value in fields
    )
    # One u32 buffer, not a join of per-token objects (~12 MB at 262K, which
    # does not fit under the suite's 90 MB address-space cap).
    digest = hashlib.sha256(framed)
    digest.update(array.array("I", tokens).tobytes())
    return digest.hexdigest()


@contextmanager
def scaled_geometry(context=SCALED_CONTEXT):
    endpoint = context - L.FULL_ENDPOINT_MAX_TOKENS
    issue13 = tuple(spec for spec in L.ISSUE13_CASE_SPECS if spec[1] < endpoint)
    full = (L.FULL_ENDPOINT_CASE_SPEC[0], endpoint, L.FULL_ENDPOINT_CASE_SPEC[2])
    semantic = (
        L.ENDPOINT_SEMANTIC_CASE_SPEC[0], endpoint, L.ENDPOINT_SEMANTIC_CASE_SPEC[2]
    )
    accept = (L.ADMISSION_ACCEPT_CASE_SPEC[0], context - 1, L.ADMISSION_ACCEPT_CASE_SPEC[2])
    reject = (L.ADMISSION_REJECT_CASE_SPEC[0], context, L.ADMISSION_REJECT_CASE_SPEC[2])
    paired = (*issue13, semantic, full)
    cases = (*paired, accept, reject)
    qualifier = {
        "ISSUE13_CASE_SPECS": issue13,
        "FULL_ENDPOINT_CASE_SPEC": full,
        "ENDPOINT_SEMANTIC_CASE_SPEC": semantic,
        "ADMISSION_ACCEPT_CASE_SPEC": accept,
        "ADMISSION_REJECT_CASE_SPEC": reject,
        "PAIRED_CASE_SPECS": paired,
        "CASE_SPECS": cases,
        "PAIRED_CASE_IDS": tuple(spec[0] for spec in paired),
        "PROMPT_TOKEN_COUNTS": tuple(dict.fromkeys(spec[1] for spec in cases)),
        "PAIRED_PROMPT_TOKEN_COUNTS": tuple(dict.fromkeys(spec[1] for spec in paired)),
        "NATIVE_CONTEXT": context,
    }
    driver = {
        "NATIVE_CONTEXT": context,
        "CASE_PROMPT_TOKENS": tuple(
            tokens for tokens in D.CASE_PROMPT_TOKENS if tokens < endpoint
        ) + (endpoint,),
    }
    with ExitStack() as stack:
        for name, value in qualifier.items():
            stack.enter_context(mock.patch.object(L, name, value))
        for name, value in driver.items():
            stack.enter_context(mock.patch.object(D, name, value))
        yield


def address_space_limit():
    soft, _hard = resource.getrlimit(resource.RLIMIT_AS)
    return None if soft == resource.RLIM_INFINITY else soft


class World:
    """Deterministic fake engine: byte tokenizer, known answers, receipts."""

    def __init__(self, mutate=None, route="row-serial", receipt_mutate=None):
        self.mutate = mutate
        self.receipt_mutate = receipt_mutate
        self.route = route
        self.arm = None
        self.log_fd = None
        self.frame_id = 0
        self.req = 0
        self.chunks = 0
        self.tokens = 0
        self.k5_events = 0
        self.responses = 0
        self.state_request_counts = {}

    def emit_state(self, body, response):
        prompt = body["prompt_token_ids"]
        completion = response["choices"][0]["token_ids"]
        fed = prompt + completion[:-1]
        blocks = math.ceil(len(fed) / 16)
        native = self.arm != "plain"
        session_bytes = b"".join(struct.pack("<I", token) for token in prompt[:1024])
        fed_digest = fixture_fed_digest(fed)
        row = {
            "schema": 1,
            "mode": "engaged-native-mtp" if native else "unloaded-target-only",
            "scheduler_path": "native-mtp" if native else "ordinary-target-only",
            "proposer_loaded": "true" if native else "false",
            "session_hash": hashlib.sha256(session_bytes).hexdigest()[:16],
            "seq_len": len(fed), "token_count": len(fed),
            "output_tokens": len(completion), "gdn_layers": 36, "attention_layers": 12,
            "logical_kv_blocks": blocks, "kv_token_positions": len(fed) * 12,
            "compared_buffers": 73 + 24 * blocks, "compared_bytes": 4096 + len(fed) * 128,
            "state_scope": "live-gdn-ple-plus-logical-bf16-kv", "checkpoints_excluded": "true",
            "fed_tokens_sha256": fed_digest,
            "target_state_sha256": hashlib.sha256(
                b"offline-only-fake-state:" + fed_digest.encode()
            ).hexdigest(),
        }
        key = (self.arm, len(prompt), body["max_tokens"])
        self.state_request_counts[key] = self.state_request_counts.get(key, 0) + 1
        if self.receipt_mutate is not None:
            self.receipt_mutate(self, body, response, row)
        fields = " ".join(f"{key}={value}" for key, value in row.items())
        self.log([D.STATE_MARKER + " " + fields])

    def log(self, lines):
        D.os_write_all(self.log_fd, ("\n".join(lines) + "\n").encode())

    def dispatch(self, prompt_tokens, native, spec=True):
        self.chunks += math.ceil(prompt_tokens / CHUNK)
        self.tokens += prompt_tokens
        first = prompt_tokens + 1
        lines = [
            (
                f"Chunked prefill start: {prompt_tokens} prompt tokens, "
                f"chunk_size={CHUNK}, max_tokens=96"
                if prompt_tokens > CHUNK
                else f"Prefilling: {prompt_tokens} prompt tokens, max_tokens=96"
            )
        ]
        if native and spec:
            self.req += 1
            if self.arm == L.EAGER_CHURN_ARM:
                self.frame_id += 1
                lines += fixtures.qsa_route_frame(
                    self.frame_id, first, route=self.route
                )
            lines.append(fixtures.mtp_frame(first, 2))
            lines.append(fixtures.spec_cycle(self.req, first, 2))
            self.k5_events += 1
        self.log(lines)

    def completion(self, body, headers):
        prompt = body["prompt_token_ids"]
        prompt_tokens = len(prompt)
        native = self.arm != "plain"
        if body["ignore_eos"] and body["max_tokens"] == 1:
            if prompt_tokens + 1 > D.NATIVE_CONTEXT:
                request_id = dict((k.lower(), v) for k, v in (headers or {}).items())[
                    "x-request-id"
                ]
                error = {
                    "message": (
                        f"Context too long: {prompt_tokens} prompt + 1 requested "
                        f"output tokens exceeds max_seq_len {D.NATIVE_CONTEXT}"
                    ),
                    "type": "invalid_request_error",
                    "param": None,
                    "code": None,
                }
                return 400, {"error": error}, [("x-request-id", request_id)]
            tokens = word_tokens("probe-" + L.token_u32le_sha256(prompt))[:1]
            text = "x"
            finish = "length"
            self.dispatch(prompt_tokens, native, spec=False)
        elif body["ignore_eos"]:
            seed = L.token_u32le_sha256(prompt)
            tokens = [
                int.from_bytes(
                    hashlib.sha256(f"{seed}:{i}".encode()).digest()[:3], "little"
                )
                for i in range(body["max_tokens"])
            ]
            text = "fill " * 16
            finish = "length"
            self.dispatch(prompt_tokens, native)
        else:
            text_prompt = decode(prompt)
            codes = {
                label: re.search(
                    rf"\[{label.upper()}\] audit code ([0-9A-F]{{16}})", text_prompt
                ).group(1)
                for label in ("early", "middle", "late")
            }
            text = " " + L.expected_answer(codes)
            tokens = word_tokens(text) + [7]
            finish = "stop"
            self.dispatch(prompt_tokens, native)
        self.responses += 1
        response = {
            "id": f"{self.arm}-{self.responses}",
            "model": body["model"],
            "choices": [
                {"index": 0, "text": text, "finish_reason": finish, "token_ids": tokens}
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": len(tokens),
                "total_tokens": prompt_tokens + len(tokens),
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        }
        if self.mutate is not None:
            self.mutate(self, body, response)
        if "token_ids" in response["choices"][0]:
            self.emit_state(body, response)
        return 200, response, []

    def exchange(self, args, method, route, body=None, headers=None, timeout=None):
        if route == "/tokenize":
            tokens = [TOKEN_BASE + value for value in body["prompt"].encode()]
            value = {"tokens": tokens, "count": len(tokens)}
            status, response_headers = 200, []
        elif route == "/metrics":
            text = (
                f"atlas_prefill_progress_chunks_total {self.chunks}\n"
                f"atlas_prefill_progress_tokens_total {self.tokens}\n"
            )
            if self.arm != "plain":
                text += (
                    'atlas_spec_decode_verify_total{k="5",outcome="accepted"} '
                    f"{self.k5_events}\n"
                )
            return 200, text.encode(), []
        elif route == "/health":
            return 200, b"{}", []
        elif route == "/v1/completions":
            status, value, response_headers = self.completion(body, headers)
        else:
            raise AssertionError(route)
        return status, S.canonical_bytes(value), response_headers


class PlainK5LongContextDriver(unittest.TestCase):
    def receipt_fixture(self, arm="plain"):
        body = {"prompt_token_ids": [1, 2], "max_tokens": 1}
        response = {"choices": [{"token_ids": [3]}]}
        with tempfile.TemporaryFile() as log:
            world = World()
            world.arm, world.log_fd = arm, log.fileno()
            world.emit_state(body, response)
            log.seek(0)
            return body, response, log.read()

    def test_fed_digest_matches_framed_v1_and_excludes_the_pending_token(self):
        self.assertEqual(
            D.fed_tokens_sha256([1, 2, 3]),
            "e8215a641ad9d534778f3db6e2cf6f5b7ee37437d23313f8a294a8cc8887545e",
        )
        self.assertNotEqual(D.fed_tokens_sha256([1, 2, 3]), L.token_u32le_sha256([1, 2, 3]))
        body, response, delta = self.receipt_fixture()
        receipt = D.capture_target_state("unit", delta, body["prompt_token_ids"], [3], "plain")
        self.assertEqual(receipt["seq_len"], 2)
        self.assertEqual(receipt["output_tokens"], 1)
        self.assertEqual(receipt["fed_tokens_sha256"], fixture_fed_digest([1, 2]))
        with self.assertRaisesRegex(ValueError, "one-token-behind"):
            D.capture_target_state("unit", delta, [1, 2], [3, 4], "plain")
        with self.assertRaisesRegex(ValueError, "unsupported empty"):
            D.capture_target_state("unit", delta, [1, 2], [], "plain")
        with self.assertRaisesRegex(ValueError, "unsupported target-fed endpoint"):
            D.capture_target_state("unit", delta, [1, 2], [3] * D.NATIVE_CONTEXT, "plain")
        with self.assertRaisesRegex(ValueError, "unknown state arm"):
            D.capture_target_state("unit", delta, [1, 2], [3], "not-a-release-arm")

    def test_source_emitter_dependency_preserves_real_proposer_modes(self):
        source = TARGET_STATE_SOURCE.read_text()
        self.assertIn("classify_capture_mode(proposer_kind, force_serial)?;", source)
        self.assertIn('(None, _) => ("unloaded-target-only", "ordinary-target-only", false)', source)
        self.assertIn('("loaded-serial", "ordinary-target-only", true)', source)
        self.assertIn('("engaged-native-mtp", "native-mtp", true)', source)
        body, _response, delta = self.receipt_fixture("k5")
        receipt = D.capture_target_state("one-token", delta, body["prompt_token_ids"], [3], "k5")
        self.assertEqual(receipt["mode"], "engaged-native-mtp")
        self.assertTrue(receipt["proposer_loaded"])
        self.assertFalse(L.parse_mtp_accept_frames(delta.decode()))

    def test_state_receipt_parser_census_modes_and_coverage_fail_closed(self):
        body, _response, delta = self.receipt_fixture()
        for bad in (
            b"", delta + delta,
            delta.replace(b"schema=1", b"schema=0"),
            delta.replace(b"unloaded-target-only", b"loaded-serial"),
            delta.replace(b"gdn_layers=36", b"gdn_layers=35"),
            delta.replace(b"seq_len=2 token_count=2", b"seq_len=3 token_count=3"),
            delta.replace(b"compared_buffers=97", b"compared_buffers=1"),
            delta.replace(b"checkpoints_excluded=true", b"checkpoints_excluded=false"),
            delta + (D.STATE_MARKER + "_FAILED detail=capture failed\n").encode(),
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                D.capture_target_state("unit", bad, body["prompt_token_ids"], [3], "plain")

    def test_identical_plain_deltas_bind_each_offset_once(self):
        _body, _response, delta = self.receipt_fixture()
        with tempfile.TemporaryFile() as log:
            capture = D.LogCapture(log.fileno(), settle_seconds=0.0)
            for _ in range(2):
                offset = capture.begin()
                os.write(log.fileno(), delta)
                capture.finish(offset)
            first = capture.claim_state_delta(D.digest(delta))[1]
            second = capture.claim_state_delta(D.digest(delta))[1]
            self.assertNotEqual(first["offset"], second["offset"])
            capture.verify_state_claims()
            with self.assertRaisesRegex(ValueError, "already claimed"):
                capture.claim_state_delta(D.digest(delta))

    def test_claimed_state_ranges_reject_later_changes_and_overlap(self):
        _body, _response, delta = self.receipt_fixture()
        with tempfile.TemporaryFile() as log:
            capture = D.LogCapture(log.fileno(), settle_seconds=0.0)
            os.write(log.fileno(), delta)
            capture.finish(0)
            capture.claim_state_delta(D.digest(delta))
            os.pwrite(log.fileno(), b"X", 0)
            with self.assertRaisesRegex(ValueError, "changed before sealing"):
                capture.verify_state_claims()
            os.pwrite(log.fileno(), delta, 0)
            capture.deltas.append(dict(capture.deltas[0]))
            capture.state_claims.add(1)
            with self.assertRaisesRegex(ValueError, "ranges overlap"):
                capture.verify_state_claims()

    def test_qualifier_pins_match_the_tracked_sources(self):
        for name, expected in D.QUALIFIER_PINS:
            raw = (ROOT / (name + ".py")).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), expected, name)

    def test_rung_census_matches_the_outer_wrapper(self):
        self.assertEqual(D.CASE_PROMPT_TOKENS, Q.LONG_CONTEXT_PROMPT_TOKENS)
        self.assertEqual(len(D.CASE_PROMPT_TOKENS), 21)
        self.assertEqual(max(D.CASE_PROMPT_TOKENS) + 64, D.NATIVE_CONTEXT)

    def test_source_dependency_hash_mismatch_fails_before_module_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            for name, _expected in D.QUALIFIER_PINS:
                (Path(temp) / (name + ".py")).write_text(
                    "raise AssertionError('must not execute')\n"
                )
            with self.assertRaisesRegex(ValueError, "dependency hash"):
                D.load_qualifier(temp)

    def test_example_plan_entry_is_runnable_against_parse_args(self):
        plan = json.loads(PLAN_EXAMPLE.read_text())
        job = plan["gate_jobs"][D.GATE]
        argv = [
            {
                "@CONTAINER_ID@": "1" * 64,
                "@IMAGE_ID@": "sha256:" + "2" * 64,
                "@IMAGE_DIGEST@": "sha256:" + "3" * 64,
                "@BASE_URL@": "http://127.0.0.1:8888",
                "@RUN_ID@": "offline-fixture",
                "@GATE_RECEIPT_FD@": "7",
                "@RAW_LOG_FD@": "8",
            }.get(item, item)
            for item in job["argv"][1:]
        ]
        self.assertNotIn("@PROCESS_PROOF_FD@", argv)
        index = argv.index("--suite-sha256") + 1
        argv[index] = "a" * 64
        with (
            mock.patch.object(D, "verify_parent_suite"),
            mock.patch.object(D, "load_module", return_value=S),
        ):
            args = D.parse_args(argv)
        self.assertEqual(args.container_port, 8888)
        self.assertEqual(
            args.qualifier_dir, str(Path(job["program_path"]).parent.parent)
        )
        self.assertTrue(
            job["program_path"].endswith("/repro-kit/r15-gates/plain-k5-long-context")
        )
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            D.parse_args(argv[:-2])

    def test_release_route_policy_is_row_serial_until_sparse_rows_qualify(self):
        self.assertFalse(D.SPARSE_ROWS_V2_QUALIFIED)
        for seq_len in (2_044, 2_045, 2_048, 2_049, 2_051, 262_081):
            self.assertEqual(D.expected_verify_route(seq_len, 5), "row-serial")
        self.assertEqual(
            D.expected_verify_route(2_047, 5, sparse_rows_v2_qualified=True),
            "row-serial",
        )
        self.assertEqual(
            D.expected_verify_route(2_048, 5, sparse_rows_v2_qualified=True),
            "sparse-rows-v2",
        )
        with self.assertRaises(ValueError):
            D.expected_verify_route(2_048, 4)

    def test_route_policy_rejects_row_serial_where_rows_v2_is_required(self):
        def frame(seq_len, route):
            counts = {"row-serial": 0, "sparse-rows-legacy": 0, "sparse-rows-v2": 0}
            counts[route] = 12
            return {
                "frame_id": 1,
                "rows": 5,
                "seq_len": seq_len,
                "route_counts": counts,
            }

        summary = D.validate_route_policy([frame(2_044, "row-serial")])
        self.assertEqual(summary["frame_route_counts"]["row-serial"], 1)
        with self.assertRaisesRegex(ValueError, "route policy"):
            D.validate_route_policy([frame(2_044, "sparse-rows-v2")])
        with self.assertRaisesRegex(ValueError, "route policy"):
            D.validate_route_policy(
                [frame(2_051, "row-serial")], sparse_rows_v2_qualified=True
            )
        D.validate_route_policy(
            [frame(2_051, "sparse-rows-v2")], sparse_rows_v2_qualified=True
        )
        with self.assertRaisesRegex(ValueError, "route policy"):
            D.validate_route_policy([frame(2_044, "sparse-rows-legacy")])
        with self.assertRaises(ValueError):
            D.validate_route_policy([])

    def test_receipt_environment_binds_to_the_release_recipe(self):
        recipe = {"ATLAS_QWEN4_VERIFY_GRAPH": "1"}
        plain = D.receipt_environment(recipe, "none", {})
        self.assertEqual(plain["ATLAS_DISABLE_SPECULATION"], "1")
        self.assertEqual(plain["ATLAS_MTP_ACCEPT_LOG"], "1")
        self.assertEqual(plain["ATLAS_QWEN4_TARGET_STATE_RECEIPT"], "1")
        self.assertIn("spec_accept_log=debug", plain["RUST_LOG"])
        native = D.receipt_environment(recipe, "native-mtp-k5", D.EAGER_ARM_ENV)
        self.assertEqual(native["ATLAS_DISABLE_SPECULATION"], "0")
        self.assertEqual(native["ATLAS_DFLASH_DEBUG_NO_GRAPH"], "1")
        self.assertEqual(native["ATLAS_QWEN4_VERIFY_GRAPH"], "1")
        pinned = dict(recipe, RUST_LOG=D.RUST_LOG_RECEIPT_DIRECTIVES)
        self.assertEqual(
            D.receipt_environment(pinned, "none", {})["RUST_LOG"], pinned["RUST_LOG"]
        )
        for conflict in (
            {"ATLAS_MTP_ACCEPT_LOG": "0"},
            {"ATLAS_DFLASH_ASYNC": "1"},
            {"RUST_LOG": "info"},
            {"RUST_LOG": "info,spark::scheduler::spec_accept_log=debug"},
            {"ATLAS_QWEN4_QSA_ROUTE_TELEMETRY": "0"},
            {"ATLAS_QWEN4_TARGET_STATE_RECEIPT": "0"},
            {"ATLAS_QWEN4_TARGET_STATE_RECEIPT_LIMIT": "1"},
        ):
            with self.subTest(conflict=conflict), self.assertRaises(ValueError):
                D.receipt_environment(
                    dict(recipe, **conflict), "native-mtp-k5", D.EAGER_ARM_ENV
                )

    def test_route_policy_widths_follow_the_k5_draft_policy(self):
        def frame(rows):
            counts = {"row-serial": 12, "sparse-rows-legacy": 0, "sparse-rows-v2": 0}
            return {"frame_id": 1, "rows": rows, "seq_len": 2_051, "route_counts": counts}

        for rows in (2, 3, 4, 5):
            with self.subTest(rows=rows):
                summary = D.validate_route_policy(
                    [frame(rows)], k5_policy=D.GATED_K5_POLICY
                )
                self.assertEqual(summary["frame_route_counts"]["row-serial"], 1)
        # The default (fixed K5) caller is unchanged: 5 rows only.
        D.validate_route_policy([frame(5)])
        for rows in (2, 3, 4):
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, "geometry"):
                D.validate_route_policy([frame(rows)])
        for rows, policy in ((1, D.GATED_K5_POLICY), (6, D.GATED_K5_POLICY),
                             (True, D.GATED_K5_POLICY), (5, "native-mtp-k5")):
            with self.subTest(rows=rows, policy=policy), self.assertRaises(ValueError):
                D.validate_route_policy([frame(rows)], k5_policy=policy)

    def test_k5_draft_policy_is_explicit_fixed_or_confidence_gated(self):
        gated = {
            prefix + suffix: value
            for prefix in D.K5_POLICY_PREFIXES
            for suffix, value in (
                ("MTP_CONF", "0.6"),
                ("MTP_CONF_EARLY", "1"),
                ("MTP_VOCAB_PRUNE", "0"),
            )
        }
        native = D.receipt_environment(gated, "native-mtp-k5", D.EAGER_ARM_ENV)
        self.assertEqual(D.k5_draft_policy(gated), D.GATED_K5_POLICY)
        self.assertEqual({key: native[key] for key in gated}, gated)
        fixed = D.receipt_environment({}, "native-mtp-k5", {})
        self.assertEqual(D.k5_draft_policy({}), D.FIXED_K5_POLICY)
        self.assertEqual({key: fixed[key] for key in D.FIXED_K5_ENV}, D.FIXED_K5_ENV)
        for value in ("1.5", "", "abc"):
            for suffix in ("MTP_CONF", "MTP_CONF_EARLY", "MTP_VOCAB_PRUNE"):
                change = {prefix + suffix: value for prefix in D.K5_POLICY_PREFIXES}
                with self.subTest(suffix=suffix, value=value), self.assertRaises(
                    ValueError
                ):
                    D.receipt_environment(dict(gated, **change), "native-mtp-k5", {})

    def test_continuation_validator_is_fail_closed(self):
        good = {
            "model": "m",
            "choices": [
                {"index": 0, "text": "x", "finish_reason": "length", "token_ids": [5]}
            ],
            "usage": {
                "prompt_tokens": 2100,
                "completion_tokens": 1,
                "total_tokens": 2101,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        }
        row = D.validate_continuation(good, 2100, "m")
        self.assertEqual(row["token_ids"], [5])
        changes = (
            lambda value: value["choices"][0].pop("token_ids"),
            lambda value: value["choices"][0].update(token_ids=[5, 6]),
            lambda value: value["choices"][0].update(finish_reason="stop"),
            lambda value: value["usage"].update(prompt_tokens=2099),
            lambda value: value["usage"]["prompt_tokens_details"].update(
                cached_tokens=3
            ),
            lambda value: value.update(model="other"),
            lambda value: value.update(error={"message": "x"}),
        )
        for change in changes:
            value = json.loads(json.dumps(good))
            change(value)
            with self.subTest(change=change), self.assertRaises(ValueError):
                D.validate_continuation(value, 2100, "m")
        with self.assertRaisesRegex(ValueError, "scheduler dispatch"):
            D.continuation_log_evidence(L, "", 2100, False)
        with self.assertRaisesRegex(ValueError, "target-only"):
            D.continuation_log_evidence(L, fixtures.request_log(2100), 2100, False)
        evidence = D.continuation_log_evidence(
            L, fixtures.request_log(2100), 2100, True
        )
        self.assertEqual(evidence["native_mtp_frames"], 1)

    def test_fatal_markers_void_the_run_and_error_lines_fail_closed(self):
        for payload in (
            b"xx NVRM: Xid (PCI:0000:01:00): 79\n",
            b"CUDA out of memory\n",
        ):
            with self.assertRaises(D.Void):
                D.scan_fatal_markers(S, L, payload)
        D.scan_fatal_markers(S, L, b"2026 INFO ok\n")
        logs = b"2026-01-01T00:00:00Z  INFO fine\n2026-01-01T00:00:01Z ERROR bad\n"
        with self.assertRaisesRegex(ValueError, "ERROR-level"):
            D.error_level_lines(logs, [])
        rows = D.error_level_lines(logs, [(logs.index(b"2026-01-01T00:00:01Z"), 40)])
        self.assertEqual([row["permitted"] for row in rows], [True])
        self.assertEqual(
            D.error_level_lines(b"INFO message with ERROR inside\n", []), []
        )

    def test_parent_verifier_pin_requires_sealed_actual_launch_bytes(self):
        descriptor = S.anonymous_writable_file("offline-parent-suite")
        S.write_all(descriptor, b"offline sealed suite fixture")
        expected = hashlib.sha256(b"offline sealed suite fixture").hexdigest()
        try:
            with (
                mock.patch.object(D.os, "getppid", return_value=123),
                mock.patch.object(
                    D.Path, "read_bytes", return_value=b"python3\0/proc/self/fd/99\0"
                ),
                mock.patch.object(
                    D.os,
                    "open",
                    side_effect=lambda *_args, **_kwargs: os.dup(descriptor),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "not sealed"):
                    D.verify_parent_suite(expected)
                S.seal_written_file(descriptor, "offline parent", 1000)
                D.verify_parent_suite(expected)
                with self.assertRaisesRegex(ValueError, "differs"):
                    D.verify_parent_suite("1" * 64)
        finally:
            os.close(descriptor)

    def exercise_mocked_run(
        self, world, *, expect=None, expect_message=None, check=None, scaled=True
    ):
        """Explicit mocked engine, never production qualification."""
        # Fault replays leave run state in exception/traceback cycles that the
        # cyclic GC rarely reaches between tests; without this, consecutive
        # replays creep past the address-space cap.
        self.addCleanup(gc.collect)
        if not scaled:
            return self.mocked_run(world, expect, expect_message, check)
        with scaled_geometry():
            if expect is not None:
                self.mocked_run(World(), None, None, self.check_scaled_pass)
            return self.mocked_run(world, expect, expect_message, check)

    def mocked_run(self, world, expect, expect_message, check):
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
                    qualifier_dir=str(ROOT),
                    runtime_dir=str(runtime),
                    receipt_fd=receipt_fd,
                    container_id="1" * 64,
                    image_id="sha256:" + "2" * 64,
                    image_digest="sha256:" + "3" * 64,
                    run_id="offline-fixture",
                    base_url="http://127.0.0.1:8888",
                )
                proofs = []

                @contextmanager
                def fixture_server(_args, mode, _plan, _recipe, model, log_fd, extra):
                    arm = next(
                        name for name, m, e in D.ARMS if m == mode and e == extra
                    )
                    world.arm, world.log_fd = arm, log_fd
                    world.log(startup_lines(arm))
                    proof = {
                        "mode": mode,
                        "host_pid": 100 + len(proofs),
                        "start_time_ticks": 5,
                        "offline_test_only": True,
                    }
                    proofs.append(proof)
                    yield proof, read_fd

                capture_class = D.LogCapture
                patches = (
                    mock.patch.object(D, "load_qualifier", return_value=L),
                    mock.patch.object(D, "inspect_container", return_value=({}, {})),
                    mock.patch.object(D, "server", side_effect=fixture_server),
                    mock.patch.object(D, "exchange", side_effect=world.exchange),
                    mock.patch.object(
                        D, "LogCapture", side_effect=lambda descriptor: capture_class(descriptor, settle_seconds=0.0)
                    ),
                )
                try:
                    with ExitStack() as stack:
                        for patch in patches:
                            stack.enter_context(patch)
                        if expect is not None:
                            with self.assertRaises(expect) as caught:
                                D.run(args)
                            if expect_message is not None:
                                self.assertIn(expect_message, str(caught.exception))
                            self.assertEqual(receipt.read_bytes(), b"")
                            self.assertFalse(
                                (
                                    runtime / "long-context/plain-k5/validator.json"
                                ).exists()
                            )
                            return None
                        result = D.run(args)
                    self.assertEqual(json.loads(receipt.read_bytes()), result)
                    if check is not None:
                        check(runtime, result)
                    return result
                finally:
                    os.close(receipt_fd)
        finally:
            os.close(read_fd)
            os.close(write_fd)

    @unittest.skipIf(
        (address_space_limit() or RELEASE_RUN_ADDRESS_SPACE) < RELEASE_RUN_ADDRESS_SPACE,
        "release-geometry mocked run needs ~140 MB of address space; "
        "run this file uncapped (the scaled replays cover the cap)",
    )
    def test_mocked_end_to_end_passes_the_outer_wrapper_and_seals_artifacts(self):
        self.assertEqual(D.NATIVE_CONTEXT, 262_144)
        self.exercise_mocked_run(World(), check=self.check_pass_artifacts, scaled=False)

    def test_scaled_mocked_end_to_end_seals_artifacts(self):
        self.exercise_mocked_run(World(), check=self.check_scaled_pass)

    def check_scaled_pass(self, runtime, result):
        self.assertEqual(D.NATIVE_CONTEXT, SCALED_CONTEXT)
        self.assertEqual(result["evidence"]["max_total_tokens"], SCALED_CONTEXT)
        self.check_run_artifacts(runtime, result)

    def check_pass_artifacts(self, runtime, result):
        evidence = result["evidence"]
        manifest = {
            "_runtime_dir": str(runtime),
            "launch": {"run_id": "offline-fixture"},
        }
        Q._validate_long_context_evidence(evidence, manifest, "1" * 64)
        self.assertEqual(evidence["case_count"], 21)
        validator_path, raw = self.check_run_artifacts(runtime, result)
        self.check_release_tampering(evidence, manifest, validator_path, raw)

    def check_run_artifacts(self, runtime, result):
        evidence = result["evidence"]
        self.assertEqual(evidence["case_count"], len(D.CASE_PROMPT_TOKENS))
        self.assertEqual(evidence["prompt_token_counts"], list(D.CASE_PROMPT_TOKENS))
        validator_path = runtime / evidence["validator_artifact"]
        raw = validator_path.read_bytes()
        self.assertEqual(
            hashlib.sha256(raw).hexdigest(), evidence["validator_receipt_sha256"]
        )
        self.assertEqual(oct(validator_path.stat().st_mode & 0o777), "0o400")
        document = json.loads(raw)
        cases = document["raw"]["cases"]
        self.assertEqual(
            [row["prompt_tokens"] for row in cases], list(D.CASE_PROMPT_TOKENS)
        )
        self.assertEqual(cases[-1]["k5_total_tokens"], D.NATIVE_CONTEXT)
        self.assertIn("ladder-4096", document["raw"]["additional_cases"])
        self.assertEqual(len(document["raw"]["process_bindings"]), 3)
        policy = document["raw"]["qsa_route_policy"]
        self.assertEqual(policy["frame_route_counts"]["sparse-rows-v2"], 0)
        self.assertGreater(policy["frame_route_counts"]["row-serial"], 0)
        admission = document["raw"]["endpoint_admission"]
        self.assertEqual(admission["rejected_262144_plus_1"]["http_status"], 400)
        k5 = document["raw"]["arms"]["k5"]
        self.assertGreater(k5["verification_metrics"]["k5_events"], 0)
        state_comparisons = document["raw"]["target_state_comparisons"]
        self.assertEqual(len(state_comparisons["rollback_vs_plain"]), 3)
        self.assertEqual(
            len(state_comparisons["plain_vs_k5"]),
            sum(len(row["observations"]) for row in k5["results"]),
        )
        endpoint = k5["endpoint_boundary_control"]
        self.assertEqual(
            endpoint["target_state_receipt"]["seq_len"], L.ADMISSION_ACCEPT_CASE_SPEC[1]
        )
        self.assertEqual(endpoint["target_state_receipt"]["mode"], "engaged-native-mtp")
        self.assertEqual(endpoint["request_log_evidence"]["native_mtp_frames"], 0)
        self.assertEqual(endpoint["request_log_evidence"]["completed_spec_cycles"], 0)
        self.assertEqual(
            document["raw"]["arms"]["plain"]["verification_metrics"]["k5_events"], 0
        )
        depth = k5["acceptance_depth"]["reporter-control-2017"]
        self.assertEqual(
            depth["accepted_histogram"]["2"], depth["completed_spec_cycles"]
        )
        continuations = k5["continuations"]
        for row in continuations:
            if not row["skipped"]:
                self.assertEqual(row["target_state_receipt"]["seq_len"], row["prompt_tokens"])
        for arm in document["raw"]["arms"].values():
            for result in arm["results"]:
                self.assertEqual(len(result["observations"]), len(result["target_state_receipts"]))
                for observation, state in zip(result["observations"], result["target_state_receipts"], strict=True):
                    self.assertEqual(state["seq_len"], observation["total_tokens"] - 1)
        self.assertTrue(
            any(
                row["skipped"]
                for row in continuations
                if row["case_id"].startswith("native-endpoint-262080-plus")
            )
        )
        self.assertTrue(
            all(
                not row["skipped"]
                for row in continuations
                if row["case_id"] == "reporter-control-2017"
            )
        )
        journal = document["raw"]["response_journal"]
        journal_raw = (runtime / journal["artifact"]).read_bytes()
        self.assertEqual(hashlib.sha256(journal_raw).hexdigest(), journal["sha256"])
        self.assertEqual(len(journal_raw.splitlines()), journal["requests"])
        for arm in ("plain", "k5", "k5_eager_churn"):
            log = document["raw"]["arms"][arm]["server_log"]
            log_raw = (runtime / log["artifact"]).read_bytes()
            self.assertEqual(hashlib.sha256(log_raw).hexdigest(), log["sha256"])
        bundle = document["raw"]["request_bundle"]
        self.assertEqual(
            hashlib.sha256((runtime / bundle["artifact"]).read_bytes()).hexdigest(),
            bundle["sha256"],
        )
        return validator_path, raw

    def check_release_tampering(self, evidence, manifest, validator_path, raw):
        # Receipt tampering: the wrapper rehashes the sealed validator and
        # recomputes the census from its raw rows.
        os.chmod(validator_path, 0o600)
        validator_path.write_bytes(
            raw.replace(b'"token_identity":true', b'"token_identity":false', 1)
        )
        with self.assertRaises(ValueError):
            Q._validate_long_context_evidence(evidence, manifest, "1" * 64)
        validator_path.write_bytes(raw)
        for tamper in (
            lambda e: e.update(
                prompt_token_counts=e["prompt_token_counts"][:-1] + [262_079]
            ),
            lambda e: e.update(native_k5_engaged=False),
            lambda e: e.update(case_count=20),
            lambda e: e.update(max_total_tokens=262_143),
            lambda e: e.update(validator_receipt_sha256="0" * 64),
        ):
            tampered = json.loads(json.dumps(evidence))
            tamper(tampered)
            with self.subTest(tamper=tamper), self.assertRaises(ValueError):
                Q._validate_long_context_evidence(tampered, manifest, "1" * 64)

    def test_mocked_run_refuses_missing_k5_token_ids(self):
        def mutate(world, body, response):
            if world.arm == "k5" and len(body["prompt_token_ids"]) == 2_017:
                response["choices"][0].pop("token_ids")

        self.exercise_mocked_run(World(mutate), expect=ValueError)

    def test_mocked_run_rejects_equal_but_wrong_fed_digests(self):
        def mutate(_world, _body, _response, row):
            row["fed_tokens_sha256"] = "0" * 64

        self.exercise_mocked_run(
            World(receipt_mutate=mutate), expect=ValueError,
            expect_message="target-fed token digest differs",
        )

    def test_mocked_run_rejects_native_target_state_mismatch(self):
        def mutate(world, body, _response, row):
            if world.arm == "k5" and len(body["prompt_token_ids"]) == 2_017:
                row["target_state_sha256"] = "1" * 64

        self.exercise_mocked_run(
            World(receipt_mutate=mutate), expect=ValueError,
            expect_message="plain/K5/reporter-control-2017: target-state mismatch",
        )

    def test_mocked_run_rejects_continuation_target_state_mismatch(self):
        def mutate(world, body, _response, row):
            if world.arm == "k5" and body["max_tokens"] == 1 and len(body["prompt_token_ids"]) < 3_000:
                row["target_state_sha256"] = "2" * 64

        self.exercise_mocked_run(
            World(receipt_mutate=mutate), expect=ValueError,
            expect_message="continuation/reporter-control-2017: target-state mismatch",
        )

    def test_mocked_run_rejects_full_endpoint_target_state_mismatch(self):
        def mutate(world, body, _response, row):
            if world.arm == "k5" and body["max_tokens"] == 64 and body["ignore_eos"]:
                row["target_state_sha256"] = "3" * 64

        self.exercise_mocked_run(
            World(receipt_mutate=mutate), expect=ValueError,
            expect_message="plain/K5/native-endpoint-262080-plus-64: target-state mismatch",
        )

    def test_mocked_run_rejects_one_token_endpoint_state_mismatch(self):
        def mutate(world, body, _response, row):
            if world.arm == "k5" and len(body["prompt_token_ids"]) == L.ADMISSION_ACCEPT_CASE_SPEC[1]:
                row["target_state_sha256"] = "4" * 64

        self.exercise_mocked_run(
            World(receipt_mutate=mutate), expect=ValueError,
            expect_message="endpoint/plain-K5: target-state mismatch",
        )

    def test_mocked_run_rejects_graph_disabled_boundary_state_mismatch(self):
        def mutate(world, body, _response, row):
            if world.arm == L.EAGER_CHURN_ARM and len(body["prompt_token_ids"]) == 2_047:
                row["target_state_sha256"] = "5" * 64

        self.exercise_mocked_run(
            World(receipt_mutate=mutate), expect=ValueError,
            expect_message="plain/forced-eager/qsa-edge-2047: target-state mismatch",
        )

    def test_mocked_run_rejects_rollback_trigger_state_mismatch(self):
        def mutate(world, body, _response, row):
            key = (world.arm, len(body["prompt_token_ids"]), body["max_tokens"])
            if world.arm == L.EAGER_CHURN_ARM and key[1] == 2_050 and world.state_request_counts[key] == 3:
                row["target_state_sha256"] = "6" * 64

        self.exercise_mocked_run(
            World(receipt_mutate=mutate), expect=ValueError,
            expect_message="rollback/rollback-trigger: target-state mismatch",
        )

    def test_mocked_run_rejects_post_rejection_canary_state_mismatch(self):
        def mutate(world, body, _response, row):
            key = (world.arm, len(body["prompt_token_ids"]), body["max_tokens"])
            if world.arm == "plain" and key[1] == 2_017 and world.state_request_counts[key] == 3:
                row["target_state_sha256"] = "7" * 64

        self.exercise_mocked_run(
            World(receipt_mutate=mutate), expect=ValueError,
            expect_message="post-rejection/plain: target-state mismatch",
        )

    def test_mocked_run_rejects_native_receipt_relabelled_loaded_serial(self):
        def mutate(world, _body, _response, row):
            if world.arm == "k5":
                row.update(mode="loaded-serial", scheduler_path="ordinary-target-only")

        self.exercise_mocked_run(
            World(receipt_mutate=mutate), expect=ValueError,
            expect_message="proposer or scheduler mode differs",
        )

    def test_mocked_run_refuses_different_k5_token_ids(self):
        def mutate(world, body, response):
            if world.arm == "k5" and len(body["prompt_token_ids"]) == 2_043:
                response["choices"][0]["token_ids"][0] ^= 1

        self.exercise_mocked_run(World(mutate), expect=ValueError)

    def test_mocked_run_refuses_mismatched_finish_reason(self):
        def mutate(world, body, response):
            if world.arm == "k5" and len(body["prompt_token_ids"]) == 2_017:
                response["choices"][0]["finish_reason"] = "length"

        self.exercise_mocked_run(World(mutate), expect=ValueError)

    def test_mocked_run_refuses_different_subsequent_target_token(self):
        def mutate(world, body, response):
            if (
                world.arm == "k5"
                and body["ignore_eos"]
                and body["max_tokens"] == 1
                and len(body["prompt_token_ids"]) < 3_000
            ):
                response["choices"][0]["token_ids"] = [1]

        self.exercise_mocked_run(World(mutate), expect=ValueError)

    def test_mocked_run_refuses_absent_native_k5_engagement(self):
        world = World()
        original = world.dispatch

        def silent(prompt_tokens, native, spec=True):
            original(prompt_tokens, False, spec)

        world.dispatch = silent
        self.exercise_mocked_run(world, expect=ValueError)

    def test_mocked_run_refuses_unqualified_sparse_route_telemetry(self):
        self.exercise_mocked_run(World(route="sparse-rows-v2"), expect=ValueError)

    def test_mocked_run_voids_on_fatal_runtime_signal(self):
        def mutate(world, body, response):
            if world.arm == "plain" and len(body["prompt_token_ids"]) == 2_112:
                world.log(
                    [
                        "NVRM: Xid (PCI:0000:01:00): 79, pid=1, GPU has fallen off the bus"
                    ]
                )

        self.exercise_mocked_run(World(mutate), expect=D.Void)

    def test_mocked_run_refuses_error_lines_outside_the_rejection_probe(self):
        def mutate(world, body, response):
            if world.arm == "plain" and len(body["prompt_token_ids"]) == 2_112:
                world.log(["2026-01-01T00:00:00Z ERROR flux_server: unrelated failure"])

        self.exercise_mocked_run(World(mutate), expect=ValueError)

    def exercise_mocked_server(
        self, mode, *, force_cleanup=False, verifier_failure=False,
        setup_failure=None, claim_failure=False, missing_ack=False,
        wait_failure=False, supervisor_forced=False, control_close_failure=False,
        capture_failure=False,
    ):
        read_fd, write_fd = os.pipe()
        out_read, out_write = os.pipe()
        log_read, log_write = os.pipe()
        dead = [False]
        process = mock.Mock()
        process.poll.return_value = None
        process.stdout = os.fdopen(out_read, "rb", buffering=0)
        output = process.stdout
        token = []
        launched = []
        daemon_alive = [False]
        duplicated = []
        threads = []
        original_dup = os.dup
        original_thread = D.threading.Thread

        class Control(io.BytesIO):
            close_failed = False

            def write(self, payload):
                if setup_failure == "control-write":
                    raise OSError("fault-control-write")
                if setup_failure == "control-short-write":
                    return 0
                daemon_alive[0] = True
                return super().write(payload)

            def flush(self):
                if setup_failure == "control-flush":
                    raise OSError("fault-control-flush")
                return super().flush()

            def close(self):
                if not self.closed:
                    if control_close_failure and not self.close_failed:
                        self.close_failed = True
                        raise OSError("control close failed")
                    started = self.getvalue() == b"G"
                    launched.append(started)
                    # The Docker client is not this daemon-owned child's parent.
                    # Only control EOF (or its proven pidfd) can retire it.
                    daemon_alive[0] = False
                    if not missing_ack:
                        try:
                            os.write(out_write, (
                                "R16_EXEC_CLEANUP token=" + token[0]
                                + " child_started=" + str(started).lower()
                                + " retired=true forced="
                                + str(supervisor_forced).lower() + "\n"
                            ).encode("ascii"))
                        except BrokenPipeError:
                            pass
                super().close()

        process.stdin = Control()

        def popen(command, **_kwargs):
            if setup_failure == "popen":
                raise OSError("fault-popen")
            token.append(command[-1])
            return process

        def duplicate(descriptor):
            if setup_failure == "dup":
                raise OSError("fault-dup")
            value = original_dup(descriptor)
            duplicated.append(value)
            return value

        def thread(**kwargs):
            if setup_failure == "thread-construction":
                raise OSError("fault-thread-construction")
            value = original_thread(**kwargs)
            threads.append(value)
            if setup_failure in ("thread-start", "thread-started"):
                original_start = value.start

                def start():
                    if setup_failure == "thread-started":
                        original_start()
                    raise OSError("fault-" + setup_failure)

                value.start = start
                value.join = mock.Mock(wraps=value.join)
            return value

        if setup_failure == "stdout-close":
            process.stdout = mock.Mock(wraps=output)
            process.stdout.closed = False
            attempts = []

            def close():
                if not attempts:
                    attempts.append(True)
                    raise OSError("fault-stdout-close")
                output.close()
                process.stdout.closed = True

            process.stdout.close.side_effect = close

        def wait(**_kwargs):
            if wait_failure:
                raise D.subprocess.TimeoutExpired("offline-docker-client", 20)
            process.poll.return_value = 0
            try:
                os.close(out_write)
            except OSError:
                pass
            return 1 if supervisor_forced else 0

        process.wait.side_effect = wait

        def kill_client():
            process.poll.return_value = -signal.SIGKILL
            try:
                os.close(out_write)
            except OSError:
                pass

        process.kill.side_effect = kill_client
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
                daemon_alive[0] = False

        def claim(*_args):
            if claim_failure:
                raise ValueError("unproven candidate")
            value = original_dup(read_fd)
            duplicated.append(value)
            return value

        with tempfile.TemporaryDirectory() as temp:
            log_fd = os.open(Path(temp) / "arm.log", os.O_RDWR | os.O_CREAT, 0o600)
            quiet = os.open(os.devnull, os.O_WRONLY)
            try:
                with (
                    mock.patch.object(D, "STDERR_FD", quiet),
                    mock.patch.object(
                        D.subprocess, "Popen", side_effect=popen
                    ) as launch,
                    mock.patch.object(D.os, "dup", side_effect=duplicate),
                    mock.patch.object(D.threading, "Thread", side_effect=thread),
                    mock.patch.object(
                        D, "os_write_all",
                        side_effect=OSError("capture write failed") if capture_failure else None,
                        wraps=D.os_write_all,
                    ),
                    mock.patch.object(D, "find_pid", return_value=123) as census,
                    mock.patch.object(
                        D, "claim_pid", side_effect=claim
                    ) as claimed,
                    mock.patch.object(D, "request", return_value=b"{}") as endpoint,
                    mock.patch.object(
                        D.select,
                        "select",
                        side_effect=lambda descriptors, _write, _error, _timeout: (
                            descriptors if dead[0] else [],
                            [],
                            [],
                        ),
                    ),
                    mock.patch.object(
                        D.signal, "pidfd_send_signal", side_effect=sent
                    ) as kill,
                ):
                    plan = {
                        "in_container_binary_path": "/opt/flux/bin/flux-server-real"
                    }
                    recipe = {"ATLAS_QWEN4_VERIFY_GRAPH": "1"}
                    os.write(out_write, b"server line one\n")
                    if (force_cleanup or verifier_failure or setup_failure
                            or claim_failure or missing_ack or wait_failure
                            or supervisor_forced or control_close_failure
                            or capture_failure):
                        expected = (
                            "cleanup" if setup_failure in ("control-write", "control-short-write") else
                            "fault-" + setup_failure if setup_failure else
                            "unproven candidate" if claim_failure else
                            "live binding" if verifier_failure else
                            "forced cleanup" if force_cleanup else
                            "capture failed" if capture_failure else
                            "cleanup"
                        )
                        with self.assertRaisesRegex(
                            (OSError, ValueError), expected,
                        ):
                            with D.server(
                                args, mode, plan, recipe, "model-x", log_fd, {}
                            ):
                                pass
                    else:
                        with D.server(
                            args, mode, plan, recipe, "model-x", log_fd, {}
                        ) as (
                            actual_proof,
                            _pidfd,
                        ):
                            self.assertEqual(actual_proof, proof)
                        self.assertEqual(verifier.call_count, 2)
                    self.assertEqual(
                        endpoint.call_count, 0 if setup_failure or claim_failure else 2
                    )
                    command = launch.call_args.args[0]
                    self.assertIn(
                        "ATLAS_DISABLE_SPECULATION=" + ("1" if mode == "none" else "0"),
                        command,
                    )
                    self.assertIn("ATLAS_MTP_ACCEPT_LOG=1", command)
                    argv = json.loads(base64.b64decode(command[-2], validate=True))
                    self.assertEqual(command[2], "-i")
                    self.assertEqual(launch.call_args.kwargs["stdin"], D.subprocess.PIPE)
                    self.assertEqual(command[-3], D.EXEC_SUPERVISOR)
                    self.assertNotIn("model-x", command)
                    self.assertIn("--no-tui", argv)
                    self.assertEqual(
                        argv[argv.index("--model-from-path") + 1], "/models"
                    )
                    self.assertEqual(
                        argv[argv.index("--model-name") + 1], "model-x"
                    )
                    self.assertNotIn("--mtp-gate", argv)
                    self.assertEqual(
                        "--speculative" in argv, mode == "native-mtp-k5"
                    )
                    if mode == "native-mtp-k5":
                        self.assertEqual(
                            argv[argv.index("--num-drafts") + 1], "4"
                        )
                    self.assertEqual(
                        [call.args[1] for call in kill.call_args_list],
                        [signal.SIGTERM, signal.SIGKILL]
                        if force_cleanup
                        else [] if setup_failure or claim_failure
                        else [signal.SIGTERM],
                    )
                    if setup_failure != "popen":
                        process.wait.assert_called()
                        self.assertTrue(process.stdin.closed)
                        self.assertFalse(daemon_alive[0])
                        self.assertEqual(
                            launched, [not setup_failure or setup_failure == "control-flush"]
                        )
                    if claim_failure:
                        self.assertEqual(claimed.call_count, 1)
                        self.assertEqual(census.call_count, 1)
                    self.assertTrue(os.pread(log_fd, 1000, 0).startswith(b"server line one\n")
                                    or setup_failure == "popen" or capture_failure)
                    for descriptor in duplicated:
                        with self.assertRaises(OSError):
                            os.fstat(descriptor)
                    for value in threads:
                        if value.ident is not None:
                            self.assertFalse(value.is_alive())
                        elif setup_failure == "thread-start":
                            value.join.assert_not_called()
            finally:
                os.close(log_fd)
                os.close(quiet)
                for descriptor in (read_fd, write_fd, log_read, log_write):
                    os.close(descriptor)
                try:
                    os.close(out_write)
                except OSError:
                    pass
                output.close()
                if not process.stdin.closed:
                    io.BytesIO.close(process.stdin)

    def test_mocked_lifecycle_uses_canonical_argv_double_health_and_exact_cleanup(self):
        for mode in ("none", "native-mtp-k5"):
            self.exercise_mocked_server(mode)

    def test_mocked_forced_cleanup_fails_the_gate(self):
        self.exercise_mocked_server("none", force_cleanup=True)

    def test_mocked_binding_failure_still_terminates_the_claimed_server(self):
        self.exercise_mocked_server("none", verifier_failure=True)

    def test_mocked_partial_setup_retires_supervisor_without_starting_engine(self):
        for failure in (
            "popen", "dup", "thread-construction", "stdout-close",
            "thread-start", "thread-started", "control-write",
            "control-short-write", "control-flush",
        ):
            with self.subTest(failure=failure):
                self.exercise_mocked_server("none", setup_failure=failure)

    def test_mocked_failed_claim_uses_control_not_an_unproven_pid(self):
        self.exercise_mocked_server("none", claim_failure=True)

    def test_mocked_lost_cleanup_ack_fails_closed(self):
        self.exercise_mocked_server("none", missing_ack=True)

    def test_mocked_docker_disconnect_is_not_child_retirement_proof(self):
        self.exercise_mocked_server("none", wait_failure=True)

    def test_mocked_supervisor_forced_cleanup_fails_closed(self):
        self.exercise_mocked_server("none", supervisor_forced=True)

    def test_mocked_failed_control_close_is_retried_but_still_fails_closed(self):
        self.exercise_mocked_server("none", control_close_failure=True)

    def test_mocked_log_capture_error_closes_owned_pidfd_and_pump_source(self):
        self.exercise_mocked_server("none", capture_failure=True)

    def exercise_supervisor(
        self, *, before_go=False, child_exited=False, pidfd_failure=False,
        launch_failure=False, forced=False, survived=False,
    ):
        namespace = {"__name__": "offline_supervisor_test"}
        exec(compile(D.EXEC_SUPERVISOR, "<sealed-exec-supervisor>", "exec"), namespace)
        child = mock.Mock(pid=4321)
        child.poll.return_value = None
        if survived:
            child.wait.side_effect = [
                D.subprocess.TimeoutExpired("owned-child", 10),
                D.subprocess.TimeoutExpired("owned-child", 5),
            ]
        elif forced:
            child.wait.side_effect = [D.subprocess.TimeoutExpired("owned-child", 10), 0]
        else:
            child.wait.return_value = 0
        selects = [([77], [], []), ([77], [], [])] if child_exited else [
            ([0], [], []), ([], [], [])
        ]
        captured = io.StringIO()
        with (
            mock.patch.object(D.os, "read", side_effect=[b"" if before_go else b"G", b""]),
            mock.patch.object(D.os, "pidfd_open", return_value=77,
                              side_effect=OSError("pidfd setup failed") if pidfd_failure else None
                              ) as opened,
            mock.patch.object(D.os, "close") as closed,
            mock.patch.object(D.subprocess, "Popen", return_value=child,
                              side_effect=OSError("child launch failed") if launch_failure else None
                              ) as popen,
            mock.patch.object(D.select, "select", side_effect=selects),
            mock.patch.object(D.signal, "pidfd_send_signal") as signalled,
            contextlib.redirect_stdout(captured),
        ):
            expected = (
                D.subprocess.TimeoutExpired if survived else
                OSError if pidfd_failure or launch_failure else None
            )
            if expected:
                with self.assertRaises(expected):
                    namespace["supervise"](["/sealed/engine", "serve"], "a" * 64)
            else:
                result = namespace["supervise"](["/sealed/engine", "serve"], "a" * 64)
                self.assertEqual(result, int(forced))
            if before_go:
                popen.assert_not_called()
                opened.assert_not_called()
                signalled.assert_not_called()
                closed.assert_not_called()
            elif launch_failure:
                child.wait.assert_not_called()
                signalled.assert_not_called()
                closed.assert_not_called()
            elif pidfd_failure:
                child.terminate.assert_called_once_with()
                child.wait.assert_called_once_with(timeout=10)
                signalled.assert_not_called()
                closed.assert_not_called()
            else:
                popen.assert_called_once_with(
                    ["/sealed/engine", "serve"], stdin=D.subprocess.DEVNULL
                )
                opened.assert_called_once_with(child.pid)
                child.poll.assert_not_called()
                child.terminate.assert_not_called()
                child.kill.assert_not_called()
                closed.assert_called_once_with(77)
                self.assertEqual(
                    [call.args for call in signalled.call_args_list],
                    [] if child_exited else
                    [(77, signal.SIGTERM), (77, signal.SIGKILL)] if forced or survived else
                    [(77, signal.SIGTERM)],
                )
            if survived:
                self.assertEqual(captured.getvalue(), "")
            else:
                self.assertEqual(
                    captured.getvalue(), "R16_EXEC_CLEANUP token=" + "a" * 64
                    + " child_started=" + str(not (before_go or launch_failure)).lower()
                    + " retired=true forced=" + str(forced).lower() + "\n"
                )

    def test_supervisor_eof_before_go_never_launches_a_child(self):
        self.exercise_supervisor(before_go=True)

    def test_supervisor_control_eof_reaps_only_its_owned_child(self):
        self.exercise_supervisor()

    def test_supervisor_exited_child_is_reaped_without_a_stale_numeric_signal(self):
        self.exercise_supervisor(child_exited=True)

    def test_supervisor_launch_failure_cannot_claim_a_started_child(self):
        self.exercise_supervisor(launch_failure=True)

    def test_supervisor_pidfd_setup_failure_retires_its_unreaped_direct_child(self):
        self.exercise_supervisor(pidfd_failure=True)

    def test_supervisor_forced_cleanup_reports_failure(self):
        self.exercise_supervisor(forced=True)

    def test_supervisor_surviving_child_cannot_emit_a_retirement_ack(self):
        self.exercise_supervisor(survived=True)

    def exercise_supervisor_signal_faults(
        self, *, term_error=None, kill_error=None, terminate_error=None,
        kill_fallback_error=None, forced=False, survived=False,
        first_wait_error=None,
    ):
        namespace = {"__name__": "offline_supervisor_signal_fault_test"}
        exec(compile(D.EXEC_SUPERVISOR, "<sealed-exec-supervisor>", "exec"), namespace)
        events = []
        child = mock.Mock(pid=4321)
        waits = iter(
            [first_wait_error or D.subprocess.TimeoutExpired("owned-child", 10),
             D.subprocess.TimeoutExpired("owned-child", 5) if survived else 0]
            if forced or survived or first_wait_error else [0]
        )

        def wait(*, timeout):
            events.append(("wait", timeout))
            result = next(waits)
            if isinstance(result, BaseException):
                raise result
            return result

        def send(descriptor, signum):
            self.assertEqual(descriptor, 77)
            events.append(("pidfd-signal", signum))
            error = term_error if signum == signal.SIGTERM else kill_error
            if error is not None:
                raise error

        def fallback(name, error):
            events.append((name,))
            if error is not None:
                raise error

        child.wait.side_effect = wait
        child.terminate.side_effect = lambda: fallback("terminate", terminate_error)
        child.kill.side_effect = lambda: fallback("kill", kill_fallback_error)
        captured = io.StringIO()
        errors = [
            error for error in (
                term_error, terminate_error, first_wait_error,
                kill_error, kill_fallback_error,
            ) if error is not None and not isinstance(error, ProcessLookupError)
        ]
        with (
            mock.patch.object(D.os, "read", side_effect=[b"G", b""]),
            mock.patch.object(D.os, "pidfd_open", return_value=77),
            mock.patch.object(D.os, "close") as closed,
            mock.patch.object(D.subprocess, "Popen", return_value=child),
            mock.patch.object(D.select, "select", side_effect=[([0], [], []), ([], [], [])]),
            mock.patch.object(D.signal, "pidfd_send_signal", side_effect=send),
            contextlib.redirect_stdout(captured),
        ):
            expected = D.subprocess.TimeoutExpired if survived else type(errors[0]) if errors else None
            if expected is not None:
                with self.assertRaises(expected) as raised:
                    namespace["supervise"](["/sealed/engine", "serve"], "a" * 64)
                if not survived:
                    self.assertIs(raised.exception, errors[0])
                self.assertEqual(captured.getvalue(), "")
            else:
                self.assertEqual(
                    namespace["supervise"](["/sealed/engine", "serve"], "a" * 64),
                    int(forced),
                )
                self.assertIn("retired=true forced=" + str(forced).lower(), captured.getvalue())
            closed.assert_called_once_with(77)
        expected_events = [("pidfd-signal", signal.SIGTERM)]
        if term_error is not None and not isinstance(term_error, ProcessLookupError):
            expected_events += [("terminate",)]
        expected_events += [("wait", 10)]
        if forced or survived or first_wait_error:
            expected_events += [("pidfd-signal", signal.SIGKILL)]
            if kill_error is not None and not isinstance(kill_error, ProcessLookupError):
                expected_events += [("kill",)]
            expected_events += [("wait", 5)]
        self.assertEqual(events, expected_events)
        child.poll.assert_not_called()

    def test_supervisor_denied_term_uses_direct_owned_child_then_reaps(self):
        self.exercise_supervisor_signal_faults(term_error=PermissionError("TERM denied"))

    def test_supervisor_denied_kill_uses_direct_owned_child_then_reaps(self):
        self.exercise_supervisor_signal_faults(
            forced=True, kill_error=PermissionError("KILL denied")
        )

    def test_supervisor_denied_term_and_kill_still_attempt_both_bounded_waits(self):
        self.exercise_supervisor_signal_faults(
            forced=True, term_error=PermissionError("TERM denied"),
            kill_error=PermissionError("KILL denied"),
            terminate_error=PermissionError("direct TERM denied"),
            kill_fallback_error=PermissionError("direct KILL denied"),
        )

    def test_supervisor_failed_signal_fallback_cannot_ack_an_unreaped_child(self):
        self.exercise_supervisor_signal_faults(
            survived=True, term_error=PermissionError("TERM denied"),
            kill_error=PermissionError("KILL denied"),
            terminate_error=PermissionError("direct TERM denied"),
            kill_fallback_error=PermissionError("direct KILL denied"),
        )

    def test_supervisor_term_exit_race_still_reaps_without_numeric_fallback(self):
        self.exercise_supervisor_signal_faults(term_error=ProcessLookupError("already exited"))

    def test_supervisor_kill_exit_race_still_reaps_without_numeric_fallback(self):
        self.exercise_supervisor_signal_faults(
            forced=True, kill_error=ProcessLookupError("already exited")
        )

    def test_supervisor_non_timeout_wait_error_still_tries_kill_and_final_wait(self):
        self.exercise_supervisor_signal_faults(
            first_wait_error=OSError("first wait failed"),
            kill_error=PermissionError("KILL denied"),
        )

    def test_symlink_runtime_artifact_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "outside").mkdir()
            (root / "long-context").symlink_to(
                root / "outside", target_is_directory=True
            )
            with self.assertRaises(OSError):
                D.make_directory(S, root, ["long-context", "plain-k5"])


if __name__ == "__main__":
    unittest.main()
