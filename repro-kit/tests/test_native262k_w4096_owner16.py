# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded offline counterexamples for the W4096/owner16 successor.

All container, process, HTTP, model and claim boundaries are mocked or denied.
Fixtures contain synthetic public-format records, never captured host evidence.
"""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def source_module(relative, name):
    path = ROOT / relative
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), "exec", dont_inherit=True), module.__dict__)
    return module


G = source_module("r15-gates/native-262k-w4096-owner16", "candidate_gate")
O = source_module("qwen4_owner16_engagement.py", "public_owner16")
E = source_module("tests/fixtures/qwen4_endpoint_admission.py", "pinned_endpoint")
RECIPE_BYTES = (ROOT / "recipes/flux-qwen4-native-262k-w4096-bf16-owner16.env").read_bytes()

READY_FIELDS = {
    "schema": "atlas.moe-bounded-transpose-owners.v1",
    "selector": "ATLAS_QWEN4_BOUNDED_TRANSPOSE_OWNERS", "owner_count": "87",
    "owner_cap_bytes": "16777216", "payload_bytes": "1415577600",
    "requested_bytes": "1415599872", "guard_and_alignment_bytes": "22272",
    "experts": "512", "projections": "3", "managed": "false",
    "base_unexposed": "true", "down_allocation_after_phase_b": "true",
}


def event(target, message, fields=None, layer=None):
    values = " ".join(key + "=" + (json.dumps(value) if key in
        ("schema", "selector", "mode", "layout") else value)
        for key, value in (fields or {}).items())
    span = "" if layer is None else (
        "bounded_transpose_layer{layer_index=%d layer_ordinal=%d}: " % (layer, layer + 1))
    return ("2026-10-23T00:00:00Z INFO " + span + target + ": " + message
            + " " + values + "\n").encode()


def ready(layer):
    return event("atlas_transposed_expert_slab", "BOUNDED_TRANSPOSE_OWNERS_READY",
                 READY_FIELDS, layer)


def startup_log():
    return b"".join([
        event("atlas_weight_allocation_audit", "FAST_WEIGHT_ROUTED_GROUP_PLAN", {
            "mode": "bounded-per-shard-layer-projection-v1", "group_count": "2304",
            "component_views": "147456", "payload_bytes": "67947724800",
            "requested_bytes": "67948314624", "guard_and_alignment_bytes": "589824",
            "maximum_group_bytes": "67108864", "base_guard_bytes": "256"}),
        event("atlas_weight_allocation_audit", "FAST_WEIGHT_ROUTED_GROUPS_EAGER",
              {"bases": "2304", "bytes": "67948314624"}),
        event("spark_runtime::cuda_backend::bounded_copy", "ENGAGED ATLAS_BOUNDED_HOST_COPY",
              {"timeout_seconds": "60"}),
        event("spark_model::factory::m2_setup", "MoE transpose pass (unified layout): "
              "ATLAS_UNIFIED_MOE_LAYOUT=1, phased transpose with frees, free pre-pass 27.2 GB \u2192 RUNNING"),
        *[ready(layer) for layer in range(48)],
        event("spark_model::factory::m2_setup", "MoE transpose pass (unified layout): done, 28.3 GB free"),
        event("atlas_transposed_expert_slab", "MOE_TRANSPOSE_PASS_MEMORY", {
            "schema": "atlas.moe-transpose-pass-memory.v1", "layout": "unified",
            "free_pre_bytes": "1000", "free_post_bytes": "1100", "free_delta_bytes": "100"}),
        b"Qwen4 QSA physical side cache initialized\n",
    ])


TERMINALS = ((253952, 4048, 258000), (258048, 3840, 261888),
             (258048, 4032, 262080), (258048, 4095, 262143))
COUNTS = {258000: 1, 261888: 1, 262080: 5, 262143: 1}


def chunk_line(start, rows, end, width=4096, tile=10, core="qsa_sparse_bf16_non_bit_exact"):
    return ("QWEN4_PREFILL_QSA_CHUNK_RECEIPT selector=ATLAS_QWEN4_PREFILL_QSA "
            "start=%d rows=%d end=%d chunk_rows=%d tile_rows=%d core=%s\n"
            % (start, rows, end, width, tile, core)).encode()


def qualified_log():
    return startup_log() + b"".join(chunk_line(*item) for item in TERMINALS) + b"".join(
        ("Chunked prefill start: %d prompt tokens,\n" % length).encode()
        for length, count in COUNTS.items() for _ in range(count))


class Offline(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        refused = mock.Mock(side_effect=AssertionError("offline boundary crossed"))
        for name in ("run", "Popen", "call", "check_call", "check_output"):
            self.stack.enter_context(mock.patch.object(G.subprocess, name, refused))
        self.stack.enter_context(mock.patch.object(G.http.client, "HTTPConnection", refused))
        self.stack.enter_context(mock.patch.object(G.os, "open", refused))


class OwnerParserTests(Offline):
    def test_exact_geometry_and_colored_original_hash(self):
        for raw in (startup_log(), startup_log().replace(b"owner_count=", b"\x1b[32mowner_count\x1b[0m=")):
            proof = O.engagement_census(raw)
            self.assertEqual(proof["destination_owners"], 48 * 87)
            self.assertEqual(proof["destination_requested_bytes"], 48 * (1415577600 + 87 * 256))
            self.assertEqual(proof["receipt_fields"], READY_FIELDS)
            self.assertEqual(proof["raw_sha256"], hashlib.sha256(raw).hexdigest())
            self.assertFalse(proof["runtime_qualified"])
            self.assertFalse(proof["release_qualified"])

    def test_each_owner_field_missing_changed_or_duplicated(self):
        for key, value in READY_FIELDS.items():
            encoded = (key + "=" + (json.dumps(value) if key in ("schema", "selector") else value)).encode()
            for altered in (b"", (key + "=wrong").encode(), encoded + b" " + encoded):
                with self.subTest(key=key, altered=altered), self.assertRaises(ValueError):
                    O.engagement_census(startup_log().replace(encoded, altered, 1))

    def test_missing_duplicate_and_unordered_layers(self):
        raw = startup_log()
        cases = (raw.replace(ready(0), b"", 1), raw + ready(0),
                 raw.replace(ready(1), ready(0), 1),
                 raw.replace(ready(0) + ready(1), ready(1) + ready(0), 1),
                 raw.replace(b"layer_index=0 layer_ordinal=1", b"layer_index=0 layer_ordinal=2", 1),
                 raw.replace(b"layer_index=47 layer_ordinal=48", b"layer_index=48 layer_ordinal=49", 1),
                 raw.replace(b"layer_index=0", b"layer_index=00", 1))
        for data in cases:
            with self.subTest(size=len(data)), self.assertRaises(ValueError):
                O.engagement_census(data)

    def test_source_group_cap_counts_and_phase_order(self):
        raw = startup_log()
        mutations = ((b"maximum_group_bytes=67108864", b"maximum_group_bytes=16777216"),
                     (b"group_count=2304", b"group_count=2303"),
                     (b"bases=2304", b"bases=2305"),
                     (b"free_delta_bytes=100", b"free_delta_bytes=99"),
                     (b"owner_count=87", b"owner_count=24"),
                     (b"requested_bytes=1415599872", b"requested_bytes=1415583744"))
        for old, new in mutations:
            with self.subTest(old=old), self.assertRaises(ValueError):
                O.engagement_census(raw.replace(old, new, 1))
        lines = raw.splitlines(keepends=True)
        with self.assertRaises(ValueError):
            O.engagement_census(b"".join([lines[1], lines[0], *lines[2:]]))

    def test_truncation_invalid_encodings_and_legacy_mode(self):
        raw = startup_log()
        for data in (b"", raw[:-1], raw + b"\x00\n", raw + b"\x1b[2J\n", raw + b"\xff\n",
                     raw + b"MOE_TRANSPOSE_ROUTED_SLAB_READY\n", bytearray(raw)):
            with self.subTest(kind=type(data)), self.assertRaises((ValueError, UnicodeError)):
                O.engagement_census(data)


class SnapshotTests(Offline):
    def test_original_bytes_preserved_for_one_and_two_streams(self):
        raw = startup_log()
        split = raw.index(b"\n") + 1
        for parts in ([raw], [raw, b""], [b"", raw], [raw[:split], raw[split:]]):
            self.assertEqual(G.engagement_snapshot(parts), raw)
            self.assertEqual(O.engagement_census(G.engagement_snapshot(parts))["target_receipts"], 48)

    def test_old_separator_join_counterexample_is_rejected(self):
        parts = [startup_log()[:-1], b""]
        self.assertEqual(O.engagement_census(b"\n".join(parts))["target_receipts"], 48)
        with self.assertRaisesRegex(ValueError, "complete original"):
            G.engagement_snapshot(parts)

    def test_each_partial_stream_types_and_aggregate_bound(self):
        for parts in ([b"x", b"done\n"], [b"done\n", b"x"], [], (), [b"", b""],
                      [b"ok\n"] * 3, [bytearray(b"ok\n")]):
            with self.subTest(parts=repr(parts)[:60]), self.assertRaises(ValueError):
                G.engagement_snapshot(parts)
        half = b"\n" * ((16 << 20) + 1)
        with self.assertRaisesRegex(ValueError, "bound"):
            G.engagement_snapshot([half, half])


def recipe_environment():
    rows = [line.split(b"=", 1) for line in RECIPE_BYTES.splitlines()
            if line and not line.startswith(b"#")]
    return rows + [[b"ATLAS_DISABLE_SPECULATION", b"1"]]


def native_argv():
    return [b"flux-server", b"serve", b"--qwen4-qsa", *sum(([
        option.encode(), value.encode()] for option, value in (
        ("--max-seq-len", "262144"), ("--kv-cache-dtype", "bf16"),
        ("--max-num-seqs", "1"), ("--max-batch-size", "1"),
        ("--max-prefill-tokens", "4096"), ("--block-size", "16"),
        ("--enable-prefix-caching", "false"), ("--ssm-cache-slots", "0"),
        ("--swap-space-gb", "0"), ("--kernel-target", "qwen3.8-flash-next"))), [])]


class LiveControlTests(Offline):
    def controls(self, rows=None, argv=None):
        env = recipe_environment() if rows is None else rows
        argsv = native_argv() if argv is None else argv
        def read(path, *unused):
            return (b"\0".join(key + b"=" + value for key, value in env) + b"\0"
                    if path.name == "environ" else b"\0".join(argsv) + b"\0")
        def values(items, option):
            return [items[i + 1] for i, item in enumerate(items[:-1]) if item == option] + [
                item.split(b"=", 1)[1] for item in items if item.startswith(option + b"=")]
        suite = types.SimpleNamespace(_read_process_bytes=read, _live_cli_values=values,
            _open_process_file=lambda *args: 81,
            _hash_regular_fd=lambda *args: (G.TOKENIZER_SHA256, b""))
        with mock.patch.object(G.os, "open", return_value=80), mock.patch.object(G.os, "close"):
            return G.native_process(types.SimpleNamespace(suite=suite), {"host_pid": 31415})

    def test_complete_recipe_pin_and_owner_aliases_are_accepted(self):
        self.assertEqual(hashlib.sha256(RECIPE_BYTES).hexdigest(), G.RECIPE_SHA256)
        rows = recipe_environment()
        self.assertEqual(len(rows), len({key for key, value in rows}))
        self.assertEqual(self.controls()["controls"]["QWEN4_BOUNDED_TRANSPOSE_OWNER_CAP_MIB"], "16")

    def test_conflicting_or_missing_aliases_fail_for_each_contract(self):
        for suffix in ("CTX_OVERFLOW_TRUNCATE", "QWEN4_PREFILL_WIDE", "QWEN4_PREFILL_QSA_FP8",
                       "FAST_LOAD_ROUTED_GROUPS", "FAST_LOAD_ROUTED_GROUPS_EAGER",
                       "QWEN4_BOUNDED_TRANSPOSE_OWNERS", "QWEN4_BOUNDED_TRANSPOSE_OWNER_CAP_MIB"):
            for prefix in ("ATLAS_", "FLUX_"):
                key = (prefix + suffix).encode()
                for changed in (None, b"wrong"):
                    rows = [[k, (changed if k == key else v)] for k, v in recipe_environment()
                            if changed is not None or k != key]
                    with self.subTest(key=key, changed=changed), self.assertRaises(ValueError):
                        self.controls(rows)

    def test_duplicate_environment_and_duplicate_cli_fail(self):
        rows = recipe_environment()
        for duplicated in ([*rows, rows[0]], [*rows, [rows[0][0], b"wrong"]]):
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                self.controls(duplicated)
        with self.assertRaises(ValueError):
            self.controls(argv=native_argv() + [b"--max-prefill-tokens", b"16384"])

    def test_auto_compact_bare_and_equals_forms_fail(self):
        for extra in ([b"--auto-compact"], [b"--auto-compact", b"0.5"], [b"--auto-compact=0.5"]):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.controls(argv=native_argv() + extra)


class RequestBoundaryTests(Offline):
    def test_observed_journals_before_send_and_keeps_raw_error_evidence(self):
        rows, sent = [], []
        raw = b'{"error":{"message":"fixture"}}'
        response = types.SimpleNamespace(status=400, read=lambda limit: raw,
            getheaders=lambda: [("Content-Length", str(len(raw))), ("x-request-id", "fixture")])
        def request(method, route, **kwargs):
            self.assertEqual(rows[-1]["event"], "request")
            self.assertEqual(rows[-1]["request"], {"max_tokens": 1})
            sent.append((method, route, kwargs))
        connection = types.SimpleNamespace(request=request, getresponse=lambda: response, close=mock.Mock())
        suite = types.SimpleNamespace(canonical_bytes=lambda value: json.dumps(value).encode())
        args = types.SimpleNamespace(base_url="http://127.0.0.1:8888", suite=suite)
        with mock.patch.object(G.http.client, "HTTPConnection", return_value=connection):
            status, body, headers = G.observed(args, types.SimpleNamespace(append=rows.append),
                "overflow-fixture", "POST", "/v1/completions", {"max_tokens": 1}, "fixture")
        self.assertEqual((status, body), (400, raw))
        self.assertEqual([row["event"] for row in rows], ["request", "response"])
        self.assertEqual(rows[1]["headers"], headers)
        self.assertEqual(rows[1]["body_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(sent[0][2]["headers"]["x-request-id"], "fixture")
        connection.close.assert_called_once()

    def test_observed_rejects_ambiguous_or_wrong_content_length_after_journaling(self):
        for headers in ([("Content-Length", "3")],
                        [("Content-Length", "2"), ("content-length", "2")]):
            rows = []
            response = types.SimpleNamespace(status=200, read=lambda limit: b"{}", getheaders=lambda: headers)
            connection = types.SimpleNamespace(request=lambda *args, **kwargs: None,
                getresponse=lambda: response, close=mock.Mock())
            args = types.SimpleNamespace(base_url="http://127.0.0.1:8888",
                suite=types.SimpleNamespace(canonical_bytes=lambda value: json.dumps(value).encode()))
            with mock.patch.object(G.http.client, "HTTPConnection", return_value=connection), \
                    self.assertRaisesRegex(ValueError, "ambiguous length"):
                G.observed(args, types.SimpleNamespace(append=rows.append), "fixture", "GET", "/health")
            self.assertEqual([row["event"] for row in rows], ["request", "response"])
            connection.close.assert_called_once()

    def test_unregistered_sealed_suite_is_rejected_before_other_helpers(self):
        argv = [value for key, item in {
            "container-id": "c" * 64, "image-id": "sha256:" + "a" * 64,
            "image-digest": "sha256:" + "b" * 64, "base-url": "http://127.0.0.1:8888",
            "run-id": "offline", "receipt-fd": "10", "raw-log-fd": "11",
            "runtime-dir": "/fixture", "suite-program": "/fixture/suite.py", "suite-sha256": "d" * 64,
            "lifecycle-program": "/fixture/lifecycle.py", "endpoint-program": "/fixture/endpoint.py",
            "engagement-program": "/fixture/engagement.py"}.items() for value in ("--" + key, item)]
        loader = mock.Mock(return_value=types.SimpleNamespace(verify_parent_suite=lambda *args: None,
            load_module=lambda *args: types.SimpleNamespace(REQUIRED_GATES=("native-262k",))))
        with mock.patch.object(G, "load_checked", loader), \
                self.assertRaisesRegex(ValueError, "explicit sealed-suite registration"):
            G.parse_args(argv)
        self.assertEqual(loader.call_count, 1)


class LogContractTests(Offline):
    def test_all_four_terminal_starts_and_full_census(self):
        proof = G.validate_logs(E, qualified_log(), COUNTS)
        self.assertTrue(proof)

    def test_missing_special_258000_receipt_and_old_width_fail(self):
        raw = qualified_log()
        for changed in (raw.replace(chunk_line(*TERMINALS[0]), b"", 1),
                        raw.replace(chunk_line(*TERMINALS[0]), chunk_line(258048, 4048, 258000), 1),
                        raw.replace(b"chunk_rows=4096", b"chunk_rows=16384", 1),
                        raw.replace(b"start=253952", b"start=245760", 1)):
            with self.assertRaises(ValueError):
                G.validate_logs(E, changed, COUNTS)

    def test_each_terminal_tuple_and_operand_policy_is_enforced(self):
        raw = qualified_log()
        for start, rows, end in TERMINALS:
            for altered in (chunk_line(start, rows - 1, end), chunk_line(start, rows, end + 1),
                            chunk_line(start, rows, end, tile=11),
                            chunk_line(start, rows, end, core="qsa_sparse_e4m3_qkpv")):
                with self.subTest(end=end), self.assertRaises(ValueError):
                    G.validate_logs(E, raw.replace(chunk_line(start, rows, end), altered, 1), COUNTS)

    def test_extra_missing_and_rejected_dispatches_fail(self):
        raw = qualified_log()
        for changed in (raw + b"Prefilling: 262144 prompt tokens,\n",
                        raw + b"Prefilling: 42 prompt tokens,\n",
                        raw.replace(b"Chunked prefill start: 258000 prompt tokens,\n", b"", 1)):
            with self.assertRaises(ValueError):
                G.validate_logs(E, changed, COUNTS)

    def test_raw_error_oracle_and_rejection_metrics_counterexamples(self):
        expected = {"error": {"message": "Context too long: 262144 prompt + 1 requested output tokens "
                    "exceeds max_seq_len 262144", "type": "invalid_request_error", "param": None, "code": None}}
        self.assertTrue(E.validate_context_length_rejection(400, expected, 262144, 1, 262144))
        for status, value in ((200, expected), (400, {"error": {**expected["error"], "param": "messages"}}),
                              (400, {"error": {**expected["error"], "code": "context_length_exceeded"}})):
            with self.assertRaises(ValueError):
                E.validate_context_length_rejection(status, value, 262144, 1, 262144)
        metric = {"values": {name: 100 for name in E.METRIC_NAMES}}
        request_id = "qualify_overflow_" + "a" * 32
        for marker in ("Chunked prefill start:", "Prefilled:", "MTP_ACCEPT frame",
                       "SPEC_CYCLE_V2 ", "Done:", "req=0123456789abcdef"):
            with self.subTest(marker=marker), self.assertRaises(ValueError):
                E.validate_no_scheduler_dispatch(marker, metric, metric, request_id)
        moved = {"values": {**metric["values"], E.METRIC_NAMES[0]: 101}}
        with self.assertRaises(ValueError):
            E.validate_no_scheduler_dispatch("", metric, moved, request_id)


class FakeRuntime:
    def stat(self):
        return types.SimpleNamespace(st_uid=G.os.getuid(), st_mode=0o700)

    def __truediv__(self, child):
        return self


class MatrixTests(Offline):
    def run_matrix(self, canary_chunks=None, corrupt_overflow=False, gate_module=None,
                   native_recipe_loader=None, image_inspector=None):
        G = gate_module or globals()["G"]
        canary_chunks = canary_chunks if canary_chunks is not None else 262144 // G.CHUNK
        bindings = source_module("r15-container-suite.py", "matrix_suite_bindings")
        image_plan = {"in_container_binary_path": "/opt/flux/bin/flux-server-real",
            "in_container_binary_sha256": "1" * 64,
            "in_container_recipe_path": "/opt/flux/repro-kit/recipes/release.env",
            "recipe_sha256": "2" * 64}
        launch_plan = {**image_plan, "schema": bindings.PLAN_SCHEMA_V4,
            "native_context_contract": {"schema": bindings.NATIVE_CONTEXT_SCHEMA,
                                        "gates": bindings.NATIVE_CONTEXT_RECIPES}}
        def select_recipe(args, image, gate):
            selected = bindings.native_context_recipe_plan(launch_plan, image, gate)
            return selected, {"ATLAS_QWEN4_PREFILL_WIDE": str(G.CHUNK)}
        state = {"log": startup_log(), "calls": [], "tokens": 0, "chunks": 0,
                 "storage": {}, "next_fd": 200}
        runtime = FakeRuntime()
        def allocate(*args):
            state["next_fd"] += 1
            state["storage"][state["next_fd"]] = bytearray()
            return state["next_fd"]
        def write(fd, data):
            state["storage"].setdefault(fd, bytearray()).extend(data)
        suite = types.SimpleNamespace(validate_inherited_receipt_fd=lambda *args: None,
            load_native_context_recipe=native_recipe_loader or select_recipe,
            normalized_absolute=lambda *args: runtime, directory_identity_sha256=lambda *args: "f" * 64,
            open_directory_chain=allocate, create_beneath=allocate, open_beneath=allocate,
            write_all=write, assert_path_identity=lambda *args: None,
            canonical_bytes=lambda obj: json.dumps(obj, sort_keys=True).encode(),
            strict_json_loads=lambda raw, *args: json.loads(raw), MAX_GATE_RECEIPT_BYTES=1 << 20)
        class Streams:
            def snapshot(self):
                return [state["log"], b""]
            def close(self):
                pass
        class Journal:
            def __init__(self, *args):
                self.fd, self.path = 150, "fixture/responses.jsonl"
            def seal(self):
                return {"artifact": self.path, "mock": True}
        @contextlib.contextmanager
        def server(*args):
            yield "fixture-model", {"host_pid": 31415}, 99
        @contextlib.contextmanager
        def kernel(*args):
            yield {"mock": True}
        answer = "ALPHA: alpha-0731826\nBETA: beta-0492175\nGAMMA: gamma-0863204"
        def observed(args, journal, label, method, route, body=None, request_id=None):
            if route == "/tokenize":
                return 200, b'{"tokens":[7],"count":1}', []
            if route == "/metrics":
                raw = ("atlas_prefill_progress_chunks_total %d\n"
                       "atlas_prefill_progress_tokens_total %d\n" % (state["chunks"], state["tokens"]))
                return 200, raw.encode(), []
            if route == "/health":
                return 200, b"{}", []
            length, budget, forced = len(body["prompt_token_ids"]), body["max_tokens"], body["ignore_eos"]
            state["calls"].append((label, length, budget, forced))
            if length == 262144:
                error = {"error": {"message": "Context too long: 262144 prompt + 1 requested output tokens "
                    "exceeds max_seq_len 262144", "type": "invalid_request_error", "param": None, "code": None}}
                if corrupt_overflow:
                    error["error"]["param"] = "messages"
                return 400, json.dumps(error).encode(), [("x-request-id", request_id)]
            state["log"] += ("Chunked prefill start: %d prompt tokens,\n" % length).encode()
            start = (length // G.CHUNK) * G.CHUNK
            state["log"] += chunk_line(start, length - start, length, width=G.CHUNK)
            state["tokens"] += length
            state["chunks"] += (canary_chunks if label.startswith("post-overflow") else
                                (length + G.CHUNK - 1) // G.CHUNK)
            count = budget if forced else 2
            response = {"model": "fixture-model", "id": "response-%d" % len(state["calls"]),
                "interventions": G.LEDGER, "choices": [{"index": 0, "text": answer if budget > 1 else "x",
                    "token_ids": [42] * count, "finish_reason": "length" if forced else "stop"}],
                "usage": {"prompt_tokens": length, "completion_tokens": count, "total_tokens": length + count,
                    "prompt_tokens_details": {"cached_tokens": 0}}}
            return 200, json.dumps(response).encode(), []
        def token_hash(values):
            return G.EXPECTED_PROMPT_SHA256.get(len(values), "fixture-%d" % len(values))
        lifecycle = types.SimpleNamespace(inspect_container=image_inspector or (
            lambda *args: (dict(image_plan), {"DEFAULT_RELEASE_CONTROL": "fixture"})),
                                          make_directory=lambda *args: None)
        args = types.SimpleNamespace(suite=suite, lifecycle=lifecycle, endpoint=E, engagement=O,
            receipt_fd=10, raw_log_fd=11, runtime_dir="fixture", container_id="c" * 64,
            image_id="sha256:" + "a" * 64, image_digest="sha256:" + "b" * 64, run_id="offline-fixture")
        with contextlib.ExitStack() as stack:
            for name, replacement in (("Streams", Streams), ("Journal", Journal), ("server", server),
                ("kernel_window", kernel), ("observed", observed), ("token_hash", token_hash),
                ("assemble", lambda length, components: ([7] * length, {})),
                ("native_process", lambda *args: {"mock": True})):
                stack.enter_context(mock.patch.object(G, name, replacement))
            stack.enter_context(mock.patch.object(G.os, "fstat", return_value=runtime.stat()))
            stack.enter_context(mock.patch.object(G.os, "fsync"))
            stack.enter_context(mock.patch.object(G.os, "close"))
            stack.enter_context(mock.patch.object(G.select, "select", return_value=([], [], [])))
            stack.enter_context(mock.patch.object(G.time, "sleep"))
            receipt = G.run(args)
        return receipt, state

    def test_real_orchestration_preserves_all_nine_cases_and_64_chunk_canary(self):
        receipt, state = self.run_matrix()
        self.assertEqual(state["calls"], [
            ("retrieval-258000", 258000, 96, False), ("retrieval-261888", 261888, 96, False),
            ("retrieval-262080", 262080, 64, False), ("retrieval-262080-repeat", 262080, 64, False),
            ("262080-plus-64", 262080, 64, True), ("262080-plus-1", 262080, 1, True),
            ("262143-plus-1", 262143, 1, True), ("262144-plus-1-rejection", 262144, 1, True),
            ("post-overflow-262080-plus-1", 262080, 1, True)])
        self.assertEqual(receipt["gate"], "native-262k-w4096-owner16")
        self.assertEqual(receipt["evidence"]["prefill_chunk_tokens"], 4096)

    def test_old_16_chunk_canary_cannot_satisfy_real_orchestration(self):
        with self.assertRaisesRegex(ValueError, "canary metric deltas"):
            self.run_matrix(canary_chunks=16)

    def test_chat_error_schema_cannot_satisfy_raw_orchestration(self):
        with self.assertRaisesRegex(ValueError, "unexpected reason"):
            self.run_matrix(corrupt_overflow=True)


if __name__ == "__main__":
    allowed = {"OwnerParserTests", "SnapshotTests", "LiveControlTests", "RequestBoundaryTests",
               "LogContractTests", "MatrixTests"}
    raw_args = sys.argv[1:]
    evidence_override = None
    selected = []
    idx = 0
    while idx < len(raw_args):
        if raw_args[idx] == "--evidence-dir":
            if idx + 1 >= len(raw_args):
                raise SystemExit("--evidence-dir requires a directory path")
            evidence_override = Path(raw_args[idx + 1])
            idx += 2
        else:
            selected.append(raw_args[idx])
            idx += 1
    if not selected or any(name not in allowed for name in selected):
        raise SystemExit("Choose explicit bounded test classes: " + ", ".join(sorted(allowed)))
    suite = unittest.defaultTestLoader.loadTestsFromNames(selected, sys.modules[__name__])
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    evidence = (
        evidence_override
        if evidence_override is not None
        else Path(tempfile.mkdtemp(prefix="native262k-offline-checks-"))
    )
    evidence.mkdir(parents=True, exist_ok=True)
    label = "-".join(selected)
    attempt = 1
    while (evidence / (label + ".attempt%d.json" % attempt)).exists():
        attempt += 1
    prefix = evidence / (label + ".attempt%d" % attempt)
    source_paths = [ROOT / "r15-gates/native-262k-w4096-owner16", ROOT / "qwen4_owner16_engagement.py",
                    ROOT / "tests/fixtures/qwen4_endpoint_admission.py", Path(__file__).resolve()]
    record = {"schema": "flux.native262k-w4096.offline-test/1", "selected_classes": selected,
              "attempt": attempt, "tests_run": result.testsRun, "failures": len(result.failures),
              "errors": len(result.errors), "passed": result.wasSuccessful(),
              "runtime_qualified": False, "release_qualified": False,
              "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in source_paths}}
    prefix.with_suffix(prefix.suffix + ".log").write_text(output.getvalue())
    prefix.with_suffix(prefix.suffix + ".json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(output.getvalue(), end="")
    print(json.dumps(record, sort_keys=True))
    raise SystemExit(0 if result.wasSuccessful() else 1)
