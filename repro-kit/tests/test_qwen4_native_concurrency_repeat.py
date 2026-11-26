# SPDX-License-Identifier: AGPL-3.0-only
"""Offline tests for the native-concurrency repeat diagnostic.

No server, network or GPU. A fake probe appends the probe's receipt grammar to
a real log file; the real runner writes records and windows; the real offline
exporter recomputes the identity data from those raw files. Divergence is
data, never a failure; only evidence gaps make an export DATA_INCOMPLETE.
"""

import ast
import hashlib
import importlib.machinery
import importlib.util
import json
from pathlib import Path
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


R = module(ROOT / "qwen4_native_concurrency_repeat.py", "qwen4_native_concurrency_repeat")
A = R.D  # the offline half: census, recomputation and export schema
PROBE_SOURCE = (ROOT / "qwen4_native_concurrency_probe.py").read_text()
MAX_TOKENS = 16


def probe_constant(name):
    """Literal value of a probe module constant (re.compile pattern or tuple)."""
    for node in ast.parse(PROBE_SOURCE).body:
        if isinstance(node, ast.Assign) and [t.id for t in node.targets if isinstance(t, ast.Name)] == [name]:
            value = node.value
            if isinstance(value, ast.Call):
                value = value.args[0]
            return ast.literal_eval(value)
    raise AssertionError(name)


def session_of(label):
    return hashlib.sha256(label.encode()).hexdigest()[:16]


def tokens_of(label):
    return [(1000 + 7 * int(label[-2:]) + index) for index in range(MAX_TOKENS)]


class FakeCursor:
    def __init__(self, path):
        self.path = path

    def mark(self):
        return self.path.stat().st_size

    def read_bytes_from(self, offset, settle_seconds=0.15, end_offset=None):
        raw = self.path.read_bytes()
        return raw[offset:end_offset]

    def read_from(self, offset, settle_seconds=0.15, end_offset=None):
        return self.read_bytes_from(offset, settle_seconds, end_offset).decode()


class FakeProbe:
    """Lockstep engine with hostile knobs; writes the probe's log grammar."""

    FORBIDDEN_LOG_MARKERS = A.FORBIDDEN_LOG_MARKERS

    def __init__(self, log, route="native", **knobs):
        self.log, self.route, self.knobs = log, route, knobs
        self.wave_index = -1
        self.c1_calls = 0
        self.LogCursor = FakeCursor

    def append(self, lines):
        with self.log.open("a") as handle:
            handle.write("".join(f"2026-10-23T00:00:00Z INFO spark: {line}\n" for line in lines))

    def make_cases(self, width):
        return [types.SimpleNamespace(label=f"row-{i:02d}", ordinal=i, prompt_sha256="0" * 64)
                for i in range(width)]

    def logits_line(self, session, index, salt=""):
        digest = hashlib.sha256(f"{session}:{index}:{salt}".encode()).hexdigest()
        return f"LOGITS_FNV session={session} idx={index} fnv={digest[:16]} sha256={digest}"

    def result(self, case, ids):
        return types.SimpleNamespace(
            label=case.label, token_ids=ids, finish_reason="length",
            usage={"prompt_tokens": 40 + case.ordinal, "completion_tokens": MAX_TOKENS},
        )

    def stream_request(self, base_url, model, case, max_tokens, timeout):
        self.c1_calls += 1
        session = session_of(case.label)
        ids = tokens_of(case.label)
        flip = self.knobs.get("c1_flip")
        if flip and self.c1_calls > len(self.knobs.get("width_cases", [])) and case.label == flip:
            ids = ids[:5] + [7] + ids[6:]
        self.append([self.logits_line(session, index) for index in range(1, max_tokens)])
        return self.result(case, ids)

    def frame(self, route, sessions, slots, step):
        return (f"QWEN4_{route.upper()}_BATCH_FRAME rows={len(sessions)} sessions={','.join(sessions)} "
                f"slots={','.join(map(str, slots))} pre_step_seq_lens={','.join(str(40 + step) for _ in sessions)} "
                + ("active_only=true target_only=true qsa=false graph=false" if route == "native"
                   else "selector_on=false qsa_required=false graph=false"))

    def run_wave(self, base_url, model, cases, max_tokens, timeout):
        self.wave_index += 1
        sessions = [session_of(case.label) for case in cases]
        slots = list(range(len(cases)))
        lines = []
        for index in range(1, max_tokens):
            route = self.route
            if self.knobs.get("collapse") == (self.wave_index, index):
                route = "serial" if route == "native" else "native"
            step_slots = list(slots)
            if self.knobs.get("slot_move") == (self.wave_index, index):
                step_slots = step_slots[::-1]
            lines.append(self.frame(route, sessions, step_slots, index))
            for case, session in zip(cases, sessions):
                salt = "x" if self.knobs.get("logits_flip") == (self.wave_index, case.label, index) else ""
                lines.append(self.logits_line(session, index, salt))
        self.append(lines)
        results = []
        for case in cases:
            ids = tokens_of(case.label)
            flip = self.knobs.get("token_flip")
            if flip and flip[:2] == (self.wave_index, case.label):
                ids = ids[: flip[2]] + [9] + ids[flip[2] + 1:]
            results.append(self.result(case, ids))
        return results

    def process_start_ticks(self, pid):
        if self.knobs.get("replaced") and self.wave_index >= 0:
            return 2
        return 1

    def validate_log_binding(self, path, pid):
        return (0, 0)

    def healthcheck(self, base_url, timeout):
        if self.knobs.get("health_fails"):
            raise RuntimeError("health returned HTTP 503")


def args_for(base, width, waves=A.MIN_WAVES, profile="native-reference"):
    return types.SimpleNamespace(
        base_url="http://127.0.0.1:18899", model="qwen38-native-repeat", width=width,
        selector_profile=profile, waves=waves, max_tokens=MAX_TOKENS, timeout=30.0,
        probe_sha256="a" * 64, server_pid=4242, server_log=str(base / f"c{width}.server.log"),
        out_dir=str(base / f"c{width}"),
    )


def run_one(base, width, waves=A.MIN_WAVES, route="native", profile="native-reference", log_prefix="", **knobs):
    args = args_for(base, width, waves, profile)
    log = Path(args.server_log)
    log.write_text(log_prefix)
    probe = FakeProbe(log, route, width_cases=list(range(width)), **knobs)
    out = R.make_out_dir(args.out_dir)
    records = R.run_width(args, probe)
    raw = A.canonical(records) + b"\n"
    A.write_new(out / "records.json", raw)
    return out, A.sha(raw), records


class RepeatTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        # Durability is not under test; fsync on tmpfs dominates the time budget.
        patcher = mock.patch.object(R.os, "fsync", lambda descriptor: None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self.temp.cleanup()

    def three(self, per_width=None):
        per_width = per_width or {}
        return [run_one(self.base, width, **per_width.get(width, {}))[:2] for width in A.WIDTHS]

    def test_grammar_and_markers_equal_the_probe(self):
        self.assertEqual(A.FRAME_RE.pattern, probe_constant("FRAME_RE"))
        self.assertEqual(A.LOGITS_RE.pattern, probe_constant("LOGITS_RE"))
        self.assertEqual(A.FORBIDDEN_LOG_MARKERS, probe_constant("FORBIDDEN_LOG_MARKERS"))
        self.assertEqual(A.RECEIPT_MARKERS, probe_constant("RECEIPT_MARKERS"))

    def test_lane03_qsa_frame_fails_closed_and_plain_frame_parses(self):
        # `qsa=true routes=..` frames need a row that can reach the 2,048-token
        # QSA gate, which the probe's plan never admits; one is a scope breach.
        qsa = (
            b"QWEN4_NATIVE_BATCH_FRAME rows=2 sessions=00000000000000aa,00000000000000bb "
            b"slots=0,1 pre_step_seq_lens=2047,9 active_only=true target_only=true "
            b"qsa=true routes=i,n graph=false\n"
        )
        with self.assertRaisesRegex(A.RepeatError, "malformed qualification evidence"):
            A.parse_window(qsa)
        frames, _ = A.parse_window(qsa.replace(b"qsa=true routes=i,n", b"qsa=false"))
        self.assertEqual([frame["route"] for frame in frames], ["native"])

    def complete(self, per_width=None):
        record = A.export(self.three(per_width))
        self.assertEqual(record["status"], A.DATA_COMPLETE, record["problems"])
        self.assertEqual(A.validate_export(record), [])
        self.assertIs(record["gating"], False)
        self.assertIs(record["release_qualified"], False)
        return record

    def test_identical_runs_export_full_identity_and_never_gate(self):
        record = self.complete()
        c8 = record["widths"]["8"]
        self.assertEqual((c8["rows_compared"], c8["fully_identical_rows"]), (8 * A.MIN_WAVES, 8 * A.MIN_WAVES))
        self.assertEqual(c8["full_identity_rate"], 1.0)
        self.assertEqual(record["widths"]["4"]["per_slot"]["3"],
                         {"rows": 20, "token_diverged": 0, "logits_diverged": 0})
        self.assertLess(record["widths"]["2"]["miss_probability_at_lf009_rate"], 0.12)
        self.assertEqual(record["cross_process_c1"], {"labels_compared": 6, "diverged": []})
        self.assertEqual([item["width"] for item in record["inputs"]], [2, 4, 8])

    def test_token_flip_is_recorded_as_data_with_slot_and_window(self):
        record = self.complete({4: {"token_flip": (13, "row-02", 9)}})
        c4 = record["widths"]["4"]
        self.assertEqual(c4["token_identical_rows"], 79)
        self.assertEqual(c4["token_identity_rate"], 79 / 80)
        self.assertEqual(c4["first_divergence_histogram"], {"9": 1})
        self.assertEqual(c4["per_wave_diverged_rows"][13], 1)
        row = c4["first_divergences"][0]
        self.assertEqual((row["wave"], row["label"], row["slot"], row["first_divergence"]), (13, "row-02", 2, 9))
        self.assertEqual(row["row_tokens"][row["first_divergence"] - row["window"][0]], 9)
        self.assertEqual(c4["per_slot"]["2"], {"rows": 20, "token_diverged": 1, "logits_diverged": 0})

    def test_logits_only_flip_is_recorded(self):
        c8 = self.complete({8: {"logits_flip": (19, "row-07", 11)}})["widths"]["8"]
        self.assertEqual((c8["token_identity_rate"], c8["logits_identical_rows"]), (1.0, 159))
        row = c8["first_divergences"][0]
        self.assertIsNone(row["first_divergence"])
        self.assertEqual(row["logits_first_divergence"], 11)

    def test_c1_self_divergence_is_recorded(self):
        c2 = self.complete({2: {"c1_flip": "row-01"}})["widths"]["2"]
        self.assertEqual(c2["c1_self"], {"rows": 2, "token_diverged": 1, "logits_diverged": 0})

    def test_validate_export_rejects_tampered_records(self):
        record = self.complete({4: {"token_flip": (1, "row-00", 3)}})
        for mutate in (
            lambda r: r.update(gating=True),
            lambda r: r.update(release_qualified=True),
            lambda r: r["widths"]["4"].update(token_identity_rate=1.0),
            lambda r: r["widths"]["4"].update(rows_compared=81),
            lambda r: r["widths"].pop("8"),
            lambda r: r.update(problems=["x"]),
            lambda r: r["inputs"].pop(),
            lambda r: r["widths"]["2"]["per_wave_diverged_rows"].append(0),
            lambda r: r["widths"]["2"].update(waves=True),
        ):
            copy = json.loads(json.dumps(record))
            mutate(copy)
            self.assertTrue(A.validate_export(copy), mutate)

    def test_incomplete_export_carries_no_rates(self):
        record = A.export(self.three()[:2])
        self.assertEqual(record["status"], A.DATA_INCOMPLETE)
        self.assertEqual((record["widths"], record["run"]), ({}, {}))
        self.assertEqual(A.validate_export(record), [])

    def test_missing_width_is_incomplete(self):
        result = A.export(self.three()[:2])
        self.assertEqual(result["status"], A.DATA_INCOMPLETE)

    def test_duplicate_width_is_incomplete(self):
        runs = self.three()
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        extra = run_one(Path(temp.name), 8)[:2]
        self.assertEqual(A.export(runs + [extra])["status"], A.DATA_INCOMPLETE)

    def test_records_sha_binding(self):
        runs = self.three()
        runs[0] = (runs[0][0], "b" * 64)
        self.assertEqual(A.export(runs)["status"], A.DATA_INCOMPLETE)

    def test_tampered_window_is_incomplete(self):
        runs = self.three()
        window = Path(runs[1][0]) / "wave-007.log"
        window.write_bytes(window.read_bytes() + b"x")
        self.assertEqual(A.export(runs)["status"], A.DATA_INCOMPLETE)

    def test_serial_collapse_in_native_run_is_incomplete(self):
        result = A.export(self.three({8: {"collapse": (3, 5)}}))
        self.assertEqual(result["status"], A.DATA_INCOMPLETE)
        self.assertTrue(any("serial batch frame" in reason for reason in result["problems"]))

    def test_slot_move_within_a_wave_is_incomplete(self):
        result = A.export(self.three({4: {"slot_move": (2, 6)}}))
        self.assertEqual(result["status"], A.DATA_INCOMPLETE)
        self.assertTrue(any("moved between slots" in reason for reason in result["problems"]))

    def test_serial_control_profile_requires_serial_frames(self):
        runs = [run_one(self.base, w, route="serial", profile="serial-control")[:2] for w in A.WIDTHS]
        record = A.export(runs)
        self.assertEqual(record["status"], A.DATA_COMPLETE)
        self.assertEqual(record["widths"]["2"]["route"], "serial")

    def test_too_few_waves_is_incomplete_and_refused_at_cli(self):
        runs = self.three({2: {"waves": A.MIN_WAVES - 1}})
        self.assertEqual(A.export(runs)["status"], A.DATA_INCOMPLETE)
        argv = ["run", "--base-url", "x", "--model", "m", "--server-log", "l", "--probe-program", "p",
                "--probe-sha256", "a" * 64, "--r9-program", "r", "--r9-sha256", "a" * 64, "--out-dir", "o",
                "--server-pid", "1", "--width", "2", "--selector-profile", "native-reference",
                "--waves", str(A.MIN_WAVES - 1)]
        with self.assertRaises(A.RepeatError):
            R.parse_args(argv)

    def test_failed_run_records_incomplete(self):
        out, digest, records = run_one(self.base, 2, health_fails=True)
        self.assertEqual(records["status"], A.RUN_INCOMPLETE)
        self.assertTrue(records["failures"])
        runs = [(out, digest)] + [run_one(self.base, w)[:2] for w in (4, 8)]
        self.assertEqual(A.export(runs)["status"], A.DATA_INCOMPLETE)

    def test_replaced_process_is_incomplete(self):
        result = A.export(self.three({4: {"replaced": True}}))
        self.assertEqual(result["status"], A.DATA_INCOMPLETE)
        self.assertIn("C4: server process was replaced during the run", result["problems"])

    def test_preexisting_receipts_refuse_the_run(self):
        prefix = "x LOGITS_FNV session=" + "0" * 16 + " idx=1 fnv=" + "0" * 16 + " sha256=" + "0" * 64 + "\n"
        _out, _digest, records = run_one(self.base, 2, log_prefix=prefix)
        self.assertEqual(records["status"], A.RUN_INCOMPLETE)

    def test_exact_type_census(self):
        runs = self.three()
        path = Path(runs[0][0]) / "records.json"
        records = json.loads(path.read_bytes())
        records["c1"][0][0]["completion_tokens"] = True
        raw = A.canonical(records) + b"\n"
        path.write_bytes(raw)
        runs[0] = (runs[0][0], A.sha(raw))
        self.assertEqual(A.export(runs)["status"], A.DATA_INCOMPLETE)

    def test_forbidden_marker_is_incomplete(self):
        runs = self.three()
        path = Path(runs[2][0]) / "records.json"
        records = json.loads(path.read_bytes())
        records["forbidden_marker_counts"]["NVRM: Xid"] = 1
        raw = A.canonical(records) + b"\n"
        path.write_bytes(raw)
        runs[2] = (runs[2][0], A.sha(raw))
        self.assertEqual(A.export(runs)["status"], A.DATA_INCOMPLETE)

    def test_malformed_wave_entry_is_incomplete(self):
        runs = self.three()
        path = Path(runs[1][0]) / "records.json"
        records = json.loads(path.read_bytes())
        records["waves"][4] = ["not", "an", "object"]
        raw = A.canonical(records) + b"\n"
        path.write_bytes(raw)
        runs[1] = (runs[1][0], A.sha(raw))
        self.assertEqual(A.export(runs)["status"], A.DATA_INCOMPLETE)

    def test_out_dir_must_be_new(self):
        (self.base / "exists").mkdir()
        with self.assertRaises(FileExistsError):
            R.make_out_dir(self.base / "exists")

    def test_miss_probability_matches_the_docstring(self):
        self.assertAlmostEqual(A.miss_probability(2, 2), (18 / 19) ** 4)
        self.assertGreater(A.miss_probability(2, 2), 0.80)
        self.assertLess(A.miss_probability(20, 2), 0.12)


if __name__ == "__main__":
    unittest.main()
