#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Run-to-run repeat data for native Qwen4 C2/C4/C8 decode.

DIAGNOSTIC DATA ONLY. Policy (user decision, 2026-10-23): batching is qualified
by scored accuracy panels plus cross-request contamination checks, owned by the
qualification lane; bit-exact per-slot identity against C1 is a diagnostic.
This tool therefore never gates: it measures how often every row of N
identical concurrent waves reproduces its own isolated C1 greedy trajectory,
where the first divergence falls, and on which physical slot, and exports
those rates in a small versioned JSON record for the qualification lane. A
one-in-nineteen flip (the LF-009 rate) goes unseen in two waves at C2 with
probability ~0.81 and in twenty waves with probability ~0.11.

Like the probe, the client never builds, launches, stops or mutates a server.
A resource owner starts one fresh process per width with the probe's
direct-binary launch contract (BF16 KV, max-seq-len 2048, prefill 1024,
speculation off) plus `ATLAS_LOGITS_FNV=1` and
`ATLAS_QWEN4_NATIVE_BATCH_RECEIPT=1`, and passes its append-only log here.

* `run` (one fresh process, one width): two isolated C1 passes over the
  probe's request plan, then N >= 20 identical forward waves of the same plan.
  The probe's own request body, SSE client, wave launcher, log cursor and
  process checks are reused from pinned bytes. Every C1 request and every wave
  keeps its raw log window. `records.json` is written even on failure, with
  status RUN_INCOMPLETE.
* `export` (offline): recomputes everything from the raw files of exactly one
  run per width 2, 4 and 8, bound by their records SHA-256, and writes one
  `flux-qwen4-native-concurrency-repeat-export/1` record (EXPORT_FIELDS).
  Evidence integrity still fails closed: any unproven prerequisite (hash,
  census, route, slot stability, process identity) makes the record
  DATA_INCOMPLETE with no rates. Divergence never does; it is data.
  `validate_export` in qwen4_native_concurrency_repeat_data.py (the offline
  half) is the consumer-side schema check.

Per row and wave it compares token IDs, finish reason, completion count and
the full-vocabulary logits SHA-256 chain against C1 pass 0, keyed by the
row's physical slot from the wave's batch frames, and records the first
divergence with a token window.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
from pathlib import Path
import sys
import types


def _load_data_module():
    """The offline half lives beside this file; load it by path, once."""
    name = "qwen4_native_concurrency_repeat_data"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().with_name(name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


D = _load_data_module()


# --- live runner (one fresh process, one width) ---------------------------------


def load_pinned(path, expected, name, injected=None):
    """Execute a module from bytes whose SHA-256 is pinned (read once)."""
    raw = D.read_regular(path, 8 << 20)
    D.require(D.sha(raw) == expected, f"{Path(path).name} differs from its pinned sha256")
    module = types.ModuleType(name)
    module.__file__ = str(Path(path).resolve())
    module.__dict__.update(injected or {})
    sys.modules[name] = module
    exec(compile(raw, module.__file__, "exec"), module.__dict__)
    return module


def request_record(result, case, window, digest):
    return {
        "label": result.label,
        "ordinal": case.ordinal,
        "prompt_sha256": case.prompt_sha256,
        "token_ids": list(result.token_ids),
        "finish_reason": result.finish_reason,
        "prompt_tokens": result.usage["prompt_tokens"],
        "completion_tokens": result.usage["completion_tokens"],
        "window": window,
        "window_sha256": digest,
    }


class Runner:
    """Drives one width against an already-running server; never launches one."""

    def __init__(self, args, probe):
        self.args, self.probe = args, probe
        self.out = Path(args.out_dir)
        self.cursor = probe.LogCursor(Path(args.server_log))

    def window(self, name, offset):
        raw = self.cursor.read_bytes_from(offset)
        D.write_new(self.out / name, raw)
        return D.sha(raw)

    def c1_pass(self, number, cases):
        rows = []
        for case in cases:
            offset = self.cursor.mark()
            result = self.probe.stream_request(
                self.args.base_url, self.args.model, case, self.args.max_tokens, self.args.timeout
            )
            name = f"c1-p{number}-{case.label}.log"
            rows.append(request_record(result, case, name, self.window(name, offset)))
        return rows

    def wave(self, index, cases):
        offset = self.cursor.mark()
        results = self.probe.run_wave(
            self.args.base_url, self.args.model, cases, self.args.max_tokens, self.args.timeout
        )
        name = f"wave-{index:03d}.log"
        digest = self.window(name, offset)
        rows = [request_record(result, case, name, digest) for result, case in zip(results, cases)]
        for row in rows:
            del row["window"], row["window_sha256"]
        return {"index": index, "rows": rows, "window": name, "window_sha256": digest}


def run_width(args, probe):
    """One fresh process, one width. Always returns a records dict (never raises)."""
    args.route = D.PROFILES[args.selector_profile]
    records = {
        "schema": D.SCHEMA, "status": D.RUN_INCOMPLETE, "failures": [], "width": args.width,
        "selector_profile": args.selector_profile, "route": args.route,
        "waves_requested": args.waves, "max_tokens": args.max_tokens,
        "probe_sha256": args.probe_sha256, "model": args.model, "server": {},
        "preexisting_receipts": -1, "forbidden_markers": list(D.FORBIDDEN_LOG_MARKERS),
        "forbidden_marker_counts": {}, "health_after": False, "c1": [], "waves": [],
    }
    try:
        run_phases(args, probe, records)
        records["status"] = D.RUN_COMPLETE
    except Exception as error:  # recorded; the D.export is then DATA_INCOMPLETE
        records["failures"].append(f"{type(error).__name__}: {error}")
    return records


def run_phases(args, probe, records):
    D.require(tuple(probe.FORBIDDEN_LOG_MARKERS) == D.FORBIDDEN_LOG_MARKERS, "probe marker set differs")
    pid = args.server_pid
    server = records["server"]
    server.update(pid=pid, start_ticks_before=probe.process_start_ticks(pid))
    probe.validate_log_binding(Path(args.server_log), pid)
    server["log_bound_fds"] = [1, 2]
    probe.healthcheck(args.base_url, args.timeout)
    runner = Runner(args, probe)
    start = runner.cursor.mark()
    prior = runner.cursor.read_from(0, settle_seconds=0.0, end_offset=start)
    records["preexisting_receipts"] = sum(prior.count(marker) for marker in D.RECEIPT_MARKERS)
    D.require(records["preexisting_receipts"] == 0, "server log already holds inference receipts")
    cases = probe.make_cases(args.width)
    for number in range(D.C1_PASSES):
        records["c1"].append(runner.c1_pass(number, cases))
    for index in range(args.waves):
        records["waves"].append(runner.wave(index, cases))
    probe.healthcheck(args.base_url, args.timeout)
    records["health_after"] = True
    lowered = runner.cursor.read_from(start).lower()
    records["forbidden_marker_counts"] = {
        marker: lowered.count(marker.lower()) for marker in D.FORBIDDEN_LOG_MARKERS
    }
    server["start_ticks_after"] = probe.process_start_ticks(pid)


def make_out_dir(path):
    path = Path(path)
    parent = path.parent
    D.require(parent.is_dir() and not parent.is_symlink(), "--out-dir parent must be a real directory")
    os.mkdir(path, 0o755)  # fails if it exists
    return path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", allow_abbrev=False)
    for name in ("base-url", "model", "server-log", "probe-program", "probe-sha256",
                 "r9-program", "r9-sha256", "out-dir"):
        run.add_argument("--" + name, required=True)
    run.add_argument("--server-pid", type=int, required=True)
    run.add_argument("--width", type=int, required=True, choices=D.WIDTHS)
    run.add_argument("--selector-profile", required=True, choices=sorted(D.PROFILES))
    run.add_argument("--waves", type=int, default=D.MIN_WAVES)
    run.add_argument("--max-tokens", type=int, default=64)
    run.add_argument("--timeout", type=float, default=900.0)
    exporter = sub.add_parser("export", allow_abbrev=False)
    exporter.add_argument("--run", nargs=2, action="append", required=True,
                          metavar=("RUN_DIR", "RECORDS_SHA256"))
    exporter.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        D.require(args.waves >= D.MIN_WAVES, f"--waves cannot go below {D.MIN_WAVES}")
        low, high = D.MAX_TOKENS_RANGE
        D.require(low <= args.max_tokens <= high, "--max-tokens must be in 16..256")
        D.require(math.isfinite(args.timeout) and args.timeout > 0, "--timeout must be positive")
        for name in ("probe_sha256", "r9_sha256"):
            D.require(D.SHA_RE.fullmatch(getattr(args, name)) is not None, f"--{name} malformed")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.command == "export":
        record = D.export([(run_dir, digest) for run_dir, digest in args.run])
        D.require(not D.validate_export(record), "export record fails its own schema")
        raw = D.canonical(record) + b"\n"
        D.write_new(args.out, raw)
        print(record["status"], D.sha(raw))
        return 0 if record["status"] == D.DATA_COMPLETE else 3
    out = make_out_dir(args.out_dir)
    r9 = load_pinned(args.r9_program, args.r9_sha256, "_repeat_r9_provenance")
    probe = load_pinned(args.probe_program, args.probe_sha256, "_repeat_probe",
                        injected={"r9_provenance": r9})
    records = run_width(args, probe)
    raw = D.canonical(records) + b"\n"
    D.write_new(out / "records.json", raw)
    print(records["status"], D.sha(raw))
    return 0 if records["status"] == D.RUN_COMPLETE else 3


if __name__ == "__main__":
    raise SystemExit(main())
