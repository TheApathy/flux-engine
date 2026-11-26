# SPDX-License-Identifier: AGPL-3.0-only
"""Offline half of the native-concurrency repeat diagnostic: grammar, typed
census, raw-file recomputation and the export schema.

Pure standard library; no network, process or server access. The live runner
(`qwen4_native_concurrency_repeat.py`) writes the raw files this module reads.
The qualification lane imports `validate_export` to check a record before
consuming it. See the runner's docstring for the policy: the identity and
divergence rates here are diagnostic data and never gate.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat

SCHEMA = "flux-qwen4-native-concurrency-repeat/1"
EXPORT_SCHEMA = "flux-qwen4-native-concurrency-repeat-export/1"
WIDTHS = (2, 4, 8)
MIN_WAVES = 20
C1_PASSES = 2
MIN_FULL_FRAMES = 8
MAX_TOKENS_RANGE = (16, 256)
WINDOW = 8
LF009_FLIP_RATE = 1.0 / 19.0
PROFILES = {
    "native-reference": "native",
    "native-optimized": "native",
    "serial-control": "serial",
}
RUN_COMPLETE, RUN_INCOMPLETE = "RUN_COMPLETE", "RUN_INCOMPLETE"
DATA_COMPLETE, DATA_INCOMPLETE = "DATA_COMPLETE", "DATA_INCOMPLETE"
POLICY = (
    "bit-exact per-slot identity against C1 is a diagnostic, not a gate; batching is "
    "qualified by accuracy panels plus contamination checks (qualification lane)"
)
SCOPE = (
    "run-to-run greedy identity of every concurrent row against its isolated C1 "
    "trajectory, over N identical waves per width, in one fresh process per width"
)
NOT_INFERRED = (
    "terminal target-state identity, cancellation/replacement, slot reuse, reverse "
    "order, accuracy, contamination and release qualification are NOT inferred"
)
# Exact copies of the probe's grammar (tests pin them to the probe source).
FRAME_RE = re.compile(
    r"QWEN4_(NATIVE|SERIAL)_BATCH_FRAME rows=([0-9]+) "
    r"sessions=([0-9a-f]{16}(?:,[0-9a-f]{16})*) "
    r"slots=([0-9]+(?:,[0-9]+)*) pre_step_seq_lens=([0-9]+(?:,[0-9]+)*) "
    r"(?:(?:active_only=true target_only=true qsa=false)|"
    r"(?:selector_on=(true|false) qsa_required=(true|false))) graph=false$"
)
LOGITS_RE = re.compile(
    r"LOGITS_FNV session=([0-9a-f]{16}) idx=([0-9]+) "
    r"fnv=([0-9a-f]{16}) sha256=([0-9a-f]{64})$"
)
FORBIDDEN_LOG_MARKERS = (
    "CUDA error",
    "illegal memory access",
    "out of memory",
    "decode_batch error",
    "aliases another request's recurrent state",
    "alias persistent slot",
    "independent rows alias a physical KV block",
    "does not exclusively own its KV block",
    "QWEN4_NATIVE_MTP_REQUEST_RECEIPT_FAILED",
    "NVRM: Xid",
    "illegal instruction",
    "segmentation fault",
    "stack backtrace:",
    "thread 'main' panicked",
)
RECEIPT_MARKERS = (
    "QWEN4_NATIVE_BATCH_FRAME",
    "QWEN4_SERIAL_BATCH_FRAME",
    "LOGITS_FNV",
    "QWEN4_NATIVE_MTP_REQUEST_RECEIPT",
)
SHA_RE = re.compile(r"[0-9a-f]{64}")
WINDOW_NAME_RE = re.compile(r"(c1-p[0-9]-row-[0-9]{2}|wave-[0-9]{3})\.log")
# Typed, non-missing census: field -> (type, required value or None for a check below).
RECORD_FIELDS = {
    "schema": (str, SCHEMA),
    "status": (str, RUN_COMPLETE),
    "failures": (list, []),
    "width": (int, None),
    "selector_profile": (str, None),
    "route": (str, None),
    "waves_requested": (int, None),
    "max_tokens": (int, None),
    "probe_sha256": (str, None),
    "model": (str, None),
    "server": (dict, None),
    "preexisting_receipts": (int, 0),
    "forbidden_markers": (list, list(FORBIDDEN_LOG_MARKERS)),
    "forbidden_marker_counts": (dict, None),
    "health_after": (bool, True),
    "c1": (list, None),
    "waves": (list, None),
}
REQUEST_FIELDS = {
    "label": (str, None),
    "ordinal": (int, None),
    "prompt_sha256": (str, None),
    "token_ids": (list, None),
    "finish_reason": (str, None),
    "prompt_tokens": (int, None),
    "completion_tokens": (int, None),
}
SERVER_FIELDS = {
    "pid": (int, None),
    "start_ticks_before": (int, None),
    "start_ticks_after": (int, None),
    "log_bound_fds": (list, [1, 2]),
}
# Export record consumed by the qualification lane. Typed, non-missing census:
# field -> (type, required value or None). Rates are identical/rows_compared.
EXPORT_FIELDS = {
    "schema": (str, EXPORT_SCHEMA),
    "status": (str, None),
    "gating": (bool, False),
    "release_qualified": (bool, False),
    "policy": (str, POLICY),
    "scope": (str, SCOPE),
    "not_inferred": (str, NOT_INFERRED),
    "problems": (list, None),
    "inputs": (list, None),
    "run": (dict, None),
    "widths": (dict, None),
    "cross_process_c1": (dict, None),
}
RUN_FIELDS = {
    "selector_profile": (str, None),
    "probe_sha256": (str, None),
    "model": (str, None),
    "max_tokens": (int, None),
}
WIDTH_FIELDS = {
    "route": (str, None),
    "waves": (int, None),
    "rows_compared": (int, None),
    "token_identical_rows": (int, None),
    "logits_identical_rows": (int, None),
    "finish_identical_rows": (int, None),
    "fully_identical_rows": (int, None),
    "token_identity_rate": (float, None),
    "logits_identity_rate": (float, None),
    "full_identity_rate": (float, None),
    "c1_self": (dict, None),
    "per_slot": (dict, None),
    "per_wave_diverged_rows": (list, None),
    "first_divergence_histogram": (dict, None),
    "first_divergences": (list, None),
    "miss_probability_at_lf009_rate": (float, None),
}
COUNT_FIELDS = {"rows": (int, None), "token_diverged": (int, None), "logits_diverged": (int, None)}
MAX_LISTED_DIVERGENCES = 16


class RepeatError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise RepeatError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def strict_json(raw):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            require(key not in out, "duplicate JSON key: " + key)
            out[key] = value
        return out

    return json.loads(raw, object_pairs_hook=unique)


def same(a, b):
    """Exact-type deep equality: 1 != True, 0.0 != 0, (x,) != [x]."""
    if type(a) is not type(b):
        return False
    if type(a) is dict:
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if type(a) in (list, tuple):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def census(value, fields, scope):
    """Typed field census; returns failure strings (never raises)."""
    if type(value) is not dict:
        return [f"{scope}: not an object"]
    failures = []
    if set(value) != set(fields):
        failures.append(f"{scope}: fields differ: {sorted(set(value) ^ set(fields))}")
    for name, (kind, expected) in fields.items():
        if name not in value:
            continue
        if type(value[name]) is not kind:
            failures.append(f"{scope}.{name}: expected {kind.__name__}")
        elif expected is not None and not same(value[name], expected):
            failures.append(f"{scope}.{name}: expected {expected!r}, got {value[name]!r}")
    return failures


def read_regular(path, limit=256 << 20):
    """Read one regular file; a final symlink is refused, never followed."""
    path = Path(path)
    require(not path.is_symlink(), f"{path.name}: symlinks are not followed")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        require(stat.S_ISREG(info.st_mode), f"{path.name}: not a regular file")
        require(info.st_size <= limit, f"{path.name}: exceeds {limit} bytes")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def write_new(path, raw):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    try:
        view = memoryview(raw)
        while view:
            view = view[os.write(descriptor, view):]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def first_divergence(left, right):
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def token_window(left, right, index):
    lo, hi = max(0, index - WINDOW), index + WINDOW + 1
    return {"window": [lo, hi], "c1_tokens": left[lo:hi], "row_tokens": right[lo:hi]}


def miss_probability(waves, width, rate=LF009_FLIP_RATE):
    """Chance that no row of `waves` x `width` shows a per-row flip at `rate`."""
    return (1.0 - rate) ** (waves * width)


# --- log windows ---------------------------------------------------------------


def parse_window(raw):
    """Batch frames and logits rows of one raw log window (strict UTF-8)."""
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise RepeatError("log window is not UTF-8") from error
    frames, logits = [], {}
    for line in text.splitlines():
        present = [marker for marker in RECEIPT_MARKERS if marker in line]
        if not present:
            continue
        frame = FRAME_RE.search(line)
        row = LOGITS_RE.search(line)
        if frame and len(present) == 1:
            route, rows, sessions, slots, seq_lens = frame.groups()[:5]
            frames.append(frame_record(route, rows, sessions, slots, seq_lens))
        elif row and present == ["LOGITS_FNV"]:
            session, index, _fnv, digest = row.groups()
            rows_for = logits.setdefault(session, {})
            require(int(index) not in rows_for, f"duplicate logits row {session}/{index}")
            rows_for[int(index)] = digest
        elif present != ["QWEN4_NATIVE_MTP_REQUEST_RECEIPT"]:
            raise RepeatError("malformed qualification evidence line")
    return frames, logits


def frame_record(route, rows, sessions, slots, seq_lens):
    record = {
        "route": route.lower(),
        "rows": int(rows),
        "sessions": sessions.split(","),
        "slots": [int(value) for value in slots.split(",")],
        "pre_step_seq_lens": [int(value) for value in seq_lens.split(",")],
    }
    count = record["rows"]
    require(
        len(record["sessions"]) == count
        and len(record["slots"]) == count
        and len(record["pre_step_seq_lens"]) == count,
        "batch frame metadata cardinality differs from rows",
    )
    require(len(set(record["sessions"])) == count, "batch frame aliases a logical session")
    require(len(set(record["slots"])) == count, "batch frame aliases a physical slot")
    return record


def frame_census(frames, sessions, width, route):
    """Route census of one wave window; returns (failures, session -> slot)."""
    failures, slots = [], {}
    allowed = set(sessions)
    full = [frame for frame in frames if frame["rows"] == width]
    other_route = "serial" if route == "native" else "native"
    if any(frame["route"] == other_route for frame in frames):
        failures.append(f"{other_route} batch frame in a {route} wave")
    if any(frame["rows"] > width or frame["rows"] < 2 for frame in frames):
        failures.append("batch frame width outside 2..C")
    if len(full) < MIN_FULL_FRAMES:
        failures.append(f"only {len(full)} full-width frames (< {MIN_FULL_FRAMES})")
    for frame in frames:
        for session, slot in zip(frame["sessions"], frame["slots"]):
            if session not in allowed:
                failures.append(f"frame session {session} is not a wave row")
            elif slots.setdefault(session, slot) != slot:
                failures.append(f"session {session} moved between slots within one wave")
    missing = allowed - set(slots)
    if missing:
        failures.append(f"{len(missing)} wave rows never appear in a batch frame")
    return failures, slots


def logits_divergence(reference, observed):
    """First index whose logits digest differs (or is missing on either side)."""
    if reference == observed:
        return None
    indices = sorted(set(reference) | set(observed))
    for index in indices:
        if reference.get(index) != observed.get(index):
            return index
    return None


# --- offline comparator (fails closed) -----------------------------------------


def load_records(run_dir, records_sha256):
    raw = read_regular(Path(run_dir) / "records.json", 64 << 20)
    require(SHA_RE.fullmatch(records_sha256 or "") is not None, "records sha256 malformed")
    require(sha(raw) == records_sha256, "records.json differs from the bound sha256")
    return strict_json(raw)


def record_census(records):
    failures = census(records, RECORD_FIELDS, "records")
    if failures:
        return failures
    if records["width"] not in WIDTHS:
        failures.append(f"width {records['width']} is not one of {WIDTHS}")
    if PROFILES.get(records["selector_profile"]) != records["route"]:
        failures.append("selector profile and route disagree")
    if records["waves_requested"] < MIN_WAVES or len(records["waves"]) != records["waves_requested"]:
        failures.append(f"fewer than {MIN_WAVES} waves or a wave is missing")
    low, high = MAX_TOKENS_RANGE
    if not low <= records["max_tokens"] <= high:
        failures.append("max_tokens outside 16..256")
    if SHA_RE.fullmatch(records["probe_sha256"]) is None:
        failures.append("probe sha256 malformed")
    failures += census(records["server"], SERVER_FIELDS, "records.server")
    server = records["server"]
    if not failures and server["start_ticks_before"] != server["start_ticks_after"]:
        failures.append("server process was replaced during the run")
    counts = records["forbidden_marker_counts"]
    if set(counts) != set(FORBIDDEN_LOG_MARKERS) or any(
        type(value) is not int or value != 0 for value in counts.values()
    ):
        failures.append(f"forbidden log markers present or census incomplete: {counts}")
    if len(records["c1"]) != C1_PASSES:
        failures.append(f"expected {C1_PASSES} C1 passes")
    return failures


def request_census(request, width, max_tokens, scope):
    fields = dict(REQUEST_FIELDS, window=(str, None), window_sha256=(str, None))
    failures = census(request, fields, scope)
    if failures:
        return failures
    if not (0 <= request["ordinal"] < width and request["label"] == f"row-{request['ordinal']:02d}"):
        failures.append(f"{scope}: label/ordinal outside the C{width} plan")
    ids = request["token_ids"]
    if not all(type(token) is int and 0 <= token <= 0xFFFF_FFFF for token in ids):
        failures.append(f"{scope}: malformed token ids")
    if len(ids) != max_tokens or request["completion_tokens"] != max_tokens:
        failures.append(f"{scope}: completion is not the exact {max_tokens}-token budget")
    if request["prompt_tokens"] + max_tokens >= 2048:
        failures.append(f"{scope}: prompt+max_tokens reaches the 2048 QSA boundary")
    if WINDOW_NAME_RE.fullmatch(request["window"]) is None:
        failures.append(f"{scope}: window name malformed")
    return failures


def read_window(run_dir, name, digest):
    require(WINDOW_NAME_RE.fullmatch(name) is not None, f"window name {name!r} malformed")
    raw = read_regular(Path(run_dir) / name)
    require(sha(raw) == digest, f"{name} differs from its recorded sha256")
    return parse_window(raw)


def c1_pass(run_dir, rows, width, max_tokens, scope):
    """Validate one isolated C1 pass; returns (failures, label -> (ids, finish, session, logits))."""
    failures, out = [], {}
    if [row.get("label") if type(row) is dict else None for row in rows] != [
        f"row-{index:02d}" for index in range(width)
    ]:
        return [f"{scope}: rows differ from the C{width} plan order"], out
    for row in rows:
        found = request_census(row, width, max_tokens, f"{scope}/{row['label']}")
        if found:
            failures += found
            continue
        frames, logits = read_window(run_dir, row["window"], row["window_sha256"])
        if frames:
            failures.append(f"{scope}/{row['label']}: isolated C1 window holds batch frames")
        if len(logits) != 1 or not next(iter(logits.values())):
            failures.append(f"{scope}/{row['label']}: expected one session of logits rows")
            continue
        session, chain = next(iter(logits.items()))
        out[row["label"]] = (row["token_ids"], row["finish_reason"], session, chain)
    return failures, out


def wave_rows(run_dir, wave, width, max_tokens, route, c1):
    """Validate one wave; returns (failures, per-row comparison rows)."""
    scope = f"wave-{wave.get('index')}"
    fields = {"index": (int, None), "rows": (list, None), "window": (str, None), "window_sha256": (str, None)}
    failures = census(wave, fields, scope)
    if failures:
        return failures, []
    rows = wave["rows"]
    if [row.get("label") if type(row) is dict else None for row in rows] != list(c1):
        return [f"{scope}: rows differ from the C1 plan"], []
    for row in rows:
        failures += request_census(
            dict(row, window=wave["window"], window_sha256=wave["window_sha256"]),
            width, max_tokens, f"{scope}/{row['label']}",
        )
    if failures:
        return failures, []
    frames, logits = read_window(run_dir, wave["window"], wave["window_sha256"])
    sessions = [c1[row["label"]][2] for row in rows]
    found, slots = frame_census(frames, sessions, width, route)
    failures += [f"{scope}: {item}" for item in found]
    table = []
    for row in rows:
        c1_ids, c1_finish, session, c1_chain = c1[row["label"]]
        table.append(compare_row(wave["index"], row, slots.get(session), c1_ids, c1_finish,
                                 c1_chain, logits.get(session, {})))
    return failures, table


def compare_row(wave_index, row, slot, c1_ids, c1_finish, c1_chain, chain):
    index = first_divergence(c1_ids, row["token_ids"])
    out = {
        "wave": wave_index,
        "label": row["label"],
        "slot": slot,
        "first_divergence": index,
        "finish_reason_equal": row["finish_reason"] == c1_finish,
        "logits_first_divergence": logits_divergence(c1_chain, chain),
    }
    if index is not None:
        out.update(token_window(c1_ids, row["token_ids"], index))
    return out


def row_diverged(row):
    return (
        row["first_divergence"] is not None
        or row["logits_first_divergence"] is not None
        or not row["finish_reason_equal"]
    )


def check_width(run_dir, records_sha256):
    """Recompute one width from raw files. Returns (problems, analysis)."""
    try:
        records = load_records(run_dir, records_sha256)
    except (OSError, ValueError) as error:
        return [f"{run_dir}: {error}"], None
    problems = record_census(records)
    if problems:
        return [f"C{records.get('width')}: {item}" for item in problems], None
    width, max_tokens, route = records["width"], records["max_tokens"], records["route"]
    passes = []
    for number, rows in enumerate(records["c1"]):
        found, out = c1_pass(run_dir, rows, width, max_tokens, f"c1-p{number}")
        problems += found
        passes.append(out)
    if problems:
        return [f"C{width}: {item}" for item in problems], None
    c1 = passes[0]
    if any(passes[1][label][2] != c1[label][2] for label in c1):
        return [f"C{width}: a prompt's session hash differs between C1 passes"], None
    c1_self = [compare_row(-1, {"label": label, "token_ids": passes[1][label][0],
                                "finish_reason": passes[1][label][1]},
                           None, *c1[label][:2], c1[label][3], passes[1][label][3])
               for label in c1]
    waves = records["waves"]
    if [wave.get("index") if type(wave) is dict else None for wave in waves] != list(range(len(waves))):
        return [f"C{width}: waves are not objects indexed 0..N-1"], None
    table = []
    for wave in waves:
        try:
            found, rows = wave_rows(run_dir, wave, width, max_tokens, route, c1)
        except (OSError, ValueError, KeyError, TypeError) as error:
            found, rows = [f"wave {wave['index']}: {error}"], []
        problems += [f"C{width}: {item}" for item in found]
        table += rows
    return problems, {
        "width": width,
        "route": route,
        "selector_profile": records["selector_profile"],
        "probe_sha256": records["probe_sha256"],
        "model": records["model"],
        "max_tokens": max_tokens,
        "waves": len(records["waves"]),
        "c1_tokens": {label: c1[label][0] for label in c1},
        "c1_self": c1_self,
        "rows": table,
    }


def counts(rows):
    return {
        "rows": len(rows),
        "token_diverged": sum(row["first_divergence"] is not None for row in rows),
        "logits_diverged": sum(row["logits_first_divergence"] is not None for row in rows),
    }


def per_slot(rows):
    slots = {}
    for row in rows:
        slots.setdefault(str(row["slot"]), []).append(row)
    return {slot: counts(group) for slot, group in sorted(slots.items())}


def cross_process_c1(analyses):
    """Identical prompts in separate fresh processes: C1 tokens compared, as data."""
    diverged, compared = [], 0
    widths = sorted(analyses)
    for left, right in zip(widths, widths[1:]):
        for label, ids in analyses[left]["c1_tokens"].items():
            compared += 1
            index = first_divergence(ids, analyses[right]["c1_tokens"].get(label))
            if index is not None:
                diverged.append({"label": label, "widths": [left, right], "first_divergence": index})
    return {"labels_compared": compared, "diverged": diverged}


def rate(part, whole):
    return float(part) / float(whole) if whole else 0.0


def width_summary(width, analysis):
    rows = analysis["rows"]
    token = sum(row["first_divergence"] is None for row in rows)
    logits = sum(row["logits_first_divergence"] is None for row in rows)
    finish = sum(row["finish_reason_equal"] for row in rows)
    full = sum(not row_diverged(row) for row in rows)
    histogram = {}
    for row in rows:
        if row["first_divergence"] is not None:
            key = str(row["first_divergence"])
            histogram[key] = histogram.get(key, 0) + 1
    return {
        "route": analysis["route"],
        "waves": analysis["waves"],
        "rows_compared": len(rows),
        "token_identical_rows": token,
        "logits_identical_rows": logits,
        "finish_identical_rows": finish,
        "fully_identical_rows": full,
        "token_identity_rate": rate(token, len(rows)),
        "logits_identity_rate": rate(logits, len(rows)),
        "full_identity_rate": rate(full, len(rows)),
        "c1_self": counts(analysis["c1_self"]),
        "per_slot": per_slot(rows),
        "per_wave_diverged_rows": [
            sum(row_diverged(row) for row in rows if row["wave"] == index)
            for index in range(analysis["waves"])
        ],
        "first_divergence_histogram": dict(sorted(histogram.items(), key=lambda item: int(item[0]))),
        "first_divergences": [row for row in rows if row_diverged(row)][:MAX_LISTED_DIVERGENCES],
        "miss_probability_at_lf009_rate": miss_probability(analysis["waves"], width),
    }


def export(runs):
    """runs: [(run_dir, records_sha256)]. Divergence is data; only evidence gaps are INCOMPLETE."""
    problems, analyses = [], {}
    inputs = [{"width": None, "records_sha256": digest} for _run_dir, digest in runs]
    for slot, (run_dir, digest) in enumerate(runs):
        found, analysis = check_width(run_dir, digest)
        problems += found
        if analysis is not None:
            inputs[slot]["width"] = analysis["width"]
            if analysis["width"] in analyses:
                problems.append(f"C{analysis['width']}: more than one run")
            analyses[analysis["width"]] = analysis
    if not problems and sorted(analyses) != list(WIDTHS):
        problems.append(f"widths {sorted(analyses)} differ from {list(WIDTHS)}")
    common = {(a["selector_profile"], a["probe_sha256"], a["model"], a["max_tokens"]) for a in analyses.values()}
    if not problems and len(common) != 1:
        problems.append("runs differ in profile, probe, model or max_tokens")
    record = {
        "schema": EXPORT_SCHEMA, "status": DATA_INCOMPLETE, "gating": False,
        "release_qualified": False, "policy": POLICY, "scope": SCOPE, "not_inferred": NOT_INFERRED,
        "problems": problems, "inputs": inputs, "run": {}, "widths": {},
        "cross_process_c1": {"labels_compared": 0, "diverged": []},
    }
    if problems:
        return record
    profile, probe_sha256, model, max_tokens = next(iter(common))
    record.update(
        status=DATA_COMPLETE,
        run={"selector_profile": profile, "probe_sha256": probe_sha256, "model": model,
             "max_tokens": max_tokens},
        widths={str(width): width_summary(width, analyses[width]) for width in sorted(analyses)},
        cross_process_c1=cross_process_c1(analyses),
    )
    return record


def validate_export(record):
    """Consumer-side schema check (qualification lane). Returns failure strings."""
    failures = census(record, EXPORT_FIELDS, "export")
    if failures:
        return failures
    if record["status"] not in (DATA_COMPLETE, DATA_INCOMPLETE):
        failures.append("export.status is not DATA_COMPLETE or DATA_INCOMPLETE")
    if record["status"] == DATA_INCOMPLETE:
        if not record["problems"] or record["widths"] or record["run"]:
            failures.append("an incomplete export carries rates or no problems")
        return failures
    if record["problems"]:
        failures.append("a complete export lists problems")
    if len(record["inputs"]) != len(WIDTHS) or any(
        type(item) is not dict or set(item) != {"width", "records_sha256"}
        or type(item["width"]) is not int or SHA_RE.fullmatch(str(item["records_sha256"])) is None
        for item in record["inputs"]
    ):
        failures.append("export.inputs must bind one records sha256 per width")
    failures += census(record["run"], RUN_FIELDS, "export.run")
    if sorted(record["widths"]) != sorted(str(width) for width in WIDTHS):
        failures.append(f"export.widths must be exactly {list(WIDTHS)}")
    for key, summary in record["widths"].items():
        failures += validate_width(key, summary)
    return failures


def validate_width(key, summary):
    scope = f"export.widths.{key}"
    failures = census(summary, WIDTH_FIELDS, scope)
    if failures:
        return failures
    rows = summary["rows_compared"]
    for name, part in (("token_identity_rate", "token_identical_rows"),
                       ("logits_identity_rate", "logits_identical_rows"),
                       ("full_identity_rate", "fully_identical_rows")):
        if not 0 <= summary[part] <= rows or summary[name] != rate(summary[part], rows):
            failures.append(f"{scope}.{name} does not equal {part}/rows_compared")
    if rows != summary["waves"] * int(key) or len(summary["per_wave_diverged_rows"]) != summary["waves"]:
        failures.append(f"{scope}: row or wave counts are inconsistent")
    if sum(summary["per_wave_diverged_rows"]) != rows - summary["fully_identical_rows"]:
        failures.append(f"{scope}: per-wave divergence does not sum to the diverged rows")
    failures += census(summary["c1_self"], COUNT_FIELDS, scope + ".c1_self")
    for slot, entry in summary["per_slot"].items():
        failures += census(entry, COUNT_FIELDS, f"{scope}.per_slot.{slot}")
    return failures
