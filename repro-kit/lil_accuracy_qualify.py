#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Fail-closed Lil v0.7.9 GSM8K and MMLU-Pro parity qualification.

This validator consumes four completed Lil reports: full GSM8K and MMLU-Pro
runs with speculation disabled, plus matched full runs with native MTP K=5.
It does not launch a server and never accesses the network.  A PASS means:

* all 2,319 pinned items were measured once in each arm at C1
* every request has exact, non-estimated token usage and no runtime artifact
* every request has a terminal, empty Atlas intervention ledger
* target-only and K=5 agree per item on answer, correctness, and token count
* the reports and all material runtime artifacts are bound by SHA-256 in an
  external provenance envelope.

The accuracy floors are provisional sanity checks.  Exact matched per-item
parity is the authoritative speculative-decoding correctness criterion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any

ENVELOPE_SCHEMA = "flux-lil-accuracy-pair-envelope-v1"
RECEIPT_SCHEMA = "flux-lil-accuracy-pair-receipt-v1"
LIL_VERSION = "0.7.9"
SCORER_VERSION = 2
PROFILE_VERSION = 1
TOKEN_BUDGET = 131_072
INTERVENTION_LEDGER_SCHEMA = "atlas.interventions.v1"
INTERVENTION_LEDGER_POLICY = "report_only"

PROFILE_CONTRACTS = {
    "gsm8k": {
        "dataset": "gsm8k_test",
        "dataset_filename": "gsm8k_test.jsonl",
        "dataset_sha256": "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
        "items": 1_319,
        "scorer": "dataset_gsm8k",
        "provisional_floor": 0.95,
    },
    "mmlu-pro": {
        "dataset": "mmlu_pro_1000",
        "dataset_filename": "mmlu_pro_1000.jsonl",
        "dataset_sha256": "a229ed37f28bea012bf08e81d8ae3358dda17ef7370dca360f632952e3bdcaca",
        "items": 1_000,
        "scorer": "dataset_mc_letter",
        "provisional_floor": 0.65,
    },
}

MMLU_CATEGORY_COUNTS = {
    "biology": 60,
    "business": 66,
    "chemistry": 94,
    "computer science": 34,
    "economics": 70,
    "engineering": 81,
    "health": 68,
    "history": 32,
    "law": 91,
    "math": 112,
    "other": 77,
    "philosophy": 41,
    "physics": 108,
    "psychology": 66,
}

ARM_MODES = {
    "no_spec": "none",
    "k5": "native-mtp-k5",
}

FORBIDDEN_FINISH_REASONS = {
    "client_cancelled",
    "length",
    "stalled",
    "timeout",
}

SPEC_BOOL_FLAGS = {"--speculative"}
SPEC_VALUE_FLAGS = {
    "--mtp-from-path",
    "--mtp-quantization",
    "--mtp-vocab",
    "--num-drafts",
}
FORBIDDEN_DFLASH_FLAGS = {
    "--dflash",
    "--dflash-gamma",
    "--draft-model",
}


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_object(value: object, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{label} must be an object")
    return value


def _require_array(value: object, label: str) -> list[Any]:
    if type(value) is not list:
        raise ValueError(f"{label} must be an array")
    return value


def _require_string(value: object, label: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        suffix = " a non-empty string" if nonempty else " a string"
        raise ValueError(f"{label} must be{suffix}")
    return value


def _require_int(value: object, expected: int, label: str) -> int:
    if type(value) is not int or value != expected:
        raise ValueError(f"{label} must equal {expected}, got {value!r}")
    return value


def _require_positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _require_bool(value: object, expected: bool, label: str) -> bool:
    if type(value) is not bool or value is not expected:
        raise ValueError(f"{label} must be {expected!r}, got {value!r}")
    return value


def _require_number(value: object, expected: float, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be a finite number")
    if not math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label} must equal {expected}, got {value!r}")
    return float(value)


def _require_keys(value: dict[str, Any], keys: set[str], label: str) -> None:
    missing = keys - set(value)
    if missing:
        raise ValueError(f"{label} is missing keys: {sorted(missing)!r}")


def _require_empty_intervention_ledger(value: object, label: str) -> dict[str, Any]:
    ledger = _require_object(value, label)
    _require_exact_keys(ledger, {"schema", "policy", "events"}, label)
    if ledger.get("schema") != INTERVENTION_LEDGER_SCHEMA:
        raise ValueError(f"{label}.schema must equal {INTERVENTION_LEDGER_SCHEMA!r}")
    if ledger.get("policy") != INTERVENTION_LEDGER_POLICY:
        raise ValueError(f"{label}.policy must equal {INTERVENTION_LEDGER_POLICY!r}")
    events = _require_array(ledger.get("events"), f"{label}.events")
    if events:
        raise ValueError(f"{label}.events must be empty, got {events!r}")
    return ledger


def _require_exact_keys(value: dict[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        missing = sorted(keys - set(value))
        extra = sorted(set(value) - keys)
        raise ValueError(f"{label} keys differ: missing={missing!r} extra={extra!r}")


def _require_sha256(value: object, label: str, *, prefixed: bool = False) -> str:
    pattern = r"sha256:[0-9a-f]{64}" if prefixed else r"[0-9a-f]{64}"
    if type(value) is not str or re.fullmatch(pattern, value) is None:
        form = (
            "sha256:<64 lowercase hex>" if prefixed else "64 lowercase hex characters"
        )
        raise ValueError(f"{label} must be {form}")
    return value


def _require_revision(value: object, label: str) -> str:
    if (
        type(value) is not str
        or re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", value) is None
    ):
        raise ValueError(
            f"{label} must be an immutable lowercase 40- or 64-hex revision"
        )
    return value


def _resolve_path(base: Path, value: object, label: str) -> Path:
    raw = _require_string(value, label)
    path = Path(raw)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _verify_regular_file(path: Path, expected_sha256: str, label: str) -> None:
    try:
        mode = path.stat().st_mode
    except OSError as error:
        raise ValueError(f"{label} is unavailable: {path}") from error
    if not stat.S_ISREG(mode):
        raise ValueError(f"{label} is not a regular file: {path}")
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {observed}"
        )


def strict_json_loads(value: str, label: str) -> Any:
    """Decode JSON while rejecting duplicate keys and non-finite numbers."""

    def reject_constant(constant: str) -> None:
        raise ValueError(f"{label} contains non-finite number {constant!r}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = item
        return result

    def reject_nonfinite_tree(item: Any, path: str = "$") -> None:
        if type(item) is float and not math.isfinite(item):
            raise ValueError(f"{label} contains non-finite number at {path}")
        if type(item) is dict:
            for key, child in item.items():
                reject_nonfinite_tree(child, f"{path}.{key}")
        elif type(item) is list:
            for index, child in enumerate(item):
                reject_nonfinite_tree(child, f"{path}[{index}]")

    try:
        decoded = json.loads(
            value,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
        reject_nonfinite_tree(decoded)
        return decoded
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is malformed JSON") from error


def _json_file(path: Path, label: str) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"{label} is not readable JSON: {path}") from error
    value = strict_json_loads(text, label)
    return _require_object(value, label)


def _dataset_dir(data_dir: Path | None = None) -> Path:
    if data_dir is not None:
        return data_dir
    return Path(__file__).resolve().parent / "llm-inference-bench" / "data"


def load_dataset_contract(
    profile: str,
    data_dir: Path | None = None,
) -> dict[str, dict[str, str]]:
    """Load and independently verify the exact item contract for one profile."""
    if profile not in PROFILE_CONTRACTS:
        raise ValueError(f"unsupported accuracy profile {profile!r}")
    contract = PROFILE_CONTRACTS[profile]
    path = _dataset_dir(data_dir) / str(contract["dataset_filename"])
    _verify_regular_file(path, str(contract["dataset_sha256"]), f"{profile} dataset")
    rows: list[dict[str, Any]] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = strict_json_loads(raw, f"{path}:{number}")
        except ValueError as error:
            raise ValueError(f"{path}:{number} is not JSON") from error
        rows.append(_require_object(row, f"{path}:{number}"))
    expected_count = int(contract["items"])
    if len(rows) != expected_count:
        raise ValueError(
            f"{profile} dataset has {len(rows)} rows, expected {expected_count}"
        )

    items: dict[str, dict[str, str]] = {}
    if profile == "gsm8k":
        for index, row in enumerate(rows):
            question = row.get("question")
            answer = row.get("answer")
            if (
                type(question) is not str
                or not question.strip()
                or type(answer) is not str
            ):
                raise ValueError(f"gsm8k dataset row {index} is malformed")
            marker = answer.rsplit("#### ", 1)
            if len(marker) != 2 or not marker[1].strip():
                raise ValueError(f"gsm8k dataset row {index} lacks a final answer")
            item_id = f"gsm8k-{index:04d}"
            items[item_id] = {
                "category": "",
                "expected_answer": marker[1].strip().replace(",", ""),
            }
    else:
        categories: dict[str, int] = {}
        for index, row in enumerate(rows):
            question_id = row.get("question_id")
            category = row.get("category")
            answer = row.get("answer")
            options = row.get("options")
            if type(question_id) is not int or question_id < 0:
                raise ValueError(
                    f"mmlu-pro dataset row {index} has an invalid question_id"
                )
            if type(category) is not str or not category:
                raise ValueError(f"mmlu-pro dataset row {index} has no category")
            if type(answer) is not str or re.fullmatch(r"[A-J]", answer) is None:
                raise ValueError(f"mmlu-pro dataset row {index} has an invalid answer")
            if type(options) is not list or not 1 <= len(options) <= 10:
                raise ValueError(f"mmlu-pro dataset row {index} has invalid options")
            item_id = f"mmlupro-{question_id}"
            categories[category] = categories.get(category, 0) + 1
            items[item_id] = {
                "category": category,
                "expected_answer": answer,
            }
        if categories != MMLU_CATEGORY_COUNTS:
            raise ValueError(f"mmlu-pro category allocation differs: {categories!r}")

    if len(items) != expected_count:
        raise ValueError(f"{profile} dataset item IDs are not unique")
    return items


def _validate_summary(
    value: object,
    expected_count: int,
    correct_count: int,
    unparseable: int,
    score_counts: dict[str, int],
    label: str,
) -> None:
    summary = _require_object(value, label)
    required = {
        "attempted",
        "completed",
        "errors",
        "correct",
        "wrong",
        "score_counts",
        "exact",
        "fail",
        "truncated",
        "stalled",
        "timeout",
        "unparseable",
        "incomplete",
        "correct_rate",
        "correct_rate_completed",
        "score_available",
        "hit_max_tokens",
        "estimated_token_runs",
    }
    _require_keys(summary, required, label)
    wrong_count = expected_count - correct_count
    exact_counts = {key: value for key, value in score_counts.items() if value}
    checks = {
        "attempted": expected_count,
        "completed": expected_count,
        "errors": 0,
        "correct": correct_count,
        "wrong": wrong_count,
        "exact": correct_count,
        "fail": wrong_count,
        "truncated": 0,
        "stalled": 0,
        "timeout": 0,
        "unparseable": unparseable,
        "incomplete": 0,
        "hit_max_tokens": 0,
        "estimated_token_runs": 0,
    }
    for key, expected in checks.items():
        _require_int(summary.get(key), expected, f"{label}.{key}")
    observed_counts = _require_object(
        summary.get("score_counts"), f"{label}.score_counts"
    )
    if any(
        type(key) is not str or type(value) is not int
        for key, value in observed_counts.items()
    ):
        raise ValueError(f"{label}.score_counts must map strings to exact integers")
    if observed_counts != exact_counts:
        raise ValueError(
            f"{label}.score_counts differs: expected {exact_counts!r}, got {observed_counts!r}"
        )
    _require_bool(summary.get("score_available"), True, f"{label}.score_available")
    expected_rate = correct_count / expected_count
    _require_number(summary.get("correct_rate"), expected_rate, f"{label}.correct_rate")
    _require_number(
        summary.get("correct_rate_completed"),
        expected_rate,
        f"{label}.correct_rate_completed",
    )


def _validate_category_summaries(
    value: object,
    runs: dict[str, dict[str, Any]],
    profile: str,
) -> None:
    rows = _require_array(value, "category_summaries")
    if profile == "gsm8k":
        if rows:
            raise ValueError("gsm8k category_summaries must be empty")
        return
    if len(rows) != len(MMLU_CATEGORY_COUNTS):
        raise ValueError("mmlu-pro must contain all 14 category summaries")
    observed: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        row = _require_object(raw, f"category_summaries[{index}]")
        _require_keys(
            row,
            {"category", "scored", "correct", "accuracy"},
            f"category_summaries[{index}]",
        )
        category = _require_string(
            row.get("category"), f"category_summaries[{index}].category"
        )
        if category in observed:
            raise ValueError(f"duplicate category summary {category!r}")
        observed[category] = row
    if set(observed) != set(MMLU_CATEGORY_COUNTS):
        raise ValueError(
            "mmlu-pro category names differ from the pinned 14-category set"
        )
    total = 0
    for category, expected_scored in MMLU_CATEGORY_COUNTS.items():
        row = observed[category]
        category_runs = [run for run in runs.values() if run["category"] == category]
        correct = sum(1 for run in category_runs if run["correct"] is True)
        _require_int(row.get("scored"), expected_scored, f"category {category}.scored")
        _require_int(row.get("correct"), correct, f"category {category}.correct")
        _require_number(
            row.get("accuracy"),
            correct / expected_scored,
            f"category {category}.accuracy",
        )
        total += expected_scored
    if total != 1_000:
        raise AssertionError("pinned MMLU-Pro category allocation does not total 1000")


def validate_report(
    report: object,
    profile: str,
    model: str,
    *,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Validate one full Lil result and return its normalized per-item view."""
    value = _require_object(report, f"{profile} report")
    _require_keys(
        value,
        {
            "metadata",
            "prefill_scout",
            "selected_concurrency",
            "selected_summary",
            "all_summary",
            "level_summaries",
            "accuracy",
            "category_summaries",
            "wrong_runs",
            "runs",
        },
        f"{profile} report",
    )
    if profile not in PROFILE_CONTRACTS:
        raise ValueError(f"unsupported accuracy profile {profile!r}")
    contract = PROFILE_CONTRACTS[profile]
    expected_count = int(contract["items"])
    expected_items = load_dataset_contract(profile, data_dir)

    metadata = _require_object(value.get("metadata"), f"{profile}.metadata")
    metadata_required = {
        "version",
        "mode",
        "interrupted",
        "model",
        "models",
        "test_profile",
        "max_tokens",
        "token_limit_field",
        "max_tokens_omitted",
        "fixed_concurrency",
        "requested_runs",
        "concurrency_levels_requested",
        "min_results",
        "score_source",
        "correct_regex",
        "profile_scorer",
        "scorer_version",
        "profile_version",
        "dataset",
        "dataset_sha256",
        "dataset_items_total",
        "dataset_items_selected",
        "prefill_scout",
        "temperature",
        "top_p",
        "seed_base",
        "reasoning_effort",
        "request_overrides",
        "system_prompt",
    }
    _require_keys(metadata, metadata_required, f"{profile}.metadata")
    exact_strings = {
        "version": LIL_VERSION,
        "mode": "completion_stats",
        "model": model,
        "test_profile": profile,
        "token_limit_field": "max_tokens",
        "score_source": "final_answer",
        "correct_regex": "",
        "profile_scorer": str(contract["scorer"]),
        "dataset": str(contract["dataset"]),
        "dataset_sha256": str(contract["dataset_sha256"]),
    }
    for key, expected in exact_strings.items():
        observed = _require_string(
            metadata.get(key), f"{profile}.metadata.{key}", nonempty=bool(expected)
        )
        if observed != expected:
            raise ValueError(
                f"{profile}.metadata.{key} must equal {expected!r}, got {observed!r}"
            )
    _require_bool(metadata.get("interrupted"), False, f"{profile}.metadata.interrupted")
    _require_bool(
        metadata.get("max_tokens_omitted"),
        False,
        f"{profile}.metadata.max_tokens_omitted",
    )
    _require_bool(
        metadata.get("prefill_scout"), False, f"{profile}.metadata.prefill_scout"
    )
    _require_bool(
        metadata.get("system_prompt"), False, f"{profile}.metadata.system_prompt"
    )
    for key, expected in {
        "max_tokens": TOKEN_BUDGET,
        "fixed_concurrency": 1,
        "requested_runs": expected_count,
        "min_results": expected_count,
        "scorer_version": SCORER_VERSION,
        "profile_version": PROFILE_VERSION,
        "dataset_items_total": expected_count,
        "dataset_items_selected": expected_count,
    }.items():
        _require_int(metadata.get(key), expected, f"{profile}.metadata.{key}")
    levels = _require_array(
        metadata.get("concurrency_levels_requested"),
        f"{profile}.metadata.concurrency_levels_requested",
    )
    if levels != [1] or any(type(item) is not int for item in levels):
        raise ValueError(f"{profile} concurrency_levels_requested must be exactly [1]")
    models = _require_array(metadata.get("models"), f"{profile}.metadata.models")
    if (
        not models
        or any(type(item) is not str for item in models)
        or model not in models
    ):
        raise ValueError(
            f"{profile}.metadata.models must contain the exact qualified model"
        )
    if type(metadata.get("temperature")) is not float or metadata["temperature"] != 0.0:
        raise ValueError(f"{profile}.metadata.temperature must be exactly 0.0")
    for key in ("top_p", "seed_base", "reasoning_effort"):
        if metadata.get(key) is not None:
            raise ValueError(f"{profile}.metadata.{key} must be null")
    overrides = _require_object(
        metadata.get("request_overrides"),
        f"{profile}.metadata.request_overrides",
    )
    expected_overrides = {"report_interventions": True}
    if overrides != expected_overrides:
        raise ValueError(
            f"{profile}.metadata.request_overrides must equal "
            f"{expected_overrides!r}, got {overrides!r}"
        )

    if value.get("prefill_scout") is not None:
        raise ValueError(f"{profile}.prefill_scout must be null")
    _require_int(
        value.get("selected_concurrency"), 1, f"{profile}.selected_concurrency"
    )

    raw_runs = _require_array(value.get("runs"), f"{profile}.runs")
    if len(raw_runs) != expected_count:
        raise ValueError(
            f"{profile} has {len(raw_runs)} runs, expected {expected_count}"
        )
    runs: dict[str, dict[str, Any]] = {}
    run_indices: set[int] = set()
    score_counts: dict[str, int] = {}
    for index, raw in enumerate(raw_runs):
        run = _require_object(raw, f"{profile}.runs[{index}]")
        required = {
            "run_index",
            "phase",
            "concurrency",
            "ok",
            "correct",
            "completion_tokens",
            "prompt_tokens",
            "finish_reason",
            "final_answer",
            "score_label",
            "score_detail",
            "parsed_answer",
            "error",
            "hit_max_tokens",
            "estimated_tokens",
            "cancelled",
            "item_id",
            "category",
            "expected_answer",
            "interventions",
        }
        _require_keys(run, required, f"{profile}.runs[{index}]")
        run_index = _require_positive_int(
            run.get("run_index"), f"{profile}.runs[{index}].run_index"
        )
        if run_index in run_indices:
            raise ValueError(f"{profile} has duplicate run_index {run_index}")
        run_indices.add(run_index)
        phase = _require_string(run.get("phase"), f"{profile}.runs[{index}].phase")
        if phase != "profile":
            raise ValueError(f"{profile} run {run_index} phase must be 'profile'")
        _require_int(
            run.get("concurrency"), 1, f"{profile} run {run_index}.concurrency"
        )
        _require_bool(run.get("ok"), True, f"{profile} run {run_index}.ok")
        if type(run.get("correct")) is not bool:
            raise ValueError(f"{profile} run {run_index}.correct must be a boolean")
        _require_positive_int(
            run.get("prompt_tokens"), f"{profile} run {run_index}.prompt_tokens"
        )
        _require_positive_int(
            run.get("completion_tokens"), f"{profile} run {run_index}.completion_tokens"
        )
        finish_reason = _require_string(
            run.get("finish_reason"),
            f"{profile} run {run_index}.finish_reason",
        ).lower()
        if finish_reason in FORBIDDEN_FINISH_REASONS:
            raise ValueError(
                f"{profile} run {run_index} has forbidden finish_reason {finish_reason!r}"
            )
        for key in (
            "final_answer",
            "score_label",
            "score_detail",
            "parsed_answer",
            "error",
            "item_id",
            "category",
            "expected_answer",
        ):
            _require_string(
                run.get(key),
                f"{profile} run {run_index}.{key}",
                nonempty=key
                not in {"final_answer", "parsed_answer", "error", "category"},
            )
        if run["error"]:
            raise ValueError(f"{profile} run {run_index} contains an error")
        _require_bool(
            run.get("hit_max_tokens"),
            False,
            f"{profile} run {run_index}.hit_max_tokens",
        )
        _require_bool(
            run.get("estimated_tokens"),
            False,
            f"{profile} run {run_index}.estimated_tokens",
        )
        _require_bool(
            run.get("cancelled"), False, f"{profile} run {run_index}.cancelled"
        )
        _require_empty_intervention_ledger(
            run.get("interventions"), f"{profile} run {run_index}.interventions"
        )
        item_id = run["item_id"]
        if item_id in runs:
            raise ValueError(f"{profile} has duplicate item_id {item_id!r}")
        if item_id not in expected_items:
            raise ValueError(f"{profile} contains unknown item_id {item_id!r}")
        expected = expected_items[item_id]
        if run["category"] != expected["category"]:
            raise ValueError(
                f"{profile} item {item_id} category differs from the pinned dataset"
            )
        if run["expected_answer"] != expected["expected_answer"]:
            raise ValueError(
                f"{profile} item {item_id} expected answer differs from the pinned dataset"
            )
        label = run["score_label"]
        if label not in ("exact", "fail"):
            raise ValueError(
                f"{profile} item {item_id} has non-final score label {label!r}"
            )
        if (label == "exact") is not run["correct"]:
            raise ValueError(
                f"{profile} item {item_id} score label and correctness differ"
            )
        score_counts[label] = score_counts.get(label, 0) + 1
        runs[item_id] = run
    if set(run_indices) != set(range(1, expected_count + 1)):
        raise ValueError(
            f"{profile} run_index coverage must be exactly 1..{expected_count}"
        )
    if set(runs) != set(expected_items):
        missing = sorted(set(expected_items) - set(runs))
        raise ValueError(f"{profile} is missing pinned item IDs: {missing[:5]!r}")

    correct_count = sum(1 for run in runs.values() if run["correct"] is True)
    wrong_count = expected_count - correct_count
    unparseable = sum(
        1 for run in runs.values() if str(run["score_detail"]).startswith("unparseable")
    )
    _validate_summary(
        value.get("selected_summary"),
        expected_count,
        correct_count,
        unparseable,
        score_counts,
        f"{profile}.selected_summary",
    )
    _validate_summary(
        value.get("all_summary"),
        expected_count,
        correct_count,
        unparseable,
        score_counts,
        f"{profile}.all_summary",
    )
    levels_summary = _require_array(
        value.get("level_summaries"), f"{profile}.level_summaries"
    )
    if len(levels_summary) != 1:
        raise ValueError(
            f"{profile}.level_summaries must contain only the fixed C1 row"
        )
    level = _require_object(levels_summary[0], f"{profile}.level_summaries[0]")
    _require_keys(
        level,
        {"concurrency", "phase", "summary", "selected", "improved"},
        f"{profile}.level_summaries[0]",
    )
    _require_int(
        level.get("concurrency"), 1, f"{profile}.level_summaries[0].concurrency"
    )
    if level.get("phase") != "profile":
        raise ValueError(f"{profile}.level_summaries[0].phase must be 'profile'")
    _require_bool(level.get("selected"), True, f"{profile}.level_summaries[0].selected")
    _require_bool(level.get("improved"), True, f"{profile}.level_summaries[0].improved")
    _validate_summary(
        level.get("summary"),
        expected_count,
        correct_count,
        unparseable,
        score_counts,
        f"{profile}.level_summaries[0].summary",
    )

    accuracy = _require_object(value.get("accuracy"), f"{profile}.accuracy")
    _require_keys(
        accuracy,
        {
            "items_total",
            "items_selected",
            "attempted",
            "scored",
            "correct",
            "accuracy",
            "unparseable",
            "truncated_no_answer",
            "hit_max_tokens",
            "errors",
            "dataset_exhausted",
        },
        f"{profile}.accuracy",
    )
    for key, expected in {
        "items_total": expected_count,
        "items_selected": expected_count,
        "attempted": expected_count,
        "scored": expected_count,
        "correct": correct_count,
        "unparseable": unparseable,
        "truncated_no_answer": 0,
        "hit_max_tokens": 0,
        "errors": 0,
    }.items():
        _require_int(accuracy.get(key), expected, f"{profile}.accuracy.{key}")
    _require_bool(
        accuracy.get("dataset_exhausted"),
        False,
        f"{profile}.accuracy.dataset_exhausted",
    )
    observed_accuracy = correct_count / expected_count
    _require_number(
        accuracy.get("accuracy"), observed_accuracy, f"{profile}.accuracy.accuracy"
    )
    floor = float(contract["provisional_floor"])
    if observed_accuracy < floor:
        raise ValueError(
            f"{profile} provisional sanity floor failed: {observed_accuracy:.6f} < {floor:.6f}"
        )

    _validate_category_summaries(value.get("category_summaries"), runs, profile)

    wrong_rows = _require_array(value.get("wrong_runs"), f"{profile}.wrong_runs")
    wrong_ids = {item_id for item_id, run in runs.items() if run["correct"] is False}
    observed_wrong_ids: set[str] = set()
    for index, raw in enumerate(wrong_rows):
        row = _require_object(raw, f"{profile}.wrong_runs[{index}]")
        item_id = _require_string(
            row.get("item_id"), f"{profile}.wrong_runs[{index}].item_id"
        )
        if item_id in observed_wrong_ids or item_id not in wrong_ids:
            raise ValueError(
                f"{profile}.wrong_runs contains an unexpected or duplicate item {item_id!r}"
            )
        observed_wrong_ids.add(item_id)
        for key in ("correct", "parsed_answer", "final_answer", "completion_tokens"):
            if row.get(key) != runs[item_id].get(key) or type(row.get(key)) is not type(
                runs[item_id].get(key)
            ):
                raise ValueError(
                    f"{profile}.wrong_runs item {item_id} differs on {key}"
                )
    if observed_wrong_ids != wrong_ids or len(wrong_rows) != wrong_count:
        raise ValueError(
            f"{profile}.wrong_runs does not exactly cover every wrong item"
        )

    normalized_items = {
        item_id: {
            "correct": run["correct"],
            "parsed_answer": run["parsed_answer"],
            "final_answer": run["final_answer"],
            "completion_tokens": run["completion_tokens"],
            "category": run["category"],
            "expected_answer": run["expected_answer"],
        }
        for item_id, run in runs.items()
    }
    return {
        "profile": profile,
        "model": model,
        "items": normalized_items,
        "item_count": expected_count,
        "correct": correct_count,
        "accuracy": observed_accuracy,
        "provisional_floor": floor,
        "provisional_floor_pass": True,
        "completion_tokens": sum(run["completion_tokens"] for run in runs.values()),
    }


def validate_pair(
    no_spec: dict[str, Any],
    k5: dict[str, Any],
) -> dict[str, Any]:
    """Require exact item-level greedy parity for one dataset profile."""
    if no_spec.get("profile") != k5.get("profile"):
        raise ValueError("cannot compare reports from different profiles")
    if no_spec.get("model") != k5.get("model"):
        raise ValueError("no-spec and K5 reports name different models")
    base_items = _require_object(no_spec.get("items"), "no_spec.items")
    k5_items = _require_object(k5.get("items"), "k5.items")
    if set(base_items) != set(k5_items):
        raise ValueError("no-spec and K5 item ID sets differ")
    parity_fields = (
        "correct",
        "parsed_answer",
        "final_answer",
        "completion_tokens",
    )
    for item_id in sorted(base_items):
        base = _require_object(base_items[item_id], f"no_spec item {item_id}")
        candidate = _require_object(k5_items[item_id], f"k5 item {item_id}")
        for field in parity_fields:
            if type(base.get(field)) is not type(candidate.get(field)) or base.get(
                field
            ) != candidate.get(field):
                raise ValueError(
                    f"matched parity failed for {item_id} field {field}: "
                    f"no_spec={base.get(field)!r} k5={candidate.get(field)!r}"
                )
    return {
        "profile": no_spec["profile"],
        "matched_items": len(base_items),
        "fields": list(parity_fields),
        "mismatches": 0,
        "status": "PASS",
        "authority": "authoritative speculative-decoding correctness gate",
    }


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            raise ValueError(f"{path}:{number}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        if re.fullmatch(r"[A-Z_][A-Z0-9_]*", key) is None:
            raise ValueError(f"{path}:{number}: invalid environment key")
        values[key] = value.strip().strip('"').strip("'")
    return values


def _argv_value(argv: list[str], flag: str) -> str | None:
    indices = [index for index, item in enumerate(argv) if item == flag]
    if len(indices) > 1:
        raise ValueError(f"argv contains duplicate {flag}")
    if not indices:
        return None
    index = indices[0]
    if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
        raise ValueError(f"argv flag {flag} lacks a value")
    return argv[index + 1]


def validate_arm_argv(argv_value: object, arm: str, binary_path: Path) -> list[str]:
    argv = _require_array(argv_value, f"arms.{arm}.argv")
    if not argv or any(type(item) is not str or not item for item in argv):
        raise ValueError(f"arms.{arm}.argv must be a non-empty string array")
    if any(item.startswith("--") and "=" in item for item in argv):
        raise ValueError(f"arms.{arm}.argv must use unambiguous separate flag values")
    argv_binary = Path(argv[0])
    if not argv_binary.is_absolute():
        argv_binary = binary_path.parent / argv_binary
    if argv_binary.resolve() != binary_path.resolve():
        raise ValueError(f"arms.{arm}.argv[0] differs from the bound binary")
    for flag in ("--max-batch-size", "--max-num-seqs"):
        if _argv_value(argv, flag) != "1":
            raise ValueError(f"arms.{arm}.argv requires {flag} 1")
    present = {item for item in argv if item.startswith("--")}
    if present & FORBIDDEN_DFLASH_FLAGS:
        raise ValueError(
            f"arms.{arm}.argv must not enable DFlash or an external draft model"
        )
    if arm == "no_spec":
        forbidden = SPEC_BOOL_FLAGS | SPEC_VALUE_FLAGS
        if present & forbidden:
            raise ValueError("no-spec argv contains speculative-decoding flags")
    elif arm == "k5":
        if "--speculative" not in present:
            raise ValueError("K5 argv lacks --speculative")
        if _argv_value(argv, "--num-drafts") != "4":
            raise ValueError("K5 argv requires --num-drafts 4")
        if _argv_value(argv, "--mtp-quantization") != "bf16":
            raise ValueError("K5 argv requires --mtp-quantization bf16")
        mtp_vocab = _argv_value(argv, "--mtp-vocab")
        if mtp_vocab is None or re.fullmatch(r"[0-9]+", mtp_vocab) is None:
            raise ValueError("K5 argv requires an explicit non-negative --mtp-vocab")
    else:
        raise ValueError(f"unknown arm {arm!r}")
    return argv


def normalized_non_spec_argv(argv: list[str]) -> list[str]:
    normalized: list[str] = []
    index = 0
    while index < len(argv):
        item = argv[index]
        if item in SPEC_BOOL_FLAGS:
            index += 1
            continue
        if item in SPEC_VALUE_FLAGS:
            if index + 1 >= len(argv):
                raise ValueError(f"argv flag {item} lacks a value")
            index += 2
            continue
        normalized.append(item)
        index += 1
    return normalized


def _validate_recipe_mode(values: dict[str, str], arm: str) -> None:
    disabled = (
        values.get("FLUX_SPEC_MODE", "").lower() in {"none", "off", "0", "false"}
        or values.get("FLUX_DISABLE_SPECULATION", "").lower() in {"1", "true", "yes"}
        or values.get("ATLAS_DISABLE_SPECULATION", "").lower() in {"1", "true", "yes"}
    )
    if arm == "no_spec":
        if not disabled:
            raise ValueError(
                "no-spec recipe lacks an explicit speculation-disable selector"
            )
        return
    if disabled:
        raise ValueError("K5 recipe disables speculation")
    k5_enabled = any(
        values.get(key) == "1"
        for key in (
            "FLUX_QWEN4_K5_HYBRID",
            "ATLAS_QWEN4_K5_HYBRID",
        )
    )
    if not k5_enabled:
        raise ValueError("K5 recipe lacks an explicit Qwen4 K5 selector")


def normalized_recipe(values: dict[str, str]) -> dict[str, str]:
    """Remove only the arm selector from an otherwise matched recipe."""
    arm_selectors = {
        "ATLAS_DISABLE_SPECULATION",
        "FLUX_DISABLE_SPECULATION",
        "FLUX_SPEC_MODE",
    }
    return {key: value for key, value in values.items() if key not in arm_selectors}


def load_and_validate_envelope(
    envelope_path: Path,
    *,
    data_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate provenance, all reports, and both matched profile pairs."""
    envelope = _json_file(envelope_path, "provenance envelope")
    _require_exact_keys(
        envelope, {"schema", "model", "artifacts", "arms"}, "provenance envelope"
    )
    if envelope.get("schema") != ENVELOPE_SCHEMA:
        raise ValueError(f"provenance envelope schema must equal {ENVELOPE_SCHEMA!r}")
    model = _require_string(envelope.get("model"), "provenance envelope.model")
    artifacts = _require_object(
        envelope.get("artifacts"), "provenance envelope.artifacts"
    )
    artifact_keys = {
        "image_digest",
        "image_id",
        "binary_path",
        "binary_sha256",
        "source_revision",
        "source_inventory_path",
        "source_inventory_sha256",
        "checkpoint_id",
        "checkpoint_revision",
        "checkpoint_inventory_path",
        "checkpoint_inventory_sha256",
    }
    _require_exact_keys(artifacts, artifact_keys, "provenance envelope.artifacts")
    _require_sha256(
        artifacts.get("image_digest"), "artifacts.image_digest", prefixed=True
    )
    _require_sha256(artifacts.get("image_id"), "artifacts.image_id", prefixed=True)
    _require_sha256(artifacts.get("binary_sha256"), "artifacts.binary_sha256")
    _require_revision(artifacts.get("source_revision"), "artifacts.source_revision")
    _require_sha256(
        artifacts.get("source_inventory_sha256"), "artifacts.source_inventory_sha256"
    )
    _require_string(artifacts.get("checkpoint_id"), "artifacts.checkpoint_id")
    _require_revision(
        artifacts.get("checkpoint_revision"), "artifacts.checkpoint_revision"
    )
    _require_sha256(
        artifacts.get("checkpoint_inventory_sha256"),
        "artifacts.checkpoint_inventory_sha256",
    )
    base = envelope_path.resolve().parent
    binary_path = _resolve_path(
        base, artifacts.get("binary_path"), "artifacts.binary_path"
    )
    source_inventory_path = _resolve_path(
        base,
        artifacts.get("source_inventory_path"),
        "artifacts.source_inventory_path",
    )
    checkpoint_inventory_path = _resolve_path(
        base,
        artifacts.get("checkpoint_inventory_path"),
        "artifacts.checkpoint_inventory_path",
    )
    _verify_regular_file(binary_path, artifacts["binary_sha256"], "bound server binary")
    if not os.access(binary_path, os.X_OK):
        raise ValueError("bound server binary is not executable")
    _verify_regular_file(
        source_inventory_path,
        artifacts["source_inventory_sha256"],
        "source inventory",
    )
    _verify_regular_file(
        checkpoint_inventory_path,
        artifacts["checkpoint_inventory_sha256"],
        "checkpoint inventory",
    )
    _json_file(source_inventory_path, "source inventory")
    _json_file(checkpoint_inventory_path, "checkpoint inventory")

    arms = _require_object(envelope.get("arms"), "provenance envelope.arms")
    _require_exact_keys(arms, set(ARM_MODES), "provenance envelope.arms")
    arm_reports: dict[str, dict[str, dict[str, Any]]] = {}
    arm_receipts: dict[str, dict[str, Any]] = {}
    resolved_report_paths: set[Path] = set()
    resolved_recipe_paths: dict[str, Path] = {}
    recipe_values_by_arm: dict[str, dict[str, str]] = {}
    validated_argv: dict[str, list[str]] = {}
    report_hashes: dict[tuple[str, str], str] = {}
    for arm, expected_mode in ARM_MODES.items():
        raw_arm = _require_object(arms.get(arm), f"arms.{arm}")
        _require_exact_keys(
            raw_arm,
            {"mode", "recipe_path", "recipe_sha256", "argv", "argv_sha256", "reports"},
            f"arms.{arm}",
        )
        mode = _require_string(raw_arm.get("mode"), f"arms.{arm}.mode")
        if mode != expected_mode:
            raise ValueError(f"arms.{arm}.mode must equal {expected_mode!r}")
        recipe_sha = _require_sha256(
            raw_arm.get("recipe_sha256"), f"arms.{arm}.recipe_sha256"
        )
        argv_sha = _require_sha256(
            raw_arm.get("argv_sha256"), f"arms.{arm}.argv_sha256"
        )
        recipe_path = _resolve_path(
            base, raw_arm.get("recipe_path"), f"arms.{arm}.recipe_path"
        )
        _verify_regular_file(recipe_path, recipe_sha, f"{arm} recipe")
        recipe_values = parse_env_file(recipe_path)
        _validate_recipe_mode(recipe_values, arm)
        argv = validate_arm_argv(raw_arm.get("argv"), arm, binary_path)
        observed_argv_sha = sha256_bytes(canonical_bytes(argv))
        if observed_argv_sha != argv_sha:
            raise ValueError(
                f"arms.{arm}.argv_sha256 mismatch: expected {argv_sha}, got {observed_argv_sha}"
            )
        resolved_recipe_paths[arm] = recipe_path
        recipe_values_by_arm[arm] = recipe_values
        validated_argv[arm] = argv

        refs = _require_object(raw_arm.get("reports"), f"arms.{arm}.reports")
        _require_exact_keys(refs, set(PROFILE_CONTRACTS), f"arms.{arm}.reports")
        arm_reports[arm] = {}
        bound_reports: dict[str, Any] = {}
        for profile in PROFILE_CONTRACTS:
            ref = _require_object(refs.get(profile), f"arms.{arm}.reports.{profile}")
            _require_exact_keys(
                ref, {"path", "sha256"}, f"arms.{arm}.reports.{profile}"
            )
            report_sha = _require_sha256(
                ref.get("sha256"),
                f"arms.{arm}.reports.{profile}.sha256",
            )
            report_path = _resolve_path(
                base,
                ref.get("path"),
                f"arms.{arm}.reports.{profile}.path",
            )
            if report_path in resolved_report_paths:
                raise ValueError("each qualification report must be a distinct file")
            resolved_report_paths.add(report_path)
            _verify_regular_file(report_path, report_sha, f"{arm} {profile} report")
            report_value = _json_file(report_path, f"{arm} {profile} report")
            normalized = validate_report(
                report_value, profile, model, data_dir=data_dir
            )
            arm_reports[arm][profile] = normalized
            report_hashes[(arm, profile)] = report_sha
            bound_reports[profile] = {
                "sha256": report_sha,
                "items": normalized["item_count"],
                "correct": normalized["correct"],
                "accuracy": normalized["accuracy"],
                "provisional_floor": normalized["provisional_floor"],
                "provisional_floor_pass": True,
            }
        arm_receipts[arm] = {
            "mode": expected_mode,
            "recipe_sha256": recipe_sha,
            "argv_sha256": argv_sha,
            "reports": bound_reports,
        }

    if resolved_recipe_paths["no_spec"] == resolved_recipe_paths["k5"]:
        raise ValueError("no-spec and K5 must use distinct, explicitly bound recipes")
    if normalized_recipe(recipe_values_by_arm["no_spec"]) != normalized_recipe(
        recipe_values_by_arm["k5"]
    ):
        raise ValueError(
            "no-spec and K5 recipes differ outside the speculation arm selector"
        )
    if normalized_non_spec_argv(validated_argv["no_spec"]) != normalized_non_spec_argv(
        validated_argv["k5"]
    ):
        raise ValueError("no-spec and K5 server argv differ outside native-MTP flags")
    for profile in PROFILE_CONTRACTS:
        if report_hashes[("no_spec", profile)] == report_hashes[("k5", profile)]:
            raise ValueError(
                f"{profile} no-spec and K5 reports are byte-identical, not distinct runs"
            )

    parity = {
        profile: validate_pair(
            arm_reports["no_spec"][profile], arm_reports["k5"][profile]
        )
        for profile in PROFILE_CONTRACTS
    }
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "PASS",
        "model": model,
        "envelope_sha256": sha256_file(envelope_path),
        "provenance": {
            "image_digest": artifacts["image_digest"],
            "image_id": artifacts["image_id"],
            "binary_sha256": artifacts["binary_sha256"],
            "source_revision": artifacts["source_revision"],
            "source_inventory_sha256": artifacts["source_inventory_sha256"],
            "checkpoint_id": artifacts["checkpoint_id"],
            "checkpoint_revision": artifacts["checkpoint_revision"],
            "checkpoint_inventory_sha256": artifacts["checkpoint_inventory_sha256"],
        },
        "arms": arm_receipts,
        "matched_parity": parity,
        "gate_policy": {
            "matched_parity": "authoritative",
            "accuracy_floors": "provisional sanity checks",
            "gsm8k_floor": PROFILE_CONTRACTS["gsm8k"]["provisional_floor"],
            "mmlu_pro_floor": PROFILE_CONTRACTS["mmlu-pro"]["provisional_floor"],
        },
    }
    return envelope, receipt


def seal_receipt(path: Path, receipt: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"receipt already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_bytes(receipt) + b"\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(fd, payload[offset:])
            if written <= 0:
                raise OSError("short write while sealing receipt")
            offset += written
        os.fchmod(fd, 0o444)
        os.fsync(fd)
    finally:
        os.close(fd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envelope", required=True, type=Path)
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path("lil-accuracy-pair-receipt.json"),
    )
    parsed = parser.parse_args(argv)
    _envelope, receipt = load_and_validate_envelope(parsed.envelope)
    seal_receipt(parsed.receipt, receipt)
    print(
        f"PASS receipt={parsed.receipt} "
        f"sha256={sha256_file(parsed.receipt)} "
        "matched=2319/2319"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
