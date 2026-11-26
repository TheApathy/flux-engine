# SPDX-License-Identifier: AGPL-3.0-only
"""Offline-only CPU regression tests for the 12-prompt fidelity harness (`fid_gate12`).

Verifies every positive and negative requirement and all four strict-contract invariants:

1. Pinned 12-prompt manifest integrity (`test_01`, `test_02`):
   - SHA-256 verification, exact 12-row count, unique IDs, contiguous index 0..11,
     `system_sha256` and `user_sha256` verification, and positive integer
     `expected_prompt_tokens_nothink`.
2. Collection (`collect_panel`) fail-closed validation with mocked HTTP (`test_03`, `test_04`, `test_05`):
   - Positive 12-prompt collection verifies default `max_tokens=64` and writes
     `fid.jsonl`, `.raw.jsonl`, and `.meta.json` with exact request caps
     (`max_tokens=8` in the fast mocked panel, `top_logprobs=5, seed=1234,
     enable_thinking=False, return_token_ids=True`).
   - Rejects missing/empty `logprobs` or `logprobs.content`, non-finite or positive
     logprobs, unsorted or wrong-cardinality `top_logprobs`, duplicate decoded token
     strings in `top_logprobs`, sampled token mismatching rank-0 `top_logprobs`,
     `len(logprobs.content) != usage.completion_tokens`, `finish_reason == "length"`
     with `len != max_tokens`, unexpected `finish_reason` or `flux_stop_reason`,
     missing `choice.token_ids` when `require_token_ids=True`, and wrong `prompt_tokens`.
3. Strict Saved-Panel Schema, Provenance, & Raw-Evidence Binding (Item 1 — `test_08`):
   - Rejects omitted required fields (`schema_version`, `index`, `finish_reason`,
     `completion_tokens`, `user_sha256`, `system_sha256`, `request_sha256`,
     `raw_response_sha256`, `requested_model`, `response_model`, `response_id`, etc.).
   - Rejects equal truncated artifacts (`finish_reason="length"` with `len < max_tokens`,
     or omitted `finish_reason` on both `ref` and `alt`).
   - Rejects provenance mismatches between `fid.jsonl`, `.meta.json` (`status != "completed"`,
     tampered `out_sha256` or `raw_out_sha256`), and `.raw.jsonl` (tampered raw response
     or panel record diverging from raw response).
   - Rejects cross-panel provenance mismatches (`seed`, `requested_model`, `prompt_tokens`).
4. Incremental Raw-Response & Failure/Progress Metadata Persistence (Item 2 — `test_09`):
   - Simulates failure after several successful requests (`e.g.`, request index `3` fails
     after requests `0, 1, 2` succeed) and verifies that:
     - `out_path` retains the 3 validated records on disk.
     - `raw_out_path` retains all 3 successful raw responses PLUS the 4th failing raw
       response/error record (`len == 4`).
     - `meta_out_path` is persisted on disk with `status == "failed"`,
       `completed_count == 3`, `failed_at_index == 3`, `failed_at_id == "HumanEval/65"`,
       and matching `out_sha256` / `raw_out_sha256`.
   - Verifies that comparing a partial failed panel against any panel fails closed.
5. Client-Bounded Wall Deadline Contract (Item 3 — `test_06`, `test_10`):
   - Tests both synthetic-clock total/request wall caps (`test_06`) and real-monotonic-clock
     bounded client wait on a slow/stalling transport (`test_10`, no live network calls),
     verifying that `collect_panel` aborts within the bounded client deadline and records
     `deadline_exceeded=True`, `uncertain_server_work=True`, and
     `server_cancellation_guaranteed=False` in `.raw.jsonl` and `.meta.json`.
6. Descriptive Comparisons by Default & Explicit Threshold Opt-In (Item 4 — `test_03`, `test_11`):
   - Verifies that `compare_panels` (and `compare` CLI) defaults to `enforce_gates=False`,
     emitting `evidence_valid=True`, `numerical_acceptance="not_evaluated"`, and
     `gate_passed=None` (never `gate_passed=True` when thresholds are disabled).
   - Verifies that complete, consistent sequences with different valid EOS lengths
     (`finish_reason="stop"` at `len=5` vs `finish_reason="length"` at `len=8`) succeed
     descriptively (`evidence_valid=True`, `diverge_at=5`,
     `diverge_reason="eos_length_difference(ref=8:length,alt=5:stop)"`, exit code `0`)
     by default, and only trigger `FidThresholdFailure` (exit code `3`) when
     `--enforce-gates` (`enforce_gates=True`) is explicitly passed and the reference
     margin at the EOS divergence point exceeds `max_diverge_margin`.
"""

from __future__ import annotations

import copy
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
FID_DIR = ROOT / "fid_gate12"
FID_MOD_PATH = FID_DIR / "fid_gate12.py"
MANIFEST_PATH = FID_DIR / "fid_gate12_manifest.jsonl"


def load_fid_module():
    loader = importlib.machinery.SourceFileLoader("fid_gate12_mod", str(FID_MOD_PATH))
    spec = importlib.util.spec_from_loader("fid_gate12_mod", loader)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fid_gate12_mod"] = mod
    loader.exec_module(mod)
    return mod


F = load_fid_module()


def make_mock_http_response(
    manifest_row: dict,
    max_tokens: int = 8,
    finish_reason: str = "length",
    base_tid: int = 1000,
    lp_shift: float = 0.0,
    diverge_at: int | None = None,
    diverge_ref_margin: float = 0.25,
    include_token_ids: bool = True,
    model: str = "default",
) -> dict:
    """Build a valid synthetic `/v1/chat/completions` JSON response for a manifest row."""
    content_lp = []
    tids = []
    pieces = []
    for pos in range(max_tokens):
        if diverge_at is not None and pos == diverge_at:
            tok0 = f"alt_{pos}"
            tid0 = base_tid + pos + 500
            lp0 = -0.20 + lp_shift
            lp1 = lp0 - diverge_ref_margin
        else:
            tok0 = f"tok_{pos}"
            tid0 = base_tid + pos
            lp0 = -0.10 + lp_shift
            lp1 = lp0 - 1.50

        top_cands = [
            {"token": tok0, "logprob": lp0, "bytes": None},
            {"token": f"cand1_{pos}", "logprob": lp1, "bytes": None},
            {"token": f"cand2_{pos}", "logprob": lp1 - 0.5, "bytes": None},
            {"token": f"cand3_{pos}", "logprob": lp1 - 1.0, "bytes": None},
            {"token": f"cand4_{pos}", "logprob": lp1 - 1.5, "bytes": None},
        ]
        content_lp.append(
            {
                "token": tok0,
                "logprob": lp0,
                "bytes": None,
                "top_logprobs": top_cands,
            }
        )
        tids.append(tid0)
        pieces.append(tok0)

    choice = {
        "index": 0,
        "message": {"role": "assistant", "content": "".join(pieces)},
        "finish_reason": finish_reason,
        "logprobs": {"content": content_lp},
    }
    if include_token_ids:
        choice["token_ids"] = tids

    return {
        "id": f"chatcmpl-mock-{manifest_row['index']}",
        "object": "chat.completion",
        "model": model,
        "choices": [choice],
        "usage": {
            "prompt_tokens": manifest_row["expected_prompt_tokens_nothink"],
            "completion_tokens": len(content_lp),
            "total_tokens": manifest_row["expected_prompt_tokens_nothink"] + len(content_lp),
        },
    }


class TestFidGate12Harness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest_rows = F.load_and_validate_manifest(MANIFEST_PATH)

    def test_01_pinned_manifest_integrity(self):
        self.assertEqual(len(self.manifest_rows), 12)
        expected_ids = [
            "LCB/abc392_d",
            "LCB/arc196_b",
            "LCB/abc400_d",
            "HumanEval/65",
            "HumanEval/153",
            "HumanEval/26",
            "LCB/abc390_b",
            "LCB/abc392_b",
            "LCB/abc391_b",
            "LCB/abc393_b",
            "long8k-a",
            "long8k-b",
        ]
        self.assertEqual([r["id"] for r in self.manifest_rows], expected_ids)

        with self.assertRaisesRegex(F.FidGateError, "manifest_sha256_mismatch"):
            F.load_and_validate_manifest(MANIFEST_PATH, expected_sha256="0" * 64)

    def test_02_manifest_rejects_duplicate_or_corrupted_rows(self):
        with tempfile.TemporaryDirectory() as td:
            bad_path = Path(td) / "bad_manifest.jsonl"
            rows = copy.deepcopy(self.manifest_rows)
            rows[1]["id"] = rows[0]["id"]
            rows[1]["user_sha256"] = F.sha256_text(rows[1]["user"])
            bad_path.write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(F.FidGateError, "manifest_duplicate_id"):
                F.load_and_validate_manifest(bad_path, expected_sha256=None)

            rows2 = copy.deepcopy(self.manifest_rows)
            rows2[0]["user_sha256"] = "f" * 64
            bad_path.write_text(
                "\n".join(json.dumps(r) for r in rows2) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(F.FidGateError, "manifest_user_sha256_mismatch"):
                F.load_and_validate_manifest(bad_path, expected_sha256=None)

    def test_03_collect_and_compare_positive_pass(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            ref_out = td_path / "ref.jsonl"
            alt_out = td_path / "alt.jsonl"

            by_user = {r["user"]: r for r in self.manifest_rows}
            self.assertEqual(F.DEFAULT_MAX_TOKENS, 64)

            def mock_ref_post(url: str, body: dict, timeout_s: float) -> dict:
                self.assertEqual(body["temperature"], 0.0)
                self.assertEqual(body["max_tokens"], 8)
                self.assertEqual(body["top_logprobs"], 5)
                self.assertEqual(body["seed"], 1234)
                self.assertTrue(body["return_token_ids"])
                self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
                mrow = by_user[body["messages"][1]["content"]]
                if mrow["index"] == 5:
                    resp = make_mock_http_response(mrow, max_tokens=8)
                    resp["choices"][0]["logprobs"]["content"][3]["top_logprobs"][1]["logprob"] = -0.35
                    return resp
                return make_mock_http_response(mrow, max_tokens=8)

            def mock_alt_post(url: str, body: dict, timeout_s: float) -> dict:
                mrow = by_user[body["messages"][1]["content"]]
                if mrow["index"] == 5:
                    return make_mock_http_response(
                        mrow, max_tokens=8, lp_shift=-0.02, diverge_at=3, diverge_ref_margin=0.25
                    )
                return make_mock_http_response(mrow, max_tokens=8, lp_shift=-0.02)

            meta_ref = F.collect_panel(
                "http://mock/v1/chat/completions",
                ref_out,
                max_tokens=8,
                http_post=mock_ref_post,
            )
            self.assertEqual(meta_ref["status"], "completed")
            self.assertEqual(meta_ref["panel_size"], 12)
            self.assertEqual(meta_ref["completed_count"], 12)
            self.assertTrue(ref_out.with_suffix(".raw.jsonl").is_file())
            self.assertTrue(ref_out.with_suffix(".meta.json").is_file())

            F.collect_panel(
                "http://mock/v1/chat/completions",
                alt_out,
                max_tokens=8,
                http_post=mock_alt_post,
            )

            # Default descriptive compare (`enforce_gates=False`):
            desc_summary = F.compare_panels(ref_out, alt_out, max_tokens=8, require_token_ids=True)
            self.assertTrue(desc_summary["evidence_valid"])
            self.assertEqual(desc_summary["numerical_acceptance"], "not_evaluated")
            self.assertFalse(desc_summary["enforce_gates"])
            self.assertIsNone(desc_summary["gate_passed"])
            self.assertEqual(desc_summary["first_token_top1_agree"], 12)
            self.assertEqual(desc_summary["continuations_identical"], 11)
            self.assertAlmostEqual(desc_summary["pos0_shared_top5_max_abs_dlp"], 0.02, places=5)

            # Explicit threshold evaluation (`enforce_gates=True`):
            gated_summary = F.compare_panels(
                ref_out,
                alt_out,
                max_tokens=8,
                require_token_ids=True,
                enforce_gates=True,
            )
            self.assertTrue(gated_summary["evidence_valid"])
            self.assertEqual(gated_summary["numerical_acceptance"], "passed")
            self.assertTrue(gated_summary["enforce_gates"])
            self.assertTrue(gated_summary["gate_passed"])

            rc = F.main(
                [
                    "compare",
                    str(ref_out),
                    str(alt_out),
                    "--max-tokens",
                    "8",
                    "--require-token-ids",
                    "--enforce-gates",
                ]
            )
            self.assertEqual(rc, 0)

    def test_04_rejects_missing_empty_or_nonfinite_logprobs(self):
        mrow = self.manifest_rows[0]

        r1 = make_mock_http_response(mrow, max_tokens=8)
        r1["choices"][0]["logprobs"] = None
        with self.assertRaisesRegex(F.FidGateError, "missing_logprobs_object"):
            F.validate_and_extract_response(mrow, r1, wall_s=0.1, max_tokens=8)

        r2 = make_mock_http_response(mrow, max_tokens=8)
        r2["choices"][0]["logprobs"]["content"] = []
        with self.assertRaisesRegex(F.FidGateError, "empty_or_missing_logprobs_content"):
            F.validate_and_extract_response(mrow, r2, wall_s=0.1, max_tokens=8)

        r3 = make_mock_http_response(mrow, max_tokens=8)
        r3["choices"][0]["logprobs"]["content"][0]["logprob"] = float("nan")
        with self.assertRaisesRegex(F.FidGateError, "non_finite_or_positive_logprob"):
            F.validate_and_extract_response(mrow, r3, wall_s=0.1, max_tokens=8)

        r4 = make_mock_http_response(mrow, max_tokens=8)
        r4["choices"][0]["logprobs"]["content"][0]["top_logprobs"][2]["logprob"] = float("-inf")
        with self.assertRaisesRegex(F.FidGateError, "top_logprobs_non_finite"):
            F.validate_and_extract_response(mrow, r4, wall_s=0.1, max_tokens=8)

        r5 = make_mock_http_response(mrow, max_tokens=8)
        r5["choices"][0]["logprobs"]["content"][0]["top_logprobs"][2]["logprob"] = -0.01
        with self.assertRaisesRegex(F.FidGateError, "top_logprobs_not_sorted_descending"):
            F.validate_and_extract_response(mrow, r5, wall_s=0.1, max_tokens=8)

        r6 = make_mock_http_response(mrow, max_tokens=8)
        r6["choices"][0]["logprobs"]["content"][0]["top_logprobs"][1]["token"] = "tok_0"
        with self.assertRaisesRegex(
            F.FidGateError, "ambiguous_duplicate_token_string_in_top_logprobs"
        ):
            F.validate_and_extract_response(mrow, r6, wall_s=0.1, max_tokens=8)

        r7 = make_mock_http_response(mrow, max_tokens=8)
        r7["choices"][0]["logprobs"]["content"][0]["logprob"] = 0.1
        with self.assertRaisesRegex(F.FidGateError, "non_finite_or_positive_logprob"):
            F.validate_and_extract_response(mrow, r7, wall_s=0.1, max_tokens=8)

        r8 = make_mock_http_response(mrow, max_tokens=8)
        r8["choices"][0]["logprobs"]["content"][0]["top_logprobs"].pop()
        with self.assertRaisesRegex(F.FidGateError, "invalid_top_logprobs_len"):
            F.validate_and_extract_response(mrow, r8, wall_s=0.1, max_tokens=8)

        r9 = make_mock_http_response(mrow, max_tokens=8)
        r9["choices"][0]["logprobs"]["content"][0]["token"] = "other_tok"
        with self.assertRaisesRegex(F.FidGateError, "greedy_sampled_token_mismatch_top1"):
            F.validate_and_extract_response(mrow, r9, wall_s=0.1, max_tokens=8)

    def test_05_rejects_length_finish_reason_and_token_id_mismatches(self):
        mrow = self.manifest_rows[0]

        r1 = make_mock_http_response(mrow, max_tokens=6, finish_reason="length")
        with self.assertRaisesRegex(F.FidGateError, "finish_reason_length_but_short_output"):
            F.validate_and_extract_response(mrow, r1, wall_s=0.1, max_tokens=8)

        r2 = make_mock_http_response(mrow, max_tokens=6, finish_reason="stop")
        rec2 = F.validate_and_extract_response(mrow, r2, wall_s=0.1, max_tokens=8)
        self.assertEqual(rec2["completion_tokens"], 6)
        self.assertEqual(rec2["finish_reason"], "stop")

        r3 = make_mock_http_response(mrow, max_tokens=6, finish_reason="stop")
        r3["choices"][0]["flux_stop_reason"] = "watchdog"
        with self.assertRaisesRegex(F.FidGateError, "unexpected_flux_stop_reason"):
            F.validate_and_extract_response(mrow, r3, wall_s=0.1, max_tokens=8)

        r4 = make_mock_http_response(mrow, max_tokens=8, include_token_ids=False)
        with self.assertRaisesRegex(F.FidGateError, "missing_token_ids"):
            F.validate_and_extract_response(
                mrow, r4, wall_s=0.1, max_tokens=8, require_token_ids=True
            )

        r5 = make_mock_http_response(mrow, max_tokens=8)
        r5["usage"]["prompt_tokens"] += 1
        with self.assertRaisesRegex(F.FidGateError, "prompt_tokens_mismatch"):
            F.validate_and_extract_response(mrow, r5, wall_s=0.1, max_tokens=8)

        r6 = make_mock_http_response(mrow, max_tokens=8)
        r6["usage"]["completion_tokens"] = 7
        with self.assertRaisesRegex(F.FidGateError, "logprobs_len_mismatch"):
            F.validate_and_extract_response(mrow, r6, wall_s=0.1, max_tokens=8)

        r7 = make_mock_http_response(mrow, max_tokens=8, finish_reason="tool_calls")
        with self.assertRaisesRegex(F.FidGateError, "invalid_finish_reason"):
            F.validate_and_extract_response(mrow, r7, wall_s=0.1, max_tokens=8)

    def test_06_enforces_request_and_whole_panel_wall_caps(self):
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "out.jsonl"
            by_user = {r["user"]: r for r in self.manifest_rows}
            fake_time = [0.0]

            def clock() -> float:
                return fake_time[0]

            def slow_post(url: str, body: dict, timeout_s: float) -> dict:
                fake_time[0] += 60.0
                mrow = by_user[body["messages"][1]["content"]]
                return make_mock_http_response(mrow, max_tokens=8)

            with self.assertRaisesRegex(F.FidGateError, "total_wall_cap_exceeded"):
                F.collect_panel(
                    "http://mock/v1/chat/completions",
                    out_path,
                    max_tokens=8,
                    request_timeout_s=100.0,
                    total_wall_cap_s=150.0,
                    http_post=slow_post,
                    clock=clock,
                )

            req_out = Path(td) / "req_cap.jsonl"
            fake_req_time = [0.0]

            def req_clock() -> float:
                return fake_req_time[0]

            def slow_single_post(url: str, body: dict, timeout_s: float) -> dict:
                fake_req_time[0] += 120.0
                mrow = by_user[body["messages"][1]["content"]]
                return make_mock_http_response(mrow, max_tokens=8)

            with self.assertRaisesRegex(F.FidGateError, "request_wall_exceeded"):
                F.collect_panel(
                    "http://mock/v1/chat/completions",
                    req_out,
                    max_tokens=8,
                    request_timeout_s=100.0,
                    total_wall_cap_s=500.0,
                    http_post=slow_single_post,
                    clock=req_clock,
                )

    def test_07_compare_rejects_partial_duplicate_extra_and_ambiguous_panels(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            ref_path = td_path / "ref.jsonl"
            alt_path = td_path / "alt.jsonl"
            by_user = {r["user"]: r for r in self.manifest_rows}

            F.collect_panel(
                "http://mock/v1/chat/completions",
                ref_path,
                max_tokens=8,
                http_post=lambda u, b, t: make_mock_http_response(
                    by_user[b["messages"][1]["content"]], max_tokens=8
                ),
            )
            ref_lines = ref_path.read_text(encoding="utf-8").splitlines()

            # 1. Partial panel (11 of 12 rows) -> must fail closed!
            alt_path.write_text("\n".join(ref_lines[:-1]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(F.FidGateError, "missing_manifest_ids"):
                F.compare_panels(
                    ref_path, alt_path, max_tokens=8, require_provenance_files=False
                )
            self.assertEqual(
                F.main(
                    [
                        "compare",
                        str(ref_path),
                        str(alt_path),
                        "--max-tokens",
                        "8",
                        "--no-require-provenance-files",
                    ]
                ),
                2,
            )

            # 2. Duplicate ID -> must fail closed!
            alt_path.write_text(
                "\n".join(ref_lines[:-1] + [ref_lines[0]]) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(F.FidGateError, "duplicate_id"):
                F.compare_panels(
                    ref_path, alt_path, max_tokens=8, require_provenance_files=False
                )

            # 3. Extra unknown ID -> must fail closed!
            extra_row = json.loads(ref_lines[-1])
            extra_row["id"] = "LCB/unknown_task"
            alt_path.write_text(
                "\n".join(ref_lines[:-1] + [json.dumps(extra_row)]) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(F.FidGateError, "extra_unexpected_id"):
                F.compare_panels(
                    ref_path, alt_path, max_tokens=8, require_provenance_files=False
                )

            # 4. Ambiguous string collision: identical decoded string at pos 2, but different token_id!
            collision_row = json.loads(ref_lines[0])
            collision_row["token_ids"][2] += 9999
            alt_path.write_text(
                "\n".join([json.dumps(collision_row)] + ref_lines[1:]) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                F.FidGateError, "ambiguous_decoded_string_collision at pos 2"
            ):
                F.compare_panels(
                    ref_path, alt_path, max_tokens=8, require_provenance_files=False
                )

    def test_08_strict_saved_panel_schema_omission_truncation_and_provenance_binding(self):
        """Contract invariant 1: omitted fields, equal truncated panels, and provenance mismatches fail closed."""
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            ref_path = td_path / "ref.jsonl"
            alt_path = td_path / "alt.jsonl"
            by_user = {r["user"]: r for r in self.manifest_rows}

            F.collect_panel(
                "http://mock/v1/chat/completions",
                ref_path,
                max_tokens=8,
                http_post=lambda u, b, t: make_mock_http_response(
                    by_user[b["messages"][1]["content"]], max_tokens=8
                ),
            )
            ref_lines = ref_path.read_text(encoding="utf-8").splitlines()

            # (A) Omitted required fields must fail closed even if `--no-require-provenance-files` is used
            for required_field in (
                "schema_version",
                "index",
                "finish_reason",
                "completion_tokens",
                "user_sha256",
                "system_sha256",
                "request_sha256",
                "raw_response_sha256",
                "requested_model",
                "response_model",
                "response_id",
            ):
                mut_row = json.loads(ref_lines[0])
                del mut_row[required_field]
                alt_path.write_text(
                    "\n".join([json.dumps(mut_row)] + ref_lines[1:]) + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    F.FidGateError, f"missing_required_field: '{required_field}'"
                ):
                    F.compare_panels(
                        ref_path, alt_path, max_tokens=8, require_provenance_files=False
                    )

            # (B) Equal truncated artifacts (`finish_reason` omitted or `"length"` with 5 < 8 tokens on BOTH panels)
            # Previously, if `finish_reason` was omitted and both panels were truncated to 5 tokens,
            # `load_and_validate_panel_jsonl` allowed both and `compare_panels` reported 12/12 identical!
            trunc_a_path = td_path / "trunc_a.jsonl"
            trunc_b_path = td_path / "trunc_b.jsonl"
            trunc_lines = []
            for line in ref_lines:
                r = json.loads(line)
                r["finish_reason"] = None
                r["completion_tokens"] = 5
                r["tokens"] = r["tokens"][:5]
                r["token_ids"] = r["token_ids"][:5]
                r["logprob"] = r["logprob"][:5]
                r["top"] = r["top"][:5]
                trunc_lines.append(json.dumps(r))
            trunc_a_path.write_text("\n".join(trunc_lines) + "\n", encoding="utf-8")
            trunc_b_path.write_text("\n".join(trunc_lines) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(F.FidGateError, "invalid_or_missing_finish_reason"):
                F.compare_panels(
                    trunc_a_path, trunc_b_path, max_tokens=8, require_provenance_files=False
                )

            # Also if `finish_reason == "length"` on both truncated panels (5 < 8 tokens):
            trunc_len_lines = []
            for line in ref_lines:
                r = json.loads(line)
                r["finish_reason"] = "length"
                r["completion_tokens"] = 5
                r["tokens"] = r["tokens"][:5]
                r["token_ids"] = r["token_ids"][:5]
                r["logprob"] = r["logprob"][:5]
                r["top"] = r["top"][:5]
                trunc_len_lines.append(json.dumps(r))
            trunc_a_path.write_text("\n".join(trunc_len_lines) + "\n", encoding="utf-8")
            trunc_b_path.write_text("\n".join(trunc_len_lines) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                F.FidGateError, "finish_reason_length_but_tokens_len=5 != 8"
            ):
                F.compare_panels(
                    trunc_a_path, trunc_b_path, max_tokens=8, require_provenance_files=False
                )

            # (C) Provenance mismatch: missing `.meta.json` or `.raw.jsonl` fails closed by default
            alt_no_prov = td_path / "alt_no_prov.jsonl"
            alt_no_prov.write_text("\n".join(ref_lines) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(F.FidGateError, "missing_meta_provenance_file"):
                F.compare_panels(ref_path, alt_no_prov, max_tokens=8)

            # (D) Provenance mismatch: `.meta.json` status, out_sha256, raw_out_sha256, and tampered raw response
            tampered_out = td_path / "tampered.jsonl"
            F.collect_panel(
                "http://mock/v1/chat/completions",
                tampered_out,
                max_tokens=8,
                http_post=lambda u, b, t: make_mock_http_response(
                    by_user[b["messages"][1]["content"]], max_tokens=8
                ),
            )
            raw_p = tampered_out.with_suffix(".raw.jsonl")
            meta_p = tampered_out.with_suffix(".meta.json")
            orig_meta_text = meta_p.read_text(encoding="utf-8")

            bad_status_meta = json.loads(orig_meta_text)
            bad_status_meta["status"] = "failed"
            meta_p.write_text(json.dumps(bad_status_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(F.FidGateError, "meta_status_not_completed"):
                F.compare_panels(ref_path, tampered_out, max_tokens=8)

            bad_out_sha_meta = json.loads(orig_meta_text)
            bad_out_sha_meta["out_sha256"] = "0" * 64
            meta_p.write_text(json.dumps(bad_out_sha_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(F.FidGateError, "meta_out_sha256_mismatch"):
                F.compare_panels(ref_path, tampered_out, max_tokens=8)

            bad_raw_sha_meta = json.loads(orig_meta_text)
            bad_raw_sha_meta["raw_out_sha256"] = "0" * 64
            meta_p.write_text(json.dumps(bad_raw_sha_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(F.FidGateError, "meta_raw_out_sha256_mismatch"):
                F.compare_panels(ref_path, tampered_out, max_tokens=8)

            meta_p.write_text(orig_meta_text, encoding="utf-8")
            raw_rows = [json.loads(ln) for ln in raw_p.read_text(encoding="utf-8").splitlines()]
            # Tamper with the raw response content inside `.raw.jsonl` and recompute `meta.raw_out_sha256`
            raw_rows[0]["response"]["choices"][0]["message"]["content"] = "tampered_content"
            raw_p.write_text(
                "\n".join(json.dumps(r, sort_keys=True) for r in raw_rows) + "\n",
                encoding="utf-8",
            )
            meta_obj = json.loads(meta_p.read_text(encoding="utf-8"))
            meta_obj["raw_out_sha256"] = F.sha256_bytes(raw_p.read_bytes())
            meta_p.write_text(json.dumps(meta_obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                F.FidGateError, "raw_response_sha256_mismatch_with_panel"
            ):
                F.compare_panels(ref_path, tampered_out, max_tokens=8)

            # (E) Cross-panel provenance mismatch: `ref` collected with `seed=1234` vs `alt` collected with `seed=9999`
            alt_seed_out = td_path / "alt_seed.jsonl"
            F.collect_panel(
                "http://mock/v1/chat/completions",
                alt_seed_out,
                max_tokens=8,
                seed=9999,
                http_post=lambda u, b, t: make_mock_http_response(
                    by_user[b["messages"][1]["content"]], max_tokens=8
                ),
            )
            with self.assertRaisesRegex(F.FidGateError, "ref_vs_alt_sampling_config_mismatch"):
                F.compare_panels(ref_path, alt_seed_out, max_tokens=8)

            # (F) Cross-panel response_model mismatch under same requested alias "default"
            alt_resp_model_out = td_path / "alt_resp_model.jsonl"
            F.collect_panel(
                "http://mock/v1/chat/completions",
                alt_resp_model_out,
                max_tokens=8,
                http_post=lambda u, b, t: make_mock_http_response(
                    by_user[b["messages"][1]["content"]],
                    max_tokens=8,
                    model="Qwen3.8-Flash-Next-Other-Checkpoint",
                ),
            )
            with self.assertRaisesRegex(F.FidGateError, "ref_vs_alt_response_model_mismatch"):
                F.compare_panels(ref_path, alt_resp_model_out, max_tokens=8)

            # (G) Cross-panel requested_model and prompt_tokens mismatches
            alt_req_model_lines = []
            for mrow, ln in zip(self.manifest_rows, ref_lines):
                r = json.loads(ln)
                r["requested_model"] = "alternate-alias"
                r["request_sha256"] = F.canonical_json_sha256(
                    F.build_request_body(
                        mrow,
                        max_tokens=8,
                        top_logprobs_k=5,
                        seed=r["seed"],
                        return_token_ids=True,
                        model="alternate-alias",
                    )
                )
                alt_req_model_lines.append(json.dumps(r))
            alt_req_model_out = td_path / "alt_req_model.jsonl"
            alt_req_model_out.write_text("\n".join(alt_req_model_lines) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(F.FidGateError, "ref_vs_alt_requested_model_mismatch"):
                F.compare_panels(
                    ref_path, alt_req_model_out, max_tokens=8, require_provenance_files=False
                )

            alt_ptok_lines = []
            for idx, ln in enumerate(ref_lines):
                r = json.loads(ln)
                if idx == 0:
                    r["prompt_tokens"] += 1
                alt_ptok_lines.append(json.dumps(r))
            alt_ptok_path = td_path / "alt_ptok.jsonl"
            alt_ptok_path.write_text("\n".join(alt_ptok_lines) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(F.FidGateError, "ref_vs_alt_prompt_tokens_mismatch"):
                F.compare_panels(
                    ref_path,
                    alt_ptok_path,
                    max_tokens=8,
                    enforce_prompt_tokens=False,
                    require_provenance_files=False,
                )

    def test_09_incremental_persistence_retains_partial_and_failing_evidence(self):
        """Contract invariant 2: partial/failing evidence is persisted before/after each request."""
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            out_path = td_path / "partial.jsonl"
            raw_path = out_path.with_suffix(".raw.jsonl")
            meta_path = out_path.with_suffix(".meta.json")
            by_user = {r["user"]: r for r in self.manifest_rows}

            seen_in_progress_snapshots = []

            def failing_on_4th_post(url: str, body: dict, timeout_s: float) -> dict:
                # Verify that `.meta.json` was updated BEFORE this request started!
                meta_now = json.loads(meta_path.read_text(encoding="utf-8"))
                seen_in_progress_snapshots.append(meta_now)
                mrow = by_user[body["messages"][1]["content"]]
                if mrow["index"] == 3:
                    # Return a malformed payload (NaN logprob) on the 4th prompt (index 3)
                    bad_resp = make_mock_http_response(mrow, max_tokens=8)
                    bad_resp["choices"][0]["logprobs"]["content"][0]["logprob"] = float("nan")
                    return bad_resp
                return make_mock_http_response(mrow, max_tokens=8)

            with self.assertRaisesRegex(F.FidGateError, "non_finite_or_positive_logprob"):
                F.collect_panel(
                    "http://mock/v1/chat/completions",
                    out_path,
                    max_tokens=8,
                    http_post=failing_on_4th_post,
                )

            # Verify pre-request in-progress metadata updates occurred for indices 0, 1, 2, 3
            self.assertEqual(len(seen_in_progress_snapshots), 4)
            self.assertEqual(seen_in_progress_snapshots[3]["status"], "in_progress")
            self.assertEqual(seen_in_progress_snapshots[3]["completed_count"], 3)
            self.assertEqual(seen_in_progress_snapshots[3]["current_request_index"], 3)
            self.assertEqual(seen_in_progress_snapshots[3]["current_request_id"], "HumanEval/65")

            # 1. `out_path` must exist and retain the 3 validated rows from indices 0, 1, 2
            self.assertTrue(out_path.is_file())
            out_lines = [json.loads(ln) for ln in out_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(out_lines), 3)
            self.assertEqual([r["id"] for r in out_lines], ["LCB/abc392_d", "LCB/arc196_b", "LCB/abc400_d"])

            # 2. `raw_path` must exist and retain 4 rows: 3 `"ok"` rows + 1 `"validation_error"` row containing the exact failing response!
            self.assertTrue(raw_path.is_file())
            raw_lines = [json.loads(ln) for ln in raw_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(raw_lines), 4)
            self.assertEqual([r["status"] for r in raw_lines], ["ok", "ok", "ok", "validation_error"])
            self.assertEqual(raw_lines[3]["id"], "HumanEval/65")
            self.assertIsNotNone(raw_lines[3]["response"])
            self.assertIn("non_finite_or_positive_logprob", raw_lines[3]["error"])

            # 3. `meta_path` must record `status="failed"`, `completed_count=3`, `failed_at_index=3`, and exact partial file hashes
            self.assertTrue(meta_path.is_file())
            failed_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(failed_meta["status"], "failed")
            self.assertEqual(failed_meta["completed_count"], 3)
            self.assertEqual(failed_meta["failed_at_index"], 3)
            self.assertEqual(failed_meta["failed_at_id"], "HumanEval/65")
            self.assertEqual(failed_meta["out_sha256"], F.sha256_bytes(out_path.read_bytes()))
            self.assertEqual(failed_meta["raw_out_sha256"], F.sha256_bytes(raw_path.read_bytes()))

            # 4. Attempting to load or compare a partial failed panel must fail closed
            with self.assertRaisesRegex(F.FidGateError, "missing_manifest_ids"):
                F.load_and_validate_panel_jsonl(out_path, self.manifest_rows, max_tokens=8)
            full_path = td_path / "full_ref.jsonl"
            F.collect_panel(
                "http://mock/v1/chat/completions",
                full_path,
                max_tokens=8,
                http_post=lambda u, b, t: make_mock_http_response(
                    by_user[b["messages"][1]["content"]], max_tokens=8
                ),
            )
            with self.assertRaisesRegex(F.FidGateError, "missing_manifest_ids"):
                F.compare_panels(full_path, out_path, max_tokens=8)

    def test_10_real_clock_client_wall_deadline_and_uncertain_server_work(self):
        """Contract invariant 3: bounded client wall deadline on slow/hanging call and honest server-work metadata."""
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            out_path = td_path / "deadline.jsonl"
            raw_path = out_path.with_suffix(".raw.jsonl")
            meta_path = out_path.with_suffix(".meta.json")
            by_user = {r["user"]: r for r in self.manifest_rows}

            import threading

            release_stall = threading.Event()
            worker_done = threading.Event()

            def stalling_on_2nd_post(url: str, body: dict, timeout_s: float) -> dict:
                mrow = by_user[body["messages"][1]["content"]]
                if mrow["index"] == 1:
                    # Simulate a transport/socket call that ignores `timeout_s` and stalls
                    try:
                        release_stall.wait(0.50)
                    finally:
                        worker_done.set()
                return make_mock_http_response(mrow, max_tokens=8)

            t0 = time.monotonic()
            try:
                with self.assertRaisesRegex(F.FidGateError, "client_wall_deadline_exceeded"):
                    F.collect_panel(
                        "http://mock/v1/chat/completions",
                        out_path,
                        max_tokens=8,
                        request_timeout_s=0.08,
                        total_wall_cap_s=5.0,
                        http_post=stalling_on_2nd_post,
                        clock=time.monotonic,
                    )
                elapsed = time.monotonic() - t0
                # Client bounded wait must return promptly after the 0.08s deadline without waiting 0.50s
                self.assertLess(elapsed, 0.35)
            finally:
                release_stall.set()
                worker_done.wait(1.0)

            # Verify retained metadata records `deadline_exceeded=True`, `uncertain_server_work=True`,
            # and `server_cancellation_guaranteed=False`
            meta_obj = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(meta_obj["status"], "failed")
            self.assertEqual(meta_obj["completed_count"], 1)
            self.assertEqual(meta_obj["failed_at_index"], 1)
            self.assertTrue(meta_obj["deadline_exceeded"])
            self.assertTrue(meta_obj["uncertain_server_work"])
            self.assertFalse(meta_obj["server_cancellation_guaranteed"])

            raw_rows = [json.loads(ln) for ln in raw_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(raw_rows), 2)
            self.assertEqual(raw_rows[0]["status"], "ok")
            self.assertEqual(raw_rows[1]["status"], "transport_or_deadline_error")
            self.assertTrue(raw_rows[1]["deadline_exceeded"])
            self.assertTrue(raw_rows[1]["uncertain_server_work"])
            self.assertFalse(raw_rows[1]["server_cancellation_guaranteed"])

    def test_11_descriptive_default_and_valid_eos_length_difference(self):
        """Contract invariant 4: descriptive comparison by default, valid EOS length difference is not an integrity error."""
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            ref_path = td_path / "ref.jsonl"
            alt_path = td_path / "alt.jsonl"
            by_user = {r["user"]: r for r in self.manifest_rows}

            # `ref` generates 8 tokens (`finish_reason="length"`) on all 12 prompts
            F.collect_panel(
                "http://mock/v1/chat/completions",
                ref_path,
                max_tokens=8,
                http_post=lambda u, b, t: make_mock_http_response(
                    by_user[b["messages"][1]["content"]], max_tokens=8
                ),
            )

            # `alt` stops early with valid EOS (`finish_reason="stop"`, 5 tokens) on prompt 0,
            # and matches `ref` on prompts 1..11
            def mock_alt_eos(url: str, body: dict, timeout_s: float) -> dict:
                mrow = by_user[body["messages"][1]["content"]]
                if mrow["index"] == 0:
                    return make_mock_http_response(mrow, max_tokens=5, finish_reason="stop")
                return make_mock_http_response(mrow, max_tokens=8, finish_reason="length")

            F.collect_panel(
                "http://mock/v1/chat/completions",
                alt_path,
                max_tokens=8,
                http_post=mock_alt_eos,
            )

            # 1. Default comparison (`enforce_gates=False`) succeeds descriptively (exit code 0)
            summary = F.compare_panels(ref_path, alt_path, max_tokens=8)
            self.assertTrue(summary["evidence_valid"])
            self.assertEqual(summary["numerical_acceptance"], "not_evaluated")
            self.assertFalse(summary["enforce_gates"])
            self.assertIsNone(summary["gate_passed"])
            self.assertEqual(summary["continuations_identical"], 11)
            p0 = summary["per_prompt"][0]
            self.assertEqual(p0["ref_len"], 8)
            self.assertEqual(p0["alt_len"], 5)
            self.assertEqual(p0["ref_finish_reason"], "length")
            self.assertEqual(p0["alt_finish_reason"], "stop")
            self.assertEqual(p0["diverge_at"], 5)
            self.assertEqual(
                p0["diverge_reason"], "eos_length_difference(ref=8:length,alt=5:stop)"
            )
            self.assertAlmostEqual(p0["ref_margin_at_diverge"], 1.50, places=5)

            json_out = td_path / "compare_desc.json"
            rc_desc = F.main(
                [
                    "compare",
                    str(ref_path),
                    str(alt_path),
                    "--max-tokens",
                    "8",
                    "--json-out",
                    str(json_out),
                ]
            )
            self.assertEqual(rc_desc, 0)
            saved_desc = json.loads(json_out.read_text(encoding="utf-8"))
            self.assertTrue(saved_desc["evidence_valid"])
            self.assertEqual(saved_desc["numerical_acceptance"], "not_evaluated")
            self.assertIsNone(saved_desc["gate_passed"])

            # 2. When `--enforce-gates` is explicitly requested, `ref_margin_at_diverge = 1.50 > 0.65`
            # triggers `FidThresholdFailure` (exit code 3, with `evidence_valid=True` and `gate_passed=False`)
            with self.assertRaisesRegex(F.FidThresholdFailure, "eos_length_difference"):
                F.compare_panels(ref_path, alt_path, max_tokens=8, enforce_gates=True)

            json_gated = td_path / "compare_gated.json"
            rc_gated = F.main(
                [
                    "compare",
                    str(ref_path),
                    str(alt_path),
                    "--max-tokens",
                    "8",
                    "--enforce-gates",
                    "--json-out",
                    str(json_gated),
                ]
            )
            self.assertEqual(rc_gated, 3)
            saved_gated = json.loads(json_gated.read_text(encoding="utf-8"))
            self.assertTrue(saved_gated["evidence_valid"])
            self.assertEqual(saved_gated["numerical_acceptance"], "failed")
            self.assertFalse(saved_gated["gate_passed"])


if __name__ == "__main__":
    unittest.main()
