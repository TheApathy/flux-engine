# SPDX-License-Identifier: AGPL-3.0-only

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "lil_accuracy_qualify.py"
EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "lil-accuracy-pair-envelope.example.json"
)
SPEC = importlib.util.spec_from_file_location("lil_accuracy_qualify", MODULE_PATH)
assert SPEC and SPEC.loader
Q = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(Q)

MODEL = "JMNI-Labs/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid"


def make_summary(count, correct):
    wrong = count - correct
    score_counts = {"exact": correct}
    if wrong:
        score_counts["fail"] = wrong
    rate = correct / count
    return {
        "attempted": count,
        "completed": count,
        "errors": 0,
        "correct": correct,
        "wrong": wrong,
        "score_counts": score_counts,
        "exact": correct,
        "near": 0,
        "fail": wrong,
        "truncated": 0,
        "stalled": 0,
        "timeout": 0,
        "unparseable": 0,
        "incomplete": 0,
        "correct_rate": rate,
        "correct_rate_completed": rate,
        "score_available": True,
        "hit_max_tokens": 0,
        "estimated_token_runs": 0,
    }


def make_report(profile, arm):
    dataset = Q.load_dataset_contract(profile)
    contract = Q.PROFILE_CONTRACTS[profile]
    count = contract["items"]
    runs = []
    category_correct = {}
    for index, (item_id, expected) in enumerate(dataset.items(), 1):
        answer = expected["expected_answer"]
        category = expected["category"]
        if category:
            category_correct[category] = category_correct.get(category, 0) + 1
        runs.append(
            {
                "run_index": index,
                "phase": "profile",
                "concurrency": 1,
                "ok": True,
                "correct": True,
                "completion_tokens": 32 + index % 17,
                "prompt_tokens": 64 + index % 13,
                "finish_reason": "stop",
                "final_answer": answer,
                "score_label": "exact",
                "score_detail": "exact",
                "parsed_answer": answer,
                "output_excerpt": answer,
                "output_text": "",
                "reasoning_text": "",
                "content_text": "",
                "error": "",
                "hit_max_tokens": False,
                "estimated_tokens": False,
                "cancelled": False,
                "item_id": item_id,
                "category": category,
                "expected_answer": answer,
                "interventions": {
                    "schema": Q.INTERVENTION_LEDGER_SCHEMA,
                    "policy": Q.INTERVENTION_LEDGER_POLICY,
                    "events": [],
                },
            }
        )
    summary = make_summary(count, count)
    categories = (
        [
            {
                "category": category,
                "scored": scored,
                "correct": category_correct[category],
                "accuracy": 1.0,
            }
            for category, scored in Q.MMLU_CATEGORY_COUNTS.items()
        ]
        if profile == "mmlu-pro"
        else []
    )
    return {
        "metadata": {
            "version": Q.LIL_VERSION,
            "mode": "completion_stats",
            "interrupted": False,
            "server": "http://127.0.0.1:8899",
            "model": MODEL,
            "models": [MODEL],
            "test_profile": profile,
            "timestamp": f"2026-10-18T00:00:00+00:00/{arm}",
            "max_tokens": Q.TOKEN_BUDGET,
            "token_limit_field": "max_tokens",
            "max_tokens_omitted": False,
            "fixed_concurrency": 1,
            "requested_runs": count,
            "concurrency_levels_requested": [1],
            "min_results": count,
            "score_source": "final_answer",
            "correct_regex": "",
            "profile_scorer": contract["scorer"],
            "scorer_version": Q.SCORER_VERSION,
            "profile_version": Q.PROFILE_VERSION,
            "dataset": contract["dataset"],
            "dataset_sha256": contract["dataset_sha256"],
            "dataset_items_total": count,
            "dataset_items_selected": count,
            "prefill_scout": False,
            "temperature": 0.0,
            "top_p": None,
            "seed_base": None,
            "reasoning_effort": None,
            "request_overrides": {"report_interventions": True},
            "system_prompt": False,
        },
        "prefill_scout": None,
        "selected_concurrency": 1,
        "selected_summary": copy.deepcopy(summary),
        "all_summary": copy.deepcopy(summary),
        "level_summaries": [
            {
                "concurrency": 1,
                "phase": "profile",
                "summary": copy.deepcopy(summary),
                "selected": True,
                "improved": True,
            }
        ],
        "accuracy": {
            "items_total": count,
            "items_selected": count,
            "attempted": count,
            "scored": count,
            "correct": count,
            "accuracy": 1.0,
            "wilson95_low": 0.99,
            "wilson95_high": 1.0,
            "unparseable": 0,
            "truncated_no_answer": 0,
            "hit_max_tokens": 0,
            "errors": 0,
            "dataset_exhausted": False,
        },
        "category_summaries": categories,
        "wrong_runs": [],
        "runs": runs,
    }


def with_gsm_failures(report, failures):
    value = copy.deepcopy(report)
    for run in value["runs"][:failures]:
        run["correct"] = False
        run["score_label"] = "fail"
        run["score_detail"] = "expected reference, got wrong"
        run["parsed_answer"] = "wrong"
        run["final_answer"] = "wrong"
    count = len(value["runs"])
    correct = count - failures
    summary = make_summary(count, correct)
    value["selected_summary"] = copy.deepcopy(summary)
    value["all_summary"] = copy.deepcopy(summary)
    value["level_summaries"][0]["summary"] = copy.deepcopy(summary)
    value["accuracy"]["correct"] = correct
    value["accuracy"]["accuracy"] = correct / count
    value["wrong_runs"] = copy.deepcopy(value["runs"][:failures])
    return value


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


class LilAccuracyContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gsm_no_spec = make_report("gsm8k", "no-spec")
        cls.gsm_k5 = make_report("gsm8k", "k5")
        cls.mmlu_no_spec = make_report("mmlu-pro", "no-spec")
        cls.mmlu_k5 = make_report("mmlu-pro", "k5")

    def test_json_consumers_reject_duplicate_keys_and_nonfinite_numbers(self):
        for raw, expected in (
            ('{"schema":1,"schema":2}', "duplicate key 'schema'"),
            ('{"accuracy":NaN}', "non-finite number 'NaN'"),
            ('{"accuracy":Infinity}', "non-finite number 'Infinity'"),
            ('{"accuracy":-Infinity}', "non-finite number '-Infinity'"),
            ('{"accuracy":1e9999}', "non-finite number"),
            ('{"accuracy":-1e9999}', "non-finite number"),
        ):
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(ValueError, expected):
                    Q.strict_json_loads(raw, "qualification artifact")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "duplicate.json"
            path.write_text('{"runs":[],"runs":[]}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate key 'runs'"):
                Q._json_file(path, "qualification artifact")

    def test_full_reports_and_authoritative_matched_pair_pass(self):
        for profile, no_spec_report, k5_report, expected in (
            ("gsm8k", self.gsm_no_spec, self.gsm_k5, 1319),
            ("mmlu-pro", self.mmlu_no_spec, self.mmlu_k5, 1000),
        ):
            no_spec = Q.validate_report(no_spec_report, profile, MODEL)
            k5 = Q.validate_report(k5_report, profile, MODEL)
            parity = Q.validate_pair(no_spec, k5)
            self.assertEqual(parity["matched_items"], expected)
            self.assertEqual(parity["mismatches"], 0)
            self.assertIn("authoritative", parity["authority"])

    def test_rejects_partial_interrupted_or_runtime_artifact_runs(self):
        mutations = []
        partial = copy.deepcopy(self.gsm_no_spec)
        partial["runs"].pop()
        mutations.append(partial)
        interrupted = copy.deepcopy(self.gsm_no_spec)
        interrupted["metadata"]["interrupted"] = True
        mutations.append(interrupted)
        estimated = copy.deepcopy(self.gsm_no_spec)
        estimated["runs"][0]["estimated_tokens"] = True
        mutations.append(estimated)
        cancelled = copy.deepcopy(self.gsm_no_spec)
        cancelled["runs"][0]["cancelled"] = True
        mutations.append(cancelled)
        truncated = copy.deepcopy(self.gsm_no_spec)
        truncated["runs"][0]["finish_reason"] = "length"
        truncated["runs"][0]["hit_max_tokens"] = True
        mutations.append(truncated)
        errored = copy.deepcopy(self.gsm_no_spec)
        errored["runs"][0]["error"] = "connection reset"
        mutations.append(errored)
        zero_usage = copy.deepcopy(self.gsm_no_spec)
        zero_usage["runs"][0]["completion_tokens"] = 0
        mutations.append(zero_usage)
        duplicate_item = copy.deepcopy(self.gsm_no_spec)
        duplicate_item["runs"][1]["item_id"] = duplicate_item["runs"][0]["item_id"]
        mutations.append(duplicate_item)
        for report in mutations:
            with self.subTest(kind=report["runs"][0].get("error") or "mutation"):
                with self.assertRaises(ValueError):
                    Q.validate_report(report, "gsm8k", MODEL)

    def test_exact_lil_profile_sampling_model_c1_and_budget_are_required(self):
        mutations = {
            "version": "0.7.8",
            "profile_scorer": "regex",
            "temperature": 0,
            "model": "lookalike-model",
            "fixed_concurrency": 2,
            "max_tokens": Q.TOKEN_BUDGET - 1,
            "top_p": 1.0,
            "seed_base": 0,
        }
        for key, value in mutations.items():
            report = copy.deepcopy(self.gsm_no_spec)
            report["metadata"][key] = value
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    Q.validate_report(report, "gsm8k", MODEL)

    def test_intervention_ledger_is_requested_and_must_be_terminal_empty(self):
        cases = []
        missing_opt_in = copy.deepcopy(self.gsm_no_spec)
        missing_opt_in["metadata"]["request_overrides"] = {}
        cases.append(("request_overrides", missing_opt_in))

        missing = copy.deepcopy(self.gsm_no_spec)
        del missing["runs"][0]["interventions"]
        cases.append(("missing keys", missing))

        wrong_schema = copy.deepcopy(self.gsm_no_spec)
        wrong_schema["runs"][0]["interventions"]["schema"] = "unknown"
        cases.append(("schema", wrong_schema))

        wrong_policy = copy.deepcopy(self.gsm_no_spec)
        wrong_policy["runs"][0]["interventions"]["policy"] = "disabled"
        cases.append(("policy", wrong_policy))

        intervened = copy.deepcopy(self.gsm_no_spec)
        intervened["runs"][0]["interventions"]["events"] = [
            {"code": "F32", "action": "duplicate_failed_tool_result", "count": 1}
        ]
        cases.append(("events must be empty", intervened))

        for error, report in cases:
            with self.subTest(error=error):
                with self.assertRaisesRegex(ValueError, error):
                    Q.validate_report(report, "gsm8k", MODEL)

    def test_mmlu_requires_every_pinned_category_and_all_1000_items(self):
        missing_category = copy.deepcopy(self.mmlu_no_spec)
        missing_category["category_summaries"].pop()
        with self.assertRaisesRegex(ValueError, "14 category"):
            Q.validate_report(missing_category, "mmlu-pro", MODEL)

        wrong_category = copy.deepcopy(self.mmlu_no_spec)
        wrong_category["runs"][0]["category"] = "other"
        with self.assertRaisesRegex(ValueError, "category differs"):
            Q.validate_report(wrong_category, "mmlu-pro", MODEL)

    def test_pair_rejects_answer_correctness_or_token_drift(self):
        no_spec = Q.validate_report(self.gsm_no_spec, "gsm8k", MODEL)
        for field, value in (
            ("final_answer", "different"),
            ("parsed_answer", "different"),
            ("correct", False),
            ("completion_tokens", 999),
        ):
            k5 = Q.validate_report(self.gsm_k5, "gsm8k", MODEL)
            first_id = sorted(k5["items"])[0]
            k5["items"][first_id][field] = value
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "matched parity failed"):
                    Q.validate_pair(no_spec, k5)

    def test_accuracy_floor_is_enforced_but_reported_as_provisional(self):
        at_floor = with_gsm_failures(self.gsm_no_spec, 65)
        normalized = Q.validate_report(at_floor, "gsm8k", MODEL)
        self.assertTrue(normalized["provisional_floor_pass"])
        self.assertEqual(normalized["provisional_floor"], 0.95)

        below_floor = with_gsm_failures(self.gsm_no_spec, 66)
        with self.assertRaisesRegex(ValueError, "provisional sanity floor failed"):
            Q.validate_report(below_floor, "gsm8k", MODEL)

    def test_external_envelope_binds_artifacts_recipes_argv_and_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "flux-server"
            binary.write_bytes(b"sealed test binary\n")
            binary.chmod(0o755)
            source_inventory = root / "source.json"
            checkpoint_inventory = root / "checkpoint.json"
            write_json(source_inventory, {"revision": "c" * 40})
            write_json(checkpoint_inventory, {"checkpoint": MODEL, "files": []})
            no_recipe = root / "no-spec.env"
            k5_recipe = root / "k5.env"
            no_recipe.write_text(
                "ATLAS_QWEN4_K5_HYBRID=1\nFLUX_SPEC_MODE=none\n",
                encoding="utf-8",
            )
            k5_recipe.write_text("ATLAS_QWEN4_K5_HYBRID=1\n", encoding="utf-8")

            reports = {
                ("no_spec", "gsm8k"): self.gsm_no_spec,
                ("no_spec", "mmlu-pro"): self.mmlu_no_spec,
                ("k5", "gsm8k"): self.gsm_k5,
                ("k5", "mmlu-pro"): self.mmlu_k5,
            }
            refs = {"no_spec": {}, "k5": {}}
            for (arm, profile), report in reports.items():
                path = root / f"{arm}-{profile}.json"
                write_json(path, report)
                refs[arm][profile] = {"path": path.name, "sha256": Q.sha256_file(path)}

            common_argv = [
                str(binary),
                "serve",
                "--max-batch-size",
                "1",
                "--max-num-seqs",
                "1",
                "--model",
                MODEL,
            ]
            k5_argv = common_argv + [
                "--speculative",
                "--num-drafts",
                "4",
                "--mtp-vocab",
                "0",
                "--mtp-quantization",
                "bf16",
            ]
            envelope = {
                "schema": Q.ENVELOPE_SCHEMA,
                "model": MODEL,
                "artifacts": {
                    "image_digest": "sha256:" + "a" * 64,
                    "image_id": "sha256:" + "b" * 64,
                    "binary_path": binary.name,
                    "binary_sha256": Q.sha256_file(binary),
                    "source_revision": "c" * 40,
                    "source_inventory_path": source_inventory.name,
                    "source_inventory_sha256": Q.sha256_file(source_inventory),
                    "checkpoint_id": MODEL,
                    "checkpoint_revision": "d" * 40,
                    "checkpoint_inventory_path": checkpoint_inventory.name,
                    "checkpoint_inventory_sha256": Q.sha256_file(checkpoint_inventory),
                },
                "arms": {
                    "no_spec": {
                        "mode": "none",
                        "recipe_path": no_recipe.name,
                        "recipe_sha256": Q.sha256_file(no_recipe),
                        "argv": common_argv,
                        "argv_sha256": Q.sha256_bytes(Q.canonical_bytes(common_argv)),
                        "reports": refs["no_spec"],
                    },
                    "k5": {
                        "mode": "native-mtp-k5",
                        "recipe_path": k5_recipe.name,
                        "recipe_sha256": Q.sha256_file(k5_recipe),
                        "argv": k5_argv,
                        "argv_sha256": Q.sha256_bytes(Q.canonical_bytes(k5_argv)),
                        "reports": refs["k5"],
                    },
                },
            }
            envelope_path = root / "envelope.json"
            write_json(envelope_path, envelope)
            _loaded, receipt = Q.load_and_validate_envelope(envelope_path)
            self.assertEqual(receipt["status"], "PASS")
            self.assertEqual(receipt["gate_policy"]["matched_parity"], "authoritative")
            self.assertIn("provisional", receipt["gate_policy"]["accuracy_floors"])

            bad = copy.deepcopy(envelope)
            bad["arms"]["k5"]["argv_sha256"] = "0" * 64
            bad_path = root / "bad-envelope.json"
            write_json(bad_path, bad)
            with self.assertRaisesRegex(ValueError, "argv_sha256 mismatch"):
                Q.load_and_validate_envelope(bad_path)

    def test_receipt_is_exclusive_and_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt_path = Path(directory) / "receipt.json"
            Q.seal_receipt(receipt_path, {"schema": Q.RECEIPT_SCHEMA, "status": "PASS"})
            self.assertEqual(receipt_path.stat().st_mode & 0o777, 0o444)
            with self.assertRaises(FileExistsError):
                Q.seal_receipt(receipt_path, {"status": "replacement"})

    def test_documented_envelope_example_is_structurally_matched(self):
        example = json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(example["schema"], Q.ENVELOPE_SCHEMA)
        self.assertEqual(set(example["arms"]), set(Q.ARM_MODES))
        no_spec = example["arms"]["no_spec"]["argv"]
        k5 = example["arms"]["k5"]["argv"]
        self.assertEqual(
            Q.normalized_non_spec_argv(no_spec), Q.normalized_non_spec_argv(k5)
        )
        for arm in Q.ARM_MODES:
            self.assertEqual(
                set(example["arms"][arm]["reports"]),
                set(Q.PROFILE_CONTRACTS),
            )


if __name__ == "__main__":
    unittest.main()
