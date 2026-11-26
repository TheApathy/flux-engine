# SPDX-License-Identifier: AGPL-3.0-only

import importlib.machinery
import importlib.util
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "lil-bench"
LOADER = importlib.machinery.SourceFileLoader("flux_lil_bench_launcher", str(MODULE_PATH))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC
L = importlib.util.module_from_spec(SPEC)
LOADER.exec_module(L)


class LilBenchLauncherTests(unittest.TestCase):
    def run_main(self, argv):
        calls = []
        original_argv = sys.argv
        original_run = L.run_lil_cmd
        try:
            sys.argv = ["lil-bench", *argv]

            def capture(command, output, options):
                calls.append((command, output, options))
                return 0

            L.run_lil_cmd = capture
            rc = L.main()
        finally:
            L.run_lil_cmd = original_run
            sys.argv = original_argv
        return rc, calls

    def test_gsm8k_shortcut_uses_real_v079_profile_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "gsm8k.json"
            rc, calls = self.run_main(
                [
                    "accuracy", "gsm8k", "--concurrency", "1",
                    "--model", "qad5500", "--output", str(output),
                ]
            )
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 1)
        command = calls[0][0]
        self.assertEqual(command[command.index("--test-profile") + 1], "gsm8k")
        self.assertEqual(command[command.index("--profile-concurrency") + 1], "1")
        self.assertEqual(command[command.index("--profile-runs") + 1], "1319")
        self.assertEqual(command[command.index("--completion-stats-temperature") + 1], "0")
        self.assertEqual(command[command.index("--completion-stats-top-p") + 1], "1")
        self.assertEqual(command[command.index("--completion-stats-seed") + 1], "20261018")
        self.assertEqual(command[command.index("--model") + 1], "qad5500")
        self.assertNotIn("--accuracy-suite", command)
        self.assertNotIn("--accuracy-concurrency", command)

    def test_mmlu_pro_and_all_use_one_pinned_profile_per_result(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "accuracy-all.json"
            rc, calls = self.run_main(
                ["accuracy", "all", "--model", "qad5500", "--output", str(output)]
            )
        self.assertEqual(rc, 0)
        self.assertEqual(
            [call[0][call[0].index("--test-profile") + 1] for call in calls],
            ["gsm8k", "mmlu-pro", "gpqa-diamond"],
        )
        self.assertEqual(
            [call[1].name for call in calls],
            ["accuracy-gsm8k.json", "accuracy-mmlu-pro.json", "accuracy-gpqa-diamond.json"],
        )

    def test_direct_passthrough_preserves_explicit_model(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "mmlu.json"
            rc, calls = self.run_main(
                [
                    "--test-profile", "mmlu-pro", "--model", "qad5500",
                    "--output", str(output),
                ]
            )
        self.assertEqual(rc, 0)
        command = calls[0][0]
        self.assertEqual(command.count("--model"), 1)
        self.assertEqual(command[command.index("--model") + 1], "qad5500")

    def test_subprocess_update_check_is_disabled(self):
        observed = {}
        original = L.subprocess.run
        try:
            class Result:
                returncode = 0

            def fake_run(command, **kwargs):
                observed.update(kwargs)
                return Result()

            L.subprocess.run = fake_run
            with redirect_stdout(io.StringIO()):
                rc = L.run_lil_cmd(["bench"], None, {})
        finally:
            L.subprocess.run = original
        self.assertEqual(rc, 0)
        self.assertEqual(observed["env"]["LLM_BENCH_NO_UPDATE_CHECK"], "1")
        self.assertEqual(os.environ["LLM_BENCH_NO_UPDATE_CHECK"], "1")


class LilBenchCardSchemaTests(unittest.TestCase):
    """llm-inference-bench v0.7.x JSON must render with real numbers, not empty cells."""

    V079 = {
        "metadata": {"model": "Qwen3.8-Flash-Next-NVFP4-main", "version": "0.7.9", "duration_per_test": 30.0},
        "startup_diagnostics": {"args": {"max_tokens": 2048, "kv_budget": 131072},
                                "nvidia_smi_query": {"stdout": "0, NVIDIA GB10, 580.178.04, 0000000F:01:00.0"}},
        "prefill": {"8192": {"tok_per_sec": 5671.0, "ttft_seconds": 1.445, "prompt_tokens": 8192}},
        "results": [
            {"concurrency": 1, "context_tokens": 0, "aggregate_tps": 48.7, "ttft_avg": 0.35, "failure_reason": ""},
            {"concurrency": 4, "context_tokens": 8192, "aggregate_tps": -3.0, "ttft_avg": 0.0, "failure_reason": ""},
        ],
    }

    def test_v079_schema_is_normalized(self):
        import copy
        run = L._normalize_run(copy.deepcopy(self.V079))
        self.assertEqual(run["model"], "Qwen3.8-Flash-Next-NVFP4-main")
        self.assertEqual(run["prefill_Load_Sweep"][0]["prefill_tok_s"], 5671.0)
        ok, bad = run["results"]
        self.assertEqual((ok["context"], ok["throughput_tok_s"], ok["cell_status"]), (0, 48.7, "ok"))
        self.assertIsNone(bad["throughput_tok_s"])
        self.assertNotEqual(bad["cell_status"], "ok")
        self.assertEqual(run["hardware"]["driver_version"], "580.178.04")

    def test_card_renders_numbers_and_model(self):
        import copy, json as _json
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "rep.json"
            src.write_text(_json.dumps(copy.deepcopy(self.V079)))
            L.generate_cards(Path(tmp) / "card", [src], {"host": True}, title_override="")
            svg = (Path(tmp) / "card.svg").read_text()
        self.assertIn("Qwen3.8-Flash-Next-NVFP4-main", svg)
        self.assertIn("5,671.0", svg)
        self.assertIn("48.7", svg)
        self.assertNotIn("Qwen3.8-35B-A3B-NVFP4", svg)

    def test_no_hardcoded_unrelated_model_fallback(self):
        self.assertNotIn("Qwen3.8-35B-A3B-NVFP4", MODULE_PATH.read_text())

    def test_accuracy_max_tokens_fits_server_context(self):
        original = L._server_context_len
        try:
            L._server_context_len = lambda host, port: 65536
            self.assertEqual(L._accuracy_max_tokens("127.0.0.1", "8899"), 65536 - L.ACCURACY_PROMPT_HEADROOM)
            L._server_context_len = lambda host, port: None
            self.assertEqual(L._accuracy_max_tokens("127.0.0.1", "8899"), L.ACCURACY_MAX_TOKENS_CAP)
        finally:
            L._server_context_len = original


class LilBenchPromptModeTests(unittest.TestCase):
    """Distinct-prompt / prefill-warmup fields of the patched bench reach the card."""

    def run_concurrency(self, extra):
        calls = []
        original_argv, original_run, original_cards = sys.argv, L.run_lil_cmd, L.generate_cards
        try:
            with tempfile.TemporaryDirectory() as directory:
                sys.argv = ["lil-bench", "concurrency", "1,4", "--model", "m",
                            "--output", str(Path(directory) / "c.json"), *extra]
                L.run_lil_cmd = lambda command, output, options: calls.append(command) or 0
                L.generate_cards = lambda *a, **k: {}
                rc = L.main()
        finally:
            sys.argv, L.run_lil_cmd, L.generate_cards = original_argv, original_run, original_cards
        self.assertEqual(rc, 0)
        return calls[0]

    def test_concurrency_defaults_to_distinct_prompts_with_prefill_headroom(self):
        command = self.run_concurrency([])
        self.assertNotIn("--shared-decode-prompt", command)
        self.assertEqual(command[command.index("--cell-warmup-timeout-seconds") + 1], "120")

    def test_concurrency_passes_opt_out_flags_through(self):
        command = self.run_concurrency(["--shared-decode-prompt", "--no-prefill-context-warmup"])
        self.assertIn("--shared-decode-prompt", command)
        self.assertIn("--no-prefill-context-warmup", command)

    def test_unpatched_multi_stream_json_is_labelled_shared(self):
        import copy
        run = L._normalize_run(copy.deepcopy(LilBenchCardSchemaTests.V079))
        self.assertEqual(run["decode_prompt_mode"], "shared")
        single = copy.deepcopy(LilBenchCardSchemaTests.V079)
        single["results"] = single["results"][:1]
        self.assertIsNone(L._normalize_run(single)["decode_prompt_mode"])

    def test_card_shows_prompt_mode_and_cold_ttft(self):
        import copy, json as _json
        data = copy.deepcopy(LilBenchCardSchemaTests.V079)
        data["metadata"]["decode_prompt_mode"] = "distinct_per_stream"
        data["prefill"]["8192"]["cold_ttft_seconds"] = 2.875
        run = L._normalize_run(copy.deepcopy(data))
        self.assertEqual(run["decode_prompt_mode"], "distinct_per_stream")
        self.assertEqual(L._aggregate_prefill([run])[0]["mean_cold_ttft_s"], 2.875)
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "rep.json"
            src.write_text(_json.dumps(data))
            L.generate_cards(Path(tmp) / "card", [src], {"host": True}, title_override="")
            svg = (Path(tmp) / "card.svg").read_text()
            md = (Path(tmp) / "MODEL_CARD.md").read_text()
        self.assertIn("prompts=distinct_per_stream", svg)
        self.assertIn("| **Decode Prompts (C>1)** | `distinct_per_stream` |", md)
        self.assertIn("Cold TTFT (s)", md)
        self.assertIn("| 1.45 | 2.88 |", md)


if __name__ == "__main__":
    unittest.main()
