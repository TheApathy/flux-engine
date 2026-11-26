# SPDX-License-Identifier: AGPL-3.0-only
"""Offline recipe and operational-driver argument integration regressions."""

import hashlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

KIT = Path(__file__).resolve().parents[1]
# 295a6aaa1 binds the example plan to the real suite program digest.
SUITE_SHA256 = hashlib.sha256((KIT / "r15-container-suite.py").read_bytes()).hexdigest()


def load(path, name):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


SUITE = load(KIT / "r15-container-suite.py", "r16_recipe_suite")


class ReleaseRecipeWiringTests(unittest.TestCase):
    def test_native262_recipe_is_accepted_by_the_sealed_literal_parser(self):
        recipe = KIT / "recipes/flux-qwen4-native-262k-bf16.env"
        environment = SUITE._recipe_environment(recipe.read_bytes())
        expected = {
            "FLUX_MODE": "256k",
            "FLUX_MAX_SEQ_LEN": "262144",
            "FLUX_MAX_PREFILL_TOKENS": "16384",
            "FLUX_CONCURRENCY": "1",
            "FLUX_KV_CACHE_DTYPE": "bf16",
            "FLUX_SPEC_MODE": "none",
            "FLUX_CTX_OVERFLOW_TRUNCATE": "0",
            "ATLAS_CTX_OVERFLOW_TRUNCATE": "0",
            "FLUX_QWEN4_PREFILL_QSA_FP8": "0",
            "ATLAS_QWEN4_PREFILL_QSA_FP8": "0",
            "ATLAS_QWEN4_PREFILL_QSA": "1",
            "ATLAS_QWEN4_PREFILL_MULTI_CHUNK": "1",
            "ATLAS_QWEN4_PREFILL_WIDE": "16384",
            "ATLAS_QWEN4_QSA_VERIFY_ROWS": "0",
            "FLUX_QWEN4_FAST_PROJ_FP8_WEIGHT_CACHE_RESERVE_GB": "10",
            "ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_WEIGHT_CACHE_RESERVE_GB": "10",
        }
        for key, value in expected.items():
            self.assertEqual(environment[key.encode()], value.encode(), key)

    def test_recipe_does_not_advertise_inherited_performance_as_qualification(self):
        source = (KIT / "recipes/flux-qwen4-native-262k-bf16.env").read_text()
        self.assertNotIn("Certified Performance", source)
        self.assertNotIn("tok/s peak", source)
        self.assertNotIn("last assignment wins", source)
        self.assertIn("Source-candidate correctness recipe", source)

    def test_accuracy_example_argv_satisfies_real_driver_required_arguments(self):
        plan = json.loads(
            (KIT / "examples/r15-container-suite-plan.example.json").read_text()
        )
        replacements = {
            "@CONTAINER_ID@": "c" * 64,
            "@IMAGE_ID@": "sha256:" + "a" * 64,
            "@IMAGE_DIGEST@": "sha256:" + "b" * 64,
            "@BASE_URL@": "http://127.0.0.1:8899",
            "@RUN_ID@": "r16-example-parse-only",
            "@GATE_RECEIPT_FD@": "42",
            "@RAW_LOG_FD@": "43",
        }
        for gate in ("mmlu-pro", "gsm8k"):
            driver = load(KIT / "r15-gates" / gate, "r16_example_" + gate)
            argv = plan["gate_jobs"][gate]["argv"]
            arguments = [replacements.get(item, item) for item in argv[1:]]
            index = arguments.index("--suite-sha256") + 1
            self.assertEqual(arguments[index], SUITE_SHA256)
            arguments[index] = "1" * 64
            with (
                mock.patch.object(driver, "verify_parent_suite") as parent,
                mock.patch.object(driver, "load_module", return_value=SUITE),
            ):
                parsed = driver.parse_args(arguments)
            parent.assert_called_once_with("1" * 64)
            self.assertEqual(parsed.runtime_dir, plan["runtime_dir"])
            self.assertEqual(
                parsed.suite_program,
                plan["source_root_path"] + "/repro-kit/r15-container-suite.py",
            )
            self.assertEqual(
                parsed.lil_program,
                plan["source_root_path"]
                + "/repro-kit/llm-inference-bench/llm_decode_bench.py",
            )

    def test_native_and_memory_example_argv_satisfy_actual_driver_parsers(self):
        plan = json.loads(
            (KIT / "examples/r15-container-suite-plan.example.json").read_text()
        )
        for gate in ("native-262k", "memory-floor"):
            with self.subTest(gate=gate), tempfile.TemporaryDirectory() as directory:
                driver = load(KIT / "r15-gates" / gate, "r16_wiring_" + gate)
                helper = driver.load_checked(
                    KIT / "r15-gates/mmlu-pro",
                    driver.LIFECYCLE_SHA256,
                    "r16_wiring_lifecycle_" + gate,
                )
                raw_path = Path(directory) / "captured.log"
                raw_path.touch()
                raw_fd = os.open(raw_path, os.O_RDONLY | os.O_CLOEXEC)
                try:
                    with tempfile.TemporaryFile() as receipt:
                        os.set_inheritable(receipt.fileno(), True)
                        os.set_inheritable(raw_fd, True)
                        replacements = {
                            "@CONTAINER_ID@": "c" * 64,
                            "@IMAGE_ID@": "sha256:" + "a" * 64,
                            "@IMAGE_DIGEST@": "sha256:" + "b" * 64,
                            "@BASE_URL@": "http://127.0.0.1:8899",
                            "@RUN_ID@": "r16-wiring-parse-only",
                            "@GATE_RECEIPT_FD@": str(receipt.fileno()),
                            "@RAW_LOG_FD@": str(raw_fd),
                        }
                        arguments = [
                            replacements.get(item, item)
                            for item in plan["gate_jobs"][gate]["argv"][1:]
                        ]
                        index = arguments.index("--suite-sha256") + 1
                        self.assertEqual(arguments[index], SUITE_SHA256)
                        arguments[index] = "1" * 64
                        modules = [helper]
                        if gate == "native-262k":
                            modules.append(
                                driver.load_checked(
                                    KIT / "qwen4_endpoint_admission.py",
                                    driver.ENDPOINT_SHA256,
                                    "r16_wiring_endpoint",
                                )
                            )
                        # Real parsers execute; memory additionally validates
                        # actual inherited FDs. Parent authority is mocked.
                        with (
                            mock.patch.object(
                                driver, "load_checked", side_effect=modules
                            ) as checked,
                            mock.patch.object(helper, "verify_parent_suite") as parent,
                            mock.patch.object(
                                helper, "load_module", return_value=SUITE
                            ),
                        ):
                            parsed = driver.parse_args(arguments)
                        parent.assert_called_once_with("1" * 64)
                        self.assertEqual(checked.call_count, len(modules))
                        self.assertEqual(parsed.runtime_dir, plan["runtime_dir"])
                        self.assertEqual(
                            parsed.lifecycle_program,
                            plan["source_root_path"] + "/repro-kit/r15-gates/mmlu-pro",
                        )
                        self.assertEqual(parsed.raw_log_fd, raw_fd)
                        self.assertEqual(parsed.receipt_fd, receipt.fileno())
                        if gate == "native-262k":
                            self.assertEqual(
                                parsed.endpoint_program,
                                plan["source_root_path"]
                                + "/repro-kit/qwen4_endpoint_admission.py",
                            )
                finally:
                    os.close(raw_fd)


if __name__ == "__main__":
    unittest.main()
