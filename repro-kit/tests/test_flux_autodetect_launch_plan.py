# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only regression tests for the container launch-plan wrapper."""

import importlib.machinery
import io
import importlib.util
import json
import os
import re
import tempfile
import tomllib
from pathlib import Path
from unittest import TestCase, mock

SCRIPT = Path(__file__).resolve().parents[1] / "flux-autodetect"


def _load_flux_autodetect():
    loader = importlib.machinery.SourceFileLoader(
        "flux_autodetect_under_test", str(SCRIPT)
    )
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:
        raise RuntimeError(f"could not import {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


flux_autodetect = _load_flux_autodetect()


def _qsa_model_info():
    return {
        "kernel_target": "qwen3.8-flash-next",
        "selected_binary": "/opt/flux/bin/flux-server-real",
        "path": "/models/qwen38-flash-next",
        "name": "qwen38-flash-next",
        "ple_dir": None,
        "speculative": {"mode": "none"},
        "requires_qwen4_qsa": True,
        "recommended_max_seq_len": 262144,
        "max_position_embeddings": 262144,
        "recommended_tool_call_parser": "qwen3_coder",
    }


def _hardware_info():
    return {
        "hardware_recipe": "flux-nextgen-max-prefill-decode.env",
        "unified_memory_architecture": True,
    }


def _arg_value(command, flag):
    return command[command.index(flag) + 1]


class QsaNativeContextLaunchPlanTests(TestCase):
    def _build(self, env=None, mode="256k", recipe_env=None):
        with (
            mock.patch.dict(os.environ, env or {}, clear=True),
            mock.patch.object(
                flux_autodetect,
                "detect_hardware_topology",
                return_value=_hardware_info(),
            ),
            mock.patch.object(
                flux_autodetect,
                "_load_recipe_env",
                return_value=recipe_env or {},
            ),
        ):
            return flux_autodetect.build_launch_plan(_qsa_model_info(), mode, [])

    def test_qsa_native_262k_defaults_to_bf16_without_yarn(self):
        plan = self._build(
            recipe_env={
                "FLUX_QWEN4_PREFILL_QSA_FP8": "qkpv",
                "ATLAS_QWEN4_PREFILL_QSA_FP8": "qkpv",
            }
        )

        self.assertEqual(plan["max_seq_len"], 262144)
        self.assertEqual(plan["kv_cache_dtype"], "bf16")
        self.assertEqual(_arg_value(plan["command"], "--kv-cache-dtype"), "bf16")
        self.assertIsNone(plan["rope_yarn_factor"])
        self.assertNotIn("--rope-yarn-factor", plan["command"])
        self.assertNotIn("--kv-high-precision-layers", plan["command"])
        self.assertEqual(plan["env"]["FLUX_QWEN4_PREFILL_QSA_FP8"], "0")
        self.assertEqual(plan["env"]["ATLAS_QWEN4_PREFILL_QSA_FP8"], "0")
        self.assertEqual(plan["env"]["ATLAS_QWEN4_QSA_VERIFY_ROWS"], "0")
        self.assertEqual(plan["env"]["ATLAS_CTX_OVERFLOW_TRUNCATE"], "0")
        self.assertGreaterEqual(
            int(
                plan["env"]["ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_WEIGHT_CACHE_RESERVE_GB"]
            ),
            10,
        )
        self.assertEqual(plan["tool_call_parser"], "qwen3_coder")
        self.assertEqual(
            _arg_value(plan["command"], "--tool-call-parser"),
            "qwen3_coder",
        )

    def test_qsa_explicit_hermes_override_is_preserved(self):
        plan = self._build({"FLUX_TOOL_CALL_PARSER": "hermes"})

        self.assertEqual(plan["tool_call_parser"], "hermes")
        self.assertEqual(_arg_value(plan["command"], "--tool-call-parser"), "hermes")

    def test_flash_next_reports_model_profile_thinking_budget_without_overriding_it(
        self,
    ):
        plan = self._build()

        self.assertEqual(plan["thinking_budget"], 262144)
        self.assertEqual(plan["thinking_budget_source"], "model_profile")
        self.assertNotIn("--max-thinking-budget", plan["command"])

    def test_flash_next_passes_its_default_max_tokens(self):
        plan = self._build()

        self.assertEqual(_arg_value(plan["command"], "--default-max-tokens"), "262144")
        self.assertEqual(plan["default_max_tokens"], 262144)
        self.assertEqual(plan["default_max_tokens_source"], "model_profile")

    def test_default_max_tokens_environment_override(self):
        plan = self._build({"FLUX_DEFAULT_MAX_TOKENS": "65536"})
        self.assertEqual(_arg_value(plan["command"], "--default-max-tokens"), "65536")
        self.assertEqual(plan["default_max_tokens_source"], "environment")

        plan = self._build({"FLUX_DEFAULT_MAX_TOKENS": "0"})
        self.assertNotIn("--default-max-tokens", plan["command"])
        self.assertIsNone(plan["default_max_tokens"])

        with self.assertRaises(ValueError):
            self._build({"FLUX_DEFAULT_MAX_TOKENS": "lots"})

    def test_model_profile_default_max_tokens_table_is_exact(self):
        self.assertEqual(
            flux_autodetect.MODEL_PROFILE_DEFAULT_MAX_TOKENS,
            {
                "qwen3.8-flash-next": 262144,
                "qwen3.8-27b": 262144,
                "glm5.3-flash": 262144,
                "deepseek-v4.1": 262144,
                "gemma-4-31b": 262144,
                "gemma-4-26b-a4b": 262144,
            },
        )
        self.assertNotIn("qwen3.6-27b", flux_autodetect.MODEL_PROFILE_DEFAULT_MAX_TOKENS)

    def test_reported_flash_next_budget_matches_model_profile(self):
        profile_path = (
            SCRIPT.parents[1] / "kernels" / "gb10" / "qwen3.8-flash-next" / "MODEL.toml"
        )
        profile = tomllib.loads(profile_path.read_text(encoding="utf-8"))

        self.assertEqual(
            flux_autodetect.MODEL_PROFILE_THINKING_BUDGETS["qwen3.8-flash-next"],
            profile["behavior"]["max_thinking_budget"],
        )

    def test_explicit_thinking_budget_overrides_model_profile_and_is_reported(self):
        plan = self._build({"FLUX_THINKING_BUDGET": "1024"})

        self.assertEqual(plan["thinking_budget"], 1024)
        self.assertEqual(plan["thinking_budget_source"], "environment")
        self.assertEqual(_arg_value(plan["command"], "--max-thinking-budget"), "1024")

    def test_cli_thinking_budget_has_highest_launcher_precedence(self):
        with (
            mock.patch.dict(
                os.environ,
                {"FLUX_THINKING_BUDGET": "1024"},
                clear=True,
            ),
            mock.patch.object(
                flux_autodetect,
                "detect_hardware_topology",
                return_value=_hardware_info(),
            ),
            mock.patch.object(flux_autodetect, "_load_recipe_env", return_value={}),
        ):
            plan = flux_autodetect.build_launch_plan(
                _qsa_model_info(),
                "interactive",
                ["--max-thinking-budget", "512"],
            )

        self.assertEqual(plan["thinking_budget"], 512)
        self.assertEqual(plan["thinking_budget_source"], "cli")
        self.assertEqual(_arg_value(plan["command"], "--max-thinking-budget"), "512")

    def test_native_262k_qsa_precision_override_is_scoped_to_256k_mode(self):
        qsa_fp8_recipe = {
            "FLUX_QWEN4_PREFILL_QSA_FP8": "qkpv",
            "ATLAS_QWEN4_PREFILL_QSA_FP8": "qkpv",
        }

        plan = self._build(mode="max-prefill", recipe_env=qsa_fp8_recipe)

        self.assertEqual(plan["max_seq_len"], 262144)
        self.assertEqual(plan["kv_cache_dtype"], "bf16")
        self.assertEqual(plan["env"]["FLUX_QWEN4_PREFILL_QSA_FP8"], "qkpv")
        self.assertEqual(plan["env"]["ATLAS_QWEN4_PREFILL_QSA_FP8"], "qkpv")

    def test_container_default_seq_len_is_a_sentinel_for_native_256k(self):
        plan = self._build({"FLUX_MAX_SEQ_LEN": "131072"})

        self.assertEqual(plan["max_seq_len"], 262144)
        self.assertEqual(_arg_value(plan["command"], "--max-seq-len"), "262144")

    def test_long_qsa_keeps_native_mtp_and_enables_verify_rows(self):
        model_info = _qsa_model_info()
        model_info["speculative"] = {
            "mode": "native-mtp",
            "drafter_path": "/opt/flux/models/Qwen3.8-Flash-Next-Inferact-MTP",
            "gamma": 4,
            "num_drafts": 4,
            "native_mtp_k": 5,
            "mtp_layers": 1,
        }
        with (
            mock.patch.dict(os.environ, {"FLUX_MAX_SEQ_LEN": "8192"}, clear=True),
            mock.patch.object(
                flux_autodetect,
                "detect_hardware_topology",
                return_value=_hardware_info(),
            ),
            mock.patch.object(flux_autodetect, "_load_recipe_env", return_value={}),
        ):
            plan = flux_autodetect.build_launch_plan(model_info, "interactive", [])

        self.assertTrue(plan["speculative_enabled"])
        self.assertIn("--speculative", plan["command"])
        self.assertEqual(_arg_value(plan["command"], "--num-drafts"), "4")
        self.assertEqual(_arg_value(plan["command"], "--mtp-vocab"), "96000")
        self.assertEqual(
            _arg_value(plan["command"], "--mtp-from-path"),
            "/opt/flux/models/Qwen3.8-Flash-Next-Inferact-MTP",
        )
        self.assertEqual(plan["env"]["ATLAS_QWEN4_QSA_VERIFY_ROWS"], "1")
        self.assertEqual(plan["env"]["FLUX_QWEN4_QSA_VERIFY_ROWS"], "1")

        with (
            mock.patch.dict(
                os.environ, {"FLUX_CONCURRENCY": "8"}, clear=True
            ),
            mock.patch.object(
                flux_autodetect,
                "detect_hardware_topology",
                return_value=_hardware_info(),
            ),
            mock.patch.object(flux_autodetect, "_load_recipe_env", return_value={}),
        ):
            conc_plan = flux_autodetect.build_launch_plan(
                model_info, "max-prefill", []
            )

        self.assertTrue(conc_plan["speculative_enabled"])
        self.assertIn("--speculative", conc_plan["command"])
        self.assertEqual(conc_plan["max_seq_len"], 262144)
        self.assertEqual(_arg_value(conc_plan["command"], "--kv-pool-tokens"), "393216")
        self.assertEqual(conc_plan["max_prefill_tokens"], 8192)
        self.assertEqual(conc_plan["env"]["ATLAS_QWEN4_BATCHED_K5"], "1")
        self.assertEqual(conc_plan["env"]["ATLAS_QWEN4_BATCHED_K5_GDN_MULTI"], "1")
        self.assertEqual(conc_plan["env"]["ATLAS_SCHED_BURST_GATHER_MS"], "3")
        self.assertNotIn("ATLAS_MOE_T_LANES_WORKLIST_R8", conc_plan["env"])
        self.assertEqual(conc_plan["env"]["ATLAS_QWEN4_MTP_CONF_CUM"], "0.4")

    def test_mtp_source_selector_reaches_the_server_env(self):
        model_info = _qsa_model_info()
        model_info["speculative"] = {
            "mode": "native-mtp",
            "drafter_path": "/opt/flux/models/Qwen3.8-Flash-Next-Inferact-MTP",
            "gamma": 4,
            "num_drafts": 4,
            "native_mtp_k": 5,
            "mtp_layers": 1,
        }
        for value in ("sidecar", "checkpoint"):
            with (
                mock.patch.dict(
                    os.environ,
                    {"FLUX_MAX_SEQ_LEN": "8192", "FLUX_MTP_SOURCE": value},
                    clear=True,
                ),
                mock.patch.object(
                    flux_autodetect,
                    "detect_hardware_topology",
                    return_value=_hardware_info(),
                ),
                mock.patch.object(flux_autodetect, "_load_recipe_env", return_value={}),
            ):
                plan = flux_autodetect.build_launch_plan(model_info, "interactive", [])

            self.assertEqual(plan["env"]["ATLAS_MTP_SOURCE"], value)
            self.assertEqual(
                _arg_value(plan["command"], "--mtp-from-path"),
                "/opt/flux/models/Qwen3.8-Flash-Next-Inferact-MTP",
            )

    def test_explicit_nvfp4_remains_an_opt_in_at_native_262k(self):
        plan = self._build({"FLUX_KV_CACHE_DTYPE": "nvfp4"})

        self.assertEqual(plan["kv_cache_dtype"], "nvfp4")
        self.assertEqual(_arg_value(plan["command"], "--kv-cache-dtype"), "nvfp4")
        self.assertEqual(
            _arg_value(plan["command"], "--kv-high-precision-layers"), "auto"
        )

    def test_flash_next_beyond_native_is_refused_in_this_release(self):
        cases = (
            ({}, "512k"),
            ({}, "1m"),
            ({}, "yarn"),
            ({"FLUX_MAX_SEQ_LEN": "524288"}, "512k"),
            ({"FLUX_MAX_SEQ_LEN": "262145"}, "auto"),
        )
        for env, mode in cases:
            with self.subTest(mode=mode, env=env):
                ver = re.escape(flux_autodetect.engine_version())
                with self.assertRaisesRegex(
                    ValueError,
                    rf"not supported in v{ver} for this mode; max is 262,144 via FLUX_MODE=256k",
                ):
                    self._build(env, mode=mode)
        # The native window itself is untouched.
        plan = self._build()
        self.assertEqual(plan["max_seq_len"], 262144)

    def test_qsa_c4_uses_serial_oracle_contract(self):
        plan = self._build({"FLUX_CONCURRENCY": "4", "FLUX_MAX_SEQ_LEN": "2048"})

        self.assertEqual(plan["concurrency"], 4)
        self.assertEqual(_arg_value(plan["command"], "--max-batch-size"), "4")
        self.assertEqual(_arg_value(plan["command"], "--max-num-seqs"), "4")
        self.assertEqual(_arg_value(plan["command"], "--swap-space-gb"), "0")
        self.assertEqual(
            _arg_value(plan["command"], "--enable-prefix-caching"), "false"
        )
        self.assertEqual(_arg_value(plan["command"], "--ssm-cache-slots"), "0")
        self.assertFalse(plan["speculative_enabled"])
        self.assertNotIn("--speculative", plan["command"])

    def test_qsa_short_context_c128_is_admitted_with_fail_closed_profile(self):
        # Explicit serial opt-out (ATLAS_QWEN4_NATIVE_BATCH=0) remains admitted up to C128 with a fail-closed profile.
        plan = self._build(
            {
                "FLUX_CONCURRENCY": "128",
                "FLUX_MAX_SEQ_LEN": "2048",
                "ATLAS_QWEN4_NATIVE_BATCH": "0",
            }
        )

        self.assertEqual(plan["concurrency"], 128)
        self.assertEqual(_arg_value(plan["command"], "--max-batch-size"), "128")
        self.assertEqual(_arg_value(plan["command"], "--max-num-seqs"), "128")
        self.assertEqual(_arg_value(plan["command"], "--swap-space-gb"), "0")
        self.assertFalse(plan["speculative_enabled"])

    def test_qsa_native_batching_default_caps_concurrency_at_c128(self):
        for width in ("32", "64", "128"):
            plan = self._build({"FLUX_CONCURRENCY": width, "FLUX_MAX_SEQ_LEN": "2048"})
            self.assertEqual(_arg_value(plan["command"], "--max-batch-size"), width)
            # The launcher leaves the selector unset: the server default is native.
            self.assertNotIn("ATLAS_QWEN4_NATIVE_BATCH", plan.get("env", {}))
        for env in (
            {"FLUX_CONCURRENCY": "129"},
            {"FLUX_CONCURRENCY": "129", "ATLAS_QWEN4_NATIVE_BATCH": "1"},
        ):
            with self.assertRaisesRegex(ValueError, "logical C128 admission ceiling"):
                self._build({**env, "FLUX_MAX_SEQ_LEN": "2048"})
        # The FLUX_ spelling of the opt-out is synchronized like every selector.
        plan = self._build(
            {
                "FLUX_CONCURRENCY": "64",
                "FLUX_MAX_SEQ_LEN": "2048",
                "FLUX_QWEN4_NATIVE_BATCH": "0",
            }
        )
        self.assertEqual(_arg_value(plan["command"], "--max-batch-size"), "64")

    def test_qsa_rows_without_native_batching_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "QSA requires native batching"):
            self._build(
                {
                    "FLUX_CONCURRENCY": "4",
                    "FLUX_MAX_SEQ_LEN": "8192",
                    "ATLAS_QWEN4_NATIVE_BATCH": "0",
                    "ATLAS_QWEN4_NATIVE_BATCH_QSA": "1",
                }
            )

    def test_native_qsa_rows_require_the_qsa_side_index(self):
        model_info = _qsa_model_info()
        model_info["requires_qwen4_qsa"] = False
        env = {
            "FLUX_CONCURRENCY": "4",
            "FLUX_MAX_SEQ_LEN": "2048",
            "ATLAS_QWEN4_NATIVE_BATCH_QSA": "1",
        }
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch.object(
                flux_autodetect,
                "detect_hardware_topology",
                return_value=_hardware_info(),
            ),
            mock.patch.object(flux_autodetect, "_load_recipe_env", return_value={}),
        ):
            with self.assertRaisesRegex(ValueError, "--qwen4-qsa"):
                flux_autodetect.build_launch_plan(model_info, "256k", [])
            # An explicit --qwen4-qsa satisfies it.
            plan = flux_autodetect.build_launch_plan(model_info, "256k", ["--qwen4-qsa"])
            self.assertIn("--qwen4-qsa", plan["command"])
        # The QSA model enables the side index itself.
        plan = self._build(env)
        self.assertIn("--qwen4-qsa", plan["command"])

    def test_qsa_native_batch_selector_is_exact(self):
        for value in ("", "true", "2"):
            with self.assertRaisesRegex(ValueError, "exactly 0 or 1"):
                self._build(
                    {
                        "FLUX_CONCURRENCY": "2",
                        "FLUX_MAX_SEQ_LEN": "2048",
                        "ATLAS_QWEN4_NATIVE_BATCH": value,
                    }
                )

    def test_concurrency_above_c128_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "logical C128 admission ceiling"):
            self._build({"FLUX_CONCURRENCY": "129"})

    def test_zero_concurrency_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "between 1"):
            self._build({"FLUX_CONCURRENCY": "0"})

    def test_qsa_concurrency_accepts_nvfp4_and_rejects_fp8_and_extended_context(self):
        plan = self._build({"FLUX_CONCURRENCY": "2", "FLUX_KV_CACHE_DTYPE": "nvfp4"})
        self.assertEqual(plan["kv_cache_dtype"], "nvfp4")
        self.assertEqual(_arg_value(plan["command"], "--kv-cache-dtype"), "nvfp4")
        for unsupported in ("fp8", "turbo4"):
            with self.subTest(kv_cache_dtype=unsupported):
                with self.assertRaisesRegex(ValueError, "bf16 or nvfp4"):
                    self._build({"FLUX_CONCURRENCY": "2", "FLUX_KV_CACHE_DTYPE": unsupported})
        ver = re.escape(flux_autodetect.engine_version())
        with self.assertRaisesRegex(ValueError, rf"not supported in v{ver}"):
            self._build(
                {"FLUX_CONCURRENCY": "2", "FLUX_MAX_SEQ_LEN": "524288"}, mode="512k"
            )

    def test_kv_pool_and_queue_depth_are_absent_by_default(self):
        plan = self._build({"FLUX_CONCURRENCY": "2", "FLUX_MAX_SEQ_LEN": "2048"})
        self.assertNotIn("--kv-pool-tokens", plan["command"])
        self.assertNotIn("--max-queue-depth", plan["command"])
        self.assertNotIn("FLUX_KV_POOL_TOKENS", plan["env"])
        self.assertNotIn("ATLAS_KV_POOL_TOKENS", plan["env"])

    def test_shared_kv_pool_admits_two_256k_sessions_queued_to_c4(self):
        plan = self._build(
            {
                "FLUX_CONCURRENCY": "4",
                "FLUX_MAX_SEQ_LEN": "262144",
                "FLUX_KV_CACHE_DTYPE": "bf16",
                "FLUX_KV_POOL_TOKENS": "524288",
                "FLUX_QUEUE_DEPTH": "2",
            },
            mode="256k",
        )
        self.assertEqual(_arg_value(plan["command"], "--kv-pool-tokens"), "524288")
        self.assertEqual(_arg_value(plan["command"], "--max-queue-depth"), "2")
        self.assertEqual(_arg_value(plan["command"], "--max-batch-size"), "4")
        self.assertNotIn("ATLAS_KV_POOL_TOKENS", plan["env"])
        self.assertNotIn("ATLAS_QUEUE_DEPTH", plan["env"])

    def test_shared_kv_pool_admits_nvfp4_and_rejects_fp8_at_c2(self):
        plan = self._build(
            {
                "FLUX_CONCURRENCY": "2",
                "FLUX_MAX_SEQ_LEN": "262144",
                "FLUX_KV_CACHE_DTYPE": "nvfp4",
                "FLUX_KV_POOL_TOKENS": "524288",
            },
            mode="256k",
        )
        self.assertEqual(plan["kv_cache_dtype"], "nvfp4")
        self.assertEqual(_arg_value(plan["command"], "--kv-cache-dtype"), "nvfp4")
        self.assertEqual(_arg_value(plan["command"], "--kv-pool-tokens"), "524288")
        with self.assertRaisesRegex(ValueError, "bf16 or nvfp4"):
            self._build(
                {
                    "FLUX_CONCURRENCY": "2",
                    "FLUX_MAX_SEQ_LEN": "262144",
                    "FLUX_KV_CACHE_DTYPE": "fp8",
                    "FLUX_KV_POOL_TOKENS": "524288",
                },
                mode="256k",
            )

    def test_routed_groups_stay_enabled_for_dflash_and_suppress_self_and_ngram_spec(self):
        dflash_info = _qsa_model_info()
        dflash_info["speculative"] = {
            "mode": "dflash",
            "drafter_path": "/opt/flux/models/Qwen3.8-Flash-Next-DFlash",
            "gamma": 7,
        }
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(
                flux_autodetect,
                "detect_hardware_topology",
                return_value=_hardware_info(),
            ),
            mock.patch.object(flux_autodetect, "_load_recipe_env", return_value={}),
        ):
            dflash_plan = flux_autodetect.build_launch_plan(dflash_info, "256k", [])
            self_plan = flux_autodetect.build_launch_plan(
                _qsa_model_info(), "256k", ["--self-speculative"]
            )
            ngram_plan = flux_autodetect.build_launch_plan(
                _qsa_model_info(), "256k", ["--ngram-speculative"]
            )
        self.assertEqual(dflash_plan["env"]["ATLAS_FAST_LOAD_ROUTED_GROUPS"], "1")
        self.assertEqual(dflash_plan["env"]["ATLAS_FAST_LOAD_ROUTED_GROUPS_EAGER"], "1")
        self.assertEqual(self_plan["env"]["ATLAS_FAST_LOAD_ROUTED_GROUPS"], "0")
        self.assertEqual(self_plan["env"]["ATLAS_FAST_LOAD_ROUTED_GROUPS_EAGER"], "0")
        self.assertEqual(ngram_plan["env"]["ATLAS_FAST_LOAD_ROUTED_GROUPS"], "0")
        self.assertEqual(ngram_plan["env"]["ATLAS_FAST_LOAD_ROUTED_GROUPS_EAGER"], "0")

    def test_qsa_concurrency_preserves_recipe_mtp_conf_and_history_drafts(self):
        model_info = _qsa_model_info()
        model_info["speculative"] = {
            "mode": "native-mtp",
            "drafter_path": None,
            "gamma": 4,
            "num_drafts": 4,
            "native_mtp_k": 5,
            "mtp_layers": 1,
        }
        recipe_env = {
            "ATLAS_QWEN4_MTP_CONF": "0.6",
            "ATLAS_QWEN4_MTP_CONF_EARLY": "1",
            "ATLAS_HISTORY_DRAFTS": "1",
            "ATLAS_SPEC_ADAPTIVE_DEPTH": "1",
            "ATLAS_QWEN4_MTP_ALLOW_ZERO": "1",
        }
        with (
            mock.patch.dict(os.environ, {"FLUX_CONCURRENCY": "16"}, clear=True),
            mock.patch.object(
                flux_autodetect,
                "detect_hardware_topology",
                return_value=_hardware_info(),
            ),
            mock.patch.object(
                flux_autodetect, "_load_recipe_env", return_value=recipe_env
            ),
        ):
            plan = flux_autodetect.build_launch_plan(model_info, "max-prefill", [])
        self.assertEqual(plan["env"]["ATLAS_QWEN4_MTP_CONF"], "0.6")
        self.assertEqual(plan["env"]["ATLAS_QWEN4_MTP_CONF_EARLY"], "1")
        self.assertEqual(plan["env"]["ATLAS_HISTORY_DRAFTS"], "1")
        self.assertEqual(plan["env"]["ATLAS_SPEC_ADAPTIVE_DEPTH"], "1")
        self.assertEqual(plan["env"]["ATLAS_QWEN4_MTP_ALLOW_ZERO"], "1")
        self.assertEqual(plan["env"]["ATLAS_QWEN4_MTP_VOCAB_PRUNE"], "0")

    def test_kv_pool_and_queue_depth_values_are_validated(self):
        for name, value in (
            ("FLUX_KV_POOL_TOKENS", "0"),
            ("FLUX_KV_POOL_TOKENS", "1M"),
            ("FLUX_QUEUE_DEPTH", "-1"),
        ):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                self._build({"FLUX_MAX_SEQ_LEN": "2048", name: value})
        with self.assertRaisesRegex(ValueError, "must hold one"):
            self._build({"FLUX_MAX_SEQ_LEN": "8192", "FLUX_KV_POOL_TOKENS": "4096"})

    def test_native_262k_recipe_alone_controls_public_plan_selectors(self):
        model_info = _qsa_model_info()
        model_info["recommended_max_seq_len"] = 135168
        native_recipe = {
            "FLUX_MODE": "256k",
            "FLUX_MAX_SEQ_LEN": "262144",
            "FLUX_MAX_PREFILL_TOKENS": "16384",
            "FLUX_CONCURRENCY": "1",
            "FLUX_KV_CACHE_DTYPE": "bf16",
            "FLUX_SPEC_MODE": "none",
        }

        def load_recipe(name):
            return native_recipe if name == "native-262k.env" else {}

        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(
                flux_autodetect,
                "detect_hardware_topology",
                return_value=_hardware_info(),
            ),
            mock.patch.object(
                flux_autodetect, "_load_recipe_env", side_effect=load_recipe
            ),
        ):
            plan = flux_autodetect.build_launch_plan(
                model_info,
                "auto",
                [],
                custom_recipe="native-262k.env",
            )

        self.assertEqual(plan["mode"], "256k")
        self.assertEqual(plan["max_seq_len"], 262144)
        self.assertEqual(plan["max_prefill_tokens"], 16384)
        self.assertEqual(plan["concurrency"], 1)
        self.assertEqual(plan["kv_cache_dtype"], "bf16")
        self.assertFalse(plan["speculative_enabled"])
        self.assertEqual(_arg_value(plan["command"], "--max-seq-len"), "262144")
        self.assertEqual(_arg_value(plan["command"], "--kv-cache-dtype"), "bf16")

    def test_explicit_mode_and_public_selectors_override_recipe(self):
        native_recipe = {
            "FLUX_MODE": "256k",
            "FLUX_MAX_SEQ_LEN": "262144",
            "FLUX_MAX_PREFILL_TOKENS": "16384",
            "FLUX_KV_CACHE_DTYPE": "bf16",
            "FLUX_SPEC_MODE": "none",
        }

        def load_recipe(name):
            return native_recipe if name == "native-262k.env" else {}

        explicit = {
            "FLUX_MAX_SEQ_LEN": "147456",
            "FLUX_KV_CACHE_DTYPE": "nvfp4",
        }
        with (
            mock.patch.dict(os.environ, explicit, clear=True),
            mock.patch.object(
                flux_autodetect,
                "detect_hardware_topology",
                return_value=_hardware_info(),
            ),
            mock.patch.object(
                flux_autodetect, "_load_recipe_env", side_effect=load_recipe
            ),
        ):
            plan = flux_autodetect.build_launch_plan(
                _qsa_model_info(),
                "interactive",
                [],
                custom_recipe="native-262k.env",
                custom_env={"FLUX_MAX_PREFILL_TOKENS": "4096"},
                mode_explicit=True,
            )

        self.assertEqual(plan["mode"], "interactive")
        self.assertEqual(plan["max_seq_len"], 147456)
        self.assertEqual(plan["max_prefill_tokens"], 4096)
        self.assertEqual(plan["kv_cache_dtype"], "nvfp4")
        self.assertEqual(_arg_value(plan["command"], "--max-seq-len"), "147456")
        self.assertEqual(_arg_value(plan["command"], "--kv-cache-dtype"), "nvfp4")
        self.assertEqual(plan["env"]["FLUX_MODE"], "interactive")
        self.assertEqual(plan["env"]["FLUX_MAX_SEQ_LEN"], "147456")
        self.assertEqual(plan["env"]["FLUX_MAX_PREFILL_TOKENS"], "4096")
        self.assertEqual(plan["env"]["FLUX_KV_CACHE_DTYPE"], "nvfp4")

    def test_qsa_agent_mode_enables_prefix_caching_and_ple_ssm_slots_at_c1_and_disables_at_cmulti(self):
        plan = self._build(mode="agent")

        self.assertEqual(plan["mode"], "agent")
        self.assertEqual(plan["max_prefill_tokens"], 4096)
        self.assertEqual(plan["recipe_file"], "flux-v9-w4096-interactive.env")
        self.assertTrue(plan["prefix_caching"])
        self.assertEqual(plan["ssm_cache_slots"], 16)
        self.assertEqual(_arg_value(plan["command"], "--enable-prefix-caching"), "true")
        self.assertEqual(_arg_value(plan["command"], "--ssm-cache-slots"), "16")
        self.assertEqual(plan["env"]["ATLAS_QWEN4_PREFILL_ATTN_CORE16"], "0")
        self.assertEqual(plan["tool_call_parser"], "qwen3_coder")
        self.assertEqual(_arg_value(plan["command"], "--tool-call-parser"), "qwen3_coder")
        self.assertEqual(plan["tool_max_tokens"], 262144)
        self.assertEqual(_arg_value(plan["command"], "--tool-max-tokens"), "262144")

        cmulti = self._build(mode="agent", env={"FLUX_CONCURRENCY": "4"})
        self.assertFalse(cmulti["prefix_caching"])
        self.assertEqual(cmulti["ssm_cache_slots"], 0)
        self.assertEqual(_arg_value(cmulti["command"], "--enable-prefix-caching"), "false")
        self.assertEqual(_arg_value(cmulti["command"], "--ssm-cache-slots"), "0")

    def test_qsa_decode_mode_aligns_prefill_chunk_to_8192(self):
        plan = self._build(mode="decode")

        self.assertEqual(plan["mode"], "decode")
        self.assertEqual(plan["max_prefill_tokens"], 8192)
        self.assertEqual(_arg_value(plan["command"], "--max-prefill-tokens"), "8192")
        self.assertEqual(plan["env"]["ATLAS_QWEN4_PREFILL_WIDE"], "8192")
        self.assertEqual(plan["env"]["FLUX_MAX_PREFILL_TOKENS"], "8192")

    def test_qsa_native_262k_enables_fast_routed_group_loader_and_w16384(self):
        plan = self._build(mode="256k")

        self.assertEqual(plan["mode"], "256k")
        self.assertEqual(plan["max_seq_len"], 262144)
        self.assertEqual(plan["max_prefill_tokens"], 16384)
        self.assertEqual(plan["env"]["ATLAS_FAST_LOAD_ROUTED_GROUPS"], "1")
        self.assertEqual(plan["env"]["ATLAS_FAST_LOAD_ROUTED_GROUPS_EAGER"], "1")
        self.assertEqual(plan["env"]["ATLAS_FAST_LOAD_DIRECT_IO_TENSOR_CAP"], "20000")
        self.assertEqual(plan["env"]["ATLAS_QWEN4_BOUNDED_TRANSPOSE_OWNERS"], "0")
        self.assertEqual(plan["env"]["ATLAS_MEMAVAILABLE_FLOOR_OVERRIDE"], "1")
        self.assertEqual(plan["env"]["ATLAS_QWEN4_PREFILL_QSA_FP8"], "0")

    def test_qsa_concurrency_c8_disables_c1_qsa_fp8_shadows_and_keeps_single_seq_verify_graph(self):
        plan = self._build(env={"FLUX_CONCURRENCY": "8"}, mode="auto")

        self.assertEqual(plan["concurrency"], 8)
        self.assertEqual(plan["max_seq_len"], 262144)
        self.assertEqual(_arg_value(plan["command"], "--kv-pool-tokens"), "393216")
        self.assertEqual(plan["max_prefill_tokens"], 8192)
        self.assertEqual(plan["request_timeout"], 1800)
        self.assertEqual(_arg_value(plan["command"], "--request-timeout"), "1800")
        self.assertEqual(plan["env"]["ATLAS_KV_ADMIT_WATERMARK"], "32768")
        self.assertEqual(plan["env"]["FLUX_KV_ADMIT_WATERMARK"], "32768")
        self.assertEqual(plan["env"]["ATLAS_QWEN4_BOUNDED_TRANSPOSE_OWNERS"], "0")
        self.assertEqual(plan["env"]["ATLAS_QWEN4_PREFILL_QSA_FP8"], "0")
        self.assertNotEqual(
            plan["env"].get("ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_WEIGHT_CACHE"),
            "0",
        )
        self.assertEqual(plan["env"]["ATLAS_QWEN4_VERIFY_GRAPH"], "1")
        self.assertEqual(plan["env"]["FLUX_QWEN4_VERIFY_GRAPH"], "1")
        self.assertEqual(plan["env"]["ATLAS_SCHED_BURST_GATHER_MS"], "3")
        self.assertNotIn("ATLAS_DFLASH_SPEC_CYCLE_V2", plan["env"])
        self.assertNotIn("ATLAS_MOE_T_LANES_WORKLIST_R8", plan["env"])
        self.assertEqual(plan["env"]["ATLAS_QWEN4_MTP_CONF_CUM"], "0.4")

        c16_plan = self._build(env={"FLUX_CONCURRENCY": "16"}, mode="auto")
        self.assertEqual(c16_plan["env"]["ATLAS_QWEN4_VERIFY_GRAPH"], "1")
        self.assertEqual(c16_plan["env"]["FLUX_QWEN4_VERIFY_GRAPH"], "1")

        explicit_off = self._build(
            env={"FLUX_CONCURRENCY": "8", "ATLAS_QWEN4_VERIFY_GRAPH": "0"},
            mode="auto",
        )
        self.assertEqual(explicit_off["env"]["ATLAS_QWEN4_VERIFY_GRAPH"], "0")
        self.assertEqual(explicit_off["env"]["FLUX_QWEN4_VERIFY_GRAPH"], "0")

        c32_plan = self._build(env={"FLUX_CONCURRENCY": "32"}, mode="auto")
        self.assertEqual(c32_plan["env"]["ATLAS_QWEN4_VERIFY_GRAPH"], "0")
        self.assertEqual(c32_plan["env"]["FLUX_QWEN4_VERIFY_GRAPH"], "0")

        wide_override = self._build(
            env={"FLUX_CONCURRENCY": "8", "FLUX_MAX_PREFILL_TOKENS": "8192"},
            mode="auto",
        )
        self.assertEqual(wide_override["max_prefill_tokens"], 8192)
        self.assertEqual(wide_override["env"]["ATLAS_QWEN4_PREFILL_WIDE"], "8192")
        self.assertEqual(wide_override["env"]["FLUX_QWEN4_PREFILL_WIDE"], "8192")

    def test_qsa_1m_nvfp4_and_wide_concurrency_modes(self):
        plan_1m = self._build(mode="1m-nvfp4")
        self.assertEqual(plan_1m["mode"], "1m-nvfp4")
        self.assertEqual(plan_1m["max_seq_len"], 1048576)
        self.assertEqual(plan_1m["kv_cache_dtype"], "nvfp4")
        self.assertEqual(_arg_value(plan_1m["command"], "--rope-yarn-factor"), "4")
        self.assertEqual(plan_1m["env"]["ATLAS_QWEN4_QSA_SELECT_V3"], "1")
        self.assertEqual(plan_1m["env"]["ATLAS_QWEN4_PREFILL_QSA_FP8"], "0")

        plan_manual_1m = self._build(
            env={"FLUX_MAX_SEQ_LEN": "1048576", "FLUX_KV_CACHE_DTYPE": "nvfp4"},
            mode="nextgen",
        )
        self.assertEqual(plan_manual_1m["max_seq_len"], 1048576)
        self.assertEqual(plan_manual_1m["kv_cache_dtype"], "nvfp4")
        self.assertEqual(_arg_value(plan_manual_1m["command"], "--rope-yarn-factor"), "4")
        self.assertEqual(plan_manual_1m["env"]["ATLAS_QWEN4_QSA_SELECT_V3"], "1")

        model_mtp = _qsa_model_info()
        model_mtp["speculative"] = {
            "mode": "native-mtp",
            "drafter_path": None,
            "gamma": 4,
            "num_drafts": 4,
            "native_mtp_k": 5,
            "mtp_layers": 1,
        }
        with (
            mock.patch.dict(os.environ, {"FLUX_CONCURRENCY": "64"}, clear=True),
            mock.patch.object(
                flux_autodetect,
                "detect_hardware_topology",
                return_value=_hardware_info(),
            ),
            mock.patch.object(
                flux_autodetect,
                "_load_recipe_env",
                return_value={"ATLAS_QWEN4_HYBRID_ALL_K": "1"},
            ),
        ):
            c64 = flux_autodetect.build_launch_plan(model_mtp, "nextgen", [])
        self.assertTrue(c64["speculative_enabled"])
        self.assertEqual(c64["env"]["ATLAS_QWEN4_BATCHED_ADAPTIVE_K"], "1")
        self.assertEqual(_arg_value(c64["command"], "--kv-pool-tokens"), "65536")

    def test_two_spark_1m_bf16_and_parallel_262k_pool(self):
        plan_2x_1m = self._build(mode="2x-spark-1m")
        self.assertEqual(plan_2x_1m["mode"], "2x-spark-1m")
        self.assertEqual(plan_2x_1m["max_seq_len"], 1048576)
        self.assertEqual(plan_2x_1m["kv_cache_dtype"], "bf16")
        self.assertEqual(_arg_value(plan_2x_1m["command"], "--world-size"), "2")
        self.assertEqual(_arg_value(plan_2x_1m["command"], "--tp-size"), "2")
        self.assertEqual(_arg_value(plan_2x_1m["command"], "--ep-size"), "2")
        self.assertEqual(_arg_value(plan_2x_1m["command"], "--rope-yarn-factor"), "4")
        self.assertEqual(plan_2x_1m["env"]["ATLAS_QWEN4_QSA_SELECT_V3"], "1")
        self.assertEqual(plan_2x_1m["env"]["ATLAS_FAST_LOAD_ROUTED_GROUPS"], "0")

        plan_2x_c8 = self._build(env={"FLUX_CONCURRENCY": "8"}, mode="2x-spark")
        self.assertEqual(plan_2x_c8["concurrency"], 8)
        self.assertEqual(plan_2x_c8["max_seq_len"], 262144)
        self.assertEqual(_arg_value(plan_2x_c8["command"], "--kv-pool-tokens"), "2621440")
        self.assertEqual(_arg_value(plan_2x_c8["command"], "--world-size"), "2")
        self.assertEqual(_arg_value(plan_2x_c8["command"], "--tp-size"), "2")
        self.assertEqual(_arg_value(plan_2x_c8["command"], "--ep-size"), "2")
        self.assertEqual(plan_2x_c8["env"]["ATLAS_KV_ADMIT_WATERMARK"], "32768")
        self.assertEqual(plan_2x_c8["env"]["ATLAS_FAST_LOAD_ROUTED_GROUPS"], "0")

    def test_nextgen_c128_and_exact_modes_set_1800s_request_timeout(self):
        for mode in ("nextgen", "c128", "exact"):
            plan = self._build(mode=mode)
            self.assertEqual(plan["request_timeout"], 1800, mode)
            self.assertEqual(_arg_value(plan["command"], "--request-timeout"), "1800", mode)

    def test_qsa_default_drafts_arms_k7_through_c16_under_adaptive_k_and_caps_c_gt_16_or_fixed_k5(self):
        model_info = _qsa_model_info()
        model_info["speculative"] = {
            "mode": "native-mtp",
            "drafter_path": None,
            "gamma": 6,
            "num_drafts": 6,
            "native_mtp_k": 7,
            "mtp_layers": 1,
        }

        def build(mode, env=None, recipe_env=None):
            with (
                mock.patch.dict(os.environ, env or {}, clear=True),
                mock.patch.object(
                    flux_autodetect,
                    "detect_hardware_topology",
                    return_value=_hardware_info(),
                ),
                mock.patch.object(
                    flux_autodetect,
                    "_load_recipe_env",
                    return_value=recipe_env or {},
                ),
            ):
                return flux_autodetect.build_launch_plan(model_info, mode, [])

        c1_plan = build("auto")
        self.assertEqual(_arg_value(c1_plan["command"], "--num-drafts"), "6")
        self.assertEqual(c1_plan["env"]["ATLAS_QWEN4_MTP_CONF_CUM"], "0.4")
        self.assertEqual(c1_plan["env"]["ATLAS_SCHED_BURST_GATHER_MS"], "3")
        self.assertNotIn("ATLAS_MOE_T_LANES_WORKLIST_R8", c1_plan["env"])

        # Fixed-K5 without HYBRID_ALL_K / ADAPTIVE_K keeps --num-drafts 4 (K=5).
        c16_fixed = build("max-prefill", {"FLUX_CONCURRENCY": "16"})
        self.assertEqual(_arg_value(c16_fixed["command"], "--num-drafts"), "4")
        self.assertEqual(c16_fixed["env"]["ATLAS_QWEN4_BATCHED_K5"], "1")
        self.assertEqual(c16_fixed["env"]["ATLAS_QWEN4_MTP_CONF_CUM"], "0.4")
        self.assertEqual(c16_fixed["env"]["ATLAS_SCHED_BURST_GATHER_MS"], "3")
        self.assertNotIn("ATLAS_MOE_T_LANES_WORKLIST_R8", c16_fixed["env"])
        self.assertNotIn("ATLAS_SPEC_ADAPTIVE_DEPTH", c16_fixed["env"])
        self.assertNotIn("ATLAS_QWEN4_MTP_ALLOW_ZERO", c16_fixed["env"])
        self.assertNotIn("ATLAS_DFLASH_SPEC_CYCLE_V2", c16_fixed["env"])

        # Both FLUX_MODE=auto and FLUX_MODE=nextgen arm K=7 (--num-drafts 6) across
        # C=1..16 (16x7=112 <= 128) so default container launches keep full K=7 parity.
        for mode in ("auto", "nextgen"):
            for conc in ("1", "4", "8", "16"):
                with self.subTest(mode=mode, concurrency=conc):
                    plan = build(mode, {"FLUX_CONCURRENCY": conc})
                    self.assertEqual(_arg_value(plan["command"], "--num-drafts"), "6")
                    self.assertIn("K=7", plan["mode_summary"])
                    if int(conc) >= 2:
                        self.assertEqual(plan["env"]["ATLAS_QWEN4_BATCHED_K5"], "1")
                        self.assertEqual(plan["env"]["ATLAS_QWEN4_BATCHED_ADAPTIVE_K"], "1")

            # C>16 in auto/nextgen caps default drafts back to 4 (K=5) while adaptive-K narrows at runtime.
            for conc in ("32", "64"):
                with self.subTest(mode=mode, concurrency=conc):
                    plan = build(mode, {"FLUX_CONCURRENCY": conc})
                    self.assertEqual(_arg_value(plan["command"], "--num-drafts"), "4")
                    self.assertIn("K=5", plan["mode_summary"])
                    self.assertEqual(plan["env"]["ATLAS_QWEN4_BATCHED_ADAPTIVE_K"], "1")

        # Real auto and nextgen recipes both arm K=7 at C=16 end-to-end.
        for mode in ("auto", "nextgen"):
            with (
                self.subTest(real_recipe_mode=mode),
                mock.patch.dict(os.environ, {"FLUX_CONCURRENCY": "16"}, clear=True),
                mock.patch.object(
                    flux_autodetect,
                    "detect_hardware_topology",
                    return_value=_hardware_info(),
                ),
            ):
                c16_real = flux_autodetect.build_launch_plan(model_info, mode, [])
            self.assertEqual(_arg_value(c16_real["command"], "--num-drafts"), "6")
            self.assertEqual(c16_real["env"]["ATLAS_QWEN4_BATCHED_K5"], "1")
            self.assertEqual(c16_real["env"]["ATLAS_QWEN4_BATCHED_ADAPTIVE_K"], "1")

        # Explicit adaptive-K opt-out or serial K5 caps C<=16 back to K=5 (4 drafts).
        c16_adaptive_off = build(
            "nextgen",
            {"FLUX_CONCURRENCY": "16", "ATLAS_QWEN4_BATCHED_ADAPTIVE_K": "0"},
        )
        self.assertEqual(_arg_value(c16_adaptive_off["command"], "--num-drafts"), "4")

        c8_serial_k5 = build(
            "nextgen",
            {"FLUX_CONCURRENCY": "8", "ATLAS_QWEN4_K5_CMULTI": "1"},
        )
        self.assertEqual(_arg_value(c8_serial_k5["command"], "--num-drafts"), "4")

        # Explicit FLUX_GAMMA=7 (K=8, 16x8=128) is preserved at C=16.
        c16_gamma7 = build("nextgen", {"FLUX_CONCURRENCY": "16", "FLUX_GAMMA": "7"})
        self.assertEqual(_arg_value(c16_gamma7["command"], "--num-drafts"), "7")
        self.assertIn("K=8", c16_gamma7["mode_summary"])

        # Original fixture dict must remain unmutated so subsequent C=1 calls keep 6.
        self.assertEqual(model_info["speculative"]["num_drafts"], 6)

        opt_out = build("auto", {"ATLAS_QWEN4_MTP_CONF_CUM": "0"})
        self.assertEqual(opt_out["env"]["ATLAS_QWEN4_MTP_CONF_CUM"], "0")


class QsaConcurrencyMoeExactRowsTests(TestCase):
    """H13: FLUX_MODE=fast has no exact-rows MoE route, so C2+ must opt out of
    native QSA rows (the server refuses them as moe_exact_rows_unavailable).
    The server reads an UNSET ATLAS_QWEN4_NATIVE_BATCH_QSA as on, so the opt-out
    must be an explicit 0; absence is the rc8 R5 failure."""

    def _build(self, mode, env):
        flux_autodetect._SERIAL_QSA_NOTE_PRINTED = False
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch.object(
                flux_autodetect,
                "detect_hardware_topology",
                return_value=_hardware_info(),
            ),
            mock.patch("sys.stderr", new_callable=io.StringIO) as stderr,
        ):
            plan = flux_autodetect.build_launch_plan(_qsa_model_info(), mode, [])
        return plan, stderr.getvalue()

    def test_moe_exact_rows_mirror_matches_the_server_parse(self):
        selected = flux_autodetect._moe_exact_rows_selected
        self.assertTrue(selected({"ATLAS_MOE_T_LANES": "1"}))
        self.assertFalse(selected({}))
        self.assertFalse(selected({"ATLAS_MOE_T_LANES": "true"}))
        self.assertTrue(selected({"ATLAS_MOE_T_LANES": "1", "ATLAS_UNIFIED_MOE_LAYOUT": "TRUE"}))
        self.assertFalse(selected({"ATLAS_MOE_T_LANES": "1", "ATLAS_UNIFIED_MOE_LAYOUT": "0"}))
        self.assertFalse(selected({"ATLAS_MOE_T_LANES": "1", "ATLAS_HYBRID_MOE_LAYOUT": "1"}))

    def test_fast_mode_concurrency_serves_serially_with_an_info_line(self):
        plan, stderr = self._build("fast", {"FLUX_CONCURRENCY": "4"})
        self.assertEqual(_arg_value(plan["command"], "--max-batch-size"), "4")
        # plan["env"] is exactly the environment the server process receives.
        self.assertEqual(plan["env"]["ATLAS_QWEN4_NATIVE_BATCH_QSA"], "0")
        self.assertEqual(plan["env"]["FLUX_QWEN4_NATIVE_BATCH_QSA"], "0")
        # The precision recipe itself is untouched: no T-lanes, no selector.
        self.assertNotIn("ATLAS_MOE_T_LANES", plan["env"])
        self.assertNotIn("ATLAS_QWEN4_NATIVE_BATCH", plan["env"])
        self.assertIn("[flux] INFO: FLUX_MODE=fast has no exact-rows MoE route", stderr)
        self.assertEqual(stderr.count("exact-rows MoE route"), 1)

    def test_fast_mode_short_context_concurrency_also_opts_out(self):
        plan, stderr = self._build(
            "fast", {"FLUX_CONCURRENCY": "4", "FLUX_MAX_SEQ_LEN": "2048"}
        )
        self.assertEqual(plan["env"]["ATLAS_QWEN4_NATIVE_BATCH_QSA"], "0")
        self.assertIn("exact-rows MoE route", stderr)

    def test_fast_mode_c1_is_unchanged(self):
        plan, stderr = self._build("fast", {})
        self.assertNotIn("ATLAS_QWEN4_NATIVE_BATCH_QSA", plan["env"])
        self.assertNotIn("exact-rows MoE route", stderr)

    def test_max_prefill_concurrency_still_requests_native_qsa_rows(self):
        plan, stderr = self._build("max-prefill", {"FLUX_CONCURRENCY": "4"})
        self.assertEqual(plan["env"]["ATLAS_MOE_T_LANES"], "1")
        self.assertEqual(plan["env"]["ATLAS_QWEN4_NATIVE_BATCH_QSA"], "1")
        self.assertEqual(plan["env"]["FLUX_QWEN4_NATIVE_BATCH_QSA"], "1")
        self.assertNotIn("exact-rows MoE route", stderr)

    def test_fast_mode_with_operator_t_lanes_batches_natively(self):
        plan, stderr = self._build("fast", {"FLUX_CONCURRENCY": "4", "ATLAS_MOE_T_LANES": "1"})
        self.assertEqual(plan["env"]["ATLAS_QWEN4_NATIVE_BATCH_QSA"], "1")
        self.assertNotIn("exact-rows MoE route", stderr)

    def test_fast_mode_explicit_qsa_selector_is_left_to_the_server(self):
        plan, stderr = self._build(
            "fast", {"FLUX_CONCURRENCY": "4", "ATLAS_QWEN4_NATIVE_BATCH_QSA": "0"}
        )
        self.assertEqual(plan["env"]["ATLAS_QWEN4_NATIVE_BATCH_QSA"], "0")
        self.assertNotIn("exact-rows MoE route", stderr)


class NonFlashNext512kModeTests(TestCase):
    """512k/1m stay available to models without the Qwen4 QSA requirement."""

    def _build(self, mode):
        model_info = dict(_qsa_model_info())
        model_info.update(
            {
                "kernel_target": "qwen3.8-27b",
                "selected_binary": "/opt/flux/bin/flux-server-qwen38-27b",
                "path": "/models/qwen38-27b",
                "name": "qwen38-27b",
                "requires_qwen4_qsa": False,
            }
        )
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(
                flux_autodetect, "detect_hardware_topology", return_value=_hardware_info()
            ),
            mock.patch.object(flux_autodetect, "_load_recipe_env", return_value={}),
        ):
            return flux_autodetect.build_launch_plan(model_info, mode, [])

    def test_dense_model_512k_and_1m_are_not_refused(self):
        # The guard is Flash-Next only: the dense plan is whatever 9.15.5
        # already produced for these modes.
        for mode in ("512k", "1m"):
            plan = self._build(mode)
            self.assertIn("--max-seq-len", plan["command"])


class PlanRefusalOutputTests(TestCase):
    """A plan-time ValueError is one `[flux] ERROR:` line and exit 2, not a
    traceback, unless FLUX_DEBUG=1 or FLUX_LOAD_DEBUG=always."""

    def _serve(self, env):
        stderr = io.StringIO()
        with (
            mock.patch.dict(os.environ, {"FLUX_MODE": "512k", **env}, clear=True),
            mock.patch.object(flux_autodetect.sys, "argv", ["flux", "serve", "--dry-run"]),
            mock.patch.object(
                flux_autodetect, "resolve_model_query", return_value=_qsa_model_info()
            ),
            mock.patch.object(
                flux_autodetect, "detect_hardware_topology", return_value=_hardware_info()
            ),
            mock.patch.object(flux_autodetect, "_load_recipe_env", return_value={}),
            mock.patch("sys.stderr", stderr),
            mock.patch("sys.stdout", io.StringIO()),
        ):
            rc = flux_autodetect.main()
        return rc, stderr.getvalue()

    def test_refusal_is_one_clean_line_and_exit_2(self):
        rc, stderr = self._serve({})
        self.assertEqual(rc, 2)
        self.assertNotIn("Traceback", stderr)
        errors = [line for line in stderr.splitlines() if line.startswith("[flux] ERROR: ")]
        self.assertEqual(len(errors), 1)
        self.assertIn(f"not supported in v{flux_autodetect.engine_version()}", errors[0])

    def test_debug_keeps_the_traceback(self):
        for env in ({"FLUX_DEBUG": "1"}, {"FLUX_LOAD_DEBUG": "always"}):
            with self.subTest(env=env):
                rc, stderr = self._serve(env)
                self.assertEqual(rc, 2)
                self.assertIn("Traceback", stderr)
                self.assertIn("[flux] ERROR: ", stderr)


class OutOfTheBoxNextgenAndDflash2ParityTests(TestCase):
    """Zero-flag container parity: FLUX_MODE=auto loads the certified NextGen
    stack and auto-pairs a qualified incoai-dflash2-b8 drafter with all
    certified DFlash2 runtime defaults at C=1 and C=2..16."""

    NEXTGEN_ACCELERATORS = {
        "ATLAS_QWEN4_FP8_PAD128": "1",
        "ATLAS_CUBLASLT_CACHE": "1",
        "ATLAS_QWEN4_GDN_VERIFY_FUSED": "1",
        "ATLAS_QWEN4_QSA_FUSED_GATE": "1",
        "ATLAS_CUDA_PDL": "1",
        "ATLAS_CUDA_WFET_SYNC": "1",
        "ATLAS_HISTORY_DRAFTS": "1",
        "ATLAS_SAMPLER_DEVICE_TOPK": "auto",
        "ATLAS_QWEN4_SEQSTATE": "1",
        "ATLAS_MOE_T_LANES_ROWS_DOWN_T8": "1",
        "ATLAS_PLE_READ_RINGS": "4",
        "ATLAS_QWEN4_K5_BATCH_CONV": "1",
    }

    CONTAINER_DEFAULT_ENV = {
        "FLUX_TARGET": "qwen3.8-flash-next",
        "FLUX_MODE": "auto",
        "FLUX_HOST": "0.0.0.0",
        "FLUX_PORT": "8899",
        "FLUX_CONCURRENCY": "1",
        "FLUX_MAX_SEQ_LEN": "131072",
    }

    def _write_flash_next_base(self, root: Path, *, with_ple: bool = True) -> Path:
        model_dir = root / "Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["Qwen4ExpForCausalLM"],
                    "model_type": "qwen4_exp",
                    "hidden_size": 2560,
                    "num_hidden_layers": 48,
                    "num_local_experts": 512,
                    "max_position_embeddings": 262144,
                    "mtp_num_hidden_layers": 1,
                }
            ),
            encoding="utf-8",
        )
        (model_dir / "model-00001.safetensors").write_bytes(b"x")
        (model_dir / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "mtp.layers.0.mlp.experts.0.gate_proj.weight": "model-00001.safetensors"
                    }
                }
            ),
            encoding="utf-8",
        )
        if with_ple:
            (model_dir / "ple-offload").mkdir()
        return model_dir

    def _write_qualified_dflash2(self, root: Path) -> Path:
        drafter_dir = root / "Qwen3.8-Flash-Next-DFlash2"
        drafter_dir.mkdir()
        (drafter_dir / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["DFlash2DraftModel"],
                    "model_type": "qwen3",
                    "hidden_size": 2560,
                    "intermediate_size": 7680,
                    "num_hidden_layers": 5,
                    "num_target_layers": 48,
                    "num_attention_heads": 32,
                    "num_key_value_heads": 8,
                    "head_dim": 128,
                    "vocab_size": 248320,
                    "is_causal": False,
                    "dflash_config": {
                        "block_size": 8,
                        "mask_token_id": 248077,
                        "target_layer_ids": [5, 15, 25, 35, 45],
                        "conv_kernel_size": 2,
                        "conv_group_size": 16,
                        "selector_rank": 256,
                        "selector_top_k": 16,
                    },
                }
            ),
            encoding="utf-8",
        )
        (drafter_dir / "model.safetensors").write_bytes(b"x")
        return drafter_dir

    def _write_unqualified_native_dflash2(self, root: Path) -> Path:
        drafter_dir = root / "Qwen3.8-Flash-Next-Native-DFlash2"
        drafter_dir.mkdir()
        (drafter_dir / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["DFlash2DraftModel"],
                    "model_type": "qwen3",
                    "hidden_size": 2560,
                    "intermediate_size": 8704,
                    "num_hidden_layers": 6,
                    "num_target_layers": 48,
                    "block_size": 16,
                    "dflash_config": {
                        "block_size": 16,
                        "target_layer_ids": [1, 7, 13, 20, 26, 33, 39, 46],
                        "selector_vocab_size": 248077,
                    },
                }
            ),
            encoding="utf-8",
        )
        (drafter_dir / "model.safetensors").write_bytes(b"x")
        return drafter_dir

    def _write_inferact_mtp(self, root: Path) -> Path:
        mtp_dir = root / "Qwen3.8-Flash-Next-Inferact-MTP"
        mtp_dir.mkdir()
        (mtp_dir / "config.json").write_text(
            json.dumps({"hidden_size": 2560}), encoding="utf-8"
        )
        (mtp_dir / "mtp-00001.safetensors").write_bytes(b"x")
        return mtp_dir

    def _inspect_and_plan(self, model_dir: Path, env=None, extra_args=None, hw=None):
        orig_is_dir = flux_autodetect._safe_is_dir
        blocked_roots = {"/models", "/workspace/models", "/opt/flux/models", "/root/.cache/huggingface/hub"}

        def isolated_is_dir(p):
            if str(p) in blocked_roots:
                return False
            return orig_is_dir(p)

        full_env = dict(self.CONTAINER_DEFAULT_ENV)
        if env:
            full_env.update(env)
        hardware = hw or {
            "sm_family": "gb10-sm121a",
            "hardware_recipe": "flux-nextgen-max-prefill-decode.env",
            "unified_memory_architecture": True,
            "hardware_default_seq_len": 131072,
            "hardware_max_seq_len_fp8_kv": 1048576,
        }
        with (
            mock.patch.dict(os.environ, full_env, clear=True),
            mock.patch.object(flux_autodetect, "_safe_is_dir", side_effect=isolated_is_dir),
            mock.patch.object(flux_autodetect, "detect_hardware_topology", return_value=hardware),
        ):
            info = flux_autodetect.inspect_model(model_dir)
            plan = flux_autodetect.build_launch_plan(
                info, full_env.get("FLUX_MODE", "auto"), list(extra_args or [])
            )
        return info, plan

    def test_detect_hardware_topology_gb10_defaults_to_nextgen_recipe(self):
        with (
            mock.patch.object(flux_autodetect.shutil, "which", return_value="/usr/bin/nvidia-smi"),
            mock.patch.object(
                flux_autodetect.subprocess,
                "check_output",
                return_value="NVIDIA GB10, 128000, 12.1\n",
            ),
        ):
            hw = flux_autodetect.detect_hardware_topology()
        self.assertEqual(hw["sm_family"], "gb10-sm121a")
        self.assertEqual(hw["hardware_recipe"], "flux-nextgen-max-prefill-decode.env")
        self.assertEqual(
            flux_autodetect.MODES_CATALOG["auto"]["recipe_file"],
            "flux-nextgen-max-prefill-decode.env",
        )

    def test_zero_flag_auto_base_model_only_uses_nextgen_recipe_and_native_mtp(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = self._write_flash_next_base(root)
            info, plan = self._inspect_and_plan(model_dir)

        self.assertEqual(info["speculative"]["mode"], "native-mtp")
        self.assertIsNone(info["speculative"]["drafter_path"])
        self.assertEqual(plan["recipe_file"], "flux-nextgen-max-prefill-decode.env")
        self.assertEqual(plan["max_seq_len"], 262144)
        self.assertIn("--speculative", plan["command"])
        self.assertEqual(_arg_value(plan["command"], "--num-drafts"), "6")
        for key, value in self.NEXTGEN_ACCELERATORS.items():
            self.assertEqual(plan["env"][key], value, key)
        self.assertEqual(plan["env"]["ATLAS_SPEC_VERIFY"], "gumbel")
        self.assertEqual(plan["env"]["ATLAS_QWEN4_PREFILL_QSA_FP8"], "qkpv")
        self.assertEqual(plan["env"]["ATLAS_QWEN4_PREFILL_QSA_PIPE"], "1")
        self.assertEqual(plan["env"]["ATLAS_QWEN4_HYBRID_ALL_K"], "1")
        self.assertEqual(plan["env"]["ATLAS_MOE_SCALE_TS4E"], "0")
        self.assertEqual(plan["env"]["ATLAS_QWEN4_FP8_WEIGHT_CACHE_FLOOR_MARGIN_MIB"], "1024")

    def test_zero_flag_auto_auto_pairs_qualified_dflash2_at_c1_and_c2_to_c16(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = self._write_flash_next_base(root)
            drafter_dir = self._write_qualified_dflash2(root)
            self._write_inferact_mtp(root)

            # C=1 zero-flag launch
            info_c1, plan_c1 = self._inspect_and_plan(model_dir)
            self.assertEqual(info_c1["speculative"]["mode"], "dflash")
            self.assertEqual(info_c1["speculative"]["drafter_path"], str(drafter_dir))
            self.assertEqual(plan_c1["recipe_file"], "flux-nextgen-max-prefill-decode.env")
            self.assertIn("--dflash", plan_c1["command"])
            self.assertEqual(_arg_value(plan_c1["command"], "--draft-model"), str(drafter_dir))
            self.assertEqual(_arg_value(plan_c1["command"], "--dflash-gamma"), "7")
            self.assertNotIn("--kv-pool-tokens", plan_c1["command"])
            for key, value in self.NEXTGEN_ACCELERATORS.items():
                self.assertEqual(plan_c1["env"][key], value, key)
            for key, value in {
                "ATLAS_DFLASH_QUANTIZATION": "nvfp4",
                "FLUX_DFLASH_QUANTIZATION": "nvfp4",
                "ATLAS_DFLASH_NOISE_ONLY": "1",
                "ATLAS_DFLASH_NOISE_GRAPH": "1",
                "ATLAS_DISABLE_TREE_WY": "1",
                "ATLAS_MEMLEAN_DRAFTER_BF16": "1",
                "ATLAS_THINK_SPEC": "1",
                "ATLAS_MOE_SCALE_TS4E": "1",
                "FLUX_MOE_SCALE_TS4E": "1",
                "ATLAS_SPEC_VERIFY": "rejection",
                "FLUX_SPEC_VERIFY": "rejection",
                "ATLAS_QWEN4_DFLASH_REJECT_TEMP_SCALE": "1.15",
                "ATLAS_QWEN4_FP8_WEIGHT_CACHE_FLOOR_MARGIN_MIB": "1024",
                "FLUX_QWEN4_FP8_WEIGHT_CACHE_FLOOR_MARGIN_MIB": "1024",
                "ATLAS_QWEN4_PREFILL_QSA_FP8": "qkpv",
                "ATLAS_QWEN4_PREFILL_QSA_PIPE": "1",
                "ATLAS_QWEN4_HYBRID_ALL_K": "1",
            }.items():
                self.assertEqual(plan_c1["env"][key], value, key)
            self.assertNotIn("ATLAS_SSM_DECODE_RING", plan_c1["env"])

            # C=2, C=8, C=16 zero-flag DFlash2 launches
            for conc in ("2", "8", "16"):
                with self.subTest(concurrency=conc):
                    info_c, plan_c = self._inspect_and_plan(
                        model_dir, env={"FLUX_CONCURRENCY": conc}
                    )
                    self.assertEqual(info_c["speculative"]["mode"], "dflash")
                    self.assertIn("--dflash", plan_c["command"])
                    self.assertEqual(_arg_value(plan_c["command"], "--kv-pool-tokens"), "393216")
                    self.assertEqual(plan_c["env"]["ATLAS_DFLASH_QUANTIZATION"], "nvfp4")
                    self.assertEqual(plan_c["env"]["ATLAS_DFLASH_NOISE_ONLY"], "1")
                    self.assertEqual(plan_c["env"]["ATLAS_DFLASH_NOISE_GRAPH"], "1")
                    self.assertEqual(plan_c["env"]["ATLAS_DISABLE_TREE_WY"], "1")
                    self.assertEqual(plan_c["env"]["ATLAS_MEMLEAN_DRAFTER_BF16"], "1")
                    self.assertEqual(plan_c["env"]["ATLAS_THINK_SPEC"], "1")
                    self.assertEqual(plan_c["env"]["ATLAS_MOE_SCALE_TS4E"], "1")
                    self.assertEqual(plan_c["env"]["ATLAS_SSM_DECODE_RING"], "0")
                    self.assertEqual(plan_c["env"]["FLUX_SSM_DECODE_RING"], "0")
                    self.assertEqual(plan_c["env"]["ATLAS_SPEC_VERIFY"], "rejection")
                    self.assertEqual(plan_c["env"]["ATLAS_QWEN4_DFLASH_REJECT_TEMP_SCALE"], "1.15")
                    expected_fp8_margin = "4608" if int(conc) >= 8 else "2048"
                    self.assertEqual(
                        plan_c["env"]["ATLAS_QWEN4_FP8_WEIGHT_CACHE_FLOOR_MARGIN_MIB"],
                        expected_fp8_margin,
                    )
                    self.assertEqual(plan_c["env"]["ATLAS_KV_ADMIT_WATERMARK"], "32768")
                    self.assertEqual(plan_c["env"]["FLUX_KV_ADMIT_WATERMARK"], "32768")
                    self.assertEqual(plan_c["env"]["ATLAS_QWEN4_BATCHED_K5"], "1")
                    self.assertEqual(plan_c["env"]["ATLAS_QWEN4_BATCHED_ADAPTIVE_K"], "1")

    def test_dynamic_uma_kv_pools_high_kv_and_multi_spark_2x_3x_4x_unlocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = self._write_flash_next_base(root)
            self._write_qualified_dflash2(root)

            def plan_with_hw(mem_gib, mode="auto", env=None):
                hw = {
                    "sm_family": "gb10-sm121a",
                    "hardware_recipe": "flux-nextgen-max-prefill-decode.env",
                    "unified_memory_architecture": True,
                    "hardware_default_seq_len": 131072,
                    "hardware_max_seq_len_fp8_kv": 1048576,
                    "mem_available_gib": mem_gib,
                }
                full_env = {"FLUX_MODE": mode}
                if env:
                    full_env.update(env)
                return self._inspect_and_plan(model_dir, env=full_env, hw=hw)

            # 1x Spark headless (118 GiB): 512K BF16, 1M mixed NVFP4 (high-kv), watermark 32768.
            _, plan_bf16_118 = plan_with_hw(118.0, env={"FLUX_CONCURRENCY": "16"})
            self.assertEqual(_arg_value(plan_bf16_118["command"], "--kv-pool-tokens"), "524288")
            self.assertEqual(plan_bf16_118["env"]["ATLAS_KV_ADMIT_WATERMARK"], "32768")
            self.assertIn("--dflash", plan_bf16_118["command"])

            _, plan_high_kv_118 = plan_with_hw(118.0, mode="high-kv", env={"FLUX_CONCURRENCY": "16"})
            self.assertEqual(plan_high_kv_118["kv_cache_dtype"], "nvfp4")
            self.assertEqual(_arg_value(plan_high_kv_118["command"], "--kv-pool-tokens"), "1048576")
            self.assertEqual(_arg_value(plan_high_kv_118["command"], "--kv-high-precision-layers"), "auto")
            self.assertEqual(plan_high_kv_118["env"]["ATLAS_KV_ADMIT_WATERMARK"], "32768")
            self.assertIn("--dflash", plan_high_kv_118["command"])

            # 1x Spark desktop (101 GiB): 384K BF16, 768K NVFP4.
            _, plan_bf16_101 = plan_with_hw(101.0, env={"FLUX_CONCURRENCY": "16"})
            self.assertEqual(_arg_value(plan_bf16_101["command"], "--kv-pool-tokens"), "393216")
            _, plan_nvfp4_101 = plan_with_hw(101.0, mode="high-kv", env={"FLUX_CONCURRENCY": "16"})
            self.assertEqual(_arg_value(plan_nvfp4_101["command"], "--kv-pool-tokens"), "786432")

            # 1x Spark tight host (90 GiB): 256K BF16, 65536 watermark (4 slots instead of 2).
            _, plan_bf16_90 = plan_with_hw(90.0, env={"FLUX_CONCURRENCY": "16"})
            self.assertEqual(_arg_value(plan_bf16_90["command"], "--kv-pool-tokens"), "262144")
            self.assertEqual(plan_bf16_90["env"]["ATLAS_KV_ADMIT_WATERMARK"], "65536")

            # 2x Spark TP=2 + EP=2 (PP=1): unlocks W=8 DFlash2 at C=1 and C=16, 4M BF16 / 7M NVFP4.
            _, plan_2x_c1 = plan_with_hw(118.0, mode="2x-spark")
            self.assertIn("--dflash", plan_2x_c1["command"])
            self.assertEqual(_arg_value(plan_2x_c1["command"], "--world-size"), "2")
            self.assertEqual(_arg_value(plan_2x_c1["command"], "--tp-size"), "2")
            self.assertEqual(_arg_value(plan_2x_c1["command"], "--ep-size"), "2")
            self.assertNotIn("--pp-size", plan_2x_c1["command"])
            self.assertEqual(plan_2x_c1["env"]["ATLAS_FAST_LOAD_ROUTED_GROUPS"], "0")

            _, plan_2x_c16 = plan_with_hw(118.0, mode="2x-spark", env={"FLUX_CONCURRENCY": "16"})
            self.assertIn("--dflash", plan_2x_c16["command"])
            self.assertEqual(_arg_value(plan_2x_c16["command"], "--kv-pool-tokens"), "4194304")
            self.assertEqual(plan_2x_c16["env"]["ATLAS_KV_ADMIT_WATERMARK"], "32768")

            _, plan_2x_nvfp4 = plan_with_hw(
                118.0,
                mode="2x-spark",
                env={"FLUX_CONCURRENCY": "16", "FLUX_KV_CACHE_DTYPE": "nvfp4"},
            )
            self.assertEqual(_arg_value(plan_2x_nvfp4["command"], "--kv-pool-tokens"), "7340032")

            # 2x Spark desktop (101 GiB) and tight (<98 GiB) UMA tiers.
            _, plan_2x_desktop = plan_with_hw(101.0, mode="2x-spark", env={"FLUX_CONCURRENCY": "16"})
            self.assertEqual(_arg_value(plan_2x_desktop["command"], "--kv-pool-tokens"), "2621440")
            _, plan_2x_tight = plan_with_hw(90.0, mode="2x-spark", env={"FLUX_CONCURRENCY": "16"})
            self.assertEqual(_arg_value(plan_2x_tight["command"], "--kv-pool-tokens"), "2097152")

            # 2x Spark PP=2 override: 4.5M BF16 / 8M NVFP4 KV, emits --pp-size 2.
            _, plan_2x_pp2 = plan_with_hw(
                118.0,
                mode="2x-spark",
                env={"FLUX_CONCURRENCY": "16", "FLUX_PP_SIZE": "2"},
            )
            self.assertEqual(_arg_value(plan_2x_pp2["command"], "--tp-size"), "1")
            self.assertEqual(_arg_value(plan_2x_pp2["command"], "--ep-size"), "1")
            self.assertEqual(_arg_value(plan_2x_pp2["command"], "--pp-size"), "2")
            self.assertEqual(_arg_value(plan_2x_pp2["command"], "--kv-pool-tokens"), "4718592")
            self.assertNotIn("--dflash", plan_2x_pp2["command"])
            self.assertIn("--speculative", plan_2x_pp2["command"])

            # 3x Spark PP=3: emits --pp-size 3, 7M BF16 / 12M NVFP4 KV, C=1..16 batched K=5 MTP.
            _, plan_3x_c1 = plan_with_hw(118.0, mode="3x-spark")
            self.assertEqual(_arg_value(plan_3x_c1["command"], "--world-size"), "3")
            self.assertEqual(_arg_value(plan_3x_c1["command"], "--pp-size"), "3")
            self.assertIn("--speculative", plan_3x_c1["command"])
            self.assertNotIn("--dflash", plan_3x_c1["command"])
            self.assertEqual(plan_3x_c1["env"]["ATLAS_FAST_LOAD_ROUTED_GROUPS"], "0")

            _, plan_3x_c16 = plan_with_hw(118.0, mode="3x-spark", env={"FLUX_CONCURRENCY": "16"})
            self.assertEqual(_arg_value(plan_3x_c16["command"], "--pp-size"), "3")
            self.assertEqual(_arg_value(plan_3x_c16["command"], "--kv-pool-tokens"), "7340032")
            self.assertNotIn("--dflash", plan_3x_c16["command"])
            self.assertIn("--speculative", plan_3x_c16["command"])

            _, plan_3x_nvfp4 = plan_with_hw(
                118.0,
                mode="3x-spark",
                env={"FLUX_CONCURRENCY": "16", "FLUX_KV_CACHE_DTYPE": "nvfp4"},
            )
            self.assertEqual(_arg_value(plan_3x_nvfp4["command"], "--kv-pool-tokens"), "12582912")

            _, plan_3x_desktop = plan_with_hw(101.0, mode="3x-spark", env={"FLUX_CONCURRENCY": "16"})
            self.assertEqual(_arg_value(plan_3x_desktop["command"], "--kv-pool-tokens"), "5242880")
            _, plan_3x_tight = plan_with_hw(90.0, mode="3x-spark", env={"FLUX_CONCURRENCY": "16"})
            self.assertEqual(_arg_value(plan_3x_tight["command"], "--kv-pool-tokens"), "3145728")

            # 4x Spark default TP=4 + EP=4 (PP=1): W=8 DFlash2 C=1..16, 5M BF16 / 10M NVFP4 KV.
            _, plan_4x_c16 = plan_with_hw(118.0, mode="4x-spark", env={"FLUX_CONCURRENCY": "16"})
            self.assertEqual(_arg_value(plan_4x_c16["command"], "--world-size"), "4")
            self.assertEqual(_arg_value(plan_4x_c16["command"], "--tp-size"), "4")
            self.assertEqual(_arg_value(plan_4x_c16["command"], "--ep-size"), "4")
            self.assertNotIn("--pp-size", plan_4x_c16["command"])
            self.assertIn("--dflash", plan_4x_c16["command"])
            self.assertEqual(_arg_value(plan_4x_c16["command"], "--kv-pool-tokens"), "5242880")
            self.assertEqual(plan_4x_c16["env"]["ATLAS_FAST_LOAD_ROUTED_GROUPS"], "0")

            _, plan_4x_nvfp4 = plan_with_hw(
                118.0,
                mode="4x-spark",
                env={"FLUX_CONCURRENCY": "16", "FLUX_KV_CACHE_DTYPE": "nvfp4"},
            )
            self.assertEqual(_arg_value(plan_4x_nvfp4["command"], "--kv-pool-tokens"), "10485760")

            _, plan_4x_desktop = plan_with_hw(101.0, mode="4x-spark", env={"FLUX_CONCURRENCY": "16"})
            self.assertEqual(_arg_value(plan_4x_desktop["command"], "--kv-pool-tokens"), "4194304")
            _, plan_4x_tight = plan_with_hw(90.0, mode="4x-spark", env={"FLUX_CONCURRENCY": "16"})
            self.assertEqual(_arg_value(plan_4x_tight["command"], "--kv-pool-tokens"), "3145728")

            # 4x Spark PP=4 override: 10M BF16 / 16M NVFP4 KV, TP=1 EP=1, batched K=5 MTP.
            _, plan_4x_pp4 = plan_with_hw(
                118.0,
                mode="4x-spark",
                env={"FLUX_CONCURRENCY": "16", "FLUX_PP_SIZE": "4"},
            )
            self.assertEqual(_arg_value(plan_4x_pp4["command"], "--tp-size"), "1")
            self.assertEqual(_arg_value(plan_4x_pp4["command"], "--ep-size"), "1")
            self.assertEqual(_arg_value(plan_4x_pp4["command"], "--pp-size"), "4")
            self.assertEqual(_arg_value(plan_4x_pp4["command"], "--kv-pool-tokens"), "10485760")
            self.assertNotIn("--dflash", plan_4x_pp4["command"])
            self.assertIn("--speculative", plan_4x_pp4["command"])

            _, plan_4x_pp4_nvfp4 = plan_with_hw(
                118.0,
                mode="4x-spark",
                env={"FLUX_CONCURRENCY": "16", "FLUX_PP_SIZE": "4", "FLUX_KV_CACHE_DTYPE": "nvfp4"},
            )
            self.assertEqual(_arg_value(plan_4x_pp4_nvfp4["command"], "--kv-pool-tokens"), "16777216")

            _, plan_4x_pp4_desktop = plan_with_hw(
                101.0,
                mode="4x-spark",
                env={"FLUX_CONCURRENCY": "16", "FLUX_PP_SIZE": "4"},
            )
            self.assertEqual(_arg_value(plan_4x_pp4_desktop["command"], "--kv-pool-tokens"), "6291456")
            _, plan_4x_pp4_tight = plan_with_hw(
                90.0,
                mode="4x-spark",
                env={"FLUX_CONCURRENCY": "16", "FLUX_PP_SIZE": "4"},
            )
            self.assertEqual(_arg_value(plan_4x_pp4_tight["command"], "--kv-pool-tokens"), "4194304")

            # Bare `--enable-prefix-caching` (arity 1) must flip the emitted
            # `--enable-prefix-caching false` default to `true`, matching
            # `--enable-prefix-caching true` and `--enable-prefix-caching=true`.
            hw_118 = {
                "sm_family": "gb10-sm121a",
                "hardware_recipe": "flux-nextgen-max-prefill-decode.env",
                "unified_memory_architecture": True,
                "hardware_default_seq_len": 131072,
                "hardware_max_seq_len_fp8_kv": 1048576,
                "mem_available_gib": 118.0,
            }
            pp4_env = {"FLUX_MODE": "4x-spark", "FLUX_PP_SIZE": "4", "FLUX_CONCURRENCY": "16"}
            _, plan_bare_pc = self._inspect_and_plan(
                model_dir, env=pp4_env, hw=hw_118, extra_args=["--enable-prefix-caching"]
            )
            self.assertEqual(_arg_value(plan_bare_pc["command"], "--enable-prefix-caching"), "true")
            _, plan_true_pc = self._inspect_and_plan(
                model_dir, env=pp4_env, hw=hw_118, extra_args=["--enable-prefix-caching", "true"]
            )
            self.assertEqual(_arg_value(plan_true_pc["command"], "--enable-prefix-caching"), "true")
            _, plan_eq_pc = self._inspect_and_plan(
                model_dir, env=pp4_env, hw=hw_118, extra_args=["--enable-prefix-caching=true"]
            )
            self.assertEqual(_arg_value(plan_eq_pc["command"], "--enable-prefix-caching"), "true")
            _, plan_false_pc = self._inspect_and_plan(
                model_dir, env=pp4_env, hw=hw_118, extra_args=["--enable-prefix-caching", "false"]
            )
            self.assertEqual(_arg_value(plan_false_pc["command"], "--enable-prefix-caching"), "false")

    def test_dflash2_fallback_rules_and_user_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = self._write_flash_next_base(root)
            self._write_unqualified_native_dflash2(root)
            self._write_inferact_mtp(root)

            # Unqualified native-dflash2 + Inferact-MTP never supersede in-checkpoint Native-MTP.
            info_unqual, plan_unqual = self._inspect_and_plan(model_dir)
            self.assertEqual(info_unqual["speculative"]["mode"], "native-mtp")
            self.assertIsNone(info_unqual["speculative"]["drafter_path"])
            self.assertIn("--speculative", plan_unqual["command"])

            # Now add the qualified incoai-dflash2-b8 companion.
            self._write_qualified_dflash2(root)

            # Explicit FLUX_SPEC_MODE=mtp / native-mtp selects Native-MTP (and 393216 pool at C=8).
            for spec_mode in ("mtp", "native-mtp"):
                with self.subTest(spec_mode=spec_mode):
                    info_mtp, plan_mtp = self._inspect_and_plan(
                        model_dir, env={"FLUX_SPEC_MODE": spec_mode, "FLUX_CONCURRENCY": "8"}
                    )
                    self.assertEqual(info_mtp["speculative"]["mode"], "native-mtp")
                    self.assertIn("--speculative", plan_mtp["command"])
                    self.assertNotIn("--dflash", plan_mtp["command"])
                    self.assertEqual(_arg_value(plan_mtp["command"], "--kv-pool-tokens"), "393216")
                    self.assertEqual(plan_mtp["env"]["ATLAS_SPEC_VERIFY"], "gumbel")

            # Concurrency > 16 falls back to Native-MTP (adaptive-K supports C=32..64).
            info_c32, plan_c32 = self._inspect_and_plan(
                model_dir, env={"FLUX_CONCURRENCY": "32"}
            )
            self.assertEqual(info_c32["speculative"]["mode"], "native-mtp")
            self.assertIn("--speculative", plan_c32["command"])
            self.assertNotIn("--dflash", plan_c32["command"])

            # FLUX_SPEC_MODE=none and FLUX_DRAFT_MODEL=none disable speculation cleanly.
            for disable_env in ({"FLUX_SPEC_MODE": "none"}, {"FLUX_DRAFT_MODEL": "none"}):
                with self.subTest(disable_env=disable_env):
                    info_off, plan_off = self._inspect_and_plan(model_dir, env=disable_env)
                    self.assertEqual(info_off["speculative"]["mode"], "none")
                    self.assertFalse(plan_off["speculative_enabled"])
                    self.assertNotIn("--dflash", plan_off["command"])
                    self.assertNotIn("--speculative", plan_off["command"])

            # Explicit user env overrides win for every DFlash2 lever.
            _, plan_ovr = self._inspect_and_plan(
                model_dir,
                env={
                    "FLUX_CONCURRENCY": "8",
                    "FLUX_DFLASH_QUANTIZATION": "fp8",
                    "ATLAS_DFLASH_NOISE_ONLY": "0",
                    "ATLAS_DFLASH_NOISE_GRAPH": "0",
                    "ATLAS_DISABLE_TREE_WY": "0",
                    "ATLAS_MEMLEAN_DRAFTER_BF16": "0",
                    "ATLAS_THINK_SPEC": "0",
                    "ATLAS_MOE_SCALE_TS4E": "0",
                    "ATLAS_SSM_DECODE_RING": "1",
                    "FLUX_SPEC_VERIFY": "gumbel",
                    "ATLAS_QWEN4_DFLASH_REJECT_TEMP_SCALE": "1.0",
                    "FLUX_QWEN4_FP8_WEIGHT_CACHE_FLOOR_MARGIN_MIB": "1024",
                    "FLUX_KV_POOL_TOKENS": "393216",
                },
            )
            self.assertEqual(plan_ovr["env"]["ATLAS_DFLASH_QUANTIZATION"], "fp8")
            self.assertEqual(plan_ovr["env"]["ATLAS_DFLASH_NOISE_ONLY"], "0")
            self.assertEqual(plan_ovr["env"]["ATLAS_DFLASH_NOISE_GRAPH"], "0")
            self.assertEqual(plan_ovr["env"]["ATLAS_DISABLE_TREE_WY"], "0")
            self.assertEqual(plan_ovr["env"]["ATLAS_MEMLEAN_DRAFTER_BF16"], "0")
            self.assertEqual(plan_ovr["env"]["ATLAS_THINK_SPEC"], "0")
            self.assertEqual(plan_ovr["env"]["ATLAS_MOE_SCALE_TS4E"], "0")
            self.assertEqual(plan_ovr["env"]["ATLAS_SSM_DECODE_RING"], "1")
            self.assertEqual(plan_ovr["env"]["ATLAS_SPEC_VERIFY"], "gumbel")
            self.assertEqual(plan_ovr["env"]["ATLAS_QWEN4_DFLASH_REJECT_TEMP_SCALE"], "1.0")
            self.assertEqual(plan_ovr["env"]["ATLAS_QWEN4_FP8_WEIGHT_CACHE_FLOOR_MARGIN_MIB"], "1024")
            self.assertEqual(_arg_value(plan_ovr["command"], "--kv-pool-tokens"), "393216")

            # Explicit CLI --dflash-quantization is not shadowed by an injected env var.
            _, plan_cli_q = self._inspect_and_plan(
                model_dir, extra_args=["--dflash-quantization", "bf16"]
            )
            self.assertNotIn("ATLAS_DFLASH_QUANTIZATION", plan_cli_q["env"])
            self.assertEqual(_arg_value(plan_cli_q["command"], "--dflash-quantization"), "bf16")

    def test_non_gb10_hopper_auto_disables_uma_ts4e_and_targets_sm90a(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = self._write_flash_next_base(root)
            self._write_qualified_dflash2(root)
            hopper_hw = {
                "sm_family": "hopper-sm90",
                "hardware_recipe": "nvidia-hopper-sm90.env",
                "unified_memory_architecture": False,
                "hardware_default_seq_len": 131072,
                "hardware_max_seq_len_fp8_kv": 1048576,
            }
            _, plan_h100 = self._inspect_and_plan(
                model_dir,
                env={"FLUX_TOPOLOGY": "sm90"},
                hw=hopper_hw,
            )
        # Non-UMA Hopper does not enable UMA-only TS4E scale compression.
        self.assertEqual(plan_h100["env"]["ATLAS_MOE_SCALE_TS4E"], "0")
        self.assertEqual(plan_h100["env"]["FLUX_ARCH_TARGET"], "sm_90a")

    def test_interactive_tty_enables_rust_tui_while_non_tty_keeps_no_tui(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = self._write_flash_next_base(root)
            with (
                mock.patch.object(flux_autodetect.sys.stdout, "isatty", return_value=True),
                mock.patch.object(flux_autodetect.sys.stdin, "isatty", return_value=True),
            ):
                _, plan_tty = self._inspect_and_plan(
                    model_dir, env={"TERM": "xterm-256color"}
                )
                _, plan_optout = self._inspect_and_plan(
                    model_dir, env={"TERM": "xterm-256color", "FLUX_NO_TUI": "1"}
                )
            with (
                mock.patch.object(flux_autodetect.sys.stdout, "isatty", return_value=False),
                mock.patch.object(flux_autodetect.sys.stdin, "isatty", return_value=False),
            ):
                _, plan_plain = self._inspect_and_plan(
                    model_dir, env={"TERM": "xterm-256color"}
                )
        self.assertNotIn("--no-tui", plan_tty["command"])
        self.assertIn("--no-tui", plan_optout["command"])
        self.assertIn("--no-tui", plan_plain["command"])

    def test_ple_ram_stage_c16_plan_tunes_128k_nvfp4_kv_and_override_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = self._write_flash_next_base(root, with_ple=True)
            self._write_qualified_dflash2(root)
            ple_dir = model_dir / "ple-offload"
            (ple_dir / "ple-ngram-nvfp4-000.bin").write_bytes(b"\x00" * 4096)
            (ple_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "format": "nvfp4",
                        "entries": [
                            {"shard": 0, "file": "ple-ngram-nvfp4-000.bin", "bytes": 4096}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            ram_base = root / "ram-stage"
            _, plan = self._inspect_and_plan(
                model_dir,
                env={
                    **self.CONTAINER_DEFAULT_ENV,
                    "FLUX_CONCURRENCY": "16",
                    "FLUX_PLE_RAM_STAGE": "1",
                    "FLUX_PLE_RAM_DIR": str(ram_base),
                },
            )
        ram_info = plan["ple_ram_stage"]
        self.assertIsNotNone(ram_info)
        self.assertTrue(ram_info["enabled"])
        self.assertTrue(ram_info["stage_dir"].startswith(str(ram_base) + "/"))
        self.assertEqual(ram_info["shard_count"], 1)
        self.assertEqual(plan["max_seq_len"], 131056)
        self.assertEqual(plan["kv_cache_dtype"], "nvfp4")
        self.assertEqual(plan["env"]["FLUX_KV_POOL_TOKENS"], "131056")
        self.assertEqual(plan["env"]["ATLAS_KV_ADMIT_WATERMARK"], "40960")
        self.assertEqual(plan["env"]["ATLAS_PLE_OFFLOAD_DIR"], ram_info["stage_dir"])
        self.assertEqual(plan["env"]["ATLAS_PLE_PAGE_CACHE"], "1")
        self.assertEqual(plan["env"]["ATLAS_PLE_PARKED_CACHE_MB"], "0")
        self.assertEqual(plan["env"]["ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_WEIGHT_CACHE"], "0")
        cmd = plan["command"]
        self.assertEqual(cmd[cmd.index("--kv-cache-dtype") + 1], "nvfp4")
        self.assertEqual(cmd[cmd.index("--kv-high-precision-layers") + 1], "auto")
        self.assertEqual(cmd[cmd.index("--max-seq-len") + 1], "131056")
        self.assertEqual(cmd[cmd.index("--kv-pool-tokens") + 1], "131056")
        self.assertEqual(cmd[cmd.index("--oom-guard-mb") + 1], "1024")

    def test_ple_ram_stage_honors_explicit_overrides_and_refuses_non_ple_or_bad_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = self._write_flash_next_base(root, with_ple=True)
            _, plan = self._inspect_and_plan(
                model_dir,
                env={
                    **self.CONTAINER_DEFAULT_ENV,
                    "FLUX_CONCURRENCY": "16",
                    "FLUX_PLE_RAM_STAGE": "1",
                    "FLUX_MAX_SEQ_LEN": "65536",
                    "FLUX_KV_POOL_TOKENS": "65536",
                    "FLUX_KV_CACHE_DTYPE": "bf16",
                    "ATLAS_PLE_PARKED_CACHE_MB": "256",
                },
                extra_args=["--oom-guard-mb", "2048"],
            )
            self.assertEqual(plan["max_seq_len"], 65536)
            self.assertEqual(plan["kv_cache_dtype"], "bf16")
            self.assertEqual(plan["env"]["ATLAS_PLE_PARKED_CACHE_MB"], "256")
            cmd = plan["command"]
            self.assertEqual(cmd[cmd.index("--oom-guard-mb") + 1], "2048")

            with self.assertRaises(ValueError):
                self._inspect_and_plan(
                    model_dir,
                    env={**self.CONTAINER_DEFAULT_ENV, "FLUX_PLE_RAM_STAGE": "maybe"},
                )

        with tempfile.TemporaryDirectory() as tmp:
            no_ple = self._write_flash_next_base(Path(tmp), with_ple=False)
            with self.assertRaises(ValueError):
                self._inspect_and_plan(
                    no_ple,
                    env={**self.CONTAINER_DEFAULT_ENV, "FLUX_PLE_RAM_STAGE": "1"},
                )

    def test_stage_ple_to_ram_is_idempotent_readonly_safe_and_repairs_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "ple-offload"
            src.mkdir()
            payload0 = b"shard-zero-bytes" * 64
            payload1 = b"shard-one-bytes!" * 32
            (src / "ple-ngram-nvfp4-000.bin").write_bytes(payload0)
            (src / "ple-ngram-nvfp4-001.bin").write_bytes(payload1)
            manifest = {
                "format": "nvfp4",
                "entries": [
                    {"shard": 0, "file": "ple-ngram-nvfp4-000.bin", "bytes": len(payload0)},
                    {"shard": 1, "file": "ple-ngram-nvfp4-001.bin", "bytes": len(payload1)},
                ],
            }
            (src / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            # Make source read-only to prove staging never mutates /models:ro
            os.chmod(src, 0o555)
            dst = root / "shm" / "abc123"
            try:
                reused1, cnt1, bytes1, _ = flux_autodetect._stage_ple_to_ram(src, dst)
                self.assertFalse(reused1)
                self.assertEqual(cnt1, 2)
                self.assertGreater(bytes1, len(payload0) + len(payload1))
                self.assertEqual((dst / "ple-ngram-nvfp4-000.bin").read_bytes(), payload0)
                self.assertEqual((dst / "ple-ngram-nvfp4-001.bin").read_bytes(), payload1)

                # Second call reuses without re-copying
                reused2, cnt2, bytes2, _ = flux_autodetect._stage_ple_to_ram(src, dst)
                self.assertTrue(reused2)
                self.assertEqual(cnt2, 2)
                self.assertEqual(bytes2, bytes1)

                # Truncated shard is detected and repaired
                (dst / "ple-ngram-nvfp4-001.bin").write_bytes(b"short")
                reused3, _, _, _ = flux_autodetect._stage_ple_to_ram(src, dst)
                self.assertFalse(reused3)
                self.assertEqual((dst / "ple-ngram-nvfp4-001.bin").read_bytes(), payload1)
            finally:
                os.chmod(src, 0o755)

            # Unsafe path traversal in manifest must be rejected
            bad_src = root / "bad-ple"
            bad_src.mkdir()
            (bad_src / "manifest.json").write_text(
                json.dumps({"entries": [{"shard": 0, "file": "../escape.bin"}]}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                flux_autodetect._stage_ple_to_ram(bad_src, root / "shm" / "bad")
