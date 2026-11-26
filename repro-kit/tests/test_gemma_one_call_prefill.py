# SPDX-License-Identifier: AGPL-3.0-only
"""Gemma-4-31B launcher profile policy: single-call prefill budget and no prefix caching.

The `gemma-4-31b` profile in `flux-autodetect` pins `max_prefill_tokens = max_seq_len`
and `prefix_cache_supported = False`, refusing smaller prefill overrides and
prefix-caching flags at launch planning time.
"""

import importlib.machinery
import importlib.util
import os
from pathlib import Path
from unittest import TestCase, mock

SCRIPT = Path(__file__).resolve().parents[1] / "flux-autodetect"


def _load_flux_autodetect():
    loader = importlib.machinery.SourceFileLoader("flux_autodetect_gemma_prefill", str(SCRIPT))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


flux_autodetect = _load_flux_autodetect()


def _gemma_model_info():
    return {
        "kernel_target": "gemma-4-31b",
        "selected_binary": "/opt/flux/bin/flux-server-universal",
        "path": "/models/Gemma-4-31B-IT-NVFP4",
        "name": "Gemma-4-31B-IT-NVFP4",
        "ple_dir": None,
        "speculative": {"mode": "none"},
        "recommended_max_seq_len": 262144,
        "max_position_embeddings": 262144,
    }


def _build(env, extra_args=None):
    with (
        mock.patch.dict(os.environ, env, clear=True),
        mock.patch.object(
            flux_autodetect,
            "detect_hardware_topology",
            return_value={"hardware_recipe": "flux-v9-max-prefill.env", "unified_memory_architecture": True},
        ),
    ):
        return flux_autodetect.build_launch_plan(_gemma_model_info(), "auto", list(extra_args or []))


def _arg(command, flag):
    return int(command[command.index(flag) + 1])


class GemmaOneCallPrefillTests(TestCase):
    def test_profile_sizes_prefill_to_the_context_cap(self):
        self.assertEqual(flux_autodetect.MULTI_MODEL_PROFILES["gemma-4-31b"]["max_prefill_tokens"], "max_seq_len")
        plan = _build({})
        self.assertEqual(plan["max_seq_len"], 36864)
        self.assertEqual(plan["max_prefill_tokens"], 36864)
        self.assertEqual(_arg(plan["command"], "--max-prefill-tokens"), _arg(plan["command"], "--max-seq-len"))

    def test_a_smaller_context_cap_shrinks_the_prefill_budget_with_it(self):
        plan = _build({"FLUX_MAX_SEQ_LEN": "32768"})
        self.assertEqual(plan["max_seq_len"], 32768)
        self.assertEqual(plan["max_prefill_tokens"], 32768)

    def test_a_chunked_prefill_override_is_refused(self):
        with self.assertRaisesRegex(ValueError, "prefills a prompt in one call"):
            _build({"FLUX_MAX_PREFILL_TOKENS": "16384"})

    def test_prefix_caching_is_refused(self):
        # The gemma-4-31b launcher profile pins prefix_cache_supported = False.
        for value in ("1", "true", "yes"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "cannot serve with prefix caching"):
                _build({"FLUX_PREFIX_CACHE": value})
        with self.assertRaisesRegex(ValueError, "cannot serve with prefix caching"):
            _build({}, ["--enable-prefix-caching", "true"])
        plan = _build({"FLUX_PREFIX_CACHE": "0"})
        self.assertFalse(plan["prefix_caching"])
        self.assertEqual(plan["command"][plan["command"].index("--enable-prefix-caching") + 1], "false")

    def test_other_profiles_still_opt_in(self):
        self.assertTrue(flux_autodetect.MULTI_MODEL_PROFILES["qwen3.8-27b"].get("prefix_cache_supported", True))

    def test_prefix_cache_final_argv_is_validated(self):
        for args in (
            ["--enable-prefix-caching=true"],
            ["--enable-prefix-caching", "false", "--enable-prefix-caching", "true"],
            ["--enable-prefix-caching=false", "--enable-prefix-caching=true"],
        ):
            with self.subTest(args=args), self.assertRaisesRegex(ValueError, "cannot serve with prefix caching"):
                _build({}, args)
        for args in (
            ["--enable-prefix-caching=false"],
            ["--enable-prefix-caching", "true", "--enable-prefix-caching", "false"],
        ):
            plan = _build({}, args)
            self.assertEqual(plan["command"][plan["command"].index("--enable-prefix-caching") + 1], "false")
