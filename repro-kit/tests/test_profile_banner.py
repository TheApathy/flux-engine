# SPDX-License-Identifier: AGPL-3.0-only
"""Per-model startup banner: profile models never print the Flash-Next mode table."""

import contextlib
import importlib.machinery
import importlib.util
import io
import os
from pathlib import Path
from unittest import TestCase, mock

SCRIPT = Path(__file__).resolve().parents[1] / "flux-autodetect"


def _load_flux_autodetect():
    loader = importlib.machinery.SourceFileLoader("flux_autodetect_profile_banner", str(SCRIPT))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


flux_autodetect = _load_flux_autodetect()

FLASH_NEXT_CLAIMS = ("SELF-GUIDING RECIPE MODES", "K=5", "Native QSA 262K", "6,335 tok/s", "max-prefill")


def _render(kernel_target, speculative_enabled=False, spec=None):
    model_info = {
        "name": f"{kernel_target}-checkpoint",
        "quantization": "NVFP4",
        "num_hidden_layers": 48,
        "kernel_target": kernel_target,
        "speculative": spec or {"mode": "none"},
        "ple_dir": None,
    }
    plan = {
        "kernel_target": kernel_target,
        "binary": "/opt/flux/bin/flux-server-universal",
        "recipe_file": "model.env",
        "mode": "auto",
        "speculative_enabled": speculative_enabled,
        "concurrency": 1,
        "max_seq_len": 36864,
        "max_prefill_tokens": 16384,
        "kv_cache_dtype": "bf16",
        "port": 8899,
    }
    out = io.StringIO()
    with mock.patch.dict(os.environ, {"NO_COLOR": "1"}, clear=True), contextlib.redirect_stdout(out):
        flux_autodetect.render_startup_interstitial(model_info, plan, {})
    return out.getvalue()


class ProfileBannerTests(TestCase):
    def test_profile_models_print_their_own_banner(self):
        for target, profile in flux_autodetect.MULTI_MODEL_PROFILES.items():
            with self.subTest(target=target):
                text = _render(target)
                self.assertIn(f"{profile['label']} launch profile", text)
                self.assertIn("Speculation    : disabled (target-only)", text)
                for claim in FLASH_NEXT_CLAIMS:
                    self.assertNotIn(claim, text)

    def test_speculating_profile_names_its_drafter(self):
        text = _render(
            "qwen3.8-27b",
            speculative_enabled=True,
            spec={"mode": "dflash", "drafter_path": "/models/Qwen3.8-27B-DFlash2", "gamma": 7},
        )
        self.assertIn("Speculation    : dflash (drafter=Qwen3.8-27B-DFlash2)", text)

    def test_flash_next_keeps_the_mode_table(self):
        text = _render("qwen3.8-flash-next")
        self.assertIn("SELF-GUIDING RECIPE MODES", text)
        self.assertIn("Lossless K=5 MTP", text)

    def test_profile_mode_summary_has_no_flash_next_numbers(self):
        for profile in flux_autodetect.MULTI_MODEL_PROFILES.values():
            summary = flux_autodetect.profile_mode_summary(profile, False)
            self.assertIn(profile["recipe_file"], summary)
            self.assertNotIn("K=5", summary)
            self.assertNotIn("tok/s", summary)
