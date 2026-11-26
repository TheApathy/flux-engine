# SPDX-License-Identifier: AGPL-3.0-only
"""GLM-5.3 auto-detected launches run target-only: no drafter search and no --dflash.

`flux-autodetect` skips companion drafter discovery and refuses `FLUX_DRAFT_MODEL`
for `glm5_next`, because `serve_load` refuses `--speculative` and restricts
`--dflash` to explicit EXL3 runs with `--max-seq-len <= 2048` rather than generic
auto-detected launches.
"""

import importlib.machinery
import importlib.util
import json
import os
import tempfile
from pathlib import Path
from unittest import TestCase, mock

SCRIPT = Path(__file__).resolve().parents[1] / "flux-autodetect"


def _load_flux_autodetect():
    loader = importlib.machinery.SourceFileLoader("flux_autodetect_glm_target_only", str(SCRIPT))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


flux_autodetect = _load_flux_autodetect()


def _write_model(directory: Path, config: dict) -> None:
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(config))
    (directory / "model.safetensors").write_bytes(b"x")


class GlmTargetOnlyLaunchTests(TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.model = root / "GLM-5.3-Flash-EXL3"
        _write_model(
            self.model,
            {
                "model_type": "glm5_next",
                "hidden_size": 4096,
                "num_hidden_layers": 46,
                "quantization_config": {"quant_method": "exl3", "bits": 2.05},
            },
        )
        # Passes every discovery filter: hidden 4096, max tap 42 < 46, "glm" in the name.
        _write_model(
            root / "GLM-5.3-Flash-DFlash2-bf582e4",
            {
                "architectures": ["DFlashDraftModel"],
                "hidden_size": 4096,
                "block_size": 8,
                "dflash_config": {"target_layer_ids": [5, 14, 24, 33, 42]},
            },
        )

    def tearDown(self):
        self._tmp.cleanup()

    def _inspect(self, spec_mode):
        env = {"FLUX_SPEC_MODE": spec_mode} if spec_mode else {}
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch.object(
                flux_autodetect,
                "find_companion_drafter",
                side_effect=AssertionError("GLM must not search for a drafter"),
            ),
        ):
            return flux_autodetect.inspect_model(self.model)

    def test_no_drafter_search_and_no_dflash_flag(self):
        for spec_mode in ("", "dflash", "mtp"):
            with self.subTest(spec_mode=spec_mode):
                info = self._inspect(spec_mode)
                self.assertEqual(info["kernel_target"], "glm5.3-flash")
                self.assertEqual(info["speculative"]["mode"], "none")
                env = {"FLUX_SPEC_MODE": spec_mode} if spec_mode else {}
                with (
                    mock.patch.dict(os.environ, env, clear=True),
                    mock.patch.object(
                        flux_autodetect,
                        "detect_hardware_topology",
                        return_value={"hardware_recipe": "flux-v9-max-prefill.env", "unified_memory_architecture": True},
                    ),
                    mock.patch.object(flux_autodetect, "_load_recipe_env", return_value={}),
                ):
                    plan = flux_autodetect.build_launch_plan(info, "auto", [])
                self.assertNotIn("--dflash", plan["command"])
                self.assertNotIn("--speculative", plan["command"])
                self.assertFalse(plan["speculative_enabled"])

    def test_requested_speculation_is_reported_as_ignored(self):
        self.assertFalse(any("target-only" in w for w in self._inspect("").get("launch_warnings", [])))
        warnings = self._inspect("dflash").get("launch_warnings", [])
        self.assertIn("GLM-5.3 serves target-only; FLUX_SPEC_MODE=dflash is ignored.", warnings)

    def test_explicit_drafter_is_refused(self):
        info = self._inspect("")
        with (
            mock.patch.dict(os.environ, {"FLUX_DRAFT_MODEL": "/models/GLM-5.3-Flash-DFlash2-bf582e4"}, clear=True),
            mock.patch.object(
                flux_autodetect,
                "detect_hardware_topology",
                return_value={"hardware_recipe": "flux-v9-max-prefill.env", "unified_memory_architecture": True},
            ),
            mock.patch.object(flux_autodetect, "_load_recipe_env", return_value={}),
        ):
            with self.assertRaisesRegex(ValueError, "serves target-only"):
                flux_autodetect.build_launch_plan(info, "auto", [])
