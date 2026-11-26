# SPDX-License-Identifier: AGPL-3.0-only
"""The launcher's drafter allow-list is the MODEL.toml [[dflash.profile]] table.

The engine compiles kernels/gb10/*/MODEL.toml [[dflash.profile]] into its
drafter admission. flux-autodetect reads the export in
recipes/drafter-profiles.json. These tests keep the export equal to MODEL.toml
and keep the allow-list equal to the literal it replaced (f95594dcc).
"""

import importlib.machinery
import importlib.util
import sys
import unittest
from unittest import mock
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT / "scripts"))
import drafter_profiles  # noqa: E402


def _load_flux_autodetect():
    loader = importlib.machinery.SourceFileLoader("flux_autodetect_profiles", str(KIT / "flux-autodetect"))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


flux_autodetect = _load_flux_autodetect()

# find_companion_drafter's hard-coded tuple before the table.
LEGACY_COMPANION_TARGETS = {
    "qwen3.8-flash-next",
    "qwen3.8-27b",
    "qwen3.6-27b",
    "qwen3.5-27b",
    "qwen3.6-35b-a3b",
    "qwen3.5-35b-a3b",
    "glm5.3-flash",
}


class DrafterProfileExportTests(unittest.TestCase):
    def test_non_object_export_turns_discovery_off(self):
        for document in (None, [], "invalid-root", 1):
            with (
                self.subTest(document=document),
                mock.patch.object(flux_autodetect, "_DRAFTER_PROFILES", None),
                mock.patch.object(flux_autodetect, "_safe_is_file", return_value=True),
                mock.patch.object(flux_autodetect, "_read_json", return_value=document),
            ):
                self.assertEqual(flux_autodetect.companion_drafter_targets(), [])

    def test_export_matches_model_toml(self):
        if not drafter_profiles.KERNELS.is_dir():
            self.skipTest("repro-kit exported without kernels/")
        self.assertEqual(
            drafter_profiles.EXPORT.read_text(encoding="utf-8"),
            drafter_profiles.render(drafter_profiles.collect()),
            "recipes/drafter-profiles.json is stale; run scripts/drafter_profiles.py",
        )

    def test_allow_list_is_unchanged_by_the_table(self):
        self.assertEqual(set(flux_autodetect.companion_drafter_targets()), LEGACY_COMPANION_TARGETS)

    def test_refusals_have_no_companion(self):
        targets = set(flux_autodetect.companion_drafter_targets())
        self.assertNotIn("gemma-4-31b", targets)
        self.assertNotIn("deepseek-v4.1", targets)
        glm = [r for r in flux_autodetect.load_drafter_profiles() if r["target_model"] == "glm5.3-flash"]
        self.assertTrue(glm)
        self.assertTrue(all(r["admission"] == "refused" for r in glm))

    def test_launcher_speculation_defaults_agree_with_the_table(self):
        defaults = {
            r["target_model"]: r["kind"]
            for r in flux_autodetect.load_drafter_profiles()
            if r["launcher_default"]
        }
        for target, profile in flux_autodetect.MULTI_MODEL_PROFILES.items():
            with self.subTest(target=target):
                if profile["speculation"] == "none":
                    self.assertNotIn(target, defaults)
                else:
                    self.assertEqual(defaults.get(target), profile["speculation"])
        for row in flux_autodetect.load_drafter_profiles():
            if row["launcher_default"]:
                self.assertTrue(row["qualified"], row)

    def test_flash_next_incoai_dflash2_is_graduated_and_launcher_default(self):
        rows = {
            r["name"]: r
            for r in flux_autodetect.load_drafter_profiles()
            if r["target_model"] == "qwen3.8-flash-next"
        }
        incoai = rows["incoai-dflash2-b8"]
        self.assertEqual(incoai["kind"], "dflash2")
        self.assertEqual(incoai["admission"], "quorum")
        self.assertTrue(incoai["qualified"])
        self.assertTrue(incoai["launcher_default"])
        self.assertEqual(incoai["capture_mode"], "qwen4_hc_mean")

        default_row = flux_autodetect.launcher_default_drafter_profile("qwen3.8-flash-next")
        self.assertIsNotNone(default_row)
        self.assertEqual(default_row["name"], "incoai-dflash2-b8")

        for experimental in ("native-v3", "native-dflash2", "pixelml-qwen3dspark"):
            self.assertFalse(rows[experimental]["qualified"], experimental)
            self.assertFalse(rows[experimental]["launcher_default"], experimental)

    def test_missing_export_turns_discovery_off(self):
        saved = flux_autodetect._DRAFTER_PROFILES
        saved_file = flux_autodetect.DRAFTER_PROFILES_FILE
        try:
            flux_autodetect._DRAFTER_PROFILES = None
            flux_autodetect.DRAFTER_PROFILES_FILE = "no-such-drafter-profiles.json"
            self.assertEqual(flux_autodetect.companion_drafter_targets(), [])
            self.assertIsNone(
                flux_autodetect.find_companion_drafter(Path("/nonexistent/model"), "qwen3.8-27b", 5120, 64)
            )
        finally:
            flux_autodetect._DRAFTER_PROFILES = saved
            flux_autodetect.DRAFTER_PROFILES_FILE = saved_file


if __name__ == "__main__":
    unittest.main()
