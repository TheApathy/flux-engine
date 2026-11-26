# SPDX-License-Identifier: AGPL-3.0-only
"""FLUX_DFLASH_LOOKUP: the launcher half of the reviewed lookup-drafter opt-in."""

import contextlib
import importlib.machinery
import importlib.util
import io
import os
import re
from pathlib import Path
from unittest import TestCase, mock

SCRIPT = Path(__file__).resolve().parents[1] / "flux-autodetect"
RUST_SELECTOR = Path(__file__).resolve().parents[2] / "crates/spark-server/src/cli_dflash_lookup.rs"


def _load_flux_autodetect():
    loader = importlib.machinery.SourceFileLoader("flux_autodetect_dflash_lookup", str(SCRIPT))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


flux_autodetect = _load_flux_autodetect()


def _model_info(kernel_target="qwen3.8-27b"):
    if kernel_target == "qwen3.8-27b":
        speculative = {"mode": "dflash", "drafter_path": "/models/Qwen3.8-27B-DFlash2", "gamma": 7}
    else:
        speculative = {"mode": "none"}
    return {
        "kernel_target": kernel_target,
        "selected_binary": "/opt/flux/bin/flux-server-universal",
        "path": f"/models/{kernel_target}",
        "name": kernel_target,
        "ple_dir": None,
        "speculative": speculative,
        "max_position_embeddings": 262144,
        "recommended_tool_call_parser": "qwen3_coder",
    }


def _build(env=None, kernel_target="qwen3.8-27b", custom_env=None):
    stderr = io.StringIO()
    with (
        mock.patch.dict(os.environ, env or {}, clear=True),
        mock.patch.object(
            flux_autodetect,
            "detect_hardware_topology",
            return_value={"hardware_recipe": "flux-v9-max-prefill.env", "unified_memory_architecture": True},
        ),
        contextlib.redirect_stderr(stderr),
    ):
        plan = flux_autodetect.build_launch_plan(
            _model_info(kernel_target), "auto", [], custom_env=custom_env
        )
    return plan, stderr.getvalue()


class DflashLookupLauncherTests(TestCase):
    def test_default_launch_is_unchanged(self):
        plan, _ = _build()
        self.assertIn("--dflash", plan["command"])
        self.assertIsNone(plan["dflash_lookup"])
        self.assertEqual(plan["env"]["ATLAS_DFLASH_SAM"], "0")
        self.assertNotIn("ATLAS_DFLASH_LOOKUP", plan["env"])
        self.assertNotIn("FLUX_DFLASH_LOOKUP", plan["env"])
        self.assertNotIn("ATLAS_DFLASH_RETRIEVAL", plan["env"])

    def test_sam_maps_to_exactly_its_engine_control(self):
        for spelling in ("FLUX_DFLASH_LOOKUP", "ATLAS_DFLASH_LOOKUP"):
            with self.subTest(spelling=spelling):
                plan, _ = _build({spelling: "sam"})
                env = plan["env"]
                self.assertEqual(plan["dflash_lookup"], "sam")
                self.assertEqual(env["ATLAS_DFLASH_LOOKUP"], "sam")
                self.assertEqual(env["FLUX_DFLASH_LOOKUP"], "sam")
                for prefix in ("ATLAS_", "FLUX_"):
                    self.assertEqual(env[f"{prefix}DFLASH_SAM"], "1")
                    self.assertEqual(env[f"{prefix}DFLASH_RETRIEVAL"], "0")
                # Every other DFlash control stays at the sealed recipe value.
                self.assertEqual(env["ATLAS_DFLASH_RETR_WIDE"], "31")
                self.assertEqual(env["ATLAS_DFLASH_ASYNC"], "0")
                self.assertNotIn("ATLAS_DFLASH_SAM_ADAPTIVE", env)

    def test_retrieval_maps_to_the_fixed_window_matcher(self):
        plan, _ = _build({"FLUX_DFLASH_LOOKUP": "retrieval"})
        self.assertEqual(plan["dflash_lookup"], "retrieval")
        self.assertEqual(plan["env"]["ATLAS_DFLASH_RETRIEVAL"], "1")
        self.assertEqual(plan["env"]["ATLAS_DFLASH_SAM"], "0")

    def test_custom_selector_overrides_either_inherited_alias(self):
        for inherited in ("FLUX_DFLASH_LOOKUP", "ATLAS_DFLASH_LOOKUP"):
            for override in ("FLUX_DFLASH_LOOKUP", "ATLAS_DFLASH_LOOKUP"):
                for mode in ("off", "sam", "retrieval"):
                    with self.subTest(inherited=inherited, override=override, mode=mode):
                        plan, _ = _build({inherited: "sam"}, custom_env={override: mode})
                        self.assertEqual(plan["dflash_lookup"], None if mode == "off" else mode)
                        for prefix in ("FLUX_", "ATLAS_"):
                            self.assertEqual(plan["env"][prefix + "DFLASH_LOOKUP"], mode)
                            self.assertEqual(plan["env"][prefix + "DFLASH_SAM"], "1" if mode == "sam" else "0")
                            self.assertEqual(plan["env"][prefix + "DFLASH_RETRIEVAL"], "1" if mode == "retrieval" else "0")

    def test_explicit_off_pins_both_controls_off(self):
        plan, _ = _build({"FLUX_DFLASH_LOOKUP": "off"})
        self.assertIsNone(plan["dflash_lookup"])
        self.assertEqual(plan["env"]["ATLAS_DFLASH_SAM"], "0")
        self.assertEqual(plan["env"]["ATLAS_DFLASH_RETRIEVAL"], "0")

    def test_malformed_selector_is_refused(self):
        for value in ("1", "SAM", "on", "pld", ""):
            with self.subTest(value=value), self.assertRaises(ValueError) as caught:
                _build({"FLUX_DFLASH_LOOKUP": value})
            self.assertIn("FLUX_DFLASH_LOOKUP", str(caught.exception))

    def test_conflicting_operator_control_is_refused_not_overridden(self):
        with self.assertRaises(ValueError) as caught:
            _build({"FLUX_DFLASH_LOOKUP": "sam", "ATLAS_DFLASH_SAM": "0"})
        self.assertIn("ATLAS_DFLASH_SAM", str(caught.exception))
        with self.assertRaises(ValueError):
            _build({"FLUX_DFLASH_LOOKUP": "sam"}, custom_env={"FLUX_DFLASH_RETRIEVAL": "1"})
        with self.assertRaises(ValueError):
            _build({"ATLAS_DFLASH_SAM": "1"}, custom_env={"FLUX_DFLASH_LOOKUP": "off"})
        plan, _ = _build({"FLUX_DFLASH_LOOKUP": "sam", "ATLAS_DFLASH_SAM": "1"})
        self.assertEqual(plan["dflash_lookup"], "sam")

    def test_target_only_27b_keeps_lookup_off(self):
        plan, err = _build({"FLUX_DFLASH_LOOKUP": "sam", "FLUX_SPEC_MODE": "none"})
        self.assertNotIn("--dflash", plan["command"])
        self.assertIsNone(plan["dflash_lookup"])
        self.assertEqual(plan["env"]["ATLAS_DFLASH_LOOKUP"], "off")
        self.assertEqual(plan["env"]["ATLAS_DFLASH_SAM"], "0")
        self.assertIn("lookup drafting stays off", err)

    def test_flash_next_never_engages_lookup(self):
        baseline, _ = _build(kernel_target="qwen3.8-flash-next")
        plan, err = _build({"FLUX_DFLASH_LOOKUP": "sam"}, kernel_target="qwen3.8-flash-next")
        self.assertIsNone(plan["dflash_lookup"])
        self.assertEqual(plan["env"]["ATLAS_DFLASH_LOOKUP"], "off")
        self.assertEqual(plan["env"]["FLUX_DFLASH_LOOKUP"], "off")
        self.assertIn("lookup drafting stays off", err)
        lookup_keys = {"ATLAS_DFLASH_LOOKUP", "FLUX_DFLASH_LOOKUP"}
        self.assertEqual(
            {k: v for k, v in plan["env"].items() if k not in lookup_keys},
            baseline["env"],
        )
        self.assertEqual(plan["command"], baseline["command"])

    def test_launcher_modes_match_the_engine_selector(self):
        source = RUST_SELECTOR.read_text(encoding="utf-8")
        self.assertIn('LOOKUP_ENV: &str = "ATLAS_DFLASH_LOOKUP"', source)
        engine_modes = set(re.findall(r'Some\("([a-z]+)"\) => Ok\(Self::', source))
        engine_modes.add("off")
        self.assertEqual(engine_modes, set(flux_autodetect.DFLASH_LOOKUP_MODES))
        self.assertEqual(
            flux_autodetect.DFLASH_LOOKUP_MODES["sam"]["DFLASH_SAM"], "1"
        )
        self.assertEqual(
            flux_autodetect.DFLASH_LOOKUP_MODES["retrieval"]["DFLASH_RETRIEVAL"], "1"
        )
