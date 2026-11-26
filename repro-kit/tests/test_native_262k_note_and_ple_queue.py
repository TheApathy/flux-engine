# SPDX-License-Identifier: AGPL-3.0-only
"""FLUX_MODE=256k announces its BF16 correctness profile; the PLE queue-depth
memlock fallback sets both spellings."""

import contextlib
import importlib.machinery
import importlib.util
import io
import os
import resource
from pathlib import Path
from unittest import TestCase, mock

SCRIPT = Path(__file__).resolve().parents[1] / "flux-autodetect"


def _load_flux_autodetect():
    loader = importlib.machinery.SourceFileLoader("flux_autodetect_native_262k_note", str(SCRIPT))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


flux_autodetect = _load_flux_autodetect()


def _flash_next(ple_dir=None):
    return {
        "kernel_target": "qwen3.8-flash-next",
        "selected_binary": "/opt/flux/bin/flux-server-real",
        "path": "/models/qwen38-flash-next",
        "name": "qwen38-flash-next",
        "ple_dir": ple_dir,
        "speculative": {"mode": "none"},
        "requires_qwen4_qsa": True,
        "recommended_max_seq_len": 262144,
        "max_position_embeddings": 262144,
        "recommended_tool_call_parser": "qwen3_coder",
    }


def _build(mode, model_info, recipe_env=None, getrlimit=None):
    stderr = io.StringIO()
    patches = [
        mock.patch.dict(os.environ, {}, clear=True),
        mock.patch.object(
            flux_autodetect,
            "detect_hardware_topology",
            return_value={"hardware_recipe": "flux-nextgen-max-prefill-decode.env", "unified_memory_architecture": True},
        ),
        mock.patch.object(flux_autodetect, "_load_recipe_env", return_value=dict(recipe_env or {})),
    ]
    if getrlimit is not None:
        patches.append(mock.patch.object(resource, "getrlimit", return_value=getrlimit))
        patches.append(mock.patch.object(resource, "setrlimit", side_effect=PermissionError))
    with contextlib.ExitStack() as stack:
        for patch in patches:
            stack.enter_context(patch)
        stack.enter_context(contextlib.redirect_stderr(stderr))
        plan = flux_autodetect.build_launch_plan(model_info, mode, [])
    return plan, stderr.getvalue()


class Native262kNoteTests(TestCase):
    def setUp(self):
        flux_autodetect._NATIVE_262K_NOTE_PRINTED = False

    def test_256k_logs_the_correctness_profile_once(self):
        plan, stderr = _build("256k", _flash_next())
        self.assertEqual(plan["env"]["ATLAS_QWEN4_PREFILL_QSA_FP8"], "0")
        self.assertEqual(plan["env"]["ATLAS_QWEN4_QSA_VERIFY_ROWS"], "0")
        self.assertEqual(stderr.count(flux_autodetect.NATIVE_262K_PROFILE_NOTE), 1)
        self.assertIn("[flux] INFO: FLUX_MODE=256k selected the native-262K BF16 correctness profile", stderr)
        self.assertIn("Use FLUX_MODE=auto (135K context)", stderr)

    def test_auto_does_not_log_it(self):
        _, stderr = _build("auto", _flash_next())
        self.assertNotIn("native-262K", stderr)

    def _serve_then_diagnose(self, mode):
        """`flux serve` with /results writable: main builds the plan, then
        collect_diagnostics builds it again for flux-diagnostics.json."""
        stderr = io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.dict(os.environ, {"FLUX_MODE": mode}, clear=True))
            stack.enter_context(mock.patch.object(
                flux_autodetect,
                "detect_hardware_topology",
                return_value={"hardware_recipe": "flux-nextgen-max-prefill-decode.env", "unified_memory_architecture": True},
            ))
            stack.enter_context(mock.patch.object(flux_autodetect, "_load_recipe_env", return_value={}))
            stack.enter_context(mock.patch.object(flux_autodetect, "scan_models_root", return_value=[]))
            stack.enter_context(mock.patch.object(flux_autodetect, "select_best_model", return_value=_flash_next()))
            stack.enter_context(mock.patch.object(flux_autodetect, "_check_io_uring_support", return_value=(True, "ok")))
            stack.enter_context(mock.patch.object(flux_autodetect.subprocess, "check_output", side_effect=OSError))
            stack.enter_context(mock.patch.object(flux_autodetect.shutil, "which", return_value=None))
            stack.enter_context(mock.patch.object(flux_autodetect.urllib.request, "urlopen", side_effect=OSError))
            stack.enter_context(contextlib.redirect_stderr(stderr))
            flux_autodetect.build_launch_plan(_flash_next(), mode, [])
            diagnostics = flux_autodetect.collect_diagnostics({})
        self.assertIsNotNone(diagnostics)
        return stderr.getvalue()

    def test_serve_plus_diagnostics_prints_the_note_once(self):
        stderr = self._serve_then_diagnose("256k")
        self.assertEqual(stderr.count(flux_autodetect.NATIVE_262K_PROFILE_NOTE), 1, stderr)

    def test_serve_plus_diagnostics_in_auto_prints_no_note(self):
        stderr = self._serve_then_diagnose("auto")
        self.assertEqual(stderr.count("native-262K"), 0, stderr)


class PleQueueDepthTests(TestCase):
    RECIPE = {"ATLAS_QWEN4_PLE_IO_QUEUE_DEPTH": "256", "FLUX_QWEN4_PLE_IO_QUEUE_DEPTH": "256"}

    def test_low_memlock_sets_both_spellings(self):
        plan, _ = _build("auto", _flash_next("/models/qwen38-flash-next/ple-offload"), self.RECIPE, (8 << 20, 8 << 20))
        self.assertEqual(plan["env"]["ATLAS_QWEN4_PLE_IO_QUEUE_DEPTH"], "32")
        self.assertEqual(plan["env"]["FLUX_QWEN4_PLE_IO_QUEUE_DEPTH"], "32")

    def test_unlimited_memlock_keeps_the_recipe_depth(self):
        unlimited = (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
        plan, _ = _build("auto", _flash_next("/models/qwen38-flash-next/ple-offload"), self.RECIPE, unlimited)
        self.assertEqual(plan["env"]["ATLAS_QWEN4_PLE_IO_QUEUE_DEPTH"], "256")
        self.assertEqual(plan["env"]["FLUX_QWEN4_PLE_IO_QUEUE_DEPTH"], "256")
