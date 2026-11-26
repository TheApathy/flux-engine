# SPDX-License-Identifier: AGPL-3.0-only
"""FLUX_LOAD_DEBUG: the launcher half of load-failure-only debug output."""

import importlib.machinery
import importlib.util
import io
import os
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest import TestCase, mock

SCRIPT = Path(__file__).resolve().parents[1] / "flux-autodetect"
RUST_CAPTURE = Path(__file__).resolve().parents[2] / "crates/spark-server/src/load_debug.rs"


def _load_flux_autodetect():
    loader = importlib.machinery.SourceFileLoader("flux_autodetect_load_debug", str(SCRIPT))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


fa = _load_flux_autodetect()


class ResolveLoadDebugEnvTest(TestCase):
    def test_default_is_auto_with_results_file(self):
        self.assertEqual(
            fa.resolve_load_debug_env({}, results_writable=True),
            {"FLUX_LOAD_DEBUG": "auto", "FLUX_LOAD_DEBUG_FILE": "/results/flux-load-debug.log"},
        )

    def test_no_file_when_results_not_writable(self):
        self.assertEqual(fa.resolve_load_debug_env({}, results_writable=False), {"FLUX_LOAD_DEBUG": "auto"})

    def test_explicit_file_and_mode_are_kept(self):
        env = {"FLUX_LOAD_DEBUG": " Always ", "FLUX_LOAD_DEBUG_FILE": "none"}
        self.assertEqual(fa.resolve_load_debug_env(env, results_writable=True), {"FLUX_LOAD_DEBUG": "always"})

    def test_off_sets_no_file(self):
        self.assertEqual(
            fa.resolve_load_debug_env({"FLUX_LOAD_DEBUG": "off"}, results_writable=True), {"FLUX_LOAD_DEBUG": "off"}
        )

    def test_rust_log_is_not_touched(self):
        out = fa.resolve_load_debug_env({"RUST_LOG": "warn"}, results_writable=True)
        self.assertNotIn("RUST_LOG", out)

    def test_bad_mode_fails_fast(self):
        with self.assertRaisesRegex(ValueError, "auto|always|off"):
            fa.resolve_load_debug_env({"FLUX_LOAD_DEBUG": "verbose"}, results_writable=True)

    def test_mode_keys_are_not_aliased_to_atlas(self):
        self.assertIn("FLUX_LOAD_DEBUG", fa._FLUX_META_ENV_KEYS)
        self.assertIn("FLUX_LOAD_DEBUG_FILE", fa._FLUX_META_ENV_KEYS)


class LoadDebugTailTest(TestCase):
    def _file(self, text):
        fh = tempfile.NamedTemporaryFile("w", suffix=".log", delete=False)
        fh.write(text)
        fh.close()
        self.addCleanup(os.unlink, fh.name)
        return fh.name

    def test_killed_load_prints_tail(self):
        path = self._file("".join(f"DEBUG line {i}\n" for i in range(300)))
        tail = fa.load_debug_tail(path)
        self.assertIn("last 200 of 300 lines", tail)
        self.assertIn("DEBUG line 299", tail)
        self.assertNotIn("DEBUG line 99\n", tail)
        self.assertIn(f"full log: {path}", tail)

    def test_server_dumped_itself(self):
        self.assertIsNone(fa.load_debug_tail(self._file("x\nflux-load-debug: dumped to stderr\n")))

    def test_load_completed_then_crashed(self):
        self.assertIsNone(fa.load_debug_tail(self._file("x\nflux-load-debug: load complete\n")))

    def test_missing_empty_or_disabled(self):
        self.assertIsNone(fa.load_debug_tail(None))
        self.assertIsNone(fa.load_debug_tail("none"))
        self.assertIsNone(fa.load_debug_tail("/nonexistent/flux-load-debug.log"))
        self.assertIsNone(fa.load_debug_tail(self._file("")))

    def test_markers_match_the_server(self):
        src = RUST_CAPTURE.read_text(encoding="utf-8")
        self.assertIn(f'"{fa.LOAD_DEBUG_DUMPED_MARKER}"', src)
        self.assertIn(f'"{fa.LOAD_DEBUG_LOADED_MARKER}"', src)


class LoadDebugHelpTest(TestCase):
    def test_env_help_documents_the_knob(self):
        buf = io.StringIO()
        with mock.patch.object(fa.sys, "argv", ["flux", "env"]), redirect_stdout(buf):
            self.assertEqual(fa.main(), 0)
        self.assertIn("FLUX_LOAD_DEBUG            Load-failure diagnostics: auto (default) | always | off", buf.getvalue())
        self.assertIn("FLUX_LOAD_DEBUG_FILE", buf.getvalue())
