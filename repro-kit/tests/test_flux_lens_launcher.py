# SPDX-License-Identifier: AGPL-3.0-only
"""Offline checks for the original exact-token trial CLI entry point."""

import importlib.machinery
import importlib.util
import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import sys
import unittest
from unittest import mock

KIT = Path(__file__).resolve().parents[1]
LOADER = importlib.machinery.SourceFileLoader(
    "flux_lens_launcher", str(KIT / "flux-autodetect")
)
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
LAUNCHER = importlib.util.module_from_spec(SPEC)
LOADER.exec_module(LAUNCHER)


class FluxLensLauncherTests(unittest.TestCase):
    def invoke(self, arguments, program, result=0):
        with (
            mock.patch.object(sys, "argv", ["flux", "lens", *arguments]),
            mock.patch.object(LAUNCHER, "_sanitize_cuda_ld_library_path"),
            mock.patch.object(LAUNCHER, "_find_lens_script", return_value=program),
            mock.patch.object(LAUNCHER.subprocess, "call", return_value=result) as call,
            mock.patch.object(LAUNCHER, "resolve_model_query") as discover,
        ):
            code = LAUNCHER.main()
        discover.assert_not_called()
        return code, call

    def test_forwards_arguments_as_tokens_without_a_shell(self):
        arguments = ["run", "--plan", "/private/trial plan.json", "--output", "trial"]
        program = Path("/usr/local/bin/flux-lens")
        code, call = self.invoke(arguments, program, result=7)
        self.assertEqual(code, 7)
        call.assert_called_once_with([sys.executable, str(program), *arguments])

    def test_bare_command_only_shows_help(self):
        program = KIT / "flux_lens.py"
        code, call = self.invoke([], program)
        self.assertEqual(code, 0)
        call.assert_called_once_with([sys.executable, str(program), "--help"])

    def test_missing_tool_does_not_launch_a_model_or_another_benchmark(self):
        error = io.StringIO()
        with redirect_stderr(error):
            code, call = self.invoke(["plan"], None)
        self.assertEqual(code, 1)
        self.assertIn("Flux Lens is not installed", error.getvalue())
        call.assert_not_called()

    def test_checkout_local_script_precedes_installed_program(self):
        with mock.patch.object(LAUNCHER, "_safe_is_file", return_value=True) as is_file:
            self.assertEqual(LAUNCHER._find_lens_script(), KIT / "flux_lens.py")
        is_file.assert_called_once_with(KIT / "flux_lens.py")

    def test_installed_script_is_resolved_without_path_search(self):
        with mock.patch.object(
            LAUNCHER, "_safe_is_file", side_effect=[False, True]
        ) as is_file:
            self.assertEqual(
                LAUNCHER._find_lens_script(), Path("/usr/local/bin/flux-lens")
            )
        self.assertEqual(is_file.call_count, 2)

    def test_help_exposes_original_cli(self):
        output = io.StringIO()
        with redirect_stdout(output):
            LAUNCHER.print_help()
        self.assertIn("lens [args...]", output.getvalue())

    def test_image_installs_code_and_documentation(self):
        dockerfile = (KIT / "Dockerfile").read_text()
        self.assertIn("COPY flux_lens.py /usr/local/bin/flux-lens", dockerfile)
        self.assertIn("COPY FLUX_LENS.md /opt/flux/repro-kit/FLUX_LENS.md", dockerfile)
        self.assertIn("        /usr/local/bin/flux-lens \\", dockerfile)


if __name__ == "__main__":
    unittest.main()
