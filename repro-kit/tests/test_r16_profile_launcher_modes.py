# SPDX-License-Identifier: AGPL-3.0-only
"""Run profile launchers against a local argv-only shim, never the engine."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
PROFILES = ROOT / "bench/qwen38-flash-next-gb10"


class ProfileLauncherModes(unittest.TestCase):
    def launch(self, launcher, mode, *, kill_switch=None, extra=()):
        source = (PROFILES / "serve.sh").read_text()
        symbols = source.split("REQUIRED_SYMBOLS=(", 1)[1].split("\n)", 1)[0]
        names = re.findall(r'"([^"\n]+)"', symbols)
        names = ["qwen3.8-flash-next" if name == "$KTARGET" else name for name in names]
        names.extend(
            re.findall(
                r'"(ATLAS_[A-Z0-9_]+)"',
                (PROFILES / "serve-fast-prefill.sh").read_text(),
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory)
            model = task_dir / "model"
            model.mkdir()
            shim = task_dir / "argv-only-engine"
            shim.write_text(
                "#!"
                + sys.executable
                + "\n"
                + "# "
                + " ".join(names)
                + "\n"
                + "import json, os, sys\n"
                + "print(json.dumps({'argv': sys.argv[1:], "
                + "'disabled': os.environ.get('ATLAS_DISABLE_SPECULATION')}))\n"
            )
            shim.chmod(0o700)
            environment = os.environ.copy()
            for key in ("MTP_DIR", "ATLAS_DISABLE_SPECULATION"):
                environment.pop(key, None)
            environment.update(MODEL_DIR=str(model), BIN=str(shim), RUNTIME_MODE=mode)
            if kill_switch is not None:
                environment["ATLAS_DISABLE_SPECULATION"] = kill_switch
            result = subprocess.run(
                ["bash", str(PROFILES / launcher), *extra],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=20,
                check=True,
            )
            return json.loads(result.stdout)

    def test_plain_profiles_omit_all_speculative_cli_selectors(self):
        for launcher in ("serve.sh", "serve-w4096.sh", "serve-fast-prefill.sh"):
            with self.subTest(launcher=launcher):
                captured = self.launch(launcher, "no-spec", kill_switch="0")
                self.assertEqual(captured["disabled"], "1")
                self.assertEqual(captured["argv"][0], "serve")
                for selector in (
                    "--no-speculative",
                    "--speculative",
                    "--mtp-from-path",
                    "--num-drafts",
                    "--mtp-vocab",
                    "--mtp-quantization",
                ):
                    self.assertNotIn(selector, captured["argv"])

    def test_native_profiles_retain_k5_cli_geometry(self):
        for launcher, chunk in (("serve.sh", "16384"), ("serve-w4096.sh", "4096")):
            with self.subTest(launcher=launcher):
                captured = self.launch(launcher, "mtp-k5")
                argv = captured["argv"]
                self.assertIn("--speculative", argv)
                self.assertNotIn("--no-speculative", argv)
                self.assertEqual(argv[argv.index("--num-drafts") + 1], "4")
                self.assertEqual(argv[argv.index("--max-prefill-tokens") + 1], chunk)
                self.assertIsNone(captured["disabled"])

    def test_native_launch_does_not_silently_override_explicit_kill_switch(self):
        captured = self.launch("serve.sh", "mtp-k5", kill_switch="1")
        self.assertEqual(captured["disabled"], "1")
        self.assertIn("--speculative", captured["argv"])

    def test_plain_kill_switch_still_protects_forwarded_speculative_override(self):
        for launcher in ("serve.sh", "serve-fast-prefill.sh"):
            with self.subTest(launcher=launcher):
                captured = self.launch(launcher, "no-spec", extra=("--speculative",))
                self.assertEqual(captured["disabled"], "1")
                self.assertEqual(captured["argv"].count("--speculative"), 1)


if __name__ == "__main__":
    unittest.main()
