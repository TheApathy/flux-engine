# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only checks for build-time and generated release metadata."""

import argparse
import importlib.machinery
import importlib.util
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


KIT_DIR = Path(__file__).resolve().parents[1]
STABLE_VERSION = "1.0.3"


def _load_script(module_name: str, script_name: str):
    script = KIT_DIR / script_name
    loader = importlib.machinery.SourceFileLoader(module_name, str(script))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:
        raise RuntimeError(f"could not import {script}")
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


FLUX = _load_script("release_metadata_flux", "flux-autodetect")
BENCH = _load_script("release_metadata_bench", "flux-bench")
LIL = _load_script("release_metadata_lil", "lil-bench")


class DockerReleaseMetadataTests(unittest.TestCase):
    def test_build_arg_drives_labels_and_runtime_environment(self):
        dockerfile = (KIT_DIR / "Dockerfile").read_text(encoding="utf-8")

        self.assertIn(f'ARG FLUX_ENGINE_VERSION="{STABLE_VERSION}"', dockerfile)
        self.assertIn(f'org.opencontainers.image.ref.name="{STABLE_VERSION}"', dockerfile)
        for argument in (
            "FLUX_SOURCE_REVISION",
            "FLUX_SOURCE_TREE",
            "FLUX_BINARY_SHA256",
            "FLUX_RECIPE_SHA256",
        ):
            self.assertIn(f"ARG {argument}", dockerfile.splitlines())
            self.assertNotIn(f"ARG {argument}=", dockerfile)
        self.assertIn(
            'org.opencontainers.image.version="${FLUX_ENGINE_VERSION}"', dockerfile
        )
        self.assertIn(
            'org.opencontainers.image.revision="${FLUX_SOURCE_REVISION}"', dockerfile
        )
        self.assertIn('com.theapathy.flux.version="${FLUX_ENGINE_VERSION}"', dockerfile)
        self.assertIn(
            'com.theapathy.flux.source-tree="${FLUX_SOURCE_TREE}"', dockerfile
        )
        self.assertIn(
            'com.theapathy.flux.binary-sha256="${FLUX_BINARY_SHA256}"', dockerfile
        )
        self.assertIn(
            'com.theapathy.flux.recipe-sha256="${FLUX_RECIPE_SHA256}"', dockerfile
        )
        self.assertIn('FLUX_ENGINE_VERSION="${FLUX_ENGINE_VERSION}"', dockerfile)
        self.assertIn('FLUX_MAX_SEQ_LEN="131072"', dockerfile)
        self.assertNotIn(
            f'org.opencontainers.image.version="{STABLE_VERSION}"', dockerfile
        )


class DynamicReleaseMetadataTests(unittest.TestCase):
    def test_tools_use_stable_fallback_and_runtime_override(self):
        for module in (FLUX, BENCH, LIL):
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertEqual(module.engine_version(), STABLE_VERSION)
            with mock.patch.dict(
                os.environ, {"FLUX_ENGINE_VERSION": "0.9.13-r9"}, clear=True
            ):
                self.assertEqual(module.engine_version(), "0.9.13-r9")
            with mock.patch.dict(
                os.environ, {"FLUX_ENGINE_VERSION": "   "}, clear=True
            ):
                self.assertEqual(module.engine_version(), STABLE_VERSION)

    def test_flux_version_command_reports_built_version(self):
        with (
            mock.patch.dict(
                os.environ, {"FLUX_ENGINE_VERSION": "0.9.13-r9"}, clear=True
            ),
            mock.patch.object(FLUX.sys, "argv", ["flux", "--version"]),
            redirect_stdout(io.StringIO()) as stdout,
        ):
            self.assertEqual(FLUX.main(), 0)

        self.assertEqual(stdout.getvalue().strip(), "flux-engine v0.9.13-r9")

    def test_flux_bench_launch_banner_reports_built_version(self):
        hardware = {
            "gpu": {"topology": "1x-spark", "target_id": "nvidia_gb10_sm121a"},
            "cpu": {"recommended_affinity": ""},
            "auto_selected_profile": {"mode": "max-prefill"},
        }
        args = argparse.Namespace(
            topology="1x-spark", mode="max-prefill", dry_run=True
        )

        with (
            mock.patch.dict(
                os.environ, {"FLUX_ENGINE_VERSION": "0.9.13-r9"}, clear=True
            ),
            mock.patch.object(BENCH, "detect_hardware", return_value=hardware),
            mock.patch.object(BENCH, "parse_env_file", return_value={}),
            mock.patch.object(BENCH, "parse_argv_file", return_value=["--version"]),
            mock.patch.object(BENCH.shutil, "which", return_value=None),
            redirect_stdout(io.StringIO()) as stdout,
        ):
            self.assertEqual(BENCH.cmd_serve(args), 0)

        self.assertIn(
            "Flux Engine Server Launch Plan (v0.9.13-r9)", stdout.getvalue()
        )

    def test_lil_bench_cards_report_built_version(self):
        fixture = {
            "model": "fixture-model",
            "benchmark_version": "0.7.9",
            "hardware": {
                "gpus": [{"name": "fixture-gpu", "memory_total_mb": 1024}],
                "driver_version": "fixture-driver",
            },
            "prefill_Load_Sweep": [],
            "results": [],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result_path = root / "run.json"
            result_path.write_text(json.dumps(fixture), encoding="utf-8")
            with (
                mock.patch.dict(
                    os.environ, {"FLUX_ENGINE_VERSION": "0.9.13-r9"}, clear=True
                ),
                mock.patch.object(LIL.shutil, "which", return_value=None),
            ):
                generated = LIL.generate_cards(
                    root / "card", [result_path], {"host": False, "logs": False}
                )

            markdown = Path(generated["markdown"]).read_text(encoding="utf-8")
            svg = Path(generated["svg"]).read_text(encoding="utf-8")

        self.assertIn("flux-engine v0.9.13-r9", markdown)
        self.assertIn("Flux Engine v0.9.13-r9", svg)


if __name__ == "__main__":
    unittest.main()
