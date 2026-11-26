# SPDX-License-Identifier: AGPL-3.0-only
"""image_labels.py: the build-time copy of the image's OCI labels."""

import importlib.machinery
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import TestCase

SCRIPT = Path(__file__).resolve().parents[1] / "image_labels.py"


def _load():
    loader = importlib.machinery.SourceFileLoader("flux_image_labels", str(SCRIPT))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


il = _load()

# The shape of the release build.sh --label set (values shortened).
PAIRS = [
    "com.theapathy.flux.version=9.15.6-rc1",
    "org.opencontainers.image.version=9.15.6-rc1",
    "org.opencontainers.image.revision=3620f27a9",
    "com.theapathy.flux.binary-sha256=" + "a" * 64,
    "com.theapathy.flux.performance.interactive.recipe-sha256=" + "b" * 64,
    "com.theapathy.flux.target=qwen3.8-flash-next,qwen3.8-27b",
    "com.theapathy.flux.changes=v0.9.15.6: x=y; FLUX_DEBUG=1 debug log",
]


class BuildLabelsTest(TestCase):
    def test_pairs_override_base_and_earlier_pairs_like_docker_label(self):
        base = {"org.opencontainers.image.version": "9.15.3", "maintainer": "NVIDIA CORPORATION"}
        labels = il.build_labels(PAIRS + ["com.theapathy.flux.target=last-wins"], base)
        self.assertEqual(labels["org.opencontainers.image.version"], "9.15.6-rc1")
        self.assertEqual(labels["maintainer"], "NVIDIA CORPORATION")
        self.assertEqual(labels["com.theapathy.flux.target"], "last-wins")
        # Only the first '=' splits; the value keeps the rest verbatim.
        self.assertEqual(labels["com.theapathy.flux.changes"], "v0.9.15.6: x=y; FLUX_DEBUG=1 debug log")
        self.assertEqual(list(labels), sorted(labels))

    def test_empty_value_is_kept(self):
        self.assertEqual(il.build_labels(["a.b="]), {"a.b": ""})

    def test_malformed_pairs_are_refused(self):
        for bad in ("no-equals", "=value", "bad key=v"):
            with self.assertRaises(il.LabelError, msg=bad):
                il.build_labels([bad])

    def test_credential_like_labels_are_refused(self):
        for bad in ("com.example.api-key=x", "com.example.token=x", "com.example.password=x",
                    "a.b=hf_PlantedHuggingFace0123", "a.b=Bearer abcDEF123456", "a.b=sk-ant-planted-0123456789",
                    "a.b=https://user:pw@example.com/x", "a.b=/home/alice/build"):
            with self.assertRaises(il.LabelError, msg=bad):
                il.build_labels([bad])
        with self.assertRaises(il.LabelError):
            il.build_labels([], {"com.example.secret": "x"})

    def test_sha_and_url_labels_are_not_mistaken_for_secrets(self):
        labels = il.build_labels(PAIRS + ["org.opencontainers.image.source=https://github.com/TheApathy/flux-engine"])
        self.assertEqual(len(labels), len(PAIRS) + 1)


class CliTest(TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def _run(self, *args, stdin=""):
        return subprocess.run([sys.executable, "-B", str(SCRIPT), *args], input=stdin, text=True,
                              capture_output=True, timeout=30)

    def test_writes_the_document_with_base_labels_from_stdin(self):
        out = self.dir / "ctx" / "IMAGE_LABELS.json"
        r = self._run("--out", str(out), "--base-labels-json", "-", *PAIRS,
                      stdin=json.dumps({"maintainer": "NVIDIA CORPORATION"}))
        self.assertEqual(r.returncode, 0, r.stderr)
        doc = json.loads(out.read_text())
        self.assertEqual(doc["schema"], "flux-image-labels-v1")
        self.assertEqual(doc["path"], "/opt/flux/IMAGE_LABELS.json")
        self.assertEqual(doc["labels"], il.build_labels(PAIRS, {"maintainer": "NVIDIA CORPORATION"}))
        self.assertEqual(out.stat().st_mode & 0o777, 0o644)
        self.assertEqual(list(out.parent.iterdir()), [out])

    def test_null_base_labels_mean_none(self):
        out = self.dir / "IMAGE_LABELS.json"
        r = self._run("--out", str(out), "--base-labels-json", "-", "a.b=c", stdin="null\n")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(json.loads(out.read_text())["labels"], {"a.b": "c"})

    def test_refused_label_fails_and_writes_nothing(self):
        out = self.dir / "IMAGE_LABELS.json"
        r = self._run("--out", str(out), "a.b=ghp_PlantedGitHub012345")
        self.assertEqual(r.returncode, 1)
        self.assertIn("credential", r.stderr)
        self.assertNotIn("ghp_PlantedGitHub012345", r.stderr)
        self.assertFalse(out.exists())
