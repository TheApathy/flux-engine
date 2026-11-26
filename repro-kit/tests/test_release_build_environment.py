# SPDX-License-Identifier: AGPL-3.0-only
"""Source regression for the sealed native build's stub bypass protection."""

import shlex
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]


class ReleaseBuildEnvironmentTests(unittest.TestCase):
    def test_sealed_native_builder_unsets_every_skip_build_alias(self):
        source = (ROOT / "scripts/build-qwen38-dual-profiled.sh").read_text()
        invocation = source.split("setsid nice -n 19 env ", 1)[1].split(
            "timeout -s TERM", 1
        )[0]
        arguments = shlex.split(invocation.replace("\\\n", " "))
        removed = {
            arguments[index + 1]
            for index, value in enumerate(arguments[:-1])
            if value == "-u"
        }
        storage = (ROOT / "crates/spark-storage/build.rs").read_text()
        for alias in ("ATLAS_SKIP_BUILD", "SKIP_ATLAS_BUILD"):
            self.assertIn(f'truthy("{alias}")', storage)
            self.assertIn(alias, removed)
        self.assertIn("ATLAS_SKIP_NATIVE_BUILD", removed)

    def test_sealed_native_builder_does_not_inherit_an_unrecorded_bridge_arch(self):
        source = (ROOT / "scripts/build-qwen38-dual-profiled.sh").read_text()
        invocation = source.split("setsid nice -n 19 env ", 1)[1].split(
            "timeout -s TERM", 1
        )[0]
        arguments = shlex.split(invocation.replace("\\\n", " "))
        self.assertTrue(
            any(
                arguments[index : index + 2] == ["-u", "ATLAS_FI_GDN_ARCH"]
                for index in range(len(arguments) - 1)
            )
        )
        self.assertFalse(
            any(argument.startswith("ATLAS_FI_GDN_ARCH=") for argument in arguments)
        )
        bridge = (ROOT / "crates/spark-model/build.rs").read_text()
        self.assertIn('std::env::var("ATLAS_FI_GDN_ARCH")', bridge)
        self.assertIn('.join("HARDWARE.toml")', bridge)

    def test_image_does_not_mislabel_mixed_kernels_as_one_architecture(self):
        source = (ROOT / "repro-kit/Dockerfile").read_text()
        self.assertNotIn("com.theapathy.flux.arch=", source)
        self.assertNotIn("Native Rust/CUDA SM121a inference engine", source)


if __name__ == "__main__":
    unittest.main()
