# SPDX-License-Identifier: AGPL-3.0-only
"""The top-level recipes/ mirror must match repro-kit/recipes/ byte for byte.

The image, launcher (flux-autodetect), suite, gates and their tests read
repro-kit/recipes/. The host-side scripts/flux_env_bridge.sh and the operator
docs read recipes/. A recipe edited in one tree only (9.15.3.1 shipped
MTP_CONF=0.6 in repro-kit/recipes/ while recipes/ kept 0) gives host and
container different policies under the same mode name.
"""

from pathlib import Path
import unittest

KIT = Path(__file__).resolve().parents[1]
CANONICAL = KIT / "recipes"
MIRROR = KIT.parent / "recipes"
# Each tree keeps its own index page; only operating files are mirrored.
MIRRORED_SUFFIXES = (".env", ".argv", ".toml")


def operating_files(root):
    return {path.name for path in root.iterdir() if path.is_file() and path.suffix in MIRRORED_SUFFIXES}


class RecipeMirrorParityTests(unittest.TestCase):
    def setUp(self):
        if not MIRROR.is_dir():
            self.skipTest("repro-kit exported without the top-level recipes/ mirror")

    def test_both_trees_carry_the_same_operating_files(self):
        self.assertEqual(operating_files(MIRROR), operating_files(CANONICAL))
        stray = {path.name for path in MIRROR.iterdir()} - {path.name for path in CANONICAL.iterdir()}
        self.assertEqual(stray, set(), "recipes/ files with no repro-kit/recipes/ source")

    def test_every_mirrored_recipe_is_byte_identical(self):
        for name in sorted(operating_files(CANONICAL)):
            with self.subTest(recipe=name):
                self.assertEqual((MIRROR / name).read_bytes(), (CANONICAL / name).read_bytes(),
                                 "recipes/" + name + " differs from repro-kit/recipes/" + name
                                 + "; copy the repro-kit file over the mirror")


if __name__ == "__main__":
    unittest.main()
