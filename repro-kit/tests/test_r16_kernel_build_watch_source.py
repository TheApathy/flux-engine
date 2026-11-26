# SPDX-License-Identifier: AGPL-3.0-only
"""Static guards only: these do not execute Rust, Cargo, or CUDA compilation."""

import re
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT / "crates/atlas-kernels/build.rs").read_text()


class KernelBuildWatchSourceTests(unittest.TestCase):
    def test_model_and_common_directories_use_the_existing_ancestor_helper(self):
        watch_sites = SOURCE.split("all_target_modules.push(modules);", 1)[1].split(
            "let n_overrides", 1
        )[0]
        self.assertIn("hw_dir.join(&target.model).join(&target.quant),", watch_sites)
        self.assertIn("hw_dir.join(&target.quant),", watch_sites)
        self.assertIn('hw_dir.join("common"),', watch_sites)
        self.assertIn("nearest_existing_path(&candidate)", watch_sites)
        self.assertIn('println!("cargo:rerun-if-changed={}", dir.display());', watch_sites)

    def test_helper_emits_only_an_existing_path_and_walks_missing_ancestors(self):
        ancestor = SOURCE.split("fn nearest_existing_path(", 1)[1].split(
            "\nfn rerun_if_changed(", 1
        )[0]
        self.assertIn("let mut watched = path;", ancestor)
        self.assertIn("loop {", ancestor)
        self.assertIn("if watched.exists()", ancestor)
        self.assertIn("return Some(watched.to_path_buf());", ancestor)
        self.assertIn("watched = watched.parent()?;", ancestor)
        helper = SOURCE.split("fn rerun_if_changed(", 1)[1].split("\nfn main()", 1)[0]
        self.assertRegex(
            helper,
            r"if let Some\(watched\) = nearest_existing_path\(path\)\s*\{\s*"
            r'println!\("cargo:rerun-if-changed=\{\}", watched\.display\(\)\);'
            r"\s*\}",
        )
        self.assertEqual(helper.count("println!"), 1)

    def test_common_only_target_resolution_remains_supported(self):
        self.assertIn("hw_dir.join(model).join(quant).is_dir()", SOURCE)
        self.assertIn("|| hw_dir.join(quant).is_dir()", SOURCE)
        self.assertIn('|| hw_dir.join("common").is_dir()', SOURCE)

    def test_existing_source_and_hardware_file_watchers_are_preserved(self):
        self.assertIn('println!("cargo:rerun-if-changed={}", cu_file.display());', SOURCE)
        self.assertIn("for toml_path in &hw_chain.hw_toml_paths {", SOURCE)
        self.assertIn('println!("cargo:rerun-if-changed={}", toml_path.display());', SOURCE)

    def test_all_target_selection_environment_aliases_still_invalidate_the_build(self):
        for namespace in ("ATLAS", "FLUX"):
            for selector in ("HW", "MODEL", "QUANT"):
                name = f"{namespace}_TARGET_{selector}"
                self.assertIn(f'println!("cargo:rerun-if-env-changed={name}");', SOURCE)


if __name__ == "__main__":
    unittest.main()
