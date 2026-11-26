#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Offline unit tests for scripts/generate_cargo_spdx_sbom.py."""

from __future__ import annotations

import importlib.util
import json
import re
import tempfile
import tomllib
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts/generate_cargo_spdx_sbom.py"
SPDX_ID_RE = re.compile(r"^SPDXRef-[A-Za-z0-9.-]+$")


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_cargo_spdx_sbom", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SBOM = _load_module()


class GenerateCargoSpdxSbomTests(unittest.TestCase):
    def test_deterministic_output_and_schema_on_real_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out1 = Path(tmp) / "sbom1.spdx.json"
            out2 = Path(tmp) / "sbom2.spdx.json"
            argv_common = [
                "--repo-root",
                str(REPO_ROOT),
                "--release-tag",
                "v1.0.3",
                "--commit",
                "0123456789abcdef0123456789abcdef01234567",
                "--created",
                "2026-11-26T00:00:00Z",
            ]
            self.assertEqual(SBOM.main([*argv_common, "--output", str(out1)]), 0)
            self.assertEqual(SBOM.main([*argv_common, "--output", str(out2)]), 0)

            raw1 = out1.read_bytes()
            raw2 = out2.read_bytes()
            self.assertEqual(raw1, raw2, "SBOM output must be byte-for-byte deterministic")

            doc = json.loads(raw1.decode("utf-8"))
            self.assertEqual(doc["spdxVersion"], "SPDX-2.3")
            self.assertEqual(doc["dataLicense"], "CC0-1.0")
            self.assertEqual(doc["SPDXID"], "SPDXRef-DOCUMENT")
            self.assertEqual(doc["creationInfo"]["created"], "2026-11-26T00:00:00Z")

            lock = tomllib.loads((REPO_ROOT / "Cargo.lock").read_text(encoding="utf-8"))
            lock_pkgs = lock["package"]
            # Root package + every entry in Cargo.lock.
            self.assertEqual(len(doc["packages"]), len(lock_pkgs) + 1)

            spdx_ids = [pkg["SPDXID"] for pkg in doc["packages"]]
            self.assertEqual(len(spdx_ids), len(set(spdx_ids)), "SPDXIDs must be unique")
            for spdx_id in spdx_ids:
                self.assertRegex(spdx_id, SPDX_ID_RE)

            by_name = {pkg["name"]: pkg for pkg in doc["packages"]}
            self.assertEqual(by_name["flux-engine"]["licenseDeclared"], "AGPL-3.0-only")
            self.assertEqual(by_name["spark-server"]["licenseDeclared"], "AGPL-3.0-only")
            self.assertEqual(by_name["xgrammar"]["licenseDeclared"], "AGPL-3.0-only")

            # Every external crates.io entry with a checksum in Cargo.lock carries SHA256.
            crates_io_with_checksum = sum(1 for p in lock_pkgs if "checksum" in p)
            sbom_with_sha256 = sum(
                1
                for p in doc["packages"]
                if any(c["algorithm"] == "SHA256" for c in p.get("checksums", []))
            )
            self.assertEqual(sbom_with_sha256, crates_io_with_checksum)

            # Every relationship references valid declared SPDXIDs.
            valid_ids = set(spdx_ids) | {"SPDXRef-DOCUMENT"}
            self.assertGreater(len(doc["relationships"]), len(lock_pkgs))
            for rel in doc["relationships"]:
                self.assertIn(rel["spdxElementId"], valid_ids)
                self.assertIn(rel["relatedSpdxElement"], valid_ids)

    def test_duplicate_crate_versions_and_epoch_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "crates/demo").mkdir(parents=True)
            (root / "Cargo.toml").write_text(
                '[workspace]\nmembers = ["crates/demo"]\n'
                '[workspace.package]\nversion = "1.2.3"\nlicense = "AGPL-3.0-only"\n',
                encoding="utf-8",
            )
            (root / "crates/demo/Cargo.toml").write_text(
                '[package]\nname = "demo"\nversion.workspace = true\nlicense.workspace = true\n',
                encoding="utf-8",
            )
            (root / "Cargo.lock").write_text(
                'version = 4\n\n'
                '[[package]]\nname = "demo"\nversion = "1.2.3"\n'
                'dependencies = ["dup 0.1.0", "dup 0.2.0"]\n\n'
                '[[package]]\nname = "dup"\nversion = "0.1.0"\n'
                'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
                'checksum = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"\n\n'
                '[[package]]\nname = "dup"\nversion = "0.2.0"\n'
                'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
                'checksum = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"\n',
                encoding="utf-8",
            )
            doc = SBOM.build_spdx_document(
                repo_root=root,
                release_tag="v1.2.3",
                commit="abcdef0",
                created="1700000000",
            )
            self.assertEqual(doc["creationInfo"]["created"], "2023-11-14T22:13:20Z")
            dep_rels = [r for r in doc["relationships"] if r["relationshipType"] == "DEPENDS_ON"]
            self.assertEqual(len(dep_rels), 2)
            self.assertEqual(len({r["relatedSpdxElement"] for r in dep_rels}), 2)


if __name__ == "__main__":
    unittest.main()
