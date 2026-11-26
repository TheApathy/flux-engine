# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import importlib.util
import io
import json
import os
import subprocess
import tarfile
import tempfile
import unittest
from unittest import mock
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "r9_source_to_binary.py"
SPEC = importlib.util.spec_from_file_location("r9_source_to_binary", MODULE_PATH)
assert SPEC and SPEC.loader
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)


class ProvenanceFixture:
    def __init__(self, root: Path):
        self.root = root
        self.source = root / "source"
        self.flashinfer = root / "flashinfer"
        self.source.mkdir()
        self.flashinfer.mkdir()
        subprocess.run(["git", "init", "-q", str(self.source)], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(self.source),
                "config",
                "user.email",
                "fixture@example.invalid",
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(self.source), "config", "user.name", "Fixture"],
            check=True,
        )

        (self.source / "src/nested").mkdir(parents=True)
        (self.source / "src/nested/lib.rs").write_text("pub fn fixture() {}\n")
        profile_root = self.source / "crates/spark-server/src/recipe/builtin"
        profile_root.mkdir(parents=True)
        self.profiles = []
        for name in (
            "qwen3.8-flash-next-prefill4k-champion.yaml",
            "qwen3.8-flash-next-native-mtp-safe-local.yaml",
            "qwen3.8-27b-dflash2-v15-safe-local.yaml",
        ):
            path = profile_root / name
            path.write_text(f"profile: {name}\n")
            self.profiles.append(path)

        header = self.flashinfer / "include/flashinfer/example.cuh"
        header.parent.mkdir(parents=True)
        header.write_bytes(b"// pinned fixture\n")
        fi_manifest = self.source / P.FLASHINFER_MANIFEST_RELATIVE
        fi_manifest.parent.mkdir(parents=True)
        fi_manifest.write_text(
            "include/flashinfer/example.cuh\t"
            f"{P.sha256_file(header)}\t{header.stat().st_size}\n"
        )

        subprocess.run(["git", "-C", str(self.source), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", str(self.source), "commit", "-q", "-m", "fixture"],
            check=True,
        )
        self.commit = self.git("rev-parse", "HEAD")
        self.tree = self.git("rev-parse", "HEAD^{tree}")

        self.bundle_dir = root / "bundle"
        self.bundle, self.manifest = P.build_source_bundle(
            self.source,
            self.commit,
            self.tree,
            self.flashinfer,
            self.bundle_dir,
        )
        self.binary = root / "spark-r9"
        self.binary.write_bytes(b"\x7fELFfixture\0" + self.commit.encode("ascii"))
        self.binary.chmod(0o555)
        self.log = root / "build.log"
        self.log.write_text(
            "compiled 1 kernels for target 0\ncompiled 1 kernels for target 1\n"
        )
        self.build_receipt = root / "build-receipt.json"
        track = "dense-qwen-dflash-v15"
        build = {
            "schema": P.BUILD_RECEIPT_SCHEMA,
            "source": {
                "path": str(self.source.resolve()),
                "commit": self.commit,
                "tree": self.tree,
                "status": "clean",
            },
            "build": {
                "track": track,
                "command": (
                    "ATLAS_TARGET_MODEL=qwen3.8-27b,qwen3.8-flash-next "
                    f"ATLAS_RELEASE_PROFILE_TRACK={track} "
                    "cargo build --locked --offline --release -j2 -p spark-server --bin spark"
                ),
                "started_utc": "2026-10-18T00:00:00Z",
                "finished_utc": "2026-10-18T00:01:00Z",
                "log": str(self.log),
                "log_sha256": P.sha256_file(self.log),
                "flashinfer_input_manifest_sha256": P.sha256_file(
                    self.source / P.FLASHINFER_MANIFEST_RELATIVE
                ),
                "flashinfer_input_count": 1,
            },
            "kernel_targets": list(P.EXPECTED_KERNEL_TARGETS),
            "profiles": [
                {
                    "path": str(path),
                    "sha256": P.sha256_file(path),
                    "profile_kind": "fixture",
                    "environment_sha256": "0" * 64,
                }
                for path in self.profiles
            ],
            "binary": {
                "path": str(self.binary),
                "sha256": P.sha256_file(self.binary),
                "build_id": "fixture",
                "bytes": self.binary.stat().st_size,
                "mode": "0555",
                "nlink": 1,
            },
        }
        self.build_receipt.write_bytes(P.canonical_bytes(build))
        self.source_receipt = root / P.SOURCE_TO_BINARY_NAME
        receipt = P.make_source_to_binary_receipt(
            self.source,
            self.commit,
            self.tree,
            self.manifest,
            self.bundle,
            self.build_receipt,
            self.binary,
        )
        self.source_receipt.write_bytes(P.canonical_bytes(receipt))

    def git(self, *args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(self.source), *args], text=True
        ).strip()

    def provenance(self) -> dict[str, object]:
        return {
            "release_version": P.RELEASE_VERSION,
            "source_revision": self.commit,
            "source_tree": self.tree,
            "source_root_path": str(self.source),
            "source_bundle_manifest_path": str(self.manifest),
            "source_bundle_manifest_sha256": P.sha256_file(self.manifest),
            "source_bundle_path": str(self.bundle),
            "source_bundle_sha256": P.sha256_file(self.bundle),
            "build_receipt_path": str(self.build_receipt),
            "build_receipt_sha256": P.sha256_file(self.build_receipt),
            "source_to_binary_receipt_path": str(self.source_receipt),
            "source_to_binary_receipt_sha256": P.sha256_file(self.source_receipt),
            "binary_path": str(self.binary),
            "binary_sha256": P.sha256_file(self.binary),
        }


def rewrite_tar(source: Path, destination: Path, mutate) -> None:
    with tarfile.open(source, "r:gz") as original:
        members = []
        for member in original:
            stream = original.extractfile(member)
            assert stream is not None
            members.append((member, stream.read()))
    with tarfile.open(destination, "w:gz") as rewritten:
        for member, data in mutate(members):
            rewritten.addfile(member, io.BytesIO(data))


class R9SourceToBinaryTests(unittest.TestCase):
    def fixture(self, directory: str) -> ProvenanceFixture:
        return ProvenanceFixture(Path(directory))

    def test_json_loader_rejects_duplicate_keys_and_nonfinite_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            path.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
            with self.assertRaisesRegex(P.ProvenanceError, "duplicate JSON key"):
                P.load_object(path, "fixture")
            for value in ("NaN", "Infinity", "-Infinity", "1e9999", "-1e9999"):
                with self.subTest(value=value):
                    path.write_text(f'{{"value":{value}}}', encoding="utf-8")
                    with self.assertRaisesRegex(
                        P.ProvenanceError, "non-finite JSON number"
                    ):
                        P.load_object(path, "fixture")

    def test_every_git_read_is_sanitized_and_replace_refs_are_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            original_tree = fixture.tree
            replacement = fixture.root / "replacement"
            replacement.mkdir()
            subprocess.run(["git", "init", "--quiet", str(replacement)], check=True)
            (replacement / "different.txt").write_text("different\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(replacement), "add", "."], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(replacement),
                    "-c",
                    "user.name=fixture",
                    "-c",
                    "user.email=fixture@example.invalid",
                    "commit",
                    "--quiet",
                    "-m",
                    "replacement",
                ],
                check=True,
            )
            replacement_commit = subprocess.check_output(
                ["git", "-C", str(replacement), "rev-parse", "HEAD"], text=True
            ).strip()
            replacement_object = subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(replacement),
                    "cat-file",
                    "commit",
                    replacement_commit,
                ]
            )
            imported_commit = (
                subprocess.check_output(
                    [
                        "git",
                        "-C",
                        str(fixture.source),
                        "hash-object",
                        "-t",
                        "commit",
                        "-w",
                        "--stdin",
                    ],
                    input=replacement_object,
                )
                .decode()
                .strip()
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(fixture.source),
                    "replace",
                    fixture.commit,
                    imported_commit,
                ],
                check=True,
            )
            hostile = {
                "GIT_DIR": str(replacement / ".git"),
                "GIT_WORK_TREE": str(replacement),
                "GIT_OBJECT_DIRECTORY": str(replacement / ".git/objects"),
                "GIT_ALTERNATE_OBJECT_DIRECTORIES": str(replacement / ".git/objects"),
                "GIT_CONFIG_COUNT": "1",
                "GIT_CONFIG_KEY_0": "core.repositoryformatversion",
                "GIT_CONFIG_VALUE_0": "999",
            }
            calls = []
            original_run = P.subprocess.run

            def inspect_run(*args, **kwargs):
                calls.append((args, kwargs))
                return original_run(*args, **kwargs)

            with mock.patch.dict(os.environ, hostile, clear=False), mock.patch.object(
                P.subprocess, "run", side_effect=inspect_run
            ):
                self.assertEqual(
                    P.validate_clean_source(
                        fixture.source, fixture.commit, original_tree
                    ),
                    fixture.source.resolve(),
                )
            self.assertTrue(calls)
            for args, kwargs in calls:
                command = args[0]
                self.assertEqual(command[0:2], ["git", "--no-replace-objects"])
                self.assertEqual(
                    kwargs["env"],
                    {
                        "PATH": os.defpath,
                        "LC_ALL": "C",
                        "LANG": "C",
                        "GIT_NO_LAZY_FETCH": "1",
                    },
                )
            flattened = [item for args, _ in calls for item in args[0]]
            self.assertIn(f"{fixture.commit}^{{tree}}", flattened)
            self.assertNotIn("HEAD^{tree}", flattened)

    def test_verified_chain_reopens_recursive_tree_archive_build_and_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            P.validate_qualification_provenance(fixture.provenance())
            manifest = json.loads(fixture.manifest.read_text())
            paths = {entry["path"] for entry in manifest["files"]}
            self.assertIn("flux-engine/src/nested/lib.rs", paths)

    def test_release_identity_is_derived_from_validated_cli_value(self):
        original = P.RELEASE_VERSION
        try:
            P.configure_release("v0.9.13-r12")
            self.assertEqual(P.RELEASE_VERSION, "v0.9.13-r12")
            self.assertEqual(
                P.SOURCE_MANIFEST_SCHEMA, "flux-r12-source-bundle-manifest/2"
            )
            self.assertEqual(
                P.SOURCE_BUNDLE_NAME, "flux-engine-source-v0.9.13-r12.tar.gz"
            )
            with self.assertRaisesRegex(P.ProvenanceError, "release version"):
                P.configure_release("r12")
            with self.assertRaisesRegex(P.ProvenanceError, "release version"):
                P.configure_release("v0.9.13-r0")
        finally:
            P.configure_release(original)

    def test_r12_bundle_and_manifest_use_the_selected_release_identity(self):
        original = P.RELEASE_VERSION
        try:
            with tempfile.TemporaryDirectory() as directory:
                fixture = self.fixture(directory)
                P.configure_release("v0.9.13-r12")
                bundle, manifest_path = P.build_source_bundle(
                    fixture.source,
                    fixture.commit,
                    fixture.tree,
                    fixture.flashinfer,
                    fixture.root / "bundle-r12",
                )
                manifest = json.loads(manifest_path.read_text())
                self.assertEqual(bundle.name, "flux-engine-source-v0.9.13-r12.tar.gz")
                self.assertEqual(manifest["release_version"], "v0.9.13-r12")
                self.assertEqual(
                    manifest["schema"], "flux-r12-source-bundle-manifest/2"
                )
                P.validate_source_bundle(
                    fixture.source,
                    fixture.commit,
                    fixture.tree,
                    manifest_path,
                    bundle,
                )
        finally:
            P.configure_release(original)

    def test_dirty_or_symlink_swapped_inputs_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            (fixture.source / "untracked.txt").write_text("dirty")
            with self.assertRaisesRegex(P.ProvenanceError, "dirty"):
                P.validate_qualification_provenance(fixture.provenance())
            (fixture.source / "untracked.txt").unlink()

            original = fixture.binary.read_bytes()
            fixture.binary.unlink()
            target = fixture.root / "other-binary"
            target.write_bytes(original)
            target.chmod(0o555)
            fixture.binary.symlink_to(target)
            with self.assertRaisesRegex(P.ProvenanceError, "non-symlink"):
                P.validate_qualification_provenance(fixture.provenance())

    def test_bundle_producer_rejects_output_inside_source_and_symlinked_fi_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            inside = fixture.source / "generated-provenance"
            with self.assertRaisesRegex(P.ProvenanceError, "outside"):
                P.build_source_bundle(
                    fixture.source,
                    fixture.commit,
                    fixture.tree,
                    fixture.flashinfer,
                    inside,
                )
            self.assertFalse(inside.exists())

            original_include = fixture.flashinfer / "include"
            relocated_include = fixture.root / "relocated-include"
            original_include.rename(relocated_include)
            original_include.symlink_to(relocated_include, target_is_directory=True)
            with self.assertRaisesRegex(P.ProvenanceError, "symlink component"):
                P.build_source_bundle(
                    fixture.source,
                    fixture.commit,
                    fixture.tree,
                    fixture.flashinfer,
                    fixture.root / "symlinked-fi-bundle",
                )

    def test_archive_mode_and_path_escape_are_rejected_even_if_rehashed(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            mode_bundle = fixture.root / "mode.tar.gz"

            def change_mode(members):
                members[0][0].mode = 0o777
                return members

            rewrite_tar(fixture.bundle, mode_bundle, change_mode)
            manifest = json.loads(fixture.manifest.read_text())
            manifest["bundle"]["sha256"] = P.sha256_file(mode_bundle)
            manifest["bundle"]["bytes"] = mode_bundle.stat().st_size
            mode_manifest = fixture.root / "mode-manifest.json"
            mode_manifest.write_bytes(P.canonical_bytes(manifest))
            with self.assertRaisesRegex(P.ProvenanceError, "archive content differs"):
                P.validate_source_bundle(
                    fixture.source,
                    fixture.commit,
                    fixture.tree,
                    mode_manifest,
                    mode_bundle,
                )

            escape_bundle = fixture.root / "escape.tar.gz"

            def add_escape(members):
                escaped = tarfile.TarInfo("../escape")
                escaped.type = tarfile.REGTYPE
                escaped.mode = 0o644
                escaped.size = 1
                members.append((escaped, b"x"))
                return members

            rewrite_tar(fixture.bundle, escape_bundle, add_escape)
            manifest["bundle"]["sha256"] = P.sha256_file(escape_bundle)
            manifest["bundle"]["bytes"] = escape_bundle.stat().st_size
            escape_manifest = fixture.root / "escape-manifest.json"
            escape_manifest.write_bytes(P.canonical_bytes(manifest))
            with self.assertRaisesRegex(P.ProvenanceError, "canonical relative"):
                P.validate_source_bundle(
                    fixture.source,
                    fixture.commit,
                    fixture.tree,
                    escape_manifest,
                    escape_bundle,
                )

    def test_self_attested_manifest_cannot_replace_clean_source_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            manifest = json.loads(fixture.manifest.read_text())
            manifest["files"][0]["sha256"] = "f" * 64
            forged = fixture.root / "forged-manifest.json"
            forged.write_bytes(P.canonical_bytes(manifest))
            with self.assertRaisesRegex(P.ProvenanceError, "clean source closure"):
                P.validate_source_bundle(
                    fixture.source,
                    fixture.commit,
                    fixture.tree,
                    forged,
                    fixture.bundle,
                )

    def test_minimal_hand_written_receipt_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            forged = {
                "schema": P.SOURCE_TO_BINARY_SCHEMA,
                "verdict": "PASS",
                "release_version": P.RELEASE_VERSION,
                "release_commit": fixture.commit,
                "source_bundle_manifest_sha256": P.sha256_file(fixture.manifest),
                "source_bundle_sha256": P.sha256_file(fixture.bundle),
                "binary": {"sha256": P.sha256_file(fixture.binary)},
            }
            fixture.source_receipt.write_bytes(P.canonical_bytes(forged))
            provenance = fixture.provenance()
            provenance["source_to_binary_receipt_sha256"] = P.sha256_file(
                fixture.source_receipt
            )
            with self.assertRaisesRegex(
                P.ProvenanceError, "differs from verified inputs"
            ):
                P.validate_qualification_provenance(provenance)


if __name__ == "__main__":
    unittest.main()
