#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Create and verify a release source bundle and binary provenance.

The source-to-binary receipt is deliberately not a self-attested JSON shim.  A
live verifier must also receive the clean Git worktree, source bundle, source
manifest, and build receipt, then independently reopen and validate all of
them.  This does not replace a signed remote build attestation, but it prevents
the R9 qualification harness from accepting a hand-written receipt by itself.
"""

from __future__ import annotations

import argparse
from datetime import date
import gzip
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

DEFAULT_RELEASE_VERSION = "v0.9.13-r9"
RELEASE_VERSION = DEFAULT_RELEASE_VERSION
SOURCE_MANIFEST_SCHEMA = "flux-r9-source-bundle-manifest/2"
SOURCE_TO_BINARY_SCHEMA = "flux-source-to-binary-receipt/1"
BUILD_RECEIPT_SCHEMA = "atlas-qwen38-dual-build-v1"
SOURCE_BUNDLE_NAME = "flux-engine-source-v0.9.13-r9.tar.gz"
SOURCE_MANIFEST_NAME = "SOURCE_BUNDLE_MANIFEST.json"
SOURCE_TO_BINARY_NAME = "SOURCE_TO_BINARY_RECEIPT.json"
ENGINE_PREFIX = "flux-engine"
FLASHINFER_PREFIX = "third_party/flashinfer-build-inputs"
FLASHINFER_MANIFEST_RELATIVE = (
    "repro-kit/source-closure/v0.9.13-r8/" "flashinfer-0.6.14-build-inputs.manifest.tsv"
)
EXPECTED_KERNEL_TARGETS = [
    "gb10/qwen3.8-27b/nvfp4",
    "gb10/qwen3.8-flash-next/nvfp4",
]
EXPECTED_PROFILE_NAMES = {
    "qwen3.8-flash-next-prefill4k-champion.yaml",
    "qwen3.8-flash-next-native-mtp-safe-local.yaml",
    "qwen3.8-27b-dflash2-safe-local.yaml",
    "qwen3.8-27b-dflash2-v15-safe-local.yaml",
}
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
RELEASE_VERSION_PATTERN = re.compile(
    r"v[0-9]+\.[0-9]+\.[0-9]+-(?:(r[1-9][0-9]*)|rc\.(r[1-9][0-9]*)\.([0-9]{8}))"
)


class ProvenanceError(ValueError):
    """Raised when a provenance input is incomplete or inconsistent."""


def release_identity(value: object) -> dict[str, str]:
    """Derive public labels and the source-schema lane from one exact version."""
    if type(value) is not str or not 0 < len(value) <= 96:
        raise ValueError("release version must be one bounded exact string")
    match = RELEASE_VERSION_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError("release version must be vX.Y.Z-rN or vX.Y.Z-rc.rN.YYYYMMDD")
    if match.group(3) is not None:
        stamp = match.group(3)
        date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:]))
    return {"release_version": value, "oci_version": value[1:],
            "source_lane": match.group(1) or match.group(2)}


def configure_release(release_version: str) -> None:
    """Select the release identity used by producers and live validators.

    The R9 defaults remain import-compatible for existing qualification tools.
    CLI callers may select a numbered release or a dated release candidate.
    Schema and filenames remain derived from that validated exact identity.
    """

    try:
        identity = release_identity(release_version)
    except ValueError as error:
        raise ProvenanceError(str(error)) from error
    release_lane = identity["source_lane"]
    global RELEASE_VERSION, SOURCE_MANIFEST_SCHEMA, SOURCE_BUNDLE_NAME
    RELEASE_VERSION = release_version
    SOURCE_MANIFEST_SCHEMA = f"flux-{release_lane}-source-bundle-manifest/2"
    SOURCE_BUNDLE_NAME = f"flux-engine-source-{release_version}.tar.gz"


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path, label: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ProvenanceError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ProvenanceError(f"{label} contains non-finite JSON number {value!r}")

    try:
        value = json.loads(
            path.read_bytes(),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except ProvenanceError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"cannot read {label}: {exc}") from exc
    validate_finite_json_numbers(value, label)
    if type(value) is not dict:
        raise ProvenanceError(f"{label} must be a JSON object")
    return value


def validate_finite_json_numbers(value: Any, label: str) -> None:
    """Reject exponent-overflow floats that ``parse_constant`` cannot see."""

    if type(value) is float and not math.isfinite(value):
        raise ProvenanceError(f"{label} contains a non-finite JSON number")
    if isinstance(value, list):
        for item in value:
            validate_finite_json_numbers(item, label)
    elif isinstance(value, dict):
        for item in value.values():
            validate_finite_json_numbers(item, label)


def require_regular(path: Path, label: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise ProvenanceError(f"{label} must be a regular non-symlink file")
    return path


def write_new(path: Path, payload: bytes, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    finally:
        os.close(descriptor)


def sanitized_git_environment() -> dict[str, str]:
    """Return a deterministic Git environment with lazy fetching disabled."""

    return {
        "PATH": os.defpath,
        "LC_ALL": "C",
        "LANG": "C",
        "GIT_NO_LAZY_FETCH": "1",
    }


def run_git(source_root: Path, *args: str, text: bool = True) -> str | bytes:
    try:
        result = subprocess.run(
            ["git", "--no-replace-objects", "-C", str(source_root), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=text,
            env=sanitized_git_environment(),
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "")
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", errors="replace")
        raise ProvenanceError(f"Git verification failed: {detail.strip()}") from exc
    return result.stdout


def validate_clean_source(
    source_root: Path, expected_commit: str, expected_tree: str
) -> Path:
    if (
        HEX40.fullmatch(expected_commit) is None
        or HEX40.fullmatch(expected_tree) is None
    ):
        raise ProvenanceError("source commit and tree must be lowercase 40-hex IDs")
    if source_root.is_symlink():
        raise ProvenanceError("source root must not be a symlink")
    source_root = source_root.resolve()
    if not source_root.is_dir() or not (
        (source_root / ".git").is_dir() or (source_root / ".git").is_file()
    ):
        raise ProvenanceError("source root is not a Git worktree")
    commit = str(run_git(source_root, "rev-parse", "--verify", "HEAD^{commit}")).strip()
    tree = str(
        run_git(source_root, "rev-parse", "--verify", f"{commit}^{{tree}}")
    ).strip()
    if commit != expected_commit:
        raise ProvenanceError("source worktree commit differs")
    if tree != expected_tree:
        raise ProvenanceError("source worktree tree differs")
    status = str(
        run_git(source_root, "status", "--porcelain=v1", "--untracked-files=all")
    )
    if status:
        raise ProvenanceError("source worktree is dirty, including untracked files")
    final_commit = str(
        run_git(source_root, "rev-parse", "--verify", "HEAD^{commit}")
    ).strip()
    if final_commit != commit:
        raise ProvenanceError("source worktree commit changed during validation")
    return source_root


def safe_relative(value: object, label: str) -> str:
    if type(value) is not str:
        raise ProvenanceError(f"{label} must be a relative POSIX path")
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or str(path) != value
    ):
        raise ProvenanceError(f"{label} is not a canonical relative POSIX path")
    return value


def parse_flashinfer_manifest(path: Path) -> list[tuple[str, str, int]]:
    require_regular(path, "FlashInfer build-input manifest")
    result: list[tuple[str, str, int]] = []
    seen: set[str] = set()
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.split("\t")
        if len(fields) != 3:
            raise ProvenanceError(f"FlashInfer manifest line {number} is malformed")
        relative, digest, size_text = fields
        safe_relative(relative, f"FlashInfer manifest line {number}")
        if relative in seen or HEX64.fullmatch(digest) is None:
            raise ProvenanceError(f"FlashInfer manifest line {number} is invalid")
        try:
            size = int(size_text)
        except ValueError as exc:
            raise ProvenanceError(
                f"FlashInfer manifest line {number} has an invalid size"
            ) from exc
        if size < 0 or str(size) != size_text:
            raise ProvenanceError(
                f"FlashInfer manifest line {number} has a non-canonical size"
            )
        seen.add(relative)
        result.append((relative, digest, size))
    if not result:
        raise ProvenanceError("FlashInfer build-input manifest is empty")
    return result


def regular_beneath(root: Path, relative: str, label: str) -> Path:
    """Resolve one manifest path without accepting symlink components."""

    safe_relative(relative, label)
    root = root.resolve()
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.is_symlink():
            raise ProvenanceError(f"{label} contains a symlink component")
    try:
        current.resolve().relative_to(root)
    except ValueError as exc:
        raise ProvenanceError(f"{label} escapes its declared root") from exc
    return require_regular(current, label)


def tracked_source_entries(
    source_root: Path, commit: str
) -> dict[str, tuple[int, int, str]]:
    raw = run_git(source_root, "ls-tree", "-r", "-z", "--full-tree", commit, text=False)
    assert isinstance(raw, bytes)
    result: dict[str, tuple[int, int, str]] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            metadata, encoded_path = record.split(b"\t", 1)
            mode_text, kind, _object_id = metadata.decode("ascii").split(" ", 2)
            relative = encoded_path.decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise ProvenanceError("Git tree contains an unparsable entry") from exc
        safe_relative(relative, "Git tree path")
        if kind != "blob" or mode_text not in ("100644", "100755"):
            raise ProvenanceError(f"unsupported Git tree entry: {relative}")
        path = source_root / relative
        require_regular(path, f"tracked source {relative}")
        data = path.read_bytes()
        mode = 0o755 if mode_text == "100755" else 0o644
        result[relative] = (mode, len(data), sha256_bytes(data))
    if not result:
        raise ProvenanceError("Git source tree is empty")
    return result


def add_tar_bytes(archive: tarfile.TarFile, name: str, data: bytes, mode: int) -> None:
    info = tarfile.TarInfo(name)
    info.type = tarfile.REGTYPE
    info.mode = mode
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.size = len(data)
    info.mtime = 0
    import io

    archive.addfile(info, io.BytesIO(data))


def build_source_bundle(
    source_root: Path,
    commit: str,
    tree: str,
    flashinfer_home: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    source_root = validate_clean_source(source_root, commit, tree)
    output_dir = output_dir.resolve()
    try:
        output_dir.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise ProvenanceError(
            "source-bundle output must be outside the source worktree"
        )
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ProvenanceError("source-bundle output directory already exists") from exc

    if flashinfer_home.is_symlink() or not flashinfer_home.is_dir():
        raise ProvenanceError("FlashInfer home must be a non-symlink directory")
    flashinfer_home = flashinfer_home.resolve()
    fi_manifest_path = source_root / FLASHINFER_MANIFEST_RELATIVE
    fi_entries = parse_flashinfer_manifest(fi_manifest_path)
    tracked = tracked_source_entries(source_root, commit)

    inventory: list[dict[str, object]] = []
    bundle_path = output_dir / SOURCE_BUNDLE_NAME
    try:
        raw = bundle_path.open("xb")
        try:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
                with tarfile.open(
                    fileobj=zipped, mode="w|", format=tarfile.GNU_FORMAT
                ) as archive:
                    for relative in sorted(tracked):
                        mode, size, digest = tracked[relative]
                        data = (source_root / relative).read_bytes()
                        if len(data) != size or sha256_bytes(data) != digest:
                            raise ProvenanceError(
                                f"tracked source changed while bundling: {relative}"
                            )
                        archive_name = f"{ENGINE_PREFIX}/{relative}"
                        add_tar_bytes(archive, archive_name, data, mode)
                        inventory.append(
                            {
                                "path": archive_name,
                                "bytes": size,
                                "sha256": digest,
                                "mode": mode,
                                "role": "engine-source",
                            }
                        )
                    for relative, expected_digest, expected_size in sorted(fi_entries):
                        path = regular_beneath(
                            flashinfer_home,
                            relative,
                            f"FlashInfer build input {relative}",
                        )
                        data = path.read_bytes()
                        if (
                            len(data) != expected_size
                            or sha256_bytes(data) != expected_digest
                        ):
                            raise ProvenanceError(
                                f"FlashInfer build input differs: {relative}"
                            )
                        archive_name = f"{FLASHINFER_PREFIX}/{relative}"
                        add_tar_bytes(archive, archive_name, data, 0o644)
                        inventory.append(
                            {
                                "path": archive_name,
                                "bytes": expected_size,
                                "sha256": expected_digest,
                                "mode": 0o644,
                                "role": "flashinfer-build-input",
                            }
                        )
            raw.flush()
            os.fsync(raw.fileno())
        finally:
            raw.close()
    except BaseException:
        bundle_path.unlink(missing_ok=True)
        raise

    validate_clean_source(source_root, commit, tree)
    manifest = {
        "schema": SOURCE_MANIFEST_SCHEMA,
        "release_version": RELEASE_VERSION,
        "source_revision": commit,
        "source_tree": tree,
        "clean_including_untracked": True,
        "flashinfer_input_manifest_sha256": sha256_file(fi_manifest_path),
        "flashinfer_input_count": len(fi_entries),
        "bundle": {
            "name": SOURCE_BUNDLE_NAME,
            "format": "tar.gz",
            "sha256": sha256_file(bundle_path),
            "bytes": bundle_path.stat().st_size,
        },
        "file_count": len(inventory),
        "files": inventory,
    }
    manifest_path = output_dir / SOURCE_MANIFEST_NAME
    write_new(manifest_path, canonical_bytes(manifest))
    return bundle_path, manifest_path


def expected_inventory(
    source_root: Path, commit: str
) -> dict[str, tuple[int, int, str, str]]:
    result: dict[str, tuple[int, int, str, str]] = {}
    for relative, (mode, size, digest) in tracked_source_entries(
        source_root, commit
    ).items():
        result[f"{ENGINE_PREFIX}/{relative}"] = (mode, size, digest, "engine-source")
    manifest_path = source_root / FLASHINFER_MANIFEST_RELATIVE
    for relative, digest, size in parse_flashinfer_manifest(manifest_path):
        result[f"{FLASHINFER_PREFIX}/{relative}"] = (
            0o644,
            size,
            digest,
            "flashinfer-build-input",
        )
    return result


def validate_source_bundle(
    source_root: Path,
    commit: str,
    tree: str,
    manifest_path: Path,
    bundle_path: Path,
) -> dict[str, Any]:
    source_root = validate_clean_source(source_root, commit, tree)
    require_regular(manifest_path, "source bundle manifest")
    require_regular(bundle_path, "source bundle")
    manifest = load_object(manifest_path, "source bundle manifest")
    required = {
        "schema",
        "release_version",
        "source_revision",
        "source_tree",
        "clean_including_untracked",
        "flashinfer_input_manifest_sha256",
        "flashinfer_input_count",
        "bundle",
        "file_count",
        "files",
    }
    if set(manifest) != required:
        raise ProvenanceError("source bundle manifest keys differ")
    if (
        manifest["schema"] != SOURCE_MANIFEST_SCHEMA
        or manifest["release_version"] != RELEASE_VERSION
        or manifest["source_revision"] != commit
        or manifest["source_tree"] != tree
        or manifest["clean_including_untracked"] is not True
    ):
        raise ProvenanceError("source bundle manifest identity differs")
    fi_manifest = source_root / FLASHINFER_MANIFEST_RELATIVE
    fi_entries = parse_flashinfer_manifest(fi_manifest)
    if manifest["flashinfer_input_manifest_sha256"] != sha256_file(
        fi_manifest
    ) or manifest["flashinfer_input_count"] != len(fi_entries):
        raise ProvenanceError("source bundle FlashInfer closure differs")
    bundle = manifest["bundle"]
    if type(bundle) is not dict or set(bundle) != {"name", "format", "sha256", "bytes"}:
        raise ProvenanceError("source bundle descriptor keys differ")
    if (
        bundle["name"] != SOURCE_BUNDLE_NAME
        or bundle["format"] != "tar.gz"
        or bundle["sha256"] != sha256_file(bundle_path)
        or bundle["bytes"] != bundle_path.stat().st_size
    ):
        raise ProvenanceError(
            "source bundle descriptor differs from the selected bundle"
        )

    expected = expected_inventory(source_root, commit)
    files = manifest["files"]
    if type(files) is not list or manifest["file_count"] != len(files):
        raise ProvenanceError("source bundle inventory count differs")
    normalized: dict[str, tuple[int, int, str, str]] = {}
    for entry in files:
        if type(entry) is not dict or set(entry) != {
            "path",
            "bytes",
            "sha256",
            "mode",
            "role",
        }:
            raise ProvenanceError("source bundle inventory entry keys differ")
        relative = safe_relative(entry["path"], "source bundle inventory path")
        if relative in normalized:
            raise ProvenanceError("source bundle inventory contains a duplicate path")
        size = entry["bytes"]
        mode = entry["mode"]
        digest = entry["sha256"]
        role = entry["role"]
        if (
            type(size) is not int
            or size < 0
            or type(mode) is not int
            or mode not in (0o644, 0o755)
            or type(digest) is not str
            or HEX64.fullmatch(digest) is None
            or role not in ("engine-source", "flashinfer-build-input")
        ):
            raise ProvenanceError("source bundle inventory contains invalid metadata")
        normalized[relative] = (mode, size, digest, role)
    if normalized != expected:
        raise ProvenanceError(
            "source bundle inventory does not match the clean source closure"
        )

    actual: dict[str, tuple[int, int, str]] = {}
    try:
        with tarfile.open(bundle_path, mode="r:gz") as archive:
            for member in archive:
                relative = safe_relative(member.name, "source bundle archive path")
                if not member.isfile() or member.issym() or member.islnk():
                    raise ProvenanceError("source bundle contains a non-regular member")
                if relative in actual:
                    raise ProvenanceError("source bundle contains a duplicate member")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ProvenanceError("source bundle member cannot be read")
                digest = hashlib.sha256()
                size = 0
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    size += len(block)
                    digest.update(block)
                actual[relative] = (member.mode & 0o777, size, digest.hexdigest())
    except (OSError, tarfile.TarError) as exc:
        raise ProvenanceError(f"cannot inspect source bundle: {exc}") from exc
    expected_archive = {
        path: (mode, size, digest)
        for path, (mode, size, digest, _role) in expected.items()
    }
    if actual != expected_archive:
        raise ProvenanceError(
            "source bundle archive content differs from its clean source closure"
        )
    validate_clean_source(source_root, commit, tree)
    return manifest


def validate_build_receipt(
    receipt_path: Path,
    source_root: Path,
    commit: str,
    tree: str,
    binary_path: Path,
) -> dict[str, Any]:
    require_regular(receipt_path, "build receipt")
    require_regular(binary_path, "selected binary")
    source_root = validate_clean_source(source_root, commit, tree)
    receipt = load_object(receipt_path, "build receipt")
    if set(receipt) != {
        "schema",
        "source",
        "build",
        "kernel_targets",
        "profiles",
        "binary",
    }:
        raise ProvenanceError("build receipt keys differ")
    if receipt["schema"] != BUILD_RECEIPT_SCHEMA:
        raise ProvenanceError("build receipt schema differs")
    source = receipt["source"]
    if type(source) is not dict or set(source) != {"path", "commit", "tree", "status"}:
        raise ProvenanceError("build receipt source keys differ")
    try:
        recorded_source = Path(source["path"]).resolve()
    except TypeError as exc:
        raise ProvenanceError("build receipt source path is invalid") from exc
    if (
        recorded_source != source_root
        or source["commit"] != commit
        or source["tree"] != tree
        or source["status"] != "clean"
    ):
        raise ProvenanceError("build receipt source identity differs")
    if receipt["kernel_targets"] != EXPECTED_KERNEL_TARGETS:
        raise ProvenanceError("build receipt kernel targets differ")

    build = receipt["build"]
    expected_build_keys = {
        "track",
        "command",
        "started_utc",
        "finished_utc",
        "log",
        "log_sha256",
        "flashinfer_input_manifest_sha256",
        "flashinfer_input_count",
    }
    if type(build) is not dict or set(build) != expected_build_keys:
        raise ProvenanceError("build receipt build keys differ")
    if build["track"] not in ("dense-qwen-dflash-v14", "dense-qwen-dflash-v15"):
        raise ProvenanceError("build receipt track differs")
    expected_command = (
        "ATLAS_TARGET_MODEL=qwen3.8-27b,qwen3.8-flash-next "
        f"ATLAS_RELEASE_PROFILE_TRACK={build['track']} "
        "cargo build --locked --offline --release -j2 -p spark-server --bin spark"
    )
    if build["command"] != expected_command:
        raise ProvenanceError("build receipt command differs")
    log_path = Path(build["log"])
    require_regular(log_path, "build log")
    if build["log_sha256"] != sha256_file(log_path):
        raise ProvenanceError("build log SHA-256 differs")
    fi_manifest = source_root / FLASHINFER_MANIFEST_RELATIVE
    fi_count = len(parse_flashinfer_manifest(fi_manifest))
    if (
        build["flashinfer_input_manifest_sha256"] != sha256_file(fi_manifest)
        or build["flashinfer_input_count"] != fi_count
    ):
        raise ProvenanceError("build receipt FlashInfer closure differs")

    profiles = receipt["profiles"]
    if type(profiles) is not list or len(profiles) != 3:
        raise ProvenanceError("build receipt profiles differ")
    observed_names: set[str] = set()
    for profile in profiles:
        if type(profile) is not dict or set(profile) != {
            "path",
            "sha256",
            "profile_kind",
            "environment_sha256",
        }:
            raise ProvenanceError("build receipt profile keys differ")
        path = Path(profile["path"]).resolve()
        try:
            path.relative_to(source_root)
        except ValueError as exc:
            raise ProvenanceError(
                "build receipt profile is outside the source root"
            ) from exc
        require_regular(path, "build profile")
        if profile["sha256"] != sha256_file(path):
            raise ProvenanceError("build profile SHA-256 differs")
        observed_names.add(path.name)
    required_names = {
        "qwen3.8-flash-next-prefill4k-champion.yaml",
        "qwen3.8-flash-next-native-mtp-safe-local.yaml",
    }
    if not required_names.issubset(observed_names) or not observed_names.issubset(
        EXPECTED_PROFILE_NAMES
    ):
        raise ProvenanceError("build receipt profile set differs")

    binary = receipt["binary"]
    if type(binary) is not dict or set(binary) != {
        "path",
        "sha256",
        "build_id",
        "bytes",
        "mode",
        "nlink",
    }:
        raise ProvenanceError("build receipt binary keys differ")
    try:
        recorded_binary = Path(binary["path"]).resolve()
    except TypeError as exc:
        raise ProvenanceError("build receipt binary path is invalid") from exc
    binary_bytes = binary_path.read_bytes()
    binary_stat = binary_path.stat()
    if (
        recorded_binary != binary_path.resolve()
        or binary["sha256"] != sha256_bytes(binary_bytes)
        or binary["bytes"] != len(binary_bytes)
        or binary["mode"] != "0555"
        or binary["nlink"] != 1
        or stat.S_IMODE(binary_stat.st_mode) != 0o555
        or binary_stat.st_nlink != 1
        or commit.encode("ascii") not in binary_bytes
    ):
        raise ProvenanceError("build receipt does not bind the selected binary")
    validate_clean_source(source_root, commit, tree)
    return receipt


def make_source_to_binary_receipt(
    source_root: Path,
    commit: str,
    tree: str,
    manifest_path: Path,
    bundle_path: Path,
    build_receipt_path: Path,
    binary_path: Path,
) -> dict[str, object]:
    manifest = validate_source_bundle(
        source_root, commit, tree, manifest_path, bundle_path
    )
    build = validate_build_receipt(
        build_receipt_path, source_root, commit, tree, binary_path
    )
    binary_bytes = binary_path.read_bytes()
    return {
        "schema": SOURCE_TO_BINARY_SCHEMA,
        "verdict": "PASS",
        "release_version": RELEASE_VERSION,
        "release_commit": commit,
        "release_tree": tree,
        "source_bundle_manifest_schema": SOURCE_MANIFEST_SCHEMA,
        "source_bundle_manifest_sha256": sha256_file(manifest_path),
        "source_bundle_sha256": sha256_file(bundle_path),
        "source_file_count": manifest["file_count"],
        "flashinfer_input_manifest_sha256": manifest[
            "flashinfer_input_manifest_sha256"
        ],
        "flashinfer_input_count": manifest["flashinfer_input_count"],
        "binary": {
            "sha256": sha256_bytes(binary_bytes),
            "bytes": len(binary_bytes),
            "embedded_build_commit": commit,
        },
        "build": {
            "receipt_schema": BUILD_RECEIPT_SCHEMA,
            "receipt_sha256": sha256_file(build_receipt_path),
            "log_sha256": build["build"]["log_sha256"],
            "track": build["build"]["track"],
            "kernel_targets": EXPECTED_KERNEL_TARGETS,
        },
        "claims": {
            "source_to_binary": True,
            "runtime_correctness": False,
            "performance": False,
            "container": False,
        },
    }


def validate_qualification_provenance(provenance: dict[str, Any]) -> None:
    required = {
        "release_version",
        "source_revision",
        "source_tree",
        "source_root_path",
        "source_bundle_manifest_path",
        "source_bundle_manifest_sha256",
        "source_bundle_path",
        "source_bundle_sha256",
        "build_receipt_path",
        "build_receipt_sha256",
        "source_to_binary_receipt_path",
        "source_to_binary_receipt_sha256",
        "binary_path",
        "binary_sha256",
    }
    if not required.issubset(provenance):
        raise ProvenanceError("qualification provenance is incomplete")
    if provenance["release_version"] != RELEASE_VERSION:
        raise ProvenanceError("qualification release version differs")
    source_root = Path(provenance["source_root_path"])
    manifest_path = Path(provenance["source_bundle_manifest_path"])
    bundle_path = Path(provenance["source_bundle_path"])
    build_receipt_path = Path(provenance["build_receipt_path"])
    binary_path = Path(provenance["binary_path"])
    source_receipt_path = Path(provenance["source_to_binary_receipt_path"])
    commit = provenance["source_revision"]
    tree = provenance["source_tree"]

    for path, expected, label in (
        (manifest_path, provenance["source_bundle_manifest_sha256"], "source manifest"),
        (bundle_path, provenance["source_bundle_sha256"], "source bundle"),
        (build_receipt_path, provenance["build_receipt_sha256"], "build receipt"),
        (binary_path, provenance["binary_sha256"], "binary"),
        (
            source_receipt_path,
            provenance["source_to_binary_receipt_sha256"],
            "source-to-binary receipt",
        ),
    ):
        require_regular(path, label)
        if sha256_file(path) != expected:
            raise ProvenanceError(f"{label} SHA-256 mismatch")

    manifest = validate_source_bundle(
        source_root, commit, tree, manifest_path, bundle_path
    )
    build = validate_build_receipt(
        build_receipt_path, source_root, commit, tree, binary_path
    )
    receipt = load_object(source_receipt_path, "source-to-binary receipt")
    expected_receipt = make_source_to_binary_receipt(
        source_root,
        commit,
        tree,
        manifest_path,
        bundle_path,
        build_receipt_path,
        binary_path,
    )
    if receipt != expected_receipt:
        raise ProvenanceError("source-to-binary receipt differs from verified inputs")
    if (
        receipt["source_bundle_manifest_sha256"]
        != provenance["source_bundle_manifest_sha256"]
        or receipt["source_bundle_sha256"] != provenance["source_bundle_sha256"]
        or receipt["build"]["receipt_sha256"] != provenance["build_receipt_sha256"]
        or receipt["binary"]["sha256"] != provenance["binary_sha256"]
        or receipt["source_file_count"] != manifest["file_count"]
        or receipt["build"]["log_sha256"] != build["build"]["log_sha256"]
    ):
        raise ProvenanceError("source-to-binary receipt bindings differ")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    bundle_parser = subparsers.add_parser("bundle")
    bundle_parser.add_argument("--source-root", required=True, type=Path)
    bundle_parser.add_argument("--source-revision", required=True)
    bundle_parser.add_argument("--source-tree", required=True)
    bundle_parser.add_argument("--flashinfer-home", required=True, type=Path)
    bundle_parser.add_argument("--output-dir", required=True, type=Path)
    bundle_parser.add_argument(
        "--release-version", default=DEFAULT_RELEASE_VERSION, metavar="VERSION"
    )

    receipt_parser = subparsers.add_parser("receipt")
    for current in (receipt_parser,):
        current.add_argument("--source-root", required=True, type=Path)
        current.add_argument("--source-revision", required=True)
        current.add_argument("--source-tree", required=True)
        current.add_argument("--source-manifest", required=True, type=Path)
        current.add_argument("--source-bundle", required=True, type=Path)
        current.add_argument("--build-receipt", required=True, type=Path)
        current.add_argument("--binary", required=True, type=Path)
        current.add_argument("--output", required=True, type=Path)
        current.add_argument(
            "--release-version", default=DEFAULT_RELEASE_VERSION, metavar="VERSION"
        )

    parsed = parser.parse_args(argv)
    try:
        configure_release(parsed.release_version)
        if parsed.command == "bundle":
            bundle, manifest = build_source_bundle(
                parsed.source_root,
                parsed.source_revision,
                parsed.source_tree,
                parsed.flashinfer_home,
                parsed.output_dir,
            )
            result = {
                "bundle": str(bundle),
                "bundle_sha256": sha256_file(bundle),
                "manifest": str(manifest),
                "manifest_sha256": sha256_file(manifest),
            }
        else:
            receipt = make_source_to_binary_receipt(
                parsed.source_root,
                parsed.source_revision,
                parsed.source_tree,
                parsed.source_manifest,
                parsed.source_bundle,
                parsed.build_receipt,
                parsed.binary,
            )
            write_new(parsed.output, canonical_bytes(receipt))
            result = {
                "receipt": str(parsed.output),
                "receipt_sha256": sha256_file(parsed.output),
                "binary_sha256": receipt["binary"]["sha256"],
            }
    except (OSError, ProvenanceError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
