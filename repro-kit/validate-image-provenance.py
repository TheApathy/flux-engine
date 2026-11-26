#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Validate copied image-input bytes, not runtime or release qualification.

The outer inspector also checks these bindings against copied image bytes and
proven source. Live broker support and per-profile geometry remain on HOLD,
so these labels alone cannot qualify a runtime performance profile.
"""

import argparse
import hashlib
import json
import os
import re
import stat
from pathlib import Path

BINARY_NAMES = frozenset(
    {
        "flux-server-real",
        "flux-server-qwen38-27b",
        "flux-server-qwen36-27b",
        "flux-server-universal",
    }
)
PERFORMANCE_RECIPE_PATHS = {
    "interactive": "/opt/flux/repro-kit/recipes/flux-v9-w4096-interactive.env",
    "high-prefill": "/opt/flux/repro-kit/recipes/flux-v9-max-prefill.env",
}


def _open_without_symlinks(path: Path) -> int:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    directory = os.open(path.anchor, flags | os.O_DIRECTORY)
    try:
        for component in path.parts[1:-1]:
            child = os.open(component, flags | os.O_DIRECTORY, dir_fd=directory)
            os.close(directory)
            directory = child
        return os.open(path.name, flags, dir_fd=directory)
    finally:
        os.close(directory)


def _file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _verify_performance_file(path: Path, expected: str) -> tuple[int, int]:
    # Walk with directory descriptors so a path check cannot be followed by a
    # symlink substitution before the recipe bytes are opened.
    descriptor = _open_without_symlinks(path)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size == 0:
            raise ValueError(
                f"performance recipe must be a nonempty regular file: {path}"
            )
        digest = hashlib.sha256()
        for block in iter(lambda: os.read(descriptor, 1024 * 1024), b""):
            digest.update(block)
        if _file_identity(os.fstat(descriptor)) != _file_identity(before):
            raise ValueError(f"performance recipe changed during validation: {path}")
    finally:
        os.close(descriptor)
    reopened = _open_without_symlinks(path)
    try:
        if _file_identity(os.fstat(reopened)) != _file_identity(before):
            raise ValueError(
                f"performance recipe path changed during validation: {path}"
            )
    finally:
        os.close(reopened)
    if digest.hexdigest() != expected:
        raise ValueError(f"performance recipe SHA-256 mismatch: {path}")
    return before.st_dev, before.st_ino


def validate_performance_recipes(
    root: Path, bindings: dict[str, dict[str, str]]
) -> dict[str, str]:
    if type(bindings) is not dict or set(bindings) != set(PERFORMANCE_RECIPE_PATHS):
        raise ValueError(
            "performance recipes must bind exactly interactive and high-prefill"
        )
    labels = {}
    identities = set()
    digests = set()
    for profile, fixed_path in PERFORMANCE_RECIPE_PATHS.items():
        binding = bindings[profile]
        if type(binding) is not dict or set(binding) != {"path", "sha256"}:
            raise ValueError(f"invalid performance recipe binding: {profile}")
        if type(binding["path"]) is not str or binding["path"] != fixed_path:
            raise ValueError(f"performance recipe path is not canonical: {profile}")
        digest = binding["sha256"]
        if type(digest) is not str:
            raise ValueError(f"performance recipe SHA-256 must be a string: {profile}")
        _require_hex(digest, 64, f"{profile} recipe SHA-256")
        path = root.absolute() / Path(fixed_path).relative_to("/opt/flux")
        identity = _verify_performance_file(path, digest)
        if identity in identities or digest in digests:
            raise ValueError("performance recipe bindings must not alias each other")
        identities.add(identity)
        digests.add(digest)
        prefix = f"com.theapathy.flux.performance.{profile}.recipe"
        labels[f"{prefix}-path"] = fixed_path
        labels[f"{prefix}-sha256"] = digest
    return labels


def _require_hex(value: str, width: int, label: str) -> None:
    if re.fullmatch(rf"[0-9a-f]{{{width}}}", value) is None:
        raise ValueError(f"{label} must be a {width}-character lowercase hex value")


def _verify_file(path: Path, expected: str, *, executable: bool) -> None:
    if path.resolve(strict=True) != path.absolute():
        raise ValueError(f"release input has a symlink component: {path}")
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size == 0:
        raise ValueError(f"release input must be a nonempty regular file: {path}")
    if executable and not metadata.st_mode & 0o111:
        raise ValueError(f"selected binary must be executable: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f"release input SHA-256 mismatch: {path}")


def validate_image_inputs(
    root: Path,
    *,
    version: str,
    source_revision: str,
    source_tree: str,
    binary_name: str,
    binary_sha256: str,
    recipe_name: str,
    recipe_sha256: str,
    performance_recipes: dict[str, dict[str, str]],
) -> dict[str, str]:
    if re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:[-+][A-Za-z0-9.-]+)?", version) is None:
        raise ValueError("version must be one release version without a v prefix")
    _require_hex(source_revision, 40, "source revision")
    _require_hex(source_tree, 40, "source tree")
    _require_hex(binary_sha256, 64, "binary SHA-256")
    _require_hex(recipe_sha256, 64, "recipe SHA-256")
    if binary_name not in BINARY_NAMES:
        raise ValueError("selected binary is not one of the image's engine binaries")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.env", recipe_name) is None:
        raise ValueError("selected recipe must be one .env basename")
    root = root.absolute()
    _verify_file(root / "bin" / binary_name, binary_sha256, executable=True)
    _verify_file(
        root / "repro-kit" / "recipes" / recipe_name,
        recipe_sha256,
        executable=False,
    )
    labels = {
        "org.opencontainers.image.version": version,
        "org.opencontainers.image.revision": source_revision,
        "com.theapathy.flux.source-tree": source_tree,
        "com.theapathy.flux.binary-sha256": binary_sha256,
        "com.theapathy.flux.recipe-sha256": recipe_sha256,
    }
    labels.update(validate_performance_recipes(root, performance_recipes))
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/opt/flux"))
    for name in (
        "version",
        "source-revision",
        "source-tree",
        "binary-name",
        "binary-sha256",
        "recipe-name",
        "recipe-sha256",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--interactive-recipe-sha256", required=True)
    parser.add_argument("--high-prefill-recipe-sha256", required=True)
    args = vars(parser.parse_args())
    root = args.pop("root")
    args["performance_recipes"] = {
        profile: {
            "path": path,
            "sha256": args.pop(f"{profile.replace('-', '_')}_recipe_sha256"),
        }
        for profile, path in PERFORMANCE_RECIPE_PATHS.items()
    }
    try:
        labels = validate_image_inputs(root, **args)
    except (OSError, ValueError) as error:
        parser.exit(1, f"invalid image release provenance: {error}\n")
    print(json.dumps(labels, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
