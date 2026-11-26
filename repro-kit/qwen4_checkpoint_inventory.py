#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Create a content-hashed checkpoint inventory for R9 qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


SCHEMA = "flux-checkpoint-inventory/1"


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )


def stable_file_record(root: Path, path: Path) -> dict[str, Any]:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise RuntimeError(f"checkpoint file changed while hashing: {path}")
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": after.st_size,
        "sha256": digest.hexdigest(),
    }


def inventory(root: Path, checkpoint_id: str, checkpoint_revision: str) -> dict[str, Any]:
    root = root.resolve()
    if not root.is_dir():
        raise ValueError("checkpoint root is not a directory")
    if not checkpoint_id:
        raise ValueError("checkpoint id is empty")
    if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", checkpoint_revision):
        raise ValueError("checkpoint revision must be a lowercase immutable hash")
    paths = []
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"checkpoint contains a non-file entry: {path}")
        paths.append(path)
    paths.sort()
    if not paths:
        raise ValueError("checkpoint root has no files")
    records = [stable_file_record(root, path) for path in paths]
    return {
        "schema": SCHEMA,
        "checkpoint_id": checkpoint_id,
        "checkpoint_revision": checkpoint_revision,
        "file_count": len(records),
        "total_bytes": sum(record["bytes"] for record in records),
        "files": records,
    }


def write_all(fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(fd, payload[offset:])
        if written <= 0:
            raise OSError("inventory write made no progress")
        offset += written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    root = args.checkpoint_root.resolve()
    output = args.output.resolve()
    try:
        output.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError("inventory output must be outside the checkpoint root")
    document = inventory(root, args.checkpoint_id, args.checkpoint_revision)
    payload = canonical_bytes(document)
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        write_all(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    print(
        f"PASS inventory={output} files={document['file_count']} "
        f"bytes={document['total_bytes']} sha256={hashlib.sha256(payload).hexdigest()}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
