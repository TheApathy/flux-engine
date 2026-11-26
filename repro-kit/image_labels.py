#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Write the image's label set to a JSON file that is COPYed into the image.

OCI labels live in the image config and are not visible from inside the
container, so `flux bundle` cannot report them. The image build script passes
the SAME `key=value` pairs it gives `docker build --label` (plus the base
image's inherited labels, if any) to this tool, which writes them to the build
context; the Dockerfile copies the file to /opt/flux/IMAGE_LABELS.json.

    python3 image_labels.py --out ctx/IMAGE_LABELS.json \
        [--base-labels-json base-labels.json] key=value [key=value ...]

Later pairs override earlier ones and the base labels, exactly as `--label`
overrides inherited labels. Labels are public metadata: a key or value that
looks like a credential or a host home path is refused, not written.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

SCHEMA = "flux-image-labels-v1"
IMAGE_PATH = "/opt/flux/IMAGE_LABELS.json"

_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
_SECRET_KEY_RE = re.compile(r"(?i)(?:^|[._-])(?:token|secret|password|passwd|credentials?|api[._-]?key|auth)(?:$|[._-])")
_SECRET_VALUE_RES = (
    re.compile(r"\b(?:Bearer|Basic)\s+[A-Za-z0-9._~+/=-]{6,}", re.I),
    re.compile(r"\bhf_[A-Za-z0-9]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{8,}"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{8,}"),
    re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{8,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\b[A-Za-z][A-Za-z0-9+.-]*://[^/\s:@]+:[^/\s@]*@"),
    re.compile(r"/(?:home|Users)/[^/\s\"':]+"),
)


class LabelError(ValueError):
    """A label pair is malformed or would publish a secret."""


def parse_pair(pair: str) -> "tuple[str, str]":
    key, sep, value = pair.partition("=")
    if not sep:
        raise LabelError(f"label {pair!r} is not key=value")
    if not _KEY_RE.match(key):
        raise LabelError(f"label key {key!r} is not a valid label name")
    return key, value


def check_label(key: str, value: str) -> None:
    if _SECRET_KEY_RE.search(key):
        raise LabelError(f"label key {key!r} looks like a credential; labels are public")
    for rule in _SECRET_VALUE_RES:
        if rule.search(value):
            raise LabelError(f"label {key!r} value looks like a credential or host path; labels are public")


def build_labels(pairs: Iterable[str], base: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """Return the effective label map: base labels, then each pair in order (last wins)."""
    labels: Dict[str, str] = {}
    for key, value in (base or {}).items():
        labels[str(key)] = "" if value is None else str(value)
    for pair in pairs:
        key, value = parse_pair(pair)
        labels[key] = value
    for key, value in labels.items():
        check_label(key, value)
    return dict(sorted(labels.items()))


def document(labels: Mapping[str, str]) -> Dict[str, object]:
    return {"schema": SCHEMA, "path": IMAGE_PATH, "labels": dict(labels)}


def write_labels(out: Path, labels: Mapping[str, str]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.partial")
    tmp.write_text(json.dumps(document(labels), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(tmp, 0o644)
    os.replace(tmp, out)


def _load_base(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    text = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
    data = json.loads(text) if text.strip() else None
    if data is None:  # `docker inspect` prints null for an image without labels
        return {}
    if not isinstance(data, dict):
        raise LabelError("--base-labels-json must hold a JSON object (docker inspect .Config.Labels)")
    return data


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="where to write IMAGE_LABELS.json in the build context")
    ap.add_argument("--base-labels-json", help="file (or - for stdin) with the base image's .Config.Labels")
    ap.add_argument("pairs", nargs="*", metavar="key=value", help="the exact --label pairs given to docker build")
    args = ap.parse_args(list(argv) if argv is not None else None)
    try:
        labels = build_labels(args.pairs, _load_base(args.base_labels_json))
        write_labels(Path(args.out), labels)
    except (LabelError, OSError, ValueError) as e:
        print(f"image_labels: {e}", file=sys.stderr)
        return 1
    print(f"image_labels: wrote {len(labels)} labels to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
