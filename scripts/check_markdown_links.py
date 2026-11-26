#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Offline deterministic checker for relative Markdown links.

Scans root `*.md`, `docs/**/*.md`, and `.github/**/*.md` (zero network I/O) and
verifies that every relative Markdown or HTML `href`/`src` link resolves to an
existing file or directory inside the repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from urllib.parse import unquote

FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
MD_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
HTML_LINK_RE = re.compile(r"""(?:href|src)\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
EXTERNAL_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")

# Local private agent notes are gitignored; public AGENTS.md and CLAUDE.md are checked.
EXCLUDED_ROOT_FILES = {"AGENTS.local.md", "CLAUDE.local.md"}


def iter_target_markdown_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for p in sorted(root.glob("*.md")):
        if p.is_file() and p.name not in EXCLUDED_ROOT_FILES:
            files.append(p)
    for subdir in ("docs", ".github"):
        base = root / subdir
        if base.is_dir():
            for p in sorted(base.rglob("*.md")):
                if p.is_file():
                    files.append(p)
    return files


def extract_relative_targets(text: str) -> list[tuple[int, str]]:
    """Return (1-based line_no, raw_target) pairs outside fenced/inline code."""
    results: list[tuple[int, str]] = []
    in_fence = False
    fence_char = ""
    fence_len = 0

    for lineno, line in enumerate(text.splitlines(), 1):
        m = FENCE_RE.match(line)
        if m:
            marker = m.group(1)
            if not in_fence:
                in_fence = True
                fence_char = marker[0]
                fence_len = len(marker)
                continue
            if marker[0] == fence_char and len(marker) >= fence_len:
                in_fence = False
                continue
        if in_fence:
            continue

        cleaned = INLINE_CODE_RE.sub("", line)
        for pattern in (MD_LINK_RE, HTML_LINK_RE):
            for match in pattern.finditer(cleaned):
                target = match.group(1).strip()
                if target.startswith("<") and target.endswith(">"):
                    target = target[1:-1].strip()
                results.append((lineno, target))
    return results


def is_external_or_anchor(target: str) -> bool:
    if not target or target.startswith("#") or target.startswith("//"):
        return True
    if EXTERNAL_SCHEME_RE.match(target):
        return True
    return False


def check_links(root: Path) -> tuple[int, int, list[str]]:
    root = root.resolve()
    md_files = iter_target_markdown_files(root)
    checked = 0
    broken: list[str] = []

    for md_path in md_files:
        rel_md = md_path.relative_to(root)
        text = md_path.read_text(encoding="utf-8", errors="replace")
        for lineno, raw in extract_relative_targets(text):
            if is_external_or_anchor(raw):
                continue
            path_part = raw.split("#", 1)[0].split("?", 1)[0]
            if not path_part:
                continue
            path_part = unquote(path_part)
            checked += 1

            if path_part.startswith("/"):
                resolved = (root / path_part.lstrip("/")).resolve()
            else:
                resolved = (md_path.parent / path_part).resolve()

            try:
                resolved.relative_to(root)
            except ValueError:
                broken.append(f"{rel_md}:{lineno}: link escapes repository root: {raw}")
                continue

            if not resolved.exists():
                broken.append(f"{rel_md}:{lineno}: broken relative link -> {raw}")

    return len(md_files), checked, broken


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify offline relative Markdown links.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Repository root directory (default: current directory).",
    )
    args = parser.parse_args(argv)

    file_count, link_count, broken = check_links(args.root)
    if broken:
        for item in broken:
            print(f"[BROKEN] {item}", file=sys.stderr)
        print(
            f"[FAIL] check_markdown_links: {len(broken)} broken link(s) across "
            f"{file_count} Markdown file(s) ({link_count} relative links checked).",
            file=sys.stderr,
        )
        return 1

    print(
        f"[PASS] check_markdown_links: {file_count} Markdown file(s), "
        f"{link_count} relative links checked — 0 broken links."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
