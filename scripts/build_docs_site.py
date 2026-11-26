#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Build the Flux Engine GitHub Pages documentation site (mdBook + rustdoc + API Explorer).

Stages canonical root and `docs/` Markdown files into a temporary mdBook tree
without duplicating documentation in Git, runs `mdbook build`, and bundles
rustdoc (`/api/`), interactive Scalar API Explorer (`/api-explorer/`),
`openapi.json`, `llms.txt`, and `assets/` into the output directory.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

BOOK_TOML = """[book]
title = "Flux Engine Documentation"
description = "Pure Rust + Blackwell CUDA/PTX LLM inference runtime (GB10 & SM120)."
authors = ["TheApathy"]
language = "en"
src = "src"

[output.html]
default-theme = "navy"
preferred-dark-theme = "navy"
git-repository-url = "https://github.com/TheApathy/flux-engine"
edit-url-template = "https://github.com/TheApathy/flux-engine/edit/main/{path}"
no-section-label = true
"""

PAGES: list[tuple[str, str, str]] = [
    ("Overview", "README.md", "index.md"),
    ("Getting Started", "GETTING_STARTED.md", "getting-started.md"),
    ("HTTP API Reference", "docs/API.md", "api-reference.md"),
    ("Claude Code & Coding Agents", "docs/CLAUDE_CODE.md", "claude-code.md"),
    ("CLI Flags Reference", "FLAGS.md", "flags.md"),
    ("Architecture", "ARCHITECTURE.md", "architecture.md"),
    ("Discrete Blackwell (SM120 / SM103)", "docs/SM120_SM103_BRINGUP.md", "sm120-bringup.md"),
    ("Hardware Support", "docs/HARDWARE.md", "hardware.md"),
    ("Container Guide", "docs/CONTAINER_GUIDE.md", "container-guide.md"),
    ("Release Signing & SLSA Provenance", "SIGNING.md", "signing.md"),
    ("Reproducing Benchmarks", "REPRODUCING.md", "reproducing.md"),
    ("Contributing", "CONTRIBUTING.md", "contributing.md"),
    ("Changelog", "CHANGELOG.md", "changelog.md"),
    ("Credits & Attribution", "CREDITS.md", "credits.md"),
    ("Trademarks", "TRADEMARKS.md", "trademarks.md"),
]


def build_site(repo_root: Path, output_dir: Path, rustdoc_dir: Path | None) -> None:
    with tempfile.TemporaryDirectory(prefix="flux-mdbook-") as tmp:
        work = Path(tmp)
        src_dir = work / "src"
        src_dir.mkdir(parents=True)
        (work / "book.toml").write_text(BOOK_TOML, encoding="utf-8")

        summary_lines = [
            "# Summary",
            "",
            "[Overview](index.md)",
            "",
            "# Guides & Reference",
            "",
        ]
        for title, rel_src, dest_name in PAGES:
            src_path = repo_root / rel_src
            if not src_path.is_file():
                continue
            shutil.copy2(src_path, src_dir / dest_name)
            if dest_name != "index.md":
                summary_lines.append(f"- [{title}]({dest_name})")

        summary_lines.extend(
            [
                "",
                "# Interactive & Machine-Readable",
                "",
                "- [Interactive API Explorer](api-explorer/index.html)",
                "- [Rustdoc Crate API](api/index.html)",
            ]
        )
        (src_dir / "SUMMARY.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

        if (repo_root / "assets").is_dir():
            shutil.copytree(repo_root / "assets", src_dir / "assets")

        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)

        subprocess.run(
            ["mdbook", "build", str(work), "--dest-dir", str(output_dir.resolve())],
            check=True,
        )

    # Copy top-level assets, llms.txt, OpenAPI spec, interactive explorer, and rustdoc.
    for extra_file in ("llms.txt",):
        src = repo_root / extra_file
        if src.is_file():
            shutil.copy2(src, output_dir / extra_file)

    openapi = repo_root / "docs/openapi.json"
    if openapi.is_file():
        shutil.copy2(openapi, output_dir / "openapi.json")

    explorer = repo_root / "docs/api-explorer"
    if explorer.is_dir():
        shutil.copytree(explorer, output_dir / "api-explorer", dirs_exist_ok=True)

    if rustdoc_dir and rustdoc_dir.is_dir():
        shutil.copytree(rustdoc_dir, output_dir / "api", dirs_exist_ok=True)

    (output_dir / ".nojekyll").write_text("", encoding="utf-8")
    print(f"[PASS] Built Flux Engine docs site into {output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Flux Engine GitHub Pages site.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("_site"))
    parser.add_argument("--rustdoc", type=Path, default=None)
    args = parser.parse_args()
    build_site(args.repo_root.resolve(), args.output.resolve(), args.rustdoc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
