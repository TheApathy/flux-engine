#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Build the Flux Engine GitHub Pages documentation site (mdBook + rustdoc + API Explorer).

Stages canonical root and `docs/` Markdown files into a temporary mdBook tree
without duplicating documentation in Git, rewrites repo-relative links so every
front-page link works seamlessly inside mdBook as well as on GitHub, runs
`mdbook build`, and bundles rustdoc (`/api/`), interactive Scalar API Explorer
(`/api-explorer/`), `openapi.json`, `llms.txt`, and `assets/` into the output.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO_BLOB_BASE = "https://github.com/TheApathy/flux-engine/blob/main"
REPO_TREE_BASE = "https://github.com/TheApathy/flux-engine/tree/main"
SITE_BASE_URL = "https://theapathy.github.io/flux-engine"

BOOK_TOML = """[book]
title = "Flux Engine Documentation"
description = "Self-contained Rust + Blackwell CUDA/PTX LLM inference server."
authors = ["TheApathy"]
language = "en"
src = "src"

[output.html]
default-theme = "navy"
preferred-dark-theme = "navy"
git-repository-url = "https://github.com/TheApathy/flux-engine"
site-url = "/flux-engine/"
no-section-label = true
"""

PAGES: list[tuple[str, str, str]] = [
    ("Overview", "README.md", "index.md"),
    ("Getting Started", "docs/GETTING_STARTED.md", "getting-started.md"),
    ("HTTP API Reference", "docs/API.md", "api-reference.md"),
    ("Coding Clients (Claude Code, Codex, OpenCode)", "docs/clients/README.md", "clients.md"),
    ("Claude Code Integration", "docs/CLAUDE_CODE.md", "claude-code.md"),
    ("Production Deployment (systemd, K8s, Grafana)", "deploy/README.md", "deploy.md"),
    ("CLI Flags Reference", "docs/FLAGS.md", "flags.md"),
    ("Architecture", "docs/ARCHITECTURE.md", "architecture.md"),
    ("Kernel Profile", "docs/KERNEL-PROFILE.md", "kernel-profile.md"),
    ("Discrete Blackwell (SM120 / SM103)", "docs/SM120_SM103_BRINGUP.md", "sm120-bringup.md"),
    ("Hardware Support", "docs/HARDWARE.md", "hardware.md"),
    ("Container Guide", "docs/CONTAINER_GUIDE.md", "container-guide.md"),
    ("Benchmark Verification", "docs/verification/BENCHMARK_VERIFICATION.md", "benchmark-verification.md"),
    ("Reproducing Benchmarks", "docs/REPRODUCING.md", "reproducing.md"),
    ("Agent & Engineering Contract (AGENTS.md)", "AGENTS.md", "agents.md"),
    ("Release Signing & SLSA Provenance", "SIGNING.md", "signing.md"),
    ("Contributing", "CONTRIBUTING.md", "contributing.md"),
    ("Changelog", "CHANGELOG.md", "changelog.md"),
    ("Credits & Attribution", "CREDITS.md", "credits.md"),
    ("Trademarks", "docs/TRADEMARKS.md", "trademarks.md"),
]

MD_LINK_RE = re.compile(r'(!?\[[^\]]*\]\()([^)\s]+)(\))')
HREF_RE = re.compile(r'(href=")([^"]+)(")')
SRC_RE = re.compile(r'(src=")([^"]+)(")')


def _normalize_mdbook_fragment(frag: str) -> str:
    """Collapse GitHub double-hyphen heading slugs (`--`) to mdBook single-hyphen slugs."""
    return re.sub(r"-{2,}", "-", frag)


def rewrite_links(content: str, src_rel: str, dest_map: dict[str, str], repo_root: Path) -> str:
    """Rewrite repo-relative links to mdBook destinations or GitHub blob URLs."""
    src_dir = Path(src_rel).parent

    def _map_target(target: str, *, html_attr: bool = False) -> str:
        if target.startswith(("http://", "https://", "mailto:", "assets/")):
            return target
        if target.startswith("#"):
            return "#" + _normalize_mdbook_fragment(target[1:])
        base, frag = (target.split("#", 1) + [""])[:2]
        frag_suffix = f"#{_normalize_mdbook_fragment(frag)}" if frag else ""
        if not base:
            return target
        resolved = (src_dir / base).as_posix()
        # Normalize '..' segments without escaping repo root.
        parts: list[str] = []
        for part in resolved.split("/"):
            if part in ("", "."):
                continue
            if part == ".." and parts:
                parts.pop()
            else:
                parts.append(part)
        norm = "/".join(parts)
        if norm in dest_map:
            dest = dest_map[norm]
            if html_attr and dest.endswith(".md"):
                dest = dest[:-3] + ".html"
            return dest + frag_suffix
        if norm.startswith("assets/"):
            return norm + frag_suffix
        candidate = repo_root / norm
        if candidate.is_dir():
            if f"{norm}/README.md" in dest_map:
                dest = dest_map[f"{norm}/README.md"]
                if html_attr and dest.endswith(".md"):
                    dest = dest[:-3] + ".html"
                return dest + frag_suffix
            return f"{REPO_TREE_BASE}/{norm}{frag_suffix}"
        if candidate.is_file():
            return f"{REPO_BLOB_BASE}/{norm}{frag_suffix}"
        return target

    content = MD_LINK_RE.sub(lambda m: f"{m.group(1)}{_map_target(m.group(2))}{m.group(3)}", content)
    content = HREF_RE.sub(lambda m: f"{m.group(1)}{_map_target(m.group(2), html_attr=True)}{m.group(3)}", content)
    content = SRC_RE.sub(lambda m: f"{m.group(1)}{_map_target(m.group(2), html_attr=True)}{m.group(3)}", content)
    return content


def rewrite_llms_txt(content: str, dest_map: dict[str, str], repo_root: Path) -> str:
    """Rewrite repo-relative Markdown links in llms.txt to live GitHub Pages or GitHub blob URLs."""

    def _replace(m: re.Match[str]) -> str:
        prefix, target, suffix = m.group(1), m.group(2), m.group(3)
        if target.startswith(("http://", "https://", "mailto:", "#")):
            return m.group(0)
        norm = target.lstrip("./")
        if norm in dest_map:
            html_page = dest_map[norm].removesuffix(".md") + ".html"
            return f"{prefix}{SITE_BASE_URL}/{html_page}{suffix}"
        candidate = repo_root / norm
        if candidate.is_dir():
            return f"{prefix}{REPO_TREE_BASE}/{norm}{suffix}"
        if candidate.is_file():
            return f"{prefix}{REPO_BLOB_BASE}/{norm}{suffix}"
        return m.group(0)

    rewritten = MD_LINK_RE.sub(_replace, content)
    if "/api-explorer/" not in rewritten:
        rewritten = rewritten.rstrip() + (
            "\n\n## Interactive & Machine-Readable Endpoints\n\n"
            f"- [Interactive API Explorer (Scalar)]({SITE_BASE_URL}/api-explorer/): Live OpenAPI 3.1 endpoint explorer.\n"
            f"- [OpenAPI 3.1 JSON]({SITE_BASE_URL}/openapi.json): Machine-readable HTTP API schema.\n"
            f"- [Rustdoc Crate API]({SITE_BASE_URL}/api/): Generated Rust workspace API documentation.\n"
        )
    return rewritten


def build_site(repo_root: Path, output_dir: Path, rustdoc_dir: Path | None) -> None:
    dest_map = {rel_src: dest_name for _, rel_src, dest_name in PAGES}
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
            raw = src_path.read_text(encoding="utf-8")
            rewritten = rewrite_links(raw, rel_src, dest_map, repo_root)
            (src_dir / dest_name).write_text(rewritten, encoding="utf-8")
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

    llms_src = repo_root / "llms.txt"
    if llms_src.is_file():
        llms_out = rewrite_llms_txt(llms_src.read_text(encoding="utf-8"), dest_map, repo_root)
        (output_dir / "llms.txt").write_text(llms_out, encoding="utf-8")

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
