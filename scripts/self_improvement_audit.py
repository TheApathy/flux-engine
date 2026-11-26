#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Weekly Engineering Health & Self-Improvement Audit for Flux Engine.

Collects objective engineering, security, formal-verification, and supply-chain
metrics and writes a structured Markdown scorecard to stdout (and optionally
to $GITHUB_STEP_SUMMARY) so regressions and improvement opportunities surface
automatically every week.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCORECARD_URL = "https://api.scorecard.dev/projects/github.com/TheApathy/flux-engine"


def fetch_scorecard() -> tuple[str, list[str]]:
    try:
        req = urllib.request.Request(SCORECARD_URL, headers={"User-Agent": "flux-engine-health"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        overall = f"{data.get('score', '?')}/10"
        gaps = []
        for chk in sorted(data.get("checks", []), key=lambda c: (c.get("score", -1), c.get("name", ""))):
            s = chk.get("score", -1)
            if 0 <= s < 8:
                gaps.append(f"- `{chk['name']}`: **{s}/10** — {chk.get('reason', '')}")
        return overall, gaps
    except Exception as exc:
        return f"unavailable ({exc})", []


def check_unpinned_dockerfiles() -> list[str]:
    unpinned: list[str] = []
    for base in (ROOT / "docker", ROOT / "repro-kit", ROOT / ".clusterfuzzlite"):
        if not base.exists():
            continue
        for df in sorted(base.rglob("Dockerfile*")):
            if not df.is_file():
                continue
            for idx, line in enumerate(df.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                if re.match(r"^\s*FROM\s+", line) and "@sha256:" not in line:
                    unpinned.append(f"{df.relative_to(ROOT)}:{idx}")
    return unpinned


def count_large_rust_files(threshold: int = 500) -> tuple[int, list[tuple[int, str]]]:
    large: list[tuple[int, str]] = []
    for rs in sorted((ROOT / "crates").rglob("*.rs")):
        lines = sum(1 for _ in rs.open("r", encoding="utf-8", errors="ignore"))
        if lines > threshold:
            large.append((lines, str(rs.relative_to(ROOT))))
    large.sort(reverse=True)
    return len(large), large[:10]


def run_formal_check(script: str) -> str:
    proc = subprocess.run(
        ["python3", "-B", str(ROOT / script)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return f"FAIL (exit {proc.returncode})"
    try:
        payload = json.loads(proc.stdout)
        if "total_proofs" in payload:
            return f"PASS ({payload['total_proofs']} Z3 UNSAT proofs)"
        if "reachable_states" in payload:
            return f"PASS ({payload['reachable_states']:,} states, {payload['deadlocks']} deadlocks)"
    except Exception:
        pass
    return "PASS"


def main() -> int:
    score, scorecard_gaps = fetch_scorecard()
    unpinned_df = check_unpinned_dockerfiles()
    large_count, top_large = count_large_rust_files(500)
    z3_status = run_formal_check("repro-kit/formal/verify_nextgen_invariants.py")
    tla_status = run_formal_check("repro-kit/formal/verify_scheduler_tla_model.py")

    lines = [
        "# Flux Engine — Weekly Engineering Health & Self-Improvement Audit",
        "",
        "| Signal | Status |",
        "|---|---|",
        f"| **OpenSSF Scorecard** | `{score}` |",
        f"| **Unpinned Dockerfiles** | `{len(unpinned_df)}` |",
        f"| **Z3 SMT Formal Proofs** | {z3_status} |",
        f"| **TLA+ BFS Scheduler Model** | {tla_status} |",
        f"| **Rust files > 500 LoC** | `{large_count}` |",
        "",
    ]

    if scorecard_gaps:
        lines.append("## OpenSSF Scorecard Improvement Targets (< 8/10)")
        lines.extend(scorecard_gaps)
        lines.append("")

    if unpinned_df:
        lines.append("## Unpinned Dockerfiles Found")
        lines.extend(f"- `{item}`" for item in unpinned_df)
        lines.append("")

    if top_large:
        lines.append("## Top Refactoring Candidates (> 500 LoC)")
        lines.extend(f"- `{path}` ({cnt} lines)" for cnt, path in top_large)
        lines.append("")

    report = "\n".join(lines) + "\n"
    print(report)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write(report)
    return 1 if unpinned_df or z3_status.startswith("FAIL") or tla_status.startswith("FAIL") else 0


if __name__ == "__main__":
    raise SystemExit(main())
