#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
Flux Engine — Zero-Tolerance Security & Sanitization Verification Gate

Scans tracked/repository files under the Flux Engine root directory for:
  1. Local POSIX usernames and raw home directory paths (/home/<user>, /Users/<user>)
  2. Known leaked internal identifiers, hostnames, and broken sanitizer placeholders
  3. GitHub, Hugging Face, OpenAI/Anthropic, or AWS tokens/keys
  4. Cryptographic private key headers (BEGIN ... PRIVATE KEY)
  5. URL-embedded basic auth credentials
  6. Prohibited credential/key files (.env*, *.key, *.pem, *.p12, *.credentials*)

Exits 0 only if 100% of scanned files are clean.
"""

from __future__ import annotations

import argparse
import getpass
import os
import re
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Pattern, Tuple


@dataclass(frozen=True)
class LeakRule:
    rule_id: str
    description: str
    regex: Pattern[str]
    only_prefixes: Tuple[str, ...] = ()


@dataclass(frozen=True)
class LeakFinding:
    relative_path: str
    line_number: int
    rule_id: str
    description: str


SKIP_DIRS = {
    ".git",
    "target",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}

SKIP_FILENAMES = {
    ".git",  # Git worktree pointer file
}

SKIP_DIR_PREFIXES = ("target-",)
SKIP_DIR_SUFFIXES = ("-target",)

# Self-skip so intentional regex definitions in privacy scanners do not trigger false positives.
SELF_SKIP_PATHS = {
    "scripts/verify_no_leaks.py",
    "qualification/audit_rootfs_privacy.py",
}

# Unit tests whose purpose is testing secret/URL redaction with synthetic dummy tokens.
# They are still scanned for real usernames, hostnames, local paths, and internal dirs.
MOCK_SECRET_TEST_PATHS = {
    "repro-kit/tests/test_debug_bundle_launcher.py",
    "repro-kit/tests/test_flux_lens.py",
    "repro-kit/tests/test_image_labels.py",
    "crates/spark-server/src/api/disagg.rs",
}

MOCK_SECRET_RULE_IDS = {
    "TIER0-GITHUB-CLASSIC-TOKEN",
    "TIER0-GITHUB-FINE-PAT",
    "TIER0-HF-TOKEN",
    "TIER0-API-SECRET-KEY",
    "TIER0-AWS-ACCESS-KEY",
    "TIER0-URL-BASIC-AUTH",
}

OPERATIONAL_PREFIXES = (
    ".github/",
    "crates/",
    "docker/",
    "scripts/",
    "tools/",
    "recipes/",
)

FORBIDDEN_FILENAME_PATTERNS: Tuple[Tuple[str, Pattern[str]], ...] = (
    ("FILE-001-DOTENV", re.compile(r"^\.env(?:\..+)?$|^.+\.env$", re.IGNORECASE)),
    ("FILE-002-PRIVKEY-EXT", re.compile(r"^.+?\.(?:key|pem|p12|pfx|jks|keystore)$", re.IGNORECASE)),
    ("FILE-003-SSH-KEY", re.compile(r"^id_(?:rsa|ed25519|ecdsa|dsa)(?:\..+)?$", re.IGNORECASE)),
    ("FILE-004-CREDENTIALS", re.compile(r"^.+?\.credentials.*$", re.IGNORECASE)),
)


def resolve_runtime_identities() -> Tuple[List[str], List[str], List[str]]:
    """Dynamically obtain local usernames, home paths, and hostnames without hardcoding."""
    users: List[str] = []
    for candidate in (
        getpass.getuser(),
        os.environ.get("USER", ""),
        os.environ.get("LOGNAME", ""),
        Path.home().name,
    ):
        cleaned = candidate.strip()
        if cleaned and cleaned not in {"root", "runner", "ubuntu", "user"} and cleaned not in users:
            users.append(cleaned)

    homes: List[str] = []
    for candidate in (str(Path.home()), os.environ.get("HOME", "")):
        cleaned = candidate.strip().rstrip("/")
        if cleaned and cleaned not in {"/", "/root"} and cleaned not in homes:
            homes.append(cleaned)

    hostnames: List[str] = []
    for candidate in (
        socket.gethostname(),
        socket.getfqdn(),
        os.uname().nodename if hasattr(os, "uname") else "",
    ):
        short_host = candidate.strip().split(".")[0]
        if (
            short_host
            and short_host.lower() not in {"localhost", "localdomain"}
            and short_host not in hostnames
        ):
            hostnames.append(short_host)

    return users, homes, hostnames


def build_leak_rules() -> List[LeakRule]:
    """Construct static + runtime regex rules for Tier 0 leak detection."""
    users, homes, hostnames = resolve_runtime_identities()
    rules: List[LeakRule] = []

    for home_dir in homes:
        rules.append(
            LeakRule(
                rule_id="TIER0-HOME-RUNTIME",
                description="Runtime local home directory path detected",
                regex=re.compile(re.escape(home_dir), re.IGNORECASE),
            )
        )

    rules.extend(
        [
            LeakRule(
                rule_id="TIER0-HOME-PATH",
                description="Unredacted /home/<user> path detected",
                regex=re.compile(
                    r"/home/(?!<|(?:a|u|user|alice|bob|op|private|test|runner|ubuntu|builder|spark)\b)[A-Za-z0-9._-]+"
                ),
            ),
            LeakRule(
                rule_id="TIER0-MACOS-HOME-PATH",
                description="Unredacted /Users/<user> path detected",
                regex=re.compile(r"/Users/(?!<|(?:name|user|runner)\b)[A-Za-z0-9._-]+"),
            ),
            LeakRule(
                rule_id="TIER0-DENYLIST-INTERNAL-DIR",
                description="Internal scratch directory reference detected",
                regex=re.compile(r"PREFILL4K"),
            ),
            LeakRule(
                rule_id="TIER0-UPSTREAM-CROSSWIRE",
                description="Stale upstream Avarok-Cybersecurity/atlas reference in operational path",
                regex=re.compile(r"Avarok-Cybersecurity/atlas"),
                only_prefixes=OPERATIONAL_PREFIXES,
            ),
            LeakRule(
                rule_id="TIER0-BROKEN-PLACEHOLDER",
                description="Unexpanded sanitizer placeholder detected",
                regex=re.compile(r"<(?:ATLAS_ROOT|HOME_ROOT|FLUX_FOUNDRY_ROOT)>"),
            ),
        ]
    )

    # Optional maintainer-supplied extra denylist terms (comma-separated, never committed).
    extra_terms = [
        item.strip()
        for item in os.environ.get("FLUX_PRIVACY_PATTERNS", "").split(",")
        if item.strip()
    ]
    for term in extra_terms:
        rules.append(
            LeakRule(
                rule_id="TIER0-EXTRA-PATTERN",
                description="Configured private denylist pattern detected",
                regex=re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE),
            )
        )

    for user in users:
        rules.append(
            LeakRule(
                rule_id="TIER0-LOCAL-USERNAME",
                description="Local POSIX username detected",
                regex=re.compile(rf"\b{re.escape(user)}\b", re.IGNORECASE),
            )
        )

    for host in hostnames:
        rules.append(
            LeakRule(
                rule_id="TIER0-LOCAL-HOSTNAME",
                description="Local machine hostname detected",
                regex=re.compile(rf"\b{re.escape(host)}\b", re.IGNORECASE),
            )
        )

    # Token & Private Key patterns
    rules.extend(
        [
            LeakRule(
                rule_id="TIER0-GITHUB-CLASSIC-TOKEN",
                description="GitHub token pattern (ghp_/gho_/ghu_/ghs_/ghr_) detected",
                regex=re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
            ),
            LeakRule(
                rule_id="TIER0-GITHUB-FINE-PAT",
                description="GitHub fine-grained token pattern (github_pat_) detected",
                regex=re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
            ),
            LeakRule(
                rule_id="TIER0-HF-TOKEN",
                description="Hugging Face token pattern (hf_) detected",
                regex=re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
            ),
            LeakRule(
                rule_id="TIER0-API-SECRET-KEY",
                description="API secret key pattern (sk-) detected",
                regex=re.compile(r"\bsk-(?!test)[A-Za-z0-9_-]{20,}\b"),
            ),
            LeakRule(
                rule_id="TIER0-AWS-ACCESS-KEY",
                description="AWS access key ID pattern (AKIA/ASIA) detected",
                regex=re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{12,}\b"),
            ),
            LeakRule(
                rule_id="TIER0-PRIVATE-KEY",
                description="Cryptographic private key block header detected",
                regex=re.compile(r"-{5}BEGIN [A-Z0-9 ]*PRIVATE KEY-{5}"),
            ),
            LeakRule(
                rule_id="TIER0-URL-BASIC-AUTH",
                description="URL-embedded basic authentication credential detected",
                regex=re.compile(r"https?://[^/\s:@<]+:[^/\s:@>]+@"),
            ),
        ]
    )

    return rules


def _should_skip_dir(dirname: str) -> bool:
    return (
        dirname in SKIP_DIRS
        or dirname.startswith(SKIP_DIR_PREFIXES)
        or dirname.endswith(SKIP_DIR_SUFFIXES)
    )


def iter_files(root: Path) -> Iterable[Path]:
    """Yield all regular files under root, excluding VCS and build/cache directories."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not _should_skip_dir(d))
        for filename in sorted(filenames):
            if filename in SKIP_FILENAMES:
                continue
            filepath = Path(dirpath) / filename
            if filepath.is_file() and not filepath.is_symlink():
                yield filepath


def scan_repository(root: Path) -> Tuple[int, List[LeakFinding]]:
    """Scan all files under root and return (scanned_file_count, findings)."""
    rules = build_leak_rules()
    findings: List[LeakFinding] = []
    scanned_count = 0

    self_path = Path(__file__).resolve()
    for filepath in iter_files(root):
        rel_path = filepath.relative_to(root).as_posix()
        if rel_path in SELF_SKIP_PATHS or filepath.resolve() == self_path:
            continue
        scanned_count += 1

        # 1. Check filename against forbidden secret file patterns
        for file_rule_id, file_regex in FORBIDDEN_FILENAME_PATTERNS:
            if file_regex.match(filepath.name):
                if (
                    file_rule_id == "FILE-001-DOTENV"
                    and not filepath.name.startswith(".env")
                    and rel_path.startswith(
                        ("recipes/", "repro-kit/recipes/", "bench/", "qualification/accuracy/")
                    )
                ):
                    continue
                findings.append(
                    LeakFinding(
                        relative_path=rel_path,
                        line_number=0,
                        rule_id=file_rule_id,
                        description=f"Prohibited credential/secret file present: {filepath.name}",
                    )
                )

        # 2. Check file contents line by line
        try:
            raw_bytes = filepath.read_bytes()
        except OSError as exc:
            findings.append(
                LeakFinding(
                    relative_path=rel_path,
                    line_number=0,
                    rule_id="SCAN-READ-ERROR",
                    description=f"Unable to read file ({exc.__class__.__name__})",
                )
            )
            continue

        is_mock_secret_test = rel_path in MOCK_SECRET_TEST_PATHS
        text = raw_bytes.decode("utf-8", errors="replace")
        for line_idx, line in enumerate(text.splitlines(), start=1):
            for rule in rules:
                if rule.only_prefixes and not rel_path.startswith(rule.only_prefixes):
                    continue
                if is_mock_secret_test and rule.rule_id in MOCK_SECRET_RULE_IDS:
                    continue
                if rule.regex.search(line):
                    findings.append(
                        LeakFinding(
                            relative_path=rel_path,
                            line_number=line_idx,
                            rule_id=rule.rule_id,
                            description=rule.description,
                        )
                    )

    return scanned_count, findings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify that no credentials, local paths, usernames, or hostnames exist under the repository root."
    )
    default_root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help="Root directory to scan (defaults to repository root).",
    )
    args = parser.parse_args()

    target_root = args.root.resolve()
    if not target_root.is_dir():
        print(f"[FAIL] Target root directory does not exist: {target_root}", file=sys.stderr)
        return 2

    scanned_count, findings = scan_repository(target_root)

    if findings:
        print(
            f"[FAIL] verify_no_leaks: {len(findings)} security/sanitization violation(s) found across {scanned_count} scanned file(s):",
            file=sys.stderr,
        )
        for finding in findings:
            loc = f"{finding.relative_path}:{finding.line_number}" if finding.line_number > 0 else finding.relative_path
            print(f"  - [{finding.rule_id}] {loc} -> {finding.description}", file=sys.stderr)
        return 1

    print(
        f"[PASS] verify_no_leaks: {scanned_count} file(s) scanned under {target_root} — 100% clean (0 leaks)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
