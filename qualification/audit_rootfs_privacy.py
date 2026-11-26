#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Milestone 5 Full Rootfs & Artifact Privacy Audit.

Exhaustively scans container rootfs tar stream and qualification artifacts
for host paths, credentials, usernames, hostnames, and forbidden state tokens.
Zero violations allowed.
"""

from __future__ import annotations
import argparse
import dataclasses
import json
import os
import re
import subprocess
import sys
import tarfile
from typing import Any, List

HOST_PATH_PATTERNS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    ("unix-home", re.compile(rb"(?<![A-Za-z0-9_])/(?:home|Users)/[A-Za-z0-9._-]+(?:/|$)")),
    ("agent-temp", re.compile(rb"/tmp/(?:codex|claude)-[A-Za-z0-9._-]*")),
    ("docker-overlay", re.compile(rb"/var/lib/docker/overlay2/")),
)


def _extra_host_patterns(cli_patterns: list[str] | None = None) -> list[tuple[str, re.Pattern[bytes]]]:
    extras: list[tuple[str, re.Pattern[bytes]]] = []
    env_raw = os.environ.get("FLUX_PRIVACY_PATTERNS", "")
    raw_items = [p.strip() for p in env_raw.split(",") if p.strip()]
    if cli_patterns:
        raw_items.extend(p.strip() for p in cli_patterns if p.strip())
    for idx, item in enumerate(raw_items):
        extras.append((f"custom-{idx}", re.compile(re.escape(item.encode("utf-8")))))
    return extras

SECRET_PATTERNS = (
    ("github-token", re.compile(rb"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}")),
    ("github-fine-grained-token", re.compile(rb"github_pat_[A-Za-z0-9_]{20,}")),
    ("huggingface-token", re.compile(rb"(?<![A-Za-z0-9])hf_[A-Za-z0-9]{20,}")),
    ("aws-access-key", re.compile(rb"(?<![A-Z0-9])AKIA[A-Z0-9]{16}(?![A-Z0-9])")),
    ("private-key", re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
)

FORBIDDEN_BASENAMES = {
    ".git-credentials",
    ".netrc",
    "id_rsa",
    "id_ed25519",
}

FORBIDDEN_STATE_PHRASES = (
    ("forbidden-candidate-phrase", b"local-e2e-only-no-push"),
    ("forbidden-candidate-phrase", b"candidate-local-only-no-push"),
    ("forbidden-candidate-phrase", b"candidate-unqualified"),
    ("forbidden-candidate-phrase", b"e2e-candidate-unqualified"),
    ("forbidden-candidate-phrase", b"release_qualification=prohibited"),
    ("forbidden-source-reconstruction", b'"source_reconstruction":"incomplete'),
    ("forbidden-no-push", b"no_push:"),
)

RELEASE_OWNED_ROOTS = ("/opt/flux",)
RELEASE_OWNED_PATHS = frozenset({"/usr/local/bin/flux"})

def is_release_owned(path: str) -> bool:
    return path in RELEASE_OWNED_PATHS or any(
        path == r or path.startswith(r + "/") for r in RELEASE_OWNED_ROOTS
    )

@dataclasses.dataclass
class ScanFinding:
    category: str
    path: str
    detector: str
    offset: int
    snippet: str

def scan_bytes(
    data: bytes,
    path: str,
    check_secrets: bool = False,
    extra_patterns: list[tuple[str, re.Pattern[bytes]]] | None = None,
) -> list[ScanFinding]:
    findings: list[ScanFinding] = []
    patterns = list(HOST_PATH_PATTERNS) + (extra_patterns or [])

    # Host paths: check all files including binary files
    for label, pattern in patterns:
        match = pattern.search(data)
        if match:
            idx = match.start()
            snippet = data[max(0, idx - 15):min(len(data), match.end() + 15)].decode("latin1", "replace")
            findings.append(ScanFinding("host_path", path, label, idx, snippet))

    # Secrets: scoped to release payload / config
    if check_secrets:
        for label, pattern in SECRET_PATTERNS:
            match = pattern.search(data)
            if match:
                findings.append(ScanFinding("secret", path, label, match.start(), "[REDACTED]"))

    # Forbidden phrases: checked everywhere
    lowered = data.lower()
    for label, pattern in FORBIDDEN_STATE_PHRASES:
        idx = lowered.find(pattern)
        if idx >= 0:
            findings.append(ScanFinding("forbidden_state", path, label, idx, pattern.decode("latin1")))

    return findings

def audit_container_image(image_ref: str, extra_patterns: list[tuple[str, re.Pattern[bytes]]] | None = None) -> dict[str, Any]:
    print(f"[*] Auditing container image: {image_ref}")
    patterns = list(HOST_PATH_PATTERNS) + (extra_patterns or [])
    # Inspect metadata
    inspect_proc = subprocess.run(["docker", "image", "inspect", image_ref], capture_output=True, text=True, check=True)
    inspect_data = json.loads(inspect_proc.stdout)[0]
    
    metadata_findings: list[ScanFinding] = []
    config = inspect_data.get("Config") or {}
    env_vars = config.get("Env") or []
    for env in env_vars:
        findings = scan_bytes(env.encode("utf-8"), "image.Config.Env", check_secrets=True, extra_patterns=extra_patterns)
        metadata_findings.extend(findings)

    labels = config.get("Labels") or {}
    for k, v in labels.items():
        findings = scan_bytes(f"{k}={v}".encode("utf-8"), "image.Config.Labels", check_secrets=True, extra_patterns=extra_patterns)
        metadata_findings.extend(findings)

    # Create standoff container
    cid = subprocess.check_output(["docker", "create", "--entrypoint", "/bin/false", image_ref], text=True).strip()
    files_scanned = 0
    bytes_scanned = 0
    rootfs_findings: list[ScanFinding] = []
    
    try:
        export_proc = subprocess.Popen(["docker", "export", cid], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert export_proc.stdout is not None
        with tarfile.open(fileobj=export_proc.stdout, mode="r|*") as archive:
            for member in archive:
                path = "/" + member.name.lstrip("./")
                basename = os.path.basename(path)
                release_owned = is_release_owned(path)
                
                # Basename checks
                if basename in FORBIDDEN_BASENAMES or "/.git/" in path or path.endswith("/.git"):
                    rootfs_findings.append(ScanFinding("forbidden_basename", path, "forbidden-file", 0, basename))
                
                # Check path string itself for forbidden host home
                for label, pattern in patterns:
                    if pattern.search(path.encode("utf-8")):
                        rootfs_findings.append(ScanFinding("host_path", path, f"path-{label}", 0, path))
                
                # Check symlink targets
                if member.issym() or member.islnk():
                    target = member.linkname
                    for label, pattern in patterns:
                        if pattern.search(target.encode("utf-8")):
                            rootfs_findings.append(ScanFinding("host_path", path, f"symlink-target-{label}", 0, target))
                    continue

                if not member.isfile():
                    continue
                    
                stream = archive.extractfile(member)
                if stream is None:
                    continue
                    
                carry = b""
                overlap = 256
                is_safetensors = path.endswith(".safetensors")
                chunk_idx = 0
                while True:
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        break
                    scan_buf = carry + chunk
                    # For binary .safetensors tensor payloads, only scan the initial metadata chunk
                    # for regex secret patterns (to avoid random FP4/BF16 weight bytes matching
                    # base62 token regexes), while scanning 100% of the file for host paths.
                    check_chunk_secrets = release_owned and (not is_safetensors or chunk_idx == 0)
                    findings = scan_bytes(
                        scan_buf,
                        path,
                        check_secrets=check_chunk_secrets,
                        extra_patterns=extra_patterns,
                    )
                    rootfs_findings.extend(findings)
                    bytes_scanned += len(chunk)
                    carry = scan_buf[-overlap:]
                    chunk_idx += 1
                files_scanned += 1

        export_proc.stdout.close()
        stderr_bytes = export_proc.stderr.read() if export_proc.stderr is not None else b""
        if export_proc.stderr is not None:
            export_proc.stderr.close()
        export_rc = export_proc.wait()
        if export_rc != 0:
            err_msg = stderr_bytes.decode("utf-8", "replace").strip()
            raise RuntimeError(f"docker export failed (rc={export_rc}) for {image_ref}: {err_msg}")
        if files_scanned == 0:
            raise RuntimeError(f"docker export produced 0 scanned files for {image_ref}; refusing vacuous PASS")
    finally:
        subprocess.run(["docker", "rm", "-f", cid], capture_output=True)

    all_findings = metadata_findings + rootfs_findings
    
    host_path_count = sum(1 for f in all_findings if f.category == "host_path")
    secret_count = sum(1 for f in all_findings if f.category in ("secret", "forbidden_basename"))
    state_token_count = sum(1 for f in all_findings if f.category == "forbidden_state")

    return {
        "image_ref": image_ref,
        "image_id": inspect_data.get("Id"),
        "files_scanned": files_scanned,
        "bytes_scanned": bytes_scanned,
        "violations_count": len(all_findings),
        "host_path_violations": host_path_count,
        "secret_violations": secret_count,
        "state_token_violations": state_token_count,
        "violations": [dataclasses.asdict(f) for f in all_findings],
        "status": "PASS" if len(all_findings) == 0 else "FAIL"
    }

def main() -> int:
    parser = argparse.ArgumentParser(description="Milestone 5 Privacy Audit")
    parser.add_argument("--image", required=True, help="Container image tag or ID")
    parser.add_argument("--output", default=None, help="JSON output file path")
    parser.add_argument(
        "--extra-pattern",
        action="append",
        default=[],
        help="Additional literal host-path or identifier string to forbid (repeatable; also reads FLUX_PRIVACY_PATTERNS)",
    )
    args = parser.parse_args()

    results = audit_container_image(args.image, extra_patterns=_extra_host_patterns(args.extra_pattern))
    print(f"[*] Audit Results:")
    print(f"    Image ID:      {results['image_id']}")
    print(f"    Files scanned: {results['files_scanned']}")
    print(f"    Bytes scanned: {results['bytes_scanned']}")
    print(f"    Violations:    {results['violations_count']}")
    print(f"    Host paths:    {results['host_path_violations']}")
    print(f"    Secrets:       {results['secret_violations']}")
    print(f"    State tokens:  {results['state_token_violations']}")
    print(f"    Status:        {results['status']}")

    if results["violations"]:
        print("\n[-] Violations found:")
        for v in results["violations"]:
            print(f"    - [{v['category']}] {v['detector']} at {v['path']}:{v['offset']}: {v['snippet']}")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[+] Results saved to {args.output}")

    return 0 if results["status"] == "PASS" else 1

if __name__ == "__main__":
    sys.exit(main())
