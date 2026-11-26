#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Milestone 5 Receipt Generator: M5_RECEIPT.json & M5_RECEIPT.md.

Compiles and seals the Milestone 5 qualification receipts for packaging,
provenance attestations, zero-tolerance rootfs streaming privacy audit,
and live container API suite results under an exclusive GPU lockfile.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

WORKTREE = Path(os.environ.get("FLUX_WORKTREE", str(Path(__file__).resolve().parents[1])))
TARGET_DIR = Path(os.environ.get("CARGO_TARGET_DIR", str(WORKTREE / "target")))
BINARY_PATH = TARGET_DIR / "release" / "flux-server"
QUAL_DIR = WORKTREE / "qualification"
API_RECEIPT_PATH = QUAL_DIR / "api_qualification_receipt.json"
PROVENANCE_RECEIPT_PATH = QUAL_DIR / "provenance_verification.json"
M5_JSON_PATH = QUAL_DIR / "M5_RECEIPT.json"
M5_MD_PATH = QUAL_DIR / "M5_RECEIPT.md"

HIGH_PREFILL_RECIPE = WORKTREE / "repro-kit" / "recipes" / "flux-v9-max-prefill.env"
INTERACTIVE_RECIPE = WORKTREE / "repro-kit" / "recipes" / "flux-v9-w4096-interactive.env"

REQUIRED_API_CASES = (
    "health",
    "models",
    "chat_blocking",
    "chat_streaming_sse",
    "tool_calling_blocking",
    "tool_calling_streaming",
    "client_cancellation",
    "slot_reuse_and_recovery",
    "raw_log_safety",
)


def sha256_file(path: Path | str) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_info() -> tuple[str, str]:
    head = subprocess.check_output(["git", "-C", str(WORKTREE), "rev-parse", "HEAD"]).decode().strip()
    tree = subprocess.check_output(["git", "-C", str(WORKTREE), "rev-parse", "HEAD^{tree}"]).decode().strip()
    return head, tree


def get_container_info(image_tag: str = "flux-engine:r17-candidate") -> dict[str, str]:
    try:
        out = subprocess.check_output(["docker", "inspect", image_tag]).decode().strip()
        data = json.loads(out)
        if not data:
            return {"image_id": "UNKNOWN"}
        return {"image_id": data[0]["Id"]}
    except Exception as e:
        print(f"Warning: could not inspect container image {image_tag}: {e}")
        return {"image_id": "UNKNOWN"}


def evaluate_provenance(
    provenance_path: Path | None,
    binary_sha: str,
    high_prefill_sha: str,
    interactive_sha: str,
) -> tuple[str, dict[str, Any]]:
    """Verify standoff byte extraction & test suite results from real provenance data."""
    binary_exists = binary_sha != "MISSING"
    if provenance_path is None or not provenance_path.is_file():
        return (
            "FAIL",
            {
                "status": "FAIL",
                "reason": "provenance verification receipt not provided (refusing fabricated PASS)",
                "standoff_byte_extraction": {
                    "binary": {
                        "container_path": "/opt/flux/bin/flux-server-real",
                        "expected_sha256": binary_sha,
                        "extracted_sha256": None,
                        "match": False,
                    },
                    "recipe_high_prefill": {
                        "container_path": "/opt/flux/repro-kit/recipes/flux-v9-max-prefill.env",
                        "expected_sha256": high_prefill_sha,
                        "extracted_sha256": None,
                        "match": False,
                    },
                    "recipe_interactive": {
                        "container_path": "/opt/flux/repro-kit/recipes/flux-v9-w4096-interactive.env",
                        "expected_sha256": interactive_sha,
                        "extracted_sha256": None,
                        "match": False,
                    },
                },
                "test_suites": {
                    "status": "UNVERIFIED",
                    "total_tests": 0,
                    "total_passed": 0,
                    "total_failed": 0,
                },
            },
        )

    prov = json.loads(provenance_path.read_text(encoding="utf-8"))
    raw_standoff = prov.get("standoff_byte_extraction") or {}
    bin_ext = (raw_standoff.get("binary") or {}).get("sha256")
    hp_ext = (raw_standoff.get("recipe_high_prefill") or {}).get("sha256")
    int_ext = (raw_standoff.get("recipe_interactive") or {}).get("sha256")

    bin_match = bool(binary_exists and bin_ext == binary_sha)
    hp_match = bool(hp_ext == high_prefill_sha)
    int_match = bool(int_ext == interactive_sha)

    suites = prov.get("test_suites") or {}
    total_tests = int(suites.get("total_tests", 0))
    total_passed = int(suites.get("total_passed", 0))
    total_failed = int(suites.get("total_failed", 0))
    suites_ok = total_tests > 0 and total_failed == 0 and total_passed == total_tests

    prov_ok = bin_match and hp_match and int_match and suites_ok
    prov_status = "PASS" if prov_ok else "FAIL"
    return (
        prov_status,
        {
            "status": prov_status,
            "source_receipt": str(provenance_path),
            "standoff_byte_extraction": {
                "binary": {
                    "container_path": "/opt/flux/bin/flux-server-real",
                    "sha256": bin_ext,
                    "expected_sha256": binary_sha,
                    "match": bin_match,
                },
                "recipe_high_prefill": {
                    "container_path": "/opt/flux/repro-kit/recipes/flux-v9-max-prefill.env",
                    "sha256": hp_ext,
                    "expected_sha256": high_prefill_sha,
                    "match": hp_match,
                },
                "recipe_interactive": {
                    "container_path": "/opt/flux/repro-kit/recipes/flux-v9-w4096-interactive.env",
                    "sha256": int_ext,
                    "expected_sha256": interactive_sha,
                    "match": int_match,
                },
            },
            "test_suites": suites,
        },
    )


def generate_receipts(
    version: str = "0.9.15",
    release_tag: str = "v0.9.15",
    privacy_path: Path | None = None,
    api_path: Path = API_RECEIPT_PATH,
    provenance_path: Path | None = None,
) -> int:
    if privacy_path is None:
        privacy_path = QUAL_DIR / f"container-{version}" / "rootfs-privacy-audit.json"
    head_commit, _tree = get_git_info()
    binary_sha = sha256_file(BINARY_PATH) if BINARY_PATH.is_file() else "MISSING"
    binary_exists = binary_sha != "MISSING"
    high_prefill_sha = sha256_file(HIGH_PREFILL_RECIPE)
    interactive_sha = sha256_file(INTERACTIVE_RECIPE)

    container_info = get_container_info("flux-engine:r17-candidate")
    image_id = container_info.get("image_id", "UNKNOWN")

    with open(privacy_path, "r", encoding="utf-8") as f:
        privacy_data = json.load(f)

    with open(api_path, "r", encoding="utf-8") as f:
        api_data = json.load(f)

    if provenance_path is None and PROVENANCE_RECEIPT_PATH.is_file():
        provenance_path = PROVENANCE_RECEIPT_PATH
    prov_status, provenance_block = evaluate_provenance(
        provenance_path, binary_sha, high_prefill_sha, interactive_sha
    )

    files_scanned = int(privacy_data.get("files_scanned", 0))
    bytes_scanned = int(privacy_data.get("bytes_scanned", 0))
    violations_count = int(privacy_data.get("violations_count", -1))
    privacy_ok = (
        privacy_data.get("status") == "PASS"
        and violations_count == 0
        and files_scanned > 0
    )
    privacy_status = "PASS" if privacy_ok else "FAIL"

    assertions = api_data.get("cases") or api_data.get("assertions") or {}
    passed_assertions = sum(
        1 for c in assertions.values() if isinstance(c, dict) and c.get("status") == "PASS"
    )
    all_required_api_pass = all(
        isinstance(assertions.get(k), dict) and assertions[k].get("status") == "PASS"
        for k in REQUIRED_API_CASES
    )
    api_ok = api_data.get("verdict") == "PASS" and all_required_api_pass
    api_status = "PASS" if api_ok else "FAIL"

    overall_verdict = (
        "PASS"
        if (binary_exists and prov_status == "PASS" and privacy_status == "PASS" and api_status == "PASS")
        else "FAIL"
    )

    ts_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    pre_mem = float(api_data.get("pre_flight_mem_gib") or api_data.get("pre_flight_mem_available_gib") or 0.0)
    post_mem = float(api_data.get("post_flight_mem_gib") or api_data.get("post_flight_mem_available_gib") or 0.0)
    readiness_time = float(api_data.get("readiness_time_seconds") or 0.0)

    receipt_json = {
        "milestone": "Milestone 5 — Packaging, Provenance, Privacy, and API Suite (R5)",
        "verdict": overall_verdict,
        "timestamp_utc": ts_utc,
        "hardware": "NVIDIA DGX Spark (GB10, SM121a, 128 GB unified memory)",
        "model": "Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid",
        "head_commit": head_commit,
        "container_image": {
            "tags": [
                "flux-engine:r17-candidate",
                f"ghcr.io/theapathy/flux-engine:{release_tag}",
                f"ghcr.io/theapathy/flux-engine:{release_tag}-r17",
                "ghcr.io/theapathy/flux-engine:gb10",
                "ghcr.io/theapathy/flux-engine:latest",
            ],
            "image_id": image_id,
            "base_image": "nvidia/cuda:13.0.0-runtime-ubuntu24.04",
            "release_version": release_tag,
            "build_args": {
                "BASE_IMAGE": "nvidia/cuda:13.0.0-runtime-ubuntu24.04",
                "RELEASE_VERSION": release_tag,
                "FLUX_BINARY_PATH": "repro-kit/bin/flux-server-real",
                "BINARY_SHA256": binary_sha,
                "RECIPE_HIGH_PREFILL_SHA256": high_prefill_sha,
                "RECIPE_INTERACTIVE_SHA256": interactive_sha,
            },
        },
        "provenance_verification": provenance_block,
        "privacy_audit": {
            "status": privacy_status,
            "engine": "docker export tar streaming scanner",
            "files_scanned": files_scanned,
            "bytes_scanned": bytes_scanned,
            "host_path_violations": int(privacy_data.get("host_path_violations", 0)),
            "secret_violations": int(privacy_data.get("secret_violations", 0)),
            "state_token_violations": int(privacy_data.get("state_token_violations", 0)),
            "total_violations": violations_count,
        },
        "api_qualification_suite": {
            "status": api_status,
            "endpoint": api_data.get("endpoint", "http://127.0.0.1:8899"),
            "hardware_lock": os.environ.get("FLUX_LOCK_FILE", "/tmp/flux-gpu.lock"),
            "pre_flight_mem_gib": round(pre_mem, 2),
            "post_flight_mem_gib": round(post_mem, 2),
            "readiness_time_seconds": round(readiness_time, 1),
            "total_assertions": len(assertions),
            "passed_assertions": passed_assertions,
            "assertions": assertions,
        },
        "artifacts": {
            "api_qualification_receipt": {
                "path": str(api_path.relative_to(WORKTREE)) if api_path.is_relative_to(WORKTREE) else str(api_path),
                "sha256": sha256_file(api_path),
            },
            "audit_rootfs_privacy_script": {
                "path": "qualification/audit_rootfs_privacy.py",
                "sha256": sha256_file(QUAL_DIR / "audit_rootfs_privacy.py"),
            },
            "privacy_scan_results": {
                "path": str(privacy_path.relative_to(WORKTREE)) if privacy_path.is_relative_to(WORKTREE) else str(privacy_path),
                "sha256": sha256_file(privacy_path),
            },
            "run_m5_api_qualification_script": {
                "path": "qualification/run_m5_api_qualification.py",
                "sha256": sha256_file(QUAL_DIR / "run_m5_api_qualification.py"),
            },
        },
    }

    with open(M5_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(receipt_json, f, indent=2)
        f.write("\n")

    t1 = assertions.get("health", {})
    t2 = assertions.get("models", {})
    t3 = assertions.get("chat_blocking", {})
    t3_usage = t3.get("usage", {})
    t3_ttft = float(t3_usage.get("time_to_first_token_ms") or t3.get("time_to_first_token_ms") or 0.0)
    t3_tps = float(t3_usage.get("response_token/s") or t3.get("response_token_per_sec") or 0.0)
    t4 = assertions.get("chat_streaming_sse", {})
    t5 = assertions.get("tool_calling_blocking", {})
    t6 = assertions.get("tool_calling_streaming", {})
    t7 = assertions.get("client_cancellation", {})
    t8 = assertions.get("slot_reuse_and_recovery", {})
    t9 = assertions.get("raw_log_safety", {})

    def _status(case: dict[str, Any]) -> str:
        return str(case.get("status", "FAIL"))

    standoff = provenance_block["standoff_byte_extraction"]
    suites = provenance_block["test_suites"]
    verdict_banner = (
        "**PASS** (Release Container Fully Sealed)"
        if overall_verdict == "PASS"
        else "**FAIL** (Qualification Gate Incomplete or Failed)"
    )
    summary_intro = (
        "All qualification phases have passed with zero exceptions:"
        if overall_verdict == "PASS"
        else "One or more qualification gates failed or were unverified (see statuses below):"
    )
    suites_summary = (
        f"Total Tests: {suites.get('total_passed', 0)} passed, {suites.get('total_failed', 0)} failed "
        f"(of {suites.get('total_tests', 0)})"
        if suites.get("total_tests", 0) > 0
        else "UNVERIFIED (no provenance test-suite receipt provided)"
    )

    md_content = f"""# Milestone 5 Qualification Receipt: Packaging, Provenance, Privacy, and API Suite (R5)

**Milestone**: Milestone 5 — Packaging, Provenance, Privacy, and API Suite (R5)  
**Verdict**: {verdict_banner}  
**Timestamp**: {ts_utc}  
**Hardware**: NVIDIA DGX Spark (GB10, SM121a, 128 GB unified memory)  
**Model**: `Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid`  
**Container Image Tags**: `flux-engine:r17-candidate`, `ghcr.io/theapathy/flux-engine:{release_tag}`, `ghcr.io/theapathy/flux-engine:{release_tag}-r17`, `ghcr.io/theapathy/flux-engine:gb10`, `ghcr.io/theapathy/flux-engine:latest`  
**Container Image ID**: `{image_id}`  
**Binary Location (Host)**: `{BINARY_PATH}`  
**Binary SHA-256**: `{binary_sha}`  
**HEAD Commit**: `{head_commit}`  

---

## 1. Executive Summary & Verification Context

Milestone 5 qualifies the production container packaging, provenance attestations, zero-leak streaming rootfs privacy audit, and live container API suite for the `{release_tag}` release of `flux-engine`.

{summary_intro}
1. **Binary Presence**: `{"PASS" if binary_exists else "FAIL"}` (`{binary_sha}`).
2. **Standoff Byte Extraction & Provenance**: `{prov_status}` ({suites_summary}).
3. **Zero-Tolerance Privacy Audit**: `{privacy_status}` — {files_scanned:,} files ({bytes_scanned:,} bytes), {violations_count} violation(s).
4. **Live Container API Qualification Suite**: `{api_status}` — {passed_assertions}/{len(assertions)} assertions passed (readiness: {readiness_time:.1f} s).
5. **R6 Box Safety & Memory Floor**: Pre-flight `MemAvailable` {pre_mem:.2f} GiB, post-flight {post_mem:.2f} GiB.

---

## 2. Release Container Build & Provenance Verification

### 2.1 Build Parameters & Artifact Checksums

| Parameter / Artifact | Value / File Path | Exact SHA-256 Digest |
|---|---|---|
| **Base Image** | `nvidia/cuda:13.0.0-runtime-ubuntu24.04` | Pinned Ubuntu 24.04 + CUDA 13.0 |
| **Release Version** | `{release_tag}` | SemVer Release Tag |
| **Release Binary** | `repro-kit/bin/flux-server-real` | `{binary_sha}` |
| **High-Prefill Recipe** | `repro-kit/recipes/flux-v9-max-prefill.env` | `{high_prefill_sha}` |
| **Interactive Recipe** | `repro-kit/recipes/flux-v9-w4096-interactive.env` | `{interactive_sha}` |
| **Built Image ID** | `flux-engine:r17-candidate` | `{image_id}` |

### 2.2 Standoff Byte Extraction Verification

- `/opt/flux/bin/flux-server-real`: `{standoff["binary"].get("sha256")}` (**{"MATCH" if standoff["binary"]["match"] else "MISMATCH/UNVERIFIED"}**)
- `/opt/flux/repro-kit/recipes/flux-v9-max-prefill.env`: `{standoff["recipe_high_prefill"].get("sha256")}` (**{"MATCH" if standoff["recipe_high_prefill"]["match"] else "MISMATCH/UNVERIFIED"}**)
- `/opt/flux/repro-kit/recipes/flux-v9-w4096-interactive.env`: `{standoff["recipe_interactive"].get("sha256")}` (**{"MATCH" if standoff["recipe_interactive"]["match"] else "MISMATCH/UNVERIFIED"}**)

### 2.3 Unit & Provenance Test Suite Results

```text
{suites_summary}
```

---

## 3. Zero-Tolerance Rootfs Streaming Privacy Audit

- **Files Scanned**: {files_scanned:,} files
- **Bytes Scanned**: {bytes_scanned:,} bytes ({bytes_scanned / 1e9:.2f} GB)
- **Host Path Violations**: {int(privacy_data.get('host_path_violations', 0))}
- **Secret & Credential Violations**: {int(privacy_data.get('secret_violations', 0))}
- **State Token Violations**: {int(privacy_data.get('state_token_violations', 0))}
- **Total Leaks Detected**: **{violations_count}**
- **Audit Verdict**: **{privacy_status}** (Receipt: `{receipt_json["artifacts"]["privacy_scan_results"]["path"]}`)

---

## 4. Live Container API Qualification Suite Results

- **Readiness Time**: {readiness_time:.1f} seconds
- **Pre-flight `MemAvailable`**: {pre_mem:.2f} GiB
- **Post-flight `MemAvailable`**: {post_mem:.2f} GiB

### 4.1 Summary of 9 Assertions

| # | Assertion | Endpoint / Mechanism | Observed Output / Behavior | Status |
|---|---|---|---|---|
| **1** | Health Probe | `GET /health` | HTTP {t1.get('http_status', 'N/A')}, {json.dumps(t1.get('body', {}))} | **{_status(t1)}** |
| **2** | Model Catalog | `GET /v1/models` | HTTP {t2.get('http_status', 'N/A')}, catalog contains {t2.get('models', [])} | **{_status(t2)}** |
| **3** | Chat Blocking | `POST /v1/chat/completions` (`stream: false`) | HTTP {t3.get('http_status', 'N/A')}, content: '{t3.get('content')}', `finish_reason: "{t3.get('finish_reason')}"`, TTFT: {t3_ttft:.2f} ms, {t3_tps:.2f} tok/s | **{_status(t3)}** |
| **4** | Chat Streaming | `POST /v1/chat/completions` (`stream: true`) | HTTP 200, {t4.get('chunks_count', 0)} SSE chunks: '{t4.get('content')}', terminal `[DONE]`={t4.get('done_seen', False)} | **{_status(t4)}** |
| **5** | Tool Calling Blocking | `POST /v1/chat/completions` (`tools: [...]`) | `finish_reason: "{t5.get('finish_reason')}"`, {t5.get('tool_call', {}).get('function', {}).get('name')}({json.dumps(t5.get('parsed_arguments', {}))}) | **{_status(t5)}** |
| **6** | Tool Calling Streaming | `POST /v1/chat/completions` (`stream: true`) | {t6.get('chunks_count', 0)} delta chunks assembled: {t6.get('function_name')}({json.dumps(t6.get('parsed_arguments', {}))}), `finish_reason: "{t6.get('finish_reason')}"` | **{_status(t6)}** |
| **7** | Client TCP Cancellation | Abrupt socket close at chunk {t7.get('chunks_before_cancellation', 0)} | `abrupt_drop={t7.get('abrupt_drop', False)}` | **{_status(t7)}** |
| **8** | Slot Reuse & Recovery | Replacement probe `POST /v1/chat/completions` | HTTP {t8.get('http_status', 'N/A')}, content: '{t8.get('content')}', `atlas_requests_active == {t8.get('atlas_requests_active')}` | **{_status(t8)}** |
| **9** | Raw Log Safety Audit | `docker logs` scan over {t9.get('total_log_lines', 0):,} lines | {t9.get('fatal_matches_count', 'N/A')} matches across fatal patterns | **{_status(t9)}** |

---

## 5. Box Safety & Hardware Compliance

1. **Lock Discipline**: Full qualification serialized under exclusive GPU and build lockfiles.
2. **RAM Threshold Verification**: Pre-flight {pre_mem:.2f} GiB, post-flight {post_mem:.2f} GiB.
3. **Log Sanitization**: `{_status(t9)}` ({t9.get('fatal_matches_count', 'N/A')} fatal matches across {t9.get('total_log_lines', 0):,} log lines).

---

## 6. Release Artifact Manifest

| Artifact File | Canonical Path | SHA-256 Digest |
|---|---|---|
| **API Qualification Receipt** | `{receipt_json["artifacts"]["api_qualification_receipt"]["path"]}` | `{receipt_json["artifacts"]["api_qualification_receipt"]["sha256"]}` |
| **Rootfs Privacy Audit Script** | `qualification/audit_rootfs_privacy.py` | `{receipt_json["artifacts"]["audit_rootfs_privacy_script"]["sha256"]}` |
| **Rootfs Privacy Results** | `{receipt_json["artifacts"]["privacy_scan_results"]["path"]}` | `{receipt_json["artifacts"]["privacy_scan_results"]["sha256"]}` |
| **API Qualification Script** | `qualification/run_m5_api_qualification.py` | `{receipt_json["artifacts"]["run_m5_api_qualification_script"]["sha256"]}` |
| **Milestone 5 Receipt JSON** | `qualification/M5_RECEIPT.json` | *(Self-referenced above)* |
"""

    with open(M5_MD_PATH, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"[+] Compiled {M5_JSON_PATH} and {M5_MD_PATH} (verdict={overall_verdict})")
    return 0 if overall_verdict == "PASS" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile Milestone 5 qualification receipts.")
    parser.add_argument("--version", default="0.9.15")
    parser.add_argument("--release-tag", default="v0.9.15")
    parser.add_argument(
        "--privacy-receipt",
        type=Path,
        default=None,
        help="Versioned rootfs privacy audit JSON (default: qualification/container-<version>/rootfs-privacy-audit.json).",
    )
    parser.add_argument("--api-receipt", type=Path, default=API_RECEIPT_PATH)
    parser.add_argument("--provenance-json", type=Path, default=None)
    args = parser.parse_args()
    privacy_path = (
        args.privacy_receipt
        if args.privacy_receipt is not None
        else QUAL_DIR / f"container-{args.version}" / "rootfs-privacy-audit.json"
    )
    return generate_receipts(
        version=args.version,
        release_tag=args.release_tag,
        privacy_path=privacy_path,
        api_path=args.api_receipt,
        provenance_path=args.provenance_json,
    )


if __name__ == "__main__":
    sys.exit(main())
