# Milestone 5 Qualification Receipt: Packaging, Provenance, Privacy, and API Suite (R5)

**Milestone**: Milestone 5 — Packaging, Provenance, Privacy, and API Suite (R5)  
**Verdict**: **PASS** (Release Container Fully Sealed)  
**Timestamp**: 2026-10-25T17:01:49Z  
**Hardware**: NVIDIA DGX Spark (GB10, SM121a, 128 GB unified memory)  
**Model**: `Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid`  
**Container Image Tags**: `flux-engine:r17-candidate`, `ghcr.io/theapathy/flux-engine:v0.9.14`, `ghcr.io/theapathy/flux-engine:v0.9.14-r17`, `ghcr.io/theapathy/flux-engine:gb10`, `ghcr.io/theapathy/flux-engine:latest`  
**Container Image ID**: `sha256:7bf7806eae0d7cd7cbad5c8c345ad91215ce7033c57506a493785bf508ae157c`  
**Binary Location (Host)**: `target/release/flux-server`  
**Binary SHA-256**: `a3b49aa3e1e3eb41b20bc28c4193026522865ee083d2d3e0cc1135d333da8b5e`  
**HEAD Commit**: `b47b9fe6b0c5826df769277bb306f5b41e2372c7`  

---

## 1. Executive Summary & Verification Context

Milestone 5 qualifies the production container packaging, provenance attestations, zero-leak streaming rootfs privacy audit, and live container API suite for the `v0.9.14` release of `flux-engine`.

All qualification phases have passed with zero exceptions:
1. **Container Image Build**: Docker release container image `flux-engine:r17-candidate` (`ghcr.io/theapathy/flux-engine:v0.9.14`) built with verified build arguments, passing in-container provenance validation during assembly.
2. **Standoff Byte Extraction & Provenance**: Standoff extracted binary and recipes from the built image match host input SHA-256 digests bit-for-bit. 100% of unit tests pass (41/41 across 3 test suites).
3. **Zero-Tolerance Privacy Audit**: Streaming audit of 9,455 files (2,955,569,430 bytes) from `docker export` reveals strictly 0 leaks of host home-directory paths, credentials, machine names, or candidate tokens.
4. **Live Container API Qualification Suite**: Live container launched under exclusive `$FLUX_LOCK_FILE` on `http://127.0.0.1:8899`, reaching `/health` readiness in 180.1 seconds. All 9 live API assertions passed (readiness, model catalog, blocking chat, streaming SSE chat, blocking tool calling, streaming delta tool calling, client TCP cancellation, slot recovery with `atlas_requests_active == 0.0`, and zero fatal raw log patterns across 9,500 lines).
5. **R6 Box Safety & Memory Floor**: `MemAvailable` held strictly $\ge 100.0\text{ GiB}$ before container launch (105.19 GiB) and fully restored post-flight (105.13 GiB). Hardware lock cleanly released.

---

## 2. Release Container Build & Provenance Verification

### 2.1 Build Parameters & Artifact Checksums

The release container was assembled using the pinned recipe configs and verified host binary:

| Parameter / Artifact | Value / File Path | Exact SHA-256 Digest |
|---|---|---|
| **Base Image** | `nvidia/cuda:13.0.0-runtime-ubuntu24.04` | Pinned Ubuntu 24.04 + CUDA 13.0 |
| **Release Version** | `v0.9.14` | SemVer Release Tag |
| **Release Binary** | `repro-kit/bin/flux-server-real` | `a3b49aa3e1e3eb41b20bc28c4193026522865ee083d2d3e0cc1135d333da8b5e` |
| **High-Prefill Recipe** | `repro-kit/recipes/flux-v9-max-prefill.env` | `00ef1a839bcfd2d2b65799846efe3441dcd267857354e48e5f0335f2f54aa3c6` *(historical `v0.9.14` image build; live repo post-sanitization: `2942f6ae474c84a175bb3969f2d6eedad43c60bd917162f76ad443f706420889`)* |
| **Interactive Recipe** | `repro-kit/recipes/flux-v9-w4096-interactive.env` | `dce6f9e14fc03f908eab406bfbb4f24755bf4447e198faec16dc37bc5f861869` *(historical `v0.9.14` image build; live repo post-sanitization: `318b36a75b485a4b73cc20b4b13285d21a1c66327f1de5a95d574c6a03039663`)* |
| **Built Image ID** | `flux-engine:r17-candidate` | `sha256:7bf7806eae0d7cd7cbad5c8c345ad91215ce7033c57506a493785bf508ae157c` |

### 2.2 Standoff Byte Extraction Verification

All release artifacts were copied out of a running standoff container instance via `docker cp` and hashed independently:
- `/opt/flux/bin/flux-server-real`: `a3b49aa3e1e3eb41b20bc28c4193026522865ee083d2d3e0cc1135d333da8b5e` (**MATCH**)
- `/opt/flux/repro-kit/recipes/flux-v9-max-prefill.env`: `00ef1a839bcfd2d2b65799846efe3441dcd267857354e48e5f0335f2f54aa3c6` (**MATCH**)
- `/opt/flux/repro-kit/recipes/flux-v9-w4096-interactive.env`: `dce6f9e14fc03f908eab406bfbb4f24755bf4447e198faec16dc37bc5f861869` (**MATCH**)

### 2.3 Unit & Provenance Test Suite Results

```text
repro-kit/tests/test_image_provenance.py: 24 passed (100%)
repro-kit/tests/test_r17_release_binding_successor.py: 12 passed (100%)
repro-kit/tests/test_release_metadata.py: 5 passed (100%)
Total Tests: 41 passed, 0 failed (100% OK)
```

---

## 3. Zero-Tolerance Rootfs Streaming Privacy Audit

An exhaustive streaming privacy scan was performed over `docker export` of `flux-engine:r17-candidate` without staging tar archives to disk:

- **Files Scanned**: 9,455 files
- **Bytes Scanned**: 2,955,569,430 bytes (2.95 GB)
- **Host Path Violations (host home-directory paths)**: 0
- **Secret & Credential Violations**: 0
- **State Token Violations**: 0
- **Total Leaks Detected**: **0 (Strict Zero Leaks)**
- **Audit Verdict**: **PASS** *(historical `v0.9.14` receipt; the original shared path `qualification/privacy_scan_results.json` was subsequently overwritten by a `9.15.2` scan, now relocated to `qualification/container-9.15.2/rootfs-privacy-audit-7b6b57df.json`)*

---

## 4. Live Container API Qualification Suite Results

The live container was executed under `$FLUX_LOCK_FILE` on loopback `http://127.0.0.1:8899` with model `Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid` mounted read-only.

- **Readiness Time**: 180.1 seconds
- **Pre-flight `MemAvailable`**: 105.19 GiB
- **Post-flight `MemAvailable`**: 105.13 GiB

### 4.1 Summary of 9 Assertions

| # | Assertion | Endpoint / Mechanism | Observed Output / Behavior | Status |
|---|---|---|---|---|
| **1** | Health Probe | `GET /health` | HTTP 200, {"status": "ready", "model": "qad5500"} | **PASS** |
| **2** | Model Catalog | `GET /v1/models` | HTTP 200, catalog contains ['qad5500'] | **PASS** |
| **3** | Chat Blocking | `POST /v1/chat/completions` (`stream: false`) | HTTP 200, content: 'READY', `finish_reason: "stop"`, TTFT: 514.14 ms, 31.99 tok/s | **PASS** |
| **4** | Chat Streaming | `POST /v1/chat/completions` (`stream: true`) | HTTP 200, 3 SSE chunks: '1, 2, 3, 4, 5', terminal `[DONE]` | **PASS** |
| **5** | Tool Calling Blocking | `POST /v1/chat/completions` (`tools: [...]`) | HTTP 200, `finish_reason: "tool_calls"`, get_weather({"city": "Paris"}) | **PASS** |
| **6** | Tool Calling Streaming | `POST /v1/chat/completions` (`stream: true`) | HTTP 200, 4 delta chunks assembled: get_weather({"city": "Paris"}), `finish_reason: "tool_calls"` | **PASS** |
| **7** | Client TCP Cancellation | Abrupt socket close at chunk 8 | Server gracefully aborts stream, no hanging threads or crash | **PASS** |
| **8** | Slot Reuse & Recovery | Replacement probe `POST /v1/chat/completions` | HTTP 200, content: 'OK', `atlas_requests_active == 0.0` | **PASS** |
| **9** | Raw Log Safety Audit | `docker logs` scan over 9,500 lines | 0 matches across fatal patterns | **PASS** |

---

## 5. Box Safety & Hardware Compliance

1. **Lock Discipline**: Full qualification serialized under `$FLUX_LOCK_FILE` and `$FLUX_BUILD_LOCK_FILE`.
2. **RAM Threshold Verification**: Host RAM headroom remained $\ge 100\text{ GiB}$ throughout (105.19 GiB pre-flight, 105.13 GiB post-flight).
3. **Log Sanitization**: Container stdout/stderr contained zero hardware, driver, or memory integrity exceptions.

---

## 6. Release Artifact Manifest

| Artifact File | Canonical Path | SHA-256 Digest |
|---|---|---|
| **API Qualification Receipt** | `qualification/api_qualification_receipt.json` | `8d1e2a219f1093201ad66bfe36b87794bcb3a95f41acf191bdb01dce587005fd` *(historical pre-sanitization: `a86b42f728cf1d3c6d5811e7640d410fac63a91011e4781a4e8152a7711fc7a9`)* |
| **Rootfs Privacy Audit Script** | `qualification/audit_rootfs_privacy.py` | `f0abd645e463ef5c5041e58b80c1d5b4654c039e54a27fa0c42bf4a39071dde7` *(historical pre-sanitization: `edf07d05b0aa5f0b22e893b99287d5ade02238f6866b0bab7f495d76a2f01923`)* |
| **Rootfs Privacy Results** | `qualification/privacy_scan_results.json` *(historical `v0.9.14` record; overwritten by later `9.15.2` scan relocated to `qualification/container-9.15.2/rootfs-privacy-audit-7b6b57df.json`)* | `c575a600bf7595db430e7801444e7badaa82903cad1e8e2f304725887e6705fc` *(historical `v0.9.14`)* |
| **API Qualification Script** | `qualification/run_m5_api_qualification.py` | `ee17fb9036ea70c3ac0a7a3b9cf9589dcccc9b3c7e8285b9c71b6a138ab33d84` *(historical pre-sanitization: `ad4706e4f6cc02f6d7466a5f7e54e4747aad571f7014863c1c30d7b795234bb1`)* |
| **Milestone 5 Receipt JSON** | `qualification/M5_RECEIPT.json` | *(Self-referenced above)* |
