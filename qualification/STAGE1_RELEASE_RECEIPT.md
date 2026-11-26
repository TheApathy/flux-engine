# Milestone Receipt: Stage 1 Fast-Track Release (v0.9.14 / v0.9.14-r17)

**Release Version**: `v0.9.14` (Fast-Track Scope)  
**Release Tags**: `ghcr.io/theapathy/flux-engine:v0.9.14`, `ghcr.io/theapathy/flux-engine:v0.9.14-r17`, `flux-engine:r17-candidate`  
**Verdict**: **PARTIAL** (Fast-Track Scope — Build, Packaging, Privacy Audit, and Plain GSM8K Complete; Plain MMLU-Pro Partial 340/1,000; Step C K=5 Parity Arm Skipped)  
**Date/Timestamp**: 2026-10-25T17:06:00Z  
**Target Hardware**: NVIDIA DGX Spark (GB10, SM121a, 128 GB unified memory)  
**Model**: `Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid` with Native MTP Drafter  

---

## 1. Executive Summary

Stage 1 records the fast-track qualification run for `flux-engine:v0.9.14`. Code packaging, path-remapped release binary compilation, container assembly, offline unit/provenance suites, rootfs streaming privacy audits, and Plain GSM8K evaluation completed with zero integrity violations; Plain MMLU-Pro concluded early at 340/1,000 items and the Step C K=5 parity arm was skipped (superseded by subsequent v1.0.x qualification runs).

Key highlights:
1. **Concurrency Port**: Integrated concurrency commits (`50a2f6781`, `6a875b9ca`, `81afe8261`, `e0babc015`, and `a975235db`) into the release worktree.
2. **Recipe Configuration**: Configured `repro-kit/recipes/flux-qwen4-native-concurrency-c8-owner16.env` with `export ATLAS_SCHED_BURST_GATHER_MS=20` and `export ATLAS_MOE_T_LANES_WORKLIST_R8=1`.
3. **Remapped Binary Build**: Compiled release binary with `--remap-path-prefix=$HOME=/opt/fluxsrc` (`-j2`). Verified 0 embedded host home-directory paths in the ELF binary.
4. **Release Container Packaging**: Assembled Docker release container image `sha256:7bf7806eae0d7cd7cbad5c8c345ad91215ce7033c57506a493785bf508ae157c` tagged as `v0.9.14` and `v0.9.14-r17`.
5. **Unit & Provenance Tests**: 41/41 unit tests PASS, 9/9 recipe tests PASS, 38/38 closure tests PASS.
6. **Zero-Tolerance Privacy Audit**: Exhaustive scan of 9,455 files (2,955,569,430 bytes) revealed strictly 0 host path, credential, or state token violations.
7. **Step C Accuracy Scope (Partial)**: Evaluated under exclusive GPU lock. Plain GSM8K completed 1,319/1,319 items (97.27% accuracy, 0 `atlas.interventions.v1` events); Plain MMLU-Pro completed 340/1,000 items (0 interventions) before early stop; the Step C Native-MTP K=5 parity arm was skipped in this fast-track run. Host memory floor stayed >= 6,656 MiB throughout.

---

## 2. Artifact Provenance & Checksums

| Artifact | Location | SHA-256 Digest |
|---|---|---|
| **Host Release Binary** | `target/release/flux-server` | `a3b49aa3e1e3eb41b20bc28c4193026522865ee083d2d3e0cc1135d333da8b5e` |
| **Container Release Binary** | `repro-kit/bin/flux-server-real` | `a3b49aa3e1e3eb41b20bc28c4193026522865ee083d2d3e0cc1135d333da8b5e` |
| **Release Container Image** | `ghcr.io/theapathy/flux-engine:v0.9.14` | `sha256:7bf7806eae0d7cd7cbad5c8c345ad91215ce7033c57506a493785bf508ae157c` |
| **High-Prefill Recipe** | `repro-kit/recipes/flux-v9-max-prefill.env` | `00ef1a839bcfd2d2b65799846efe3441dcd267857354e48e5f0335f2f54aa3c6` *(historical `v0.9.14` image build; live repo post-sanitization: `2942f6ae474c84a175bb3969f2d6eedad43c60bd917162f76ad443f706420889`)* |
| **Interactive Recipe** | `repro-kit/recipes/flux-v9-w4096-interactive.env` | `dce6f9e14fc03f908eab406bfbb4f24755bf4447e198faec16dc37bc5f861869` *(historical `v0.9.14` image build; live repo post-sanitization: `318b36a75b485a4b73cc20b4b13285d21a1c66327f1de5a95d574c6a03039663`)* |
| **Concurrency Recipe** | `repro-kit/recipes/flux-qwen4-native-concurrency-c8-owner16.env` | Verified with R8 additions |

---

## 3. Verification & Test Suite Summary

- **Standoff Byte Verification**: Extracted container binaries match host inputs bit-for-bit.
- **Image Provenance Suite**: `test_image_provenance.py` (24/24 PASS).
- **Release Binding Successor Suite**: `test_r17_release_binding_successor.py` (12/12 PASS).
- **Release Metadata Suite**: `test_release_metadata.py` (5/5 PASS).
- **Recipe Compliance Suite**: `test_qwen4_native_concurrency_owner16_recipe.py` (9/9 PASS).
- **Source Closure Suite**: `test_ignored_source_closure.py` (38/38 PASS).
- **Total Tests Passed**: 88/88 tests PASS.

---

## 4. Rootfs Privacy & Security Audit

- **Files Scanned**: 9,455 files
- **Total Bytes Scanned**: 2,955,569,430 bytes
- **Host Path Leaks**: 0
- **Secret & Token Patterns**: 0
- **Prohibited State Tokens**: 0
- **Verdict**: **PASS (0 Leaks)**

---

## 5. Step C Accuracy Execution (Partial Fast-Track Scope)

- **Execution Mode**: Live Docker container (`flux-engine:r17-candidate`)
- **GPU Lock**: Serialized via exclusive GPU lockfile
- **Pre-flight Available Memory**: $\ge 100.0\text{ GiB}$ verified before launch
- **Concurrency**: `CONCURRENCY_PLAIN=8` (8-way native batched QSA)
- **Plain GSM8K Benchmark**: Full 1,319 / 1,319 items COMPLETE (`100% OK`, 1,283 correct, `97.27% accuracy`, `0 interventions`)
  - Report file: `qualification/accuracy/reports/plain-gsm8k.json` (SHA-256: `f9381bb9ec99500c9e925fea6aa78edaad8a8db6ae22accb6554503da83a78dd`; historical pre-sanitization: `551aa5c3dcf90be675dff377430c3a212827c0405915a43caa4c88fef5294b85`)
- **Plain MMLU-Pro Benchmark**: **PARTIAL** — 340 / 1,000 items completed at concurrency 8 with 0 interventions
- **Step C Native-MTP K=5 Arm**: **SKIPPED** (2026-10-25T20:34:36Z) — run concluded early to yield the GPU to subsequent benchmark gating; K=5 parity was not evaluated in this receipt
- **Intervention Ledger**: 100% empty `atlas.interventions.v1` ledgers (0 non-empty interventions across all 1,659 evaluated items)
- **Memory Compliance**: Host memory floor maintained $\ge 6,656\text{ MiB}$ throughout all phases (lowest measured `11,430 MiB` across 43,756 samples; > 4.7 GiB margin above requirement)

---

## 6. Distribution & Registry Status

- Local Tags Built and Validated:
  - `flux-engine:r17-candidate`
  - `ghcr.io/theapathy/flux-engine:v0.9.14`
  - `ghcr.io/theapathy/flux-engine:v0.9.14-r17`
- Remote Registry: `ghcr.io/theapathy/flux-engine` is public; anonymous and authorized pulls validated.
- Protected Tag Status: `v0.9.12-6k-stable` strictly untouched and preserved.
