# Flux Engine `9.15.1` (`v0.9.15.1`) Hermetic Container Validation Receipt

## 1. Image & Provenance Summary

- **Image**: `ghcr.io/theapathy/flux-engine:9.15.1`
- **Published GHCR Tags**: `9.15.1`, `v9.15.1`, `v0.9.15.1`, `0.9.15.1`, `v0.9.15`, `9.15`, `0.9.15`, `v0.9.15-r17`, `gb10`, `latest`
- **Registry Manifest Digest**: `sha256:a90548fa801f61936fe2dc8b93431b050656574672dc6f138cb565e7b5ec8eda`
- **Local Image ID**: `sha256:a686822cd8ddda1aefc8c9a5692e2c8368bc0858117c6332a907f2e6b49186e6`
- **Source Revision**: `2f7d72d7dee9b29a26bb071229caad222dcc2851` (`TREE=f9cdb8e363c006f3c0ff9cdd37fafa0898e87f3c`)
- **In-Container Engine Binary**: `/opt/flux/bin/flux-server-real` (`SHA-256: d4bce90069810049f88d92c0f171bb3cb45f12e0dc86d179e2a36b5cbbbed271`, 85 MiB stripped)
- **Container Binary Hygiene Audit**: `PASS` (stripped release binary, `0` embedded host paths)
- **Zero-Flag Container Compatibility**: `PASS` (runs directly via `docker run --rm --gpus all --ipc=host --network=host` without requiring `--ulimit memlock=-1` or `--security-opt seccomp=unconfined`; 4-lane parallel `pread64` fallback + `PINNED_HOST_POOL` + `record_cache` fast-plan enabled)
- **Bundled Native MTP Sidecar**: `/opt/flux/models/Qwen3.8-Flash-Next-Inferact-MTP` (`nvfp4_experts_mtp.safetensors`, `SHA-256: 9fcde463262080a95caebbcfa99ba4fb17990a9cd1465c87e65a31099d7a98a5`)
- **Rootfs Privacy Scan**: `PASS` (`9,458` files, `4,361,252,193` bytes scanned, `0` violations — historical `9.15.1` scan record; its original shared output file `qualification/privacy_scan_results.json` was subsequently overwritten by a `9.15.2` scan, now relocated to `qualification/container-9.15.2/rootfs-privacy-audit-7b6b57df.json`)

---

## 2. Live Container Prefill Validation (`W16384` Default & `W4096` Interactive)

| Context | Prompt Tokens | `W16384` TTFT (s) | `W16384` Container (`tok/s`) | `W4096` TTFT (s) | `W4096` Container (`tok/s`) | `W4096` Same-Prompt Determinism |
|:---:|---:|---:|---:|---:|---:|:---:|
| **8K** | 8,192 | `1.369` | **`5,983.9`** | `1.513` | **`5,413.7`** | `3/3 IDENTICAL` |
| **16K** | 16,384 | `2.627` | **`6,237.4`** | `3.038` | **`5,392.5`** | `3/3 IDENTICAL` |
| **32K** | 32,768 | `5.338` | **`6,138.3`** | `6.128` | **`5,347.5`** | `3/3 IDENTICAL` |
| **64K** | 65,536 | `11.059` | **`5,926.2`** | `12.556` | **`5,219.7`** | `3/3 IDENTICAL` |
| **128K** | 131,072 | `22.542` | **`5,814.6`** | `25.637` | **`5,112.7`** | `3/3 IDENTICAL` |

- **`W16384` JSON Receipt**: `qualification/container-9.15.1/fn-grid-container-9.15.1-w16384.json`
- **`W4096` JSON Receipt**: `qualification/container-9.15.1/fn-grid-container-9.15.1-w4096-v3.json` (`15/15` identical across all 5 context groups)

---

## 3. Live Container Native-MTP ($K=5$) Decode Validation ($C=1$)

| Workload | Tokens | Decode-Only (`tok/s`) | End-to-End Wall (`tok/s`) | Wall Time (s) |
|---|---:|---:|---:|---:|
| **`sparkdash_code_400`** | 400 | **`93.13`** | **`87.95`** | `4.28s` |
| **`tf_streams_code_256`** | 256 | **`84.98`** | **`78.60`** | `3.00s` |
| **`tf_streams_prose_256`** | 256 | **`65.48`** | **`61.63`** | `3.89s` |
| **`sparkdash_prose_400`** | 400 | **`64.12`** | **`61.90`** | `6.22s` |

- **Decode Receipt**: `qualification/container-9.15.1/decode-k5.stdout`
