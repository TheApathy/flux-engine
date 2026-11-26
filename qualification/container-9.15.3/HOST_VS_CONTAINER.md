# Host vs. Zero-Flag Container Parity Qualification Report
## Release Candidate: `flux-engine:9.15.3` (NVIDIA Blackwell DGX Spark GB10 / SM121a)
**Date**: 2026-10-26 *(amended against checked-in JSON summaries)*  
**Status**: **FAIL / PARTIAL (Parity & SLA Gaps Observed — see Audit Summary)**  
**Host Binary SHA-256** (`host_validation_summary.json`): `a7d164092103048edffef502fe8ebe1589794359cf3e9978ce284768c2ccb78f`  
**Reported Container Binary SHA-256** *(not recorded in `validation_summary.json`)*: `58641187c7ed267a1ffe8769a55b248753027c41dcea4a76a0c39b04a52bd9c1`  
**Image Config Digest** (`rootfs_audit.json`): `sha256:04a1e514c257380433df7fc8e7403bd57b9edec91d4e1716e87daf749753085d`  
**Source Commit**: `30cc416d9` (`perf(qwen4): batched C=8 verify/MTP fusion, 2-slab UMA headroom, and cohort prefill overlap (v9.15.3)`)  

---

### Executive Summary

This report compares the checked-in bare-metal DGX Spark host summary (`host_validation_summary.json`), zero-flag Docker container summary (`validation_summary.json`), and rootfs privacy audit (`rootfs_audit.json`) for `flux-engine:9.15.3`.

**Passing gates in the checked-in evidence:**
- Rootfs privacy audit (`rootfs_audit.json`): 9,458 files, `4,593,819,053` bytes, **0 violations** (host paths, secrets, forbidden state tokens).
- Cold start readiness across all tested modes (`<= 120s` target): Host `69.03s–82.06s`, Container `107.26s–113.36s`.
- Issue #13 needle retrieval: **8/8 PASS** on both Host and Container.
- Native 262K battery (`FLUX_MODE=256k`): **PASS** on both Host and Container (`depths: [0.01, 0.25, 0.50, 0.75, 0.99]`).
- C=8 concurrency scaling (`0..8K`): completed across all 5 context points on both Host and Container.

**Failing / mismatched gates in the checked-in evidence (previously misreported as PASS):**
1. **Binary Identity — FAIL (Mismatch)**: `host_validation_summary.json` records `a7d164092103048e...`, whereas the report previously claimed identical `58641187c7ed267a...` (and `validation_summary.json` does not record `binary_sha256`).
2. **PLE Placement SLA (`< 400 ms`) — FAIL**: `validation_summary.json` records `placement_sub_400ms: false`, `placement_max_ms: 875.8 ms`, with samples `[31.5, 45.3, 37.1, 126.0, 47.9, 481.3, 113.5, 151.9, 123.1, 591.9]`.
3. **8-Prompt Greedy Decode Identity — FAIL (4/8)**: `validation_summary.json` records `id_ok: false` on 4 of 8 prompts (`prose8k`, `legal32k`, `doc64k`, `mix128k`), with differing Host vs Container completion lengths on `prose8k` (72 vs 256) and `mix128k` (30 vs 50).
4. **Prefill Parity Band (`|delta| <= 3%` vs Host) — Outside ±3% on 16K & 32K**: Using Host as the denominator, Container prefill is `+3.41%` at 16K and `+6.74%` at 32K (faster than Host, but outside a symmetric `±3%` parity band).

---

### Section 1: Image Build & IP Integrity

| Verification Gate | Specification | Host Baseline | Container (`flux-engine:9.15.3`) | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Binary Identity** | Exact SHA-256 Match | `a7d164092103048e...` | `58641187c7ed267a...` *(unrecorded in `validation_summary.json`)* | **FAIL (Mismatch)** |
| **Privacy Audit** | `qualification/audit_rootfs_privacy.py` | N/A | 9,458 files, `4,593,819,053` bytes, **0 violations** | **PASS** |
| **Host Path Leakage** | Zero host home-directory paths | N/A | **0 matches** (`host_path_violations: 0`) | **PASS** |
| **Secrets & Forbidden Tokens** | Zero secrets / state tokens | N/A | **0 matches** (`secret_violations: 0`, `state_token_violations: 0`) | **PASS** |

---

### Section 2: Cold Start & Mode Boots

Target: readiness `<= 120s`.

| Operating Mode | Host Cold Start | Container Cold Start | Target | Status | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`c1` (`max-prefill`)** | 82.06s | 109.34s | $\le 120$s | **PASS** | Ready with Web Studio UI & `/flux/modes` API |
| **`c8` (Concurrency)** | 69.05s | 109.28s | $\le 120$s | **PASS** | 8 verify lanes, Host `ready_memavail_gib=9.77` |
| **`auto`** | — | 109.30s | $\le 120$s | **PASS** | `completion_ok: true` |
| **`max-prefill`** | — | 109.25s | $\le 120$s | **PASS** | `completion_ok: true` |
| **`interactive`** | — | 107.26s | $\le 120$s | **PASS** | `completion_ok: true` |
| **`agent`** | — | 108.26s | $\le 120$s | **PASS** | `completion_ok: true`, `tool_call_ok: true` |
| **`decode`** | — | 108.26s | $\le 120$s | **PASS** | `completion_ok: true` |
| **`256k`** | — | 113.36s | $\le 120$s | **PASS** | Recorded in `validation_summary.json` (`mode_boots.256k.ready_s`) |
| **`exact`** | — | 108.26s | $\le 120$s | **PASS** | `completion_ok: true` |
| **`long-context`** | — | 108.28s | $\le 120$s | **PASS** | `completion_ok: true` |
| **`fast`** | — | 109.28s | $\le 120$s | **PASS** | `completion_ok: true` |

Web Studio UI (`web_studio_ok: true`) and `/flux/modes` API (`modes_count: 6`) verified in `validation_summary.json`.

---

### Section 3: Prefill Throughput Parity ($W=16,384$, 8K..128K)

Dedicated prefill grid benchmark (`n=1` per context in both JSON summaries). Relative delta uses **Host** as denominator: `(Container - Host) / Host * 100`.

| Context | Host Prefill (tok/s) | Container Prefill (tok/s) | Relative Delta (vs Host) | Symmetric Parity Gate ($\lvert\Delta\rvert \le 3\%$) |
| :--- | :--- | :--- | :--- | :--- |
| **8K** | 5,871.4 | 5,922.2 | +0.86% (Container faster) | **PASS** |
| **16K** | 6,001.4 | 6,205.8 | +3.41% (Container faster) | **OUTSIDE ±3% (Container faster)** |
| **32K** | 5,745.7 | 6,133.2 | +6.74% (Container faster) | **OUTSIDE ±3% (Container faster)** |
| **64K** | 6,033.6 | 6,053.9 | +0.34% (Container faster) | **PASS** |
| **128K** | 5,881.7 | 5,930.7 | +0.83% (Container faster) | **PASS** |

- **PLE Placement IO Latency (`validation_summary.json`)**:
  * Container `engaged_256_lanes: true`, `plan_ran_256_lanes: true`.
  * Container `placement_samples_ms`: `[31.5, 45.3, 37.1, 126.0, 47.9, 481.3, 113.5, 151.9, 123.1, 591.9]`.
  * Container **`placement_max_ms: 875.8 ms`**, **`placement_sub_400ms: false`** — **FAIL** against the `< 400 ms` SLA requirement.

---

### Section 4: Decode Parity & Needle Retrieval

#### 1. Issue #13 Needle In A Haystack (8/8 Probes)
Passkey: `The secret vault code is 4719-ALPHA-82.`

| Probe Context | Depth | Rep | Host Answer | Host Wall | Container Answer | Container Wall | Result |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2,017 | 0.50 | 0 | `4719-ALPHA-82` | 0.640s | `4719-ALPHA-82` | 0.655s | **PASS** |
| 2,112 | 0.50 | 0 | `4719-ALPHA-82` | 0.616s | `4719-ALPHA-82` | 0.902s | **PASS** |
| 7,710 | 0.50 | 0 | `4719-ALPHA-82` | 1.502s | `4719-ALPHA-82` | 2.266s | **PASS** |
| 7,710 | 0.50 | 1 | `4719-ALPHA-82` | 1.469s | `4719-ALPHA-82` | 2.130s | **PASS** |
| 7,710 | 0.50 | 2 | `4719-ALPHA-82` | 1.466s | `4719-ALPHA-82` | 1.617s | **PASS** |
| 7,710 | 0.98 | 0 | `4719-ALPHA-82` | 1.468s | `4719-ALPHA-82` | 1.474s | **PASS** |
| 30,647 | 0.50 | 0 | `4719-ALPHA-82` | 5.115s | `4719-ALPHA-82` | 6.790s | **PASS** |
| 65,536 | 0.50 | 0 | `4719-ALPHA-82` | 10.655s | `4719-ALPHA-82` | 11.160s | **PASS** |

#### 2. 8-Prompt Greedy Decode (CONF=0) — **4/8 Match (`FAIL`)**

TTFT delta uses **Host** as denominator: `(Container - Host) / Host * 100`. Greedy identity reflects `id_ok` in `validation_summary.json` (and note the mismatched binary SHA-256 in Section 1).

| Prompt | Context | Host Tok | Container Tok | Host TTFT | Host tok/s | Container TTFT | Container tok/s | TTFT Delta (vs Host) | Greedy Identity (`id_ok`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `code0` | 64 | 256 | 256 | 160.9ms | 63.61 | 159.9ms | 51.52 | -0.62% | **PASS (`true`)** |
| `prose0` | 64 | 133 | 133 | 170.2ms | 46.20 | 172.4ms | 33.76 | +1.29% | **PASS (`true`)** |
| `chat0` | 46 | 256 | 256 | 199.8ms | 62.72 | 226.0ms | 45.28 | +13.11% | **PASS (`true`)** |
| `rep0` | 104 | 256 | 256 | 181.9ms | 99.17 | 182.5ms | 97.50 | +0.33% | **PASS (`true`)** |
| `prose8k` | 8,192 | 72 | 256 | 1,468.0ms | 43.26 | 1,645.2ms | 52.82 | +12.07% | **FAIL (`false`, 72 vs 256 tok)** |
| `legal32k` | 32,768 | 256 | 256 | 5,411.5ms | 56.63 | 5,704.2ms | 48.05 | +5.41% | **FAIL (`false`)** |
| `doc64k` | 65,536 | 256 | 256 | 10,923.0ms | 54.55 | 10,958.1ms | 47.03 | +0.32% | **FAIL (`false`)** |
| `mix128k` | 126,976 | 30 | 50 | 21,562.8ms | 44.35 | 21,428.2ms | 40.97 | -0.62% | **FAIL (`false`, 30 vs 50 tok)** |

---

### Section 5: C=8 Concurrency Scaling

Measured with non-overlapping random prompt seeds (`100 + i`, 128 generated tokens per sequence = 1,024 completion tokens per point). E2E delta uses **Host** as denominator: `(Container - Host) / Host * 100`.

| Context Point | Avg Prompt Tok | Host Agg E2E (tok/s) | Host Sum Decode (tok/s) | Host Decode-Only (tok/s) | Host Avg TTFT (ms) | Container Agg E2E (tok/s) | Container Sum Decode (tok/s) | Container Decode-Only (tok/s) | Container Avg TTFT (ms) | E2E Delta (vs Host) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`ctx=0`** | 29.0 | **130.82** | **152.04** | **141.91** | 611.7 | **124.79** | **143.79** | **134.79** | 608.6 | -4.61% |
| **`ctx=1024`** | 990.1 | **62.52** | **87.50** | **72.12** | 2,181.3 | **59.59** | **79.92** | **67.36** | 1,982.4 | -4.69% |
| **`ctx=2048`** | 1,980.8 | **60.01** | **87.98** | **70.76** | 2,590.8 | **56.32** | **84.11** | **69.30** | 3,407.0 | -6.15% |
| **`ctx=4096`** | 3,976.0 | **54.64** | **90.37** | **67.85** | 3,647.4 | **53.04** | **84.41** | **65.30** | 3,624.3 | -2.93% |
| **`ctx=8192`** | 7,966.6 | **43.24** | **91.75** | **60.96** | 6,885.1 | **42.01** | **87.45** | **58.95** | 7,005.4 | -2.84% |

---

### Section 6: Native 262K Battery Parity (`FLUX_MODE=256k`)

Executed via `m2_262k_harness.py` (`verdict: PASS` in both `host_validation_summary.json` and `validation_summary.json`).

| Test Description | Container Result | Host Result | Parity Gate |
| :--- | :--- | :--- | :--- |
| **5-Depth Passkeys @ 262,080 (`0.01, 0.25, 0.50, 0.75, 0.99`)** | **100% Found (5/5)** | **100% Found (5/5)** | **PASS (100% Parity)** |
| **Deterministic Repeat (`repeat_identical`)** | **`true`** | **`true`** | **PASS (100% Parity)** |
| **Forced Fill (`262,080 + 64 = 262,144`)** | **`succeeded: true`** | **`succeeded: true`** | **PASS (100% Parity)** |
| **Boundary (`262,143 + 1 = 262,144`, token 15)** | **`boundary_matches_control: true`** | **`boundary_matches_control: true`** | **PASS (100% Parity)** |
| **Overflow (`262,144 + 1 = 262,145`)** | **HTTP 400 `invalid_request_error`** | **HTTP 400 `invalid_request_error`** | **PASS (100% Parity)** |
| **Canary Recovery (`262,080` tokens, 16 chunks)** | **`recovered: true`, token 15** | **`recovered: true`, token 15** | **PASS (100% Parity)** |

---

### Section 7: Rootfs Privacy Audit Receipt

1. **Rootfs Privacy Audit (`rootfs_audit.json`)**: `qualification/audit_rootfs_privacy.py` verified **9,458 files** (`4,593,819,053` bytes) for `flux-engine:9.15.3` (`sha256:04a1e514c257380433df7fc8e7403bd57b9edec91d4e1716e87daf749753085d`) with **0 violations** (`host_path_violations: 0`, `secret_violations: 0`, `state_token_violations: 0`). *(Note: `audit_rootfs_privacy.py` audits host paths, secrets, forbidden basenames, and forbidden state tokens; it does not scan for PTX kernels.)*
2. **Re-run Requirement for Full Parity Certification**: Closing the remaining parity gaps requires re-running Host and Container side-by-side with an identical verified binary SHA-256, resolving the 4 long-context greedy identity mismatches (`prose8k`, `legal32k`, `doc64k`, `mix128k`), and verifying `placement_max_ms < 400 ms` across all PLE placement samples.
