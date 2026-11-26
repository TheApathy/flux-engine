# Milestone 4 Qualification Receipt: Performance Profiles & Fast Loading (R4)

**Milestone**: Milestone 4 — Performance Profiles & Fast Loading (R4)  
**Verdict**: **FAIL** *(strict `<= 5.0%` decode noise band vs `CERTIFIED_HOST` failed on all 12 cells; see audit note below)*  
**Timestamp**: 2026-10-24T21:26:11Z  
**Hardware**: NVIDIA DGX Spark (GB10, SM121a, 128 GB unified memory)  
**Model**: `Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid` with native MTP head  
**Binary Location**: `target/release/flux-server` *(bare-metal host run)*  
**Binary SHA-256**: `e742cb9b808b1f224f3684adda501354782cab7116a77c1f1e1a70bddc78e4ab`  
**HEAD Commit**: `6498ad9ebaa1d8a4db04cef6ec425fd6004b9e40`  

> **Truth Audit Amendment Note**: This historical receipt was corrected against its checked-in `qualification/profiles/sustained_results.json` and `coldstart_results.json` artifacts. Cold-start readiness, prefill throughput vs baselines, K=5/K=1 greedy bit-identity, 3/3 passkey retrieval, and host memory floor all passed; however, the earlier receipt over-claimed `1.67x – 2.05x` decode speedup (actual: `1.18x – 1.41x`), claimed `>= 5,000 tok/s` across all Interactive contexts (262k measured `4,990.7 tok/s`), used `(budget / rate)` instead of `sustained_results.json`'s `decode_wall_s` (`(budget - 1) / rate`), and marked all 12 rows `PASS (<= 5.0%)` despite K=5 decode wall regressions of `-12.70%` to `-33.17%` (masked by tautological `or` escapes in `run_m4_sustained_matrix.py`).

---

## 1. Executive Summary

Milestone 4 evaluates the **Interactive (`W4096`)** and **High-Prefill (`W16384`)** sustained performance profiles across both **Plain ($K=1$)** and **Native-MTP ($K=5$)** decoding arms across all context lengths (`8k`, `16k`, `32k`, `64k`, `128k`, `262k`), alongside **3 consecutive cold starts** targeting readiness in $\le 150.0\text{ s}$ median.

Gate results from the recorded run:
1. **Cold Start Fast Loading (PASS)**: 3 consecutive bare-metal cold starts achieved a median readiness time of **139.23 s** (target $\le 150.0\text{ s}$) with zero fatal log patterns and anchor verification.
2. **High-Prefill Profile ($W=16384$, PASS)**: Strictly beats the historical champion `64136c5d9-r1` baseline at every context length (`+197.3` to `+724.7 tok/s`).
3. **Interactive Profile ($W=4096$, PASS vs baseline)**: Exceeds `510b8f5de-r1` at every context length (`+205.5` to `+390.7 tok/s`), reaching `5,138.9 – 5,294.5 tok/s` across `8k – 128k` and `4,990.7 tok/s` at `262k`.
4. **Native-MTP $K=5$ Decode Acceleration (PASS)**: Losslessly accelerates decode throughput over Plain ($K=1$) by **1.18x – 1.41x** across the 12 context cells with 100% token bit-identity (0 diff).
5. **Noise Band Compliance (FAIL)**: **0/12** cells fall within the strict **$\le 5.0\%$** noise band vs `CERTIFIED_HOST` references (K=5 decode wall times are `-12.70%` to `-33.17%` slower than reference; Plain decode hits `-5.50%` to `-10.52%` at 128k/262k; prefill is faster than reference by `+5.05%` at Interactive 128k and `+10.94%` at High-Prefill 262k).
6. **Passkey Needle Retrieval (PASS)**: 100% of embedded passkeys (ALPHA, BETA, GAMMA at depths 0.05, 0.50, 0.95) retrieved successfully across all runs.
7. **Box Safety & Memory Floor (PASS)**: Host available memory remained safely above the required threshold (lowest observed floor: **9240.0 MiB** vs 6,656 MiB required, 0 watchdog kills).

---

## 2. Cold Start Readiness Attestation

| Attempt | Launch Monotonic (ns) | Ready Monotonic (ns) | Elapsed Seconds | HTTP Status | Log Anchors | Fatal Patterns | Result |
|---|---|---|---|---|---|---|---|
| Run 1 | 1051424496020467 | 1051565004956481 | **140.51 s** | 200 OK | Verified In Order | 0 Matches | **PASS** |
| Run 2 | 1051568534637430 | 1051707759680542 | **139.23 s** | 200 OK | Verified In Order | 0 Matches | **PASS** |
| Run 3 | 1051711287196404 | 1051845521205176 | **134.23 s** | 200 OK | Verified In Order | 0 Matches | **PASS** |

- **Median Readiness Time**: **139.23 s** (Target: $\le 150.0\text{ s}$) — **PASS**
- **Page-Cache Eviction Method**: Per-file `posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)` on model safetensors shards and metadata files.
- **R6 Box Safety**: Zero global cache drops (`drop_caches`), zero `swapoff`, zero service terminations. Pre-flight memory restored to $\ge 100\text{ GiB}$ before every run.

---

## 3. High-Prefill Profile Throughput (`W16384`) vs Baseline `64136c5d9-r1`

| Context | Prompt Tokens | TTFT (ms) | Measured Throughput (tok/s) | Baseline Throughput (tok/s) | Margin (tok/s) | Status |
|---|---|---|---|---|---|---|
| **8k** | 8,192 | 1399.1 | **5844.4** | 5508.7 | **+335.7** | **PASS** |
| **16k** | 16,384 | 2704.4 | **6054.1** | 5813.2 | **+240.9** | **PASS** |
| **32k** | 32,768 | 5516.5 | **5935.5** | 5738.2 | **+197.3** | **PASS** |
| **64k** | 65,536 | 11175.0 | **5860.7** | 5637.1 | **+223.6** | **PASS** |
| **128k** | 131,072 | 22690.8 | **5773.1** | 5551.4 | **+221.7** | **PASS** |
| **262k** | 262,080 | 47415.3 | **5524.7** | 4800.0 | **+724.7** | **PASS** |

---

## 4. Interactive Profile Throughput (`W4096`) vs Baseline `510b8f5de-r1`

| Context | Prompt Tokens | TTFT (ms) | Measured Throughput (tok/s) | Baseline Throughput (tok/s) | Margin (tok/s) | Status |
|---|---|---|---|---|---|---|
| **8k** | 8,192 | 1549.4 | **5280.0** | 5074.5 | **+205.5** | **PASS** |
| **16k** | 16,384 | 3092.2 | **5294.5** | 5054.7 | **+239.8** | **PASS** |
| **32k** | 32,768 | 6190.1 | **5289.9** | 4992.2 | **+297.7** | **PASS** |
| **64k** | 65,536 | 12513.8 | **5233.9** | 4884.1 | **+349.8** | **PASS** |
| **128k** | 131,072 | 25493.3 | **5138.9** | 4768.2 | **+370.7** | **PASS** |
| **262k** | 262,080 | 52500.0 | **4990.7** | 4600.0 | **+390.7** | **PASS** |

---

## 5. Speculative Decode Acceleration & Lossless Parity

| Profile | Context | Plain Decode (tok/s) | Native-MTP K=5 Decode (tok/s) | Speedup Ratio | Token Bit-Identity | Passkeys (ALPHA, BETA, GAMMA) |
|---|---|---|---|---|---|---|
| High-Prefill | 8k | 35.2 | **48.7** | **1.38x** | **100% Identical (0 Diff)** | **3/3 Verified** |
| High-Prefill | 16k | 34.8 | **49.1** | **1.41x** | **100% Identical (0 Diff)** | **3/3 Verified** |
| High-Prefill | 32k | 34.7 | **45.9** | **1.32x** | **100% Identical (0 Diff)** | **3/3 Verified** |
| High-Prefill | 64k | 34.2 | **44.6** | **1.31x** | **100% Identical (0 Diff)** | **3/3 Verified** |
| High-Prefill | 128k | 33.2 | **46.1** | **1.39x** | **100% Identical (0 Diff)** | **3/3 Verified** |
| High-Prefill | 262k | 32.0 | **41.8** | **1.31x** | **100% Identical (0 Diff)** | **3/3 Verified** |
| Interactive | 8k | 35.1 | **44.3** | **1.26x** | **100% Identical (0 Diff)** | **3/3 Verified** |
| Interactive | 16k | 35.1 | **46.1** | **1.31x** | **100% Identical (0 Diff)** | **3/3 Verified** |
| Interactive | 32k | 34.5 | **46.6** | **1.35x** | **100% Identical (0 Diff)** | **3/3 Verified** |
| Interactive | 64k | 34.3 | **44.5** | **1.30x** | **100% Identical (0 Diff)** | **3/3 Verified** |
| Interactive | 128k | 33.7 | **43.9** | **1.30x** | **100% Identical (0 Diff)** | **3/3 Verified** |
| Interactive | 262k | 31.9 | **37.6** | **1.18x** | **100% Identical (0 Diff)** | **3/3 Verified** |

---

## 6. Host Noise Band Compliance ($\le 5.0\%$)

*(K=5 decode wall times below are taken verbatim from `qualification/profiles/sustained_results.json` using `(budget - 1) / rate`.)*

| Profile | Context | Measured Prefill Wall (s) | Prefill Noise Δ (%) | Measured K=5 Decode Wall (s) | K=5 Decode Noise Δ (%) | Noise Band Status |
|---|---|---|---|---|---|---|
| High-Prefill | 8k | 1.402 | +0.38% | 5.235 | -13.47% | **FAIL (> 5.0%)** |
| High-Prefill | 16k | 2.706 | -0.38% | 5.189 | -12.70% | **FAIL (> 5.0%)** |
| High-Prefill | 32k | 5.521 | -0.28% | 5.558 | -18.50% | **FAIL (> 5.0%)** |
| High-Prefill | 64k | 11.182 | +1.18% | 5.718 | -20.78% | **FAIL (> 5.0%)** |
| High-Prefill | 128k | 22.704 | +1.91% | 5.536 | -18.17% | **FAIL (> 5.0%)** |
| High-Prefill | 262k | 47.438 | +10.94% | 1.506 | -25.63% | **FAIL (> 5.0%)** |
| Interactive | 8k | 1.552 | +0.23% | 5.761 | -21.37% | **FAIL (> 5.0%)** |
| Interactive | 16k | 3.095 | +2.50% | 5.527 | -18.04% | **FAIL (> 5.0%)** |
| Interactive | 32k | 6.194 | +3.40% | 5.466 | -17.12% | **FAIL (> 5.0%)** |
| Interactive | 64k | 12.521 | +4.34% | 5.728 | -20.91% | **FAIL (> 5.0%)** |
| Interactive | 128k | 25.506 | +5.05% | 5.815 | -22.10% | **FAIL (> 5.0%)** |
| Interactive | 262k | 52.514 | +2.90% | 1.676 | -33.17% | **FAIL (> 5.0%)** |

---

## 7. Shared-Hardware Safety & Memory Telemetry

- **Lock Acquisition**: Held `$FLUX_LOCK_FILE` exclusively throughout the entire benchmark execution.
- **Pre-Flight Memory**: Certified $\ge 100\text{ GiB}$ MemAvailable before every server invocation.
- **Memory Floor**: Lowest measured available memory during sustained runs was **9240.0 MiB** (Safety bar: $\ge 6,656\text{ MiB}$).
- **Watchdog Kills**: **0** aborts or OOM events across all benchmark cells.
- **Speculative Telemetry**:
  - Plain Arm verify events: **0** (`high-prefill-plain=0`, `interactive-plain=0`; speculation strictly dormant)
  - Native-MTP K=5 verify events: **2,432** (`high-prefill-k5=1,186`, `interactive-k5=1,246`; native batching engaged)
  - Serial fallback events: **0**
- **Orderly Teardown**: Server processes cleanly terminated; unified host/device memory completely reclaimed ($\ge 100\text{ GiB}$) before lock release.

---

## 8. Artifact Checksums & Signoff

*(Recipe paths point to the checked-in `repro-kit/recipes/` files; `coldstart_results.json` digest reflects the checked-in sanitized artifact; pre-sanitization historical digests are retained in `qualification/M4_RECEIPT.json`.)*

| Artifact | Path | SHA-256 Digest |
|---|---|---|
| Release Binary *(historical run)* | `target/release/flux-server` | `e742cb9b808b1f224f3684adda501354782cab7116a77c1f1e1a70bddc78e4ab` |
| Recipe (High-Prefill) | `repro-kit/recipes/flux-v9-max-prefill.env` | `2942f6ae474c84a175bb3969f2d6eedad43c60bd917162f76ad443f706420889` |
| Recipe (Interactive) | `repro-kit/recipes/flux-v9-w4096-interactive.env` | `318b36a75b485a4b73cc20b4b13285d21a1c66327f1de5a95d574c6a03039663` |
| Cold Start Results | `qualification/profiles/coldstart_results.json` | `e5e80fb1513d481df8ca6fc54fcddc3124d5be63a64215098fc6ac0d8aaee3b3` |
| Sustained Results | `qualification/profiles/sustained_results.json` | `e4f480d17cf9239ee2f2817d756ec4db8b408360c09d838c60dac9a84b5cf379` |
| Response Journal | `qualification/profiles/responses.jsonl` | `eb7707c69dd9100da3c0bb0372e377d25bf4a8be26c4fb0d7ce99221b1dbca12` |
| Memory Samples | `qualification/profiles/memory_samples.csv` | `bf63b603262376546f2a548ec567d040f48a12d142d72db6b53b25240735a748` |

### Independent Checklist Signoff
- [x] Git HEAD is at `6498ad9eb` on `.`.
- [x] Candidate `flux-server` compiled incrementally under `$FLUX_BUILD_LOCK_FILE` (`-j2`).
- [x] Three consecutive cold starts completed with median time $\le 150.0\text{ s}$ (139.23 s).
- [x] Log anchor sequence verified in order with 0 fatal patterns.
- [x] High-Prefill ($W=16384$) strictly beats `64136c5d9-r1` across all contexts.
- [x] Interactive ($W=4096$) exceeds `510b8f5de-r1` across all contexts (`4,990.7 – 5,294.5 tok/s`).
- [x] Native-MTP $K=5$ decode throughput accelerates over Plain ($K=1$) (`1.18x – 1.41x`) with 100% token bit-identity.
- [ ] Noise band $\le 5.0\%$ satisfied across all cells *(FAIL: 0/12 cells within strict `+-5.0%` band due to K=5 decode wall delta `-12.70%` to `-33.17%`)*.
- [x] 3 passkeys (ALPHA, BETA, GAMMA) verified at 5%, 50%, 95% depths.
- [x] Host memory floor $\ge 6,656\text{ MiB}$ maintained throughout with 0 watchdog kills.
- [x] Receipts committed to git repository.
