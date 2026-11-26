# Milestone 1 Qualification Receipt: True Concurrency (R3)

**Verdict**: **PASS**  
**Timestamp**: `2026-10-24T12:35:59.149284+00:00`  
**Hardware**: `NVIDIA DGX Spark (GB10, SM121a, 128 GB unified memory)`  
**Model**: `Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid`  
**Binary**: `target/release/flux-server`  
**Recipe**: `flux-qwen4-native-concurrency-c8-owner16.env`  

---

## 1. Concurrency Parity Attestation

All concurrent request waves (C2..C8) were validated for exact greedy token-ID bit identity against their respective isolated C1 references.

| Test Case | Description | Result | Parity vs C1 Reference |
|---|---|---|---|
| **C1 Baseline & Repeat** | Janet repeat determinism | PASS | 100% token-ID identity |
| **C2 Same** | 2x simultaneous Janet requests | PASS | 100% token-ID identity |
| **C2 Diff** | Janet + Robe simultaneous requests | PASS | 100% token-ID identity |
| **C4** | Janet + Janet + Robe + Fib simultaneous requests | PASS | 100% token-ID identity |
| **C8** | 8 diverse prompts simultaneous requests | PASS | 100% token-ID identity |
| **Cancel & Slot Reuse** | Cancel slot 7 mid-stream, immediately reuse slot | PASS | 100% token-ID identity |
| **Post-Cancel Sanity** | Standalone C1 post cancellation | PASS | 100% token-ID identity |

---

## 2. Concurrency Throughput Scaling

Measured across fixed 128-token generation (`ignore_eos: true`):

| Concurrency Level | Mean TTFT (s) | Max TTFT (s) | Mean Per-Req Tok/s | Sum Per-Req Tok/s | Aggregate Decode Tok/s | Parity vs C1 |
|---|---|---|---|---|---|---|
| **C1** | 0.301 s | 0.301 s | 45.46 tok/s | 45.46 tok/s | **45.82 tok/s** | PASS |
| **C2** | 0.556 s | 0.615 s | 26.33 tok/s | 52.66 tok/s | **49.25 tok/s** | PASS |
| **C4** | 1.110 s | 1.231 s | 14.03 tok/s | 56.13 tok/s | **52.63 tok/s** | PASS |
| **C8** | 2.108 s | 3.156 s | 6.78 tok/s | 54.25 tok/s | **49.90 tok/s** | PASS |

*Scaling Observation*: Aggregate decode throughput scales monotonically from C1 (45.82 tok/s) through C4 (52.63 tok/s) and sustains 49.90 tok/s at C8.

---

## 3. Bounded-Transpose MoE & Zero-Fallback Telemetry

Telemetry extracted from `server.log` confirms 100% native multi-sequence execution with zero serial fallback occurrences:

- **Bounded Transpose MoE Owners Ready**: `48 / 48` layers (all 48 layers configured with 16 MiB owner partitions).
- **Native Batched K=5 Verify Frames**: `560` executed frames.
- **Batched K=5 Refusals**: `0` (0 refusals).
- **Serial Fallback Count (`SerialReason`)**: `0` (0 fallback events).
- **Serial Batch Frames**: `0` (0 frames).
- **Serial Frame First Alerts**: `0` (0 alerts).

---

## 4. Host Memory Floor Compliance

- **Required Minimum Floor**: `6,656 MiB` (Safety margin +512 MiB above 6,144 MiB emergency watchdog).
- **Measured Lowest Floor**: `9712 MiB`
- **Watchdog Kills**: `0`
- **Status**: **PASS** (Strictly compliant with DGX Spark shared hardware memory rules).

---

## 5. Artifact Provenance

- `server.log`: SHA-256 `538923a6637fe1d0c1449228b492a2d708a5b609210b209eb8411b21ec5aada9`
- `report.json`: SHA-256 `7f4ff510de69318e2c71efbbd1e45a496bce410b69c208213c89903c978aa4e6`
- `memory_samples.csv`: SHA-256 `5f36a96d04aca45e1934b8a2c81552895184576d2b98bb93115928d5222ae945`
