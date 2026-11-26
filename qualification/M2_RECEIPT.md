# Milestone 2 Qualification Receipt: Native 262K Context Qualification (R2)

**Verdict**: **PASS**  
**Timestamp**: `2026-10-24T14:33:17Z`  
**Hardware**: `NVIDIA DGX Spark (GB10, SM121a, 128 GB unified memory)`  
**Model**: `Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid`  
**Binary**: `target/release/flux-server`  
**Binary SHA-256**: `4eebc7dfe11bf97919b4df33c663fca0835d6a34f267b84a6bcbbb4b81784549`  
**Recipe**: `flux-qwen4-native-262k-w4096-bf16-owner16.env`  
**Gate**: `native-262k-w4096-owner16`  

---

## 1. Native 262K Allocation & Memory Floor Compliance

All native context allocations were verified under the certified `flux-qwen4-native-262k-w4096-bf16-owner16.env` operating environment on the single GB10 GPU.

- **Configured Context Limit (`max_seq_len`)**: `262,144` tokens
- **Prefill Chunk Width (`chunk_width_tokens`)**: `4,096` tokens
- **Full Endpoint Fill Capacity**: Succeeded to exactly `262,144` total tokens (`finish_reason="length"`)
- **Required Minimum Free Memory Floor**: `6,656 MiB` (Watchdog threshold 6,144 MiB + 512 MiB safety margin)
- **Lowest Measured Host Available Memory (`measured_floor_mib`)**: **`21,016 MiB`** (20.52 GiB available)
- **Pre-flight Available Memory**: `108 GiB`
- **Post-run Reclaimed Memory**: `107 GiB`
- **Watchdog Emergency Kills**: `0`
- **Floor Compliance**: **PASS** (Zero memory pressure, margin of +14,360 MiB above required floor)

---

## 2. 5-Depth Needle Retrieval Attestation

Five unique audit passkeys were inserted into the 262,080-token context across depths spanning 0.01 to 0.99. All passkeys were retrieved with 100% precision in both initial and repeated greedy decoding passes.

| Depth | Needle Token Offset | Passkey Label | Expected Value | Observed Output | Result |
|---|---|---|---|---|---|
| **0.01** (1%) | 2,641 | `ALPHA` | `alpha-0731826` | `alpha-0731826` | PASS |
| **0.25** (25%) | 65,553 | `BETA` | `beta-0492175` | `beta-0492175` | PASS |
| **0.50** (50%) | 131,079 | `GAMMA` | `gamma-0863204` | `gamma-0863204` | PASS |
| **0.75** (75%) | 196,595 | `DELTA` | `delta-0184932` | `delta-0184932` | PASS |
| **0.99** (99%) | 259,483 | `EPSILON` | `epsilon-0952174` | `epsilon-0952174` | PASS |

- **All Passkeys Retrieved**: `true` (5/5 correct)
- **Greedy Repeat Determinism**: `true` (100% bit-identical token IDs and completion text between Case 1 and Case 2)
- **Forced Endpoint Fill Passkeys Preserved**: `true` (all 5 passkeys present in Case 3 forced 262,144-token completion)

---

## 3. Boundary & Overflow Admission Attestation

The engine strictly enforces boundary admission and correlated overflow rejection at the exact 262,144-token context horizon without scheduler contamination.

| Test Case | Prompt Tokens | Requested Output | Total Tokens | Expected Response | Observed Response | Status |
|---|---|---|---|---|---|---|
| **Boundary Control Baseline** | 262,080 | 1 | 262,081 | HTTP 200, Token ID 15 | HTTP 200, Token ID 15 | PASS |
| **Inclusive Boundary Admission** | 262,143 | 1 | 262,144 | HTTP 200, Token ID 15 | HTTP 200, Token ID 15 | PASS |
| **Exclusive Overflow Rejection** | 262,144 | 1 | 262,145 | HTTP 400, `invalid_request_error` | HTTP 400, `invalid_request_error` | PASS |

- **Overflow Correlation**: `x-request-id` correctly echoed in error headers.
- **Zero Scheduler Dispatch**:
  - `atlas_prefill_progress_chunks_total`: 320 -> 320 ($\Delta = 0$)
  - `atlas_prefill_progress_tokens_total`: 1,310,463 -> 1,310,463 ($\Delta = 0$)
  - Confirmed 0 GPU scheduler dispatches during overflow attempt.

---

## 4. Post-Overflow Canary Recovery

Following the rejected overflow request, the engine was probed for execution health and state machine recovery.

1. **Service Health Probe**:
   - `GET /health` returned `HTTP 200` (`{}`).
2. **Canary Execution**:
   - Prompt: 262,080 tokens + 1 requested token (`ignore_eos=True`).
   - Observed Output Token ID: **`15`** (matches baseline control token ID `15` exactly).
3. **Canary Metric Counter Advances**:
   - $\Delta \text{ tokens}$: **`+262,080`** (prefill tokens advanced accurately).
   - $\Delta \text{ chunks}$: **`+64`** (64 prefill chunks processed).
   - Recovery Verified: **PASS**

---

## 5. MoE & QSA Telemetry Attestation

Telemetry captured across the entire server execution lifecycle confirms full architectural compliance:

- **Bounded-Transpose MoE Owner Census**:
  - `48 / 48` layers initialized with `BOUNDED_TRANSPOSE_OWNERS_READY`.
  - Exactly `87` destination owners per layer (`4,176` total destination owners).
  - Owner partition cap: `16,777,216 bytes` (`16 MiB`).
  - Source owner group cap: `67,108,864 bytes` (`64 MiB`).
  - Total requested payload: `67,948,314,624 bytes` eager pre-allocated.
- **QSA Sparse Side-Cache Geometry**:
  - `72` `QWEN4_PREFILL_QSA_CHUNK_RECEIPT` events recorded.
  - Terminal chunk ranges: `258048..262080` (rows=4032, tile_rows=10, chunk_rows=4096).
  - Operand Core: `qsa_sparse_bf16_non_bit_exact` (pure BF16 operands, zero quantized FP8 routes).
- **Engine & Kernel Safety Scan**:
  - Fatal log markers in `server.log`: **`0`**
  - Kernel errors / Xid / MMU faults in `journalctl --dmesg`: **`0`**

---

## 6. Artifact Provenance & Signoff

All artifacts are persisted under `qualification/native262k/`:

| Artifact | File Path | SHA-256 Digest |
|---|---|---|
| **Candidate Binary** | `target/release/flux-server` | `4eebc7dfe11bf97919b4df33c663fca0835d6a34f267b84a6bcbbb4b81784549` |
| **Server Log** | `qualification/native262k/server.log` *(omitted from public checkout; historical digest)* | `58c8b895d8d06f461b221a996629069bfd844363a0299a0768e41e3977cee70e` |
| **Test Responses** | `qualification/native262k/responses.jsonl` | `efff093f1fd6a7c75db500d055adf20606629d8566b29b798c278fdb8aba8049` |
| **Summary Report** | `qualification/native262k/report.json` | `473352696fc01225d505d461ebb57226e8eae331d0cfd504534c17aa1e14ff1a` |
| **Memory Samples** | `qualification/native262k/memory_samples.csv` | `7ab7a45c3bd8ccb0d69b6030c3ad0cbf75dcbc291292c72d484087bb7b982704` |
| **Kernel Journal** | `qualification/native262k/kernel-summary.json` | `360988afcade0738dd5a730d42fbef02ae4f275aedf2aaf3d2ccdd53ffd61637` *(historical pre-sanitization: `eaf8c6d989da982446e681110f60f4052b08aba10de38a366f6cef37854db7ed`)* |

### Independent Review Checklist
- [x] Context capacity $\ge 262,144$ tokens verified with zero memory exhaustion.
- [x] Needle retrieval validated across 5 distinct depths including $\le 1\%$ and $\ge 99\%$.
- [x] Greedy repetition deterministic bit identity verified.
- [x] Boundary admission and overflow rejection verified at token-accurate resolution.
- [x] Zero scheduler dispatch proven during overflow rejection.
- [x] Host memory floor ($\ge 6,656$ MiB) maintained throughout entire run.
- [x] 48/48 bounded transpose MoE receipts present with 16 MiB owner partitions.
- [x] Zero kernel fatal events, OOM kills, or MMU faults observed.
