# Benchmark Methodology & Verification: Roofline Physics, Prefix-Cache Controls, Speculative Accounting, and Quality Gates

> **Hardware Target:** Single NVIDIA GB10 Grace Blackwell Superchip (`sm_121a`, 48 SMs, `65,536` 32-bit registers/SM = `256 KiB` RF/SM, `30–32 MiB` L2 cache @ `2,188 GB/s` measured bisection bandwidth, `128 GB` 256-bit unified LPDDR5X-9400 @ `301 GB/s` theoretical peak / **`273 GB/s` sustained STREAM**, 20-core ARMv9.2 Grace CPU: 10× Cortex-X925 + 10× Cortex-A725, PCIe Gen5 x4 NVMe @ `~5.4–6.5 GB/s` `io_uring` `O_DIRECT`).
>
> **Audited Primary Models:**
> 1. **`Qwen3.8-Flash-Next` (`qwen4_exp`, 125B+51B PLE, 6B active; 48 layers: 36 GDN + 12 QSA, 512 routed experts top-10 + 1 shared expert, 4-stream mHC, 128-shard NVFP4 PLE):**
>    - **Prefill (`w16384` max-prefill profile):** **`6,082.2 tok/s` peak / `6,076.7 tok/s` median** @ 16K (`6,291.0 tok/s` median / `6,420.3 tok/s` single-chunk peak in `v0.9.15+` / `v1.0.x`), **`5,830.2 tok/s` @ 8K**, **`5,969.6 tok/s` @ 32K**, **`5,828.3 tok/s` @ 64K**, **`5,694.4 tok/s` @ 128K** (`5,905.6 tok/s` @ 128K and `5,570.6 tok/s` @ 262K in `v0.9.15+` / `v1.0.x`).
>    - **Prefill (`w4096` memory-constrained profile):** **`5,266.8 tok/s` @ 8K / `5,164.5 tok/s` @ 16K / `5,116.2 tok/s` @ 32K / `5,016.1 tok/s` @ 64K / `4,891.7 tok/s` @ 128K** under 4,096-token chunks (`5,447.7` → `5,003.1 tok/s` across `8K`–`262K` in `v0.9.15+` / `v1.0.x`).
>    - **Decode (`K=5` lossless native MTP, `--speculative --num-drafts 4 --mtp-vocab 96000`):** **`56.3 tok/s` client median / `56.5 tok/s` server median (`57.4 tok/s` client / `57.6 tok/s` server peak)** on standard mixed coding benchmarks, verified **`LOSSLESS 10/10` (`2,560 / 2,560` tokens bit-exact to `K=1` greedy decode)**; short-context `v0.9.15+` coding bursts reach `87.7–91.95 tok/s` (`C=1`) and `580.66 tok/s` aggregate (`C=8`).
>    - **Ultra-Long-Context (`136,593–136,612` tokens, Estonia 6-hop needle benchmark):** **`5/5` (`100%` completion, `0` OOM, `8,431 MiB` minimum `MemAvailable` floor)** at **`5,400.4–5,428.3 tok/s` prefill** and **`48.2 tok/s` client / `48.6 tok/s` server decode**.
> 2. **`Qwen3.8-27B` (`qwen3_5_text`, 64 layers: 48 GDN + 16 Full GQA Attention, dense 27B NVFP4):**
>    - **Decode (`DFlash` block-diffusion drafter, `gamma=15` => `K=16` verify batch, BF16 KV):** **`63.9 tok/s`** (`63.96 / 63.77 tok/s` on the 400-token coding probe; `8.33` mean emitted tokens / `129.8 ms` cycle = `64.2 tok/s`) and **`59.46 tok/s`** on ABI-verified `DFlash v3` coding decode.

---

## Key Verification Questions & Roofline Analysis

Hardware systems reviewers evaluating `6,082.2 tok/s` prefill and `56.3–63.9 tok/s` decode on a `273 GB/s` unified-memory Grace Blackwell (`sm_121a`) system typically examine four core questions. Each is answered below with roofline math, code anchors, and reproduction controls:

| # | Verification Question | Core Physics / Code Answer | Primary Code Anchor & Section |
|---:|---|---|---|
| **1** | **How do these numbers stay within the `273 GB/s` LPDDR5X bandwidth roofline?** | Prefill (`T=16,384`) amortizes the routed-expert weight stream (`~66.0 GB` per 16K chunk, `97.2%` of the `67.95 GB` expert bank touched) across `16,384` tokens (`~4.03 MB/token` => `241.9 ms` weight read per 16K chunk, `~9%` of the `2,693 ms` chunk time), so prefill is **Tensor-Core MMA + in-register `NVFP4 -> MXFP8` transcode + 4-stream mHC L2/DRAM bound**. Decode (`K=5` MTP on MoE, `K=16` DFlash on 27B dense) verifies `5` or `16` tokens in a **single batched Tensor-Core weight read**, multiplying the single-token bandwidth roofline by the mean speculative acceptance length (`accept_len = 3.02–3.36` for `K=5` MTP, `8.33` on the 400-token `K=16` DFlash coding probe). | Section 1, `crates/spark-model/src/layers/moe/qwen4_compact_ops.rs`, `crates/spark-model/src/layers/dense_ffn.rs` |
| **2** | **Are prefill throughput measurements hitting a prefix cache or reusing KV blocks?** | **No.** All cold-prefill benchmarks disable server-side prefix caching (`--enable-prefix-caching false`) and recurrent SSM caching (`--ssm-cache-slots 0`), advance prime-strided offsets (`STRIDE = 104729`) through distinct corpus windows, verify `usage.prompt_tokens_details.cached_tokens == 0`, drop warmup reps, and zero freshly allocated KV blocks on every request. | Section 2, `crates/spark-model/src/model/block_mgmt.rs`, `repro-kit/METHODOLOGY.md` |
| **3** | **Are decode `tok/s` inflated by counting rejected draft tokens or excluding drafter/verify overhead?** | **No.** `usage.completion_tokens` counts **only committed target-verified tokens** emitted to the client. Both **client wall-clock tok/s** (`completion_tokens / latency_s`) and **server decode-only tok/s** (`usage["response_token/s"]`, which includes 100% of drafter propose + target verify + rollback journal + sampler time) are reported side by side, and engine step rate (`steps_s = tok_s / accept_len`) is cross-checked via the Prometheus counter `atlas_spec_decode_verify_total`. | Section 3, `crates/spark-model/src/model/impl_b3.rs` |
| **4** | **Do `NVFP4` + `MXFP8` (`v28`/`v17qr2`) + `E4M3` QSA (`qkpv`) + `GDN_NORM_FP8=2` degrade output fidelity?** | **No.** Qualified serving profiles pass four verification gates: runtime receipt and poison checks, long-context corpus NLL (`delta_NLL <= +0.005 nats`, measured `+0.0001..+0.0017`), `15/15` multi-depth long-context needle retrieval (`32K/64K/128K`), and `LOSSLESS 10/10` (`2,560 / 2,560` tokens bit-exact to `K=1` greedy decode). Non-deterministic or high-NLL modes (such as `GDN_NORM_FP8=3` and E4M3 mHC residual storage) were explicitly rejected and quarantined. | Section 4, `crates/spark-model/src/layers/qwen4_qsa.rs` |

---

## 1. Roofline Physics Against the `273 GB/s` LPDDR5X Bandwidth Ceiling

### 1.1 Hardware Roofline Baseline of NVIDIA GB10 (`sm_121a`)

| Hardware Parameter | Physical Specification | Measured / Effective Value on GB10 (`sm_121a`) |
|---|---|---|
| Streaming Multiprocessors (SMs) | `48 SMs` (Blackwell `sm_121a` ISA) | `48 SMs`, `128 threads/warp-group`, `65,536` 32-bit regs/SM (`256 KiB` RF/SM) |
| Shared Memory / L1 per SM | Up to `128 KiB` configurable SMEM/L1 | `99 KiB` max dynamic SMEM per CTA without occupancy drop, `24–48 KiB` typical for 2–4 CTAs/SM |
| L2 Cache Capacity & Bandwidth | `30–32 MiB` unified die-level L2 | **`2,188 GB/s`** measured L2 read/write roof |
| System / GPU Memory | `128 GB` 256-bit LPDDR5X-9400 (Unified CPU+GPU) | `301 GB/s` theoretical peak, **`273 GB/s` sustained STREAM triad** (`~213 GB/s` random read/write roof) |
| Tensor Core Throughput (`sm_121a`) | FP8 (`e4m3` / `mxfp8` `mma.sync.aligned.m16n8k32`) & FP16 (`m16n8k16`) | `~180–200 TFLOPS` effective FP8 MMA, `~90–100 TFLOPS` effective FP16/BF16 MMA |
| NVMe Storage Subsystem | PCIe Gen5 x4 NVMe SSD | **`~5.4–6.5 GB/s`** sustained via 4-ring `io_uring` `O_DIRECT` (`2,048 B` sector reads on Cortex-X925 cores) |

A first-order calculation assumes every token must stream all model weights from LPDDR5X DRAM at `273 GB/s`. That bound applies **only to unbatched `K=1` single-token decode**. Below is the exact byte-and-FLOP roofline accounting for all three headline regimes.

---

### 1.2 Regime A: `Qwen3.8-Flash-Next` Prefill (`6,082.2 tok/s` Peak / `6,076.7 tok/s` Median @ 16K = `1,346.9 ms` per 8K Tokens)

#### Model Geometry (`qwen4_exp`)
- **Layers:** `L = 48` total layers = `36` Gated DeltaNet (`GDN`) linear-attention layers (`75%`) + `12` Query-Sparse Attention (`QSA`) layers (`25%`, every 4th layer: `3, 7, 11, ..., 47`).
- **Hidden Dimension & 4-Stream mHC:** `D = 2560`, expanded into `S = 4` Multi-Stream Hyper-Connection (`mHC`) residual streams (`hc_count = 4`, `hc_lowrank = 320`), carrying `4 x 2560 x 2 B (BF16) = 32 KiB/token` across `96` sub-layer boundaries (`2 x 48`).
- **MoE Sub-Layer (48 layers):** `E = 512` routed experts per layer (`I_moe = 640`, top-`k = 10` active per token) + `1` shared expert (`I_shared = 640`).
- **Per-Layer Embeddings (PLE):** 128-shard NVFP4 lookup table (`D_ple = 2560` across 48 layers), offloaded to NVMe (`ple-offload/` auto-detected or via `ATLAS_PLE_OFFLOAD_DIR`) with deduplicated sector DMA (`ATLAS_PLE_SECTOR_READ=1`, `ATLAS_PLE_READ_RINGS=4`).

#### Active Weight Traffic vs. Token Amortization at `T = 16,384` (`w16384`)
1. **Per-Expert NVFP4 Footprint:**
   Each routed expert has three matrices (`gate_proj`, `up_proj`, `down_proj`) of shape `2560 x 640`:
   ```text
   Params_expert = 3 * 2560 * 640 = 4,915,200 parameters
   ```
   In `NVFP4` (`0.5 B/param` packed `E2M1` + `1 B` `UE4M3` block scale per 16 elements = `0.5625 B/param`):
   ```text
   Bytes_expert = 4,915,200 * 0.5625 B = 2,764,800 B (~2.637 MiB)
   ```
2. **Active Experts per Layer during a 16K Chunk (`T = 16,384`):**
   Each chunk dispatches `16,384 x 10 = 163,840` `(token, expert)` pairs across `E = 512` routed experts (`mean_M_expert = 320.0` tokens/expert). Empirically, `97.2%` of the `67.95 GB` expert bank is touched at `16,384` rows, i.e. **`~66.0 GB` of routed expert weights per 16,384-token chunk** (`44.0 GB` gate/up + `22.0 GB` down), before the shared expert and the `2.08 GiB` cached FP8 dense projections (`FAST_PROJ_FP8_WEIGHT_CACHE_ENGAGED`). `ATLAS_QWEN4_PREFILL_MOE_PLAN_ORDER=inter` (`qwen4_moe_sort_plan_t_v8`) keeps consecutive 128-row tiles of each expert warm in L2 so each touched expert is streamed from DRAM roughly once per chunk.
3. **Amortized Weight Bandwidth Cost per Token:**
   ```text
   66.0 GB / 16,384 tokens = 4.03 MB/token
   t_weight_DRAM = 66.0 GB / 273 GB/s = 241.9 ms per 16K chunk (121.0 ms per 8K tokens)
   ```
   Out of the measured **`1,346.9 ms` per 8K tokens** (`6,082.2 tok/s` @ 16K), once-per-chunk routed-weight DRAM streaming accounts for **about `9%` of wall-clock time**, and roughly one-third of the routed MoE's `690 ms` per 16K chunk: gate/up is MMA + transcode bound, and down is bound by MXFP8 activation traffic.

#### Verified CUDA-Event Stage Census (`ATLAS_QWEN4_PREFILL_STAGES=1`)

Measured directly via CUDA-event stage markers (`ATLAS_QWEN4_PREFILL_STAGES=1`, see `crates/spark-runtime/src/cublaslt/stage_timer.rs` and [`docs/KERNEL-PROFILE.md`](../KERNEL-PROFILE.md)):

| Subsystem Stage | Measured Time (`ms / 8K` @ `w16384`) | Share (`%`) | Measured Time (`ms / 8K` @ `w4096`) | Primary Hardware Bottleneck on GB10 (`sm_121a`) |
|---|---:|---:|---:|---|
| **1. Routed + Shared MoE (`48` layers)** | **`684.5 ms`** | **`50.8%`** | `648.8 ms` (`362.1 ms` `gate_up` `v28` + `221.2 ms` `down` `v17qr2` + `48.3 ms` `shared_v7` + `8.6 ms` `v16q4_gate` + `8.6 ms` `sort_plan_v8`) | `mma.sync.aligned.m16n8k32.f32.e4m3.e4m3.f32` Tensor Core issue rate + in-register `NVFP4 -> MXFP8` weight transcode across `163,840` routed token-expert rows/layer |
| **2. GDN Linear Attention + Dense FP8 (`36` layers)** | **`312.8 ms`** | **`23.2%`** | `510.7 ms` (`242.9 ms` `in_proj_qkvz` FP8 GEMM + `94.1 ms` `out_proj` FP8 GEMM + `58.1 ms` `conv_v3` + `51.0 ms` `fi_gdn_sm121a` + `45.3 ms` `gdn_norm_fp8_v2` + `19.3 ms` `ba_gates`) | Cached FP8 `cuBLASLt` GEMMs (`qkvz` + `out_proj`) + `128 x 128` FP32 register-resident state recurrence (`gated_delta_rule`) |
| **3. QSA Sparse Attention (`12` layers)** | **`198.4 ms`** | **`14.7%`** | `223.9 ms` (`109.1 ms` `qsa_sparse_e4m3_qkpv` + `41.3 ms` `score_v2` + `fused_topk` + `43.3 ms` `o_proj_amax` + `30.2 ms` `attn_prep_fused_fp8`) | Top-2048 block-sparse FP8 `QK^T` and `PV` Tensor Core MMA + 64-head compressed indexer score scan (`ldmatrix.x4`) |
| **4. 4-Stream mHC Hyper-Connections (`96` boundaries)** | **`94.2 ms`** *(un-overlapped)* | **`7.0%`** | `375.7 ms` standalone (`112.1 ms` `upmix_roof_t128_amax` + `263.6 ms` `inject_gemm` fused into MoE/dense prologue) | `32 KiB/token` 4-stream BF16 residual read/write against GB10's `2,188 GB/s` L2 and `213 GB/s` random DRAM roof |
| **5. 4-Ring NVFP4 PLE Offload + Host/KV Setup** | **`57.0 ms`** | **`4.2%`** | `14.3 ms` (`ple_prefill_fused` + 4-ring `io_uring` `O_DIRECT` sector DMA on Cortex-X925 cores) + `2.1 ms` coalesced KV-zero | Overlapped PCIe Gen5 x4 NVMe DMA (`~5.4 GB/s`) + `qwen4_ple_conv_inject_fused_prefill` |
| **Total per 8,192 Tokens** | **`1,346.9 ms`** | **`100.0%`** | **`1,554.6 ms`** (`5,266.8 tok/s` @ 8K `w4096`) | **`8,192 / 1.3469 s = 6,082.2 tok/s` (`w16384` peak)** |

---

### 1.3 Regime B: `Qwen3.8-Flash-Next` Decode (`56.3–56.5 tok/s` Median / `57.4–57.6 tok/s` Peak, `K=5` Lossless Native MTP)

#### Why Plain `K=1` Decode Is Limited to `31.0–38.7 tok/s`
In single-token (`K=1`) autoregressive decode, every generated token activates `10` routed experts + `1` shared expert per layer across 48 layers, plus dense projections (`in_proj_qkvz`, `out_proj`, `q_proj`, `kv_proj`, `o_proj`), mHC mixing weights, and `lm_head`:
- **MoE weights read per token (`K=1`):** **`48 x (10 + 1) x 2.7648 MB = 1.46 GB`**
- **Dense projections + mHC + `lm_head` (`248,320` vocab):** **`~6.65 GB`**
- **Total DRAM weight read at `K=1`:** **`~8.11 GB/token`**
- At `273 GB/s` sustained LPDDR5X bandwidth, the minimum weight-read time is `8.11 GB / 273 GB/s = 29.7 ms/token` (`33.7 tok/s`), matching the measured non-speculative `K=1` decode baseline of **`31.0–38.7 tok/s`** (`25.8–32.2 ms/token`).

#### How `K=5` Native MTP Reaches `56.3–57.4 tok/s` Within the `273 GB/s` Roofline
When native `K=5` Multi-Token Prediction (`--speculative --num-drafts 4 --mtp-vocab 96000`, `ATLAS_QWEN4_MTP_CHAIN=1`) is enabled:
1. **Batched 5-Row Target Verification (`R = 5` candidate tokens: `1` bonus + `4` MTP drafts):**
   - All **dense weights** (`~6.65 GB`) are read **once per 5-row verify step** via multi-row batch GEMVs (`dense_gemv_batchn`, `dense_gemv_bf16_batchm`), amortizing `6.65 GB` across all `5` candidate rows (`1.33 GB/row`).
   - For the **routed MoE layers**, `5` candidate tokens × `10` top-k experts = `50` expert assignments per layer. Because consecutive draft tokens are correlated, `ATLAS_QWEN4_MOE_DEDUP=1` deduplicates shared expert hits across the 5 rows, reading on average `~31–36` unique experts per layer (`~4.1–4.8 GB` across 48 layers) instead of `50`.
   - Total target verify DRAM traffic for all `5` rows is **`~10.8–11.5 GB/step`**, plus `~1.2 GB` for the 1-layer device-chained 4-step MTP drafter (`qwen4_mtp_chain`, `--mtp-vocab 96000`), totaling **`~12.0–12.7 GB` per speculative cycle**.
2. **Cycle Time and Effective Token Rate:**
   - Theoretical bandwidth floor per 5-row cycle at `273 GB/s`: **`12.3 GB / 273 GB/s ~= 45.1 ms/cycle`**.
   - Measured cycle time: **`53.5–59.5 ms/cycle`** (`16.8–18.7 verify steps/s`, including `~6.0–7.2 ms` 4-step MTP propose + `~46–52 ms` 5-row target verify).
   - Measured acceptance length (`accept_len = (completion_tokens - conc) / verify_steps`): **`3.02` to `3.36` tokens/step** across standard mixed benchmarks (and up to `4.35` on structured code).
   - **Resulting throughput:**
     ```text
     Decode tok/s = 17.6 steps/s * 3.20 accepted tokens/step = 56.3 tok/s median (57.4 tok/s peak)
     ```
   - Effective DRAM traffic per committed token drops from `8.11 GB/token` (`K=1`) to **`12.3 GB / 3.20 = 3.84 GB/token`** (`56.3 tok/s * 3.84 GB/tok = 216.2 GB/s`, or **`79.2%` of sustained STREAM bandwidth**).

---

### 1.4 Regime C: `Qwen3.8-27B` Dense Hybrid Decode (`63.9 tok/s` `DFlash` `gamma=15`, BF16 KV)

#### Why Plain `K=1` Dense 27B Decode Is Bandwidth-Bound at `~13.9–19.5 tok/s`
`Qwen3.8-27B` is a **dense** 27B hybrid model (64 layers: 48 GDN + 16 Full GQA Attention, `hidden_size = 5120`, `intermediate_size = 17408`).
- In NVFP4 (`0.5625 B/param`), the 64-layer target weights occupy **`~14.0 GB`** (with full 64-layer streaming at `273 GB/s`, non-speculative `K=1` throughput is memory-bandwidth limited to `~13.9–19.5 tok/s`).

#### How `DFlash` (`gamma=15` => `K=16`) Achieves `63.9 tok/s` (`BF16 KV`)
1. **Single-Pass 16-Row Tensor-Core Target Verification:**
   - The block-diffusion drafter (`z-lab/Qwen3.8-27B-DFlash2`, **5 layers** B8 qualified default; or legacy `z-lab/Qwen3.6-27B-DFlash`, **8 layers** B16; `~0.85 GB` NVFP4) proposes a draft block in parallel in a single lightweight pass.
   - The 64-layer 27B target model verifies **all `K = gamma + 1 = 16` candidate tokens simultaneously** in a single `M=16` Tensor Core pass using custom split-K `M16` NVFP4 kernels (`ffn_gateup_fused_kgamma` in `crates/spark-model/src/layers/dense_ffn.rs`, `ssm_ffn_kgamma_dense` in `crates/spark-model/src/layers/qwen3_ssm/trait_decode_batched.rs`, and `gated_delta_rule_wy17_vsplit` in `kernels/gb10/qwen3.8-27b/nvfp4/gated_delta_rule_wy17_vsplit.cu`).
   - Because `M = 16` maps directly onto Tensor Core MMA tiles (`m16n8k16` / `m16n8k32`), **reading `14.0 GB` of target weights once verifies all `16` tokens** in **`~129.8 ms/cycle`**.
2. **Bandwidth Amortization Math (400-Token Coding Probe, BF16 KV):**
   - On the qualified 400-token coding probe with `gamma=15` (see [`docs/QWEN38_PERFORMANCE_RECIPE.md`](../QWEN38_PERFORMANCE_RECIPE.md)), mean emitted tokens per cycle is **`8.33`** (`7.33` accepted drafts + `1` bonus token) at **`129.8 ms/cycle`**:
     ```text
     Throughput = 8.33 emitted tokens / 0.1298 s/cycle = 64.18 tok/s (63.96 / 63.77 tok/s across repeated BF16-KV runs)
     ```
   - Weight DRAM traffic per committed token drops from `~14.0 GB/token` (`K=1`) to **`(14.0 + 0.85) GB / 8.33 = 1.78 GB/token`** (`63.9 tok/s * 1.78 GB/tok = 113.9 GB/s`, well within GB10's `273 GB/s` sustained LPDDR5X ceiling).
   - *(On the longer 800-token coding probe with lower mean acceptance of `5.81` accepted drafts + `1` bonus = `6.81` emitted tokens over a `131.25 ms` cycle, throughput scales directly with acceptance length: `6.81 / 0.13125 = 51.89 tok/s`.)*

---

## 2. Prefix-Cache Falsification Controls

Five independent mechanisms guarantee **zero prefix-cache hits** during cold-prefill measurements (see also [`repro-kit/METHODOLOGY.md`](../../repro-kit/METHODOLOGY.md)):

1. **Server-Side Cache Disabling:** Cold-prefill benchmark profiles launch `flux-server` with `--enable-prefix-caching false` and `--ssm-cache-slots 0`, disabling both radix-tree KV prefix reuse and recurrent SSM state caching at the server CLI boundary.
2. **Prime-Stride Fresh Corpus Windows (`STRIDE = 104729`):** Every measured prefill repetition slices a distinct window of the on-disk benchmark corpus using a prime stride so consecutive requests share zero leading tokens.
3. **Per-Request SHA-256 Prompt Receipts:** Benchmark harnesses log `prompt_sha256` and `ids_sha256` per request so auditors can verify distinct prompt payloads across every repetition.
4. **Explicit `cached_tokens_max == 0` Assertion:** Harnesses read `usage.prompt_tokens_details.cached_tokens` from every OpenAI response and assert `cached_tokens_max == 0` across all timed repetitions.
5. **Fresh Sequence Allocation & Warmup Exclusion:** Every `/v1/completions` request allocates a fresh sequence slot (`alloc_seq_async` in `crates/spark-model/src/model/trait_impl/meta.rs`) and zeroes newly allocated paged KV blocks (`zero_kv_blocks_coalesced` in `crates/spark-model/src/model/block_mgmt.rs`). Warmup requests and post-run determinism checks are excluded from timed throughput aggregates.

---

## 3. Decode Token & Wall-Clock Accounting

### 3.1 Only Committed Target-Verified Tokens Are Counted
In `spark-server`:
- A speculative verify step with `K=5` (`1` bonus token + `4` MTP drafts) or `K=16` (`1` bonus token + `15` DFlash drafts) submits `K` candidate tokens to the target model.
- If the target model accepts `a in {0..K-1}` draft tokens plus the `1` guaranteed bonus token, `1 + a` tokens are committed to the output sequence and the remaining `(K - 1 - a)` rejected suffixes are rolled back via the QSA undo journal (`crates/spark-model/src/layers/qwen4_qsa/device.rs` & `crates/spark-model/src/layers/qwen3_attention/qwen4_device_attention.rs`) and GDN state checkpoint restoration.
- **`usage.completion_tokens` increments strictly by `(1 + a)`** — rejected draft tokens are discarded on-device and never counted.

### 3.2 Wall-Clock Metrics Reported Side by Side
1. **`agg_tok_s` (Client Aggregate `tok/s`, Prefill-Included):** Total committed completion tokens divided by full client wall-clock latency (including HTTP overhead and TTFT).
2. **`decode_only_tok_s` (Client Prefill-Excluded `tok/s`):** Committed completion tokens divided by `(client_latency_s - measured_ttft_s)`.
3. **`server_decode_tps` (`usage["response_token/s"]`, Server-Side Decode Wall Clock):** Measured inside `spark-server` from Token 1 emission to final token emission, including 100% of drafter propose time, target verify time, QSA/GDN rollback journal time, PLE lookups, and sampling.
4. **`accept_len` and `steps_s` (Hardware Verify Step Rate):** Cross-checked against the Prometheus counter `atlas_spec_decode_verify_total`:
   ```text
   accept_len = (completion_tokens - concurrency) / delta(atlas_spec_decode_verify_total)
   steps_s    = decode_only_tok_s / accept_len
   ```

---

## 4. Numerical Fidelity, Quality Gates & Quarantined Regressions

### 4.1 Precision Architecture
- **Weights & MoE Activation Quantization:** Model weights are stored in NVIDIA ModelOpt `NVFP4` (`E2M1` + `UE4M3` per 16 elements + FP32 `weight_scale_2`). During MoE prefill, BF16 activations quantize to 32-element `MXFP8` (`E4M3` + `UE8M0`), NVFP4 weights transcode in registers to matching `E4M3` tiles, and every `m16n8k32` Tensor Core MMA accumulates in **full `FP32`**.
- **QSA Sparse Attention (`qkpv`):** Per-head/per-row `E4M3` Q/K/V operands accumulate `Q*K^T`, online softmax rescaling, and `P*V` in **full `FP32`** (`kernels/gb10/qwen3.8-flash-next/nvfp4/qwen4_qsa_prefill_attn_fp8.cu`).
- **GDN Recurrent State (`FP32` + `GDN_NORM_FP8=2`):** All 36 GDN layers maintain the `128 x 128` per-head state matrix `H` in **full `FP32`**. `ATLAS_QWEN4_PREFILL_GDN_NORM_FP8=2` (`qwen4_gdn_norm_fp8_v2`) is deterministic and passes `10/10` lossless `K=5` parity; `mode = 3` (store-warp `nf8`) exhibited a multi-chunk ordering race that broke lossless parity and was quarantined from all serving profiles.

### 4.2 Four-Tier Quality Gate Summary

| Tier | Gate | Protocol | Acceptance Bar | Measured Result |
|---:|---|---|---|---|
| **Tier 1** | **Runtime Receipt & Determinism Gate** | Mandatory kernel-engagement receipts + scratch poison (`ATLAS_ALLOC_POISON=1`) + cross-process repeatability | `100%` receipts present, `100%` bit-exact across repeats | **PASS** (`20/20` bit-identical determinism) |
| **Tier 2** | **Long-Context Corpus NLL (`delta_NLL`)** | Teacher-forced log-probability across `8K`–`128K` vs exact reference | `abs(delta_NLL) <= +0.0050 nats` | **PASS** (`+0.0001..+0.0017 nats`; lossy E4M3 mHC residual storage scored `+4.05%..+6.51%` and was rejected) |
| **Tier 3** | **Multi-Depth Long-Context Needle Retrieval** | `5` passkey needles at `5%..95%` depth across `32K`, `64K`, `128K` (`15` cells) + `262K` (`5/5`) + Estonia `136.5K` (`5/5`) | `100%` exact needle retrieval | **PASS** (`15/15` @ `32K`–`128K`, `5/5` @ `136.5K`, `5/5` @ `262K`) |
| **Tier 4** | **Speculative Decode `LOSSLESS 10/10` Gate** | `10` diverse coding/reasoning/math prompts × `256` tokens (`2,560` tokens) under `K=5` MTP (`temp=0`) vs `K=1` greedy decode | `2,560 / 2,560` bit-exact tokens + `0` corrupted runs | **PASS** (`10/10` prompts, `2,560 / 2,560` exact token IDs) |

### 4.3 Engineering Fixes Behind `LOSSLESS 10/10` `K=5` Parity
1. **QSA Transactional Undo Journal (`crates/spark-model/src/layers/qwen4_qsa/device.rs`, `crates/spark-model/src/layers/qwen3_attention/qwen4_device_attention.rs`):** Snapshots modified QSA pool and indexer slots before a multi-row verify pass and restores rejected suffixes on partial acceptance.
2. **Final-Layer Crop Trailing `MTP_LASTK=64` Window (`crates/spark-model/src/layers/qwen4_prefill_moe/last_layer_crop.rs`, `crates/spark-model/src/model/impl_b3.rs`):** Retains the trailing 64 final-layer hidden states on the last prefill chunk so the 1-layer MTP drafter warms its KV cache identically to uncropped prefill.
3. **4-Byte Device Scale Fence (`crates/spark-model/src/layers/qwen4_fast_proj.rs`):** Stream-orders activation `amax` completion ahead of cached FP8 `cuBLASLt` GEMMs on short prompts.
4. **Long-Context (`>131K`) Verify Pool Endpoint (`crates/spark-model/src/layers/qwen4_qsa.rs`):** Computes `pool_endpoint = seq_len + rows - 1` so `K=5` verification crossing 131,072 tokens succeeds reliably (`5/5 PASS` on Estonia 136.6K with `8,431 MiB` minimum free memory floor).

### 4.4 Transparent Disclosure: Quarantined `72.25 tok/s` Prefill-Attention ABI Bug (`Qwen3.8-27B`)
During early `Qwen3.8-27B` `DFlash` development, an experimental `NVFP4-KV` prefill-attention path recorded `~72.25 tok/s` median across 5 runs before a PTX ABI audit revealed that the host launcher passed **13 kernel arguments** to an **11-parameter PTX entrypoint** (causing `query_start=0` to be decoded as `num_q_heads=0` so chunk-0 attention returned without writing output). That measurement was immediately **quarantined as invalid performance evidence** and removed from all release targets. The verified, ABI-compliant `Qwen3.8-27B` baseline is **`63.9 tok/s` (`BF16 KV`, `DFlash` `gamma=15`)** (`59.46 tok/s` on ABI-corrected `DFlash v3`).
