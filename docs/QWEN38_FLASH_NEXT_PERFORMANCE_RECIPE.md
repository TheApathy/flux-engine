# Qwen3.8-Flash-Next on GB10 — production recipe and measurement record

Status: current as of v1.0.3 (initial measurement campaign 2026-10-17). Every number here is measured on a single NVIDIA DGX Spark (GB10 `sm_121a`, 128 GB unified LPDDR5X @ ~273 GB/s). Estimates are labeled as such. Where a lever was tested and found null or regressive, it is recorded as null: the negative results are the most expensive part of this document and the most useful.

---

## 1. Headline numbers

Canonical harnesses: [`bench/qwen38-flash-next-gb10/lil_decode_bench.py`](../bench/qwen38-flash-next-gb10/lil_decode_bench.py),
[`bench/qwen38-flash-next-gb10/estonia_bench.py`](../bench/qwen38-flash-next-gb10/estonia_bench.py),
and [`repro-kit/flux-bench`](../repro-kit/flux-bench): single stream (`C=1`,
`batch_size=1`), greedy (`temperature=0`), zero prefix cache (`--enable-prefix-caching false --ssm-cache-slots 0`).

| Configuration | Short / 4K Prefill | 8K Prefill | 16K Prefill | 128K Prefill | Decode (`K=5` MTP) | Free UMA Floor |
|---|---:|---:|---:|---:|---:|---:|
| **`v9` (`w16384`, qualified champion)** | **6,021.8** | **5,969.4** (`5,830.2`–`5,969.4`) | **6,082.2** peak (`6,076.7` med) | **5,694.4** peak (`5,684.0` med) | **56.5** med (`57.4` peak) | `7,072 – 8,450 MiB` |
| **`v9-w4096` (`w4096` interactive)** | **5,046.0** | **5,276.3** peak (`5,266.8` med) | **5,164.5** med | **4,899.8** peak (`4,891.7` med) | **56.5** med (`57.4` peak) | **11,907 – 12,695 MiB** |
| **`estonia-136k` (`136,576` tok, 6-hop)** | — | — | — | **5,400.4** (`25.29 s` TTFT) | **48.2** client (`48.6` srv) | **8,412 – 8,431 MiB** |
| **`k5-lossless-decode` (`w8192`, cache off)** | ~4,400 | ~4,410 | — | — | **56.5** med (`57.4` peak) | **10,500+ MiB** |
| **`exact-reference` (`K=1`, no speculation)** | 3,420 | 3,390 | 3,350 | — | **33.4 – 33.7** | `8,800+ MiB` |

Every `v9` and `v9-w4096` prefill sweep executes 1 uncounted transition request
followed by `N=3` timed repetitions per context length (`8K, 16K, 32K, 64K, 128K`)
on the same token slice from the 395,923-token corpus (`0f214e7481...`). All 20
requests per run produce byte-identical output hashes (`dbba921ce205` at 8K,
`0695b563acde` at 16K, `e47125968b3b` at 32K, `0695b563acde` at 64K, and
`c51cdffb6e38` at 128K).

Across the 10-prompt greedy verification suite (`0K..64K` context, 256 generated
tokens per prompt = 2,560 tokens total), `K=5` Inferact MTP speculative decode
matches `K=1` non-speculative greedy decode **10/10 bit-for-bit (`2,560 / 2,560`
tokens identical, `0` differences)** with mean acceptance length `3.02–3.15`
tokens/cycle.

On the 136,576-token 6-hop Estonia needle-in-a-haystack benchmark (`136,584`
active KV tokens during `K=5` verify), the `estonia-136k` profile scores **5/5
(100% PASS)** across two independent 5-run campaigns (`10/10` total), resolving
the 6-hop chain past the planted `Latvia` decoy in `25.29 s` median TTFT.

---

## 2. The serve recipes

Six profiles live in [`recipes/`](../recipes/) and are launched by
[`bench/qwen38-flash-next-gb10/serve.sh`](../bench/qwen38-flash-next-gb10/serve.sh):

```bash
MODEL_DIR=/path/to/Qwen3.8-Flash-Next-NVFP4-Offload \
MTP_DIR=/path/to/Qwen3.8-Flash-Next-Inferact-MTP \
./bench/qwen38-flash-next-gb10/serve.sh
```

Core `v9` (`flux-v9-max-prefill.env`) settings:

```text
FLUX_MODE=max-prefill  FLUX_QWEN4_PREFILL_WIDE=16384
FLUX_QWEN4_PREFILL_MOE_V28Q=2  FLUX_QWEN4_PREFILL_MHC_ROOF=t128
FLUX_QWEN4_PREFILL_QSA_FP8=qkpv  FLUX_QWEN4_PREFILL_QSA_PIPE=1
FLUX_QWEN4_PREFILL_GDN_CONV_V3=1  FLUX_QWEN4_PREFILL_GDN_NORM_FP8_V2=1
FLUX_PLE_READ_RINGS=4  FLUX_PLE_SECTOR_READ=1
FLUX_QWEN4_FAST_PROJ_FP8_WEIGHT_CACHE_RESERVE_GB=7
```

Why `w16384` vs `w4096` is an explicit choice:
- `w16384` (`--max-prefill-tokens 16384`) feeds `16,384 * 10 = 163,840` routed
  token-expert pairs per layer into the 512-expert MoE grouped GEMM (~320 rows
  per expert per chunk), saturating the 48 SMs on GB10 and pushing 16K prefill
  above **6,000 tok/s** (`1,348 ms` per 8K tokens). Its peak scratch leaves a
  `7,072–8,450 MiB` free UMA floor (safely above the `6,144 MiB` GB10 hardware
  watchdog).
- `w4096` (`--max-prefill-tokens 4096`, [`serve-w4096.sh`](../bench/qwen38-flash-next-gb10/serve-w4096.sh))
  quarters the transient prefill activation footprint, raising free unified
  memory to **11,907–12,695 MiB** (`+4.2 GiB` extra headroom) while still
  delivering **5,046–5,276 tok/s** prefill and identical **56.5 tok/s** decode.

---

## 3. Where prefill time goes (`1,348.0 ms / 8K` across 48 layers)

Measured at 16K context (`w16384`, `1,348.0 ms` per 8,192 tokens = `6,076.7 tok/s`):

| Subsystem | `ms / 8K` | Share | Kernel Path |
|---|---:|---:|---|
| **1. MoE Routed (512 top-10) + Shared Experts** | `684.5 ms` | `50.8%` | `cutlass_5222` / `qwen4_moe_w4a8_v28` (`gate_up`), `v17q` (`down_proj`), `v16q4_gate` fused quant |
| **2. Gated DeltaNet Linear Attention (36 layers)** | `312.8 ms` | `23.2%` | `gdn_umma_prefill` (`atlas_gdn_fi_sm121a_prefill_f16`), `conv1d_v3_commit`, `fp8_act_and_ba_gates_mma` |
| **3. QSA Sparse Attention + Projections (12 layers)** | `198.4 ms` | `14.7%` | `qwen4_qsa_prefill_select` (`ldmatrix.x4` score + `topk_v2`/`c3_v2`), `qkpv_amax_pipe` |
| **4. 4-Stream mHC Hyper-Connections (48 layers)** | `94.2 ms` | `7.0%` | `qwen4_hc_upmix_roof_t128`, `qwen4_hc_combine_moe_gemm_mx_gatepre`, `PIN_M=4096` cuBLASLt |
| **5. 4-Ring NVFP4 PLE Streaming + Glue** | `58.1 ms` | `4.3%` | `PackedPleBlock` cache + `qwen4_ple_prefill_fused` (`14.3 ms/8K` pure PLE) |
| **Total per 8,192 tokens** | **`1,348.0 ms`** | **`100.0%`** | **`6,076.7 – 6,082.2 tok/s` single-request (`C=1`)** |

### 3.1 Compact transposed W4A8 MoE (`cutlass_5222` / `v28q` & `v17q`)
- Routed MoE weights are stored in **NVFP4** (`E2M1` nibbles + `UE4M3` block-16
  scales) and multiplied against dynamic **MXFP8** activations on `sm_121a`
  block-scaled Tensor Cores (`mma.sync.aligned.m16n8k64.row.col.f32.e2m1.e2m1.f32.Kind::f8f6f4.block_scale.scale_vec::2X`).
- `gate_up` (`v28`) uses 16-warp prologue and in-loop B-weight NVFP4-to-MXFP8
  transcoding (`tid < 512`) with hoisted `sfa[2]` scales and a zero-`sE`
  register-quad shuffle MXFP8 epilogue (`0` register spills, `0` epilogue barriers).
- `down_proj` (`v17q`) runs at **2 CTAs/SM** (`256` threads, `122` registers,
  `0` spills), overlapping shared-memory B-tile loads across concurrent CTAs.
- `SHARED_W4A8_PRETRANSCODE` redirects its `5.07 MiB` per-layer scratch into the
  idle `ssm_deinterleaved` arena during MoE execution, saving `243.4 MiB` of
  persistent unified memory.

### 3.2 Gated DeltaNet (`gdn_umma_prefill` + `conv1d_v3_commit` + `rmsnorm + mxfp8_quant`)
- Input activations and `beta/alpha` gates are fused into a 64-row tile kernel
  (`qwen4_fp8_act_and_ba_gates_mma`, `72` registers, `0` spills).
- Causal `conv1d` + `L2` normalization runs in `qwen4_conv1d_l2norm_f16_prefill_v3_commit`
  (`TOKENS=64` with shared-memory filter staging and in-place state commit).
- The chunked WY recurrence dispatches to the optional locally AOT-compiled
  SM121a FlashInfer CuTe-DSL kernel (`atlas_gdn_fi_sm121a_prefill_f16`, see
  `crates/spark-model/csrc/fi_gdn/NOTICE`; falls back to native CUDA GDN
  prefill when AOT objects are absent), followed by fused output
  RMSNorm + FP8 E4M3 quantization (`GDN_NORM_FP8=2` + `GDN_NORM_FP8_V2=1`).

### 3.3 Query-Sparse Attention (`qwen4_qsa_prefill_select` + `qkpv_amax_pipe`)
- Block scoring (`qwen4_qsa_prefill_score` and `score_gmax`) uses
  `ldmatrix.sync.aligned.m8n8.x4.shared.b16` (`4x` fewer shared loads) at
  **5 CTAs/SM** (`96` registers, `0` spills).
- Block selection dispatches bimodally: `qwen4_qsa_prefill_topk_v2` for `<= 8,192`
  blocks and `qwen4_qsa_prefill_topk_fused_c3_v2` for `> 8,192` blocks.
- Sparse attention (`qwen4_qsa_prefill_attn_fp8_qkpv_amax_pipe`) uses `cp.async`
  double-buffering, per-half (`16`-token) `do_h0 / do_h1` HMMA skipping, and
  folds the `o_proj` absmax calculation into the attention epilogue.
- On Chunk 0 (`rows <= 2,048`), `index_qk_proj` and `iq_norm_rope_fused` are
  skipped on the dense-window prefix where sparse block indexing is unused.

### 3.4 4-Stream Multi-Stream Hyper-Connections (`mHC`)
- Clamping `query_m` to `PIN_M = 4096` for `ref_m >= 545` in
  `prefill_inject_gemm_admit.rs` keeps cuBLASLt's heuristic order consistent at
  `M = 16,384`, saving `228 ms` per 16K chunk.
- Splitting `qwen4_hc_prep_up` immediately ahead of `qwen4_hc_upmix_roof_t128`
  keeps `s.up` 100% L2-resident inside GB10's 32 MiB L2 cache.
- `qwen4_hc_combine_moe_gemm_mx_gatepre` unrolls period-6 warp-per-row combine +
  router gate projection (`128` registers, `0` spills, `0` `token_to_perm` D2D copies).

---

## 4. What measured null or regressed (do not re-try)

| Candidate / Lever | Hypothesis | Measured Effect | Why it failed |
|---|---|---|---|
| `GDN_NORM_FP8=3` during `K=5` speculative decode | Use store-warp asynchronous absmax reduction in unified serving | Breaks `LOSSLESS 10/10` (`2,560/2,560`) token parity on multi-row verify rollback | Store-warp reduction order is not state-deterministic across partial draft rollbacks. Kept in `fast-prefill` (prefill-only) and replaced by `GDN_NORM_FP8=2` + `V2=1` in `v9`. |
| Unbounded `FAST_PROJ_FP8_WEIGHT_CACHE=1` at 144K context | Cache all 96 FP8 projection matrices regardless of KV size | Drops `MemAvailable` below `6,144 MiB` at 136.5K tokens | Triggers the GB10 unified-memory watchdog. Fixed by `_RESERVE_GB=7` (`w16384`) and `_RESERVE_GB=10` (`estonia-136k`). |
| 3-stage `cp.async` SMEM buffering in `v17q` MoE `down_proj` | Deeper prefetch pipeline to hide LPDDR5x latency | **−6.4% MoE throughput** (`+44 ms/8K`) | Raises per-CTA shared memory above the 2-CTA/SM boundary (`99 KB/SM` on GB10), halving occupancy from 2 CTAs/SM to 1 CTA/SM. |
| Per-request `mmap`/`munmap` on NVFP4 PLE shards | Map PLE shards on demand to save host RAM | **+110 ms/8K** kernel-launch jitter | Causes kernel-wide ARMv9.2 TLB shootdowns across the 10 `Cortex-X925` cores. Replaced by persistent `Arc<PackedPleBlock>` cache (`14.3 ms/8K`). |
| `QSA_VERIFY_ROWS=1` above 131,072 tokens | Use multi-row Flash-Decoding split workspace at 136.5K | Scratch tile wrap past 128K boundary | Explicitly disabled (`QSA_VERIFY_ROWS=0`) in `flux-estonia-136k.env` for >131K contexts. |
