# Flux Engine `sm_121a` Kernel Registry

Tracking the custom CUDA/PTX and CuTe-DSL kernels shipped in `TheApathy/flux-engine`
for NVIDIA Grace Blackwell (`GB10`, `sm_121a`, 48 SMs, 32 MiB L2, 128 GB unified
LPDDR5x @ ~273 GB/s).

## Hardware Envelope: NVIDIA GB10 (`sm_121a`)

| Attribute | Value | Implication for Kernel Design |
|---|---|---|
| Streaming Multiprocessors | 48 SMs (Compute Capability `12.1a`) | Small grids starve, split-K and interleaved MoE tile planners (`MOE_PLAN_ORDER=inter`) are mandatory. |
| Shared Memory | 99 KB / SM | 2-CTA/SM occupancy requires `<= 48 KB` SMEM/CTA, 5-CTA/SM QSA scoring requires `<= 19 KB` SMEM/CTA. |
| L2 Cache | 32 MiB coherent L2 | Staging intermediate activations (`s.up` in 4-stream mHC, router weights at `w4096`) inside 32 MiB avoids LPDDR5x round-trips. |
| Unified Memory BW | 273 GB/s sustained (`301 GB/s` peak) | Decode is bandwidth-bound, prefill requires W4A8/FP8 Tensor Core fusion with zero register spills. |
| Register File | 255 registers / thread (`65,536` / SM) | Every production kernel is compiled with `-Xptxas -v -fmad=false` and gated on `0 bytes stack frame, 0 spill stores, 0 spill loads`. |
| Block-Scaled Tensor Cores | `mma.sync.aligned.m16n8k64...f8f6f4.block_scale` | Native hardware support for `NVFP4` (`E2M1` + `UE4M3`/`UE8M0`) and `MXFP8` (`E4M3` + `UE8M0`) register-backed MMA. |

---

## 1. `Qwen3.8-Flash-Next` (`qwen4_exp_text`) Kernel Stack (`kernels/gb10/qwen3.8-flash-next/nvfp4/`)

### 1.1 Compact Transposed W4A8 MoE (`cutlass_5222` / `v28q` & `v17q`)
- **`qwen4_moe_w4a8_v28` (`gate_up`)**:
  - 16-warp prologue and in-loop B-weight transcoding (`NVFP4` -> `MXFP8`, `tid < 512`) with hoisted `sfa[2]` scales and a zero-`sE` register-quad shuffle MXFP8 epilogue (`v28_mxfp8_reg_quad`).
  - `0` register spills, `0` epilogue barriers.
- **`qwen4_moe_compact_t_gemm_k32_v17q` (`down_proj`)**:
  - Dual-occupancy **2 CTAs/SM** (`256` threads, `122` registers, `0` spills) with `v17_mxfp8_reg_quad` epilogue.
- **`qwen4_mxfp8_quant_rows_v16q4_gate`**:
  - Fuses MoE router gate projection and dynamic MXFP8 slot quantization into a single pass.

### 1.2 Query-Sparse Attention (`qwen4_qsa_prefill_select` + `qkpv_amax_pipe`)
- **`qwen4_qsa_prefill_score` & `qwen4_qsa_prefill_score_gmax`**:
  - Uses `ldmatrix.sync.aligned.m8n8.x4.shared.b16` (`4x` fewer shared loads), shift/mask paging, and streaming score stores at **5 CTAs/SM** (`96` registers, `0` spills).
- **`qwen4_qsa_prefill_topk_v2` & `qwen4_qsa_prefill_topk_fused_c3_v2`**:
  - Bimodal block selection (`qwen4_qsa_prefill_select`): register top-k for `<= 8,192` blocks and radix-fused `c3_v2` (`32 MiB` score scratch) for `> 8,192` blocks.
- **`qwen4_qsa_prefill_attn_fp8_qkpv_amax_pipe`**:
  - `cp.async` double-buffered FP8 E4M3 sparse attention with per-half (`16`-token) `do_h0 / do_h1` HMMA skipping and fused `o_proj` absmax epilogue.

### 1.3 Gated DeltaNet Linear Attention (`gdn_umma_prefill`)
- **`qwen4_fp8_act_and_ba_gates_mma`**:
  - 64-row tile fused input activation FP8 quantization and `beta/alpha` gate MMA (`72` registers, `0` spills).
- **`qwen4_conv1d_l2norm_f16_prefill_v3_commit`**:
  - Shared-memory staged `TOKENS=64` causal `conv1d` + per-head `L2` normalization with direct recurrent state commit.
- **`atlas_gdn_fi_sm121a_prefill_f16` (`gdn_umma_prefill`)**:
  - Optional locally AOT-compiled SM121a FlashInfer chunked WY state recurrence (`crates/spark-model/csrc/fi_gdn/`; defaults to native CUDA GDN prefill when AOT objects are absent — see `csrc/fi_gdn/NOTICE`), paired with fused `rmsnorm + mxfp8_quant` (`GDN_NORM_FP8_V2=1`).

### 1.4 4-Stream Multi-Stream Hyper-Connections (`mHC`)
- **`qwen4_hc_prep_up` & `qwen4_hc_upmix_roof_t128`**:
  - Split preparation + 128-row roofline upmix keeping intermediate stream state 100% resident in GB10's 32 MiB L2 cache.
- **`qwen4_hc_combine_moe_gemm_mx_gatepre`**:
  - Warp-per-row period-6 unrolled stream combine + MoE router gate projection (`128` registers, `0` spills, `0` `token_to_perm` D2D copies).

### 1.5 Per-Layer Embeddings (`PLE`) & Lossless `K=5` MTP Verify
- **`qwen4_ple_prefill_fused`**:
  - Consumes 4-ring sector-aligned NVFP4 shards from the pre-deinterleaved `Arc<PackedPleBlock>` cache (`14.3 ms/8K` across 48 layers).
- **Multi-row `K=5` Verify Kernels**:
  - Batched 5-row QSA Flash-Decoding (`partial_rows_v2` / `reduce_rows`), 5-row GDN recurrence with undo rollback journaling, and device-chained 4-draft MTP proposer.

---

## 2. Shared & Multi-Model `sm_121a` Kernels (`kernels/gb10/common/`)

| Kernel Family | Files | Models Using It | Key Technique |
|---|---|---|---|
| **TurboQuant+ KV Cache (`turbo2`/`3`/`4`/`8`, `fp8k_turbo4v`, `bf16k_turbo4v`, `turbo4k_turbo3v`)** | `wht_bf16.cu`, `reshape_and_cache_turbo.cu`, `paged_decode_attn_turbo*.cu`, `tq_plus_innerq.cu` | All supported models (`--kv-cache-dtype turbo4` / `fp8k_turbo4v` / `turbo8`) | Randomized Walsh-Hadamard Transform + Lloyd-Max codebooks + matched-`L2`-norm scaling + asymmetric K/V & InnerQ channel equalization. |
| **Transposed W4A16 & Split-K GEMMs** | `w4a16_gemm.cu`, `w4a16_gemm_v2.cu`, `w4a16_gemv_sparse_cols.cu` | `Qwen3.8-27B`, `Qwen3.6-35B-A3B`, `SuperGemma4` | Double-buffered `cp.async` M16/M32/M128 transposed NVFP4 tiles + lossless FP32 split-K reduction. |
| **Multi-Sequence & Tree GDN (`WY17`)** | `gated_delta_rule_wy.cu`, `gated_delta_rule_tree_wy.cu`, `causal_conv1d_multi_seq.cu` | `Qwen3.8-27B` (DFlash), `Qwen3.5/3.6` | Lazy WY state commit (`ATLAS_WY17_LAZY=1`) and ancestor-correct tree recurrence. |
| **Multi-Head Latent Attention (`MLA`)** | `mla_prefill.cu`, `mla_decode.cu`, `kernels/gb10/deepseek-v4.1/` | `DeepSeek-V4.1`, `Mistral-Small-4` | Absorbed-latent MLA (`head_dim=512`, `qk_rope_head_dim=64`, `q_lora_rank=1280` on DeepSeek-V4.1 `deepseek_v41`; `HDIM=128` compressed KV on Mistral-Small-4). |
| **GQA Paged FlashAttention-4** | `paged_decode_attn.cu`, `inferspark_prefill_paged_*.cu` | All models | Conditional softmax rescaling, polynomial `sw_exp` (3 FMA + `ldexpf`), and BR64/BR128 tiled prefill. |
