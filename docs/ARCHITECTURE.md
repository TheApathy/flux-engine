# Flux Engine Architecture

A technical walkthrough of the Cargo workspace, the `sm_121a` CUDA/PTX kernel
stack, the KV cache quantization pipeline, and the request lifecycle.

---

## 1. Crate Graph

Flux Engine is a single Cargo workspace with 16 crates organized into four layers:

```
                         ┌─────────────────────────┐
   HTTP / scheduling     │      spark-server       │
                         │  (flux-server, flux-cli,│
                         │  spark: OpenAI +        │
                         │  Anthropic API, SLAi    │
                         │  scheduler, xgrammar)   │
                         └──────────┬──────────────┘
                                    │
                         ┌──────────┴──────────────┐
   model + layers        │       spark-model       │
                         │  (TransformerModel,     │
                         │  per-arch layers, MoE,  │
                         │  GDN/SSM, QSA/GQA/MLA,  │
                         │  mHC, PLE, MTP/DFlash)  │
                         └──────────┬──────────────┘
                                    │
            ┌───────────────────────┼─────────────────────┐
            │                       │                     │
   GPU runtime          inter-rank comm           NVMe-backed swap
   ┌────────┴────────┐ ┌────────────┴───────┐  ┌──────────┴────────┐
   │  spark-runtime  │ │    spark-comm      │  │  spark-storage    │
   │  (PagedKvCache, │ │  (NCCL backend +   │  │  (HSS, io_uring,  │
   │  buffers,       │ │  RoCEv2, all_reduce│  │  4-ring PLE DMA)  │
   │  cuda_backend,  │ │  broadcast)        │  └───────────────────┘
   │  sampler,       │ └────────────────────┘
   │  weights)       │
   └────────┬────────┘
            │
   ┌────────┴────────┐
   │  atlas-kernels  │   PTX bytes embedded at compile time
   │  (build.rs:     │   per (hardware, model, quant) tuple
   │  compile *.cu   │   from kernels/<hw>/<model>/<quant>/
   │  → PTX → embed) │
   └────────┬────────┘
            │
   ┌────────┴────────┐   Shared vocabulary across the workspace:
   │  atlas-core     │   ModelConfig, registry, hardware capability
   │                 │   enums, error types.
   └─────────────────┘

   Support & reserved interface stub crates:
   xgrammar (pure-Rust pushdown grammar engine) · rest-store · cufile-sys
   atlas-activation · atlas-norm · atlas-quant · atlas-embed · atlas-reduce (interface stubs)

   Benchmark crate:
   atlas-spark-bench (Criterion + SM121a kernel microbenchmarks)
```

---

## 2. The `sm_121a` Kernel Stack (NVIDIA Grace Blackwell GB10)

GB10 (`sm_121a`) pairs 48 Blackwell SMs and a 32 MiB L2 cache with 128 GB
(`119.7 GiB` OS-visible) of unified LPDDR5x memory at **273 GB/s** sustained
bandwidth. Unlike datacenter B200 (`sm_100a`), `sm_121a` has no Tensor Memory
(`TMEM`) and no thread-block clusters larger than `1x1x1`, but it exposes native
register-backed block-scaled FP4/FP8 Tensor Core instructions:

```ptx
mma.sync.aligned.m16n8k64.row.col.f32.e2m1.e2m1.f32.Kind::f8f6f4.block_scale.scale_vec::2X
```

Every hot-path kernel in `kernels/gb10/` is written directly against this ISA and
compiled with `nvcc -arch=sm_121a -Xptxas -v -fmad=false` under a hard
**zero-register-spill** contract (`0 bytes stack frame, 0 spill stores, 0 spill loads`).

### 2.1 `cutlass_5222` Compact Transposed W4A8 MXFP8 MoE (`v28` / `v28q` & `v17q`)

On `Qwen3.8-Flash-Next`, every layer routes each token to top-10 of 512 routed
experts plus 1 shared expert (`moe_intermediate_size = 640`). At `w16384`, a
single prefill chunk dispatches `16,384 × 10 = 163,840` token-expert pairs
(~320 rows per expert), making MoE 50.8% of total prefill time (`684.5 ms / 8K`):

- **Fused Router + Slot Quantization (`rmsnorm + mxfp8_quant`):**
  - `qwen4_mxfp8_quant_rows_v16q4_gate` fuses post-attention RMSNorm, router
    gate input preparation, and dynamic MXFP8 (`E4M3` + `UE8M0` block-32 scale)
    slot quantization into one pass.
  - Router gate projection dispatches chunk-adaptively: `fp8_gemm_ex` with L2
    `ROUTER_WEIGHT_CACHE` at `w4096` (`8.6 ms/8K`) and single-pass
    `ops::fp8_gemm_n128` at `w16384`.
  - Expert histogramming, prefix scan, and interleaved tile planning
    (`MOE_PLAN_ORDER=inter` in `qwen4_plan_order_rule.rs`) are fused into
    `ATLAS_QWEN4_PREFILL_MOE_FUSED_SORT_PLAN=1` so multi-tile and single-tile
    experts interleave cleanly across all 48 SMs without tail-wave starvation.
- **`gate_up` Grouped GEMM (`cutlass_5222` / `qwen4_moe_w4a8_v28`):**
  - Consumes compact transposed **NVFP4** (`E2M1` + `UE4M3`) weights and dynamic
    **MXFP8** activations.
  - Uses 16-warp prologue and in-loop B-weight transcoding (`tid < 512`) with
    hoisted `sfa[2]` scale registers and a zero-`sE` register-quad shuffle MXFP8
    epilogue (`v28_mxfp8_reg_quad`), eliminating shared-memory epilogue staging
    and barriers (`0` spills).
- **`down_proj` Grouped GEMM (`qwen4_moe_compact_t_gemm_k32_v17q`):**
  - Tuned for **2 CTAs/SM dual occupancy** (`256` threads, `122` registers, `0`
    spills, `<= 48 KB` SMEM/CTA) with the `v17_mxfp8_reg_quad` epilogue so
    one CTA's MMA pipeline overlaps the other CTA's LPDDR5x weight loads.
- **Zero-Allocation Shared Expert Pretranscoding:**
  - `SHARED_W4A8_PRETRANSCODE` aliases its `5.07 MiB` scratch buffer onto the idle
    `ctx.buffers.ssm_deinterleaved()` arena during MoE execution, saving
    `243.4 MiB` of persistent unified memory across 48 layers.

### 2.2 Query-Sparse Attention (`qwen4_qsa_prefill_select` & `qkpv_amax_pipe`)

Twelve of the 48 layers in `Qwen3.8-Flash-Next` use Query-Sparse Attention
(`num_attention_heads = 24`, `num_key_value_heads = 2`, `head_dim = 256`,
`indexer_budget = 2048`, `indexer_compress_ratio = 4`), accounting for 14.7% of
prefill (`198.4 ms / 8K`):

- **`ldmatrix.x4` Block Scoring (`qwen4_qsa_prefill_score` & `score_gmax`):**
  - Stages compressed query/key index tiles in shared memory and loads them via
    `ldmatrix.sync.aligned.m8n8.x4.shared.b16` (`4x` fewer shared loads than
    scalar LDS) with shift/mask paging and streaming score stores.
  - Runs at **5 CTAs/SM** (`96` registers, `0` spills).
- **Bimodal Block Selection (`qwen4_qsa_prefill_select`):**
  - For sequence depths `<= 8,192` blocks, `qwen4_qsa_prefill_topk_v2` selects
    the top-k blocks directly in registers.
  - For sequence depths `> 8,192` blocks (`32K..136.5K`),
    `qwen4_qsa_prefill_topk_fused_c3_v2` uses a radix-assisted selector backed by
    a `32 MiB` score scratch slice inside the `192 MiB` prefill reserve.
  - On Chunk 0 (`rows <= 2,048`), `index_qk_proj` and `iq_norm_rope_fused` are
    bypassed on the dense-window prefix where all blocks are attended anyway.
- **Double-Buffered FP8 Sparse Attention (`qwen4_qsa_prefill_attn_fp8_qkpv_amax_pipe`):**
  - Stages selected KV blocks via `cp.async` double-buffering, skips inactive
    16-token half-tiles (`do_h0 / do_h1` HMMA skipping), and folds the outgoing
    `o_proj` activation absmax calculation into the attention epilogue
    (`OPROJ_AMAX_FUSED=1`).

### 2.3 Gated DeltaNet Linear Attention (`gdn_umma_prefill`)

Thirty-six of the 48 layers use Gated DeltaNet (`linear_num_key_heads = 16`,
`linear_num_value_heads = 48`, `head_dim = 128`, `conv_kernel_dim = 4`),
accounting for 23.2% of prefill (`312.8 ms / 8K`):

- **64-Row Fused Input Quantization + Gate MMA (`qwen4_fp8_act_and_ba_gates_mma`):**
  - Quantizes the incoming hidden state to FP8 for the large `in_proj_qkvz` GEMM
    while simultaneously computing the `beta` and `alpha` decay gates in a
    64-row tile (`72` registers, `0` spills).
- **Shared-Memory Staged Causal Conv1d (`qwen4_conv1d_l2norm_f16_prefill_v3_commit`):**
  - Processes 64-token tiles (`TOKENS=64`) with filter weights staged in shared
    memory, applies per-head `L2` normalization on `Q` and `K`, and commits the
    final 3-token convolution history directly into the sequence's recurrent slot.
- **Chunked WY State Recurrence (`gdn_umma_prefill` / `atlas_gdn_fi_sm121a_prefill_f16`):**
  - Dispatches through the native CUDA GDN chunked WY recurrence kernel by
    default, or the optional locally AOT-compiled SM121a FlashInfer CuTe-DSL
    kernel (`crates/spark-model/csrc/fi_gdn/`, see its `NOTICE` for local
    source-build instructions), updating the FP32
    recurrent state tensor and emitting FP16 layer outputs.
- **Fused Output `rmsnorm + mxfp8_quant` (`GDN_NORM_FP8_V2=1`):**
  - Applies gated RMSNorm (`z * rms_norm(out)`) and quantizes directly to
    static-scaled FP8 E4M3 (`GDN_NORM_FP8=2`) ahead of `out_proj`, preserving
    100% state determinism across `K=5` speculative rollbacks.

### 2.4 4-Stream Multi-Stream Hyper-Connections (`mHC`)

All 48 layers maintain a 4-stream hyper-connection residual bus (`hc_count = 4`,
`hc_lowrank = 320`, `hidden_size = 2560`), taking 7.0% of prefill (`94.2 ms / 8K`):

- **`PIN_M=4096` cuBLASLt Heuristic Admission:**
  - Clamping `query_m` to `PIN_M = 4096` for `ref_m >= 545` in
    `prefill_inject_gemm_admit.rs` prevents cuBLASLt heuristic plan flips at
    `M = 16,384`, unlocking fused `injgemm`, `combine`, and `postpre` (`-228 ms`
    per 16K chunk).
- **L2-Resident Roofline Upmix (`qwen4_hc_prep_up` + `qwen4_hc_upmix_roof_t128`):**
  - Splitting `qwen4_hc_prep_up` immediately ahead of `qwen4_hc_upmix_roof_t128`
    keeps `s.up` 100% resident inside GB10's 32 MiB L2 cache.
- **Warp-Per-Row Stream Combine (`qwen4_hc_combine_moe_gemm_mx_gatepre`):**
  - Period-6 unrolled warp-per-row stream combine fused with MoE gate preparation
    (`128` registers, `0` spills, `0` `token_to_perm` D2D copies).

### 2.5 4-Ring NVFP4 Per-Layer Embedding (`PLE`) Streaming

`Qwen3.8-Flash-Next` includes a 51B-parameter (`20,000,000 × 2,560`) 3-gram
Per-Layer Embedding table split across 128 sector-aligned NVFP4 shards:

- [`scripts/repack_qwen38_flash_next_ple_nvfp4.py`](../scripts/repack_qwen38_flash_next_ple_nvfp4.py)
  converts the 128 PLE shards into group-16 NVFP4 (`E2M1` packed nibbles + `FP8`
  group scales + `FP32` global scale).
- At runtime, `PackedPleBlock` caches pre-deinterleaved shards in `Arc` host
  memory with 4 asynchronous DMA rings (`ATLAS_PLE_READ_RINGS=4`, pinned to the
  10 Grace `Cortex-X925` cores via `ATLAS_PLE_READ_AFFINITY=big`), reducing
  48-layer PLE overhead to **`14.3 ms / 8K`** with zero `munmap` TLB shootdowns.

### 2.6 Lossless `K=5` Inferact MTP & `gamma=15` DFlash Speculative Verification

- **Lossless `K=5` Inferact MTP (`Qwen3.8-Flash-Next`):**
  - A 1-layer full-attention MoE draft head proposes 4 draft tokens per cycle in
    a device-chained loop (`ATLAS_QWEN4_MTP_CHAIN=1`, single H2D / single D2H).
  - The 48-layer target model verifies all `K=5` rows (`1` bonus + `4` drafts) in
    a single batched pass (`ATLAS_QWEN4_K5_BATCH_{ATTN_QKV,GDN,HYPER,SSM,CONV}=1`)
    using 0-ULP multi-row QSA Flash-Decoding (`partial_rows_v2` / `reduce_rows`)
    and deterministic GDN undo journaling on partial acceptance.
  - Certified **`57.4 tok/s` peak (`56.3–56.5 tok/s` coding median)** and
    **`LOSSLESS 10/10`** (`2,560 / 2,560` greedy tokens bit-exact vs `K=1`).
- **`gamma=15` DFlash Speculative Decode (`Qwen3.8-27B`):**
  - A bidirectional `DFlashDraftModel` (`5`-layer DFlash2 `B8` qualified default
    `z-lab/Qwen3.8-27B-DFlash2`, or `8`-layer DFlash v1 `B16`
    `z-lab/Qwen3.6-27B-DFlash`) conditioned on 5 captured target hidden layers
    proposes a draft block per step, verified by split-K `ffn_down`
    (`ATLAS_FFN_DOWN_SPLITK=4`), `ssm_out` (`ATLAS_SSM_OUT_SPLITK=4`), and `WY17`
    lazy GDN state commit (`ATLAS_WY17_LAZY=1`), reaching **`63.9 tok/s`** on the
    qualified `BF16-KV` `gamma=15` profile.

---

## 3. TurboQuant+ KV Cache Compression (`turbo2` / `turbo3` / `turbo4` / `turbo8` & Asymmetric K/V)

Implemented in `kernels/gb10/common/wht_bf16.cu`, `reshape_and_cache_turbo.cu`,
`paged_decode_attn_turbo*.cu`, and `tq_plus_innerq.cu`, Flux Engine supports
symmetric (`turbo2`, `turbo3`, `turbo4`, `turbo8`) and asymmetric
(`fp8k_turbo4v` / `fp8k4v` = 6 bits/elem avg, `bf16k_turbo4v`, `turbo4k_turbo3v`)
online **TurboQuant+** KV cache formats:

1. **Randomized Walsh-Hadamard Transform (`wht_bf16.cu`):** Rotates per-head
   key/value vectors before quantization so channel outliers are spread into a
   near-Gaussian distribution across all head dimensions.
2. **Lloyd-Max Optimal Codebooks & Matched-`L2`-Norm Scaling:** Instead of uniform
   scalar min/max binning, `turbo2` (2-bit), `turbo3` (3-bit), `turbo4` (4-bit),
   `turbo8` (8-bit E4M3 + BF16 group scales), and asymmetric K/V modes
   (`fp8k_turbo4v` / `fp8k4v`, `bf16k_turbo4v`, `turbo4k_turbo3v`) use Gaussian
   Lloyd-Max centroids with matched-`L2`-norm scale correction to eliminate
   centroid-rounding norm shrinkage.
3. **InnerQ Per-Channel Equalization & Sparse-V Dequant (`tq_plus_innerq.cu`):**
   Applies pre-rotation channel equalization and skips V dequantization when
   attention weights fall below `1e-3`, cutting decode KV bandwidth by up to 4×
   with near-BF16 perplexity. See [`docs/turboquant-plus.md`](turboquant-plus.md).

---

## 4. Workspace Layer-by-Layer Reference

### `atlas-core`
Shared configuration and trait definitions:
- `ModelConfig`: normalized representation of HuggingFace `config.json` across
  `Qwen3.8-Flash-Next` (`qwen4_exp_text`), `Qwen3.8-27B`, `DeepSeek-V4.1`,
  `GLM-5.3`, `SuperGemma4`, `Qwen3.5/3.6`, `Nemotron-H`, `Mistral-Small-4`, and
  `MiniMax-M2.7`.
- `KernelTarget` / `TargetPtxSet`: build-time and runtime kernel target descriptors.

### `atlas-kernels`
Compile-time CUDA/PTX builder (`crates/atlas-kernels/build.rs`):
1. Reads `kernels/<hw>/HARDWARE.toml` and `kernels/<hw>/<model>/MODEL.toml`.
2. Compiles every `.cu` under `kernels/<hw>/<model>/<quant>/` and
   `kernels/<hw>/common/` via `nvcc -arch=sm_121a --ptx`.
3. Emits `OUT_DIR/target_ptx.rs` embedding the PTX modules directly into the
   Rust binary.
4. Supports `ATLAS_SKIP_BUILD=1` for GPU-free `cargo check` / `clippy` runs.

### `spark-runtime`
GPU driver abstraction (`cudarc`), paged KV cache manager (`PagedKvCache`),
pinned-host + device buffer arenas (`buffers.rs`), safetensors streamer with OOM
guard (`weights.rs`), and host/device token sampler (`sampler.rs`).

### `spark-model`
Model orchestration (`TransformerModel`), layer implementations
(`qwen3_attention`, `qwen3_ssm`, `qwen4_hyper`, `moe`, `dense_ffn`,
`dflash_head`, `mtp_head`), AOT FlashInfer SM121a GDN/Attention bridges
(`csrc/fi_gdn/`, `csrc/fi_attn/`), and per-family weight loaders (`factory.rs`).

### `spark-server`
Builds the `flux-server`, `flux-cli`, and `spark` binaries:
- Startup `sync_flux_env_aliases()` in `src/main.rs` automatically maps `FLUX_*`
  environment variables to `ATLAS_*` selectors before worker threads spawn.
- Axum HTTP server (`POST /v1/chat/completions`, `/v1/completions`,
  `/v1/responses`, `/v1/messages`), SLAi continuous batching scheduler,
  per-model tool-call parsers, and `xgrammar` structured outputs.

---

## 5. Build Pipeline

```bash
export PATH=/usr/local/cuda/bin:$PATH

# Single-target build for Qwen3.8-Flash-Next NVFP4 on GB10:
FLUX_TARGET_HW=gb10 FLUX_TARGET_MODEL=qwen3.8-flash-next FLUX_TARGET_QUANT=nvfp4 \
  cargo build --release -p spark-server

# Dual-target build for both Qwen3.8-Flash-Next and Qwen3.8-27B:
FLUX_TARGET_HW=gb10 FLUX_TARGET_MODEL='qwen3.8-flash-next,qwen3.8-27b' FLUX_TARGET_QUANT=nvfp4 \
  cargo build --release -p spark-server
```

See also:
- [`docs/FLUX_KERNELS.md`](FLUX_KERNELS.md): `sm_121a` kernel inventory.
- [`docs/QWEN38_FLASH_NEXT_PERFORMANCE_RECIPE.md`](QWEN38_FLASH_NEXT_PERFORMANCE_RECIPE.md): per-stage breakdown and null-lever record for `Qwen3.8-Flash-Next`.
- [`docs/QWEN38_PERFORMANCE_RECIPE.md`](QWEN38_PERFORMANCE_RECIPE.md): measurement record for `Qwen3.8-27B` DFlash.
- [`docs/HARDWARE.md`](HARDWARE.md): adding a new hardware target or model family.
