# Citations and Prior Art

This document records the academic papers, upstream projects, and open-source
implementations referenced or built upon by Flux Engine. For formal copyright
notices and license texts, see [`CREDITS.md`](../CREDITS.md),
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md), and
[`SOURCE_AND_LICENSES.md`](../SOURCE_AND_LICENSES.md).

## 1. Upstream Engine Foundation

- **Atlas Inference Engine** (`MonumentalSystems/atlas`, `Avarok-Cybersecurity/atlas`):
  Pure-Rust LLM inference runtime and CUDA/Metal kernel architecture from which
  Flux Engine originated. Flux Engine builds on Atlas's workspace structure,
  OpenAI/Anthropic HTTP serving stack, paged KV cache, and initial GB10 kernels
  under the AGPL-3.0-only license.

## 2. KV-Cache Quantization — TurboQuant & TurboQuant+

### TurboQuant (Canonical Paper)
- **TurboQuant: Online Vector Quantization with Near-Optimal Distortion Rate.**
  Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni.
  [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) (April 2025).
- Establishes that random rotation of input vectors concentrates coordinate
  distributions toward a Beta / near-Gaussian marginal (`N(0, 1/d)`), enabling
  a shared scalar Lloyd-Max quantizer per coordinate, paired with optional
  1-bit Quantized Johnson-Lindenstrauss (QJL) residual correction for unbiased
  inner-product estimation.
- Upstream Atlas's `wht_bf16.cu` implemented an unmasked Walsh-Hadamard
  transform; Flux Engine extends this to the Randomized Hadamard Transform
  (with per-channel random sign masks) described in Zandieh et al.

### `TheTom/turboquant_plus`
- Primary umbrella research repository for TurboQuant+ experiments, design
  notes, and cross-engine reference benchmarks spanning llama.cpp, vLLM, and
  Atlas-derived engines:
  [https://github.com/TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus).
- Relevant upstream technical notes in `TheTom/turboquant_plus` (see Flux Engine's port summary in [`docs/turboquant-plus.md`](turboquant-plus.md)):
  - `asymmetric-kv-compression.md` — higher precision on K than V (`Turbo4K` + `Turbo3V`).
  - `layer-aware-v-compression.md` (LA-V7) — preserving boundary layers in BF16.
  - `sparse-v-dequant.md` — skipping V dequantization when softmax weight is below threshold.
  - `weight-compression-tq4.md` — 4-bit weight quantization with WHT rotation.
  - `triattention-v3.md` — long-context attention eviction policies.
  - `moe-v-compression-frontier.md` — value compression trade-offs for MoE models.

### `TheTom/llama-cpp-turboquant`
- Reference llama.cpp fork implementing TurboQuant KV-cache quantization
  (`--kv-cache-dtype turbo3/turbo4/turbo8`):
  [https://github.com/TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant).
- Reference kernels: `ggml/src/ggml-cuda/turbo-wht.cu`, `turbo-quant.cuh`,
  `turbo-innerq.cu`.
- The Rademacher sign arrays in `kernels/gb10/common/tq_plus_signs.cuh` match
  `TURBO_WHT_SIGNS1` / `TURBO_WHT_SIGNS2` from `turbo-quant.cuh` (seed 42
  Rademacher draws), and the sparse-V gating design mirrors `de44bfe60` in
  `TheTom/llama-cpp-turboquant`.

### Summary of TurboQuant+ Kernel Extensions in Flux Engine
- **Two-sided Rademacher signs in WHT (`TQ_PLUS_SIGNS`)**:
  `kernels/gb10/common/tq_plus_signs.cuh` and `kernels/gb10/common/wht_bf16.cu`
  apply `S1` before and `S2` after the butterfly transform for `head_dim = 128`,
  with `wht_bf16_inplace_inv` performing the exact transpose inverse
  `(S2 · H · S1) · (S1 · H · S2)^T = I`.
- **`N(0, 1/d)` Lloyd-Max codebooks**:
  `kernels/gb10/common/reshape_and_cache_turbo.cu`,
  `kernels/gb10/common/paged_decode_attn_turbo{3,4,8}*.cu`, and
  `crates/spark-runtime/src/kv_dequant.rs` use `N(0, 1/128)` codebooks for
  post-WHT coordinates without per-group `amax` loops on the write path.
- **Attention-gated sparse V dequantization**:
  All `paged_decode_attn_turbo{2,3,4,8}*.cu` decode kernels gate per-position V
  loads and dequantization on `exp_weight > TQ_PLUS_SPARSE_V_THRESHOLD`
  (default `1e-3`) across both remainder and batched (`BC=4`) paths.
- **Boundary layer protection (LA-V7)**:
  `--kv-high-precision-layers` (default `"auto"`, protecting the first 2 and
  last 2 layers in BF16) preserves sensitive boundary attention layers.
- **Scaffolding for future extensions**:
  `kernels/gb10/common/tq_plus_innerq.{cuh,cu}` (per-channel Q/K equalization
  calibration helpers),
  `crates/spark-model/src/weight_loader/qwen35/load_layers/tq_plus_weight_rotation.rs`
  (load-time Q/K/V/O rotation helper behind `TQ_PLUS_WEIGHT_ROTATION=1`), and
  asymmetric KV enum variants (`Turbo4KTurbo3V`, `Turbo4KTurbo8V`,
  `Turbo3KTurbo8V` in `crates/spark-runtime/src/kv_cache.rs`, currently routed
  to symmetric K-side kernels pending dedicated mixed-bit kernels).

## 3. Speculative Decoding, Constrained Decoding & Attention Kernels

- **DFlash** (`z-lab/dflash`): Block-diffusion speculative decoding drafter
  architecture and draft checkpoints used for `Qwen3.8-27B` (5-layer DFlash2
  and 8-layer DFlash v1) and `Qwen3.8-Flash-Next` (5-layer DFlash2).
- **XGrammar** (`mlc-ai/xgrammar`): Grammar-guided structured output and JSON
  schema constrained decoding algorithms; re-implemented in safe pure Rust in
  `crates/xgrammar` (see `crates/xgrammar/DESIGN.md` and `crates/xgrammar/NOTICE`).
- **FlashAttention & FlashInfer** (`Dao-AILab/flash-attention`,
  `flashinfer-ai/flashinfer`): IO-aware tiled attention algorithms, online
  softmax rescaling, and Blackwell SM120/SM121 tensor-core GEMM/attention
  kernel techniques.
- **NVIDIA CUTLASS** (`NVIDIA/cutlass`): Blackwell `mma.sync` / `tcgen05`
  block-scaled NVFP4 (`E2M1` + `UE4M3`) and FP8 (`E4M3`) matrix-multiply
  programming model and PTX layouts.
- **Gated DeltaNet / Linear Attention** (Yang et al., *Gated Delta Networks*,
  ICLR 2025): Chunkwise WY-representation parallel prefill and recurrent state
  update used in Qwen3.5 / Qwen3.8 hybrid linear-attention layers.
- **ExLlamaV3 & ggml / llama.cpp**: EXL3 quantization (`turboderp-org/exllamav3`)
  and Q4_K quantization (`ggml-org/llama.cpp`) reference implementations
  attributed in `THIRD_PARTY_NOTICES.md`.

## 4. Licensing

Flux Engine is licensed under **AGPL-3.0-only** with no Contributor License
Agreement (CLA). All third-party code, vendored headers, and model templates
retain their original upstream copyright and license notices as cataloged in
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) and
[`SOURCE_AND_LICENSES.md`](../SOURCE_AND_LICENSES.md).
