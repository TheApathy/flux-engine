# Flux Engine on DGX Spark (GB10): Single-Node Prefill, Speculative Decode, and Engineering Architecture

This document traces the engineering progression behind `TheApathy/flux-engine`
on a single **NVIDIA DGX Spark (`GB10`, `sm_121a`, 128 GB unified LPDDR5x @ 273 GB/s)**,
building on the pure-Rust foundation created by `MonumentalSystems/atlas` and
`Avarok-Cybersecurity/atlas` via `TheApathy/apathy-atlas`.

There is no PyTorch, no Python, and no vLLM in the inference hot path. Every
operation (tokenization, radix prefix caching, 4-stream Multi-Stream
Hyper-Connections `mHC`, 36-layer Gated DeltaNet `GDN` linear attention,
12-layer Query-Sparse Attention `QSA`, 512-expert block-scaled `W4A8` MoE
grouped GEMMs, 4-ring sector-aligned NVFP4 Per-Layer Embedding `PLE` DMA, and
device-chained `K=5` Multi-Token Prediction `MTP` / `DFlash` speculative
verification) runs in pure Rust over hand-tuned `sm_121a` CUDA/PTX kernels.

---

## 1. The Hardware Reality of GB10 (`sm_121a`)

The NVIDIA Grace Blackwell GB10 pairs a 20-core ARMv9.2 Grace CPU (`10x Cortex-X925`
+ `10x Cortex-A725`) with a 48-SM Blackwell GPU (`sm_121a`) over a single
128 GB (`119.7 GiB` OS-visible) LPDDR5x unified memory pool:

- **Memory Bandwidth Ceiling (`273 GB/s` sustained):** Single-stream decode is
  strictly memory-bandwidth-bound. Every byte read from LPDDR5x costs wall time.
  Compressing weights to NVFP4 (`E2M1` + `UE4M3` block-16 scales) and verifying
  `K=5` (`Qwen3.8-Flash-Next` MTP) or `gamma=15` (`Qwen3.8-27B` DFlash) draft
  tokens in a single batched pass turns a memory-bound GEMV into a compute-dense
  Tensor Core tile.
- **32 MiB L2 Cache & 99 KB/SM Shared Memory:** During large-chunk prefill
  (`w16384` and `w4096`), keeping intermediate residual streams (`s.up` in 4-stream
  `mHC`) inside the 32 MiB L2 cache and keeping register counts under 128 (`0`
  spills) so `down_proj` (`v17q`) holds 2 CTAs/SM and QSA block scoring holds
  5 CTAs/SM is the difference between 3,400 tok/s and **6,082.2 tok/s**.
- **6,144 MiB Hardware Watchdog:** Because CPU and GPU share one unified memory
  pool without PCIe isolation, dropping `MemAvailable` below `6,144 MiB` triggers
  an immediate OS/hardware watchdog kill. Every scratch buffer in Flux Engine is
  pre-sized, recycled across idle arenas (`SHARED_W4A8_PRETRANSCODE` inside
  `ssm_deinterleaved`), and guarded by dynamic reserve floors (`_RESERVE_GB=7`
  at `w16384`, `_RESERVE_GB=10` at `136.5K`).

---

## 2. Milestone Progression across Model Families

| Model Target | Architecture | Headline Prefill (`1x GB10`) | Headline Decode (`1x GB10`) | Key Breakthrough |
|---|---|---:|---:|---|
| **`Qwen3.8-Flash-Next-NVFP4`** | 48L hybrid (36 GDN + 12 QSA, 512+1 MoE top-10, 4-stream mHC, 51B PLE) | **6,082.2 tok/s** @ 16K<br/>**6,021.8 tok/s** short<br/>**5,969.4 tok/s** @ 8K<br/>**5,400.4 tok/s** @ 136.5K | **56.5 tok/s** med (`57.4` peak, `LOSSLESS 10/10`)<br/>**48.2 tok/s** @ 136.5K (`5/5 PASS`) | `cutlass_5222` `v28q`/`v17q` W4A8 MoE, `gdn_umma_prefill`, `qwen4_qsa_prefill_select`, `mHC` `t128` roof, 4-ring NVFP4 PLE, lossless `K=5` MTP. |
| **`Qwen3.8-27B-NVFP4`** | 64L dense hybrid (48 GDN + 16 GQA) + `5L` DFlash2 / `8L` DFlash v1 drafter | **1,050 – 1,411 tok/s** | **63.9 tok/s** (`BF16-KV`, `gamma=15`)<br/>**58.012 tok/s** (`v3` control) | DFlash `gamma=15` speculative verify, split-K `ffn_down`/`ssm_out`/`qkvz`, `WY17` lazy state commit. |
| **`Qwen3.5-35B-A3B` / `Qwen3.6-35B-A3B`** | 40L hybrid (GDN + GQA + 256 MoE + MTP) | **3,800+ tok/s** | **130 – 131 tok/s** (`MTP K=2`) | Fused GDN recurrence + coalesced W4A16/FP8 grouped MoE + BF16 MTP head. |
| **`Qwen3-Next-80B-A3B-NVFP4`** | 48L hybrid (36 GDN + 12 GQA + 512 MoE) | **2,900+ tok/s** | **99.1 tok/s** peak (`74–97` sustained) | Pure-Rust FP8 KV + native MTP speculative decode on a single GB10. |
| **`DeepSeek-V4.1`** | MLA (`head_dim=512`, `qk_rope_head_dim=64`, `q_lora_rank=1280`) + 384-expert CB3 MoE + DSpark | **1,980 – 2,010 tok/s** (`8K–16K`) | **14.8 tok/s** plain / **19.0 – 30.8 tok/s** (`DSpark`) | K124/K154 CB3 routed-expert pack + persistent Engram lookup + cross-layer sparse MLA. |
| **`GLM-5.3-Flash` (EXL3 2.05 bpw)** | KDA linear attention + NoPE DSA + 4-stream mHC + MoE | **694 tok/s** @ 8K / **603 tok/s** @ 32K | **14.6 tok/s** (`8.6` @ 32K, target-only) | EXL3 trellis quantization on single GB10 (target-only, no speculation). |
| **`SuperGemma4` (`26B-A4B` / `31B`)** | Sliding + full attention alternation, GeGLU MoE/Dense | **3,100+ tok/s** | **67 tok/s** (`26B-A4B`) | Fused GeGLU + sliding-window causal mask hoist + TurboQuant+ KV cache. |

For the full release history and kernel walkthrough, see
[`CHANGELOG.md`](../CHANGELOG.md), [`ARCHITECTURE.md`](ARCHITECTURE.md),
and [`docs/QWEN38_FLASH_NEXT_PERFORMANCE_RECIPE.md`](QWEN38_FLASH_NEXT_PERFORMANCE_RECIPE.md).
