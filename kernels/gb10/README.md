<p align="left"><a href="../../README.md"><b>← Flux Engine</b></a> · <a href="../README.md"><b>Kernels (<code>kernels/</code>)</b></a> · <b>NVIDIA GB10 (<code>sm_121f</code>) Kernel Targets (<code>kernels/gb10/</code>)</b></p>

# NVIDIA GB10 (`sm_121f`) CUDA / PTX Kernel Targets (`kernels/gb10/`)

Hand-tuned CUDA and inline-PTX kernels targeting **NVIDIA Grace Blackwell (`GB10`, `arch = "sm_121f"` in `HARDWARE.toml`)** on DGX Spark. Every model directory contains a `MODEL.toml` manifest (`common/` holds shared primitives with `KERNEL.toml`), compiled via `nvcc --ptx -arch=sm_121f -O3 --use_fast_math -Xptxas -O3` plus per-target `KERNEL.toml` flags (e.g., `--fmad=false`, or `--atlas-ieee-math` to opt out of `--use_fast_math`); hot-path kernels are hand-tuned for zero register spills (`0 bytes spill stores, 0 bytes spill loads`).

---

## Target Directory Index

| Directory | Model Family | Quantization | Key `sm_121f` Kernels & Features |
|---|---|---|---|
| **[`qwen3.8-flash-next/nvfp4/`](qwen3.8-flash-next/nvfp4/)** | `Qwen3.8-Flash-Next` (`125B` + `51B PLE`, `6B` active) | `NVFP4` + `MXFP8` (`W4A8`) | W4A8 / W4A4 MoE (`0` spills), `qwen4_qsa_prefill_select` (`ldmatrix.x4` @ 5 CTAs/SM), `qwen4_gdn_fi_glue.cu` (`v3`/`v4_strip` conv1d+L2 norm), 4-stream `mHC` roof (`t128`), 4-ring NVMe PLE unpack, and lossless `K=5` Inferact MTP verify. |
| **[`qwen3.8-27b/nvfp4/`](qwen3.8-27b/nvfp4/)** | `Qwen3.8-27B` Dense Hybrid | `NVFP4` + `BF16/FP8` KV | 48-layer Gated DeltaNet (`WY` chunked recurrence) + 16-layer GQA, fused `M17` QKV staging, and `gamma=15` (`K=16`) DFlash block-diffusion tree verification (`63.9 tok/s`). |
| **[`common/`](common/)** | Shared `sm_121f` Primitives | `BF16` / `FP8` / `NVFP4` / `TQ+` | FlashAttention-2/4 online softmax, `TurboQuant+` (`wht_bf16.cu`, `reshape_and_cache_turbo.cu`, `tq_plus_innerq.cu`), fused `RMSNorm + MXFP8`, and `MRoPE`. |
| **[`deepseek-v4.1/`](deepseek-v4.1/)** | `DeepSeek-V4.1` (`dsv41`) | `CB3` (+ `FP8` GEMM / sparse attention tooling) | Cross-layer sparse Multi-Head Latent Attention (`head_dim=512`, `64` Q heads, `1` KV head), Engram memory lookup, and `MROW=2` deduplicated CB3 MoE verify. |
| **[`glm5.3-flash/`](glm5.3-flash/)** | `GLM-5.3-Flash` (`glm53`) | `EXL3` / `IQ3` | KDA recurrent state + Tensor-Core Dynamic Sparse Attention (`DSA`) + row-batched DFlash2 verify. |
| **[`gemma-4-26b-a4b/nvfp4/`](gemma-4-26b-a4b/nvfp4/)** & **[`gemma-4-31b/nvfp4/`](gemma-4-31b/nvfp4/)** | `Gemma 4` (`26B-A4B` & `31B`) | `NVFP4` | Alternating sliding-window + global GQA attention, GeGLU MoE / dense FFN, and PLE sidecar support. |
| **[`qwen3.6-35b-a3b/nvfp4/`](qwen3.6-35b-a3b/nvfp4/)** & **[`qwen3.6-27b/nvfp4/`](qwen3.6-27b/nvfp4/)** | `Qwen3.6` (`35B-A3B` & `27B`) | `NVFP4` / `FP8` | Hybrid GDN + GQA + 256-expert routed MoE + `K=2` MTP / `K=17` DFlash. |
| **[`qwen3.5-35b-a3b/nvfp4/`](qwen3.5-35b-a3b/nvfp4/)**, **[`qwen3.5-122b-a10b/nvfp4/`](qwen3.5-122b-a10b/nvfp4/)**, **[`qwen3.5-27b/nvfp4/`](qwen3.5-27b/nvfp4/)** | `Qwen3.5` (`27B`, `35B-A3B`, `122B-A10B`) | `NVFP4` | Single-Spark and multi-Spark (`EP=2`) hybrid GDN + MoE kernels. |
| **[`qwen3-next-80b-a3b/nvfp4/`](qwen3-next-80b-a3b/nvfp4/)** & **[`qwen3-vl-30b-a3b/nvfp4/`](qwen3-vl-30b-a3b/nvfp4/)** | `Qwen3-Next-80B` & `Qwen3-VL-30B` | `NVFP4` | 512-expert MoE and multimodal Vision ViT + MoE kernels. |
| **[`minimax-m2-229b/nvfp4/`](minimax-m2-229b/nvfp4/)** | `MiniMax-M2.7` (`229B-A10B`) | `NVFP4` + `Turbo8` KV | 62-layer sigmoid-routed 256-expert MoE + MTP. |
| **[`mistral-small-4/nvfp4/`](mistral-small-4/nvfp4/)** | `Mistral-Small-4-119B` | `NVFP4` | 119B MoE (`6.5B` active) single-Spark kernels. |
| **[`nemotron-3-nano-30b-a3b/nvfp4/`](nemotron-3-nano-30b-a3b/nvfp4/)** & **[`nemotron-super-120b-a12b/nvfp4/`](nemotron-super-120b-a12b/nvfp4/)** | `Nemotron-3` (`Nano-30B` & `Super-120B`) | `NVFP4` | Hybrid Mamba-2 SSM + GQA attention + routed MoE. |
