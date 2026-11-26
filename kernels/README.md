<p align="left"><a href="../README.md"><b>← Flux Engine</b></a> · <b>Hardware &amp; Model Specific CUDA / Metal Kernels (<code>kernels/</code>)</b></p>

# Hardware & Model Specific Kernels (`kernels/`)

Every model kernel directory follows the **`kernels/<hardware>/<model>/<quantization>/`** hierarchy and is auto-discovered at build time by [`crates/atlas-kernels/build.rs`](../crates/atlas-kernels/build.rs) via `MODEL.toml` descriptors (shared primitives live in `<hardware>/common/` with `KERNEL.toml`). GB10 kernels compile with `nvcc --ptx -arch=sm_121f -O3 --use_fast_math -Xptxas -O3` plus per-target `KERNEL.toml` flags (e.g., `--fmad=false`, or `--atlas-ieee-math` to opt out of `--use_fast_math`); hot-path kernels are hand-tuned for **`0` register spill bytes**.

---

## Hardware Backends

| Directory | Compute Target | Architecture / ISA | Primary Model Targets |
|---|---|---|---|
| **[`gb10/`](gb10/README.md)** | NVIDIA Grace Blackwell (`DGX Spark` / `GB10`) | `sm_121f` (48 SMs, 120 GB unified LPDDR5x) | `qwen3.8-flash-next`, `qwen3.8-27b`, `deepseek-v4.1`, `glm5.3-flash`, `gemma-4-26b-a4b`, `gemma-4-31b`, `qwen3.6-35b-a3b`, `qwen3.5-35b/122b`, `minimax-m2-229b`, `nemotron-3` |
| **[`gb300/`](gb300/)** | NVIDIA Blackwell Ultra (`GB300`) | `sm_103a` (288 GB HBM3E, extends `gb10`) | Inherits `gb10` kernel targets compiled for `sm_103a` |
| **[`rtx6000/`](rtx6000/)** | NVIDIA Discrete Blackwell (`RTX PRO 6000` / `RTX 5090`) | `sm_120a` (GDDR7 discrete VRAM) | Shared Blackwell Tensor Core MMA (`f8f6f4.block_scale`) and discrete PCIe staging profiles |
| **[`metal/`](metal/)** | Apple Silicon Unified Memory (`M2/M3/M4 Max & Ultra`) | Metal Shading Language (`.metal`) | `common/` + `qwen3-5-4b-vlm-mlx-int8/` |

## Build-Time Target Selection

```bash
export PATH=/usr/local/cuda/bin:$PATH
FLUX_TARGET_MODEL='qwen3.8-flash-next,qwen3.8-27b' FLUX_TARGET_QUANT=nvfp4 \
  cargo build --release -p spark-server
```

See [`../ARCHITECTURE.md`](../docs/ARCHITECTURE.md) and [`../docs/FLUX_KERNELS.md`](../docs/FLUX_KERNELS.md) for the complete kernel architecture guide.
