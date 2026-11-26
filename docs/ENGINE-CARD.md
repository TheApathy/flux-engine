# Engine Card — Flux Engine (`sm_121a` / GB10)

Native Rust + CUDA 13.0 (`sm_121a`) inference runtime and universal local-model container appliance for **NVIDIA DGX Spark (Grace Blackwell GB10)** and discrete Blackwell/Hopper GPUs.

---

## 1. Identity

| Field | Value |
|---|---|
| **Repository** | `https://github.com/TheApathy/flux-engine` (`main`) |
| **Container appliance** | `ghcr.io/theapathy/flux-engine:v1.0.3` (`:gb10`, `:latest`) |
| **Server binaries** | `target/release/flux-server`, `target/release/flux-cli`, `target/release/spark` (`flux` smart entrypoint inside container) |
| **Cargo package** | `spark-server` (`cargo build --release -p spark-server`) |
| **Primary hardware target** | NVIDIA GB10 (`sm_121a`, 48 SMs, 32 MiB L2, 128 GB unified LPDDR5X @ 273–301 GB/s, 10× Cortex-X925 + 10× Cortex-A725) |
| **Additional targets** | 2×/4× DGX Spark (ConnectX-7 200GbE RoCEv2), RTX PRO 6000 Blackwell (`sm_120a`), Apple Metal (experimental) |
| **License** | `AGPL-3.0-only` |

> **Crate vs binary name:** `crates/spark-server/Cargo.toml` defines the `spark-server` binary crate (`flux-server`, `flux-cli`, and `spark` binaries pointing to `src/main.rs`). Always compile with `cargo build --release -p spark-server`.

---

## 2. Universal Local-Model Auto-Detection (`flux`)

The container image (`ghcr.io/theapathy/flux-engine:gb10`) and `flux` CLI inspect any local model directory (`config.json`, `hf_quant_config.json`, `ple/` shards, vision towers, and companion drafter `block_size`) and automatically select the matching `--kernel-target`, operating recipe (`FLUX_MODE`), PLE DMA ring configuration, and speculative decoding width (`gamma`):

```bash
# Serve any local checkpoint mounted under /models
docker run --rm --gpus all --ipc=host -p 8899:8899 \
  -v /path/to/models:/models:ro \
  ghcr.io/theapathy/flux-engine:gb10 \
  serve /models/Qwen3.8-Flash-Next-NVFP4-Offload

# Inspect a single model directory or scan an entire /models root
docker run --rm --gpus all -v /path/to/models:/models:ro \
  ghcr.io/theapathy/flux-engine:gb10 inspect /models

# List all discovered local models and their auto-detected targets
docker run --rm --gpus all -v /path/to/models:/models:ro \
  ghcr.io/theapathy/flux-engine:gb10 list-models /models
```

| Local Model Architecture (`config.json`) | Auto-Detected `--kernel-target` | Speculative Path | Key `sm_121a` Subsystem Kernels |
|---|---|---|---|
| `qwen4_exp_text` (`Qwen3.8-Flash-Next-NVFP4-Offload`) | `qwen3.8-flash-next` | Native 1-layer Inferact MTP (`K=5`, `num_drafts=4`) | `v28q`/`v17q` W4A8 MoE (`512` routed top-10 + `1` shared), 4-stream mHC `t128` roof + `gatepre`, QSA `qkpv_amax_pipe` + `ldmatrix.x4` score, GDN `fi_sm121a` + `conv_v3`, 4-ring NVFP4 PLE DMA |
| `qwen3_5` / `qwen3_5_text` (`Qwen3.8-27B-NVFP4`) | `qwen3.8-27b` | DFlash block-diffusion drafter (`gamma=15`, `K=16` verify) | `w4a16_gemm_t_m32_n64`, `gated_delta_rule_wy17` (`ATLAS_SSM_GDN_LAZY=1`), `paged_decode_attn_kgamma_nvfp4`, fused QKV + split-K drafter |
| `qwen3_5_moe` (`Qwen3.6-35B-A3B` / `Qwen3.5-122B-A10B`) | `qwen3.5-35b-a3b` / `qwen3.5-122b-a10b` | Native MTP / DFlash | Grouped NVFP4 MoE GEMM + WY-chunked GDN + paged GQA |
| `deepseek_v41` (`DeepSeek-V4.1`) | `deepseek-v4.1` | DSpark / Native MTP | MLA (`head_dim=512`, `qk_rope_head_dim=64`, `q_lora_rank=1280`, MQA `kv_heads=1`) + 384-expert CB3 MoE + Engram |
| `glm4_moe` / `glm5` (`GLM-5.3`) | `glm5.3-flash` | Target-only (speculation disabled at load) | EXL3 (`2.05 bpw`) routed MoE + GQA prefill/decode |
| `gemma4` (`Gemma-4-26B-A4B` / `31B-IT`) | `gemma-4-26b-a4b` / `gemma-4-31b` | Plain (target-only) | Interleaved sliding-window + global GQA + NVFP4 FFN |

---

## 3. Headline Measured Performance (`1× NVIDIA GB10`)

### 3.1 `Qwen3.8-Flash-Next-NVFP4-Offload` (`qwen3.8-flash-next`, `C=1`)

| Workload / Context | Mode (`FLUX_MODE`) | Certified Peak | Certified Median (`N=3`) | TTFT (Median / Min) | Correctness / Determinism | Free UMA Floor |
|---|---|:---:|:---:|:---:|---|:---:|
| **8K Prefill (`8,192` tok)** | `max-prefill` (`w16384`) | **`5,830.2 tok/s`** | **`5,819.3 tok/s`** | `1.408 s` (`1.405 s`) | `4/4` bit-identical (`dbba921ce205`) | `7,464 MiB` |
| **16K Prefill (`16,384` tok)** | `max-prefill` (`w16384`) | **`6,082.2 tok/s`** | **`6,076.7 tok/s`** | `2.696 s` (`2.694 s`) | `4/4` bit-identical (`0695b563acde`) | `7,252 MiB` |
| **32K Prefill (`32,768` tok)** | `max-prefill` (`w16384`) | **`5,969.6 tok/s`** | **`5,952.6 tok/s`** | `5.505 s` (`5.489 s`) | `4/4` bit-identical (`e47125968b3b`) | `7,137 MiB` |
| **64K Prefill (`65,536` tok)** | `max-prefill` (`w16384`) | **`5,828.3 tok/s`** | **`5,803.2 tok/s`** | `11.293 s` (`11.244 s`) | `4/4` bit-identical (`0695b563acde`) | `7,194 MiB` |
| **128K Prefill (`131,072` tok)** | `max-prefill` (`w16384`) | **`5,694.4 tok/s`** | **`5,684.0 tok/s`** | `23.060 s` (`23.018 s`) | `4/4` bit-identical (`c51cdffb6e38`) | `7,072 MiB` |
| **8K–64K Interactive (`w4096`)** | `interactive` (`w4096`) | **`5,276.3 tok/s`** | **`5,266.8 – 5,016.1 tok/s`** | `1.555 s – 13.065 s` | `20/20` bit-identical (`8K..128K`) | **`11,907 MiB`** |
| **Lossless `K=5` MTP Decode** | `max-prefill` / `decode` | **`57.4 tok/s`** | **`56.3 tok/s`** (`1.70×` vs `K=1`) | — | **`LOSSLESS 10/10`** (`2,560/2,560` exact) | `7,579 MiB` |
| **Estonia `136,584`-tok 6-Hop** | `long-context` (`136k`) | **`5,428.3 tok/s`** | **`5,400.4 tok/s`** (`48.6` dec) | `25.290 s` (`25.160 s`) | **`5/5 (100% PASS)`** across 2 runs (`10/10`) | `8,431 MiB` |

### 3.2 `Qwen3.8-27B-NVFP4` (`qwen3.8-27b` + DFlash `gamma=15`, `C=1`)

| Configuration | Decode Throughput (`tok/s`) | Emitted / Accept Rate | Output Reference | Launcher |
|---|:---:|:---:|---|---|
| **DFlash (`bf16` KV, `WY17` lazy GDN, drafter split-K, 400-tok probe)** | **`63.91 – 64.05 tok/s`** | `8.33` emitted / `129.8 ms` (`7.33 / 15` drafts; 800-tok probe: `5.81 / 15` @ `51.89 tok/s`) | `f3b4b2cedd78` (`5/5` bit-identical) | [`bench/qwen38-gb10/serve.sh`](../bench/qwen38-gb10/serve.sh) |
| **DFlash2 official (`incoai`, corrected 11-arg ABI)** | **`58.01 tok/s`** | `5.12 / 15` (`p = 0.831`) | `9b86c6d92123` (`5/5` bit-identical) | [`bench/qwen38-gb10/serve-dflash2-official.sh`](../bench/qwen38-gb10/serve-dflash2-official.sh) |
| **No-speculation baseline (`ATLAS_DISABLE_SPECULATION=1`)** | `12.90 – 13.35 tok/s` | — (`1` tok/step) | `f3b4b2cedd78` | `flux` launcher `--no-speculative` (raw server: `ATLAS_DISABLE_SPECULATION=1`) |

---

## 4. Build From Source

```bash
# Build Qwen3.8-Flash-Next + Qwen3.8-27B targets for GB10 (sm_121a)
export PATH="/usr/local/cuda/bin:${PATH}"
FLUX_TARGET_HW=gb10 \
FLUX_TARGET_MODEL="qwen3.8-flash-next,qwen3.8-27b" \
FLUX_TARGET_QUANT=nvfp4 \
  cargo build --release -p spark-server
```

Verify the resulting binary embeds the expected kernel target before serving:

```bash
strings target/release/spark | grep -E '^qwen3\.8-(flash-next|27b)$'
```

---

## 5. Companion Cards & Guides

- [`MODEL-CARD.md`](MODEL-CARD.md). Checkpoint architecture, quantization layout, and SHA-256 manifests.
- [`FLAGS.md`](FLAGS.md). Complete `FLUX_*` / `ATLAS_*` environment variable and CLI flag reference.
- [`docs/OPERATOR-GUIDE.md`](OPERATOR-GUIDE.md). One-command serving, OpenAI API queries, thinking/tool-call controls, and GB10 UMA gotchas.
- [`docs/CONTAINER_GUIDE.md`](CONTAINER_GUIDE.md). Universal local-model container guide (`ghcr.io/theapathy/flux-engine:v1.0.3`).
- [`docs/KERNEL-PROFILE.md`](KERNEL-PROFILE.md). Per-stage `sm_121a` kernel timing breakdown (`1,348.0 ms / 8K` prefill and `K=5` / `K=16` verify).
- [`docs/verification/BENCHMARK_VERIFICATION.md`](verification/BENCHMARK_VERIFICATION.md). Roofline physics, prefix-cache controls, and benchmark verification proofs.
