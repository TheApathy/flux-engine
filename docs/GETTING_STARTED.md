# Apathy Flux Engine — Getting Started, Benchmark Reproduction & Diagnostics Guide

**Container Image:** `ghcr.io/theapathy/flux-engine:v1.0.3` *(also tagged `:v1.0`, `:gb10`, `:latest`; prior `v0.9.x` tags remain available)*  
**Target Hardware:** NVIDIA DGX Spark (`GB10` / `sm_121a`, 128 GB Unified LPDDR5x), Multi-Spark (`2x` / `4x`), or discrete Blackwell (`sm_120a`) GPUs  
**Privacy Policy:** 100% Air-Gapped Local Execution by Default (`FLUX_OPT_OUT=all` — zero external telemetry, host/prompt logs redacted)

---

## Quick Jump

- [0. System Prerequisites Checklist](#0-system-prerequisites-checklist-before-you-start)
- [1. Downloading & Preparing Models (JMNI-Labs QAD5500-Hybrid, Stock NVIDIA NVFP4, Cyber-Frost V2 & 27B)](#1-downloading--preparing-models-critical-ple-offload-step)
- [2. Pull the Container & Run Pre-Flight Diagnostics](#2-pull-the-container--run-pre-flight-diagnostics)
- [3. Serve Any Model & Operating Mode](#3-serve-any-model--operating-mode)
- [3.5. Web Recipe Studio (`GET /`), Interactive Ratatui TUI & CLI Mode Wizard](#35-web-recipe-studio-get--interactive-ratatui-tui--cli-mode-wizard)
- [4. Reproduce the Benchmarks (All Contexts, Decode, Concurrency & Estonia 136.6K)](#4-reproduce-the-benchmarks-all-contexts-decode-concurrency--estonia-1366k)
- [5. Customizing Your Own Recipes & Kernel Knobs](#5-customizing-your-own-recipes--kernel-knobs)
- [6. Built-In Operating Modes Summary](#6-built-in-operating-modes-summary)
- [7. Live Diagnostics, Telemetry & Kernel Receipt Verification](#7-live-diagnostics-telemetry--kernel-receipt-verification)
- [8. Top 11 Hardware, Container & Model Gotchas ("Things That Will Cost You a Day")](#8-top-11-hardware-container--model-gotchas-things-that-will-cost-you-a-day)
- [9. Technical FAQ (Roofline Math, Prefix Cache, Lossless Proofs)](#9-technical-faq-roofline-math-prefix-cache-lossless-proofs)

---

## 0. System Prerequisites Checklist (Before You Start)

To reproduce the certified `6,291.0 tok/s` median (`6,299.3 tok/s` peak) `w16384` 16K prefill (`5,447.7 tok/s` @ 8K and `5,063.0 tok/s` @ 128K in `w4096`), `91.26 tok/s` (`tf_streams_code_256`), `87.72 tok/s` (`sparkdash_code_400`), `86.86 tok/s` (`rep0`), `66.20 tok/s` (`sparkdash_prose_400`), `63.98 tok/s` (`chat0`), `62.59 tok/s` (`code0`), `50.58 tok/s` (`prose0`), `55.27 tok/s` @ 8K, `53.72 tok/s` @ 32K, `53.03 tok/s` @ 64K, `46.1 tok/s` (`43.79 tok/s` `mix128k`) @ 128K, and `41.8 tok/s` @ 255K/262K single-stream `K=5` decode (`173.55 / 258.22 / 340.45 / 420.28 tok/s` at `C=2..5`, `580.66 tok/s` at `C=8`, and `85.5 tok/s` at `C=8` `K=1`), native `262,144`-token needle retrieval (`5/5 PASS`), and `97.27%` GSM8K accuracy on a single 128 GB DGX Spark (`GB10`), verify these four items before launching:

1. **Docker + NVIDIA Container Toolkit (`R580` / `R590` Driver):**
   ```bash
   docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi
   ```
2. **Local NVMe Storage (`O_DIRECT` Capable) for `/models`:**
   `Qwen3.8-Flash-Next` and `CYBER-FROST-3.8` stream their 51B-parameter Per-Layer Embedding (`PLE`) n-gram table directly from NVMe via 8-KiB page-aligned `O_DIRECT` DMA rings (`ple-offload/`). Store `/models` on a local `ext4` or `xfs` NVMe drive (not NFS, CIFS, or VirtioFS).
3. **Clean UMA Memory Headroom (`MemAvailable >= 105 GiB` Before Starting):**
   On unified-memory DGX Spark systems, the Linux filesystem page cache shares the 128 GB LPDDR5x pool with CUDA allocations. Before launching a 128K/136.6K benchmark, stop any other GPU processes and drop clean OS page cache:
   ```bash
   sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
   awk '/MemAvailable/ {printf "Free UMA Available: %.2f GiB\n", $2/1048576}' /proc/meminfo
   ```
4. **Recommended Docker Flags (`--gpus all --ipc=host --shm-size 16g --ulimit memlock=-1 --ulimit stack=67108864`):**
   Pass `--gpus all --ipc=host --shm-size 16g --ulimit memlock=-1 --ulimit stack=67108864` when running the serving container (`--ulimit memlock=-1` enables pinned host memory and `O_DIRECT` NVMe buffers). Under Docker's default seccomp profile, `ple_offload.rs` automatically falls back to `pread64`; `--security-opt seccomp=unconfined` (or a custom seccomp profile permitting `io_uring_setup`/`io_uring_enter`) is **optional** and only needed if you explicitly want async `io_uring` NVMe PLE offload on `Qwen3.8-Flash-Next`.

---

## 1. Downloading & Preparing Models (Critical PLE Offload Step)

First pull the container image and set `MODELS_DIR` to a directory on your local NVMe drive:
```bash
docker pull ghcr.io/theapathy/flux-engine:gb10

export MODELS_DIR="${MODELS_DIR:-$HOME/models}"
mkdir -p "$MODELS_DIR"
```

> [!IMPORTANT]
> **Built-In `K=5` MTP Speculative Decoding & One-Time PLE Sidecar Preparation:**  
> 1. **Built-In `mtp.*` Speculative Drafter (No External MTP Download Needed):**  
>    Flux Engine uses the checkpoint's **built-in `mtp.*` head directly** for `Qwen3.8-Flash-Next` (`--speculative --num-drafts 4 --mtp-vocab 0 --mtp-quantization bf16`), supporting:
>    - **[`JMNI-Labs/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid`](https://huggingface.co/JMNI-Labs/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid)**: `MIXED_PRECISION` target (`MXFP8` attention/GDN/shared-expert + `NVFP4` routed experts) with built-in weight-only `NVFP4` MTP experts and native `NVFP4` PLE shards.
>    - **[`Blackfrost-AI/CYBER-FROST-3.8-NVFP4-V2`](https://huggingface.co/Blackfrost-AI/CYBER-FROST-3.8-NVFP4-V2)**: `NVFP4` target with built-in `BF16` MTP experts (`mtp.*` in shard 131), auto-quantized to `NVFP4` and cached on first load.
>    - **[`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)**: `NVFP4` target with built-in packed `BF16` MTP experts (`mtp.*`), auto-quantized to `NVFP4` and cached on first load.
>    - *(Optional `--mtp-from-path <DIR>` remains supported if you want to override the built-in MTP expert bank with an external sidecar directory; with a checkpoint that already ships MTP experts this also needs `-e FLUX_MTP_SOURCE=sidecar`, otherwise the built-in bank is kept and the sidecar is skipped with a `QWEN4_MTP_SOURCE source=checkpoint` warning.)*
> 2. **One-Time 8-KiB Page-Aligned `ple-offload/` Preparation (`0 GB` GPU RAM for PLE):**  
>    Every `Qwen3.8-Flash-Next` checkpoint contains a **51B-parameter Per-Layer Embedding (`PLE`) n-gram table** (`128` shards of `2,500,012` rows x `160` dim). Running `repack-ple` once materializes `28.8 GB` of 8-KiB page-aligned (`91` records/page) NVMe sidecars (`ple-offload/ple-ngram-nvfp4-000.bin .. 127.bin` + `ple-offload/manifest.json`). Whenever `ple-offload/manifest.json` is present, `flux-server` automatically skips uploading `*.layers.1.ple.ple_embedding.ngram_embedding.*` tensors from the `.safetensors` shards into GPU memory — keeping the resident GPU footprint at **`~71 GiB`**.

### Option A — `JMNI-Labs/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid` (Native NVFP4 PLE Passthrough + Built-In NVFP4 MTP)

- **Checkpoint:** [`JMNI-Labs/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid`](https://huggingface.co/JMNI-Labs/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid)
- **MTP Drafter:** Built-in weight-only NVFP4 `mtp.*` head (`--speculative --num-drafts 4 --mtp-vocab 0 --mtp-quantization bf16`)
- **PLE Mode:** **Mode A — Native NVFP4 Passthrough** (`U8 [2,500,012, 80]` + `F8_E4M3 [2,500,012, 10]` + `F32 [1]` copied byte-for-byte into `ple-offload/` in-place with `--keep-source-shards` by default; no `.safetensors` files or `model.safetensors.index.json` are modified)

#### 1. Download `JMNI-Labs/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid`
```bash
export HF_HUB_ENABLE_HF_TRANSFER=1

hf download JMNI-Labs/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid \
  --local-dir "$MODELS_DIR/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid"
```

#### 2. Materialize `ple-offload/` In-Place (Byte-for-Byte Native NVFP4 Passthrough)
Because `JMNI-Labs/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid` already stores its 128 PLE shards in native NVFP4 (`U8 [rows, 80]` + `F8_E4M3 [rows, 10]`), `repack-ple` automatically selects **Mode A (Native NVFP4 Passthrough)** and writes `ple-offload/` directly inside the checkpoint directory without modifying or duplicating any `.safetensors` files:

```bash
# Option 2a — Run directly on host:
docker run --rm \
  ghcr.io/theapathy/flux-engine:gb10 export-repacker > "$MODELS_DIR/repack_ple.py"
python3 "$MODELS_DIR/repack_ple.py" --self-test
python3 "$MODELS_DIR/repack_ple.py" \
  --model "$MODELS_DIR/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid"

# Option 2b — Or run inside the container:
docker run --rm \
  -v "$MODELS_DIR":/models \
  ghcr.io/theapathy/flux-engine:gb10 repack-ple \
  --model /models/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid
```

---

### Option B — Stock `nvidia/Qwen3.8-Flash-Next-NVFP4` (Built-In MTP + PLE Offload Repack)

- **Checkpoint:** [`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) *(or [`Hotshot2000/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/Hotshot2000/Qwen3.8-Flash-Next-NVFP4))*
- **MTP Drafter:** Built-in packed BF16 `mtp.*` head (auto-quantized to NVFP4 and cached on first load; `--speculative --num-drafts 4 --mtp-vocab 0 --mtp-quantization bf16`)

#### 1. Download the Stock NVFP4 Checkpoint
```bash
export HF_HUB_ENABLE_HF_TRANSFER=1

hf download nvidia/Qwen3.8-Flash-Next-NVFP4 \
  --local-dir "$MODELS_DIR/Qwen3.8-Flash-Next-NVFP4-Raw"
```
*(Note: If your downloaded [`Hotshot2000/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/Hotshot2000/Qwen3.8-Flash-Next-NVFP4) checkpoint already contains `ple-offload/manifest.json` or `ple/ple-ngram-nvfp4-000.bin`, save it directly as `$MODELS_DIR/Qwen3.8-Flash-Next-NVFP4-Offload` and skip Step 2 below.)*

#### 2. Repack the PLE Table into `ple-offload/` (`~71 GiB` Resident GPU Footprint)
```bash
# Option 2a — Run the PLE repacker directly on your host (fastest if host has Python + torch + safetensors):
docker run --rm \
  ghcr.io/theapathy/flux-engine:gb10 export-repacker > "$MODELS_DIR/repack_ple.py"
python3 "$MODELS_DIR/repack_ple.py" --self-test
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python3 "$MODELS_DIR/repack_ple.py" \
  "$MODELS_DIR/Qwen3.8-Flash-Next-NVFP4-Raw" \
  "$MODELS_DIR/Qwen3.8-Flash-Next-NVFP4-Offload" \
  --mode offload

# Option 2b — Or run the PLE repacker directly inside the container (mount $MODELS_DIR read-write):
docker run --rm \
  -v "$MODELS_DIR":/models \
  ghcr.io/theapathy/flux-engine:gb10 repack-ple \
  /models/Qwen3.8-Flash-Next-NVFP4-Raw \
  /models/Qwen3.8-Flash-Next-NVFP4-Offload \
  --mode offload
```

Verify the repacked checkpoint before deleting `-Raw`:
```bash
ls -lh "$MODELS_DIR/Qwen3.8-Flash-Next-NVFP4-Offload/ple-offload/manifest.json" \
       "$MODELS_DIR/Qwen3.8-Flash-Next-NVFP4-Offload/ple-offload/ple-ngram-nvfp4-000.bin" \
       "$MODELS_DIR/Qwen3.8-Flash-Next-NVFP4-Offload/model.safetensors.index.json"
```
Each `ple-ngram-nvfp4-XXX.bin` file is `225,058,816` bytes (`27,473` pages x `8,192` bytes for `O_DIRECT`). Once verified, you can `rm -rf "$MODELS_DIR/Qwen3.8-Flash-Next-NVFP4-Raw"` to reclaim disk space (or pass `--keep-source-shards` if you prefer in-place sidecar generation without rewriting `.safetensors` files).

---

### Option C — `Blackfrost-AI/CYBER-FROST-3.8-NVFP4-V2` (180B Security Research Release)

- **Checkpoint:** [`Blackfrost-AI/CYBER-FROST-3.8-NVFP4-V2`](https://huggingface.co/Blackfrost-AI/CYBER-FROST-3.8-NVFP4-V2)
- **MTP Drafter:** Built-in BF16 `mtp.*` weights inside shard 131 (auto-quantized to NVFP4 and cached on first load; `--speculative --num-drafts 4 --mtp-vocab 0 --mtp-quantization bf16`)

[`Blackfrost-AI/CYBER-FROST-3.8-NVFP4-V2`](https://huggingface.co/Blackfrost-AI/CYBER-FROST-3.8-NVFP4-V2) uses the exact same `Qwen4ExpForConditionalGeneration` (`qwen3.8-flash-next`) kernel stack, includes V2 shared gate/up global scales (`weight_scale_2`) across all 24,576 expert pairs, and includes an in-checkpoint BF16 MTP layer.

#### 1. Download `Blackfrost-AI/CYBER-FROST-3.8-NVFP4-V2` (`173.56 GiB`, 131 Shards)
```bash
export HF_HUB_ENABLE_HF_TRANSFER=1

hf download Blackfrost-AI/CYBER-FROST-3.8-NVFP4-V2 \
  --local-dir "$MODELS_DIR/CYBER-FROST-3.8-NVFP4-V2" \
  --max-workers 8
```

#### 2. Repack its 102 GB BF16 PLE Table into `CYBER-FROST-3.8-NVFP4-Offload` (`~71 GiB` Resident)
The repacker automatically detects Cyber-Frost's unscaled BF16 PLE shards (`Mode B`, `source_scale=1.0`) and quantizes them into 8-KiB page-aligned NVFP4 `O_DIRECT` sidecars:

```bash
# Option 2a — Run on host:
docker run --rm \
  ghcr.io/theapathy/flux-engine:gb10 export-repacker > "$MODELS_DIR/repack_ple.py"
python3 "$MODELS_DIR/repack_ple.py" --self-test
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python3 "$MODELS_DIR/repack_ple.py" \
  "$MODELS_DIR/CYBER-FROST-3.8-NVFP4-V2" \
  "$MODELS_DIR/CYBER-FROST-3.8-NVFP4-Offload" \
  --mode offload

# Option 2b — Or run inside container:
docker run --rm \
  -v "$MODELS_DIR":/models \
  ghcr.io/theapathy/flux-engine:gb10 repack-ple \
  /models/CYBER-FROST-3.8-NVFP4-V2 \
  /models/CYBER-FROST-3.8-NVFP4-Offload \
  --mode offload
```

Verify that `$MODELS_DIR/CYBER-FROST-3.8-NVFP4-Offload/ple-offload/manifest.json` (`128` shards) and `model.safetensors.index.json` exist.

---

### Option D — `Qwen3.8-27B-NVFP4` + Companion `DFlash` / `DFlash2` Drafters & Other Supported Models (No Repack Needed)

All other supported architectures fit directly in unified memory without a PLE offload repack step:

| Model Family | Base Checkpoint (Hugging Face Link) | Companion Speculative Drafter (Hugging Face Link) | `block_size` / `gamma` | Auto-Detected `--kernel-target` |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen3.8-27B Dense Hybrid** *(63.9+ tok/s)* | [`nvidia/Qwen3.8-27B-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) *(canonical public ModelOpt NVFP4)* | • **Recommended (`63.9 tok/s`):** [`onewhosighs/Apathy-Qwen3.8-27B-DFlash-drafter-v2`](https://huggingface.co/onewhosighs/Apathy-Qwen3.8-27B-DFlash-drafter-v2)<br>• **V3 Control (`58.0–59.5 tok/s`):** [`onewhosighs/Apathy-Qwen3.8-27B-DFlash-drafter-v3`](https://huggingface.co/onewhosighs/Apathy-Qwen3.8-27B-DFlash-drafter-v3)<br>• **Public DFlash2 (`58.0–81.1 tok/s` GB10 / `487.2 tok/s` SM120):** [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) *(or [`incoai/Qwen3.5-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.5-27B-DFlash2))*<br>• **Public 8-tok DFlash2 (`43.9 tok/s`):** [`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) | `block_size=16` -> `gamma=15` *(for `Apathy-v2/v3`)*<br>`block_size=8` -> `gamma=7` *(for `z-lab/Qwen3.8-27B-DFlash2` & `incoai`)* | `qwen3.8-27b` |
| **Qwen3.6-27B / AEON-27B** | [`RedHatAI/Qwen3.6-27B-NVFP4`](https://huggingface.co/RedHatAI/Qwen3.6-27B-NVFP4) | [`z-lab/Qwen3.6-27B-DFlash`](https://huggingface.co/z-lab/Qwen3.6-27B-DFlash) | `block_size=16` -> `gamma=15` | `qwen3.6-27b` |
| **Qwen3.6 / 3.5 35B-A3B MoE** | [`Qwen/Qwen3.5-35B-A3B-FP8`](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8) | Built-in MTP or [`z-lab/Qwen3.5-35B-A3B-DFlash`](https://huggingface.co/z-lab/Qwen3.5-35B-A3B-DFlash) | `gamma=4` (MTP) / `15` (DFlash) | `qwen3.6-35b-a3b` / `qwen3.5-35b-a3b` |
| **DeepSeek-V4.1 Flash MoE** | `DeepSeek-V4.1-Flash-Next-DGX-Spark-512K` | In-checkpoint 3-layer MTP | `gamma=3` | `deepseek-v4-flash` |
| **GLM-5.3 Flash MoE** | `GLM-5.3-Flash-exl3-2.05bpw` | None: target-only (the engine refuses DFlash2 for GLM; the launcher never attaches one) | — | `glm5.3-flash` |
| **SuperGemma4 26B-A4B / 31B** | `SuperGemma4-26B-abl-NVFP4` | Native | — | `gemma-4-26b-a4b` |

#### Download Example for `Qwen3.8-27B-NVFP4` + Companion `DFlash` / `DFlash2` Drafter
```bash
export HF_HUB_ENABLE_HF_TRANSFER=1

# 1. Download the canonical ModelOpt Qwen3.8-27B-NVFP4 target checkpoint:
hf download nvidia/Qwen3.8-27B-NVFP4 \
  --local-dir "$MODELS_DIR/Qwen3.8-27B-NVFP4"

# 2a. Download our gamma=15 DFlash v2 drafter (63.9 tok/s, block_size=16):
hf download onewhosighs/Apathy-Qwen3.8-27B-DFlash-drafter-v2 \
  --local-dir "$MODELS_DIR/Qwen3.8-27B-DFlash-v2"

# 2b. Or download the public DFlash2 drafter (z-lab/Qwen3.8-27B-DFlash2 or incoai/Qwen3.8-27B-DFlash2):
hf download z-lab/Qwen3.8-27B-DFlash2 \
  --local-dir "$MODELS_DIR/Qwen3.8-27B-DFlash2"
```
*(Note: The container's `flux` entrypoint automatically reads `block_size` from the drafter's `config.json` and sets `--dflash-gamma = block_size - 1` — `15` for `block_size: 16` drafters like `Apathy-Qwen3.8-27B-DFlash-drafter-v2` and `z-lab/Qwen3.8-27B-DFlash2`, or `7` for `block_size: 8` drafters like `incoai/Qwen3.8-27B-DFlash2`.)*

---

## 2. Pull the Container & Run Pre-Flight Diagnostics

```bash
# 1. Pull the container image from GitHub Container Registry:
docker pull ghcr.io/theapathy/flux-engine:gb10

# 2. List all discovered models and drafters in your $MODELS_DIR:
docker run --rm --gpus all --ipc=host \
  -v "$MODELS_DIR":/models:ro \
  ghcr.io/theapathy/flux-engine:gb10 list-models

# 3. Run full pre-flight hardware, driver, memory, and model readiness diagnostics:
docker run --rm --gpus all --ipc=host \
  -v "$MODELS_DIR":/models:ro \
  ghcr.io/theapathy/flux-engine:gb10 diagnose
```

### What to Check in `diagnose` Output

```json
{
  "cuda_runtime": {
    "driver_ok": true,
    "real_driver_libs": ["/usr/lib/aarch64-linux-gnu/libcuda.so.1"],
    "using_compat_stub": false
  },
  "memory": {
    "mem_total_mb": 124620,
    "mem_available_mb": 116400
  },
  "selected_model": {
    "name": "Qwen3.8-Flash-Next-NVFP4-Offload",
    "kernel_target": "qwen3.8-flash-next",
    "quant": "nvfp4",
    "ple_dir": "/models/Qwen3.8-Flash-Next-NVFP4-Offload/ple-offload",
    "speculative": {
      "mode": "native-mtp",
      "gamma": 5
    }
  },
  "readiness": {
    "ready": true,
    "issues": []
  }
}
```
- **`cuda_runtime.using_compat_stub: false`** — Confirms the container is bound to your host's real NVIDIA GPU driver (`libcuda.so.1`) and not the forward-compat stub.
- **`memory.mem_available_mb >= 105000`** — Confirms sufficient free unified memory before loading a 71 GiB checkpoint + 128K KV cache.
- **`selected_model.ple_dir: "/models/.../ple-offload"`** — Confirms your repacked `ple-offload/` sidecar directory was detected. If `ple_dir` is `null` on `Qwen3.8-Flash-Next` or `CYBER-FROST-3.8`, stop and run the repacker in [Section 1](#1-downloading--preparing-models-critical-ple-offload-step).
- **`selected_model.speculative`** — Confirms `"mode": "native-mtp"` (`gamma: 5`) or `"mode": "dflash"` (`gamma: 15`) is paired and active.

---

## 3. Serve Any Model & Operating Mode

### A. Serve `JMNI-Labs/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid` or Stock `Qwen3.8-Flash-Next-NVFP4-Offload`
*(Add optional `--security-opt seccomp=unconfined` if you want kernel `io_uring` async NVMe PLE DMA instead of the default `pread64` fallback.)*
```bash
# Serve JMNI-Labs/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid (v9 Max-Prefill + built-in NVFP4 MTP):
docker run --rm -it --gpus all --ipc=host --shm-size 16g \
  --ulimit memlock=-1 --ulimit stack=67108864 -p 8899:8899 \
  -v "$MODELS_DIR":/models:ro \
  ghcr.io/theapathy/flux-engine:gb10 serve \
  --model Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid \
  --mode max-prefill

# Default v9 Max-Prefill on Stock Qwen3.8-Flash-Next-NVFP4-Offload (16K chunking, 6,082+ tok/s prefill, 56.5+ tok/s K=5 MTP decode):
docker run --rm -it --gpus all --ipc=host --shm-size 16g \
  --ulimit memlock=-1 --ulimit stack=67108864 -p 8899:8899 \
  -v "$MODELS_DIR":/models:ro \
  ghcr.io/theapathy/flux-engine:gb10 serve \
  --model Qwen3.8-Flash-Next-NVFP4-Offload \
  --mode max-prefill

# Interactive v9-w4096 Mode (4K chunking, preserves >12.5 GiB free UMA memory floor):
docker run --rm -it --gpus all --ipc=host --shm-size 16g \
  --ulimit memlock=-1 --ulimit stack=67108864 -p 8899:8899 \
  -v "$MODELS_DIR":/models:ro \
  ghcr.io/theapathy/flux-engine:gb10 serve \
  --model Qwen3.8-Flash-Next-NVFP4-Offload \
  --mode interactive
```

### B. Serve `CYBER-FROST-3.8-NVFP4-Offload`
```bash
# Serve CYBER-FROST-3.8-NVFP4-Offload with v9 Max-Prefill + Built-In MTP:
docker run --rm -it --gpus all --ipc=host --shm-size 16g \
  --ulimit memlock=-1 --ulimit stack=67108864 -p 8899:8899 \
  -v "$MODELS_DIR":/models:ro \
  ghcr.io/theapathy/flux-engine:gb10 serve \
  --model CYBER-FROST-3.8-NVFP4-Offload \
  --mode max-prefill
```
> [!TIP]
> **Built-In `K=5` MTP Speculative Decoding Across All Three Checkpoints:**  
> `JMNI-Labs/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid`, `Blackfrost-AI/CYBER-FROST-3.8-NVFP4-V2`, and `nvidia/Qwen3.8-Flash-Next-NVFP4` all use their built-in `mtp.*` head directly (`--speculative --num-drafts 4 --mtp-vocab 0 --mtp-quantization bf16`). If you want to compare raw `K=1` non-speculative trunk output against `K=5` MTP speculative decoding, pass `-e FLUX_DRAFT_MODEL=none` (or `--no-speculative`). If you want to override the built-in MTP expert bank with an external sidecar directory, pass `--mtp-from-path <DIR>` together with `-e FLUX_MTP_SOURCE=sidecar`; without it the built-in bank wins and the sidecar is skipped.

### C. Serve `Qwen3.8-27B-NVFP4` with `gamma=15` DFlash Tree Speculative Decode
```bash
docker run --rm -it --gpus all --ipc=host --shm-size 16g \
  --ulimit memlock=-1 --ulimit stack=67108864 -p 8899:8899 \
  -v "$MODELS_DIR":/models:ro \
  ghcr.io/theapathy/flux-engine:gb10 serve \
  --model Qwen3.8-27B-NVFP4
```

### D. Verify the Live HTTP Endpoint (`localhost:8899`)
Wait ~35–55 seconds for weight loading + warmup to complete (`Listening on http://0.0.0.0:8899`), then test from a second terminal:

```bash
# 1. Health check:
curl -s http://localhost:8899/health
# {"status":"ok"}

# 2. Standard chat completion. Qwen3.8-Flash-Next defaults to thinking off and
#    greedy decoding (temperature 0) since 9.15.3.1, so no sampling fields are needed:
curl -s http://localhost:8899/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Write a warp-aggregated atomic counter in CUDA."}],
    "max_tokens": 256
  }' | jq .

# 3. Enable <think> reasoning on a per-request basis via chat_template_kwargs:
curl -s http://localhost:8899/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Prove that every finite integral domain is a field."}],
    "chat_template_kwargs": {"enable_thinking": true},
    "max_tokens": 1024,
    "temperature": 0.6
  }' | jq .
```

> [!NOTE]
> **Qwen3.8-Flash-Next sampling defaults (since `9.15.3.1`):** the thinking-off `MODEL.toml` presets (`non_thinking`, `tools`) use `temperature 0.0` and `presence_penalty 0.0`, and `thinking_default = false`. A request with no sampling fields is greedy and identical to an explicit `"temperature": 0`. Since `9.15.6`, a request that turns thinking on and sends no sampling fields uses the checkpoint's `generation_config.json` (`temperature 1.0`, `top_p 0.95`, `top_k 20`); greedy long reasoning tends to loop. Explicit request fields override the preset, so the 9.15.3 behaviour is one request away: `"chat_template_kwargs": {"enable_thinking": true}` for reasoning, and an explicit `"temperature"` (9.15.3 used `1.0` with thinking, and `0.7` with `presence_penalty 1.5` without thinking). Thinking and `temperature > 0` lower MTP acceptance, so decode is slower. On a GB10 with the QAD5500 Hybrid (`FLUX_MODE=256k`, 1024 tokens, code / prose / user question), 9.15.3.1 measured `90.8 / 65.0 / 63.7 tok/s` with no sampling fields. 9.15.3 measured `38.4 / 37.0 / 30.1 tok/s` with its old defaults.

---

## 3.5. Web Recipe Studio (`GET /`), Interactive Ratatui TUI & CLI Mode Wizard

Flux Engine ships with three self-guiding interfaces so you can inspect live telemetry, switch workload profiles, and export `.env` recipes without memorizing environment variables:

1. **Web Mission Control & Self-Guiding Recipe Studio (`http://localhost:8899/`)**:
   - Open `http://localhost:8899/` in any browser while the server is running to access the dark-mode **Apathy Flux Engine Mission Control & Recipe Studio** (featuring the official hexagonal Tensor Core emblem, live status & tok/s badge, 4 hardware roofline cards, 6 clickable workload mode cards, one-click Docker/`.env` recipe generator, and a built-in `/v1/chat/completions` probe).
   - Programmatic JSON catalog endpoints are served at `GET /flux/modes`, `GET /recipes`, and `GET /v1/flux/modes`:
     ```bash
     curl -s http://localhost:8899/flux/modes | jq .
     ```
2. **Interactive Ratatui TUI (`docker run -it`)**:
   - Whenever you launch the container with an interactive TTY (`docker run --rm -it ...`), `flux-autodetect` prints the startup interstitial banner and hands off directly to the live **Ratatui TUI** (`--no-tui` is only added in non-interactive/background runs).
   - Inside the TUI, press `1..6` to switch tabs (`Main`, `Chat`, `Stats`, `Library`, `Logs`, `Help`), or type `/recipes` (or `/modes`) in the command bar to inspect all 6 self-guiding recipe modes.
   - To force plain structured logs even in an interactive terminal, pass `--no-tui` or `-e FLUX_NO_TUI=1`.
3. **Interactive CLI Recipe Wizard (`flux wizard` / `flux recipes`)**:
   - Run the interactive mode advisor before launching to get a tailored `docker run` and `.env` export command for your exact workload:
     ```bash
     # Interactive menu advisor:
     docker run --rm -it -v "$MODELS_DIR":/models:ro ghcr.io/theapathy/flux-engine:latest wizard

     # Or query a specific workload directly (e.g. agent, interactive, decode, 256k, concurrency):
     docker run --rm -v "$MODELS_DIR":/models:ro ghcr.io/theapathy/flux-engine:latest wizard agent
     ```

---

## 4. Reproduce the Benchmarks (All Contexts, Decode, Concurrency & Estonia 136.6K)

Create a local `./results` folder on your host so all JSON artifacts, server logs, and SVG/PNG Model Cards are saved to disk:
```bash
mkdir -p ./results
```

### Expected Reference Numbers on a Single DGX Spark (`GB10`, `sm_121a`)

| Benchmark Suite | Profile (`--mode`) | Metric | Certified Reference (`GB10`) | Free UMA Floor |
| :--- | :--- | :--- | :--- | :--- |
| **`Qwen3.8-Flash-Next` Prefill (`8K..255K`)** | `max-prefill` (`w16384`) | Prefill `tok/s` | **`6,291.0 tok/s` med (`6,299.3` pk, `6,420.3` chunk pk) @ 16K** (`6,168.5` @ 32K, `6,028.5` @ 8K, `6,006.5` @ 64K, `5,905.6` @ 128K, `5,524.7` @ 255K) | `>= 7,429 MiB` |
| **`Qwen3.8-Flash-Next` Interactive Prefill** | `interactive` (`w4096`) | Prefill `tok/s` | **`5,447.7 tok/s` (`5,449.8` pk) @ 8K** (`5,326.8` @ 16K, `5,294.8` @ 32K, `5,158.2` @ 64K, `5,063.0` @ 128K, `4,990.7` @ 255K) | **`>= 9,890 MiB`** |
| **`Qwen3.8-Flash-Next` Single-Stream `K=5` Decode** | `max-prefill` / `interactive` | Pure Decode `tok/s` (`K=5` MTP) | **`91.26 tok/s` (`tf_code`) · `87.72` (`sd_code`) · `62.59–66.20` (mixed)** · `53.03` @ 64K · `46.1` @ 128K · `41.8` @ 255K (`100%` 0-ULP greedy parity; `v1.0.x` + DFlash2 reaches `82.4` greedy / `52.0` @ 128K) | `>= 7,429 – 9,890 MiB` |
| **`Qwen3.8-Flash-Next` Concurrent `K=5` Decode** | Bounded-transpose concurrency (`C=5` / `C=8` batched `K=5`; `C=16` in `v1.0.x`) | Aggregate Decode `tok/s` | **`420.28 tok/s` (`C=5` code)** · **`295.73 tok/s` (`C=5` prose)** · **`580.66 tok/s` (`C=8`)** · **`298.8 tok/s` (`C=16` prose, `v1.0.x`)** · **`85.5 tok/s` (`C=8` `K=1` QSA)** | `>= 9,500 MiB` |
| **`Qwen3.8-Flash-Next` Native `262K` & GSM8K** | `256k` (`262,144` tok) & `gsm8k` | Needle Retrieval & Accuracy | **`5/5 (100% PASS)`** needles (`1%..99%` of `262,144` tok) · **GSM8K `97.27%`** (`1,283 / 1,319`, `0` interventions) | **`>= 11,452 MiB`** |
| **`Qwen3.8-Flash-Next` Estonia `136.6K`** | `long-context` (`estonia`) | `136,576`-tok Prefill + 6-Hop Decode | **`5,428.3 tok/s` prefill** (`25.16 s` TTFT) + **`48.2 tok/s` peak / `43.8 tok/s` avg decode** (`5/5 PASS`) | **`8,431 – 8,847 MiB`** |
| **`Qwen3.8-27B-NVFP4` Speculative Decode** | `auto` (`gamma=15` `DFlash`) | Pure Decode `tok/s` (`K=16` Tree) | **`63.9 tok/s` BF16 KV** (`8.33` emitted / `129.8 ms` on 400-tok coding probe; `58.0 tok/s` on corrected-ABI `v3` control) | `> 45,000 MiB` |

---

### Method 1 — One-Shot Automated Benchmark Runs (Starts Server -> Runs Benchmark -> Renders Cards -> Exits)

```bash
# 1. All-Context Prefill + Decode Matrix (0, 8K, 16K, 32K, 64K, 128K) + Visual Model Card:
docker run --rm --gpus all --ipc=host --shm-size 16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$MODELS_DIR":/models:ro \
  -v "$(pwd)/results":/results \
  ghcr.io/theapathy/flux-engine:gb10 serve \
  --model Qwen3.8-Flash-Next-NVFP4-Offload \
  --mode max-prefill \
  --with-lil matrix \
  --with-card

# 2. Estonia 136.6K Long-Context 6-Hop Needle-in-a-Haystack & Coherent Decode Benchmark:
docker run --rm --gpus all --ipc=host --shm-size 16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$MODELS_DIR":/models:ro \
  -v "$(pwd)/results":/results \
  ghcr.io/theapathy/flux-engine:gb10 serve \
  --model Qwen3.8-Flash-Next-NVFP4-Offload \
  --mode long-context \
  --with-estonia \
  --with-card

# 3. Fast Prefill Reproducibility Sweep (8K, 16K, 32K, 64K, 128K TTFT + 20/20 SHA-256 Identity Gate):
docker run --rm --gpus all --ipc=host --shm-size 16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$MODELS_DIR":/models:ro \
  -v "$(pwd)/results":/results \
  ghcr.io/theapathy/flux-engine:gb10 serve \
  --model Qwen3.8-Flash-Next-NVFP4-Offload \
  --mode max-prefill \
  --with-bench \
  --with-card
```
*(To run the exact same one-shot suites on `JMNI-Labs/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid` or `CYBER-FROST-3.8-NVFP4-V2`, simply change `--model` to `Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid` or `CYBER-FROST-3.8-NVFP4-Offload`.)*

---

### Method 2 — Interactive Benchmarking Against a Running Container

Start the server in the background named `flux`:
```bash
docker run --rm -d --name flux --gpus all --ipc=host --shm-size 16g \
  --ulimit memlock=-1 --ulimit stack=67108864 -p 8899:8899 \
  -v "$MODELS_DIR":/models:ro \
  -v "$(pwd)/results":/results \
  ghcr.io/theapathy/flux-engine:gb10 serve \
  --model Qwen3.8-Flash-Next-NVFP4-Offload \
  --mode max-prefill
```

Wait until `docker logs -f flux` shows `Listening on http://0.0.0.0:8899`, then run any benchmark command:

```bash
# A. Prefill-only sweep across specific context depths (8k, 16k, 32k, 64k, 128k):
docker exec -it flux lil-bench prefill \
  --prefill-contexts 8k,16k,32k,64k,128k \
  --output /results/prefill-sweep.json

# B. Single-stream Coding Peak decode benchmark (5 runs x 2,000 tokens, temperature=0):
docker exec -it flux lil-bench coding \
  --runs 5 \
  --output /results/coding-peak.json

# C. Full prefill + decode matrix across 0, 8k, 16k, 32k, 64k, 128k (e.g., 3 repetitions):
docker exec -it flux lil-bench matrix \
  --contexts 0,8k,16k,32k,64k,128k \
  --reps 3

# D. Multi-stream concurrency sweep (C=1,2,4,8 across 0, 8k, 32k):
docker exec -it flux lil-bench concurrency 1,2,4,8 --contexts 0,8k,32k

# E. Estonia 136.6K long-context needle-in-a-haystack test:
#    (Ensure server was started with --mode long-context or --max-seq-len 139264)
docker exec -it flux lil-bench estonia --output /results/estonia-results.json

# F. Render dark-mode SVG/PNG benchmark cards + MODEL_CARD.md + privacy-scrubbed bundle:
docker exec -it flux flux model-card
```

All outputs are written directly to `./results/` on your host:
- `results/MODEL_CARD.md` & `results/summary.md`
- `results/card.svg` & `results/card.png`
- `results/submission-bundle/` (privacy-scrubbed bundle with `SUBMISSION_MANIFEST.json`)

---

## 5. Customizing Your Own Recipes & Kernel Knobs

Every kernel selector, prefill chunk width, MoE compact GEMM variant, PLE DMA ring setting, and speculative decoding threshold can be customized without rebuilding the container.

### 1. Export Any Built-In Recipe to a Local `.env` File
```bash
# List all built-in recipes:
docker run --rm \
  ghcr.io/theapathy/flux-engine:gb10 list-recipes

# Export the v9 max-prefill recipe to my-recipe.env on your host:
docker run --rm \
  ghcr.io/theapathy/flux-engine:gb10 export-recipe max-prefill > my-recipe.env
```

### 2. Run With Your Custom `.env` File (`--recipe`) or Inline Overrides (`--set`)
Precedence order (highest to lowest): `--set KEY=VAL` CLI flags -> `-e FLUX_*/ATLAS_*` container environment variables -> `--recipe /path/to/file.env` -> built-in `--mode` defaults.

```bash
docker run --rm -it --gpus all --ipc=host --shm-size 16g \
  --ulimit memlock=-1 --ulimit stack=67108864 -p 8899:8899 \
  -v "$MODELS_DIR":/models:ro \
  -v "$(pwd)/my-recipe.env":/custom.env:ro \
  ghcr.io/theapathy/flux-engine:gb10 serve \
  --model CYBER-FROST-3.8-NVFP4-Offload \
  --recipe /custom.env \
  --set FLUX_QWEN4_PREFILL_WIDE=16384 \
  --set FLUX_QWEN4_PREFILL_MOE_V28Q=2 \
  --set FLUX_QWEN4_MTP_CONF=0.6 \
  --set FLUX_PLE_READ_RINGS=4
```

*(Add `--dry-run` to any `serve` command to print the complete resolved environment and CLI command JSON without starting the GPU.)*

### 3. Complete Knob Reference (`FLUX_*` / `ATLAS_*`)

| Variable (`--set KEY=VAL` or `-e`) | Default (`v9`) | What It Controls |
| :--- | :--- | :--- |
| `FLUX_QWEN4_PREFILL_WIDE` | `16384` (`4096` in `w4096`) | Prefill chunk size in tokens (`4096`, `8192`, or `16384`). Lowering to `4096` increases free UMA RAM by `~4.2–4.5 GiB`. |
| `FLUX_QWEN4_PREFILL_MOE_V28Q` | `2` | Selects `v28q` (`gateup`) + `v17qr2` (`down`) W4A8 MXFP8 compact MoE prefill kernels. |
| `FLUX_QWEN4_PREFILL_MOE_PLAN_ORDER` | `inter` | Enables `v8` inter-expert tile ordering (`qwen4_moe_sort_plan_t_v8`) to maximize L2 weight reuse across tiles. |
| `FLUX_QWEN4_PREFILL_MHC_ROOF` | `t128` | Selects fused `t128` Manifold Hyper-Connection (`mHC`) pre-attention roofline kernel (`87%` of GB10 DRAM roof). |
| `FLUX_QWEN4_PREFILL_QSA_FP8` | `qkpv` | Enables FP8 E4M3 `Q*K` and `P*V` tensor-core attention in the 12 QSA sparse-attention layers. |
| `FLUX_QWEN4_PREFILL_GDN_NORM_FP8` | `2` (`3` in `fast-prefill`) | Gated DeltaNet RMSNorm + FP8 quantization fusion tier (`2` = lossless under `K=5` MTP, `3` = pure-prefill only). |
| `FLUX_QWEN4_FAST_PROJ_FP8_WEIGHT_CACHE_RESERVE_GB` | `7` (`10` in `long-context`) | Free unified memory floor (in GiB) below which FP8 fast-projection weight caching stops allocating. |
| `FLUX_PLE_READ_RINGS` | `4` | Number of asynchronous `io_uring` `O_DIRECT` NVMe DMA read rings for `ple-offload/`. |
| `FLUX_PLE_READ_AFFINITY` | `big` | Pins PLE NVMe reader threads to Cortex-X925 performance cores (`5-9,15-19`) on DGX Spark. |
| `FLUX_QWEN4_MTP_CONF` | `0.6` in `auto` / `max-prefill` / `256k` / `512k` / `1m` (since `9.15.3.1`), `decode`, `long-context`; `0` (off) in `interactive` | Confidence threshold for early-exit gating in the 4-step native MTP draft proposer. The launcher forces `0` when `FLUX_CONCURRENCY > 1`. The server logs `MTP_DRAFT_CONF_ENGAGED ... threshold=Some(0.6) early=true` when the gate is on. |
| `FLUX_QWEN4_MTP_CONF_EARLY` | `1` wherever `FLUX_QWEN4_MTP_CONF=0.6` | `1` stops drafting at the first draft step below the threshold, so a weak draft does not pay for the full `K=5` verify. |
| `FLUX_DRAFT_MODEL` | *(built-in `mtp.*` or auto-paired DFlash)* | Optional override path to an external MTP/DFlash drafter directory, or `none` to run pure `K=1` non-speculative decoding. |
| `FLUX_GAMMA` | `4` (MTP `K=5`) / `15` (DFlash) | Number of speculative draft tokens proposed per verify step. |
| `FLUX_DFLASH_LOOKUP` | `off` | DFlash only (Qwen3.8-27B with its DFlash2 drafter): adds a context-lookup drafter, lossless (byte-identical to the TC-parity reference). `sam` (suffix automaton) or `retrieval`. Measured A/B on 27B: edit-heavy prompts +33% (`sam`) / +37% (`retrieval`); 32K matrix decode 14.8 -> 16.7 (`sam`) / 18.9 (`retrieval`) tok/s. `retrieval` measured faster here but costs up to ~3.7 ms per step on a lookup miss at 262K context (microbench). Forced off with a warning when the launch does not run `--dflash`; set it instead of `ATLAS_DFLASH_SAM` / `ATLAS_DFLASH_RETRIEVAL`. |
| `FLUX_CPU_AFFINITY` | `5-9,15-19` | CPU core affinity mask (`taskset -c`) on 20-core DGX Spark (`GB10`). |

---

## 6. Built-In Operating Modes Summary

> **Which mode to run:** use `FLUX_MODE=auto` (the default, = `max-prefill`, `135,168`-token context) unless a request needs more than ~`135K` tokens. `256k` is the native-`262K` **BF16 correctness profile** (BF16 QSA Q/K/V prefill operands, QSA verify rows `0`): it costs **~5% prefill** against `auto` / `max-prefill` at `16K`–`128K` and ~`15%` at `8K` (measured A/B on GB10), and the launcher logs one `INFO` line saying so when it engages.

| Mode (`--mode`) | Recipe File | Default Chunk / Max Seq | Description |
| :--- | :--- | :--- | :--- |
| `max-prefill` *(or `v9`, `auto`)* | `flux-v9-max-prefill.env` | `16,384` / `135,168` | Maximum prefill (`6,291.0 / 6,299.3 tok/s` @ 16K, `5,905.6 tok/s` @ 128K) + `91.26 tok/s` (`tf_code`) / `86.86 tok/s` (`rep0`) / `62.59 tok/s` (`code0`) lossless `K=5` MTP decode (`>= 7,429 MiB` free UMA). |
| `interactive` *(or `w4096`)* | `flux-v9-w4096-interactive.env` | `4,096` / `135,168` | Interactive 4K chunking with `>= 9,890 MiB` free UMA memory floor (`5,447.7 tok/s` @ 8K, `5,063.0 tok/s` @ 128K, `5/5 >= 5,000 tok/s`). |
| `decode` *(or `k5-decode`)* | `flux-k5-lossless-decode.env` | `4,096` / `135,168` | Dedicated lossless `K=5` MTP & DFlash speculative decode profile (`91.26 tok/s` `tf_code`, `87.72 tok/s` `sd_code`, `55.27 tok/s` @ 8K, `53.03 tok/s` @ 64K, `8/8 id=OK`). |
| `long-context` *(or `estonia`)* | `flux-estonia-136k.env` | `16,384` / `139,264` | 136.6K+ long-context needle & synthesis profile (`100%` Estonia 6-hop pass rate, `8.4+ GiB` free UMA floor). |
| `fast-prefill` *(or `fast`)* | `flux-fast-prefill.env` | `16,384` / `135,168` | Pure prefill velocity (`GDN_NORM_FP8=3` Ultra tier + prefill logit skip, for batch ingestion / RAG / TTFT). |
| `agent` *(or `coding`)* | `flux-v9-w4096-interactive.env` | `4,096` / `131,072` | Enables Radix prefix caching, `16` SSM state slots, and `16K` tool-call token budget for OpenAI/Anthropic agentic coding clients (`docker run ... ghcr.io/theapathy/flux-engine:gb10 agents`). |
| `256k` / `512k` / `1m` | Extended Context + Auto-YaRN | `16,384` / `262K–1M` | Only for requests above ~`135K` tokens; otherwise use `auto` / `max-prefill`. `256k` is the BF16 correctness profile (BF16 QSA Q/K/V prefill operands, QSA verify rows `0`, ~5% slower prefill than `auto` at `16K`–`128K`, ~`15%` at `8K`). Native `262,144`-token context (`5,524.7 tok/s` `w16384` / `4,990.7 tok/s` `w4096` prefill, `41.8 tok/s` decode, `5/5` needles `1%..99%`, `>= 11,452 MiB` free UMA). |
| `exact` *(or `strict`)* | `flux-exact-reference.env` | `4,096` / `135,168` | Unfused W4A16 bit-exact reference baseline (`K=1`, `35.2 tok/s` @ `C=1`, `85.5 tok/s` @ `C=8` native batched QSA). |

---

## 7. Live Diagnostics, Telemetry & Kernel Receipt Verification

When running benchmarks or debugging a custom configuration, Flux Engine exposes deterministic runtime receipts and hardware telemetry so you never have to guess what the GPU is doing.

### 7.1 Monitoring Real Unified Memory on DGX Spark (`Why nvidia-smi Shows [N/A]`)
On NVIDIA GB10, the Grace CPU and Blackwell GPU share 128 GB of coherent LPDDR5x memory over NVLink-C2C. Because there is no separate PCIe BAR VRAM counter, `nvidia-smi` always displays `[N/A]` for per-process memory.

Always monitor `/proc/meminfo` (`MemAvailable`) in a side terminal:
```bash
watch -n 1 "awk '/MemAvailable/ {printf \"Free UMA Available: %.2f GiB (%d MiB)\n\", \$2/1048576, \$2/1024}' /proc/meminfo"
```
- **Safe Operating Zone:** `MemAvailable >= 7,000 MiB` (`6.8+ GiB`).
- **GB10 Hardware Watchdog Floor:** `6,144 MiB` (`6.0 GiB`). All built-in Flux recipes enforce a reserve floor (`FLUX_QWEN4_FAST_PROJ_FP8_WEIGHT_CACHE_RESERVE_GB=7` for 128K, `=10` for 136.6K Estonia) so free memory stays comfortably above `7,072–12,695 MiB`.

### 7.2 Verifying Runtime Kernel Engagement Receipts (`docker logs`)
Every major `sm_121a` kernel path logs a one-time `[INFO]` receipt string to `stderr` the first time it executes. You can inspect `docker logs flux` (or `/results/server.log`) to confirm that all champion kernels engaged:

```bash
docker logs flux 2>&1 | grep -E "ENGAGED|PLAN_ORDER_RAN|OPROJ_AMAX_FUSED|PLE_OFFLOAD|WEIGHT_CACHE"
```

Key receipts you should see on `Qwen3.8-Flash-Next` / `CYBER-FROST-3.8` (`--mode max-prefill` or `interactive`):
- `MOE_PREFILL_W4A8_ENGAGED kernels=v28` — Confirms the `v28q` (`gateup`) + `v17qr2` (`down`) W4A8 MXFP8 Tensor Core MoE prefill kernels are active.
- `MOE_PREFILL_PLAN_ORDER_RAN order=inter planner=v8` — Confirms the `v8` inter-expert tile planner is grouping consecutive tiles of the same expert for L2 weight reuse.
- `OPROJ_AMAX_FUSED_CHUNK ... hits=12 refused=0 missed=0` — Confirms all 12 QSA `o_proj` layers fused the downstream mHC amax reduction (`0` fallbacks).
- `FAST_PROJ_FP8_WEIGHT_CACHE_ENGAGED` — Confirms the on-device FP8 E4M3 fast-projection weight cache is active while respecting the free UMA reserve floor.
- `FAST_PROJ_FP8_WEIGHT_CACHE_FILLED` — Logged once, at the first cache reuse (the end of the first prefill pass): `weights` resident, `uncached` refused by the MemAvailable stop line (those run the per-call NVFP4 -> e4m3 conversion every pass), `resident_mib`, `stop_line_mib`, and `complete=true` only when no weight was refused.
- `KV_ZERO_COALESCE_ENGAGED` & `ALLOC_SEQ_ASYNC_ENGAGED` — Confirms zero-blocking sequence allocation and coalesced `cuMemsetD8Async` KV page zeroing.

### 7.3 Enabling Per-Stage CUDA Event Hardware Timers (`QWEN4_PREFILL_STAGES`)
Want to see the exact millisecond breakdown across MoE `compact_gateup`, `compact_down`, `shared_expert`, GDN `proj_qkvz` / `gdn_core`, QSA `attn.core`, `mHC`, and `PLE`? Pass `--set ATLAS_QWEN4_PREFILL_STAGES=1`:

```bash
docker run --rm -it --gpus all --ipc=host --shm-size 16g \
  --ulimit memlock=-1 --ulimit stack=67108864 -p 8899:8899 \
  -v "$MODELS_DIR":/models:ro \
  ghcr.io/theapathy/flux-engine:gb10 serve \
  --model Qwen3.8-Flash-Next-NVFP4-Offload \
  --mode max-prefill \
  --set ATLAS_QWEN4_PREFILL_STAGES=1
```
Each prefill chunk will emit a structured `[QWEN4_PREFILL_STAGES]` line in the server log showing the exact hardware CUDA-event milliseconds spent in every subsystem across the 36 GDN layers and 12 QSA layers.

### 7.4 Scraping Live Speculative Decode Acceptance Metrics (`/metrics`)
To inspect the hardware speculative verify step counter and verify acceptance length (`accept_len`) directly from the server's Prometheus endpoint:
```bash
curl -s http://localhost:8899/metrics | grep -E "atlas_spec_decode|flux_spec_decode"
```
Dividing `completion_tokens` by the delta in `atlas_spec_decode_verify_total` gives the exact hardware acceptance length (`accept_len = 3.02–3.36` tokens/step on `K=5` MTP, and `8.33` emitted tokens/step on the 400-token `Qwen3.8-27B` `gamma=15` `DFlash` coding probe).

---

## 8. Top 11 Hardware, Container & Model Gotchas ("Things That Will Cost You a Day")

### Gotcha 1: OOM Kill or Node Freeze During Model Load (`122–174 GiB` Raw Checkpoint)
- **Symptom:** The container gets killed (`exit code 137` / `SIGKILL`) while loading SafeTensors shards for `nvidia/Qwen3.8-Flash-Next-NVFP4` (`122 GiB`) or `Blackfrost-AI/CYBER-FROST-3.8-NVFP4-V2` (`173.56 GiB`).
- **Root Cause:** You pointed `--model` at the un-repacked raw Hugging Face download where the 51B-parameter PLE table is still embedded inside the SafeTensors shards without `ple-offload/manifest.json` (`51.2 GB` FP8 or `102.4 GB` BF16), or Linux page cache is holding 40+ GiB of dirty pages.
- **Fix:**
  1. Run `repack-ple` once ([Section 1](#1-downloading--preparing-models-critical-ple-offload-step)) so `ple-offload/manifest.json` and `ple-offload/*.bin` exist (`0 GB` GPU RAM for PLE, `~71 GiB` resident checkpoint).
  2. Run `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` before starting the container.

### Gotcha 2: `CUDA_ERROR_SYSTEM_NOT_READY` or `using_compat_stub: true`
- **Symptom:** Server fails at startup with `CUDA driver initialization failed` or `diagnose` shows `"using_compat_stub": true`.
- **Root Cause:** Either `--gpus all` was omitted from `docker run`, or `LD_LIBRARY_PATH` inside a custom wrapper put `/usr/local/cuda/compat` ahead of `/usr/lib/aarch64-linux-gnu/libcuda.so.1`.
- **Fix:** Always run with `--gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864` (add optional `--security-opt seccomp=unconfined` only if you want async `io_uring` NVMe PLE offload). The built-in `/usr/local/bin/flux` entrypoint automatically strips `/usr/local/cuda/compat` whenever the host's real `libcuda.so.1` is mounted.

### Gotcha 3: `EINVAL (os error 22)` When Opening `ple-offload/ple-ngram-nvfp4-*.bin`
- **Symptom:** Server logs an I/O error (`Invalid argument / os error 22`) when initializing the 4-ring PLE NVMe reader.
- **Root Cause:** By default, the PLE NVMe reader uses Linux `O_DIRECT` bypass DMA, which requires `8,192`-byte page alignment and a local filesystem (`ext4` or `xfs` on NVMe) that supports `O_DIRECT`. Network mounts (`NFS`, `SMB`), `VirtioFS`, or `tmpfs` reject `O_DIRECT` with `EINVAL`.
- **Fix:** Place `$MODELS_DIR` on a local NVMe drive. If you are forced to test on a filesystem without `O_DIRECT` support, pass `--set FLUX_PLE_PAGE_CACHE=1` (alias `ATLAS_PLE_PAGE_CACHE=1`, switches PLE reads to buffered OS page cache).

### Gotcha 4: Watchdog Reset at `128K–136.6K` Context (`MemAvailable < 6.0 GiB`)
- **Symptom:** Running the `136,576`-token Estonia benchmark under a custom recipe triggers a node reset or OOM near the end of prefill.
- **Root Cause:** Unbounded on-device FP8 fast-projection weight caching (`ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_WEIGHT_CACHE=1` without a reserve cap) consumes an extra `~4.6 GiB` of unified memory, dropping free memory at 136.6K KV context to `~3.8 GiB`.
- **Fix:** Always use `--mode long-context` for `128K–139K` workloads (which sets `FLUX_QWEN4_FAST_PROJ_FP8_WEIGHT_CACHE_RESERVE_GB=10` and `FLUX_QWEN4_QSA_VERIFY_ROWS=0`, keeping `8,431–8,847 MiB` free throughout Estonia), or use `--mode interactive` (`w4096`, which keeps `>11,900 MiB` free).

### Gotcha 5: Using `GDN_NORM_FP8=3` With `K=5` Speculative Decode
- **Symptom:** You manually set `--set FLUX_QWEN4_PREFILL_GDN_NORM_FP8=3` while `K=5` MTP speculative decoding is active, and greedy `temperature=0` outputs occasionally diverge from `K=1` plain decode.
- **Root Cause:** `GDN_NORM_FP8=3` enables an asynchronous store-warp E4M3 reduction inside the FlashInfer GDN kernel that is designed for **pure prefill (`--mode fast-prefill`, `K=1`)**. Under multi-row `K=5` speculative verification rollbacks, store-warp ordering is not state-deterministic across chunk boundaries.
- **Fix:** Keep `FLUX_QWEN4_PREFILL_GDN_NORM_FP8=2` (with `ATLAS_QWEN4_PREFILL_GDN_NORM_FP8_V2=1`, the default in `max-prefill` and `interactive`), which delivers `6,082.2 tok/s` prefill and `10/10` (`2,560 / 2,560` tokens) bit-exact greedy equivalence under `K=5` MTP.

### Gotcha 6: Enabling `LAST_LAYER_CROP=1` Without `MTP_LASTK_DEFER=1`
- **Symptom:** Custom recipe sets `ATLAS_QWEN4_PREFILL_LAST_LAYER_CROP=1` alone and either refuses to enable crop or reduces first-step MTP acceptance after prefill.
- **Root Cause:** Cropping unused layer-47 logits during prefill must still preserve the trailing hidden states across chunk boundaries for the native MTP proposer's first draft step.
- **Fix:** Built-in recipes automatically pair `ATLAS_QWEN4_PREFILL_LAST_LAYER_CROP=1` with `ATLAS_MTP_LASTK_DEFER=1` and `ATLAS_QWEN4_MTP_LASTK_SKIP_INTERMEDIATE=1`. Keep all three enabled together.

### Gotcha 7: HTTP `400 Bad Request` When Client Sends `"reasoning_effort"`
- **Symptom:** An OpenAI SDK client or agent harness sends `"reasoning_effort": "high"` and receives an HTTP `400` template error.
- **Root Cause:** Qwen3.8's upstream Jinja chat template explicitly calls `raise_exception` when `reasoning_effort` is passed.
- **Fix:** Control thinking via `"chat_template_kwargs": {"enable_thinking": true}` (or `false`) in the request JSON, or set `-e FLUX_THINKING=on` at container launch. `FLUX_THINKING_BUDGET=1024` is an explicit override; when it is omitted, Qwen3.8-Flash-Next keeps its `MODEL.toml` ceiling (`max_thinking_budget = 262144`). A thinking request without a budget may think for at most 90% of `max_tokens` (70% with tools) and must leave `max(15%, 1024)` tokens for the answer, so at the default `max_tokens` of `4096` it thinks for at most `3072` tokens; send a larger `max_tokens` for long reasoning.

### Gotcha 8: Passing Dense 27B DFlash Flags (`--dflash-gamma 15`) to `Qwen3.8-Flash-Next` or `CYBER-FROST-3.8`
- **Symptom:** Forcing `--gamma 15` (`K=16` verify rows) on `Qwen3.8-Flash-Next` causes decode speed to drop to `~3 tok/s`.
- **Root Cause:** Dense `Qwen3.8-27B` uses a bidirectional block-diffusion drafter (`DFlash`, `gamma=15`) where all 16 tokens fit into a single dense Tensor Core `m16n8k32` tile. `Qwen3.8-Flash-Next` is a 512-expert MoE model whose batched expert-deduplicated verify kernels (`moe_expert_*_r5`) are tuned for native MTP `K=5` (`--speculative --num-drafts 4`).
- **Fix:** Let the container auto-detect speculative settings per model (`K=5` native MTP for `Qwen3.8-Flash-Next` / `CYBER-FROST-3.8`, and `gamma=15` `DFlash` for `Qwen3.8-27B`).

### Gotcha 9: Raising `--gpu-memory-utilization` Above `0.55` or Exceeding UMA Concurrency Headroom on GB10
- **Symptom:** Passing `--gpu-memory-utilization 0.85` or unthrottled high concurrency on long contexts causes an OS OOM during wide prefill chunks.
- **Root Cause:** Unlike discrete GPUs where model weights live in dedicated GDDR/HBM and `--gpu-memory-utilization` only sizes the KV pool, on GB10 unified memory the `71 GiB` model weights, OS kernel, page cache, and CUDA workspaces all share the same 128 GB pool, and every active sequence in a 36-layer GDN model owns a persistent FP32 recurrent state slot.
- **Fix:** Keep `--gpu-memory-utilization 0.55` on single-node GB10. For high-concurrency serving (`FLUX_CONCURRENCY=16` in `v1.0.x`, or `C=32..128` via the lazy-H SSM pool), use `FLUX_MODE=auto` / `interactive` (`w4096`) and let startup memory admission cap concurrent slots safely.

### Gotcha 10: Mixing Up Throughput Counters (`TTFT` vs Pure Decode vs Total Wall Time)
- **Symptom:** A custom script computes `completion_tokens / total_request_time` on an 8K prompt with `max_tokens=128` and reports a lower decode number than `lil-bench`.
- **Root Cause:** Dividing `completion_tokens` by `total_request_time` includes the `1.4 s` 8K prefill (`TTFT`) inside the decode denominator!
- **Fix:** Compare like with like:
  - **Prefill (`tok/s`):** `prompt_tokens / ttft_seconds` (or server `Done:` prefill tok/s).
  - **Pure Decode (`tok/s`):** `completion_tokens / (total_time - ttft_seconds)`, `usage["response_token/s"]` in the JSON response, or the server `Done:` decode log line.

### Gotcha 11: `K=5` Decode Stuck at `~30 tok/s` on Legacy `9.15.3` With Default Sampling
- **Symptom:** On legacy `9.15.3` images, `Qwen3.8-Flash-Next` decodes at `29–33 tok/s` sustained, below the `35–41 tok/s` of speculation off and far below current `v1.0.3` numbers. A `temperature: 0` request with thinking off on the same server is much faster.
- **Root Cause:** Legacy `9.15.3` defaults (thinking on at `temperature 1.0`, or `temperature 0.7` + `presence_penalty 1.5`) kept MTP acceptance at about `2.1–2.4` tokens/step, and the draft-confidence gate was off so every step paid for a full `K=5` verify (`60–71 ms`: about `14 steps/s x 2.1 ≈ 30 tok/s`).
- **Fix:** Upgrade to **`v1.0.3`** (fixed since `9.15.3.1`), which uses greedy / thinking-off defaults and enables `FLUX_QWEN4_MTP_CONF=0.6` + `FLUX_QWEN4_MTP_CONF_EARLY=1`. If pinned to legacy `9.15.3`, pass `-e ATLAS_QWEN4_MTP_CONF=0.6 -e ATLAS_QWEN4_MTP_CONF_EARLY=1` and send `"temperature": 0, "chat_template_kwargs": {"enable_thinking": false}` with each request.

---

## 9. Technical FAQ (Roofline Math, Prefix Cache, Lossless Proofs)

### Q1: How can a `273 GB/s` LPDDR5x system achieve `6,082+ tok/s` prefill and `56.5–63.9 tok/s` decode without violating the memory bandwidth roofline?
A first-order calculation assumes every token must read all model weights from DRAM at `273 GB/s`. That only applies to **unbatched `K=1` single-token decode** (`~8.11 GB/token` active weights on `Qwen3.8-Flash-Next` -> `~30 ms/token` = `33.5 tok/s` measured `K=1` baseline).
- **Why Prefill Reaches `6,082+ tok/s` (`1,346.9 ms` per 8K tokens):** During a `16,384`-token prefill chunk (`w16384`), each weight is read **once per chunk** and amortized across all `16,384` tokens. The routed experts alone are `67.95 GB` (512 experts x 48 layers, NVFP4 + e4m3 scales), and `97.2%` of them are touched at `16,384` rows, so the routed stream is **`~66.0 GB` per 16K chunk** (`44.0 GB` gate/up + `22.0 GB` down, `~4.0 MB/token`), before the shared expert and the cached FP8 dense projections. At `273 GB/s` that is **`241.9 ms` per 16K chunk (`121.0 ms` per 8K tokens, `~9%` of total prefill time)**, about a third of the `690 ms` the routed MoE actually takes per 16K chunk. Prefill on GB10 is therefore governed by compute, not the weight stream: the MoE gate/up GEMMs are bound by `m16n8k32` FP8/MXFP8 Tensor Core MMA and the in-register `NVFP4 -> MXFP8` transcode (`v28q` / `v17qr2`, `~123 TF/s` of the `246 TF/s` QMMA roof), the down GEMMs by their MXFP8 activation traffic, and the 4-stream `mHC` by L2 bandwidth (`2,188 GB/s`).
- **Why `K=5` MTP Decode Reaches `56.5–57.6 tok/s` on `Qwen3.8-Flash-Next`:** Native `K=5` MTP verifies `5` candidate tokens (`1` bonus + `4` drafts) in a single batched pass. Dense weights (`~6.65 GB`) are read once for all 5 rows, and `ATLAS_QWEN4_MOE_DEDUP=1` deduplicates shared expert hits across the 5 correlated tokens (`~31–36` unique experts/layer instead of `50`). Total DRAM traffic is `~12.3 GB` per 5-row verify cycle (`~55 ms/cycle` = `17.6–18.2 steps/s`). Multiplied by a measured acceptance of `accept_len = 3.02–3.20` committed tokens/step, effective DRAM read drops to **`3.84 GB/token`** (`216 GB/s` = `79%` of STREAM ceiling), yielding **`56.5 tok/s` median (`57.6 tok/s` peak)**.
- **Why `gamma=15` `DFlash` Reaches `63.9 tok/s` on Dense `Qwen3.8-27B`:** The `14.0 GB` NVFP4 target model verifies all `K=16` tokens (`1` bonus + `15` block-diffusion drafts) in a single `m16n8k32` Tensor Core tile (`~129.8 ms/cycle`). On the 400-token coding probe with `8.33` mean emitted tokens/cycle (`7.33` accepted drafts + `1` bonus), weight bandwidth per committed token drops to **`1.78 GB/token`** (`14.85 GB / 8.33`), achieving **`63.9 tok/s` (`BF16 KV`)** (`8.33 / 0.1298 = 64.18 tok/s`).

### Q2: Is prefix caching or KV block reuse inflating the prefill benchmark numbers?
**No.** Both `flux-bench` and `lil-bench` enforce five independent safeguards against prefix caching during prefill benchmarks:
1. Server is launched with `--enable-prefix-caching false` and `--ssm-cache-slots 0` in benchmark mode.
2. Every request advances a prime token stride (`STRIDE = 104729`) through the on-disk evaluation corpus so every prompt window has a distinct `prompt_sha256` and shares zero leading tokens with any prior request.
3. Every response receipt asserts `usage.prompt_tokens_details.cached_tokens == 0` (`cached_tokens_max == 0`).
4. Every request allocates a fresh sequence slot and explicitly zeroes its KV blocks via `cuMemsetD8Async` (`KV_ZERO_COALESCE_ENGAGED`).
5. Warmup iterations (`drop_first = 1`) and post-run determinism checks (`20/20` SHA-256 identity gate) are strictly excluded from timed measurements.

### Q3: Does `Qwen3.8-Flash-Next` require an external MTP drafter repository, or does it use the checkpoint's built-in `mtp.*` head?
Flux Engine uses the **checkpoint's built-in `mtp.*` layer directly** (`--speculative --num-drafts 4 --mtp-vocab 0 --mtp-quantization bf16`) across all supported `Qwen3.8-Flash-Next` checkpoints:
- **[`JMNI-Labs/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid`](https://huggingface.co/JMNI-Labs/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid)**: `MIXED_PRECISION` target (`MXFP8` attention/GDN/shared-expert + `NVFP4` routed experts) with built-in weight-only `NVFP4` MTP experts (`mtp.layers.0.mlp.experts.{0..511}.{gate,up,down}_proj.{weight,weight_scale,weight_scale_2}`).
- **[`Blackfrost-AI/CYBER-FROST-3.8-NVFP4-V2`](https://huggingface.co/Blackfrost-AI/CYBER-FROST-3.8-NVFP4-V2)**: Built-in BF16 `mtp.*` layer inside shard 131, automatically quantized to NVFP4 and cached on first load.
- **[`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)**: Built-in packed BF16 `mtp.*` experts, automatically quantized to NVFP4 and cached on first load.
- Optional `--mtp-from-path <DIR>` remains supported if you want to override the built-in MTP expert bank with an external sidecar directory (add `-e FLUX_MTP_SOURCE=sidecar` when the checkpoint ships its own MTP experts), and `-e FLUX_DRAFT_MODEL=none` (or `--no-speculative`) forces pure `K=1` non-speculative trunk decoding.

### Q4: How do I verify for myself that `K=5` MTP speculative decode is 100% lossless (`2,560 / 2,560` tokens bit-identical to `K=1` greedy decode)?
In greedy mode (`temperature: 0.0`), speculative verification accepts a draft token if and only if it equals the exact `argmax` of the target model's logits. You can verify bit-for-bit identity on your own GPU in two commands:
1. Run a greedy request (`temperature: 0.0`, `max_tokens: 256`) with default `K=5` MTP enabled and hash the output text (`sha256sum`).
2. Restart with `-e FLUX_DRAFT_MODEL=none` (`K=1` non-speculative decode), send the exact same request, and compare the `sha256sum` — across our 10-prompt `0..64K` audit suite (`2,560` tokens total), `K=5` and `K=1` match **`10/10` (`2,560 / 2,560` tokens exact)** while running **`1.26x–1.55x` faster**.
