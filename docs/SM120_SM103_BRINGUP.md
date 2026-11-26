# Blackwell SM120 (`x86_64`) & SM103 (`B300`/`GB300`) Fast Bringup Guide

This guide covers building, deploying, and tuning **Flux Engine** (`v1.0.3`) on discrete NVIDIA Blackwell **SM120** (`sm_120a` — RTX 50-series & RTX PRO Blackwell workstations/servers) and Blackwell Ultra **SM103** (`sm_103a` — B300 / GB300) hosts, plus the 4-step checklist for adding future NVIDIA SM architectures quickly.

---

## 1. Hardware & VRAM Fit Matrix

Unlike unified-memory **GB10 (`SM121a`)**, discrete Blackwell GPUs strictly separate device GDDR7 / HBM3e VRAM from host DRAM. Flux Engine automatically detects `CU_DEVICE_ATTRIBUTE_INTEGRATED == 0`, excludes `/proc/meminfo` host RAM from VRAM budgeting, scales CTA/split-K occupancy to the live SM count (`active_sm_count()`), and stages `Qwen3.8-Flash-Next` PLE offload shards in host RAM (`/dev/shm`) with zero-syscall `mmap`.

| GPU | Arch (`FLUX_TARGET_HW`) | SMs | VRAM / BW | `Qwen3.8-27B-NVFP4` (+ DFlash2) | `Qwen3.8-Flash-Next-NVFP4` (+ DFlash2) |
|---|---|---:|---:|---|---|
| **RTX 5060 Ti / 5070 Ti / 5080** | `sm_120a` (`rtx6000` / `sm120`) | 36–84 | 16 GB GDDR7 | Does not fit (use `<=14B` NVFP4) | Does not fit |
| **RTX 5090** | `sm_120a` (`rtx6000` / `sm120`) | 170 | 32 GB · 1,792 GB/s | **Fits (`1x`)** w/ ~10+ GiB free KV | Needs `4x` (`TP=4`) |
| **RTX PRO 5000 Blackwell** | `sm_120a` (`rtx6000` / `sm120`) | 110 | 48 / 72 GB GDDR7 | **Fits (`1x`)** w/ 262K+ KV | **Fits (`2x` `TP=2 + EP=2`)** |
| **RTX PRO 6000 Blackwell WS/Server** | `sm_120a` (`rtx6000` / `sm120`) | 188 | 96 GB · 1,792 GB/s | **Fits (`1x`)** w/ 262K+ KV | **Fits (`1x`)** w/ `--ple-ram` + DFlash2 + 131K–248K KV |
| **B300 / GB300 (Blackwell Ultra)** | `sm_103a` (`gb300` / `sm103`) | 160+ | 288 GB HBM3e · 8 TB/s | **Fits (`1x`)** w/ massive KV | **Fits (`1x`)** inline or offloaded + 1M+ KV |

---

## 2. Five-Minute Bare-Metal / Cloud Spinup (Ubuntu 24.04 + CUDA 13.0)

### Step 1: System Dependencies & Toolchain

```bash
export PATH=/usr/local/cuda/bin:$PATH

# Install build tools + NCCL >= 2.28 (Ubuntu CUDA images often hold libnccl2)
apt-get update
apt-get install -y --allow-change-held-packages \
  build-essential clang lld cmake ninja-build pkg-config libssl-dev \
  git curl jq python3 python3-pip python3-venv \
  libnccl2 libnccl-dev

# Verify CUDA 13.0+ and NCCL >= 2.28
nvcc --version
dpkg -l libnccl2 | grep libnccl2

# Install Rust (if not already present)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
source "$HOME/.cargo/env"
```

> **Note:** Always ensure `/usr/local/cuda/bin` is on `PATH` before running `cargo build`, or `cudarc` / `atlas-kernels` will fail to locate `nvcc` and `ptxas`.

### Step 2: Optional FlashInfer Headers (`fi_attn` & GDN Prefill Acceleration)

Precompiled FlashInfer CuTe-DSL AOT object files (`fi_gdn/*.o`) are intentionally **not** committed to this repository (preserving full AGPL-3.0 Corresponding Source compliance; see [`crates/spark-model/csrc/fi_gdn/NOTICE`](../crates/spark-model/csrc/fi_gdn/NOTICE)). When `FLASHINFER_HOME` is unset, Flux Engine builds cleanly and uses its native CUDA GDN (`qwen4_gdn_fi_glue.cu`) and native CUDA attention prefill fallbacks.

To compile `libatlas_qwen4_fi_attn.a` (`crates/spark-model/csrc/fi_attn/qwen4_fi_attn.cu` — FlashInfer ragged HD256 attention for short/medium Qwen3.8-Flash-Next and Qwen3.8-27B prompts) directly from upstream Apache-2.0 FlashInfer C++ headers (`qwen4_fi_attn.cu` auto-adapts across FlashInfer `0.6.x`–`0.7.5+` via C++17 SFINAE), clone FlashInfer with its pinned `3rdparty/cccl` submodule:

```bash
git clone --depth 1 https://github.com/flashinfer-ai/flashinfer.git /workspace/fi_src
git -C /workspace/fi_src submodule update --init --depth 1 3rdparty/cccl
export FLASHINFER_HOME=/workspace/fi_src
```

Operators who also want optional local source-built FlashInfer CuTe-DSL `fi_gdn` AOT objects can export them locally from FlashInfer source per [`crates/spark-model/csrc/fi_gdn/NOTICE`](../crates/spark-model/csrc/fi_gdn/NOTICE) (Section 7 below).

### Step 3: Build Native `sm_120a` or `sm_103a` Binary

`atlas-kernels` uses hardware inheritance (`kernels/rtx6000/HARDWARE.toml` and `kernels/gb300/HARDWARE.toml` both inherit `extends = "gb10"`), compiling all 408 `.cu` kernels + 5 `spark-storage` kernels directly to native PTX and AOT CUBINs:

```bash
export PATH=/usr/local/cuda/bin:$PATH

# Discrete SM120 (RTX 5090 / RTX PRO 5000 / RTX PRO 6000) — all models:
ATLAS_TARGET_HW=rtx6000 cargo build --release --bin flux-server

# Fast dual-model build (Flash-Next + 27B NVFP4 only):
ATLAS_TARGET_HW=rtx6000 ATLAS_TARGET_MODEL="qwen3.8-flash-next,qwen3.8-27b" ATLAS_TARGET_QUANT=nvfp4 \
  cargo build --release --bin flux-server

# Blackwell Ultra SM103 (B300 / GB300):
ATLAS_TARGET_HW=gb300 cargo build --release --bin flux-server
```

> **Cross-CPU Portability Note:** `.cargo/config.toml` keeps portable CPU defaults so binaries transfer cleanly across `x86_64` and `aarch64` hosts (e.g., Intel Xeon Emerald Rapids and AMD EPYC Genoa). For single-host local builds where maximum host SIMD tuning is desired, opt in via `RUSTFLAGS="-C target-cpu=native"` (avoid `-C target-cpu=native` for redistributable binaries or containers).

Supported `ATLAS_TARGET_HW` / `FLUX_TARGET_HW` aliases: `rtx6000` / `sm120` -> `sm_120a`, `gb10` / `sm121` -> `sm_121a`, `gb300` / `sm103` -> `sm_103a`.

### Step 4: Container Build (Pre-Packaged `linux/amd64` Image)

```bash
docker build --platform linux/amd64 \
  -f docker/sm120/Dockerfile \
  --build-arg FLUX_TARGET_HW=rtx6000 \
  --build-arg FLUX_TARGET_MODEL='*' \
  --build-arg FLUX_TARGET_QUANT='*' \
  -t ghcr.io/theapathy/flux-engine:sm120 .
```

See [docker/sm120/README.md](../docker/sm120/README.md) for cross-building from `arm64` via Buildx and running single-GPU or multi-GPU containers.

---

## 3. Model Preparation (`Qwen3.8-Flash-Next` PLE Offload)

`Qwen3.8-Flash-Next-NVFP4` contains ~25 GiB of per-layer embedding (PLE) tables. On a 96 GB discrete GPU (`RTX PRO 6000`), offloading PLE shards to host RAM (`/dev/shm`) drops resident VRAM from ~99 GiB to **~73.5 GiB**, leaving ~22+ GiB free for KV cache, DFlash2, and FP8 prefill weight caches.

If your checkpoint does not already have a `ple-offload/` directory:

```bash
# Lossless NVFP4 passthrough repack (~45 seconds, preserves original shards):
python3 repro-kit/scripts/repack_qwen38_flash_next_ple_nvfp4.py \
  /workspace/models/Qwen3.8-Flash-Next-NVFP4 \
  --mode offload --keep-source-shards
```

When launched with `--ple-ram` (via `flux-autodetect`), Flux Engine stages `ple-offload/` into `/dev/shm/flux-ple-offload/<hash>` (~12s from NVMe) and automatically engages the **zero-syscall `mmap` PLE reader** on `tmpfs` (bypassing `pread` / `io_uring` completely).

---

## 4. Recommended Launch Commands

> **Network Security Note:** The examples below bind to loopback (`--bind 127.0.0.1`) by default. When exposing the server on a LAN or non-loopback interface (`--bind 0.0.0.0`), always pair it with `--api-key "$FLUX_API_KEY"` (or a reverse proxy enforcing authentication).

### 4.1 `Qwen3.8-Flash-Next-NVFP4` — Interactive & Max-Prefill (`C=1..16`, DFlash2 + BF16 KV)

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}

FLUX_SERVER_BIN=./target/release/flux-server \
  python3 repro-kit/flux-autodetect serve \
    /workspace/models/Qwen3.8-Flash-Next-NVFP4 \
    --ple-ram --kv-cache-dtype bf16 --kv-pool-tokens 131072 \
    --max-prefill-tokens 16384 --concurrency 16 \
    --port 8000 --bind 127.0.0.1
```

### 4.2 `Qwen3.8-Flash-Next-NVFP4` — High-Concurrency Batched Serving (`C=24..64` DFlash2)

On discrete SM120 (`96 GB`), `flux-autodetect` automatically keeps `--kv-cache-dtype bf16` and `ATLAS_QWEN4_VERIFY_GRAPH=1` even under `--ple-ram` and `C > 16`, unlocking `gqa12_ragged` and single-frame `256`-row batched verify CUDA graphs across **`C=1..64` DFlash2** (`32×K8=256`, `48×K5=240`, `64×K4=256`), while auto-sizing the lossless drafter context ring (`2048` for `C=17..32`, `1024` for `C=33..48`, `256` for `C=49..64`) to fit cleanly in `96 GB` VRAM:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}

# C=24..48 DFlash2 (256-row batched verify CUDA graphs, BF16 KV):
FLUX_SERVER_BIN=./target/release/flux-server \
FLUX_SPEC_MODE=dflash \
FLUX_DRAFT_MODEL=/workspace/models/Qwen3.8-Flash-Next-DFlash2 \
ATLAS_SSM_DECODE_RING=0 \
ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_WEIGHT_CACHE=0 \
ATLAS_QWEN4_COHORT_ADAPTIVE_K=0 \
ATLAS_QWEN4_VERIFY_GRAPH=1 \
  python3 repro-kit/flux-autodetect serve \
    /workspace/models/Qwen3.8-Flash-Next-NVFP4 \
    --ple-ram --max-seq-len 4096 --max-prefill-tokens 2048 \
    --concurrency 48 --kv-cache-dtype bf16 \
    --speculative-cuda-headroom-mb 1024 --decode-gap-target-ms 0 \
    --port 8000 --bind 127.0.0.1

# C=64 DFlash2 (K=4, 256-row batched verify CUDA graph, auto window=256 + PCUM=0):
FLUX_SERVER_BIN=./target/release/flux-server \
FLUX_SPEC_MODE=dflash \
FLUX_DRAFT_MODEL=/workspace/models/Qwen3.8-Flash-Next-DFlash2 \
ATLAS_SSM_DECODE_RING=0 \
ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_WEIGHT_CACHE=0 \
ATLAS_QWEN4_COHORT_ADAPTIVE_K=0 \
ATLAS_QWEN4_VERIFY_GRAPH=1 \
  python3 repro-kit/flux-autodetect serve \
    /workspace/models/Qwen3.8-Flash-Next-NVFP4 \
    --ple-ram --max-seq-len 2048 --max-prefill-tokens 2048 \
    --concurrency 64 --kv-cache-dtype bf16 --kv-pool-tokens 32768 \
    --speculative-cuda-headroom-mb 512 --decode-gap-target-ms 0 \
    --port 8000 --bind 127.0.0.1
```

For opt-in ultra-fast Flash-Next prefill (`W4A4` NVFP4×NVFP4 MoE + `mHC` NVFP4 2-CTA), pass `ATLAS_QWEN4_PREFILL_MOE_W4A4=all ATLAS_QWEN4_PREFILL_MHC_NVFP4=1` (experimental opt-in; default recipe stays calibrated exact-route `W4A8` `V31_WS` prefill — `5/5` needles verified in §6.1 — with lossless greedy speculative verification).

For plain target-only exact decode (disabling speculation):

```bash
FLUX_SERVER_BIN=./target/release/flux-server \
  python3 repro-kit/flux-autodetect serve \
    /workspace/models/Qwen3.8-Flash-Next-NVFP4 \
    --ple-ram --no-speculative --concurrency 1 \
    --port 8000 --bind 127.0.0.1
```

### 4.3 `Qwen3.8-27B-NVFP4` (`C=1` DFlash2 & `C=1..64` Multi-Seq Piecewise CUDA Graphs)

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}

# C=1 Interactive + DFlash2 (NVFP4 drafter + CUDA-graphed noise pass):
FLUX_SERVER_BIN=./target/release/flux-server \
  python3 repro-kit/flux-autodetect serve \
    /workspace/models/Qwen3.8-27B-NVFP4 \
    --dflash-quantization nvfp4 --dflash-draft-sample-method greedy \
    --port 8000 --bind 127.0.0.1

# C=1..64 High-Concurrency Plain Decode (piecewise CUDA graphs + merged gate/up cuBLASLt NVFP4 FFN):
FLUX_SERVER_BIN=./target/release/flux-server \
FLUX_SPEC_MODE=off \
FLUX_CONCURRENCY=64 \
ATLAS_MULTISEQ_GRAPHS=1 \
  python3 repro-kit/flux-autodetect serve \
    /workspace/models/Qwen3.8-27B-NVFP4 \
    --max-seq-len 20480 --kv-pool-tokens 131072 \
    --port 8000 --bind 127.0.0.1
```

### 4.4 Single-Host Multi-GPU (`2x` / `4x` `TP + EP`)

`flux-autodetect` automatically detects multi-GPU discrete hosts (e.g., `2x RTX PRO 5000 72GB` or `2x/4x RTX PRO 6000 96GB`), sets `NCCL_P2P_LEVEL=SYS` (or NVLink on NVSwitch/B300), preserves `LL128` for intra-node collectives, and generates per-rank `--gpu-ordinal` / `--world-size` commands on `127.0.0.1:29500`:

```bash
# Inspect the auto-generated multi-GPU launch plan before starting:
python3 repro-kit/flux-autodetect inspect --model /workspace/models/Qwen3.8-Flash-Next-NVFP4
```

---

## 5. Key SM120 / SM103 Performance Levers

Most of these auto-engage when `active_sm_count() > 64` or `CC == (10, 3)`:

| Knob | Default on SM120 (`188 SMs`) | Effect |
|---|---|---|
| `ATLAS_PLE_MMAP` | Auto-ON on `tmpfs` (`/dev/shm`) | Zero-syscall `mmap` PLE reads instead of 262K `pread()` calls (`41.9 ms -> 13.7–17.5 ms` per 16K chunk). |
| `ATLAS_QWEN4_COALESCED_SM_TILES` | Auto-ON (`sm_count > 64`) | Selects higher-CTA NVFP4 coalesced tiles (`N8Sk8`/`N16Sk4`/`N32`) so narrow-N projections (`ssm.out`, `attn.oproj`, `hc.up`) saturate 188 SMs. |
| `ATLAS_QWEN4_GDN_SPLIT_V` | Auto-ON (`C=1`, `sm_count > 64`) | Splits each GDN value head's `128x128` H-state across 4 V-column CTAs (`48 -> 192 CTAs`), cutting 36-layer GDN decode to `0.30 ms`. |
| `GQA12_MAX_PARTS` (built-in) | `64` partitions (`sm_count > 64`, `48` on GB10) | Scales ragged GQA-12 attention decode from 96 to 128 CTAs on SM120's 2 KV heads. |
| `ATLAS_QWEN4_MOE_DECODE_M16` | Auto-ON (`rows < 96`, `sm_count > 64`) | 4-CTA/SM (`752` resident CTAs) MoE verify for `C=1..12` DFlash2 and low-row frames (`+7%` to `+12%`). |
| `ATLAS_QWEN4_PREFILL_MOE_V31_WS` | Auto-ON (`rows >= 33`, `grid=188`, `sm_count > 64`) | 16-warp persistent warp-specialized MoE kernel delivering both peak `C=8..64` verify and peak 16K/32K exact `W4A8` prefill. |
| `ATLAS_QWEN4_MOE_THIN_SK_MAX_ROWS` | Auto `32` (`sm_count > 64`) | Thin split-K MoE verify for `rows <= 32` (`C=1..4` DFlash2); hands off to `V31_WS` at `rows >= 33`. |
| `ATLAS_QWEN4_COHORT_ADAPTIVE_K` | `0` in `nvidia-blackwell-sm120.env` | Keeps full `K=8` (`C=24/32`), `K=5` (`C=48`), and `K=4` (`C=64`) single-frame `256`-row speculative verify lanes active on SM120. |
| `ATLAS_QWEN4_PREFILL_GDN_FI` | `2` (FP16 `nf8_strip`, `strip=4096`) | FlashInfer CuTe-DSL chunked delta-rule GDN prefill (`conv_v3` + `conv_v4_strip`) tuned for SM120's 96 MiB L2 cache. |
| `ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_WEIGHT_CACHE` | `1` (`RESERVE_GB=1` on 96 GB discrete) | Caches all `120/120` dequantized FP8 projection weights (`2,550 MiB`) cleanly above discrete VRAM stop line. |
| `ATLAS_CUBLASLT_NVFP4` | Auto-ON on SM103 & SM120 (when FI FFN absent) | Native CUDA 13.0 `cuBLASLt` `VEC16_UE4M3` NVFP4 Tensor Core GEMM for `Qwen3.8-27B` prefill & wide `C=32/64` multi-seq decode. |
| `ATLAS_GDN_PREFILL_FI` | Auto-ON (`rows > 32`, `sm_count > 64`) | FlashInfer CuTe-DSL BF16 chunked GDN prefill for `Qwen3.8-27B` (`48` SSM layers, `2.45x` faster 8K prefill). |
| `ATLAS_GATEUP_N32=0` + coalesced `M<=16` | Auto-ON (`sm_count > 64`) | Single-launch BF16×BF16 MMA intra-CTA warp-split-K fused `gate+up+silu` (`v2_n16_sk4_m16_gateup_silu`, 1,088 CTAs) + `down`/`QKV`/`SSM`/`lm_head` (`M<=16`), plus 3-CTA/SM streaming (`__ldcs`/`__stcs`) multi-seq GDN (`-3.04 ms/step` on `27B C=1` DFlash2, `+50–103 tok/s` across `27B C=8..64`). |
| `ATLAS_WY17_LAZY_COMMIT` | `1` in `nvidia-blackwell-sm120.env` | Deferred WY17 GDN commit on `Qwen3.8-27B` DFlash2 (`+22–66 tok/s` on `C=1`). |
| `ATLAS_DFLASH_BATCHED_VERIFY` | Auto-ON via `flux-autodetect` for `27B C>=2` (opt-in `=1` on raw `flux-server`) | Cross-sequence batched flat DFlash2 verify (`C=2..16`, up to `128` rows/pass) with per-slot `gdn_lazy_retain` lanes, hoisted 48L SSM (`gdn_f32_sequence_lazyfinal_multi` at `C>=4` / Split-V=4 at `C<4`), hoisted 16L Attn QKV/O/FFN, `cuBLASLt NVFP4` `lm_head_batched` (`LM_CHUNK=128`, 1-launch `argmax_bf16_rows`), 1-launch `batched_embed`, and post-thinking (`--disable-thinking` / post-`</think>`) masked-argmax support. |
| `ATLAS_DFLASH_BATCHED_VERIFY_GRAPH` | Auto-ON (`sm_count > 64`), OFF on GB10 | Full-step CUDA graph capture of `decode_verify_dflash_batched` (embedding + 64 layers + final norm + `lm_head_batched` + `argmax_bf16_rows`) over a fixed-address pre-staged metadata slab. |
| `ATLAS_QWEN4_PLE_UNIFIED_GRAPH` | Default ON (`1`) | Unified 48-layer `Qwen3.8-Flash-Next` verify CUDA graph overlapping `mmap` PLE `dequant + key_proj + value_proj` on the secondary stream with Layer 0, combined with `sk_dn=1` + fused `shared_gate` in `qwen4_moe_thin_sk_v33`, high-SM `N8Sk8M32V2` mHC routing, `N=512` Split-K router, inter-layer `mlp_hyper` deferral, and fused `scatter + deinterleave_qg`. |
| `ATLAS_DFLASH_STAGE1_GRAPH` | Default ON (`1`) | Bucketed `1..=8`-row Stage-1 KV CUDA graphs, coalesced context-capture D2D runs, suffix-only position upload, and single `216 B` Stage-2 bundle H2D in `dflash_head`. |
| `ATLAS_QWEN4_RELEASE_UNUSED` | `1` in `nvidia-blackwell-sm120.env` | Reclaims dead `mtp.*` weights (`+1.52–4.80 GiB`) and unused BF16 `lm_head` (`+1.18 GiB`) under `Qwen4ReleaseMode::Dflash`, unlocking `1.5M–2.0M` `1×` 96 GB NVFP4 KV pools (`FLUX_MODE=1m-nvfp4`, `C=16` YaRN4) and `6M–16M` multi-GPU (`2×/4×/8×` TP/EP/PP) pooled KV budgets. |
| `ATLAS_DFLASH_NOISE_ONLY` | `1` (set by `flux-autodetect`) | Restricts DFlash2 drafter layer math to noise tokens only (avoids recomputing full context per step). |

---

## 6. Verified Live SM120 Performance Reference (`1x RTX PRO 6000 Blackwell WS`, `600 W`)

Measured on `1x NVIDIA RTX PRO 6000 Blackwell Workstation Edition` (`SM12.0`, `188 SMs`, `3,090 MHz` boost, `96 GB GDDR7` @ `14,001 MHz` / `1,792 GB/s`, `96 MiB L2`, `600 W` TDP, PCIe Gen5 x16, CUDA `13.0`, Ubuntu `24.04 x86_64`).

> **Build & Edition Notes:**
> - **Peak Prefill (`fi_gdn` AOT):** The headline cold-prefill tables below (§6.1 and §6.3) were measured with locally source-built FlashInfer CuTe-DSL `fi_gdn` AOT objects (`rebuild-aot.sh`) + `FLASHINFER_HOME` enabled — the exact configuration built into the published `ghcr.io/theapathy/flux-engine:sm120` (`:v1.0.3` `linux/amd64`) container image. Pure-source builds without `fi_gdn` AOT compile and run cleanly via the native CUDA GDN fallback at lower prefill speed.
> - **Workstation vs. Server Edition:** Passive-cooled `RTX PRO 6000 Blackwell Server Edition` SKUs cap core clocks at `2,430 MHz` and GDDR7 at `12,481 MHz` (`1,600 GB/s`), running ~8–12% lower on decode while achieving `15,151 tok/s` (`532.5 ms`) 8K `27B` prefill and `32,674 tok/s` (`495.9 ms`) 16K Flash-Next `W4A4` prefill.

### 6.1 `Qwen3.8-Flash-Next-NVFP4` — Cold Prefill (`1x RTX PRO 6000 WS`, `600 W`, BF16 KV, `mmap` PLE)

| Prompt Length | Default Exact `W4A8` (`5/5` Needles, Server / Client TTFT) | Opt-In `W4A4=all` + `MHC_NVFP4=1` (Server TTFT) |
|---|---:|---:|
| `4K` (`4,016` tok) | **`168.6 ms` (`23,820 tok/s`)** / `197.1 ms` (`20,375 tok/s`) | `151.2 ms` (`26,475 tok/s`) |
| `8K` (`8,067` tok) | **`275.5 ms` (`29,281 tok/s`)** / `293.7 ms` (`27,468 tok/s`) | `244.3 ms` (`33,013 tok/s`) |
| **`16K` (`16,206` tok)** | **`525.0 ms` (`30,869 tok/s`)** / `538.4 ms` (`30,102 tok/s`) | `472.2 ms` (`34,308 tok/s`) |
| **`32K` (`32,713` tok)** | **`1,076.3 ms` (`30,394 tok/s`)** / `1,136.3 ms` (`28,789 tok/s`) | `973.4 ms` (`33,601 tok/s`) |

### 6.2 `Qwen3.8-Flash-Next-NVFP4` — Highest Verified Concurrency `C=1..64` (`1x RTX PRO 6000 WS`, `600 W`, BF16 KV)

| Concurrency | Best Mode | Steady-State Aggregate | Per-Request Steady | End-to-End Wall Aggregate | Peak Power |
|---:|---|---:|---:|---:|---:|
| **`C=1`** (`arith` / `code_edit` / `code` / `chat_reason` / `prose`) | **DFlash2 (`W=8`, `FP8` default `lm_head`, `11.66 ms/round`, batched mRoPE+GQA12+`PCUM_THRESH=0`)** | **`661.1` / `507.0` / `370.1` / `330.9` / `202.5 tok/s`** | `661.1` / `507.0` / `370.1` / `330.9` / `202.5 tok/s` | `587.1` / `478.0` / `353.9` / `315.8` / `198.3 tok/s` | `447.3 W` |
| **`C=1` (opt-in `NVFP4` `lm_head`)** | **DFlash2 (`W=8`, `ATLAS_QWEN4_TARGET_LMHEAD_MODE=nvfp4`, `11.29–11.40 ms/round`)** | **`678.3` / `550.5` / `358.8` / `316.2` / `208.4 tok/s`** | `678.3` / `550.5` / `358.8` / `316.2` / `208.4 tok/s` | `606.2` / `505.4` / `343.9` / `302.6` / `204.0 tok/s` | — |
| **`C=2`** | DFlash2 (`W=8`, `256`-row graph) | **`432.7 tok/s`** | `216.4 tok/s` | `355.5 tok/s` | `396.0 W` |
| **`C=4`** | DFlash2 (`W=8`, `256`-row graph) | **`672.0 tok/s`** | `168.0 tok/s` | `475.9 tok/s` | `434.8 W` |
| **`C=8`** | **DFlash2 (`W=8`, `256`-row graph, `GQA12_RAGGED` gated)** | **`1,133.6 tok/s`** (mixed) | `141.7 tok/s` (mixed) | **`712.5 tok/s`** (mixed) / **`2,357.0 tok/s`** (arith wall, `294.6/req`) | `525.2 W` |
| **`C=16`** | **DFlash2 (`W=8`, `256`-row graph, `GQA12_RAGGED` gated)** | **`1,886.0 tok/s`** (mixed) | `117.9 tok/s` (mixed) | **`1,399.2 tok/s`** (mixed) / **`3,687.1 tok/s`** (arith wall, `230.4/req`) | `507.8 W` |
| **`C=24`** | DFlash2 (`W=8`, `256`-row graph) | **`2,356.0 tok/s`** | `98.2 tok/s` | `1,693.2 tok/s` | `527.2 W` |
| **`C=32`** | DFlash2 (`W=8`, `256`-row graph) | **`2,765.3 tok/s`** | `86.4 tok/s` | `1,882.0 tok/s` | `552.6 W` |
| **`C=48`** | **DFlash2 (`K=5`, `win=1024`, `rows=240`)** | **`3,068.6 tok/s`** | **`63.9 tok/s`** | `1,686.7 tok/s` | `544.3 W` |
| **`C=64`** | **DFlash2 (`K=4`, `win=256`, `PCUM_THRESH=0`, `rows=256`)** | **`3,405.7 tok/s`** | **`53.2 tok/s`** | **`2,023.5 tok/s`** | **`562.9 W`** |

### 6.3 `Qwen3.8-27B-NVFP4` — Prefill & Highest Verified Concurrency `C=1..64` (`1x RTX PRO 6000 WS`, `600 W`)

- **Cold Prefill (`cuBLASLt NVFP4` merged gate/up + `L2_STRIP=1024` + fused residual-norm-quant + `cuBLASLt NVFP4` projections + FlashInfer GDN + `fi_attn`, `3/3` exact needles):**
  - `4K` (`4,012` tok): **`526.7 ms` server TTFT (`7,617 tok/s`)** (`477.1 ms` / `8,390 tok/s` warm)
  - **`8K` (`8,067` tok):** **`496.5 ms` server TTFT (`16,248 tok/s` · `15,727 tok/s` client)** — **`+65%`** over `9,831 tok/s` baseline (`+7.9x` over initial `2,054 tok/s`)
  - **`16K` (`16,203` tok):** **`1,079.3 ms` server TTFT (`15,012 tok/s` · `14,581 tok/s` client)** — **`+70%`** over `8,856 tok/s` baseline
- **Highest Verified Concurrency `C=1..64` (Hoisted Cross-Sequence Batched `DFlash2` Verify Graph for `C=1..8`; Plain Multi-Seq Piecewise Graphs + Coalesced `M<=16` / `cuBLASLt NVFP4` `M>=24` + PTX dynamic quant + 3-CTA/SM Streaming GDN for `C=16..64`):**

| Concurrency | Best Mode | Steady-State Aggregate | Per-Request Steady | End-to-End Wall Aggregate | Peak Power |
|---:|---|---:|---:|---:|---:|
| **`C=1`** (`arith` / `code_edit` / `code` / `prose`) | **DFlash2 (`W=8`, coalesced `v2_n16_sk4` + `WY17`)** | **`497.1` / `312.3` / `300.0` / `266.0 tok/s`** | `497.1` / `312.3` / `300.0` / `266.0 tok/s` | `461.3` / `299.7` / `286.9` / `258.7 tok/s` | — |
| **`C=2`** | **Hoisted Batched DFlash2 (`W=8`, `forward=15.97 ms`, `20.1 ms/round`, graphed)** | — *(see wall)* | — | **`590.1 tok/s`** (arith) / **`575.5 tok/s`** (mixed) | `568.0 W` |
| **`C=4`** | **Hoisted Batched DFlash2 (`W=8`, `forward=18.47 ms`, `23.5 ms/round`, graphed)** | — *(see wall)* | — | **`600.1 tok/s`** (arith) / **`514.3 tok/s`** (mixed) | `568.0 W` |
| **`C=8`** | **Hoisted Batched DFlash2 (`W=8`, `forward=21.09 ms`, `27.5 ms/round`, graphed)** | **`1,118.9 tok/s`** (mixed; plain `561.5`) | **`139.9 tok/s`** | **`1,073.8 tok/s`** (arith) / **`711.4 tok/s`** (mixed) | `568.0 W` |
| **`C=16`** | Plain Multi-Seq (`GRAPHS=1`, coalesced `M<=16` + 3-CTA GDN) | **`932.5 tok/s`** | **`58.3 tok/s`** | `802.3 tok/s` | **`605.5 W`** |
| **`C=24`** | Plain Multi-Seq (`GRAPHS=1`, `cuBLASLt NVFP4` `M>=24` + PTX absmax + 3-CTA GDN) | **`1,161.6 tok/s`** | **`48.4 tok/s`** | `957.0 tok/s` | `431.3 W` |
| **`C=32`** | Plain Multi-Seq (`GRAPHS=1`, `cuBLASLt NVFP4` `M>=24` + PTX absmax + 3-CTA GDN) | **`1,394.4 tok/s`** | **`43.6 tok/s`** | `1,102.6 tok/s` | `437.1 W` |
| **`C=48`** | Plain Multi-Seq (`GRAPHS=1`, `cuBLASLt NVFP4` `M>=24` + PTX absmax + 3-CTA GDN) | **`1,724.8 tok/s`** | **`35.9 tok/s`** | `1,300.9 tok/s` | `452.9 W` |
| **`C=64`** | Plain Multi-Seq (`GRAPHS=1`, `cuBLASLt NVFP4` `M>=24` + PTX absmax + 3-CTA GDN) | **`1,974.8 tok/s`** | **`30.9 tok/s`** | **`1,479.8 tok/s`** | `454.3 W` |

*(Note: For `C=2`/`C=4` batched DFlash2, `C×8 / round_ms` gives a 100%-acceptance upper bound only; the table reports strictly measured wall-clock throughput and measured mixed steady-state `1,118.9 tok/s` at `C=8`.)*

### 6.4 `Qwen3.8-Flash-Next-NVFP4` — `1M` YaRN4 Long-Context Recall & `1x`–`8x` Pooled NVFP4 KV Capacity (`600 W` SM120)

- **`1,011,019`-Token YaRN4 Passkey Recall (`1x RTX PRO 6000 WS`, `FLUX_MODE=1m-nvfp4`, `hp=auto` `4xBF16 + 8xNVFP4`, `W=8192`, `588 W` peak):**
  - Depth `0.1`: `1,011,019` prompt tokens · **`82.07 s` TTFT (`12,318.8 tok/s`)** · exact needle recall (**PASS**)
  - Depth `0.5`: `1,011,019` prompt tokens · **`82.12 s` TTFT (`12,312.0 tok/s`)** · exact needle recall (**PASS**)
  - Depth `0.9`: `1,011,019` prompt tokens · **`82.18 s` TTFT (`12,302.8 tok/s`)** · exact needle recall (**PASS**)
- **Resident KV Pool Capacity (`1x 96 GB` live-verified; `2x`/`4x`/`8x` planner-calculated):**
  - `1x 96 GB` live-verified (`hp=auto`, `4xBF16 + 8xNVFP4`): **`1,458,544` (`1.46M`) max KV tokens** (`budget_blocks=91,159`, `14.11 KiB/tok` total incl. QSA indexer)
  - `1x 96 GB` live-verified (`hp=0`, `12xNVFP4`, `C=4`): **`2,304,528` (`2.30M`) max KV tokens** (`budget_blocks=144,033`; `2,097,232` = `2.0M` tokens allocate **`13.50 GiB` raw NVFP4 KV** @ `6.75 KiB/tok` or **`16.72 GiB` total incl. QSA indexer** @ `8.36 KiB/tok`)
  - `2x / 4x / 8x 96 GB` **planner-calculated** via `flux-autodetect` (not yet live-benchmarked on multi-GPU SM120 hardware): **`6.3M` (`2x TP2+EP2`) -> `8.4M` (`4x TP4+EP4`) -> `16.8M` (`4x/8x PP`)** pooled NVFP4 KV tokens.

---

## 7. Architecture Notes: SM103 (`B300`/`GB300`) & Adding Future Architectures

### SM103 Phase-A Architecture Support
- **ISA difference:** `sm_103a` (datacenter Blackwell Ultra) replaces workstation `mma.sync.aligned.block_scale` (`sm_120a`/`sm_121a`) with datacenter `tcgen05` tensor-memory instructions, while keeping standard FP8/BF16 `mma.sync` and native CUDA 13.0 `cuBLASLt` `VEC16_UE4M3` (NVFP4) / `VEC32_UE8M0` (MXFP8) Tensor Core GEMMs.
- **Phase-A (implemented):**
  - `kernels/gb300/HARDWARE.toml` (`arch = "sm_103a"`, `extends = "gb10"`, `8000 GB/s`, `288 GB`).
  - All 18 `block_scale` CUDA sources guarded with `#if __CUDA_ARCH__ >= 1200` (unit-`0x7f` FP8 -> standard `e4m3` MMA; MXFP8 `scale_vec::1X` -> exact branchless software post-scale proven 0-ULP; NVFP4 `4X` -> clean trap + `cuBLASLt` `VEC16_UE4M3` dispatch in `dense_ffn.rs`).
  - All 408/408 model kernels + 5/5 storage kernels compile cleanly to `.target sm_103a` PTX and AOT CUBINs with 0 skipped.
- **Phase-B roadmap (future):** CUTLASS 4.x `tcgen05.mma.cta_group::2` grouped MoE persistent kernels (`NVFP4` & `MXFP8`) to push SM103 MoE prefill to full 8 TB/s HBM3e / Blackwell Ultra Tensor Core throughput.

### Optional Local Source-Built `fi_gdn` AOT Objects (Any SM Architecture)
By default, Flux Engine ships **zero** precompiled `fi_gdn/*.o` binary objects and falls back cleanly to its native CUDA GDN prefill kernels (`qwen4_gdn_fi_glue.cu`). Operators who want optional local FlashInfer CuTe-DSL GDN prefill acceleration on `sm_121*`, `sm_120a`, or a new architecture (`sm_103a`) can export the three `atlas_fi_gdn_<arch>_<cpu>_prefill{,_f16,_f16_nf8}.o` objects and matching headers locally from FlashInfer's CuTe-DSL `delta_rule_sm120.py` using `repro-kit/source-closure/v0.9.13-r8/gdn-aot/rebuild-aot.sh` (see [`crates/spark-model/csrc/fi_gdn/NOTICE`](../crates/spark-model/csrc/fi_gdn/NOTICE)), then rebuild `spark-model` so `crates/spark-model/build.rs` auto-detects the trio and enables `cfg(atlas_fi_gdn)`.

### Cross-Generation GPU Architecture Support Matrix

Flux Engine's `HARDWARE.toml` inheritance (`extends = "gb10"`), `GpuBackend` trait, dynamic `active_sm_count()`, and `compute_capability()` guards isolate hardware-specific fast paths cleanly:

| GPU Family | Representative SKUs | Status | Key Technical Notes / Roadmap |
|---|---|---|---|
| **Blackwell Consumer/Pro (`SM120`)** | RTX 5060–5090, RTX PRO 4000/5000/6000 | **Production & Certified** | Native `sm_120a` PTX/CUBINs, native CUDA GDN fallback + optional source-built `FLASHINFER_HOME` `fi_attn`/`fi_gdn`, `cuBLASLt` NVFP4/MXFP8, discrete VRAM budgeting & `/dev/shm` `mmap` PLE. |
| **DGX Spark (`SM121`)** | GB10 (UMA 128 GB) | **Production & Certified** | Primary `sm_121a`/`sm_121f` reference target with UMA budgeting, native CUDA GDN fallback + optional local `fi_gdn`/`fi_attn`, and 1x–4x RoCEv2 scaling. |
| **Blackwell Datacenter (`SM100`) & Ultra (`SM103`)** | B200, GB200 (`SM100`); B300, GB300 (`SM103`) | **SM103 Phase-A builds cleanly**; SM100 autodetect stub | Phase-A uses guarded warp-MMA fallbacks + `cuBLASLt` NVFP4/MXFP8; Phase-B adds CUTLASS 4.x `tcgen05.mma.cta_group::2` persistent MoE + optional local `fi_gdn` AOT. |
| **Hopper (`SM90`)** | H100, H200, GH200 | **Documented Roadmap** (recipe stub) | Requires `sm_90a` `HARDWARE.toml`, guarding 5 Blackwell `cvt.*e2m1x2` quantizers, `CC >= 10` guard on `cuBLASLt` NVFP4 (landed), and W4A8/W4A16 via FP8 MMA + `cuBLASLt` FP8 (Phase-B: WGMMA+TMA MoE). |
| **Ada Lovelace (`SM89`)** | L40S, RTX 4090, RTX 6000 Ada | **Documented Roadmap** | Native FP8 MMA works; additionally requires guarding SM90+ thread-block cluster/DSMEM intrinsics in `qwen4_hc_combine_gemm.cu` and `CC >= 9` PDL gate (landed). |
| **Ampere (`SM80` / `SM86`)** | A100, A6000, RTX 3090 | **Documented Roadmap** | No hardware FP8/FP4: requires `#if __CUDA_ARCH__ < 890` software-dequant BF16 `mma.sync` fallbacks for FP8 MMA/convert kernels + BF16 `cuBLASLt`. |
| **Apple Silicon (`Metal`)** | M2 / M3 / M4 | **Vertical Slice Implemented** | `MetalGpuBackend` + 30 `.metal` kernels + `metal3.1` target; expanding to full server requires MPS/simdgroup GEMM + MoE/NVFP4 kernels. |
| **AMD ROCm (`HIP`) / Intel (`oneAPI`)** | MI300X / MI350X; Gaudi / Arc | **Trait Seams Only** (`Vendor` enum) | Requires HIP/MFMA wavefront-64 or SYCL/XMX backend & kernel port behind `ComputeTarget` / `GpuBackend` / `CommBackend`. |

### 4-Step Checklist to Add Any New NVIDIA SM Architecture
1. **Hardware config:** Add `kernels/<hw>/HARDWARE.toml` with `arch = "sm_XXXa"`, `extends = "gb10"`, memory bandwidth, and VRAM specs.
2. **Build alias:** Add the target alias in `crates/atlas-kernels/build_hw.rs` and (if needed) AOT object selection in `crates/spark-model/build.rs`.
3. **Launcher detection:** Add the compute capability (`cc.startswith("X.Y")`) and container tag in `repro-kit/flux-autodetect` + unit tests in `repro-kit/tests/test_sm120_sm103_discrete.py`.
4. **Verify:** Run `FLUX_TARGET_HW=<hw> cargo test -p atlas-kernels`, check `ptxas` spill/stack output, verify `cuBLASLt` NVFP4/MXFP8 parity, and benchmark `C=1` decode, `C=16` concurrency, and `16K+` cold prefill.

---

## 8. Troubleshooting Common Bringup Pitfalls

| Symptom | Root Cause | Fix |
|---|---|---|
| `nvcc not found` during `cargo build` | `/usr/local/cuda/bin` missing from `PATH` | `export PATH=/usr/local/cuda/bin:$PATH` |
| `apt-get install libnccl2` fails (held packages) | Base NVIDIA image pins `libnccl2` | Pass `--allow-change-held-packages` |
| `QWEN4_PLE_PREAD_LANES ... reason=io_uring_unavailable` | Docker seccomp blocks `io_uring_setup` | Stage PLE to `/dev/shm` (`--ple-ram`); Flux auto-engages zero-syscall `mmap` |
| `cuBLASLt tuneAllocW failed: status 2` or `cuGraphInstantiateWithFlags: status 2` | KV cache claimed >92% of discrete VRAM leaving no workspace for graphs/cuBLASLt | Lower `--gpu-memory-utilization` (e.g. `0.80`–`0.85`), cap `--kv-pool-tokens`, or use `--kv-cache-dtype nvfp4` |
| `ATLAS_QWEN4_PREFILL_QSA_FP8: requires BF16 KV` | Mixed NVFP4 KV selected with `QSA_FP8=qkpv` | Either pass `--kv-cache-dtype bf16` (with `--kv-pool-tokens 131072` on 96 GB) or let `flux-autodetect` choose mixed NVFP4 KV without `QSA_FP8` |
| Slow DFlash2 decode at `>=4K` context (`propose > 50 ms`) | `ATLAS_DFLASH_NOISE_ONLY` not enabled | Use `flux-autodetect` or set `ATLAS_DFLASH_NOISE_ONLY=1` |
| `libnccl.so.2: cannot open shared object file` | Runtime `LD_LIBRARY_PATH` missing CUDA/NCCL dirs | `export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH` |
