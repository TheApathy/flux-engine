<p align="left"><a href="../../README.md"><b>← Flux Engine</b></a> · <a href="../README.md"><b>Benchmarks (<code>bench/</code>)</b></a> · <b><code>qwen38-flash-next-gb10</code></b></p>

# Qwen3.8-Flash-Next single-stream prefill & speculative decode on GB10 — reproduction

Reproduces measured single-stream prefill, long-context needle retrieval, and
lossless `K=5` speculative decode profiles for **Qwen3.8-Flash-Next** (48-layer
hybrid: 36 Gated DeltaNet + 12 Query-Sparse Attention `QSA` layers, 512 routed
+ 1 shared MoE experts top-10, 4-stream `mHC` hyper-connections, and 320M-row
NVFP4 Per-Layer Embedding `PLE` offload) on a single NVIDIA DGX Spark (**GB10**,
`sm_121a`, 128 GB unified LPDDR5X).

Qualification scope: the measurements below belong to the historical named
binaries, recipes, checkpoints and Inferact sidecar runs. They do not qualify
the current integration, its checkpoint-embedded MTP head, or a new container.
The built-in and external heads need separately identified, matched receipts.
For the raw engine CLI, Plain mode omits all speculative selectors and uses
`ATLAS_DISABLE_SPECULATION=1`. There is no `--no-speculative` engine flag.

Without pruning experts, layers, vocabulary, vision tower, or PLE rows, the
certified `unified13` (`v0.9.12-6k-stable`) stack reaches:

- **`6,082.2 tok/s` peak (`6,076.7 tok/s` Wave 9 median / `6,063.5 tok/s` Wave 11 median) single-request prefill at 16K context** (`2.694 s` TTFT), sustaining **`5,830.2 tok/s` peak (`5,819.3` median) at 8K**, **`5,969.6 tok/s` peak (`5,952.6` median) at 32K**, **`5,828.3 tok/s` peak (`5,803.2` median) at 64K**, and **`5,694.4 tok/s` peak (`5,684.0` median) at 128K** (`23.018 s` min / `23.060 s` median TTFT) under the 16K-chunk `v9` profile (`serve.sh`).
- **`5,276.3 tok/s` peak (`5,266.8 tok/s` median) at 8K** and **`4,899.8 tok/s` peak (`4,891.7` median) at 128K** under the low-memory 4K-chunk `v9-w4096` profile (`serve-w4096.sh`), while retaining **`11,907–12,695 MiB` free unified memory** (`+4.2 GiB` extra UMA headroom vs 16K chunks).
- **`57.4 tok/s` client peak (`56.3 tok/s` client median, `57.6 tok/s` server peak / `56.5 tok/s` server median) speculative decode** (`accept_len = 3.02–3.15` tokens/step) using the native 1-layer MoE **Inferact MTP** head at `K=5` (`--speculative --num-drafts 4`), verified **`LOSSLESS: 10 identical, 0 different` (`2,560 / 2,560` greedy tokens bit-for-bit identical to Plain (`ATLAS_DISABLE_SPECULATION=1`) `K=1` decode)** across `0` to `64K` context depths.
- **`5/5 = 100.0% PASS` (`10/10` across two independent certified campaigns) on the 136,576-token (`136,584` active KV tokens) LIL `v0.7.9` `--test-profile estonia` 6-hop needle benchmark**, sustaining **`5,400.4 tok/s` client (`5,428.3 tok/s` server median) prefill** (`25.29 s` avg TTFT) and **`43.8 tok/s` mean / `48.2 tok/s` client peak (`48.6 tok/s` server peak) decode** (`accept_len = 2.88`) with **`8,431–8,847 MiB` free UMA floor** and zero watchdog terminations.

```bash
git clone https://github.com/TheApathy/flux-engine.git
cd flux-engine
git checkout main   # unified13 / v0.9.12-6k-stable (commit ae58bc27efd8)
```

Historical baseline context: on the earlier `origin/perf/qwen38-flash-next`
branch, target-only decode with PLE-boundary segmented CUDA graphs reached
**42.1010 tok/s** median on the 400-token Weschera MinHeap probe (stable output
SHA-256 `f811cdc565dff063074f8bb1d0bb3fd55c8b10f3438c517cf4d59128f23cf790`)
over a 38.3594 tok/s eager control, while serialized prefill ran at ~40.4 tok/s
and row-wise GDN prefill reached 111.8 tok/s. The `unified13` K1–K12 waves
replaced per-token prefill with compact transposed W4A8/NVFP4 MoE (`v28q=2` +
`v17q`), FlashInfer SM121a CuTe-DSL chunked GDN scan (`GDN_FI=2`), 5-CTA/SM
`ldmatrix.x4` QSA sparse attention (`QSA_PIPE=1`), 4-stream `mHC` roofline
fusions (`MHC_ROOF=t128`), 4-ring sector-aligned NVMe PLE DMA, and 5-row
batched target verification for Inferact MTP (`K=5`).

## Hardware and build

GB10 / DGX Spark, coherent LPDDR5X unified memory (`128 GB` physical, `119.7 GiB`
OS-visible), `sm_121a` (`sm_121f`, 48 SMs, 32 MiB L2). Sustained memory
bandwidth is roughly 273 GB/s (`301 GB/s` theoretical peak), so single-token
plain decode (`K=1`) is bandwidth-bound at `31.0–33.7 tok/s`.

**Build the kernels for the right target.** This is the single most common way
to get a wrong binary:

```bash
# nvcc must be ON PATH. cudarc's build script resolves the CUDA version by
# shelling out to a bare `nvcc --version`; it does NOT consult CUDA_HOME. A
# clean checkout has no cached build-script output, so this fails there even
# though it succeeds on a tree that has built before.
export PATH=/usr/local/cuda/bin:$PATH

touch crates/atlas-kernels/build.rs
FLUX_TARGET_MODEL=qwen3.8-flash-next FLUX_TARGET_QUANT=nvfp4 \
  cargo build --release -p spark-server
```

*(Note: `ATLAS_TARGET_MODEL=qwen3.8-flash-next ATLAS_TARGET_QUANT=nvfp4` is
accepted as a 100% compatible build-time alias.)*

Without those two variables the build **silently** defaults to
`qwen3-next-80b-a3b` and produces a binary that will not serve Flash-Next. The
build log line is the check:

```text
atlas-kernels: compiled N kernels for target 0 (gb10, qwen3.8-flash-next, nvfp4)
```

*(or `flux-kernels: compiled N kernels for target 0 (gb10, qwen3.8-flash-next, nvfp4)`).*

If that line names any other model, the binary is wrong. `target/release/spark`
(or `target/release/flux-server`) is overwritten in place by any other build in
the workspace, so verify its embedded target strings or SHA-256 immediately
before launching rather than trusting an earlier build. Every `serve*.sh`
script in this directory runs that check automatically before `exec`.

## Weights and checkpoints

- **Target (Public NVFP4 Checkpoints)**:
  - **Option A — [`JMNI-Labs/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid`](https://huggingface.co/JMNI-Labs/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid)** (`41` shards, `~69.3 GiB` total): Both the 48 backbone layers and the 128 PLE shards (`model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{0..127}.weight`) are already NVFP4-quantized (`U8 [2500012, 80]` + `F8_E4M3 [2500012, 10]`), and shard 41 embeds a built-in NVFP4 `mtp.*` head. Running `scripts/repack_qwen38_flash_next_ple_nvfp4.py` automatically selects **Mode A (Native NVFP4 Passthrough)**, copying the 80 packed bytes + 10 scale bytes per row byte-for-byte into the 91-record / 8192-byte page-aligned `ple-offload/ple-ngram-nvfp4-{shard:03d}.bin` sidecar files (`28,807,528,448` bytes across 128 shards).
  - **Option B — [`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)** (`53` shards): Official NVIDIA ModelOpt NVFP4 backbone (`48` layers) with built-in packed BF16 `mtp.*` expert tensors (`mtp.layers.0.mlp.experts.{gate_up_proj,down_proj}`) and BF16 PLE shards (`~102.4 GB`), which `scripts/repack_qwen38_flash_next_ple_nvfp4.py` quantizes in **Mode B (BF16/FP8 Quantize)** into `ple-offload/` (`28.8 GB`).
  - **Option C — [`Blackfrost-AI/CYBER-FROST-3.8-NVFP4-V2`](https://huggingface.co/Blackfrost-AI/CYBER-FROST-3.8-NVFP4-V2)** (`131` shards): NVFP4 backbone with built-in per-expert BF16 `mtp.*` weights in shard 131 and BF16 PLE shards (`00003..00130`), which `scripts/repack_qwen38_flash_next_ple_nvfp4.py` quantizes in **Mode B** into `ple-offload/`.
  - **What the `ple-offload/` sidecar layout does:** streams the 320M-row Per-Layer Embedding (`PLE`) table from 128 checksummed, 8192-byte page-aligned (`91` records/page) NVFP4 sidecar shards (`28,807,528,448` bytes on NVMe) via 4 `io_uring` DMA rings while keeping all 48 layers, all 512 routed + 1 shared experts, the full 248,077-token runtime vocabulary, and the vision tower resident in unified memory.
  - **Official FP8 comparison checkpoint**: pinned at
    `bcd9f01ddc9cff2316eb84281bebcd5b058bddce` (173 GB, which does not fit single-node
    128 GB UMA without expert pruning).
- **Speculative Drafter (Built-in Checkpoint `mtp.*` Head — No External Download Required)**:
  All three public checkpoints above (`JMNI-Labs/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid`, `nvidia/Qwen3.8-Flash-Next-NVFP4`, and `Blackfrost-AI/CYBER-FROST-3.8-NVFP4-V2`) include the 1-layer full-attention MoE Multi-Token Prediction (`mtp.*`, 512 experts, top-10 routing) head directly inside the checkpoint shards.
  Served out of the box with `--speculative --num-drafts 4 --mtp-vocab 0 --mtp-quantization bf16`
  (proposing 4 drafts per cycle -> target verifies `K=5` rows in one batched forward pass).
  An optional external MTP sidecar directory (`129972269565f7f4f664fdf8dd42268d3bbda9fd`) can also be supplied via `MTP_DIR=/path/to/mtp` (`--mtp-from-path "$MTP_DIR" --mtp-vocab 96000`).

| Speculative / Target Pairing | Draft Width | Verify Batch (`K`) | Coding Decode (tok/s) | Greedy Parity vs `K=1` | Verdict |
|---|---:|---:|---:|---|---|
| **Inferact Native MTP (`v9-w4096` / `k5-decode`)** | `4` | **`5`** | **`56.3` med / `57.4` max** (`57.6` srv) | **`LOSSLESS 10/10` (`2,560/2,560` exact)** | **Promoted production default** |
| **Inferact Native MTP (`v9` `w16384`)** | `4` | **`5`** | **`52.1` med / `53.6` mean / `58.7` max** | **`LOSSLESS 10/10` (`2,560/2,560` exact)** | **Promoted 6K-prefill default** |
| Target-only (no speculative selector, segmented graph) | `0` | `1` | `42.10` (MinHeap 400-tok) / `33.7` (LIL) | Reference oracle (`K=1`) | Exact non-speculative control |
| Dense Qwen3.8 DFlash V3 donor bridge | `15` | `16` | `2.88 – 3.32` | Exact (`0.32/15` accept) | **Rejected** (`K=16` unbatched verify = 418 ms) |
| Public DFlash2 (`incoai`) | `7` / `15` | `8` / `16` | Refused at load | — | **Incompatible** (`hidden_size` 5120 vs 2560) |

## Operating profiles: `v9` (16K-chunk) vs `v9-w4096` (4K-chunk) vs `fast-prefill` vs `k5-lossless-decode`

Four self-verifying launchers in this directory cover the four operating
regimes on GB10. Each script accepts both `FLUX_*` and `ATLAS_*` environment
variables, validates every parameter, and asserts that the binary embeds the
`qwen3.8-flash-next` kernel target and all exported runtime gates before
exec'ing the server:

| Launcher | Mode / Recipe | Prefill Chunk (`WIDE`) | 16K Prefill (tok/s) | 128K Prefill (tok/s) | Coding Decode (tok/s) | Free UMA Floor (`MemAvailable`) | When to use |
|---|---|---:|---:|---:|---:|---:|---|
| [`serve.sh`](serve.sh) | `v9` (`max-prefill` / `w16384`) | `16,384` | **`6,082.2` peak**<br/>(`6,076.7` W9 med) | **`5,694.4` peak**<br/>(`5,684.0` W11 med) | **`52.1` med / `58.7` max**<br/>(`LOSSLESS 10/10`) | `7,072 – 8,450 MiB`<br/>(`8,431–8,847` @ Estonia) | Default flagship for maximum 8K–128K prefill velocity + lossless `K=5` MTP decode |
| [`serve-w4096.sh`](serve-w4096.sh) | `v9-w4096` (`interactive-w4096`) | `4,096` | **`5,176.9` peak**<br/>(`5,164.5` med, `5,276.3` @ 8K) | **`4,899.8` peak**<br/>(`4,891.7` med) | **`56.3` med / `57.4` max**<br/>(`57.6` srv, `LOSSLESS 10/10`) | **`11,907 – 12,695 MiB`** | Interactive chat/IDE serving with `+4.2 GiB` extra free unified memory and zero re-prefill windows |
| [`serve-fast-prefill.sh`](serve-fast-prefill.sh) | `fast-prefill` (`GDN_NORM_FP8=3`) | `16,384` | **Ultra-tier prefill**<br/>(store-warp E4M3 norm) | **Ultra-tier prefill** | Plain `K=1` (no speculative selector) | `8,500+ MiB` | Pure prefill-only batch ingestion, RAG indexing, rubric scoring, or single-token TTFT sweeps |
| [`serve-k5-decode.sh`](serve-k5-decode.sh) | `k5-lossless-decode` (`k5all` + K11-DEC) | `8,192` | `~4,400` (FP8 cache off) | `~4,200` | **`56.3` med / `57.4` max**<br/>(`3.02–3.15` accept len) | **`10,500+ MiB`** | Decode-dominated sessions (`max_tokens >= 512`) maximizing L2/UMA residency for 5-row verify |

### Launch examples

All launchers use the checkpoint's built-in `mtp.*` head (`--mtp-vocab 0`) by default when `MTP_DIR` is omitted, or an external MTP sidecar (`--mtp-from-path "$MTP_DIR" --mtp-vocab 96000`) when `MTP_DIR` is set:

```bash
# 1. Flagship v9 (w16384: 6,082.2 tok/s @ 16K prefill + lossless K=5 built-in MTP decode)
MODEL_DIR=/path/to/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid \
./bench/qwen38-flash-next-gb10/serve.sh

# 2. Interactive v9-w4096 (w4096: 5,276.3 tok/s @ 8K prefill + 57.4 tok/s decode + 12 GiB free UMA)
MODEL_DIR=/path/to/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid \
./bench/qwen38-flash-next-gb10/serve-w4096.sh

# 3. Pure maximum-velocity prefill-only (GDN_NORM_FP8=3 store-warp E4M3, no speculation)
MODEL_DIR=/path/to/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid \
./bench/qwen38-flash-next-gb10/serve-fast-prefill.sh

# 4. Dedicated K=5 lossless speculative decode (FP8 projection weight cache off for max L2 headroom)
MODEL_DIR=/path/to/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid \
./bench/qwen38-flash-next-gb10/serve-k5-decode.sh

# 5. Estonia 136.6K ultra-long-context needle profile (143,360 max-seq-len, 10 GiB FP8 cache reserve)
ESTONIA_136K=1 \
MODEL_DIR=/path/to/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid \
./bench/qwen38-flash-next-gb10/serve.sh

# 6. Verify binary & environment gates offline without starting the HTTP server:
MODEL_DIR=/path/to/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid \
./bench/qwen38-flash-next-gb10/serve.sh --verify-only
```

Once the server reports ready on `http://127.0.0.1:8899`, run the public-safe
reproduction verifier:

```bash
python3 repro/qwen38-flash-next/run.py \
  --endpoint http://127.0.0.1:8899/v1/chat/completions \
  --profile coding-k5
```

## All-context prefill (`8K..128K`) & Estonia `136.6K` needle benchmark

Every prefill figure below is **single-request (`concurrency = 1`, `batch_size = 1`)
end-to-end HTTP client wall-clock throughput** (`prompt_tokens / ttft_seconds`)
with prefix caching disabled (`--enable-prefix-caching false`) and recurrent
state caching disabled (`--ssm-cache-slots 0`). At each context depth, 1
uncounted warmup transition (`rep 0`) is followed by `N=3` timed repetitions
(`rep 1..3`) using deterministic slices of the 395,923-token corpus (`ids_sha256:
0f214e7481731a78a91f68a68bb09210f8151179d8d7e86d886511b2b8e93ad1`).

### All-context prefill table (`v9` 16K-chunk vs `v9-w4096` 4K-chunk)

| Context Depth | Prompt Tokens | `v9` (`w16384`) Certified Peak (tok/s) | `v9` (`w16384`) Wave 9 Med (tok/s) | `v9` (`w16384`) Wave 11 Med (tok/s) | `v9` (`w16384`) Min TTFT (s) | `v9-w4096` (`w4096`) Peak (tok/s) | `v9-w4096` (`w4096`) Med (tok/s) | `v9-w4096` Med TTFT (s) | Output SHA-256 Prefix (`20/20` Exact) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **`8K`** | `8,192` | **`5,830.2`** | `5,819.0` | `5,815.7` | `1.405 s` | **`5,276.3`** | **`5,266.8`** | `1.555 s` | `dbba921ce205` (`PASS`) |
| **`16K`** | `16,384` | **`6,082.2`** | **`6,076.7`** | `6,063.5` | `2.694 s` | **`5,176.9`** | **`5,164.5`** | `3.172 s` | `0695b563acde` (`PASS`) |
| **`32K`** | `32,768` | **`5,969.6`** | **`5,952.6`** | `5,945.0` | `5.489 s` | **`5,125.1`** | **`5,116.2`** | `6.405 s` | `e47125968b3b` (`PASS`) |
| **`64K`** | `65,536` | **`5,828.3`** | `5,792.4` | **`5,803.2`** | `11.244 s` | **`5,026.5`** | **`5,016.1`** | `13.065 s` | `0695b563acde` (`PASS`) |
| **`128K`** | `131,072` | **`5,694.4`** | `5,664.8` | **`5,684.0`** | `23.018 s` | **`4,899.8`** | **`4,891.7`** | `26.795 s` | `c51cdffb6e38` (`PASS`) |
| **`136.6K` (`Estonia`)** | `136,576` | **`5,429.4`** (srv) | **`5,400.4`** (avg) | **`5,428.3`** (srv med) | `25.160 s` | — | — | — | `5/5 = 100% PASS` (`Estonia`) |

### Per-stage prefill decomposition (`1,348.0 ms` per 8K tokens at 16K context)

| Subsystem Stage (48 Layers) | Time per 8K (`ms`) | Share (`%`) | Dominant SM121a Kernel / Route |
|---|---:|---:|---|
| **Routed + Shared MoE (`512` routed top-10 + `1` shared)** | `684.5 ms` | `50.8%` | `qwen4_moe_w4a8_v28` / `v28q=2` (`mma.sync.aligned.m16n8k64...e2m1.block_scale`), `inter` planner, `v16q4` slot quant |
| **Gated DeltaNet Linear Attention (`36` layers)** | `312.8 ms` | `23.2%` | `atlas_gdn_fi_sm121a_prefill_f16` (`GDN_FI=2` CuTe-DSL WY chunk scan), `gdn_conv_v3`, `gdn_norm_fp8_v2` |
| **Query-Sparse Attention `QSA` + Projections (`12` layers)** | `198.4 ms` | `14.7%` | `qsa_sparse_e4m3_qkpv` (`QSA_FP8=qkpv`, `QSA_PIPE=1`), `ldmatrix.x4` `score_v2`, fused group-max top-k, `o_proj` amax fold |
| **4-Stream `mHC` Hyper-Connections (`48` layers)** | `94.2 ms` | `7.0%` | `qwen4_hc_upmix_roof_t128_amax` (`MHC_ROOF=t128`), `MHC_GATEPRE=1`, fused `inject,cross,combine,nostage,upmix,injgemm,postpre` |
| **4-Ring NVFP4 PLE N-Gram Offload + Stream Glue** | `58.1 ms` | `4.3%` | 4-ring `io_uring` sector DMA (`PLE_READ_RINGS=4`, `PLE_SECTOR_READ=1`, `PLE_READ_AFFINITY=big`), `qwen4_ple_prefill_fused` |
| **Total per 8,192 Tokens (`16K` Chunk)** | **`1,348.0 ms`** | **`100.0%`** | **`6,076.7 – 6,082.2 tok/s` single-request (`C=1`)** |

### Estonia 136.6K (`136,576` prompt tokens / `136,584` active KV tokens) prefill & decode table

Measured with `llm-inference-bench` (`LIL v0.7.9`, commit `b11c2390262aa2b736402c23aa9ffaedb6a407b5`)
`--test-profile estonia` (`707,468` chars, prompt SHA-256 `8c045ba5e8299b2516d81603ba1bc27bad9fcf3583de0aecbf89a9d66b131942`,
6-hop cross-reference chain past planted `"Latvia"` decoy to expected `"Estonia"`,
`C=1`, `max_tokens=5120`, seed `20261016 + run_index`). Raising
`ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_WEIGHT_CACHE_RESERVE_GB=10` and setting
`ATLAS_QWEN4_QSA_VERIFY_ROWS=0` raised the minimum free UMA floor at 136.6K
active KV tokens from `3,848 MiB` (`v8c` watchdog abort) to **`8,431–8,847 MiB`
(`0` watchdog kills, `10/10` combined PASS)**:

| Campaign / Trial | Binary | Prompt Tokens | Active KV | Client TTFT (s) | Prefill (tok/s) | Client Decode (tok/s) | Server Decode (tok/s) | `accept_len` | Free UMA Floor | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `v8c` Baseline (`20261016T152317Z`) | `6f6f15fc2` | `136,576` | `136,584` | DNF | DNF | DNF | DNF | — | `3,848 MiB` | `0/5` (Watchdog Kill) |
| **Cert 1: Scout (`20261016T210345Z`)** | `64136c5d9-r1` | `136,576` | `136,584` | `25.69 s` | `5,316.9` | — | — | — | `8,431 MiB` | `SCOUT` (`max_tokens=1`) |
| **Cert 1: Run 1** | `64136c5d9-r1` | `136,576` | `136,584` | `25.18 s` | `5,428.7` (srv) | `44.3` | `44.5` | `2.91` | `8,431 MiB` | **PASS (`Estonia`)** |
| **Cert 1: Run 2** | `64136c5d9-r1` | `136,576` | `136,584` | `25.16 s` | **`5,429.4`** (srv) | **`48.2`** | **`48.6`** | **`3.06`** | `8,431 MiB` | **PASS (`Estonia`)** |
| **Cert 1: Run 3** | `64136c5d9-r1` | `136,576` | `136,584` | `25.17 s` | `5,428.3` (srv) | `42.3` | `42.4` | `2.81` | `8,431 MiB` | **PASS (`Estonia`)** |
| **Cert 1: Run 4** | `64136c5d9-r1` | `136,576` | `136,584` | `25.46 s` | `5,368.1` (srv) | `41.8` | `41.9` | `2.79` | `8,431 MiB` | **PASS (`Estonia`)** |
| **Cert 1: Run 5** | `64136c5d9-r1` | `136,576` | `136,584` | `25.48 s` | `5,364.1` (srv) | `42.4` | `42.3` | `2.83` | `8,431 MiB` | **PASS (`Estonia`)** |
| **Cert 1: Summary (`N=5`)** | `64136c5d9-r1` | **`136,576`** | **`136,584`** | **`25.29 s` avg** (`25.16` med) | **`5,400.4` avg** (`5,428.3` med) | **`43.8` avg** (`48.2` peak) | **`43.9` avg** (`48.6` peak) | **`2.88`** | **`8,431 MiB`** | **`5/5` (`100.0% PASS`)** |
| **Cert 2: Summary (`20261016T225440Z`, `N=5`)** | `b80fbf309-r1` | **`136,576`** | **`136,584`** | **`26.09 s` avg** (`26.07` scout) | **`5,239.6` scout** | **`43.6` agg** (`45.7` peak) | **`43.3` med** | **`2.85`** | **`8,847 MiB`** | **`5/5` (`100.0% PASS`)** |

Sanitized evidence record: [`repro/qwen38-flash-next/evidence/estonia-v9.json`](../../repro/qwen38-flash-next/evidence/estonia-v9.json).

## Lil Decode Bench table (`v9` vs `v9-w4096` + 10-prompt `LOSSLESS 10/10` gate)

### 1. Coding Peak & `0..128K` Context Sweep (`LIL v0.7.9`, `C=1`, 3 repetitions per context)

Sanitized evidence records:
- [`repro/qwen38-flash-next/evidence/lil-decode-v9.json`](../../repro/qwen38-flash-next/evidence/lil-decode-v9.json) (`lil-09c-serve-v9-20261016T165118Z-campaign-full`, `w16384`)
- [`repro/qwen38-flash-next/evidence/lil-decode-v9-w4096.json`](../../repro/qwen38-flash-next/evidence/lil-decode-v9-w4096.json) (`lil-09c-serve-v9-w4096-20261016T214937Z-campaign-full`, `w4096`)

| Workload / Context Row | `v9` (`w16384`) Client Med (tok/s) | `v9` (`w16384`) Server `Done:` Med (tok/s) | `v9` (`w16384`) Mean `accept_len` | `v9` (`w16384`) Prefill Med (tok/s) | `v9-w4096` (`w4096`) Client Med (tok/s) | `v9-w4096` (`w4096`) Server `Done:` Med (tok/s) | `v9-w4096` (`w4096`) Mean `accept_len` | `v9-w4096` Re-Prefills |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Coding Peak (`5 x 1024` tok)** | **`52.15` med** (`53.56` mean / **`58.70` max**) | `52.3` med (`59.0` max) | **`3.15`** | — | **`56.34` med** (`55.71` mean / **`57.42` max**) | **`56.5` med** (**`57.6` max**) | **`3.02`** | `0` |
| **Context `0` (cold prompt)** | `42.24` | `42.2` | `2.15` | — | **`42.84`** | **`43.0`** | `2.34` | `0` |
| **Context `8K` (`8,192`)** | **`42.91`** | `42.6` | `2.22` | `5,342.0` | `42.74` | **`42.7`** | `2.26` | `0` |
| **Context `16K` (`16,384`)** | `40.87` | `40.8` | `1.95` | **`5,754.0`** | **`43.52`** | **`43.1`** | **`2.41`** | `0` |
| **Context `32K` (`32,768`)** | `39.63` | `39.2` | `1.97` | **`5,722.0`** | **`43.16`** | **`43.1`** | `2.31` | `0` |
| **Context `64K` (`65,536`)** | `39.00` | `40.3` | `1.95` | **`5,627.0`** | **`42.12`** | **`41.6`** | `2.33` | `0` |
| **Context `128K` (`131,072`)** | `39.04` | `40.1` | `1.92` | **`5,508.0`** | **`40.72`** | **`41.9`** | `2.16` | `0` |

Why `v9-w4096` beats `v9` (`w16384`) on sustained mixed-context decode (`40.7–43.5 tok/s`
with `0` re-prefill windows vs `39.0–42.9 tok/s`): capping the prefill scratch
arena at 4,096 tokens frees `+4.2 GiB` of unified memory (`11,907–12,695 MiB`
free floor), eliminating KV block eviction and re-prefill windows across the
entire `0..128K` sweep.

### 2. 10-Prompt Greedy Lossless Gate (`K=5` MTP vs `K=1` Plain Decode, `2,560 / 2,560` Exact)

Every promotion in `05e-serve-v9-gate` (`vdec-serve-v9-20261016T1809`) requires
100% bit-for-bit greedy token identity (`temperature = 0.0`, `256` tokens per
prompt = `2,560` tokens total) between `K=5` Inferact MTP speculative decode
and `K=1` non-speculative decode (`ATLAS_DISABLE_SPECULATION=1`):

| Prompt ID | Context Depth | Plain `K=1` (tok/s) | Speculative `K=5` (tok/s) | Speedup | Plain SHA-256 Prefix | `K=5` SHA-256 Prefix | Greedy Token Identity (`256/256`) |
|---|---:|---:|---:|---:|---|---|---|
| `d0a` | `0` | `33.3` | `42.1` | `1.26x` | `6a4797a22931` | `6a4797a22931` | **PASS (`256/256` Exact)** |
| `dc0a` (coding) | `0` | `33.7` | **`51.3`** | **`1.52x`** | `e479aec17a0b` | `e479aec17a0b` | **PASS (`256/256` Exact)** |
| `d8ka` | `8K` | `31.8` | `45.6` | `1.43x` | `d3e66510cb16` | `d3e66510cb16` | **PASS (`256/256` Exact)** |
| `d32ka` | `32K` | `31.1` | `46.1` | `1.48x` | `1ad002470c4a` | `1ad002470c4a` | **PASS (`256/256` Exact)** |
| `d64ka` | `64K` | `31.1` | `46.4` | `1.49x` | `eee781f4eb2c` | `eee781f4eb2c` | **PASS (`256/256` Exact)** |
| `d0b` | `0` | `33.4` | `45.8` | `1.37x` | `6a4797a22931` | `6a4797a22931` | **PASS (`256/256` Exact)** |
| `dc0b` (coding) | `0` | `33.4` | **`51.9`** | **`1.55x`** | `e479aec17a0b` | `e479aec17a0b` | **PASS (`256/256` Exact)** |
| `d8kb` | `8K` | `31.5` | `45.9` | `1.46x` | `d3e66510cb16` | `d3e66510cb16` | **PASS (`256/256` Exact)** |
| `d32kb` | `32K` | `31.5` | `45.0` | `1.43x` | `1ad002470c4a` | `1ad002470c4a` | **PASS (`256/256` Exact)** |
| `d64kb` | `64K` | `31.0` | `46.5` | `1.50x` | `eee781f4eb2c` | `eee781f4eb2c` | **PASS (`256/256` Exact)** |
| **Total / Gate Verdict** | **`0..64K`** | **`31.0 – 33.7`** | **`42.1 – 51.9`** | **`1.26x – 1.55x`** | — | — | **`LOSSLESS: 10 identical, 0 different` (`2,560/2,560`)** |

## Things that will cost you a day

**1. `FLUX_TARGET_MODEL=qwen3.8-flash-next` at build time is mandatory, and omitting it is silent.**
`crates/atlas-kernels/build.rs` defaults silently to `qwen3-next-80b-a3b`. A
binary built without `FLUX_TARGET_MODEL=qwen3.8-flash-next FLUX_TARGET_QUANT=nvfp4`
(or the `ATLAS_TARGET_*` equivalents) compiles cleanly and fails only when
`serve` tries to resolve `--kernel-target qwen3.8-flash-next`. All four
`serve*.sh` scripts in this directory check `strings -a "$BIN"` for
`qwen3.8-flash-next` before launching.

**2. `GDN_NORM_FP8=3` is for pure prefill only: it breaks `K=5` speculative decode determinism.**
`FLUX_QWEN4_PREFILL_GDN_NORM_FP8=3` (`ATLAS_QWEN4_PREFILL_GDN_NORM_FP8=3`, used
in `serve-fast-prefill.sh`) enables the store-warp E4M3 normalization inside the
FlashInfer GDN kernel. Under multi-row `K=5` speculative verification rollbacks,
that asynchronous store-warp reduction is not state-deterministic (`LOSSLESS: 4
identical, 6 different` in `05e` gate trial 1). Unified serving with `K=5` MTP
(`serve.sh` and `serve-w4096.sh`) **must** use `GDN_NORM_FP8=2` paired with
`GDN_NORM_FP8_V2=1`, which restores `LOSSLESS: 10 identical, 0 different` while
retaining 6,082.2 tok/s prefill.

**3. `LAST_LAYER_CROP=1` without `MTP_LASTK_DEFER=1` corrupts the MTP proposer's prompt handoff.**
During prefill, only the final token of a prompt chunk needs logits from layer
47, except when native MTP is active, where the proposer needs the trailing
hidden states across chunk boundaries. Enabling `ATLAS_QWEN4_PREFILL_LAST_LAYER_CROP=1`
without `ATLAS_MTP_LASTK_DEFER=1` and `ATLAS_QWEN4_MTP_LASTK_SKIP_INTERMEDIATE=1`
either refuses at runtime or feeds stale hidden states into the first MTP draft
cycle. `serve.sh` and `serve-w4096.sh` export all three together and verify
their presence in the binary.

**4. Unbounded FP8 projection weight caching (`FAST_PROJ_FP8_WEIGHT_CACHE=1`) will trip the GB10 UMA watchdog at 128K–144K.**
Caching pre-quantized FP8 E4M3 weights for the 96 GDN/QSA fast projections is
worth `+300+ tok/s` in prefill, but on a 128 GB unified-memory system an
unbounded cache leaves only `3,848 MiB` free at 136.6K context and triggers the
UMA safety watchdog (`watchdog_kills=1`). Always pair
`ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_WEIGHT_CACHE=1` with
`ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_WEIGHT_CACHE_RESERVE_GB=7` for `128K`
(`serve.sh` / `serve-w4096.sh`, `7,072–12,695 MiB` floor) or `_RESERVE_GB=10`
plus `ATLAS_QWEN4_QSA_VERIFY_ROWS=0` for `136.6K` Estonia (`ESTONIA_136K=1`,
`8,431–8,847 MiB` floor).

**5. Do not copy dense `Qwen3.8-27B` DFlash flags (`--dflash-gamma 15`) or legacy `SSM_PREFILL_BATCH=1` onto Flash-Next.**
Dense Qwen3.8 V3 uses bidirectional B16 attention and forces `gamma=15`
(`K=16` verify rows). On Flash-Next, `K=16` has no batched verify route and
falls back to the 418 ms row-serial oracle (`2.88 tok/s`, slower than plain
decode even at 100% acceptance). Likewise, legacy `ATLAS_QWEN4_SSM_PREFILL_BATCH=1`
and `ATLAS_QWEN4_SSM_PREFILL_TILE>=2` from early bring-up corrupt GDN state from
token 0. Use the qualified `ATLAS_QWEN4_PREFILL_GDN_FI=2` + `GDN_CONV_V3=1` +
`QSA=1` pipeline configured in `serve*.sh`.

**6. Know which throughput counter you are reading.**
- **Prefill (`prompt_tokens / ttft_seconds`)**: client wall-clock TTFT and
  server `Done:` prefill tok/s agree within `~0.5%` at 16K–136.6K (`5,400.4`
  client vs `5,428.3` server on Estonia).
- **Decode**: `usage.response_token/s` (non-streaming) and server `Done:` log
  lines measure pure decode after the first token (`56.5–57.6 tok/s` on Coding
  Peak), whereas streaming client `openai_continuous_usage` (`56.3–57.4 tok/s`)
  includes per-chunk SSE transport, and naive `completion_tokens / total_elapsed`
  includes TTFT. Compare like with like.

Every `FLUX_*` and `ATLAS_*` variable set by the four launchers is documented in
[`FLAGS.md`](FLAGS.md). Cryptographic digests for checkpoints, binaries, recipes,
and evidence files are in [`ARTIFACTS.sha256`](ARTIFACTS.sha256).
