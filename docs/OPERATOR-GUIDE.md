# Flux Engine — Operator Guide (GB10 / `sm_121a`)

Everything you need to serve local models on **NVIDIA DGX Spark (GB10)**: either via the universal container appliance (`ghcr.io/theapathy/flux-engine:gb10`) or directly from a host build, without rediscovering the unified-memory and build gotchas the hard way.

---

## TL;DR — Start Serving in One Command

### Option A: Universal Container Appliance (Any Local Model)

Mount your local models directory at `/models` and point `flux serve` at any checkpoint folder. The entrypoint inspects `config.json`, `hf_quant_config.json`, `ple/` shards, and companion drafters, then launches the matching kernel target and operating profile automatically:

```bash
# Serve Qwen3.8-Flash-Next-NVFP4-Offload (6,082.2 tok/s @ 16K prefill + 57.4 tok/s lossless K=5 MTP decode)
docker run --rm --gpus all --ipc=host -p 8899:8899 \
  -v /path/to/models:/models:ro \
  ghcr.io/theapathy/flux-engine:gb10 \
  serve /models/Qwen3.8-Flash-Next-NVFP4-Offload --mode max-prefill

# Or serve via environment variables (interactive w4096 mode keeps ~12 GiB free UMA headroom)
docker run --rm --gpus all --ipc=host -p 8899:8899 \
  -v /path/to/models:/models:ro \
  -e FLUX_MODEL=/models/Qwen3.8-Flash-Next-NVFP4-Offload \
  -e FLUX_MODE=interactive \
  ghcr.io/theapathy/flux-engine:gb10

# Inspect all local checkpoints under /models without starting a server
docker run --rm --gpus all -v /path/to/models:/models:ro \
  ghcr.io/theapathy/flux-engine:gb10 list-models /models
```

### Option B: Direct Host Launchers

```bash
# 1. Qwen3.8-Flash-Next-NVFP4-Offload — Flagship 6K+ prefill + 57.4 tok/s K=5 decode (port 8899)
MODELS_ROOT=/path/to/models \
  ./bench/qwen38-flash-next-gb10/serve.sh

# 2. Qwen3.8-Flash-Next-NVFP4-Offload — Interactive w4096 (~12 GiB free UMA floor, port 8899)
MODELS_ROOT=/path/to/models \
  ./bench/qwen38-flash-next-gb10/serve-w4096.sh

# 3. Qwen3.8-27B-NVFP4 + DFlash v2 (gamma=15, 64.0 tok/s decode, port 8896)
MODEL=/path/to/models/Qwen3.8-27B-NVFP4 \
DRAFTER=/path/to/models/Qwen3.8-27B-DFlash-v2 \
  ./bench/qwen38-gb10/serve.sh
```

Wait ~30–60 s for `/health` to return `200 OK`:

```bash
curl -s http://127.0.0.1:8899/health
# {"status":"ok"}
```

---

## Where the Weights Are

| Role | Checkpoint Directory | Size | What It Is |
|---|---|---:|---|
| **Flash-Next Target** | `/models/Qwen3.8-Flash-Next-NVFP4-Offload` | `~70 GiB` | 48-layer hybrid (`36` GDN + `12` QSA), 512 routed + 1 shared MoE experts (`NVFP4`), 4-stream mHC, 128 sector-aligned NVFP4 `ple/` shards (`548` files total) |
| **Flash-Next MTP Drafter** | `/models/Qwen3.8-Flash-Next-Inferact-MTP` | `~4.2 GiB` | Native 1-layer full-attention MoE Multi-Token Prediction head (`num_drafts=4`, `K=5` lossless verify) |
| **Dense 27B Target** | `/models/Qwen3.8-27B-NVFP4` | `21.8 GiB` | `unsloth/Qwen3.8-27B-NVFP4` (`64` layers: `48` GDN + `16` GQA, dense FFN) |
| **Dense 27B Drafter (v1 B16)** | `/models/Qwen3.8-27B-DFlash-v2` | `3.96 GiB` | 8-layer BF16 block-diffusion drafter (`block_size=16`, proposes `gamma=15` tokens/step, `63.9–64.0 tok/s` on 400-tok coding probe) |
| **Dense 27B Drafter (DFlash2 B8)** | `/models/Qwen3.8-27B-DFlash2` | `~2.5 GiB` | 5-layer qualified default `z-lab/Qwen3.8-27B-DFlash2` (`block_size=8`, `gamma=7`) or `incoai/Qwen3.5-27B-DFlash2` (`block_size=16`, `58.01 tok/s`) |

---

## Where the Engine Is (and Build Rules)

| Item | Path / Command |
|---|---|
| **Compiled binaries** | `target/release/flux-server`, `target/release/flux-cli`, `target/release/spark` (`/usr/local/bin/flux-server` in container) |
| **Smart CLI wrapper** | [`repro-kit/flux-autodetect`](../repro-kit/flux-autodetect) (`/usr/local/bin/flux` in the container) |
| **Environment bridge** | [`scripts/flux_env_bridge.sh`](../scripts/flux_env_bridge.sh) (`FLUX_*` <-> `ATLAS_*` translator) |
| **Operating recipes** | [`recipes/`](../recipes/) (`flux-nextgen-max-prefill-decode.env`, `flux-v9-max-prefill.env`, `flux-v9-w4096-interactive.env`, `flux-fast-prefill.env`, `flux-k5-lossless-decode.env`, `flux-estonia-136k.env`, `flux-exact-reference.env`) |

> **Three build rules:**
> 1. **Always build `-p spark-server`:** `crates/spark-server` is the binary crate defining `flux-server`, `flux-cli`, and `spark`.
> 2. **Always pass `FLUX_TARGET_MODEL` (or `ATLAS_TARGET_MODEL`) when compiling:** `crates/atlas-kernels/build.rs` only embeds PTX for the requested model targets. Build with:
>    ```bash
>    export PATH="/usr/local/cuda/bin:${PATH}"
>    FLUX_TARGET_HW=gb10 \
>    FLUX_TARGET_MODEL="qwen3.8-flash-next,qwen3.8-27b" \
>    FLUX_TARGET_QUANT=nvfp4 \
>      cargo build --release -p spark-server
>    ```
> 3. **Touch `crates/atlas-kernels/build.rs` after editing `MODEL.toml`:** Cargo's build-script cache does not track `kernels/gb10/*/MODEL.toml` changes unless `build.rs` is touched.

---

## Operating Modes & Recipes (`FLUX_MODE`)

| `FLUX_MODE` Alias | Recipe File | Chunk (`WIDE`) | Headline Prefill | Headline Decode | Free UMA Floor | When to Use |
|---|---|---:|---|---|:---:|---|
| **`auto`** *(Container default)* | *(Auto-selected per model)* | Auto | Auto | Auto | `>= 7,072 MiB` | Inspects local checkpoint under `/models` and picks the best recipe automatically. |
| **`max-prefill`** (`v9`) | [`recipes/flux-v9-max-prefill.env`](../recipes/flux-v9-max-prefill.env) | `16,384` | **`6,082.2 tok/s` @ 16K**<br/>`5,694.4 tok/s` @ 128K | **`57.4 tok/s` peak**<br/>(`LOSSLESS 10/10`) | `7,072 – 8,450 MiB` | Dedicated single-stream / agentic serving with maximum 8K–128K prefill + lossless `K=5` decode. |
| **`interactive`** (`interactive-w4096`) | [`recipes/flux-v9-w4096-interactive.env`](../recipes/flux-v9-w4096-interactive.env) | `4,096` | **`5,276.3 tok/s` @ 8K**<br/>`5,016.1 tok/s` @ 64K | **`57.4 tok/s` peak**<br/>(`LOSSLESS 10/10`) | **`11,907 – 12,695 MiB`** | Interactive chat, IDE coding assistants, or shared workstations needing `+4.2 GiB` extra free memory. |
| **`fast-prefill`** (`fast`) | [`recipes/flux-fast-prefill.env`](../recipes/flux-fast-prefill.env) | `16,384` | **Ultra prefill velocity**<br/>(`GDN_NORM_FP8=3`) | Plain (`K=1`) | `8,500+ MiB` | Batch prompt ingestion, RAG indexing, logprob scoring, or single-token classification. |
| **`decode`** (`k5-decode`) | [`recipes/flux-k5-lossless-decode.env`](../recipes/flux-k5-lossless-decode.env) | `8,192` | `~4,400 tok/s` | **`57.4 tok/s` peak**<br/>(`56.3` median) | **`10,500+ MiB`** | Long generation sessions (`max_tokens >= 512`) with FP8 weight cache disabled for maximum L2/UMA headroom. |
| **`long-context`** (`long-context-136k`) | [`recipes/flux-estonia-136k.env`](../recipes/flux-estonia-136k.env) | `16,384` | **`5,400.4 tok/s` @ 136.5K** | **`48.6 tok/s` server** | **`8,412+ MiB`** | `128K – 144K` ultra-long context (`_RESERVE_GB=10`, `QSA_VERIFY_ROWS=0`, `--max-seq-len 143360`). |
| **`exact`** (`exact-reference`) | [`recipes/flux-exact-reference.env`](../recipes/flux-exact-reference.env) | `16,384` | Reference exact W4A16/BF16 | `33.4 tok/s` (`K=1`) | `8,800+ MiB` | Differential 0-ULP and `LOSSLESS 10/10` token-identity auditing baseline (`DISABLE_SPECULATION=1`). |

---

## Complete Environment Variable Reference (`FLUX_*`)

All `FLUX_*` variables can be passed via `docker run -e FLUX_...=...` or exported on the host before sourcing [`scripts/flux_env_bridge.sh`](../scripts/flux_env_bridge.sh):

| Variable | Default | Description |
|---|---|---|
| `FLUX_MODEL` | *(auto-discovered in `/models`)* | Path to local target model directory (e.g. `/models/Qwen3.8-Flash-Next-NVFP4-Offload`). |
| `FLUX_MODE` | `auto` | Operating mode: `auto`, `max-prefill`, `interactive` (`interactive-w4096`), `fast-prefill` (`fast`), `decode` (`k5-decode`), `long-context` (`long-context-136k`), `exact` (`exact-reference`). |
| `FLUX_PORT` | `8899` | HTTP listening port for the OpenAI-compatible API (`/v1/chat/completions`, `/v1/models`, `/health`). |
| `FLUX_HOST` | `127.0.0.1` (host) / `0.0.0.0` (container) | Bind address passed to `--bind` (`--host`). |
| `FLUX_MAX_SEQ_LEN` | Mode-dependent (`40960` / `135168` / `143360`) | Maximum context length in tokens (`--max-seq-len`). |
| `FLUX_KV_CACHE_DTYPE` | `bf16` (or `fp8` / `turboquant`) | KV cache precision (`bf16` required for lossless Qwen3.8-Flash-Next and Qwen3.8-27B recipes). |
| `FLUX_GPU_MEMORY_UTILIZATION` | `0.55` (GB10 UMA) / `0.90` (discrete) | Fraction of memory budget for runtime arenas (keep `<= 0.55` on GB10 UMA to preserve OS and runtime headroom). |
| `FLUX_DRAFT_MODEL` | *(auto-detected if present)* | Path to companion speculative drafter (`Inferact-MTP` or `DFlash`). |
| `FLUX_GAMMA` | `4` (MTP `K=5`) / `15` (DFlash `K=16`) | Number of speculative draft tokens proposed per step (`gamma`). |
| `FLUX_DISABLE_SPECULATION` | `0` *(unset)* | Set to `1` to disable all speculative decoding (equivalent to `--no-speculative` / `ATLAS_DISABLE_SPECULATION=1`; use `FLUX_DRAFT_MODEL=none` to disable external DFlash/MTP sidecars). |
| `FLUX_THINKING` | unset (`MODEL.toml` `thinking_default` applies) | Default `<think>` behavior (`on`, `off`, or unset for model default). Can also be toggled per-request via `chat_template_kwargs`. |
| `FLUX_THINKING_BUDGET` | MODEL profile (`262144` in `MODEL.toml` for Qwen3.8-Flash-Next & Qwen3.8-27B) | Optional `--max-thinking-budget` server-ceiling override. A request may select a lower budget; when unset, the selected `MODEL.toml` profile remains authoritative. |
| `FLUX_CPU_AFFINITY` | `5-9,15-19` (GB10 ARM64) | CPU core mask passed to `taskset -c` (pins server + PLE DMA to the 10 Cortex-X925 performance cores). |
| `FLUX_CONCURRENCY` | `1` | Max concurrent sequences (`--max-num-seqs` / `--max-batch-size`). Qwen3.8-Flash-Next batches `2..32` natively by default (`ATLAS_QWEN4_NATIVE_BATCH` unset; `v1.0.x` supports `FLUX_CONCURRENCY=16` with lazy-H SSM pool); `33..128` requires `ATLAS_QWEN4_NATIVE_BATCH=0` and decodes C1 once per row. Startup memory admission still applies. |
| `FLUX_KERNEL_TARGET` | `auto` | Explicit `--kernel-target` override matching a directory under `kernels/gb10/` (e.g. `qwen3.8-flash-next`, `qwen3.8-27b`, `qwen3.6-27b`, `deepseek-v4.1`, `glm5.3-flash`, `gemma-4-26b-a4b`, `gemma-4-31b`). |
| `FLUX_WEIGHT_CACHE_DIR` | `/var/cache/flux-weight-cache` | Directory for persistent post-transform weight cache shards. |
| `FLUX_QWEN4_PREFILL_WIDE` | `16384` (`max-prefill`) / `4096` (`interactive`) | Prefill chunk token width (`4096`, `8192`, `16384`). Must match `--max-prefill-tokens`. |
| `ATLAS_QWEN4_NATIVE_BATCH` | unset (native) | Qwen4 C2+ decode route. Unset: native one-forward batching for C2-C32, with every serialized frame logged and counted by reason (`atlas_qwen4_decode_frames_total`). `1`: native or refuse to start. `0`: serial C1 per row. |
| `FLUX_QWEN4_FAST_PROJ_FP8_WEIGHT_CACHE_RESERVE_GB` | `7` (`128K`) / `10` (`136K+`) | Free unified memory floor (in GiB) below which FP8 fast-projection weight caching stops allocating. |

---

## How to Query It (OpenAI-Compatible API)

### 1. Standard Chat Completion (Streaming or Non-Streaming)

```bash
curl -s http://127.0.0.1:8899/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen38-native-mtp-frontier-v1",
    "messages": [
      {"role": "user", "content": "Write a lock-free ring buffer in Rust with atomic acquire/release indices."}
    ],
    "max_tokens": 512,
    "temperature": 0.0
  }' | jq -r '.choices[0].message.content'
```

### 2. Toggling `<think>` Reasoning Per Request (`chat_template_kwargs`)

Even if the server was launched with `--default-chat-template-kwargs '{"enable_thinking":false}'` (default for throughput benchmarking), any client request can override thinking on the fly:

The request selects a budget, which is capped by the effective server ceiling.
That ceiling comes from an explicit `FLUX_THINKING_BUDGET` /
`--max-thinking-budget` override when present, otherwise from the selected
`MODEL.toml` profile (`max_thinking_budget = 262144` on Qwen3.8-Flash-Next). The
launcher reports the ceiling and its source without replacing the model profile.

```bash
# Enable thinking for a hard math / multi-hop reasoning query:
curl -s http://127.0.0.1:8899/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen38-native-mtp-frontier-v1",
    "messages": [{"role": "user", "content": "Find all integer solutions to x^3 + y^3 = 1729."}],
    "chat_template_kwargs": {"enable_thinking": true},
    "max_tokens": 2048,
    "temperature": 0.6
  }'
```

### 3. Tool Calling (`tools` / `tool_calls`)

```bash
curl -s http://127.0.0.1:8899/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen38-native-mtp-frontier-v1",
    "messages": [{"role": "user", "content": "Check the current free memory floor on node spark-01."}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_node_meminfo",
        "description": "Read /proc/meminfo MemAvailable in MiB on a cluster node",
        "parameters": {
          "type": "object",
          "properties": {"node": {"type": "string"}},
          "required": ["node"]
        }
      }
    }],
    "max_tokens": 256,
    "temperature": 0.0
  }' | jq '.choices[0].message.tool_calls'
```

---

## GLM-5.3 speculative decoding (DFlash2): not admitted yet

GLM-5.3 Flash serves target-only by default. Speculative decoding with the DFlash2 drafter is built in,
but its admission gate is **closed** until a Flux measurement certifies it. Until then, every number from
the speculative path is unvalidated.

- **Drafter license.** The DFlash2 drafter checkpoint is **CC BY-NC-ND 4.0** (non-commercial, no
  derivatives). It is not shipped. Operators supply their own copy, for example
  `--draft-model /models/GLM-5.3-Flash-DFlash2-bf582e4`, and are responsible for the license terms.
- **Why it is closed.** `GLM53_SPECULATIVE_EVIDENCE` (`speculative_admission.rs`) is a pending Flux
  record. It opens only for evidence that meets all of the following:
  - it was measured on `flux-engine`;
  - it names the same binary for both measurement runs;
  - it matches the speculation-path sources compiled into the running binary;
  - it passes the identity, state and negative-control gates.

  Earlier measurements from the legacy `apathy-atlas` branch are not transferable to Flux Engine and must be re-qualified on the current runtime.
- **Bring-up only.** With `ATLAS_GLM53_UNVALIDATED_BRINGUP=1` explicitly set, and only then, the server
  loads the drafter while the gate is closed. It logs a warning that output is unvalidated. Use this only
  for the measurement runs below, never for serving users.
  ```
  ATLAS_GLM53_UNVALIDATED_BRINGUP=1 flux-server-glm53 serve --model-from-path <GLM-5.3 EXL3> \
    --dflash --draft-model <DFlash2 dir> --dflash-gamma 8 --glm-dflash-max-drafts 3 --max-num-seqs 1
  ```
- **Limits.**
  - Batch 1 only: the driver refuses more than one active sequence.
  - The context cap stays at 2048 until longer-context evidence exists.
  - Greedy only.
- **Startup log.** It always prints `GLM53_SPEC_SOURCES fnv1a64=<hash> evidence=<valid|not-admitting>`. That
  is the binary's own statement of which sources it would certify.
- **Admitting it** takes one Spark, alone, for about 4 h. The runbook is `bench/glm53-spec/README.md`:
  1. P2 output identity.
  2. P3 state identity and negative controls.
  3. The executed kernel-equivalence receipt.
  4. `make_evidence.py --flux`, then a rebuild.
  5. An `--admitted` P2 re-run as the admission receipt.

## Gotchas (Learned the Hard Way on GB10)

1. **`nvidia-smi` reports `[N/A]` for per-process memory on GB10.**
   Because Grace and Blackwell share 128 GB of coherent LPDDR5X over NVLink-C2C, there is no separate PCIe BAR VRAM counter. Always monitor `/proc/meminfo` (`awk '/MemAvailable/ {printf "%.1f GiB\n", $2/1048576}' /proc/meminfo`) and poll `GET /health` to check server liveness.
2. **Never let `MemAvailable` drop below `6,656 MiB` (`6.5 GiB`).**
   The GB10 hardware watchdog resets the node if free unified memory hits `6,144 MiB` (`6.0 GiB`). All Flux recipes enforce `MEASURED_FLOOR_MB >= 6,656 MiB` via `FLUX_QWEN4_FAST_PROJ_FP8_WEIGHT_CACHE_RESERVE_GB=7` (or `10` at 136K+ context).
3. **Never pass `"reasoning_effort"` to Qwen3.8 chat templates.**
   Qwen3.8's Jinja template explicitly calls `raise_exception` if `reasoning_effort` is present, causing an HTTP `400`. Use `"chat_template_kwargs": {"enable_thinking": true|false}` instead.
4. **Do not raise `--gpu-memory-utilization` above `0.55` on GB10.**
   On unified memory, raising `--gpu-memory-utilization` from `0.55` to `0.68` shrinks allocatable headroom (`16.6 GiB` -> `5.7 GiB`) and risks an OS OOM during wide prefill chunks.
5. **Size concurrency (`FLUX_CONCURRENCY` / `--max-num-seqs`) carefully on GB10 UMA.**
   Each active sequence in a hybrid GDN model owns an FP32 recurrent state slot across 36–48 layers. Under legacy eager state pools, keep `--max-num-seqs <= 8` on long contexts; in `v1.0.x` with the lazy-H SSM pool and startup memory admission, `FLUX_CONCURRENCY=16` (up to `32` native batched) is supported when free UMA headroom allows.

---

## One-Liner Glossary

- **GDN (Gated DeltaNet):** Linear-attention recurrent layer (`36` of `48` layers in `Qwen3.8-Flash-Next`, `48` of `64` in `Qwen3.8-27B`) with constant-size FP32 state `[H_v, d_k, d_v]` updated via WY chunkwise scan.
- **QSA (Query-Sparse Attention):** Full-attention layer (`12` of `48` layers in `Qwen3.8-Flash-Next`) that scores 64-token KV pages using compressed indexer keys and attends only to the top-`k` (`2,048`-token budget) blocks in FP8 `E4M3`.
- **mHC (Multi-Stream Hyper-Connections):** 4-stream residual highway (`4 × 2,560`) mixed before and after every GDN, QSA, and MoE block via low-rank (`r = 320`) dynamic gates.
- **PLE (Per-Layer Embeddings):** 20-million-entry 3-gram embedding table (`128` NVFP4 shards on NVMe) streamed asynchronously via 4 `O_DIRECT` DMA rings and folded into each layer's residual stream.
- **Inferact MTP (`K=5`):** Native 1-layer Multi-Token Prediction head that proposes 4 draft tokens on device, verified in a single 5-row target pass with `10/10` bit-for-bit greedy equivalence.
- **DFlash (`gamma=7` / `gamma=15`):** Bidirectional block-diffusion drafter for `Qwen3.8-27B` (`5`-layer DFlash2 `B8` qualified default `z-lab/Qwen3.8-27B-DFlash2`, or `8`-layer DFlash v1 `B16` proposing 15 tokens per pass conditioned on 5 target hidden layers `[1, 16, 31, 46, 61]`).
