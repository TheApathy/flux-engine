<p align="left"><a href="../README.md"><b>← Flux Engine</b></a> · <b>Operating Mode Recipes (<code>recipes/</code>)</b></p>

# Operating Modes & Recipes (`recipes/`)

Flux Engine ships curated operating recipes for **NVIDIA Grace Blackwell GB10 (`sm_121a`)**, **discrete NVIDIA Blackwell SM120 (`sm_120a`)**, and **multi-Spark (`2×`/`3×`/`4×`)** topologies covering bulk context prefill, low-TTFT interactive serving, `K=5` speculative decoding, ultra-long-context reasoning, concurrency, model-specific profiles, and schema-1 TOML recipe contracts.

All recipes support **dual-namespace environment configuration** via [`scripts/flux_env_bridge.sh`](../scripts/flux_env_bridge.sh):
1. **`FLUX_*` Selectors:** Users, systemd units, and containers can set `FLUX_MODE=<mode>` or individual `FLUX_*` variables (e.g., `FLUX_QWEN4_PREFILL_WIDE=16384`, `FLUX_QWEN4_PREFILL_MOE_V28Q=2`).
2. **Runtime `ATLAS_*` Synchronization:** Sourcing [`scripts/flux_env_bridge.sh`](../scripts/flux_env_bridge.sh) loads the selected recipe and mirrors every `FLUX_*` variable to its corresponding `ATLAS_*` runtime selector automatically.

---

## 1. Mode Selection Matrix

| Mode Alias (`--mode` / `FLUX_MODE`) | Recipe File | Paired Argv File | Prefill Chunk (`WIDE`) | Headline Prefill Throughput | Headline Decode Throughput | Free Memory Floor (`MemAvailable`) | Primary Use Case |
|---|---|---|---:|---|---|---:|---|
| **`max-prefill`** *(Default Flagship)* | [`flux-v9-max-prefill.env`](flux-v9-max-prefill.env) | [`flux-serve.argv`](flux-serve.argv) | `16,384` | **`6,082.2 tok/s` @ 16K**<br/>`5,969.6` @ 32K \| `5,830.2` @ 8K<br/>`5,828.3` @ 64K \| `5,694.4` @ 128K | **`57.4 tok/s` peak**<br/>(`56.3` median, `LOSSLESS 10/10`) | `7,072 – 8,450 MiB` | Maximum prefill + decode throughput across `8K..128K` context lengths. |
| **`interactive-w4096`** (`interactive`) | [`flux-v9-w4096-interactive.env`](flux-v9-w4096-interactive.env) | [`flux-serve-w4096.argv`](flux-serve-w4096.argv) | `4,096` | **`5,276.3 tok/s` peak @ 8K**<br/>(`5,266.8` @ 8K \| `5,164.5` @ 16K<br/>`5,116.2` @ 32K \| `5,016.1` @ 64K) | **`57.4 tok/s` peak**<br/>(`56.3` median, `LOSSLESS 10/10`) | **`11,907 – 12,695 MiB`** | Low-TTFT interactive chat/IDE serving with `+4.2 GiB` extra unified memory headroom. |
| **`fast-prefill`** (`fast`) | [`flux-fast-prefill.env`](flux-fast-prefill.env) | [`flux-serve.argv`](flux-serve.argv) | `16,384` | **Ultra-tier prefill velocity**<br/>(`GDN_NORM_FP8=3` store-warp E4M3) | Plain / single-token emit | `8,500+ MiB` | Pure prefill-only batch ingestion, RAG indexing, summarization, and classification. |
| **`k5-decode`** (`decode`) | [`flux-k5-lossless-decode.env`](flux-k5-lossless-decode.env) | [`flux-serve.argv`](flux-serve.argv) | `8,192` | `~4,400 tok/s` (conservative `w8192`, FP8 weight cache off) | **`57.4 tok/s` peak / `56.3` median**<br/>(`3.02–3.15` mean accept len) | **`10,500+ MiB`** | Dedicated speculative decode serving with maximum L2/UMA headroom (`LOSSLESS 10/10` greedy identity). |
| **`long-context-136k`** (`long-context`) | [`flux-estonia-136k.env`](flux-estonia-136k.env) | [`flux-serve.argv`](flux-serve.argv)<br/>*(+ `--max-seq-len 143360`)* | `16,384` | **`5,400.4 tok/s` @ `136,584` tok**<br/>(`25.29s` TTFT on 707 KB prompt) | **`48.2 tok/s` client**<br/>(`48.6` server, `2.88` accept len) | **`8,412 – 8,431+ MiB`** | Certified `136.5K+` needle-in-a-haystack & 6-hop reasoning (`5/5 = 100% PASS` on `--test-profile estonia`). |
| **`exact-reference`** (`exact`) | [`flux-exact-reference.env`](flux-exact-reference.env) | [`flux-serve.argv`](flux-serve.argv)<br/>*(plain argv without `--speculative`; `ATLAS_DISABLE_SPECULATION=1`)* | `16,384` | Reference exact W4A16/BF16-QSA (`GDN_NORM_FP8=1`) | `33.4 – 33.7 tok/s` plain greedy (`DISABLE_SPECULATION=1`) | `8,800+ MiB` | Differential 0-ULP, teacher-forced NLL, and `LOSSLESS 10/10` token-identity auditing baseline. |

---

## 2. Operating Mode Details

### 2.1 `flux-v9-max-prefill.env` (`--mode max-prefill` / `w16384`)

Default production profile for single-stream or high-velocity agentic workloads on a dedicated 128 GB GB10 node where maximum prefill throughput (`> 6,000 tok/s` at 16K and `> 5,680 tok/s` at 128K) and `57.4 tok/s` lossless `K=5` speculative decoding are required simultaneously:
- **16,384-Token Wide Chunking:** `FLUX_QWEN4_PREFILL_WIDE=16384` paired with `--max-prefill-tokens 16384`.
- **Compact Transposed W4A8 MoE (`v28q=2` + `v17q`):** `ATLAS_QWEN4_PREFILL_MOE_W4A8_WIDE=7`, `ATLAS_QWEN4_PREFILL_MOE_V28Q=2`, `ATLAS_QWEN4_PREFILL_MOE_FUSED_SORT_PLAN=1`, `ATLAS_QWEN4_PREFILL_ROUTER_GEMM_PIPE=1`, `ATLAS_QWEN4_PREFILL_SHARED_W4A8_PRETRANSCODE=1`.
- **4-Stream mHC Roofline Pipeline:** `ATLAS_QWEN4_PREFILL_MHC_ROOF=t128`, `ATLAS_QWEN4_PREFILL_HC_FUSED_PREP=1`, `ATLAS_QWEN4_PREFILL_MHC_GATEPRE=1`.
- **QSA Sparse Attention Pipeline:** `ATLAS_QWEN4_PREFILL_QSA_FP8=qkpv`, `ATLAS_QWEN4_PREFILL_QSA_PIPE=1`, `ATLAS_QWEN4_PREFILL_QSA_PLAN_FUSED=1`, `ATLAS_QWEN4_PREFILL_QSA_PREP_FUSED=1`, `ATLAS_QWEN4_PREFILL_OPROJ_AMAX_FUSED=1`.
- **GDN Linear Attention Fusions:** `ATLAS_QWEN4_PREFILL_GDN_NORM_FP8=2`, `ATLAS_QWEN4_PREFILL_GDN_NORM_FP8_V2=1`, `ATLAS_QWEN4_PREFILL_GDN_CONV_V3=1`, `ATLAS_QWEN4_PREFILL_FP8_ACT_AMAX_FUSED=1`.
- **4-Ring Sector-Aligned NVMe PLE Streaming:** `ATLAS_PLE_FAST_PLAN=1`, `ATLAS_PLE_SECTOR_READ=1`, `ATLAS_PLE_READ_RINGS=4`, `ATLAS_QWEN4_PLE_EARLY_ISSUE=1`, `ATLAS_QWEN4_PLE_PREFILL_ASYNC=1`.
- **Watchdog-Safe FP8 Weight Cache:** `ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_WEIGHT_CACHE=1` with `ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_WEIGHT_CACHE_RESERVE_GB=6`.

### 2.2 `flux-v9-w4096-interactive.env` (`--mode interactive-w4096` / `w4096`)

Designed for interactive serving, multi-tool developer environments, or memory-constrained deployments where keeping `~12 GiB` (`11,907 – 12,695 MiB`) of free unified LPDDR5X memory is prioritized over the final `~12%` of 16K-chunk prefill speed:
- Uses the same `sm_121a` kernel stack as `flux-v9-max-prefill.env`, with prefill chunk width capped to **`4,096` tokens** (`FLUX_QWEN4_PREFILL_WIDE=4096`) and paired with [`flux-serve-w4096.argv`](flux-serve-w4096.argv) (`--max-prefill-tokens 4096`).
- Sustains **`5,266.8 tok/s` median (`5,276.3` peak) @ 8K**, **`5,164.5` @ 16K**, **`5,116.2` @ 32K**, and **`5,016.1` @ 64K**.

### 2.3 `flux-fast-prefill.env` (`--mode fast-prefill`)

Pure prefill-only batch processing profile (prompt cache warming, rubric scoring, or single-token logprob extraction) where `K=5` speculative decoding is not active:
- Uses `FLUX_QWEN4_PREFILL_GDN_NORM_FP8=3` (store-warp E4M3 normalization) alongside MXFP8 MoE slot quantization (`ATLAS_QWEN4_PREFILL_MOE_SLOT_MXFP8=1`), `ATLAS_QWEN4_PREFILL_MOE_XQ_REUSE=1`, and `ATLAS_QWEN4_PREFILL_LAST_LAYER_CROP=1`.
- Because `GDN_NORM_FP8=3` uses an asynchronous store-warp reduction that is not state-deterministic across multi-row speculative verify rollbacks, unified serving with `K=5` MTP uses `GDN_NORM_FP8=2` + `GDN_NORM_FP8_V2=1` (`flux-v9-max-prefill.env`).

### 2.4 `flux-k5-lossless-decode.env` (`--mode k5-decode`)

Decode-dominated interactive coding, reasoning, and long-generation sessions (`max_tokens >= 512`) where preserving maximum L2 cache residency and unified memory headroom (`FAST_PROJ_FP8_WEIGHT_CACHE=0`, `PREFILL_WIDE=8192`) for the 4-draft native MTP proposer and 5-row target verifier is paramount:
- **Built-in Checkpoint `mtp.*` Head Support:** Paired argv files ([`flux-serve.argv`](flux-serve.argv) and [`flux-serve-w4096.argv`](flux-serve-w4096.argv)) pass `--speculative --num-drafts 4 --mtp-vocab 0 --mtp-quantization bf16`, loading the checkpoint's built-in `mtp.*` weights directly from `JMNI-Labs/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid` (native NVFP4 MTP), `Blackfrost-AI/CYBER-FROST-3.8-NVFP4-V2` (per-expert BF16 MTP), or `nvidia/Qwen3.8-Flash-Next-NVFP4` (packed BF16 MTP) without requiring a separate MTP download (or pass `--mtp-from-path <DIR> --mtp-vocab 96000` to override with an external sidecar).
- **Device-Chained `K=5` MTP Proposer:** `ATLAS_QWEN4_MTP_CHAIN=1`, `ATLAS_QWEN4_K5_HYBRID=1`, `ATLAS_QWEN4_K5_NATIVE_MOE=1`, `ATLAS_QWEN4_MTP_CONF=0.6`, `ATLAS_QWEN4_MTP_CONF_EARLY=1`.
- **Batched 5-Row Target Verification:** `ATLAS_QWEN4_K5_BATCH_ATTN_QKV=1`, `ATLAS_QWEN4_K5_BATCH_GDN=1`, `ATLAS_QWEN4_K5_BATCH_SSM=1`, `ATLAS_QWEN4_K5_BATCH_HYPER=1`, `ATLAS_QWEN4_K5_BATCH_CONV=1`, `ATLAS_QWEN4_PLE_VERIFY_BATCH=1`, `ATLAS_QWEN4_VERIFY_FAST_HOST=1`.
- **Lossless 10/10 Guarantee:** `2,560 / 2,560` greedy tokens match plain decode (`ATLAS_DISABLE_SPECULATION=1`, omitting `--speculative`) bit-for-bit across all 10 verification prompts (`0` to `64K` context).

### 2.5 `flux-estonia-136k.env` (`--mode long-context-136k`)

Ultra-long-context profile (`128K – 144K` tokens), including the `llm-inference-bench` (`v0.7.9`) `--test-profile estonia` 136,584-token 6-hop reasoning benchmark:
- **10 GiB FP8 Weight Cache Reserve Guard:** `ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_WEIGHT_CACHE_RESERVE_GB=10` stops FP8 projection weight caching once `MemAvailable` approaches `10 GiB`, guaranteeing `8,412 – 8,431+ MiB` free memory floor even with a 144K KV cache allocated.
- **Long-Context QSA Verify Stability:** `ATLAS_QWEN4_QSA_VERIFY_ROWS=0` and `ATLAS_CTX_OVERFLOW_TRUNCATE=0` (fail-closed if prompt exceeds `--max-seq-len`).
- **Required CLI Argv Overrides:** When launching the server for `> 131,072` tokens, pass `--max-seq-len 143360` (or `147456`) and `--request-timeout 1800`.

### 2.6 `flux-exact-reference.env` (`--mode exact-reference`)

Ground-truth numerical verification, 0-ULP kernel differential testing, teacher-forced NLL/top-1 quality gates, and generating the reference greedy token stream for `LOSSLESS 10/10` speculative decode audits:
- Forces `ATLAS_DISABLE_SPECULATION=1`, `ATLAS_QWEN4_K16_EXACT=1`, `ATLAS_QWEN4_PREFILL_HC_EXACT=1`, `ATLAS_QWEN4_PREFILL_SSM_EXACT=1`, `ATLAS_QWEN4_PREFILL_GDN_NORM_FP8=1`, and BF16 QSA attention (`ATLAS_QWEN4_PREFILL_QSA_FP8` unset).

---

## 3. Usage Examples (`scripts/flux_env_bridge.sh`)

```bash
# 1. Preview the translated FLUX_* -> ATLAS_* environment for --mode max-prefill
bash scripts/flux_env_bridge.sh --mode max-prefill --print-env

# 2. Start from interactive-w4096 mode and override reserve memory using FLUX_* syntax
FLUX_QWEN4_FAST_PROJ_FP8_WEIGHT_CACHE_RESERVE_GB=9 \
  bash scripts/flux_env_bridge.sh --mode interactive-w4096 --print-env

# 3. Source the flagship v9 max-prefill recipe into the current shell
source scripts/flux_env_bridge.sh --mode max-prefill
```

---

## 4. Additional Recipes in `recipes/`

| Category | Files |
|---|---|
| **Next-Gen & High-Concurrency (GB10)** | `flux-nextgen-max-prefill-decode.env`, `flux-nextgen-1m-nvfp4.env`, `flux-nextgen-c128-concurrency.env`, `flux-qwen4-native-262k-bf16.env`, `flux-qwen4-native-262k-w4096-bf16-owner16.env`, `flux-qwen4-native-concurrency-c8.env`, `flux-qwen4-native-concurrency-c8-owner16.env` |
| **Multi-Spark Clusters (`2×`/`3×`/`4×`)** | `2x-spark-cluster.env`, `2x-spark-tp2-disagg.env`, `3x-spark-pp3-disagg.env`, `4x-spark-cluster.env` |
| **Model-Specific Profiles** | `flux-qwen38-27b-dflash2.env`, `flux-dsv41-k124.env`, `flux-gemma4-31b.env`, `flux-glm53-exl3.env` |
| **Discrete Blackwell & Experimental Hardware** | `nvidia-blackwell-sm120.env` (SM120 discrete GPUs), `nvidia-hopper-sm90.env` *(experimental placeholder)* |
| **Schema-1 TOML Recipe Contracts** | `qwen38-flash-next-qad5500-hybrid-default.toml`, `qwen38-flash-next-qad5500-hybrid-c1-harness.toml`, `qwen38-flash-next-fp8-dense.toml`, `qwen38-flash-next-native-dense.toml` |
| **Paired CLI Argv Templates** | `flux-serve.argv`, `flux-serve-w4096.argv` |

> **Note on sealed R15/R16/R17 qualification recipes:** `flux-v9-max-prefill.env`, `flux-v9-w4096-interactive.env`, `flux-qwen4-native-262k-bf16.env`, `flux-qwen4-native-262k-w4096-bf16-owner16.env`, `flux-qwen4-native-concurrency-c8.env`, and `flux-qwen4-native-concurrency-c8-owner16.env` are SHA-256-pinned historical qualification contracts under `repro-kit/`. When loading them via `scripts/flux_env_bridge.sh`:
> - Use `--mode 256k` for `flux-qwen4-native-262k-bf16.env` and `--env-file recipes/flux-qwen4-native-262k-w4096-bf16-owner16.env` paired with `recipes/flux-serve-w4096.argv` (`--max-prefill-tokens 4096`) for the W4096 owner16 profile.
> - Live `ATLAS_QWEN4_PREFILL_WIDE` admits `0 / 2048 / 4096 / 8192 / 16384 / 32768` (`crates/spark-server/src/main_modules/serve_phases/kv_cache.rs`).
