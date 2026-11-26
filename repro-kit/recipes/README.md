<p align="left"><a href="../../README.md"><b>← Flux Engine</b></a> · <a href="../README.md"><b>Reproducibility Kit (<code>repro-kit/</code>)</b></a> · <b>Operating Mode Recipes (<code>repro-kit/recipes/</code>)</b></p>

# Reproducibility Kit Operating Recipes (`repro-kit/recipes/`)

Pinned `.env`, `.argv`, `.toml`, and `.json` profiles bundled inside `ghcr.io/theapathy/flux-engine:gb10` and invoked via `flux serve --mode <mode>` or `flux-bench serve --mode <mode>`. All `.env` and `.argv` files here are byte-identical mirrors of the repo-root [`recipes/`](../../recipes/README.md) directory (verified by `repro-kit/tests/test_recipe_mirror_parity.py`).

---

## 1. `flux-bench` Operating Modes & Single-Node Recipes

| File | Mode / Target | Description |
|---|---|---|
| [`flux-v9-max-prefill.env`](flux-v9-max-prefill.env) & [`flux-serve.argv`](flux-serve.argv) | `max-prefill` (`v9`, `w16384`) | Certified `6,082.2 tok/s` peak (`6,076.7` med) @ 16K prefill + `56.5 tok/s` (`57.4` peak) lossless `K=5` MTP decode. |
| [`flux-v9-w4096-interactive.env`](flux-v9-w4096-interactive.env) & [`flux-serve-w4096.argv`](flux-serve-w4096.argv) | `interactive-w4096` (`w4096`) | `5,276.3 tok/s` @ 8K (`5,164.5` @ 16K) prefill + `56.5 tok/s` lossless `K=5` decode with `11.9–12.7 GiB` free UMA floor. |
| [`flux-fast-prefill.env`](flux-fast-prefill.env) | `fast-prefill` | Pure prefill throughput profile (`GDN_NORM_FP8=3`, single-token decode). |
| [`flux-k5-lossless-decode.env`](flux-k5-lossless-decode.env) | `k5-decode` | Dedicated low-latency `K=5` MTP coding decode profile (`w8192`). |
| [`flux-estonia-136k.env`](flux-estonia-136k.env) | `long-context-136k` | Long-context `136,576`-token 6-hop needle profile (`RESERVE_GB=10`, `QSA_VERIFY_ROWS=0`, `5/5 PASS`). |
| [`flux-exact-reference.env`](flux-exact-reference.env) | `exact-reference` (`K=1`) | Non-speculative `K=1` greedy oracle for bit-exact differential verification (`ATLAS_DISABLE_SPECULATION=1`). |
| [`flux-nextgen-max-prefill-decode.env`](flux-nextgen-max-prefill-decode.env) | `nextgen` (`1x Spark`) | Unified `7K–8K` prefill (TMA pad, fused MoE, warm Marconi, `cuBLASLt` cache) + `81–88 tok/s` `0-ULP` lossless `K=5`/`K=7` decode (`1x1x1` fused GDN, `PLE` suffix graph, `PDL`, `WFET`) + `SLO` interleave. |
| [`flux-nextgen-1m-nvfp4.env`](flux-nextgen-1m-nvfp4.env) | `1m-nvfp4` (`1,048,576` tok) | Full `1M`-context single-Spark profile (`--rope-yarn-factor 4.0` unit `attention_factor`, 4-bit `NVFP4` KV pool, slabbed `QSA v3` selector). |
| [`flux-nextgen-c128-concurrency.env`](flux-nextgen-c128-concurrency.env) | `c128` (`C=2..128`) | `C=128` native batching & `C=2..128` two-frame speculative cycles (up to `256` verify rows/cycle, `64`-seq `propose_batch`, `12:1` ragged GQA, `-42.7 GiB` `decode_ring=0`). |

## 2. Qualification, Model-Specific & Multi-Node Recipes

| File | Mode / Target | Description |
|---|---|---|
| [`flux-qwen4-native-262k-bf16.env`](flux-qwen4-native-262k-bf16.env) | `256k` native qualification | Sealed native 262K BF16 KV qualification recipe (`--mode 256k` via `scripts/flux_env_bridge.sh`). |
| [`flux-qwen4-native-262k-w4096-bf16-owner16.env`](flux-qwen4-native-262k-w4096-bf16-owner16.env) | `256k` W4096 owner16 qualification | Sealed 262K BF16 W4096 qualification recipe with 16 MiB bounded transpose owners (pair with `flux-serve-w4096.argv`). |
| [`flux-qwen4-native-concurrency-c8.env`](flux-qwen4-native-concurrency-c8.env) | `native-concurrency` qualification | Frozen snapshot of `flux-v9-max-prefill.env` with `ATLAS_QWEN4_PREFILL_WIDE=0` for the plain native C2/C4/C8 gate (`--max-prefill-tokens 1024`, `--max-seq-len 2048`). |
| [`flux-qwen4-native-concurrency-c8-owner16.env`](flux-qwen4-native-concurrency-c8-owner16.env) | QAD5500 owner16 plain qualification | Dedicated C2/C4/C8 target-only candidate retaining bounded 16 MiB transpose owners, eager grouped loading and the 8192 MiB allocator floor. |
| [`flux-qwen38-27b-dflash2.env`](flux-qwen38-27b-dflash2.env) | `qwen3.8-27b` | Dense Qwen3.8-27B NVFP4 + DFlash2 profile (`--kv-high-precision-layers 8`). |
| [`flux-glm53-exl3.env`](flux-glm53-exl3.env) | `glm5.3-flash` | GLM-5.3-Flash EXL3 target-only profile. |
| [`flux-dsv41-k124.env`](flux-dsv41-k124.env) | `deepseek-v4.1` | DeepSeek-V4.1 NVFP4/MLA profile (`--max-seq-len 65536`). |
| [`flux-gemma4-31b.env`](flux-gemma4-31b.env) | `gemma-4-31b` | Gemma-4-31B-IT NVFP4 single-call prefill profile. |
| [`2x-spark-tp2-disagg.env`](2x-spark-tp2-disagg.env) | `2x-spark` (`TP=2` / `1P+1D` / `DP=2`) | `2× DGX Spark` 200GbE ConnectX-7 RoCEv2 profile supporting `TP=2` `(8, 24)` GDN + batched verify, `1P+1D` disaggregation, and sticky `flux-router`. |
| [`3x-spark-pp3-disagg.env`](3x-spark-pp3-disagg.env) | `3x-spark` (`PP=3` / `1P+2D` / `DP=3`) | `3× DGX Spark` `PP=3` `16`-layer (`4`-hybrid-block) pipeline ring + `1P+2D` disaggregated serving with `flux-router`. |
| [`2x-spark-cluster.env`](2x-spark-cluster.env) & [`4x-spark-cluster.env`](4x-spark-cluster.env) | `2x-spark` / `4x-spark` EP cluster | 200GbE ConnectX-7 RoCEv2 Expert-Parallel (`EP=2` / `EP=4`) cluster profiles. |
| [`nvidia-blackwell-sm120.env`](nvidia-blackwell-sm120.env) & [`nvidia-hopper-sm90.env`](nvidia-hopper-sm90.env) | Discrete `SM120` / `SM90a` | Discrete GPU profiles for RTX PRO 6000 Blackwell and H100 Hopper. |

## 3. Schema-1 TOML Contracts & JSON Metadata

| File | Kind | Description |
|---|---|---|
| [`qwen38-flash-next-qad5500-hybrid-default.toml`](qwen38-flash-next-qad5500-hybrid-default.toml) | Schema-1 TOML contract | Default QAD5500 hybrid serving contract for Qwen3.8-Flash-Next. |
| [`qwen38-flash-next-qad5500-hybrid-c1-harness.toml`](qwen38-flash-next-qad5500-hybrid-c1-harness.toml) | Schema-1 TOML contract | C1 harness contract for QAD5500 hybrid verification. |
| [`qwen38-flash-next-fp8-dense.toml`](qwen38-flash-next-fp8-dense.toml) | Schema-1 TOML contract | FP8-dense operating contract for Qwen3.8-Flash-Next. |
| [`qwen38-flash-next-native-dense.toml`](qwen38-flash-next-native-dense.toml) | Schema-1 TOML contract | Native-dense BF16 operating contract for Qwen3.8-Flash-Next. |
| [`drafter-profiles.json`](drafter-profiles.json) | JSON profile registry | Companion speculative drafter compatibility & gamma table consumed by `flux-autodetect`. |
| [`flux-profile-high-prefill-w16384.json`](flux-profile-high-prefill-w16384.json) & [`flux-profile-interactive-w4096.json`](flux-profile-interactive-w4096.json) | JSON profile metadata | Bundled profile descriptors for high-prefill (`w16384`) and interactive (`w4096`) serving contracts. |

> **Note on sealed qualification recipes:** `flux-v9-max-prefill.env`, `flux-v9-w4096-interactive.env`, `flux-qwen4-native-262k-bf16.env`, `flux-qwen4-native-262k-w4096-bf16-owner16.env`, `flux-qwen4-native-concurrency-c8.env`, and `flux-qwen4-native-concurrency-c8-owner16.env` are SHA-256-pinned historical qualification contracts. Live `ATLAS_QWEN4_PREFILL_WIDE` admits `0 / 2048 / 4096 / 8192 / 16384 / 32768` (`crates/spark-server/src/main_modules/serve_phases/kv_cache.rs`). See [`recipes/README.md`](../../recipes/README.md) for the full mode selection matrix and bridge usage.
