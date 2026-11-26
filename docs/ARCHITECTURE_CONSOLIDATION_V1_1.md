# Flux Engine v1.1 — Architecture Consolidation

This document describes the **Flux Engine v1.1 Architecture Consolidation** effort,
which systematically addresses external systems-review feedback around
configuration-surface clarity, CUDA source deduplication, Rust monolith
decomposition, and kernel shape templatization — without altering served
numerics, breaking shipped recipes, or regressing throughput.

---

## 1. Consolidation Program Overview

The v1.1 consolidation runs across four orthogonal engineering tracks with strict
file ownership and machine-verified regression gates:

| Track | Focus Area | Key Deliverable | Verified Impact (v1.0.3 → v1.1) |
|---|---|---|---|
| **Track A** | CUDA `.cu` Deduplication | Layer-2 `kernels/gb10/nvfp4/` hoist + 14 canonical headers in `kernels/gb10/common/shared/` + thin `#include` shims; zero-ref legacy kernel removal | `kernels/gb10/`: **-32,554 net LOC** (`-40,422` `.cu` LOC, `+7,868` shared `.cuh` LOC), **15 duplicate groups (34 copies) → 0**, 2 zero-ref `.cu` deleted; **0 stem/symbol diffs** across all 57 `(hw, model, quant)` targets & 153 `nvcc --ptx` compiles (`sm_121f`/`sm_120a`/`sm_103a`) |
| **Track B** | Env-Var Governance & `FluxConfig` | Typed 40-knob Tier-1 `FluxConfig` surface, `EnvTier` catalog boundary, lexer-aware `env_inventory.py --check`, dead-reader pruning | `env_catalog.tsv`: **1,275 → 1,189** (`DEBUG` **150 → 54**, `-96`; **111** dead/stale/false-positive/test-only entries removed; **25** previously-hidden production readers recovered; **40** curated Tier-1 public knobs vs **1,155** Tier-2 internal catalog entries) |
| **Track C** | Rust Monolith Decomposition | Rust 2018+ modular decomposition of the 4 largest `.rs` files (`dense_ffn`, `verify_qwen4_batched_step`, `qwen4_mtp`, `noise_pass`) | **35,175 LOC** across 4 monoliths (max **11,867 LOC**) → 49 cohesive modules, **max 1,471 LOC** (`-87.6%` peak file size); zero API/behavior changes, zero `env_catalog.tsv` drift, `clippy --workspace --all-targets -- -D warnings` clean |
| **Track D** | CUDA Kernel Modularization & Shape Templatization | Canonical `Qwen4MoeGeometry<Hidden, Inter, NumExperts, TopK, MhcStreams, MhcRank>` + `Qwen4Ts4eLayout` in `qwen4_geometry.cuh`; split 4 hottest Flash-Next `.cu` monoliths into 26 `detail/*.cuh` headers | Top-level Flash-Next `.cu`: **18,281 → 571 LOC** (`v28` `8,175 → 52`, `hc_combine` `6,614 → 25`, `v32` `1,815 → 272`, `v33` `1,677 → 222`; all `.cuh` ≤ **1,398 LOC**); **408 / 408** `(file, arch, entry)` `ptxas -v` rows **0 diffs / 0 regressions / 0 B spills** on `sm_121a`/`sm_120a`/`sm_103a`; live GB10 parity **4/4 PASS** (`cos = 1.000000`) |

---

## 2. Two-Tier Environment Configuration Architecture (Track B)

In v1.0.3, [`env_catalog.tsv`](../crates/spark-server/src/recipe_contract/env_catalog.tsv)
listed **1,275** `ATLAS_*` / `FLUX_*` environment literals flatly alongside each
other, mixing operator-facing knobs (bind port, API key, memory reserve, KV cache
precision, concurrency) with internal kernel-route switches, recipe-pinned
numerics gates, and ~90 test/bench/example-only literals.

In v1.1, Flux Engine formalizes a machine-enforced **Two-Tier Configuration
Architecture**:

```
┌───────────────────────────────────────────────────────────────────────────┐
│ Tier 1 — Public Operator Surface (`spark_server::flux_config::FluxConfig`)│
│   40 curated `FLUX_*` knobs (28 runtime, 10 launcher, 1 build, 1 recipe) │
│   Typed enums, bounds & cross-field validation, redacted secret `Debug`, │
│   deterministic `from_map()` parsing, and single-source spec table.      │
└───────────────────────────────────┬───────────────────────────────────────┘
                                    │ `Catalog::tier()` / `tier_counts()`
┌───────────────────────────────────▼───────────────────────────────────────┐
│ Tier 2 — Internal Recipe & Kernel Switches (`EnvTier::InternalRuntime`)   │
│   1,155 cataloged engine switches governed by fail-closed `recipe.toml`  │
│   contracts (`NUMERICS`, `ROUTE`, `RESOURCE`, `OBSERVABILITY`, `DEBUG`). │
│   Operators select named profiles (`FLUX_MODE`) or pinned recipes        │
│   (`FLUX_RECIPE`) rather than hand-tuning Tier-2 selectors.              │
└───────────────────────────────────────────────────────────────────────────┘
```

### Key Architectural Guarantees

1. **Single Typed Public Surface ([`flux_config.rs`](../crates/spark-server/src/flux_config.rs)):**
   - Exposes `FluxConfig::from_env()` and deterministic `FluxConfig::from_map(&BTreeMap<String, String>)` returning `Result<FluxConfig, FluxConfigError>`.
   - Enforces domain enums (`KvCacheDtype`, `NativeBatchMode`, `MemWatchMode`, `MtpSource`, `DflashLookupMode`, `LoadDebugMode`, `SettingsFileMode`), numeric range checks, cross-field invariants (e.g., `FLUX_KV_POOL_TOKENS <= FLUX_KV_MAX_TOKENS`, `FLUX_QWEN4_NATIVE_BATCH=1` requires `FLUX_CONCURRENCY <= 32`), and `Debug` redaction for `FLUX_API_KEY` (`SecretString`).
   - Defines `PUBLIC_OPERATOR_VARS: &[PublicVarSpec]` as the canonical metadata table consumed by docs, tests, and the recipe-contract catalog.
2. **Exact Alias Parity:**
   - Preserves startup `FLUX_*` → `ATLAS_*` alias semantics (`main::flux_env_alias_key` and `flux_config::flux_env_alias_key`): when both spellings are present, the canonical `ATLAS_*` value wins. Shorthand aliases (`FLUX_PREFILL_WIDE` → `ATLAS_QWEN4_PREFILL_WIDE` and `FLUX_QWEN4_FAST_PROJ_FP8_WEIGHT_CACHE_RESERVE_GB` → `ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_WEIGHT_CACHE_RESERVE_GB`) resolve consistently in both layers.
3. **Machine-Enforced Catalog Boundary ([`catalog.rs`](../crates/spark-server/src/recipe_contract/catalog.rs)):**
   - Adds `EnvTier::{PublicOperator, InternalRuntime}`, `Catalog::tier(&self, name: &str) -> Option<EnvTier>`, and `Catalog::tier_counts(&self) -> (usize, usize)` while keeping `env_catalog.tsv` and `RECIPE_RECEIPT` JSON 100% backward compatible.
   - Unit tests in [`tier_tests.rs`](../crates/spark-server/src/recipe_contract/tier_tests.rs) assert that every `RuntimeEngine` Tier-1 variable in `PUBLIC_OPERATOR_VARS` exists in `Catalog::builtin()`, resolves to `EnvTier::PublicOperator`, matches its declared `EnvClass`, and agrees on alias mapping with `main.rs`.
4. **Zero-Drift Inventory Tooling ([`env_inventory.py`](../repro-kit/tools/env_inventory.py)):**
   - Upgraded to a Rust lexer-aware `#[cfg(test)]` mask (fixing brace-balance drift across raw strings/comments), automatic exclusion of trailing-underscore prefix fragments, standalone `tracing::*!` / `log::*!` event tags, exclusion tables (`META_KEYS`, `NON_NUMERIC_ENV_PREFIXES`), and test/bench/example-only literals.
   - Added suffix-helper discovery (`env_setting`, `mem_watch`, `think_nudge`) so dynamic `FLUX_<suffix>` / `ATLAS_<suffix>` production `NUMERICS` and `RESOURCE` readers cannot evade catalog tracking.
   - Supports `--check` for non-mutating drift detection in CI.

---

## 3. Tier-1 Public Operator Surface (`FluxConfig` — 40 Variables)

### 3.1 Target Hardware, Model, Mode & Contract (7)

| Canonical `FLUX_*` Name | `ATLAS_*` Alias | Scope | Class | Default | Description |
|---|---|---|---|---|---|
| `FLUX_TARGET_HW` | `ATLAS_TARGET_HW` | BuildTime | `RESOURCE` | `gb10` | Compile-time target GPU/backend architecture (`gb10`, `rtx6000`, `sm120`, `metal`) |
| `FLUX_TARGET_MODEL` | `ATLAS_TARGET_MODEL` | RuntimeEngine | `RESOURCE` | `unset` | Target kernel model family selector and transform-cache key |
| `FLUX_TARGET_QUANT` | `ATLAS_TARGET_QUANT` | RuntimeEngine | `RESOURCE` | `unset` | Target weight quantization family (`nvfp4`, `fp8`, `bf16`, `exl3`) |
| `FLUX_MODEL` | *(launcher)* | Launcher | `RESOURCE` | `auto` | Path to the local model checkpoint directory |
| `FLUX_MODE` | *(meta)* | RuntimeEngine | `OBSERVABILITY` | `auto` | Named operating profile (`auto`, `max-prefill`, `interactive`, `fast-prefill`, `decode`, `long-context`, `exact`) |
| `FLUX_RECIPE` | *(contract)* | ContractControl | `RESOURCE` | `unset` | Path to a fail-closed `recipe.toml` contract enforced at startup |
| `FLUX_WEIGHT_CACHE_DIR` | `ATLAS_WEIGHT_CACHE_DIR` | RuntimeEngine | `RESOURCE` | `auto` | Directory for persistent post-transform weight cache shards |

### 3.2 Concurrency, Batching & Sequence Limits (6)

| Canonical `FLUX_*` Name | `ATLAS_*` Alias | Scope | Class | Default | Description |
|---|---|---|---|---|---|
| `FLUX_MAX_SEQ_LEN` | *(launcher)* | Launcher | `RESOURCE` | `mode-dependent` | Maximum context length in tokens (`--max-seq-len`) |
| `FLUX_CONCURRENCY` | *(launcher)* | Launcher | `RESOURCE` | `1` | Maximum concurrent sequences (`--max-batch-size` / `--max-num-seqs`) |
| `FLUX_MAX_BATCH_TOKENS` | `ATLAS_MAX_BATCH_TOKENS` | RuntimeEngine | `RESOURCE` | `unset` | Maximum combined tokens scheduled in a single forward batch |
| `FLUX_QWEN4_PREFILL_WIDE` | `ATLAS_QWEN4_PREFILL_WIDE` | RuntimeEngine | `ROUTE` | `unset` | Prefill chunk width in tokens (`4096`, `8192`, `16384`; shorthand `FLUX_PREFILL_WIDE`) |
| `FLUX_QWEN4_NATIVE_BATCH` | `ATLAS_QWEN4_NATIVE_BATCH` | RuntimeEngine | `ROUTE` | `auto` | Qwen4 C2–C32 native batched decode mode (`auto`, `1`=require native, `0`=serial) |
| `FLUX_QUEUE_TIMEOUT` | *(FLUX-native)* | RuntimeEngine | `RESOURCE` | `0` | Maximum seconds a request may wait in the admission queue (`0` = disabled) |

### 3.3 KV Cache & VRAM/UMA Memory Headroom (9)

| Canonical `FLUX_*` Name | `ATLAS_*` Alias | Scope | Class | Default | Description |
|---|---|---|---|---|---|
| `FLUX_KV_CACHE_DTYPE` | *(launcher)* | Launcher | `NUMERICS` | `auto` | KV cache precision (`auto`, `bf16`, `fp8`, `turboquant`, `nvfp4`) |
| `FLUX_KV_POOL_TOKENS` | *(launcher)* | Launcher | `RESOURCE` | `auto` | Explicit KV pool token capacity passed to `--kv-pool-tokens` |
| `FLUX_KV_MAX_TOKENS` | *(FLUX-native)* | RuntimeEngine | `RESOURCE` | `unset` | Hard runtime ceiling on allocated KV pool tokens |
| `FLUX_KV_POOL_UNCAPPED` | *(FLUX-native)* | RuntimeEngine | `RESOURCE` | `0` | Allow KV pool sizing to bypass the default conservative token cap |
| `FLUX_GPU_MEMORY_UTILIZATION` | *(launcher)* | Launcher | `RESOURCE` | `0.55` | Fraction of GPU/UMA memory budgeted for runtime arenas (`<=0.55` on GB10 UMA) |
| `FLUX_MEM_WATCH` | `ATLAS_MEM_WATCH` | RuntimeEngine | `RESOURCE` | `auto` | In-process `MemAvailable` floor watchdog (`auto`, `on`, `off`) |
| `FLUX_MEM_FLOOR_MIB` | `ATLAS_MEM_FLOOR_MIB` | RuntimeEngine | `RESOURCE` | `6656` | `MemAvailable` floor in MiB before watchdog load-abort or serve-503 refusal |
| `FLUX_MEMAVAILABLE_FLOOR_MIB` | `ATLAS_MEMAVAILABLE_FLOOR_MIB` | RuntimeEngine | `RESOURCE` | `auto` | Startup memory-admission free-RAM floor in MiB |
| `FLUX_QWEN4_FAST_PROJ_FP8_WEIGHT_CACHE_RESERVE_GB` | `ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_WEIGHT_CACHE_RESERVE_GB` | RuntimeEngine | `RESOURCE` | `7` | Unified-memory reserve in GiB below which FP8 weight caching stops allocating |

### 3.4 Speculative Decoding — MTP & DFlash2 (6)

| Canonical `FLUX_*` Name | `ATLAS_*` Alias | Scope | Class | Default | Description |
|---|---|---|---|---|---|
| `FLUX_DISABLE_SPECULATION` | `ATLAS_DISABLE_SPECULATION` | RuntimeEngine | `ROUTE` | `0` | Disable all speculative decoding (MTP, DFlash, self/n-gram) |
| `FLUX_DRAFT_MODEL` | *(launcher)* | Launcher | `RESOURCE` | `auto` | Path to companion speculative drafter directory (or `none` to disable sidecars) |
| `FLUX_GAMMA` | *(launcher)* | Launcher | `ROUTE` | `auto` | Speculative draft tokens proposed per step (`4` for MTP K=5, `block_size - 1` for DFlash) |
| `FLUX_MTP_SOURCE` | `ATLAS_MTP_SOURCE` | RuntimeEngine | `ROUTE` | `auto` | MTP routed-expert weight source (`auto`, `checkpoint`, `sidecar`) |
| `FLUX_DFLASH_DRAFT_CAP` | `ATLAS_DFLASH_DRAFT_CAP` | RuntimeEngine | `ROUTE` | `unset` | Hard upper bound on DFlash proposed draft tokens per step |
| `FLUX_DFLASH_LOOKUP` | `ATLAS_DFLASH_LOOKUP` | RuntimeEngine | `ROUTE` | `off` | Opt-in DFlash lookup drafter mode (`off`, `sam`, `retrieval`) |

### 3.5 PLE RAM Staging & Offload (3)

| Canonical `FLUX_*` Name | `ATLAS_*` Alias | Scope | Class | Default | Description |
|---|---|---|---|---|---|
| `FLUX_PLE_MMAP` | `ATLAS_PLE_MMAP` | RuntimeEngine | `RESOURCE` | `0` | Memory-map PLE offload shards instead of `O_DIRECT` asynchronous reads |
| `FLUX_PLE_CACHE_MB` | `ATLAS_PLE_CACHE_MB` | RuntimeEngine | `RESOURCE` | `auto` | Host RAM staging cache budget in MiB for active PLE embeddings |
| `FLUX_PLE_PARKED_CACHE_MB` | `ATLAS_PLE_PARKED_CACHE_MB` | RuntimeEngine | `RESOURCE` | `auto` | Host RAM cache budget in MiB for parked PLE offload shards |

### 3.6 Network, Auth, Persistence & Observability (9)

| Canonical `FLUX_*` Name | `ATLAS_*` Alias | Scope | Class | Default | Description |
|---|---|---|---|---|---|
| `FLUX_HOST` | *(launcher)* | Launcher | `RESOURCE` | `127.0.0.1` | HTTP API server bind host address |
| `FLUX_PORT` | *(launcher)* | Launcher | `RESOURCE` | `8899` | HTTP API server listening TCP port |
| `FLUX_API_KEY` | `ATLAS_API_KEY` | RuntimeEngine | `RESOURCE` | `unset` | Bearer API key activating HTTP authentication (redacted in `Debug` output) |
| `FLUX_TRUST_PROXY` | `ATLAS_TRUST_PROXY` | RuntimeEngine | `RESOURCE` | `0` | Trust `X-Forwarded-For` / `X-Real-IP` headers for per-client rate limiting |
| `FLUX_SETTINGS_FILE` | `ATLAS_SETTINGS_FILE` | RuntimeEngine | `RESOURCE` | `off` | Persistent runtime settings JSON file (`off`, `on`, or explicit path) |
| `FLUX_HOME` | `ATLAS_HOME` | RuntimeEngine | `RESOURCE` | `auto` | Base state and configuration directory for Flux Engine |
| `FLUX_DEBUG` | *(FLUX-native)* | RuntimeEngine | `DEBUG` | `0` | Enable structured rolling debug log file (`FLUX_DEBUG_FILE` / `_MAX_MB` / `_FILTER`) |
| `FLUX_LOAD_DEBUG` | *(FLUX-native)* | RuntimeEngine | `DEBUG` | `auto` | Load-phase diagnostic capture mode (`auto`, `always`, `off`) |
| `FLUX_PROFILE` | `ATLAS_PROFILE` | RuntimeEngine | `OBSERVABILITY` | `0` | Enable per-step latency and CUDA kernel profiling instrumentation |

---

## 4. Machine-Verified `env_catalog.tsv` Audit (Before vs After)

Regenerated directly from source via `python3 repro-kit/tools/env_inventory.py . <out-dir>` and verified with `--check`:

| Class | Baseline (v1.0.3) | Consolidated (v1.1) | Delta | Explanation |
|---|---:|---:|---:|---|
| `ROUTE` | 760 | 759 | `-1` | `-1` stale, `-5` non-reader tags/meta keys, `-2` `TRUST_PROXY` reclassified to `RESOURCE`, `+8` recently added active route toggles synced |
| `DEBUG` | 150 | 54 | `-96` | `-89` test/bench/example-only literals removed from runtime catalog, `-2` stale, `-5` real `LOG_TOOL_RAW`/`MEM_WATCH` vars upgraded to `OBSERVABILITY`/`RESOURCE` |
| `NUMERICS` | 132 | 142 | `+10` | `+10` previously hidden dynamic `THINK_NUDGE*` / `THINK_BUDGET_NUDGE` numerics gates recovered; `*_TARGET_LMHEAD_MODE` pinned in `overrides.tsv` |
| `RESOURCE` | 132 | 142 | `+10` | `-2` non-reader prefix/meta keys (`ATLAS_PORT`, `FLUX_ACTIVE_ARGV_FILE`), `+7` dynamic `SETTINGS_FILE`/`ADMIN_TOKEN_FILE`/`MEM_FLOOR_*` readers recovered, `+5` upgraded from `DEBUG`/`ROUTE` |
| `OBSERVABILITY` | 101 | 92 | `-9` | `-3` dead `ATLAS_GDN_DUMP*` source readers pruned, `-1` stale `ATLAS_QWEN4_VERIFY_DUMP`, `-7` non-reader log tags & prefix strings removed, `+2` `LOG_TOOL_RAW` recovered from fixed `test_line_mask` |
| **Total Runtime Catalog** | **1,275** | **1,189** | **`-86`** | **111** removed, **25** recovered/synced, **7** accurately reclassified |

### Tier Partition of the Runtime Catalog (`Catalog::builtin().tier_counts()`)

| Governance Tier | Catalog Keys | Share of Catalog | Notes |
|---|---:|---:|---|
| **Tier 1 (`EnvTier::PublicOperator`)** | **34** | **2.9%** | Backed by 28 `RuntimeEngine` specs in `PUBLIC_OPERATOR_VARS` (6 expose direct dual `FLUX_*`/`ATLAS_*` readers; the remaining 12 Tier-1 specs are launcher, build-time, or contract-control knobs) |
| **Tier 2 (`EnvTier::InternalRuntime`)** | **1,155** | **97.1%** | Recipe-governed kernel/precision/diagnostic switches; operators interact with these through named modes and `recipe.toml` contracts rather than direct env mutation |
| **Total** | **1,189** | **100.0%** | Exact partition (`public + internal == catalog.len()`) enforced in unit tests |

### Summary of Pruned Entries (`111` Removed from Runtime Catalog)

1. **Confirmed-Dead Source Readers Pruned (3):**
   - `ATLAS_GDN_DUMP`, `ATLAS_GDN_DUMP_LAYERS`, `ATLAS_GDN_DUMP_N_SSM` removed from [`qwen3_ssm/debug.rs`](../crates/spark-model/src/layers/qwen3_ssm/debug.rs) (zero callers workspace-wide; `#![allow(dead_code)]` removed).
   - Unused duplicate helper `ffn_kgamma_wide_enabled()` removed from [`layers/mod.rs`](../crates/spark-model/src/layers/mod.rs) (zero callers; canonical reader remains in `dense_ffn.rs`).
2. **Stale Catalog Entries Removed (4):**
   - `ATLAS_DEBUG_SYNC_KERNELS`, `ATLAS_QWEN4_SEQSTATE_RAM_MARGIN_GIB`, `ATLAS_QWEN4_VERIFY_DUMP`, `ATLAS_QWEN4_VERIFY_TOKEN_MAJOR` (source readers already removed in prior commits).
3. **False-Positive Non-Reader Log Tags & Exclusion Tables Excluded (15):**
   - Tracing event tags (3): `ATLAS_PROFILE_RECEIPT`, `ATLAS_PROFILE_LOOKUP_OVERLAY`, `ATLAS_TS4E128_MOE_SCALES_READY`.
   - `NON_NUMERIC_ENV_PREFIXES` filter strings (8): `ATLAS_DUMP`, `ATLAS_HOST`, `ATLAS_LOG`, `ATLAS_METRICS`, `ATLAS_PORT`, `ATLAS_REQUEST_DUMP`, `ATLAS_STARTUP_LEDGER`, `ATLAS_TRACE`.
   - Launcher-only `META_KEYS` (4): `FLUX_ACTIVE_ARGV_FILE`, `FLUX_MODELS_ROOT`, `FLUX_OPT_OUT`, `FLUX_QUIET`.
   - Programmatic exclusion of trailing-underscore prefix fragments (`ATLAS_GLM53_`, `ATLAS_QWEN4_PLE_PARITY_`, `ATLAS_DDTREE_`, `ATLAS_DFLASH_`) and test dummies (`ATLAS_X`, `ATLAS_Y`) in `env_inventory.py`.
4. **Test / Bench / Example-Only Literals Excluded from Runtime Catalog (89):**
   - 51 `crates/*/examples/*` microgate harnesses, 8 `crates/*/benches/*` benchmark dimensions, and 30 `tests/` / `#[cfg(test)]` harnesses that have zero production readers in the serving binary (still audited in `env_inventory.csv` under `DEAD`).

---

## 5. CUDA Kernel Deduplication & Shared Layering (Track A)

`crates/atlas-kernels/build_hw.rs` resolves `.cu` sources through a three-layer priority hierarchy (`$hw/common/` → `$hw/$quant/` → `$hw/$model/$quant/`), with recursive quoted `#include` dependency tracking in `build_deps.rs`. Track A eliminated all **15 non-shim code-identical duplicate groups (34 redundant copies)** under `kernels/gb10/` (**`-32,554` net LOC** across `.cu` + `.cuh`):

- **Layer-2 NVFP4 Hoist (`w4a16_gemm.cu`):** Created [`kernels/gb10/common/shared/w4a16_gemm_nvfp4_v1.cuh`](../kernels/gb10/common/shared/w4a16_gemm_nvfp4_v1.cuh) and Layer-2 [`kernels/gb10/nvfp4/w4a16_gemm.cu`](../kernels/gb10/nvfp4/w4a16_gemm.cu), removing 10 identical per-model copies and reducing `gemma-4-31b` from 1,394 → 6 LOC while leaving non-`nvfp4` targets (`cb3`, `exl3`, `iq3`) and specialized overrides untouched.
- **Canonical Shared Headers (`kernels/gb10/common/shared/`):** Consolidated `moe_w4a16_grouped_gemm` (10 targets via 2 shared headers), `vision_encoder` (6 targets), `rms_norm` (7 targets via `rms_norm_standard_core.cuh` + Mamba/Gemma-4 extensions), `gated_delta_rule` (3 targets), `moe_shared_expert_fused_batch2` (2 targets), and all 5 Gemma-4 512-dim/activation kernel pairs.
- **Zero-Ref Dead Kernels Removed:** Deleted unreferenced `kernels/gb10/common/inferspark_prefill_v47.cu` (494 LOC) and `w4a16_dequant_prmt_proto.cu` (270 LOC), saving 38 unnecessary `nvcc` compilations across a 19-target build.
- **Bit-Exact Verification:** Across all **57** `(hw, model, quant)` targets (`gb10`, `rtx6000`, `gb300`), `stem_mismatches = 0` and `sym_mismatches = 0`; across all **153** direct `nvcc --ptx` compiles (`sm_121f`/`sm_120a`/`sm_103a`), `entry_mismatches = 0` with zero per-entry PTX instruction diffs.

---

## 6. Rust Monolith Decomposition (Track C)

The four largest Rust files (**35,175 LOC** total) were decomposed into cohesive Rust 2018+ module trees (`foo.rs` coordinator + `foo/*.rs` submodules + `*_tests.rs` test suites) with zero public API changes, zero performance regressions, and **every resulting file ≤ 1,471 LOC**:

| Module | Before (LOC) | After Layout | Peak File (LOC) |
|---|---:|---|---:|
| [`crates/spark-model/src/layers/dense_ffn.rs`](../crates/spark-model/src/layers/dense_ffn.rs) | 11,867 | 17 files (`dense_ffn.rs` = 86 LOC + 16 submodules) | 1,471 |
| [`crates/spark-server/src/scheduler/verify_qwen4_batched_step.rs`](../crates/spark-server/src/scheduler/verify_qwen4_batched_step.rs) | 8,504 | 12 files (`verify_qwen4_batched_step.rs` = 103 LOC + 11 submodules) | 1,026 |
| [`crates/spark-model/src/layers/qwen4_mtp.rs`](../crates/spark-model/src/layers/qwen4_mtp.rs) | 7,632 | 11 files (`qwen4_mtp.rs` = 46 LOC + 10 submodules) | 1,457 |
| [`crates/spark-model/src/layers/dflash_head/noise_pass.rs`](../crates/spark-model/src/layers/dflash_head/noise_pass.rs) | 7,172 | 9 files (`noise_pass.rs` = 60 LOC + 8 submodules) | 1,229 |

All static `include_str!` contract tests, microgate provenance SHA checks, and `env_inventory.py --check` invariants were updated without weakening a single assertion.

---

## 7. Flash-Next CUDA Modularization & Shape Templatization (Track D)

The four hottest Flash-Next Blackwell NVFP4 MoE and mHC kernels (**18,281 LOC** total) were factored into compile-time geometry traits and focused `.cuh` modules while keeping identical top-level `.cu` stems and `extern "C" __global__` entry symbols:

- **Compile-Time Shape Traits ([`qwen4_geometry.cuh`](../kernels/gb10/qwen3.8-flash-next/nvfp4/qwen4_geometry.cuh)):** Replaced hardcoded `2560`/`640`/`512`/`10`/`4`/`320` literals with `Qwen4MoeGeometry<Hidden, Inter, NumExperts, TopK, MhcStreams, MhcRank>` (with divisibility `static_assert` invariants and `Qwen4Ts4eLayout<N, K>`), instantiated for `Qwen4FlashNextShape = Qwen4MoeGeometry<2560, 640, 512, 10, 4, 320>` and compile-verified on alternate geometry `<3072, 768, 256, 8, 4, 384>`.
- **Modular Header Decomposition:**
  - `qwen4_moe_compact_t_k32_v28.cu`: **8,175 → 52 LOC** (`detail/moe_compact/`, 11 headers, max 1,398 LOC)
  - `qwen4_hc_combine_gemm.cu`: **6,614 → 25 LOC** (`detail/hc_combine/`, 9 headers, max 1,203 LOC)
  - `qwen4_moe_w4a4_v32.cu`: **1,815 → 272 LOC** (`detail/moe_w4a4/`, 2 headers, max 1,172 LOC)
  - `qwen4_moe_thin_sk_v33.cu`: **1,677 → 222 LOC** (`detail/moe_thin_sk/`, 2 headers, max 1,161 LOC)
  - Shared FP4/MMA/PDL primitives extracted to `detail/qwen4_nvfp4_common.cuh` (283 LOC).
- **Exact Compiler & Live GB10 Verification:** Automated `ptxas -v` diffing across all **408** `(file, arch, entry)` rows (`sm_121a`, `sm_120a`, `sm_103a`) reported **0 missing, 0 extra, 0 diffs, 0 regressions**, with **0 B stack / 0 B spills** on SM121a/SM120a. Formal PTX+SASS verification (`verify_ptx_isa_contracts.py`) proved 21 kernels / 377 entries (`0` odd `desc[UR]`), and live NVIDIA GB10 GPU parity tests passed **4/4** (`cos = 1.000000`, `max = 0.000e0`).

---

## 8. Related Documentation

- [ADR-0011: Fail-Closed Recipe Contract](adr/recipe-contract.md)
- [Operator Guide](OPERATOR-GUIDE.md)
- [Flags & Environment Reference](FLAGS.md)
- [Flux Engine Architecture](ARCHITECTURE.md)
