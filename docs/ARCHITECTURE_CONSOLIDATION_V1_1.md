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
| **Track A** | CUDA `.cu` Deduplication | Shared common kernel headers / deduplicated Blackwell `.cu` sources | *[TBD — filled by Track A on merge]* |
| **Track B** | Env-Var Governance & `FluxConfig` | Typed 40-knob Tier-1 `FluxConfig` surface, `EnvTier` catalog boundary, lexer-aware `env_inventory.py --check`, dead-reader pruning | `env_catalog.tsv`: **1,275 → 1,189** (`DEBUG` **150 → 54**, `-96`; **111** dead/stale/false-positive/test-only entries removed; **25** previously-hidden production readers recovered; **40** curated Tier-1 public knobs vs **1,155** Tier-2 internal catalog entries) |
| **Track C** | Rust Monolith Decomposition | Modular decomposition of oversized model/scheduler modules (`dense_ffn`, `verify_qwen4_batched_step`, `qwen4_mtp`, `noise_pass`) | *[TBD — filled by Track C on merge]* |
| **Track D** | CUDA Kernel Shape Templatization | Parameterized shape templates replacing copy-pasted kernel instantiations | *[TBD — filled by Track D on merge]* |

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

## 5. Related Documentation

- [ADR-0011: Fail-Closed Recipe Contract](adr/recipe-contract.md)
- [Operator Guide](OPERATOR-GUIDE.md)
- [Flags & Environment Reference](FLAGS.md)
- [Flux Engine Architecture](ARCHITECTURE.md)
