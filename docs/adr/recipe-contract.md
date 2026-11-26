# ADR-0011: Fail-closed recipe contract; retire numerics env gates

**Status:** Proposed
**Date:** 2026-10-30

## Context

Flux decides served arithmetic through environment variables. The tree has
1,100 distinct `ATLAS_*`/`FLUX_*` literals under `crates/` (tracked in
`docs/env-inventory/env_inventory_summary.md`). Nothing ties a
running server to the set it was qualified with, so a stray or missing
variable silently changes output:

- A precision census shows every prefill approximation (FP8 projections,
  FlashInfer GDN and attention, W4A8 MoE, FP8 QSA) is an env gate, and that
  a single run's numerics can only be reconstructed from its effective
  environment snapshot.
- An explicit `--kv-cache-dtype fp8` can be silently overridden to BF16 when
  a `MODEL.toml` default takes precedence over CLI flags unless verified
  after load.
- Pinning one declarative recipe per checkpoint/profile prevents silent
  numeric drift and operability incidents caused by untracked environment
  mutations.

## Decision

### 1. Classify every env literal

`repro-kit/tools/env_inventory.py` scans `crates/` and writes
`docs/env-inventory/env_inventory.csv` plus a summary. Each literal gets one
class (current counts from `docs/env-inventory/env_inventory_summary.md`):

| class | meaning | count |
|---|---|---|
| NUMERICS | changes served output, precision or kernel arithmetic | 105 |
| ROUTE | changes kernel or route but claims identical output; must be verified | 643 |
| RESOURCE | memory, threads, limits, paths, I/O shape | 112 |
| OBSERVABILITY | logs, receipts, dumps, profiles | 98 |
| DEBUG | debug or test-only (incl. read only by tests/examples/benches) | 134 |
| DEAD | no reader | 8 |

106 names are classified by hand in `env_inventory_overrides.tsv`. The rest
use name keywords; ROUTE is the fallthrough and means
"kernel selector, identity unverified", not "verified identical". Of the 105
NUMERICS gates, 48 have no `ENGAGED`/receipt line anywhere near their reader
or name, and 34 have an effective value (production recipe value, else code
default) that is not checkpoint-faithful: an approximation that is on, or a
faithful BF16/NATIVE/EXACT path that is off.

Build scripts (`build.rs` and the `build_*.rs` modules beside it) are
skipped, and `option_env!` is not a reader: those are compile-time only.
`ATLAS_TARGET_QUANT`/`ATLAS_TARGET_MODEL` are RESOURCE, because at runtime
they are only a transform-cache key, so a build shell's export does not
refuse a serve. A gate read through clap or a helper (for example
`ATLAS_ALLOW_SPECIAL_TOKENS_IN_CONTENT`) still counts, because any quoted
non-test occurrence is a reader.

The script also writes `crates/spark-server/src/recipe_contract/env_catalog.tsv`,
which the server embeds. A name missing from it has no reader in this build.

### 2. `recipe.toml`, schema 1

One file per checkpoint and serve shape. Unknown keys are refused.

- `[checkpoint]`: `model_type`, `model_id`, `revision`, and `sha256` per file
  (config, safetensors index, quant config, HYBRID.json, generation config,
  tokenizer, chat template).
- `[precision]`: the served format per module family as fields
  (`router`, `hyper_connection`, `gdn_proj`, `attention_proj`,
  `shared_expert`, `lm_head` in `nvfp4_requant | bf16 | fp8_e4m3`;
  `prefill_dense` in `fast_fp8 | native`; `routed_experts = nvfp4`;
  `gdn_state = fp32`). The loader maps them to today's selectors with the
  dense-native precedence rule: `ATLAS_QWEN4_DENSE_NATIVE=1` implies every knob
  unless that knob is `0`. `fp8_e4m3` is one selector, so the three families
  must agree.
- `[numerics_env]` and `[route_env]`: exact values for every other NUMERICS
  and ROUTE gate the recipe was qualified with. `route_policy` is `flag` or
  `refuse` for undeclared ROUTE gates.
- `[sampling]`: defaults, a thinking preset and ranges. Receipt-only in
  schema 1 (`"enforced": false`).
- `[limits]`: inclusive ranges for `--max-seq-len`, `--max-prefill-tokens`,
  `--max-batch-size`, and `qwen4_qsa`. `kv_cache_dtype`. `[speculation]`:
  `off` or `mtp` with `num_drafts`.
- `[operational_env]`: RESOURCE/OBSERVABILITY/DEBUG names with `values` or
  `min`/`max`.

### 3. Loader behavior

`--recipe PATH` or `FLUX_RECIPE=PATH` (the flag wins) turns the contract on.
The check runs in `serve()` before any process setup or weight load. The env
it sees is after `main`'s `FLUX_X` to `ATLAS_X` alias sync, and it folds names
with the same function. Startup refuses with one line,
`RECIPE_REFUSED <reason> ; <reason> ...`, listing every problem:

| reason | when |
|---|---|
| `invalid_recipe` | parse error, unknown key, bad sha, empty range, sampling default outside its range, partial FP8 plan, a gate in the wrong table, a declared name with no reader in this build |
| `unknown_numerics_env=X` | a NUMERICS gate is set that the recipe does not declare |
| `env_mismatch=X want got` | a declared gate has another value |
| `declared_env_missing=X` | a declared gate is unset, so the code default would apply |
| `unknown_route_env=X` | an undeclared ROUTE gate under `route_policy = "refuse"` |
| `non_utf8_env=X` | an `ATLAS_*`/`FLUX_*` name or value is not UTF-8 (the engine reads it with `var_os`) |
| `resolved_mismatch=field` | after load, the MODEL.toml-resolved KV dtype, `num_drafts` or prefill budget differs from the recipe |
| `model_swap_under_recipe` | any model swap while a recipe is active |
| `operational_env_out_of_range=X` | outside its declared values or range |
| `precision_mismatch=family` | the env-derived served format differs from the plan |
| `precision_unsupported_by_build=family` | the plan needs a selector this build cannot read |
| `flag_mismatch=--flag` | a serve flag outside the recipe's value or range, or a numerics flag (`--rope-yarn-factor`, `--rope-theta-override`, `--kv-high-precision-layers(-set)`, `--fp8-kv-calibration-tokens`, `--max-thinking-budget`, `--disable-thinking`, `--default-chat-template-kwargs`, `--allow-special-tokens-in-content`, `--kernel-target`, `--mtp-vocab`, `--mtp-quantization`) that differs from its `[cli]` pin or, undeclared, from its default |
| `checkpoint_pin=file` | missing file, sha256 mismatch, wrong `model_type`, or a hub id with no local directory |

On success it logs one line, `RECIPE_RECEIPT {json}`, with key order fixed:
`schema, recipe_id, recipe_sha256, checkpoint, precision, flags, sampling,
numerics_env, route_env, route_undeclared, operational_env,
ignored_env_no_reader, alias_shadowed`. Every NUMERICS gate appears with its value and a
source of `recipe` (pinned and matching) or `default` (unset, code default).
Operational and flagged ROUTE vars carry `env-allowed`.

Alias precedence: `main` copies `FLUX_X` onto `ATLAS_X` only when `ATLAS_X`
is unset, so the `ATLAS_` value is what the engine reads. The contract uses
the same rule. A differing `FLUX_` value is not refused; it is listed under
`alias_shadowed` in the receipt. A `FLUX_` name with its own reader and no
`ATLAS_` reader keeps its own spelling.

KV dtype, `num_drafts` and the prefill budget can change after the startup
check (MODEL.toml `default_kv_dtype`, `default_num_drafts`, budget
resolution). They are checked again once resolved, each logging a
`RECIPE_RESOLVED field=value` line or refusing. Model swaps are refused while
a recipe is active.

Without the flag and the env var nothing is read or checked; one
`RECIPE_LEGACY_MODE` line is logged. This branch changes no default.

Shipped recipes, each derived from a captured launch environment, not
hand-written:

- `qwen38-flash-next-qad5500-hybrid-default.toml`: plain `flux serve`
  (`FLUX_MODE=auto`, MTP K=4), from the v0.9.15.6 launcher's own dry-run plan.
- `qwen38-flash-next-qad5500-hybrid-c1-harness.toml`: the accuracy-harness
  launch (speculation off, segmented PLE graphs off, FP8 weight cache off),
  derived from the single-stream accuracy-harness launch plan.
- `qwen38-flash-next-native-dense.toml`: the harness env with
  `ATLAS_QWEN4_RELEASE_BF16_QKVZ=0` and the all-BF16 plan plus native
  prefill (`ATLAS_QWEN4_DENSE_NATIVE=1`). Recipe availability only; not GPU-qualified.
- `qwen38-flash-next-fp8-dense.toml`: the harness env with
  `ATLAS_QWEN4_DENSE_FP8=1` (`gdn_proj`, `attention_proj`, `shared_expert` served as
  `fp8_e4m3`) and `ATLAS_QWEN4_{ROUTER,HC,LMHEAD}_BF16=1` (`router`, `hyper_connection`,
  `lm_head` served as `bf16`). Recipe availability only; not GPU-qualified.

Values the launcher varies by host are ranges, not pins: the PLE I/O queue
depth (32 under a small memlock limit, else 256) and the FP8 weight cache
(a memory-for-speed cache of the same bytes).

## Migration: retiring NUMERICS gates

**Phase 0 (this branch).** Inventory, contract, receipt, two recipes. Opt-in.
No engine code reads the recipe; the recipe checks the env the launcher
already sets.

**Phase 1: launcher passes the recipe.** `flux` passes `--recipe` for
Flash-Next and stops emitting gates the recipe does not declare. Qualify
the hybrid recipe on the GPU once: the receipt must match the census
`env.effective`. Regenerate the catalog in CI and fail on drift, so a new
`env::var("ATLAS_...")` must be classified before merge. Flip
`route_policy` to `refuse` for qualified recipes.

**Phase 2: fold into the recipe (engine reads fields, not env).**

| gates | recipe field |
|---|---|
| `ATLAS_QWEN4_{ROUTER,HC,GDN,ATTN,SHARED,LMHEAD}_BF16`, `DENSE_NATIVE`, `DENSE_FP8`, `PREFILL_NATIVE` | `[precision]` per family |
| `PREFILL_FAST_PROJ*`, `PREFILL_FP8_ACT_*`, `PREFILL_OPROJ_AMAX_*` | `precision.prefill_proj = fp8_tensor / bf16 / native` |
| `PREFILL_GDN_FI`, `PREFILL_GDN_CONV_V2/V3`, `PREFILL_GDN_NORM_FP8*`, `PREFILL_SSM_EXACT` | `precision.prefill_gdn = flashinfer_f16 / exact_fp32` |
| `PREFILL_ATTN_FI` | `precision.prefill_attn = flashinfer_fa2 / exact` |
| `PREFILL_MOE_W4A8*`, `PREFILL_SHARED_W4A8*`, `PREFILL_MOE_SLOT_*`, `PREFILL_MOE_V28Q` | `precision.prefill_moe = w4a8_mxfp8 / w4a16` |
| `PREFILL_QSA_FP8*` | `precision.prefill_qsa = e4m3 / bf16` |
| `PREFILL_HC_*`, `PREFILL_MHC_FUSED*` | `precision.prefill_hc = fused / exact` |
| `--kv-cache-dtype`, `--kv-high-precision-layers`, MODEL.toml `default_kv_dtype` | `kv_cache_dtype` only (removes the masking trap) |
| `ATLAS_SSM_H_FP16` | delete; `gdn_state = fp32` is the only state format |
| `ATLAS_CTX_OVERFLOW_TRUNCATE`, thinking budget gates | `[limits]` / `[sampling]` |

Each fold lands with a receipt line per engaged path and a test that the
env name is no longer read.

**Phase 3: delete.** Remove gates for other models' experiments that no
recipe declares after two releases (the DSV41 FP8 policy, GEMMA4 FP32 and
BF16, `E2M1_*`, `UNSAFE_UNVERIFIED_*_KGAMMA_EXACT`, `FORCE_NVFP4_MOE`,
`FP32_RESIDUAL`). Move DEBUG and OBSERVABILITY names under one documented
`ATLAS_DIAG_*` namespace. Delete the 8 DEAD names now and the 134 DEBUG
names that only tests, examples or benches read once their harnesses move to
CLI flags.

**Silent fallbacks (refuse-or-receipt).** Some ROUTE gates fall back
without a per-request signal. `admit_or_suppress`
(`crates/spark-model/src/layers/qwen4_prefill_moe/attn16.rs:231-289`, selector
`ATLAS_QWEN4_PREFILL_ATTN_CORE16`) sends any prefill it does not admit down the
ordinary path and logs that once per process. A recipe that declares exact prefill can then
serve mixed paths with nothing in the request record. Rule: under a recipe
that declares the admitted path, a non-admitted prefill refuses the request.
The minimum is a per-request receipt field naming the path taken, plus a
fallback counter in `/metrics`. Every gate with a fallback gets the same
treatment, found by grepping `call_once` next to a downgrade.

**Phase 4: ROUTE identity.** For each ROUTE gate a recipe declares, add a
bit-exactness test (or reclassify it NUMERICS). Gates that fail become
`[precision]` fields. Then make the contract mandatory for Flash-Next
(no legacy mode) and make the recipe hash the `.flux-pack` key.

## Consequences

- A qualified serve is one artifact plus a receipt. Drift refuses at startup,
  before weights load, with the full list of reasons.
- Model swaps are refused under a recipe; Phase 1 can add a per-swap check.
- Optional checkpoint sidecars (`quantization_config.json`,
  `ple-offload/manifest.json`) are not yet required to be absent when
  unpinned; Phase 1 adds that.
- The catalog is generated from source. A build whose catalog is stale
  misclassifies new gates; Phase 1's CI drift check closes this.
- Keyword classification is a first pass. The manual table is authoritative,
  and every NUMERICS fold in Phase 2 re-reads the gate's code.
- Sampling is not enforced in schema 1. Enforcing omitted-field defaults
  changes request behavior and will be evaluated separately with dedicated A/B tests.
