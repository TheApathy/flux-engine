# Serve-profile flag reference

Every `ATLAS_*` variable `serve.sh` sets (plus the build-time target selector
`ATLAS_TARGET_MODEL`), with the value used for the published decode result and
the description taken from the source that reads it. Where the source carries no
comment this table says so rather than guessing: those are flags whose behaviour
you should read at the use site before changing.

Values here are the measured configuration. Changing any of them invalidates the
comparison to the published number. Several are not bit-exact (see the numerics
section of the performance recipe).

| Flag | Value | What it does |
|---|---|---|
| `ATLAS_ACCEPT_FAST_ARGMAX` | `1` | Environment kill-switch for the raw-BF16 accept fast path (default ON). Set `ATLAS_ACCEPT_FAST_ARGMAX=0` to force every row through the full `process_seq_logits` walk when debugging a drift. |
| `ATLAS_ATTN_GATE_BATCHED` | `0` | Default-off exact gated-attention launch-batching candidate. For M=5..17 it replaces one sigmoid-gate launch per row with the already-loaded strided batched kernel, preserving BF16/FP32 arithmetic and traffic while saving 64–256 launches across 16 attention layers. Strict 0/1 parsing and an ENGAGED marker make qualification fail closed. |
| `ATLAS_ATTN_QKV_BATCHED` | `0` | *(no comment at the use site: read the source before changing)* |
| `ATLAS_ATTN_QKV_EXACT_M17_ASTAGE` | `0` | Default-off exact M=5..17 QG/dual-KV experiment, launcher alias `ATTN_QKV_M17_ASTAGE`. Both names accept exactly `0` or `1`; disagreeing aliases and explicit empty values fail closed. Cooperatively stages each 512-column raw-BF16 activation tile once per CTA for reuse by four output groups. |
| `ATLAS_ATTN_QKV_EXACT_STRIDED` | `1` | Writes QG/K/V directly to strided destinations instead of staging through scratch and scattering with per-row D2D copies. Bit-exact (same bytes to the same addresses; the exact kernels already deinterleave QG at the store), removing 3 copy launches per row plus their bandwidth. |
| `ATLAS_ATTN_QKV_FUSED` | `1` | Cached `ATLAS_ATTN_QKV_FUSED` lookup. When `1`/`true`, the batch-3 QKV projection writes directly into the interleaved `qkv_buf` layout and uses a single batched RMS norm launch instead of 6 separate ones. Default off for A/B safety. |
| `ATLAS_ATTN_QKV_SPLITK` | `4` | Split-K factor for attention QKV projection, clamped to `[2, 8]`. Raises SM occupancy and effective bandwidth on the narrow K/V weights. |
| `ATLAS_DDTREE_MAX_NODES` | `$((GAMMA + 1))` | Caps DDTree node allocation and forces `has_mtp` on whenever DFlash is active so the partial-accept rollback checkpoint pools exist. |
| `ATLAS_DDTREE_TREE_AWARE_VERIFY` | `0` | Enables tree-aware parent indirection during DDTree verification (`depth[t]` derived from `parent_ids` stashed by `set_ddtree_parent_ids`). Default off for flat chains. |
| `ATLAS_DDTREE_TREE_CONV_EXACT` | `0` | Enables tree-aware GDN Conv1D verification using device `parent_ids_dev` uploaded from `pending_tree_payload`, so each token loads state from `parent_ids[i]` instead of `i - 1`. Default off for flat chains. |
| `ATLAS_DDTREE_TREE_TOKENS_VERIFY` | `0` | Uses tree-ordered token IDs rather than flat `drafts` during non-flat DDTree verification. Identical to flat mode when `chain_only=1` with no branching. |
| `ATLAS_DDTREE_UNCAP` | `0` | Uncaps DDTree verification width above the flat `gamma` cap (raising `physical_drafts` beyond `ATLAS_DDTREE_MAX_NODES`). Default off. |
| `ATLAS_DFLASH_ASYNC` | `0` | Master gate for asynchronous DFlash proposal overlap (default OFF, cached). Companion `ATLAS_DFLASH_FUSED=1` overlaps drafter execution with SSM commit and KV reshape when async is enabled. |
| `ATLAS_DFLASH_ATTN_KGAMMA` | `1` | Batches DFlash drafter attention projections across `K=gamma` using transposed `nvfp4_t` weights (one-time `transpose_for_gemm` at load). Default off; cached via `OnceLock`. |
| `ATLAS_DFLASH_CTX_WINDOW` | `4096` | Controls how many captured target positions the drafter attends to per step (matching the drafter's trained prefix window rather than capping at `gamma`). |
| `ATLAS_DFLASH_DRAFT_CAP` | `$GAMMA` | *(no comment at the use site: read the source before changing)* |
| `ATLAS_DFLASH_DRAFT_SPLITK` | `8` | Split-K factor for DFlash drafter GEMMs (values `<= 1` disable; clamped to `<= 8` to match workspace sizing). Reassociates the K reduction loop, so bit-exactness is not guaranteed. |
| `ATLAS_DFLASH_ECHO` | `0` | Default-off echo-drafting / Jacobi salvage mode (`crates/spark-model/src/layers/dflash_head/echo.rs`). `0` keeps output byte-identical to standard DFlash. |
| `ATLAS_DFLASH_FFN_KGAMMA` | `1` | Batches DFlash drafter FFN projections across `K=gamma` using transposed `nvfp4_t` weights (one-time `transpose_for_gemm` at model build). Default off until A/B-validated. |
| `ATLAS_DFLASH_FREE_SLOTS` | `0` | Number of sibling branch slots to populate in DDTree free-slot mode (default `0`, byte-identical to flat `gamma=16`). |
| `ATLAS_DFLASH_FREE_SLOTS_TAIL` | `4` | Per-branch tail length carried on each free-slot sibling (`0` = bare 1-node fork leaves; default `4` to capture post-cliff continuation). |
| `ATLAS_DFLASH_KERNEL_PROFILE` | `0` | Enables separate kernel timing accumulation for the DFlash drafter's transformer layers (which are excluded from `ATLAS_FULL_PROFILE`). |
| `ATLAS_DFLASH_LM_HEAD_FP8` | `1` | Uses a pre-scaled FP8 E4M3 `lm_head` weight slice for the DFlash drafter's propose step, folding the compensating `1/s` into the final norm weight while preserving ~3 mantissa bits of logit fidelity. |
| `ATLAS_DFLASH_LM_HEAD_NVFP4` | `1` | Enables NVFP4 `lm_head` weight path for the DFlash drafter (superseded by FP8 when `ATLAS_DFLASH_LM_HEAD_FP8=1` is also set). |
| `ATLAS_DFLASH_NOISE_ONLY` | `1` | Restricts per-row drafter ops to the noise slice `[eff_ctx .. n_attn)`, leaving context-covering ops (`k_norm`, RoPE, attention) unchanged. |
| `ATLAS_DFLASH_PREFILL_PIPE` | `1` | Uses the byte-exact `cp.async` W4A16 pipeline for V3 large-M ingestion after target prefill (FC plus shared-input dual-output context K/V). Small-M decode retains existing kernels; fails closed if required symbols are stale. |
| `ATLAS_DFLASH_QUANT` | `nvfp4` | *(no comment at the use site: read the source before changing)* |
| `ATLAS_DFLASH_RETR_WIDE` | `31` | Clamps retrieval width to `[gamma_eff, K_MAX - 1]` (where `K_MAX=32` is the tree-WY / DDTree buffer capacity). |
| `ATLAS_DFLASH_SAM` | `0` | Enables suffix-automaton (SAM) retrieval drafting in DFlash. Disabled here (`0`); `ATLAS_DFLASH_SAM_ASYNC=1` overrides when async collect precedes propose. |
| `ATLAS_DFLASH_SPEC_CYCLE_V2` | `1` | *(no comment at the use site: read the source before changing)* |
| `ATLAS_DFLASH_TREE_COMMIT` | `0` | Commits a possibly non-contiguous accepted path (including sibling-fork tails) via `tree_accepted_path` during emit and KV compaction. Default off for flat chains. |
| `ATLAS_DISABLE_TREE_WY` | `1` | Disables automatic selection of the tree-WY GDN kernel (which is not bit-equivalent to `wy17` on flat chains and causes drafter acceptance drift). |
| `ATLAS_DSPARK_ASYMMETRIC_ATTN` | `1` | *(no comment at the use site: read the source before changing)* |
| `ATLAS_E2M1_GEMM` | `0` | Default-off whole-FFN W4A4 candidate. Quantizes BF16 activations to NVFP4 before native block-scaled OMMA (not byte-exact to W4A16). Gate/up share one prepared activation; fails closed if selected dependencies are missing. |
| `ATLAS_E2M1_GEMM_DOWN_ONLY` | `0` | Default-off down-projection-only W4A4 candidate. Gate/up retain the transposed W4A16 M128 kernel while only the down projection uses W4A4. Full W4A4 takes precedence when both are set. |
| `ATLAS_E2M1_KMAJOR` | `0` | Default-off refinement for W4A4 candidates using retained `[K/2, N]` packed and `[K/16, N]` scale transforms with an 8-warp M128 native-OMMA kernel and double-buffered B tiles. Requires full or down-only W4A4. |
| `ATLAS_E2M1_KMAJOR_M256` | `0` | Default-off long-prefill (`M >= 2048`) 16-warp M256 refinement of `ATLAS_E2M1_KMAJOR` sharing one transformed weight tile across 256 rows with double-buffered B tiles. Shorter rows fall back to M128. |
| `ATLAS_E2M1_SILU_QUANT` | `0` | Default-off fused SwiGLU-to-NVFP4 boundary for W4A4 down projection. Reproduces `SiLU(gate) * up` BF16 rounding in registers and writes E2M1 nibbles and E4M3 group scales directly. Requires W4A4 plus static checkpoint scales. |
| `ATLAS_E2M1_STATIC_SCALE` | `0` | Default-off ModelOpt-compatible static W4A4 input scaling using checkpoint `.input_scale` metadata (`max(gate, up)` for shared input), bypassing runtime activation absmax and D2H sync. |
| `ATLAS_FA2_KGAMMA` | `1` | Selects the FlashAttention-2 `K=gamma` paged-decode kernel (`paged_decode_kgamma_fa2_k`), taking precedence over the VEC variant when both are enabled. Cached via `OnceLock`. |
| `ATLAS_FFN_DOWN_SPLITK` | `4` | Split-K factor for `w4a16_gemm_t` FFN down-projection (`N=5120`), increasing CTA count and DRAM bandwidth utilization with FP32 partial reductions. |
| `ATLAS_FFN_FUSED_GATEUP` | `1` | *(no comment at the use site: read the source before changing)* |
| `ATLAS_FFN_KGAMMA_M128` | `1` | Uses the M=128 batched NVFP4 kernel for `K=gamma` verification so `M=17` completes in one tile (single weight read) on bandwidth-bound GB10 decode. |
| `ATLAS_FFN_KGAMMA_M16` | `1` | Batches `K=gamma` target FFN verification across all verify rows in a single weight load per layer instead of a per-token loop. Default off until A/B-validated. |
| `ATLAS_FFN_M16_TRANSPOSED` | `1` | Uses transposed `w4a16_gemm_t_m16` weights (one-time host-side transpose at load) for batched FFN verification; falls back to `M_TILE=64` when disabled or symbol-absent. |
| `ATLAS_FFN_TC` | `1` | Enables Tensor Core MMA FFN kernels (`m32_n64` / `m128`). MMA reduction order, BF16 weight rounding, and dequant re-association differ from the serial FMA oracle, so enabling this can change token output (requires reference hash refreeze). |
| `ATLAS_FFN_W3_LAYERS` | *(empty)* | Optional layer list or ranges (`"3,7,12"` or `"a-b"`) selecting layers to load from an external W3 FFN sidecar (`crates/spark-model/src/weight_map/w3_sidecar.rs`). Empty disables W3 sidecar loading. |
| `ATLAS_FFN_W3_SIDECAR` | *(empty)* | Canonical path to an optional W3 FFN sidecar safetensors file (validated alongside `ATLAS_FFN_W3_SIDECAR_SHA256` and `ATLAS_FFN_W3_SIDECAR_SIZE`). Empty disables W3 sidecar loading. |
| `ATLAS_FLASH_ATTN_KGAMMA_SPLITK` | `1` | Enables split-K across long-context `K=gamma` FlashAttention decode to raise SM occupancy (4 CTAs -> 48 CTAs on GB10). Consulted only when `flash_attn_kgamma` is enabled; cached via `OnceLock`. |
| `ATLAS_GDN_C143_PREFILL` | `0` | Default-off eager-only sealed SM121 C143 GDN prefill route (`crates/spark-model/src/layers/qwen3_ssm/gdn_c143_prefill.rs`). Strict `0`/`1`; requires `ATLAS_GDN_C143_SM121_LIB` and is mutually exclusive with `ATLAS_GDN_PREFILL_GATECACHE=1`. |
| `ATLAS_GDN_PREFILL_GATECACHE` | `0` | Default-off exact `C=1` WY32 candidate that replaces redundant per-thread coefficient multiplies with rolling scans in shared memory while preserving every parent FP32 product bit and 4-warp reduction tree. |
| `ATLAS_LM_HEAD_T` | `1` | Retains a transposed NVFP4 `lm_head` copy at load for post-construction sharing with the DFlash drafter's propose head (`dflash_lm_head_t`). |
| `ATLAS_LM_HEAD_TC` | `1` | Enables Tensor Core MMA for target `lm_head`. Target committed tokens may differ from the scalar FMA oracle by MMA-vs-FMA rounding (same re-reference class as `ATLAS_FFN_TC=1`). Default off. |
| `ATLAS_MULTISEQ_GRAPHS` | `0` | Enables multi-sequence CUDA graph capture/replay with persistent-batch gather-before-replay metadata uploads outside the captured region. Default off; cached via `OnceLock`. |
| `ATLAS_NO_GEMV_SW` | `1` | Disables software-pipelined GEMV variant; resolved once per process to avoid per-hop `std::env::var` overhead. |
| `ATLAS_NVFP4_GATE_UP_M128` | `1` | *(no comment at the use site: read the source before changing)* |
| `ATLAS_PAGED_DECODE_SPLITK` | `1` | Allocates the split-K paged-decode arena (split-K does not use `BC=4` batching), sized for `K=gamma` DFlash verification (`max_decode_seqs ~= 32`, `MAX_SPLITS = max_seq_len / 512`). |
| `ATLAS_PREFILL_ATTN_BR128` | `0` | Default-off NVFP4 HD256 paged-attention shadow for prompt chunks `>= 2,048` tokens, merging two BR64 tiles to share one dequantized K/V tile while keeping identical causal loop limits. Fails closed on stale bundles and emits an ENGAGED marker. |
| `ATLAS_PREFILL_ATTN_GATE_FUSED` | `0` | Default-off dense-Qwen3.8 `C=1` chunk-0 candidate folding the sigmoid gate after BR64 contiguous FlashAttention into the final store while preserving exact scalar `expf`, multiply, and BF16 rounding order. |
| `ATLAS_PREFILL_FFN_DUAL_FUSED` | `0` | Default-off exact gate+up+SiLU pipe kernel that loads activation `A` once and eliminates both BF16 intermediate round trips plus two launches. Authoritative over `ATLAS_PREFILL_FFN_FUSED_EPILOGUE`; fails closed on missing dependencies. |
| `ATLAS_PREFILL_FFN_FAST` | `0` | Routes prefill FFN through transposed `w4a16_gemm_t_m128` when available (`crates/spark-model/src/layers/qwen3_attention/prefill_weights.rs`); falls back to `M_TILE=64` when disabled or missing. Default off; cached via `OnceLock`. |
| `ATLAS_PREFILL_FFN_FLASHINFER` | `0` | Default-off exact-Qwen3.8 FFN prefill route using the pinned native FlashInfer SM121 C ABI (`ATLAS_FLASHINFER_SM121_LIB`, verified against compiled-in SHA-256 and loaded via sealed memfd). Prefill-only; does not affect decode. |
| `ATLAS_PREFILL_FFN_FUSED_EPILOGUE` | `0` | Default-off candidate keeping the pipe up-GEMM and BF16 round trip while folding `SiLU(gate) * up` into its epilogue, removing one launch and activation round trip. Fails closed if parent pipe or symbol is missing. |
| `ATLAS_PREFILL_FFN_PIPE` | `1` | Byte-exact `cp.async` shadow of the non-transposed prefill FFN path. Preserves dequantization and MMA reduction order; fails before projection if unavailable and emits an ENGAGED marker. |
| `ATLAS_PREFILL_KV_DUAL` | `0` | Default-off exact attention prefill candidate projecting K and V from one original-layout pipe kernel and single `A` load into their independent BF16 buffers in both chunk-0 and paged paths. |
| `ATLAS_PREFILL_PROJ_FAST` | `0` | Selects transposed fast prefill projections when `1`; when `0`, routes prefill projections through the non-transposed `M_TILE=64` `w4a16_gemm` while keeping transposed weights installed for decode/verify. Cached via `OnceLock`. |
| `ATLAS_PREFILL_PROJ_FLASHINFER` | `0` | Default-off exact-Qwen3.8 attention QKV/O prefill route via the pinned FlashInfer SM121 C ABI (`crates/spark-model/src/layers/qwen3_attention/prefill/flashinfer_projection.rs`). Strict `0`/`1`; requires `ATLAS_FLASHINFER_SM121_LIB` and a standard ModelOpt NVFP4 checkpoint. |
| `ATLAS_PREFILL_PROJ_PIPE` | `1` | Byte-exact pipelined shadow for the non-transposed attention and SSM prefill projections selected when `ATLAS_PREFILL_PROJ_FAST=0`. Fails before projection if missing and emits per-family ENGAGED markers. |
| `ATLAS_PREFILL_QKNORM_ROPE` | `0` | Default-off exact dense-Qwen3.8 `C=1` fusion of Q/G deinterleave, per-head Q/K RMSNorm, and RoPE, preserving distinct reduction trees and normalized-BF16 boundaries across chunk-0 scalar RoPE and paged M-RoPE paths. |
| `ATLAS_PREFILL_SSM_FLASHINFER` | `0` | Default-off exact-Qwen3.8 SSM QKVZ and output prefill route via the pinned FlashInfer SM121 C ABI (`crates/spark-model/src/layers/qwen3_ssm/trait_prefill_helper.rs`). Strict `0`/`1`; requires `ATLAS_FLASHINFER_SM121_LIB`. |
| `ATLAS_SSM_BA_BATCH` | `1` | Collapses the `K=gamma+1` per-token `in_proj_ba` GEMVs (`N=64, K=5120`) per SSM layer into a single batched `dense_gemv_bf16_batchn` launch, saving hundreds of small launches per verify step. |
| `ATLAS_SSM_GDN_LAZY` | `1` | *(no comment at the use site: read the source before changing)* |
| `ATLAS_SSM_GDN_SEQ_PERSISTENT` | `1` | *(no comment at the use site: read the source before changing)* |
| `ATLAS_SSM_OUT_SPLITK` | `4` | Split-K factor for SSM `out_proj` (`0` when unset/`0`/`1`, else clamped to `[2, 8]`). Uses lossless FP32 partials plus `reduce_splitk_f32_to_bf16`, matching `ffn_down` split-K. Default off. |
| `ATLAS_SSM_PREFILL_CONV_L2` | `0` | Default-off exact Qwen3.8 candidate fusing sequential Conv1D, projected-Z persistence, and per-head Q/K L2-norm while preserving the standalone BF16 conv round trip and reduction tree. |
| `ATLAS_SSM_PREFILL_PACK` | `1` | Writes two-phase prefill QKV and gate/beta directly to their full-sequence destinations and copies projected Z inside an exact conv-prefill shadow, removing ~8,194 extra D2D copies per recurrent layer at 8K. |
| `ATLAS_SSM_PROJ_TC` | `1` | Enables Tensor Core MMA for SSM projections with FP32 partials and `reduce_splitk_f32_to_bf16`. Reduction order differs from the serial FMA oracle and can change token output (requires reference hash refreeze). Default off. |
| `ATLAS_SSM_QKVZ_SPLITK` | `4` | Split-K factor for SSM `in_proj_qkvz` (`0` when unset/`0`/`1`, else clamped to `[2, 8]`), using lossless FP32 partials to raise DRAM bandwidth utilization across the 48 SSM layers. Default off. |
| `ATLAS_TARGET_MODEL` | `qwen3.8-27b` *(build-time)* | Compile-time kernel-bundle selector in `crates/atlas-kernels/build.rs`. Must be set to `qwen3.8-27b` (with `ATLAS_TARGET_QUANT=nvfp4`) when building `spark-server`; selected at runtime via `--kernel-target qwen3.8-27b`. |
| `ATLAS_TC_NVFP4_M16` | `0` | Enables the `M_TILE=16` Tensor Core NVFP4 kernel (`M <= 32`) instead of `w4a16_gemm_n128` (`M_TILE=64`). Kept off (`0`) at `K=17` because `M_TILE=16` corrupts deep-slot verification. Cached via `OnceLock`. |
| `ATLAS_TC_NVFP4_M16_MS_ATTN` | `0` | Enables `M_TILE=16` NVFP4 multi-step attention/SSM path (`q/k/v_nvfp4_t` with `deinterleave_qg`). Kept off (`0`) alongside `ATLAS_TC_NVFP4_M16=0`. |
| `ATLAS_THINK_SPEC` | `1` | Keeps DFlash speculation active across thinking blocks. `</think>` and EOS are phase-boundary tokens that end the walk as the bonus token so `a.last_token` preserves the plain-path contract; not byte-identical to sequential decode on SSM layers. |
| `ATLAS_TREE_AWARE_ATTN` | `0` | Enables tree-aware ancestor indirection in paged decode attention. Default off until the single-position path is batched over scattered ancestor blocks. |
| `ATLAS_WEIGHT_CACHE` | `1` | *(no comment at the use site: read the source before changing)* |
| `ATLAS_WY17_LAZY` | `1` | Controls `wy17` intermediate state write stride (`1` = write all, bit-identical to historical kernel; `>= 2` clamped to `[2, 16]`, outputs and final `h_state` byte-identical). Cached via `OnceLock`. |
| `ATLAS_WY17_LAZY_COMMIT` | `0` | Uses `gated_delta_rule_wy17_replay` instead of a plain intermediate-to-`h_state` D2D copy during lazy commit (requires per-layer `wy17_kv_retain` / `wy17_gate_retain` buffers). Default off; cached via `OnceLock`. |
| `ATLAS_WY17_SPLIT` | `2` | Splits `wy17` across V columns (`0` when unset/`0`/`1`, else clamped to `[2, 4]`), keeping per-column FP32 math and reduction order unchanged. |

77 of 87 carry a description at their use site in `crates/`.

Undocumented at source: `ATLAS_ATTN_QKV_BATCHED`, `ATLAS_DFLASH_DRAFT_CAP`, `ATLAS_DFLASH_QUANT`, `ATLAS_DFLASH_SPEC_CYCLE_V2`, `ATLAS_DSPARK_ASYMMETRIC_ATTN`, `ATLAS_FFN_FUSED_GATEUP`, `ATLAS_NVFP4_GATE_UP_M128`, `ATLAS_SSM_GDN_LAZY`, `ATLAS_SSM_GDN_SEQ_PERSISTENT`, `ATLAS_WEIGHT_CACHE`.
