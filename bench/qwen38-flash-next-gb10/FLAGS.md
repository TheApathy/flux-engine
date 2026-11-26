# Qwen3.8-Flash-Next serve-profile flag reference (`FLAGS.md`)

Every `FLUX_*` and `ATLAS_*` environment variable set by [`serve.sh`](serve.sh)
(`v9` 16K-chunk), [`serve-w4096.sh`](serve-w4096.sh) (`v9-w4096` 4K-chunk),
[`serve-fast-prefill.sh`](serve-fast-prefill.sh), and
[`serve-k5-decode.sh`](serve-k5-decode.sh), with its profile value, its binary
default when unset, what it controls in the Rust/CUDA stack, and the campaign
gate or microbenchmark evidence that qualified it.

`flux-server` bridges `FLUX_<SUFFIX>` -> `ATLAS_<SUFFIX>` inside the binary at
startup (`crates/spark-server/src/main.rs`). At the shell launcher layer, only
the parameters explicitly listed with both names in `serve.sh` (12 knobs) — plus
`PREFILL_WIDE` in `serve-k5-decode.sh` and `PREFILL_WIDE`/`GDN_NORM_FP8` in
`serve-fast-prefill.sh` — are shell-overridable; all remaining `ATLAS_*` values
are pinned by each launcher. Changing any value in this table invalidates
attribution against the published `6,082.2 tok/s` prefill, `57.4 tok/s`
`LOSSLESS 10/10` `K=5` decode, and `5/5` Estonia `136.6K` numbers.

## 1. High-level Flux wrapper & build selectors

| Flag (`FLUX_*` / `ATLAS_*`) | `v9` (`serve.sh`) | `v9-w4096` | `fast-prefill` | `k5-decode` | Default | What it controls & qualification evidence |
|---|---|---|---|---|---|---|
| `FLUX_TARGET_MODEL` / `ATLAS_TARGET_MODEL` | `qwen3.8-flash-next` | `qwen3.8-flash-next` | `qwen3.8-flash-next` | `qwen3.8-flash-next` | `qwen3-next-80b-a3b` | Compile-time selector in `crates/atlas-kernels/build.rs`. Must be set to `qwen3.8-flash-next` when building `spark-server`, and verified in `strings -a "$BIN"` before launch. |
| `FLUX_TARGET_QUANT` / `ATLAS_TARGET_QUANT` | `nvfp4` | `nvfp4` | `nvfp4` | `nvfp4` | `nvfp4` | Compile-time quantization selector in `crates/atlas-kernels/build.rs` for the `sm_121a` (`gb10`) PTX bundle. |
| `FLUX_MODE` | `max-prefill` | `interactive-w4096` | `fast-prefill` | `k5-decode` | unset | Wrapper-level mode identifier consumed by `scripts/flux_env_bridge.sh` and logged by the self-verifying launchers (`long-context-136k` when `ESTONIA_136K=1`). |

## 2. Prefill chunking, memory safety & runtime lifecycle

| Flag (`ATLAS_*` / `FLUX_*`) | `v9` (`serve.sh`) | `v9-w4096` | `fast-prefill` | `k5-decode` | Default | What it controls in Rust/CUDA & evidence |
|---|---|---|---|---|---|---|
| `ATLAS_QWEN4_PREFILL_WIDE`<br/>(`FLUX_QWEN4_PREFILL_WIDE`) | `16384` | `4096` | `16384` | `8192` | `2048` | Maximum token width of the prefill workspace arena in `crates/spark-model/src/layers/qwen4_prefill_moe.rs`. Must match `--max-prefill-tokens`. `16384` unlocks `6,082.2 tok/s` at 16K (`M-0001`), while `4096` saves `+4.2 GiB` UMA (`11,907–12,695 MiB` floor) and sustains `5,276.3 tok/s` at 8K (`05f-serve-v9-w4096-gate`). |
| `ATLAS_QWEN4_PREFILL_MULTI_CHUNK` | `1` | `1` | `1` | `1` | `0` | Admits multi-chunk prefill execution up to `--max-seq-len` (`135168` / `147456`) across GDN recurrent states, QSA paged KV blocks, and PLE DMA rings without falling back to row-serial prefill. |
| `ATLAS_PREFILL_KV_ZERO_COALESCE` | `1` | `1` | `0` | `0` | `0` | Coalesces per-sequence KV and GDN state zeroing (`BufferArena::ZERO_ALL_BUFFERS`) into a single batched clear at sequence allocation (`04o-serve-v6-gate`, `KV_ZERO_COALESCE_ENGAGED`). |
| `ATLAS_ALLOC_SEQ_ASYNC` | `1` | `1` | `0` | `0` | `0` | Runs sequence slot allocation and state zeroing asynchronously on the prefill stream instead of blocking the scheduler thread (`04s/04t` Wave 8, `ALLOC_SEQ_ASYNC_ENGAGED`). |
| `ATLAS_PREFILL_LEAN_FINALIZE` | `1` | `1` | `0` | `0` | `0` | Elides redundant stream synchronizations and intermediate host readbacks at the end of each prefill chunk (`05e-serve-v9-gate`). |
| `ATLAS_QWEN4_PREFILL_LAST_LAYER_CROP`<br/>(`FLUX_QWEN4_PREFILL_LAST_LAYER_CROP`) | `1` | `1` | `1` | `0` | `0` | Computes final-layer (`layer 47`) post-attention/FFN norm and `lm_head` only for the trailing token(s) needed by sampling and MTP last-`K` capture. Refuses if MTP capture is unguarded (`612de1ad3`). |
| `ATLAS_QWEN4_RELEASE_BF16_QKVZ` | `1` | `1` | `1` | `1` | `0` | Frees temporary BF16 `in_proj_qkvz` weights after transcoding/packing into the runtime layout at load time, reclaiming unified memory for KV blocks. |
| `ATLAS_QWEN4_RELEASE_UNUSED` | `1` | `1` | `1` | `1` | `0` | Releases pre-transpose staging buffers immediately after model construction in `load_layers.rs`. |
| `ATLAS_QWEN4_RELEASE_SPEC_UNUSED` | `1` | `1` | `0` | `1` | `0` | Releases unused speculative scratch buffers when native MTP `K=5` is selected instead of `K=16` DFlash. |
| `ATLAS_BOUNDED_HOST_COPY` | `1` | `1` | `0` | `1` | `0` | Enforces bounded timeouts on pageable/pinned H2D/D2H host copies in `spark-runtime` (`1fc7ec6b2`, `a3069305b`) so a stalled DMA fails closed rather than hanging the worker. |
| `ATLAS_CTX_OVERFLOW_TRUNCATE`<br/>(`FLUX_CTX_OVERFLOW_TRUNCATE`) | `0` *(Estonia)* | unset | unset | unset | `0` | Set to `0` in `ESTONIA_136K=1` (`flux-estonia-136k.env`) so any prompt exceeding `--max-seq-len` fails closed instead of silently truncating context tokens. |
| `ATLAS_CUBLAS_TUNED` | `0` | `0` | `0` | `0` | `0` | Disables runtime cuBLAS heuristic auto-tuning sweeps during model startup in favor of deterministic pre-selected algorithms. |

## 3. Routed & shared MoE prefill kernels (`512` routed top-10 + `1` shared)

| Flag (`ATLAS_*` / `FLUX_*`) | `v9` (`serve.sh`) | `v9-w4096` | `fast-prefill` | `k5-decode` | Default | What it controls in Rust/CUDA & evidence |
|---|---|---|---|---|---|---|
| `ATLAS_UNIFIED_MOE_LAYOUT` | `1` | `1` | `1` | `1` | `0` | Installs the unified expert-major transposed weight layout shared between compact prefill GEMMs and multi-row speculative verify (`k5-bisect/20261014`). |
| `ATLAS_QWEN4_PREFILL_MOE_BATCH` | `1` | `1` | `1` | `1` | `0` | Enables batched expert-grouped MoE prefill execution across all active token-expert assignments (`163,840` pairs per 16K chunk). |
| `ATLAS_QWEN4_PREFILL_MOE_COMPACT` | `1` | `1` | `1` | `1` | `0` | Routes grouped MoE prefill through the compact expert-major tile planner (`qwen4_moe_compact*`). |
| `ATLAS_QWEN4_PREFILL_MOE_COMPACT_K32` | `1` | `1` | `1` | `1` | `0` | Selects the K32 reduction-tile family in the compact MoE kernel suite. |
| `ATLAS_QWEN4_PREFILL_MOE_COMPACT_T` | `1` | `1` | `1` | `1` | `0` | Consumes transposed expert weights (`qwen4_moe_compact_t_k32*`) for coalesced global loads. |
| `ATLAS_QWEN4_PREFILL_MOE_COMPACT_V2` | `8` | `8` | `8` | `8` | `0` | Selects V2 compact descriptor layout (`version 8`) admitting up to 65,536 token-expert rows per launch (`50b0d1626`). |
| `ATLAS_QWEN4_PREFILL_MOE_W4A8` | `1` | `1` | `1` | `1` | `0` | Enables native SM121a block-scaled W4A8 (`NVFP4 E2M1` weights x `MXFP8 E4M3` activations) Tensor Core MMA (`mma.sync.aligned.m16n8k64...e2m1.block_scale`) for routed experts. |
| `ATLAS_QWEN4_PREFILL_MOE_W4A8_K64` | `1` | `1` | `1` | `1` | `0` | Selects the `k64` Tensor Core MMA instruction shape across routed expert projections. |
| `ATLAS_QWEN4_PREFILL_MOE_W4A8_WIDE`<br/>(`FLUX_QWEN4_PREFILL_MOE_W4A8_WIDE`) | `7` | `7` | `2` | `2` | `0` | Selects wide expert-tile variant: `7` activates the `v28` W4A8 MoE kernel with packed B-column scales (`14f4425b1`, `04k-serve-v5-gate`), whereas `2` selects the conservative `v16` W4A8 kernel (`6c6eaa6ea`). |
| `ATLAS_QWEN4_PREFILL_MOE_V28Q`<br/>(`FLUX_QWEN4_PREFILL_MOE_V28Q`) | `2` | `2` | `0` | `0` | `0` | Routes the MoE down-projection through `v28q=2` (`e962ed334`, `ae58bc27e`) with interleaved `mma`/`ldmatrix`, 0-SMEM quad-BF16 epilogue, and adaptive 512-thread transcode (`05e-serve-v9-gate`). |
| `ATLAS_QWEN4_PREFILL_MOE_SLOT_MXFP8`<br/>(`FLUX_QWEN4_PREFILL_MOE_SLOT_MXFP8`) | `1` | `1` | `1` | `0` | `0` | Quantizes MoE dispatch slot rows directly into MXFP8 (`v16q4` fused gate+up prologue, `8f2e417c0`, `076086763`). |
| `ATLAS_QWEN4_PREFILL_MOE_SLOT_BF16_FUSED` | `1` | `1` | `0` | `0` | `0` | Fuses BF16 slot writeback into the MoE down-projection epilogue (`04o-serve-v6-gate`). |
| `ATLAS_QWEN4_PREFILL_MOE_XQ_REUSE`<br/>(`FLUX_QWEN4_PREFILL_MOE_XQ_REUSE`) | `1` | `1` | `1` | `0` | `0` | Reuses the shared expert's pre-quantized MXFP8 input activation `xq` across all 10 routed expert dispatches per token (`cf4e6198a`), eliminating redundant input quantization. |
| `ATLAS_QWEN4_PREFILL_SHARED_W4A8` | `1` | `1` | `1` | `1` | `0` | Routes the per-layer dense shared expert through the W4A8 Tensor Core pipeline (`1c09e9331`). |
| `ATLAS_QWEN4_PREFILL_SHARED_W4A8_PRETRANSCODE` | `1` | `1` | `0` | `0` | `0` | Uses the 2-CTA/SM persistent pretranscode kernel for shared expert weights with zero D2D status copies (`19e0aaf77`, `05e-serve-v9-gate`). |
| `ATLAS_QWEN4_PREFILL_ROUTER_PIPE` | `1` | `1` | `0` | `0` | `0` | Enables pipelined 512-expert router projection during prefill (`05e-serve-v9-gate`). |
| `ATLAS_QWEN4_PREFILL_ROUTER_GEMM_PIPE` | `1` | `1` | `0` | `0` | `0` | Dispatches the `pipe_m128n128` FP8 router GEMM at all chunks `>= 1024` (`bdd769a8c`, `19e0aaf77`). |
| `ATLAS_QWEN4_PREFILL_MOE_TOPK_WARP` | `1` | `1` | `0` | `0` | `0` | Uses warp-aggregated top-10 expert selection (`topk_v2` hybrid `rk[16]+tail`, `04o-serve-v6-gate`). |
| `ATLAS_QWEN4_PREFILL_MOE_FUSED_SORT_PLAN` | `1` | `1` | `0` | `0` | `0` | Fuses expert histogram sort and tile descriptor planning into a single 4-CTA/SM SMEM pass (`K11-A3`, `12656e104`). |
| `ATLAS_QWEN4_PREFILL_MOE_SORT_PLAN_FUSED` | `1` | `1` | `0` | `0` | `0` | Companion alias for `MOE_FUSED_SORT_PLAN` (`qwen4_moe_sort_plan_t_v8` with `uint4` histogram and threadfence elision, `811ebe5cc`). |
| `ATLAS_QWEN4_PREFILL_MOE_PLAN_ORDER` | `inter` | `inter` | unset | unset | `linear` | Orders expert tile execution in interleaved L2-friendly order (`K10 D`, `04s/04t` Wave 8). |
| `ATLAS_QWEN4_PREFILL_MOE_STATUS_ASYNC` | `1` | `1` | `1` | `1` | `0` | Defers compact-MoE overflow status checks so no per-layer D2H synchronization stalls the prefill stream (`3acff6da1`). |

## 4. Fast projections (`FP8` cuBLASLt + weight cache)

| Flag (`ATLAS_*` / `FLUX_*`) | `v9` (`serve.sh`) | `v9-w4096` | `fast-prefill` | `k5-decode` | Default | What it controls in Rust/CUDA & evidence |
|---|---|---|---|---|---|---|
| `ATLAS_QWEN4_PREFILL_FAST_PROJ` | `all` | `all` | `all` | `all` | `none` | Routes all eligible prefill linear projections (`in_proj_qkvz`, `out_proj`, `q_proj`, `k_proj`, `v_proj`, `o_proj`) through the batched fast-projection dispatcher (`qwen4_fast_proj.rs`). |
| `ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8` | `ssm,attn` | `ssm,attn` | `ssm,attn` | `ssm,attn` | `none` | Enables FP8 E4M3 cuBLASLt Tensor Core GEMMs for both the 36 GDN (`ssm`) and 12 QSA (`attn`) projection families. |
| `ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_SCALE` | `tensor` | `tensor` | `tensor` | `tensor` | `tensor` | Uses per-tensor `WEIGHT_AMAX_CACHE` and dynamic activation absmax scales for FP8 cuBLASLt projections. |
| `ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_SHARED_ACT` | `1` | `1` | `1` | `1` | `0` | Quantizes shared input activations to FP8 E4M3 once per layer (`qwen4_fp8_act_rows_tensor_v2` with 5-step register pipeline and `L2::evict_first` hints, `c0d38114c`) and reuses across sibling projections. |
| `ATLAS_QWEN4_PREFILL_FP8_ACT_AMAX_FUSED` | `1` | `1` | `0` | `0` | `0` | Fuses the FP8 activation absmax reduction into the preceding normalization epilogue (`04k-serve-v5-gate`). |
| `ATLAS_QWEN4_PREFILL_OPROJ_AMAX_FUSED` | `1` | `1` | `0` | `0` | `0` | Folds `o_proj` input absmax computation into the E4M3 QSA attention core epilogue (`K8 D` / `K10 C`, `6e30a52a0`, `91649cabe`). |
| `ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_WEIGHT_CACHE` | `1` | `1` | `1` | `0` | `0` | Caches pre-transcoded FP8 E4M3 weights for the 96 fast projections in unified memory. Enabled in prefill-heavy profiles (`+300+ tok/s`), and disabled (`0`) in `serve-k5-decode.sh` to maximize free UMA and L2 residency for decode. |
| `ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_WEIGHT_CACHE_RESERVE_GB`<br/>(`FLUX_QWEN4_FAST_PROJ_FP8_WEIGHT_CACHE_RESERVE_GB`) | `7`<br/>(`10` Estonia) | `7` | unset | unset | `0` | Stops populating the FP8 projection weight cache once OS `MemAvailable` drops to the reserve threshold (`7 GiB` for 128K in `v9`/`v9-w4096`, `10 GiB` for 144K in `ESTONIA_136K=1`), preventing UMA watchdog kills (`bdb486eab`, `M-0003`). |

## 5. Gated DeltaNet (`GDN`) 36-layer linear attention stack

| Flag (`ATLAS_*` / `FLUX_*`) | `v9` (`serve.sh`) | `v9-w4096` | `fast-prefill` | `k5-decode` | Default | What it controls in Rust/CUDA & evidence |
|---|---|---|---|---|---|---|
| `ATLAS_QWEN4_PREFILL_SSM_EXACT`<br/>(`FLUX_QWEN4_PREFILL_SSM_EXACT`) | `1` | `1` | `1` | `1` | `0` | Selects the numerically verified two-phase GDN prefill pipeline instead of approximate/unqualified tile paths. |
| `ATLAS_QWEN4_PREFILL_SSM_GRID32` | `1` | `1` | `1` | `1` | `0` | Uses 32-CTA grid partitioning across the 48 value heads (`head_dim = 128`) during GDN prefill. |
| `ATLAS_QWEN4_PREFILL_BA_GEMM` | `1` | `1` | `1` | `1` | `0` | Batches the `in_proj_ba` beta/alpha gate projection into a single MMA GEMM with `cp.async` preload (`c58ad3aa7`) instead of per-token GEMVs. |
| `ATLAS_QWEN4_PREFILL_CONV_PARALLEL` | `1` | `1` | `1` | `1` | `0` | Executes the 4-tap causal depthwise Conv1D across prompt chunks in parallel across tokens. |
| `ATLAS_QWEN4_PREFILL_GDN_CONV_V2` | `1` | `1` | `0` | `0` | `0` | Enables the V2 fused causal Conv1D + SiLU + Q/K L2-norm kernel (`04o-serve-v6-gate`). |
| `ATLAS_QWEN4_PREFILL_GDN_CONV_V3`<br/>(`FLUX_QWEN4_PREFILL_GDN_CONV_V3`) | `1` | `1` | `0` | `0` | `0` | Upgrades GDN Conv1D to the 2-stage `qwen4_conv1d_l2norm_prefill_v3_commit` kernel with `LDS.64` loads and direct state commit (`19e0aaf77`, `c58ad3aa7`, `05e-serve-v9-gate`). |
| `ATLAS_QWEN4_PREFILL_GDN_FI` | `2` | `2` | `2` | `2` | `0` | Dispatches the native FlashInfer SM121a CuTe-DSL chunked GDN recurrence kernel (`atlas_gdn_fi_sm121a_prefill_f16` in `csrc/fi_gdn/`) with TMA L2 cache-policy hints and `v2.f32` state load/store (`K11-G14`, `409016ba9`). |
| `ATLAS_QWEN4_PREFILL_GDN_LAZYFINAL` | `4` | `4` | `4` | `4` | `0` | Lazily materializes intermediate GDN chunk states every 4 sub-chunks while committing the exact final recurrent state `S` (`[1536, 128, 128]`). |
| `ATLAS_QWEN4_PREFILL_GDN_NORM_FP8`<br/>(`FLUX_QWEN4_PREFILL_GDN_NORM_FP8`) | `2` | `2` | `3` | `1` | `0` | Controls GDN output gated-RMSNorm -> FP8 E4M3 quantization before `out_proj`: `1` = standalone exact kernel (`k5-decode` / `exact-reference`), `2` = static-scale deterministic fused norm (`v9` / `v9-w4096`, `LOSSLESS 10/10`), and `3` = store-warp Ultra tier inside the FI kernel (`fast-prefill` only, non-deterministic under `K=5` verify rollback). |
| `ATLAS_QWEN4_PREFILL_GDN_NORM_FP8_V2`<br/>(`FLUX_QWEN4_PREFILL_GDN_NORM_FP8_V2`) | `1` | `1` | `0` | `0` | `0` | Enables the pipelined V2 `gdn_norm_fp8_v2` kernel when `GDN_NORM_FP8=2` (`5c7118c5d`, `e962ed334`, `05e-serve-v9-gate`). |
| `ATLAS_SSM_GDN_LAZY` | `1` | `1` | `0` | `1` | `0` | Defers intermediate GDN state writebacks during decode/verify until the accepted token index is known. |
| `ATLAS_SSM_DUMMY_STATE_ONLY` | `1` | `1` | `0` | `1` | `0` | Skips intermediate state writes on padding/dummy verify slots (`fc02ee695`, `04k-serve-v5-gate`). |
| `ATLAS_QWEN4_GDN_COMMIT_NO_MIRROR` | `1` | `1` | `0` | `1` | `0` | Commits accepted GDN state directly in place without a redundant mirror buffer D2D copy (`05e-serve-v9-gate`). |

## 6. Query-Sparse Attention (`QSA`) 12-layer full/sparse attention stack

| Flag (`ATLAS_*` / `FLUX_*`) | `v9` (`serve.sh`) | `v9-w4096` | `fast-prefill` | `k5-decode` | Default | What it controls in Rust/CUDA & evidence |
|---|---|---|---|---|---|---|
| `ATLAS_QWEN4_PREFILL_QSA` | `1` | `1` | `1` | `1` | `0` | Enables native Query-Sparse Attention (`QSA`, `indexer_budget = 2048`, `indexer_compress_ratio = 4`) across the 12 full-attention layers during prefill. |
| `ATLAS_QWEN4_PREFILL_QSA_V2` | `1` | `1` | `1` | `1` | `0` | Uses the V2 QSA prefill pipeline with 16/32 MiB batched score scratch (`80d8d7a6d`, `c58ad3aa7`). |
| `ATLAS_QWEN4_PREFILL_QSA_TILE` | `1` | `1` | `1` | `1` | `0` | Enables tiled QSA block-score evaluation across long prompts (`8K..144K`). |
| `ATLAS_QWEN4_PREFILL_QSA_KEY_GEMM` | `1` | `1` | `1` | `1` | `0` | Computes compressed indexer keys via batched GEMM + 4-warp `qsa_pool_norm_rope` (`c58ad3aa7`). |
| `ATLAS_QWEN4_QSA_SCORE_V2` | `1` | `1` | `0` | `1` | `0` | Dispatches the 5-CTA/SM `qsa_prefill_score_v2` / `qsa_prefill_score_gmax` kernel using `ldmatrix.x4`, `cp.async.cg.L2::128B`, and streaming stores (`63618f661`, `61d332229`). |
| `ATLAS_QWEN4_PREFILL_QSA_FUSED_TOPK` | `1` | `1` | `0` | `0` | `0` | Fuses group-max reduction and top-`k` block selection (`4096`-block threshold, `63618f661`). |
| `ATLAS_QWEN4_QSA_SELECT_PAR` | `1` | `1` | `1` | `1` | `0` | Parallelizes selected KV page gathering across CTAs (`80d8d7a6d`). |
| `ATLAS_QWEN4_PREFILL_QSA_PLAN_FUSED` | `1` | `1` | `0` | `0` | `0` | Fuses QSA sparse block plan construction (`14fbea1de`, `05e-serve-v9-gate`). |
| `ATLAS_QWEN4_PREFILL_QSA_PREP_FUSED` | `1` | `1` | `0` | `0` | `0` | Fuses QSA query/key norm + RoPE preparation (`05e-serve-v9-gate`). |
| `ATLAS_QWEN4_PREFILL_QSA_FP8`<br/>(`FLUX_QWEN4_PREFILL_QSA_FP8`) | `qkpv` | `qkpv` | unset | unset | unset | Runs both `Q*K^T` and `P*V` inside the QSA sparse attention core in FP8 E4M3 (`qsa_sparse_e4m3_qkpv`, `4b6da6d7c`, `04k-serve-v5-gate`). |
| `ATLAS_QWEN4_PREFILL_QSA_PIPE`<br/>(`FLUX_QWEN4_PREFILL_QSA_PIPE`) | `1` | `1` | `0` | `0` | `0` | Enables the software-pipelined `qsa_prefill_attn_fp8_pipe` kernel with 16-token half-tile causal softmax skipping (`ae58bc27e`, `80d8d7a6d`, `05e-serve-v9-gate`). |
| `ATLAS_QWEN4_QSA_ATTN_V2` | `1` | `1` | `0` | `1` | `0` | Selects the V2 multi-split QSA decode/verify attention kernel. |
| `ATLAS_QWEN4_QSA_GRAPH_SAFE` | `1` | `1` | `0` | `1` | `0` | Uses fixed-address device metadata so QSA decode and `K=5` verify can be captured inside CUDA graphs (`9ba551b87`). |
| `ATLAS_QWEN4_PREFILL_ATTN_CORE16` | `1` | `1` | `1` | `1` | `0` | Enables 16-head grouped query tile packing in full-attention prefill layers. |
| `ATLAS_QWEN4_PREFILL_ATTN_CORE32` | `0` | `0` | `0` | `0` | `0` | Keeps 32-row core tiling disabled in favor of `CORE16`. |
| `ATLAS_QWEN4_PREFILL_ATTN_DEVICE16` | `1` | `1` | `1` | `1` | `0` | Keeps 16-head attention descriptors device-resident. |
| `ATLAS_QWEN4_PREFILL_ATTN_FI` | `1` | `1` | `1` | `1` | `0` | Admits the FlashInfer attention bridge for dense short-chunk fallbacks (`csrc/fi_attn`). |
| `ATLAS_QWEN4_PREFILL_ATTN_FLASH` | `1` | `1` | `1` | `1` | `0` | Enables FlashAttention-2 tiling on chunk-0 / dense attention paths. |
| `ATLAS_QWEN4_PREFILL_ATTN_FULLROW` | `1` | `1` | `1` | `1` | `0` | Processes full query rows per CTA without splitting across head dimensions (`head_dim = 256`). |
| `ATLAS_QWEN4_PREFILL_ATTN_GQA` | `1` | `1` | `1` | `1` | `0` | Uses native 12:1 GQA head sharing (`24` Q heads / `2` KV heads) in prefill attention. |
| `ATLAS_QWEN4_PREFILL_ATTN_HC16` | `1` | `1` | `1` | `1` | `0` | Aligns attention entry/exit staging with 16-row `mHC` tiles. |
| `ATLAS_QWEN4_PREFILL_ATTN_O16` | `1` | `1` | `1` | `1` | `0` | Uses 16-row tiled output accumulation before `o_proj`. |
| `ATLAS_QWEN4_PREFILL_ATTN_PREP_FUSED` | `1` | `1` | `1` | `1` | `0` | Fuses Q/G deinterleave, per-head Q/K RMSNorm, and partial RoPE preparation (`19e0aaf77`). |
| `ATLAS_QWEN4_PREFILL_ATTN_QKV16` | `1` | `1` | `1` | `1` | `0` | Packs QKV prefill projections into 16-row aligned tiles. |

## 7. 4-Stream Multi-Hyper-Connections (`mHC`) residual mixer

| Flag (`ATLAS_*` / `FLUX_*`) | `v9` (`serve.sh`) | `v9-w4096` | `fast-prefill` | `k5-decode` | Default | What it controls in Rust/CUDA & evidence |
|---|---|---|---|---|---|---|
| `ATLAS_QWEN4_PREFILL_HC_EXACT`<br/>(`FLUX_QWEN4_PREFILL_HC_EXACT`) | `1` | `1` | `1` | `1` | `0` | Preserves exact FP32 accumulation across the 4-stream hyper-connection (`hc_count = 4`, `hc_lowrank = 320`) residual mixer. |
| `ATLAS_QWEN4_PREFILL_HC_FUSED` | `2` | `2` | `2` | `2` | `0` | Enables V2 fused multi-stream hyper-connection kernels (`qwen4_hc_fused.cu`). |
| `ATLAS_QWEN4_PREFILL_MHC_FUSED`<br/>(`FLUX_QWEN4_PREFILL_MHC_FUSED`) | `inject,cross,combine,nostage,upmix,injgemm,postpre` | `inject,cross,combine,nostage,upmix,injgemm,postpre` | `inject,cross,combine,nostage,upmix,injgemm,postpre` | `inject,cross,combine,nostage,upmix,injgemm` | `none` | Comma-separated list of fused `mHC` stages: `inject` + `injgemm` (`0`-spill `hc_inject_gemm_combine`, `c58ad3aa7`), `cross`, `combine` (period-6 unrolled `hc_combine_gemm`, `076086763`), `nostage`, `upmix`, and `postpre` (layer-boundary deferred MoE combine, `a3cf75081`). |
| `ATLAS_QWEN4_PREFILL_MHC_ROOF`<br/>(`FLUX_QWEN4_PREFILL_MHC_ROOF`) | `t128` | `t128` | unset | unset | `off` | Dispatches the 128-thread `qwen4_hc_upmix_roof_t128_amax` kernel with stage-9 `MUFU` / `cp.async` overlap (`c353ccc06`, `bdd769a8c`, `04q-serve-v7-gate`). |
| `ATLAS_QWEN4_PREFILL_MHC_GATEPRE` | `1` | `1` | `0` | `0` | `0` | Enables 0-SMEM warp-shuffle `mHC` `moe_prologue_gatepre` with early B/hyper weight preload (`ae58bc27e`, `05e-serve-v9-gate`). |
| `ATLAS_QWEN4_PREFILL_HC_FUSED_PREP` | `1` | `1` | `0` | `0` | `0` | Fast-paths `hc_prep_weights` and hoists `srstd`/permutation loads (`ed8f43fe4`, `05e-serve-v9-gate`). |
| `ATLAS_QWEN4_HC_GEMV_V2` | `1` | `1` | `0` | `1` | `0` | Uses the V2 low-rank `mHC` GEMV kernel during decode and speculative verification. |
| `ATLAS_QWEN4_HC_DUAL_ROWS` | `1` | `1` | `0` | `1` | `0` | Batches dual `mHC` stream projections across all `K=5` verify rows in a single launch. |
| `ATLAS_QWEN4_HC_TAIL_ROWS` | `1` | `1` | `0` | `1` | `0` | Batches the fused `mHC` tail combine across all `K=5` verify rows. |

## 8. 4-Ring NVFP4 Per-Layer Embedding (`PLE`) offload stack

| Flag (`ATLAS_*` / `FLUX_*`) | `v9` (`serve.sh`) | `v9-w4096` | `fast-prefill` | `k5-decode` | Default | What it controls in Rust/CUDA & evidence |
|---|---|---|---|---|---|---|
| `ATLAS_PLE_CACHE_MB` | `0` | `0` | `0` | `0` | `0` *(row cache; ~192 MiB chunk cache)* | System-RAM hot-page row cache size in MiB (`crates/spark-model/src/layers/qwen4_ple.rs`). Kept at `0` in `unified13` (`aad7f8525`) to avoid duplicate page caching on GB10 UMA where Linux page cache and 4-ring direct sector DMA already share LPDDR5X. |
| `ATLAS_PLE_HALF_PAGE` | `1` | `1` | `1` | `1` | `0` | Reads 2,048-byte half-pages matching the exact NVFP4 PLE row stride instead of over-fetching full 4,096-byte pages. |
| `ATLAS_PLE_FAST_PLAN` | `1` | `1` | `0` | `0` | `0` | Uses the sorted deduplicated n-gram read planner for bulk prefill chunks (`04q-serve-v7-gate`). |
| `ATLAS_PLE_SECTOR_READ`<br/>(`FLUX_PLE_SECTOR_READ`) | `1` | `1` | `0` | `0` | `0` | Issues sector-aligned NVMe reads directly into pinned staging rings (`04q-serve-v7-gate`). |
| `ATLAS_PLE_READ_RINGS`<br/>(`FLUX_PLE_READ_RINGS`) | `4` | `4` | unset | unset | `1` | Spreads planned PLE prefill reads across 4 concurrent `io_uring` rings (`fa6da111f`, `baca894b6`, `04q-serve-v7-gate`). |
| `ATLAS_PLE_READ_AFFINITY` | `big` | `big` | unset | unset | `none` | Pins PLE DMA reader threads to the 10 ARM Cortex-X925 performance cores (`cpus 5-9,15-19`, `04s/04t` Wave 8). |
| `ATLAS_QWEN4_PLE_EARLY_ISSUE` | `1` | `1` | `0` | `0` | `0` | Issues all PLE n-gram reads for the chunk at token-ingestion time before layer 0 begins (`04q-serve-v7-gate`). |
| `ATLAS_QWEN4_PLE_IO_QUEUE_DEPTH` | `256` | `256` | `256` | `256` | `256` *(valid: `32` or `256`)* | Per-ring `io_uring` submission queue depth for NVMe PLE sidecar shards (`qwen4_ple.rs` rejects values other than `32` or `256`). |
| `ATLAS_QWEN4_PLE_IO_TIMEOUT_MS` | `30000` | `30000` | `30000` | `30000` | `required` *(no default)* | Bounded 30-second timeout per PLE I/O batch (`qwen4_ple.rs` fails closed if unset) so NVMe contention under 16K prefill never hangs indefinitely. |
| `ATLAS_QWEN4_PLE_WARM_READ` | `2` | `2` | `2` | `2` | `0` | Pre-touches PLE shard file descriptors and ring buffers during server warmup. |
| `ATLAS_QWEN4_PLE_PREFILL_BATCH` | `1` | `1` | `1` | `1` | `0` | Batches PLE dequantization and low-rank projection across the full prefill chunk. |
| `ATLAS_QWEN4_PLE_PREFILL_COMBINE` | `1` | `1` | `1` | `1` | `0` | Combines per-layer PLE projections with the residual stream in one batched kernel. |
| `ATLAS_QWEN4_PLE_PREFILL_STREAM` | `3` | `3` | `3` | `3` | `0` | Overlaps PLE H2D copy, dequant, and projection on a dedicated auxiliary CUDA stream (`mode 3`). |
| `ATLAS_QWEN4_PLE_PREFILL_PIPELINE` | `1` | `1` | `0` | `0` | `0` | Pipelines layer `L+1` PLE staging behind layer `L` MoE compute (`04k-serve-v5-gate`). |
| `ATLAS_QWEN4_PLE_PREFILL_FUSED` | `1` | `1` | `0` | `0` | `0` | Uses `qwen4_ple_prefill_fused` to join PLE embeddings without FP32 intermediate buffers (`9fde3da3b`, `04o-serve-v6-gate`). |
| `ATLAS_QWEN4_PLE_PREFILL_ASYNC` | `1` | `1` | `0` | `0` | `0` | Makes PLE host-to-device staging fully asynchronous across all 48 layers (`05e-serve-v9-gate`). |
| `ATLAS_QWEN4_PLE_VERIFY_BATCH` | `1` | `1` | `0` | `1` | `0` | Batches PLE n-gram lookup and H2D staging across all `K=5` speculative verify tokens in one pass. |
| `ATLAS_QWEN4_PREFILL_EMBED_BATCH` | `1` | `1` | `1` | `1` | `0` | Batches token embedding lookups across the prefill chunk. |

## 9. Native Inferact MTP (`K=5`) speculative decode & verification stack

| Flag (`ATLAS_*` / `FLUX_*`) | `v9` (`serve.sh`) | `v9-w4096` | `fast-prefill` | `k5-decode` | Default | What it controls in Rust/CUDA & evidence |
|---|---|---|---|---|---|---|
| `ATLAS_DISABLE_SPECULATION`<br/>(`FLUX_DISABLE_SPECULATION`) | unset (`0`) | unset (`0`) | `1` | unset (`0`) | `0` | Forces non-speculative `K=1` plain decode (`serve-fast-prefill.sh` and `flux-exact-reference.env`). |
| `ATLAS_QWEN4_MTP_CHAIN`<br/>(`FLUX_QWEN4_MTP_CHAIN`) | `1` | `1` | `0` | `1` | `0` | Chains all 4 autoregressive draft steps of the 1-layer Inferact MTP proposer on device (`single-H2D / single-D2H`). |
| `ATLAS_QWEN4_MTP_CONF`<br/>(`FLUX_QWEN4_MTP_CONF`) | `0.6` | `0.6` | unset | `0.6` | `0.0` | Confidence threshold for MTP draft continuation (`M-0002`). |
| `ATLAS_QWEN4_MTP_CONF_EARLY`<br/>(`FLUX_QWEN4_MTP_CONF_EARLY`) | `1` | `1` | `0` | `1` | `0` | Truncates the draft chain early when draft token confidence falls below `MTP_CONF`, avoiding wasted verify rows on low-confidence tails. |
| `ATLAS_QWEN4_MTP_UNIFIED_T` | `1` | `1` | `0` | `1` | `0` | Shares the unified transposed MoE weight layout with the 1-layer MTP draft head (`fc02ee695`, `04k-serve-v5-gate`). |
| `ATLAS_MTP_LASTK_DEFER` | `1` | `1` | `0` | `0` | `0` | Race-free deferral of MTP last-`K` hidden-state capture until the final prefill chunk completes (`570efc606`, `04q-serve-v7-gate`). |
| `ATLAS_QWEN4_MTP_LASTK_SKIP_INTERMEDIATE` | `1` | `1` | `0` | `0` | `0` | Skips MTP hidden-state capture on non-terminal prefill chunks (`743479c83`, `05e-serve-v9-gate`). |
| `ATLAS_QWEN4_MTP_KEEP_LAST_ROW` | `1` | `1` | `0` | `1` | `0` | Retains the accepted bonus row's hidden state in device memory for the next MTP propose step without re-uploading (`05e-serve-v9-gate`). |
| `ATLAS_QWEN4_MTP_BATCH_REPLAY` | `1` | `1` | `0` | `1` | `0` | Replays accepted draft tokens through the MTP proposer's KV/hidden state in a single batched pass (`05e-serve-v9-gate`). |
| `ATLAS_QWEN4_MTP_KV_ONLY_REPLAY` | `1` | `1` | `0` | `1` | `0` | Restricts MTP post-verify catchup to KV projection/cache updates when hidden states are already current (`05e-serve-v9-gate`). |
| `ATLAS_MTP_THINK_SPEC` | `1` | `1` | `0` | `1` | `0` | Keeps native MTP speculation active across `<think>...</think>` reasoning blocks and phase boundaries. |
| `ATLAS_QWEN4_K5_HYBRID`<br/>(`FLUX_QWEN4_K5_HYBRID`) | `1` | `1` | `0` | `1` | `0` | Master gate enabling the 5-row (`1 + 4` drafts) hybrid GDN + QSA target verification path in `verify_d.rs`. |
| `ATLAS_QWEN4_K4_HYBRID` | `1` | `1` | `0` | `1` | `0` | Enables batched hybrid verification when confidence early-stop shortens the draft chain to `K <= 4`. |
| `ATLAS_QWEN4_HYBRID_ALL_K` | `1` | `1` | `0` | `1` | `0` | Admits all dynamic draft widths `K in [2..5]` into the batched hybrid verifier without falling back to row-serial decode. |
| `ATLAS_QWEN4_K16_EXACT`<br/>(`FLUX_QWEN4_K16_EXACT`) | `1` | `1` | `1` | `1` | `0` | Enforces exact per-row reduction order in target verification so `K=5` greedy output matches `K=1` bit-for-bit (`2,560/2,560` in `05e-serve-v9-gate`). |
| `ATLAS_QWEN4_K5_NATIVE_MOE`<br/>(`FLUX_QWEN4_K5_NATIVE_MOE`) | `1` | `1` | `0` | `1` | `0` | Routes `K=5` verify MoE layers through the native multi-row transposed expert kernels (`936983436`). |
| `ATLAS_QWEN4_MOE_DEDUP` | `1` | `1` | `0` | `1` | `0` | Deduplicates shared expert weight loads across the 5 verify candidate rows when multiple draft tokens route to the same expert. |
| `ATLAS_QWEN4_MOE_DOWN_ACT` | `1` | `1` | `0` | `1` | `0` | Fuses activation scaling into the multi-row MoE down-projection during verify. |
| `ATLAS_MOE_T_LANES` | `1` | `1` | `0` | `1` | `0` | Enables transposed-lane W4A16/NVFP4 verify MoE GEMV kernels (`24a6a4b15`). |
| `ATLAS_MOE_T_LANES_DOWN_T8` | `1` | `1` | `0` | `1` | `0` | Uses 8-lane down-projection tiling for single-row decode (`05e-serve-v9-gate`). |
| `ATLAS_MOE_T_LANES_ROWS_DOWN_T8` | `1` | `1` | `0` | `1` | `0` | Uses 8-lane down-projection tiling across multi-row `K=5` verify batches. |
| `ATLAS_QWEN4_K5_BATCH_ATTN_QKV` | `1` | `1` | `0` | `1` | `0` | Batches full-attention QKV projections across all 5 verify rows (`19ef1d8ea`). |
| `ATLAS_QWEN4_K5_BATCH_GDN` | `1` | `1` | `0` | `1` | `0` | Batches the 36-layer Gated DeltaNet recurrence across all 5 verify rows (`39f68603f`, `9fd3f285b`). |
| `ATLAS_QWEN4_K5_BATCH_SSM` | `1` | `1` | `0` | `1` | `0` | Batches `in_proj_qkvz`, `in_proj_ba`, and `out_proj` across all 5 verify rows. |
| `ATLAS_QWEN4_K5_BATCH_HYPER` | `1` | `1` | `0` | `1` | `0` | Batches 4-stream `mHC` projections across all 5 verify rows. |
| `ATLAS_QWEN4_K5_BATCH_CONV` | `1` | `1` | `0` | `1` | `0` | Batches causal Conv1D across all 5 verify rows with exact per-row rollback snapshots (`04bbfb7ca`). |
| `ATLAS_QWEN4_QSA_VERIFY_ROWS`<br/>(`FLUX_QWEN4_QSA_VERIFY_ROWS`) | `1`<br/>(`0` Estonia) | `1` | `0` | `1` | `0` | Batches QSA selector preparation only when the verify frame ends at or before position `2048`. Sparse multi-row attention separately requires every row to be at or beyond `2048`, so the predicates do not overlap in this revision: executed verifier attention remains row-serial. Do not attribute endpoint results to `partial_rows_v2`. `ESTONIA_136K=1` also sets the flag to `0` in `flux-estonia-136k.env`. |
| `ATLAS_QWEN4_QSA_ROUTE_TELEMETRY` | `0` | `0` | `0` | `0` | `0` | Qualification-only receipts for the executed verifier route. Row-serial receipts contain exact dense-paged, sparse-QSA, HSS, and MLA row counts; separate counters cover legacy/V2 multi-row frames if a later revision makes them reachable. Enable only in a fresh-process, eager, `C=1`, `TP=1`, HSS-off arm. Group by request-scoped log delta plus `frame_id` and require the exact QSA layer set. Receipts prove host enqueue, not GPU completion, correctness, latency, or throughput. |
| `ATLAS_QWEN4_VERIFY_OPROJ_ROWS` | `1` | `1` | `0` | `1` | `0` | Batches QSA `o_proj` across all 5 verify rows. |
| `ATLAS_QWEN4_VERIFY_GNORM_ROWS` | `1` | `1` | `0` | `1` | `0` | Batches GDN output gated-RMSNorm across all 5 verify rows. |
| `ATLAS_VERIFY_ARGMAX_ROWS` | `1` | `1` | `0` | `1` | `0` | Computes target verify argmax across all 5 rows in a single device reduction before D2H copy. |
| `ATLAS_QWEN4_VERIFY_GRAPH`<br/>(`FLUX_QWEN4_VERIFY_GRAPH`) | `1` | `1` | `0` | `1` | `0` | Captures and replays PLE-segmented CUDA graphs for the 5-row target verification pass (`9ba551b87`). |
| `ATLAS_QWEN4_VERIFY_FAST_HOST`<br/>(`FLUX_QWEN4_VERIFY_FAST_HOST`) | `1` | `1` | `0` | `1` | `0` | Coalesces verify-step host metadata updates and token acceptance accounting into a single pinned D2H read (`05e-serve-v9-gate`). |
| `ATLAS_W4A16_GEMV_RT2` | `1` | `1` | `1` | `1` | `0` | Selects the RT2 register-tiled W4A16 GEMV kernel for dense projections. |
| `ATLAS_W4A16_EXACT_ROWPAR` | `1` | `1` | `0` | `1` | `0` | Enables row-parallel exact W4A16 GEMV across small verify batches while preserving single-row reduction order. |
| `ATLAS_W4A16_EXACT_SWK` | `1` | `1` | `0` | `1` | `0` | Enables software-pipelined K-loop in exact W4A16 verify GEMVs. |
| `ATLAS_W4A16_EXACT_SW4` | `1` | `1` | `0` | `1` | `0` | Uses 4-warp split reduction in exact W4A16 verify GEMVs. |
| `ATLAS_DFLASH_ASYNC` | `0` | `0` | `0` | `0` | `0` | Keeps asynchronous DFlash stream overlap off (native MTP uses synchronous device-chained `MTP_CHAIN=1`). |
| `ATLAS_DFLASH_SPEC_CYCLE_V2` | `1` | `1` | `0` | `1` | `0` | Emits structured per-cycle speculative acceptance telemetry (`accepted=N/M`) used by `repro/qwen38-flash-next/run.py` and LIL `v0.7.9`. |
