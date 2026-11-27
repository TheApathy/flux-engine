// SPDX-License-Identifier: AGPL-3.0-only

//! DFlash block-diffusion draft head implementing [`DraftProposer`].
//!
//! Block-diffusion drafter (Z Lab, arXiv 2602.06036): a small Qwen3-architecture
//! transformer (8 layers, hidden=2048, GQA 32:4, head_dim=128) that emits γ
//! draft tokens **in a single forward pass** via bidirectional in-block
//! attention. The training `block_size` includes one known anchor row, so a
//! block-16 checkpoint has γ=15 trained drafts.
//! Conditioned on five intermediate hidden states captured from the target
//! model at `target_layer_ids` (e.g., `[1, 10, 19, 28, 37]` for
//! Qwen3.6-35B-A3B-DFlash), projected through a single `fc` layer at model
//! entry — NOT per-layer KV injection (early plan was wrong; cf. vLLM
//! `qwen3_dflash.py`).
//!
//! Implements the full neural [`BlockDiffusionDraftHead`] proposer across
//! single-sequence, cross-sequence batched (`DFlash2`), `DDTree`, `DSpark`
//! (with optional Markov head), probabilistic sampling, and async overlapped
//! propose paths, supporting `Bf16`, `Fp8`, and `Nvfp4` drafter weights.

use parking_lot::Mutex;
use std::any::Any;
use std::time::Instant;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle, PinnedHostBuffer};

use crate::device_allocations::DeviceAllocations;
use crate::speculative::{DraftProposer, ProposerState};
use crate::weight_map::{DenseWeight, QuantizedWeight};

/// Default per-head cap on pooled [`DflashProposerState`] buffer sets.
///
/// Matches steady-state `C=16` concurrency: during active `C=16` decode all 16
/// live sequences already hold their `ctx_hidden_acc`, `ctx_fc_cache`, and
/// `ctx_{k,v}_cache` allocations in UMA, so retaining up to 16 idle sets across
/// batches adds zero peak UMA while eliminating per-sequence `cuMemAlloc_v2`,
/// `cuMemFree_v2`, `cuMemAllocHost_v2`, and `cuMemFreeHost` stalls.
const DEFAULT_SEQ_BUF_POOL_CAP: usize = 16;

/// Hard upper bound on pooled [`DflashProposerState`] buffer sets per head
/// when explicitly raised via `ATLAS_QWEN4_SEQ_BUF_POOL=<N>`.
const MAX_SEQ_BUF_POOL_CAP: usize = 64;

/// Environment selector for per-sequence DFlash buffer pooling.
/// Default ON (`1` / unset -> `min(max_batch_slots, 16)`); `0` disables
/// pooling and frees buffers immediately; numeric `N >= 2` sets an explicit
/// cap clamped to `MAX_SEQ_BUF_POOL_CAP` (64).
const SEQ_BUF_POOL_ENV: &str = "ATLAS_QWEN4_SEQ_BUF_POOL";

/// Exact byte geometry of every device and page-locked host buffer owned by a
/// [`DflashProposerState`]. Pool reuse requires bit-for-bit equality across all
/// fields so incompatible heads or geometries can never cross-pollinate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DflashStateBufferGeometry {
    ctx_allocation_bytes: usize,
    ctx_capacity: usize,
    ctx_slot_bytes: usize,
    max_ctx_len: usize,
    ctx_fc_bytes: usize,
    ctx_kv_bytes: usize,
    num_layers: usize,
    pos_pinned_bytes: usize,
    token_ids_pinned_bytes: usize,
    draft_output_pinned_bytes: usize,
    probabilistic: bool,
    selector_uniforms_bytes: usize,
    selector_q_modifiers_bytes: usize,
    selector_row_bytes: usize,
}

/// One retired [`DflashProposerState`] whose device and pinned-host buffers are
/// idle and safe to reuse by the same owning [`BlockDiffusionDraftHead`].
struct PooledDflashStateBuffers {
    geometry: DflashStateBufferGeometry,
    state: Box<dyn ProposerState>,
}

type DflashStatePoolRegistry =
    Mutex<std::collections::HashMap<usize, Vec<PooledDflashStateBuffers>>>;

fn dflash_state_pool_registry() -> &'static DflashStatePoolRegistry {
    static REGISTRY: std::sync::OnceLock<DflashStatePoolRegistry> = std::sync::OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

/// Resolve the per-head sequence buffer pool cap from an optional
/// `ATLAS_QWEN4_SEQ_BUF_POOL` value and the model's `max_batch_slots`.
fn resolve_seq_buf_pool_cap(raw_env: Option<&str>, max_batch_slots: usize) -> usize {
    let default_cap = if max_batch_slots > 0 {
        max_batch_slots.min(DEFAULT_SEQ_BUF_POOL_CAP)
    } else {
        DEFAULT_SEQ_BUF_POOL_CAP
    };
    let Some(raw) = raw_env else {
        return default_cap;
    };
    let trimmed = raw.trim();
    if trimmed.is_empty()
        || trimmed.eq_ignore_ascii_case("true")
        || trimmed.eq_ignore_ascii_case("on")
    {
        return default_cap;
    }
    if trimmed == "0"
        || trimmed.eq_ignore_ascii_case("false")
        || trimmed.eq_ignore_ascii_case("off")
    {
        return 0;
    }
    match trimmed.parse::<usize>() {
        Ok(0) => 0,
        Ok(1) => default_cap,
        Ok(n) => n.min(MAX_SEQ_BUF_POOL_CAP),
        Err(_) => default_cap,
    }
}

/// True when host `MemAvailable` is at or above the configured UMA floor, or
/// when either reading is unavailable (non-Linux or unset/mocked floor).
fn seq_buf_pool_admits_under_mem_floor(
    mem_available: Option<usize>,
    floor_bytes: Option<usize>,
) -> bool {
    match (mem_available, floor_bytes) {
        (Some(avail), Some(floor)) => avail >= floor,
        _ => true,
    }
}

fn seq_buf_pool_cap_from_env(max_batch_slots: usize) -> usize {
    static ENV_VAL: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let raw = ENV_VAL.get_or_init(|| std::env::var(SEQ_BUF_POOL_ENV).ok());
    resolve_seq_buf_pool_cap(raw.as_deref(), max_batch_slots)
}

/// Discrete-GPU VRAM floor (`1 GiB`) below which idle pooled DFlash sequence
/// buffers are drained and freed back to CUDA instead of held in the pool.
/// On discrete GPUs, sequence KV/hidden caches live in dedicated VRAM, so
/// gating them against host `/proc/meminfo` vs the `12 GiB` UMA floor would
/// either drain on tight host RAM or miss real VRAM pressure.
const DISCRETE_SEQ_BUF_POOL_VRAM_FLOOR_BYTES: usize = 1024 * 1024 * 1024;

#[cfg(feature = "cuda")]
fn discrete_seq_buf_pool_vram_free_bytes() -> Option<usize> {
    spark_runtime::cuda_backend::cuda_free_memory_bytes()
}

#[cfg(not(feature = "cuda"))]
fn discrete_seq_buf_pool_vram_free_bytes() -> Option<usize> {
    None
}

fn seq_buf_pool_mem_floor_ok_for(
    is_integrated: Option<bool>,
    discrete_vram_free: impl FnOnce() -> Option<usize>,
    host_mem_available: impl FnOnce() -> Option<usize>,
    uma_floor_bytes: impl FnOnce() -> Option<usize>,
) -> bool {
    if is_integrated == Some(false) {
        return seq_buf_pool_admits_under_mem_floor(
            discrete_vram_free(),
            Some(DISCRETE_SEQ_BUF_POOL_VRAM_FLOOR_BYTES),
        );
    }
    seq_buf_pool_admits_under_mem_floor(host_mem_available(), uma_floor_bytes())
}

fn seq_buf_pool_mem_floor_ok() -> bool {
    static FLOOR_BYTES: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    seq_buf_pool_mem_floor_ok_for(
        atlas_core::device::active_device_is_integrated(),
        discrete_seq_buf_pool_vram_free_bytes,
        spark_runtime::startup_ledger::mem_available_bytes,
        || {
            *FLOOR_BYTES.get_or_init(|| {
                spark_runtime::startup_ledger::effective_mem_floor_bytes()
                    .ok()
                    .or_else(|| spark_runtime::startup_ledger::memavailable_floor_bytes().ok())
            })
        },
    )
}

fn free_dflash_state_device_buffers(gpu: &dyn GpuBackend, ds: &mut DflashProposerState) {
    let acc = std::mem::replace(&mut ds.ctx_hidden_acc, DevicePtr::NULL);
    let fc = std::mem::replace(&mut ds.ctx_fc_cache, DevicePtr::NULL);
    let k_ptrs = std::mem::take(&mut ds.ctx_k_cache);
    let v_ptrs = std::mem::take(&mut ds.ctx_v_cache);
    for ptr in std::iter::once(acc)
        .chain(std::iter::once(fc))
        .chain(k_ptrs)
        .chain(v_ptrs)
    {
        if !ptr.is_null() {
            crate::gpu_lifetime::require_completion(gpu.free(ptr), "DFlash state buffer free");
        }
    }
}

fn free_dflash_state_device_buffers_best_effort(
    gpu: &dyn GpuBackend,
    ds: &mut DflashProposerState,
) {
    let acc = std::mem::replace(&mut ds.ctx_hidden_acc, DevicePtr::NULL);
    let fc = std::mem::replace(&mut ds.ctx_fc_cache, DevicePtr::NULL);
    let k_ptrs = std::mem::take(&mut ds.ctx_k_cache);
    let v_ptrs = std::mem::take(&mut ds.ctx_v_cache);
    for ptr in std::iter::once(acc)
        .chain(std::iter::once(fc))
        .chain(k_ptrs)
        .chain(v_ptrs)
    {
        if !ptr.is_null()
            && let Err(error) = gpu.free(ptr)
        {
            tracing::error!(
                "BlockDiffusionDraftHead::release_rejection_resources: free(state buffer {ptr}) failed: {error:#}"
            );
        }
    }
}

/// Reclaim a retired [`DflashProposerState`] into its owning head's bounded
/// buffer pool, or free its device buffers immediately when pooling is disabled
/// (`ATLAS_QWEN4_SEQ_BUF_POOL=0`), the pool is at capacity, or host
/// `MemAvailable` is below the effective floor
/// (`max(ATLAS_MEMAVAILABLE_FLOOR_MIB, FLUX_MEM_FLOOR_MIB)`). Under a floor
/// breach, already-pooled idle states for the same head are also drained and
/// freed back to UMA.
///
/// Caller must have synchronized all GPU streams that could read or write this
/// state's device or pinned-host buffers before invoking this function.
pub(crate) fn reclaim_dflash_state_buffers(
    gpu: &dyn GpuBackend,
    state: &mut Option<Box<dyn ProposerState>>,
    max_batch_slots: usize,
) {
    let pool_cap = seq_buf_pool_cap_from_env(max_batch_slots);
    let mem_floor_ok = pool_cap == 0 || seq_buf_pool_mem_floor_ok();
    reclaim_dflash_state_buffers_with_policy(gpu, state, pool_cap, mem_floor_ok);
}

fn reclaim_dflash_state_buffers_with_policy(
    gpu: &dyn GpuBackend,
    state: &mut Option<Box<dyn ProposerState>>,
    pool_cap: usize,
    mem_floor_ok: bool,
) {
    if !state
        .as_deref()
        .is_some_and(|ps| ps.as_any().is::<DflashProposerState>())
    {
        return;
    }
    let Some(mut boxed) = state.take() else {
        return;
    };
    let (owner_head, geometry, buffers_complete) = {
        let Some(ds) = boxed.as_any_mut().downcast_mut::<DflashProposerState>() else {
            return;
        };

        // Always scrub tenant tokens, pinned host mirrors, cursors, and
        // transient proposal metadata before either pooling or freeing.
        ds.scrub_for_retirement();

        let owner_head = ds.pool_owner_head;
        let geometry = ds.buffer_geometry();
        let has_sel_uniforms = ds.selector_uniforms_pinned.is_some();
        let selector_pins_consistent = has_sel_uniforms == ds.selector_q_modifiers_pinned.is_some()
            && has_sel_uniforms == ds.selector_candidates_pinned.is_some()
            && has_sel_uniforms == ds.selector_q_pinned.is_some()
            && ds
                .selector_candidates_pinned
                .as_ref()
                .map(PinnedHostBuffer::len)
                == ds.selector_q_pinned.as_ref().map(PinnedHostBuffer::len);
        let buffers_complete = owner_head != 0
            && !ds.ctx_hidden_acc.is_null()
            && (geometry.ctx_fc_bytes == 0 || !ds.ctx_fc_cache.is_null())
            && ds.ctx_k_cache.len() == ds.ctx_v_cache.len()
            && ds.ctx_k_cache.iter().all(|p| !p.is_null())
            && ds.ctx_v_cache.iter().all(|p| !p.is_null())
            && selector_pins_consistent;
        (owner_head, geometry, buffers_complete)
    };

    if !mem_floor_ok {
        // Under UMA floor pressure, release already-pooled idle states for
        // this head back to the driver instead of leaving them pinned.
        if owner_head != 0 {
            let drained = dflash_state_pool_registry().lock().remove(&owner_head);
            if let Some(entries) = drained {
                for mut entry in entries {
                    if let Some(ds) = entry
                        .state
                        .as_any_mut()
                        .downcast_mut::<DflashProposerState>()
                    {
                        free_dflash_state_device_buffers(gpu, ds);
                    }
                }
            }
        }
    } else if pool_cap > 0 && buffers_complete {
        let mut registry = dflash_state_pool_registry().lock();
        let entries = registry.entry(owner_head).or_default();
        if entries.len() < pool_cap {
            entries.push(PooledDflashStateBuffers {
                geometry,
                state: boxed,
            });
            return;
        }
    }

    if let Some(ds) = boxed.as_any_mut().downcast_mut::<DflashProposerState>() {
        free_dflash_state_device_buffers(gpu, ds);
    }
}

/// Drain and free every pooled [`DflashProposerState`] owned by `head_id`.
/// Called from [`BlockDiffusionDraftHead::release_rejection_resources`] during
/// model teardown so zero pooled GPU allocations leak on drop.
fn drain_dflash_state_pool_for_head(head_id: usize, gpu: &dyn GpuBackend) {
    let drained = dflash_state_pool_registry().lock().remove(&head_id);
    if let Some(entries) = drained {
        for mut entry in entries {
            if let Some(ds) = entry
                .state
                .as_any_mut()
                .downcast_mut::<DflashProposerState>()
            {
                free_dflash_state_device_buffers_best_effort(gpu, ds);
            }
        }
    }
}

/// Remove any remaining pool registry entry for `head_id` when
/// [`BlockDiffusionDraftHead`] drops, preventing address-reuse collisions if a
/// caller dropped the head without invoking `release_rejection_resources`.
fn drop_dflash_state_pool_entry(head_id: usize) -> usize {
    let removed = dflash_state_pool_registry().lock().remove(&head_id);
    removed.as_ref().map_or(0, Vec::len)
}

#[cfg(test)]
fn pooled_state_count_for_head(head_id: usize) -> usize {
    dflash_state_pool_registry()
        .lock()
        .get(&head_id)
        .map_or(0, Vec::len)
}

/// Cached gate for `ATLAS_DFLASH_DEEPLOOP=1` (multi-pass denoise residual scaling).
///
/// Defined in the head module so `forward_block_layer` and `noise_pass` can
/// both import it without a circular dependency.
pub(crate) fn dflash_deeploop_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        let on = std::env::var("ATLAS_DFLASH_DEEPLOOP").ok().as_deref() == Some("1");
        if on {
            tracing::info!("DFlash DeepLoop residual scaling ENABLED (ATLAS_DFLASH_DEEPLOOP=1)");
        }
        on
    })
}

/// Resolve the effective neural/host-draft width against the trained maximum.
/// The scheduler's runtime arm override has precedence over the environment,
/// matching the forward path's established contract.
fn resolve_effective_draft_width(
    trained_drafts: usize,
    runtime_override: Option<usize>,
    env_cap: Option<usize>,
) -> usize {
    runtime_override
        .or(env_cap)
        .unwrap_or(trained_drafts)
        .min(trained_drafts)
        .max(1)
}

pub(crate) fn effective_draft_width(trained_drafts: usize) -> usize {
    let env_cap = std::env::var("ATLAS_DFLASH_DRAFT_CAP")
        .ok()
        .and_then(|s| s.parse::<usize>().ok());
    resolve_effective_draft_width(
        trained_drafts,
        crate::speculative::dflash_gamma_override(),
        env_cap,
    )
}

/// DFlash verify width (drafts + 1 rows) at the current launch/runtime cap.
pub fn effective_verify_rows(trained_drafts: usize) -> usize {
    effective_draft_width(trained_drafts) + 1
}

#[cfg(test)]
mod effective_draft_width_tests {
    use super::resolve_effective_draft_width;

    #[test]
    fn width_never_exceeds_the_trained_rows() {
        assert_eq!(resolve_effective_draft_width(15, None, None), 15);
        assert_eq!(resolve_effective_draft_width(15, None, Some(16)), 15);
        assert_eq!(resolve_effective_draft_width(6, None, Some(4)), 4);
    }

    #[test]
    fn runtime_arm_cap_precedes_environment_cap() {
        assert_eq!(resolve_effective_draft_width(15, Some(3), Some(8)), 3);
        assert_eq!(resolve_effective_draft_width(15, Some(0), Some(8)), 1);
    }

    #[test]
    fn pld_effective_width_cannot_recreate_the_untrained_tail() {
        // `propose.rs` uses this same resolver before slicing a PLD hit.
        assert_eq!(resolve_effective_draft_width(15, None, Some(16)), 15);
        assert_eq!(resolve_effective_draft_width(15, Some(4), Some(12)), 4);
        assert_eq!(resolve_effective_draft_width(6, None, None), 6);
    }
}

/// Compile-time cap on the per-position top-K used by the DDTree (M4B v2)
/// builder. Must match `MAX_TOP_K` in `kernels/gb10/common/argmax_bf16.cu`.
/// Runtime `top_k` comes from `ATLAS_DDTREE_TOP_K` (default 8) and is
/// validated against this maximum; invalid values fail closed without launch.
pub const DDTREE_TOP_K_MAX: usize = 16;

/// Maximum cross-sequence cohort width for batched DFlash2 Stage-2 metadata.
pub const DFLASH_BATCH_MAX_SEQS: usize = 64;

/// Minimum row capacity for DFlash2 selector scratch buffers so cross-sequence
/// batched proposals up to 256 total output rows (`C=32 x K=8`) fit without
/// extra allocation.
pub const DFLASH_BATCH_MAX_SEL_ROWS: usize = 256;

/// Widened selector shortlist capacity (`K <= 32`) used by `topk_bf16_radix`
/// and `dflash2_selector_walk_gumbel_batched_k32` so `top_k=20, top_p=0.95`
/// sampling does not prematurely truncate at `DDTREE_TOP_K_MAX` (16).
pub const DFLASH2_GUMBEL_SHORTLIST_K: usize = 32;

/// Minimum f32 slot count allocated for `DflashScratch::batch_out_probs`
/// (`64 * 8 = 512` f32s = 2048 B, shared by C=1 and batched C=2..64).
pub const DFLASH_BATCH_OUT_PROBS_MIN_SLOTS: usize = DFLASH_BATCH_MAX_SEQS * 8;

/// Kernel handles for the DFlash γ-block forward chain. All resolved once
/// at `BlockDiffusionDraftHead::from_weights` against the active GPU backend
/// (which compiles target-specific PTX at startup); subsequent
/// `propose()` calls just `KernelLaunch::new(...).launch(stream)`.
pub struct DflashKernels {
    pub rms_norm: KernelHandle,
    pub dense_gemv: KernelHandle,
    pub dense_gemm: KernelHandle,
    pub rope_qwen3: KernelHandle,
    pub silu_mul: KernelHandle,
    pub residual_add: KernelHandle,
    /// BF16 scaled accumulate: `output[i] += scale * src[i]`. Same module
    /// as `residual_add` (`residual_add.cu`). Used by DeepLoop multi-pass
    /// residual scaling (`ATLAS_DFLASH_DEEPLOOP=1`).
    pub scaled_add: KernelHandle,
    /// GPU-side token recommit for DeepLoop multi-pass: reconstructs
    /// `draft_tokens_dev` for pass N+1 without host D2H (async-safe).
    /// Same module as `scaled_add` (`residual_add.cu`).
    pub token_recommit: KernelHandle,
    pub argmax: KernelHandle,
    /// Row-parallel `argmax_bf16_rows` (`argmax` module); sentinel 0 on miss,
    /// falling back to the per-row `argmax_bf16` loop.
    pub argmax_rows: KernelHandle,
    /// DFlash 2 grouped-dynamic-conv stage 0 (`prepare`): convolves the
    /// normed noise rows in place and exports the stage-1 dynamic rows.
    /// Resolved only for DFlash2 checkpoints; sentinel zero otherwise.
    pub dflash2_conv_prepare: KernelHandle,
    /// DFlash 2 grouped-dynamic-conv stage 1 (`finish`): convolves the
    /// sublayer output with the stage-1 dynamic rows. Resolved only for
    /// DFlash2 checkpoints; sentinel zero otherwise.
    pub dflash2_conv_finish: KernelHandle,
    /// Cross-sequence batched DFlash 2 conv stage 0 (`dflash2_conv_prepare_batched`),
    /// resetting causal padding every `block_len` rows. Sentinel zero on miss.
    pub dflash2_conv_prepare_batched: KernelHandle,
    /// Cross-sequence batched DFlash 2 conv stage 1 (`dflash2_conv_finish_batched`),
    /// resetting causal padding every `block_len` rows. Sentinel zero on miss.
    pub dflash2_conv_finish_batched: KernelHandle,
    /// DFlash 2 candidate-selector greedy walk (top-k codebook scores).
    /// Resolved only for DFlash2 checkpoints; sentinel zero otherwise.
    pub dflash2_selector_walk: KernelHandle,
    /// Cross-sequence batched DFlash 2 candidate-selector greedy walk
    /// (`dflash2_selector_walk_batched`), reading per-seq `anchor_ids` from device memory.
    /// Sentinel zero on miss.
    pub dflash2_selector_walk_batched: KernelHandle,
    /// Cross-sequence batched DFlash 2 CRN-coupled Gumbel-Max candidate-selector walk
    /// (`dflash2_selector_walk_gumbel_batched`). Handles both greedy (`temp <= 0`)
    /// and noise-coupled Gumbel-Max (`temp > 0`) in a single kernel ABI so C=1
    /// Stage-2 CUDA graphs and C=2..16 batched cohorts share one symbol without
    /// recapturing. Sentinel zero on miss.
    pub dflash2_selector_walk_gumbel_batched: KernelHandle,
    /// Widened K<=32 cross-sequence batched DFlash 2 CRN-coupled Gumbel-Max
    /// candidate-selector walk (`dflash2_selector_walk_gumbel_batched_k32`),
    /// supporting dynamic `top_k_shortlist` up to `DFLASH2_GUMBEL_SHORTLIST_K` (32)
    /// and writing per-token selected draft probabilities into `batch_out_probs`.
    /// Sentinel zero on miss.
    pub dflash2_selector_walk_gumbel_batched_k32: KernelHandle,
    /// Sampled DFlash2 selector. Separate from the released greedy symbol so
    /// greedy launch arguments and generated code remain unchanged.
    pub dflash2_selector_walk_probabilistic: KernelHandle,
    /// Cross-sequence batched probabilistic DFlash2 selector for maximal-coupling
    /// rejection sampling (`dflash2_selector_walk_probabilistic_batched`), writing
    /// normalized shortlist `(q_cand_ids, q_probs)` directly into GPU shared scratch
    /// at per-sequence `out_q_row_bases[b]`. Sentinel zero on miss.
    pub dflash2_selector_walk_probabilistic_batched: KernelHandle,
    /// Opt-in 4-gate safe in-context n-gram bonus DFlash2 selector walk
    /// (`dflash2_selector_walk_ngram_batched`, `ATLAS_DFLASH_NGRAM_BONUS=1`, T=0 only).
    /// Sentinel zero on miss.
    pub dflash2_selector_walk_ngram_batched: KernelHandle,
    /// Cross-sequence batched HD=128 split-context attention
    /// (`inferspark_prefill_split_ctx_h128_batched`) reading circular PostRope
    /// per-sequence `ctx_{k,v}_cache` via pointer tables. Sentinel zero on miss.
    pub split_ctx_h128_batched: KernelHandle,
    /// Exact `argmax(base_logits + markov_bias)` with lowest-token tie-break.
    /// Keeps the DSpark left-to-right Markov chain on the producer stream.
    /// Resolved fail-closed for Markov checkpoints; sentinel zero for generic
    /// DFlash so this algorithm-specific kernel does not broaden its contract.
    pub argmax_add: KernelHandle,
    /// Top-K over BF16 logits — used by the DDTree (M4B v2) propose path
    /// to seed per-position branch candidates. Same `argmax` module as
    /// `argmax_bf16` (shared .cu file). Resolved unconditionally; never
    /// invoked under flat DFlash, only when the propose path opts into
    /// the real (non-chain-only) DDTree topology builder.
    pub topk: KernelHandle,
    /// Fast radix top-K over BF16 logits (`topk_bf16_radix`, `k in 1..=32`).
    /// Bit-exact with `topk_bf16` for `k <= 16` and supports widened `k = 32`
    /// shortlist extraction for DFlash2 Gumbel-Max sampling. Sentinel zero on miss.
    pub topk_bf16_radix: KernelHandle,
    /// Byte-exact 16-byte-aligned strided row copy (`strided_copy_rows_16`),
    /// used by batched Stage 2 to compact `B` sequences' output rows in one
    /// CUDA-graph-capturable launch. Sentinel zero on miss.
    pub strided_copy_rows: KernelHandle,
    pub batched_embed: KernelHandle,
    /// Non-paged prefill attention (used for the γ-block self-attention
    /// when there's no persistent K/V cache to walk).
    pub prefill_attn: KernelHandle,
    /// NVFP4 W4A16 GEMV (M=1) — used for the per-row `fc` projection when
    /// the drafter is built with `DflashQuantization::Nvfp4`. BF16 build
    /// leaves this set to the same handle as `dense_gemv` and never
    /// dispatches to it.
    pub w4a16_gemv: KernelHandle,
    /// NVFP4 W4A16 GEMM (M>1) — used for q/k/v/o + gate/up/down per-layer
    /// projections under NVFP4. BF16 build leaves this set to the same
    /// handle as `dense_gemm` and never dispatches to it.
    pub w4a16_gemm: KernelHandle,
    /// Byte-exact cp.async-pipelined shadow of `w4a16_gemm`. V3 uses this
    /// only for large-M prompt-context ingestion when
    /// `ATLAS_DFLASH_PREFILL_PIPE=1`; small-M decode remains on the dedicated
    /// transposed gamma kernels. Optional for the default-off policy; an
    /// explicit bulk-prefill request fails closed when an older kernel cache
    /// lacks the symbol, instead of silently benchmarking the baseline.
    pub w4a16_gemm_pipe: KernelHandle,
    /// Shared-input dual-output shadow of `w4a16_gemm_pipe`. During V3 bulk
    /// context ingestion it projects K and V in one launch while retaining
    /// the same two independent BF16 cache buffers. The explicit prefill gate
    /// fails closed when this optional symbol is absent.
    pub w4a16_gemm_pipe_dual: KernelHandle,
    /// NVFP4 W4A16 GEMM M_TILE=16 specialization (`w4a16_gemm_t_m16`) —
    /// requires transposed `nvfp4_t` weight layout. Only used by the
    /// drafter FFN when `ATLAS_DFLASH_FFN_KGAMMA=1`. Resolved via
    /// `try_kernel`; handle is `KernelHandle(0)` (sentinel) on miss so
    /// the dispatch can degrade gracefully to the M_TILE=64 path.
    pub w4a16_gemm_t_m16: KernelHandle,
    /// `w4a16_gemm_t_m32_n64` — single B read × full occupancy at M ≤ 32.
    /// Preferred over m16 (2× B reads at M=17) for drafter kgamma GEMMs.
    pub w4a16_gemm_t_m32_n64: KernelHandle,
    /// `w4a16_gemm_t_m32_n64_gateup_silu` — FUSED gate_proj + up_proj +
    /// SiLU·mul in one launch (A loaded once, both B streams, one [M,N]
    /// write). The drafter FFN dispatches to it when the transposed gate/up
    /// weights are present, replacing two m32_n64 GEMMs + the standalone
    /// silu_mul launch. `try_kernel`; sentinel 0 on miss → fall back to the
    /// separate-GEMM path.
    pub w4a16_gemm_t_m32_n64_gateup_silu: KernelHandle,
    /// `w4a16_gemm_t_m32_n32_gateup_silu_pipe_k64` — `N_TILE=32, K_STEP=64`
    /// register-dequant fused gate+up+SiLU for wide discrete GPUs (`> 64` SMs).
    pub w4a16_gemm_t_m32_n32_gateup_silu_pipe_k64: KernelHandle,
    /// `w4a16_gemm_t_m32_n64_splitk` + `reduce_splitk_f32_to_bf16` — the
    /// K-sliced variant of `w4a16_gemm_t_m32_n64` and its FP32 band reducer.
    /// Used only when `ATLAS_DFLASH_DRAFT_SPLITK` >= 2; see `draft_splitk.rs`
    /// for why the drafter's narrow-N projections are occupancy-starved.
    pub w4a16_gemm_t_m32_n64_splitk: KernelHandle,
    pub reduce_splitk_k: KernelHandle,
    /// `fp8_gemm_t` (BF16 A × FP8-E4M3 B) — the propose lm_head FP8 fast
    /// path (`ATLAS_DFLASH_LM_HEAD_FP8=1`). `try_kernel`; sentinel 0 on miss.
    pub fp8_gemm_t: KernelHandle,
    /// `gemv_fp8w::dense_gemv_fp8w` — M=1 row-scaled FP8 GEMV for
    /// `DflashQuantization::Fp8`.
    pub dense_gemv_fp8w: KernelHandle,
    /// `fp8_gemm_dflash::fp8_gemm_t_m16_n64_rowscale` — small-M row-scaled
    /// FP8 GEMM for `DflashQuantization::Fp8`.
    pub fp8_gemm_dflash: KernelHandle,
}

/// Per-step scratch buffers for the γ-block forward.
///
/// Sized for `n_attn_slots = ctx_window + γ` rows, where ctx_window is the
/// max number of past target positions the drafter attends to per step. The
/// first `ctx_window` slots hold post-`fc` projected target context (K/V
/// only — Q is zero-padded); the next γ slots hold the noise tokens.
///
/// At γ=16 and ctx_window=γ=16: 32 rows × 2048 BF16 × ~10 buffers = ~1.3 MB
/// per head. lm_head logits buffer is the largest single alloc:
/// 32 × 248320 × 2 = 15 MB.
pub struct DflashScratch {
    pub stream_buf: DevicePtr,
    pub norm_buf: DevicePtr,
    pub q_buf: DevicePtr,
    pub k_buf: DevicePtr,
    pub v_buf: DevicePtr,
    pub attn_out: DevicePtr,
    pub mlp_intermediate: DevicePtr,
    pub mlp_up: DevicePtr,
    pub stream_acc: DevicePtr,
    /// `[ctx_window, draft_hidden]` BF16 — fc-projected + hidden_norm'd
    /// ctx for the most recent `ctx_window` target positions.
    pub fc_proj: DevicePtr,
    pub logits: DevicePtr,
    pub draft_tokens_dev: DevicePtr,
    /// Secondary double-buffered draft token output slot for consecutive
    /// overlapping async batched (`C >= 2`) proposals (zero aliasing with
    /// `draft_tokens_dev`).
    pub draft_tokens_dev_alt: DevicePtr,
    /// `[ctx_window + γ]` i32 positions. First ctx_window are
    /// historical target positions (decoded indices); last γ are
    /// the to-be-predicted noise positions.
    pub position_ids: DevicePtr,
    /// DDTree M4B v2: per-MASK-position top-K token IDs. Layout
    /// `[γ, DDTREE_TOP_K_MAX]` u32, written by the `topk_bf16` kernel
    /// after lm_head and read host-side to seed the DDTree builder.
    /// Allocated for the maximum compile-time K (16); per-call `k` may
    /// be smaller (env-configurable via `ATLAS_DDTREE_TOP_K`).
    pub topk_tokens_dev: DevicePtr,
    /// Matching f32 logit scores for `topk_tokens_dev`, shape
    /// `[γ, DDTREE_TOP_K_MAX]`. Caller converts to log-probs in Rust by
    /// row-wise softmax over the K selected logits (or simpler heuristic
    /// — see `propose.rs` for the score-translation policy).
    pub topk_logits_dev: DevicePtr,
    /// EAGLE-3.1 per-layer FC-normalization scratch (ATLAS_DFLASH_FC_LAYERNORM=1).
    /// Holds one fc-input slot's `[n_target_layers * target_hidden]` BF16 after
    /// each captured target-layer slice has been unit-variance RMS-normalized
    /// independently (so no single high-magnitude late-layer capture dominates
    /// the fused fc input). The fc GEMV reads from here instead of the raw
    /// `ctx_hidden_acc` slot when the flag is on. Sized
    /// `n_target_layers * target_hidden` BF16. Unused (and may be a null/stub
    /// pointer) when the flag is off — Step 0 only touches it on the ON path.
    pub fc_norm_in: DevicePtr,
    /// All-ones BF16 (`0x3F80`) weight of length `target_hidden`, used as the
    /// per-slice RMS-norm scale for the unit-variance FC-layernorm variant.
    /// `rms_norm_vanilla` computes `x * w / RMS(x)`, so BF16 `1.0` weights
    /// yield plain unit-variance normalization (`x / RMS(x)`, variant (a): no
    /// learned gamma, the safest / least-OOD choice given `self.fc` was trained
    /// on un-normalized concat). Allocated and initialized to `1.0` once at
    /// construction.
    pub fc_norm_ones_w: DevicePtr,
    /// DSpark Markov head scratch — the gathered `markov_w1[prev]` row
    /// (`[rank]` BF16). Only allocated (non-null) when the head is present;
    /// `DevicePtr::NULL` otherwise. Input to the `w2` bias GEMV.
    pub markov_w1_row: DevicePtr,
    /// DSpark Markov head scratch — the per-position bias `B(prev)`
    /// (`[vocab]` BF16), the output of `dense_gemv(w1_row, w2)`. Added to the
    /// base logit row before argmax. `DevicePtr::NULL` when no Markov head.
    pub markov_bias: DevicePtr,
    /// DSpark Markov head scratch — the seed token id (`[1]` u32) fed to the
    /// first `batched_embed` gather. Later positions read the preceding u32
    /// directly from `draft_tokens_dev`, so no host round-trip or D2D copy is
    /// needed. `DevicePtr::NULL` when no Markov head.
    pub markov_prev_dev: DevicePtr,
    /// DFlash 2 conv scratch — the `kernel_projection` output
    /// `[n_attn, 2*kernel_size*groups]` BF16 (both stages). Allocated only
    /// for DFlash2 checkpoints; `DevicePtr::NULL` otherwise.
    pub conv_dyn: DevicePtr,
    /// DFlash 2 conv scratch — the stage-1 (finish) dynamic rows exported by
    /// `prepare`: `[n_attn, kernel_size*groups]` BF16. `DevicePtr::NULL` for
    /// non-DFlash2 checkpoints.
    pub conv_dyn1: DevicePtr,
    /// DFlash 2 conv scratch — non-aliased conv output `[n_attn, hidden]`
    /// BF16. The conv kernel's causal taps read a *previous row* of the same
    /// buffer it writes, so in-place would race; the kernel writes this
    /// buffer and the caller D2D-copies the noise slice back to
    /// norm_buf/stream_acc. `DevicePtr::NULL` for non-DFlash2 checkpoints.
    pub conv_out: DevicePtr,
    /// DFlash 2 selector scratch — the `hidden_projection` output
    /// `[gamma, selector_rank]` BF16. `DevicePtr::NULL` for non-DFlash2
    /// checkpoints.
    pub selector_hidden: DevicePtr,
    /// Sampled-selector uniforms `[gamma]` f32. NULL for released greedy.
    pub selector_uniforms: DevicePtr,
    /// Fixed-layout sampled-q modifier table: bounded token IDs followed by
    /// f32 finite biases or internal `-inf` hard-mask sentinels. NULL for
    /// released greedy.
    pub selector_q_modifiers: DevicePtr,
    /// Sampled-selector normalized sparse q `[gamma, 16]` f32. NULL for
    /// released greedy.
    pub selector_q_probabilities: DevicePtr,
    /// Candidate IDs paired with `selector_q_probabilities`, `[gamma, 16]`
    /// u32. The explicit output prevents a later top-k reuse from changing
    /// the row that is transported to the scheduler.
    pub selector_q_candidates: DevicePtr,
    /// FP32 split-K partial-product workspace `[k_splits, 32, n]` for the
    /// drafter's own projections. `DevicePtr::NULL` unless
    /// `ATLAS_DFLASH_DRAFT_SPLITK` >= 2 was set at load time (allocation must
    /// happen before CUDA graph capture, so it is not lazily created).
    pub splitk_ws: DevicePtr,
    /// Fixed-capacity (`[num_layers, DFLASH_BATCH_MAX_SEQS]` u64) device pointer
    /// table holding per-sequence `ctx_k_cache[l]` bases for cross-sequence
    /// batched Stage-2 split-context attention. Pre-uploaded once before layer 0.
    pub batch_ctx_k_ptrs: DevicePtr,
    /// Fixed-capacity (`[num_layers, DFLASH_BATCH_MAX_SEQS]` u64) device pointer
    /// table holding per-sequence `ctx_v_cache[l]` bases for cross-sequence
    /// batched Stage-2 split-context attention. Pre-uploaded once before layer 0.
    pub batch_ctx_v_ptrs: DevicePtr,
    /// Fixed-capacity (`[DFLASH_BATCH_MAX_SEQS]` u32) per-sequence effective
    /// context lengths for cross-sequence batched Stage-2 attention.
    pub batch_eff_ctx: DevicePtr,
    /// Fixed-capacity (`[DFLASH_BATCH_MAX_SEQS]` u32) per-sequence circular
    /// ring start indices (`needed_start % ctx_window`) for batched Stage-2.
    pub batch_ring_start: DevicePtr,
    /// Fixed-capacity (`[DFLASH_BATCH_MAX_SEQS]` u32) per-sequence anchor token
    /// IDs (`last_tokens`) for `dflash2_selector_walk_batched`.
    pub batch_anchor_ids: DevicePtr,
    /// Fixed-capacity (`[DFLASH_BATCH_MAX_SEQS]` u64) per-sequence SplitMix64
    /// CRN seeds for `dflash2_selector_walk_gumbel_batched`.
    pub batch_gumbel_seeds: DevicePtr,
    /// Fixed-capacity (`[DFLASH_BATCH_MAX_SEQS]` u64) per-sequence step-0
    /// absolute positions for `dflash2_selector_walk_gumbel_batched`.
    pub batch_gumbel_step0_pos: DevicePtr,
    /// Fixed-capacity (`[DFLASH_BATCH_MAX_SEQS]` f32) per-sequence sampling
    /// temperatures (`0.0` = exact greedy argmax) for `dflash2_selector_walk_gumbel_batched`.
    pub batch_gumbel_temps: DevicePtr,
    /// Fixed-capacity (`[DFLASH_BATCH_MAX_SEQS]` u32) per-sequence top-k caps
    /// (`0` = disabled) for `dflash2_selector_walk_gumbel_batched`.
    pub batch_gumbel_top_ks: DevicePtr,
    /// Fixed-capacity (`[DFLASH_BATCH_MAX_SEQS]` f32) per-sequence nucleus top-p
    /// thresholds for `dflash2_selector_walk_gumbel_batched`.
    pub batch_gumbel_top_ps: DevicePtr,
    /// Fixed-capacity (`[DFLASH_BATCH_MAX_SEQS]` f32) per-sequence min-p
    /// thresholds for `dflash2_selector_walk_gumbel_batched`.
    pub batch_gumbel_min_ps: DevicePtr,
    /// Fixed-capacity per-token selected draft probabilities (`f32`) written by
    /// `dflash2_selector_walk_gumbel_batched_k32` (at least `64 * 8 = 512` f32s = 2048 B,
    /// shared by C=1 and batched C=2..64).
    pub batch_out_probs: DevicePtr,
    /// Secondary double-buffered draft probability output slot paired with
    /// `draft_tokens_dev_alt` for consecutive overlapping async batched proposals.
    pub batch_out_probs_alt: DevicePtr,
    /// Opt-in (`ATLAS_DFLASH_EARLY_STAGE1=1`) isolated staging slab
    /// `[physical_verify_k, hidden_size]` BF16 for early-started Stage-1
    /// `fc` + `hidden_norm` projections computed on `pstream` after Layer 61 / 45
    /// completes. `DevicePtr::NULL` when the flag is OFF.
    pub early_stage1_fc: DevicePtr,
    /// Fixed-capacity (`[DFLASH_BATCH_MAX_SEQS]` u32 = 256 B) per-sequence
    /// base row indices (`slot_idx * 8`) into the shared `(q_cand_ids, q_probs)`
    /// GPU buffer for `dflash2_selector_walk_probabilistic_batched`.
    pub q_row_bases: DevicePtr,
    /// Non-owning pointer to model-owned `q_cand_ids: [row_cap, q_stride] u32`
    /// bound via `DraftProposer::bind_rejection_q_scratch`.
    pub shared_q_ids: DevicePtr,
    /// Non-owning pointer to model-owned `q_probs: [row_cap, q_stride] f32`
    /// bound via `DraftProposer::bind_rejection_q_scratch`.
    pub shared_q_probs: DevicePtr,
    /// Row capacity of `shared_q_ids` / `shared_q_probs` (512 when bound).
    pub shared_q_row_cap: usize,
    /// Candidate stride of `shared_q_ids` / `shared_q_probs` (20 when bound).
    pub shared_q_stride: usize,
    /// Per-sequence open-addressed n-gram tables `[DFLASH_BATCH_MAX_SEQS, 512]`
    /// (`NgramTableEntry`, 512 KiB) for `dflash2_selector_walk_ngram_batched`.
    /// Allocated ONLY when `ATLAS_DFLASH_NGRAM_BONUS=1` and a selector is present;
    /// `DevicePtr::NULL` when OFF (zero VRAM overhead on baseline).
    pub ngram_tables_dev: DevicePtr,
    /// Per-sequence `[t-3, t-2, t-1]` history tokens `[DFLASH_BATCH_MAX_SEQS, 3]` u32
    /// (768 B) for `dflash2_selector_walk_ngram_batched`. `DevicePtr::NULL` when OFF.
    pub ngram_prev3_dev: DevicePtr,
}

impl DflashScratch {
    /// Select the primary or secondary double-buffered draft-token output device pointer.
    #[inline]
    #[must_use]
    pub const fn draft_tokens_dev_for_slot(
        &self,
        slot: async_propose::AsyncScratchSlot,
    ) -> DevicePtr {
        match slot {
            async_propose::AsyncScratchSlot::Primary => self.draft_tokens_dev,
            async_propose::AsyncScratchSlot::Secondary => self.draft_tokens_dev_alt,
        }
    }

    /// Select the primary or secondary double-buffered draft-probability output device pointer.
    #[inline]
    #[must_use]
    pub const fn batch_out_probs_for_slot(
        &self,
        slot: async_propose::AsyncScratchSlot,
    ) -> DevicePtr {
        match slot {
            async_propose::AsyncScratchSlot::Primary => self.batch_out_probs,
            async_propose::AsyncScratchSlot::Secondary => self.batch_out_probs_alt,
        }
    }
}

/// Drafter-side weight precision.
///
/// * `Bf16` — default. Loads weights verbatim from the drafter checkpoint
///   (BF16 in both `z-lab/Qwen3.6-{27B,35B-A3B}-DFlash`). Stable but slow:
///   the BF16 `dense_gemm` is bandwidth-bound on GB10 and dominates the
///   ~134 ms / propose-step at γ=16, ctx_window=16.
/// * `Fp8` — online-quantize the 7 dense projections per layer plus `fc` to
///   row-scaled FP8-E4M3 at model-load time and dispatch through the drafter
///   FP8 GEMV/GEMM kernels (`dense_gemv_fp8w` / `fp8_gemm_dflash`), halving
///   weight bandwidth vs BF16 with higher numerical fidelity than 4-bit.
/// * `Nvfp4` — runtime-quantize (or consume pre-quantized ModelOpt NVFP4 for)
///   the 7 dense projections per layer plus the `fc` target-context projection
///   at model-load time, then dispatch the per-step forward through the same
///   fast `w4a16_gemv` / `w4a16_gemm` kernels the target model already uses.
///   RMSNorm + bias weights stay BF16 (kernels expect BF16 inputs for those
///   small ops). Cuts propose latency by ~3-5× on the GEMM-dominated path.
///   Acceptance parity vs BF16 has been observed for the target NVFP4 path;
///   matching parity for the drafter is acceptable because the verify step
///   always uses the target's logits, so a small drafter-side numerical drift
///   just reduces accept rate, never produces wrong tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DflashQuantization {
    Bf16,
    Fp8,
    Nvfp4,
}

/// DSpark VanillaMarkov head (runtime form) — the low-rank bigram bias
/// `B(prev) = markov_w2 @ markov_w1[prev]` applied per block position.
///
/// Both weights are BF16 on device. `w1` is `[vocab, rank]` (an embedding
/// gather picks row `prev`); `w2` is `[vocab, rank]` (`nn.Linear(rank, vocab)`
/// weight), so `B = dense_gemv(input=w1[prev], weight=w2, n=vocab, k=rank)`.
/// The per-block bias scratch (`[vocab]` BF16) lives in [`DflashScratch`].
pub struct MarkovHead {
    pub w1: DenseWeight,
    pub w2: DenseWeight,
    pub rank: usize,
}

/// FP8 propose LM head and the matching final-norm weight.
///
/// The FP8 weight is multiplied by a power-of-two scale at load time, so only
/// this projection may consume activations produced by `input_norm`. Keeping
/// the two values in one type prevents that compensation from leaking into
/// the NVFP4/BF16 heads or the DFlash2 candidate selector.
#[derive(Debug, Clone, Copy)]
pub struct DflashFp8LmHead {
    pub weight: DevicePtr,
    pub input_norm: DenseWeight,
}

/// Per-drafter-layer Qwen3-style BF16 weights (default path).
#[allow(dead_code)]
pub struct DflashLayer {
    // Norms
    pub input_layernorm: DenseWeight,
    pub post_attention_layernorm: DenseWeight,
    // Attention (Qwen3: per-head Q/K RMSNorm)
    pub q_proj: DenseWeight,
    pub k_proj: DenseWeight,
    pub v_proj: DenseWeight,
    pub o_proj: DenseWeight,
    pub q_norm: DenseWeight,
    pub k_norm: DenseWeight,
    // MLP
    pub gate_proj: DenseWeight,
    pub up_proj: DenseWeight,
    pub down_proj: DenseWeight,
    /// DFlash 2 two-tap dynamic grouped conv wrapping the attention sublayer
    /// (`attention_conv.base_kernel` + `attention_conv.kernel_projection`),
    /// present iff the checkpoint is DFlash2 (`incoai/Qwen3.8-27B-DFlash2`).
    /// `None` for plain DFlash / DSpark drafters — `forward_block_layer`
    /// skips the conv entirely in that case.
    pub attention_conv: Option<crate::weight_loader::DflashConvWeights>,
    /// DFlash 2 two-tap dynamic grouped conv wrapping the MLP sublayer.
    /// `None` for plain DFlash / DSpark drafters.
    pub mlp_conv: Option<crate::weight_loader::DflashConvWeights>,
}

/// Per-drafter-layer NVFP4 weights (one of these is built per layer when
/// `DflashQuantization::Nvfp4` is selected). The 7 dense projections
/// (q/k/v/o + gate/up/down) are quantized via `quantize_to_nvfp4`; the
/// RMSNorm weights (`input_layernorm`, `post_attention_layernorm`,
/// `q_norm`, `k_norm`) stay BF16 because the rms_norm kernel reads them
/// as BF16 scaling factors.
#[allow(dead_code)]
pub struct DflashLayerNvfp4 {
    // Norms — BF16 (small, kernel-required dtype).
    pub input_layernorm: DenseWeight,
    pub post_attention_layernorm: DenseWeight,
    pub q_norm: DenseWeight,
    pub k_norm: DenseWeight,
    // Attention projections — NVFP4 packed.
    pub q_proj: QuantizedWeight,
    pub k_proj: QuantizedWeight,
    pub v_proj: QuantizedWeight,
    pub o_proj: QuantizedWeight,
    // MLP projections — NVFP4 packed.
    pub gate_proj: QuantizedWeight,
    pub up_proj: QuantizedWeight,
    pub down_proj: QuantizedWeight,
    /// Checkpoint-calibrated activation-scale receipts. Present only for the
    /// exact ModelOpt RTNcal DFlash2 artifact; runtime-quantized BF16 weights
    /// deliberately leave this `None` and cannot enter the W4A4 route.
    input_scales: Option<w4a4_admission::DflashNvfp4InputScales>,
    /// Transposed (`nvfp4_t` layout) MLP projections — populated only when
    /// `ATLAS_DFLASH_FFN_KGAMMA=1` was set at model build time and the
    /// quantization path took the NVFP4 branch. When `Some`, the per-layer
    /// forward routes the gate/up/down FFN GEMMs through
    /// `w4a16_gemm_n128_m16` (M_TILE=16) instead of the standard M_TILE=64
    /// `w4a16_gemm`. Each is a freshly allocated GPU buffer pair (packed
    /// weight + scales transposed via `QuantizedWeight::transpose_for_gemm`).
    /// `None` falls back to the original M_TILE=64 path.
    pub gate_proj_t: Option<QuantizedWeight>,
    pub up_proj_t: Option<QuantizedWeight>,
    pub down_proj_t: Option<QuantizedWeight>,
    /// DFlash 2 two-tap dynamic grouped conv weights (BF16 — the checkpoint
    /// ships them BF16 and they are small; they stay unquantized even in the
    /// NVFP4 drafter variant). `None` for plain DFlash / DSpark drafters.
    pub attention_conv: Option<crate::weight_loader::DflashConvWeights>,
    /// DFlash 2 MLP conv (see `attention_conv`).
    pub mlp_conv: Option<crate::weight_loader::DflashConvWeights>,
    /// Transposed (`nvfp4_t` layout) attention projections — populated only
    /// when `ATLAS_DFLASH_ATTN_KGAMMA=1` was set at model build time and the
    /// quantization path took the NVFP4 branch. When `Some`, the per-layer
    /// forward routes q/k/v/o GEMMs through `w4a16_gemm_n128_m16` (M_TILE=16)
    /// instead of the standard M_TILE=64 `w4a16_gemm`. Each is a freshly
    /// allocated GPU buffer pair (packed weight + scales transposed via
    /// `QuantizedWeight::transpose_for_gemm`). `None` falls back to the
    /// original M_TILE=64 path. Mirrors the FFN-kgamma pattern (cf.
    /// `gate_proj_t` / `up_proj_t` / `down_proj_t` above).
    pub q_proj_t: Option<QuantizedWeight>,
    pub k_proj_t: Option<QuantizedWeight>,
    pub v_proj_t: Option<QuantizedWeight>,
    pub o_proj_t: Option<QuantizedWeight>,
}

/// Per-drafter-layer online-quantized FP8 E4M3 weights (built when
/// `DflashQuantization::Fp8` is selected via `ATLAS_DFLASH_QUANTIZATION=fp8`
/// or `FLUX_DFLASH_QUANTIZATION=fp8`). Norms and optional DFlash2 convs stay
/// BF16; the 7 dense projections are quantized from BF16 with per-row scales.
#[allow(dead_code)]
pub struct DflashLayerFp8Weights {
    pub input_layernorm: DenseWeight,
    pub post_attention_layernorm: DenseWeight,
    pub q_norm: DenseWeight,
    pub k_norm: DenseWeight,
    pub q_proj: crate::weight_map::Fp8DenseWeight,
    pub k_proj: crate::weight_map::Fp8DenseWeight,
    pub v_proj: crate::weight_map::Fp8DenseWeight,
    pub o_proj: crate::weight_map::Fp8DenseWeight,
    pub gate_proj: crate::weight_map::Fp8DenseWeight,
    pub up_proj: crate::weight_map::Fp8DenseWeight,
    pub down_proj: crate::weight_map::Fp8DenseWeight,
    pub attention_conv: Option<crate::weight_loader::DflashConvWeights>,
    pub mlp_conv: Option<crate::weight_loader::DflashConvWeights>,
}

/// Per-layer weight bundle dispatched on at forward time. Holds BF16
/// [`DflashLayer`], online-quantized FP8 [`DflashLayerFp8Weights`], or NVFP4
/// [`DflashLayerNvfp4`]; `forward_block_layer` dispatches every projection
/// through the matching kernel.
#[allow(clippy::large_enum_variant)]
pub enum DflashLayerQuantWeights {
    Bf16(DflashLayer),
    Fp8(DflashLayerFp8Weights),
    Nvfp4(DflashLayerNvfp4),
}

/// Per-sequence DFlash drafter state for the active BF16 circular-context path.
///
/// There is deliberately no paged draft-KV block table here. The old state
/// carried a table, sequence length, and prefill flag for an FP8 paged design,
/// but no production forward ever read them. Context projection plus every
/// layer's K/V are instead retained in the circular buffers below.
pub struct DflashProposerState {
    /// Drafts produced in the last `propose()` call. `after_verify` consults
    /// this to know how many KV positions to roll back when the accept
    /// prefix is shorter than γ.
    pub last_num_drafted: usize,
    /// Immutable limits from the most recent outer `propose` call.
    pub draft_budget: Option<draft_budget::DflashDraftBudget>,
    /// Multi-token accumulator for captured target hidden states. Physical
    /// layout: `[ctx_capacity, captures * target_hidden]` BF16 packed. The scheduler appends
    /// the model's `dflash_hidden_save` (latest decoded position's 5 hiddens)
    /// into physical slot `ctx_len % ctx_capacity` after each successful verify. `propose()` reads
    /// the latest resident window and projects it through `fc` at forward
    /// time. Absolute positions wrap only at the physical-storage boundary.
    pub ctx_hidden_acc: DevicePtr,
    /// Absolute one-past-last captured target position. It never wraps.
    pub ctx_len: usize,
    /// Number of latest absolute positions currently resident in the ring.
    pub ctx_resident_len: usize,
    /// Lowest absolute position whose target hidden state is actually populated
    /// in `ctx_hidden_acc`. Zero on uncached prefills; advanced to `M` on a
    /// Marconi prefix-cache hit covering `[0..M)` so cross-attention never
    /// reads uninitialized slots `< M`.
    pub ctx_valid_from_abs: usize,
    /// Physical slot count allocated for `ctx_hidden_acc`.
    pub ctx_capacity: usize,
    /// Exact pool/allocation key for `ctx_hidden_acc`.
    pub ctx_allocation_bytes: usize,
    /// Absolute context limit from the model build.
    pub max_ctx_len: usize,
    /// Width (bytes) of one `ctx_hidden_acc` slot — `5 * target_hidden * bf16`.
    /// Stored to avoid re-deriving on every append.
    pub ctx_slot_bytes: usize,
    /// Index into the multi-token `dflash_hidden_save` buffer of the
    /// last accepted token's hidden state. Set by `after_verify` to
    /// `num_accepted` so `propose_drafts` appends the correct capture.
    pub last_capture_idx: usize,
    /// Number of drafts accepted in the previous verify step.
    /// Used by `propose_drafts` to know how many verify positions to
    /// append to `ctx_hidden_acc` (positions 1..=last_num_accepted for
    /// accepted drafts; position 0 when zero drafts were accepted).
    pub last_num_accepted: usize,
    /// FIX 1 (ATLAS_DFLASH_TREE_COMMIT): the accepted-path COMPACT indices
    /// from the last tree-fork verify (e.g. `[1, 2, 3, 7]`). Empty on every
    /// flat-chain / non-tree step. When non-empty, `propose_drafts` reads the
    /// captured ctx hiddens from these sparse `dflash_hidden_save` rows
    /// (verify slot 0 = last_token, slot `c` = compact index `c`) instead of
    /// the contiguous `0..last_num_accepted+1` — because a fork accept's
    /// hiddens are scattered, and reading `0..N+1` would pick up rejected
    /// sibling rows. Stays a strict superset-safe override: when the path is
    /// the contiguous `[1..N]` it produces identical reads.
    pub last_accepted_compact: Vec<usize>,
    /// Host copy of the sequence's committed tokens, refreshed by the
    /// caller each propose when ATLAS_DFLASH_PLD=1 (prompt-lookup drafts).
    pub pld_tokens: Vec<u32>,
    /// Whether `propose_drafts` has been called at least once. Used to
    /// skip the post-prefill append on the first call because
    /// `dflash_hidden_save` hasn't been populated yet.
    pub first_propose_done: bool,
    /// Page-locked host mirror of the per-propose position-id buffer
    /// (`[ctx_pos_0..ctx_pos_{eff_ctx-1}, seq_pos..seq_pos+γ-1]`). The
    /// forward writes positions here host-side and enqueues a stream-ordered
    /// async H2D on the propose stream, avoiding the per-propose
    /// `cuStreamSynchronize(default_stream)` drain that `copy_h2d` performs
    /// (measured ~8.7 ms of hidden serialization per cycle). Sized for
    /// `(ctx_window + γ + 1) * 4` bytes.
    pub pos_pinned: PinnedHostBuffer,
    /// Separate page-locked mirror for `[context zeros, anchor, noise]` token
    /// IDs. It cannot alias `pos_pinned`: both asynchronous H2D copies may be
    /// pending concurrently on the proposal stream.
    pub token_ids_pinned: PinnedHostBuffer,
    /// Page-locked D2H staging for final draft IDs. Shared by synchronous and
    /// async collection only after the owning proposal stream has completed.
    pub draft_output_pinned: PinnedHostBuffer,
    /// One-shot sampled-selector inputs installed by the serving layer. The
    /// outer `propose` call moves this into `active_sampling_request` and
    /// clears it on every success/error exit.
    pub next_sampling_request: Option<crate::dflash_sampling::DflashDraftSamplingRequest>,
    /// Sampling inputs owned by the currently executing neural proposal.
    pub active_sampling_request: Option<crate::dflash_sampling::DflashDraftSamplingRequest>,
    /// Pinned H2D staging for `[gamma]` f32 uniforms. None for released greedy.
    pub selector_uniforms_pinned: Option<PinnedHostBuffer>,
    /// Pinned H2D staging for the fixed-layout sparse q modifier table. None
    /// for released greedy.
    pub selector_q_modifiers_pinned: Option<PinnedHostBuffer>,
    /// Pinned D2H staging for `[gamma, 16]` u32 candidate IDs. None for greedy.
    pub selector_candidates_pinned: Option<PinnedHostBuffer>,
    /// Pinned D2H staging for `[gamma, 16]` f32 q. None for greedy.
    pub selector_q_pinned: Option<PinnedHostBuffer>,
    /// Drain-once sampled proposal produced by the most recent successful
    /// neural proposal. Cleared with every other proposal output.
    pub pending_sampled_proposal: Option<crate::dflash_sampling::DflashSampledProposalFrame>,

    // ── Adaptive retrieval gate (ATLAS_DFLASH_SAM auto-disable) ──
    /// Whether the PREVIOUS propose pre-empted the neural drafter with a
    /// retrieval (SAM) draft. Set when the retrieval path fires; read on the
    /// next propose to attribute `last_num_accepted` to retrieval vs drafter.
    pub retr_used_last: bool,
    /// Consecutive retrieval steps whose accept came back poor. When it
    /// crosses the limit, retrieval enters a cooldown — this auto-disables SAM
    /// on content where its strong suffix matches mis-predict (e.g. counting:
    /// digit runs match but the next number is always new), while leaving it
    /// fully active on reuse-heavy code editing (where it keeps accepting).
    pub retr_misfire_streak: u32,
    /// Remaining propose steps to SKIP retrieval (cooldown). Decremented each
    /// step; retrieval is suppressed while > 0, then retried.
    pub retr_cooldown: u32,
    /// Per-request lookup receipt counters (host-only, observational).
    pub lookup_stats: lookup_stats::LookupStats,

    // ── Persistent context cache (eliminates O(seq_len) recompute) ──
    /// Cached fc_proj outputs for context tokens. Circular buffer of
    /// `[ctx_window, hidden_size]` BF16. Slot `p` holds absolute position
    /// `p % ctx_window`.
    pub ctx_fc_cache: DevicePtr,
    /// Cached K projections per layer. One buffer per layer,
    /// `[ctx_window, num_kv_heads * head_dim]` BF16.
    pub ctx_k_cache: Vec<DevicePtr>,
    /// Cached V projections per layer.
    pub ctx_v_cache: Vec<DevicePtr>,
    /// Per-layer cache range: first valid absolute position in the circular
    /// K cache for each layer. All layers cache the same positions in
    /// practice, but they must be tracked independently so layer N knows
    /// which positions layer N has already cached.
    pub cache_k_start: Vec<usize>,
    /// Per-layer K cache range end (one past last valid).
    pub cache_k_end: Vec<usize>,
    /// Per-layer V cache range start.
    pub cache_v_start: Vec<usize>,
    /// Per-layer V cache range end.
    pub cache_v_end: Vec<usize>,
    /// fc_proj cache range start (shared across layers).
    pub cache_fc_start: usize,
    /// fc_proj cache range end.
    pub cache_fc_end: usize,
    /// Representation currently stored in `ctx_k_cache` (`PreRope` for eager
    /// `forward_block_layer`, `PostRope` for Stage-1 / batched DFlash2).
    /// `None` when empty / freshly allocated.
    ctx_kv_repr: Option<noise_pass::CtxKvCacheRepr>,
    /// DDTree M4B: optional tree payload built by the drafter's most recent
    /// `propose_drafts()` when `--dflash-method=ddtree` is active. The
    /// scheduler drains this via `take_pending_tree_payload()` after each
    /// propose call and stashes it on `ActiveSeq.pending_tree_payload`
    /// (M3 plumbing). `None` for flat DFlash → preserves legacy behavior.
    pub pending_tree_payload: Option<ddtree::TreePayload>,
    /// Ring buffer of accept counts from the last `ACCEPT_HISTORY` verifies.
    /// `accept_history[accept_history_pos]` is the next slot to overwrite.
    /// Used by `ATLAS_DFLASH_ADAPTIVE_GAMMA=1` in `forward_block` to shrink
    /// the drafter noise block when recent accept is low.
    pub accept_history: [u8; 8],
    /// Next write index into `accept_history` (mod 8).
    pub accept_history_pos: usize,
    /// Number of verifies recorded so far (capped at `accept_history.len()`).
    /// Below 4 we don't adapt — let the drafter run full γ until we have a
    /// stable signal.
    pub accept_history_count: usize,
    /// Monotonic count of adaptive-engaged propose() calls. Unlike
    /// `accept_history_count` (saturates at 8), this counter never wraps
    /// during a sequence so it can drive periodic γ_max reprobes via
    /// `ATLAS_DFLASH_ADAPTIVE_PROBE_INTERVAL`. Without periodic reprobe,
    /// adaptive truncate is self-limiting: once γ_eff truncates to K=2/3/4,
    /// the history fills with the small accept counts from those truncated
    /// steps and the cutoff stays low forever even when content becomes
    /// more predictable (counting, lists, structured output). Reprobing
    /// re-measures the true accept ceiling.
    pub propose_steps: usize,
    /// Throughput router state. Timing begins immediately before neural
    /// proposal and ends in `after_verify`, covering the cost that matters to
    /// server decode throughput rather than drafter acceptance in isolation.
    throughput_router: throughput_router::ThroughputRouter,
    /// Climb/drop adaptive depth controller (ATLAS_DFLASH_TPS_ROUTER_MODE=
    /// climbdrop, ported from llama.cpp PR #27210). Fed by `after_verify`
    /// with (last_num_drafted, num_accepted) and queried by `forward_block`
    /// for the draft cutoff. Mutually exclusive with the EWMA router.
    climbdrop_router: throughput_router::ClimbDropRouter,
    throughput_cycle_started: Option<Instant>,
    throughput_last_width: usize,
    /// ATLAS_DFLASH_ACCEPT_FALLBACK: number of remaining steps this sequence
    /// must spend in plain single-token decode (speculation suppressed)
    /// before the next re-probe. When > 0, `propose_drafts` returns an empty
    /// draft vector so the scheduler routes the sequence through the bootstrap
    /// plain-decode path. Decremented once per suppressed propose call. When
    /// it reaches 0, the next propose runs a full γ probe to re-measure
    /// acceptance. 0 = not suppressed (normal full-γ speculation).
    pub fallback_suppressed_remaining: usize,

    // ── ATLAS_DFLASH_RECYCLE=1: discarded-draft-tail recycling (default off) ──
    /// The tail of the PREVIOUS step's drafts that verify discarded after the
    /// first content miss — `drafts[num_accepted+1 .. γ_eff]`. The drafter
    /// often gets the STRUCTURAL continuation right even when the corrected
    /// token (committed by the target at the mismatch) differs, so re-offering
    /// this tail re-accepts the free structural part. Populated by
    /// `dflash_stash_recycle` from `verify_dflash_step` after `num_accepted`
    /// is known; consumed by `propose_drafts` on the next call. Empty ⇒ no
    /// tail available. LOSSLESS: these tokens are only PROPOSED — verify still
    /// commits target-greedy, so a wrong recycle costs one rejected
    /// speculation, never output.
    pub recycle_tail: Vec<u32>,
    /// The corrected token committed by the target at the mismatch position
    /// (= `verified[num_accepted]` = the bonus = next step's `last_token`).
    /// `propose_drafts` offers `recycle_tail` only when the new `last_token`
    /// equals this key — the tail was the drafter's continuation conditioned
    /// on this exact corrected token, so re-offering it is principled.
    pub recycle_key: u32,
    /// Whether `recycle_tail`/`recycle_key` hold a valid stash (distinguishes
    /// "empty tail because full-accept" from "no stash yet"; the offer path
    /// needs both a matching key AND a non-empty tail anyway, so this is
    /// belt-and-suspenders for the key==0 corner case).
    pub recycle_valid: bool,
    /// Whether the PREVIOUS propose returned a recycled tail. Prevents the
    /// self-sustaining low-accept trap: a recycled tail that re-accepts poorly
    /// keeps `last_num_accepted` low, which would re-open the recycle gate
    /// forever and STARVE the neural drafter (measured: counting collapsed
    /// 82→~9 tok/s because one early recycle offer trapped the sequence). With
    /// this flag, recycle never fires two steps in a row — after any offer the
    /// next step runs the real drafter, which re-establishes the true accept
    /// signal and produces a fresh tail. Recycle thus fires at most every other
    /// step, capping its worst-case throughput cost at one wasted speculation
    /// per two steps while still recovering the discarded tail on genuinely
    /// weak content.
    pub recycle_last_offered: bool,

    // ── ATLAS_DFLASH_ECHO=1: echo-drafting / Jacobi salvage (default off) ──
    /// The TARGET'S OWN verify argmaxes downstream of the bonus —
    /// `verified[num_accepted+1 ..]` from the previous flat-chain verify.
    /// Unlike `recycle_tail` (drafter-authored), these are target-authored:
    /// conditioned on a near-miss prefix, they are usually still right after
    /// the one-token bonus substitution. Populated by `dflash_stash_echo`
    /// (verify_dflash_step.rs, flat path only, gated on
    /// `num_accepted >= min_accept` + `tail >= min_tail`); consumed by
    /// `propose_drafts` on the next call, skipping the drafter forward
    /// entirely (the 25-50ms propose slice). LOSSLESS: proposal-only.
    pub echo_tail: Vec<u32>,
    /// The bonus token committed at the rejection (= next step's
    /// `last_token`). The echo tail is the target's continuation after this
    /// exact token's context, so it is offered only when the key matches.
    pub echo_key: u32,
    /// Whether `echo_tail`/`echo_key` hold a valid stash (belt-and-suspenders
    /// for the key==0 corner, mirroring `recycle_valid`).
    pub echo_valid: bool,
    /// Consecutive echo offers. Capped at `ATLAS_DFLASH_ECHO_MAX_STREAK`
    /// (default 2) so an echo-drafted step that itself rejects can not keep
    /// salvaging its own wreckage forever — after the cap the real drafter
    /// runs and re-establishes the true accept signal. Reset to 0 on any
    /// propose where echo does not fire.
    pub echo_streak: u32,
    /// Whether the PREVIOUS propose returned an echo tail. Read (and
    /// cleared) by the next propose to attribute `last_num_accepted` to the
    /// echo draft — the salvage-accept telemetry line.
    pub echo_offered_last: bool,
    /// ATLAS_DFLASH_ASYNC: this sequence's `pending_drafts` currently holds
    /// a PLACEHOLDER chain from an async (second-stream) propose launch; the
    /// real drafts are collected via `collect_async_drafts` at the top of
    /// the next scheduler step. Cleared on collect / resolve.
    pub async_placeholder: bool,
    /// Sequence-scoped CRN Gumbel draft spec (`Some(Some(spec))` = coupled Gumbel,
    /// `Some(None)` = explicit greedy/non-Gumbel, `None` = fallback to global ring).
    pending_gumbel_spec:
        std::sync::Mutex<Option<Option<spark_runtime::sampler::gumbel::GumbelDraftSpec>>>,
    /// Per-token draft probabilities `q_draft[0..drafts.len()]` populated by
    /// `dflash2_selector_walk_gumbel_batched_k32` on the most recent proposal.
    /// Cleared at the start of every `propose` / `propose_batch` and on `free_state`
    /// so stale probabilities can never leak across steps.
    pub(super) last_draft_probs: Vec<f32>,
    /// Number of C=1 draft probabilities written into `scratch.batch_out_probs`
    /// during the active `run_noise_pass`, pending D2H readback into `last_draft_probs`.
    pub(super) pending_draft_probs_gamma: usize,
    /// Pending model slot index staged before probabilistic rejection proposal
    /// so the selector writes `(q_cand_ids, q_probs)` at row base `slot_idx * 8`.
    pending_rejection_slot: std::sync::Mutex<Option<usize>>,
    /// Metadata `(slot_idx, step0_abs_pos, valid_len)` for the GPU draft
    /// distribution written into the shared rejection `(q_cand_ids, q_probs)`
    /// scratch by the most recent probabilistic proposal.
    last_rejection_q: std::sync::Mutex<Option<(usize, u64, usize)>>,
    /// Address identity of the owning [`BlockDiffusionDraftHead`] pool (`0`
    /// when constructed directly in unit tests without a head).
    pool_owner_head: usize,
    /// Exact byte size allocated for `ctx_fc_cache`, used for strict geometry
    /// matching on pool reuse.
    ctx_fc_bytes: usize,
    /// Exact byte size allocated for each `ctx_k_cache[l]` / `ctx_v_cache[l]`
    /// buffer, used for strict geometry matching on pool reuse.
    ctx_kv_bytes: usize,
    /// Whether the bound target model is MoE (`Some(true)`), dense (`Some(false)`),
    /// or unspecified (`None`, e.g. unit-test fixtures without `sync_target_config`).
    pub(super) target_is_moe: Option<bool>,
}

fn checked_ctx_capture_debt(
    cursor: usize,
    position: usize,
    last_num_accepted: usize,
) -> Result<usize> {
    anyhow::ensure!(
        position >= cursor,
        "DFlash context cursor {cursor} is ahead of target position {position}",
    );
    let pending_rows = position - cursor;
    if pending_rows == 0 {
        return Ok(0);
    }
    let recorded_rows = last_num_accepted
        .checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("DFlash accepted-row count overflowed"))?;
    anyhow::ensure!(
        pending_rows == recorded_rows,
        "DFlash context debt disagrees with captured verify rows: cursor={cursor} position={position} pending={pending_rows} recorded={recorded_rows}",
    );
    Ok(pending_rows)
}

/// Coalesce consecutive `(source_row, dst_slot)` capture copies into contiguous
/// `(src_start, dst_start, run_len)` D2D runs so flat C=1 appends issue 1 D2D
/// copy (or 2 on a ring wrap) instead of `pending_rows` per-row copies.
fn coalesce_ctx_capture_runs(
    source_rows: &[usize],
    write: &ring_window::RingSpanPlan,
) -> Result<Vec<(usize, usize, usize)>> {
    let mut runs: Vec<(usize, usize, usize)> = Vec::with_capacity(2.min(source_rows.len()));
    for (linear_row, &source_row) in source_rows.iter().enumerate() {
        let dst_slot = write.physical_slot_for(linear_row)?;
        if let Some(last) = runs.last_mut()
            && source_row == last.0 + last.2
            && dst_slot == last.1 + last.2
        {
            last.2 += 1;
        } else {
            runs.push((source_row, dst_slot, 1));
        }
    }
    Ok(runs)
}

impl DflashProposerState {
    pub(crate) fn ctx_ring_state(&self) -> Result<ring_window::RingState> {
        Ok(ring_window::RingState::from_lengths_with_valid_from(
            self.max_ctx_len,
            self.ctx_capacity,
            self.ctx_len,
            self.ctx_resident_len,
            self.ctx_valid_from_abs,
        )?)
    }

    pub(crate) fn apply_ctx_ring_state(&mut self, state: ring_window::RingState) -> Result<()> {
        anyhow::ensure!(
            state.max_context == self.max_ctx_len && state.capacity == self.ctx_capacity,
            "DFlash ring state geometry changed while applying an update"
        );
        self.ctx_len = state.absolute_len;
        self.ctx_resident_len = state.resident_len;
        self.ctx_valid_from_abs = state.ctx_valid_from_abs;
        Ok(())
    }

    /// Advance `ctx_len` past a Marconi prefix-cache hit `[0..skip_to)` without
    /// claiming those skipped positions are resident in `ctx_hidden_acc`.
    /// Invalidates all derived FC/K/V context caches whenever the cursor moves.
    pub(crate) fn advance_marconi_skip(&mut self, skip_to: usize) -> Result<()> {
        if skip_to <= self.ctx_len {
            return Ok(());
        }
        let next = self.ctx_ring_state()?.advance_skipped_prefix(skip_to)?;
        self.apply_ctx_ring_state(next)?;
        self.cache_fc_start = 0;
        self.cache_fc_end = 0;
        self.cache_k_start.fill(0);
        self.cache_k_end.fill(0);
        self.cache_v_start.fill(0);
        self.cache_v_end.fill(0);
        self.ctx_kv_repr = None;
        Ok(())
    }

    /// Append verifier captures that have not yet reached the per-sequence
    /// context ring.
    ///
    /// The ring cursor is authoritative.  A successful verify advances
    /// `seq.seq_len` by the root plus the accepted drafts, while the matching
    /// capture rows remain in `dflash_hidden_save` until the next proposal.
    /// Most cycles retire that debt at proposal entry.  A policy/structural
    /// boundary can deliberately skip the proposal, however, so the scheduler
    /// must retire the same rows before an ordinary bootstrap decode overwrites
    /// capture row zero.
    ///
    /// `position` is the target sequence's current absolute one-past-last-fed
    /// position.  A zero debt is an idempotent no-op (normal after prefill).
    /// Any debt other than exactly `last_num_accepted + 1` is rejected: merely
    /// assigning the cursor would hide a missing/overwritten capture and poison
    /// all later drafter conditioning.
    pub(crate) fn append_pending_ctx_captures(
        &mut self,
        gpu: &dyn GpuBackend,
        captured: DevicePtr,
        position: usize,
        physical_verify_k: usize,
        stream: u64,
    ) -> Result<usize> {
        let ring = self.ctx_ring_state()?;
        let pending_rows =
            checked_ctx_capture_debt(ring.absolute_len, position, self.last_num_accepted)?;
        if pending_rows == 0 {
            return Ok(0);
        }

        // Row zero is the verify root.  Flat accepts occupy contiguous rows;
        // a tree-fork accept stores the scheduler-provided kernel-frame rows.
        let source_rows: Vec<usize> = if self.last_accepted_compact.is_empty() {
            (0..pending_rows).collect()
        } else {
            let mut rows = Vec::with_capacity(pending_rows);
            rows.push(0);
            rows.extend_from_slice(&self.last_accepted_compact);
            rows
        };
        anyhow::ensure!(
            source_rows.len() == pending_rows,
            "DFlash sparse accepted-row map has {} rows; expected {pending_rows}",
            source_rows.len(),
        );
        anyhow::ensure!(
            pending_rows <= physical_verify_k
                && source_rows
                    .iter()
                    .all(|source_row| *source_row < physical_verify_k),
            "DFlash accepted-row source exceeds physical verify K={physical_verify_k}: {source_rows:?}",
        );

        let append = ring.plan_append_at(ring.absolute_len, pending_rows)?;
        let runs = coalesce_ctx_capture_runs(&source_rows, &append.write)?;
        for (src_start, dst_start, run_len) in runs {
            let src = captured.offset(src_start * self.ctx_slot_bytes);
            let dst = self.ctx_hidden_acc.offset(dst_start * self.ctx_slot_bytes);
            gpu.copy_d2d_async(src, dst, run_len * self.ctx_slot_bytes, stream)?;
        }
        self.apply_ctx_ring_state(append.next)?;
        Ok(pending_rows)
    }

    /// Commit one pre-planned serial-decode target-hidden row (`captured_row0`)
    /// into the bounded context ring and clear any stale verify-debt metadata.
    ///
    /// Returns `Ok(false)` when `seq_len == 0` or `self.ctx_len == seq_len` so
    /// duplicate capture hooks on the same decoded token are idempotent.
    pub(crate) fn commit_serial_decode_capture(
        &mut self,
        gpu: &dyn GpuBackend,
        captured_row0: DevicePtr,
        seq_len: usize,
        append: ring_window::AppendPlan,
        stream: u64,
    ) -> Result<bool> {
        if seq_len == 0 || self.ctx_len == seq_len {
            return Ok(false);
        }
        let slot_bytes = self.ctx_slot_bytes;
        let dst_slot = append.write.physical_slot_for(0)?;
        let dst = self.ctx_hidden_acc.offset(dst_slot * slot_bytes);
        gpu.copy_d2d_async(captured_row0, dst, slot_bytes, stream)?;
        self.apply_ctx_ring_state(append.next)?;
        self.last_capture_idx = 0;
        self.last_num_accepted = 0;
        self.last_accepted_compact.clear();
        Ok(true)
    }

    /// Stamp this sequence's upcoming CRN Gumbel draft spec (`None` = explicit
    /// greedy / uncoupled, suppressing any global-ring fallback).
    pub fn set_gumbel_spec(&self, spec: Option<spark_runtime::sampler::gumbel::GumbelDraftSpec>) {
        *self
            .pending_gumbel_spec
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(spec);
    }

    fn take_gumbel_spec_with(
        &self,
        position: usize,
        enabled: bool,
    ) -> Option<spark_runtime::sampler::gumbel::GumbelDraftSpec> {
        let stamped = self
            .pending_gumbel_spec
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        if !enabled {
            return None;
        }
        if let Some(published) = stamped {
            return published.filter(|s| s.is_active());
        }
        spark_runtime::sampler::gumbel::lookup_gumbel_draft_spec(position)
    }

    /// Consume this sequence's pending CRN Gumbel/rejection draft spec for `position`.
    ///
    /// Returns `None` when neither Gumbel nor rejection verification is enabled
    /// (still draining any stamped slot), when `Some(None)` was explicitly
    /// stamped for a greedy sequence (without consulting the global ring), or
    /// when no active spec is registered.
    pub fn take_gumbel_spec(
        &self,
        position: usize,
    ) -> Option<spark_runtime::sampler::gumbel::GumbelDraftSpec> {
        self.take_gumbel_spec_with(
            position,
            spark_runtime::sampler::gumbel::gumbel_sampling_enabled()
                || spark_runtime::sampler::gumbel::rejection_sampling_enabled(),
        )
    }

    /// Stage the model slot index to receive GPU draft shortlist rows
    /// (`slot_idx * 8`) on the next probabilistic rejection proposal.
    pub fn set_pending_rejection_slot(&self, slot_idx: Option<usize>) {
        *self
            .pending_rejection_slot
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = slot_idx;
    }

    pub(super) fn take_pending_rejection_slot(&self) -> Option<usize> {
        self.pending_rejection_slot
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
    }

    /// Return `(slot_idx, step0_abs_pos, valid_len)` for the GPU draft
    /// distribution written by the most recent probabilistic proposal, if valid.
    #[must_use]
    pub fn last_rejection_q(&self) -> Option<(usize, u64, usize)> {
        *self
            .last_rejection_q
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub(super) fn set_last_rejection_q(&self, entry: Option<(usize, u64, usize)>) {
        *self
            .last_rejection_q
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) =
            entry.filter(|(_, _, len)| *len > 0);
    }

    /// Invalidate any recorded GPU draft distribution from prior proposals.
    pub fn invalidate_rejection_q(&self) {
        *self
            .last_rejection_q
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
    }

    /// Clear both the staged rejection slot and any cached GPU draft shortlist
    /// metadata (e.g. when a sequence compacts from `old_slot` to `new_slot`).
    pub fn invalidate_rejection_slot_state(&self) {
        self.set_pending_rejection_slot(None);
        self.invalidate_rejection_q();
    }

    #[cfg(test)]
    pub(crate) fn pending_rejection_slot_for_test(&self) -> Option<usize> {
        *self
            .pending_rejection_slot
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    #[cfg(test)]
    pub(crate) fn set_last_rejection_q_for_test(&self, entry: Option<(usize, u64, usize)>) {
        self.set_last_rejection_q(entry);
    }

    #[cfg(test)]
    pub(crate) fn new_for_compaction_test(gpu: &dyn GpuBackend) -> Self {
        let pos_pinned = gpu.alloc_host_pinned(64).expect("alloc pos_pinned");
        let token_ids_pinned = gpu.alloc_host_pinned(64).expect("alloc token_ids_pinned");
        let draft_output_pinned = gpu
            .alloc_host_pinned(64)
            .expect("alloc draft_output_pinned");
        Self {
            last_num_drafted: 0,
            draft_budget: None,
            ctx_hidden_acc: DevicePtr::NULL,
            ctx_len: 0,
            ctx_resident_len: 0,
            ctx_valid_from_abs: 0,
            ctx_capacity: 0,
            ctx_allocation_bytes: 0,
            max_ctx_len: 0,
            ctx_slot_bytes: 0,
            last_capture_idx: 0,
            last_num_accepted: 0,
            last_accepted_compact: Vec::new(),
            pld_tokens: Vec::new(),
            first_propose_done: false,
            pos_pinned,
            token_ids_pinned,
            draft_output_pinned,
            next_sampling_request: None,
            active_sampling_request: None,
            selector_uniforms_pinned: None,
            selector_q_modifiers_pinned: None,
            selector_candidates_pinned: None,
            selector_q_pinned: None,
            pending_sampled_proposal: None,
            retr_used_last: false,
            retr_misfire_streak: 0,
            retr_cooldown: 0,
            lookup_stats: lookup_stats::LookupStats::default(),
            ctx_fc_cache: DevicePtr::NULL,
            ctx_k_cache: Vec::new(),
            ctx_v_cache: Vec::new(),
            cache_k_start: Vec::new(),
            cache_k_end: Vec::new(),
            cache_v_start: Vec::new(),
            cache_v_end: Vec::new(),
            cache_fc_start: 0,
            cache_fc_end: 0,
            ctx_kv_repr: None,
            pending_tree_payload: None,
            accept_history: [0u8; 8],
            accept_history_pos: 0,
            accept_history_count: 0,
            propose_steps: 0,
            throughput_router: throughput_router::ThroughputRouter::default(),
            climbdrop_router: throughput_router::ClimbDropRouter::default(),
            throughput_cycle_started: None,
            throughput_last_width: 0,
            fallback_suppressed_remaining: 0,
            recycle_tail: Vec::new(),
            recycle_key: 0,
            recycle_valid: false,
            recycle_last_offered: false,
            echo_tail: Vec::new(),
            echo_key: 0,
            echo_valid: false,
            echo_streak: 0,
            echo_offered_last: false,
            async_placeholder: false,
            pending_gumbel_spec: std::sync::Mutex::new(None),
            last_draft_probs: Vec::new(),
            pending_draft_probs_gamma: 0,
            pending_rejection_slot: std::sync::Mutex::new(None),
            last_rejection_q: std::sync::Mutex::new(None),
            pool_owner_head: 0,
            ctx_fc_bytes: 0,
            ctx_kv_bytes: 0,
            target_is_moe: None,
        }
    }

    /// Whether the bound target model is MoE (`Some(true)`), dense (`Some(false)`),
    /// or unspecified (`None`).
    #[must_use]
    pub fn target_is_moe(&self) -> Option<bool> {
        self.target_is_moe
    }

    /// Clamp the recorded valid length of `last_rejection_q` to at most
    /// `max_len` (invalidating it if `max_len == 0`).
    pub fn truncate_rejection_q(&self, max_len: usize) {
        let mut guard = self
            .last_rejection_q
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some((_, _, valid_len)) = guard.as_mut() {
            *valid_len = (*valid_len).min(max_len);
            if *valid_len == 0 {
                *guard = None;
            }
        }
    }

    /// Selected draft token probabilities from the most recent DFlash2 K<=32
    /// selector proposal, or an empty slice when unavailable or cleared.
    #[must_use]
    pub fn last_draft_probs(&self) -> &[f32] {
        &self.last_draft_probs
    }

    fn buffer_geometry(&self) -> DflashStateBufferGeometry {
        let probabilistic = self.selector_uniforms_pinned.is_some()
            && self.selector_q_modifiers_pinned.is_some()
            && self.selector_candidates_pinned.is_some()
            && self.selector_q_pinned.is_some();
        DflashStateBufferGeometry {
            ctx_allocation_bytes: self.ctx_allocation_bytes,
            ctx_capacity: self.ctx_capacity,
            ctx_slot_bytes: self.ctx_slot_bytes,
            max_ctx_len: self.max_ctx_len,
            ctx_fc_bytes: self.ctx_fc_bytes,
            ctx_kv_bytes: self.ctx_kv_bytes,
            num_layers: self.ctx_k_cache.len(),
            pos_pinned_bytes: self.pos_pinned.len(),
            token_ids_pinned_bytes: self.token_ids_pinned.len(),
            draft_output_pinned_bytes: self.draft_output_pinned.len(),
            probabilistic,
            selector_uniforms_bytes: self
                .selector_uniforms_pinned
                .as_ref()
                .map_or(0, PinnedHostBuffer::len),
            selector_q_modifiers_bytes: self
                .selector_q_modifiers_pinned
                .as_ref()
                .map_or(0, PinnedHostBuffer::len),
            selector_row_bytes: self
                .selector_candidates_pinned
                .as_ref()
                .map_or(0, PinnedHostBuffer::len),
        }
    }

    /// Exhaustively reset every non-buffer field and zero every host/pinned
    /// staging buffer so a retired or reused [`DflashProposerState`] cannot
    /// leak tokens, positions, sampling specs, caches, or router history across
    /// requests.
    ///
    /// Uses an exhaustive struct destructure (no `..`) so adding a field to
    /// [`DflashProposerState`] fails compilation until its reset policy is
    /// explicitly handled here.
    fn reset_all_transient_state(
        &mut self,
        initial_ring: Option<ring_window::RingState>,
        num_layers: usize,
    ) {
        const HOST_VEC_RETAIN_CAP: usize = 4096;
        let Self {
            last_num_drafted,
            draft_budget,
            ctx_hidden_acc: _,
            ctx_len,
            ctx_resident_len,
            ctx_valid_from_abs,
            ctx_capacity: _,
            ctx_allocation_bytes: _,
            max_ctx_len: _,
            ctx_slot_bytes: _,
            last_capture_idx,
            last_num_accepted,
            last_accepted_compact,
            pld_tokens,
            first_propose_done,
            pos_pinned,
            token_ids_pinned,
            draft_output_pinned,
            next_sampling_request,
            active_sampling_request,
            selector_uniforms_pinned,
            selector_q_modifiers_pinned,
            selector_candidates_pinned,
            selector_q_pinned,
            pending_sampled_proposal,
            retr_used_last,
            retr_misfire_streak,
            retr_cooldown,
            lookup_stats,
            ctx_fc_cache: _,
            ctx_k_cache: _,
            ctx_v_cache: _,
            cache_k_start,
            cache_k_end,
            cache_v_start,
            cache_v_end,
            cache_fc_start,
            cache_fc_end,
            ctx_kv_repr,
            pending_tree_payload,
            accept_history,
            accept_history_pos,
            accept_history_count,
            propose_steps,
            throughput_router,
            climbdrop_router,
            throughput_cycle_started,
            throughput_last_width,
            fallback_suppressed_remaining,
            recycle_tail,
            recycle_key,
            recycle_valid,
            recycle_last_offered,
            echo_tail,
            echo_key,
            echo_valid,
            echo_streak,
            echo_offered_last,
            async_placeholder,
            pending_gumbel_spec,
            last_draft_probs,
            pending_draft_probs_gamma,
            pending_rejection_slot,
            last_rejection_q,
            pool_owner_head: _,
            ctx_fc_bytes: _,
            ctx_kv_bytes: _,
            target_is_moe: _,
        } = self;

        *last_num_drafted = 0;
        *draft_budget = None;
        if let Some(ring) = initial_ring {
            *ctx_len = ring.absolute_len;
            *ctx_resident_len = ring.resident_len;
            *ctx_valid_from_abs = ring.ctx_valid_from_abs;
        } else {
            *ctx_len = 0;
            *ctx_resident_len = 0;
            *ctx_valid_from_abs = 0;
        }
        *last_capture_idx = 0;
        *last_num_accepted = 0;
        last_accepted_compact.clear();
        if last_accepted_compact.capacity() > HOST_VEC_RETAIN_CAP {
            last_accepted_compact.shrink_to(0);
        }
        pld_tokens.fill(0);
        pld_tokens.clear();
        if pld_tokens.capacity() > HOST_VEC_RETAIN_CAP {
            pld_tokens.shrink_to(0);
        }
        *first_propose_done = false;

        pos_pinned.as_mut_slice().fill(0);
        token_ids_pinned.as_mut_slice().fill(0);
        draft_output_pinned.as_mut_slice().fill(0);
        if let Some(buf) = selector_uniforms_pinned.as_mut() {
            buf.as_mut_slice().fill(0);
        }
        if let Some(buf) = selector_q_modifiers_pinned.as_mut() {
            buf.as_mut_slice().fill(0);
        }
        if let Some(buf) = selector_candidates_pinned.as_mut() {
            buf.as_mut_slice().fill(0);
        }
        if let Some(buf) = selector_q_pinned.as_mut() {
            buf.as_mut_slice().fill(0);
        }

        *next_sampling_request = None;
        *active_sampling_request = None;
        *pending_sampled_proposal = None;
        *retr_used_last = false;
        *retr_misfire_streak = 0;
        *retr_cooldown = 0;
        *lookup_stats = lookup_stats::LookupStats::default();

        cache_k_start.clear();
        cache_k_start.resize(num_layers, 0);
        cache_k_end.clear();
        cache_k_end.resize(num_layers, 0);
        cache_v_start.clear();
        cache_v_start.resize(num_layers, 0);
        cache_v_end.clear();
        cache_v_end.resize(num_layers, 0);
        *cache_fc_start = 0;
        *cache_fc_end = 0;
        *ctx_kv_repr = None;

        *pending_tree_payload = None;
        *accept_history = [0u8; 8];
        *accept_history_pos = 0;
        *accept_history_count = 0;
        *propose_steps = 0;
        *throughput_router = throughput_router::ThroughputRouter::default();
        *climbdrop_router = throughput_router::ClimbDropRouter::default();
        *throughput_cycle_started = None;
        *throughput_last_width = 0;
        *fallback_suppressed_remaining = 0;

        recycle_tail.fill(0);
        recycle_tail.clear();
        if recycle_tail.capacity() > HOST_VEC_RETAIN_CAP {
            recycle_tail.shrink_to(0);
        }
        *recycle_key = 0;
        *recycle_valid = false;
        *recycle_last_offered = false;

        echo_tail.fill(0);
        echo_tail.clear();
        if echo_tail.capacity() > HOST_VEC_RETAIN_CAP {
            echo_tail.shrink_to(0);
        }
        *echo_key = 0;
        *echo_valid = false;
        *echo_streak = 0;
        *echo_offered_last = false;

        *async_placeholder = false;
        *pending_gumbel_spec
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
        last_draft_probs.clear();
        if last_draft_probs.capacity() > HOST_VEC_RETAIN_CAP {
            last_draft_probs.shrink_to(0);
        }
        *pending_draft_probs_gamma = 0;
        *pending_rejection_slot
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
        *last_rejection_q
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
    }

    fn scrub_for_retirement(&mut self) {
        let num_layers = self.ctx_k_cache.len();
        self.reset_all_transient_state(None, num_layers);
    }
}

#[cfg(test)]
mod context_debt_contract_tests {
    use super::{DflashProposerState, checked_ctx_capture_debt, lookup_stats, throughput_router};
    use spark_runtime::gpu::GpuBackend;
    use spark_runtime::gpu::mock::MockGpuBackend;

    fn test_dflash_state(
        gpu: &MockGpuBackend,
        max_ctx_len: usize,
        ctx_capacity: usize,
        ctx_slot_bytes: usize,
        initial_ctx_len: usize,
    ) -> DflashProposerState {
        let total = ctx_capacity * ctx_slot_bytes;
        let ctx_hidden_acc = gpu.alloc(total).expect("alloc ctx_hidden_acc");
        let ctx_fc_cache = gpu.alloc(64).expect("alloc ctx_fc_cache");
        let pos_pinned = gpu.alloc_host_pinned(64).expect("alloc pos_pinned");
        let token_ids_pinned = gpu.alloc_host_pinned(64).expect("alloc token_ids_pinned");
        let draft_output_pinned = gpu
            .alloc_host_pinned(64)
            .expect("alloc draft_output_pinned");
        DflashProposerState {
            last_num_drafted: 0,
            draft_budget: None,
            ctx_hidden_acc,
            ctx_len: initial_ctx_len,
            ctx_resident_len: initial_ctx_len.min(ctx_capacity),
            ctx_valid_from_abs: 0,
            ctx_capacity,
            ctx_allocation_bytes: total,
            max_ctx_len,
            ctx_slot_bytes,
            last_capture_idx: 0,
            last_num_accepted: 0,
            last_accepted_compact: Vec::new(),
            pld_tokens: Vec::new(),
            first_propose_done: true,
            pos_pinned,
            token_ids_pinned,
            draft_output_pinned,
            next_sampling_request: None,
            active_sampling_request: None,
            selector_uniforms_pinned: None,
            selector_q_modifiers_pinned: None,
            selector_candidates_pinned: None,
            selector_q_pinned: None,
            pending_sampled_proposal: None,
            retr_used_last: false,
            retr_misfire_streak: 0,
            retr_cooldown: 0,
            lookup_stats: lookup_stats::LookupStats::default(),
            ctx_fc_cache,
            ctx_k_cache: Vec::new(),
            ctx_v_cache: Vec::new(),
            cache_k_start: Vec::new(),
            cache_k_end: Vec::new(),
            cache_v_start: Vec::new(),
            cache_v_end: Vec::new(),
            cache_fc_start: 0,
            cache_fc_end: 0,
            ctx_kv_repr: None,
            pending_tree_payload: None,
            accept_history: [0u8; 8],
            accept_history_pos: 0,
            accept_history_count: 0,
            propose_steps: 0,
            throughput_router: throughput_router::ThroughputRouter::default(),
            climbdrop_router: throughput_router::ClimbDropRouter::default(),
            throughput_cycle_started: None,
            throughput_last_width: 0,
            fallback_suppressed_remaining: 0,
            recycle_tail: Vec::new(),
            recycle_key: 0,
            recycle_valid: false,
            recycle_last_offered: false,
            echo_tail: Vec::new(),
            echo_key: 0,
            echo_valid: false,
            echo_streak: 0,
            echo_offered_last: false,
            async_placeholder: false,
            pending_gumbel_spec: std::sync::Mutex::new(None),
            last_draft_probs: Vec::new(),
            pending_draft_probs_gamma: 0,
            pending_rejection_slot: std::sync::Mutex::new(None),
            last_rejection_q: std::sync::Mutex::new(None),
            pool_owner_head: 0,
            ctx_fc_bytes: 64,
            ctx_kv_bytes: 0,
            target_is_moe: None,
        }
    }

    #[test]
    fn sampled_boundary_debt_is_root_plus_accepted_prefix() {
        // Production failure trace: proposal appended [167,171), verify then
        // accepted two drafts and left captures for [171,174).  The boundary
        // must retire all three before bootstrap decode overwrites row zero.
        assert_eq!(checked_ctx_capture_debt(171, 174, 2).unwrap(), 3);
    }

    #[test]
    fn reset_prefill_and_repeated_bootstrap_have_exact_debt() {
        assert_eq!(checked_ctx_capture_debt(0, 0, 0).unwrap(), 0);
        assert_eq!(checked_ctx_capture_debt(8192, 8192, 0).unwrap(), 0);
        assert_eq!(checked_ctx_capture_debt(8192, 8193, 0).unwrap(), 1);
    }

    #[test]
    fn cursor_drift_and_accept_overflow_fail_closed() {
        assert!(checked_ctx_capture_debt(171, 175, 2).is_err());
        assert!(checked_ctx_capture_debt(175, 174, 0).is_err());
        assert!(checked_ctx_capture_debt(0, 1, usize::MAX).is_err());
    }

    #[test]
    fn serial_capture_and_reconcile_round_trip_is_exact_and_idempotent() {
        let gpu = MockGpuBackend::new();
        let slot_bytes = 16usize;
        let capacity = 8usize;
        let mut dstate = test_dflash_state(&gpu, 64, capacity, slot_bytes, 2);
        let captured = gpu.alloc(4 * slot_bytes).unwrap();

        // Simulate a verify step that accepted 2 drafts (3 pending rows at [2, 5)).
        dstate.last_capture_idx = 2;
        dstate.last_num_accepted = 2;
        let verify_rows: Vec<u8> = (0..3u8)
            .flat_map(|r| std::iter::repeat_n(0x10 + r, slot_bytes))
            .collect();
        gpu.copy_h2d(&verify_rows, captured).unwrap();

        // Pre-serial-decode reconcile retires the 3 verify rows before decode overwrites row 0.
        let retired = dstate
            .append_pending_ctx_captures(&gpu, captured, 5, 4, 0)
            .unwrap();
        assert_eq!(retired, 3);
        dstate.last_capture_idx = 0;
        dstate.last_num_accepted = 0;
        dstate.last_accepted_compact.clear();
        assert_eq!(dstate.ctx_len, 5);
        assert_eq!(dstate.ctx_resident_len, 5);

        // Serial decode step produces row 0 at absolute position 5 (`seq_len = 6`).
        let serial_row_5 = vec![0x55u8; slot_bytes];
        gpu.copy_h2d(&serial_row_5, captured).unwrap();
        let append = dstate
            .ctx_ring_state()
            .unwrap()
            .plan_append_at(5, 1)
            .unwrap();
        assert!(
            dstate
                .commit_serial_decode_capture(&gpu, captured, 6, append, 0)
                .unwrap()
        );
        assert_eq!(dstate.ctx_len, 6);
        assert_eq!(dstate.ctx_resident_len, 6);
        assert_eq!(dstate.last_num_accepted, 0);
        assert!(dstate.last_accepted_compact.is_empty());

        // Repeating the capture for the same `seq_len` is an idempotent no-op.
        assert!(
            !dstate
                .commit_serial_decode_capture(&gpu, captured, 6, append, 0)
                .unwrap()
        );
        assert_eq!(dstate.ctx_len, 6);

        // Re-entering bootstrap reconcile after serial decode sees zero debt (no cursor mismatch).
        assert_eq!(
            dstate
                .append_pending_ctx_captures(&gpu, captured, 6, 4, 0)
                .unwrap(),
            0
        );

        // Verify the exact bytes landed in slots 2, 3, 4, 5 of `ctx_hidden_acc`.
        let acc = gpu.read_alloc(dstate.ctx_hidden_acc).unwrap();
        assert_eq!(&acc[2 * slot_bytes..3 * slot_bytes], &[0x10; 16]);
        assert_eq!(&acc[3 * slot_bytes..4 * slot_bytes], &[0x11; 16]);
        assert_eq!(&acc[4 * slot_bytes..5 * slot_bytes], &[0x12; 16]);
        assert_eq!(&acc[5 * slot_bytes..6 * slot_bytes], &[0x55; 16]);
    }

    #[test]
    fn serial_capture_wraps_physical_ring_slot_without_absolute_offset_overflow() {
        let gpu = MockGpuBackend::new();
        let slot_bytes = 8usize;
        let capacity = 4usize;
        let mut dstate = test_dflash_state(&gpu, 16, capacity, slot_bytes, 4);
        let captured = gpu.alloc(slot_bytes).unwrap();
        gpu.copy_h2d(&[0xAB; 8], captured).unwrap();

        // Absolute position 4 wraps into physical slot `4 % 4 == 0`.
        let append = dstate
            .ctx_ring_state()
            .unwrap()
            .plan_append_at(4, 1)
            .unwrap();
        assert!(
            dstate
                .commit_serial_decode_capture(&gpu, captured, 5, append, 0)
                .unwrap()
        );
        assert_eq!(dstate.ctx_len, 5);
        assert_eq!(dstate.ctx_resident_len, 4);
        let acc = gpu.read_alloc(dstate.ctx_hidden_acc).unwrap();
        assert_eq!(&acc[0..slot_bytes], &[0xAB; 8]);
        assert_eq!(
            dstate
                .append_pending_ctx_captures(&gpu, captured, 5, 4, 0)
                .unwrap(),
            0
        );
    }

    #[test]
    fn multiturn_6799_bootstrap_external_draft_reconcile_keeps_verify_cursor_lockstep() {
        let gpu = MockGpuBackend::new();
        let slot_bytes = 16usize;
        let capacity = 4_096usize;
        let mut dstate = test_dflash_state(&gpu, 16_384, capacity, slot_bytes, 6_799);
        let captured = gpu.alloc(2 * slot_bytes).unwrap();

        // Failing to retire the 1-row bootstrap capture at position 6800 before
        // verifying 1 accepted draft to position 6802 reproduces the exact
        // fail-closed cursor disagreement observed on multi-turn TC-88.
        let err = checked_ctx_capture_debt(6_799, 6_802, 1).unwrap_err();
        assert!(
            err.to_string()
                .contains("cursor=6799 position=6802 pending=3 recorded=2"),
            "{err:#}"
        );

        // Bootstrap decode advances seq_len 6799 -> 6800, captures row 0, and
        // stamps `last_num_accepted = 0`.
        dstate.last_capture_idx = 0;
        dstate.last_num_accepted = 0;
        dstate.last_accepted_compact.clear();
        gpu.copy_h2d(&[0xB0u8; 16], captured).unwrap();

        // Reconciling before installing an external draft chain retires row 0
        // into physical ring slot `6799 % 4096 == 2703` and advances ctx_len to 6800.
        assert_eq!(
            dstate
                .append_pending_ctx_captures(&gpu, captured, 6_800, 8, 0)
                .unwrap(),
            1
        );
        dstate.last_capture_idx = 0;
        dstate.last_num_accepted = 0;
        dstate.last_accepted_compact.clear();
        assert_eq!(dstate.ctx_len, 6_800);
        assert_eq!(dstate.ctx_resident_len, capacity);

        // Repeating reconcile at position 6800 is an idempotent zero-row no-op.
        assert_eq!(
            dstate
                .append_pending_ctx_captures(&gpu, captured, 6_800, 8, 0)
                .unwrap(),
            0
        );
        assert_eq!(dstate.ctx_len, 6_800);

        // Verify then runs at position 6800 and accepts 1 draft (`seq_len = 6802`,
        // `last_num_accepted = 1`), leaving rows 0 and 1 in `captured`.
        dstate.last_capture_idx = 1;
        dstate.last_num_accepted = 1;
        let verify_rows: Vec<u8> = [0xC0u8, 0xC1u8]
            .into_iter()
            .flat_map(|b| std::iter::repeat_n(b, slot_bytes))
            .collect();
        gpu.copy_h2d(&verify_rows, captured).unwrap();

        assert_eq!(
            dstate
                .append_pending_ctx_captures(&gpu, captured, 6_802, 8, 0)
                .unwrap(),
            2
        );
        assert_eq!(dstate.ctx_len, 6_802);
        assert_eq!(dstate.ctx_resident_len, capacity);

        let acc = gpu.read_alloc(dstate.ctx_hidden_acc).unwrap();
        let slot_6799 = 6_799 % capacity;
        let slot_6800 = 6_800 % capacity;
        let slot_6801 = 6_801 % capacity;
        assert_eq!(
            &acc[slot_6799 * slot_bytes..(slot_6799 + 1) * slot_bytes],
            &[0xB0; 16]
        );
        assert_eq!(
            &acc[slot_6800 * slot_bytes..(slot_6800 + 1) * slot_bytes],
            &[0xC0; 16]
        );
        assert_eq!(
            &acc[slot_6801 * slot_bytes..(slot_6801 + 1) * slot_bytes],
            &[0xC1; 16]
        );
    }

    #[test]
    fn gumbel_spec_stamp_take_isolates_greedy_and_falls_back_to_ring() {
        use spark_runtime::sampler::gumbel::{
            GumbelDraftSpec, lookup_gumbel_draft_spec, register_gumbel_draft_spec,
        };

        let gpu = MockGpuBackend::new();
        let dstate = test_dflash_state(&gpu, 64, 8, 16, 4);
        let active = GumbelDraftSpec {
            seed: 0x1234_5678,
            temperature: 0.7,
            top_k: 20,
            top_p: 0.95,
            min_p: 0.05,
            step0_abs_pos: 42,
        };
        let inactive = GumbelDraftSpec {
            temperature: 0.0,
            ..active
        };

        // Disabled mode drains any stamped spec and returns None.
        dstate.set_gumbel_spec(Some(active));
        assert_eq!(dstate.take_gumbel_spec_with(999_912_341, false), None);
        // Drained slot does not replay when enabled later.
        assert_eq!(dstate.take_gumbel_spec_with(999_912_341, true), None);

        // Active stamped spec is returned exactly once.
        dstate.set_gumbel_spec(Some(active));
        assert_eq!(
            dstate.take_gumbel_spec_with(999_912_342, true),
            Some(active)
        );
        assert_eq!(dstate.take_gumbel_spec_with(999_912_342, true), None);

        // Explicit greedy `Some(None)` suppresses global-ring fallback without consuming it.
        let ring_pos = 999_912_343usize;
        register_gumbel_draft_spec(ring_pos, active);
        dstate.set_gumbel_spec(None);
        assert_eq!(dstate.take_gumbel_spec_with(ring_pos, true), None);

        // Inactive stamped spec (`temperature == 0.0`) also suppresses global-ring fallback.
        dstate.set_gumbel_spec(Some(inactive));
        assert_eq!(dstate.take_gumbel_spec_with(ring_pos, true), None);

        // Unstamped state falls back to the global ring entry registered at `ring_pos`.
        assert_eq!(dstate.take_gumbel_spec_with(ring_pos, true), Some(active));
        // Clean up / verify the ring entry was consumed.
        assert_eq!(lookup_gumbel_draft_spec(ring_pos), None);
    }

    #[test]
    fn last_draft_probs_initializes_empty_reads_f32_and_never_leaks_stale_values() {
        use super::{
            DDTREE_TOP_K_MAX, DFLASH_BATCH_OUT_PROBS_MIN_SLOTS, DFLASH2_GUMBEL_SHORTLIST_K,
            read_draft_probs_into,
        };
        use spark_runtime::gpu::DevicePtr;

        assert_eq!(DDTREE_TOP_K_MAX, 16);
        assert_eq!(DFLASH2_GUMBEL_SHORTLIST_K, 32);
        assert_eq!(DFLASH_BATCH_OUT_PROBS_MIN_SLOTS * size_of::<f32>(), 2048);

        let gpu = MockGpuBackend::new();
        let mut dstate = test_dflash_state(&gpu, 64, 8, 16, 4);
        assert!(dstate.last_draft_probs().is_empty());

        let expected = [0.91f32, 0.42, 0.08, 0.77];
        let bytes: Vec<u8> = expected.iter().flat_map(|v| v.to_le_bytes()).collect();
        let dev = gpu.alloc(bytes.len()).unwrap();
        gpu.copy_h2d(&bytes, dev).unwrap();

        read_draft_probs_into(
            &gpu,
            dev,
            expected.len(),
            Some(0),
            &mut dstate.last_draft_probs,
        )
        .unwrap();
        assert_eq!(dstate.last_draft_probs(), &expected);

        // Smaller read overwrites without leaking trailing elements from prior step.
        read_draft_probs_into(&gpu, dev, 2, None, &mut dstate.last_draft_probs).unwrap();
        assert_eq!(dstate.last_draft_probs(), &expected[..2]);

        // Zero-count or null pointer clears cleanly.
        read_draft_probs_into(
            &gpu,
            DevicePtr::NULL,
            4,
            Some(0),
            &mut dstate.last_draft_probs,
        )
        .unwrap();
        assert!(dstate.last_draft_probs().is_empty());
    }

    #[test]
    fn rejection_q_metadata_and_marconi_skip_state_transitions() {
        let gpu = MockGpuBackend::new();
        let mut dstate = test_dflash_state(&gpu, 4096, 256, 16, 0);

        assert_eq!(dstate.last_rejection_q(), None);
        dstate.set_pending_rejection_slot(Some(3));
        assert_eq!(dstate.take_pending_rejection_slot(), Some(3));
        assert_eq!(dstate.take_pending_rejection_slot(), None);

        dstate.set_last_rejection_q(Some((3, 512, 4)));
        assert_eq!(dstate.last_rejection_q(), Some((3, 512, 4)));
        dstate.truncate_rejection_q(2);
        assert_eq!(dstate.last_rejection_q(), Some((3, 512, 2)));
        dstate.truncate_rejection_q(0);
        assert_eq!(dstate.last_rejection_q(), None);
        dstate.set_last_rejection_q(Some((3, 512, 4)));
        dstate.invalidate_rejection_q();
        assert_eq!(dstate.last_rejection_q(), None);

        // Marconi skip advances ctx_len + ctx_valid_from_abs and invalidates caches.
        dstate.cache_fc_start = 1;
        dstate.cache_fc_end = 4;
        dstate.advance_marconi_skip(128).unwrap();
        assert_eq!(dstate.ctx_len, 128);
        assert_eq!(dstate.ctx_resident_len, 0);
        assert_eq!(dstate.ctx_valid_from_abs, 128);
        assert_eq!((dstate.cache_fc_start, dstate.cache_fc_end), (0, 0));
        let ring = dstate.ctx_ring_state().unwrap();
        assert_eq!(ring.resident_start(), 128);
        assert!(ring.slot_for(127).is_err());
    }
}

impl ProposerState for DflashProposerState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Family-specific draft-query/output geometry.
///
/// Generic DFlash trains an anchor followed by `gamma` MASK rows and emits
/// logits only from the MASK rows. DSpark instead runs exactly `gamma` query
/// rows (anchor plus `gamma - 1` MASK rows) and emits logits from every row,
/// including the anchor. The latter matches SGLang's DSpark proposal path:
/// row 0 is the target bonus token and all `gamma` raw hidden rows feed the
/// shared LM head and Markov sampler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct DraftRowLayout {
    pub query_rows: usize,
    pub output_start: usize,
}

impl DraftRowLayout {
    pub fn for_family(family: crate::weight_loader::DrafterCheckpointFamily, gamma: usize) -> Self {
        match family {
            crate::weight_loader::DrafterCheckpointFamily::Dflash => Self {
                query_rows: gamma + 1,
                output_start: 1,
            },
            crate::weight_loader::DrafterCheckpointFamily::Dspark => Self {
                query_rows: gamma,
                output_start: 0,
            },
        }
    }

    pub fn feedback_rows(self) -> usize {
        self.query_rows.saturating_sub(1)
    }
}

/// Block-diffusion draft head. Public API is the [`DraftProposer`] trait.
///
/// The drafter shares `embed_tokens` and `lm_head` with the target — these
/// are NOT in the drafter's safetensors checkpoint (verified against
/// `z-lab/Qwen3.6-35B-A3B-DFlash` commit 42d3b34). The constructor takes
/// the target's `embed_tokens_shared` and `lm_head_shared` device pointers
/// at build time and slots them in alongside the drafter's own `fc`,
/// `hidden_norm`, `norm`, and per-layer weights.
#[allow(dead_code)]
pub struct BlockDiffusionDraftHead {
    // Drafter-architecture config (mirrors the drafter's HF config.json).
    pub checkpoint_family: crate::weight_loader::DrafterCheckpointFamily,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub draft_vocab_size: usize,
    pub gamma: usize,
    /// Draft-token selection contract admitted at model construction.
    pub draft_sample_method: crate::dflash_sampling::DflashDraftSampleMethod,
    /// Verify-side physical token capacity, including the bonus/root row.
    pub physical_verify_k: usize,
    pub mask_token_id: u32,
    pub window_size: Option<usize>,
    /// Per-layer attention window in tokens. Non-zero `w` for `sliding_attention`
    /// layers (vLLM PR #40898 — Qwen3.6-27B-DFlash has 4× sliding + 1× full),
    /// `0` for `full_attention`. Length == `num_layers`. Empty Vec when
    /// the drafter omits `layer_types` (older 3.5 drafter — treated as all
    /// full-attention via the propose codepath).
    pub layer_window_sizes: Vec<u32>,
    /// Per-layer causal mask flag (vLLM PR #40898 — DFlash uses `causal=True`
    /// for `sliding_attention` layers and `causal=False` for `full_attention`).
    /// Same length as `layer_window_sizes`. Empty Vec falls back to `false`
    /// for all layers (the original DFlash 3.5 behavior).
    pub layer_causal: Vec<bool>,
    /// `target_layer_ids`. Same data as `TransformerModel::dflash_capture_layers`,
    /// repeated here so the loader is the single source of truth; the model
    /// reads these to size its capture buffer.
    pub target_layer_ids: Vec<usize>,
    /// Target-side hidden_size (used for the `fc` projection input width:
    /// `target_layer_ids.len() * target_hidden_size`).
    pub target_hidden_size: usize,
    /// Target-side vocab_size. The drafter's `vocab_size` may be larger
    /// (e.g. 248320 vs target 248077) but the shared lm_head only has
    /// `target_vocab_size` rows. Capped at construction time.
    pub target_vocab_size: usize,

    // === Weights shared with the target ===
    /// Target's embed_tokens GPU pointer. The drafter's checkpoint has no
    /// own embeddings — both vocab and embedding dim must match the target
    /// (Qwen3.6-35B-A3B-DFlash: vocab=248320, hidden=2048 — same as target).
    pub embed_tokens_shared: DevicePtr,
    /// Target's lm_head GPU pointer. Used for the drafter's per-position
    /// argmax over `[γ, vocab]` logits.
    pub lm_head_shared: DevicePtr,
    /// Target's transposed NVFP4 lm_head (shared pointer, no extra memory)
    /// for the propose lm_head fast path (`ATLAS_DFLASH_LM_HEAD_NVFP4=1`):
    /// reads the `--mtp-vocab` column prefix of the T-weight via
    /// `w4a16_gemm_t_m32_n64` (~0.25 GB at 96k vocab) instead of streaming
    /// the ~1 GB BF16 slice through scalar `dense_gemm`. Drafter-only —
    /// verify commits the target's own argmax, so this affects acceptance,
    /// never committed tokens. Wired post-construction in `factory/build.rs`.
    pub lm_head_shared_t: Option<crate::weight_map::QuantizedWeight>,
    /// Row stride (padded full-vocab N) of `lm_head_shared_t`.
    pub lm_head_shared_t_ldb: u32,
    /// Target's already-resident untransposed NVFP4 `lm_head` (shared pointer,
    /// zero extra GPU memory) and exact multi-row NVFP4 LM-head kernel bundle.
    /// Used on the default `LmHeadPath::Bf16` path to avoid streaming the BF16
    /// LM head through scalar `dense_gemm` while preserving the exact
    /// `LmHeadPath` resolution and norm contracts.
    pub lm_head_shared_nvfp4_exact: Option<crate::weight_map::QuantizedWeight>,
    pub lm_head_exact_kernels: Option<crate::layers::ops::W4a16ExactLmHeadKernels>,
    /// Pre-scaled FP8-E4M3 copy of the `--mtp-vocab` lm_head slice and its
    /// paired final-norm weight
    /// (`ATLAS_DFLASH_LM_HEAD_FP8=1`, built at load in `from_weights`):
    /// `[lm_vocab, K]` row-major, weights multiplied by a power-of-2 scale
    /// `s` chosen so absmax·s ≈ 256 (keeps small weights out of the E4M3
    /// subnormal floor). The compensating `1/s` is folded into
    /// `DflashFp8LmHead::input_norm`; the canonical `norm` stays unscaled for
    /// NVFP4, BF16, and selector consumers. Halves the propose lm_head read vs
    /// BF16 (0.98→0.49 GB) at far higher logit fidelity than the NVFP4 slice
    /// (which measured accepted 5.88→5.36 and was refuted, 2026-07-31).
    pub lm_head_shared_fp8: Option<DflashFp8LmHead>,

    // === Weights from the drafter checkpoint ===
    /// Hidden-norm applied to the projected target context before mixing
    /// with the embedded tokens (Qwen3-DFlash convention; see vLLM
    /// `DFlashQwen3Model.hidden_norm`).
    pub hidden_norm: DenseWeight,
    /// Final RMSNorm before LM head.
    pub norm: DenseWeight,
    /// `fc` projection — `[draft_hidden, target_layer_ids.len() * target_hidden_size]`
    /// BF16. Maps the stack of captured target hiddens to drafter's input space
    /// once at model entry. Replaces the earlier (incorrect) "per-layer KV
    /// injection" design.
    ///
    /// Populated under `DflashQuantization::Bf16`. Under `Nvfp4` this field
    /// holds a stale (zeroed) pointer after the BF16 source is freed, and
    /// the forward path reads `fc_nvfp4` instead. Kept as a typed field
    /// (rather than `Option`) so the rest of `BlockDiffusionDraftHead` stays
    /// shape-stable.
    pub fc: DenseWeight,
    /// NVFP4-quantized version of `fc`. `Some` only when the drafter was
    /// built with `DflashQuantization::Nvfp4`. `forward_block` Step 0 reads
    /// this through `w4a16_gemv` when present, else falls back to `fc` via
    /// `dense_gemv`.
    pub fc_nvfp4: Option<QuantizedWeight>,
    /// Online-quantized FP8 version of `fc` (`Some` only under
    /// `DflashQuantization::Fp8`).
    pub fc_fp8: Option<crate::weight_map::Fp8DenseWeight>,
    /// BF16 drafter-store tensors (`fc` and each layer's seven projections)
    /// that `DflashQuantization::Nvfp4` quantized and no longer reads.
    /// Released by the factory under `ATLAS_MEMLEAN_DRAFTER_BF16=1`, then
    /// cleared; empty in every other mode.
    pub(crate) quantized_bf16_sources: Vec<DevicePtr>,
    /// Optional draft-vocab-id → target-vocab-id remap. `None` when the
    /// drafter shares vocab with the target (Qwen3.6-35B-A3B-DFlash case:
    /// vocab_size == draft_vocab_size == 248320).
    pub draft_id_to_target_id: Option<DevicePtr>,

    /// DSpark VanillaMarkov head. `Some` when the checkpoint ships the
    /// `markov_head.markov_w{1,2}.weight` tensors and `markov_rank > 0` and
    /// `ATLAS_DFLASH_MARKOV != 0`. When present, `propose_drafts` applies a
    /// per-position bigram logit bias `B(prev) = W2(W1[prev])` and samples the
    /// γ-block LEFT-TO-RIGHT (each position's chosen token biases the next),
    /// semi-autoregressively repairing suffix decay. LOSSLESS w.r.t. committed
    /// output — only changes which tokens are *proposed*; the target verify
    /// still commits its own greedy token.
    pub markov: Option<MarkovHead>,
    /// DFlash 2 candidate selector (3 BF16 tensors) replacing the per-row
    /// argmax in the propose tail. `Some` only for DFlash2 checkpoints
    /// (`candidate_selector.*` present + `selector_rank > 0`). When present,
    /// the propose tail runs top-16 per position and walks a coherent path
    /// via low-rank codebook scores instead of plain per-row argmax
    /// (reference: `CandidateSelector.select`, z-lab/dflash dflash/model.py).
    /// Drafter-only — the target verify still commits its own greedy token.
    pub selector: Option<crate::weight_loader::DflashSelectorWeights>,
    /// Native DSpine (`arXiv:2609.36173`) Adjacent Causal Injection bridge +
    /// Pass-2 exact-sender `serial_table` refiner (`None` for non-DSpine drafters).
    pub dspine: Option<dspine::DspineCodeBridge>,
    /// Drafter transformer layers (8 for Qwen3.6-35B-A3B-DFlash). Each
    /// layer carries BF16, FP8, or NVFP4 weights — the `forward_block_layer`
    /// helper match-dispatches on the variant.
    pub layers: Vec<DflashLayerQuantWeights>,

    /// Default-off calibrated W4A4 execution state. `None` preserves the
    /// established W4A16 path and holds no additional activation scratch.
    w4a4_shadow: Option<w4a4_shadow::DflashW4a4Shadow>,

    /// Per-step scratch buffers (allocated once at construction, reused).
    pub scratch: DflashScratch,

    /// All kernel handles needed by `propose()` and the eventual prefill
    /// projection (`precompute_and_store_context_kv`).
    pub kernels: DflashKernels,

    /// Absolute logical sequence-length limit (mirrors model's `max_seq_len`),
    /// passed to `ring_window::plan_accumulator_layout` / `RingState::new`.
    /// Physical `ctx_hidden_acc` capacity in `alloc_state` is bounded by
    /// `min(max_seq_len, ctx_window.clamp(1, ring_window::NATIVE_V3_RING_SLOTS))`.
    pub max_seq_len: usize,

    /// Pre-computed yarn inv_freq table (`[head_dim/2]` f32 on GPU).
    /// Drafter rope_scaling: factor=64, beta_fast=32, beta_slow=1,
    /// original_max_position_embeddings=4096 (per drafter config.json).
    pub yarn_inv_freq: DevicePtr,

    /// YaRN's post-RoPE cos/sin multiplier. Transformers derives this as
    /// `1 + 0.1 * ln(factor)` unless the checkpoint supplies an explicit
    /// `attention_factor`; vanilla RoPE uses 1.0.
    pub rope_attention_factor: f32,

    /// rope_theta (10000000 for Qwen3.6-DFlash). Stored to pass into the
    /// rope_yarn kernel each step.
    pub rope_theta: f32,

    /// rotary_dim. Drafter uses full-rotation (rotary_dim = head_dim = 128).
    pub rotary_dim: usize,

    /// RMSNorm epsilon (drafter inherits Qwen3 default 1e-6).
    pub rms_norm_eps: f32,

    /// Max number of past target positions injected into the drafter's K/V
    /// per step. Default γ — drafter sees at most γ ctx + γ noise = 2γ
    /// attention positions per step. ctx_window=0 disables ctx conditioning
    /// (degraded quality, ablation only).
    pub ctx_window: usize,

    /// Process-environment proposal controls, frozen on first use. Runtime
    /// profiler signals are intentionally separate and remain dynamic.
    proposal_policy: std::sync::OnceLock<proposal_policy::ProposalPolicy>,

    /// Active drafter weight quantization mode (`Bf16`, `Fp8`, or `Nvfp4`).
    pub quant: DflashQuantization,

    // ── ATLAS_DFLASH_ASYNC (task #20) ──
    /// At most one in-flight async (second-stream) propose — the head owns a
    /// single scratch buffer set. `None` when idle / flag off. See
    /// `async_propose.rs` for the shared-scratch discipline.
    pub async_inflight: Mutex<Option<async_propose::AsyncInflight>>,
    /// At most one in-flight async batched (`C >= 2`) propose using alternating
    /// double-buffered output scratch slots (`Primary` / `Secondary`) so
    /// consecutive launches never alias.
    pub async_batched_inflight: Mutex<Option<async_propose::AsyncBatchedInflight>>,
    /// Alternating slot counter (`0` / `1`) toggled per async batched launch.
    pub async_batch_slot_toggle: std::sync::atomic::AtomicU8,
    /// Optional staged early-start Stage-1 (`ATLAS_DFLASH_EARLY_STAGE1=1`)
    /// `fc` + `hidden_norm` projection launched on `pstream` after the target's
    /// final capture layer.
    pub early_stage1_staged: Mutex<Option<async_propose::EarlyStage1Staged>>,
    /// Dedicated non-blocking CUDA stream for async propose launches.
    /// Lazily created on first eligible launch; `0` = creation failed →
    /// async permanently disabled.
    pub async_propose_stream: std::sync::OnceLock<u64>,
    /// Event used to order the propose stream after the default stream's
    /// prior writes (ctx-append D2Ds, verify captures). Set together with
    /// `async_propose_stream`.
    pub async_order_event: std::sync::atomic::AtomicU64,

    // ── ATLAS_DFLASH_FUSED ──
    /// Set by `arm_propose_overlap` immediately after verify returns (before
    /// commit is enqueued on the default stream). When true,
    /// `try_launch_async_propose` skips re-recording `async_order_event`
    /// (the pre-commit snapshot is already in place) so the propose stream
    /// only waits for verify, not for the ~10ms SSM commit + KV reshape.
    /// Cleared on each consume in `try_launch_async_propose`.
    pub fused_event_armed: std::sync::atomic::AtomicBool,
    /// Stage-2 CUDA graph cache and rotating pinned metadata staging for
    /// cross-sequence batched DFlash2 proposals (`batched_propose.rs`).
    batched_noise_graph: Mutex<batched_propose::BatchedNoiseGraphCache>,
    /// Guards one-time release of `scratch.q_row_bases` during model teardown.
    q_row_bases_released: std::sync::atomic::AtomicBool,
    /// When true (batched DFlash2 mode), omit per-sequence `ctx_fc_cache`
    /// allocations (`-10..20 MiB/seq`) since Stage-1/Stage-2 batched proposals
    /// exclusively consume per-layer PostRope `ctx_{k,v}_cache`.
    skip_ctx_fc_cache: bool,
    /// Whether the bound target model is MoE (`Some(true)` when `num_experts > 1`),
    /// dense (`Some(false)`), or unspecified (`None` before `sync_target_config`).
    pub(crate) target_is_moe: Option<bool>,
}

impl BlockDiffusionDraftHead {
    /// Copy a range of cached context slots from a circular buffer into a
    /// linear destination. Handles wrap-around with at most two D2D copies.
    ///
    /// `cache` is a circular buffer of `window` slots, each `slot_bytes`.
    /// Valid absolute positions are `[cache_start..cache_end)`.
    /// We need positions `[needed_start..needed_end)`.
    /// Copies overlap into `dst` at offsets `[pos - needed_start)`.
    fn cache_copy_range(
        &self,
        gpu: &dyn GpuBackend,
        cache: DevicePtr,
        cache_start: usize,
        cache_end: usize,
        window: usize,
        needed_start: usize,
        needed_end: usize,
        dst: DevicePtr,
        slot_bytes: usize,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            !cache.is_null(),
            "DFlash cache_copy_range called with NULL cache pointer"
        );
        let plan =
            ring_window::plan_ring_copy(cache_start, cache_end, window, needed_start, needed_end)?;
        for span in plan.spans() {
            let src_offset = span
                .src_slot
                .checked_mul(slot_bytes)
                .ok_or_else(|| anyhow::anyhow!("DFlash ring source byte offset overflow"))?;
            let dst_offset = span
                .dst_slot
                .checked_mul(slot_bytes)
                .ok_or_else(|| anyhow::anyhow!("DFlash ring destination byte offset overflow"))?;
            let bytes = span
                .slot_count
                .checked_mul(slot_bytes)
                .ok_or_else(|| anyhow::anyhow!("DFlash ring copy byte count overflow"))?;
            gpu.copy_d2d_async(
                cache.offset(src_offset),
                dst.offset(dst_offset),
                bytes,
                stream,
            )?;
        }
        Ok(())
    }

    /// Write newly-computed context slots from a linear source into a
    /// circular cache. `src` holds `write_count` slots starting at absolute
    /// position `write_start`. Returns the updated `(cache_start, cache_end)`.
    fn cache_write_range(
        &self,
        gpu: &dyn GpuBackend,
        src: DevicePtr,
        write_start: usize,
        write_count: usize,
        cache: DevicePtr,
        cache_start: usize,
        cache_end: usize,
        window: usize,
        slot_bytes: usize,
        stream: u64,
    ) -> Result<(usize, usize)> {
        if cache.is_null() {
            // `ctx_fc_cache` was intentionally skipped in batched DFlash2 mode;
            // keep the FC cache range empty `(0, 0)` so any C=1 fallback never
            // reads a NULL `ctx_fc_cache` pointer.
            return Ok((0, 0));
        }
        if write_count == 0 {
            return Ok((cache_start, cache_end));
        }
        let write_end = write_start
            .checked_add(write_count)
            .ok_or_else(|| anyhow::anyhow!("DFlash cache write end overflowed"))?;
        let plan =
            ring_window::plan_write_chunk(write_start, write_count, write_end.max(window), window)?;
        for span in plan.spans() {
            gpu.copy_d2d_async(
                src.offset(span.linear_slot * slot_bytes),
                cache.offset(span.physical_slot * slot_bytes),
                span.slot_count * slot_bytes,
                stream,
            )?;
        }
        // Expand cache range to include the new positions.
        let (mut new_start, new_end) = if cache_start == 0 && cache_end == 0 {
            (write_start, write_end)
        } else {
            (cache_start.min(write_start), cache_end.max(write_end))
        };
        // Cap at window size (oldest positions are implicitly evicted).
        if new_end - new_start > window {
            new_start = new_end - window;
        }
        Ok((new_start, new_end))
    }

    /// True when all six batched Gumbel selector metadata scratch buffers are allocated.
    pub(super) fn gumbel_selector_scratch_ready(&self) -> bool {
        !self.scratch.batch_gumbel_seeds.is_null()
            && !self.scratch.batch_gumbel_step0_pos.is_null()
            && !self.scratch.batch_gumbel_temps.is_null()
            && !self.scratch.batch_gumbel_top_ks.is_null()
            && !self.scratch.batch_gumbel_top_ps.is_null()
            && !self.scratch.batch_gumbel_min_ps.is_null()
    }

    /// True when the K<=32 Gumbel-Max selector kernel and its `batch_out_probs`
    /// output scratch are both available.
    pub(super) fn gumbel_selector_k32_ready(&self) -> bool {
        self.kernels.dflash2_selector_walk_gumbel_batched_k32.0 != 0
            && self.gumbel_selector_scratch_ready()
            && !self.scratch.batch_out_probs.is_null()
    }

    /// True when the batched probabilistic rejection selector kernel,
    /// per-sequence `q_row_bases`, and model-bound shared `(q_cand_ids, q_probs)`
    /// scratch are all available.
    pub(super) fn rejection_selector_ready(&self) -> bool {
        self.kernels.dflash2_selector_walk_probabilistic_batched.0 != 0
            && self.gumbel_selector_scratch_ready()
            && !self.scratch.q_row_bases.is_null()
            && !self.scratch.shared_q_ids.is_null()
            && !self.scratch.shared_q_probs.is_null()
            && self.scratch.shared_q_row_cap > 0
            && self.scratch.shared_q_stride > 0
            && !self.scratch.batch_out_probs.is_null()
    }

    #[inline]
    fn pool_head_id(&self) -> usize {
        self as *const Self as usize
    }

    /// Pop the most recently retired [`DflashProposerState`] from this head's
    /// free-list whose buffer geometry matches `geometry` exactly.
    fn take_pooled_state(
        &self,
        geometry: &DflashStateBufferGeometry,
    ) -> Option<Box<dyn ProposerState>> {
        let head_id = self.pool_head_id();
        let mut registry = dflash_state_pool_registry().lock();
        let entries = registry.get_mut(&head_id)?;
        let idx = entries
            .iter()
            .rposition(|entry| entry.geometry == *geometry)?;
        Some(entries.swap_remove(idx).state)
    }
}

impl Drop for BlockDiffusionDraftHead {
    fn drop(&mut self) {
        let orphaned = drop_dflash_state_pool_entry(self.pool_head_id());
        if orphaned > 0 {
            tracing::warn!(
                head_id = self.pool_head_id(),
                orphaned_entries = orphaned,
                "BlockDiffusionDraftHead dropped before release_rejection_resources drained pooled state buffers"
            );
        }
        if noise_pass::drop_noise_graph_head_entry(self.pool_head_id()) {
            tracing::warn!(
                head_id = self.pool_head_id(),
                "BlockDiffusionDraftHead dropped before release_rejection_resources cleared cached noise graph state"
            );
        }
    }
}

/// Opt-in dense-target `p_cum` policy (`ATLAS_DFLASH_DENSE_PCUM=1`, default OFF):
/// preserves full fixed-width `K` on dense `C = 1` non-thinking steps while
/// engaging `p_cum` confidence truncation on dense cohorts `C >= 4` or inside
/// `<think>` reasoning blocks.
pub const DFLASH_DENSE_PCUM_ENV: &str = "ATLAS_DFLASH_DENSE_PCUM";

/// Minimum dense cohort concurrency admitted by `ATLAS_DFLASH_DENSE_PCUM=1`.
pub const DFLASH_DENSE_PCUM_MIN_COHORT: usize = 4;

/// Parse `ATLAS_DFLASH_DENSE_PCUM`: exact `"1"` (trimmed) opts in; default OFF.
#[inline]
#[must_use]
pub fn parse_dflash_dense_pcum_opt_in(raw: Option<&str>) -> bool {
    raw.map(str::trim) == Some("1")
}

/// Process-lifetime cached gate for `ATLAS_DFLASH_DENSE_PCUM=1` (default OFF).
#[must_use]
pub fn dflash_dense_pcum_opt_in() -> bool {
    let read_live = || match std::env::var(DFLASH_DENSE_PCUM_ENV) {
        Ok(val) => Some(val),
        Err(std::env::VarError::NotPresent) => None,
        Err(std::env::VarError::NotUnicode(_)) => {
            tracing::warn!(
                "{DFLASH_DENSE_PCUM_ENV} contains non-UTF-8 bytes; failing closed to OFF"
            );
            None
        }
    };
    #[cfg(not(test))]
    let raw = {
        static RAW: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
        RAW.get_or_init(read_live).as_deref()
    };
    #[cfg(test)]
    let raw_owned = read_live();
    #[cfg(test)]
    let raw = raw_owned.as_deref();
    parse_dflash_dense_pcum_opt_in(raw)
}

/// Whether DFlash2 `p_cum` confidence truncation (and its `batch_out_probs`
/// D2H readback) should engage for a target model.
///
/// - Explicit `ATLAS_QWEN4_DFLASH_PCUM_THRESH` overrides the default: finite
///   `> 0.0` enables it on any model, while finite `<= 0.0` disables it on all
///   models.
/// - When unset or invalid, MoE targets (`Some(true)`) and unspecified fixtures
///   (`None`) stay enabled, while dense targets (`Some(false)`) disable `p_cum`
///   truncation so full `K` acceptance and fixed-width CUDA graphs are preserved.
#[must_use]
pub fn dflash_pcum_enabled_for_target(target_is_moe: Option<bool>, raw_env: Option<&str>) -> bool {
    if let Some(trimmed) = raw_env.map(str::trim).filter(|s| !s.is_empty())
        && let Ok(tau) = trimmed.parse::<f32>()
        && tau.is_finite()
    {
        return tau > 0.0;
    }
    !matches!(target_is_moe, Some(false))
}

/// Context-aware DFlash2 `p_cum` policy supporting opt-in dense 27B truncation
/// at `cohort_size >= 4` or `inside_thinking` (`ATLAS_DFLASH_DENSE_PCUM=1`)
/// while keeping dense `C = 1` non-thinking default behavior byte-identical.
#[must_use]
pub fn dflash_pcum_enabled_for_target_context(
    target_is_moe: Option<bool>,
    cohort_size: usize,
    inside_thinking: bool,
    dense_opt_in: bool,
    explicit_thresh_env: Option<&str>,
) -> bool {
    if let Some(trimmed) = explicit_thresh_env.map(str::trim).filter(|s| !s.is_empty())
        && let Ok(tau) = trimmed.parse::<f32>()
        && tau.is_finite()
    {
        return tau > 0.0;
    }
    if !matches!(target_is_moe, Some(false)) {
        return true;
    }
    dense_opt_in && (cohort_size >= DFLASH_DENSE_PCUM_MIN_COHORT || inside_thinking)
}

/// Runtime wrapper over [`dflash_pcum_enabled_for_target_context`].
#[must_use]
pub fn dflash_pcum_enabled_for_target_context_env(
    target_is_moe: Option<bool>,
    cohort_size: usize,
    inside_thinking: bool,
) -> bool {
    let read_live = || std::env::var("ATLAS_QWEN4_DFLASH_PCUM_THRESH").ok();
    #[cfg(not(test))]
    let raw = {
        static RAW: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
        RAW.get_or_init(read_live).as_deref()
    };
    #[cfg(test)]
    let raw_owned = read_live();
    #[cfg(test)]
    let raw = raw_owned.as_deref();
    dflash_pcum_enabled_for_target_context(
        target_is_moe,
        cohort_size,
        inside_thinking,
        dflash_dense_pcum_opt_in(),
        raw,
    )
}

/// Runtime wrapper over [`dflash_pcum_enabled_for_target`] reading
/// `ATLAS_QWEN4_DFLASH_PCUM_THRESH`.
#[must_use]
pub fn dflash_pcum_enabled_for_target_env(target_is_moe: Option<bool>) -> bool {
    let read_live = || std::env::var("ATLAS_QWEN4_DFLASH_PCUM_THRESH").ok();
    #[cfg(not(test))]
    let raw = {
        static RAW: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
        RAW.get_or_init(read_live).as_deref()
    };
    #[cfg(test)]
    let raw_owned = read_live();
    #[cfg(test)]
    let raw = raw_owned.as_deref();
    dflash_pcum_enabled_for_target(target_is_moe, raw)
}

/// True when `batch_out_probs` should be read back alongside draft tokens so
/// either MoE/explicit `p_cum` or opt-in `ATLAS_DFLASH_DENSE_PCUM=1` can inspect them.
#[must_use]
pub fn dflash_pcum_probs_readback_enabled(target_is_moe: Option<bool>) -> bool {
    dflash_pcum_enabled_for_target_env(target_is_moe)
        || (matches!(target_is_moe, Some(false)) && dflash_dense_pcum_opt_in())
}

/// Read `count` little-endian `f32` draft probabilities from `dev_ptr` into
/// `out` without per-step heap allocation for `count <= 64`. When `stream` is
/// `Some(s)`, performs `copy_d2h_on_stream`; when `None`, performs `copy_d2h`
/// (for callers that already synchronized the producing stream).
pub(super) fn read_draft_probs_into(
    gpu: &dyn GpuBackend,
    dev_ptr: DevicePtr,
    count: usize,
    stream: Option<u64>,
    out: &mut Vec<f32>,
) -> Result<()> {
    out.clear();
    if count == 0 || dev_ptr.is_null() {
        return Ok(());
    }
    let bytes = count
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| anyhow::anyhow!("DFlash draft probs byte length overflow"))?;
    let mut stack_buf = [0u8; 64 * size_of::<f32>()];
    let mut heap_buf = Vec::new();
    let dst: &mut [u8] = if bytes <= stack_buf.len() {
        &mut stack_buf[..bytes]
    } else {
        heap_buf.resize(bytes, 0);
        &mut heap_buf
    };
    match stream {
        Some(s) => gpu.copy_d2h_on_stream(dev_ptr, dst, s)?,
        None => gpu.copy_d2h(dev_ptr, dst)?,
    }
    out.reserve(count);
    for chunk in dst.chunks_exact(size_of::<f32>()) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(())
}

pub mod async_propose;
mod batched_propose;
pub mod dcoco;
pub mod dcoco_weights;
pub mod ddtree;
pub mod ddtree_gdn_contract;
pub mod ddtree_gdn_dispatch;
pub mod dflash3;
mod draft_budget;
mod draft_splitk;
pub mod dspine;
pub mod echo;
// Source-pinned by device_transport_tests: lint here, not in the file.
#[allow(clippy::let_and_return)]
mod forward_block;
mod forward_block_layer;
mod from_weights;
mod logits_layout;
pub mod lookup_stats;
mod markov;
pub mod ngram_bonus;
mod noise_pass;
mod pctree;
mod proposal_inputs;
mod proposal_policy;
mod propose;
pub mod retrieval;
pub(crate) mod ring_window;
mod selector_policy;
mod throughput_router;
mod w4a4_admission;
#[cfg(test)]
mod w4a4_admission_tests;
mod w4a4_shadow;

// Re-export DDTree payload so the scheduler can carry it as Option<DDTreePayload>
// in ActiveSeq (M3 milestone — pure plumbing, no behavior change).
pub use ddtree::TreePayload as DDTreePayload;

// ── Kernel profiler accumulator (ATLAS_DFLASH_KERNEL_PROFILE=1) ──
// Per-kernel μs accumulated across all drafter layers in one propose()
// call. Reset by `kprof_reset_layers()` at the top of forward_block;
// `forward_block_layer` adds into each field via `kp!`; snapshot is
// returned by `kprof_snapshot_layers()` at end of forward_block.
//
// Thread-local because Atlas serves one request per stream and we only
// want per-step aggregation, not global. Using `Cell<KprofAcc>` keeps
// the helpers cheap and Copy.
#[derive(Clone, Copy, Default)]
pub(super) struct KprofAcc {
    pub input_norm_us: u128,
    pub q_proj_us: u128,
    pub kv_ctx_copy_us: u128,
    pub kv_ctx_new_us: u128,
    pub kv_noise_us: u128,
    pub qk_norm_us: u128,
    pub rope_us: u128,
    pub cache_write_us: u128,
    pub prefill_attn_us: u128,
    pub o_proj_us: u128,
    pub resid1_us: u128,
    pub post_norm_us: u128,
    pub gate_up_us: u128,
    pub silu_mul_us: u128,
    pub down_proj_us: u128,
    pub resid2_us: u128,
    /// DFlash2 attention-conv `prepare` / `finish` (kernel_projection GEMM +
    /// conv + the D2D copy-back). Previously unlabelled on the NVFP4 path, so
    /// a profile could not tell "ran unwrapped" from "never ran".
    pub conv_prepare_us: u128,
    pub conv_finish_us: u128,
}

thread_local! {
    static KPROF_ACC: std::cell::Cell<KprofAcc> = const { std::cell::Cell::new(KprofAcc {
        input_norm_us: 0, q_proj_us: 0, kv_ctx_copy_us: 0, kv_ctx_new_us: 0,
        kv_noise_us: 0, qk_norm_us: 0, rope_us: 0, cache_write_us: 0,
        prefill_attn_us: 0, o_proj_us: 0, resid1_us: 0, post_norm_us: 0,
        gate_up_us: 0, silu_mul_us: 0, down_proj_us: 0, resid2_us: 0,
        conv_prepare_us: 0, conv_finish_us: 0,
    }) };
}

pub(super) fn kprof_reset_layers() {
    KPROF_ACC.with(|c| c.set(KprofAcc::default()));
}

pub(super) fn kprof_snapshot_layers() -> KprofAcc {
    KPROF_ACC.with(|c| c.get())
}

impl KprofAcc {
    /// Every field paired with the label used in the `DFLASH_KP` log line and
    /// in the `KPROF` table. Keeping one list means a new field cannot be
    /// added to the report without also being published to `ATLAS_FULL_PROFILE`.
    pub(super) fn labelled(&self) -> [(&'static str, u128); 18] {
        [
            ("draft_input_norm", self.input_norm_us),
            ("draft_q_proj", self.q_proj_us),
            ("draft_kv_ctx_copy", self.kv_ctx_copy_us),
            ("draft_kv_ctx_new", self.kv_ctx_new_us),
            ("draft_kv_noise", self.kv_noise_us),
            ("draft_qk_norm", self.qk_norm_us),
            ("draft_rope", self.rope_us),
            ("draft_cache_write", self.cache_write_us),
            ("draft_prefill_attn", self.prefill_attn_us),
            ("draft_o_proj", self.o_proj_us),
            ("draft_resid1", self.resid1_us),
            ("draft_post_norm", self.post_norm_us),
            ("draft_gate_up", self.gate_up_us),
            ("draft_silu_mul", self.silu_mul_us),
            ("draft_down_proj", self.down_proj_us),
            ("draft_resid2", self.resid2_us),
            ("draft_conv_prepare", self.conv_prepare_us),
            ("draft_conv_finish", self.conv_finish_us),
        ]
    }

    /// Sum of every attributed field, in microseconds.
    pub(super) fn attributed_us(&self) -> u128 {
        self.labelled().iter().map(|(_, v)| *v).sum()
    }
}

/// True when the per-kernel propose profiler should run: either its own
/// `ATLAS_DFLASH_KERNEL_PROFILE=1`, or `ATLAS_FULL_PROFILE=1` (including the
/// SIGUSR1 runtime override), so ONE flag attributes both the verify path and
/// the drafter's internal kernels.
///
/// Historically only the verify path answered to `ATLAS_FULL_PROFILE`; the
/// drafter's 6 transformer layers used this separate accumulator and so were
/// invisible in a full profile — the drafter's lm_head was the only kernel
/// that appeared, because it is the one propose-side launch wrapped in
/// `kprof!` rather than the layer-local `kp!`.
pub(super) fn kernel_profile_enabled() -> bool {
    std::env::var("ATLAS_DFLASH_KERNEL_PROFILE").ok().as_deref() == Some("1")
        || crate::full_profile::is_enabled()
}

pub(super) fn kprof_add(f: impl FnOnce(&mut KprofAcc)) {
    KPROF_ACC.with(|c| {
        let mut a = c.get();
        f(&mut a);
        c.set(a);
    });
}

impl DraftProposer for BlockDiffusionDraftHead {
    fn kind(&self) -> crate::speculative::DraftProposerKind {
        crate::speculative::DraftProposerKind::DFlash
    }

    fn is_dflash(&self) -> bool {
        true
    }

    fn dflash_draft_sample_method(&self) -> crate::dflash_sampling::DflashDraftSampleMethod {
        self.draft_sample_method
    }

    fn dflash_sampled_proposal_width(&self, requested: usize) -> Result<usize> {
        let budget = draft_budget::DflashDraftBudget::new(
            requested,
            effective_draft_width(self.gamma),
            self.physical_verify_k,
        )?;
        budget.validate_flat_width(budget.flat)?;
        Ok(budget.flat)
    }

    fn stage_dflash_sampling_request(
        &self,
        state: &mut dyn ProposerState,
        request: crate::dflash_sampling::DflashDraftSamplingRequest,
    ) -> Result<()> {
        anyhow::ensure!(
            self.draft_sample_method
                == crate::dflash_sampling::DflashDraftSampleMethod::Probabilistic,
            "sampled DFlash request cannot be staged on a greedy proposer"
        );
        anyhow::ensure!(
            self.markov.is_none(),
            "probabilistic DFlash2 sampling cannot be combined with a DSpark Markov rewrite"
        );
        let request_gamma = if request.request_sampler.temperature > 0.0 {
            request.draft_uniforms.len()
        } else {
            self.gamma
        };
        request.validate(request_gamma)?;
        anyhow::ensure!(
            request_gamma <= self.gamma,
            "sampled DFlash request width exceeds the trained selector width"
        );
        let dstate = state
            .as_any_mut()
            .downcast_mut::<DflashProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid DFlash proposer state"))?;
        anyhow::ensure!(
            dstate.next_sampling_request.is_none() && dstate.active_sampling_request.is_none(),
            "DFlash sampling request slot is already occupied"
        );
        dstate.next_sampling_request = Some(request);
        Ok(())
    }

    fn take_pending_sampled_proposal(
        &self,
        state: &mut dyn ProposerState,
    ) -> Option<crate::dflash_sampling::DflashSampledProposalFrame> {
        state
            .as_any_mut()
            .downcast_mut::<DflashProposerState>()?
            .pending_sampled_proposal
            .take()
    }

    fn physical_verify_k(&self) -> Option<usize> {
        Some(self.physical_verify_k)
    }

    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn ProposerState>> {
        anyhow::ensure!(
            self.max_seq_len <= ring_window::MAX_ABSOLUTE_CONTEXT,
            "DFlash absolute context {} exceeds the checked ring limit {}",
            self.max_seq_len,
            ring_window::MAX_ABSOLUTE_CONTEXT,
        );
        anyhow::ensure!(
            self.ctx_window <= ring_window::NATIVE_V3_RING_SLOTS,
            "DFlash requested context window {} exceeds the qualified resident ring capacity {}",
            self.ctx_window,
            ring_window::NATIVE_V3_RING_SLOTS,
        );
        let bf16 = 2usize;
        let native_v3_geometry = self.target_layer_ids.len() == ring_window::NATIVE_V3_CAPTURE_TAPS
            && self.target_hidden_size == ring_window::NATIVE_V3_HIDDEN_SIZE;
        let ring_slots = self.ctx_window.clamp(1, ring_window::NATIVE_V3_RING_SLOTS);
        let layout: ring_window::AccumulatorLayout = ring_window::plan_accumulator_layout(
            self.max_seq_len,
            ring_slots,
            self.target_layer_ids.len(),
            self.target_hidden_size,
        )?;
        let initial_ring = ring_window::RingState::new(self.max_seq_len, layout.capacity)?;
        let ctx_slot_bytes = layout.slot_bytes;
        let total = layout.allocation_bytes;
        let ctx_fc_bytes = if self.skip_ctx_fc_cache {
            0
        } else {
            self.ctx_window
                .checked_mul(self.hidden_size)
                .and_then(|n| n.checked_mul(bf16))
                .ok_or_else(|| anyhow::anyhow!("DFlash ctx_fc_cache byte size overflow"))?
        };
        let kv_dim = self
            .num_kv_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| anyhow::anyhow!("DFlash kv_dim overflow"))?;
        let ctx_kv_bytes = self
            .ctx_window
            .checked_mul(kv_dim)
            .and_then(|n| n.checked_mul(bf16))
            .ok_or_else(|| anyhow::anyhow!("DFlash ctx_kv_cache byte size overflow"))?;
        let pos_pinned_bytes = self
            .ctx_window
            .checked_add(self.gamma)
            .and_then(|n| n.checked_add(1))
            .and_then(|n| n.checked_mul(size_of::<u32>()))
            .ok_or_else(|| anyhow::anyhow!("DFlash position staging size overflow"))?;
        let token_ids_pinned_bytes = pos_pinned_bytes;
        let draft_output_bytes = self
            .gamma
            .checked_mul(size_of::<u32>())
            .ok_or_else(|| anyhow::anyhow!("DFlash draft output staging size overflow"))?;
        let probabilistic = self.draft_sample_method
            == crate::dflash_sampling::DflashDraftSampleMethod::Probabilistic;
        let (selector_uniforms_bytes, selector_q_modifiers_bytes, selector_row_bytes) =
            if probabilistic {
                anyhow::ensure!(
                    self.selector.is_some()
                        && self.kernels.dflash2_selector_walk_probabilistic.0 != 0,
                    "probabilistic DFlash selector ABI is unavailable"
                );
                anyhow::ensure!(
                    self.markov.is_none(),
                    "probabilistic DFlash2 cannot allocate state with a DSpark Markov rewrite"
                );
                let uniforms_bytes = self
                    .gamma
                    .checked_mul(size_of::<f32>())
                    .ok_or_else(|| anyhow::anyhow!("DFlash sampled uniforms staging overflow"))?;
                let row_bytes = self
                    .gamma
                    .checked_mul(crate::dflash_sampling::DFLASH2_SELECTOR_TOP_K)
                    .and_then(|elements| elements.checked_mul(size_of::<u32>()))
                    .ok_or_else(|| anyhow::anyhow!("DFlash sampled row staging overflow"))?;
                (
                    uniforms_bytes,
                    selector_policy::SELECTOR_Q_MODIFIER_BYTES,
                    row_bytes,
                )
            } else {
                (0, 0, 0)
            };

        let geometry = DflashStateBufferGeometry {
            ctx_allocation_bytes: total,
            ctx_capacity: layout.capacity,
            ctx_slot_bytes,
            max_ctx_len: self.max_seq_len,
            ctx_fc_bytes,
            ctx_kv_bytes,
            num_layers: self.num_layers,
            pos_pinned_bytes,
            token_ids_pinned_bytes,
            draft_output_pinned_bytes: draft_output_bytes,
            probabilistic,
            selector_uniforms_bytes,
            selector_q_modifiers_bytes,
            selector_row_bytes,
        };

        let alloc_t0 = std::time::Instant::now();
        let mut allocations = DeviceAllocations::new(gpu);

        // Reuse an idle retired state matching exact buffer geometry (zero
        // cuMemAlloc_v2 / cuMemAllocHost_v2 on steady-state waves). Adopt every
        // device pointer into `DeviceAllocations` before zeroing so a failed
        // memset releases all buffers cleanly without leaking.
        if let Some(mut pooled) = self.take_pooled_state(&geometry) {
            let ds = pooled
                .as_any_mut()
                .downcast_mut::<DflashProposerState>()
                .ok_or_else(|| anyhow::anyhow!("Pooled state was not DflashProposerState"))?;
            for p in std::iter::once(ds.ctx_hidden_acc)
                .chain((!ds.ctx_fc_cache.is_null()).then_some(ds.ctx_fc_cache))
                .chain(ds.ctx_k_cache.iter().copied())
                .chain(ds.ctx_v_cache.iter().copied())
            {
                allocations.adopt(p);
            }
            let alloc_us = alloc_t0.elapsed().as_secs_f64() * 1e6;
            let memset_t0 = std::time::Instant::now();
            // Zero ctx_hidden_acc as the defensive cross-tenant baseline.
            gpu.memset(ds.ctx_hidden_acc, 0, total)?;
            let memset_us = memset_t0.elapsed().as_secs_f64() * 1e6;
            ds.reset_all_transient_state(Some(initial_ring), self.num_layers);
            if std::env::var("ATLAS_PREFILL_PHASE_PROFILE").ok().as_deref() == Some("1") {
                tracing::info!(
                    "DFLASH ALLOC_STATE | ctx_total_bytes={:.2}GB pool_hit=true alloc={:.1}ms memset={:.1}ms",
                    total as f64 / 1e9,
                    alloc_us / 1000.0,
                    memset_us / 1000.0,
                );
            }
            allocations.commit();
            return Ok(pooled);
        }

        let ctx_hidden_acc = allocations.alloc(total)?;
        let alloc_us = alloc_t0.elapsed().as_secs_f64() * 1e6;
        let memset_t0 = std::time::Instant::now();
        // Initialize to zero so stale data doesn't leak between sequences.
        gpu.memset(ctx_hidden_acc, 0, total)?;
        let memset_us = memset_t0.elapsed().as_secs_f64() * 1e6;
        if std::env::var("ATLAS_PREFILL_PHASE_PROFILE").ok().as_deref() == Some("1") {
            tracing::info!(
                "DFLASH ALLOC_STATE | ctx_total_bytes={:.2}GB pool_hit=false alloc={:.1}ms memset={:.1}ms",
                total as f64 / 1e9,
                alloc_us / 1000.0,
                memset_us / 1000.0,
            );
        }
        tracing::warn!(
            "DFlash bounded target-hidden ring: native_v3_geometry={} absolute_limit={} resident_slots={} resident_bytes={} active_attention_window={}. The drafter sees only the latest resident window; training and acceptance qualification must use this same bounded {}-slot contract.",
            native_v3_geometry,
            self.max_seq_len,
            layout.capacity,
            total,
            self.ctx_window,
            layout.capacity,
        );
        // Allocate per-sequence persistent context caches.
        // These eliminate O(seq_len) recomputation of fc_proj and k_proj/v_proj
        // for previously-seen context positions.
        let ctx_fc_cache = if ctx_fc_bytes > 0 {
            allocations.alloc(ctx_fc_bytes)?
        } else {
            DevicePtr::NULL
        };
        let mut ctx_k_cache = Vec::with_capacity(self.num_layers);
        let mut ctx_v_cache = Vec::with_capacity(self.num_layers);
        for _ in 0..self.num_layers {
            ctx_k_cache.push(allocations.alloc(ctx_kv_bytes)?);
            ctx_v_cache.push(allocations.alloc(ctx_kv_bytes)?);
        }
        let zeros = vec![0usize; self.num_layers];
        // Page-locked host buffer sized for the max position-id layout
        // (`ctx_window` context positions + up to γ+1 noise rows).
        let pos_pinned = gpu.alloc_host_pinned(pos_pinned_bytes)?;
        let token_ids_pinned = gpu.alloc_host_pinned(token_ids_pinned_bytes)?;
        let draft_output_pinned = gpu.alloc_host_pinned(draft_output_bytes)?;
        let (
            selector_uniforms_pinned,
            selector_q_modifiers_pinned,
            selector_candidates_pinned,
            selector_q_pinned,
        ) = if probabilistic {
            (
                Some(gpu.alloc_host_pinned(selector_uniforms_bytes)?),
                Some(gpu.alloc_host_pinned(selector_q_modifiers_bytes)?),
                Some(gpu.alloc_host_pinned(selector_row_bytes)?),
                Some(gpu.alloc_host_pinned(selector_row_bytes)?),
            )
        } else {
            (None, None, None, None)
        };

        let state = Box::new(DflashProposerState {
            last_num_drafted: 0,
            draft_budget: None,
            ctx_hidden_acc,
            ctx_len: initial_ring.absolute_len,
            ctx_resident_len: initial_ring.resident_len,
            ctx_valid_from_abs: initial_ring.ctx_valid_from_abs,
            ctx_capacity: layout.capacity,
            ctx_allocation_bytes: total,
            max_ctx_len: self.max_seq_len,
            ctx_slot_bytes,
            last_capture_idx: 0,
            last_num_accepted: 0,
            last_accepted_compact: Vec::new(),
            pld_tokens: Vec::new(),
            first_propose_done: false,
            pos_pinned,
            token_ids_pinned,
            draft_output_pinned,
            next_sampling_request: None,
            active_sampling_request: None,
            selector_uniforms_pinned,
            selector_q_modifiers_pinned,
            selector_candidates_pinned,
            selector_q_pinned,
            pending_sampled_proposal: None,
            retr_used_last: false,
            retr_misfire_streak: 0,
            retr_cooldown: 0,
            lookup_stats: lookup_stats::LookupStats::default(),
            ctx_fc_cache,
            ctx_k_cache,
            ctx_v_cache,
            cache_k_start: zeros.clone(),
            cache_k_end: zeros.clone(),
            cache_v_start: zeros.clone(),
            cache_v_end: zeros,
            cache_fc_start: 0,
            cache_fc_end: 0,
            ctx_kv_repr: None,
            pending_tree_payload: None,
            accept_history: [0u8; 8],
            accept_history_pos: 0,
            accept_history_count: 0,
            propose_steps: 0,
            throughput_router: throughput_router::ThroughputRouter::default(),
            climbdrop_router: throughput_router::ClimbDropRouter::default(),
            throughput_cycle_started: None,
            throughput_last_width: 0,
            fallback_suppressed_remaining: 0,
            recycle_tail: Vec::new(),
            recycle_key: 0,
            recycle_valid: false,
            recycle_last_offered: false,
            echo_tail: Vec::new(),
            echo_key: 0,
            echo_valid: false,
            echo_streak: 0,
            echo_offered_last: false,
            async_placeholder: false,
            pending_gumbel_spec: std::sync::Mutex::new(None),
            last_draft_probs: Vec::new(),
            pending_draft_probs_gamma: 0,
            pending_rejection_slot: std::sync::Mutex::new(None),
            last_rejection_q: std::sync::Mutex::new(None),
            pool_owner_head: self.pool_head_id(),
            ctx_fc_bytes,
            ctx_kv_bytes,
            target_is_moe: self.target_is_moe,
        });
        allocations.commit();
        Ok(state)
    }

    fn propose(
        &self,
        last_token: u32,
        target_hidden: spark_runtime::gpu::DevicePtr,
        position: usize,
        num_drafts: usize,
        state: &mut dyn ProposerState,
        ctx: &crate::layer::ForwardContext,
        stream: u64,
        draft_embed_target: Option<spark_runtime::gpu::DevicePtr>,
        grammar_bitmask: Option<&[i32]>,
        target_hidden_stack: Option<spark_runtime::gpu::DevicePtr>,
    ) -> Result<Vec<u32>> {
        {
            let dstate = state
                .as_any_mut()
                .downcast_mut::<DflashProposerState>()
                .ok_or_else(|| anyhow::anyhow!("Invalid DFlash proposer state"))?;
            anyhow::ensure!(
                dstate.active_sampling_request.is_none(),
                "DFlash sampling request leaked across proposal frames"
            );
            dstate.pending_sampled_proposal = None;
            dstate.last_draft_probs.clear();
            dstate.pending_draft_probs_gamma = 0;
            dstate.target_is_moe = self.target_is_moe;
            dstate.invalidate_rejection_q();
            if dstate.ctx_valid_from_abs > 0 && position <= dstate.ctx_valid_from_abs {
                self.resolve_async_inflight_impl(ctx.gpu, Some(dstate))?;
                dstate.next_sampling_request = None;
                *dstate
                    .pending_gumbel_spec
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
                let _ = dstate.take_pending_rejection_slot();
                draft_budget::clear_proposal_outputs(dstate);
                return Ok(Vec::new());
            }
            dstate.active_sampling_request = dstate.next_sampling_request.take();
            if self.draft_sample_method
                == crate::dflash_sampling::DflashDraftSampleMethod::Probabilistic
            {
                anyhow::ensure!(
                    dstate.active_sampling_request.is_some(),
                    "probabilistic DFlash proposal is missing its one-shot sampling request"
                );
            }
        }
        let result = self.propose_drafts(
            last_token,
            target_hidden,
            position,
            num_drafts,
            state,
            ctx,
            stream,
            draft_embed_target,
            grammar_bitmask,
            target_hidden_stack,
        );
        let dstate = state
            .as_any_mut()
            .downcast_mut::<DflashProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid DFlash proposer state"))?;
        // Temperature zero deliberately keeps the legacy greedy output
        // contract even when the proposer is configured as probabilistic.
        let requires_sampled_q = dstate
            .active_sampling_request
            .as_ref()
            .is_some_and(|request| request.request_sampler.temperature > 0.0);
        let probs_gamma = dstate.pending_draft_probs_gamma;
        dstate.pending_draft_probs_gamma = 0;
        dstate.active_sampling_request = None;
        *dstate
            .pending_gumbel_spec
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
        let _ = dstate.take_pending_rejection_slot();
        match result {
            Ok(drafts) => {
                let Some(budget) = dstate.draft_budget else {
                    dstate.last_draft_probs.clear();
                    dstate.invalidate_rejection_q();
                    draft_budget::clear_proposal_outputs(dstate);
                    anyhow::bail!("DFlash proposal returned without an outer draft budget")
                };
                let had_drafts = !drafts.is_empty();
                let finalized =
                    draft_budget::finalize_proposal(dstate, budget, drafts, requires_sampled_q);
                if had_drafts && finalized.is_empty() {
                    self.resolve_async_inflight_impl(ctx.gpu, Some(dstate))?;
                }
                if finalized.is_empty() {
                    dstate.last_draft_probs.clear();
                    dstate.invalidate_rejection_q();
                } else {
                    dstate.truncate_rejection_q(finalized.len());
                    if dstate.async_placeholder {
                        dstate.last_draft_probs.clear();
                        dstate.pending_draft_probs_gamma = probs_gamma;
                    } else if dstate.pending_tree_payload.is_none()
                        && !dstate.last_draft_probs.is_empty()
                        && dstate.last_draft_probs.len() >= finalized.len()
                        && dflash_pcum_probs_readback_enabled(self.target_is_moe)
                    {
                        // Already packed into `copy_d2h_pair_on_stream` alongside
                        // `draft_tokens_dev` in `forward_block.rs` (zero extra D2H sync).
                        dstate.last_draft_probs.truncate(finalized.len());
                    } else if dstate.pending_tree_payload.is_none()
                        && probs_gamma > 0
                        && finalized.len() <= probs_gamma
                        && !self.scratch.batch_out_probs.is_null()
                        && dflash_pcum_probs_readback_enabled(self.target_is_moe)
                    {
                        read_draft_probs_into(
                            ctx.gpu,
                            self.scratch.batch_out_probs,
                            finalized.len(),
                            Some(stream),
                            &mut dstate.last_draft_probs,
                        )?;
                    } else {
                        dstate.last_draft_probs.clear();
                    }
                }
                Ok(finalized)
            }
            Err(error) => {
                // Drain any in-flight pinned H2D transfers referencing `dstate`
                // before the caller can drop or reuse its pinned host buffers.
                let _ = ctx.gpu.synchronize(stream);
                dstate.last_draft_probs.clear();
                dstate.invalidate_rejection_q();
                draft_budget::clear_proposal_outputs(dstate);
                Err(error)
            }
        }
    }

    fn propose_batch(
        &self,
        last_tokens: &[u32],
        target_hiddens: &[DevicePtr],
        positions: &[usize],
        num_drafts: usize,
        states: &mut [&mut dyn ProposerState],
        ctx: &crate::layer::ForwardContext,
        stream: u64,
        target_hidden_stacks: &[Option<DevicePtr>],
    ) -> Result<Option<Vec<Vec<u32>>>> {
        self.propose_batch_dflash2(
            last_tokens,
            target_hiddens,
            positions,
            num_drafts,
            states,
            ctx,
            stream,
            target_hidden_stacks,
        )
    }

    fn after_verify(
        &self,
        num_accepted: usize,
        state: &mut dyn ProposerState,
        _stream: u64,
    ) -> Result<()> {
        let dstate = state
            .as_any_mut()
            .downcast_mut::<DflashProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid DFlash proposer state"))?;
        // `num_accepted` equals the index of the last accepted token in the
        // verify batch (0 = prefix token, 1 = first draft, etc.). The
        // multi-token `dflash_hidden_save` buffer holds hidden states for
        // every verify position; `last_capture_idx` selects the correct one.
        dstate.last_capture_idx = num_accepted;
        dstate.last_num_accepted = num_accepted;
        // Exactly one verify outcome per lookup pre-empt; independent of the
        // adaptive gate's own `retr_used_last` bookkeeping.
        dstate.lookup_stats.take_pending_verify(num_accepted);
        if let Some(started) = dstate.throughput_cycle_started.take() {
            let elapsed = started.elapsed().as_secs_f64();
            let alpha = std::env::var("ATLAS_DFLASH_TPS_ROUTER_ALPHA")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.30);
            let width = dstate.throughput_last_width;
            let climbdrop = std::env::var("ATLAS_DFLASH_TPS_ROUTER_MODE")
                .ok()
                .as_deref()
                == Some("climbdrop");
            if climbdrop {
                // Climb/drop mode: feed (drafts offered, drafts accepted).
                // `num_accepted` is the count of accepted drafts (0 = only
                // the prefix/bonus token matched). The controller needs the
                // number of drafts actually verified this cycle, which is
                // `last_num_drafted` when it was non-zero.
                let offered = if dstate.last_num_drafted > 0 {
                    dstate.last_num_drafted
                } else {
                    width
                };
                dstate
                    .climbdrop_router
                    .update(offered, num_accepted.min(offered));
            } else {
                dstate
                    .throughput_router
                    .observe(width, num_accepted + 1, elapsed, alpha);
            }
            if let Some(score) = dstate.throughput_router.score(width) {
                tracing::debug!(
                    "DFlash TPS router observe: width={} delivered={} elapsed_ms={:.3} ewma_tps={:.3}",
                    width,
                    num_accepted + 1,
                    elapsed * 1000.0,
                    score
                );
            }
        }
        // Clear any stale tree-fork path; the scheduler re-stamps it via
        // `set_dflash_accepted_compact` AFTER this when the step forked.
        dstate.last_accepted_compact.clear();
        dstate.last_num_drafted = 0;
        // Push accept count into the ring buffer for ATLAS_DFLASH_ADAPTIVE_GAMMA.
        // Saturating cast: num_accepted >= 256 cannot happen because dflash_kgamma
        // <= 64 in practice and the buffer width is u8.
        let slot = dstate.accept_history_pos % dstate.accept_history.len();
        dstate.accept_history[slot] = num_accepted.min(u8::MAX as usize) as u8;
        dstate.accept_history_pos = (dstate.accept_history_pos + 1) % dstate.accept_history.len();
        dstate.accept_history_count =
            (dstate.accept_history_count + 1).min(dstate.accept_history.len());
        Ok(())
    }

    /// DDTree M6: drain the tree payload stashed by `propose()` (if any).
    /// Returns + clears `dstate.pending_tree_payload`.
    fn take_pending_tree_payload(
        &self,
        state: &mut dyn ProposerState,
    ) -> Option<crate::layers::DDTreePayload> {
        let dstate = state.as_any_mut().downcast_mut::<DflashProposerState>()?;
        let payload = dstate.pending_tree_payload.take()?;
        let Some(budget) = dstate.draft_budget else {
            tracing::warn!("DFlash tree rejected without an outer draft budget");
            return None;
        };
        if let Err(error) = budget.validate_tree(&payload) {
            tracing::warn!("DFlash tree rejected before scheduler exposure: {error:#}");
            return None;
        }
        Some(payload)
    }

    fn free_state(&self, state: &mut dyn ProposerState) -> Result<()> {
        let dstate = state
            .as_any_mut()
            .downcast_mut::<DflashProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid DFlash proposer state"))?;
        dstate.last_draft_probs.clear();
        dstate.pending_draft_probs_gamma = 0;
        dstate.set_pending_rejection_slot(None);
        dstate.invalidate_rejection_q();
        draft_budget::clear_proposal_outputs(dstate);
        *dstate
            .pending_gumbel_spec
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
        // Host-side state is cleared here; per-sequence device allocations
        // (`ctx_hidden_acc`, `ctx_fc_cache`, `ctx_k_cache`, `ctx_v_cache`) are
        // pooled or freed with `&dyn GpuBackend` via
        // `reclaim_dflash_state_buffers` during sequence retirement.
        Ok(())
    }

    fn collect_async_drafts(
        &self,
        gpu: &dyn GpuBackend,
        state: &mut dyn ProposerState,
    ) -> Result<Option<Vec<u32>>> {
        let dstate = state
            .as_any_mut()
            .downcast_mut::<DflashProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid DFlash proposer state"))?;
        let result = self.collect_async_drafts_impl(gpu, dstate);
        match &result {
            Ok(Some(drafts)) if !drafts.is_empty() && dstate.pending_tree_payload.is_none() => {
                let probs_gamma = dstate.pending_draft_probs_gamma;
                dstate.pending_draft_probs_gamma = 0;
                if !dstate.last_draft_probs.is_empty() {
                    // Batched async collection already materialized `last_draft_probs`
                    // from the double-buffered slot alongside `drafts`.
                    dstate.last_draft_probs.truncate(drafts.len());
                } else if probs_gamma > 0
                    && drafts.len() <= probs_gamma
                    && !self.scratch.batch_out_probs.is_null()
                    && dflash_pcum_probs_readback_enabled(self.target_is_moe)
                {
                    read_draft_probs_into(
                        gpu,
                        self.scratch.batch_out_probs,
                        drafts.len(),
                        None,
                        &mut dstate.last_draft_probs,
                    )?;
                } else {
                    dstate.last_draft_probs.clear();
                }
            }
            Ok(Some(_)) | Err(_) => {
                dstate.pending_draft_probs_gamma = 0;
                dstate.last_draft_probs.clear();
            }
            Ok(None) => {}
        }
        result
    }

    fn resolve_async_inflight(
        &self,
        gpu: &dyn GpuBackend,
        state: Option<&mut dyn ProposerState>,
    ) -> Result<()> {
        let mut dstate = state.and_then(|s| s.as_any_mut().downcast_mut::<DflashProposerState>());
        if let Some(ds) = dstate.as_deref_mut() {
            ds.pending_draft_probs_gamma = 0;
            ds.last_draft_probs.clear();
            ds.invalidate_rejection_q();
        }
        self.resolve_async_inflight_impl(gpu, dstate)
    }

    fn arm_propose_overlap(&self, gpu: &dyn GpuBackend, default_stream: u64) -> Result<()> {
        BlockDiffusionDraftHead::arm_propose_overlap(self, gpu, default_stream)
    }

    fn launch_dflash_early_stage1(
        &self,
        gpu: &dyn GpuBackend,
        dflash_hidden_save: DevicePtr,
        num_rows: usize,
        start_pos: usize,
        capture_done_event: u64,
        state: &mut dyn ProposerState,
    ) -> Result<()> {
        let Some(dstate) = state.as_any_mut().downcast_mut::<DflashProposerState>() else {
            return Ok(());
        };
        self.launch_early_stage1_from_capture_event(
            gpu,
            dflash_hidden_save,
            num_rows,
            start_pos,
            capture_done_event,
            dstate,
        )
    }

    fn drain_dflash_early_stage1(&self, gpu: &dyn GpuBackend) -> Result<()> {
        self.drain_early_stage1_impl(gpu)
    }

    fn sync_target_config(
        &mut self,
        target_config: &atlas_core::config::ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        self.target_is_moe = Some(target_config.num_experts > 1);
        self.sync_target_qwen4_yarn(target_config, gpu)
    }

    fn bind_rejection_q_scratch(
        &mut self,
        q_ids: DevicePtr,
        q_probs: DevicePtr,
        row_cap: usize,
        q_stride: usize,
    ) {
        self.scratch.shared_q_ids = q_ids;
        self.scratch.shared_q_probs = q_probs;
        self.scratch.shared_q_row_cap = row_cap;
        self.scratch.shared_q_stride = q_stride;
    }

    fn release_rejection_resources(&self, gpu: &dyn GpuBackend) {
        noise_pass::clear_noise_graph_head_state(self as *const Self as usize, gpu);
        self.batched_noise_graph.lock().clear(gpu);
        drain_dflash_state_pool_for_head(self.pool_head_id(), gpu);
        if !self.scratch.q_row_bases.is_null()
            && !self
                .q_row_bases_released
                .swap(true, std::sync::atomic::Ordering::AcqRel)
            && let Err(error) = gpu.free(self.scratch.q_row_bases)
        {
            tracing::error!(
                "BlockDiffusionDraftHead::release_rejection_resources: free(q_row_bases) failed: {error:#}"
            );
        }
    }
}

#[cfg(test)]
mod draft_kv_storage_contract_tests {
    #[test]
    fn dead_paged_fp8_path_cannot_silently_return() {
        let head = include_str!("dflash_head.rs");
        let constructor = include_str!("dflash_head/from_weights.rs");
        for dead in [
            concat!("Paged", "KvCache"),
            concat!("KvCache", "Config"),
            concat!("reshape_cache", "_fp8"),
            concat!("prefill_attn_dflash", "_fp8"),
        ] {
            assert!(!head.contains(dead), "head reintroduced dead `{dead}`");
            assert!(
                !constructor.contains(dead),
                "constructor reintroduced dead `{dead}`"
            );
        }
    }

    #[test]
    fn active_context_path_remains_bf16_circular() {
        let head = include_str!("dflash_head.rs");
        let layer = include_str!("dflash_head/forward_block_layer.rs");
        for required in ["ctx_fc_cache", "ctx_k_cache", "ctx_v_cache"] {
            assert!(head.contains(required), "missing active `{required}` state");
        }
        assert!(layer.contains("cache_write_range("));
        assert!(layer.contains("ops::prefill_attention("));
        assert!(!layer.contains(concat!("prefill_attention", "_paged")));
    }

    #[test]
    fn bounded_hidden_ring_is_wired_through_all_production_boundaries() {
        let head = include_str!("dflash_head.rs");
        let prefill = include_str!("../model/impl_b3.rs");
        let propose = include_str!("dflash_head/propose.rs");
        let forward = include_str!("dflash_head/forward_block.rs");
        let lifecycle = include_str!("../model/trait_impl/sequence.rs");

        for required in [
            "plan_accumulator_layout(",
            "ctx_resident_len",
            "ctx_capacity",
            "ctx_allocation_bytes",
        ] {
            assert!(head.contains(required), "head is missing `{required}`");
        }
        // The per-chunk advance moved into RingState::advanced_after_chunk (f018dce11), which
        // calls plan_append_at itself; the prefill keeps its two direct capture sites.
        let ring_state = include_str!("dflash_head/ring_window_state.rs");
        assert!(prefill.matches(".plan_append_at(").count() >= 2);
        assert!(prefill.contains(".advanced_after_chunk("));
        assert!(ring_state.contains("Ok(Some(self.plan_append_at(absolute_start, rows)?.next))"));
        assert!(prefill.contains(".plan_gather("));
        assert!(propose.contains(".append_pending_ctx_captures("));
        assert!(
            head.contains("let append = ring.plan_append_at(ring.absolute_len, pending_rows)?")
        );
        assert!(head.contains(".physical_slot_for("));
        assert!(forward.matches(".plan_gather(").count() >= 2);
        assert!(forward.contains(".slot_for("));
        assert!(lifecycle.contains("ds.ctx_allocation_bytes"));

        for forbidden in [
            concat!("ctx_hidden_acc.offset(abs_pos", " * slot_bytes)"),
            concat!("base.offset(old_fc_end", " * ctx_slot_bytes)"),
            concat!("ds.max_ctx_len", " * ds.ctx_slot_bytes"),
        ] {
            assert!(
                !head.contains(forbidden)
                    && !prefill.contains(forbidden)
                    && !propose.contains(forbidden)
                    && !forward.contains(forbidden)
                    && !lifecycle.contains(forbidden),
                "absolute/nonresident ring access returned: `{forbidden}`"
            );
        }
    }
}

#[cfg(test)]
mod seq_buf_pool_tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use anyhow::Result;
    use parking_lot::Mutex;
    use spark_runtime::gpu::mock::MockGpuBackend;
    use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle, PinnedHostBuffer};
    use spark_runtime::sampler::gumbel::GumbelDraftSpec;

    use super::{
        BlockDiffusionDraftHead, DflashKernels, DflashProposerState, DflashQuantization,
        DflashScratch, batched_propose, noise_pass, pooled_state_count_for_head,
        reclaim_dflash_state_buffers_with_policy, resolve_seq_buf_pool_cap,
        seq_buf_pool_admits_under_mem_floor,
    };
    use crate::dflash_sampling::DflashDraftSampleMethod;
    use crate::speculative::DraftProposer;
    use crate::weight_loader::DrafterCheckpointFamily;
    use crate::weight_map::DenseWeight;

    struct CountingGpu {
        inner: MockGpuBackend,
        alloc_calls: AtomicUsize,
        free_calls: AtomicUsize,
        pinned_alloc_calls: AtomicUsize,
        fail_next_memset: AtomicBool,
    }

    impl CountingGpu {
        fn new() -> Self {
            Self {
                inner: MockGpuBackend::new(),
                alloc_calls: AtomicUsize::new(0),
                free_calls: AtomicUsize::new(0),
                pinned_alloc_calls: AtomicUsize::new(0),
                fail_next_memset: AtomicBool::new(false),
            }
        }

        fn alloc_calls(&self) -> usize {
            self.alloc_calls.load(Ordering::SeqCst)
        }

        fn free_calls(&self) -> usize {
            self.free_calls.load(Ordering::SeqCst)
        }

        fn pinned_alloc_calls(&self) -> usize {
            self.pinned_alloc_calls.load(Ordering::SeqCst)
        }
    }

    impl GpuBackend for CountingGpu {
        fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
            self.alloc_calls.fetch_add(1, Ordering::SeqCst);
            self.inner.alloc(bytes)
        }

        fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr> {
            self.alloc(bytes)
        }

        fn alloc_host_pinned(&self, bytes: usize) -> Result<PinnedHostBuffer> {
            self.pinned_alloc_calls.fetch_add(1, Ordering::SeqCst);
            self.inner.alloc_host_pinned(bytes)
        }

        fn free(&self, ptr: DevicePtr) -> Result<()> {
            if !ptr.is_null() {
                self.free_calls.fetch_add(1, Ordering::SeqCst);
            }
            self.inner.free(ptr)
        }

        fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> Result<()> {
            self.inner.copy_h2d(src, dst)
        }

        fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()> {
            self.inner.copy_d2h(src, dst)
        }

        fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
            self.inner.copy_d2d(src, dst, bytes)
        }

        fn launch(
            &self,
            func: KernelHandle,
            grid: [u32; 3],
            block: [u32; 3],
            shared_mem: u32,
            stream: u64,
            params: &mut [*mut std::ffi::c_void],
        ) -> Result<()> {
            self.inner
                .launch(func, grid, block, shared_mem, stream, params)
        }

        fn synchronize(&self, stream: u64) -> Result<()> {
            self.inner.synchronize(stream)
        }

        fn default_stream(&self) -> u64 {
            self.inner.default_stream()
        }

        fn kernel(&self, module: &str, func_name: &str) -> Result<KernelHandle> {
            self.inner.kernel(module, func_name)
        }

        fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()> {
            if self.fail_next_memset.swap(false, Ordering::SeqCst) {
                anyhow::bail!("injected memset failure");
            }
            self.inner.memset(ptr, value, bytes)
        }

        fn memset_async(&self, ptr: DevicePtr, value: u8, bytes: usize, stream: u64) -> Result<()> {
            self.inner.memset_async(ptr, value, bytes, stream)
        }

        fn total_memory(&self) -> Result<usize> {
            self.inner.total_memory()
        }

        fn free_memory(&self) -> Result<usize> {
            self.inner.free_memory()
        }
    }

    pub(super) fn test_draft_head(num_layers: usize) -> Box<BlockDiffusionDraftHead> {
        let k0 = KernelHandle(0);
        let p0 = DevicePtr::NULL;
        let w0 = DenseWeight { weight: p0 };
        let kernels = DflashKernels {
            rms_norm: k0,
            dense_gemv: k0,
            dense_gemm: k0,
            rope_qwen3: k0,
            silu_mul: k0,
            residual_add: k0,
            scaled_add: k0,
            token_recommit: k0,
            argmax: k0,
            argmax_rows: k0,
            dflash2_conv_prepare: k0,
            dflash2_conv_finish: k0,
            dflash2_conv_prepare_batched: k0,
            dflash2_conv_finish_batched: k0,
            dflash2_selector_walk: k0,
            dflash2_selector_walk_batched: k0,
            dflash2_selector_walk_gumbel_batched: k0,
            dflash2_selector_walk_gumbel_batched_k32: k0,
            dflash2_selector_walk_probabilistic: k0,
            dflash2_selector_walk_probabilistic_batched: k0,
            dflash2_selector_walk_ngram_batched: k0,
            split_ctx_h128_batched: k0,
            argmax_add: k0,
            topk: k0,
            topk_bf16_radix: k0,
            strided_copy_rows: k0,
            batched_embed: k0,
            prefill_attn: k0,
            w4a16_gemv: k0,
            w4a16_gemm: k0,
            w4a16_gemm_pipe: k0,
            w4a16_gemm_pipe_dual: k0,
            w4a16_gemm_t_m16: k0,
            w4a16_gemm_t_m32_n64: k0,
            w4a16_gemm_t_m32_n64_gateup_silu: k0,
            w4a16_gemm_t_m32_n32_gateup_silu_pipe_k64: k0,
            w4a16_gemm_t_m32_n64_splitk: k0,
            reduce_splitk_k: k0,
            fp8_gemm_t: k0,
            dense_gemv_fp8w: k0,
            fp8_gemm_dflash: k0,
        };
        let scratch = DflashScratch {
            stream_buf: p0,
            norm_buf: p0,
            q_buf: p0,
            k_buf: p0,
            v_buf: p0,
            attn_out: p0,
            mlp_intermediate: p0,
            mlp_up: p0,
            stream_acc: p0,
            fc_proj: p0,
            logits: p0,
            draft_tokens_dev: p0,
            draft_tokens_dev_alt: p0,
            position_ids: p0,
            topk_tokens_dev: p0,
            topk_logits_dev: p0,
            fc_norm_in: p0,
            fc_norm_ones_w: p0,
            markov_w1_row: p0,
            markov_bias: p0,
            markov_prev_dev: p0,
            conv_dyn: p0,
            conv_dyn1: p0,
            conv_out: p0,
            selector_hidden: p0,
            selector_uniforms: p0,
            selector_q_modifiers: p0,
            selector_q_probabilities: p0,
            selector_q_candidates: p0,
            splitk_ws: p0,
            batch_ctx_k_ptrs: p0,
            batch_ctx_v_ptrs: p0,
            batch_eff_ctx: p0,
            batch_ring_start: p0,
            batch_anchor_ids: p0,
            batch_gumbel_seeds: p0,
            batch_gumbel_step0_pos: p0,
            batch_gumbel_temps: p0,
            batch_gumbel_top_ks: p0,
            batch_gumbel_top_ps: p0,
            batch_gumbel_min_ps: p0,
            batch_out_probs: p0,
            batch_out_probs_alt: p0,
            early_stage1_fc: p0,
            q_row_bases: p0,
            shared_q_ids: p0,
            shared_q_probs: p0,
            shared_q_row_cap: 0,
            shared_q_stride: 0,
            ngram_tables_dev: p0,
            ngram_prev3_dev: p0,
        };
        Box::new(BlockDiffusionDraftHead {
            checkpoint_family: DrafterCheckpointFamily::Dflash,
            num_layers,
            hidden_size: 32,
            intermediate_size: 64,
            num_q_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            vocab_size: 128,
            draft_vocab_size: 128,
            gamma: 7,
            draft_sample_method: DflashDraftSampleMethod::Greedy,
            physical_verify_k: 8,
            mask_token_id: 0,
            window_size: Some(16),
            layer_window_sizes: Vec::new(),
            layer_causal: Vec::new(),
            target_layer_ids: vec![1, 2, 3, 4, 5],
            target_hidden_size: 32,
            target_vocab_size: 128,
            embed_tokens_shared: p0,
            lm_head_shared: p0,
            lm_head_shared_t: None,
            lm_head_shared_t_ldb: 0,
            lm_head_shared_nvfp4_exact: None,
            lm_head_exact_kernels: None,
            lm_head_shared_fp8: None,
            hidden_norm: w0,
            norm: w0,
            fc: w0,
            fc_nvfp4: None,
            fc_fp8: None,
            quantized_bf16_sources: Vec::new(),
            draft_id_to_target_id: None,
            markov: None,
            selector: None,
            dspine: None,
            layers: Vec::new(),
            w4a4_shadow: None,
            scratch,
            kernels,
            max_seq_len: 64,
            yarn_inv_freq: p0,
            rope_attention_factor: 1.0,
            rope_theta: 10_000_000.0,
            rotary_dim: 16,
            rms_norm_eps: 1e-6,
            ctx_window: 16,
            proposal_policy: std::sync::OnceLock::new(),
            quant: DflashQuantization::Bf16,
            async_inflight: Mutex::new(None),
            async_batched_inflight: Mutex::new(None),
            async_batch_slot_toggle: std::sync::atomic::AtomicU8::new(0),
            early_stage1_staged: Mutex::new(None),
            async_propose_stream: std::sync::OnceLock::new(),
            async_order_event: std::sync::atomic::AtomicU64::new(0),
            fused_event_armed: std::sync::atomic::AtomicBool::new(false),
            batched_noise_graph: Mutex::new(batched_propose::BatchedNoiseGraphCache::new()),
            q_row_bases_released: std::sync::atomic::AtomicBool::new(false),
            skip_ctx_fc_cache: false,
            target_is_moe: None,
        })
    }

    fn state_device_ptrs(
        state: &dyn crate::speculative::ProposerState,
    ) -> (DevicePtr, DevicePtr, Vec<DevicePtr>, Vec<DevicePtr>) {
        let ds = state
            .as_any()
            .downcast_ref::<DflashProposerState>()
            .expect("DflashProposerState");
        (
            ds.ctx_hidden_acc,
            ds.ctx_fc_cache,
            ds.ctx_k_cache.clone(),
            ds.ctx_v_cache.clone(),
        )
    }

    #[test]
    fn wave2_alloc_reclaim_alloc_performs_zero_gpu_allocs_and_frees() {
        let gpu = CountingGpu::new();
        let num_layers = 5usize;
        let expected_dev_allocs_per_seq = 1 + 1 + 2 * num_layers;
        let head = test_draft_head(num_layers);
        let head_id = head.pool_head_id();

        // Wave 1 (cold): allocate 2 concurrent sequences.
        let mut s0 = Some(head.alloc_state(&gpu).expect("wave1 s0"));
        let mut s1 = Some(head.alloc_state(&gpu).expect("wave1 s1"));
        assert_eq!(gpu.alloc_calls(), 2 * expected_dev_allocs_per_seq);
        assert_eq!(gpu.free_calls(), 0);
        assert_eq!(gpu.pinned_alloc_calls(), 6);
        assert_eq!(gpu.inner.alloc_count(), 2 * expected_dev_allocs_per_seq);

        let ptrs0 = state_device_ptrs(s0.as_deref().expect("s0 present"));
        let ptrs1 = state_device_ptrs(s1.as_deref().expect("s1 present"));

        // Retire both sequences into the bounded pool -> 0 frees.
        reclaim_dflash_state_buffers_with_policy(&gpu, &mut s0, 16, true);
        reclaim_dflash_state_buffers_with_policy(&gpu, &mut s1, 16, true);
        assert!(s0.is_none() && s1.is_none());
        assert_eq!(gpu.free_calls(), 0);
        assert_eq!(pooled_state_count_for_head(head_id), 2);

        // Wave 2 (warm): re-allocating both performs 0 device allocs, 0 frees,
        // and 0 pinned host allocs while reusing the exact LIFO pointer sets.
        let allocs_before = gpu.alloc_calls();
        let frees_before = gpu.free_calls();
        let pinned_before = gpu.pinned_alloc_calls();
        let mut r0 = Some(head.alloc_state(&gpu).expect("wave2 r0"));
        let mut r1 = Some(head.alloc_state(&gpu).expect("wave2 r1"));
        assert_eq!(gpu.alloc_calls(), allocs_before);
        assert_eq!(gpu.free_calls(), frees_before);
        assert_eq!(gpu.pinned_alloc_calls(), pinned_before);
        assert_eq!(pooled_state_count_for_head(head_id), 0);
        assert_eq!(state_device_ptrs(r0.as_deref().expect("r0")), ptrs1);
        assert_eq!(state_device_ptrs(r1.as_deref().expect("r1")), ptrs0);

        // Retire again and release on teardown -> 100% freed, 0 leaks.
        reclaim_dflash_state_buffers_with_policy(&gpu, &mut r0, 16, true);
        reclaim_dflash_state_buffers_with_policy(&gpu, &mut r1, 16, true);
        assert_eq!(gpu.free_calls(), 0);
        head.release_rejection_resources(&gpu);
        drop(head);
        assert_eq!(gpu.inner.alloc_count(), 0);
        assert_eq!(gpu.free_calls(), gpu.alloc_calls());
        assert_eq!(gpu.inner.invalid_free_count(), 0);
    }

    #[test]
    fn reused_state_resets_all_tenant_and_scalar_fields() {
        let gpu = CountingGpu::new();
        let num_layers = 5usize;
        let head = test_draft_head(num_layers);

        let mut s = Some(head.alloc_state(&gpu).expect("cold alloc"));
        let acc_ptr = {
            let ds = s
                .as_deref_mut()
                .expect("state present")
                .as_any_mut()
                .downcast_mut::<DflashProposerState>()
                .expect("DflashProposerState");
            ds.ctx_len = 32;
            ds.ctx_resident_len = 16;
            ds.ctx_valid_from_abs = 16;
            ds.cache_fc_start = 4;
            ds.cache_fc_end = 16;
            ds.cache_k_start.fill(4);
            ds.cache_k_end.fill(16);
            ds.cache_v_start.fill(4);
            ds.cache_v_end.fill(16);
            ds.ctx_kv_repr = Some(noise_pass::CtxKvCacheRepr::PostRope);
            ds.last_num_drafted = 7;
            ds.last_capture_idx = 5;
            ds.last_num_accepted = 5;
            ds.last_accepted_compact = vec![1, 2, 3];
            ds.pld_tokens = vec![42, 99, 1234];
            ds.first_propose_done = true;
            ds.pos_pinned.as_mut_slice().fill(0xAA);
            ds.token_ids_pinned.as_mut_slice().fill(0xBB);
            ds.draft_output_pinned.as_mut_slice().fill(0xCC);
            ds.retr_used_last = true;
            ds.retr_misfire_streak = 3;
            ds.retr_cooldown = 9;
            ds.accept_history = [1, 2, 3, 4, 5, 6, 7, 8];
            ds.accept_history_pos = 5;
            ds.accept_history_count = 8;
            ds.propose_steps = 17;
            ds.fallback_suppressed_remaining = 4;
            ds.recycle_tail = vec![7, 8];
            ds.recycle_key = 99;
            ds.recycle_valid = true;
            ds.recycle_last_offered = true;
            ds.echo_tail = vec![10, 11];
            ds.echo_key = 88;
            ds.echo_valid = true;
            ds.echo_streak = 2;
            ds.echo_offered_last = true;
            ds.async_placeholder = true;
            ds.last_draft_probs = vec![0.5, 0.25];
            ds.pending_draft_probs_gamma = 2;
            ds.set_pending_rejection_slot(Some(7));
            ds.set_last_rejection_q(Some((7, 1024, 5)));
            ds.set_gumbel_spec(Some(GumbelDraftSpec {
                seed: 0xDEAD_BEEF,
                temperature: 0.7,
                top_k: 20,
                top_p: 0.95,
                min_p: 0.05,
                step0_abs_pos: 100,
            }));
            gpu.copy_h2d(&vec![0xEEu8; ds.ctx_allocation_bytes], ds.ctx_hidden_acc)
                .expect("dirty ctx_hidden_acc");
            ds.ctx_hidden_acc
        };

        let allocs_before = gpu.alloc_calls();
        reclaim_dflash_state_buffers_with_policy(&gpu, &mut s, 4, true);
        let mut reused = Some(head.alloc_state(&gpu).expect("warm reuse"));
        assert_eq!(gpu.alloc_calls(), allocs_before);
        assert_eq!(gpu.free_calls(), 0);

        {
            let ds = reused
                .as_deref_mut()
                .expect("reused present")
                .as_any_mut()
                .downcast_mut::<DflashProposerState>()
                .expect("DflashProposerState");
            assert_eq!(ds.ctx_hidden_acc, acc_ptr);
            assert_eq!(ds.ctx_len, 0);
            assert_eq!(ds.ctx_resident_len, 0);
            assert_eq!(ds.ctx_valid_from_abs, 0);
            assert_eq!(ds.cache_fc_start, 0);
            assert_eq!(ds.cache_fc_end, 0);
            assert_eq!(ds.cache_k_start, vec![0usize; num_layers]);
            assert_eq!(ds.cache_k_end, vec![0usize; num_layers]);
            assert_eq!(ds.cache_v_start, vec![0usize; num_layers]);
            assert_eq!(ds.cache_v_end, vec![0usize; num_layers]);
            assert!(ds.ctx_kv_repr.is_none());
            assert_eq!(ds.last_num_drafted, 0);
            assert_eq!(ds.last_capture_idx, 0);
            assert_eq!(ds.last_num_accepted, 0);
            assert!(ds.last_accepted_compact.is_empty());
            assert!(ds.pld_tokens.is_empty());
            assert!(!ds.first_propose_done);
            assert!(ds.pos_pinned.as_mut_slice().iter().all(|&b| b == 0));
            assert!(ds.token_ids_pinned.as_mut_slice().iter().all(|&b| b == 0));
            assert!(
                ds.draft_output_pinned
                    .as_mut_slice()
                    .iter()
                    .all(|&b| b == 0)
            );
            assert!(!ds.retr_used_last);
            assert_eq!(ds.retr_misfire_streak, 0);
            assert_eq!(ds.retr_cooldown, 0);
            assert_eq!(ds.accept_history, [0u8; 8]);
            assert_eq!(ds.accept_history_pos, 0);
            assert_eq!(ds.accept_history_count, 0);
            assert_eq!(ds.propose_steps, 0);
            assert_eq!(ds.fallback_suppressed_remaining, 0);
            assert!(ds.recycle_tail.is_empty());
            assert_eq!(ds.recycle_key, 0);
            assert!(!ds.recycle_valid);
            assert!(!ds.recycle_last_offered);
            assert!(ds.echo_tail.is_empty());
            assert_eq!(ds.echo_key, 0);
            assert!(!ds.echo_valid);
            assert_eq!(ds.echo_streak, 0);
            assert!(!ds.echo_offered_last);
            assert!(!ds.async_placeholder);
            assert!(ds.last_draft_probs().is_empty());
            assert_eq!(ds.pending_draft_probs_gamma, 0);
            assert_eq!(ds.take_pending_rejection_slot(), None);
            assert_eq!(ds.last_rejection_q(), None);
            assert_eq!(ds.take_gumbel_spec_with(999_912_349, true), None);
            let acc_bytes = gpu
                .inner
                .read_alloc(ds.ctx_hidden_acc)
                .expect("read ctx_hidden_acc");
            assert!(acc_bytes.iter().all(|&b| b == 0));
        }

        reclaim_dflash_state_buffers_with_policy(&gpu, &mut reused, 4, true);
        head.release_rejection_resources(&gpu);
        drop(head);
        assert_eq!(gpu.inner.alloc_count(), 0);
        assert_eq!(gpu.free_calls(), gpu.alloc_calls());
        assert_eq!(gpu.inner.invalid_free_count(), 0);
    }

    #[test]
    fn pool_cap_disabled_mem_floor_and_teardown_free_every_device_pointer() {
        let gpu = CountingGpu::new();
        let num_layers = 5usize;
        let per_seq = 1 + 1 + 2 * num_layers;
        let head = test_draft_head(num_layers);
        let head_id = head.pool_head_id();

        // Allocate 4 states, reclaim with pool_cap=2 -> 2 pooled, 2 immediately freed.
        let mut states: Vec<_> = (0..4)
            .map(|_| Some(head.alloc_state(&gpu).expect("alloc")))
            .collect();
        assert_eq!(gpu.alloc_calls(), 4 * per_seq);
        for st in &mut states {
            reclaim_dflash_state_buffers_with_policy(&gpu, st, 2, true);
        }
        assert_eq!(pooled_state_count_for_head(head_id), 2);
        assert_eq!(gpu.free_calls(), 2 * per_seq);
        assert_eq!(gpu.inner.alloc_count(), 2 * per_seq);

        // UMA floor breach (`mem_floor_ok = false`) drains already-pooled idle
        // buffers AND frees the retiring buffer immediately.
        let mut s_floor = Some(head.alloc_state(&gpu).expect("pop for floor test"));
        assert_eq!(pooled_state_count_for_head(head_id), 1);
        reclaim_dflash_state_buffers_with_policy(&gpu, &mut s_floor, 16, false);
        assert_eq!(pooled_state_count_for_head(head_id), 0);
        assert_eq!(gpu.free_calls(), 4 * per_seq);
        assert_eq!(gpu.inner.alloc_count(), 0);

        // Disabled pool (`pool_cap = 0`, i.e. ATLAS_QWEN4_SEQ_BUF_POOL=0) frees immediately.
        let mut s_disabled = Some(head.alloc_state(&gpu).expect("alloc for disabled test"));
        assert_eq!(pooled_state_count_for_head(head_id), 0);
        reclaim_dflash_state_buffers_with_policy(&gpu, &mut s_disabled, 0, true);
        assert_eq!(pooled_state_count_for_head(head_id), 0);
        assert_eq!(gpu.free_calls(), 5 * per_seq);
        assert_eq!(gpu.inner.alloc_count(), 0);

        // Teardown via `release_rejection_resources` + `Drop` frees 100% of pooled
        // device pointers and is idempotent on repeat calls.
        let mut t0 = Some(head.alloc_state(&gpu).expect("teardown t0"));
        let mut t1 = Some(head.alloc_state(&gpu).expect("teardown t1"));
        reclaim_dflash_state_buffers_with_policy(&gpu, &mut t0, 16, true);
        reclaim_dflash_state_buffers_with_policy(&gpu, &mut t1, 16, true);
        assert_eq!(pooled_state_count_for_head(head_id), 2);
        assert_eq!(gpu.inner.alloc_count(), 2 * per_seq);

        head.release_rejection_resources(&gpu);
        assert_eq!(pooled_state_count_for_head(head_id), 0);
        assert_eq!(gpu.inner.alloc_count(), 0);
        assert_eq!(gpu.free_calls(), gpu.alloc_calls());
        assert_eq!(gpu.inner.invalid_free_count(), 0);

        // Second release call is a clean no-op with zero double-frees.
        head.release_rejection_resources(&gpu);
        drop(head);
        assert_eq!(gpu.inner.alloc_count(), 0);
        assert_eq!(gpu.free_calls(), gpu.alloc_calls());
        assert_eq!(gpu.inner.invalid_free_count(), 0);
    }

    #[test]
    fn mem_floor_breach_drains_idle_pooled_buffers_and_frees_retiring_state() {
        let gpu = CountingGpu::new();
        let num_layers = 5usize;
        let per_seq = 1 + 1 + 2 * num_layers;
        let head = test_draft_head(num_layers);
        let head_id = head.pool_head_id();

        let mut s0 = Some(head.alloc_state(&gpu).expect("alloc s0"));
        let mut s1 = Some(head.alloc_state(&gpu).expect("alloc s1"));
        let mut s2 = Some(head.alloc_state(&gpu).expect("alloc s2"));
        let mut s_active = Some(head.alloc_state(&gpu).expect("alloc s_active"));
        assert_eq!(gpu.alloc_calls(), 4 * per_seq);

        // Pool 3 idle buffers while 1 remains active.
        reclaim_dflash_state_buffers_with_policy(&gpu, &mut s0, 16, true);
        reclaim_dflash_state_buffers_with_policy(&gpu, &mut s1, 16, true);
        reclaim_dflash_state_buffers_with_policy(&gpu, &mut s2, 16, true);
        assert_eq!(pooled_state_count_for_head(head_id), 3);
        assert_eq!(gpu.free_calls(), 0);

        // Floor breach must drain all 3 idle pooled entries AND free s_active.
        reclaim_dflash_state_buffers_with_policy(&gpu, &mut s_active, 16, false);
        assert!(s_active.is_none());
        assert_eq!(pooled_state_count_for_head(head_id), 0);
        assert_eq!(gpu.free_calls(), 4 * per_seq);
        assert_eq!(gpu.inner.alloc_count(), 0);
        assert_eq!(gpu.inner.invalid_free_count(), 0);

        head.release_rejection_resources(&gpu);
        drop(head);
    }

    #[test]
    fn pool_hit_failed_memset_rolls_back_all_adopted_device_buffers() {
        let gpu = CountingGpu::new();
        let num_layers = 5usize;
        let head = test_draft_head(num_layers);
        let head_id = head.pool_head_id();

        let mut s = Some(head.alloc_state(&gpu).expect("cold alloc"));
        reclaim_dflash_state_buffers_with_policy(&gpu, &mut s, 4, true);
        assert_eq!(pooled_state_count_for_head(head_id), 1);

        gpu.fail_next_memset.store(true, Ordering::SeqCst);
        let err = head.alloc_state(&gpu);
        assert!(err.is_err(), "injected memset failure must surface");
        assert_eq!(pooled_state_count_for_head(head_id), 0);
        assert_eq!(gpu.inner.alloc_count(), 0);
        assert_eq!(gpu.free_calls(), gpu.alloc_calls());
        assert_eq!(gpu.inner.invalid_free_count(), 0);

        head.release_rejection_resources(&gpu);
        drop(head);
    }

    #[test]
    fn seq_buf_pool_policy_parsing_and_uma_floor_contract() {
        assert_eq!(resolve_seq_buf_pool_cap(None, 16), 16);
        assert_eq!(resolve_seq_buf_pool_cap(None, 32), 16);
        assert_eq!(resolve_seq_buf_pool_cap(None, 4), 4);
        assert_eq!(resolve_seq_buf_pool_cap(None, 0), 16);
        assert_eq!(resolve_seq_buf_pool_cap(Some("0"), 16), 0);
        assert_eq!(resolve_seq_buf_pool_cap(Some("false"), 16), 0);
        assert_eq!(resolve_seq_buf_pool_cap(Some("OFF"), 16), 0);
        assert_eq!(resolve_seq_buf_pool_cap(Some("1"), 16), 16);
        assert_eq!(resolve_seq_buf_pool_cap(Some("true"), 8), 8);
        assert_eq!(resolve_seq_buf_pool_cap(Some("on"), 32), 16);
        assert_eq!(resolve_seq_buf_pool_cap(Some("32"), 16), 32);
        assert_eq!(resolve_seq_buf_pool_cap(Some("64"), 16), 64);
        assert_eq!(resolve_seq_buf_pool_cap(Some("999"), 16), 64);
        assert_eq!(resolve_seq_buf_pool_cap(Some("invalid"), 16), 16);

        assert!(seq_buf_pool_admits_under_mem_floor(Some(8000), Some(6656)));
        assert!(seq_buf_pool_admits_under_mem_floor(Some(6656), Some(6656)));
        assert!(!seq_buf_pool_admits_under_mem_floor(Some(6655), Some(6656)));
        assert!(seq_buf_pool_admits_under_mem_floor(None, Some(6656)));
        assert!(seq_buf_pool_admits_under_mem_floor(Some(100), None));

        // Discrete GPU (`is_integrated == Some(false)`) checks VRAM against 1 GiB
        // instead of host `/proc/meminfo` vs 12 GiB.
        use super::seq_buf_pool_mem_floor_ok_for;
        const GIB: usize = 1024 * 1024 * 1024;
        assert!(seq_buf_pool_mem_floor_ok_for(
            Some(false),
            || Some(2 * GIB),
            || panic!("discrete GPU must not read host MemAvailable"),
            || panic!("discrete GPU must not read UMA floor"),
        ));
        assert!(seq_buf_pool_mem_floor_ok_for(
            Some(false),
            || Some(GIB),
            || panic!("discrete GPU must not read host MemAvailable"),
            || panic!("discrete GPU must not read UMA floor"),
        ));
        assert!(!seq_buf_pool_mem_floor_ok_for(
            Some(false),
            || Some(GIB - 1),
            || panic!("discrete GPU must not read host MemAvailable"),
            || panic!("discrete GPU must not read UMA floor"),
        ));

        // GB10 UMA (`Some(true)`) and unset unit tests (`None`) keep host MemAvailable vs 12 GiB.
        for topology in [Some(true), None] {
            assert!(seq_buf_pool_mem_floor_ok_for(
                topology,
                || panic!("UMA path must not query discrete VRAM"),
                || Some(16 * GIB),
                || Some(12 * GIB),
            ));
            assert!(!seq_buf_pool_mem_floor_ok_for(
                topology,
                || panic!("UMA path must not query discrete VRAM"),
                || Some(8 * GIB),
                || Some(12 * GIB),
            ));
        }
    }

    #[test]
    fn batched_skip_ctx_fc_cache_omits_fc_alloc_and_pools_cleanly() {
        let gpu = CountingGpu::new();
        let num_layers = 5usize;
        let expected_dev_allocs_per_seq = 1 + 2 * num_layers; // ctx_hidden_acc + K/V only (no ctx_fc_cache)
        let mut head = test_draft_head(num_layers);
        head.skip_ctx_fc_cache = true;
        let head_id = head.pool_head_id();

        let mut s0 = Some(head.alloc_state(&gpu).expect("alloc without ctx_fc_cache"));
        assert_eq!(gpu.alloc_calls(), expected_dev_allocs_per_seq);
        let (acc, fc, ks, vs) = state_device_ptrs(s0.as_deref().expect("s0"));
        assert!(!acc.is_null());
        assert!(fc.is_null(), "ctx_fc_cache must be NULL when skipped");
        assert_eq!(ks.len(), num_layers);
        assert_eq!(vs.len(), num_layers);

        // cache_write_range on NULL cache is a clean no-op returning (0, 0).
        let range = head
            .cache_write_range(&gpu, acc, 0, 8, DevicePtr::NULL, 0, 0, 16, 64, 0)
            .expect("NULL cache_write_range must succeed");
        assert_eq!(range, (0, 0));

        // Reclaim into pool -> reuse -> teardown with zero leaks / invalid frees.
        reclaim_dflash_state_buffers_with_policy(&gpu, &mut s0, 16, true);
        assert_eq!(pooled_state_count_for_head(head_id), 1);
        let allocs_before = gpu.alloc_calls();
        let mut reused = Some(head.alloc_state(&gpu).expect("pool reuse"));
        assert_eq!(gpu.alloc_calls(), allocs_before);
        let (acc2, fc2, _, _) = state_device_ptrs(reused.as_deref().expect("reused"));
        assert_eq!(acc2, acc);
        assert!(fc2.is_null());

        reclaim_dflash_state_buffers_with_policy(&gpu, &mut reused, 16, true);
        head.release_rejection_resources(&gpu);
        drop(head);
        assert_eq!(gpu.inner.alloc_count(), 0);
        assert_eq!(gpu.free_calls(), gpu.alloc_calls());
        assert_eq!(gpu.inner.invalid_free_count(), 0);
    }

    #[test]
    fn dflash_pcum_disabled_on_dense_targets_unless_explicitly_overridden() {
        use super::dflash_pcum_enabled_for_target;

        // Dense targets disable p_cum truncation + probs D2H by default.
        assert!(!dflash_pcum_enabled_for_target(Some(false), None));
        assert!(!dflash_pcum_enabled_for_target(Some(false), Some("")));
        assert!(!dflash_pcum_enabled_for_target(Some(false), Some("0")));
        assert!(!dflash_pcum_enabled_for_target(Some(false), Some("0.0")));
        // Explicit positive env override re-enables on dense targets.
        assert!(dflash_pcum_enabled_for_target(Some(false), Some("0.25")));

        // MoE targets and unspecified unit-test fixtures stay enabled by default.
        assert!(dflash_pcum_enabled_for_target(Some(true), None));
        assert!(dflash_pcum_enabled_for_target(None, None));
        // Explicit <= 0 disables everywhere (skips redundant D2H sync).
        assert!(!dflash_pcum_enabled_for_target(Some(true), Some("0")));
        assert!(!dflash_pcum_enabled_for_target(None, Some("0.0")));
        assert!(dflash_pcum_enabled_for_target(Some(true), Some("0.55")));
    }

    #[test]
    fn coalesce_ctx_capture_runs_merges_flat_splits_wrap_and_preserves_sparse_tree() {
        use super::coalesce_ctx_capture_runs;
        use super::ring_window::RingState;

        // 1. Flat non-wrapping append collapses 6 rows into 1 contiguous D2D run.
        let ring = RingState::from_lengths(16, 16, 4, 4).unwrap();
        let append = ring.plan_append_at(4, 6).unwrap();
        let flat_src: Vec<usize> = (0..6).collect();
        assert_eq!(
            coalesce_ctx_capture_runs(&flat_src, &append.write).unwrap(),
            vec![(0, 4, 6)]
        );

        // 2. Ring-wrapping append splits cleanly into exactly 2 contiguous D2D runs.
        let ring_wrap = RingState::from_lengths(64, 16, 14, 14).unwrap();
        let append_wrap = ring_wrap.plan_append_at(14, 5).unwrap();
        let wrap_src: Vec<usize> = (0..5).collect();
        assert_eq!(
            coalesce_ctx_capture_runs(&wrap_src, &append_wrap.write).unwrap(),
            vec![(0, 14, 2), (2, 0, 3)]
        );

        // 3. Sparse tree-fork accepts coalesce contiguous sub-runs while breaking on gaps.
        let sparse_src = vec![0, 1, 2, 5, 6];
        let append_sparse = ring.plan_append_at(4, 5).unwrap();
        assert_eq!(
            coalesce_ctx_capture_runs(&sparse_src, &append_sparse.write).unwrap(),
            vec![(0, 4, 3), (5, 7, 2)]
        );
    }
}
