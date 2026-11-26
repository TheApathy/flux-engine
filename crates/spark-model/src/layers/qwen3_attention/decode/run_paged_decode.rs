// SPDX-License-Identifier: AGPL-3.0-only

//! Split out of `super::super::decode.rs` for file-size budget.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kv_cache::{KvCacheDtype, PagedKvCache};
use spark_runtime::kv_dequant::{
    NVFP4_E2M1_LUT, TURBO4_LUT, dequant_4bit_block_to_bf16, dequant_fp8_to_bf16,
    dequant_turbo3_block_to_bf16, dequant_turbo8_block_to_bf16,
};

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

/// Cached env-var lookup for `ATLAS_PAGED_DECODE_SPLITK`. Read once on first
/// access; subsequent calls are a relaxed atomic load.
fn paged_decode_splitk_enabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static STATE: AtomicU8 = AtomicU8::new(2); // 2 = uninit, 0 = off, 1 = on
    let s = STATE.load(Ordering::Relaxed);
    if s != 2 {
        return s == 1;
    }
    let on = std::env::var("ATLAS_PAGED_DECODE_SPLITK")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    STATE.store(if on { 1 } else { 0 }, Ordering::Relaxed);
    on
}

/// Compute `num_splits` for paged-decode split-K.
///
/// Two regimes:
///   • Legacy (default, `ATLAS_PAGED_DECODE_SPLITK` unset / 0): fill SMs only
///     when `current_ctas < NUM_SMS`. Once num_q_heads * num_seqs ≥ NUM_SMS the
///     dispatch falls back to the single-CTA kernel, which serializes the
///     entire KV history through 8 warps. For K=3 verify on aeon-27b
///     (24×3 = 72 CTAs ≥ 48 SMs → num_splits = 1), each CTA scans the full KV
///     history; this is the production setting.
///   • Aggressive (`ATLAS_PAGED_DECODE_SPLITK=1`, EXPERIMENTAL): use the same
///     base split geometry as an independent K1 row, then additionally split
///     KV when `seq_len` is long. Empirically REGRESSES throughput on
///     FP8 KV at ctx=4K (measured 9 → 4 tok/s on aeon-27b K=3) because the
///     `paged_decode_attn_splitk_fp8` kernel processes one KV position at a
///     time (no BC=4 batching), while the non-splitk `paged_decode_attn_fp8`
///     kernel batches BC=4 K/V loads. Per-CTA throughput is ~4× lower in
///     split-K, and we add a reduce kernel launch on top. Each split processes
///     ~`SPLIT_TILE` KV positions; bounded by `MAX_SPLITS_CAP` so the workspace
///     allocated in `BufferSizes::splitk_workspace` always fits.
///
/// The aggressive regime is kept for completeness — it can be useful as a
/// scaffold for a future BC=4-aware split-K kernel that combines work-stealing
/// with batched loads — but the env var must remain OFF by default.
fn compute_num_splits_for_sms(
    num_q_heads: u32,
    num_seqs: u32,
    max_seq_len_host: u32,
    aggressive: bool,
    sm_count: u32,
) -> u32 {
    // Non-aggressive `BufferSizes::splitk_workspace` allocates a fixed 48-slot
    // (`sm121::NUM_SMS`) partial buffer. Cap legacy occupancy splits to that
    // slot envelope so high-SM discrete GPUs (e.g. 170-188 SMs) never overflow
    // the default workspace.
    const LEGACY_WORKSPACE_SLOTS: u32 = atlas_core::device::sm121::NUM_SMS;
    const SPLIT_TILE: u32 = 512;
    const MAX_SPLITS_CAP: u32 = 64;
    const MAX_SPLITS_CAP_GQA8: u32 = 16;

    let num_sms = sm_count.max(1);
    let legacy_budget = num_sms.min(LEGACY_WORKSPACE_SLOTS);
    let current_ctas = num_q_heads.saturating_mul(num_seqs);
    let legacy = if current_ctas == 0 || current_ctas >= legacy_budget {
        1u32
    } else {
        legacy_budget / current_ctas
    };

    if !aggressive {
        return legacy;
    }

    // The aggressive regime owns an extended [seq, head, split] arena with a
    // guaranteed minimum of two splits per row. Use independent K1 geometry
    // only when it fits that guarantee (Qwen3.8: 48 / 24 heads = 2). Smaller
    // head counts or higher SM counts that would exceed two splits per row
    // retain the bounded batch-derived occupancy.
    // Keep the legacy/default regime above byte-for-byte unchanged on GB10.
    let k1_legacy = if num_q_heads == 0 || num_q_heads >= num_sms {
        1u32
    } else {
        num_sms / num_q_heads
    };
    let occupancy_floor = if k1_legacy <= 2 { k1_legacy } else { legacy };
    // Cap n_splits <= 16 when `num_q_heads == 16` so MTP / 16-head decode at
    // 136K context does not launch 64 splits per head and bottleneck the reduce stage.
    let max_splits_cap = if num_q_heads == 16 {
        MAX_SPLITS_CAP_GQA8
    } else {
        MAX_SPLITS_CAP
    };
    let seq_target = max_seq_len_host
        .div_ceil(SPLIT_TILE)
        .clamp(1, max_splits_cap);
    seq_target.max(occupancy_floor).min(max_splits_cap)
}

fn compute_num_splits_for(
    num_q_heads: u32,
    num_seqs: u32,
    max_seq_len_host: u32,
    aggressive: bool,
) -> u32 {
    compute_num_splits_for_sms(
        num_q_heads,
        num_seqs,
        max_seq_len_host,
        aggressive,
        atlas_core::device::active_sm_count(),
    )
}

const fn flat_window_min_seq_len_host(num_seqs: u32, max_seq_len_host: u32) -> u32 {
    max_seq_len_host.saturating_sub(num_seqs.saturating_sub(1))
}

fn flat_window_crosses_k1_split_boundary_for_sms(
    num_q_heads: u32,
    num_seqs: u32,
    max_seq_len_host: u32,
    aggressive: bool,
    sm_count: u32,
) -> bool {
    if !aggressive || num_seqs <= 1 {
        return false;
    }
    let num_sms = sm_count.max(1);
    let k1_legacy = if num_q_heads == 0 || num_q_heads >= num_sms {
        1
    } else {
        num_sms / num_q_heads
    };
    if k1_legacy > 2 {
        return false;
    }
    let min_seq_len_host = flat_window_min_seq_len_host(num_seqs, max_seq_len_host);
    compute_num_splits_for_sms(num_q_heads, 1, min_seq_len_host, true, num_sms)
        != compute_num_splits_for_sms(num_q_heads, 1, max_seq_len_host, true, num_sms)
}

fn flat_window_crosses_k1_split_boundary_for(
    num_q_heads: u32,
    num_seqs: u32,
    max_seq_len_host: u32,
    aggressive: bool,
) -> bool {
    flat_window_crosses_k1_split_boundary_for_sms(
        num_q_heads,
        num_seqs,
        max_seq_len_host,
        aggressive,
        atlas_core::device::active_sm_count(),
    )
}

fn flat_window_crosses_k1_split_boundary(
    num_q_heads: u32,
    num_seqs: u32,
    max_seq_len_host: u32,
) -> bool {
    flat_window_crosses_k1_split_boundary_for(
        num_q_heads,
        num_seqs,
        max_seq_len_host,
        paged_decode_splitk_enabled(),
    )
}

fn compute_num_splits(num_q_heads: u32, num_seqs: u32, max_seq_len_host: u32) -> u32 {
    compute_num_splits_for(
        num_q_heads,
        num_seqs,
        max_seq_len_host,
        paged_decode_splitk_enabled(),
    )
}

const GQA4_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_ATTN_GQA";

fn gqa4_selected() -> bool {
    static SEL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SEL.get_or_init(|| std::env::var(GQA4_SELECTOR).ok().as_deref() == Some("1"))
}

fn gqa4_kernel(gpu: &dyn GpuBackend) -> Result<KernelHandle> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<HashMap<usize, Result<KernelHandle, String>>>> = OnceLock::new();
    let key = (gpu as *const dyn GpuBackend).cast::<()>() as usize;
    let map = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = map
        .lock()
        .map_err(|_| anyhow::anyhow!("{GQA4_SELECTOR}: kernel cache lock poisoned"))?;
    let entry = guard.entry(key).or_insert_with(|| {
        gpu.kernel("paged_decode_attn_gqa4", "paged_decode_attn_gqa4")
            .map_err(|e| e.to_string())
    });
    entry
        .clone()
        .map_err(|e| anyhow::anyhow!("{GQA4_SELECTOR}: kernel unavailable: {e}"))
}

const fn bf16_gqa4_geometry(
    num_seqs: u32,
    sliding: u32,
    head_dim: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
) -> bool {
    num_seqs > 1
        && sliding == 0
        && head_dim == 256
        && num_kv_heads != 0
        && num_q_heads.is_multiple_of(4)
        && num_q_heads.is_multiple_of(num_kv_heads)
        && (num_q_heads / num_kv_heads).is_multiple_of(4)
}

const GQA12_RAGGED_SELECTOR: &str = "ATLAS_PAGED_DECODE_RAGGED_GQA";
const GQA12_MAX_HPC: u32 = 12;
const GQA12_MAX_SEQS: u32 = 256;
const GQA12_MAX_PARTS: u32 = 64;
/// Wide-SM (`> 64` SMs) BF16 partition cap matching the 4-word ballot in
/// `paged_decode_attn_gqa4.cu` (`GQA12_MAX_PARTS = 128`). Lifts 2-KV-head
/// Qwen4 Flash-Next on 188-SM GPUs from 128 -> 256 CTAs while keeping GB10
/// (`<= 64` SMs) at 48 partitions (96 CTAs) and FP8 at 64.
const GQA12_MAX_PARTS_BF16_WIDE: u32 = 128;
const GQA12_WIDE_SM_THRESHOLD: u32 = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Gqa12RaggedMode {
    Auto,
    ForceOn,
    ForceOff,
}

fn gqa12_ragged_mode_from(raw: Option<&str>) -> Gqa12RaggedMode {
    match raw.map(str::trim) {
        Some("1") => Gqa12RaggedMode::ForceOn,
        Some("0") => Gqa12RaggedMode::ForceOff,
        Some(v) if v.eq_ignore_ascii_case("true") => Gqa12RaggedMode::ForceOn,
        Some(v) if v.eq_ignore_ascii_case("false") => Gqa12RaggedMode::ForceOff,
        _ => Gqa12RaggedMode::Auto,
    }
}

fn gqa12_ragged_mode() -> Gqa12RaggedMode {
    static MODE: std::sync::OnceLock<Gqa12RaggedMode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        gqa12_ragged_mode_from(std::env::var(GQA12_RAGGED_SELECTOR).ok().as_deref())
    })
}

const fn gqa12_ragged_eligible(
    mode: Gqa12RaggedMode,
    num_seqs: u32,
    head_dim: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    tree_or_indir: bool,
) -> bool {
    if tree_or_indir
        || head_dim != 256
        || num_kv_heads == 0
        || num_seqs == 0
        || num_seqs > GQA12_MAX_SEQS
        || !num_q_heads.is_multiple_of(num_kv_heads)
    {
        return false;
    }
    let gqa_ratio = num_q_heads / num_kv_heads;
    match mode {
        Gqa12RaggedMode::ForceOff => false,
        Gqa12RaggedMode::ForceOn => gqa_ratio >= 4,
        // Default-on at 8..=256 rows for >=8:1 GQA (12:1 Qwen3.8-Flash-Next and
        // 8:1 Qwen3.6-35B / Qwen3-Next-80B), where C=1 K-verify stays at <=5
        // rows so 8..=256 covers concurrent multi-request and batched verify.
        Gqa12RaggedMode::Auto => gqa_ratio >= 8 && num_seqs >= 8,
    }
}

#[inline]
#[must_use]
const fn gqa12_ragged_max_parts_for_kv_and_sm(kv_dtype: KvCacheDtype, sm_count: u32) -> u32 {
    if matches!(kv_dtype, KvCacheDtype::Bf16) && sm_count > GQA12_WIDE_SM_THRESHOLD {
        GQA12_MAX_PARTS_BF16_WIDE
    } else {
        GQA12_MAX_PARTS
    }
}

fn gqa12_ragged_grid_and_workspace_bytes_for_sms_with_cap(
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    sm_count: u32,
    max_parts: u32,
) -> Option<(u32, u32, usize)> {
    if num_kv_heads == 0 || !num_q_heads.is_multiple_of(num_kv_heads) {
        return None;
    }
    let gqa_ratio = num_q_heads / num_kv_heads;
    if gqa_ratio == 0 {
        return None;
    }
    let groups_per_kv = gqa_ratio.div_ceil(GQA12_MAX_HPC);
    let total_groups = num_kv_heads.checked_mul(groups_per_kv)?;
    let target_ctas = sm_count.max(1).saturating_mul(2);
    // Clamped to `max_parts` (64 for GB10 / FP8 2-word ballot, 128 for BF16
    // 4-word ballot on >64 SMs); caller verifies `workspace_cap_bytes >= workspace_bytes`.
    let num_partitions = (target_ctas / total_groups).clamp(1, max_parts);
    let slots_per_group = (num_partitions as usize).checked_mul(2)?;
    let stride_f32 = (head_dim as usize).checked_add(2)?;
    let workspace_bytes = (total_groups as usize)
        .checked_mul(slots_per_group)?
        .checked_mul(GQA12_MAX_HPC as usize)?
        .checked_mul(stride_f32)?
        .checked_mul(std::mem::size_of::<f32>())?;
    Some((num_partitions, total_groups, workspace_bytes))
}

#[cfg(test)]
fn gqa12_ragged_grid_and_workspace_bytes_for_sms(
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    sm_count: u32,
) -> Option<(u32, u32, usize)> {
    gqa12_ragged_grid_and_workspace_bytes_for_sms_with_cap(
        num_q_heads,
        num_kv_heads,
        head_dim,
        sm_count,
        GQA12_MAX_PARTS,
    )
}

fn gqa12_ragged_grid_and_workspace_bytes_for_kv(
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    kv_dtype: KvCacheDtype,
) -> Option<(u32, u32, usize)> {
    let sm_count = atlas_core::device::active_sm_count();
    let max_parts = gqa12_ragged_max_parts_for_kv_and_sm(kv_dtype, sm_count);
    gqa12_ragged_grid_and_workspace_bytes_for_sms_with_cap(
        num_q_heads,
        num_kv_heads,
        head_dim,
        sm_count,
        max_parts,
    )
}

#[cfg(test)]
fn gqa12_ragged_grid_and_workspace_bytes(
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
) -> Option<(u32, u32, usize)> {
    gqa12_ragged_grid_and_workspace_bytes_for_kv(
        num_q_heads,
        num_kv_heads,
        head_dim,
        KvCacheDtype::Bf16,
    )
}

impl Qwen3AttentionLayer {
    /// Resolve the lazily loaded BF16 GQA4 candidate before an enclosing model
    /// frame performs any request-owned work.
    pub(in super::super) fn preflight_bf16_gqa4_kernel(
        &self,
        gpu: &dyn GpuBackend,
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        num_seqs: u32,
    ) -> Result<()> {
        if bf16_gqa4_geometry(
            num_seqs,
            self.sliding_window.unwrap_or(0),
            head_dim,
            num_q_heads,
            num_kv_heads,
        ) && gqa4_selected()
        {
            gqa4_kernel(gpu)?;
        }
        Ok(())
    }

    pub(in crate::layers::qwen3_attention) fn flat_window_needs_k1_paged_rows(
        &self,
        num_q_heads: u32,
        num_seqs: u32,
        max_seq_len_host: u32,
    ) -> bool {
        flat_window_crosses_k1_split_boundary(num_q_heads, num_seqs, max_seq_len_host)
    }

    /// Attempt to dispatch the 12:1 GQA-grouped (`SK=32`) + Warp-0 ragged
    /// partitioner kernel pair when eligible and backed by sufficient scratch.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::layers::qwen3_attention) fn try_run_gqa12_ragged_paged_decode(
        &self,
        gpu: &dyn GpuBackend,
        q: DevicePtr,
        kv_cache: &PagedKvCache,
        output: DevicePtr,
        workspace: DevicePtr,
        workspace_cap_bytes: usize,
        block_table: DevicePtr,
        seq_lens: DevicePtr,
        max_blocks_per_seq: u32,
        num_seqs: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        block_size: u32,
        inv_sqrt_d: f32,
        q_stride: u32,
        tree_or_indir: bool,
        stream: u64,
    ) -> Result<bool> {
        let Some((main_k, red_k)) = self.paged_decode_gqa12_ragged_k else {
            return Ok(false);
        };
        if !gqa12_ragged_eligible(
            gqa12_ragged_mode(),
            num_seqs,
            head_dim,
            num_q_heads,
            num_kv_heads,
            tree_or_indir,
        ) {
            return Ok(false);
        }
        let Some((num_partitions, total_groups, needed_bytes)) =
            gqa12_ragged_grid_and_workspace_bytes_for_kv(
                num_q_heads,
                num_kv_heads,
                head_dim,
                self.kv_dtype,
            )
        else {
            return Ok(false);
        };
        if workspace == DevicePtr::NULL || workspace_cap_bytes < needed_bytes {
            return Ok(false);
        }

        let sliding = self.sliding_window.unwrap_or(0);
        let k_pool = kv_cache.k_pool_ptr(self.attn_layer_idx);
        let v_pool = kv_cache.v_pool_ptr(self.attn_layer_idx);

        match self.kv_dtype {
            KvCacheDtype::Bf16 => {
                static LOGGED_BF16: std::sync::Once = std::sync::Once::new();
                LOGGED_BF16.call_once(|| {
                    tracing::info!(
                        "ATTN_GQA12_RAGGED_BF16_ENGAGED rows={num_seqs} partitions={num_partitions} groups={total_groups}"
                    );
                });
                ops::paged_decode_attn_gqa12_ragged_bf16(
                    gpu,
                    main_k,
                    q,
                    k_pool,
                    v_pool,
                    output,
                    workspace,
                    block_table,
                    seq_lens,
                    max_blocks_per_seq,
                    num_seqs,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    block_size,
                    inv_sqrt_d,
                    q_stride,
                    sliding,
                    num_partitions,
                    total_groups,
                    stream,
                )?;
                ops::paged_decode_attn_gqa12_ragged_reduce(
                    gpu,
                    red_k,
                    workspace,
                    output,
                    seq_lens,
                    num_seqs,
                    num_q_heads,
                    num_kv_heads,
                    num_partitions,
                    sliding,
                    stream,
                )?;
                Ok(true)
            }
            KvCacheDtype::Fp8 => {
                static LOGGED_FP8: std::sync::Once = std::sync::Once::new();
                LOGGED_FP8.call_once(|| {
                    tracing::info!(
                        "ATTN_GQA12_RAGGED_FP8_ENGAGED rows={num_seqs} partitions={num_partitions} groups={total_groups}"
                    );
                });
                let (k_scale, v_scale) = self.effective_fp8_scales();
                ops::paged_decode_attn_gqa12_ragged_fp8(
                    gpu,
                    main_k,
                    q,
                    k_pool,
                    v_pool,
                    output,
                    workspace,
                    block_table,
                    seq_lens,
                    max_blocks_per_seq,
                    num_seqs,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    block_size,
                    inv_sqrt_d,
                    k_scale,
                    v_scale,
                    q_stride,
                    kv_cache.cache_stride() as u64,
                    sliding,
                    num_partitions,
                    total_groups,
                    stream,
                )?;
                ops::paged_decode_attn_gqa12_ragged_reduce(
                    gpu,
                    red_k,
                    workspace,
                    output,
                    seq_lens,
                    num_seqs,
                    num_q_heads,
                    num_kv_heads,
                    num_partitions,
                    sliding,
                    stream,
                )?;
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn run_paged_decode(
        &self,
        gpu: &dyn GpuBackend,
        q: DevicePtr,
        kv_cache: &PagedKvCache,
        output: DevicePtr,
        block_table: DevicePtr,
        seq_lens: DevicePtr,
        max_blocks_per_seq: u32,
        num_seqs: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        block_size: u32,
        inv_sqrt_d: f32,
        q_stride: u32,
        workspace: DevicePtr,
        kv_indirection: DevicePtr,
        kv_indir_base_ptr: DevicePtr,
        kv_indir_stride: u32,
        max_seq_len_host: u32,
        stream: u64,
    ) -> Result<()> {
        self.run_paged_decode_with_gate(
            gpu,
            q,
            kv_cache,
            output,
            block_table,
            seq_lens,
            max_blocks_per_seq,
            num_seqs,
            num_q_heads,
            num_kv_heads,
            head_dim,
            block_size,
            inv_sqrt_d,
            q_stride,
            workspace,
            kv_indirection,
            kv_indir_base_ptr,
            kv_indir_stride,
            max_seq_len_host,
            None,
            stream,
        )
        .map(|_| ())
    }

    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn run_paged_decode_with_gate(
        &self,
        gpu: &dyn GpuBackend,
        q: DevicePtr,
        kv_cache: &PagedKvCache,
        output: DevicePtr,
        block_table: DevicePtr,
        seq_lens: DevicePtr,
        max_blocks_per_seq: u32,
        num_seqs: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        block_size: u32,
        inv_sqrt_d: f32,
        q_stride: u32,
        workspace: DevicePtr,
        kv_indirection: DevicePtr,
        kv_indir_base_ptr: DevicePtr,
        kv_indir_stride: u32,
        max_seq_len_host: u32,
        gate: Option<(DevicePtr, u32)>,
        stream: u64,
    ) -> Result<bool> {
        let sliding = self.sliding_window.unwrap_or(0);
        if let Some((gate_ptr, gate_stride)) = gate
            && self.kv_dtype == KvCacheDtype::Bf16
            && kv_indirection == DevicePtr::NULL
            && head_dim == 256
            && self.paged_decode_bf16_gated_k.0 != 0
            && crate::layers::qwen4_qsa::qwen4_qsa_fused_gate_enabled()
            && !(bf16_gqa4_geometry(num_seqs, sliding, head_dim, num_q_heads, num_kv_heads)
                && gqa4_selected())
        {
            ops::paged_decode_attn_bf16_gated(
                gpu,
                self.paged_decode_bf16_gated_k,
                q,
                kv_cache.k_pool_ptr(self.attn_layer_idx),
                kv_cache.v_pool_ptr(self.attn_layer_idx),
                gate_ptr,
                output,
                block_table,
                seq_lens,
                max_blocks_per_seq,
                num_seqs,
                num_q_heads,
                num_kv_heads,
                head_dim,
                block_size,
                inv_sqrt_d,
                q_stride,
                gate_stride,
                sliding,
                stream,
            )?;
            return Ok(true);
        }
        self.run_paged_decode_ungated(
            gpu,
            q,
            kv_cache,
            output,
            block_table,
            seq_lens,
            max_blocks_per_seq,
            num_seqs,
            num_q_heads,
            num_kv_heads,
            head_dim,
            block_size,
            inv_sqrt_d,
            q_stride,
            workspace,
            kv_indirection,
            kv_indir_base_ptr,
            kv_indir_stride,
            max_seq_len_host,
            stream,
        )?;
        Ok(false)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_paged_decode_ungated(
        &self,
        gpu: &dyn GpuBackend,
        q: DevicePtr,
        kv_cache: &PagedKvCache,
        output: DevicePtr,
        block_table: DevicePtr,
        seq_lens: DevicePtr,
        max_blocks_per_seq: u32,
        num_seqs: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        block_size: u32,
        inv_sqrt_d: f32,
        q_stride: u32,
        workspace: DevicePtr,
        kv_indirection: DevicePtr,
        kv_indir_base_ptr: DevicePtr,
        kv_indir_stride: u32,
        max_seq_len_host: u32,
        stream: u64,
    ) -> Result<()> {
        match self.kv_dtype {
            KvCacheDtype::Nvfp4 => {
                let num_splits = compute_num_splits(num_q_heads, num_seqs, max_seq_len_host);

                if num_splits > 1 && head_dim <= 256 {
                    let splitk_k = self
                        .paged_decode_splitk_k
                        .expect("split-K kernel required for NVFP4");
                    let reduce_k = self
                        .paged_decode_reduce_k
                        .expect("reduce kernel required for NVFP4");
                    ops::paged_decode_attn_splitk_nvfp4(
                        gpu,
                        splitk_k,
                        q,
                        kv_cache.k_pool_ptr(self.attn_layer_idx),
                        kv_cache.v_pool_ptr(self.attn_layer_idx),
                        workspace,
                        block_table,
                        seq_lens,
                        max_blocks_per_seq,
                        num_q_heads,
                        num_kv_heads,
                        head_dim,
                        block_size,
                        inv_sqrt_d,
                        num_splits,
                        q_stride,
                        kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                        kv_cache.nvfp4_data_bytes() as u64,
                        num_seqs,
                        kv_indirection,
                        kv_indir_base_ptr,
                        kv_indir_stride,
                        stream,
                    )?;
                    ops::paged_decode_attn_reduce_nvfp4(
                        gpu,
                        reduce_k,
                        workspace,
                        output,
                        seq_lens,
                        num_q_heads,
                        head_dim,
                        num_splits,
                        num_seqs,
                        stream,
                    )
                } else {
                    ops::paged_decode_attn_nvfp4(
                        gpu,
                        self.paged_decode_k,
                        q,
                        kv_cache.k_pool_ptr(self.attn_layer_idx),
                        kv_cache.v_pool_ptr(self.attn_layer_idx),
                        output,
                        block_table,
                        seq_lens,
                        max_blocks_per_seq,
                        num_seqs,
                        num_q_heads,
                        num_kv_heads,
                        head_dim,
                        block_size,
                        inv_sqrt_d,
                        q_stride,
                        kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                        kv_cache.nvfp4_data_bytes() as u64,
                        kv_indirection,
                        kv_indir_base_ptr,
                        kv_indir_stride,
                        stream,
                    )
                }
            }
            // Turbo4/3: same 4-bit interface as NVFP4 (block_stride + data_section layout).
            KvCacheDtype::Turbo4 | KvCacheDtype::Turbo3 | KvCacheDtype::Turbo2 => {
                let kernel = if head_dim > 256 && self.paged_decode_512_k.0 != 0 {
                    self.paged_decode_512_k
                } else {
                    self.paged_decode_k
                };
                let data_bytes = match self.kv_dtype {
                    KvCacheDtype::Turbo3 => kv_cache.turbo3_data_bytes() as u64,
                    KvCacheDtype::Turbo2 => kv_cache.turbo2_data_bytes() as u64,
                    _ => kv_cache.turbo4_data_bytes() as u64,
                };
                ops::paged_decode_attn_nvfp4(
                    gpu,
                    kernel,
                    q,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    output,
                    block_table,
                    seq_lens,
                    max_blocks_per_seq,
                    num_seqs,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    block_size,
                    inv_sqrt_d,
                    q_stride,
                    kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                    data_bytes,
                    kv_indirection,
                    kv_indir_base_ptr,
                    kv_indir_stride,
                    stream,
                )
            }
            // Turbo8: WHT + FP8 — 1 byte per element + per-group BF16 scales.
            KvCacheDtype::Turbo8 => {
                let kernel = if head_dim > 256 && self.paged_decode_512_k.0 != 0 {
                    self.paged_decode_512_k
                } else {
                    self.paged_decode_k
                };
                ops::paged_decode_attn_nvfp4(
                    gpu,
                    kernel,
                    q,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    output,
                    block_table,
                    seq_lens,
                    max_blocks_per_seq,
                    num_seqs,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    block_size,
                    inv_sqrt_d,
                    q_stride,
                    kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                    kv_cache.turbo8_data_bytes() as u64,
                    kv_indirection,
                    kv_indir_base_ptr,
                    kv_indir_stride,
                    stream,
                )
            }
            KvCacheDtype::Bf16KTurbo3V => {
                // TurboQuant+ safer-asym Bf16K + Turbo3V combined paged decode.
                // K read as BF16 NHD (vector loads), V read as turbo3 (3-bit
                // packed + FP8 group scale, sparse-V threshold on batched +
                // remainder paths). Single combined kernel per HDIM variant.
                let sliding = self.sliding_window.unwrap_or(0);
                ops::paged_decode_attn_bf16k_turbo3v(
                    gpu,
                    self.paged_decode_k,
                    q,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    output,
                    block_table,
                    seq_lens,
                    max_blocks_per_seq,
                    num_seqs,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    block_size,
                    inv_sqrt_d,
                    q_stride,
                    kv_cache.v_block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                    kv_cache.turbo3_data_bytes() as u64,
                    sliding,
                    stream,
                )
            }
            KvCacheDtype::Bf16KTurbo4V => {
                // TurboQuant+ safer-asym Bf16K + Turbo4V combined paged decode.
                // K read as BF16 NHD, V read as turbo4 (4-bit packed + FP8
                // group scale, sparse-V threshold on batched + remainder paths).
                let sliding = self.sliding_window.unwrap_or(0);
                ops::paged_decode_attn_bf16k_turbo4v(
                    gpu,
                    self.paged_decode_k,
                    q,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    output,
                    block_table,
                    seq_lens,
                    max_blocks_per_seq,
                    num_seqs,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    block_size,
                    inv_sqrt_d,
                    q_stride,
                    kv_cache.v_block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                    kv_cache.nvfp4_data_bytes() as u64,
                    sliding,
                    stream,
                )
            }
            KvCacheDtype::Bf16KTurbo2V => {
                // TurboQuant+ safer-asym Bf16K + Turbo2V (6.4x V compression)
                // combined paged decode. K read as BF16 NHD, V read as turbo2
                // (2-bit packed + FP8 group scale, sparse-V threshold).
                let sliding = self.sliding_window.unwrap_or(0);
                ops::paged_decode_attn_bf16k_turbo2v(
                    gpu,
                    self.paged_decode_k,
                    q,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    output,
                    block_table,
                    seq_lens,
                    max_blocks_per_seq,
                    num_seqs,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    block_size,
                    inv_sqrt_d,
                    q_stride,
                    kv_cache.v_block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                    kv_cache.turbo2_data_bytes() as u64,
                    sliding,
                    stream,
                )
            }
            KvCacheDtype::Turbo4KTurbo3V
            | KvCacheDtype::Turbo4KTurbo8V
            | KvCacheDtype::Turbo3KTurbo8V => {
                // TurboQuant+ both-sides asym: K and V both turbo. Pass per-side
                // (block_stride, data_section) pairs since K and V pools have
                // independent byte layouts.
                let sliding = self.sliding_window.unwrap_or(0);
                let k_block_stride =
                    kv_cache.k_block_stride_bytes_for_layer(self.attn_layer_idx) as u64;
                let v_block_stride =
                    kv_cache.v_block_stride_bytes_for_layer(self.attn_layer_idx) as u64;
                let k_pool = kv_cache.k_pool_ptr(self.attn_layer_idx);
                let v_pool = kv_cache.v_pool_ptr(self.attn_layer_idx);
                match self.kv_dtype {
                    KvCacheDtype::Turbo4KTurbo3V => ops::paged_decode_attn_turbo4k_turbo3v(
                        gpu,
                        self.paged_decode_k,
                        q,
                        k_pool,
                        v_pool,
                        output,
                        block_table,
                        seq_lens,
                        max_blocks_per_seq,
                        num_seqs,
                        num_q_heads,
                        num_kv_heads,
                        head_dim,
                        block_size,
                        inv_sqrt_d,
                        q_stride,
                        k_block_stride,
                        kv_cache.nvfp4_data_bytes() as u64,
                        v_block_stride,
                        kv_cache.turbo3_data_bytes() as u64,
                        sliding,
                        stream,
                    ),
                    KvCacheDtype::Turbo4KTurbo8V => ops::paged_decode_attn_turbo4k_turbo8v(
                        gpu,
                        self.paged_decode_k,
                        q,
                        k_pool,
                        v_pool,
                        output,
                        block_table,
                        seq_lens,
                        max_blocks_per_seq,
                        num_seqs,
                        num_q_heads,
                        num_kv_heads,
                        head_dim,
                        block_size,
                        inv_sqrt_d,
                        q_stride,
                        k_block_stride,
                        kv_cache.nvfp4_data_bytes() as u64,
                        v_block_stride,
                        kv_cache.turbo8_data_bytes() as u64,
                        sliding,
                        stream,
                    ),
                    KvCacheDtype::Turbo3KTurbo8V => ops::paged_decode_attn_turbo3k_turbo8v(
                        gpu,
                        self.paged_decode_k,
                        q,
                        k_pool,
                        v_pool,
                        output,
                        block_table,
                        seq_lens,
                        max_blocks_per_seq,
                        num_seqs,
                        num_q_heads,
                        num_kv_heads,
                        head_dim,
                        block_size,
                        inv_sqrt_d,
                        q_stride,
                        k_block_stride,
                        kv_cache.turbo3_data_bytes() as u64,
                        v_block_stride,
                        kv_cache.turbo8_data_bytes() as u64,
                        sliding,
                        stream,
                    ),
                    _ => unreachable!(),
                }
            }
            KvCacheDtype::Fp8KTurbo3V | KvCacheDtype::Fp8KTurbo4V | KvCacheDtype::Fp8KTurbo2V => {
                // TurboQuant+ asym for FP8 models: K=fp8 (per-tensor scale),
                // V=turbo{3,4,2} with sparse-V threshold on batched + remainder.
                let sliding = self.sliding_window.unwrap_or(0);
                let (k_scale, _) = self.effective_fp8_scales();
                let v_block_stride =
                    kv_cache.v_block_stride_bytes_for_layer(self.attn_layer_idx) as u64;
                let k_pool = kv_cache.k_pool_ptr(self.attn_layer_idx);
                let v_pool = kv_cache.v_pool_ptr(self.attn_layer_idx);
                match self.kv_dtype {
                    KvCacheDtype::Fp8KTurbo3V => ops::paged_decode_attn_fp8k_turbo3v(
                        gpu,
                        self.paged_decode_k,
                        q,
                        k_pool,
                        v_pool,
                        output,
                        block_table,
                        seq_lens,
                        max_blocks_per_seq,
                        num_seqs,
                        num_q_heads,
                        num_kv_heads,
                        head_dim,
                        block_size,
                        inv_sqrt_d,
                        k_scale,
                        q_stride,
                        v_block_stride,
                        kv_cache.turbo3_data_bytes() as u64,
                        sliding,
                        stream,
                    ),
                    KvCacheDtype::Fp8KTurbo4V => ops::paged_decode_attn_fp8k_turbo4v(
                        gpu,
                        self.paged_decode_k,
                        q,
                        k_pool,
                        v_pool,
                        output,
                        block_table,
                        seq_lens,
                        max_blocks_per_seq,
                        num_seqs,
                        num_q_heads,
                        num_kv_heads,
                        head_dim,
                        block_size,
                        inv_sqrt_d,
                        k_scale,
                        q_stride,
                        v_block_stride,
                        kv_cache.nvfp4_data_bytes() as u64,
                        sliding,
                        stream,
                    ),
                    KvCacheDtype::Fp8KTurbo2V => ops::paged_decode_attn_fp8k_turbo2v(
                        gpu,
                        self.paged_decode_k,
                        q,
                        k_pool,
                        v_pool,
                        output,
                        block_table,
                        seq_lens,
                        max_blocks_per_seq,
                        num_seqs,
                        num_q_heads,
                        num_kv_heads,
                        head_dim,
                        block_size,
                        inv_sqrt_d,
                        k_scale,
                        q_stride,
                        v_block_stride,
                        kv_cache.turbo2_data_bytes() as u64,
                        sliding,
                        stream,
                    ),
                    _ => unreachable!(),
                }
            }
            KvCacheDtype::Bf16 => {
                // Tree verify uses a Qwen-only ABI. Keep the shared BF16
                // symbol byte-for-byte compatible with Gemma-4, whose final
                // argument is `sliding_window` rather than tree metadata.
                if kv_indirection != DevicePtr::NULL {
                    anyhow::ensure!(
                        self.paged_decode_bf16_qwen_tree_k.0 != 0,
                        "BF16 DDTree indirection requested without the dedicated Qwen kernel"
                    );
                    anyhow::ensure!(
                        kv_indir_base_ptr != DevicePtr::NULL && kv_indir_stride > 0,
                        "BF16 DDTree indirection metadata is incomplete"
                    );
                    return ops::paged_decode_attn_bf16_qwen_tree(
                        gpu,
                        self.paged_decode_bf16_qwen_tree_k,
                        q,
                        kv_cache.k_pool_ptr(self.attn_layer_idx),
                        kv_cache.v_pool_ptr(self.attn_layer_idx),
                        output,
                        block_table,
                        seq_lens,
                        max_blocks_per_seq,
                        num_seqs,
                        num_q_heads,
                        num_kv_heads,
                        head_dim,
                        block_size,
                        inv_sqrt_d,
                        q_stride,
                        kv_indirection,
                        kv_indir_base_ptr,
                        kv_indir_stride,
                        stream,
                    );
                }

                // Legacy BF16 decode — no Split-K. Gemma-4 uses the HDIM=512
                // variant and retains its unchanged sliding-window ABI.
                let kernel = if head_dim > 256 && self.paged_decode_512_k.0 != 0 {
                    self.paged_decode_512_k
                } else {
                    self.paged_decode_k
                };
                // Gemma-4 sliding layers attend only to the last `window_size`
                // KV positions; full layers (and all non-Gemma-4 models) pass 0.
                let sliding = self.sliding_window.unwrap_or(0);
                // Default-off GQA-fused twin (ATLAS_QWEN4_PREFILL_ATTN_GQA=1):
                // bit-identical per-head math, K/V loads shared by 4 heads.
                if bf16_gqa4_geometry(num_seqs, sliding, head_dim, num_q_heads, num_kv_heads)
                    && gqa4_selected()
                {
                    let kernel = gqa4_kernel(gpu)?;
                    static LOGGED: std::sync::Once = std::sync::Once::new();
                    LOGGED.call_once(|| {
                        tracing::info!(
                            "ATTN_PREFILL_GQA4_ENGAGED selector={GQA4_SELECTOR} rows={num_seqs}"
                        );
                    });
                    let k_pool = kv_cache.k_pool_ptr(self.attn_layer_idx);
                    let v_pool = kv_cache.v_pool_ptr(self.attn_layer_idx);
                    let result = ops::paged_decode_attn_bf16_gqa4(
                        gpu,
                        kernel,
                        q,
                        k_pool,
                        v_pool,
                        output,
                        block_table,
                        seq_lens,
                        max_blocks_per_seq,
                        num_seqs,
                        num_q_heads,
                        num_kv_heads,
                        head_dim,
                        block_size,
                        inv_sqrt_d,
                        q_stride,
                        stream,
                    );
                    // Env-gated oracle capture (ATLAS_QWEN4_ORACLE_DUMP): one
                    // recording of the attention core's arguments and result.
                    // Off by default; nothing runs when the variable is unset.
                    if result.is_ok()
                        && let Some(mut sink) =
                            crate::layers::qwen4_oracle::Sink::open("attn_core_gqa4")
                    {
                        gpu.synchronize(stream)?;
                        let kv_elems = max_blocks_per_seq as usize
                            * block_size as usize
                            * num_kv_heads as usize
                            * head_dim as usize;
                        sink.scalar("num_seqs", num_seqs);
                        sink.scalar("num_q_heads", num_q_heads);
                        sink.scalar("num_kv_heads", num_kv_heads);
                        sink.scalar("head_dim", head_dim);
                        sink.scalar("block_size", block_size);
                        sink.scalar("max_blocks_per_seq", max_blocks_per_seq);
                        sink.scalar("q_stride", q_stride);
                        sink.scalar("inv_sqrt_d", format!("{inv_sqrt_d:e}"));
                        sink.buf(gpu, "q_bf16", q, num_seqs as usize * q_stride as usize * 2)?;
                        sink.buf(gpu, "k_pool_bf16", k_pool, kv_elems * 2)?;
                        sink.buf(gpu, "v_pool_bf16", v_pool, kv_elems * 2)?;
                        sink.buf(
                            gpu,
                            "block_table_u32",
                            block_table,
                            num_seqs as usize * max_blocks_per_seq as usize * 4,
                        )?;
                        sink.buf(gpu, "seq_lens_u32", seq_lens, num_seqs as usize * 4)?;
                        sink.buf(
                            gpu,
                            "out_bf16",
                            output,
                            num_seqs as usize * num_q_heads as usize * head_dim as usize * 2,
                        )?;
                        sink.finish();
                    }
                    return result;
                }
                ops::paged_decode_attn_bf16(
                    gpu,
                    kernel,
                    q,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    output,
                    block_table,
                    seq_lens,
                    max_blocks_per_seq,
                    num_seqs,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    block_size,
                    inv_sqrt_d,
                    q_stride,
                    sliding,
                    stream,
                )
            }
            _ => {
                // FP8 paged decode
                let num_splits = compute_num_splits(num_q_heads, num_seqs, max_seq_len_host);

                let (k_scale, v_scale) = self.effective_fp8_scales();

                if num_splits > 1 && head_dim <= 256 {
                    let splitk_k = self
                        .paged_decode_splitk_k
                        .expect("split-K kernel required for FP8");
                    let reduce_k = self
                        .paged_decode_reduce_k
                        .expect("reduce kernel required for FP8");
                    ops::paged_decode_attn_splitk_fp8(
                        gpu,
                        splitk_k,
                        q,
                        kv_cache.k_pool_ptr(self.attn_layer_idx),
                        kv_cache.v_pool_ptr(self.attn_layer_idx),
                        workspace,
                        block_table,
                        seq_lens,
                        max_blocks_per_seq,
                        num_q_heads,
                        num_kv_heads,
                        head_dim,
                        block_size,
                        inv_sqrt_d,
                        num_splits,
                        k_scale,
                        v_scale,
                        q_stride,
                        kv_cache.cache_stride() as u64,
                        num_seqs,
                        kv_indirection,
                        kv_indir_base_ptr,
                        kv_indir_stride,
                        stream,
                    )?;
                    ops::paged_decode_attn_reduce_fp8(
                        gpu,
                        reduce_k,
                        workspace,
                        output,
                        seq_lens,
                        num_q_heads,
                        head_dim,
                        num_splits,
                        num_seqs,
                        stream,
                    )
                } else {
                    // Use HDIM=512 kernel for Gemma-4 full-attention layers
                    let fp8_kernel = if head_dim > 256 && self.paged_decode_512_k.0 != 0 {
                        self.paged_decode_512_k
                    } else {
                        self.paged_decode_k
                    };
                    ops::paged_decode_attn_fp8(
                        gpu,
                        fp8_kernel,
                        q,
                        kv_cache.k_pool_ptr(self.attn_layer_idx),
                        kv_cache.v_pool_ptr(self.attn_layer_idx),
                        output,
                        block_table,
                        seq_lens,
                        max_blocks_per_seq,
                        num_seqs,
                        num_q_heads,
                        num_kv_heads,
                        head_dim,
                        block_size,
                        inv_sqrt_d,
                        k_scale,
                        v_scale,
                        q_stride,
                        kv_cache.cache_stride() as u64,
                        kv_indirection,
                        kv_indir_base_ptr,
                        kv_indir_stride,
                        // No tree pack for the standard single-seq decode path.
                        spark_runtime::gpu::DevicePtr::NULL,
                        spark_runtime::gpu::DevicePtr::NULL,
                        0,
                        stream,
                    )
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "run_paged_decode_tests.rs"]
mod tests;

#[cfg(test)]
mod dynamic_sm_tests {
    use spark_runtime::kv_cache::KvCacheDtype;

    use super::{
        GQA12_MAX_PARTS, GQA12_MAX_PARTS_BF16_WIDE, compute_num_splits_for_sms,
        flat_window_crosses_k1_split_boundary_for_sms,
        gqa12_ragged_grid_and_workspace_bytes_for_sms,
        gqa12_ragged_grid_and_workspace_bytes_for_sms_with_cap,
        gqa12_ragged_max_parts_for_kv_and_sm,
    };

    #[test]
    fn legacy_splitk_never_overflows_fixed_48_slot_workspace_on_high_sm_gpus() {
        let workspace_slots = atlas_core::device::sm121::NUM_SMS;
        for sm_count in [24u32, 48, 160, 170, 188, 256] {
            for num_q_heads in [1u32, 4, 8, 16, 24, 32, 64] {
                for num_seqs in [1u32, 2, 3, 5, 17, 32] {
                    let splits =
                        compute_num_splits_for_sms(num_q_heads, num_seqs, 16_384, false, sm_count);
                    assert!(splits >= 1, "sm={sm_count} h={num_q_heads} s={num_seqs}");
                    if splits > 1 {
                        let used_slots = num_seqs * num_q_heads * splits;
                        assert!(
                            used_slots <= workspace_slots,
                            "legacy split-K overflowed 48-slot workspace: sm={sm_count} h={num_q_heads} \
                             s={num_seqs} splits={splits} used_slots={used_slots}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn aggressive_splitk_respects_caps_and_workspace_bounds_on_188_sm_gpu() {
        assert_eq!(compute_num_splits_for_sms(24, 1, 94, true, 48), 2);
        assert_eq!(compute_num_splits_for_sms(24, 1, 94, true, 188), 2);
        assert_eq!(compute_num_splits_for_sms(24, 5, 1536, true, 188), 3);
        assert_eq!(compute_num_splits_for_sms(24, 5, u32::MAX, true, 188), 64);
        assert_eq!(compute_num_splits_for_sms(16, 1, 136_000, true, 188), 16);
        assert!(!flat_window_crosses_k1_split_boundary_for_sms(
            24, 5, 1025, true, 188
        ));
        assert!(flat_window_crosses_k1_split_boundary_for_sms(
            24, 5, 1025, true, 48
        ));
    }

    #[test]
    fn gqa12_ragged_scales_with_high_sm_count_within_partition_cap() {
        // 48 SMs (GB10): 24Q/2KV -> 2 groups, 48 partitions (96 CTAs); 24Q/4KV -> 4 groups, 24 partitions.
        assert_eq!(
            gqa12_ragged_grid_and_workspace_bytes_for_sms(24, 2, 256, 48).map(|(p, g, _)| (p, g)),
            Some((48, 2))
        );
        assert_eq!(
            gqa12_ragged_grid_and_workspace_bytes_for_sms(24, 4, 256, 48).map(|(p, g, _)| (p, g)),
            Some((24, 4))
        );
        // 188 SMs with default 64-part cap (FP8 2-word ballot): scales 4-group and 8-group occupancy.
        assert_eq!(
            gqa12_ragged_grid_and_workspace_bytes_for_sms(24, 4, 256, 188).map(|(p, g, _)| (p, g)),
            Some((GQA12_MAX_PARTS, 4))
        );
        assert_eq!(
            gqa12_ragged_grid_and_workspace_bytes_for_sms(32, 8, 256, 188).map(|(p, g, _)| (p, g)),
            Some((47, 8))
        );
        assert_eq!(
            gqa12_ragged_grid_and_workspace_bytes_for_sms(24, 2, 256, 188).map(|(p, g, _)| (p, g)),
            Some((GQA12_MAX_PARTS, 2))
        );

        // Wide-SM (> 64 SMs) BF16 4-word ballot lifts cap to 128 (256 CTAs on 2 KV heads,
        // 376 CTAs on 4 KV heads) while preserving GB10 (<= 64 SMs) at 48 partitions and FP8 at 64.
        assert_eq!(
            gqa12_ragged_max_parts_for_kv_and_sm(KvCacheDtype::Bf16, 48),
            GQA12_MAX_PARTS
        );
        assert_eq!(
            gqa12_ragged_max_parts_for_kv_and_sm(KvCacheDtype::Bf16, 64),
            GQA12_MAX_PARTS
        );
        assert_eq!(
            gqa12_ragged_max_parts_for_kv_and_sm(KvCacheDtype::Fp8, 188),
            GQA12_MAX_PARTS
        );
        assert_eq!(
            gqa12_ragged_max_parts_for_kv_and_sm(KvCacheDtype::Bf16, 188),
            GQA12_MAX_PARTS_BF16_WIDE
        );
        let bf16_wide = gqa12_ragged_max_parts_for_kv_and_sm(KvCacheDtype::Bf16, 188);
        assert_eq!(
            gqa12_ragged_grid_and_workspace_bytes_for_sms_with_cap(24, 2, 256, 188, bf16_wide)
                .map(|(p, g, _)| (p, g)),
            Some((128, 2))
        );
        assert_eq!(
            gqa12_ragged_grid_and_workspace_bytes_for_sms_with_cap(24, 4, 256, 188, bf16_wide)
                .map(|(p, g, _)| (p, g)),
            Some((94, 4))
        );
        assert_eq!(
            gqa12_ragged_grid_and_workspace_bytes_for_sms_with_cap(32, 8, 256, 188, bf16_wide)
                .map(|(p, g, _)| (p, g)),
            Some((47, 8))
        );
        let gb10_cap = gqa12_ragged_max_parts_for_kv_and_sm(KvCacheDtype::Bf16, 48);
        assert_eq!(
            gqa12_ragged_grid_and_workspace_bytes_for_sms_with_cap(24, 2, 256, 48, gb10_cap)
                .map(|(p, g, _)| (p, g)),
            Some((48, 2))
        );
        let fp8_wide = gqa12_ragged_max_parts_for_kv_and_sm(KvCacheDtype::Fp8, 188);
        assert_eq!(
            gqa12_ragged_grid_and_workspace_bytes_for_sms_with_cap(24, 2, 256, 188, fp8_wide)
                .map(|(p, g, _)| (p, g)),
            Some((64, 2))
        );
    }
}
