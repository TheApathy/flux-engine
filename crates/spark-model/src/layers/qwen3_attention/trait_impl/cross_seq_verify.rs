// SPDX-License-Identifier: AGPL-3.0-only

//! Cross-sequence Qwen4 K-row verify for one attention layer
//! (`ATLAS_QWEN4_BATCHED_K5`).
//!
//! The single-sequence K5 verifier (`decode_qwen4_batched_inner`, production
//! route: batched hyper rows, exact K5 QKV, deferred multi-row O projection)
//! runs for its K rows: attention-hyperconnection prep into `gate_logits`,
//! exact QKV into `qkv_output`, the QSA journal/index/select, the row-ordered
//! attention core staging gated rows into `ssm_qkvz`, one exact O projection
//! over the K rows and a saved-scale inject, then the MLP half.
//!
//! This frame keeps the attention core per sequence while avoiding per-sequence
//! copies into the single-sequence arenas: `cross_seq_attention_core` shadows
//! `gate_logits()` and `ssm_qkvz()` directly with the sequence's `mixed_rows`
//! slice and verifier-owned `raw_rows_dst` staging slice at its row base. QSA
//! journal snapshots run inside the sequence's verify lane. The hyperconnection
//! prep, the O projection, the inject and the whole MLP half run once over every
//! row on the exact multi-row tiers.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::kv_cache::{KvCacheDtype, PagedKvCache};

use super::super::Qwen3AttentionLayer;
use crate::layer::{
    AttnMetadataDev, ForwardContext, Qwen4CrossSeqFrame, Qwen4CrossSeqRows, Qwen4CrossSeqStaging,
};
use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen4K5DeviceAttentionPlan;
use crate::weight_map::QuantizedWeight;

#[inline]
#[must_use]
const fn attn_oproj_tc_dense_fp8_shape_valid(rows: usize, h: usize, attn_width: usize) -> bool {
    rows > 0 && h > 0 && attn_width > 0 && h.is_multiple_of(16) && attn_width.is_multiple_of(16)
}

fn attn_oproj_tc_dense_fp8_available(
    gpu: &dyn GpuBackend,
    weight: &QuantizedWeight,
    rows: usize,
    h: usize,
    attn_width: usize,
) -> bool {
    if !attn_oproj_tc_dense_fp8_shape_valid(rows, h, attn_width)
        || weight.weight.is_null()
        || weight.weight_scale.is_null()
        || !weight.weight_scale_2.is_finite()
    {
        return false;
    }
    if crate::layers::qwen4_fast_proj::fp8_tensor_scales() {
        crate::layers::qwen4_fast_proj::fp8_tensor_kernels(gpu).is_ok()
    } else {
        crate::layers::qwen4_hyper::fp8_row_scale_kernels_available(gpu)
    }
}

/// Cross-sequence mixer preflight admits CUDA graph capture when the batch is
/// strictly `DenseAll` (`!any_qsa`) or pure QSA dense-prefix
/// (`!any_qsa_non_dense_prefix`, where every active QSA sequence has
/// `seq_len + k <= VERIFY_ROWS_DENSE_ENDPOINT` and runs exact dense attention
/// plus static-grid multi-seq QSA staging).
#[inline]
#[must_use]
const fn cross_seq_mixer_graph_capture_admitted(
    graph_capture: bool,
    any_qsa_non_dense_prefix: bool,
) -> bool {
    !graph_capture || !any_qsa_non_dense_prefix
}

/// True when a sequence with `qsa_required` stays strictly inside the dense
/// pre-2048 QSA prefix (`seq_len + k <= VERIFY_ROWS_DENSE_ENDPOINT`).
#[inline]
#[must_use]
const fn qsa_seq_in_dense_prefix(qsa_required: bool, seq_len: usize, k: usize) -> bool {
    if !qsa_required {
        return true;
    }
    match seq_len.checked_add(k) {
        Some(end) => end <= crate::layers::qwen4_qsa::VERIFY_ROWS_DENSE_ENDPOINT,
        None => false,
    }
}

/// Partition `rows` (`1..=256`) for `try_run_gqa12_ragged_paged_decode`.
/// `paged_decode_attn_gqa12_ragged_{bf16,fp8}` natively supports up to
/// `GQA12_MAX_SEQS = 256` (`== QWEN4_NATIVE_MAX_ROWS`), so all valid verify
/// frames run in a single globally balanced ragged kernel pair (`1` chunk)
/// instead of 2..4 sequential 64-row chunks across 12 attention layers.
#[inline]
#[must_use]
const fn dense_gqa12_ragged_chunks(rows: usize) -> ([usize; 4], usize) {
    if rows == 0 || rows > crate::layers::QWEN4_NATIVE_MAX_ROWS {
        ([0, 0, 0, 0], 0)
    } else {
        ([rows, 0, 0, 0], 1)
    }
}

const GQA12_RAGGED_SELECTOR: &str = "ATLAS_PAGED_DECODE_RAGGED_GQA";
const GQA12_MAX_HPC: u32 = 12;
const GQA12_MAX_SEQS: u32 = 256;
const GQA12_MAX_PARTS: u32 = 64;
/// Wide-SM (`> 64` SMs) BF16 partition cap matching the 4-word ballot in
/// `paged_decode_attn_gqa4.cu` (`GQA12_MAX_PARTS = 128`). Lifts 2-KV-head
/// Qwen4 Flash-Next on 188-SM GPUs from 128 -> 256 CTAs (100% SM saturation).
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

#[inline]
#[must_use]
const fn gqa12_ragged_eligible_with_policy(
    mode: Gqa12RaggedMode,
    num_seqs: u32,
    head_dim: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    tree_or_indir: bool,
    det_batch: crate::layers::deterministic_batch::DeterministicBatchPolicy,
) -> bool {
    if det_batch.is_enabled() {
        return false;
    }
    gqa12_ragged_eligible(
        mode,
        num_seqs,
        head_dim,
        num_q_heads,
        num_kv_heads,
        tree_or_indir,
    )
}

#[inline]
#[must_use]
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

fn gqa12_ragged_grid_and_workspace_bytes(
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

/// Alignment and non-aliasing preconditions for
/// `paged_decode_attn_gqa12_ragged_reduce_gated_bf16`.
#[inline]
#[must_use]
const fn gqa12_ragged_reduce_gated_eligible(
    reduce_gated_handle: u64,
    output_ptr: u64,
    gate_ptr: u64,
    gate_stride: u32,
) -> bool {
    reduce_gated_handle != 0
        && output_ptr != 0
        && gate_ptr != 0
        && output_ptr != gate_ptr
        && output_ptr.is_multiple_of(4)
        && gate_ptr.is_multiple_of(4)
        && gate_stride.is_multiple_of(2)
}

struct Gqa12RaggedReduceGatedCache {
    primary: OnceLock<(usize, KernelHandle)>,
    fallback: OnceLock<Mutex<HashMap<usize, KernelHandle>>>,
}

impl Gqa12RaggedReduceGatedCache {
    const fn new() -> Self {
        Self {
            primary: OnceLock::new(),
            fallback: OnceLock::new(),
        }
    }

    fn get_or_load(&self, gpu: &dyn GpuBackend) -> KernelHandle {
        let key = std::ptr::from_ref(gpu) as *const () as usize;
        if let Some(&(cached_key, cached_val)) = self.primary.get() {
            if cached_key == key {
                return cached_val;
            }
        } else {
            let val = crate::layers::try_kernel(
                gpu,
                "paged_decode_attn_gqa4",
                "paged_decode_attn_gqa12_ragged_reduce_gated_bf16",
            );
            let &(stored_key, stored_val) = self.primary.get_or_init(|| (key, val));
            if stored_key == key {
                return stored_val;
            }
            let map = self.fallback.get_or_init(|| Mutex::new(HashMap::new()));
            return *map
                .lock()
                .expect("gqa12 ragged reduce gated cache lock poisoned")
                .entry(key)
                .or_insert(val);
        }
        let map = self.fallback.get_or_init(|| Mutex::new(HashMap::new()));
        let mut guard = map
            .lock()
            .expect("gqa12 ragged reduce gated cache lock poisoned");
        if let Some(&val) = guard.get(&key) {
            return val;
        }
        let val = crate::layers::try_kernel(
            gpu,
            "paged_decode_attn_gqa4",
            "paged_decode_attn_gqa12_ragged_reduce_gated_bf16",
        );
        *guard.entry(key).or_insert(val)
    }
}

fn gqa12_ragged_reduce_gated_kernel(gpu: &dyn GpuBackend) -> KernelHandle {
    static CACHE: Gqa12RaggedReduceGatedCache = Gqa12RaggedReduceGatedCache::new();
    CACHE.get_or_load(gpu)
}

/// Lifted RoPE kernel selection for multi-sequence cross-seq verify.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CrossSeqBatchRope {
    /// `rope_forward_strided_yarn_scaled` with precomputed `qwen4_yarn_inv_freq`.
    StridedYarnScaled,
    /// `rope_forward_strided_b3`: instruction-identical (0-ULP) to both scalar
    /// `rope_forward` and interleaved `rope_forward_mrope_interleaved` whenever
    /// `(pos_t, pos_h, pos_w)` alias the same scalar position stream.
    StridedScalar,
}

/// Parameters determining whether `try_cross_seq_attention_multi` can lift
/// per-row RoPE into a single strided launch across all `C * k` rows.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CrossSeqBatchRopePolicy {
    has_qwen4_yarn_inv_freq: bool,
    has_yarn: bool,
    rope_proportional: bool,
    mrope_interleaved: bool,
    uniform_positions: bool,
    rotary_dim: u32,
    has_strided_yarn_kernel: bool,
    has_strided_b3_kernel: bool,
}

impl CrossSeqBatchRopePolicy {
    #[inline]
    #[must_use]
    const fn select(self) -> Option<CrossSeqBatchRope> {
        // Both strided kernels launch at most 128 threads (one rotary pair per
        // thread -> rotary_dim <= 256) and read a single scalar position stream.
        if (self.mrope_interleaved && !self.uniform_positions)
            || self.rotary_dim == 0
            || self.rotary_dim > 256
            || !self.rotary_dim.is_multiple_of(2)
        {
            return None;
        }
        if self.has_qwen4_yarn_inv_freq {
            if self.has_strided_yarn_kernel {
                Some(CrossSeqBatchRope::StridedYarnScaled)
            } else {
                None
            }
        } else if !self.has_yarn && !self.rope_proportional && self.has_strided_b3_kernel {
            Some(CrossSeqBatchRope::StridedScalar)
        } else {
            None
        }
    }
}

/// Whether `try_cross_seq_attention_multi` must probe the single-sequence
/// `ATLAS_QWEN4_K5_DEVICE_ATTN_GRAPH` guard before mutating state.
///
/// `DenseAll` (`!any_qsa`) frames are never governed by
/// [`Qwen4K5DeviceAttentionPlan`] (which applies only when `qwen4_qsa_required`
/// is set) and skip the probe completely across all context lengths.
#[inline]
#[must_use]
const fn cross_seq_device_plan_probe_needed(any_qsa: bool, env_present: bool) -> bool {
    any_qsa && env_present
}

impl Qwen3AttentionLayer {
    fn cross_seq_attn_row_bytes(&self, ctx: &ForwardContext) -> usize {
        self.num_q_heads_override
            .unwrap_or(ctx.config.num_attention_heads)
            * self.head_dim_override.unwrap_or(ctx.config.head_dim)
            * 2
    }

    pub(super) fn preflight_qwen4_cross_seq_mixer_only_inner(
        &self,
        frame: Qwen4CrossSeqFrame,
        qsa_required: &[bool],
        ctx: &ForwardContext,
    ) -> Result<()> {
        let any_qsa = qsa_required.iter().any(|&required| required);
        self.preflight_qwen4_cross_seq_mixer_only_with_qsa_mode(frame, qsa_required, any_qsa, ctx)
    }

    fn preflight_qwen4_cross_seq_mixer_only_with_qsa_mode(
        &self,
        frame: Qwen4CrossSeqFrame,
        qsa_required: &[bool],
        any_qsa_non_dense_prefix: bool,
        ctx: &ForwardContext,
    ) -> Result<()> {
        ensure!(
            qsa_required.len() == frame.seqs(),
            "cross-sequence attention preflight has {} QSA flags for {} sequences",
            qsa_required.len(),
            frame.seqs()
        );
        ensure!(
            self.qwen4_attn_hyper.is_some() && self.qwen4_mlp_hyper.is_some(),
            "cross-sequence Qwen4 attention verify requires both hyperconnections"
        );
        let any_qsa = qsa_required.iter().any(|&required| required);
        ensure!(
            ctx.config.is_qwen4_exp()
                && cross_seq_mixer_graph_capture_admitted(
                    ctx.graph_capture,
                    any_qsa_non_dense_prefix
                )
                && ctx.config.tp_world_size <= 4
                && ctx.config.ep_world_size <= 4
                && ctx.ddtree_parent_ids_dev.is_none()
                && ctx.tree_aware_attn.is_none()
                && ctx.ffn_defer.is_none()
                && ctx.self_spec_sparse_draft.is_none(),
            "cross-sequence Qwen4 attention verify requires flat-chain TP<=4/EP<=4 (graph capture requires DenseAll or QSA dense-prefix)"
        );
        let max_k = frame.max_k();
        let rows = frame.rows();
        let hybrid_ok = if frame.is_ragged() {
            let mut checked_ks = 0u16;
            let mut all_ok = true;
            for s in 0..frame.seqs() {
                let k_s = frame.seq_k(s);
                let k_bit = if k_s < 16 { 1u16 << k_s } else { 0 };
                if k_bit != 0 && (checked_ks & k_bit) != 0 {
                    continue;
                }
                if !crate::layers::qwen4_hybrid_cross_seq_rows(k_s) {
                    all_ok = false;
                    break;
                }
                checked_ks |= k_bit;
            }
            all_ok
        } else {
            crate::layers::qwen4_hybrid_cross_seq_rows(max_k)
        };
        ensure!(
            hybrid_ok && k5_batch_attn_qkv_selected(),
            "cross-sequence Qwen4 attention verify K{}..=K{max_k} requires the exact K-row QKV route \
             (ATLAS_QWEN4_K5_HYBRID=1, ATLAS_QWEN4_K5_BATCH_ATTN_QKV=1)",
            frame.min_k()
        );
        let h = ctx.config.hidden_size;
        let core_bytes = h * 2;
        let attn_row_bytes = self.cross_seq_attn_row_bytes(ctx);
        // The single-sequence verifier must itself defer its O projection to
        // the multi-row tier (ATLAS_QWEN4_VERIFY_OPROJ_ROWS); the frame then
        // runs the same exact tier at the wider row count.
        ensure!(
            (max_k == 1 || self.verify_oproj_rows_admits(max_k, attn_row_bytes, core_bytes, ctx))
                && self.verify_oproj_rows_admits(rows, attn_row_bytes, core_bytes, ctx),
            "cross-sequence Qwen4 attention verify requires the exact multi-row O projection \
             (ATLAS_QWEN4_VERIFY_OPROJ_ROWS=1) at K{max_k} and at {rows} rows"
        );
        let attn_hyper = self
            .qwen4_attn_hyper
            .as_ref()
            .expect("checked hyperconnections above");
        attn_hyper.preflight_qwen4_native_batched(rows, true)?;
        let sizes = ctx.buffers.sizes();
        ensure!(
            max_k * core_bytes <= sizes.gate_logits && rows * core_bytes <= sizes.norm_output,
            "cross-sequence Qwen4 attention verify rows exceed the serving arenas"
        );
        if any_qsa {
            ensure!(
                self.qwen4_qsa.is_some(),
                "cross-sequence Qwen4 QSA verify lacks an indexer on attention layer {}",
                self.attn_layer_idx
            );
        }
        let nq = self
            .num_q_heads_override
            .unwrap_or(ctx.config.num_attention_heads) as u32;
        let nkv = self
            .num_kv_heads_override
            .unwrap_or(ctx.config.num_key_value_heads) as u32;
        let hd = self.head_dim_override.unwrap_or(ctx.config.head_dim) as u32;
        self.preflight_bf16_gqa4_kernel(ctx.gpu, nq, nkv, hd, max_k as u32)?;
        if self.gated && self.paged_decode_gqa12_ragged_k.is_some() {
            let _ = gqa12_ragged_reduce_gated_kernel(ctx.gpu);
        }
        if frame.is_ragged() {
            let mut checked_ks = if max_k < 16 { 1u16 << max_k } else { 0 };
            for s in 0..frame.seqs() {
                let k_s = frame.seq_k(s);
                let k_bit = if k_s < 16 { 1u16 << k_s } else { 0 };
                if k_bit != 0 && (checked_ks & k_bit) != 0 {
                    continue;
                }
                self.preflight_bf16_gqa4_kernel(ctx.gpu, nq, nkv, hd, k_s as u32)?;
                checked_ks |= k_bit;
            }
        }
        Ok(())
    }

    pub(super) fn preflight_qwen4_cross_seq_verify_inner(
        &self,
        frame: Qwen4CrossSeqFrame,
        qsa_required: &[bool],
        ctx: &ForwardContext,
    ) -> Result<()> {
        self.preflight_qwen4_cross_seq_mixer_only_inner(frame, qsa_required, ctx)?;
        crate::layers::qwen4_cross_seq_mlp::preflight(
            self.qwen4_mlp_hyper.as_ref(),
            &self.ffn,
            frame,
            ctx,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_qwen4_cross_seq_mixer_only_inner(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        frame: Qwen4CrossSeqFrame,
        seqs: &mut [Qwen4CrossSeqRows<'_>],
        kv_cache: &mut PagedKvCache,
        staging: Qwen4CrossSeqStaging,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let _ = self.decode_qwen4_cross_seq_mixer_core(
            hidden, residual, frame, seqs, kv_cache, staging, ctx, false, stream,
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_qwen4_cross_seq_mixer_core(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        frame: Qwen4CrossSeqFrame,
        seqs: &mut [Qwen4CrossSeqRows<'_>],
        kv_cache: &mut PagedKvCache,
        staging: Qwen4CrossSeqStaging,
        ctx: &ForwardContext,
        defer_inject: bool,
        stream: u64,
    ) -> Result<Option<DevicePtr>> {
        ensure!(
            seqs.len() == frame.seqs(),
            "cross-sequence attention frame has {} sequence views for {} sequences",
            seqs.len(),
            frame.seqs()
        );
        let mut qsa_required_stack = [false; 64];
        let qsa_required_vec;
        let qsa_required: &[bool] = if seqs.len() <= qsa_required_stack.len() {
            for (dst, view) in qsa_required_stack.iter_mut().zip(seqs.iter()) {
                *dst = view.metadata.qwen4_qsa_required;
            }
            &qsa_required_stack[..seqs.len()]
        } else {
            qsa_required_vec = seqs
                .iter()
                .map(|view| view.metadata.qwen4_qsa_required)
                .collect::<Vec<_>>();
            &qsa_required_vec
        };
        let any_qsa = qsa_required.iter().any(|&r| r);
        let any_qsa_non_dense_prefix = seqs.iter().enumerate().any(|(s, view)| {
            !qsa_seq_in_dense_prefix(
                view.metadata.qwen4_qsa_required,
                view.seq_len,
                frame.seq_k(s),
            )
        });
        self.preflight_qwen4_cross_seq_mixer_only_with_qsa_mode(
            frame,
            qsa_required,
            any_qsa_non_dense_prefix,
            ctx,
        )?;
        let attn_hyper = self.qwen4_attn_hyper.as_ref().ok_or_else(|| {
            anyhow::anyhow!("cross-sequence Qwen4 attention verify lacks attention hyperconnection")
        })?;
        let rows = frame.rows();
        let h = ctx.config.hidden_size;
        let core_bytes = h * 2;
        let attn_row_bytes = self.cross_seq_attn_row_bytes(ctx);
        let attn_width = attn_row_bytes / 2;
        ensure!(
            rows * core_bytes <= staging.attn_inputs_bytes
                && rows * attn_row_bytes <= staging.raw_rows_bytes,
            "cross-sequence attention staging is smaller than the {rows}-row frame"
        );
        let eps = ctx.config.rms_norm_eps as f32;
        let tc_dense_scope =
            crate::layers::qwen4_hyper::batched_verify_tc_dense_active(rows, ctx.graph_capture);
        let _tc_dense_guard =
            crate::layers::qwen4_hyper::BatchedVerifyTcDenseGuard::enter(tc_dense_scope);
        let tc_dense_fp8 = crate::layers::qwen4_hyper::batched_verify_tc_component_active(
            crate::layers::qwen4_hyper::BatchedVerifyTcComponent::Fp8Dense,
            rows,
            ctx.graph_capture,
        );

        // ── Row-parallel: attention hyperconnection for every row ──
        let attn_inputs = match crate::layers::qwen4_cross_seq_mlp::take_pending_inject() {
            Some(pending)
                if pending.matches(hidden, residual, rows, stream)
                    && !any_qsa
                    && self.mla.is_none()
                    && rows * core_bytes <= ctx.buffers.sizes().norm_output =>
            {
                crate::layers::qwen4_cross_seq_mlp::log_batched_hc_fuse_inject_gn_engaged(rows);
                attn_hyper.prepare_batched_after_inject(
                    hidden,
                    pending.core,
                    residual,
                    rows,
                    ctx.buffers,
                    ctx.gpu,
                    eps,
                    stream,
                )?
            }
            other => {
                if let Some(pending) = other {
                    attn_hyper.inject_saved_batched(
                        pending.hyper,
                        pending.core,
                        pending.residual,
                        pending.rows,
                        ctx.gpu,
                        pending.stream,
                    )?;
                }
                attn_hyper.prepare_batched_into(
                    hidden,
                    residual,
                    rows,
                    ctx.buffers,
                    Some(staging.attn_inputs),
                    ctx.gpu,
                    eps,
                    stream,
                )?
            }
        };
        if !ctx.graph_capture {
            spark_runtime::cublaslt::stage_timer::mark("x_hc_in", stream);
        }

        // ── Row-parallel: QKV projection + QK-norm for every row ──
        let frame_qkv = match self.qwen4_k5_project_qkv_tc_dense(attn_inputs, rows, ctx, stream)? {
            Some(qkv) => Some(qkv),
            None => self.qwen4_k5_project_qkv_exact(attn_inputs, rows, ctx, stream)?,
        };
        let frame_qsa_qk = if any_qsa && let Some(qsa) = self.qwen4_qsa.as_ref() {
            qsa.project_index_qk_rows(
                attn_inputs,
                ctx.buffers.norm_output(),
                rows,
                h as u32,
                ctx.gpu,
                stream,
            )?
        } else {
            None
        };
        if !ctx.graph_capture {
            spark_runtime::cublaslt::stage_timer::mark("x_qkv", stream);
        }

        // ── Batched multi-sequence QSA + RoPE + KV write + sparse/dense attention ──
        let ran_multi = self.try_cross_seq_attention_multi(
            frame,
            seqs,
            frame_qkv,
            frame_qsa_qk,
            attn_row_bytes,
            kv_cache,
            staging,
            ctx,
            stream,
        )?;
        if !ran_multi {
            ensure!(
                !(ctx.graph_capture && any_qsa),
                "cross-sequence QSA dense-prefix graph capture requires multi-seq QSA staging path"
            );
            // ── Per sequence: exact K_s-row attention core, C1 buffer footprint ──
            for (s, view) in seqs.iter_mut().enumerate() {
                let k_s = frame.seq_k(s);
                let base = frame.row_base(s);
                let seq_qkv =
                    frame_qkv.map(|(qkv, row_bytes)| (qkv.offset(base * row_bytes), row_bytes));
                let seq_qsa_qk =
                    frame_qsa_qk.map(|(qk, qk_row_bytes)| qk.offset(base * qk_row_bytes));
                let _lane =
                    crate::layers::qwen4_verify_lanes::LaneGuard::enter_for_slot(view.slot_idx)?;
                self.cross_seq_attention_core(
                    k_s,
                    attn_inputs.offset(base * core_bytes),
                    seq_qkv,
                    seq_qsa_qk,
                    staging.raw_rows.offset(base * attn_row_bytes),
                    hidden.offset(base * ctx.config.residual_width() * 2),
                    residual.offset(base * ctx.config.residual_width() * 2),
                    view,
                    kv_cache,
                    ctx,
                    stream,
                )?;
            }
        }
        if !ctx.graph_capture {
            spark_runtime::cublaslt::stage_timer::mark(
                if ran_multi { "x_multi_attn" } else { "x_attn" },
                stream,
            );
        }

        // ── Row-parallel: O projection and inject for every row ──
        let o_out = ctx.buffers.norm_output();
        let oproj_tc_fp8 = tc_dense_fp8
            && crate::layers::qwen4_fast_proj::batched_verify_dense_proj_tc_admits_with_policy(
                rows,
                h,
                crate::layers::deterministic_batch::deterministic_batch_policy_or_disabled(),
            );
        if oproj_tc_fp8
            && self.try_qwen4_oproj_direct_nvfp4(
                staging.raw_rows,
                o_out,
                rows,
                h,
                attn_width,
                ctx,
                stream,
            )?
        {
            // Executed directly on NVFP4 weights via w4a16_gemm_coalesced* (or opt-in w4a16_gemm_pipe*).
        } else if oproj_tc_fp8
            && (!ctx.graph_capture
                || (crate::layers::qwen4_fast_proj::fp8_tensor_scales()
                    && crate::layers::qwen4_fast_proj::has_cached_weight_tensor(
                        &self.attn.o_proj,
                        h,
                        attn_width,
                        None,
                    )
                    && crate::layers::qwen4_fast_proj::fp8_scratch_ready_for_capture(
                        h,
                        rows.max(crate::layers::qwen4_fast_proj::resolve_fp8_pad_m(rows, h)),
                        attn_width,
                    )))
            && attn_oproj_tc_dense_fp8_available(ctx.gpu, &self.attn.o_proj, rows, h, attn_width)
        {
            crate::layers::qwen4_fast_proj::clear_prequant_fp8_act();
            crate::layers::qwen4_fp8_act_amax::clear();
            crate::layers::qwen4_fast_proj::fp8_gemm_ex(
                ctx.gpu,
                staging.raw_rows,
                &self.attn.o_proj,
                o_out,
                rows,
                h,
                attn_width,
                h,
                None,
                stream,
            )?;
            static OPROJ_TC_LOGGED: std::sync::Once = std::sync::Once::new();
            OPROJ_TC_LOGGED.call_once(|| {
                tracing::info!(
                    selector = crate::layers::qwen4_hyper::BATCHED_VERIFY_TC_DENSE_ENV,
                    component = "attn_o_proj",
                    route = "qwen4_fast_proj_fp8",
                    tensor_scales = crate::layers::qwen4_fast_proj::fp8_tensor_scales(),
                    cached_weight = crate::layers::qwen4_fast_proj::has_cached_weight_tensor(
                        &self.attn.o_proj,
                        h,
                        attn_width,
                        None,
                    ),
                    rows,
                    n = h,
                    k = attn_width,
                    "QWEN4_BATCHED_VERIFY_TC_DENSE_ENGAGED"
                );
            });
        } else {
            ops::w4a16_gemv_batch_logits_exact_with(
                ctx.gpu,
                self.w4a16_exact_o_proj_kernels,
                staging.raw_rows,
                &self.attn.o_proj,
                o_out,
                rows as u32,
                h as u32,
                attn_width as u32,
                stream,
                ops::w4a16_exact_rt2_for_rows(
                    ops::w4a16_gemv_rt2_enabled(),
                    ops::w4a16_exact_rt2_min_rows(),
                    rows as u32,
                ),
            )?;
        }
        if ctx.config.tp_world_size > 1
            && let Some(comm) = ctx.comm
        {
            comm.all_reduce_async(o_out.0, rows * core_bytes, stream)?;
        }
        let deferred_out = if defer_inject {
            Some(o_out)
        } else {
            attn_hyper.inject_saved_batched(hidden, o_out, residual, rows, ctx.gpu, stream)?;
            None
        };
        if !ctx.graph_capture {
            spark_runtime::cublaslt::stage_timer::mark("x_oproj", stream);
        }
        Ok(deferred_out)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_qwen4_cross_seq_verify_inner(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        frame: Qwen4CrossSeqFrame,
        seqs: &mut [Qwen4CrossSeqRows<'_>],
        kv_cache: &mut PagedKvCache,
        staging: Qwen4CrossSeqStaging,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        crate::layers::qwen4_cross_seq_mlp::preflight(
            self.qwen4_mlp_hyper.as_ref(),
            &self.ffn,
            frame,
            ctx,
        )?;
        let mlp_hyper = self.qwen4_mlp_hyper.as_ref().ok_or_else(|| {
            anyhow::anyhow!("cross-sequence Qwen4 attention verify lacks MLP hyperconnection")
        })?;
        let defer_inject = crate::layers::qwen4_cross_seq_mlp::batched_hc_fuse_inject_gn_enabled();
        let deferred_mixer_out = self.decode_qwen4_cross_seq_mixer_core(
            hidden,
            residual,
            frame,
            seqs,
            kv_cache,
            staging,
            ctx,
            defer_inject,
            stream,
        )?;
        crate::layers::qwen4_cross_seq_mlp::run(
            mlp_hyper,
            &self.ffn,
            hidden,
            residual,
            deferred_mixer_out,
            frame,
            ctx,
            stream,
        )
    }

    /// One sequence's K-row attention core: the middle of
    /// `decode_qwen4_batched_inner` on its production route (batched hyper,
    /// exact QKV, no device-attention plan, deferred O projection).
    #[allow(clippy::too_many_arguments)]
    fn cross_seq_attention_core(
        &self,
        k: usize,
        mixed_rows: DevicePtr,
        preprojected_qkv: Option<(DevicePtr, usize)>,
        preprojected_qsa_qk: Option<DevicePtr>,
        raw_rows_dst: DevicePtr,
        hidden_rows: DevicePtr,
        residual_rows: DevicePtr,
        view: &mut Qwen4CrossSeqRows<'_>,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let metadata = view.metadata;
        ensure!(
            metadata.num_seqs as usize == k,
            "cross-sequence attention metadata has {} rows for K{k}",
            metadata.num_seqs
        );
        let seq_len = view.seq_len;
        let seq_ctx = ForwardContext {
            attn_metadata: Some(metadata),
            ..*ctx
        };
        let h = ctx.config.hidden_size;
        let core_bytes = h * 2;
        let row_bytes = ctx.config.residual_width() * 2;
        let attn_row_bytes = self.cross_seq_attn_row_bytes(ctx);
        let eps = ctx.config.rms_norm_eps as f32;
        let mb = metadata.max_blocks_per_seq as usize;

        let qsa_journal =
            metadata.qwen4_qsa_required && crate::layers::qwen4_qsa::verify_journal_rows(k);
        if qsa_journal {
            self.qwen4_qsa
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Qwen4 K-row QSA transaction lacks an indexer"))?
                .prepare_verify_k5_transaction(kv_cache, ctx.gpu)?;
        }
        ensure!(
            Qwen4K5DeviceAttentionPlan::from_host(ctx.config, k, seq_len, metadata)?.is_none(),
            "cross-sequence Qwen4 attention verify does not run the K5 device-attention plan \
             (unset ATLAS_QWEN4_K5_DEVICE_ATTN_GRAPH)"
        );

        // Shadow `gate_logits()` directly with the sequence's `mixed_rows`
        // staging slice (zero-copy; binding retained for C1 parity source checks).
        let attn_inputs = ctx.buffers.gate_logits();
        let _ = (attn_inputs, core_bytes);
        let attn_inputs = mixed_rows;
        let (qkv, qkv_row_bytes) = match preprojected_qkv {
            Some(pair) => pair,
            None => self
                .qwen4_k5_project_qkv_exact(attn_inputs, k, &seq_ctx, stream)?
                .ok_or_else(|| {
                    anyhow::anyhow!("cross-sequence Qwen4 attention verify lost exact K{k} QKV")
                })?,
        };
        let raw_rows = ctx.buffers.ssm_qkvz();
        let _ = raw_rows;
        let raw_rows = raw_rows_dst;
        let qsa_rows_precomputed = if let Some(qsa) = self.qwen4_qsa.as_ref() {
            qsa.update_and_select_verify_rows(
                preprojected_qsa_qk.unwrap_or(attn_inputs),
                k,
                seq_len,
                qsa_journal,
                ctx.graph_capture,
                None,
                kv_cache,
                metadata,
                if preprojected_qsa_qk.is_some() {
                    0
                } else {
                    h as u32
                },
                eps,
                ctx.config.rope_theta as f32,
                ctx.config.rotary_dim() as u32,
                ctx.gpu,
                stream,
            )?
        } else {
            false
        };
        let route_telemetry =
            qsa_journal && crate::layers::qwen4_qsa::verify_route_telemetry_enabled();
        let attention_result =
            (|| -> Result<Option<crate::layers::qwen4_qsa::VerifyAttentionRoute>> {
                if qsa_rows_precomputed
                    && let Some(route) = self.try_qsa_sparse_attention_rows(
                        hidden_rows,
                        residual_rows,
                        raw_rows,
                        k,
                        seq_len,
                        Some((qkv, qkv_row_bytes)),
                        None,
                        true,
                        attn_row_bytes,
                        row_bytes,
                        kv_cache,
                        metadata,
                        &seq_ctx,
                        stream,
                    )?
                {
                    return Ok(route_telemetry.then_some(route));
                }
                if k >= 2
                    && let Some(route) = self.try_dense_paged_attention_rows(
                        raw_rows,
                        k,
                        seq_len,
                        qkv,
                        qkv_row_bytes,
                        attn_row_bytes,
                        qsa_journal,
                        qsa_rows_precomputed,
                        kv_cache,
                        metadata,
                        &seq_ctx,
                        stream,
                    )?
                {
                    return Ok(route_telemetry.then_some(route));
                }
                let mut row_routes = crate::layers::qwen4_qsa::VerifyRowRouteCounts::default();
                for row in 0..k {
                    let token_metadata = AttnMetadataDev {
                        positions: metadata.positions.offset(row * 4),
                        positions_h: metadata.positions_h.offset(row * 4),
                        positions_w: metadata.positions_w.offset(row * 4),
                        slot: metadata.slot.offset(row * 8),
                        seq_len: metadata.seq_len.offset(row * 4),
                        block_table: metadata.block_table.offset(row * mb * 4),
                        num_seqs: 1,
                        ..metadata
                    };
                    let token_ctx = ForwardContext {
                        attn_metadata: Some(token_metadata),
                        ..*ctx
                    };
                    if qsa_journal && !qsa_rows_precomputed {
                        self.qwen4_qsa
                            .as_ref()
                            .ok_or_else(|| {
                                anyhow::anyhow!("Qwen4 K-row QSA transaction lacks an indexer")
                            })?
                            .snapshot_verify_k5_row(
                                kv_cache,
                                token_metadata.slot,
                                row,
                                ctx.gpu,
                                stream,
                            )?;
                    }
                    let (_, row_route) = self.attention_forward_preprojected_deferred_into(
                        attn_inputs.offset(row * core_bytes),
                        qkv.offset(row * qkv_row_bytes),
                        seq_len + row,
                        Some(raw_rows.offset(row * attn_row_bytes)),
                        qsa_journal,
                        view.block_table,
                        view.disk_block_ids,
                        view.disk_last_offloaded_per_layer,
                        kv_cache,
                        &token_ctx,
                        stream,
                    )?;
                    if route_telemetry {
                        row_routes.record(row_route);
                    }
                }
                Ok(route_telemetry.then_some(
                    crate::layers::qwen4_qsa::VerifyAttentionRoute::RowSerial(row_routes),
                ))
            })();
        if qsa_rows_precomputed && let Some(qsa) = self.qwen4_qsa.as_ref() {
            qsa.clear_precomputed_verify_rows();
        }
        if let Some(attention_route) = attention_result? {
            crate::layers::qwen4_qsa::record_verify_route(
                attention_route,
                k,
                seq_len,
                self.attn_layer_idx,
            );
        }
        // Hand the gated rows to the frame before the next sequence's core
        // reuses the single-sequence staging rows.
        if raw_rows != raw_rows_dst {
            ctx.gpu
                .copy_d2d_async(raw_rows, raw_rows_dst, k * attn_row_bytes, stream)?;
        }
        Ok(())
    }

    /// Dispatch the 12:1 GQA-grouped (`SK=32`) ragged paged-decode kernel paired
    /// with `paged_decode_attn_gqa12_ragged_reduce_gated_bf16`, fusing the exact
    /// BF16-rounded sigmoid output gate into the reduce pass so no separate
    /// `sigmoid_gate_mul_batched` launch is needed.
    #[allow(clippy::too_many_arguments)]
    fn try_run_gqa12_ragged_paged_decode_gated(
        &self,
        gpu: &dyn GpuBackend,
        reduce_gated_k: KernelHandle,
        q: DevicePtr,
        gate: DevicePtr,
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
        gate_stride: u32,
        tree_or_indir: bool,
        stream: u64,
    ) -> Result<bool> {
        let Some((main_k, _)) = self.paged_decode_gqa12_ragged_k else {
            return Ok(false);
        };
        if !gqa12_ragged_reduce_gated_eligible(reduce_gated_k.0, output.0, gate.0, gate_stride) {
            return Ok(false);
        }
        if !gqa12_ragged_eligible_with_policy(
            gqa12_ragged_mode(),
            num_seqs,
            head_dim,
            num_q_heads,
            num_kv_heads,
            tree_or_indir,
            crate::layers::deterministic_batch::deterministic_batch_policy_or_disabled(),
        ) {
            return Ok(false);
        }
        let Some((num_partitions, total_groups, needed_bytes)) =
            gqa12_ragged_grid_and_workspace_bytes(
                num_q_heads,
                num_kv_heads,
                head_dim,
                self.kv_dtype,
            )
        else {
            return Ok(false);
        };
        if workspace.is_null() || workspace_cap_bytes < needed_bytes {
            return Ok(false);
        }

        let sliding = self.sliding_window.unwrap_or(0);
        let k_pool = kv_cache.k_pool_ptr(self.attn_layer_idx);
        let v_pool = kv_cache.v_pool_ptr(self.attn_layer_idx);

        match self.kv_dtype {
            KvCacheDtype::Bf16 => {
                static LOGGED_BF16_GATED: std::sync::Once = std::sync::Once::new();
                LOGGED_BF16_GATED.call_once(|| {
                    tracing::info!(
                        "ATTN_GQA12_RAGGED_BF16_GATED_ENGAGED rows={num_seqs} partitions={num_partitions} groups={total_groups}"
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
            }
            KvCacheDtype::Fp8 => {
                static LOGGED_FP8_GATED: std::sync::Once = std::sync::Once::new();
                LOGGED_FP8_GATED.call_once(|| {
                    tracing::info!(
                        "ATTN_GQA12_RAGGED_FP8_GATED_ENGAGED rows={num_seqs} partitions={num_partitions} groups={total_groups}"
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
            }
            _ => return Ok(false),
        }

        KernelLaunch::new(gpu, reduce_gated_k)
            .grid([num_q_heads, num_seqs, 1])
            .block([32, 1, 1])
            .arg_ptr(workspace)
            .arg_ptr(output)
            .arg_ptr(gate)
            .arg_ptr(seq_lens)
            .arg_u32(num_seqs)
            .arg_u32(num_q_heads)
            .arg_u32(num_kv_heads)
            .arg_u32(num_partitions)
            .arg_u32(sliding)
            .arg_u32(gate_stride)
            .launch(stream)?;
        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_cross_seq_sigmoid_gate(
        &self,
        raw_out: DevicePtr,
        gate_base: DevicePtr,
        num_rows: usize,
        q_dim: u32,
        attn_row_bytes: usize,
        qkv_row_bytes: usize,
        qkv_row_stride_elems: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if self.sigmoid_gate_mul_batched_k.0 != 0 && attn_row_bytes == (q_dim as usize) * 2 {
            ops::sigmoid_gate_mul_batched(
                ctx.gpu,
                self.sigmoid_gate_mul_batched_k,
                raw_out,
                gate_base,
                raw_out,
                q_dim,
                qkv_row_stride_elems,
                num_rows as u32,
                stream,
            )
        } else {
            for row in 0..num_rows {
                let row_attn = raw_out.offset(row * attn_row_bytes);
                let gate_ptr = gate_base.offset(row * qkv_row_bytes);
                ops::sigmoid_gate_mul(
                    ctx.gpu,
                    self.sigmoid_gate_mul_k,
                    row_attn,
                    gate_ptr,
                    row_attn,
                    q_dim,
                    stream,
                )?;
            }
            Ok(())
        }
    }

    /// Batched multi-sequence QSA verify + lifted RoPE + lifted KV write +
    /// batched sparse/dense/mixed attention across all `frame.seqs()` sequences.
    ///
    /// Returns `Ok(true)` when executed, or `Ok(false)` BEFORE any KV/QSA state
    /// mutation so the caller falls back cleanly to per-sequence
    /// [`Self::cross_seq_attention_core`] (for example on boundary-straddling
    /// `2044..=2047` sequences, oversized `batch_logits`, or
    /// `ATLAS_QWEN4_BATCHED_QSA_MULTI=0`).
    #[allow(clippy::too_many_arguments)]
    fn try_cross_seq_attention_multi(
        &self,
        frame: Qwen4CrossSeqFrame,
        seqs: &[Qwen4CrossSeqRows<'_>],
        frame_qkv: Option<(DevicePtr, usize)>,
        frame_qsa_qk: Option<(DevicePtr, usize)>,
        attn_row_bytes: usize,
        kv_cache: &mut PagedKvCache,
        staging: Qwen4CrossSeqStaging,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        if !crate::layers::qwen4_qsa::batched_qsa_multi_enabled()
            || frame.seqs() <= 1
            || self.mla.is_some()
            || self.high_speed_swap_engaged(kv_cache)
        {
            return Ok(false);
        }
        let Some((qkv, qkv_row_bytes)) = frame_qkv else {
            return Ok(false);
        };
        if !qkv_row_bytes.is_multiple_of(2) {
            return Ok(false);
        }
        let kv_dtype = kv_cache.dtype_for_layer(self.attn_layer_idx);
        if !matches!(self.kv_dtype, KvCacheDtype::Bf16 | KvCacheDtype::Nvfp4)
            || kv_dtype != self.kv_dtype
        {
            return Ok(false);
        }
        let max_k = frame.max_k();
        let rows = frame.rows();
        let frame_meta = seqs[0].metadata;
        let mb = frame_meta.max_blocks_per_seq as usize;
        let mut uniform_positions = true;
        for (s, view) in seqs.iter().enumerate() {
            let k_s = frame.seq_k(s);
            let base = frame.row_base(s);
            if view.metadata.num_seqs as usize != k_s
                || view.metadata.max_blocks_per_seq != frame_meta.max_blocks_per_seq
                || view.metadata.positions != frame_meta.positions.offset(base * 4)
                || view.metadata.slot != frame_meta.slot.offset(base * 8)
                || view.metadata.seq_len != frame_meta.seq_len.offset(base * 4)
                || view.metadata.block_table != frame_meta.block_table.offset(base * mb * 4)
            {
                return Ok(false);
            }
            if view.metadata.positions_h != view.metadata.positions
                || view.metadata.positions_w != view.metadata.positions
            {
                uniform_positions = false;
            }
        }
        let rotary_dim = self
            .rotary_dim_override
            .unwrap_or(ctx.config.rotary_dim() as u32);
        let Some(batch_rope) = (CrossSeqBatchRopePolicy {
            has_qwen4_yarn_inv_freq: !self.qwen4_yarn_inv_freq.is_null(),
            has_yarn: self.yarn.is_some(),
            rope_proportional: self.rope_proportional,
            mrope_interleaved: self.mrope_interleaved,
            uniform_positions,
            rotary_dim,
            has_strided_yarn_kernel: self.rope_strided_yarn_scaled_k.0 != 0,
            has_strided_b3_kernel: self.rope_strided_b3_k.0 != 0,
        })
        .select() else {
            return Ok(false);
        };

        if seqs.len() > 64 {
            return Ok(false);
        }
        let mut any_qsa = false;
        let mut any_qsa_journal = false;
        let mut sparse_seqs = 0usize;
        let mut dense_seqs = 0usize;
        let mut total_logits_floats = 0usize;
        let mut total_score_blocks = 0u32;
        let mut max_visible_groups = 0usize;
        let mut max_journal_lane = 0usize;
        let mut sparse_mask = [0u64; 4];
        let mut seen_qsa_lanes = [0usize; 64];
        let mut seen_qsa_len = 0usize;
        let mut seq_is_sparse = [false; 64];
        for (s, view) in seqs.iter().enumerate() {
            let k_s = frame.seq_k(s);
            if view.metadata.qwen4_qsa_required {
                any_qsa = true;
                if !crate::layers::qwen4_qsa::verify_rows_context_safe(view.seq_len, k_s) {
                    return Ok(false);
                }
                if seen_qsa_lanes[..seen_qsa_len].contains(&view.slot_idx) {
                    return Ok(false);
                }
                seen_qsa_lanes[seen_qsa_len] = view.slot_idx;
                seen_qsa_len += 1;
                if crate::layers::qwen4_qsa::verify_journal_rows(k_s) {
                    any_qsa_journal = true;
                    max_journal_lane = max_journal_lane.max(view.slot_idx);
                }
                if view.seq_len >= crate::layers::qwen4_qsa::VERIFY_ROWS_DENSE_ENDPOINT {
                    sparse_seqs += 1;
                    seq_is_sparse[s] = true;
                    let max_vis = (view.seq_len + k_s) / 4;
                    max_visible_groups = max_visible_groups.max(max_vis);
                    total_logits_floats =
                        total_logits_floats.saturating_add(k_s.saturating_mul(max_vis));
                    total_score_blocks = total_score_blocks.saturating_add(
                        max_vis.div_ceil(crate::layers::qwen4_qsa::SCORE_V2_GROUPS_PER_BLOCK)
                            as u32,
                    );
                    let base = frame.row_base(s);
                    for r in 0..k_s {
                        let row = base + r;
                        if row < crate::layers::QWEN4_NATIVE_MAX_ROWS {
                            sparse_mask[row / 64] |= 1u64 << (row % 64);
                        }
                    }
                } else {
                    dense_seqs += 1;
                }
            } else {
                dense_seqs += 1;
            }
        }
        let device_plan_env_present = any_qsa && k5_device_attn_graph_env_present();
        if cross_seq_device_plan_probe_needed(any_qsa, device_plan_env_present) {
            for (s, view) in seqs.iter().enumerate() {
                if view.metadata.qwen4_qsa_required
                    && Qwen4K5DeviceAttentionPlan::from_host(
                        ctx.config,
                        frame.seq_k(s),
                        view.seq_len,
                        view.metadata,
                    )?
                    .is_some()
                {
                    return Ok(false);
                }
            }
        }
        let mixed = sparse_seqs > 0 && dense_seqs > 0;
        let qsa_dense_prefix_graph_safe = sparse_seqs == 0
            && seqs.iter().enumerate().all(|(s, view)| {
                qsa_seq_in_dense_prefix(
                    view.metadata.qwen4_qsa_required,
                    view.seq_len,
                    frame.seq_k(s),
                )
            });
        if any_qsa {
            if ctx.graph_capture && !qsa_dense_prefix_graph_safe {
                return Ok(false);
            }
            let Some(qsa) = self.qwen4_qsa.as_ref() else {
                return Ok(false);
            };
            if frame_qsa_qk.is_none()
                || staging.verify_meta_dev.is_null()
                || seqs.len() * crate::layers::qwen4_qsa::QsaBatchSeqDesc::SIZE_BYTES
                    > staging.verify_meta_bytes
                || !staging.qsa_scratch.is_allocated()
                || total_logits_floats > staging.qsa_scratch.batch_logits_floats
                || !qsa.has_multi_seq_verify_kernels(kv_dtype, sparse_seqs > 0, mixed)
                || (any_qsa_journal
                    && max_journal_lane >= crate::layers::qwen4_verify_lanes::lanes())
            {
                return Ok(false);
            }
        }

        let eps = ctx.config.rms_norm_eps as f32;
        let nq = self
            .num_q_heads_override
            .unwrap_or(ctx.config.num_attention_heads) as u32;
        let nkv = self
            .num_kv_heads_override
            .unwrap_or(ctx.config.num_key_value_heads) as u32;
        let hd = self.head_dim_override.unwrap_or(ctx.config.head_dim) as u32;
        let bs = kv_cache.block_size() as u32;
        let q_dim = nq * hd;
        let q_proj_dim = if self.gated { q_dim * 2 } else { q_dim };
        let q_proj_bytes = q_proj_dim as usize * 2;
        let kv_bytes = (nkv * hd) as usize * 2;
        let qkv_row_stride_elems = (qkv_row_bytes / 2) as u32;
        let k_offset_bf16 = (q_proj_bytes / 2) as u32;
        let rope_theta = self
            .rope_theta_override
            .unwrap_or(ctx.config.rope_theta as f32);

        // 1. Batched multi-sequence QSA stage/pool/RoPE/store + score + select-expand
        if any_qsa {
            let qsa = self.qwen4_qsa.as_ref().expect("checked above");
            let (projected_qk, _) = frame_qsa_qk.expect("checked above");
            if any_qsa_journal {
                qsa.prepare_verify_k5_transaction(kv_cache, ctx.gpu)?;
            }
            qsa.update_and_select_verify_multi_seq(
                projected_qk,
                seqs.len(),
                max_k,
                rows,
                total_score_blocks,
                max_visible_groups,
                any_qsa_journal,
                max_journal_lane,
                staging.verify_meta_dev,
                staging.qsa_scratch,
                kv_cache,
                frame_meta,
                eps,
                rope_theta,
                rotary_dim,
                ctx.gpu,
                stream,
            )?;
        }

        // 2. Lifted RoPE across all `rows = C * k`
        match batch_rope {
            CrossSeqBatchRope::StridedYarnScaled => {
                let block_x = (rotary_dim / 2).next_power_of_two().clamp(32, 128);
                spark_runtime::kernel_args::KernelLaunch::new(
                    ctx.gpu,
                    self.rope_strided_yarn_scaled_k,
                )
                .grid([nq + nkv, rows as u32, 1])
                .block([block_x, 1, 1])
                .arg_ptr(qkv)
                .arg_ptr(frame_meta.positions)
                .arg_u32(rows as u32)
                .arg_u32(qkv_row_stride_elems)
                .arg_u32(k_offset_bf16)
                .arg_u32(nq)
                .arg_u32(nkv)
                .arg_u32(hd)
                .arg_u32(rotary_dim)
                .arg_ptr(self.qwen4_yarn_inv_freq)
                .arg_f32(rope_theta)
                .arg_f32(self.qwen4_yarn_attention_factor)
                .launch(stream)?;
            }
            CrossSeqBatchRope::StridedScalar => {
                ops::rope_strided_b3(
                    ctx.gpu,
                    self.rope_strided_b3_k,
                    qkv,
                    frame_meta.positions,
                    rows as u32,
                    qkv_row_stride_elems,
                    k_offset_bf16,
                    nq,
                    nkv,
                    hd,
                    rotary_dim,
                    rope_theta,
                    stream,
                )?;
            }
        }

        // 3. Lifted KV cache write across all `rows = C * k`
        let k_base = qkv.offset(q_proj_bytes);
        let v_base = k_base.offset(kv_bytes);
        self.write_kv_cache(
            ctx.gpu,
            k_base,
            v_base,
            kv_cache,
            frame_meta.slot,
            rows as u32,
            nkv,
            hd,
            bs,
            qkv_row_stride_elems,
            qkv_row_stride_elems,
            stream,
            ctx.graph_capture,
        )?;
        if !ctx.graph_capture {
            spark_runtime::cublaslt::stage_timer::mark("x_rope_kvw", stream);
        }

        // 4. Batched sparse / dense / mixed attention + sigmoid gating
        let inv_sqrt_d = self.effective_attn_scale(hd);
        let gate_offset_bytes = q_dim as usize * 2;
        if dense_seqs == 0 {
            // All-sparse batch: reuse unmasked partial + reduce_gated across all `C * k` rows.
            let qsa = self.qwen4_qsa.as_ref().expect("sparse requires QSA");
            let fused_gate = qsa.sparse_attention_multi_seq_rows(
                qkv,
                qkv_row_stride_elems,
                staging.raw_rows,
                rows,
                None,
                staging.qsa_scratch,
                kv_cache,
                frame_meta,
                self.attn_layer_idx,
                nq,
                nkv,
                hd,
                inv_sqrt_d,
                self.gated
                    .then(|| (qkv.offset(gate_offset_bytes), qkv_row_stride_elems)),
                ctx.gpu,
                stream,
            )?;
            if self.gated && !fused_gate {
                self.apply_cross_seq_sigmoid_gate(
                    staging.raw_rows,
                    qkv.offset(gate_offset_bytes),
                    rows,
                    q_dim,
                    attn_row_bytes,
                    qkv_row_bytes,
                    qkv_row_stride_elems,
                    ctx,
                    stream,
                )?;
            }
        } else if sparse_seqs == 0 {
            // All-dense batch: prefer the 12:1 GQA-grouped ragged paged-decode
            // kernel pair (static grid independent of seq_len, 12x less KV BW,
            // 100% CUDA-graph safe), natively covering up to 256 rows in a
            // single globally balanced launch.
            // When gated, fuse exact BF16-rounded sigmoid gating into the reduce pass.
            let mut ran_ragged = false;
            let mut ragged_gated = false;
            let ragged_ws = ctx.buffers.ssm_deinterleaved();
            let ragged_ws_cap = ctx.buffers.sizes().ssm_deinterleaved;
            if attn_row_bytes == (q_dim as usize) * 2
                && !ragged_ws.is_null()
                && ragged_ws != qkv
                && ragged_ws != staging.raw_rows
            {
                let (chunks, num_chunks) = dense_gqa12_ragged_chunks(rows);
                if num_chunks > 0 {
                    let reduce_gated_k = if self.gated {
                        let k = gqa12_ragged_reduce_gated_kernel(ctx.gpu);
                        let gate_base = qkv.offset(gate_offset_bytes);
                        if gqa12_ragged_reduce_gated_eligible(
                            k.0,
                            staging.raw_rows.0,
                            gate_base.0,
                            qkv_row_stride_elems,
                        ) {
                            Some(k)
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                    let mut row_base = 0usize;
                    let mut all_ok = true;
                    for &chunk_rows in &chunks[..num_chunks] {
                        let chunk_qkv = qkv.offset(row_base * qkv_row_bytes);
                        let chunk_out = staging.raw_rows.offset(row_base * attn_row_bytes);
                        let ok = if let Some(gated_k) = reduce_gated_k {
                            self.try_run_gqa12_ragged_paged_decode_gated(
                                ctx.gpu,
                                gated_k,
                                chunk_qkv,
                                chunk_qkv.offset(gate_offset_bytes),
                                kv_cache,
                                chunk_out,
                                ragged_ws,
                                ragged_ws_cap,
                                frame_meta.block_table.offset(row_base * mb * 4),
                                frame_meta.seq_len.offset(row_base * 4),
                                frame_meta.max_blocks_per_seq,
                                chunk_rows as u32,
                                nq,
                                nkv,
                                hd,
                                bs,
                                inv_sqrt_d,
                                qkv_row_stride_elems,
                                qkv_row_stride_elems,
                                false,
                                stream,
                            )?
                        } else {
                            self.try_run_gqa12_ragged_paged_decode(
                                ctx.gpu,
                                chunk_qkv,
                                kv_cache,
                                chunk_out,
                                ragged_ws,
                                ragged_ws_cap,
                                frame_meta.block_table.offset(row_base * mb * 4),
                                frame_meta.seq_len.offset(row_base * 4),
                                frame_meta.max_blocks_per_seq,
                                chunk_rows as u32,
                                nq,
                                nkv,
                                hd,
                                bs,
                                inv_sqrt_d,
                                qkv_row_stride_elems,
                                false,
                                stream,
                            )?
                        };
                        if !ok {
                            all_ok = false;
                            break;
                        }
                        row_base += chunk_rows;
                    }
                    ran_ragged = all_ok;
                    ragged_gated = all_ok && reduce_gated_k.is_some();
                }
            }
            if ran_ragged {
                if self.gated && !ragged_gated {
                    self.apply_cross_seq_sigmoid_gate(
                        staging.raw_rows,
                        qkv.offset(gate_offset_bytes),
                        rows,
                        q_dim,
                        attn_row_bytes,
                        qkv_row_bytes,
                        qkv_row_stride_elems,
                        ctx,
                        stream,
                    )?;
                }
            } else if self.kv_dtype == KvCacheDtype::Bf16 || rows <= 32 {
                // Under CUDA graph capture, keep split-K grid selection independent
                // of per-request host sequence lengths across bucket replays.
                let max_seq_len_host = if ctx.graph_capture {
                    0
                } else {
                    seqs.iter()
                        .enumerate()
                        .map(|(s, v)| v.seq_len + frame.seq_k(s))
                        .max()
                        .unwrap_or(0) as u32
                };
                let fused_gate = self.run_paged_decode_with_gate(
                    ctx.gpu,
                    qkv,
                    kv_cache,
                    staging.raw_rows,
                    frame_meta.block_table,
                    frame_meta.seq_len,
                    frame_meta.max_blocks_per_seq,
                    rows as u32,
                    nq,
                    nkv,
                    hd,
                    bs,
                    inv_sqrt_d,
                    qkv_row_stride_elems,
                    ctx.buffers.splitk_workspace(),
                    DevicePtr::NULL,
                    DevicePtr::NULL,
                    0,
                    max_seq_len_host,
                    self.gated
                        .then(|| (qkv.offset(gate_offset_bytes), qkv_row_stride_elems)),
                    stream,
                )?;
                if self.gated && !fused_gate {
                    self.apply_cross_seq_sigmoid_gate(
                        staging.raw_rows,
                        qkv.offset(gate_offset_bytes),
                        rows,
                        q_dim,
                        attn_row_bytes,
                        qkv_row_bytes,
                        qkv_row_stride_elems,
                        ctx,
                        stream,
                    )?;
                }
            } else {
                for (s, view) in seqs.iter().enumerate() {
                    let k_s = frame.seq_k(s);
                    let base = frame.row_base(s);
                    let seq_qkv = qkv.offset(base * qkv_row_bytes);
                    let seq_raw = staging.raw_rows.offset(base * attn_row_bytes);
                    let seq_max_len = if ctx.graph_capture {
                        0
                    } else {
                        (view.seq_len + k_s) as u32
                    };
                    let fused_gate = self.run_paged_decode_with_gate(
                        ctx.gpu,
                        seq_qkv,
                        kv_cache,
                        seq_raw,
                        view.metadata.block_table,
                        view.metadata.seq_len,
                        view.metadata.max_blocks_per_seq,
                        k_s as u32,
                        nq,
                        nkv,
                        hd,
                        bs,
                        inv_sqrt_d,
                        qkv_row_stride_elems,
                        ctx.buffers.splitk_workspace(),
                        DevicePtr::NULL,
                        DevicePtr::NULL,
                        0,
                        seq_max_len,
                        self.gated
                            .then(|| (seq_qkv.offset(gate_offset_bytes), qkv_row_stride_elems)),
                        stream,
                    )?;
                    if self.gated && !fused_gate {
                        self.apply_cross_seq_sigmoid_gate(
                            seq_raw,
                            seq_qkv.offset(gate_offset_bytes),
                            k_s,
                            q_dim,
                            attn_row_bytes,
                            qkv_row_bytes,
                            qkv_row_stride_elems,
                            ctx,
                            stream,
                        )?;
                    }
                }
            }
        } else {
            // Mixed batch: masked sparse launches for sparse rows + paged decode per dense seq.
            let qsa = self.qwen4_qsa.as_ref().expect("sparse requires QSA");
            let fused_sparse_gate = qsa.sparse_attention_multi_seq_rows(
                qkv,
                qkv_row_stride_elems,
                staging.raw_rows,
                rows,
                Some(sparse_mask),
                staging.qsa_scratch,
                kv_cache,
                frame_meta,
                self.attn_layer_idx,
                nq,
                nkv,
                hd,
                inv_sqrt_d,
                self.gated
                    .then(|| (qkv.offset(gate_offset_bytes), qkv_row_stride_elems)),
                ctx.gpu,
                stream,
            )?;
            for (s, view) in seqs.iter().enumerate() {
                let k_s = frame.seq_k(s);
                let base = frame.row_base(s);
                let seq_qkv = qkv.offset(base * qkv_row_bytes);
                let seq_raw = staging.raw_rows.offset(base * attn_row_bytes);
                if seq_is_sparse[s] {
                    if self.gated && !fused_sparse_gate {
                        self.apply_cross_seq_sigmoid_gate(
                            seq_raw,
                            seq_qkv.offset(gate_offset_bytes),
                            k_s,
                            q_dim,
                            attn_row_bytes,
                            qkv_row_bytes,
                            qkv_row_stride_elems,
                            ctx,
                            stream,
                        )?;
                    }
                } else {
                    let fused_dense_gate = self.run_paged_decode_with_gate(
                        ctx.gpu,
                        seq_qkv,
                        kv_cache,
                        seq_raw,
                        view.metadata.block_table,
                        view.metadata.seq_len,
                        view.metadata.max_blocks_per_seq,
                        k_s as u32,
                        nq,
                        nkv,
                        hd,
                        bs,
                        inv_sqrt_d,
                        qkv_row_stride_elems,
                        ctx.buffers.splitk_workspace(),
                        DevicePtr::NULL,
                        DevicePtr::NULL,
                        0,
                        (view.seq_len + k_s) as u32,
                        self.gated
                            .then(|| (seq_qkv.offset(gate_offset_bytes), qkv_row_stride_elems)),
                        stream,
                    )?;
                    if self.gated && !fused_dense_gate {
                        self.apply_cross_seq_sigmoid_gate(
                            seq_raw,
                            seq_qkv.offset(gate_offset_bytes),
                            k_s,
                            q_dim,
                            attn_row_bytes,
                            qkv_row_bytes,
                            qkv_row_stride_elems,
                            ctx,
                            stream,
                        )?;
                    }
                }
            }
        }

        if any_qsa && any_qsa_journal && crate::layers::qwen4_qsa::verify_route_telemetry_enabled()
        {
            for (s, view) in seqs.iter().enumerate() {
                let k_s = frame.seq_k(s);
                if view.metadata.qwen4_qsa_required
                    && crate::layers::qwen4_qsa::verify_journal_rows(k_s)
                {
                    let route = if seq_is_sparse[s] {
                        crate::layers::qwen4_qsa::VerifyAttentionRoute::SparseRowsV2
                    } else {
                        let mut counts = crate::layers::qwen4_qsa::VerifyRowRouteCounts::default();
                        for _ in 0..k_s {
                            counts.record(
                                crate::layers::qwen4_qsa::VerifyRowAttentionRoute::DensePaged,
                            );
                        }
                        crate::layers::qwen4_qsa::VerifyAttentionRoute::RowSerial(counts)
                    };
                    crate::layers::qwen4_qsa::record_verify_route(
                        route,
                        k_s,
                        view.seq_len,
                        self.attn_layer_idx,
                    );
                }
            }
        }

        static ENGAGED: std::sync::Once = std::sync::Once::new();
        ENGAGED.call_once(|| {
            tracing::info!(
                seqs = seqs.len(),
                min_k = frame.min_k(),
                max_k,
                rows,
                sparse_seqs,
                dense_seqs,
                "QWEN4_BATCHED_QSA_MULTI_ENGAGED"
            );
        });
        Ok(true)
    }
}

/// `ATLAS_PAGED_DECODE_RAGGED_GQA` mode (cached via `OnceLock` outside unit
/// tests). Placed after `impl Qwen3AttentionLayer` so `c1_parity_source_tests`'s
/// `body()` slice still covers the entire attention implementation.
#[inline]
fn gqa12_ragged_mode() -> Gqa12RaggedMode {
    let read_live = || gqa12_ragged_mode_from(std::env::var(GQA12_RAGGED_SELECTOR).ok().as_deref());
    #[cfg(not(test))]
    {
        static MODE: OnceLock<Gqa12RaggedMode> = OnceLock::new();
        *MODE.get_or_init(read_live)
    }
    #[cfg(test)]
    {
        read_live()
    }
}

/// `ATLAS_QWEN4_K5_BATCH_ATTN_QKV=1` (cached via `OnceLock` outside unit tests).
#[inline]
fn k5_batch_attn_qkv_selected() -> bool {
    let read_live = || {
        std::env::var("ATLAS_QWEN4_K5_BATCH_ATTN_QKV")
            .ok()
            .as_deref()
            == Some("1")
    };
    #[cfg(not(test))]
    {
        static GATE: OnceLock<bool> = OnceLock::new();
        *GATE.get_or_init(read_live)
    }
    #[cfg(test)]
    {
        read_live()
    }
}

/// Presence of `ATLAS_QWEN4_K5_DEVICE_ATTN_GRAPH` (cached via `OnceLock`
/// outside unit tests).
#[inline]
fn k5_device_attn_graph_env_present() -> bool {
    let read_live = || std::env::var_os("ATLAS_QWEN4_K5_DEVICE_ATTN_GRAPH").is_some();
    #[cfg(not(test))]
    {
        static PRESENT: OnceLock<bool> = OnceLock::new();
        *PRESENT.get_or_init(read_live)
    }
    #[cfg(test)]
    {
        read_live()
    }
}

#[cfg(test)]
mod c1_parity_source_tests {
    const CROSS: &str = include_str!("cross_seq_verify.rs");
    const C1: &str = include_str!("decode_inner.rs");

    fn body(source: &str) -> &str {
        source.split("#[cfg(test)]").next().unwrap()
    }

    #[test]
    fn attention_core_uses_the_c1_helpers_and_transaction_kernels() {
        for call in [
            ".prepare_verify_k5_transaction(",
            ".qwen4_k5_project_qkv_exact(",
            ".update_and_select_verify_rows(",
            ".try_qsa_sparse_attention_rows(",
            ".snapshot_verify_k5_row(",
            ".attention_forward_preprojected_deferred_into(",
            ".clear_precomputed_verify_rows(",
            "ops::w4a16_gemv_batch_logits_exact_with(",
            ".inject_saved_batched(",
        ] {
            assert!(
                body(C1).contains(call),
                "C1 verifier no longer calls {call}"
            );
            assert!(
                body(CROSS).contains(call),
                "cross-sequence core lost {call}"
            );
        }
    }

    #[test]
    fn attention_core_runs_inside_the_sequence_lane() {
        let core = body(CROSS);
        let lane = core.find("LaneGuard::enter_for_slot(").unwrap();
        let call = core.find("self.cross_seq_attention_core(").unwrap();
        assert!(lane < call, "the lane must be entered before the core runs");
    }

    #[test]
    fn attention_core_keeps_the_c1_staging_footprint() {
        let core = body(CROSS);
        assert!(core.contains("let attn_inputs = ctx.buffers.gate_logits();"));
        assert!(core.contains("let raw_rows = ctx.buffers.ssm_qkvz();"));
        assert!(core.contains("copy_d2d_async(raw_rows, raw_rows_dst"));
    }

    #[test]
    fn attn_oproj_tc_dense_fp8_shape_admits_canonical_and_tp2_shapes_only() {
        use super::attn_oproj_tc_dense_fp8_shape_valid;
        use crate::layers::qwen4_fast_proj::{
            batched_verify_dense_proj_tc_admits, coalesced_nvfp4_prefers_fp8_cached,
        };
        for rows in [0usize, 1, 2, 3] {
            assert!(!batched_verify_dense_proj_tc_admits(rows, 2_560));
        }
        for rows in [4usize, 5, 6, 7, 8, 9, 10, 12, 16, 20, 40, 64, 80, 128] {
            assert!(attn_oproj_tc_dense_fp8_shape_valid(rows, 2_560, 6_144));
            assert!(attn_oproj_tc_dense_fp8_shape_valid(rows, 2_560, 3_072));
            assert!(batched_verify_dense_proj_tc_admits(rows, 2_560));
            assert!(
                !coalesced_nvfp4_prefers_fp8_cached(rows, 2_560),
                "rows={rows}: attn.oproj stays on zero-VRAM coalesced/Warp-Split-K NVFP4 through 128 rows"
            );
        }
        assert!(!attn_oproj_tc_dense_fp8_shape_valid(0, 2_560, 6_144));
        assert!(!attn_oproj_tc_dense_fp8_shape_valid(20, 2_556, 6_144));
        assert!(!attn_oproj_tc_dense_fp8_shape_valid(20, 2_560, 6_140));
    }

    #[test]
    fn dense_all_graph_capture_and_gqa12_ragged_chunks_cover_b2_to_b16() {
        use super::{cross_seq_mixer_graph_capture_admitted, dense_gqa12_ragged_chunks};
        assert!(cross_seq_mixer_graph_capture_admitted(false, false));
        assert!(cross_seq_mixer_graph_capture_admitted(false, true));
        assert!(cross_seq_mixer_graph_capture_admitted(true, false));
        assert!(!cross_seq_mixer_graph_capture_admitted(true, true));

        assert_eq!(dense_gqa12_ragged_chunks(0), ([0, 0, 0, 0], 0));
        assert_eq!(dense_gqa12_ragged_chunks(257), ([0, 0, 0, 0], 0));
        assert_eq!(dense_gqa12_ragged_chunks(10), ([10, 0, 0, 0], 1));
        assert_eq!(dense_gqa12_ragged_chunks(20), ([20, 0, 0, 0], 1));
        assert_eq!(dense_gqa12_ragged_chunks(40), ([40, 0, 0, 0], 1));
        assert_eq!(dense_gqa12_ragged_chunks(64), ([64, 0, 0, 0], 1));
        assert_eq!(dense_gqa12_ragged_chunks(65), ([65, 0, 0, 0], 1));
        assert_eq!(dense_gqa12_ragged_chunks(80), ([80, 0, 0, 0], 1));
        assert_eq!(dense_gqa12_ragged_chunks(128), ([128, 0, 0, 0], 1));
        assert_eq!(dense_gqa12_ragged_chunks(129), ([129, 0, 0, 0], 1));
        assert_eq!(dense_gqa12_ragged_chunks(192), ([192, 0, 0, 0], 1));
        assert_eq!(dense_gqa12_ragged_chunks(193), ([193, 0, 0, 0], 1));
        assert_eq!(dense_gqa12_ragged_chunks(256), ([256, 0, 0, 0], 1));
        for rows in 1usize..=256 {
            let (chunks, n) = dense_gqa12_ragged_chunks(rows);
            assert_eq!(n, 1, "rows={rows}");
            assert_eq!(chunks[0], rows, "rows={rows}");
            for &trailing in &chunks[1..] {
                assert_eq!(trailing, 0, "rows={rows} trailing={trailing}");
            }
        }
    }

    #[test]
    fn batch_rope_policy_admits_flash_next_mrope_with_and_without_yarn() {
        use super::{CrossSeqBatchRope, CrossSeqBatchRopePolicy};

        let flash_next_default = CrossSeqBatchRopePolicy {
            has_qwen4_yarn_inv_freq: false,
            has_yarn: false,
            rope_proportional: false,
            mrope_interleaved: true,
            uniform_positions: true,
            rotary_dim: 64,
            has_strided_yarn_kernel: true,
            has_strided_b3_kernel: true,
        };
        assert_eq!(
            flash_next_default.select(),
            Some(CrossSeqBatchRope::StridedScalar),
            "canonical Flash-Next interleaved MRoPE with uniform scalar positions must lift RoPE"
        );

        let flash_next_yarn = CrossSeqBatchRopePolicy {
            has_qwen4_yarn_inv_freq: true,
            has_yarn: true,
            ..flash_next_default
        };
        assert_eq!(
            flash_next_yarn.select(),
            Some(CrossSeqBatchRope::StridedYarnScaled),
            "Flash-Next 1M YaRN must lift strided YaRN RoPE"
        );

        let non_mrope_scalar = CrossSeqBatchRopePolicy {
            mrope_interleaved: false,
            uniform_positions: false,
            ..flash_next_default
        };
        assert_eq!(
            non_mrope_scalar.select(),
            Some(CrossSeqBatchRope::StridedScalar)
        );

        // Non-uniform 3D MRoPE positions, unbacked YaRN, proportional RoPE,
        // missing kernels, and out-of-range rotary dims must fail closed.
        assert_eq!(
            CrossSeqBatchRopePolicy {
                uniform_positions: false,
                ..flash_next_default
            }
            .select(),
            None
        );
        assert_eq!(
            CrossSeqBatchRopePolicy {
                has_yarn: true,
                ..flash_next_default
            }
            .select(),
            None
        );
        assert_eq!(
            CrossSeqBatchRopePolicy {
                rope_proportional: true,
                ..flash_next_default
            }
            .select(),
            None
        );
        assert_eq!(
            CrossSeqBatchRopePolicy {
                has_strided_b3_kernel: false,
                ..flash_next_default
            }
            .select(),
            None
        );
        assert_eq!(
            CrossSeqBatchRopePolicy {
                has_strided_yarn_kernel: false,
                ..flash_next_yarn
            }
            .select(),
            None
        );
        for bad_dim in [0u32, 63, 258] {
            assert_eq!(
                CrossSeqBatchRopePolicy {
                    rotary_dim: bad_dim,
                    ..flash_next_default
                }
                .select(),
                None,
                "rotary_dim={bad_dim}"
            );
        }
        assert_eq!(
            CrossSeqBatchRopePolicy {
                rotary_dim: 256,
                ..flash_next_default
            }
            .select(),
            Some(CrossSeqBatchRope::StridedScalar)
        );
    }

    #[test]
    fn dense_all_skips_device_plan_probe_and_attributes_multi_attn_stages() {
        use super::cross_seq_device_plan_probe_needed;

        // DenseAll (!any_qsa) never probes Qwen4K5DeviceAttentionPlan regardless
        // of context length (< 16384 or >= 16384) or env selector presence.
        assert!(!cross_seq_device_plan_probe_needed(false, false));
        assert!(!cross_seq_device_plan_probe_needed(false, true));
        assert!(!cross_seq_device_plan_probe_needed(true, false));
        assert!(cross_seq_device_plan_probe_needed(true, true));

        let src = body(CROSS);
        let multi = src
            .split("fn try_cross_seq_attention_multi(")
            .nth(1)
            .expect("try_cross_seq_attention_multi present");
        assert!(
            !multi.contains("!self.mrope_interleaved"),
            "try_cross_seq_attention_multi must not reject Flash-Next interleaved MRoPE"
        );
        assert!(
            multi.contains("CrossSeqBatchRopePolicy"),
            "try_cross_seq_attention_multi must select lifted RoPE via CrossSeqBatchRopePolicy"
        );
        assert!(
            multi.contains("cross_seq_device_plan_probe_needed("),
            "device-attention plan probe must be gated behind cross_seq_device_plan_probe_needed"
        );

        // Stage timer marks cleanly separate QKV, lifted RoPE+KV write, and multi-seq attention.
        let hc_in = src.find("mark(\"x_hc_in\"").unwrap();
        let qkv = src.find("mark(\"x_qkv\"").unwrap();
        let call_multi = src.find("self.try_cross_seq_attention_multi(").unwrap();
        let multi_attn = src.find("\"x_multi_attn\"").unwrap();
        let oproj = src.find("mark(\"x_oproj\"").unwrap();
        assert!(hc_in < qkv && qkv < call_multi && call_multi < multi_attn && multi_attn < oproj);

        let kvw = multi.find("self.write_kv_cache(").unwrap();
        let rope_kvw = multi.find("mark(\"x_rope_kvw\"").unwrap();
        let ragged = multi
            .find("self.try_run_gqa12_ragged_paged_decode(")
            .unwrap();
        let ragged_gated = multi
            .find("self.try_run_gqa12_ragged_paged_decode_gated(")
            .unwrap();
        assert!(kvw < rope_kvw && rope_kvw < ragged_gated && ragged_gated < ragged);
        assert!(multi.contains("if self.gated && !ragged_gated {"));
    }

    #[test]
    fn gqa12_ragged_reduce_gated_validates_alignment_and_disjoint_ptrs() {
        use super::gqa12_ragged_reduce_gated_eligible;

        assert!(gqa12_ragged_reduce_gated_eligible(
            1, 0x1000, 0x2000, 13_312
        ));
        assert!(!gqa12_ragged_reduce_gated_eligible(
            0, 0x1000, 0x2000, 13_312
        ));
        assert!(!gqa12_ragged_reduce_gated_eligible(1, 0, 0x2000, 13_312));
        assert!(!gqa12_ragged_reduce_gated_eligible(1, 0x1000, 0, 13_312));
        assert!(!gqa12_ragged_reduce_gated_eligible(
            1, 0x1000, 0x1000, 13_312
        ));
        assert!(!gqa12_ragged_reduce_gated_eligible(
            1, 0x1002, 0x2000, 13_312
        ));
        assert!(!gqa12_ragged_reduce_gated_eligible(
            1, 0x1000, 0x2002, 13_312
        ));
        assert!(!gqa12_ragged_reduce_gated_eligible(
            1, 0x1000, 0x2000, 13_311
        ));
    }

    #[test]
    fn deterministic_batch_disables_cross_seq_gqa12_ragged_across_1_to_128() {
        use super::{Gqa12RaggedMode, gqa12_ragged_eligible_with_policy};
        use crate::layers::deterministic_batch::DeterministicBatchPolicy;

        for m in [1u32, 2, 4, 8, 16, 32, 64, 128] {
            for mode in [Gqa12RaggedMode::Auto, Gqa12RaggedMode::ForceOn] {
                assert!(
                    !gqa12_ragged_eligible_with_policy(
                        mode,
                        m,
                        256,
                        24,
                        2,
                        false,
                        DeterministicBatchPolicy::Enabled,
                    ),
                    "cross-seq gqa12_ragged must stay disabled at M={m} mode={mode:?}"
                );
            }
        }
        assert!(!gqa12_ragged_eligible_with_policy(
            Gqa12RaggedMode::Auto,
            1,
            256,
            24,
            2,
            false,
            DeterministicBatchPolicy::Disabled,
        ));
        assert!(gqa12_ragged_eligible_with_policy(
            Gqa12RaggedMode::Auto,
            8,
            256,
            24,
            2,
            false,
            DeterministicBatchPolicy::Disabled,
        ));
    }

    #[test]
    fn cached_attention_cross_seq_env_helpers_track_live_env_in_tests() {
        use super::{
            GQA12_RAGGED_SELECTOR, Gqa12RaggedMode, gqa12_ragged_mode, k5_batch_attn_qkv_selected,
            k5_device_attn_graph_env_present,
        };
        use crate::layers::qwen4_verify_lanes::{ScopedEnvVar, test_env_lock};

        let _lock = test_env_lock();

        {
            let _g = ScopedEnvVar::remove(GQA12_RAGGED_SELECTOR);
            assert_eq!(gqa12_ragged_mode(), Gqa12RaggedMode::Auto);
            let _g1 = ScopedEnvVar::set(GQA12_RAGGED_SELECTOR, "1");
            assert_eq!(gqa12_ragged_mode(), Gqa12RaggedMode::ForceOn);
            let _g0 = ScopedEnvVar::set(GQA12_RAGGED_SELECTOR, "0");
            assert_eq!(gqa12_ragged_mode(), Gqa12RaggedMode::ForceOff);
        }
        {
            let _g = ScopedEnvVar::remove("ATLAS_QWEN4_K5_BATCH_ATTN_QKV");
            assert!(!k5_batch_attn_qkv_selected());
            let _g1 = ScopedEnvVar::set("ATLAS_QWEN4_K5_BATCH_ATTN_QKV", "1");
            assert!(k5_batch_attn_qkv_selected());
            let _g0 = ScopedEnvVar::set("ATLAS_QWEN4_K5_BATCH_ATTN_QKV", "0");
            assert!(!k5_batch_attn_qkv_selected());
        }
        {
            let _g = ScopedEnvVar::remove("ATLAS_QWEN4_K5_DEVICE_ATTN_GRAPH");
            assert!(!k5_device_attn_graph_env_present());
            let _g1 = ScopedEnvVar::set("ATLAS_QWEN4_K5_DEVICE_ATTN_GRAPH", "1");
            assert!(k5_device_attn_graph_env_present());
        }
    }
}

#[cfg(test)]
mod dynamic_sm_tests {
    use spark_runtime::kv_cache::KvCacheDtype;

    use super::{
        GQA12_MAX_PARTS, GQA12_MAX_PARTS_BF16_WIDE,
        gqa12_ragged_grid_and_workspace_bytes_for_sms_with_cap,
        gqa12_ragged_max_parts_for_kv_and_sm, qsa_seq_in_dense_prefix,
    };

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

    #[test]
    fn cross_seq_gqa12_ragged_preserves_gb10_and_caps_high_sm_partitions() {
        // 48-SM GB10 parity.
        assert_eq!(
            gqa12_ragged_grid_and_workspace_bytes_for_sms(24, 2, 256, 48).map(|(p, g, _)| (p, g)),
            Some((48, 2))
        );
        assert_eq!(
            gqa12_ragged_grid_and_workspace_bytes_for_sms(24, 4, 256, 48).map(|(p, g, _)| (p, g)),
            Some((24, 4))
        );
        // 188-SM RTX PRO 6000 scales 4-group and 8-group occupancy while staying <= GQA12_MAX_PARTS.
        assert_eq!(
            gqa12_ragged_grid_and_workspace_bytes_for_sms(24, 4, 256, 188).map(|(p, g, _)| (p, g)),
            Some((GQA12_MAX_PARTS, 4))
        );
        assert_eq!(
            gqa12_ragged_grid_and_workspace_bytes_for_sms(32, 8, 256, 188).map(|(p, g, _)| (p, g)),
            Some((47, 8))
        );
        // Extreme SM count never overflows GQA12_MAX_PARTS (48 <= 64 kernel cap).
        let (parts, _, _) = gqa12_ragged_grid_and_workspace_bytes_for_sms(24, 2, 256, 512).unwrap();
        assert_eq!(parts, GQA12_MAX_PARTS);

        // BF16 on >64 SMs uses 4-word ballot cap (128), lifting 24Q/2KV to 128*2 = 256 CTAs
        // and 24Q/4KV to 94*4 = 376 CTAs while keeping GB10 and FP8 at 64.
        assert_eq!(
            gqa12_ragged_max_parts_for_kv_and_sm(KvCacheDtype::Bf16, 48),
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
    }

    #[test]
    fn qsa_dense_prefix_boundary_matches_2048_endpoint() {
        assert!(qsa_seq_in_dense_prefix(false, 8192, 8));
        assert!(qsa_seq_in_dense_prefix(true, 0, 8));
        assert!(qsa_seq_in_dense_prefix(true, 2040, 8));
        assert!(!qsa_seq_in_dense_prefix(true, 2041, 8));
        assert!(!qsa_seq_in_dense_prefix(true, 2048, 2));
        assert!(!qsa_seq_in_dense_prefix(true, usize::MAX, 1));
    }

    #[test]
    #[ignore = "needs an idle GB10 GPU and the flash-next kernel target"]
    fn gqa12_ragged_bf16_128_parts_four_word_ballot_parity_on_gb10() {
        use crate::layers::ops;
        use half::bf16;
        use spark_runtime::cuda_backend::AtlasCudaBackend;
        use spark_runtime::gpu::{DevicePtr, GpuBackend};
        use spark_runtime::kernel_args::KernelLaunch;

        let ptx = atlas_kernels::ptx_for_model("flash-next").expect("flash-next PTX target");
        let gpu = AtlasCudaBackend::new(0, &ptx.modules).unwrap();
        let stream = gpu.default_stream();
        let main_k = gpu
            .kernel(
                "paged_decode_attn_gqa4",
                "paged_decode_attn_gqa12_ragged_bf16",
            )
            .unwrap();
        let red_k = gpu
            .kernel(
                "paged_decode_attn_gqa4",
                "paged_decode_attn_gqa12_ragged_reduce_bf16",
            )
            .unwrap();
        let red_gated_k = gpu
            .kernel(
                "paged_decode_attn_gqa4",
                "paged_decode_attn_gqa12_ragged_reduce_gated_bf16",
            )
            .unwrap();

        let num_q_heads = 24u32;
        let num_kv_heads = 2u32;
        let head_dim = 256u32;
        let block_size = 16u32;
        let q_stride = num_q_heads * head_dim;
        let seq_lens_host: [i32; 8] = [33, 64, 127, 256, 385, 512, 769, 1024];
        let num_seqs = seq_lens_host.len() as u32;
        let max_blocks_per_seq = 64u32; // 64 * 16 = 1024
        let total_blocks = (num_seqs * max_blocks_per_seq) as usize;
        let inv_sqrt_d = 1.0f32 / (head_dim as f32).sqrt();

        let lcg_bf16 = |len: usize, mut s: u64, scale: f32| -> Vec<u8> {
            let mut out = Vec::with_capacity(len * 2);
            for _ in 0..len {
                s = s
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let v = (((s >> 40) as f32) / ((1u32 << 24) as f32) * 2.0 - 1.0) * scale;
                out.extend_from_slice(&bf16::from_f32(v).to_le_bytes());
            }
            out
        };
        let upload = |bytes: &[u8]| -> DevicePtr {
            let p = gpu.alloc(bytes.len()).unwrap();
            gpu.copy_h2d(bytes, p).unwrap();
            p
        };
        let download = |p: DevicePtr, n: usize| -> Vec<u8> {
            gpu.synchronize(stream).unwrap();
            let mut out = vec![0u8; n];
            gpu.copy_d2h(p, &mut out).unwrap();
            out
        };

        let q_dev = upload(&lcg_bf16((num_seqs * q_stride) as usize, 11, 0.5));
        let gate_dev = upload(&lcg_bf16((num_seqs * q_stride) as usize, 22, 1.0));
        let kv_elems = total_blocks * (block_size * num_kv_heads * head_dim) as usize;
        let k_dev = upload(&lcg_bf16(kv_elems, 33, 0.4));
        let v_dev = upload(&lcg_bf16(kv_elems, 44, 0.4));
        let mut bt_bytes = Vec::with_capacity(total_blocks * 4);
        for blk in 0..total_blocks {
            bt_bytes.extend_from_slice(&(blk as i32).to_le_bytes());
        }
        let bt_dev = upload(&bt_bytes);
        let mut sl_bytes = Vec::with_capacity(seq_lens_host.len() * 4);
        for &l in &seq_lens_host {
            sl_bytes.extend_from_slice(&l.to_le_bytes());
        }
        let sl_dev = upload(&sl_bytes);

        let out_bytes = (num_seqs * q_stride) as usize * 2;
        let (_, total_groups, max_ws_bytes) =
            gqa12_ragged_grid_and_workspace_bytes_for_sms_with_cap(
                num_q_heads,
                num_kv_heads,
                head_dim,
                188,
                GQA12_MAX_PARTS_BF16_WIDE,
            )
            .unwrap();
        let ws_dev = gpu.alloc(max_ws_bytes).unwrap();
        let out_dev = gpu.alloc(out_bytes).unwrap();

        let run_case = |num_parts: u32, gated: bool| -> Vec<u8> {
            gpu.memset(ws_dev, 0, max_ws_bytes).unwrap();
            gpu.memset(out_dev, 0, out_bytes).unwrap();
            ops::paged_decode_attn_gqa12_ragged_bf16(
                &gpu,
                main_k,
                q_dev,
                k_dev,
                v_dev,
                out_dev,
                ws_dev,
                bt_dev,
                sl_dev,
                max_blocks_per_seq,
                num_seqs,
                num_q_heads,
                num_kv_heads,
                head_dim,
                block_size,
                inv_sqrt_d,
                q_stride,
                0,
                num_parts,
                total_groups,
                stream,
            )
            .unwrap();
            if gated {
                KernelLaunch::new(&gpu, red_gated_k)
                    .grid([num_q_heads, num_seqs, 1])
                    .block([32, 1, 1])
                    .arg_ptr(ws_dev)
                    .arg_ptr(out_dev)
                    .arg_ptr(gate_dev)
                    .arg_ptr(sl_dev)
                    .arg_u32(num_seqs)
                    .arg_u32(num_q_heads)
                    .arg_u32(num_kv_heads)
                    .arg_u32(num_parts)
                    .arg_u32(0)
                    .arg_u32(q_stride)
                    .launch(stream)
                    .unwrap();
            } else {
                ops::paged_decode_attn_gqa12_ragged_reduce(
                    &gpu,
                    red_k,
                    ws_dev,
                    out_dev,
                    sl_dev,
                    num_seqs,
                    num_q_heads,
                    num_kv_heads,
                    num_parts,
                    0,
                    stream,
                )
                .unwrap();
            }
            download(out_dev, out_bytes)
        };

        for gated in [false, true] {
            let ref_out = run_case(1, gated);
            for &parts in &[48u32, 64, 65, 96, 128] {
                let got = run_case(parts, gated);
                let mut max_abs = 0.0f32;
                let mut max_ulp_sig = 0u32;
                for (ca, cb) in got.chunks_exact(2).zip(ref_out.chunks_exact(2)) {
                    let ua = u16::from_le_bytes([ca[0], ca[1]]);
                    let ub = u16::from_le_bytes([cb[0], cb[1]]);
                    let fa = bf16::from_bits(ua).to_f32();
                    let fb = bf16::from_bits(ub).to_f32();
                    max_abs = max_abs.max((fa - fb).abs());
                    if fa.abs().max(fb.abs()) >= 1e-2 && (ua >> 15) == (ub >> 15) {
                        max_ulp_sig = max_ulp_sig.max(u32::from(ua.abs_diff(ub)));
                    }
                }
                eprintln!(
                    "gqa12_ragged_bf16 gated={gated} parts={parts}: max_abs={max_abs:.3e} max_ulp_sig={max_ulp_sig}"
                );
                assert!(
                    max_abs <= 1e-3 && max_ulp_sig <= 1,
                    "gated={gated} parts={parts} failed: max_abs={max_abs} max_ulp_sig={max_ulp_sig}"
                );
            }
        }
    }

    #[test]
    #[ignore = "needs an idle GB10 GPU and the flash-next kernel target"]
    fn gqa12_ragged_bf16_and_fp8_256_seqs_single_launch_parity_on_gb10() {
        use crate::layers::ops;
        use half::bf16;
        use spark_runtime::cuda_backend::AtlasCudaBackend;
        use spark_runtime::gpu::{DevicePtr, GpuBackend};
        use spark_runtime::kernel_args::KernelLaunch;

        let ptx = atlas_kernels::ptx_for_model("flash-next").expect("flash-next PTX target");
        let gpu = AtlasCudaBackend::new(0, &ptx.modules).unwrap();
        let stream = gpu.default_stream();
        let main_bf16_k = gpu
            .kernel(
                "paged_decode_attn_gqa4",
                "paged_decode_attn_gqa12_ragged_bf16",
            )
            .unwrap();
        let red_bf16_k = gpu
            .kernel(
                "paged_decode_attn_gqa4",
                "paged_decode_attn_gqa12_ragged_reduce_bf16",
            )
            .unwrap();
        let red_gated_k = gpu
            .kernel(
                "paged_decode_attn_gqa4",
                "paged_decode_attn_gqa12_ragged_reduce_gated_bf16",
            )
            .unwrap();
        let main_fp8_k = gpu
            .kernel("paged_decode_fp8", "paged_decode_attn_gqa12_ragged_fp8")
            .unwrap();
        let red_fp8_k = gpu
            .kernel(
                "paged_decode_fp8",
                "paged_decode_attn_gqa12_ragged_reduce_fp8",
            )
            .unwrap();

        let num_q_heads = 24u32;
        let num_kv_heads = 2u32;
        let head_dim = 256u32;
        let block_size = 16u32;
        let q_stride = num_q_heads * head_dim;
        let inv_sqrt_d = 1.0f32 / (head_dim as f32).sqrt();
        let max_blocks_per_seq = 16u32; // up to 256 tokens per seq
        let kv_block_elems = (block_size * num_kv_heads * head_dim) as usize;
        // Share a 64-block KV pool across all 256 sequences via ragged block tables.
        let pool_blocks = 64usize;

        let lcg_bf16 = |len: usize, mut s: u64, scale: f32| -> Vec<u8> {
            let mut out = Vec::with_capacity(len * 2);
            for _ in 0..len {
                s = s
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let v = (((s >> 40) as f32) / ((1u32 << 24) as f32) * 2.0 - 1.0) * scale;
                out.extend_from_slice(&bf16::from_f32(v).to_le_bytes());
            }
            out
        };
        let lcg_fp8_e4m3 = |len: usize, mut s: u64| -> Vec<u8> {
            // Finite E4M3 codes (sign bit 0/1, exp 1..=10, mantissa 0..=7) -> safe non-NaN values.
            let mut out = Vec::with_capacity(len);
            for _ in 0..len {
                s = s
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let sign = ((s >> 63) as u8) << 7;
                let exp = ((((s >> 56) as u8) % 8) + 2) << 3;
                let mant = ((s >> 52) as u8) & 0x7;
                out.push(sign | exp | mant);
            }
            out
        };
        let upload = |bytes: &[u8]| -> DevicePtr {
            let p = gpu.alloc(bytes.len()).unwrap();
            gpu.copy_h2d(bytes, p).unwrap();
            p
        };
        let download = |p: DevicePtr, n: usize| -> Vec<u8> {
            gpu.synchronize(stream).unwrap();
            let mut out = vec![0u8; n];
            gpu.copy_d2h(p, &mut out).unwrap();
            out
        };

        let k_bf16_dev = upload(&lcg_bf16(pool_blocks * kv_block_elems, 33, 0.35));
        let v_bf16_dev = upload(&lcg_bf16(pool_blocks * kv_block_elems, 44, 0.35));
        let k_fp8_dev = upload(&lcg_fp8_e4m3(pool_blocks * kv_block_elems, 55));
        let v_fp8_dev = upload(&lcg_fp8_e4m3(pool_blocks * kv_block_elems, 66));

        let (_, total_groups, max_ws_bytes) =
            gqa12_ragged_grid_and_workspace_bytes_for_sms_with_cap(
                num_q_heads,
                num_kv_heads,
                head_dim,
                188,
                GQA12_MAX_PARTS_BF16_WIDE,
            )
            .unwrap();
        let ws_dev = gpu.alloc(max_ws_bytes).unwrap();

        for &num_seqs in &[128u32, 256u32] {
            let q_dev = upload(&lcg_bf16(
                (num_seqs * q_stride) as usize,
                11 + u64::from(num_seqs),
                0.5,
            ));
            let gate_dev = upload(&lcg_bf16(
                (num_seqs * q_stride) as usize,
                22 + u64::from(num_seqs),
                1.0,
            ));
            let mut bt_bytes = Vec::with_capacity((num_seqs * max_blocks_per_seq) as usize * 4);
            let mut sl_bytes = Vec::with_capacity(num_seqs as usize * 4);
            for s in 0..num_seqs {
                // Ragged lengths covering 1 tile, partial tiles, multi-block, and boundary rows (63/64/127/128/255).
                let len = match s % 8 {
                    0 => 17i32,
                    1 => 32,
                    2 => 33,
                    3 => 64,
                    4 => 97,
                    5 => 128,
                    6 => 193,
                    _ => 256,
                };
                sl_bytes.extend_from_slice(&len.to_le_bytes());
                for b in 0..max_blocks_per_seq {
                    let blk = ((s * 7 + b * 3) as usize % pool_blocks) as i32;
                    bt_bytes.extend_from_slice(&blk.to_le_bytes());
                }
            }
            let bt_dev = upload(&bt_bytes);
            let sl_dev = upload(&sl_bytes);
            let out_bytes = (num_seqs * q_stride) as usize * 2;
            let out_dev = gpu.alloc(out_bytes).unwrap();

            let check_parity = |label: &str, got: &[u8], reference: &[u8]| {
                let mut max_abs = 0.0f32;
                let mut max_ulp_sig = 0u32;
                for (ca, cb) in got.chunks_exact(2).zip(reference.chunks_exact(2)) {
                    let ua = u16::from_le_bytes([ca[0], ca[1]]);
                    let ub = u16::from_le_bytes([cb[0], cb[1]]);
                    let fa = bf16::from_bits(ua).to_f32();
                    let fb = bf16::from_bits(ub).to_f32();
                    max_abs = max_abs.max((fa - fb).abs());
                    if fa.abs().max(fb.abs()) >= 1e-2 && (ua >> 15) == (ub >> 15) {
                        max_ulp_sig = max_ulp_sig.max(u32::from(ua.abs_diff(ub)));
                    }
                }
                eprintln!("{label}: max_abs={max_abs:.3e} max_ulp_sig={max_ulp_sig}");
                assert!(
                    max_abs <= 1e-3 && max_ulp_sig <= 1,
                    "{label} failed: max_abs={max_abs} max_ulp_sig={max_ulp_sig}"
                );
            };

            let run_bf16 = |num_parts: u32, gated: bool| -> Vec<u8> {
                gpu.memset(ws_dev, 0, max_ws_bytes).unwrap();
                gpu.memset(out_dev, 0, out_bytes).unwrap();
                ops::paged_decode_attn_gqa12_ragged_bf16(
                    &gpu,
                    main_bf16_k,
                    q_dev,
                    k_bf16_dev,
                    v_bf16_dev,
                    out_dev,
                    ws_dev,
                    bt_dev,
                    sl_dev,
                    max_blocks_per_seq,
                    num_seqs,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    block_size,
                    inv_sqrt_d,
                    q_stride,
                    0,
                    num_parts,
                    total_groups,
                    stream,
                )
                .unwrap();
                if gated {
                    KernelLaunch::new(&gpu, red_gated_k)
                        .grid([num_q_heads, num_seqs, 1])
                        .block([32, 1, 1])
                        .arg_ptr(ws_dev)
                        .arg_ptr(out_dev)
                        .arg_ptr(gate_dev)
                        .arg_ptr(sl_dev)
                        .arg_u32(num_seqs)
                        .arg_u32(num_q_heads)
                        .arg_u32(num_kv_heads)
                        .arg_u32(num_parts)
                        .arg_u32(0)
                        .arg_u32(q_stride)
                        .launch(stream)
                        .unwrap();
                } else {
                    ops::paged_decode_attn_gqa12_ragged_reduce(
                        &gpu,
                        red_bf16_k,
                        ws_dev,
                        out_dev,
                        sl_dev,
                        num_seqs,
                        num_q_heads,
                        num_kv_heads,
                        num_parts,
                        0,
                        stream,
                    )
                    .unwrap();
                }
                download(out_dev, out_bytes)
            };

            for gated in [false, true] {
                let ref_bf16 = run_bf16(1, gated);
                for &parts in &[48u32, 64, 128] {
                    let got = run_bf16(parts, gated);
                    check_parity(
                        &format!("gqa12_bf16 seqs={num_seqs} gated={gated} parts={parts}"),
                        &got,
                        &ref_bf16,
                    );
                }
            }

            let run_fp8 = |num_parts: u32| -> Vec<u8> {
                gpu.memset(ws_dev, 0, max_ws_bytes).unwrap();
                gpu.memset(out_dev, 0, out_bytes).unwrap();
                ops::paged_decode_attn_gqa12_ragged_fp8(
                    &gpu,
                    main_fp8_k,
                    q_dev,
                    k_fp8_dev,
                    v_fp8_dev,
                    out_dev,
                    ws_dev,
                    bt_dev,
                    sl_dev,
                    max_blocks_per_seq,
                    num_seqs,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    block_size,
                    inv_sqrt_d,
                    0.5,
                    0.5,
                    q_stride,
                    kv_block_elems as u64,
                    0,
                    num_parts,
                    total_groups,
                    stream,
                )
                .unwrap();
                ops::paged_decode_attn_gqa12_ragged_reduce(
                    &gpu,
                    red_fp8_k,
                    ws_dev,
                    out_dev,
                    sl_dev,
                    num_seqs,
                    num_q_heads,
                    num_kv_heads,
                    num_parts,
                    0,
                    stream,
                )
                .unwrap();
                download(out_dev, out_bytes)
            };

            let ref_fp8 = run_fp8(1);
            for &parts in &[48u32, 64] {
                let got = run_fp8(parts);
                check_parity(
                    &format!("gqa12_fp8 seqs={num_seqs} parts={parts}"),
                    &got,
                    &ref_fp8,
                );
            }
        }
    }
}
