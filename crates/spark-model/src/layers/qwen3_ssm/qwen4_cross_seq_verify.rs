// SPDX-License-Identifier: AGPL-3.0-only

//! Cross-sequence Qwen4 K-row verify for one GDN layer
//! (`ATLAS_QWEN4_BATCHED_K5`).
//!
//! The single-sequence K5 verifier (`ssm_forward_qwen4_exact_rows`, legacy
//! K5/K9 route) runs, for its K rows: attention-hyperconnection prep, exact
//! QKVZ projection, exact BA gates, the ordered conv sequence kernel, the
//! ordered GDN sequence kernel (lazyfinal / persistent / snapshot), the gated
//! RMS norm, the exact output projection and the saved-scale inject.
//!
//! This frame runs the row-parallel stages once over every row of every
//! sequence (the exact tiers are K1-lane-exact per row) and the two ordered
//! recurrent stages plus the gated norm per sequence, with the same kernels,
//! the same row count K, the sequence's own live state and intermediates,
//! and its own GDN retain lane. Kernel choice is decided by the same
//! predicates the single-sequence verifier and the async commit use, with
//! the per-sequence width K.

use anyhow::Context;

use super::qwen4_k5_ssm::{
    Qwen4ExactSsmRowsRoute, qwen4_exact_ssm_rows_route, qwen4_k16_batched_verify_requested,
    qwen4_k16_exact_requested, qwen4_verify_gnorm_rows_selected,
};
use super::*;
use crate::layer::{Qwen4CrossSeqFrame, Qwen4CrossSeqRows};

const BF16: usize = 2;
const FP32: usize = 4;

/// SM count threshold above which cross-sequence verify routes `ssm.in_proj_ba`
/// (`rows >= 32`) through cuBLASLt BF16 -> FP32 GEMM + `qwen4_ba_gates_from_f32_v2`
/// instead of scalar `dense_gemv_ba_gates_batchn` (GB10 `<= 64` SMs stays exact).
const VERIFY_BA_GEMM_HIGH_SM_THRESHOLD: u32 = 64;
const VERIFY_BA_GEMM_MIN_ROWS: usize = 32;
const VERIFY_BA_GEMM_SCRATCH_MIN_ROWS: usize = 256;

#[expect(
    clippy::too_many_arguments,
    reason = "pure const policy predicate mirroring every cross-seq verify BA GEMM check"
)]
#[inline]
#[must_use]
const fn cross_seq_verify_ba_gemm_eligible(
    tc_dense_scope: bool,
    rows: usize,
    sm_count: u32,
    hidden_size: usize,
    ba_size: usize,
    nk: usize,
    nv: usize,
    input_ptr: u64,
    weight_ptr: u64,
    gates_ptr: u64,
) -> bool {
    tc_dense_scope
        && sm_count > VERIFY_BA_GEMM_HIGH_SM_THRESHOLD
        && rows >= VERIFY_BA_GEMM_MIN_ROWS
        && hidden_size > 0
        && hidden_size.is_multiple_of(8)
        && nk > 0
        && nv > 0
        && nv.is_multiple_of(nk)
        && ba_size == nv * 2
        && input_ptr != 0
        && weight_ptr != 0
        && gates_ptr != 0
        && input_ptr.is_multiple_of(16)
        && weight_ptr.is_multiple_of(16)
        && gates_ptr.is_multiple_of(4)
}

fn verify_ba_gates_kernel(gpu: &dyn GpuBackend) -> Option<KernelHandle> {
    use std::collections::HashMap;
    use std::sync::OnceLock;

    let key = std::ptr::from_ref(gpu) as *const () as usize;
    static CACHE: OnceLock<parking_lot::Mutex<HashMap<usize, KernelHandle>>> = OnceLock::new();
    let map = CACHE.get_or_init(|| parking_lot::Mutex::new(HashMap::new()));
    if let Some(&cached) = map.lock().get(&key) {
        return (cached.0 != 0).then_some(cached);
    }
    let v2 = crate::layers::try_kernel(gpu, "qwen4_wide_glue", "qwen4_ba_gates_from_f32_v2");
    let handle = if v2.0 != 0 {
        v2
    } else {
        crate::layers::try_kernel(gpu, "qwen4_wide_glue", "qwen4_ba_gates_from_f32")
    };
    map.lock().insert(key, handle);
    (handle.0 != 0).then_some(handle)
}

/// Graph-capture-safe FP32 `[rows, ba_size]` scratch for cross-sequence verify
/// BA GEMM. Pre-allocates at least `256` rows outside capture and refuses to
/// allocate during CUDA graph capture so capture never fails.
fn ensure_verify_ba_f32_scratch(
    gpu: &dyn GpuBackend,
    rows: usize,
    ba_size: usize,
    capturing: bool,
) -> Result<Option<DevicePtr>> {
    use std::collections::HashMap;
    use std::sync::Mutex;

    let Some(need) = rows.checked_mul(ba_size).and_then(|n| n.checked_mul(FP32)) else {
        return Ok(None);
    };
    let key = std::ptr::from_ref(gpu) as *const () as usize;
    static SCRATCH: Mutex<Option<HashMap<usize, (DevicePtr, usize)>>> = Mutex::new(None);
    let mut guard = SCRATCH
        .lock()
        .expect("cross-seq verify BA GEMM scratch lock poisoned");
    let map = guard.get_or_insert_with(HashMap::new);
    if let Some(&(ptr, cap)) = map.get(&key)
        && cap >= need
        && !ptr.is_null()
    {
        return Ok(Some(ptr));
    }
    if capturing || spark_runtime::gpu_capture_region::active_capture_region().is_some() {
        return Ok(None);
    }
    let alloc_rows = rows.max(VERIFY_BA_GEMM_SCRATCH_MIN_ROWS);
    let cap = alloc_rows
        .saturating_mul(ba_size)
        .saturating_mul(FP32)
        .next_multiple_of(1 << 20);
    let ptr = gpu
        .alloc(cap)
        .with_context(|| format!("cross-seq verify BA GEMM scratch alloc {cap} B"))?;
    spark_runtime::scratch_init::init_grown(gpu, ptr, cap, "ssm_verify_ba_f32")?;
    map.insert(key, (ptr, cap));
    Ok(Some(ptr))
}

/// Maximum concurrent sequences (`C = 1..=4`) that engage Split-V (`SPLIT_V = 4`)
/// multi-CTA fused GDN verify on wide GPUs (`> 64` SMs: `C=2 -> 384 CTAs`,
/// `C=4 -> 768 CTAs`; `C >= 5` already fields `>= 240 CTAs` without splitting V).
const GDN_MULTI_SEQ_SPLIT_V_MAX_SEQS: usize = 4;

/// Resolved Split-V dispatch choice for one multi-sequence fused GDN verify launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CrossSeqSplitV4Plan {
    Disabled,
    Fp32Only,
    ScaledFp16Capable,
}

#[inline]
#[must_use]
const fn plan_cross_seq_splitv4_with_policy(
    split_v_enabled: bool,
    num_seqs: usize,
    lazy: bool,
    needs_scaled_fp16: bool,
    has_fp32_kernel: bool,
    has_scaled_fp16_kernel: bool,
    det_batch: crate::layers::deterministic_batch::DeterministicBatchPolicy,
) -> CrossSeqSplitV4Plan {
    if det_batch.is_enabled() {
        return CrossSeqSplitV4Plan::Disabled;
    }
    plan_cross_seq_splitv4(
        split_v_enabled,
        num_seqs,
        lazy,
        needs_scaled_fp16,
        has_fp32_kernel,
        has_scaled_fp16_kernel,
    )
}

#[inline]
#[must_use]
const fn plan_cross_seq_splitv4(
    split_v_enabled: bool,
    num_seqs: usize,
    lazy: bool,
    needs_scaled_fp16: bool,
    has_fp32_kernel: bool,
    has_scaled_fp16_kernel: bool,
) -> CrossSeqSplitV4Plan {
    if !split_v_enabled || num_seqs == 0 || num_seqs > GDN_MULTI_SEQ_SPLIT_V_MAX_SEQS {
        return CrossSeqSplitV4Plan::Disabled;
    }
    if needs_scaled_fp16 {
        if lazy && has_scaled_fp16_kernel {
            CrossSeqSplitV4Plan::ScaledFp16Capable
        } else {
            CrossSeqSplitV4Plan::Disabled
        }
    } else if has_fp32_kernel {
        CrossSeqSplitV4Plan::Fp32Only
    } else {
        CrossSeqSplitV4Plan::Disabled
    }
}

/// Byte layout inside `ssm_conv_out_f32` for multi-sequence Scaled-FP16
/// `Split-V=4` GDN verify (`scratch_gdn` + `scratch_ss` + L2-staged
/// `scratch_h_flush` + `scratch_amax`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GdnSplitVScaledFp16ScratchLayout {
    ss_offset_bytes: usize,
    h_flush_offset_bytes: usize,
    amax_offset_bytes: usize,
    total_bytes: usize,
}

#[must_use]
const fn gdn_split_v_scaled_fp16_scratch_layout(
    total_rows: usize,
    num_seqs: usize,
    num_v_heads: usize,
) -> Option<GdnSplitVScaledFp16ScratchLayout> {
    const HEAD_ELEMS: usize = 128 * 128;
    const ALIGN_BYTES: usize = 256;
    let Some(base) = ops::gdn_split_v_scratch_layout(total_rows, num_v_heads) else {
        return None;
    };
    let rem = base.total_bytes % ALIGN_BYTES;
    let pad = if rem == 0 { 0 } else { ALIGN_BYTES - rem };
    let Some(h_flush_offset_bytes) = base.total_bytes.checked_add(pad) else {
        return None;
    };
    let Some(total_heads) = num_seqs.checked_mul(num_v_heads) else {
        return None;
    };
    let Some(h_flush_elems) = total_heads.checked_mul(HEAD_ELEMS) else {
        return None;
    };
    let Some(h_flush_bytes) = h_flush_elems.checked_mul(FP32) else {
        return None;
    };
    let Some(amax_offset_bytes) = h_flush_offset_bytes.checked_add(h_flush_bytes) else {
        return None;
    };
    let Some(amax_elems) = total_heads.checked_mul(ops::GDN_SPLIT_V_FACTOR as usize) else {
        return None;
    };
    let Some(amax_bytes) = amax_elems.checked_mul(FP32) else {
        return None;
    };
    let Some(total_bytes) = amax_offset_bytes.checked_add(amax_bytes) else {
        return None;
    };
    Some(GdnSplitVScaledFp16ScratchLayout {
        ss_offset_bytes: base.ss_offset_bytes,
        h_flush_offset_bytes,
        amax_offset_bytes,
        total_bytes,
    })
}

struct GdnScaledFp16SplitV4KernelCache {
    primary: std::sync::OnceLock<(usize, KernelHandle)>,
    fallback: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<usize, KernelHandle>>>,
}

impl GdnScaledFp16SplitV4KernelCache {
    const fn new() -> Self {
        Self {
            primary: std::sync::OnceLock::new(),
            fallback: std::sync::OnceLock::new(),
        }
    }

    fn resolve(&self, gpu: &dyn GpuBackend) -> KernelHandle {
        let key = (gpu as *const dyn GpuBackend as *const ()) as usize;
        if let Some(&(cached_key, handle)) = self.primary.get()
            && cached_key == key
        {
            return handle;
        }
        let lookup = || {
            crate::layers::try_kernel(
                gpu,
                "qwen4_gdn_verify_fused",
                "qwen4_gdn_verify_fused_lazyfinal_multi_seq_scaled_fp16_splitv4",
            )
        };
        if self.primary.get().is_none() {
            let handle = lookup();
            let _ = self.primary.set((key, handle));
            return handle;
        }
        let map = self
            .fallback
            .get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
        let mut guard = map
            .lock()
            .expect("GDN scaled_fp16_splitv4 kernel cache lock poisoned");
        *guard.entry(key).or_insert_with(lookup)
    }
}

fn qwen4_gdn_scaled_fp16_splitv4_kernel(gpu: &dyn GpuBackend) -> KernelHandle {
    static CACHE: GdnScaledFp16SplitV4KernelCache = GdnScaledFp16SplitV4KernelCache::new();
    CACHE.resolve(gpu)
}

#[expect(
    clippy::too_many_arguments,
    reason = "matches CUDA kernel ABI directly"
)]
fn launch_gdn_verify_fused_lazyfinal_multi_seq_scaled_fp16_splitv4(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    seq_descs: DevicePtr,
    deinterleaved: DevicePtr,
    conv_weight: &crate::weight_map::DenseWeight,
    gate: DevicePtr,
    beta: DevicePtr,
    rms_weight: &crate::weight_map::DenseWeight,
    normed_out: DevicePtr,
    num_seqs: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    l2_eps: f32,
    rms_eps: f32,
    input_stride: u32,
    gate_beta_stride: u32,
    normed_stride: u32,
    retain_stride: u32,
    conv_inter_stride: u32,
    h_inter_stride: u32,
    scratch_gdn: DevicePtr,
    scratch_ss: DevicePtr,
    split_done: DevicePtr,
    scratch_h_flush: DevicePtr,
    scratch_amax: DevicePtr,
    stream: u64,
) -> Result<()> {
    spark_runtime::kernel_args::KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads, ops::GDN_SPLIT_V_FACTOR, num_seqs])
        .block([128, 1, 1])
        .arg_ptr(seq_descs)
        .arg_ptr(deinterleaved)
        .arg_ptr(conv_weight.weight)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(rms_weight.weight)
        .arg_ptr(normed_out)
        .arg_u32(num_seqs)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_f32(l2_eps)
        .arg_f32(rms_eps)
        .arg_u32(input_stride)
        .arg_u32(gate_beta_stride)
        .arg_u32(normed_stride)
        .arg_u32(retain_stride)
        .arg_u32(conv_inter_stride)
        .arg_u32(h_inter_stride)
        .arg_ptr(scratch_gdn)
        .arg_ptr(scratch_ss)
        .arg_ptr(split_done)
        .arg_ptr(scratch_h_flush)
        .arg_ptr(scratch_amax)
        .launch(stream)
}

fn checked_rows_bytes(rows: usize, width: usize, element: usize, name: &str) -> Result<usize> {
    rows.checked_mul(width)
        .and_then(|elements| elements.checked_mul(element))
        .ok_or_else(|| anyhow::anyhow!("cross-sequence SSM {name} extent overflow"))
}

#[inline]
#[must_use]
const fn ssm_tc_dense_fp8_shape_valid(rows: usize, n: usize, k: usize) -> bool {
    rows > 0 && n > 0 && k > 0 && n.is_multiple_of(16) && k.is_multiple_of(16)
}

fn ssm_tc_dense_fp8_available(
    gpu: &dyn GpuBackend,
    weight: &QuantizedWeight,
    rows: usize,
    n: usize,
    k: usize,
) -> bool {
    if !ssm_tc_dense_fp8_shape_valid(rows, n, k)
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

/// `ATLAS_QWEN4_K5_BATCH_SSM=1` (cached via `OnceLock` outside unit tests).
#[inline]
fn qwen4_k5_batch_ssm_selected() -> bool {
    let read_live = || std::env::var("ATLAS_QWEN4_K5_BATCH_SSM").ok().as_deref() == Some("1");
    #[cfg(not(test))]
    {
        static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *GATE.get_or_init(read_live)
    }
    #[cfg(test)]
    {
        read_live()
    }
}

/// Cached wrapper around [`qwen4_k16_batched_verify_requested`] (preserves
/// fail-closed parse errors while eliminating per-layer/per-seq env syscalls).
#[inline]
fn qwen4_k16_batched_verify_cached() -> Result<bool> {
    let read_live = || qwen4_k16_batched_verify_requested().map_err(|err| format!("{err:#}"));
    #[cfg(not(test))]
    {
        static CACHED: std::sync::OnceLock<std::result::Result<bool, String>> =
            std::sync::OnceLock::new();
        CACHED
            .get_or_init(read_live)
            .as_ref()
            .copied()
            .map_err(|msg| anyhow::Error::msg(msg.clone()))
    }
    #[cfg(test)]
    {
        read_live().map_err(anyhow::Error::msg)
    }
}

/// Cached wrapper around [`qwen4_k16_exact_requested`] (preserves fail-closed
/// parse errors while eliminating per-layer/per-seq env syscalls).
#[inline]
fn qwen4_k16_exact_cached() -> Result<bool> {
    let read_live = || qwen4_k16_exact_requested().map_err(|err| format!("{err:#}"));
    #[cfg(not(test))]
    {
        static CACHED: std::sync::OnceLock<std::result::Result<bool, String>> =
            std::sync::OnceLock::new();
        CACHED
            .get_or_init(read_live)
            .as_ref()
            .copied()
            .map_err(|msg| anyhow::Error::msg(msg.clone()))
    }
    #[cfg(test)]
    {
        read_live().map_err(anyhow::Error::msg)
    }
}

/// `ATLAS_QWEN4_VERIFY_DIRECT_RETAIN != "0"` (default ON for direct retain
/// staging, cached via `OnceLock` outside unit tests). Separate from
/// `ATLAS_QWEN4_VERIFY_FAST_HOST` (which is opt-in / default OFF across the
/// codebase) so direct-retain staging has one unambiguous default polarity.
#[inline]
pub(super) fn qwen4_verify_direct_retain_enabled() -> bool {
    let read_live = || {
        std::env::var("ATLAS_QWEN4_VERIFY_DIRECT_RETAIN")
            .ok()
            .as_deref()
            != Some("0")
    };
    #[cfg(not(test))]
    {
        static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *GATE.get_or_init(read_live)
    }
    #[cfg(test)]
    {
        read_live()
    }
}

/// `ATLAS_SSM_GDN_SEQ_PERSISTENT != "0"` (default ON, cached via `OnceLock`
/// outside unit tests).
#[inline]
pub(super) fn ssm_gdn_seq_persistent_enabled() -> bool {
    let read_live = || {
        std::env::var("ATLAS_SSM_GDN_SEQ_PERSISTENT")
            .ok()
            .as_deref()
            != Some("0")
    };
    #[cfg(not(test))]
    {
        static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *GATE.get_or_init(read_live)
    }
    #[cfg(test)]
    {
        read_live()
    }
}

impl Qwen3SsmLayer {
    /// Route the single-sequence verifier would take for one K-row frame.
    /// The cross-sequence frame admits only the legacy exact K5/K9 rows route
    /// (plus K=1 anchor-only slots inside ragged frames via `qwen4_hybrid_cross_seq_rows`).
    fn cross_seq_single_route(&self, k: usize) -> Result<Option<Qwen4ExactSsmRowsRoute>> {
        let hybrid = crate::layers::qwen4_hybrid_cross_seq_rows(k);
        let legacy_batch_ssm = hybrid && qwen4_k5_batch_ssm_selected();
        Ok(qwen4_exact_ssm_rows_route(
            k,
            hybrid,
            legacy_batch_ssm,
            qwen4_k16_batched_verify_cached()?,
            qwen4_k16_exact_cached()?,
            true,
        ))
    }

    pub(super) fn preflight_qwen4_cross_seq_mixer_only_inner(
        &self,
        frame: Qwen4CrossSeqFrame,
        ctx: &ForwardContext,
    ) -> Result<()> {
        ensure!(
            self.qwen4_attn_hyper.is_some() && self.qwen4_mlp_hyper.is_some(),
            "cross-sequence Qwen4 SSM verify requires both hyperconnections"
        );
        let nk = ctx.config.linear_num_key_heads;
        let kd = ctx.config.linear_key_head_dim;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let rows = frame.rows();
        let fused_graph_safe = !ctx.graph_capture
            || self
                .cross_seq_fused_multi_gdn_kernel(frame, (nk, kd, nv, vd), ctx)
                .is_some();
        ensure!(
            ctx.config.is_qwen4_exp()
                && fused_graph_safe
                && ctx.config.tp_world_size <= 4
                && (ctx.config.ep_world_size <= 1
                    || ctx.config.ep_world_size == ctx.config.tp_world_size)
                && ctx.ddtree_parent_ids_dev.is_none()
                && ctx.tree_aware_attn.is_none()
                && ctx.ffn_defer.is_none()
                && ctx.self_spec_sparse_draft.is_none(),
            "cross-sequence Qwen4 SSM verify is flat-chain TP<=4 (+ matching EP) (and requires fused multi-seq GDN under graph capture)"
        );
        let mut checked_ks = 0u16;
        for s in 0..frame.seqs() {
            let seq_k = frame.seq_k(s);
            let k_bit = if seq_k < 16 { 1u16 << seq_k } else { 0 };
            if k_bit != 0 && (checked_ks & k_bit) != 0 {
                continue;
            }
            ensure!(
                self.cross_seq_single_route(seq_k)? == Some(Qwen4ExactSsmRowsRoute::LegacyK5OrK9),
                "cross-sequence Qwen4 SSM verify seq {s} K{seq_k} requires the exact K5/K9 rows route \
                 (ATLAS_QWEN4_K5_HYBRID=1, ATLAS_QWEN4_K5_BATCH_SSM=1)"
            );
            let seq_lazy = self.gdn_seq_lazy_engaged_inner(seq_k);
            ensure!(
                !seq_lazy || self.gdn_f32_sequence_lazyfinal_k.0 != 0,
                "cross-sequence Qwen4 SSM lazy verify seq {s} K{seq_k} is missing its lazyfinal kernel"
            );
            checked_ks |= k_bit;
        }
        ensure!(
            self.sequential_qkvz && self.qkvz_nvfp4.is_some(),
            "cross-sequence Qwen4 SSM verify requires sequential NVFP4 QKVZ"
        );
        ensure!(
            self.ba_gates_batchn_exact_k.0 != 0
                && self.conv1d_l2norm_f32_sequence_k.0 != 0
                && self.gdn_f32_sequence_k.0 != 0
                && self.gated_rms_norm_f32_k.0 != 0,
            "cross-sequence Qwen4 SSM verify kernels are incomplete"
        );
        let attn_hyper = self
            .qwen4_attn_hyper
            .as_ref()
            .expect("checked hyperconnections above");
        attn_hyper.preflight_qwen4_native_batched(rows, true)?;

        let h = ctx.config.hidden_size;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let qkvz_size = ctx.config.ssm_qkvz_size();
        let sizes = ctx.buffers.sizes();
        ensure!(
            checked_rows_bytes(rows, h, BF16, "attention input")? <= sizes.gate_logits
                && checked_rows_bytes(rows, qkvz_size, BF16, "deinterleaved")?
                    <= sizes.ssm_deinterleaved
                && checked_rows_bytes(rows, nv * 2, FP32, "gate")? <= sizes.ssm_gates
                && checked_rows_bytes(rows, qkvz_size, FP32, "conv/GDN")? <= sizes.ssm_conv_out_f32
                && checked_rows_bytes(rows, nv * vd, BF16, "normalization")? <= sizes.ssm_qkvz
                && checked_rows_bytes(rows, h, BF16, "output")? <= sizes.moe_output,
            "cross-sequence Qwen4 SSM verify rows exceed the serving arenas"
        );
        Ok(())
    }

    pub(super) fn preflight_qwen4_cross_seq_verify_inner(
        &self,
        frame: Qwen4CrossSeqFrame,
        ctx: &ForwardContext,
    ) -> Result<()> {
        self.preflight_qwen4_cross_seq_mixer_only_inner(frame, ctx)?;
        crate::layers::qwen4_cross_seq_mlp::preflight(
            self.qwen4_mlp_hyper.as_ref(),
            &self.ffn,
            frame,
            ctx,
        )
    }

    /// On high-SM GPUs (`sm_count > 64`) at `rows >= 32`, compute the row-interleaved
    /// `[gate(nv) | beta(nv)]` FP32 GDN gates via cuBLASLt BF16 -> FP32 GEMM +
    /// `qwen4_ba_gates_from_f32_v2` instead of scalar `dense_gemv_ba_gates_batchn`.
    /// Returns `Ok(false)` when ineligible, on GB10 (`<= 64` SMs), when the kernel
    /// is absent, or when scratch is not yet pre-allocated during graph capture.
    #[allow(clippy::too_many_arguments)]
    fn try_cross_seq_verify_ba_gates_gemm(
        &self,
        input: DevicePtr,
        gates: DevicePtr,
        rows: usize,
        h: usize,
        ba_size: usize,
        nk: usize,
        nv: usize,
        vpg: usize,
        tc_dense_scope: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        if crate::layers::deterministic_batch::deterministic_batch_enabled()
            || crate::layers::qwen4_oracle::root().is_some()
            || !cross_seq_verify_ba_gemm_eligible(
                tc_dense_scope,
                rows,
                atlas_core::device::active_sm_count(),
                h,
                ba_size,
                nk,
                nv,
                input.0,
                self.ssm.in_proj_ba.weight.0,
                gates.0,
            )
        {
            return Ok(false);
        }
        let Some(ba_k) = verify_ba_gates_kernel(ctx.gpu) else {
            return Ok(false);
        };
        let capturing = ctx.graph_capture
            || spark_runtime::gpu_capture_region::active_capture_region().is_some();
        let Some(ba_buf) = ensure_verify_ba_f32_scratch(ctx.gpu, rows, ba_size, capturing)? else {
            return Ok(false);
        };
        let h_u32 = u32::try_from(h)?;
        let rows_u32 = u32::try_from(rows)?;
        let ba_u32 = u32::try_from(ba_size)?;
        let vpg_u32 = u32::try_from(vpg)?;
        spark_runtime::cublaslt::gemm_act_weight_t_typed_pinned(
            input.0,
            h_u32,
            self.ssm.in_proj_ba.weight.0,
            ba_buf.0,
            ba_u32,
            rows_u32,
            ba_u32,
            h_u32,
            spark_runtime::cublaslt::GemmDtype::Bf16,
            spark_runtime::cublaslt::GemmDtype::F32,
            false,
            rows_u32,
            stream,
        )?;
        spark_runtime::kernel_args::KernelLaunch::new(ctx.gpu, ba_k)
            .grid([rows_u32, 1, 1])
            .block([ba_u32, 1, 1])
            .arg_ptr(ba_buf)
            .arg_ptr(self.ssm.a_log.weight)
            .arg_ptr(self.ssm.dt_bias.weight)
            .arg_ptr(gates)
            .arg_u32(ba_u32)
            .arg_u32(vpg_u32)
            .arg_u32(ba_u32)
            .launch(stream)?;
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                rows,
                hidden = h,
                ba_size,
                "QWEN4_CROSS_SEQ_VERIFY_BA_GEMM_ENGAGED"
            );
        });
        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_qwen4_cross_seq_mixer_only_inner(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        frame: Qwen4CrossSeqFrame,
        seqs: &mut [Qwen4CrossSeqRows<'_>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let _ = self
            .decode_qwen4_cross_seq_mixer_core(hidden, residual, frame, seqs, ctx, false, stream)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_qwen4_cross_seq_mixer_core(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        frame: Qwen4CrossSeqFrame,
        seqs: &mut [Qwen4CrossSeqRows<'_>],
        ctx: &ForwardContext,
        defer_inject: bool,
        stream: u64,
    ) -> Result<Option<DevicePtr>> {
        ensure!(
            seqs.len() == frame.seqs(),
            "cross-sequence SSM frame has {} sequence views for {} sequences",
            seqs.len(),
            frame.seqs()
        );
        self.preflight_qwen4_cross_seq_mixer_only_inner(frame, ctx)?;
        let attn_hyper = self.qwen4_attn_hyper.as_ref().ok_or_else(|| {
            anyhow::anyhow!("cross-sequence Qwen4 SSM verify lacks attention hyperconnection")
        })?;
        let k = frame.k();
        let rows = frame.rows();
        let h = ctx.config.hidden_size;
        let nk = ctx.config.linear_num_key_heads;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let kd = ctx.config.linear_key_head_dim;
        let vpg = nv / nk;
        let value_dim = nv * vd;
        let qkvz_size = ctx.config.ssm_qkvz_size();
        let ba_size = ctx.config.ssm_ba_size();
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

        // ── Row-parallel: attention hyperconnection, QKVZ, BA gates ──
        let attn_inputs = match crate::layers::qwen4_cross_seq_mlp::take_pending_inject() {
            Some(pending)
                if pending.matches(hidden, residual, rows, stream)
                    && rows * h * BF16 <= ctx.buffers.sizes().norm_output =>
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
                let attn_inputs = ctx.buffers.gate_logits();
                attn_hyper.prepare_batched_into(
                    hidden,
                    residual,
                    rows,
                    ctx.buffers,
                    Some(attn_inputs),
                    ctx.gpu,
                    eps,
                    stream,
                )?
            }
        };
        spark_runtime::cublaslt::stage_timer::mark("x_hc_in", stream);
        let qkvz = self
            .qkvz_nvfp4
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("cross-sequence SSM verify requires NVFP4 QKVZ"))?;
        let deinterleaved = ctx.buffers.ssm_deinterleaved();
        if tc_dense_fp8 && ssm_tc_dense_fp8_available(ctx.gpu, qkvz, rows, qkvz_size, h) {
            // `qwen4_fast_proj` uses its own private scratch and the already-cached
            // FP8 weight table, strictly disjoint from every arena in `ctx.buffers`.
            crate::layers::qwen4_fast_proj::clear_prequant_fp8_act();
            crate::layers::qwen4_fp8_act_amax::clear();
            crate::layers::qwen4_fast_proj::fp8_gemm_ex(
                ctx.gpu,
                attn_inputs,
                qkvz,
                deinterleaved,
                rows,
                qkvz_size,
                h,
                qkvz_size,
                None,
                stream,
            )?;
            static QKVZ_TC_LOGGED: std::sync::Once = std::sync::Once::new();
            QKVZ_TC_LOGGED.call_once(|| {
                tracing::info!(
                    selector = crate::layers::qwen4_hyper::BATCHED_VERIFY_TC_DENSE_ENV,
                    component = "ssm_qkvz",
                    route = "qwen4_fast_proj_fp8",
                    tensor_scales = crate::layers::qwen4_fast_proj::fp8_tensor_scales(),
                    cached_weight = crate::layers::qwen4_fast_proj::has_cached_weight_tensor(
                        qkvz, qkvz_size, h, None,
                    ),
                    rows,
                    n = qkvz_size,
                    k = h,
                    "QWEN4_BATCHED_VERIFY_TC_DENSE_ENGAGED"
                );
            });
        } else {
            self.project_nvfp4_rows_exact_or_k1(
                ctx.gpu,
                attn_inputs,
                qkvz,
                deinterleaved,
                rows as u32,
                qkvz_size as u32,
                h as u32,
                stream,
            )?;
        }
        let gates = ctx.buffers.ssm_gates();
        let ran_ba_gemm = self.try_cross_seq_verify_ba_gates_gemm(
            attn_inputs,
            gates,
            rows,
            h,
            ba_size,
            nk,
            nv,
            vpg,
            tc_dense_scope,
            ctx,
            stream,
        )?;
        if !ran_ba_gemm {
            ops::dense_gemv_ba_gates_batchn(
                ctx.gpu,
                self.ba_gates_batchn_exact_k,
                attn_inputs,
                &self.ssm.in_proj_ba,
                self.ssm.a_log.weight,
                self.ssm.dt_bias.weight,
                gates,
                gates.offset(nv * FP32),
                rows as u32,
                ba_size as u32,
                h as u32,
                vpg as u32,
                (nv * 2) as u32,
                stream,
            )?;
        }
        spark_runtime::cublaslt::stage_timer::mark("x_qkvz_ba", stream);

        // ── Recurrent cores: prefer 1-launch fused multi-sequence GDN,
        // falling back to 3-launch multi-sequence or per-sequence launches ──
        let normed = ctx.buffers.ssm_qkvz();
        let conv_arena = ctx.buffers.ssm_conv_out_f32();
        let arenas = (deinterleaved, gates, conv_arena, normed);
        let ran_fused_multi = if frame.seqs() >= 1
            && let Some((fused_multi, lazy)) =
                self.cross_seq_fused_multi_gdn_kernel(frame, (nk, kd, nv, vd), ctx)
        {
            self.cross_seq_recurrent_fused_multi(
                fused_multi,
                lazy,
                frame,
                seqs,
                (deinterleaved, gates, normed),
                (nk, kd, nv, vd),
                ctx,
                stream,
            )?
        } else {
            false
        };
        if !ran_fused_multi {
            let capturing = ctx.graph_capture
                || spark_runtime::gpu_capture_region::active_capture_region().is_some();
            ensure!(
                !capturing,
                "cross-sequence Qwen4 SSM non-fused fallback is unsupported under CUDA graph capture"
            );
            for view in seqs.iter() {
                if let Some(ssm) = view.state.as_any().downcast_ref::<SsmLayerState>() {
                    self.gdn_keep_late_flush_slot_inner(
                        ctx.gpu,
                        ctx.config,
                        view.slot_idx,
                        ssm.h_state,
                        ssm.conv_state,
                        None,
                        view.conv_intermediate,
                        view.conv_intermediate_stride,
                        view.h_intermediate,
                        stream,
                    )?;
                }
                self.gdn_keep_late_mark_verify_ran(view.slot_idx, false);
            }
            match self
                .cross_seq_multi_gdn_kernel(k, ctx.gpu)?
                .filter(|_| frame.seqs() > 1 && !frame.is_ragged())
            {
                Some(gdn_multi) => self.cross_seq_recurrent_multi(
                    gdn_multi,
                    frame,
                    seqs,
                    arenas,
                    (nk, kd, nv, vd),
                    ctx,
                    stream,
                )?,
                None => self.cross_seq_recurrent_per_seq(
                    frame,
                    seqs,
                    arenas,
                    (nk, kd, nv, vd),
                    ctx,
                    stream,
                )?,
            }
        }
        spark_runtime::cublaslt::stage_timer::mark("x_recurrent", stream);

        // ── Row-parallel: output projection and attention inject ──
        let output = ctx.buffers.moe_output();
        if tc_dense_fp8
            && crate::layers::qwen4_fast_proj::batched_verify_dense_proj_tc_admits_with_policy(
                rows,
                h,
                crate::layers::deterministic_batch::deterministic_batch_policy_or_disabled(),
            )
            && ssm_tc_dense_fp8_available(ctx.gpu, &self.ssm.out_proj, rows, h, value_dim)
        {
            crate::layers::qwen4_fast_proj::clear_prequant_fp8_act();
            crate::layers::qwen4_fp8_act_amax::clear();
            crate::layers::qwen4_fast_proj::fp8_gemm_ex(
                ctx.gpu,
                normed,
                &self.ssm.out_proj,
                output,
                rows,
                h,
                value_dim,
                h,
                None,
                stream,
            )?;
            static OUT_TC_LOGGED: std::sync::Once = std::sync::Once::new();
            OUT_TC_LOGGED.call_once(|| {
                tracing::info!(
                    selector = crate::layers::qwen4_hyper::BATCHED_VERIFY_TC_DENSE_ENV,
                    component = "ssm_out_proj",
                    route = "qwen4_fast_proj_fp8",
                    tensor_scales = crate::layers::qwen4_fast_proj::fp8_tensor_scales(),
                    cached_weight = crate::layers::qwen4_fast_proj::has_cached_weight_tensor(
                        &self.ssm.out_proj,
                        h,
                        value_dim,
                        None,
                    ),
                    rows,
                    n = h,
                    k = value_dim,
                    "QWEN4_BATCHED_VERIFY_TC_DENSE_ENGAGED"
                );
            });
        } else {
            self.project_nvfp4_rows_exact_or_k1(
                ctx.gpu,
                normed,
                &self.ssm.out_proj,
                output,
                rows as u32,
                h as u32,
                value_dim as u32,
                stream,
            )?;
        }
        if ctx.config.tp_world_size > 1
            && let Some(comm) = ctx.comm
        {
            comm.all_reduce_async(output.0, rows * h * BF16, stream)?;
        }
        let deferred_out = if defer_inject {
            Some(output)
        } else {
            attn_hyper.inject_saved_batched(hidden, output, residual, rows, ctx.gpu, stream)?;
            None
        };
        spark_runtime::cublaslt::stage_timer::mark("x_ssm_out", stream);
        Ok(deferred_out)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_qwen4_cross_seq_verify_inner(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        frame: Qwen4CrossSeqFrame,
        seqs: &mut [Qwen4CrossSeqRows<'_>],
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
            anyhow::anyhow!("cross-sequence Qwen4 SSM verify lacks MLP hyperconnection")
        })?;
        let defer_inject = crate::layers::qwen4_cross_seq_mlp::batched_hc_fuse_inject_gn_enabled();
        let deferred_mixer_out = self.decode_qwen4_cross_seq_mixer_core(
            hidden,
            residual,
            frame,
            seqs,
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

    /// One single-sequence conv/GDN/gnorm launch set per sequence.
    #[allow(clippy::too_many_arguments)]
    fn cross_seq_recurrent_per_seq(
        &self,
        frame: Qwen4CrossSeqFrame,
        seqs: &mut [Qwen4CrossSeqRows<'_>],
        (deinterleaved, gates, conv_arena, normed): (DevicePtr, DevicePtr, DevicePtr, DevicePtr),
        (nk, kd, nv, vd): (usize, usize, usize, usize),
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let qkvz_size = ctx.config.ssm_qkvz_size();
        let value_dim = nv * vd;
        for (s, view) in seqs.iter_mut().enumerate() {
            let seq_k = frame.seq_k(s);
            let base = frame.row_base(s);
            let _lane =
                crate::layers::qwen4_verify_lanes::LaneGuard::enter_for_slot(view.slot_idx)?;
            let state = view
                .state
                .as_any_mut()
                .downcast_mut::<SsmLayerState>()
                .ok_or_else(|| anyhow::anyhow!("cross-sequence SSM row {s} is not SSM state"))?;
            self.preflight_qwen4_exact_ssm_rows(
                Qwen4ExactSsmRowsRoute::LegacyK5OrK9,
                seq_k,
                state,
                view.h_intermediate,
                view.conv_intermediate,
                view.h_intermediate_stride,
                view.conv_intermediate_stride,
                ctx,
            )?;
            self.cross_seq_recurrent_core(
                seq_k,
                deinterleaved.offset(base * qkvz_size * BF16),
                gates.offset(base * nv * 2 * FP32),
                conv_arena.offset(base * qkvz_size * FP32),
                normed.offset(base * value_dim * BF16),
                state,
                view.h_intermediate,
                view.conv_intermediate,
                view.conv_intermediate_stride,
                view.h_intermediate_stride,
                (nk, kd, nv, vd),
                ctx,
                stream,
            )?;
        }
        Ok(())
    }

    /// Conv sequence, GDN sequence and gated norm for one sequence's `k`
    /// rows. Mirrors `ssm_forward_qwen4_exact_rows` for the legacy K5/K9
    /// route (which forces both sequence kernels) after its projections.
    #[allow(clippy::too_many_arguments)]
    fn cross_seq_recurrent_core(
        &self,
        k: usize,
        deinterleaved: DevicePtr,
        gates_rows: DevicePtr,
        conv_arena_rows: DevicePtr,
        normed_rows: DevicePtr,
        state: &mut SsmLayerState,
        h_intermediate: DevicePtr,
        conv_intermediate: DevicePtr,
        conv_intermediate_stride: usize,
        h_intermediate_stride: usize,
        (nk, kd, nv, vd): (usize, usize, usize, usize),
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let key_dim = nk * kd;
        let value_dim = nv * vd;
        let conv_dim = key_dim * 2 + value_dim;
        let qkvz_size = ctx.config.ssm_qkvz_size();
        let d_conv = ctx.config.linear_conv_kernel_dim;
        let qk_ch = (key_dim * 2) as u32;
        let max_k = crate::layers::qwen3_ssm::GDN_LAZY_MAX_K;

        // Identical lazy decision to the single-sequence dispatch and to the
        // async commit (`gdn_seq_lazy_engaged(k)`).
        let lazy = self.gdn_seq_lazy_engaged_inner(k);
        ensure!(
            !lazy || ctx.ddtree_parent_ids_dev.is_none(),
            "cross-sequence GDN lazy verify requires a flat chain"
        );
        let lazy_retain = if lazy {
            Some(self.gdn_lazy_retain_current(ctx.gpu, qkvz_size, nv)?)
        } else {
            None
        };
        // The single-sequence route writes conv rows and gates straight into
        // the retain lane when the conv sequence kernel runs (always, on this
        // route) and direct retain staging is not disabled.
        let direct_retain = lazy_retain.filter(|_| qwen4_verify_direct_retain_enabled());
        let gate_bytes = k * nv * 2 * FP32;
        // C1 writes the BA gates straight into the retain lane; the frame
        // computed every sequence's gates in one launch into the arena, so it
        // copies this sequence's rows into its lane (one extra D2D of
        // k * nv * 8 bytes). The GDN kernel and the commit replay then read
        // the same values from the same place as in C1.
        let gates = match direct_retain {
            Some(retain) => {
                let dst = retain.offset(max_k * qkvz_size * FP32);
                ctx.gpu
                    .copy_d2d_async(gates_rows, dst, gate_bytes, stream)?;
                dst
            }
            None => gates_rows,
        };
        let conv_rows = direct_retain.unwrap_or(conv_arena_rows);
        ops::conv1d_update_l2norm_f32_sequence(
            ctx.gpu,
            self.conv1d_l2norm_f32_sequence_k,
            state.conv_state,
            deinterleaved,
            &self.ssm.conv1d,
            conv_rows,
            conv_intermediate,
            k as u32,
            conv_dim as u32,
            d_conv as u32,
            qk_ch,
            kd as u32,
            1e-6,
            qkvz_size as u32,
            qkvz_size as u32,
            (conv_intermediate_stride / FP32) as u32,
            stream,
        )?;

        let persistent_gdn =
            !lazy && self.gdn_f32_sequence_persistent_k.0 != 0 && ssm_gdn_seq_persistent_enabled();
        let gdn_rows = conv_rows.offset(conv_dim * FP32);
        let gdn_kernel = if let Some(retain) = lazy_retain {
            if direct_retain.is_none() {
                ctx.gpu
                    .copy_d2d_async(conv_rows, retain, k * qkvz_size * FP32, stream)?;
                ctx.gpu.copy_d2d_async(
                    gates,
                    retain.offset(max_k * qkvz_size * FP32),
                    gate_bytes,
                    stream,
                )?;
            }
            Some(self.gdn_f32_sequence_lazyfinal_k)
        } else if persistent_gdn {
            Some(self.gdn_f32_sequence_persistent_k)
        } else {
            None
        };
        match gdn_kernel {
            Some(kernel) => ops::gdn_decode_f32_sequence_persistent(
                ctx.gpu,
                kernel,
                state.h_state,
                conv_rows,
                conv_rows.offset(key_dim * FP32),
                conv_rows.offset(key_dim * 2 * FP32),
                gates,
                gates.offset(nv * FP32),
                gdn_rows,
                h_intermediate,
                k as u32,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                qkvz_size as u32,
                qkvz_size as u32,
                (nv * 2) as u32,
                qkvz_size as u32,
                (h_intermediate_stride / FP32) as u32,
                stream,
            )?,
            None => ops::gdn_decode_f32_sequence(
                ctx.gpu,
                self.gdn_f32_sequence_k,
                state.h_state,
                conv_rows,
                conv_rows.offset(key_dim * FP32),
                conv_rows.offset(key_dim * 2 * FP32),
                gates,
                gates.offset(nv * FP32),
                gdn_rows,
                h_intermediate,
                k as u32,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                qkvz_size as u32,
                qkvz_size as u32,
                (nv * 2) as u32,
                qkvz_size as u32,
                (h_intermediate_stride / FP32) as u32,
                stream,
            )?,
        }

        self.cross_seq_gated_norm(
            k,
            gdn_rows,
            deinterleaved,
            normed_rows,
            (nk, kd, nv, vd),
            ctx,
            stream,
        )
    }

    /// Gated RMS norm of one sequence's `k` GDN rows (the single-sequence
    /// verifier's multi-row launch, or its per-row fallback).
    #[allow(clippy::too_many_arguments)]
    fn cross_seq_gated_norm(
        &self,
        k: usize,
        gdn_rows: DevicePtr,
        deinterleaved: DevicePtr,
        normed_rows: DevicePtr,
        (nk, kd, nv, vd): (usize, usize, usize, usize),
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let key_dim = nk * kd;
        let value_dim = nv * vd;
        let qkvz_size = ctx.config.ssm_qkvz_size();
        let eps = ctx.config.rms_norm_eps as f32;
        if qwen4_verify_gnorm_rows_selected()
            && self.gated_rms_norm_f32_multi_seq_k.0 != 0
            && ctx.config.output_gate_type == "sigmoid"
        {
            ops::gated_rms_norm_f32_multi_seq(
                ctx.gpu,
                self.gated_rms_norm_f32_multi_seq_k,
                gdn_rows,
                deinterleaved.offset((key_dim * 2 + value_dim) * BF16),
                &self.ssm.norm,
                normed_rows,
                nv as u32,
                k as u32,
                vd as u32,
                eps,
                qkvz_size as u32,
                qkvz_size as u32,
                value_dim as u32,
                stream,
            )?;
        } else {
            for row in 0..k {
                let qkvz_row = deinterleaved.offset(row * qkvz_size * BF16);
                let gdn_row = gdn_rows.offset(row * qkvz_size * FP32);
                ops::gated_rms_norm(
                    ctx.gpu,
                    self.gated_rms_norm_f32_k,
                    gdn_row,
                    qkvz_row.offset((key_dim * 2 + value_dim) * BF16),
                    &self.ssm.norm,
                    normed_rows.offset(row * value_dim * BF16),
                    nv as u32,
                    vd as u32,
                    vd as u32,
                    eps,
                    vd as u32,
                    stream,
                )?;
            }
        }
        Ok(())
    }
}

/// `ATLAS_QWEN4_BATCHED_K5_GDN_MULTI` (strict 0/1, default 0): run the
/// frame's ordered conv and GDN recurrences as one multi-sequence launch
/// each instead of one launch per sequence. Same statement sequence per
/// sequence as the single-sequence kernels.
pub(crate) const GDN_MULTI_ENV: &str = "ATLAS_QWEN4_BATCHED_K5_GDN_MULTI";

fn parse_gdn_multi(raw: Option<&str>) -> std::result::Result<bool, &'static str> {
    match raw {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(_) => Err("ATLAS_QWEN4_BATCHED_K5_GDN_MULTI must be exact value 0 or 1"),
    }
}

fn read_gdn_multi_live() -> std::result::Result<bool, String> {
    let raw = match std::env::var(GDN_MULTI_ENV) {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err(format!("{GDN_MULTI_ENV} must be valid UTF-8"));
        }
    };
    parse_gdn_multi(raw.as_deref()).map_err(str::to_owned)
}

#[inline]
fn gdn_multi_selected() -> Result<bool> {
    #[cfg(not(test))]
    {
        static CACHED: std::sync::OnceLock<std::result::Result<bool, String>> =
            std::sync::OnceLock::new();
        CACHED
            .get_or_init(read_gdn_multi_live)
            .as_ref()
            .copied()
            .map_err(|msg| anyhow::Error::msg(msg.clone()))
    }
    #[cfg(test)]
    {
        read_gdn_multi_live().map_err(anyhow::Error::msg)
    }
}

/// Sequences one pointer table can address (the 128-row frame at K2).
const MULTI_TABLE_SEQS: usize = 64;
/// Tables: h, conv rows, gates, H intermediates, conv state, conv
/// intermediates, each `MULTI_TABLE_SEQS` u64 entries.
const MULTI_TABLES: usize = 6;
/// Unified staging capacity (`64 × 64 B = 4096 B`) holding either
/// `GdnFusedMultiSeqDesc[64]` or the legacy `6 × 64 × 8 B = 3072 B` table.
pub(crate) const CROSS_SEQ_TABLE_BYTES: usize =
    MULTI_TABLE_SEQS * ops::GdnFusedMultiSeqDesc::SIZE_BYTES;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct CrossSeqDescSlabKey {
    gpu_addr: usize,
    slot_bytes: usize,
    num_ssm_layers: usize,
}

struct CrossSeqDescSlabEntry {
    slab_id: u64,
    key: CrossSeqDescSlabKey,
    base: DevicePtr,
    next_layer: usize,
    live_layers: usize,
}

#[derive(Default)]
struct CrossSeqDescSlabRegistry {
    next_slab_id: u64,
    entries: Vec<CrossSeqDescSlabEntry>,
}

static CROSS_SEQ_DESC_SLABS: std::sync::OnceLock<parking_lot::Mutex<CrossSeqDescSlabRegistry>> =
    std::sync::OnceLock::new();

/// Claim one layer slot `(device_ptr, slab_id)` from a model-lifetime contiguous
/// `[num_ssm_layers][slot_bytes]` device slab so `stage_all_ssm_fused_descs` can
/// upload all SSM layers' `GdnFusedMultiSeqDesc` tables in ONE H2D transfer.
fn alloc_cross_seq_desc_slab_slice(
    gpu: &dyn GpuBackend,
    slot_bytes: usize,
    num_ssm_layers: usize,
) -> Result<(DevicePtr, u64)> {
    let slab_bytes = num_ssm_layers
        .checked_mul(slot_bytes)
        .ok_or_else(|| anyhow::anyhow!("cross-sequence SSM descriptor slab extent overflow"))?;
    let key = CrossSeqDescSlabKey {
        gpu_addr: std::ptr::from_ref(gpu) as *const () as usize,
        slot_bytes,
        num_ssm_layers,
    };
    let mut reg = CROSS_SEQ_DESC_SLABS
        .get_or_init(|| parking_lot::Mutex::new(CrossSeqDescSlabRegistry::default()))
        .lock();
    if let Some(entry) = reg
        .entries
        .iter_mut()
        .rev()
        .find(|entry| entry.key == key && entry.next_layer < key.num_ssm_layers)
    {
        let layer_ordinal = entry.next_layer;
        entry.next_layer += 1;
        entry.live_layers += 1;
        return Ok((entry.base.offset(layer_ordinal * slot_bytes), entry.slab_id));
    }
    let base = gpu.alloc(slab_bytes)?;
    reg.next_slab_id = reg.next_slab_id.wrapping_add(1).max(1);
    let slab_id = reg.next_slab_id;
    reg.entries.push(CrossSeqDescSlabEntry {
        slab_id,
        key,
        base,
        next_layer: 1,
        live_layers: 1,
    });
    Ok((base, slab_id))
}

fn release_cross_seq_desc_slab_slice(slab_id: u64) {
    if slab_id == 0 {
        return;
    }
    let Some(lock) = CROSS_SEQ_DESC_SLABS.get() else {
        return;
    };
    let mut reg = lock.lock();
    if let Some(pos) = reg.entries.iter().position(|e| e.slab_id == slab_id) {
        let entry = &mut reg.entries[pos];
        entry.next_layer = entry.key.num_ssm_layers;
        entry.live_layers = entry.live_layers.saturating_sub(1);
        if entry.live_layers == 0 {
            reg.entries.swap_remove(pos);
        }
    }
}
/// Built cross-sequence fused GDN descriptor payload and intermediate geometry.
struct CrossSeqFusedDescs {
    descs: Vec<ops::GdnFusedMultiSeqDesc>,
    h_inter_stride: usize,
    conv_inter_stride: usize,
    use_keep_late: bool,
}

/// Number of disjoint page-locked staging slots in the fused descriptor ring,
/// amortizing host fence events by 8× across eager cohort transitions.
const FUSED_DESC_RING_SLOTS: usize = 8;

impl Qwen3SsmLayer {
    /// The multi-sequence GDN twin of the kernel the single-sequence route
    /// would launch at width `k`, when selected and loaded. `None` means
    /// the frame runs the per-sequence launches.
    fn cross_seq_multi_gdn_kernel(
        &self,
        k: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<KernelHandle>> {
        if !gdn_multi_selected()? {
            return Ok(None);
        }
        // The multi route's pointer-table fence waits are bounded host waits;
        // admit it only when the backend itself runs bounded host copies (its
        // cached mode, not the env string). Otherwise run the single-sequence
        // kernels, which need no table.
        if !gpu.bounded_host_copy() {
            static UNBOUNDED: std::sync::Once = std::sync::Once::new();
            UNBOUNDED.call_once(|| {
                tracing::warn!(
                    "{GDN_MULTI_ENV}=1 ignored: the backend is not in bounded host-copy mode \
                     (ATLAS_BOUNDED_HOST_COPY); using per-sequence GDN kernels"
                );
            });
            return Ok(None);
        }
        if self.conv1d_l2norm_f32_sequence_multi_k.0 == 0 {
            return Ok(None);
        }
        let kernel = if self.gdn_seq_lazy_engaged_inner(k) {
            self.gdn_f32_sequence_lazyfinal_multi_k
        } else if self.gdn_f32_sequence_persistent_k.0 != 0 && ssm_gdn_seq_persistent_enabled() {
            self.gdn_f32_sequence_persistent_multi_k
        } else {
            // The snapshot-per-row sequence kernel has no multi twin.
            KernelHandle(0)
        };
        Ok((kernel.0 != 0).then_some(kernel))
    }

    /// Multi-sequence conv + GDN for the whole frame, then the per-sequence
    /// gated norm. Retain-lane staging matches the single-sequence route.
    #[allow(clippy::too_many_arguments)]
    fn cross_seq_recurrent_multi(
        &self,
        gdn_multi: KernelHandle,
        frame: Qwen4CrossSeqFrame,
        seqs: &mut [Qwen4CrossSeqRows<'_>],
        (deinterleaved, gates, conv_arena, normed): (DevicePtr, DevicePtr, DevicePtr, DevicePtr),
        (nk, kd, nv, vd): (usize, usize, usize, usize),
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            !frame.is_ragged(),
            "cross-sequence GDN multi requires a uniform frame width"
        );
        let k = frame.k();
        let n = frame.seqs();
        ensure!(
            n <= MULTI_TABLE_SEQS,
            "cross-sequence GDN multi table holds {MULTI_TABLE_SEQS} sequences, got {n}"
        );
        let key_dim = nk * kd;
        let value_dim = nv * vd;
        let conv_dim = key_dim * 2 + value_dim;
        let qkvz_size = ctx.config.ssm_qkvz_size();
        let max_k = crate::layers::qwen3_ssm::GDN_LAZY_MAX_K;
        let lazy = self.gdn_seq_lazy_engaged_inner(k);
        // Honor `ATLAS_QWEN4_VERIFY_DIRECT_RETAIN != "0"` (cached outside unit tests).
        let direct = lazy && qwen4_verify_direct_retain_enabled();
        let gate_bytes = k * nv * 2 * FP32;
        let (mut h_inter_stride, mut conv_inter_stride) = (None, None);

        // Per-sequence pointers; retain lanes resolved inside each lane.
        let mut table = [[0u64; MULTI_TABLE_SEQS]; MULTI_TABLES];
        let mut retains: Vec<Option<DevicePtr>> = Vec::with_capacity(n);
        for (s, view) in seqs.iter_mut().enumerate() {
            let base = frame.row_base(s);
            let _lane =
                crate::layers::qwen4_verify_lanes::LaneGuard::enter_for_slot(view.slot_idx)?;
            let state = view
                .state
                .as_any_mut()
                .downcast_mut::<SsmLayerState>()
                .ok_or_else(|| anyhow::anyhow!("cross-sequence SSM row {s} is not SSM state"))?;
            self.preflight_qwen4_exact_ssm_rows(
                Qwen4ExactSsmRowsRoute::LegacyK5OrK9,
                k,
                state,
                view.h_intermediate,
                view.conv_intermediate,
                view.h_intermediate_stride,
                view.conv_intermediate_stride,
                ctx,
            )?;
            for (seen, stride) in [
                (&mut h_inter_stride, view.h_intermediate_stride),
                (&mut conv_inter_stride, view.conv_intermediate_stride),
            ] {
                ensure!(
                    seen.replace(stride)
                        .is_none_or(|previous| previous == stride),
                    "cross-sequence GDN multi needs one intermediate stride per frame"
                );
            }
            let retain = if lazy {
                Some(self.gdn_lazy_retain_current(ctx.gpu, qkvz_size, nv)?)
            } else {
                None
            };
            let arena_gates = gates.offset(base * nv * 2 * FP32);
            let (rows_ptr, gates_ptr) = match retain.filter(|_| direct) {
                Some(retain) => (retain, DevicePtr(arena_gates.0 | 1)),
                None => (conv_arena.offset(base * qkvz_size * FP32), arena_gates),
            };
            table[0][s] = state.h_state.0;
            table[1][s] = rows_ptr.0;
            table[2][s] = gates_ptr.0;
            table[3][s] = view.h_intermediate.0;
            table[4][s] = state.conv_state.0;
            table[5][s] = view.conv_intermediate.0;
            retains.push(retain);
        }
        let h_inter_stride = h_inter_stride.unwrap_or(0);
        let conv_inter_stride = conv_inter_stride.unwrap_or(0);
        let device = self.upload_cross_seq_ptr_table(&table, ctx, stream)?;
        let table_ptr = |index: usize| device.offset(index * MULTI_TABLE_SEQS * 8);

        ops::conv1d_update_l2norm_f32_sequence_multi(
            ctx.gpu,
            self.conv1d_l2norm_f32_sequence_multi_k,
            table_ptr(4),
            deinterleaved,
            &self.ssm.conv1d,
            table_ptr(1),
            table_ptr(5),
            n as u32,
            k as u32,
            conv_dim as u32,
            ctx.config.linear_conv_kernel_dim as u32,
            (key_dim * 2) as u32,
            kd as u32,
            1e-6,
            qkvz_size as u32,
            qkvz_size as u32,
            (conv_inter_stride / FP32) as u32,
            stream,
        )?;
        if lazy && !direct {
            // Mirror the single-sequence route: retain the exact conv rows
            // and gates before the GDN launch.
            for (s, retain) in retains.iter().enumerate() {
                let retain = retain.ok_or_else(|| anyhow::anyhow!("lazy retain lane missing"))?;
                let base = frame.row_base(s);
                ctx.gpu.copy_d2d_async(
                    conv_arena.offset(base * qkvz_size * FP32),
                    retain,
                    k * qkvz_size * FP32,
                    stream,
                )?;
                ctx.gpu.copy_d2d_async(
                    gates.offset(base * nv * 2 * FP32),
                    retain.offset(max_k * qkvz_size * FP32),
                    gate_bytes,
                    stream,
                )?;
            }
        }
        ops::gdn_decode_f32_sequence_multi(
            ctx.gpu,
            gdn_multi,
            table_ptr(0),
            table_ptr(1),
            table_ptr(2),
            table_ptr(3),
            n as u32,
            k as u32,
            nk as u32,
            nv as u32,
            kd as u32,
            vd as u32,
            key_dim as u32,
            (key_dim * 2) as u32,
            conv_dim as u32,
            nv as u32,
            qkvz_size as u32,
            qkvz_size as u32,
            (nv * 2) as u32,
            qkvz_size as u32,
            (h_inter_stride / FP32) as u32,
            stream,
        )?;
        if self.gated_rms_norm_f32_multi_seq_k.0 != 0 && qkvz_size == conv_dim + value_dim {
            let eps = ctx.config.rms_norm_eps as f32;
            spark_runtime::kernel_args::KernelLaunch::new(
                ctx.gpu,
                self.gated_rms_norm_f32_multi_seq_k,
            )
            .grid([nv as u32, k as u32, n as u32])
            .block([vd.min(1024) as u32, 1, 1])
            .arg_ptr(table_ptr(1))
            .arg_ptr(deinterleaved.offset(conv_dim * BF16))
            .arg_ptr(self.ssm.norm.weight)
            .arg_ptr(normed)
            .arg_u32(vd as u32)
            .arg_f32(eps)
            .arg_u32(qkvz_size as u32)
            .arg_u32(qkvz_size as u32)
            .arg_u32(value_dim as u32)
            .launch(stream)?;
        } else {
            for s in 0..n {
                let base = frame.row_base(s);
                let rows = DevicePtr(table[1][s]);
                self.cross_seq_gated_norm(
                    k,
                    rows.offset(conv_dim * FP32),
                    deinterleaved.offset(base * qkvz_size * BF16),
                    normed.offset(base * value_dim * BF16),
                    (nk, kd, nv, vd),
                    ctx,
                    stream,
                )?;
            }
        }
        static ENGAGED: std::sync::Once = std::sync::Once::new();
        ENGAGED.call_once(|| {
            tracing::info!(seqs = n, k, lazy, "QWEN4_BATCHED_K5_GDN_MULTI_ENGAGED");
        });
        Ok(())
    }

    /// Stage the six tables in this layer's page-locked buffer and enqueue
    /// one H2D copy fenced by `upload_done`. The host never rewrites the
    /// staging while a copy from it may still be in flight: see
    /// [`TableDma`] and [`acquire_staging`].
    fn upload_cross_seq_ptr_table(
        &self,
        table: &[[u64; MULTI_TABLE_SEQS]; MULTI_TABLES],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let bytes = MULTI_TABLES * MULTI_TABLE_SEQS * 8;
        let mut guard = self
            .cross_seq_ptr_table
            .lock()
            .map_err(|_| anyhow::anyhow!("cross-sequence pointer table lock poisoned"))?;
        if guard.is_none() {
            *guard = Some(CrossSeqPtrTable::allocate_for_layers(
                ctx.gpu,
                CROSS_SEQ_TABLE_BYTES,
                super::count_ssm_layers(ctx.config),
            )?);
        }
        let entry = guard.as_mut().expect("pointer table just initialized");
        if entry.state == TableDma::Recorded
            && entry.stream == stream
            && entry.last_uploaded.as_ref() == Some(table)
        {
            return Ok(entry.device);
        }
        let copy_stream = entry.stream;
        let (state, acquired) = acquire_staging(
            entry.state,
            entry.upload_done,
            |event| ctx.gpu.poll_event(event),
            |event| ctx.gpu.record_event(event, copy_stream),
        );
        entry.state = state;
        acquired?;
        entry.last_uploaded = None;
        entry.last_uploaded_fused = None;
        entry.prestaged_for_frame = false;
        entry.prestaged_unconfirmed = false;
        let staging = entry.host.as_mut_slice();
        for (index, value) in table.iter().flatten().enumerate() {
            staging[index * 8..index * 8 + 8].copy_from_slice(&value.to_le_bytes());
        }
        let device = entry.device;
        let source = entry.host.pinned_slice(bytes)?;
        // From here the staging may be read by the device until a fence
        // proves otherwise, whatever the enqueue calls below report.
        entry.state = TableDma::Uncertain;
        entry.stream = stream;
        // SAFETY: `source` is this layer's model-lifetime page-locked buffer.
        // `acquire_staging` proved no earlier copy from it is in flight, and
        // the state stays `Uncertain` until `upload_done` is recorded after
        // this copy, so the next rewrite waits for or drains it.
        unsafe {
            ctx.gpu.copy_h2d_pinned_async(source, device, stream)?;
        }
        entry.state = after_enqueue(entry.upload_done, || {
            ctx.gpu.record_event(entry.upload_done, stream)
        })?;
        entry.last_uploaded = Some(*table);
        Ok(device)
    }

    /// Release this layer's cross-sequence pointer table (model drop /
    /// reload). `TransformerModel::drop` has already drained the default and
    /// secondary streams (aborting on failure), so no copy can still read the
    /// staging. Only the `upload_done` event is destroyed. The table, including
    /// its pinned owner, stays in the field until automatic field teardown,
    /// and the raw device copy stays for `GpuBackend::free_all_allocations`
    /// (BUG #29 rule in `model/drop.rs`). No host-side synchronize is issued
    /// here: an unbounded `cuStreamSynchronize` could hang teardown.
    pub(super) fn release_cross_seq_ptr_table(&self, gpu: &dyn GpuBackend) -> Result<()> {
        let mut guard = self
            .cross_seq_ptr_table
            .lock()
            .map_err(|_| anyhow::anyhow!("cross-sequence pointer table lock poisoned"))?;
        let Some(entry) = guard.as_mut() else {
            return Ok(());
        };
        // Nothing may upload through this table after teardown began.
        entry.state = TableDma::Poisoned;
        release_cross_seq_desc_slab_slice(std::mem::replace(&mut entry.slab_id, 0));
        let event = std::mem::replace(&mut entry.upload_done, 0);
        if event != 0 {
            gpu.destroy_event(event)
                .map_err(|error| anyhow::anyhow!("cross-sequence upload event: {error:#}"))?;
        }
        Ok(())
    }

    /// Whether this layer's pointer table can serve another frame. A
    /// poisoned table refuses every later multi-sequence upload, so the
    /// model's read-only admission reports it and the scheduler verifies
    /// those sequences one at a time instead of failing them.
    pub(super) fn cross_seq_table_admits_frames(&self) -> bool {
        let state = self
            .cross_seq_ptr_table
            .lock()
            .ok()
            .map(|guard| guard.as_ref().map(|entry| entry.state));
        table_admits_frames(state)
    }

    /// Select the 1-launch fused multi-sequence GDN verify kernel when
    /// `ATLAS_QWEN4_GDN_VERIFY_FUSED` (default on) is enabled, the backend
    /// runs bounded host copies, and the Qwen4 geometry matches. Supports both
    /// uniform and ragged mixed-`K` frames as long as all sequences share the
    /// same lazy/persistent mode.
    fn cross_seq_fused_multi_gdn_kernel(
        &self,
        frame: Qwen4CrossSeqFrame,
        (nk, kd, nv, vd): (usize, usize, usize, usize),
        ctx: &ForwardContext,
    ) -> Option<(KernelHandle, bool)> {
        if frame.seqs() == 0
            || !super::qwen4_gdn_verify_fused_enabled()
            || !ctx.gpu.bounded_host_copy()
            || !ctx.config.is_qwen4_exp()
            || self.gdn_fused_group_done.is_null()
            || kd != 128
            || vd != 128
            || ctx.config.linear_conv_kernel_dim != 4
            || nk == 0
            || nk > 32
            || nv < nk
            || !nv.is_multiple_of(nk)
            || ctx.config.output_gate_type != "sigmoid"
        {
            return None;
        }
        let k0 = frame.seq_k(0);
        let lazy = self.gdn_seq_lazy_engaged_inner(k0);
        if frame.is_ragged() {
            let mut checked_ks = if k0 < 16 { 1u16 << k0 } else { 0 };
            for s in 1..frame.seqs() {
                let seq_k = frame.seq_k(s);
                let k_bit = if seq_k < 16 { 1u16 << seq_k } else { 0 };
                if k_bit != 0 && (checked_ks & k_bit) != 0 {
                    continue;
                }
                if self.gdn_seq_lazy_engaged_inner(seq_k) != lazy {
                    return None;
                }
                checked_ks |= k_bit;
            }
        }
        let kernel = if lazy {
            self.qwen4_gdn_verify_fused_lazyfinal_multi_k
        } else if self.gdn_f32_sequence_persistent_k.0 != 0 && ssm_gdn_seq_persistent_enabled() {
            self.qwen4_gdn_verify_fused_persistent_multi_k
        } else {
            KernelHandle(0)
        };
        (kernel.0 != 0).then_some((kernel, lazy))
    }

    /// Pre-allocate this layer's cross-sequence fused descriptor table and
    /// lazy-retain pool outside CUDA graph capture so zero `gpu.alloc` occurs
    /// inside capture.
    pub(super) fn ensure_qwen4_cross_seq_fused_descs_ready_inner(
        &self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
        k: usize,
    ) -> Result<()> {
        let num_ssm_layers = super::count_ssm_layers(config);
        if self.gdn_seq_lazy_engaged_inner(k) {
            let qkvz_size = config.ssm_qkvz_size();
            let nv = config.linear_num_value_heads;
            let _ =
                self.gdn_lazy_retain_pool_with_layers(gpu, qkvz_size, nv, Some(num_ssm_layers))?;
        }
        if !config.is_qwen4_exp() {
            return Ok(());
        }
        // Pre-resolve the Scaled-FP16 Split-V=4 kernel outside CUDA graph capture.
        let _ = qwen4_gdn_scaled_fp16_splitv4_kernel(gpu);
        let mut guard = self
            .cross_seq_ptr_table
            .lock()
            .map_err(|_| anyhow::anyhow!("cross-sequence pointer table lock poisoned"))?;
        if guard.is_none() {
            let capturing = spark_runtime::gpu_capture_region::active_capture_region().is_some()
                || gpu.stream_is_capturing(gpu.default_stream());
            ensure!(
                !capturing,
                "cross-sequence pointer table must be pre-allocated outside CUDA graph capture"
            );
            *guard = Some(CrossSeqPtrTable::allocate_for_layers(
                gpu,
                CROSS_SEQ_TABLE_BYTES,
                num_ssm_layers,
            )?);
        }
        if let Some(entry) = guard.as_ref() {
            ensure!(
                entry.state != TableDma::Poisoned,
                "cross-sequence pointer table is poisoned"
            );
        }
        Ok(())
    }

    /// Build `GdnFusedMultiSeqDesc[0..n]` for the frame and report whether
    /// `KEEP_LATE` (indirect `GdnKeepLateCtrl` per slot) is active.
    #[allow(clippy::too_many_arguments)]
    fn build_cross_seq_fused_descs(
        &self,
        lazy: bool,
        frame: Qwen4CrossSeqFrame,
        seqs: &[Qwen4CrossSeqRows<'_>],
        (nk, kd, nv, vd): (usize, usize, usize, usize),
        ctx: &ForwardContext,
        stream: u64,
        allow_alloc: bool,
    ) -> Result<Option<CrossSeqFusedDescs>> {
        let n = frame.seqs();
        let max_lanes = crate::layers::qwen4_verify_lanes::MAX_LANES;
        if n > MULTI_TABLE_SEQS {
            return Ok(None);
        }
        let mut h_inter_stride: Option<usize> = None;
        let mut conv_inter_stride: Option<usize> = None;
        let mut seen_slots = [false; crate::layers::qwen4_verify_lanes::MAX_LANES];
        for view in seqs.iter() {
            if view.slot_idx >= max_lanes || seen_slots[view.slot_idx] {
                return Ok(None);
            }
            seen_slots[view.slot_idx] = true;
            let Some(state) = view.state.as_any().downcast_ref::<SsmLayerState>() else {
                return Ok(None);
            };
            if state.h_is_f16 {
                return Ok(None);
            }
            if h_inter_stride
                .replace(view.h_intermediate_stride)
                .is_some_and(|prev| prev != view.h_intermediate_stride)
                || conv_inter_stride
                    .replace(view.conv_intermediate_stride)
                    .is_some_and(|prev| prev != view.conv_intermediate_stride)
            {
                return Ok(None);
            }
        }
        let h_inter_stride = h_inter_stride.unwrap_or(0);
        let conv_inter_stride = conv_inter_stride.unwrap_or(0);
        let qkvz_size = ctx.config.ssm_qkvz_size();
        let pool = if lazy {
            if allow_alloc {
                Some(self.gdn_lazy_retain_pool_with_layers(
                    ctx.gpu,
                    qkvz_size,
                    nv,
                    Some(super::count_ssm_layers(ctx.config)),
                )?)
            } else {
                self.gdn_lazy_retain.get()
            }
        } else {
            None
        };
        if lazy && pool.is_none() {
            anyhow::bail!(
                "cross-sequence GDN lazy retain pool was not pre-allocated before CUDA graph capture"
            );
        }
        let use_keep_late = lazy
            && self.gdn_keep_late_supported_inner()
            && h_inter_stride == 0
            && keep_late_ring_fits(nk, kd, nv, vd, qkvz_size)
            && seqs
                .iter()
                .all(|view| self.gdn_keep_late_slot_ctrl_ready(view.slot_idx).is_some());

        let default_stream = ctx.gpu.default_stream();
        let mut descs = Vec::with_capacity(n);
        for (s, view) in seqs.iter().enumerate() {
            let seq_k = frame.seq_k(s);
            let _lane =
                crate::layers::qwen4_verify_lanes::LaneGuard::enter_for_slot(view.slot_idx)?;
            let state = view
                .state
                .as_any()
                .downcast_ref::<SsmLayerState>()
                .ok_or_else(|| anyhow::anyhow!("cross-sequence SSM row {s} is not SSM state"))?;
            self.preflight_qwen4_exact_ssm_rows(
                Qwen4ExactSsmRowsRoute::LegacyK5OrK9,
                seq_k,
                state,
                view.h_intermediate,
                view.conv_intermediate,
                view.h_intermediate_stride,
                view.conv_intermediate_stride,
                ctx,
            )?;
            let counter_slot = if stream == default_stream {
                view.slot_idx
            } else {
                max_lanes + view.slot_idx
            };
            let group_done = self.gdn_fused_group_done.offset(counter_slot * 32 * FP32);
            if use_keep_late {
                let pool = pool.ok_or_else(|| {
                    anyhow::anyhow!("cross-sequence KEEP_LATE retain pool unavailable")
                })?;
                let staged_ctrl = self
                    .gdn_keep_late_slot_ctrl_ready(view.slot_idx)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "cross-sequence KEEP_LATE ctrl missing for slot {}",
                            view.slot_idx
                        )
                    })?;
                descs.push(ops::GdnFusedMultiSeqDesc {
                    conv_state: state.conv_state.0,
                    h_state: state.h_state.0,
                    // Constant slot lane base; kernel reads circular ring
                    // coordinates (`ring_base`, `pending_base`, `pending_rows`,
                    // `write_base`, `ring_cap`, `flush_now`) from `ctrl_ptr`.
                    retain_qkv: pool.lane_base(view.slot_idx).0,
                    // Zero dead `conv_inter` writes under KEEP_LATE circular ring.
                    conv_inter: 0,
                    h_state_inter: staged_ctrl.ctrl_ptr.0,
                    group_done: group_done.0,
                    row_base: frame.row_base(s) as u32,
                    num_tokens: seq_k as u32,
                    flags: staged_ctrl.desc_flags(),
                    _pad: 0,
                });
            } else {
                let retain = if let Some(pool) = pool {
                    let lane = crate::layers::qwen4_verify_lanes::current_lane();
                    ensure!(
                        lane < pool.lanes,
                        "GDN lazy retain has {} lanes; verify lane {lane} has none",
                        pool.lanes
                    );
                    Some(pool.retain_active_ptr(lane))
                } else {
                    None
                };
                descs.push(ops::GdnFusedMultiSeqDesc {
                    conv_state: state.conv_state.0,
                    h_state: state.h_state.0,
                    retain_qkv: retain.map_or(0, |ptr| ptr.0),
                    conv_inter: view.conv_intermediate.0,
                    h_state_inter: view.h_intermediate.0,
                    group_done: group_done.0,
                    row_base: frame.row_base(s) as u32,
                    num_tokens: seq_k as u32,
                    flags: if lazy {
                        ops::GDN_FUSED_FLAG_COPY_GATES_TO_RETAIN
                    } else {
                        0
                    },
                    _pad: 0,
                });
            }
        }
        Ok(Some(CrossSeqFusedDescs {
            descs,
            h_inter_stride,
            conv_inter_stride,
            use_keep_late,
        }))
    }

    /// Stage this layer's `GdnFusedMultiSeqDesc[0..n]` bytes and return
    /// `(dest_dev_ptr, bytes)` so callers can batch all 36 SSM layers in one
    /// Phase-1 `copy_h2d_group_fast` call before `begin_capture` or graph replay.
    pub(super) fn stage_qwen4_cross_seq_fused_desc_bytes_inner(
        &self,
        frame: Qwen4CrossSeqFrame,
        seqs: &[Qwen4CrossSeqRows<'_>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Option<(DevicePtr, Vec<u8>)>> {
        if seqs.len() != frame.seqs() || frame.seqs() == 0 {
            return Ok(None);
        }
        let nk = ctx.config.linear_num_key_heads;
        let kd = ctx.config.linear_key_head_dim;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let Some((_fused_multi, lazy)) =
            self.cross_seq_fused_multi_gdn_kernel(frame, (nk, kd, nv, vd), ctx)
        else {
            return Ok(None);
        };
        let Some(CrossSeqFusedDescs { descs, .. }) = self.build_cross_seq_fused_descs(
            lazy,
            frame,
            seqs,
            (nk, kd, nv, vd),
            ctx,
            stream,
            true,
        )?
        else {
            return Ok(None);
        };
        let bytes_len = descs.len() * ops::GdnFusedMultiSeqDesc::SIZE_BYTES;
        ensure!(
            bytes_len <= CROSS_SEQ_TABLE_BYTES,
            "cross-sequence fused GDN descriptors ({bytes_len} B) exceed {CROSS_SEQ_TABLE_BYTES} B staging"
        );
        let mut guard = self
            .cross_seq_ptr_table
            .lock()
            .map_err(|_| anyhow::anyhow!("cross-sequence pointer table lock poisoned"))?;
        if guard.is_none() {
            *guard = Some(CrossSeqPtrTable::allocate_for_layers(
                ctx.gpu,
                CROSS_SEQ_TABLE_BYTES,
                super::count_ssm_layers(ctx.config),
            )?);
        }
        let entry = guard
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("cross-sequence pointer table missing after init"))?;
        ensure!(
            entry.state != TableDma::Poisoned,
            "cross-sequence pointer table is poisoned"
        );
        // Drain any in-flight pinned-ring DMAs on a prior stream before
        // updating `fused_ring_stream` so the next `upload_cross_seq_fused_descs`
        // on `stream` cannot skip the stream-switch fence.
        if entry.fused_ring_slot > 0 && entry.fused_ring_stream != stream {
            let prev_stream = entry.fused_ring_stream;
            let (state, acquired) = acquire_staging(
                TableDma::Uncertain,
                entry.upload_done,
                |event| ctx.gpu.poll_event(event),
                |event| ctx.gpu.record_event(event, prev_stream),
            );
            entry.state = state;
            acquired?;
            entry.fused_ring_slot = 0;
        }
        let device = entry.device;
        let mut payload = vec![0u8; bytes_len];
        for (index, desc) in descs.iter().enumerate() {
            let offset = index * ops::GdnFusedMultiSeqDesc::SIZE_BYTES;
            desc.write_le_bytes(
                &mut payload[offset..offset + ops::GdnFusedMultiSeqDesc::SIZE_BYTES],
            );
        }
        entry.last_uploaded = None;
        entry.last_uploaded_fused = Some(descs);
        entry.fused_ring_stream = stream;
        entry.prestaged_for_frame = true;
        entry.prestaged_stream = stream;
        entry.prestaged_unconfirmed = true;
        Ok(Some((device, payload)))
    }

    /// Execute conv1d + SiLU + L2-norm + GDN recurrence + sigmoid gated RMS
    /// norm across all sequences in one `grid=[nv, C, 1]` fused kernel launch,
    /// writing lazy-commit retain QKV and unclamped gates in place.
    #[allow(clippy::too_many_arguments)]
    fn cross_seq_recurrent_fused_multi(
        &self,
        fused_multi: KernelHandle,
        lazy: bool,
        frame: Qwen4CrossSeqFrame,
        seqs: &mut [Qwen4CrossSeqRows<'_>],
        (deinterleaved, gates, normed): (DevicePtr, DevicePtr, DevicePtr),
        (nk, kd, nv, vd): (usize, usize, usize, usize),
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let k = frame.k();
        let n = frame.seqs();
        let capturing = ctx.graph_capture
            || spark_runtime::gpu_capture_region::active_capture_region().is_some();
        let Some(CrossSeqFusedDescs {
            descs,
            h_inter_stride,
            conv_inter_stride,
            use_keep_late,
        }) = self.build_cross_seq_fused_descs(
            lazy,
            frame,
            seqs,
            (nk, kd, nv, vd),
            ctx,
            stream,
            !capturing,
        )?
        else {
            return Ok(false);
        };
        if !use_keep_late {
            for view in seqs.iter() {
                if self.gdn_keep_late_has_pending_inner(view.slot_idx) {
                    ensure!(
                        !capturing,
                        "cannot flush pending KEEP_LATE rows inside CUDA graph capture (slot={})",
                        view.slot_idx
                    );
                    if let Some(ssm) = view.state.as_any().downcast_ref::<SsmLayerState>() {
                        self.gdn_keep_late_flush_slot_inner(
                            ctx.gpu,
                            ctx.config,
                            view.slot_idx,
                            ssm.h_state,
                            ssm.conv_state,
                            None,
                            view.conv_intermediate,
                            view.conv_intermediate_stride,
                            view.h_intermediate,
                            stream,
                        )?;
                    }
                }
            }
        }

        let value_dim = nv * vd;
        let qkvz_size = ctx.config.ssm_qkvz_size();
        let eps = ctx.config.rms_norm_eps as f32;
        let desc_dev = self.upload_cross_seq_fused_descs(&descs, ctx, stream, capturing)?;
        let fp16_flags = ops::GDN_FUSED_FLAG_LOAD_SCALED_FP16
            | ops::GDN_FUSED_FLAG_STORE_SCALED_FP16
            | super::GDN_FUSED_FLAG_LOAD_SCALED_FP8
            | super::GDN_FUSED_FLAG_STORE_SCALED_FP8;
        let any_scaled_fp16 = descs.iter().any(|desc| (desc.flags & fp16_flags) != 0);
        // Under KEEP_LATE + GDN_H_FP16 / GDN_H_FP8 for C >= 2, pin the scaled-H
        // Split-V=4 kernel even on an initial FP32 step so captured CUDA graphs
        // safely replay across subsequent Scaled-FP16/FP8 load/flush transitions.
        let needs_scaled_fp16 = any_scaled_fp16
            || (lazy
                && use_keep_late
                && (super::gdn_h_fp16_enabled() || super::gdn_h_fp8_enabled())
                && n >= super::GDN_H_FP16_MIN_COHORT);
        let splitv4_k = if lazy {
            self.qwen4_gdn_verify_fused_lazyfinal_multi_splitv4_k
        } else {
            self.qwen4_gdn_verify_fused_persistent_multi_splitv4_k
        };
        let scaled_splitv4_k = if lazy && needs_scaled_fp16 {
            qwen4_gdn_scaled_fp16_splitv4_kernel(ctx.gpu)
        } else {
            KernelHandle(0)
        };
        let splitv4_plan = plan_cross_seq_splitv4_with_policy(
            super::qwen4_gdn_split_v_enabled_for_single_seq(),
            n,
            lazy,
            needs_scaled_fp16,
            splitv4_k.0 != 0,
            scaled_splitv4_k.0 != 0,
            crate::layers::deterministic_batch::deterministic_batch_policy_or_disabled(),
        );
        let max_lanes = crate::layers::qwen4_verify_lanes::MAX_LANES;
        let split_done_ptr = if splitv4_plan != CrossSeqSplitV4Plan::Disabled && n <= max_lanes {
            let default_stream = ctx.gpu.default_stream();
            // For C=1 keep per-slot counter offset; for C=2..=4 anchor at the
            // stream-disjoint half base (0 or max_lanes) so contiguous
            // `split_done + seq_idx * nv` stays in-bounds and stream-disjoint.
            let base_slot = if stream == default_stream {
                if n == 1 { seqs[0].slot_idx } else { 0 }
            } else if n == 1 {
                max_lanes + seqs[0].slot_idx
            } else {
                max_lanes
            };
            self.gdn_split_done_ptr(base_slot, nv)
        } else {
            None
        };
        let scaled_splitv4_resources = if splitv4_plan == CrossSeqSplitV4Plan::ScaledFp16Capable {
            gdn_split_v_scaled_fp16_scratch_layout(frame.rows(), n, nv)
                .filter(|layout| layout.total_bytes <= ctx.buffers.sizes().ssm_conv_out_f32)
                .and_then(|layout| {
                    let base = ctx.buffers.ssm_conv_out_f32();
                    if base.is_null() {
                        return None;
                    }
                    Some((
                        base,
                        base.offset(layout.ss_offset_bytes),
                        base.offset(layout.h_flush_offset_bytes),
                        base.offset(layout.amax_offset_bytes),
                    ))
                })
                .zip(split_done_ptr)
        } else {
            None
        };
        let fp32_splitv4_resources = if splitv4_plan == CrossSeqSplitV4Plan::Fp32Only {
            Self::gdn_split_v_scratch_ptrs(ctx, frame.rows(), nv).zip(split_done_ptr)
        } else {
            None
        };
        if lazy {
            if let Some(((scratch_gdn, scratch_ss, scratch_h_flush, scratch_amax), split_done)) =
                scaled_splitv4_resources
            {
                launch_gdn_verify_fused_lazyfinal_multi_seq_scaled_fp16_splitv4(
                    ctx.gpu,
                    scaled_splitv4_k,
                    desc_dev,
                    deinterleaved,
                    &self.ssm.conv1d,
                    gates,
                    gates.offset(nv * FP32),
                    &self.ssm.norm,
                    normed,
                    n as u32,
                    nk as u32,
                    nv as u32,
                    1e-6,
                    eps,
                    qkvz_size as u32,
                    (nv * 2) as u32,
                    value_dim as u32,
                    qkvz_size as u32,
                    (conv_inter_stride / FP32) as u32,
                    (h_inter_stride / FP32) as u32,
                    scratch_gdn,
                    scratch_ss,
                    split_done,
                    scratch_h_flush,
                    scratch_amax,
                    stream,
                )?;
            } else if let Some(((scratch_gdn, scratch_ss), split_done)) = fp32_splitv4_resources {
                ops::qwen4_gdn_verify_fused_lazyfinal_multi_seq_splitv4(
                    ctx.gpu,
                    splitv4_k,
                    desc_dev,
                    deinterleaved,
                    &self.ssm.conv1d,
                    gates,
                    gates.offset(nv * FP32),
                    &self.ssm.norm,
                    normed,
                    n as u32,
                    nk as u32,
                    nv as u32,
                    1e-6,
                    eps,
                    qkvz_size as u32,
                    (nv * 2) as u32,
                    value_dim as u32,
                    qkvz_size as u32,
                    (conv_inter_stride / FP32) as u32,
                    (h_inter_stride / FP32) as u32,
                    scratch_gdn,
                    scratch_ss,
                    split_done,
                    stream,
                )?;
            } else {
                ops::qwen4_gdn_verify_fused_lazyfinal_multi_seq(
                    ctx.gpu,
                    fused_multi,
                    desc_dev,
                    deinterleaved,
                    &self.ssm.conv1d,
                    gates,
                    gates.offset(nv * FP32),
                    &self.ssm.norm,
                    normed,
                    n as u32,
                    nk as u32,
                    nv as u32,
                    1e-6,
                    eps,
                    qkvz_size as u32,
                    (nv * 2) as u32,
                    value_dim as u32,
                    qkvz_size as u32,
                    (conv_inter_stride / FP32) as u32,
                    (h_inter_stride / FP32) as u32,
                    stream,
                )?;
            }
        } else if let Some(((scratch_gdn, scratch_ss), split_done)) = fp32_splitv4_resources {
            ops::qwen4_gdn_verify_fused_persistent_multi_seq_splitv4(
                ctx.gpu,
                splitv4_k,
                desc_dev,
                deinterleaved,
                &self.ssm.conv1d,
                gates,
                gates.offset(nv * FP32),
                &self.ssm.norm,
                normed,
                n as u32,
                nk as u32,
                nv as u32,
                1e-6,
                eps,
                qkvz_size as u32,
                (nv * 2) as u32,
                value_dim as u32,
                (conv_inter_stride / FP32) as u32,
                (h_inter_stride / FP32) as u32,
                scratch_gdn,
                scratch_ss,
                split_done,
                stream,
            )?;
        } else {
            ops::qwen4_gdn_verify_fused_persistent_multi_seq(
                ctx.gpu,
                fused_multi,
                desc_dev,
                deinterleaved,
                &self.ssm.conv1d,
                gates,
                gates.offset(nv * FP32),
                &self.ssm.norm,
                normed,
                n as u32,
                nk as u32,
                nv as u32,
                1e-6,
                eps,
                qkvz_size as u32,
                (nv * 2) as u32,
                value_dim as u32,
                (conv_inter_stride / FP32) as u32,
                (h_inter_stride / FP32) as u32,
                stream,
            )?;
        }
        for view in seqs.iter() {
            self.gdn_keep_late_mark_verify_ran(view.slot_idx, use_keep_late);
        }
        static FUSED_MULTI_ENGAGED: std::sync::Once = std::sync::Once::new();
        FUSED_MULTI_ENGAGED.call_once(|| {
            tracing::info!(
                seqs = n,
                k,
                lazy,
                keep_late = use_keep_late,
                "QWEN4_BATCHED_K5_GDN_FUSED_MULTI_ENGAGED"
            );
        });
        Ok(true)
    }

    /// Stage `GdnFusedMultiSeqDesc[0..n]` in this layer's 8-slot page-locked
    /// descriptor ring and enqueue one bounded H2D copy. Inside CUDA graph
    /// capture or when descriptors were pre-staged in Phase 1, returns the
    /// persistent device pointer directly with zero `gpu.alloc`, zero
    /// `poll_event`, zero `record_event`, and zero H2D copies.
    fn upload_cross_seq_fused_descs(
        &self,
        descs: &[ops::GdnFusedMultiSeqDesc],
        ctx: &ForwardContext,
        stream: u64,
        capturing: bool,
    ) -> Result<DevicePtr> {
        let bytes = descs.len() * ops::GdnFusedMultiSeqDesc::SIZE_BYTES;
        ensure!(
            bytes <= CROSS_SEQ_TABLE_BYTES,
            "cross-sequence fused GDN descriptors ({bytes} B) exceed {CROSS_SEQ_TABLE_BYTES} B staging"
        );
        let mut guard = self
            .cross_seq_ptr_table
            .lock()
            .map_err(|_| anyhow::anyhow!("cross-sequence pointer table lock poisoned"))?;
        if guard.is_none() {
            ensure!(
                !capturing,
                "cross-sequence fused GDN descriptor table was not pre-allocated before CUDA graph capture"
            );
            *guard = Some(CrossSeqPtrTable::allocate_for_layers(
                ctx.gpu,
                CROSS_SEQ_TABLE_BYTES,
                super::count_ssm_layers(ctx.config),
            )?);
        }
        let entry = guard
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("cross-sequence pointer table missing after init"))?;
        ensure!(
            entry.state != TableDma::Poisoned,
            "cross-sequence pointer table is poisoned"
        );
        if entry.prestaged_for_frame
            && entry.prestaged_stream == stream
            && entry.last_uploaded_fused.as_deref() == Some(descs)
        {
            if !capturing {
                entry.prestaged_for_frame = false;
                // Do not promote an externally-staged payload into the
                // cross-frame cache because the caller has no post-H2D commit
                // acknowledgement hook.
                entry.prestaged_unconfirmed = false;
                entry.last_uploaded_fused = None;
            }
            return Ok(entry.device);
        }
        if !capturing {
            entry.prestaged_for_frame = false;
            if entry.prestaged_unconfirmed {
                entry.last_uploaded_fused = None;
                entry.prestaged_unconfirmed = false;
            }
        }
        if !entry.prestaged_unconfirmed
            && entry.last_uploaded_fused.as_deref() == Some(descs)
            && (entry.fused_ring_stream == stream
                || (entry.state == TableDma::Recorded && entry.stream == stream))
        {
            return Ok(entry.device);
        }
        ensure!(
            !capturing,
            "cross-sequence fused GDN descriptors changed during CUDA graph capture without Phase-1 pre-staging"
        );
        if entry.fused_ring_slot > 0 && entry.fused_ring_stream != stream {
            let prev_stream = entry.fused_ring_stream;
            let (state, acquired) = acquire_staging(
                TableDma::Uncertain,
                entry.upload_done,
                |event| ctx.gpu.poll_event(event),
                |event| ctx.gpu.record_event(event, prev_stream),
            );
            entry.state = state;
            acquired?;
            entry.fused_ring_slot = 0;
        }
        if entry.fused_ring_slot == 0 && entry.state != TableDma::Idle {
            let copy_stream = entry.stream;
            let (state, acquired) = acquire_staging(
                entry.state,
                entry.upload_done,
                |event| ctx.gpu.poll_event(event),
                |event| ctx.gpu.record_event(event, copy_stream),
            );
            entry.state = state;
            acquired?;
        }
        let slot = entry.fused_ring_slot;
        let is_wrap_slot = slot + 1 >= FUSED_DESC_RING_SLOTS;
        if is_wrap_slot && entry.state != TableDma::Idle && entry.stream != stream {
            let copy_stream = entry.stream;
            let (state, acquired) = acquire_staging(
                entry.state,
                entry.upload_done,
                |event| ctx.gpu.poll_event(event),
                |event| ctx.gpu.record_event(event, copy_stream),
            );
            entry.state = state;
            acquired?;
        }
        entry.last_uploaded = None;
        entry.last_uploaded_fused = None;
        let ring_offset = slot * CROSS_SEQ_TABLE_BYTES;
        let staging = &mut entry.fused_ring.as_mut_slice()[ring_offset..ring_offset + bytes];
        for (index, desc) in descs.iter().enumerate() {
            let offset = index * ops::GdnFusedMultiSeqDesc::SIZE_BYTES;
            desc.write_le_bytes(
                &mut staging[offset..offset + ops::GdnFusedMultiSeqDesc::SIZE_BYTES],
            );
        }
        let device = entry.device;
        let source = entry.fused_ring.pinned_slice_range(ring_offset, bytes)?;
        entry.fused_ring_stream = stream;
        if is_wrap_slot {
            entry.state = TableDma::Uncertain;
            entry.stream = stream;
        }
        // SAFETY: `source` is a disjoint subrange of this layer's model-lifetime
        // 8-slot page-locked ring; slot 0 always waits on `upload_done` recorded
        // after the previous cycle's final slot, and stream switches drain first.
        unsafe {
            ctx.gpu.copy_h2d_pinned_async(source, device, stream)?;
        }
        if is_wrap_slot {
            entry.state = after_enqueue(entry.upload_done, || {
                ctx.gpu.record_event(entry.upload_done, stream)
            })?;
            entry.fused_ring_slot = 0;
        } else {
            entry.fused_ring_slot = slot + 1;
        }
        entry.prestaged_unconfirmed = false;
        entry.last_uploaded_fused = Some(descs.to_vec());
        Ok(device)
    }
}

/// DMA ownership of one layer's pointer-table staging.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TableDma {
    /// No copy from the staging can be in flight.
    Idle,
    /// A copy was enqueued and `upload_done` was recorded after it.
    Recorded,
    /// A copy may be in flight with no valid fence (the copy was enqueued but
    /// the event record failed, or the backend has no events). Reuse first
    /// re-records the fence after it and waits, bounded; otherwise poisoned.
    Uncertain,
    /// No bounded fence proved the staging free; it is never rewritten again
    /// and the model's admission refuses multi-sequence frames.
    Poisoned,
}

/// Page-locked staging, device copy, fence and DMA state for one SSM
/// layer's cross-sequence pointer tables.
pub(super) struct CrossSeqPtrTable {
    host: PinnedHostBuffer,
    fused_ring: PinnedHostBuffer,
    device: DevicePtr,
    /// Nonzero registry ID when `device` is sliced from a shared contiguous
    /// `[num_ssm_layers][CROSS_SEQ_TABLE_BYTES]` slab.
    slab_id: u64,
    /// 0 when the backend has no events (every upload is then `Uncertain`).
    upload_done: u64,
    /// Stream of the last enqueued copy (where a fence re-record goes).
    stream: u64,
    state: TableDma,
    last_uploaded: Option<[[u64; MULTI_TABLE_SEQS]; MULTI_TABLES]>,
    last_uploaded_fused: Option<Vec<ops::GdnFusedMultiSeqDesc>>,
    fused_ring_slot: usize,
    fused_ring_stream: u64,
    prestaged_for_frame: bool,
    prestaged_stream: u64,
    prestaged_unconfirmed: bool,
}

impl CrossSeqPtrTable {
    fn allocate_for_layers(
        gpu: &dyn GpuBackend,
        bytes: usize,
        num_ssm_layers: usize,
    ) -> Result<Self> {
        let host = gpu.alloc_host_pinned(bytes)?;
        let fused_ring = gpu.alloc_host_pinned(bytes * FUSED_DESC_RING_SLOTS)?;
        let (device, slab_id) = if num_ssm_layers > 0 {
            alloc_cross_seq_desc_slab_slice(gpu, bytes, num_ssm_layers)?
        } else {
            (gpu.alloc(bytes)?, 0)
        };
        let upload_done = match gpu.create_event() {
            Ok(event) => event,
            Err(error) => {
                if slab_id != 0 {
                    release_cross_seq_desc_slab_slice(slab_id);
                } else {
                    let _ = gpu.free(device);
                }
                return Err(error);
            }
        };
        Ok(Self {
            host,
            fused_ring,
            device,
            slab_id,
            upload_done,
            stream: 0,
            state: TableDma::Idle,
            last_uploaded: None,
            last_uploaded_fused: None,
            fused_ring_slot: 0,
            fused_ring_stream: 0,
            prestaged_for_frame: false,
            prestaged_stream: 0,
            prestaged_unconfirmed: false,
        })
    }
}

/// `state` is the table's DMA state, `None` when no table exists yet, and the
/// outer `None` a poisoned host lock.
pub(super) fn table_admits_frames(state: Option<Option<TableDma>>) -> bool {
    !matches!(state, None | Some(Some(TableDma::Poisoned)))
}

/// Longest the host waits for the previous frame's table upload.
const UPLOAD_WAIT: std::time::Duration = std::time::Duration::from_secs(5);

fn wait_upload_done(event: u64, mut poll: impl FnMut(u64) -> Result<bool>) -> Result<()> {
    let deadline = std::time::Instant::now() + UPLOAD_WAIT;
    while !poll(event)? {
        ensure!(
            std::time::Instant::now() < deadline,
            "cross-sequence pointer-table upload did not complete within {UPLOAD_WAIT:?}"
        );
        std::hint::spin_loop();
    }
    Ok(())
}

/// Make the staging writable; returns the new state and whether the host may
/// write. Only bounded, interruptible waits are used, never a stream
/// synchronize:
/// - `Idle` may write;
/// - `Recorded` with a fence waits on it (bounded); an error or timeout
///   poisons;
/// - `Uncertain` with an event re-records the fence after the possibly
///   in-flight copy (`rerecord`, on that copy's stream) and waits on the new
///   record (bounded); a failed record poisons without polling the stale
///   event, and a wait error or timeout poisons;
/// - a missing fence (no event) or `Poisoned` refuses with `Poisoned`.
pub(super) fn acquire_staging(
    state: TableDma,
    event: u64,
    poll: impl FnMut(u64) -> Result<bool>,
    rerecord: impl FnOnce(u64) -> Result<()>,
) -> (TableDma, Result<()>) {
    let poisoned = |why: String| {
        (
            TableDma::Poisoned,
            Err(anyhow::anyhow!(
                "cross-sequence pointer-table staging could not be proven free ({why}); \
                 refusing to reuse it"
            )),
        )
    };
    match state {
        TableDma::Idle => (TableDma::Idle, Ok(())),
        TableDma::Poisoned => poisoned("poisoned earlier".to_owned()),
        _ if event == 0 => poisoned("no fence event".to_owned()),
        TableDma::Recorded => match wait_upload_done(event, poll) {
            Ok(()) => (TableDma::Idle, Ok(())),
            Err(error) => poisoned(format!("fence wait: {error:#}")),
        },
        TableDma::Uncertain => match rerecord(event) {
            Err(error) => poisoned(format!("fence re-record: {error:#}")),
            Ok(()) => match wait_upload_done(event, poll) {
                Ok(()) => (TableDma::Idle, Ok(())),
                Err(error) => poisoned(format!("re-recorded fence wait: {error:#}")),
            },
        },
    }
}

/// State after the copy was enqueued: `Recorded` only when the fence was
/// recorded; a failed record leaves the copy unfenced (`Uncertain`) and
/// returns the error.
pub(super) fn after_enqueue(event: u64, record: impl FnOnce() -> Result<()>) -> Result<TableDma> {
    if event == 0 {
        return Ok(TableDma::Uncertain);
    }
    record()?;
    Ok(TableDma::Recorded)
}

#[cfg(test)]
mod c1_parity_source_tests {
    use crate::layers::ops;

    const CROSS: &str = include_str!("qwen4_cross_seq_verify.rs");
    const C1: &str = include_str!("qwen4_k5_ssm.rs");

    fn body(source: &str) -> &str {
        // Up to the trailing test module: qwen4_k5_ssm.rs has `#[cfg(test)]`
        // items near its top, so the FIRST marker cut the verifier off.
        source.rsplit_once("#[cfg(test)]").unwrap().0
    }

    #[test]
    fn cross_seq_verify_ba_gemm_gates_on_high_sm_and_rows_ge_32() {
        use super::cross_seq_verify_ba_gemm_eligible;

        // GB10 (<=64 SMs): never diverts from exact scalar BA GEMV.
        for gb10_sms in [48u32, 64u32] {
            for rows in [16usize, 32, 64, 128] {
                assert!(
                    !cross_seq_verify_ba_gemm_eligible(
                        true, rows, gb10_sms, 2_560, 96, 16, 48, 0x1000, 0x2000, 0x3000
                    ),
                    "GB10 sm={gb10_sms} rows={rows} must stay on exact scalar BA GEMV"
                );
            }
        }
        // SM120 (>64 SMs): engages at rows >= 32 under tc_dense_scope with aligned pointers.
        for sm120_sms in [65u32, 188u32] {
            assert!(!cross_seq_verify_ba_gemm_eligible(
                true, 31, sm120_sms, 2_560, 96, 16, 48, 0x1000, 0x2000, 0x3000
            ));
            assert!(cross_seq_verify_ba_gemm_eligible(
                true, 32, sm120_sms, 2_560, 96, 16, 48, 0x1000, 0x2000, 0x3000
            ));
            assert!(cross_seq_verify_ba_gemm_eligible(
                true, 128, sm120_sms, 2_560, 96, 16, 48, 0x1000, 0x2000, 0x3000
            ));
            assert!(!cross_seq_verify_ba_gemm_eligible(
                false, 128, sm120_sms, 2_560, 96, 16, 48, 0x1000, 0x2000, 0x3000
            ));
            assert!(!cross_seq_verify_ba_gemm_eligible(
                true, 128, sm120_sms, 2_560, 96, 16, 48, 0x1008, 0x2000, 0x3000
            ));
            assert!(!cross_seq_verify_ba_gemm_eligible(
                true, 128, sm120_sms, 2_560, 96, 16, 48, 0x1000, 0x2008, 0x3000
            ));
            assert!(!cross_seq_verify_ba_gemm_eligible(
                true, 128, sm120_sms, 2_560, 96, 16, 48, 0x1000, 0x2000, 0x3002
            ));
        }
    }

    #[test]
    fn recurrent_core_uses_the_c1_kernels_and_lazy_predicate() {
        for call in [
            "self.gdn_seq_lazy_engaged_inner(",
            "self.gdn_lazy_retain_current(",
            "ops::conv1d_update_l2norm_f32_sequence(",
            "ops::gdn_decode_f32_sequence_persistent(",
            "ops::gdn_decode_f32_sequence(",
            "self.gdn_f32_sequence_lazyfinal_k",
            "self.gdn_f32_sequence_persistent_k",
            "ssm_gdn_seq_persistent_enabled()",
            "qwen4_verify_direct_retain_enabled()",
            "ops::gated_rms_norm_f32_multi_seq(",
            "ops::dense_gemv_ba_gates_batchn(",
            "self.project_nvfp4_rows_exact_or_k1(",
        ] {
            assert!(body(C1).contains(call), "C1 verifier no longer uses {call}");
            assert!(
                body(CROSS).contains(call),
                "cross-sequence core lost {call}"
            );
        }
    }

    #[test]
    fn recorded_fence_is_waited_on_before_reuse() {
        use super::{TableDma, acquire_staging};
        let mut polls = 0;
        let (state, ok) = acquire_staging(
            TableDma::Recorded,
            7,
            |_| {
                polls += 1;
                Ok(polls >= 3)
            },
            |_| panic!("a recorded fence is not re-recorded"),
        );
        assert!(ok.is_ok() && state == TableDma::Idle && polls == 3);
        let (state, ok) =
            acquire_staging(TableDma::Idle, 7, |_| unreachable!(), |_| unreachable!());
        assert!(ok.is_ok() && state == TableDma::Idle);
    }

    #[test]
    fn recorded_fence_errors_poison_instead_of_synchronizing() {
        use super::{TableDma, acquire_staging};
        let (state, ok) = acquire_staging(
            TableDma::Recorded,
            7,
            |_| anyhow::bail!("query failed"),
            |_| panic!("no re-record for a recorded fence"),
        );
        assert!(ok.is_err() && state == TableDma::Poisoned);
    }

    #[test]
    fn copy_success_then_event_failure_rerecords_and_waits_bounded() {
        use super::{TableDma, acquire_staging, after_enqueue};
        // The copy was enqueued (state already Uncertain), then the record
        // failed: the upload returns the error and stays Uncertain.
        assert!(after_enqueue(7, || anyhow::bail!("record_event failed")).is_err());
        let order = std::cell::RefCell::new(Vec::new());
        let (state, ok) = acquire_staging(
            TableDma::Uncertain,
            7,
            |_| {
                order.borrow_mut().push("poll");
                Ok(true)
            },
            |_| {
                order.borrow_mut().push("rerecord");
                Ok(())
            },
        );
        assert!(ok.is_ok() && state == TableDma::Idle);
        assert_eq!(
            *order.borrow(),
            ["rerecord", "poll"],
            "never poll before the new record"
        );
    }

    #[test]
    fn a_failed_rerecord_poisons_without_polling_the_stale_event() {
        use super::{TableDma, acquire_staging};
        let (state, ok) = acquire_staging(
            TableDma::Uncertain,
            7,
            |_| panic!("the stale event must not be polled"),
            |_| anyhow::bail!("record failed again"),
        );
        assert!(ok.is_err() && state == TableDma::Poisoned);
        // A re-recorded fence that never completes times out (bounded).
        let (state, ok) = acquire_staging(TableDma::Uncertain, 7, |_| Ok(false), |_| Ok(()));
        assert!(ok.is_err() && state == TableDma::Poisoned);
    }

    #[test]
    fn missing_fences_and_poison_refuse_permanently() {
        use super::{TableDma, acquire_staging, after_enqueue};
        for state in [TableDma::Recorded, TableDma::Uncertain] {
            let (next, ok) = acquire_staging(state, 0, |_| unreachable!(), |_| unreachable!());
            assert!(ok.is_err() && next == TableDma::Poisoned, "{state:?}");
        }
        let (next, ok) = acquire_staging(
            TableDma::Poisoned,
            7,
            |_| unreachable!(),
            |_| unreachable!(),
        );
        assert!(ok.is_err() && next == TableDma::Poisoned);
        assert_eq!(
            after_enqueue(0, || unreachable!()).unwrap(),
            TableDma::Uncertain
        );
        assert_eq!(after_enqueue(7, || Ok(())).unwrap(), TableDma::Recorded);
    }

    #[test]
    fn no_host_stream_synchronize_on_any_pointer_table_path() {
        let source = body(CROSS);
        let start = source.find("fn upload_cross_seq_ptr_table(").unwrap();
        let end = source.find("fn after_enqueue(").unwrap();
        assert!(
            !source[start..end].contains(".synchronize("),
            "upload, release and acquire use bounded event waits only"
        );
    }

    #[test]
    fn poisoned_tables_refuse_frames_so_admission_falls_back() {
        use super::{TableDma, table_admits_frames};
        assert!(table_admits_frames(Some(None)), "no table yet");
        for state in [TableDma::Idle, TableDma::Recorded, TableDma::Uncertain] {
            assert!(table_admits_frames(Some(Some(state))), "{state:?}");
        }
        assert!(!table_admits_frames(Some(Some(TableDma::Poisoned))));
        assert!(
            !table_admits_frames(None),
            "a poisoned host lock refuses too"
        );
    }

    #[test]
    fn release_destroys_only_the_event_and_keeps_the_owner_in_place() {
        let release = body(CROSS)
            .split("fn release_cross_seq_ptr_table(")
            .nth(1)
            .unwrap()
            .split("fn cross_seq_table_admits_frames(")
            .next()
            .unwrap();
        assert!(
            !release.contains("gpu.free("),
            "raw device memory stays for free_all"
        );
        assert!(
            !release.contains(".take()"),
            "the pinned owner stays in the field"
        );
        assert!(!release.contains("mem::forget"));
        assert!(release.contains("gpu.destroy_event(event)"));
        assert!(release.contains("entry.state = TableDma::Poisoned;"));
    }

    #[test]
    fn upload_marks_the_staging_uncertain_before_enqueueing_the_copy() {
        let upload = body(CROSS)
            .split("fn upload_cross_seq_ptr_table(")
            .nth(1)
            .unwrap()
            .split("fn release_cross_seq_ptr_table(")
            .next()
            .unwrap();
        let acquire = upload.find("acquire_staging(").unwrap();
        let write = upload.find("copy_from_slice(").unwrap();
        let uncertain = upload.find("entry.state = TableDma::Uncertain;").unwrap();
        let copy = upload.find("copy_h2d_pinned_async(").unwrap();
        let fenced = upload.find("entry.state = after_enqueue(").unwrap();
        assert!(acquire < write && write < uncertain && uncertain < copy && copy < fenced);
    }

    #[test]
    fn multi_route_requires_the_backends_bounded_copy_mode() {
        let route = body(CROSS)
            .split("fn cross_seq_multi_gdn_kernel(")
            .nth(1)
            .unwrap()
            .split("fn cross_seq_recurrent_multi(")
            .next()
            .unwrap();
        let bounded = route.find("gpu.bounded_host_copy()").unwrap();
        let kernel = route
            .find("self.gdn_f32_sequence_lazyfinal_multi_k")
            .unwrap();
        assert!(bounded < kernel);
        assert!(
            !route.contains("ATLAS_BOUNDED_HOST_COPY\")"),
            "never the env string"
        );
    }

    #[test]
    fn gdn_multi_selector_is_strict_and_default_off() {
        assert_eq!(super::parse_gdn_multi(None), Ok(false));
        assert_eq!(super::parse_gdn_multi(Some("0")), Ok(false));
        assert_eq!(super::parse_gdn_multi(Some("1")), Ok(true));
        for hostile in ["", "true", "2", " 1"] {
            assert!(super::parse_gdn_multi(Some(hostile)).is_err());
        }
    }

    #[test]
    fn multi_route_resolves_retain_lanes_inside_each_sequence_lane() {
        let multi = body(CROSS)
            .split("fn cross_seq_recurrent_multi(")
            .nth(1)
            .unwrap()
            .split("fn upload_cross_seq_ptr_table(")
            .next()
            .unwrap();
        let lane = multi.find("LaneGuard::enter_for_slot(").unwrap();
        let retain = multi.find("self.gdn_lazy_retain_current(").unwrap();
        let launch = multi.find("ops::gdn_decode_f32_sequence_multi(").unwrap();
        assert!(lane < retain && retain < launch);
        assert!(multi.contains("qwen4_verify_direct_retain_enabled()"));
    }

    #[test]
    fn recurrent_core_runs_inside_the_sequence_lane() {
        let core = body(CROSS);
        let lane = core.find("LaneGuard::enter_for_slot(").unwrap();
        let call = core.find("self.cross_seq_recurrent_core(").unwrap();
        assert!(lane < call, "the lane must be entered before the core runs");
    }

    #[test]
    fn ssm_tc_dense_fp8_shape_admits_canonical_and_tp2_shapes_only() {
        use super::ssm_tc_dense_fp8_shape_valid;
        // Canonical TP1 and TP2 Flash-Next SSM QKVZ and out_proj shapes.
        for rows in [16, 20, 40, 80, 128] {
            assert!(ssm_tc_dense_fp8_shape_valid(rows, 16_384, 2_560));
            assert!(ssm_tc_dense_fp8_shape_valid(rows, 8_192, 2_560));
            assert!(ssm_tc_dense_fp8_shape_valid(rows, 2_560, 6_144));
            assert!(ssm_tc_dense_fp8_shape_valid(rows, 2_560, 3_072));
        }
        assert!(!ssm_tc_dense_fp8_shape_valid(0, 16_384, 2_560));
        assert!(!ssm_tc_dense_fp8_shape_valid(20, 16_380, 2_560));
        assert!(!ssm_tc_dense_fp8_shape_valid(20, 2_560, 6_140));
    }

    #[test]
    fn ragged_frames_use_seq_k_bypass_uniform_multi_and_skip_keep_late_conv_inter() {
        let source = body(CROSS);
        assert!(
            source.contains(".filter(|_| frame.seqs() > 1 && !frame.is_ragged())"),
            "uniform-k cross_seq_recurrent_multi must bypass ragged frames"
        );
        assert!(
            source.contains("crate::layers::qwen4_hybrid_cross_seq_rows(k)"),
            "cross_seq_single_route must admit ragged K=1 slots"
        );
        let preflight = source
            .split("fn preflight_qwen4_cross_seq_mixer_only_inner(")
            .nth(1)
            .unwrap()
            .split("fn preflight_qwen4_cross_seq_verify_inner(")
            .next()
            .unwrap();
        assert!(
            preflight.contains("let seq_k = frame.seq_k(s);"),
            "preflight must validate every sequence's frame.seq_k(s)"
        );
        let build = source
            .split("fn build_cross_seq_fused_descs(")
            .nth(1)
            .unwrap()
            .split("fn stage_qwen4_cross_seq_fused_desc_bytes_inner(")
            .next()
            .unwrap();
        assert!(
            build.contains("let seq_k = frame.seq_k(s);")
                && build.contains("conv_inter: 0,")
                && build.contains("keep_late_ring_fits("),
            "build_cross_seq_fused_descs must use seq_k, keep_late_ring_fits, and conv_inter: 0 under KEEP_LATE"
        );
        let per_seq = source
            .split("fn cross_seq_recurrent_per_seq(")
            .nth(1)
            .unwrap()
            .split("fn cross_seq_recurrent_core(")
            .next()
            .unwrap();
        assert!(
            per_seq.contains("let seq_k = frame.seq_k(s);"),
            "cross_seq_recurrent_per_seq must advance each sequence by frame.seq_k(s)"
        );
    }

    #[test]
    fn fused_descs_propagate_scaled_fp16_flags_and_validate_prestaged_payload() {
        let source = body(CROSS);
        let build = source
            .split("fn build_cross_seq_fused_descs(")
            .nth(1)
            .unwrap()
            .split("fn stage_qwen4_cross_seq_fused_desc_bytes_inner(")
            .next()
            .unwrap();
        assert!(
            build.contains("flags: staged_ctrl.desc_flags(),")
                && build.contains("h_state_inter: staged_ctrl.ctrl_ptr.0,"),
            "build_cross_seq_fused_descs must emit staged Scaled-FP16 descriptor flags"
        );
        let upload = source
            .split("fn upload_cross_seq_fused_descs(")
            .nth(1)
            .unwrap()
            .split("pub(super) enum TableDma")
            .next()
            .unwrap();
        assert!(
            upload.contains("entry.prestaged_for_frame")
                && upload.contains("entry.prestaged_stream == stream")
                && upload.contains("entry.last_uploaded_fused.as_deref() == Some(descs)"),
            "upload_cross_seq_fused_descs must verify prestaged descriptors and stream match before skipping H2D"
        );
    }

    #[test]
    fn cross_seq_desc_slab_packs_all_ssm_layers_at_exact_table_stride() {
        use super::{
            CROSS_SEQ_TABLE_BYTES, alloc_cross_seq_desc_slab_slice,
            release_cross_seq_desc_slab_slice,
        };
        use spark_runtime::gpu::mock::MockGpuBackend;

        let gpu = MockGpuBackend::new();
        let num_layers = 36usize;
        let mut slices: Vec<(spark_runtime::gpu::DevicePtr, u64)> = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            let (dev, slab_id) =
                alloc_cross_seq_desc_slab_slice(&gpu, CROSS_SEQ_TABLE_BYTES, num_layers)
                    .expect("alloc cross-seq desc slab slice");
            assert_ne!(slab_id, 0);
            if layer > 0 {
                assert_eq!(slab_id, slices[0].1);
                assert_eq!(
                    dev.0,
                    slices[0].0.0 + (layer * CROSS_SEQ_TABLE_BYTES) as u64,
                    "layer {layer} must sit at exact CROSS_SEQ_TABLE_BYTES stride"
                );
            }
            slices.push((dev, slab_id));
        }

        // Next model allocates a fresh non-aliasing slab.
        let (other_dev, other_slab_id) =
            alloc_cross_seq_desc_slab_slice(&gpu, CROSS_SEQ_TABLE_BYTES, num_layers)
                .expect("second model fresh slab");
        assert_ne!(other_slab_id, slices[0].1);
        assert_ne!(other_dev, slices[0].0);

        for (_, slab_id) in slices {
            release_cross_seq_desc_slab_slice(slab_id);
        }
        release_cross_seq_desc_slab_slice(other_slab_id);
    }

    #[test]
    fn cached_ssm_cross_seq_env_helpers_preserve_defaults_toggles_and_fail_closed_errors() {
        use super::{
            GDN_MULTI_ENV, gdn_multi_selected, qwen4_k5_batch_ssm_selected,
            qwen4_k16_batched_verify_cached, qwen4_k16_exact_cached,
            qwen4_verify_direct_retain_enabled, ssm_gdn_seq_persistent_enabled,
        };
        use crate::layers::qwen4_verify_lanes::{ScopedEnvVar, test_env_lock};

        let _lock = test_env_lock();

        // `qwen4_k5_batch_ssm_selected`: default OFF, "1" enables.
        {
            let _g = ScopedEnvVar::remove("ATLAS_QWEN4_K5_BATCH_SSM");
            assert!(!qwen4_k5_batch_ssm_selected());
            let _g1 = ScopedEnvVar::set("ATLAS_QWEN4_K5_BATCH_SSM", "1");
            assert!(qwen4_k5_batch_ssm_selected());
            let _g0 = ScopedEnvVar::set("ATLAS_QWEN4_K5_BATCH_SSM", "0");
            assert!(!qwen4_k5_batch_ssm_selected());
        }

        // `qwen4_k16_batched_verify_cached` & `qwen4_k16_exact_cached`: strict 0/1, fail-closed.
        {
            let _g = ScopedEnvVar::remove("ATLAS_QWEN4_K16_BATCHED_VERIFY");
            assert!(!qwen4_k16_batched_verify_cached().unwrap());
            let _g1 = ScopedEnvVar::set("ATLAS_QWEN4_K16_BATCHED_VERIFY", "1");
            assert!(qwen4_k16_batched_verify_cached().unwrap());
            let _g0 = ScopedEnvVar::set("ATLAS_QWEN4_K16_BATCHED_VERIFY", "0");
            assert!(!qwen4_k16_batched_verify_cached().unwrap());
            let _gb = ScopedEnvVar::set("ATLAS_QWEN4_K16_BATCHED_VERIFY", "bad");
            assert!(qwen4_k16_batched_verify_cached().is_err());
        }
        {
            let _g = ScopedEnvVar::remove("ATLAS_QWEN4_K16_EXACT");
            assert!(!qwen4_k16_exact_cached().unwrap());
            let _g1 = ScopedEnvVar::set("ATLAS_QWEN4_K16_EXACT", "1");
            assert!(qwen4_k16_exact_cached().unwrap());
            let _g0 = ScopedEnvVar::set("ATLAS_QWEN4_K16_EXACT", "0");
            assert!(!qwen4_k16_exact_cached().unwrap());
            let _gb = ScopedEnvVar::set("ATLAS_QWEN4_K16_EXACT", "2");
            assert!(qwen4_k16_exact_cached().is_err());
        }

        // `qwen4_verify_direct_retain_enabled` & `ssm_gdn_seq_persistent_enabled`: default ON, "0" disables.
        {
            let _g = ScopedEnvVar::remove("ATLAS_QWEN4_VERIFY_DIRECT_RETAIN");
            assert!(qwen4_verify_direct_retain_enabled());
            let _g0 = ScopedEnvVar::set("ATLAS_QWEN4_VERIFY_DIRECT_RETAIN", "0");
            assert!(!qwen4_verify_direct_retain_enabled());
            let _g1 = ScopedEnvVar::set("ATLAS_QWEN4_VERIFY_DIRECT_RETAIN", "1");
            assert!(qwen4_verify_direct_retain_enabled());
        }
        {
            let _g = ScopedEnvVar::remove("ATLAS_SSM_GDN_SEQ_PERSISTENT");
            assert!(ssm_gdn_seq_persistent_enabled());
            let _g0 = ScopedEnvVar::set("ATLAS_SSM_GDN_SEQ_PERSISTENT", "0");
            assert!(!ssm_gdn_seq_persistent_enabled());
            let _g1 = ScopedEnvVar::set("ATLAS_SSM_GDN_SEQ_PERSISTENT", "1");
            assert!(ssm_gdn_seq_persistent_enabled());
        }

        // `gdn_multi_selected`: strict 0/1, fail-closed on invalid values.
        {
            let _g = ScopedEnvVar::remove(GDN_MULTI_ENV);
            assert!(!gdn_multi_selected().unwrap());
            let _g1 = ScopedEnvVar::set(GDN_MULTI_ENV, "1");
            assert!(gdn_multi_selected().unwrap());
            let _g0 = ScopedEnvVar::set(GDN_MULTI_ENV, "0");
            assert!(!gdn_multi_selected().unwrap());
            let _gb = ScopedEnvVar::set(GDN_MULTI_ENV, "true");
            assert!(gdn_multi_selected().is_err());
        }
    }

    #[test]
    fn multi_seq_splitv4_policy_and_scaled_fp16_scratch_layout() {
        use super::{
            CrossSeqSplitV4Plan, GDN_MULTI_SEQ_SPLIT_V_MAX_SEQS,
            gdn_split_v_scaled_fp16_scratch_layout, plan_cross_seq_splitv4,
        };

        assert_eq!(GDN_MULTI_SEQ_SPLIT_V_MAX_SEQS, 4);
        // Disabled on GB10 (split_v_enabled == false) across all C.
        for c in 1..=8 {
            assert_eq!(
                plan_cross_seq_splitv4(false, c, true, true, true, true),
                CrossSeqSplitV4Plan::Disabled
            );
        }
        // Enabled on >64 SMs: C=1..=4 engages Split-V=4 (FP32 or Scaled-FP16); C>=5 falls back.
        for c in 1..=4 {
            assert_eq!(
                plan_cross_seq_splitv4(true, c, true, false, true, true),
                CrossSeqSplitV4Plan::Fp32Only
            );
            assert_eq!(
                plan_cross_seq_splitv4(true, c, true, true, true, true),
                CrossSeqSplitV4Plan::ScaledFp16Capable
            );
            // Missing scaled_fp16 kernel fails closed to Disabled rather than running FP32-only splitv4.
            assert_eq!(
                plan_cross_seq_splitv4(true, c, true, true, true, false),
                CrossSeqSplitV4Plan::Disabled
            );
        }
        assert_eq!(
            plan_cross_seq_splitv4(true, 5, true, true, true, true),
            CrossSeqSplitV4Plan::Disabled
        );
        assert_eq!(
            plan_cross_seq_splitv4(true, 8, true, true, true, true),
            CrossSeqSplitV4Plan::Disabled
        );

        use super::plan_cross_seq_splitv4_with_policy;
        use crate::layers::deterministic_batch::DeterministicBatchPolicy;
        for c in [1usize, 2, 4, 8, 16, 32, 64, 128] {
            assert_eq!(
                plan_cross_seq_splitv4_with_policy(
                    true,
                    c,
                    true,
                    false,
                    true,
                    true,
                    DeterministicBatchPolicy::Enabled,
                ),
                CrossSeqSplitV4Plan::Disabled,
                "cross-seq split-V must stay disabled at C={c}"
            );
            assert_eq!(
                plan_cross_seq_splitv4_with_policy(
                    true,
                    c,
                    true,
                    true,
                    true,
                    true,
                    DeterministicBatchPolicy::Enabled,
                ),
                CrossSeqSplitV4Plan::Disabled,
                "cross-seq scaled-FP16 split-V must stay disabled at C={c}"
            );
        }
        assert_eq!(
            plan_cross_seq_splitv4_with_policy(
                true,
                1,
                true,
                true,
                true,
                true,
                DeterministicBatchPolicy::Disabled,
            ),
            CrossSeqSplitV4Plan::ScaledFp16Capable
        );

        // Scratch layout for C=4, K=8 (32 rows), nv=48 fits well within 40 MiB GDN_FI arena.
        let layout = gdn_split_v_scaled_fp16_scratch_layout(32, 4, 48)
            .expect("C=4 K=8 layout must not overflow");
        assert_eq!(layout.h_flush_offset_bytes % 256, 0);
        assert_eq!(
            layout.amax_offset_bytes - layout.h_flush_offset_bytes,
            4 * 48 * 128 * 128 * 4
        );
        assert_eq!(
            layout.total_bytes - layout.amax_offset_bytes,
            4 * 48 * 4 * 4
        );
        assert!(
            layout.total_bytes < (16 << 20),
            "total={}",
            layout.total_bytes
        );
    }

    #[test]
    #[ignore = "needs an idle GB10 GPU and the flash-next kernel target"]
    fn qwen4_gdn_multi_seq_scaled_fp16_splitv4_parity_on_gb10() {
        use half::bf16;
        use spark_runtime::cuda_backend::AtlasCudaBackend;
        use spark_runtime::gpu::{DevicePtr, GpuBackend};

        use super::{
            gdn_split_v_scaled_fp16_scratch_layout,
            launch_gdn_verify_fused_lazyfinal_multi_seq_scaled_fp16_splitv4,
        };

        let ptx = atlas_kernels::ptx_for_model("flash-next").expect("flash-next PTX target");
        let gpu = AtlasCudaBackend::new(0, &ptx.modules).expect("init GB10 CUDA backend");
        let stream = gpu.default_stream();

        let base_multi_k = gpu
            .kernel(
                "qwen4_gdn_verify_fused",
                "qwen4_gdn_verify_fused_lazyfinal_multi_seq",
            )
            .expect("base lazyfinal multi_seq kernel");
        let fp32_sv4_k = gpu
            .kernel(
                "qwen4_gdn_verify_fused",
                "qwen4_gdn_verify_fused_lazyfinal_multi_seq_splitv4",
            )
            .expect("fp32 lazyfinal multi_seq splitv4 kernel");
        let scaled_sv4_k = gpu
            .kernel(
                "qwen4_gdn_verify_fused",
                "qwen4_gdn_verify_fused_lazyfinal_multi_seq_scaled_fp16_splitv4",
            )
            .expect("scaled_fp16 lazyfinal multi_seq splitv4 kernel");

        let nk = 16usize;
        let nv = 48usize;
        let dim = 128usize;
        let d_conv = 4usize;
        let key_dim = nk * dim;
        let value_dim = nv * dim;
        let conv_dim = 2 * key_dim + value_dim;
        let qkvz_size = conv_dim + value_dim;
        let h_elems = nv * dim * dim;
        let ring_cap = 32usize;
        let ring_row_bytes = (conv_dim * 2 + nv * 8 + 15) & !15;
        let ring_bytes = ring_cap * ring_row_bytes;
        let eps = 1e-6f32;

        let lcg_f32 = |n: usize, mut seed: u64, scale: f32| -> Vec<f32> {
            let mut out = Vec::with_capacity(n);
            for _ in 0..n {
                seed = seed
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let u = ((seed >> 40) as f32) / ((1u32 << 24) as f32);
                out.push((u * 2.0 - 1.0) * scale);
            }
            out
        };
        let to_bf16_bytes = |vals: &[f32]| -> Vec<u8> {
            let mut out = Vec::with_capacity(vals.len() * 2);
            for &v in vals {
                out.extend_from_slice(&bf16::from_f32(v).to_bits().to_le_bytes());
            }
            out
        };
        let to_f32_bytes = |vals: &[f32]| -> Vec<u8> {
            let mut out = Vec::with_capacity(vals.len() * 4);
            for &v in vals {
                out.extend_from_slice(&v.to_le_bytes());
            }
            out
        };
        let upload = |bytes: &[u8]| -> DevicePtr {
            let ptr = gpu.alloc(bytes.len().max(16)).unwrap();
            if !bytes.is_empty() {
                gpu.copy_h2d(bytes, ptr).unwrap();
            }
            ptr
        };
        let download = |ptr: DevicePtr, len: usize| -> Vec<u8> {
            gpu.synchronize(stream).unwrap();
            let mut out = vec![0u8; len];
            gpu.copy_d2h(ptr, &mut out).unwrap();
            out
        };

        let conv_w_ptr = upload(&to_bf16_bytes(&lcg_f32(conv_dim * d_conv, 101, 0.25)));
        let norm_w_ptr = upload(&to_bf16_bytes(&lcg_f32(dim, 202, 1.0)));
        let conv_w = crate::weight_map::DenseWeight { weight: conv_w_ptr };
        let norm_w = crate::weight_map::DenseWeight { weight: norm_w_ptr };

        let make_gates = |rows: usize, seed: u64| -> Vec<f32> {
            let raw = lcg_f32(rows * nv * 2, seed, 1.0);
            let mut out = vec![0.0f32; rows * nv * 2];
            for r in 0..rows {
                for h in 0..nv {
                    out[r * nv * 2 + h] = 0.75 + 0.20 * raw[r * nv * 2 + h].abs();
                    out[r * nv * 2 + nv + h] = 0.05 + 0.35 * raw[r * nv * 2 + nv + h].abs();
                }
            }
            out
        };

        let max_bf16_ulp_and_abs = |a: &[u8], b: &[u8]| -> (u32, f32, f32) {
            assert_eq!(a.len(), b.len());
            let mut max_ulp_sig = 0u32;
            let mut max_abs_all = 0.0f32;
            let mut max_abs_near_zero = 0.0f32;
            for (ca, cb) in a.chunks_exact(2).zip(b.chunks_exact(2)) {
                let ua = u16::from_le_bytes([ca[0], ca[1]]);
                let ub = u16::from_le_bytes([cb[0], cb[1]]);
                let fa = bf16::from_bits(ua).to_f32();
                let fb = bf16::from_bits(ub).to_f32();
                let diff = (fa - fb).abs();
                max_abs_all = max_abs_all.max(diff);
                if fa.abs().max(fb.abs()) >= 5e-2 {
                    let ulp = if (ua >> 15) == (ub >> 15) {
                        u32::from(ua.abs_diff(ub))
                    } else {
                        u32::MAX
                    };
                    max_ulp_sig = max_ulp_sig.max(ulp);
                } else {
                    max_abs_near_zero = max_abs_near_zero.max(diff);
                }
            }
            (max_ulp_sig, max_abs_all, max_abs_near_zero)
        };

        // Test C = 1, 2, 4 across K = 2, 5, 8 through full Scaled-FP16 lifecycle:
        //   step 0: FP32 -> ScaledFp16 (flush=0, pending=0)
        //   step 1: FP32 -> ScaledFp16 (flush=1, pending=K)
        //   step 2: ScaledFp16 -> ScaledFp16 (flush=0, pending=0)
        //   step 3: ScaledFp16 -> ScaledFp16 (flush=1, pending=K)
        //   step 4: ScaledFp16 -> FP32 cold unpack (flush=1, pending=0, ntok=0 -> exact 0-ULP H!)
        let base_kl = ops::GDN_FUSED_FLAG_COPY_GATES_TO_RETAIN
            | ops::GDN_FUSED_FLAG_KEEP_LATE
            | ops::GDN_FUSED_FLAG_KEEP_LATE_CTRL_PTR;

        for &c in &[1usize, 2, 4] {
            for &k in &[2usize, 5, 8] {
                let total_rows = c * k;
                let layout = gdn_split_v_scaled_fp16_scratch_layout(total_rows, c, nv)
                    .expect("scratch layout");
                let scratch_base = gpu.alloc(layout.total_bytes).unwrap();
                let scratch_gdn = scratch_base;
                let scratch_ss = scratch_base.offset(layout.ss_offset_bytes);
                let scratch_h_flush = scratch_base.offset(layout.h_flush_offset_bytes);
                let scratch_amax = scratch_base.offset(layout.amax_offset_bytes);
                let split_done = gpu.alloc(c * nv * 4).unwrap();
                gpu.memset(split_done, 0, c * nv * 4).unwrap();

                let mut conv_b = Vec::with_capacity(c);
                let mut conv_s = Vec::with_capacity(c);
                let mut h_b = Vec::with_capacity(c);
                let mut h_s = Vec::with_capacity(c);
                let mut ring_b = Vec::with_capacity(c);
                let mut ring_s = Vec::with_capacity(c);
                let mut gdone_b = Vec::with_capacity(c);
                let mut gdone_s = Vec::with_capacity(c);
                let mut ctrl_b = Vec::with_capacity(c);
                let mut ctrl_s = Vec::with_capacity(c);
                for seq in 0..c {
                    let seed = (c * 1000 + k * 100 + seq * 17) as u64;
                    let init_conv = to_f32_bytes(&lcg_f32(conv_dim * d_conv, seed + 1, 0.5));
                    let init_h = to_f32_bytes(&lcg_f32(h_elems, seed + 2, 0.25));
                    conv_b.push(upload(&init_conv));
                    conv_s.push(upload(&init_conv));
                    h_b.push(upload(&init_h));
                    h_s.push(upload(&init_h));
                    let rb = gpu.alloc(ring_bytes).unwrap();
                    let rs = gpu.alloc(ring_bytes).unwrap();
                    gpu.memset(rb, 0, ring_bytes).unwrap();
                    gpu.memset(rs, 0, ring_bytes).unwrap();
                    ring_b.push(rb);
                    ring_s.push(rs);
                    let gb = gpu.alloc(32 * 4).unwrap();
                    let gs = gpu.alloc(32 * 4).unwrap();
                    gpu.memset(gb, 0, 32 * 4).unwrap();
                    gpu.memset(gs, 0, 32 * 4).unwrap();
                    gdone_b.push(gb);
                    gdone_s.push(gs);
                    ctrl_b.push(gpu.alloc(ops::GdnKeepLateCtrl::SIZE_BYTES).unwrap());
                    ctrl_s.push(gpu.alloc(ops::GdnKeepLateCtrl::SIZE_BYTES).unwrap());
                }
                let desc_b_dev = gpu
                    .alloc(c * ops::GdnFusedMultiSeqDesc::SIZE_BYTES)
                    .unwrap();
                let desc_s_dev = gpu
                    .alloc(c * ops::GdnFusedMultiSeqDesc::SIZE_BYTES)
                    .unwrap();

                // 0-ULP check: FP32-only mode of scaled_fp16_splitv4 vs fp32_splitv4.
                {
                    let qkvz = upload(&to_bf16_bytes(&lcg_f32(
                        total_rows * qkvz_size,
                        (c * 77 + k) as u64,
                        0.6,
                    )));
                    let gates = upload(&to_f32_bytes(&make_gates(total_rows, (c * 88 + k) as u64)));
                    let norm_fp32_sv4 = gpu.alloc(total_rows * value_dim * 2).unwrap();
                    let norm_scaled_sv4 = gpu.alloc(total_rows * value_dim * 2).unwrap();
                    let mut dbytes = vec![0u8; c * ops::GdnFusedMultiSeqDesc::SIZE_BYTES];
                    for seq in 0..c {
                        let ctrl = ops::GdnKeepLateCtrl {
                            ring_base: ring_s[seq].0,
                            pending_base: 0,
                            pending_rows: 0,
                            write_base: 0,
                            executed_flag: 0,
                            ring_cap: ring_cap as u32,
                            flush_now: 0,
                        };
                        let mut cb = [0u8; ops::GdnKeepLateCtrl::SIZE_BYTES];
                        ctrl.write_le_bytes(&mut cb);
                        gpu.copy_h2d(&cb, ctrl_s[seq]).unwrap();
                        let desc = ops::GdnFusedMultiSeqDesc {
                            conv_state: conv_s[seq].0,
                            h_state: h_s[seq].0,
                            retain_qkv: ring_s[seq].0,
                            conv_inter: 0,
                            h_state_inter: ctrl_s[seq].0,
                            group_done: gdone_s[seq].0,
                            row_base: (seq * k) as u32,
                            num_tokens: k as u32,
                            flags: base_kl,
                            _pad: 0,
                        };
                        desc.write_le_bytes(
                            &mut dbytes[seq * ops::GdnFusedMultiSeqDesc::SIZE_BYTES
                                ..(seq + 1) * ops::GdnFusedMultiSeqDesc::SIZE_BYTES],
                        );
                    }
                    gpu.copy_h2d(&dbytes, desc_s_dev).unwrap();
                    ops::qwen4_gdn_verify_fused_lazyfinal_multi_seq_splitv4(
                        &gpu,
                        fp32_sv4_k,
                        desc_s_dev,
                        qkvz,
                        &conv_w,
                        gates,
                        gates.offset(nv * 4),
                        &norm_w,
                        norm_fp32_sv4,
                        c as u32,
                        nk as u32,
                        nv as u32,
                        eps,
                        eps,
                        qkvz_size as u32,
                        (nv * 2) as u32,
                        value_dim as u32,
                        qkvz_size as u32,
                        0,
                        0,
                        scratch_gdn,
                        scratch_ss,
                        split_done,
                        stream,
                    )
                    .unwrap();
                    for seq in 0..c {
                        gpu.copy_d2d_async(conv_b[seq], conv_s[seq], conv_dim * d_conv * 4, stream)
                            .unwrap();
                        gpu.memset_async(ring_s[seq], 0, ring_bytes, stream)
                            .unwrap();
                    }
                    launch_gdn_verify_fused_lazyfinal_multi_seq_scaled_fp16_splitv4(
                        &gpu,
                        scaled_sv4_k,
                        desc_s_dev,
                        qkvz,
                        &conv_w,
                        gates,
                        gates.offset(nv * 4),
                        &norm_w,
                        norm_scaled_sv4,
                        c as u32,
                        nk as u32,
                        nv as u32,
                        eps,
                        eps,
                        qkvz_size as u32,
                        (nv * 2) as u32,
                        value_dim as u32,
                        qkvz_size as u32,
                        0,
                        0,
                        scratch_gdn,
                        scratch_ss,
                        split_done,
                        scratch_h_flush,
                        scratch_amax,
                        stream,
                    )
                    .unwrap();
                    assert_eq!(
                        download(norm_scaled_sv4, total_rows * value_dim * 2),
                        download(norm_fp32_sv4, total_rows * value_dim * 2),
                        "C={c} K={k} FP32-mode scaled_fp16_splitv4 must be 0-ULP identical to fp32_splitv4"
                    );
                    for seq in 0..c {
                        gpu.copy_d2d_async(conv_b[seq], conv_s[seq], conv_dim * d_conv * 4, stream)
                            .unwrap();
                        gpu.memset_async(ring_s[seq], 0, ring_bytes, stream)
                            .unwrap();
                    }
                    gpu.synchronize(stream).unwrap();
                }

                // Full Scaled-FP16 multi-step lifecycle vs non-split `lazyfinal_multi_seq`:
                //   step 0: FP32 -> ScaledFp16 (flush=0, pending=0, ntok=K)
                //   step 1: FP32 -> ScaledFp16 (flush=1, pending=K, ntok=K)
                //   step 2: ScaledFp16 -> ScaledFp16 (flush=0, pending=0, ntok=K)
                //   step 3: ScaledFp16 -> ScaledFp16 (flush=1, pending=K, ntok=K)
                //   step 4: ScaledFp16 -> FP32 cold unpack (flush=1, pending=0, ntok=0)
                //   step 5: 0-ULP bit-identical ScaledFp16 -> FP32 cold unpack from identical packed H
                let steps: [(u32, u32, u32, u32, u32, u32); 6] = [
                    // (pending_base, pending_rows, write_base, flush_now, ntok, extra_flags)
                    (0, 0, 0, 0, k as u32, ops::GDN_FUSED_FLAG_STORE_SCALED_FP16),
                    (
                        0,
                        k as u32,
                        k as u32,
                        1,
                        k as u32,
                        ops::GDN_FUSED_FLAG_STORE_SCALED_FP16,
                    ),
                    (
                        k as u32,
                        0,
                        (2 * k) as u32,
                        0,
                        k as u32,
                        ops::GDN_FUSED_FLAG_LOAD_SCALED_FP16
                            | ops::GDN_FUSED_FLAG_STORE_SCALED_FP16,
                    ),
                    (
                        k as u32,
                        k as u32,
                        (2 * k) as u32,
                        1,
                        k as u32,
                        ops::GDN_FUSED_FLAG_LOAD_SCALED_FP16
                            | ops::GDN_FUSED_FLAG_STORE_SCALED_FP16,
                    ),
                    (0, 0, 0, 1, 0, ops::GDN_FUSED_FLAG_LOAD_SCALED_FP16),
                    (0, 0, 0, 1, 0, ops::GDN_FUSED_FLAG_LOAD_SCALED_FP16),
                ];
                let mut saved_packed_h = Vec::with_capacity(c);
                for _ in 0..c {
                    saved_packed_h.push(gpu.alloc(h_elems * 4).unwrap());
                }
                for (step_idx, &(pbase, prows, wbase, flush_now, ntok, extra_flags)) in
                    steps.iter().enumerate()
                {
                    if step_idx == 5 {
                        // Seed BOTH h_b and h_s with identical valid packed Scaled-FP16 bytes
                        // captured right after step 3 to verify 0-ULP cold unpack parity.
                        for seq in 0..c {
                            gpu.copy_d2d_async(saved_packed_h[seq], h_b[seq], h_elems * 4, stream)
                                .unwrap();
                            gpu.copy_d2d_async(saved_packed_h[seq], h_s[seq], h_elems * 4, stream)
                                .unwrap();
                        }
                    }
                    let step_rows = (c * ntok as usize).max(1);
                    let qkvz = upload(&to_bf16_bytes(&lcg_f32(
                        step_rows * qkvz_size,
                        (c * 10000 + k * 100 + step_idx) as u64,
                        0.7,
                    )));
                    let gates = upload(&to_f32_bytes(&make_gates(
                        step_rows,
                        (c * 20000 + k * 100 + step_idx) as u64,
                    )));
                    let norm_b = gpu.alloc(step_rows * value_dim * 2).unwrap();
                    let norm_s = gpu.alloc(step_rows * value_dim * 2).unwrap();
                    gpu.memset(norm_b, 0, step_rows * value_dim * 2).unwrap();
                    gpu.memset(norm_s, 0, step_rows * value_dim * 2).unwrap();

                    let mut db = vec![0u8; c * ops::GdnFusedMultiSeqDesc::SIZE_BYTES];
                    let mut ds = vec![0u8; c * ops::GdnFusedMultiSeqDesc::SIZE_BYTES];
                    for seq in 0..c {
                        let cb_struct = ops::GdnKeepLateCtrl {
                            ring_base: ring_b[seq].0,
                            pending_base: pbase,
                            pending_rows: prows,
                            write_base: wbase,
                            executed_flag: 0,
                            ring_cap: ring_cap as u32,
                            flush_now,
                        };
                        let cs_struct = ops::GdnKeepLateCtrl {
                            ring_base: ring_s[seq].0,
                            ..cb_struct
                        };
                        let mut cbuf = [0u8; ops::GdnKeepLateCtrl::SIZE_BYTES];
                        cb_struct.write_le_bytes(&mut cbuf);
                        gpu.copy_h2d(&cbuf, ctrl_b[seq]).unwrap();
                        cs_struct.write_le_bytes(&mut cbuf);
                        gpu.copy_h2d(&cbuf, ctrl_s[seq]).unwrap();

                        let desc_b = ops::GdnFusedMultiSeqDesc {
                            conv_state: conv_b[seq].0,
                            h_state: h_b[seq].0,
                            retain_qkv: ring_b[seq].0,
                            conv_inter: 0,
                            h_state_inter: ctrl_b[seq].0,
                            group_done: gdone_b[seq].0,
                            row_base: (seq as u32) * ntok,
                            num_tokens: ntok,
                            flags: base_kl | extra_flags,
                            _pad: 0,
                        };
                        let desc_s = ops::GdnFusedMultiSeqDesc {
                            conv_state: conv_s[seq].0,
                            h_state: h_s[seq].0,
                            retain_qkv: ring_s[seq].0,
                            conv_inter: 0,
                            h_state_inter: ctrl_s[seq].0,
                            group_done: gdone_s[seq].0,
                            ..desc_b
                        };
                        let off = seq * ops::GdnFusedMultiSeqDesc::SIZE_BYTES;
                        desc_b.write_le_bytes(
                            &mut db[off..off + ops::GdnFusedMultiSeqDesc::SIZE_BYTES],
                        );
                        desc_s.write_le_bytes(
                            &mut ds[off..off + ops::GdnFusedMultiSeqDesc::SIZE_BYTES],
                        );
                    }
                    gpu.copy_h2d(&db, desc_b_dev).unwrap();
                    gpu.copy_h2d(&ds, desc_s_dev).unwrap();

                    ops::qwen4_gdn_verify_fused_lazyfinal_multi_seq(
                        &gpu,
                        base_multi_k,
                        desc_b_dev,
                        qkvz,
                        &conv_w,
                        gates,
                        gates.offset(nv * 4),
                        &norm_w,
                        norm_b,
                        c as u32,
                        nk as u32,
                        nv as u32,
                        eps,
                        eps,
                        qkvz_size as u32,
                        (nv * 2) as u32,
                        value_dim as u32,
                        qkvz_size as u32,
                        0,
                        0,
                        stream,
                    )
                    .unwrap();
                    launch_gdn_verify_fused_lazyfinal_multi_seq_scaled_fp16_splitv4(
                        &gpu,
                        scaled_sv4_k,
                        desc_s_dev,
                        qkvz,
                        &conv_w,
                        gates,
                        gates.offset(nv * 4),
                        &norm_w,
                        norm_s,
                        c as u32,
                        nk as u32,
                        nv as u32,
                        eps,
                        eps,
                        qkvz_size as u32,
                        (nv * 2) as u32,
                        value_dim as u32,
                        qkvz_size as u32,
                        0,
                        0,
                        scratch_gdn,
                        scratch_ss,
                        split_done,
                        scratch_h_flush,
                        scratch_amax,
                        stream,
                    )
                    .unwrap();

                    for seq in 0..c {
                        assert_eq!(
                            download(conv_s[seq], conv_dim * d_conv * 4),
                            download(conv_b[seq], conv_dim * d_conv * 4),
                            "C={c} K={k} step={step_idx} seq={seq} conv_state 0-ULP mismatch"
                        );
                        assert_eq!(
                            download(ring_s[seq], ring_bytes),
                            download(ring_b[seq], ring_bytes),
                            "C={c} K={k} step={step_idx} seq={seq} ring 0-ULP mismatch"
                        );
                        assert_eq!(
                            download(gdone_s[seq], 32 * 4),
                            vec![0u8; 32 * 4],
                            "C={c} K={k} step={step_idx} seq={seq} group_done not reset"
                        );
                        let ctrl_after = download(ctrl_s[seq], ops::GdnKeepLateCtrl::SIZE_BYTES);
                        let executed = u32::from_le_bytes(ctrl_after[20..24].try_into().unwrap());
                        assert_eq!(
                            executed, 1,
                            "C={c} K={k} step={step_idx} seq={seq} executed"
                        );
                    }
                    assert_eq!(
                        download(split_done, c * nv * 4),
                        vec![0u8; c * nv * 4],
                        "C={c} K={k} step={step_idx} split_done not self-reset"
                    );
                    if ntok > 0 {
                        let (max_ulp, max_abs, max_near_zero) = max_bf16_ulp_and_abs(
                            &download(norm_s, step_rows * value_dim * 2),
                            &download(norm_b, step_rows * value_dim * 2),
                        );
                        eprintln!(
                            "scaled_fp16_splitv4 C={c} K={k} step={step_idx}: norm_ulp={max_ulp} norm_abs={max_abs:.3e} near0={max_near_zero:.3e}"
                        );
                        assert!(
                            max_ulp <= 1 && max_near_zero <= 1e-3 && max_abs <= 4e-3,
                            "C={c} K={k} step={step_idx}: norm_ulp={max_ulp} norm_abs={max_abs} near0={max_near_zero}"
                        );
                    }
                    if step_idx == 3 {
                        for seq in 0..c {
                            gpu.copy_d2d_async(h_b[seq], saved_packed_h[seq], h_elems * 4, stream)
                                .unwrap();
                        }
                        gpu.synchronize(stream).unwrap();
                    } else if step_idx == 4 {
                        // After full 4-step Scaled-FP16 KEEP_LATE lifecycle + cold unpack,
                        // verify full dequantized FP32 H parity.
                        for seq in 0..c {
                            let hb = download(h_b[seq], h_elems * 4);
                            let hs = download(h_s[seq], h_elems * 4);
                            let mut max_h_diff = 0.0f32;
                            for (a, b) in hs.chunks_exact(4).zip(hb.chunks_exact(4)) {
                                let fa = f32::from_le_bytes([a[0], a[1], a[2], a[3]]);
                                let fb = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                                max_h_diff = max_h_diff.max((fa - fb).abs());
                            }
                            eprintln!(
                                "scaled_fp16_splitv4 C={c} K={k} seq={seq} final_unpacked_h_max_diff={max_h_diff:.3e}"
                            );
                            // 1 FP16 ULP at head_amax (~0.25) is `32 * (amax / 65000) ~ 1.2e-4`
                            // when a ~1e-7 K-reassociation diff crosses an FP16 rounding midpoint.
                            assert!(
                                max_h_diff <= 5e-4,
                                "C={c} K={k} seq={seq} unpacked FP32 H max_diff={max_h_diff}"
                            );
                        }
                    } else if step_idx == 5 {
                        // Identical packed Scaled-FP16 input -> cold unpack MUST be 0-ULP identical!
                        for seq in 0..c {
                            assert_eq!(
                                download(h_s[seq], h_elems * 4),
                                download(h_b[seq], h_elems * 4),
                                "C={c} K={k} seq={seq} cold ScaledFp16->FP32 unpack must be 0-ULP bit-identical"
                            );
                        }
                    }
                }

                // 0-ULP bit-identical FP32 -> Scaled-FP16 pack check vs non-split:
                // feed pending=1 with gate=1.0, beta=0.0 so H_reg == H_init bit-for-bit,
                // then flush with STORE_SCALED_FP16 -> packed 32,772 B/head MUST match 0-ULP!
                {
                    let mut ring_no_op = vec![0u8; ring_bytes];
                    let gates_off = conv_dim * 2;
                    for h in 0..nv {
                        let goff = gates_off + h * 4;
                        ring_no_op[goff..goff + 4].copy_from_slice(&1.0f32.to_le_bytes());
                        // beta = 0.0f32 (already zeroed)
                    }
                    let mut db = vec![0u8; c * ops::GdnFusedMultiSeqDesc::SIZE_BYTES];
                    let mut ds = vec![0u8; c * ops::GdnFusedMultiSeqDesc::SIZE_BYTES];
                    for seq in 0..c {
                        // h_b[seq] and h_s[seq] are already bit-identical FP32 after step 5!
                        gpu.copy_h2d(&ring_no_op, ring_b[seq]).unwrap();
                        gpu.copy_h2d(&ring_no_op, ring_s[seq]).unwrap();
                        let cb_struct = ops::GdnKeepLateCtrl {
                            ring_base: ring_b[seq].0,
                            pending_base: 0,
                            pending_rows: 1,
                            write_base: 0,
                            executed_flag: 0,
                            ring_cap: ring_cap as u32,
                            flush_now: 1,
                        };
                        let cs_struct = ops::GdnKeepLateCtrl {
                            ring_base: ring_s[seq].0,
                            ..cb_struct
                        };
                        let mut cbuf = [0u8; ops::GdnKeepLateCtrl::SIZE_BYTES];
                        cb_struct.write_le_bytes(&mut cbuf);
                        gpu.copy_h2d(&cbuf, ctrl_b[seq]).unwrap();
                        cs_struct.write_le_bytes(&mut cbuf);
                        gpu.copy_h2d(&cbuf, ctrl_s[seq]).unwrap();
                        let desc_b = ops::GdnFusedMultiSeqDesc {
                            conv_state: conv_b[seq].0,
                            h_state: h_b[seq].0,
                            retain_qkv: ring_b[seq].0,
                            conv_inter: 0,
                            h_state_inter: ctrl_b[seq].0,
                            group_done: gdone_b[seq].0,
                            row_base: 0,
                            num_tokens: 0,
                            flags: base_kl | ops::GDN_FUSED_FLAG_STORE_SCALED_FP16,
                            _pad: 0,
                        };
                        let desc_s = ops::GdnFusedMultiSeqDesc {
                            conv_state: conv_s[seq].0,
                            h_state: h_s[seq].0,
                            retain_qkv: ring_s[seq].0,
                            conv_inter: 0,
                            h_state_inter: ctrl_s[seq].0,
                            group_done: gdone_s[seq].0,
                            ..desc_b
                        };
                        let off = seq * ops::GdnFusedMultiSeqDesc::SIZE_BYTES;
                        desc_b.write_le_bytes(
                            &mut db[off..off + ops::GdnFusedMultiSeqDesc::SIZE_BYTES],
                        );
                        desc_s.write_le_bytes(
                            &mut ds[off..off + ops::GdnFusedMultiSeqDesc::SIZE_BYTES],
                        );
                    }
                    gpu.copy_h2d(&db, desc_b_dev).unwrap();
                    gpu.copy_h2d(&ds, desc_s_dev).unwrap();
                    let dummy = gpu.alloc(64).unwrap();
                    ops::qwen4_gdn_verify_fused_lazyfinal_multi_seq(
                        &gpu,
                        base_multi_k,
                        desc_b_dev,
                        dummy,
                        &conv_w,
                        dummy,
                        dummy,
                        &norm_w,
                        dummy,
                        c as u32,
                        nk as u32,
                        nv as u32,
                        eps,
                        eps,
                        qkvz_size as u32,
                        (nv * 2) as u32,
                        value_dim as u32,
                        qkvz_size as u32,
                        0,
                        0,
                        stream,
                    )
                    .unwrap();
                    launch_gdn_verify_fused_lazyfinal_multi_seq_scaled_fp16_splitv4(
                        &gpu,
                        scaled_sv4_k,
                        desc_s_dev,
                        dummy,
                        &conv_w,
                        dummy,
                        dummy,
                        &norm_w,
                        dummy,
                        c as u32,
                        nk as u32,
                        nv as u32,
                        eps,
                        eps,
                        qkvz_size as u32,
                        (nv * 2) as u32,
                        value_dim as u32,
                        qkvz_size as u32,
                        0,
                        0,
                        scratch_gdn,
                        scratch_ss,
                        split_done,
                        scratch_h_flush,
                        scratch_amax,
                        stream,
                    )
                    .unwrap();
                    let head_stride = dim * dim * 4;
                    for seq in 0..c {
                        let hb = download(h_b[seq], h_elems * 4);
                        let hs = download(h_s[seq], h_elems * 4);
                        for vh in 0..nv {
                            let base = vh * head_stride;
                            assert_eq!(
                                &hs[base..base + 32_772],
                                &hb[base..base + 32_772],
                                "C={c} K={k} seq={seq} vh={vh} FP32->ScaledFp16 pack must be 0-ULP bit-identical"
                            );
                        }
                    }
                }
            }
        }
    }
}
