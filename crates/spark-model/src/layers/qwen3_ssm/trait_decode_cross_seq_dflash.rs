// SPDX-License-Identifier: AGPL-3.0-only

//! Cross-sequence hoisted SSM/GDN forward for batched flat DFlash verify.
//!
//! Runs input RMS norm, `in_proj_qkvz`, `in_proj_ba` + GDN gates, `out_proj`,
//! post-attn residual RMS norm, and dense FFN ONCE across all
//! `total_rows = num_seqs * k` rows (reading every SSM weight once per layer),
//! while advancing each sequence's recurrent conv1d + GDN state in its own
//! `gdn_lazy_retain` lane (`slot_ids[s]`).
//!
//! For `num_seqs >= 4` on the lazy-commit route (when multi-seq conv1d + GDN
//! kernels are loaded), the recurrent core runs in 1 conv1d launch
//! (`grid=[ceil(conv_dim/256), C]`) + 1 GDN launch (`grid=[nv, C]`, with
//! fused `gate_ptrs[s] | 1` gate retention) reading from a pre-staged
//! fixed-address device pointer table so the full layer loop is CUDA-graph
//! capturable. For `num_seqs < 4` (e.g. `C=2`, where Split-V=4 gives 192 CTAs
//! per sequence vs 96 CTAs), each sequence runs `conv1d_update_l2norm_f32_sequence`
//! + `gdn_f32_sequence_lazyfinal_splitv4` (or the persistent/snapshot fallback).

use std::sync::OnceLock;

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::*;

const BF16: usize = 2;
const FP32: usize = 4;

/// Number of sequences accommodated in one cross-sequence DFlash SSM pointer
/// table slice (matches the process verify-lane ceiling).
pub(crate) const DFLASH_SSM_PTR_TABLE_MAX_SEQS: usize =
    crate::layers::qwen4_verify_lanes::MAX_LANES;

/// Six parallel `u64` device pointer tables per SSM layer:
/// `[0] h_ptrs, [1] rows_ptrs, [2] gate_ptrs(|1), [3] h_inter_ptrs,
///  [4] conv_ptrs, [5] conv_inter_ptrs`.
pub(crate) const DFLASH_SSM_PTR_TABLE_COUNT: usize = 6;

/// Fixed per-SSM-layer byte stride inside the pre-staged pointer-table slab.
pub(crate) const DFLASH_SSM_PTR_TABLE_BYTES: usize =
    DFLASH_SSM_PTR_TABLE_COUNT * DFLASH_SSM_PTR_TABLE_MAX_SEQS * std::mem::size_of::<u64>();

/// Minimum batch concurrency that dispatches the 1-launch multi-seq conv1d +
/// GDN lazyfinal kernels (`48 * 4 = 192` CTAs at `C=4`, `384` CTAs at `C=8`).
/// Below this (`C=2`, `48 * 2 = 96 < 188` SMs), per-sequence Split-V=4
/// (`48 * 4 = 192` CTAs per seq) delivers 2x better SM occupancy.
const MULTI_SEQ_RECURRENT_MIN_SEQS: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SmEnvOverride {
    Disabled,
    Enabled,
    Auto,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CrossSeqBaRoute {
    CublasBf16AndGates,
    PrefillFusedBaGates,
    ExactBatchnBaGates,
    SerialPerRow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CrossSeqNvfp4ProjRoute {
    TcM32N64SplitK,
    TcM32N64,
    TcM128,
    TcN128,
    ExactRows,
    LegacyGemm,
    SerialGemv,
}

fn parse_sm_env_override(raw: Option<&str>) -> SmEnvOverride {
    match raw.map(str::trim) {
        Some("0") | Some("false") | Some("FALSE") | Some("off") | Some("OFF") => {
            SmEnvOverride::Disabled
        }
        Some("1") | Some("true") | Some("TRUE") | Some("on") | Some("ON") => SmEnvOverride::Enabled,
        _ => SmEnvOverride::Auto,
    }
}

fn raw_ssm_proj_tc_override() -> SmEnvOverride {
    static RAW: OnceLock<SmEnvOverride> = OnceLock::new();
    *RAW.get_or_init(|| parse_sm_env_override(std::env::var("ATLAS_SSM_PROJ_TC").ok().as_deref()))
}

fn raw_cublas_gemm_override() -> SmEnvOverride {
    static RAW: OnceLock<SmEnvOverride> = OnceLock::new();
    *RAW.get_or_init(|| parse_sm_env_override(std::env::var("ATLAS_CUBLAS_GEMM").ok().as_deref()))
}

const fn resolve_ssm_proj_tc_for_sm_with_policy(
    env_override: SmEnvOverride,
    sm_count: u32,
    rows: u32,
    det_batch: crate::layers::deterministic_batch::DeterministicBatchPolicy,
) -> bool {
    if det_batch.is_enabled() {
        return false;
    }
    resolve_ssm_proj_tc_for_sm(env_override, sm_count, rows)
}

const fn resolve_ssm_proj_tc_for_sm(env_override: SmEnvOverride, sm_count: u32, rows: u32) -> bool {
    match env_override {
        SmEnvOverride::Disabled => false,
        SmEnvOverride::Enabled => true,
        SmEnvOverride::Auto => sm_count > 64 && rows >= 16,
    }
}

#[allow(clippy::too_many_arguments)]
const fn select_cross_seq_ba_route_with_policy(
    total_rows: usize,
    sm_count: u32,
    cuda_available: bool,
    cublas_override: SmEnvOverride,
    has_compute_gdn_gates_multi_seq: bool,
    has_ba_gates_prefill: bool,
    has_ba_gates_batchn_exact: bool,
    det_batch: crate::layers::deterministic_batch::DeterministicBatchPolicy,
) -> CrossSeqBaRoute {
    if det_batch.is_enabled() {
        return if has_ba_gates_batchn_exact {
            CrossSeqBaRoute::ExactBatchnBaGates
        } else {
            CrossSeqBaRoute::SerialPerRow
        };
    }
    select_cross_seq_ba_route(
        total_rows,
        sm_count,
        cuda_available,
        cublas_override,
        has_compute_gdn_gates_multi_seq,
        has_ba_gates_prefill,
        has_ba_gates_batchn_exact,
    )
}

#[allow(clippy::too_many_arguments)]
const fn select_cross_seq_ba_route(
    total_rows: usize,
    sm_count: u32,
    cuda_available: bool,
    cublas_override: SmEnvOverride,
    has_compute_gdn_gates_multi_seq: bool,
    has_ba_gates_prefill: bool,
    has_ba_gates_batchn_exact: bool,
) -> CrossSeqBaRoute {
    let allow_cublas = !matches!(cublas_override, SmEnvOverride::Disabled);
    if cuda_available
        && allow_cublas
        && sm_count > 64
        && total_rows >= 32
        && has_compute_gdn_gates_multi_seq
    {
        return CrossSeqBaRoute::CublasBf16AndGates;
    }
    if sm_count > 64 && has_ba_gates_prefill {
        return CrossSeqBaRoute::PrefillFusedBaGates;
    }
    if has_ba_gates_batchn_exact {
        return CrossSeqBaRoute::ExactBatchnBaGates;
    }
    if has_ba_gates_prefill {
        return CrossSeqBaRoute::PrefillFusedBaGates;
    }
    CrossSeqBaRoute::SerialPerRow
}

#[allow(clippy::too_many_arguments)]
const fn select_cross_seq_nvfp4_fallback_route_with_policy(
    rows: u32,
    n: u32,
    sm_count: u32,
    tc_enabled: bool,
    has_transposed: bool,
    splitk_splits: u32,
    has_m32_n64_splitk: bool,
    has_m32_n64: bool,
    has_m128: bool,
    has_t_n128: bool,
    has_legacy_gemm: bool,
    det_batch: crate::layers::deterministic_batch::DeterministicBatchPolicy,
) -> CrossSeqNvfp4ProjRoute {
    if det_batch.is_enabled() {
        return if rows >= 2 && rows <= crate::layers::QWEN4_NATIVE_MAX_ROWS as u32 {
            CrossSeqNvfp4ProjRoute::ExactRows
        } else {
            CrossSeqNvfp4ProjRoute::SerialGemv
        };
    }
    select_cross_seq_nvfp4_fallback_route(
        rows,
        n,
        sm_count,
        tc_enabled,
        has_transposed,
        splitk_splits,
        has_m32_n64_splitk,
        has_m32_n64,
        has_m128,
        has_t_n128,
        has_legacy_gemm,
    )
}

#[allow(clippy::too_many_arguments)]
const fn select_cross_seq_nvfp4_fallback_route(
    rows: u32,
    n: u32,
    sm_count: u32,
    tc_enabled: bool,
    has_transposed: bool,
    splitk_splits: u32,
    has_m32_n64_splitk: bool,
    has_m32_n64: bool,
    has_m128: bool,
    has_t_n128: bool,
    has_legacy_gemm: bool,
) -> CrossSeqNvfp4ProjRoute {
    if sm_count > 64
        && tc_enabled
        && has_transposed
        && rows > crate::layers::TC_VERIFY_MAX_ROWS as u32
        && n <= 8192
        && has_m32_n64
    {
        return CrossSeqNvfp4ProjRoute::TcM32N64;
    }
    if tc_enabled && has_transposed {
        if rows <= crate::layers::TC_VERIFY_MAX_ROWS as u32 {
            if splitk_splits > 0 && has_m32_n64_splitk {
                return CrossSeqNvfp4ProjRoute::TcM32N64SplitK;
            }
            if has_m32_n64 {
                return CrossSeqNvfp4ProjRoute::TcM32N64;
            }
        }
        if has_m128 {
            return CrossSeqNvfp4ProjRoute::TcM128;
        }
        if has_t_n128 {
            return CrossSeqNvfp4ProjRoute::TcN128;
        }
    }
    if rows >= 2 && rows <= crate::layers::QWEN4_NATIVE_MAX_ROWS as u32 {
        return CrossSeqNvfp4ProjRoute::ExactRows;
    }
    if has_legacy_gemm {
        CrossSeqNvfp4ProjRoute::LegacyGemm
    } else {
        CrossSeqNvfp4ProjRoute::SerialGemv
    }
}

const fn should_use_multi_seq_recurrent_with_policy(
    num_seqs: usize,
    k: usize,
    gdn_seq_lazy: bool,
    has_conv_multi: bool,
    has_gdn_lazyfinal_multi: bool,
    det_batch: crate::layers::deterministic_batch::DeterministicBatchPolicy,
) -> bool {
    if det_batch.is_enabled() {
        return false;
    }
    should_use_multi_seq_recurrent(
        num_seqs,
        k,
        gdn_seq_lazy,
        has_conv_multi,
        has_gdn_lazyfinal_multi,
    )
}

const fn should_use_multi_seq_recurrent(
    num_seqs: usize,
    k: usize,
    gdn_seq_lazy: bool,
    has_conv_multi: bool,
    has_gdn_lazyfinal_multi: bool,
) -> bool {
    num_seqs >= MULTI_SEQ_RECURRENT_MIN_SEQS
        && num_seqs <= DFLASH_SSM_PTR_TABLE_MAX_SEQS
        && k <= GDN_LAZY_MAX_K
        && gdn_seq_lazy
        && has_conv_multi
        && has_gdn_lazyfinal_multi
}

fn slot_ids_are_distinct(slot_ids: &[usize]) -> bool {
    let mut seen = [false; DFLASH_SSM_PTR_TABLE_MAX_SEQS];
    for &slot in slot_ids {
        if slot >= DFLASH_SSM_PTR_TABLE_MAX_SEQS || seen[slot] {
            return false;
        }
        seen[slot] = true;
    }
    true
}

impl Qwen3SsmLayer {
    #[allow(clippy::too_many_arguments)]
    fn launch_dflash_cublas_bf16_proj(
        act: DevicePtr,
        weight: &DenseWeight,
        out: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        graph_capture: bool,
        stream: u64,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            if graph_capture {
                return spark_runtime::cublaslt::bf16_gemm_act_weight_t(
                    act.0,
                    weight.weight.0,
                    out.0,
                    m,
                    n,
                    k,
                    stream,
                );
            }
            ops::cublas_bf16_proj_dense(act, weight, out, m, n, k, stream)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (act, weight, out, m, n, k, graph_capture, stream);
            anyhow::bail!("cuBLASLt BF16 projection requires the cuda feature")
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn project_dflash_cross_seq_nvfp4(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        weight: &QuantizedWeight,
        weight_t: Option<&QuantizedWeight>,
        splitk_splits: u32,
        output: DevicePtr,
        rows: u32,
        n: u32,
        k: u32,
        graph_capture: bool,
        label: &'static str,
        stream: u64,
    ) -> Result<()> {
        let det_batch = crate::layers::deterministic_batch::deterministic_batch_policy()?;
        let sm_count = atlas_core::device::active_sm_count();
        let tc_override = raw_ssm_proj_tc_override();
        let tc_enabled =
            resolve_ssm_proj_tc_for_sm_with_policy(tc_override, sm_count, rows, det_batch);

        // SM120/SM103 (`sm_count > 64`) small/medium-M coalesced BF16xBF16 MMA
        // warp-split-K (`M <= 16`, or `M <= 64` when wide coalesced is enabled).
        // Zero extra VRAM, zero global split-K workspace, CUDA-graph safe.
        let sm120_max_m = crate::layers::dense_ffn::sm120_coalesced_max_m();
        if !det_batch.is_enabled()
            && rows >= 4
            && (rows as usize) <= sm120_max_m
            && tc_override != SmEnvOverride::Disabled
            && crate::layers::dense_ffn::try_coalesced_nvfp4_sm120_small_m(
                gpu,
                input,
                weight,
                output,
                rows,
                n,
                k,
                sm120_max_m,
                stream,
            )?
        {
            return Ok(());
        }

        // SM120/SM103 (`sm_count > 64`) native block-scaled cuBLASLt NVFP4 for
        // `M >= 24` (graph-capture aware via cached weight scales/workspaces).
        if !det_batch.is_enabled()
            && rows >= crate::layers::dense_ffn::CUBLASLT_NVFP4_MIN_M
            && tc_override != SmEnvOverride::Disabled
            && ops::try_cublaslt_nvfp4_multiseq_single(
                gpu,
                input,
                weight,
                output,
                rows,
                n,
                k,
                n,
                graph_capture,
                label,
                stream,
            )?
        {
            return Ok(());
        }

        let ws = if splitk_splits > 0 {
            *self
                .ssm_splitk_workspace
                .lock()
                .map_err(|_| anyhow::anyhow!("ssm_splitk_workspace lock poisoned"))?
        } else {
            None
        };
        let route = select_cross_seq_nvfp4_fallback_route_with_policy(
            rows,
            n,
            sm_count,
            tc_enabled,
            weight_t.is_some(),
            splitk_splits,
            self.w4a16_gemm_t_m32_n64_splitk_k.0 != 0
                && self.reduce_splitk_k.0 != 0
                && ws.is_some(),
            self.w4a16_gemm_t_m32_n64_k.0 != 0,
            self.w4a16_gemm_t_m128_k.0 != 0,
            self.w4a16_gemm_t_k.0 != 0,
            self.w4a16_gemm_k.0 != 0,
            det_batch,
        );
        match route {
            CrossSeqNvfp4ProjRoute::TcM32N64SplitK => {
                let wt = weight_t.expect("TcM32N64SplitK requires transposed weight");
                let ws_ptr = ws.expect("TcM32N64SplitK requires split-K workspace");
                ops::w4a16_gemm_n64_m32_splitk(
                    gpu,
                    self.w4a16_gemm_t_m32_n64_splitk_k,
                    self.reduce_splitk_k,
                    input,
                    wt,
                    output,
                    ws_ptr,
                    rows,
                    n,
                    k,
                    n,
                    splitk_splits,
                    stream,
                )
            }
            CrossSeqNvfp4ProjRoute::TcM32N64 => {
                let wt = weight_t.expect("TcM32N64 requires transposed weight");
                ops::w4a16_gemm_n64_m32(
                    gpu,
                    self.w4a16_gemm_t_m32_n64_k,
                    input,
                    wt,
                    output,
                    rows,
                    n,
                    k,
                    stream,
                )
            }
            CrossSeqNvfp4ProjRoute::TcM128 => {
                let wt = weight_t.expect("TcM128 requires transposed weight");
                ops::w4a16_gemm_n128_m128(
                    gpu,
                    self.w4a16_gemm_t_m128_k,
                    input,
                    wt,
                    output,
                    rows,
                    n,
                    k,
                    stream,
                )
            }
            CrossSeqNvfp4ProjRoute::TcN128 => {
                let wt = weight_t.expect("TcN128 requires transposed weight");
                ops::w4a16_gemm_n128(
                    gpu,
                    self.w4a16_gemm_t_k,
                    input,
                    wt,
                    output,
                    rows,
                    n,
                    k,
                    stream,
                )
            }
            CrossSeqNvfp4ProjRoute::ExactRows => {
                let _ = self.project_nvfp4_rows_exact_or_k1(
                    gpu, input, weight, output, rows, n, k, stream,
                )?;
                Ok(())
            }
            CrossSeqNvfp4ProjRoute::LegacyGemm => ops::w4a16_gemm(
                gpu,
                self.w4a16_gemm_k,
                input,
                weight,
                output,
                rows,
                n,
                k,
                stream,
            ),
            CrossSeqNvfp4ProjRoute::SerialGemv => {
                for row in 0..rows as usize {
                    ops::w4a16_decode_gemv(
                        gpu,
                        self.w4a16_gemv_k,
                        self.w4a16_gemv_sw_k,
                        self.gemv_sw,
                        input.offset(row * k as usize * BF16),
                        weight,
                        output.offset(row * n as usize * BF16),
                        n,
                        k,
                        stream,
                    )?;
                }
                Ok(())
            }
        }
    }

    fn dflash_cross_seq_geometry_eligible(
        &self,
        num_seqs: usize,
        k: usize,
        slot_ids: &[usize],
        ctx: &ForwardContext,
    ) -> bool {
        if !(2..=DFLASH_SSM_PTR_TABLE_MAX_SEQS).contains(&num_seqs)
            || !(2..=GDN_LAZY_MAX_K).contains(&k)
            || slot_ids.len() != num_seqs
        {
            return false;
        }
        let Some(total_rows) = num_seqs.checked_mul(k) else {
            return false;
        };
        if total_rows > ctx.buffers.max_batch_tokens() {
            return false;
        }
        if ctx.config.use_fp32_residual()
            || self.qwen4_attn_hyper.is_some()
            || self.qwen4_mlp_hyper.is_some()
            || ctx.ddtree_parent_ids_dev.is_some()
            || super::ssm_h_fp16_enabled()
        {
            return false;
        }
        let serial = crate::model::env_diag::DflashSerialControls::current();
        if serial.layer_norms
            || serial.ffn
            || crate::layers::ssm_qkvz_serial_enabled()
            || crate::layers::ssm_out_serial_enabled()
            || crate::layers::ssm_recurrent_serial_enabled()
            || crate::layers::ssm_ba_serial_enabled()
        {
            return false;
        }
        if self.conv1d_l2norm_f32_sequence_k.0 == 0
            || self.gdn_f32_sequence_k.0 == 0
            || self.gated_rms_norm_f32_multi_seq_k.0 == 0
            || self.qkvz_nvfp4.is_none()
            || self.ssm.out_proj.is_null()
            || self.out_proj_fp8w.is_some()
            || self.qkvz_fp8w.is_some()
        {
            return false;
        }
        if !self.sequential_qkvz && self.deinterleave_k.0 == 0 {
            return false;
        }
        let nk = ctx.config.linear_num_key_heads;
        let kd = ctx.config.linear_key_head_dim;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        if nk == 0 || nv == 0 || !nv.is_multiple_of(nk) {
            return false;
        }
        let key_dim = nk * kd;
        let value_dim = nv * vd;
        let conv_dim = key_dim * 2 + value_dim;
        let qkvz_size = ctx.config.ssm_qkvz_size();
        let ba_size = ctx.config.ssm_ba_size();
        let h = ctx.config.hidden_size;
        if qkvz_size != conv_dim + value_dim {
            return false;
        }
        let sizes = ctx.buffers.sizes();
        if sizes.norm_output < total_rows * h * BF16
            || sizes.ssm_deinterleaved < total_rows * qkvz_size * BF16
            || sizes.ssm_qkvz < total_rows * qkvz_size * BF16
            || sizes.ssm_ba < total_rows * ba_size * BF16
            || sizes.ssm_gates < total_rows * nv * 2 * FP32
            || sizes.ssm_conv_out_f32 < total_rows * qkvz_size * FP32
            || sizes.moe_output < total_rows * h * BF16
        {
            return false;
        }
        let gdn_nosnap = std::env::var("ATLAS_SSM_GDN_NOSNAP").ok().as_deref() == Some("1")
            && self.gdn_f32_sequence_nosnap_k.0 != 0;
        let gdn_seq_lazy = !gdn_nosnap && self.gdn_seq_lazy_engaged_inner(k);
        if gdn_seq_lazy {
            if !slot_ids_are_distinct(slot_ids) {
                return false;
            }
            let configured_lanes = crate::layers::qwen4_verify_lanes::lanes();
            if configured_lanes < num_seqs || slot_ids.iter().any(|&slot| slot >= configured_lanes)
            {
                return false;
            }
        }
        true
    }

    fn resolve_h_inter_base_and_stride(
        &self,
        ssm: &SsmLayerState,
        k: usize,
        gdn_seq_lazy: bool,
    ) -> Result<(DevicePtr, u32)> {
        if gdn_seq_lazy && ssm.h_state_checkpoint.is_none() {
            ensure!(
                !ssm.h_state_intermediates.is_empty(),
                "cross-seq DFlash SSM missing compact lazy h_state_intermediates"
            );
            Ok((ssm.h_state_intermediates[0], 0u32))
        } else {
            let base = contiguous_intermediate_base(
                &ssm.h_state_intermediates,
                k,
                self.h_state_bytes,
                "H",
            )?;
            Ok((base, (self.h_state_bytes / FP32) as u32))
        }
    }

    /// Pre-allocate retain lanes and pack this layer's 6-table cross-sequence
    /// SSM pointer table into `dst` (`DFLASH_SSM_PTR_TABLE_BYTES` bytes) before
    /// layer 0 and before CUDA graph capture/replay.
    pub(super) fn stage_dflash_cross_seq_ssm_ptr_table_inner(
        &self,
        num_seqs: usize,
        k: usize,
        slot_ids: &[usize],
        states: &[&(dyn LayerState + 'static)],
        ctx: &ForwardContext,
        dst: &mut [u8],
    ) -> Result<bool> {
        if states.len() != num_seqs
            || !self.dflash_cross_seq_geometry_eligible(num_seqs, k, slot_ids, ctx)
        {
            return Ok(false);
        }
        let gdn_nosnap = std::env::var("ATLAS_SSM_GDN_NOSNAP").ok().as_deref() == Some("1")
            && self.gdn_f32_sequence_nosnap_k.0 != 0;
        let gdn_seq_lazy = !gdn_nosnap && self.gdn_seq_lazy_engaged_inner(k);
        if !should_use_multi_seq_recurrent_with_policy(
            num_seqs,
            k,
            gdn_seq_lazy,
            self.conv1d_l2norm_f32_sequence_multi_k.0 != 0,
            self.gdn_f32_sequence_lazyfinal_multi_k.0 != 0,
            crate::layers::deterministic_batch::deterministic_batch_policy_or_disabled(),
        ) {
            // Still ensure retain lanes exist outside capture for C < 4!
            if gdn_seq_lazy {
                let qkvz_size = ctx.config.ssm_qkvz_size();
                let nv = ctx.config.linear_num_value_heads;
                let _ = self.gdn_lazy_retain_pool_with_layers(
                    ctx.gpu,
                    qkvz_size,
                    nv,
                    Some(super::count_ssm_layers(ctx.config)),
                )?;
            }
            return Ok(false);
        }
        ensure!(
            dst.len() >= DFLASH_SSM_PTR_TABLE_BYTES,
            "DFlash SSM pointer-table staging slice too small: {} < {}",
            dst.len(),
            DFLASH_SSM_PTR_TABLE_BYTES
        );
        let qkvz_size = ctx.config.ssm_qkvz_size();
        let nv = ctx.config.linear_num_value_heads;
        let gates_base = ctx.buffers.ssm_gates();
        let gate_stride_bytes = nv * 2 * FP32;

        let _ = self.gdn_lazy_retain_pool_with_layers(
            ctx.gpu,
            qkvz_size,
            nv,
            Some(super::count_ssm_layers(ctx.config)),
        )?;

        let mut table = [[0u64; DFLASH_SSM_PTR_TABLE_MAX_SEQS]; DFLASH_SSM_PTR_TABLE_COUNT];
        let mut expected_h_inter_stride: Option<u32> = None;
        for (s, &state_dyn) in states.iter().enumerate() {
            let Some(ssm) = state_dyn.as_any().downcast_ref::<SsmLayerState>() else {
                return Ok(false);
            };
            if ssm.h_state_intermediates.len() < k || ssm.conv_state_intermediates.len() < k {
                return Ok(false);
            }
            let conv_inter = contiguous_intermediate_base(
                &ssm.conv_state_intermediates,
                k,
                self.conv_state_bytes,
                "conv",
            )?;
            let (h_inter, h_stride) = self.resolve_h_inter_base_and_stride(ssm, k, gdn_seq_lazy)?;
            if expected_h_inter_stride
                .replace(h_stride)
                .is_some_and(|prev| prev != h_stride)
            {
                return Ok(false);
            }
            let _lane = crate::layers::qwen4_verify_lanes::LaneGuard::enter_for_slot(slot_ids[s])?;
            let retain_s = self.gdn_lazy_retain_current(ctx.gpu, qkvz_size, nv)?;
            let gates_s = gates_base.offset(s * k * gate_stride_bytes);
            ensure!(
                (gates_s.0 & 1) == 0,
                "ssm_gates pointer {:#x} must be even for fused gate-retain flag",
                gates_s.0
            );
            table[0][s] = ssm.h_state.0;
            table[1][s] = retain_s.0;
            table[2][s] = gates_s.0 | 1; // Bit 0 enables fused raw_g/beta writeback into retain.
            table[3][s] = h_inter.0;
            table[4][s] = ssm.conv_state.0;
            table[5][s] = conv_inter.0;
        }

        for (idx, &val) in table.iter().flatten().enumerate() {
            dst[idx * 8..idx * 8 + 8].copy_from_slice(&val.to_le_bytes());
        }
        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_ssm_batched_cross_seq_inner<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        k: usize,
        slot_ids: &[usize],
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        if states.len() != num_seqs
            || !self.dflash_cross_seq_geometry_eligible(num_seqs, k, slot_ids, ctx)
        {
            return Ok(false);
        }

        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let nk = ctx.config.linear_num_key_heads;
        let kd = ctx.config.linear_key_head_dim;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let vpg = nv / nk;
        let key_dim = nk * kd;
        let value_dim = nv * vd;
        let conv_dim = key_dim * 2 + value_dim;
        let qk_ch = (key_dim * 2) as u32;
        let d_conv = ctx.config.linear_conv_kernel_dim;
        let qkvz_size = ctx.config.ssm_qkvz_size();
        let ba_size = ctx.config.ssm_ba_size();
        let total_rows = num_seqs * k;

        let gdn_nosnap = std::env::var("ATLAS_SSM_GDN_NOSNAP").ok().as_deref() == Some("1")
            && self.gdn_f32_sequence_nosnap_k.0 != 0;
        let gdn_seq_lazy = !gdn_nosnap && self.gdn_seq_lazy_engaged_inner(k);

        // Validate every sequence's SSM intermediates up front before launching
        // any kernel so ineligible states fail closed to the per-seq fallback.
        let mut h_states = Vec::with_capacity(num_seqs);
        let mut conv_states = Vec::with_capacity(num_seqs);
        let mut h_inters = Vec::with_capacity(num_seqs);
        let mut h_inter_strides = Vec::with_capacity(num_seqs);
        let mut conv_inters = Vec::with_capacity(num_seqs);
        for state_dyn in states.iter_mut() {
            let Some(ssm) = state_dyn.as_any_mut().downcast_mut::<SsmLayerState>() else {
                return Ok(false);
            };
            if ssm.h_state_intermediates.len() < k || ssm.conv_state_intermediates.len() < k {
                return Ok(false);
            }
            let conv_inter = match contiguous_intermediate_base(
                &ssm.conv_state_intermediates,
                k,
                self.conv_state_bytes,
                "conv",
            ) {
                Ok(ptr) => ptr,
                Err(_) => return Ok(false),
            };
            let (h_inter, h_stride) =
                match self.resolve_h_inter_base_and_stride(ssm, k, gdn_seq_lazy) {
                    Ok(pair) => pair,
                    Err(_) => return Ok(false),
                };
            h_states.push(ssm.h_state);
            conv_states.push(ssm.conv_state);
            h_inters.push(h_inter);
            h_inter_strides.push(h_stride);
            conv_inters.push(conv_inter);
        }

        let Some(qkvz_nvfp4) = self.qkvz_nvfp4.as_ref() else {
            return Ok(false);
        };

        // ── Step 1: 1-launch input RMS norm + residual save across total_rows ──
        let normed = ctx.buffers.norm_output();
        ops::rms_norm_residual(
            ctx.gpu,
            self.rms_norm_residual_k,
            hidden,
            &self.input_norm,
            normed,
            residual,
            total_rows as u32,
            h as u32,
            eps,
            stream,
        )?;

        // ── Step 2: Hoisted in_proj_qkvz [total_rows, H] -> ssm_deinterleaved ──
        let deinterleaved = ctx.buffers.ssm_deinterleaved();
        let proj_dst = if self.sequential_qkvz {
            deinterleaved
        } else {
            ctx.buffers.ssm_qkvz()
        };
        self.project_dflash_cross_seq_nvfp4(
            ctx.gpu,
            normed,
            qkvz_nvfp4,
            self.qkvz_nvfp4_t.as_ref(),
            crate::layers::ssm_qkvz_splitk(),
            proj_dst,
            total_rows as u32,
            qkvz_size as u32,
            h as u32,
            ctx.graph_capture,
            "ssm_qkvz_dflash_batched",
            stream,
        )?;
        if !self.sequential_qkvz {
            for t in 0..total_rows {
                ops::deinterleave_qkvz(
                    ctx.gpu,
                    self.deinterleave_k,
                    proj_dst.offset(t * qkvz_size * BF16),
                    deinterleaved.offset(t * qkvz_size * BF16),
                    1,
                    nk as u32,
                    kd as u32,
                    vpg as u32,
                    vd as u32,
                    stream,
                )?;
            }
        }

        // ── Step 3: Hoisted in_proj_ba + compute_gdn_gates [total_rows, 2*nv] FP32 ──
        let gates_buf = ctx.buffers.ssm_gates();
        let gate_beta_stride = nv * 2 * FP32;
        let sm_count = atlas_core::device::active_sm_count();
        let det_batch = crate::layers::deterministic_batch::deterministic_batch_policy()?;
        match select_cross_seq_ba_route_with_policy(
            total_rows,
            sm_count,
            cfg!(feature = "cuda"),
            raw_cublas_gemm_override(),
            self.compute_gdn_gates_multi_seq_k.0 != 0,
            self.ba_gates_prefill_k.0 != 0,
            self.ba_gates_batchn_exact_k.0 != 0,
            det_batch,
        ) {
            CrossSeqBaRoute::CublasBf16AndGates => {
                let ba_buf = ctx.buffers.ssm_ba();
                Self::launch_dflash_cublas_bf16_proj(
                    normed,
                    &self.ssm.in_proj_ba,
                    ba_buf,
                    total_rows as u32,
                    ba_size as u32,
                    h as u32,
                    ctx.graph_capture,
                    stream,
                )?;
                ops::compute_gdn_gates_multi_seq(
                    ctx.gpu,
                    self.compute_gdn_gates_multi_seq_k,
                    ba_buf,
                    self.ssm.a_log.weight,
                    self.ssm.dt_bias.weight,
                    gates_buf,
                    gates_buf.offset(nv * FP32),
                    total_rows as u32,
                    nv as u32,
                    nk as u32,
                    vpg as u32,
                    ba_size as u32,
                    (nv * 2) as u32,
                    stream,
                )?;
            }
            CrossSeqBaRoute::PrefillFusedBaGates => {
                ops::dense_gemm_ba_gates_prefill(
                    ctx.gpu,
                    self.ba_gates_prefill_k,
                    normed,
                    &self.ssm.in_proj_ba,
                    self.ssm.a_log.weight,
                    self.ssm.dt_bias.weight,
                    gates_buf,
                    total_rows as u32,
                    ba_size as u32,
                    h as u32,
                    h as u32,
                    (nv * 2) as u32,
                    nv as u32,
                    vpg as u32,
                    stream,
                )?;
            }
            CrossSeqBaRoute::ExactBatchnBaGates => {
                ops::dense_gemv_ba_gates_batchn(
                    ctx.gpu,
                    self.ba_gates_batchn_exact_k,
                    normed,
                    &self.ssm.in_proj_ba,
                    self.ssm.a_log.weight,
                    self.ssm.dt_bias.weight,
                    gates_buf,
                    gates_buf.offset(nv * FP32),
                    total_rows as u32,
                    ba_size as u32,
                    h as u32,
                    vpg as u32,
                    (nv * 2) as u32,
                    stream,
                )?;
            }
            CrossSeqBaRoute::SerialPerRow => {
                for t in 0..total_rows {
                    let gate_t = gates_buf.offset(t * gate_beta_stride);
                    ops::dense_gemv_ba_gates(
                        ctx.gpu,
                        self.ba_gates_k,
                        normed.offset(t * h * BF16),
                        &self.ssm.in_proj_ba,
                        self.ssm.a_log.weight,
                        self.ssm.dt_bias.weight,
                        gate_t,
                        gate_t.offset(nv * FP32),
                        ba_size as u32,
                        h as u32,
                        vpg as u32,
                        stream,
                    )?;
                }
            }
        }

        // ── Step 4: Recurrent core (Conv1d + GDN + Gated RMSNorm -> ssm_qkvz) ──
        // `ssm_qkvz` is free once `sequential_qkvz` wrote directly to
        // `ssm_deinterleaved` (or deinterleave finished above). Pack all
        // `total_rows` gated-norm outputs contiguously at `[total_rows, value_dim]`.
        let normed_out = ctx.buffers.ssm_qkvz();
        let conv_arena = ctx.buffers.ssm_conv_out_f32();
        let uniform_h_stride = h_inter_strides.iter().all(|&s| s == h_inter_strides[0]);
        let use_multi_seq = uniform_h_stride
            && ctx.ssm_multi_seq_ptr_table_override.is_some()
            && should_use_multi_seq_recurrent_with_policy(
                num_seqs,
                k,
                gdn_seq_lazy,
                self.conv1d_l2norm_f32_sequence_multi_k.0 != 0,
                self.gdn_f32_sequence_lazyfinal_multi_k.0 != 0,
                det_batch,
            );

        if use_multi_seq {
            let table_dev = ctx
                .ssm_multi_seq_ptr_table_override
                .expect("use_multi_seq requires pre-staged pointer table");
            let table_ptr = |index: usize| {
                table_dev.offset(index * DFLASH_SSM_PTR_TABLE_MAX_SEQS * std::mem::size_of::<u64>())
            };

            // 1-launch multi-seq Conv1d (`grid=[ceil(conv_dim/256), C]`).
            ops::conv1d_update_l2norm_f32_sequence_multi(
                ctx.gpu,
                self.conv1d_l2norm_f32_sequence_multi_k,
                table_ptr(4),
                deinterleaved,
                &self.ssm.conv1d,
                table_ptr(1),
                table_ptr(5),
                num_seqs as u32,
                k as u32,
                conv_dim as u32,
                d_conv as u32,
                qk_ch,
                kd as u32,
                1e-6,
                qkvz_size as u32,
                qkvz_size as u32,
                (self.conv_state_bytes / FP32) as u32,
                stream,
            )?;

            // 1-launch multi-seq GDN lazyfinal (`grid=[nv, C]`, fused gate retain).
            ops::gdn_decode_f32_sequence_multi(
                ctx.gpu,
                self.gdn_f32_sequence_lazyfinal_multi_k,
                table_ptr(0),
                table_ptr(1),
                table_ptr(2),
                table_ptr(3),
                num_seqs as u32,
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
                h_inter_strides[0],
                stream,
            )?;

            // Per-sequence gated RMS norm reading `retain_lane(slot_s) + conv_dim*4`
            // and writing compactly into `ssm_qkvz + s * k * value_dim * 2`.
            for (s, &slot_idx) in slot_ids.iter().enumerate() {
                let _lane = crate::layers::qwen4_verify_lanes::LaneGuard::enter_for_slot(slot_idx)?;
                let retain_s = self.gdn_lazy_retain_current(ctx.gpu, qkvz_size, nv)?;
                let gdn_out_s = retain_s.offset(conv_dim * FP32);
                let z_s = deinterleaved.offset(s * k * qkvz_size * BF16 + conv_dim * BF16);
                let out_s = normed_out.offset(s * k * value_dim * BF16);
                ops::gated_rms_norm_f32_multi_seq(
                    ctx.gpu,
                    self.gated_rms_norm_f32_multi_seq_k,
                    gdn_out_s,
                    z_s,
                    &self.ssm.norm,
                    out_s,
                    nv as u32,
                    k as u32,
                    vd as u32,
                    eps,
                    qkvz_size as u32,
                    qkvz_size as u32,
                    value_dim as u32,
                    stream,
                )?;
            }
        } else {
            let persistent = !gdn_nosnap
                && !gdn_seq_lazy
                && self.gdn_f32_sequence_persistent_k.0 != 0
                && std::env::var("ATLAS_SSM_GDN_SEQ_PERSISTENT")
                    .ok()
                    .as_deref()
                    == Some("1");
            let split_v =
                super::qwen4_gdn_split_v_enabled_for_single_seq() && kd == 128 && vd == 128;

            for (s, &slot_idx) in slot_ids.iter().enumerate() {
                let _lane = crate::layers::qwen4_verify_lanes::LaneGuard::enter_for_slot(slot_idx)?;
                let row_base = s * k;
                let deint_s = deinterleaved.offset(row_base * qkvz_size * BF16);
                let gates_s = gates_buf.offset(row_base * gate_beta_stride);
                let conv_f32_s = if gdn_seq_lazy {
                    let retain_s = self.gdn_lazy_retain_current(ctx.gpu, qkvz_size, nv)?;
                    ctx.gpu.copy_d2d_async(
                        gates_s,
                        retain_s.offset(GDN_LAZY_MAX_K * qkvz_size * FP32),
                        k * gate_beta_stride,
                        stream,
                    )?;
                    retain_s
                } else {
                    conv_arena.offset(row_base * qkvz_size * FP32)
                };
                let gdn_f32_s = conv_f32_s.offset(conv_dim * FP32);

                ops::conv1d_update_l2norm_f32_sequence(
                    ctx.gpu,
                    self.conv1d_l2norm_f32_sequence_k,
                    conv_states[s],
                    deint_s,
                    &self.ssm.conv1d,
                    conv_f32_s,
                    conv_inters[s],
                    k as u32,
                    conv_dim as u32,
                    d_conv as u32,
                    qk_ch,
                    kd as u32,
                    1e-6,
                    qkvz_size as u32,
                    qkvz_size as u32,
                    (self.conv_state_bytes / FP32) as u32,
                    stream,
                )?;

                if gdn_nosnap {
                    ops::gdn_decode_f32_sequence(
                        ctx.gpu,
                        self.gdn_f32_sequence_nosnap_k,
                        h_states[s],
                        conv_f32_s,
                        conv_f32_s.offset(key_dim * FP32),
                        conv_f32_s.offset(key_dim * 2 * FP32),
                        gates_s,
                        gates_s.offset(nv * FP32),
                        gdn_f32_s,
                        h_inters[s],
                        k as u32,
                        nk as u32,
                        nv as u32,
                        kd as u32,
                        vd as u32,
                        qkvz_size as u32,
                        qkvz_size as u32,
                        (nv * 2) as u32,
                        qkvz_size as u32,
                        h_inter_strides[s],
                        stream,
                    )?;
                } else if gdn_seq_lazy {
                    if split_v && self.gdn_f32_sequence_lazyfinal_splitv4_k.0 != 0 {
                        ops::gdn_decode_f32_sequence_persistent_splitv4(
                            ctx.gpu,
                            self.gdn_f32_sequence_lazyfinal_splitv4_k,
                            h_states[s],
                            conv_f32_s,
                            conv_f32_s.offset(key_dim * FP32),
                            conv_f32_s.offset(key_dim * 2 * FP32),
                            gates_s,
                            gates_s.offset(nv * FP32),
                            gdn_f32_s,
                            h_inters[s],
                            k as u32,
                            nk as u32,
                            nv as u32,
                            kd as u32,
                            vd as u32,
                            qkvz_size as u32,
                            qkvz_size as u32,
                            (nv * 2) as u32,
                            qkvz_size as u32,
                            h_inter_strides[s],
                            stream,
                        )?;
                    } else {
                        ops::gdn_decode_f32_sequence_persistent(
                            ctx.gpu,
                            self.gdn_f32_sequence_lazyfinal_k,
                            h_states[s],
                            conv_f32_s,
                            conv_f32_s.offset(key_dim * FP32),
                            conv_f32_s.offset(key_dim * 2 * FP32),
                            gates_s,
                            gates_s.offset(nv * FP32),
                            gdn_f32_s,
                            h_inters[s],
                            k as u32,
                            nk as u32,
                            nv as u32,
                            kd as u32,
                            vd as u32,
                            qkvz_size as u32,
                            qkvz_size as u32,
                            (nv * 2) as u32,
                            qkvz_size as u32,
                            h_inter_strides[s],
                            stream,
                        )?;
                    }
                } else if persistent {
                    if split_v && self.gdn_f32_sequence_persistent_splitv4_k.0 != 0 {
                        ops::gdn_decode_f32_sequence_persistent_splitv4(
                            ctx.gpu,
                            self.gdn_f32_sequence_persistent_splitv4_k,
                            h_states[s],
                            conv_f32_s,
                            conv_f32_s.offset(key_dim * FP32),
                            conv_f32_s.offset(key_dim * 2 * FP32),
                            gates_s,
                            gates_s.offset(nv * FP32),
                            gdn_f32_s,
                            h_inters[s],
                            k as u32,
                            nk as u32,
                            nv as u32,
                            kd as u32,
                            vd as u32,
                            qkvz_size as u32,
                            qkvz_size as u32,
                            (nv * 2) as u32,
                            qkvz_size as u32,
                            h_inter_strides[s],
                            stream,
                        )?;
                    } else {
                        ops::gdn_decode_f32_sequence_persistent(
                            ctx.gpu,
                            self.gdn_f32_sequence_persistent_k,
                            h_states[s],
                            conv_f32_s,
                            conv_f32_s.offset(key_dim * FP32),
                            conv_f32_s.offset(key_dim * 2 * FP32),
                            gates_s,
                            gates_s.offset(nv * FP32),
                            gdn_f32_s,
                            h_inters[s],
                            k as u32,
                            nk as u32,
                            nv as u32,
                            kd as u32,
                            vd as u32,
                            qkvz_size as u32,
                            qkvz_size as u32,
                            (nv * 2) as u32,
                            qkvz_size as u32,
                            h_inter_strides[s],
                            stream,
                        )?;
                    }
                } else {
                    ops::gdn_decode_f32_sequence(
                        ctx.gpu,
                        self.gdn_f32_sequence_k,
                        h_states[s],
                        conv_f32_s,
                        conv_f32_s.offset(key_dim * FP32),
                        conv_f32_s.offset(key_dim * 2 * FP32),
                        gates_s,
                        gates_s.offset(nv * FP32),
                        gdn_f32_s,
                        h_inters[s],
                        k as u32,
                        nk as u32,
                        nv as u32,
                        kd as u32,
                        vd as u32,
                        qkvz_size as u32,
                        qkvz_size as u32,
                        (nv * 2) as u32,
                        qkvz_size as u32,
                        h_inter_strides[s],
                        stream,
                    )?;
                }

                ops::gated_rms_norm_f32_multi_seq(
                    ctx.gpu,
                    self.gated_rms_norm_f32_multi_seq_k,
                    gdn_f32_s,
                    deint_s.offset(conv_dim * BF16),
                    &self.ssm.norm,
                    normed_out.offset(row_base * value_dim * BF16),
                    nv as u32,
                    k as u32,
                    vd as u32,
                    eps,
                    qkvz_size as u32,
                    qkvz_size as u32,
                    value_dim as u32,
                    stream,
                )?;
            }
        }

        // ── Step 5: Hoisted out_proj ssm_qkvz[total_rows, 6144] -> moe_output[total_rows, H] ──
        let ssm_out = ctx.buffers.moe_output();
        self.project_dflash_cross_seq_nvfp4(
            ctx.gpu,
            normed_out,
            &self.ssm.out_proj,
            self.out_proj_nvfp4_t.as_ref(),
            crate::layers::ssm_out_splitk(),
            ssm_out,
            total_rows as u32,
            h as u32,
            value_dim as u32,
            ctx.graph_capture,
            "ssm_out_dflash_batched",
            stream,
        )?;
        if ctx.config.tp_world_size > 1
            && let Some(comm) = ctx.comm
        {
            comm.all_reduce_async(ssm_out.0, total_rows * h * BF16, stream)?;
        }

        // ── Step 6: 1-launch residual_add_rms_norm straight into norm_output (0 copies) ──
        ops::residual_add_rms_norm(
            ctx.gpu,
            self.residual_add_rms_norm_k,
            hidden,
            ssm_out,
            &self.post_attn_norm,
            normed,
            residual,
            total_rows as u32,
            h as u32,
            eps,
            stream,
        )?;

        // ── Step 7: Hoisted dense/MoE FFN straight from norm_output -> hidden ──
        self.run_deferred_ffn_inner(normed, hidden, total_rows, ctx, stream)?;

        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ba_route_selects_prefill_for_small_m_and_cublas_for_m32_on_wide_sm_only() {
        // SM120 (188 SMs): M=16/24 -> prefill fused BA+gates; M=32/64 -> cuBLASLt + multi-seq gates.
        assert_eq!(
            select_cross_seq_ba_route(16, 188, true, SmEnvOverride::Auto, true, true, true),
            CrossSeqBaRoute::PrefillFusedBaGates
        );
        assert_eq!(
            select_cross_seq_ba_route(24, 188, true, SmEnvOverride::Auto, true, true, true),
            CrossSeqBaRoute::PrefillFusedBaGates
        );
        assert_eq!(
            select_cross_seq_ba_route(32, 188, true, SmEnvOverride::Auto, true, true, true),
            CrossSeqBaRoute::CublasBf16AndGates
        );
        assert_eq!(
            select_cross_seq_ba_route(64, 188, true, SmEnvOverride::Auto, true, true, true),
            CrossSeqBaRoute::CublasBf16AndGates
        );
        // Explicit ATLAS_CUBLAS_GEMM=0 honors disable and stays on prefill fused BA+gates.
        assert_eq!(
            select_cross_seq_ba_route(64, 188, true, SmEnvOverride::Disabled, true, true, true),
            CrossSeqBaRoute::PrefillFusedBaGates
        );
        // GB10 (48 SMs <= 64): always keeps exact batchn BA+gates.
        for rows in [16usize, 24, 32, 64] {
            assert_eq!(
                select_cross_seq_ba_route(rows, 48, true, SmEnvOverride::Auto, true, true, true),
                CrossSeqBaRoute::ExactBatchnBaGates
            );
        }
    }

    #[test]
    fn multi_seq_recurrent_engages_only_for_c_ge_4_and_lazy_multi_kernels() {
        // C=2 stays per-seq (Split-V=4 gives 192 CTAs vs 96 CTAs).
        assert!(!should_use_multi_seq_recurrent(2, 8, true, true, true));
        assert!(!should_use_multi_seq_recurrent(3, 8, true, true, true));
        // C=4, 8, 16 engage 1-launch multi-seq conv1d + lazyfinal_multi.
        assert!(should_use_multi_seq_recurrent(4, 8, true, true, true));
        assert!(should_use_multi_seq_recurrent(8, 8, true, true, true));
        assert!(should_use_multi_seq_recurrent(16, 8, true, true, true));
        // Non-lazy or missing multi kernels stay on per-seq fallback.
        assert!(!should_use_multi_seq_recurrent(4, 8, false, true, true));
        assert!(!should_use_multi_seq_recurrent(4, 8, true, false, true));
        assert!(!should_use_multi_seq_recurrent(4, 8, true, true, false));
        assert!(!should_use_multi_seq_recurrent(4, 17, true, true, true));
    }

    #[test]
    fn deterministic_batch_pins_cross_seq_dflash_ssm_across_1_to_128() {
        use crate::layers::deterministic_batch::DeterministicBatchPolicy;

        for m in [1usize, 2, 4, 8, 16, 32, 64, 128] {
            for sm in [48u32, 188u32] {
                assert!(!resolve_ssm_proj_tc_for_sm_with_policy(
                    SmEnvOverride::Auto,
                    sm,
                    m as u32,
                    DeterministicBatchPolicy::Enabled,
                ));
                assert!(!resolve_ssm_proj_tc_for_sm_with_policy(
                    SmEnvOverride::Enabled,
                    sm,
                    m as u32,
                    DeterministicBatchPolicy::Enabled,
                ));
                assert_eq!(
                    select_cross_seq_ba_route_with_policy(
                        m,
                        sm,
                        true,
                        SmEnvOverride::Auto,
                        true,
                        true,
                        true,
                        DeterministicBatchPolicy::Enabled,
                    ),
                    CrossSeqBaRoute::ExactBatchnBaGates,
                    "cross-seq BA route drifted at M={m} sm={sm}"
                );
                let expected_proj = if m == 1 {
                    CrossSeqNvfp4ProjRoute::SerialGemv
                } else {
                    CrossSeqNvfp4ProjRoute::ExactRows
                };
                assert_eq!(
                    select_cross_seq_nvfp4_fallback_route_with_policy(
                        m as u32,
                        2_560,
                        sm,
                        true,
                        true,
                        4,
                        true,
                        true,
                        true,
                        true,
                        true,
                        DeterministicBatchPolicy::Enabled,
                    ),
                    expected_proj,
                    "cross-seq NVFP4 proj route drifted at M={m} sm={sm}"
                );
            }
            assert!(
                !should_use_multi_seq_recurrent_with_policy(
                    m,
                    8,
                    true,
                    true,
                    true,
                    DeterministicBatchPolicy::Enabled,
                ),
                "cross-seq multi-seq recurrent must stay disabled at C={m}"
            );
        }

        // Baseline (Disabled) preserves wide-SM BA cuBLASLt at M>=32 and multi-seq recurrent at C>=4.
        assert_eq!(
            select_cross_seq_ba_route_with_policy(
                32,
                188,
                true,
                SmEnvOverride::Auto,
                true,
                true,
                true,
                DeterministicBatchPolicy::Disabled,
            ),
            CrossSeqBaRoute::CublasBf16AndGates
        );
        assert!(should_use_multi_seq_recurrent_with_policy(
            8,
            8,
            true,
            true,
            true,
            DeterministicBatchPolicy::Disabled,
        ));
    }

    #[test]
    fn slot_distinctness_rejects_duplicate_and_oob_slots() {
        assert!(slot_ids_are_distinct(&[0, 1, 2, 3]));
        assert!(slot_ids_are_distinct(&[7, 2, 0, 5]));
        assert!(!slot_ids_are_distinct(&[0, 1, 1, 3]));
        assert!(!slot_ids_are_distinct(&[0, DFLASH_SSM_PTR_TABLE_MAX_SEQS]));
    }

    #[test]
    fn ptr_table_bytes_matches_six_tables_of_max_lanes() {
        assert_eq!(DFLASH_SSM_PTR_TABLE_BYTES, 6 * 128 * 8);
    }

    /// Live GPU parity proof at exact Qwen3.8-27B geometry (`nk=16, nv=48,
    /// kd=128, vd=128, d_conv=4, conv_dim=10240, qkvz_size=16384`) verifying
    /// 0-ULP bit-exact equality between:
    /// 1. Cross-sequence 1-launch Conv1d + 1-launch `lazyfinal_multi` (with
    ///    `gate_ptrs[s] | 1` fused retain across `C=4` distinct retain lanes) +
    ///    `gated_rms_norm_f32_multi_seq` vs per-sequence reference.
    /// 2. Split-V=4 (`C=2` regime) `lazyfinal_splitv4` vs `lazyfinal`.
    /// 3. CUDA graph capture + fresh-input H2D replay of the `C=4, K=8`
    ///    multi-seq recurrent pipeline vs eager per-sequence reference.
    #[cfg(feature = "cuda")]
    #[test]
    fn gpu_parity_dflash_cross_seq_ssm_multi_splitv4_and_graph_at_27b_geometry()
    -> anyhow::Result<()> {
        use half::bf16;
        use spark_runtime::gpu::GpuBackend;

        let Some(ptx) = atlas_kernels::ptx_for_model("qwen3.8-27b")
            .or_else(|| atlas_kernels::all_ptx_sets().into_iter().next())
        else {
            return Ok(());
        };
        let Ok(gpu) = spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &ptx.modules) else {
            return Ok(());
        };
        let stream = gpu.default_stream();

        let conv_seq_k = crate::layers::try_kernel(
            &gpu,
            "causal_conv1d",
            "causal_conv1d_update_l2norm_f32_sequence",
        );
        let conv_multi_k = crate::layers::try_kernel(
            &gpu,
            "causal_conv1d",
            "causal_conv1d_update_l2norm_f32_sequence_multi",
        );
        let gdn_lazy_k = crate::layers::try_kernel(
            &gpu,
            "gated_delta_rule",
            "gated_delta_rule_decode_f32_sequence_lazyfinal",
        );
        let gdn_multi_k = crate::layers::try_kernel(
            &gpu,
            "gated_delta_rule",
            "gated_delta_rule_decode_f32_sequence_lazyfinal_multi",
        );
        let gdn_splitv4_k = crate::layers::try_kernel(
            &gpu,
            "gated_delta_rule",
            "gated_delta_rule_decode_f32_sequence_lazyfinal_splitv4",
        );
        let gnorm_k = crate::layers::try_kernel(
            &gpu,
            "gated_rms_norm_f32_multi_seq",
            "gated_rms_norm_f32_multi_seq",
        );
        if conv_seq_k.0 == 0
            || conv_multi_k.0 == 0
            || gdn_lazy_k.0 == 0
            || gdn_multi_k.0 == 0
            || gdn_splitv4_k.0 == 0
            || gnorm_k.0 == 0
        {
            return Ok(());
        }

        // Exact 27B SSM geometry.
        let c = 4usize;
        let k = 8usize;
        let nk = 16usize;
        let nv = 48usize;
        let kd = 128usize;
        let vd = 128usize;
        let d_conv = 4usize;
        let key_dim = nk * kd;
        let value_dim = nv * vd;
        let conv_dim = key_dim * 2 + value_dim; // 10240
        let qkvz_size = conv_dim + value_dim; // 16384
        let total_rows = c * k;
        let eps = 1e-6f32;

        let h_elems = nv * kd * vd;
        let h_bytes = h_elems * FP32;
        let conv_state_elems = conv_dim * d_conv;
        let conv_state_bytes = conv_state_elems * FP32;
        let retain_lane_elems = GDN_LAZY_MAX_K * (qkvz_size + nv * 2);
        let retain_lane_bytes = retain_lane_elems * FP32;
        let gate_stride_elems = nv * 2;
        let gate_stride_bytes = gate_stride_elems * FP32;

        let mut allocs: Vec<DevicePtr> = Vec::new();
        let mut alloc_dev = |bytes: usize| -> anyhow::Result<DevicePtr> {
            let p = gpu.alloc(bytes)?;
            allocs.push(p);
            Ok(p)
        };

        // Deterministic PRNG helper for distinct per-seq inputs/states.
        let fill_f32 = |seed: u64, len: usize, scale: f32, bias: f32| -> Vec<f32> {
            let mut s = seed;
            (0..len)
                .map(|_| {
                    s = s
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    let u = ((s >> 40) as f32) * (1.0 / ((1u32 << 24) as f32));
                    (u * 2.0 - 1.0) * scale + bias
                })
                .collect()
        };
        let to_bf16_bytes = |vals: &[f32]| -> Vec<u8> {
            let mut out = Vec::with_capacity(vals.len() * BF16);
            for &v in vals {
                out.extend_from_slice(&bf16::from_f32(v).to_le_bytes());
            }
            out
        };
        let f32_bytes = |vals: &[f32]| -> &[u8] {
            // SAFETY: f32 is POD with no padding bytes.
            unsafe {
                std::slice::from_raw_parts(vals.as_ptr().cast::<u8>(), std::mem::size_of_val(vals))
            }
        };

        // Shared weights: conv1d [conv_dim, d_conv] BF16 and gated norm [vd] BF16.
        let conv_w_host = to_bf16_bytes(&fill_f32(0x1111, conv_dim * d_conv, 0.35, 0.0));
        let norm_w_host = to_bf16_bytes(&fill_f32(0x2222, vd, 0.25, 1.0));
        let conv_w_dev = alloc_dev(conv_w_host.len())?;
        let norm_w_dev = alloc_dev(norm_w_host.len())?;
        gpu.copy_h2d(&conv_w_host, conv_w_dev)?;
        gpu.copy_h2d(&norm_w_host, norm_w_dev)?;
        let conv_weight = DenseWeight { weight: conv_w_dev };
        let norm_weight = DenseWeight { weight: norm_w_dev };

        // Shared inputs: deinterleaved [c*k, qkvz_size] BF16, gates [c*k, 2*nv] FP32.
        let deint_dev = alloc_dev(total_rows * qkvz_size * BF16)?;
        let gates_dev = alloc_dev(total_rows * gate_stride_bytes)?;
        let out_ref_dev = alloc_dev(total_rows * value_dim * BF16)?;
        let out_multi_dev = alloc_dev(total_rows * value_dim * BF16)?;
        let out_sv4_dev = alloc_dev(2 * k * value_dim * BF16)?;

        // Per-seq states for ref, multi, and splitv4 (separate allocations so zero aliasing).
        let mut h_ref = Vec::with_capacity(c);
        let mut h_multi = Vec::with_capacity(c);
        let mut h_inter_ref = Vec::with_capacity(c);
        let mut h_inter_multi = Vec::with_capacity(c);
        let mut conv_ref = Vec::with_capacity(c);
        let mut conv_multi = Vec::with_capacity(c);
        let mut conv_inter_ref = Vec::with_capacity(c);
        let mut conv_inter_multi = Vec::with_capacity(c);
        let mut retain_ref = Vec::with_capacity(c);
        let mut retain_multi = Vec::with_capacity(c);

        for _ in 0..c {
            h_ref.push(alloc_dev(h_bytes)?);
            h_multi.push(alloc_dev(h_bytes)?);
            h_inter_ref.push(alloc_dev(h_bytes)?);
            h_inter_multi.push(alloc_dev(h_bytes)?);
            conv_ref.push(alloc_dev(conv_state_bytes)?);
            conv_multi.push(alloc_dev(conv_state_bytes)?);
            conv_inter_ref.push(alloc_dev(k * conv_state_bytes)?);
            conv_inter_multi.push(alloc_dev(k * conv_state_bytes)?);
            retain_ref.push(alloc_dev(retain_lane_bytes)?);
            retain_multi.push(alloc_dev(retain_lane_bytes)?);
        }

        // Fixed-address 6-table device pointer table for multi-seq + CUDA graph.
        let ptr_table_dev = alloc_dev(DFLASH_SSM_PTR_TABLE_BYTES)?;
        let mut table = [[0u64; DFLASH_SSM_PTR_TABLE_MAX_SEQS]; DFLASH_SSM_PTR_TABLE_COUNT];
        for s in 0..c {
            let gates_s = gates_dev.offset(s * k * gate_stride_bytes);
            assert_eq!(gates_s.0 & 1, 0);
            table[0][s] = h_multi[s].0;
            table[1][s] = retain_multi[s].0;
            table[2][s] = gates_s.0 | 1;
            table[3][s] = h_inter_multi[s].0;
            table[4][s] = conv_multi[s].0;
            table[5][s] = conv_inter_multi[s].0;
        }
        let mut table_bytes = vec![0u8; DFLASH_SSM_PTR_TABLE_BYTES];
        for (idx, &val) in table.iter().flatten().enumerate() {
            table_bytes[idx * 8..idx * 8 + 8].copy_from_slice(&val.to_le_bytes());
        }
        gpu.copy_h2d(&table_bytes, ptr_table_dev)?;

        let table_ptr = |index: usize| {
            ptr_table_dev.offset(index * DFLASH_SSM_PTR_TABLE_MAX_SEQS * std::mem::size_of::<u64>())
        };

        let upload_step_inputs = |step_seed: u64| -> anyhow::Result<()> {
            let deint_vals = fill_f32(step_seed ^ 0xA001, total_rows * qkvz_size, 0.5, 0.0);
            let deint_bf16 = to_bf16_bytes(&deint_vals);
            gpu.copy_h2d(&deint_bf16, deint_dev)?;

            let mut gates_host = vec![0f32; total_rows * gate_stride_elems];
            let raw_g = fill_f32(step_seed ^ 0xB002, total_rows * nv, 0.45, 0.5);
            let raw_b = fill_f32(step_seed ^ 0xC003, total_rows * nv, 0.4, 0.5);
            for r in 0..total_rows {
                for vh in 0..nv {
                    gates_host[r * gate_stride_elems + vh] = raw_g[r * nv + vh];
                    gates_host[r * gate_stride_elems + nv + vh] = raw_b[r * nv + vh];
                }
            }
            gpu.copy_h2d(f32_bytes(&gates_host), gates_dev)?;

            for s in 0..c {
                let h_init = fill_f32(step_seed ^ (0xD000 + s as u64), h_elems, 0.15, 0.0);
                let conv_init =
                    fill_f32(step_seed ^ (0xE000 + s as u64), conv_state_elems, 0.25, 0.0);
                gpu.copy_h2d(f32_bytes(&h_init), h_ref[s])?;
                gpu.copy_h2d(f32_bytes(&h_init), h_multi[s])?;
                gpu.copy_h2d(f32_bytes(&conv_init), conv_ref[s])?;
                gpu.copy_h2d(f32_bytes(&conv_init), conv_multi[s])?;
                gpu.memset_async(retain_ref[s], 0, retain_lane_bytes, stream)?;
                gpu.memset_async(retain_multi[s], 0, retain_lane_bytes, stream)?;
            }
            gpu.synchronize(stream)?;
            Ok(())
        };

        let run_per_seq_reference = || -> anyhow::Result<()> {
            for s in 0..c {
                let deint_s = deint_dev.offset(s * k * qkvz_size * BF16);
                let gates_s = gates_dev.offset(s * k * gate_stride_bytes);
                gpu.copy_d2d_async(
                    gates_s,
                    retain_ref[s].offset(GDN_LAZY_MAX_K * qkvz_size * FP32),
                    k * gate_stride_bytes,
                    stream,
                )?;
                ops::conv1d_update_l2norm_f32_sequence(
                    &gpu,
                    conv_seq_k,
                    conv_ref[s],
                    deint_s,
                    &conv_weight,
                    retain_ref[s],
                    conv_inter_ref[s],
                    k as u32,
                    conv_dim as u32,
                    d_conv as u32,
                    (key_dim * 2) as u32,
                    kd as u32,
                    eps,
                    qkvz_size as u32,
                    qkvz_size as u32,
                    conv_state_elems as u32,
                    stream,
                )?;
                ops::gdn_decode_f32_sequence_persistent(
                    &gpu,
                    gdn_lazy_k,
                    h_ref[s],
                    retain_ref[s],
                    retain_ref[s].offset(key_dim * FP32),
                    retain_ref[s].offset(key_dim * 2 * FP32),
                    gates_s,
                    gates_s.offset(nv * FP32),
                    retain_ref[s].offset(conv_dim * FP32),
                    h_inter_ref[s],
                    k as u32,
                    nk as u32,
                    nv as u32,
                    kd as u32,
                    vd as u32,
                    qkvz_size as u32,
                    qkvz_size as u32,
                    gate_stride_elems as u32,
                    qkvz_size as u32,
                    0,
                    stream,
                )?;
                ops::gated_rms_norm_f32_multi_seq(
                    &gpu,
                    gnorm_k,
                    retain_ref[s].offset(conv_dim * FP32),
                    deint_s.offset(conv_dim * BF16),
                    &norm_weight,
                    out_ref_dev.offset(s * k * value_dim * BF16),
                    nv as u32,
                    k as u32,
                    vd as u32,
                    eps,
                    qkvz_size as u32,
                    qkvz_size as u32,
                    value_dim as u32,
                    stream,
                )?;
            }
            Ok(())
        };

        let run_multi_seq = || -> anyhow::Result<()> {
            ops::conv1d_update_l2norm_f32_sequence_multi(
                &gpu,
                conv_multi_k,
                table_ptr(4),
                deint_dev,
                &conv_weight,
                table_ptr(1),
                table_ptr(5),
                c as u32,
                k as u32,
                conv_dim as u32,
                d_conv as u32,
                (key_dim * 2) as u32,
                kd as u32,
                eps,
                qkvz_size as u32,
                qkvz_size as u32,
                conv_state_elems as u32,
                stream,
            )?;
            ops::gdn_decode_f32_sequence_multi(
                &gpu,
                gdn_multi_k,
                table_ptr(0),
                table_ptr(1),
                table_ptr(2),
                table_ptr(3),
                c as u32,
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
                gate_stride_elems as u32,
                qkvz_size as u32,
                0,
                stream,
            )?;
            for s in 0..c {
                ops::gated_rms_norm_f32_multi_seq(
                    &gpu,
                    gnorm_k,
                    retain_multi[s].offset(conv_dim * FP32),
                    deint_dev.offset(s * k * qkvz_size * BF16 + conv_dim * BF16),
                    &norm_weight,
                    out_multi_dev.offset(s * k * value_dim * BF16),
                    nv as u32,
                    k as u32,
                    vd as u32,
                    eps,
                    qkvz_size as u32,
                    qkvz_size as u32,
                    value_dim as u32,
                    stream,
                )?;
            }
            Ok(())
        };

        let assert_all_bit_exact = |label: &str| -> anyhow::Result<()> {
            gpu.synchronize(stream)?;
            let mut out_ref_host = vec![0u8; total_rows * value_dim * BF16];
            let mut out_multi_host = vec![0u8; total_rows * value_dim * BF16];
            gpu.copy_d2h(out_ref_dev, &mut out_ref_host)?;
            gpu.copy_d2h(out_multi_dev, &mut out_multi_host)?;
            assert_eq!(
                out_multi_host, out_ref_host,
                "{label}: BF16 gated-norm output mismatch"
            );

            for s in 0..c {
                let mut hr = vec![0u8; h_bytes];
                let mut hm = vec![0u8; h_bytes];
                gpu.copy_d2h(h_inter_ref[s], &mut hr)?;
                gpu.copy_d2h(h_inter_multi[s], &mut hm)?;
                assert_eq!(hm, hr, "{label}: seq {s} h_state_inter mismatch");

                let mut cr = vec![0u8; conv_state_bytes];
                let mut cm = vec![0u8; conv_state_bytes];
                gpu.copy_d2h(conv_ref[s], &mut cr)?;
                gpu.copy_d2h(conv_multi[s], &mut cm)?;
                assert_eq!(cm, cr, "{label}: seq {s} conv_state mismatch");

                let mut cir = vec![0u8; k * conv_state_bytes];
                let mut cim = vec![0u8; k * conv_state_bytes];
                gpu.copy_d2h(conv_inter_ref[s], &mut cir)?;
                gpu.copy_d2h(conv_inter_multi[s], &mut cim)?;
                assert_eq!(
                    cim, cir,
                    "{label}: seq {s} conv_state_intermediates mismatch"
                );

                let mut rr = vec![0u8; retain_lane_bytes];
                let mut rm = vec![0u8; retain_lane_bytes];
                gpu.copy_d2h(retain_ref[s], &mut rr)?;
                gpu.copy_d2h(retain_multi[s], &mut rm)?;
                assert_eq!(
                    rm, rr,
                    "{label}: seq {s} retain lane (conv_f32 + fused gate retain) mismatch"
                );
            }
            Ok(())
        };

        // 1. Eager C=4, K=8 parity vs per-sequence reference.
        upload_step_inputs(0x1000_0001)?;
        run_per_seq_reference()?;
        run_multi_seq()?;
        assert_all_bit_exact("eager C=4 K=8 multi-seq")?;

        // 2. C=2, K=8 Split-V=4 (`lazyfinal_splitv4`) parity vs `lazyfinal`.
        for s in 0..2usize {
            let h_inter_sv4 = alloc_dev(h_bytes)?;
            let retain_sv4 = alloc_dev(retain_lane_bytes)?;
            gpu.copy_d2d_async(retain_ref[s], retain_sv4, retain_lane_bytes, stream)?;
            ops::gdn_decode_f32_sequence_persistent_splitv4(
                &gpu,
                gdn_splitv4_k,
                h_ref[s],
                retain_sv4,
                retain_sv4.offset(key_dim * FP32),
                retain_sv4.offset(key_dim * 2 * FP32),
                gates_dev.offset(s * k * gate_stride_bytes),
                gates_dev.offset(s * k * gate_stride_bytes + nv * FP32),
                retain_sv4.offset(conv_dim * FP32),
                h_inter_sv4,
                k as u32,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                qkvz_size as u32,
                qkvz_size as u32,
                gate_stride_elems as u32,
                qkvz_size as u32,
                0,
                stream,
            )?;
            ops::gated_rms_norm_f32_multi_seq(
                &gpu,
                gnorm_k,
                retain_sv4.offset(conv_dim * FP32),
                deint_dev.offset(s * k * qkvz_size * BF16 + conv_dim * BF16),
                &norm_weight,
                out_sv4_dev.offset(s * k * value_dim * BF16),
                nv as u32,
                k as u32,
                vd as u32,
                eps,
                qkvz_size as u32,
                qkvz_size as u32,
                value_dim as u32,
                stream,
            )?;
            gpu.synchronize(stream)?;
            let mut h_sv4_host = vec![0u8; h_bytes];
            let mut h_ref_host = vec![0u8; h_bytes];
            gpu.copy_d2h(h_inter_sv4, &mut h_sv4_host)?;
            gpu.copy_d2h(h_inter_ref[s], &mut h_ref_host)?;
            assert_eq!(h_sv4_host, h_ref_host, "Split-V=4 seq {s} h_inter mismatch");
        }
        let mut out_sv4_host = vec![0u8; 2 * k * value_dim * BF16];
        let mut out_ref2_host = vec![0u8; 2 * k * value_dim * BF16];
        gpu.copy_d2h(out_sv4_dev, &mut out_sv4_host)?;
        gpu.copy_d2h(out_ref_dev, &mut out_ref2_host)?;
        assert_eq!(
            out_sv4_host, out_ref2_host,
            "Split-V=4 BF16 output mismatch"
        );

        // 3. CUDA graph capture + fresh-input H2D replay (P4 fixed-address slab contract).
        let capture = crate::model::trait_impl::capture_guard::CaptureGuard::begin(
            &gpu,
            stream,
            "cross_seq_dflash_ssm_test",
        )?;
        run_multi_seq()?;
        let graph = capture.end()?;
        assert_ne!(graph.0, 0, "CUDA graph capture must produce a valid handle");

        // Replay graph with completely fresh step-2 inputs + states.
        upload_step_inputs(0x2000_0002)?;
        run_per_seq_reference()?;
        gpu.launch_graph(graph, stream)?;
        assert_all_bit_exact("CUDA-graphed C=4 K=8 multi-seq replay")?;

        gpu.destroy_graph(graph)?;
        for ptr in allocs {
            gpu.free(ptr)?;
        }
        Ok(())
    }
}
