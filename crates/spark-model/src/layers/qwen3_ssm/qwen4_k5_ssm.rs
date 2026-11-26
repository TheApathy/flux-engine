// SPDX-License-Identifier: AGPL-3.0-only

//! Exact small-row Qwen4 SSM projection batching with serial recurrence.

use super::*;

pub(super) const QWEN4_K16_BATCHED_VERIFY_ENV: &str = "ATLAS_QWEN4_K16_BATCHED_VERIFY";
pub(super) const QWEN4_K16_EXACT_ENV: &str = "ATLAS_QWEN4_K16_EXACT";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Qwen4ExactSsmRowsRoute {
    LegacyK5OrK9,
    NativeK16,
}

impl Qwen4ExactSsmRowsRoute {
    fn admits(self, rows: usize) -> bool {
        match self {
            // 2..=9 via `qwen4_hybrid_rows`, plus 1 for anchor-only slots inside
            // ragged cross-sequence frames (`qwen4_hybrid_cross_seq_rows`);
            // the projection/conv/GDN sequence kernels are row-count parameterized.
            Self::LegacyK5OrK9 => (1..=crate::layers::QWEN4_HYBRID_MAX_ROWS).contains(&rows),
            Self::NativeK16 => rows == 16,
        }
    }

    fn force_sequence_kernels(self) -> bool {
        matches!(self, Self::NativeK16)
    }
}

fn parse_exact_bool_env_value(name: &str, value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => {
            anyhow::bail!("{name} must be exactly 0 or 1, got {other:?}")
        }
    }
}

fn exact_bool_env(name: &str) -> Result<bool> {
    match std::env::var(name) {
        Ok(value) => parse_exact_bool_env_value(name, Some(&value)),
        Err(std::env::VarError::NotPresent) => parse_exact_bool_env_value(name, None),
        Err(std::env::VarError::NotUnicode(_)) => {
            anyhow::bail!("{name} must be valid UTF-8 and exactly 0 or 1")
        }
    }
}

/// `ATLAS_QWEN4_VERIFY_GNORM_ROWS` (default on unless `"0"`): the K-row exact
/// route's gated RMS norm runs every row in one launch instead of one per row.
pub(super) fn qwen4_verify_gnorm_rows_selected() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("ATLAS_QWEN4_VERIFY_GNORM_ROWS").as_deref() != Ok("0");
        if on {
            tracing::info!(
                "ATLAS_QWEN4_VERIFY_GNORM_ROWS=1 requested: one gated-norm launch per K-row SSM layer"
            );
        }
        on
    })
}

/// Receipt for the first multi-row gated norm (the env read alone proves nothing).
static VERIFY_GNORM_ROWS_LAUNCHED: std::sync::Once = std::sync::Once::new();

#[cfg(test)]
pub(super) fn parse_qwen4_k16_batched_verify(value: Option<&str>) -> Result<bool> {
    parse_exact_bool_env_value(QWEN4_K16_BATCHED_VERIFY_ENV, value)
}

#[cfg(test)]
pub(super) fn parse_qwen4_k16_exact(value: Option<&str>) -> Result<bool> {
    parse_exact_bool_env_value(QWEN4_K16_EXACT_ENV, value)
}

pub(super) fn qwen4_k16_batched_verify_requested() -> Result<bool> {
    exact_bool_env(QWEN4_K16_BATCHED_VERIFY_ENV)
}

pub(super) fn qwen4_k16_exact_requested() -> Result<bool> {
    exact_bool_env(QWEN4_K16_EXACT_ENV)
}

pub(super) fn qwen4_exact_ssm_rows_route(
    rows: usize,
    legacy_hybrid: bool,
    legacy_batch_ssm: bool,
    k16_requested: bool,
    k16_exact_requested: bool,
    has_recurrent_intermediates: bool,
) -> Option<Qwen4ExactSsmRowsRoute> {
    if rows == 16 && k16_requested && k16_exact_requested && has_recurrent_intermediates {
        Some(Qwen4ExactSsmRowsRoute::NativeK16)
    } else if (1..=crate::layers::QWEN4_HYBRID_MAX_ROWS).contains(&rows)
        && legacy_hybrid
        && legacy_batch_ssm
        && has_recurrent_intermediates
    {
        Some(Qwen4ExactSsmRowsRoute::LegacyK5OrK9)
    } else {
        None
    }
}

/// Whether single-sequence Qwen4 SSM dense projections (`qkvz` `16384x2560`,
/// `out_proj` `2560x6144`) should dispatch through Tensor Core coalesced NVFP4
/// W4A16 GEMM (`try_coalesced_nvfp4_w4a16_gemm`) instead of scalar exact GEMV.
///
/// Active at `C=1` ONLY when the caller entered `BatchedVerifyTcDenseGuard`
/// (`ATLAS_QWEN4_C1_TC_VERIFY=1`, `rows >= 4`), preserving 0-ULP scalar exact
/// parity when `ATLAS_QWEN4_C1_TC_VERIFY` is OFF (`0`/unset).
#[inline]
#[must_use]
const fn ssm_k5_tc_coalesced_eligible(
    scope_active: bool,
    coalesced_enabled: bool,
    rows: usize,
    n: usize,
    k: usize,
) -> bool {
    scope_active
        && coalesced_enabled
        && rows >= crate::layers::qwen4_hyper::DEFAULT_BATCHED_VERIFY_TC_DIRECT_MIN_ROWS
        && rows <= crate::layers::QWEN4_NATIVE_MAX_ROWS
        && n > 0
        && k > 0
        && k.is_multiple_of(32)
}

fn checked_row_bytes(rows: usize, width: usize, element_bytes: usize, name: &str) -> Result<usize> {
    rows.checked_mul(width)
        .and_then(|elements| elements.checked_mul(element_bytes))
        .ok_or_else(|| anyhow::anyhow!("Qwen4 exact SSM {name} extent overflow"))
}

impl Qwen3SsmLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn preflight_qwen4_exact_ssm_rows(
        &self,
        route: Qwen4ExactSsmRowsRoute,
        rows: usize,
        state: &SsmLayerState,
        h_intermediate: DevicePtr,
        conv_intermediate: DevicePtr,
        h_intermediate_stride: usize,
        conv_intermediate_stride: usize,
        ctx: &ForwardContext,
    ) -> Result<()> {
        const BF16: usize = 2;
        const FP32: usize = 4;
        anyhow::ensure!(
            route.admits(rows),
            "Qwen4 exact SSM route does not admit {rows} rows"
        );

        let h = ctx.config.hidden_size;
        let nk = ctx.config.linear_num_key_heads;
        let kd = ctx.config.linear_key_head_dim;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let qkvz_size = ctx.config.ssm_qkvz_size();
        let ba_size = ctx.config.ssm_ba_size();
        let d_conv = ctx.config.linear_conv_kernel_dim;
        if route == Qwen4ExactSsmRowsRoute::NativeK16 {
            anyhow::ensure!(
                ctx.config.is_qwen4_exp()
                    && h == 2_560
                    && ctx.config.residual_width() == 10_240
                    && nk == 16
                    && kd == 128
                    && nv == 48
                    && vd == 128
                    && d_conv == 4
                    && qkvz_size == 16_384
                    && ba_size == 96,
                "Qwen4 native K16 SSM requires exact Flash-Next recurrent geometry"
            );
        }
        anyhow::ensure!(
            nk > 0 && nv.is_multiple_of(nk),
            "Qwen4 exact SSM requires integral value-head grouping"
        );
        anyhow::ensure!(
            self.sequential_qkvz,
            "Qwen4 exact SSM requires sequential QKVZ"
        );
        anyhow::ensure!(
            self.qkvz_nvfp4.is_some(),
            "Qwen4 exact SSM requires NVFP4 QKVZ"
        );
        anyhow::ensure!(
            !state.h_is_f16 && !state.h_state.is_null() && !state.conv_state.is_null(),
            "Qwen4 exact SSM requires FP32 live recurrent state"
        );
        anyhow::ensure!(
            !h_intermediate.is_null() && !conv_intermediate.is_null(),
            "Qwen4 exact SSM requires recurrent intermediates"
        );
        anyhow::ensure!(
            h_intermediate != state.h_state && conv_intermediate != state.conv_state,
            "Qwen4 exact SSM live and checkpoint state must not alias"
        );
        let valid_h_stride = h_intermediate_stride == self.h_state_bytes
            || (h_intermediate_stride == 0 && self.gdn_seq_lazy_engaged_inner(rows));
        anyhow::ensure!(
            valid_h_stride
                && conv_intermediate_stride == self.conv_state_bytes
                && h_intermediate_stride.is_multiple_of(FP32)
                && conv_intermediate_stride.is_multiple_of(FP32),
            "Qwen4 exact SSM recurrent intermediate stride mismatch"
        );

        let batch_gdn = route.force_sequence_kernels()
            || std::env::var("ATLAS_QWEN4_K5_BATCH_GDN").ok().as_deref() == Some("1");
        let batch_conv = route.force_sequence_kernels()
            || match std::env::var("ATLAS_QWEN4_K5_BATCH_CONV").ok().as_deref() {
                Some("0") => false,
                Some("1") => true,
                _ => batch_gdn || crate::layers::qwen4_verify_fast_host_selected(),
            };
        anyhow::ensure!(
            self.ba_gates_batchn_exact_k.0 != 0
                && self.conv1d_l2norm_f32_k.0 != 0
                && self.gdn_f32_k.0 != 0
                && self.gated_rms_norm_f32_k.0 != 0
                && (!batch_conv || self.conv1d_l2norm_f32_sequence_k.0 != 0)
                && (!batch_gdn || self.gdn_f32_sequence_k.0 != 0),
            "Qwen4 exact SSM kernels are incomplete"
        );

        let sizes = ctx.buffers.sizes();
        anyhow::ensure!(
            checked_row_bytes(rows, qkvz_size, BF16, "deinterleaved")? <= sizes.ssm_deinterleaved,
            "Qwen4 exact SSM deinterleaved staging exceeds its arena"
        );
        anyhow::ensure!(
            checked_row_bytes(rows, nv * 2, FP32, "gate")? <= sizes.ssm_gates,
            "Qwen4 exact SSM gate staging exceeds its arena"
        );
        anyhow::ensure!(
            checked_row_bytes(rows, qkvz_size, FP32, "conv/GDN")? <= sizes.ssm_conv_out_f32,
            "Qwen4 exact SSM conv/GDN staging exceeds its arena"
        );
        anyhow::ensure!(
            checked_row_bytes(rows, nv * vd, BF16, "normalization")? <= sizes.ssm_qkvz,
            "Qwen4 exact SSM normalization staging exceeds its arena"
        );
        anyhow::ensure!(
            checked_row_bytes(rows, h, BF16, "output")? <= sizes.moe_output,
            "Qwen4 exact SSM output staging exceeds its arena"
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn ssm_forward_qwen4_exact_rows(
        &self,
        route: Qwen4ExactSsmRowsRoute,
        input: DevicePtr,
        rows: usize,
        state: &mut SsmLayerState,
        h_intermediate: DevicePtr,
        conv_intermediate: DevicePtr,
        h_intermediate_stride: usize,
        conv_intermediate_stride: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        const BF16: usize = 2;
        const FP32: usize = 4;
        let h = ctx.config.hidden_size;
        let nk = ctx.config.linear_num_key_heads;
        let kd = ctx.config.linear_key_head_dim;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let vpg = nv / nk;
        let key_dim = nk * kd;
        let value_dim = nv * vd;
        let conv_dim = key_dim * 2 + value_dim;
        let qkvz_size = ctx.config.ssm_qkvz_size();
        let ba_size = ctx.config.ssm_ba_size();
        let d_conv = ctx.config.linear_conv_kernel_dim;
        let qk_ch = (key_dim * 2) as u32;
        let eps = ctx.config.rms_norm_eps as f32;
        let batch_gdn = route.force_sequence_kernels()
            || std::env::var("ATLAS_QWEN4_K5_BATCH_GDN").ok().as_deref() == Some("1");
        let batch_conv = route.force_sequence_kernels()
            || match std::env::var("ATLAS_QWEN4_K5_BATCH_CONV").ok().as_deref() {
                Some("0") => false,
                Some("1") => true,
                _ => batch_gdn || crate::layers::qwen4_verify_fast_host_selected(),
            };
        anyhow::ensure!(
            route.admits(rows),
            "Qwen4 exact SSM route does not admit {rows} rows"
        );
        anyhow::ensure!(
            self.sequential_qkvz,
            "Qwen4 exact SSM requires sequential QKVZ"
        );
        let qkvz = self
            .qkvz_nvfp4
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Qwen4 exact SSM requires NVFP4 QKVZ"))?;
        anyhow::ensure!(
            !h_intermediate.is_null() && !conv_intermediate.is_null(),
            "Qwen4 exact SSM requires recurrent intermediates"
        );
        let valid_h_stride = h_intermediate_stride == self.h_state_bytes
            || (h_intermediate_stride == 0 && self.gdn_seq_lazy_engaged_inner(rows));
        anyhow::ensure!(
            valid_h_stride && conv_intermediate_stride == self.conv_state_bytes,
            "Qwen4 exact SSM recurrent intermediate stride mismatch"
        );
        anyhow::ensure!(
            self.ba_gates_batchn_exact_k.0 != 0
                && self.conv1d_l2norm_f32_k.0 != 0
                && self.gdn_f32_k.0 != 0
                && self.gated_rms_norm_f32_k.0 != 0
                && (!batch_conv || self.conv1d_l2norm_f32_sequence_k.0 != 0)
                && (!batch_gdn || self.gdn_f32_sequence_k.0 != 0),
            "Qwen4 exact SSM kernels are incomplete"
        );

        let deinterleaved = ctx.buffers.ssm_deinterleaved();
        let tc_scope = crate::layers::qwen4_hyper::batched_verify_tc_dense_scope_active();
        let coalesced_enabled = crate::layers::qwen4_fast_proj::coalesced_nvfp4_dense_enabled();
        let qkvz_coalesced =
            ssm_k5_tc_coalesced_eligible(tc_scope, coalesced_enabled, rows, qkvz_size, h)
                && crate::layers::qwen4_fast_proj::try_coalesced_nvfp4_w4a16_gemm(
                    ctx.gpu,
                    input,
                    qkvz,
                    deinterleaved,
                    rows,
                    qkvz_size,
                    h,
                    stream,
                )?;
        if qkvz_coalesced {
            static QKVZ_COALESCED_LOGGED: std::sync::Once = std::sync::Once::new();
            QKVZ_COALESCED_LOGGED.call_once(|| {
                let route = crate::layers::qwen4_fast_proj::coalesced_nvfp4_dispatch_symbol(
                    rows, qkvz_size, h,
                );
                tracing::info!(
                    component = "ssm_k5_qkvz",
                    route,
                    rows,
                    n = qkvz_size,
                    k = h,
                    "QWEN4_K5_SSM_TC_COALESCED_ENGAGED"
                );
            });
        } else {
            self.project_nvfp4_rows_exact_or_k1(
                ctx.gpu,
                input,
                qkvz,
                deinterleaved,
                rows as u32,
                qkvz_size as u32,
                h as u32,
                stream,
            )?;
        }
        spark_runtime::cublaslt::stage_timer::mark("v_qkvz", stream);

        // ATLAS_SSM_GDN_LAZY: the async commit decides replay-vs-copy from
        // `gdn_seq_lazy_engaged(rows)`, so this dispatch must take the lazy
        // kernel exactly when that predicate holds (and fail closed where it
        // cannot), or the commit would replay a retain buffer nobody wrote.
        let lazy = self.gdn_seq_lazy_engaged_inner(rows);
        anyhow::ensure!(
            !lazy || (batch_gdn && ctx.ddtree_parent_ids_dev.is_none()),
            "ATLAS_SSM_GDN_LAZY on the Qwen4 exact route requires the batched GDN sequence \
             (ATLAS_QWEN4_K5_BATCH_GDN=1) and no tree payload"
        );
        let max_k = crate::layers::qwen3_ssm::GDN_LAZY_MAX_K;
        let lane = crate::layers::qwen4_verify_lanes::current_lane();
        let keep_late_staged = if lazy
            && h_intermediate_stride == 0
            && batch_conv
            && batch_gdn
            && !state.h_is_f16
            && kd == 128
            && vd == 128
            && d_conv == 4
            && nk > 0
            && nk <= 32
            && nv >= nk
            && nv.is_multiple_of(nk)
            && ctx.config.output_gate_type == "sigmoid"
            && self.gdn_keep_late_supported_inner()
            && keep_late_ring_fits(nk, kd, nv, vd, qkvz_size)
        {
            self.find_single_seq_staged_desc_for_h_state(state.h_state)
        } else {
            None
        };
        if keep_late_staged.is_none() {
            // Materialize any deferred KEEP_LATE rows for this state (or a
            // colliding occupant of `lane` under single-lane serving) before
            // overwriting the retain buffer or reading live conv/H state.
            self.flush_keep_late_for_h_state(
                ctx.gpu,
                ctx.config,
                state.h_state,
                ctx.graph_capture,
                stream,
            )?;
            self.flush_keep_late_for_lane_if_different_h(
                ctx.gpu,
                ctx.config,
                lane,
                state.h_state,
                ctx.graph_capture,
                stream,
            )?;
        }
        let lazy_retain = if lazy && keep_late_staged.is_none() {
            Some(self.gdn_lazy_retain_current(ctx.gpu, qkvz_size, nv)?)
        } else {
            None
        };
        // Write ba_gates and conv_rows directly into this layer's
        // `gdn_lazy_retain` buffer when `batch_conv` is active instead of
        // staging into shared arena buffers and issuing two copy_d2d_async
        // calls per SSM layer (72 launches and 23.8 MB D2D per K=5 verify).
        // Disabled under `keep_late_staged` so ping-pong retain buffers stay
        // indirect via `GdnKeepLateCtrl` and CUDA graphs never overwrite
        // `retain_prev`.
        let direct_retain = lazy_retain.filter(|_| {
            batch_conv && super::qwen4_cross_seq_verify::qwen4_verify_direct_retain_enabled()
        });
        let gates = direct_retain
            .map(|retain| retain.offset(max_k * qkvz_size * FP32))
            .unwrap_or_else(|| ctx.buffers.ssm_gates());
        ops::dense_gemv_ba_gates_batchn(
            ctx.gpu,
            self.ba_gates_batchn_exact_k,
            input,
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
        spark_runtime::cublaslt::stage_timer::mark("v_ba", stream);

        let persistent_gdn = !lazy
            && batch_gdn
            && self.gdn_f32_sequence_persistent_k.0 != 0
            && super::qwen4_cross_seq_verify::ssm_gdn_seq_persistent_enabled();
        let normed = ctx.buffers.ssm_qkvz();
        let fused_verify_k = if lazy {
            self.qwen4_gdn_verify_fused_lazyfinal_k
        } else if persistent_gdn {
            self.qwen4_gdn_verify_fused_persistent_k
        } else {
            KernelHandle(0)
        };
        let max_lanes = crate::layers::qwen4_verify_lanes::MAX_LANES;
        let use_fused_verify = super::qwen4_gdn_verify_fused_enabled()
            && ctx.config.is_qwen4_exp()
            && fused_verify_k.0 != 0
            && !self.gdn_fused_group_done.is_null()
            && batch_conv
            && batch_gdn
            && !state.h_is_f16
            && kd == 128
            && vd == 128
            && d_conv == 4
            && nk > 0
            && nk <= 32
            && nv >= nk
            && nv.is_multiple_of(nk)
            && lane < max_lanes
            && ctx.config.output_gate_type == "sigmoid";

        if let Some((staged_lane, desc_dev)) = keep_late_staged {
            let counter_slot = if stream == ctx.gpu.default_stream() {
                staged_lane
            } else {
                max_lanes + staged_lane
            };
            let use_splitv4 = super::qwen4_gdn_split_v_enabled_for_single_seq()
                && self.qwen4_gdn_verify_fused_lazyfinal_multi_splitv4_k.0 != 0
                && self.single_seq_staged_is_fp32(staged_lane);
            if use_splitv4
                && let (Some((scratch_gdn, scratch_ss)), Some(split_done)) = (
                    Self::gdn_split_v_scratch_ptrs(ctx, rows, nv),
                    self.gdn_split_done_ptr(counter_slot, nv),
                )
            {
                static FUSED_GDN_KEEP_LATE_SPLITV4_LAUNCHED: std::sync::Once =
                    std::sync::Once::new();
                FUSED_GDN_KEEP_LATE_SPLITV4_LAUNCHED.call_once(|| {
                    tracing::info!(
                        rows,
                        nv,
                        ctas = nv * ops::GDN_SPLIT_V_FACTOR as usize,
                        "ENGAGED ATLAS_QWEN4_GDN_SPLIT_V: Split-V=4 single-seq KEEP_LATE fused GDN verify"
                    );
                });
                ops::qwen4_gdn_verify_fused_lazyfinal_multi_seq_splitv4(
                    ctx.gpu,
                    self.qwen4_gdn_verify_fused_lazyfinal_multi_splitv4_k,
                    desc_dev,
                    deinterleaved,
                    &self.ssm.conv1d,
                    gates,
                    gates.offset(nv * FP32),
                    &self.ssm.norm,
                    normed,
                    1,
                    nk as u32,
                    nv as u32,
                    1e-6,
                    eps,
                    qkvz_size as u32,
                    (nv * 2) as u32,
                    value_dim as u32,
                    qkvz_size as u32,
                    (conv_intermediate_stride / FP32) as u32,
                    (h_intermediate_stride / FP32) as u32,
                    scratch_gdn,
                    scratch_ss,
                    split_done,
                    stream,
                )?;
            } else {
                static FUSED_GDN_KEEP_LATE_LAUNCHED: std::sync::Once = std::sync::Once::new();
                FUSED_GDN_KEEP_LATE_LAUNCHED.call_once(|| {
                    tracing::info!(
                        "ENGAGED ATLAS_QWEN4_GDN_KEEP_LATE: single-seq graph-safe fused GDN verify launched (rows={rows})"
                    );
                });
                ops::qwen4_gdn_verify_fused_lazyfinal_multi_seq(
                    ctx.gpu,
                    self.qwen4_gdn_verify_fused_lazyfinal_multi_k,
                    desc_dev,
                    deinterleaved,
                    &self.ssm.conv1d,
                    gates,
                    gates.offset(nv * FP32),
                    &self.ssm.norm,
                    normed,
                    1,
                    nk as u32,
                    nv as u32,
                    1e-6,
                    eps,
                    qkvz_size as u32,
                    (nv * 2) as u32,
                    value_dim as u32,
                    qkvz_size as u32,
                    (conv_intermediate_stride / FP32) as u32,
                    (h_intermediate_stride / FP32) as u32,
                    stream,
                )?;
            }
            self.gdn_keep_late_mark_verify_ran(staged_lane, true);
            spark_runtime::cublaslt::stage_timer::mark("v_gnorm", stream);
        } else if use_fused_verify {
            self.gdn_keep_late_mark_verify_ran(lane, false);
            static FUSED_GDN_VERIFY_LAUNCHED: std::sync::Once = std::sync::Once::new();
            FUSED_GDN_VERIFY_LAUNCHED.call_once(|| {
                tracing::info!(
                    "ENGAGED ATLAS_QWEN4_GDN_VERIFY_FUSED: 1x1x1-CTA fused GDN verify launched (rows={rows}, lazy={lazy})"
                );
            });
            let counter_slot = if stream == ctx.gpu.default_stream() {
                lane
            } else {
                max_lanes + lane
            };
            let group_done = self.gdn_fused_group_done.offset(counter_slot * 32 * FP32);
            let splitv4_enabled = super::qwen4_gdn_split_v_enabled_for_single_seq();
            let splitv4_resources = if splitv4_enabled {
                Self::gdn_split_v_scratch_ptrs(ctx, rows, nv)
                    .zip(self.gdn_split_done_ptr(counter_slot, nv))
            } else {
                None
            };
            if let Some(retain) = lazy_retain {
                if direct_retain.is_none() {
                    ctx.gpu.copy_d2d_async(
                        gates,
                        retain.offset(max_k * qkvz_size * FP32),
                        rows * nv * 2 * FP32,
                        stream,
                    )?;
                }
                if let Some(((scratch_gdn, scratch_ss), split_done)) = splitv4_resources
                    && self.qwen4_gdn_verify_fused_lazyfinal_splitv4_k.0 != 0
                {
                    ops::qwen4_gdn_verify_fused_lazyfinal_splitv4(
                        ctx.gpu,
                        self.qwen4_gdn_verify_fused_lazyfinal_splitv4_k,
                        state.conv_state,
                        deinterleaved,
                        &self.ssm.conv1d,
                        gates,
                        gates.offset(nv * FP32),
                        state.h_state,
                        &self.ssm.norm,
                        normed,
                        retain,
                        conv_intermediate,
                        h_intermediate,
                        group_done,
                        rows as u32,
                        nk as u32,
                        nv as u32,
                        1e-6,
                        eps,
                        qkvz_size as u32,
                        (nv * 2) as u32,
                        value_dim as u32,
                        qkvz_size as u32,
                        (conv_intermediate_stride / FP32) as u32,
                        (h_intermediate_stride / FP32) as u32,
                        scratch_gdn,
                        scratch_ss,
                        split_done,
                        stream,
                    )?;
                } else {
                    ops::qwen4_gdn_verify_fused_lazyfinal(
                        ctx.gpu,
                        fused_verify_k,
                        state.conv_state,
                        deinterleaved,
                        &self.ssm.conv1d,
                        gates,
                        gates.offset(nv * FP32),
                        state.h_state,
                        &self.ssm.norm,
                        normed,
                        retain,
                        conv_intermediate,
                        h_intermediate,
                        group_done,
                        rows as u32,
                        nk as u32,
                        nv as u32,
                        1e-6,
                        eps,
                        qkvz_size as u32,
                        (nv * 2) as u32,
                        value_dim as u32,
                        qkvz_size as u32,
                        (conv_intermediate_stride / FP32) as u32,
                        (h_intermediate_stride / FP32) as u32,
                        stream,
                    )?;
                }
            } else if let Some(((scratch_gdn, scratch_ss), split_done)) = splitv4_resources
                && self.qwen4_gdn_verify_fused_persistent_splitv4_k.0 != 0
            {
                ops::qwen4_gdn_verify_fused_persistent_splitv4(
                    ctx.gpu,
                    self.qwen4_gdn_verify_fused_persistent_splitv4_k,
                    state.conv_state,
                    deinterleaved,
                    &self.ssm.conv1d,
                    gates,
                    gates.offset(nv * FP32),
                    state.h_state,
                    &self.ssm.norm,
                    normed,
                    conv_intermediate,
                    h_intermediate,
                    group_done,
                    rows as u32,
                    nk as u32,
                    nv as u32,
                    1e-6,
                    eps,
                    qkvz_size as u32,
                    (nv * 2) as u32,
                    value_dim as u32,
                    (conv_intermediate_stride / FP32) as u32,
                    (h_intermediate_stride / FP32) as u32,
                    scratch_gdn,
                    scratch_ss,
                    split_done,
                    stream,
                )?;
            } else {
                ops::qwen4_gdn_verify_fused_persistent(
                    ctx.gpu,
                    fused_verify_k,
                    state.conv_state,
                    deinterleaved,
                    &self.ssm.conv1d,
                    gates,
                    gates.offset(nv * FP32),
                    state.h_state,
                    &self.ssm.norm,
                    normed,
                    conv_intermediate,
                    h_intermediate,
                    group_done,
                    rows as u32,
                    nk as u32,
                    nv as u32,
                    1e-6,
                    eps,
                    qkvz_size as u32,
                    (nv * 2) as u32,
                    value_dim as u32,
                    (conv_intermediate_stride / FP32) as u32,
                    (h_intermediate_stride / FP32) as u32,
                    stream,
                )?;
            }
            spark_runtime::cublaslt::stage_timer::mark("v_gnorm", stream);
        } else {
            self.gdn_keep_late_mark_verify_ran(lane, false);
            // Stage every FP32 conv row so conv batching can be qualified
            // independently from GDN. Delaying GDN is safe: the two recurrent
            // states are disjoint and GDN consumes only its corresponding conv row.
            let conv_rows = direct_retain.unwrap_or_else(|| ctx.buffers.ssm_conv_out_f32());
            if batch_conv {
                ops::conv1d_update_l2norm_f32_sequence(
                    ctx.gpu,
                    self.conv1d_l2norm_f32_sequence_k,
                    state.conv_state,
                    deinterleaved,
                    &self.ssm.conv1d,
                    conv_rows,
                    conv_intermediate,
                    rows as u32,
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
            } else {
                for row in 0..rows {
                    let qkvz_row = deinterleaved.offset(row * qkvz_size * BF16);
                    ops::conv1d_update_l2norm(
                        ctx.gpu,
                        self.conv1d_l2norm_f32_k,
                        state.conv_state,
                        qkvz_row,
                        &self.ssm.conv1d,
                        conv_rows.offset(row * qkvz_size * FP32),
                        conv_dim as u32,
                        d_conv as u32,
                        1,
                        qk_ch,
                        kd as u32,
                        1e-6,
                        stream,
                    )?;
                    ctx.gpu.copy_d2d_async(
                        state.conv_state,
                        conv_intermediate.offset(row * conv_intermediate_stride),
                        conv_intermediate_stride,
                        stream,
                    )?;
                }
            }
            spark_runtime::cublaslt::stage_timer::mark("v_conv", stream);

            let gdn_rows = conv_rows.offset(conv_dim * FP32);
            if let Some(retain) = lazy_retain {
                if direct_retain.is_none() {
                    ctx.gpu
                        .copy_d2d_async(conv_rows, retain, rows * qkvz_size * FP32, stream)?;
                    ctx.gpu.copy_d2d_async(
                        gates,
                        retain.offset(max_k * qkvz_size * FP32),
                        rows * nv * 2 * FP32,
                        stream,
                    )?;
                }
                // Final H only, into inter[rows - 1]; h_state stays step-initial.
                ops::gdn_decode_f32_sequence_persistent(
                    ctx.gpu,
                    self.gdn_f32_sequence_lazyfinal_k,
                    state.h_state,
                    conv_rows,
                    conv_rows.offset(key_dim * FP32),
                    conv_rows.offset(key_dim * 2 * FP32),
                    gates,
                    gates.offset(nv * FP32),
                    gdn_rows,
                    h_intermediate,
                    rows as u32,
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
                )?;
            } else if persistent_gdn {
                ops::gdn_decode_f32_sequence_persistent(
                    ctx.gpu,
                    self.gdn_f32_sequence_persistent_k,
                    state.h_state,
                    conv_rows,
                    conv_rows.offset(key_dim * FP32),
                    conv_rows.offset(key_dim * 2 * FP32),
                    gates,
                    gates.offset(nv * FP32),
                    gdn_rows,
                    h_intermediate,
                    rows as u32,
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
                )?;
            } else if batch_gdn {
                ops::gdn_decode_f32_sequence(
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
                    rows as u32,
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
                )?;
            } else {
                for row in 0..rows {
                    let gates_row = gates.offset(row * nv * 2 * FP32);
                    let conv_row = conv_rows.offset(row * qkvz_size * FP32);
                    let gdn_row = gdn_rows.offset(row * qkvz_size * FP32);
                    ops::gdn_decode(
                        ctx.gpu,
                        self.gdn_f32_k,
                        state.h_state,
                        conv_row,
                        conv_row.offset(key_dim * FP32),
                        conv_row.offset(key_dim * 2 * FP32),
                        gates_row,
                        gates_row.offset(nv * FP32),
                        gdn_row,
                        1,
                        nk as u32,
                        nv as u32,
                        kd as u32,
                        vd as u32,
                        stream,
                    )?;
                    ctx.gpu.copy_d2d_async(
                        state.h_state,
                        h_intermediate.offset(row * h_intermediate_stride),
                        h_intermediate_stride,
                        stream,
                    )?;
                }
            }
            spark_runtime::cublaslt::stage_timer::mark("v_gdn", stream);

            // The multi-row sigmoid kernel runs qwen4_gated_rms_sigmoid_body for
            // each (row, head) with the per-row launch's grid.x, block and gate
            // stride, offset by whole rows, so every row is byte-identical.
            if qwen4_verify_gnorm_rows_selected()
                && self.gated_rms_norm_f32_multi_seq_k.0 != 0
                && ctx.config.output_gate_type == "sigmoid"
            {
                VERIFY_GNORM_ROWS_LAUNCHED.call_once(|| {
                    tracing::info!(
                        "ENGAGED ATLAS_QWEN4_VERIFY_GNORM_ROWS: multi-row gated norm launched (rows={rows})"
                    );
                });
                ops::gated_rms_norm_f32_multi_seq(
                    ctx.gpu,
                    self.gated_rms_norm_f32_multi_seq_k,
                    gdn_rows,
                    deinterleaved.offset((key_dim * 2 + value_dim) * BF16),
                    &self.ssm.norm,
                    normed,
                    nv as u32,
                    rows as u32,
                    vd as u32,
                    eps,
                    qkvz_size as u32,
                    qkvz_size as u32,
                    value_dim as u32,
                    stream,
                )?;
            } else {
                for row in 0..rows {
                    let qkvz_row = deinterleaved.offset(row * qkvz_size * BF16);
                    let gdn_row = gdn_rows.offset(row * qkvz_size * FP32);
                    ops::gated_rms_norm(
                        ctx.gpu,
                        self.gated_rms_norm_f32_k,
                        gdn_row,
                        qkvz_row.offset((key_dim * 2 + value_dim) * BF16),
                        &self.ssm.norm,
                        normed.offset(row * value_dim * BF16),
                        nv as u32,
                        vd as u32,
                        vd as u32,
                        eps,
                        vd as u32,
                        stream,
                    )?;
                }
            }
            spark_runtime::cublaslt::stage_timer::mark("v_gnorm", stream);
        }

        let output = ctx.buffers.moe_output();
        let out_coalesced =
            ssm_k5_tc_coalesced_eligible(tc_scope, coalesced_enabled, rows, h, value_dim)
                && crate::layers::qwen4_fast_proj::try_coalesced_nvfp4_w4a16_gemm(
                    ctx.gpu,
                    normed,
                    &self.ssm.out_proj,
                    output,
                    rows,
                    h,
                    value_dim,
                    stream,
                )?;
        if out_coalesced {
            static OUT_COALESCED_LOGGED: std::sync::Once = std::sync::Once::new();
            OUT_COALESCED_LOGGED.call_once(|| {
                let route = crate::layers::qwen4_fast_proj::coalesced_nvfp4_dispatch_symbol(
                    rows, h, value_dim,
                );
                tracing::info!(
                    component = "ssm_k5_out_proj",
                    route,
                    rows,
                    n = h,
                    k = value_dim,
                    "QWEN4_K5_SSM_TC_COALESCED_ENGAGED"
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
        spark_runtime::cublaslt::stage_timer::mark("v_ssm_out", stream);
        Ok(output)
    }
}

#[cfg(test)]
mod qwen4_k16_ssm_tests {
    include!("qwen4_k16_ssm_tests.rs");
}

#[cfg(test)]
mod c1_tc_coalesced_tests {
    use super::ssm_k5_tc_coalesced_eligible;
    use crate::layers::qwen4_hyper::{
        BatchedVerifyTcDenseGuard, batched_verify_tc_dense_scope_active,
    };

    #[test]
    fn c1_ssm_coalesced_tc_preserves_default_exact_and_admits_scoped_verify() {
        // Default C=1 (`ATLAS_QWEN4_C1_TC_VERIFY` OFF -> scope_active == false)
        // must stay 100% on exact scalar GEMV for both qkvz and out_proj.
        assert!(!batched_verify_tc_dense_scope_active());
        for rows in [1usize, 4, 5, 8, 9, 16] {
            assert!(
                !ssm_k5_tc_coalesced_eligible(false, true, rows, 16_384, 2_560),
                "default C=1 qkvz rows={rows} must stay exact"
            );
            assert!(
                !ssm_k5_tc_coalesced_eligible(false, true, rows, 2_560, 6_144),
                "default C=1 out_proj rows={rows} must stay exact"
            );
        }

        // Scoped (`ATLAS_QWEN4_C1_TC_VERIFY=1` or C >= 2): admits rows in 4..=256
        // for Flash-Next qkvz (16384x2560) and out_proj (2560x6144), while
        // rejecting rows <= 3, disabled coalesced toggle, or K % 32 != 0.
        let _guard = BatchedVerifyTcDenseGuard::enter(true);
        assert!(batched_verify_tc_dense_scope_active());
        for rows in [4usize, 5, 8, 9, 16] {
            assert!(
                ssm_k5_tc_coalesced_eligible(
                    batched_verify_tc_dense_scope_active(),
                    true,
                    rows,
                    16_384,
                    2_560,
                ),
                "scoped qkvz rows={rows}"
            );
            assert!(
                ssm_k5_tc_coalesced_eligible(
                    batched_verify_tc_dense_scope_active(),
                    true,
                    rows,
                    2_560,
                    6_144,
                ),
                "scoped out_proj rows={rows}"
            );
            assert!(!ssm_k5_tc_coalesced_eligible(
                batched_verify_tc_dense_scope_active(),
                false,
                rows,
                16_384,
                2_560,
            ));
        }
        for rows in [0usize, 1, 2, 3, 257] {
            assert!(!ssm_k5_tc_coalesced_eligible(
                true, true, rows, 16_384, 2_560
            ));
            assert!(!ssm_k5_tc_coalesced_eligible(
                true, true, rows, 2_560, 6_144
            ));
        }
        assert!(!ssm_k5_tc_coalesced_eligible(true, true, 8, 16_384, 2_544));
        assert!(!ssm_k5_tc_coalesced_eligible(true, true, 8, 0, 2_560));
    }
}
