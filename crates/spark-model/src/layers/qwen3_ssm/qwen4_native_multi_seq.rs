// SPDX-License-Identifier: AGPL-3.0-only

//! Native cross-request Qwen4 SSM decode for the C2-C32 ladder.
//!
//! Speculative verification advances several causally-related rows of one
//! request through one recurrent state.  Concurrent decode is different: each
//! row belongs to a different request and must advance a different recurrent
//! state.  This lane keeps the Qwen4 hyperconnection/MoE row slab batched while
//! feeding the existing multi-sequence conv/GDN kernels a device pointer table
//! of the request-owned states.

use super::*;

const BF16: usize = 2;
const FP32: usize = 4;
const NATIVE_SSM_PTR_CACHE_ENV: &str = "ATLAS_QWEN4_NATIVE_SSM_PTR_CACHE";

fn parse_native_ssm_ptr_cache(raw: Option<&str>) -> std::result::Result<bool, &'static str> {
    match raw {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(_) => Err("ATLAS_QWEN4_NATIVE_SSM_PTR_CACHE must be exact value 0 or 1"),
    }
}

fn parse_native_ssm_ptr_cache_env(
    value: std::result::Result<String, std::env::VarError>,
) -> std::result::Result<bool, &'static str> {
    match value {
        Ok(value) => parse_native_ssm_ptr_cache(Some(value.as_str())),
        Err(std::env::VarError::NotPresent) => parse_native_ssm_ptr_cache(None),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err("ATLAS_QWEN4_NATIVE_SSM_PTR_CACHE must be valid UTF-8 and exactly 0 or 1")
        }
    }
}

fn native_ssm_ptr_cache_selected() -> Result<bool> {
    static SELECTED: std::sync::OnceLock<std::result::Result<bool, &'static str>> =
        std::sync::OnceLock::new();
    match *SELECTED
        .get_or_init(|| parse_native_ssm_ptr_cache_env(std::env::var(NATIVE_SSM_PTR_CACHE_ENV)))
    {
        Ok(selected) => Ok(selected),
        Err(message) => anyhow::bail!(message),
    }
}

pub(super) const fn qwen4_native_multi_seq_width(rows: usize) -> bool {
    rows >= 2 && rows <= crate::layers::QWEN4_NATIVE_MAX_ROWS
}

fn checked_bytes(rows: usize, width: usize, element_bytes: usize, name: &str) -> Result<usize> {
    rows.checked_mul(width)
        .and_then(|items| items.checked_mul(element_bytes))
        .ok_or_else(|| anyhow::anyhow!("Qwen4 native multi-seq {name} extent overflow"))
}

impl Qwen3SsmLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_qwen4_native_multi_seq<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        rows: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            qwen4_native_multi_seq_width(rows),
            "Qwen4 native multi-seq SSM currently admits C2-C32, got C{rows}"
        );
        ensure!(
            states.len() == rows,
            "Qwen4 native multi-seq SSM state extent {} != active rows {rows}",
            states.len()
        );
        ensure!(
            !ctx.graph_capture && ctx.ssm_multi_seq_ptr_table_override.is_none(),
            "Qwen4 native multi-seq C2-C32 is eager-only until its full HC/PLE graph audit"
        );
        ensure!(
            ctx.config.is_qwen4_exp()
                && ctx.config.hc_count == 4
                && ctx.config.hidden_size == 2_560
                && ctx.config.residual_width() == 10_240
                && !ctx.config.use_fp32_residual(),
            "Qwen4 native multi-seq SSM requires the canonical BF16 Flash-Next geometry"
        );
        let (attn_hyper, mlp_hyper) = match (&self.qwen4_attn_hyper, &self.qwen4_mlp_hyper) {
            (Some(attn), Some(mlp)) => (attn, mlp),
            _ => anyhow::bail!("Qwen4 native multi-seq SSM requires both hyperconnections"),
        };
        ensure!(
            attn_hyper.residual_width() == ctx.config.residual_width()
                && mlp_hyper.residual_width() == ctx.config.residual_width(),
            "Qwen4 native multi-seq hyperconnection width mismatch"
        );
        {
            let readonly_states: Vec<&(dyn LayerState + 'static)> = states
                .iter()
                .map(|state| &**state as &(dyn LayerState + 'static))
                .collect();
            self.preflight_qwen4_native_multi_seq(rows, &readonly_states, ctx)?;
        }

        // Batched HC preparation retains one injection vector in each
        // request's residual row.  The recurrent core below never sees or
        // manufactures padding rows: `rows` is the active C2-C32 width.
        let ssm_input = attn_hyper.prepare_batched(
            hidden,
            residual,
            rows,
            ctx.buffers,
            ctx.gpu,
            ctx.config.rms_norm_eps as f32,
            stream,
        )?;
        let ssm_output =
            self.ssm_forward_qwen4_native_multi_seq(ssm_input, rows, states, ctx, stream)?;
        attn_hyper.inject_saved_batched(hidden, ssm_output, residual, rows, ctx.gpu, stream)?;

        let ffn_input = mlp_hyper.prepare_batched(
            hidden,
            residual,
            rows,
            ctx.buffers,
            ctx.gpu,
            ctx.config.rms_norm_eps as f32,
            stream,
        )?;
        self.ffn
            .forward_qwen4_native_target_rows(ffn_input, rows, ctx, stream)?;
        mlp_hyper.inject_saved_batched(
            hidden,
            ctx.buffers.moe_output(),
            residual,
            rows,
            ctx.gpu,
            stream,
        )?;

        static ENGAGED: std::sync::Once = std::sync::Once::new();
        ENGAGED.call_once(|| {
            tracing::info!(
                "QWEN4_NATIVE_CONCURRENT_SSM_ENGAGED rows={rows} recurrence=pointer_table padding=excluded"
            );
        });
        Ok(())
    }

    /// Refuse an incomplete or aliased state lane before the first HC kernel
    /// can alter `residual`.  In particular, repeated dummy padding states
    /// cannot pass this gate.
    pub(super) fn preflight_qwen4_native_multi_seq(
        &self,
        rows: usize,
        states: &[&(dyn LayerState + 'static)],
        ctx: &ForwardContext,
    ) -> Result<()> {
        // Parse before the model-wide preflight permits any layer to mutate
        // state. Misspelled selector values never silently choose a route.
        let _ = native_ssm_ptr_cache_selected()?;
        ensure!(
            qwen4_native_multi_seq_width(rows) && states.len() == rows,
            "Qwen4 native multi-seq SSM requires one state per C2-C32 row"
        );
        self.qwen4_native_multi_seq_capability(rows, ctx)?;

        let mut h_states = Vec::with_capacity(rows);
        let mut conv_states = Vec::with_capacity(rows);
        for (row, state) in states.iter().enumerate() {
            let state = state
                .as_any()
                .downcast_ref::<SsmLayerState>()
                .ok_or_else(|| anyhow::anyhow!("Qwen4 native row {row} is not SSM state"))?;
            ensure!(
                !state.h_is_f16 && !state.h_state.is_null() && !state.conv_state.is_null(),
                "Qwen4 native row {row} requires live FP32 recurrent state"
            );
            ensure!(
                !h_states.contains(&state.h_state.0) && !conv_states.contains(&state.conv_state.0),
                "Qwen4 native row {row} aliases another request's recurrent state"
            );
            h_states.push(state.h_state.0);
            conv_states.push(state.conv_state.0);
        }
        Ok(())
    }

    /// The state-independent half of [`Self::preflight_qwen4_native_multi_seq`]:
    /// selectors, topology, geometry, hyperconnections, pointer-table capacity,
    /// kernels, exact projection tiers, the MoE half and staging arenas. The
    /// model also calls it once per width to route frames before dispatch.
    pub(super) fn qwen4_native_multi_seq_capability(
        &self,
        rows: usize,
        ctx: &ForwardContext,
    ) -> Result<()> {
        let _ = native_ssm_ptr_cache_selected()?;
        ensure!(
            qwen4_native_multi_seq_width(rows),
            "Qwen4 native multi-seq SSM admits only C2-C32"
        );
        ensure!(
            !ctx.graph_capture
                && ctx.ssm_multi_seq_ptr_table_override.is_none()
                && ctx.config.tp_world_size <= 4
                && (ctx.config.ep_world_size <= 1
                    || ctx.config.ep_world_size == ctx.config.tp_world_size),
            "Qwen4 native multi-seq SSM requires eager TP<=4 (+ matching EP) target execution"
        );
        ensure!(
            ctx.config.is_qwen4_exp()
                && ctx.config.hc_count == 4
                && ctx.config.hidden_size == 2_560
                && ctx.config.residual_width() == 10_240
                && !ctx.config.use_fp32_residual(),
            "Qwen4 native multi-seq SSM requires the canonical BF16 Flash-Next geometry"
        );
        let (attn_hyper, mlp_hyper) = match (&self.qwen4_attn_hyper, &self.qwen4_mlp_hyper) {
            (Some(attn), Some(mlp)) => (attn, mlp),
            _ => anyhow::bail!("Qwen4 native multi-seq SSM requires both hyperconnections"),
        };
        ensure!(
            attn_hyper.residual_width() == ctx.config.residual_width()
                && mlp_hyper.residual_width() == ctx.config.residual_width(),
            "Qwen4 native multi-seq hyperconnection width mismatch"
        );
        attn_hyper.preflight_qwen4_native_batched(rows, true)?;
        mlp_hyper.preflight_qwen4_native_batched(rows, true)?;
        ensure!(
            rows <= self.ssm_multi_seq_ptr_max && !self.ssm_multi_seq_ptr_scratch.is_null(),
            "Qwen4 native multi-seq C{rows} exceeds pointer-table capacity {}",
            self.ssm_multi_seq_ptr_max
        );
        ensure!(
            self.sequential_qkvz && self.qkvz_nvfp4.is_some(),
            "Qwen4 native multi-seq SSM requires sequential NVFP4 QKVZ"
        );
        ensure!(
            self.ba_gates_batchn_exact_k.0 != 0
                && self.conv1d_l2norm_f32_multi_seq_k.0 != 0
                && self.gdn_decode_f32_multi_seq_k.0 != 0
                && self.gated_rms_norm_f32_multi_seq_k.0 != 0,
            "Qwen4 native multi-seq recurrent kernel set is incomplete"
        );
        let projection_route = super::exact_projection::exact_projection_route(
            self.w4a16_exact_projection_kernels,
            rows as u32,
        )?;
        if matches!(projection_route, ops::ExactLmHeadRoute::SerialK1(_)) {
            ensure!(
                ops::use_gemv_sw(self.gemv_sw, self.w4a16_gemv_sw_k) || self.w4a16_gemv_k.0 != 0,
                "Qwen4 native multi-seq SSM is missing its K1 projection fallback kernel"
            );
        }
        self.ffn.preflight_qwen4_native_exact_rows(rows, ctx)?;

        let h = ctx.config.hidden_size;
        let nk = ctx.config.linear_num_key_heads;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        ensure!(
            nk > 0 && nv.is_multiple_of(nk),
            "invalid Qwen4 GDN head grouping"
        );
        let value_dim = nv * vd;
        let qkvz_size = ctx.config.ssm_qkvz_size();
        ensure!(
            checked_bytes(rows, qkvz_size, BF16, "QKVZ")? <= ctx.buffers.sizes().ssm_deinterleaved
                && checked_bytes(rows, nv * 2, FP32, "gates")? <= ctx.buffers.sizes().ssm_gates
                && checked_bytes(rows, qkvz_size, FP32, "conv/GDN")?
                    <= ctx.buffers.sizes().ssm_conv_out_f32
                && checked_bytes(rows, value_dim, BF16, "normalized output")?
                    <= ctx.buffers.sizes().ssm_qkvz
                && checked_bytes(rows, h, BF16, "projection output")?
                    <= ctx.buffers.sizes().moe_output,
            "Qwen4 native multi-seq SSM staging exceeds its arena"
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn ssm_forward_qwen4_native_multi_seq<'a, 'b: 'a>(
        &self,
        input: DevicePtr,
        rows: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        ensure!(!input.is_null(), "Qwen4 native multi-seq SSM input is null");
        ensure!(
            rows <= self.ssm_multi_seq_ptr_max,
            "Qwen4 native multi-seq C{rows} exceeds pointer-table capacity {}",
            self.ssm_multi_seq_ptr_max
        );
        ensure!(
            self.sequential_qkvz,
            "Qwen4 native multi-seq SSM requires sequential QKVZ"
        );
        let qkvz = self
            .qkvz_nvfp4
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Qwen4 native multi-seq SSM requires NVFP4 QKVZ"))?;
        ensure!(
            self.ba_gates_batchn_exact_k.0 != 0
                && self.conv1d_l2norm_f32_multi_seq_k.0 != 0
                && self.gdn_decode_f32_multi_seq_k.0 != 0
                && self.gated_rms_norm_f32_multi_seq_k.0 != 0,
            "Qwen4 native multi-seq recurrent kernel set is incomplete"
        );
        for (row, state) in states.iter_mut().enumerate() {
            let state = state
                .as_any_mut()
                .downcast_mut::<SsmLayerState>()
                .ok_or_else(|| anyhow::anyhow!("Qwen4 native row {row} is not SSM state"))?;
            ensure!(
                !state.h_is_f16 && !state.h_state.is_null() && !state.conv_state.is_null(),
                "Qwen4 native row {row} requires live FP32 recurrent state"
            );
        }

        let h = ctx.config.hidden_size;
        let nk = ctx.config.linear_num_key_heads;
        let kd = ctx.config.linear_key_head_dim;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        ensure!(
            nk > 0 && nv.is_multiple_of(nk),
            "invalid Qwen4 GDN head grouping"
        );
        let vpg = nv / nk;
        let key_dim = nk * kd;
        let value_dim = nv * vd;
        let conv_dim = key_dim * 2 + value_dim;
        let qkvz_size = ctx.config.ssm_qkvz_size();
        let ba_size = ctx.config.ssm_ba_size();
        ensure!(
            checked_bytes(rows, qkvz_size, BF16, "QKVZ")? <= ctx.buffers.sizes().ssm_deinterleaved
                && checked_bytes(rows, nv * 2, FP32, "gates")? <= ctx.buffers.sizes().ssm_gates
                && checked_bytes(rows, qkvz_size, FP32, "conv/GDN")?
                    <= ctx.buffers.sizes().ssm_conv_out_f32
                && checked_bytes(rows, value_dim, BF16, "normalized output")?
                    <= ctx.buffers.sizes().ssm_qkvz
                && checked_bytes(rows, h, BF16, "projection output")?
                    <= ctx.buffers.sizes().moe_output,
            "Qwen4 native multi-seq SSM staging exceeds its arena"
        );

        // Exact K1-order projections and fused BA/gate rows preserve greedy
        // output while the conv/GDN kernels parallelize over independent state
        // pointers.  This is cross-request batching, not speculative sequence
        // recurrence.
        let deinterleaved = ctx.buffers.ssm_deinterleaved();
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
        let gates = ctx.buffers.ssm_gates();
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

        let pointer_table_reused = if native_ssm_ptr_cache_selected()? {
            self.refresh_native_multi_seq_ptr_table_cached(states, rows, ctx.gpu, stream)?
        } else {
            self.refresh_multi_seq_ptr_table(states, rows, ctx.gpu, stream)?;
            false
        };
        if pointer_table_reused {
            static PTR_CACHE_ENGAGED: std::sync::Once = std::sync::Once::new();
            PTR_CACHE_ENGAGED.call_once(|| {
                tracing::info!(
                    "QWEN4_NATIVE_SSM_PTR_CACHE_ENGAGED rows={rows} table=stable h2d=elided"
                );
            });
        }
        let h_state_ptrs = self.ssm_multi_seq_ptr_scratch;
        let conv_state_ptrs = h_state_ptrs.offset(rows * std::mem::size_of::<u64>());
        let conv_rows = ctx.buffers.ssm_conv_out_f32();
        ops::conv1d_update_l2norm_f32_multi_seq(
            ctx.gpu,
            self.conv1d_l2norm_f32_multi_seq_k,
            conv_state_ptrs,
            deinterleaved,
            &self.ssm.conv1d,
            conv_rows,
            conv_dim as u32,
            ctx.config.linear_conv_kernel_dim as u32,
            rows as u32,
            (key_dim * 2) as u32,
            kd as u32,
            1e-6,
            qkvz_size as u32,
            qkvz_size as u32,
            stream,
        )?;

        let gdn_offset = conv_dim * FP32;
        let gdn_rows = conv_rows.offset(gdn_offset);
        ops::gdn_decode_f32_multi_seq(
            ctx.gpu,
            self.gdn_decode_f32_multi_seq_k,
            h_state_ptrs,
            conv_rows,
            conv_rows.offset(key_dim * FP32),
            conv_rows.offset(key_dim * 2 * FP32),
            gates,
            gates.offset(nv * FP32),
            gdn_rows,
            rows as u32,
            nk as u32,
            nv as u32,
            kd as u32,
            vd as u32,
            (nv * 2) as u32,
            qkvz_size as u32,
            qkvz_size as u32,
            qkvz_size as u32,
            stream,
        )?;

        let normed = ctx.buffers.ssm_qkvz();
        ops::gated_rms_norm_f32_multi_seq(
            ctx.gpu,
            self.gated_rms_norm_f32_multi_seq_k,
            gdn_rows,
            deinterleaved.offset(conv_dim * BF16),
            &self.ssm.norm,
            normed,
            nv as u32,
            rows as u32,
            vd as u32,
            ctx.config.rms_norm_eps as f32,
            qkvz_size as u32,
            qkvz_size as u32,
            value_dim as u32,
            stream,
        )?;
        let output = ctx.buffers.moe_output();
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
        if ctx.config.tp_world_size > 1
            && let Some(comm) = ctx.comm
        {
            comm.all_reduce_async(output.0, rows * h * BF16, stream)?;
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        parse_native_ssm_ptr_cache, parse_native_ssm_ptr_cache_env, qwen4_native_multi_seq_width,
    };

    #[test]
    fn native_ladder_matches_qwen4_native_max_rows() {
        for rows in 0..=(crate::layers::QWEN4_NATIVE_MAX_ROWS + 1) {
            assert_eq!(
                qwen4_native_multi_seq_width(rows),
                (2..=crate::layers::QWEN4_NATIVE_MAX_ROWS).contains(&rows)
            );
        }
    }

    #[test]
    fn native_pointer_cache_selector_is_strict_and_default_off() {
        assert_eq!(parse_native_ssm_ptr_cache(None), Ok(false));
        assert_eq!(parse_native_ssm_ptr_cache(Some("0")), Ok(false));
        assert_eq!(parse_native_ssm_ptr_cache(Some("1")), Ok(true));
        for invalid in ["", "true", "yes", "01", " 1"] {
            assert!(parse_native_ssm_ptr_cache(Some(invalid)).is_err());
        }
    }

    #[cfg(unix)]
    #[test]
    fn native_pointer_cache_selector_rejects_non_utf8() {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt;

        let error = parse_native_ssm_ptr_cache_env(Err(std::env::VarError::NotUnicode(
            OsString::from_vec(vec![0xff]),
        )))
        .unwrap_err();
        assert_eq!(
            error,
            "ATLAS_QWEN4_NATIVE_SSM_PTR_CACHE must be valid UTF-8 and exactly 0 or 1"
        );
    }
}
