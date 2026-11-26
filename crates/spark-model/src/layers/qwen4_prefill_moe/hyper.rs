// SPDX-License-Identifier: AGPL-3.0-only

//! Optional exact HC preparation and one saved-scale injection for grouped FFN.

use crate::{
    layer::ForwardContext,
    layers::qwen4_hyper::{pending, prefill_fused, prefill_inject::Boundary},
    layers::{FfnComponent, Qwen4HyperConnection},
};
use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

pub(super) fn finish(
    mlp: &Qwen4HyperConnection,
    ffn: &FfnComponent,
    hidden: DevicePtr,
    residual: DevicePtr,
    rows: usize,
    ctx: &ForwardContext,
    stream: u64,
) -> Result<()> {
    use spark_runtime::cublaslt::stage_timer::mark;
    mark("mixer", stream);
    // The MoE FFN reads only the packed rows in norm_output.
    let _amax = crate::layers::qwen4_fp8_act_amax::Arm::enter(
        rows >= 1024 && crate::layers::qwen4_fast_proj::fp8_tensor_scales(),
    )?;
    let mxfp8_req = match ffn {
        FfnComponent::Moe(moe)
            if rows > 16
                && ctx.config.hidden_size == 2560
                && ctx.buffers.sizes().expert_up_out >= rows * (2560 + 80) =>
        {
            let xq = ctx.buffers.expert_up_out();
            let gatepre = crate::layers::qwen4_hyper::prefill_inject::selection()?.gatepre;
            let ss_out = if gatepre && ctx.buffers.sizes().moe_gate_scalars >= rows * 4 {
                ctx.buffers.moe_gate_scalars()
            } else {
                DevicePtr::NULL
            };
            Some(prefill_fused::UpmixMxfp8Request {
                xq,
                xs: xq.offset(rows * 2560),
                shared_gate: moe.weights.shared_expert_gate.weight,
                ss_out,
                rows,
                stream,
            })
        }
        _ => None,
    };
    let _mxfp8 = prefill_fused::UpmixMxfp8Arm::enter(mxfp8_req)?;
    let input = mlp.prepare_prefill_exact_packed(
        hidden,
        residual,
        rows,
        ctx.buffers,
        ctx.gpu,
        ctx.config.rms_norm_eps as f32,
        stream,
    )?;
    mark("moe_hc_prepare", stream);
    // With MHC_FUSED=combine the FFN may also leave its output combine
    // (unpermute + shared-expert blend) to the next preparation.
    if mlp.combine_deferrable()? {
        pending::arm_combine_sink();
    }
    let ffn_result = ffn.forward_prefill(input, rows, ctx, stream);
    crate::layers::qwen4_fast_proj::clear_prequant_fp8_act();
    let combine = pending::take_combine_sink();
    ffn_result?;
    mark("moe", stream);
    // May stay pending for the next layer's attention hyperconnection.
    mlp.inject_saved_or_defer(
        Boundary::Layer,
        hidden,
        ctx.buffers.moe_output(),
        residual,
        rows,
        combine,
        ctx.gpu,
        stream,
    )?;
    mark("moe_hc_inject", stream);
    Ok(())
}

/// [`finish`] for a final-layer tail (`ATLAS_QWEN4_PREFILL_LAST_LAYER_CROP`):
/// prepare the `hc_rows` rows at `hidden`/`residual` with the per-stream GEMM
/// order of a `ref_rows` preparation, then run the FFN and its injection on
/// the last `moe_rows` of them (all of them when the two are equal).
#[allow(clippy::too_many_arguments)]
pub(super) fn finish_tail(
    mlp: &Qwen4HyperConnection,
    ffn: &FfnComponent,
    hidden: DevicePtr,
    residual: DevicePtr,
    hc_rows: usize,
    moe_rows: usize,
    ref_rows: usize,
    ctx: &ForwardContext,
    stream: u64,
) -> Result<()> {
    use spark_runtime::cublaslt::stage_timer::mark;
    mark("mixer", stream);
    let input = {
        let _order = prefill_fused::ReferenceRowsGuard::enter(ref_rows);
        mlp.prepare_prefill_exact_packed(
            hidden,
            residual,
            hc_rows,
            ctx.buffers,
            ctx.gpu,
            ctx.config.rms_norm_eps as f32,
            stream,
        )?
    };
    mark("moe_hc_prepare", stream);
    let skip = hc_rows
        .checked_sub(moe_rows)
        .filter(|&skip| skip == 0 || skip >= moe_rows)
        .ok_or_else(|| {
            anyhow::anyhow!("MoE tail of {moe_rows} rows overlaps its {hc_rows}-row staging")
        })?;
    anyhow::ensure!(
        input == ctx.buffers.norm_output(),
        "MoE tail requires the packed rows in norm_output"
    );
    // The compact MoE reads its rows from the front of norm_output.
    let core_bytes = ctx.config.hidden_size * 2;
    if skip > 0 {
        ctx.gpu.copy_d2d_async(
            input.offset(skip * core_bytes),
            input,
            moe_rows * core_bytes,
            stream,
        )?;
    }
    if mlp.combine_deferrable()? {
        pending::arm_combine_sink();
    }
    let ffn_result = ffn.forward_prefill(input, moe_rows, ctx, stream);
    let combine = pending::take_combine_sink();
    ffn_result?;
    mark("moe", stream);
    let row_bytes = ctx.config.residual_width() * 2;
    mlp.inject_saved_or_defer(
        Boundary::Layer,
        hidden.offset(skip * row_bytes),
        ctx.buffers.moe_output(),
        residual.offset(skip * row_bytes),
        moe_rows,
        combine,
        ctx.gpu,
        stream,
    )?;
    mark("moe_hc_inject", stream);
    Ok(())
}
