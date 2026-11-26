// SPDX-License-Identifier: AGPL-3.0-only

//! Phase 8a unified-layout decode dispatch — hoisted from `forward.rs`
//! to keep that file under the 500 LoC cap.
//!
//! Single helper `dispatch_unified_t_decode` runs the gate+up and silu+down
//! kernels against transposed expert weight tables (gate_t / up_t / down_t
//! plus shared_*_t). Mirrors the inline `else if self.use_t_layout_for_decode()`
//! branch 1:1.

use anyhow::Result;

use super::*;

/// `ATLAS_MOE_T_LANES=1` routes unified-layout decode through the
/// lane-parallel `_t_lanes` kernels, which match the untransposed decode
/// kernels bit for bit (the default `_t` kernels do not).
pub(super) fn t_lanes_selected() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("ATLAS_MOE_T_LANES").ok().as_deref() == Some("1");
        if on {
            tracing::info!("ENGAGED ATLAS_MOE_T_LANES: exact lane-parallel transposed decode MoE");
        }
        on
    })
}

/// `ATLAS_MOE_T_LANES_GATE_UP_T8`: eight-threads-per-group M=1 gate/up on
/// the unified layout (byte-identical to `moe_expert_gate_up_shared_t_lanes_o4t16u1`).
///
/// Explicit `"1"` enables and `"0"` disables; when unset, implicitly turns ON
/// whenever `ATLAS_MOE_T_LANES_DOWN_T8=1` or `ATLAS_QWEN4_VERIFY_FAST_HOST=1`.
fn moe_t_lanes_gate_up_t8_selected() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        match std::env::var("ATLAS_MOE_T_LANES_GATE_UP_T8")
            .ok()
            .as_deref()
        {
            Some("1") => true,
            Some("0") => false,
            _ => moe_t_lanes_down_t8_selected() || crate::layers::qwen4_verify_fast_host_selected(),
        }
    })
}

static MOE_T_LANES_GATE_UP_T8_LAUNCHED: std::sync::Once = std::sync::Once::new();

/// `ATLAS_MOE_T_LANES_DOWN_T8`: eight-threads-per-group M=1 down on the
/// unified layout (byte-identical to `moe_expert_silu_down_shared_t_lanes_o4t16u1`).
///
/// Explicit `"1"` enables and `"0"` disables; when unset, implicitly turns ON
/// whenever `ATLAS_QWEN4_VERIFY_FAST_HOST=1`.
fn moe_t_lanes_down_t8_selected() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(
        || match std::env::var("ATLAS_MOE_T_LANES_DOWN_T8").ok().as_deref() {
            Some("1") => true,
            Some("0") => false,
            _ => crate::layers::qwen4_verify_fast_host_selected(),
        },
    )
}

static MOE_T_LANES_DOWN_T8_LAUNCHED: std::sync::Once = std::sync::Once::new();
static MOE_T_LANES_TS4E_DECODE_LAUNCHED: std::sync::Once = std::sync::Once::new();

fn ts4e_t8_decode_kernels(gpu: &dyn GpuBackend) -> Result<(KernelHandle, KernelHandle)> {
    static KERNELS: std::sync::OnceLock<std::result::Result<(KernelHandle, KernelHandle), String>> =
        std::sync::OnceLock::new();
    KERNELS
        .get_or_init(|| {
            let module = "moe_shared_expert_fused_t";
            let gu = gpu
                .kernel(module, "moe_expert_gate_up_shared_t_lanes_t8_k160_ts4e")
                .map_err(|e| {
                    format!("{module}::moe_expert_gate_up_shared_t_lanes_t8_k160_ts4e: {e}")
                })?;
            let dn = gpu
                .kernel(module, "moe_expert_silu_down_shared_t_lanes_t8_k40_ts4e")
                .map_err(|e| {
                    format!("{module}::moe_expert_silu_down_shared_t_lanes_t8_k40_ts4e: {e}")
                })?;
            Ok((gu, dn))
        })
        .clone()
        .map_err(|e| anyhow::anyhow!("ATLAS_MOE_SCALE_TS4E M=1 decode kernel load failed: {e}"))
}

impl MoeLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dispatch_unified_t_decode(
        &self,
        ctx: &ForwardContext,
        expert_input: DevicePtr,
        expert_gate_out: DevicePtr,
        expert_up_out: DevicePtr,
        expert_down_out: DevicePtr,
        shared_gate_scratch: DevicePtr,
        shared_up_scratch: DevicePtr,
        shared_out: DevicePtr,
        indices_dev: DevicePtr,
        h: u32,
        inter: u32,
        top_k: u32,
        stream: u64,
    ) -> Result<()> {
        // Unified-layout decode path: transposed weight tables for all three
        // projections. Fires when `unified_layout` is enabled (defaults ON for
        // Qwen4 via `config.is_qwen4_exp()`, opt-in via `ATLAS_UNIFIED_MOE_LAYOUT=1`
        // elsewhere; override with `=0`) AND the weight loader has built
        // persistent transposed copies for gate / up / down (no lazy-scratch path).
        let gate_t = self
            .gate_ptrs_t
            .as_ref()
            .expect("gate_ptrs_t under unified_t");
        let up_t = self.up_ptrs_t.as_ref().expect("up_ptrs_t under unified_t");
        let down_t = self
            .down_ptrs_t
            .as_ref()
            .expect("down_ptrs_t under unified_t");
        let null_qw = QuantizedWeight::null();
        // ATLAS_QWEN4_DENSE_FP8=1 decode: null shared weights zero the fused
        // shared slots; the FP8 arm below writes `shared_out` instead.
        let shared_fp8 = self.qwen4_shared_fp8_active();
        let shared_nvfp4 = |w: &Option<QuantizedWeight>| match shared_fp8 {
            Some(_) => null_qw,
            None => w.unwrap_or(null_qw),
        };
        let sh_gate_t = &shared_nvfp4(&self.shared_gate_t);
        let sh_up_t = &shared_nvfp4(&self.shared_up_t);
        let sh_down_t = &shared_nvfp4(&self.shared_down_t);
        let lanes = t_lanes_selected();
        if lanes {
            anyhow::ensure!(
                self.moe_expert_gate_up_shared_t_lanes_k.0 != 0
                    && self.moe_expert_silu_down_shared_t_lanes_k.0 != 0,
                "ATLAS_MOE_T_LANES=1 but this target's moe_shared_expert_fused_t module has no _t_lanes kernels"
            );
        }
        let (gate_up_k, gu_out_block, gu_threads, silu_down_k, dn_out_block, dn_threads) = if self
            .routed_scale_layout
            == RoutedScaleLayout::Ts4e128
        {
            anyhow::ensure!(
                h == ops::T_LANES_GATE_UP_T8_K && inter == ops::T_LANES_T8_K,
                "Atlas-TS4E-128 M=1 decode requires H={} and inter={} (got H={h}, inter={inter}); set ATLAS_MOE_SCALE_TS4E=0",
                ops::T_LANES_GATE_UP_T8_K,
                ops::T_LANES_T8_K,
            );
            let (gu_k, dn_k) = ts4e_t8_decode_kernels(ctx.gpu)?;
            anyhow::ensure!(
                gu_k.0 != 0 && dn_k.0 != 0,
                "Atlas-TS4E-128 M=1 decode kernels are absent from moe_shared_expert_fused_t; set ATLAS_MOE_SCALE_TS4E=0"
            );
            MOE_T_LANES_TS4E_DECODE_LAUNCHED.call_once(|| {
                    tracing::info!(
                        "ENGAGED ATLAS_TS4E128_M1_DECODE: t8_k160_ts4e + t8_k40_ts4e launched (H={h}, inter={inter})"
                    );
                });
            (
                gu_k,
                ops::T_LANES_T8_OUT_BLOCK,
                ops::T_LANES_T8_THREADS,
                dn_k,
                ops::T_LANES_T8_OUT_BLOCK,
                ops::T_LANES_T8_THREADS,
            )
        } else if lanes {
            let gu_t8_k = self.moe_expert_gate_up_shared_t_lanes_t8_k;
            let (gu_k, gu_ob, gu_th) = if moe_t_lanes_gate_up_t8_selected()
                && gu_t8_k.0 != 0
                && h == ops::T_LANES_GATE_UP_T8_K
            {
                MOE_T_LANES_GATE_UP_T8_LAUNCHED.call_once(|| {
                    tracing::info!(
                        "ENGAGED ATLAS_MOE_T_LANES_GATE_UP_T8: t8 M=1 gate_up launched (k={h})"
                    );
                });
                (gu_t8_k, ops::T_LANES_T8_OUT_BLOCK, ops::T_LANES_T8_THREADS)
            } else {
                (
                    self.moe_expert_gate_up_shared_t_lanes_k,
                    ops::T_LANES_OUT_BLOCK,
                    ops::T_LANES_THREADS,
                )
            };
            let dn_t8_k = self.moe_expert_silu_down_shared_t_lanes_t8_k;
            let (dn_k, dn_ob, dn_th) =
                if moe_t_lanes_down_t8_selected() && dn_t8_k.0 != 0 && inter == ops::T_LANES_T8_K {
                    MOE_T_LANES_DOWN_T8_LAUNCHED.call_once(|| {
                        tracing::info!(
                            "ENGAGED ATLAS_MOE_T_LANES_DOWN_T8: t8 M=1 down launched (k={inter})"
                        );
                    });
                    (dn_t8_k, ops::T_LANES_T8_OUT_BLOCK, ops::T_LANES_T8_THREADS)
                } else {
                    (
                        self.moe_expert_silu_down_shared_t_lanes_k,
                        ops::T_LANES_OUT_BLOCK,
                        ops::T_LANES_THREADS,
                    )
                };
            (gu_k, gu_ob, gu_th, dn_k, dn_ob, dn_th)
        } else {
            (
                self.moe_expert_gate_up_shared_t_k,
                ops::T_BLOCK,
                ops::T_BLOCK,
                self.moe_expert_silu_down_shared_t_k,
                ops::T_BLOCK,
                ops::T_BLOCK,
            )
        };
        ops::moe_expert_gate_up_shared_t_shape(
            ctx.gpu,
            gate_up_k,
            gu_out_block,
            gu_threads,
            expert_input,
            gate_t.packed_ptrs,
            gate_t.scale_ptrs,
            gate_t.scale2_vals,
            expert_gate_out,
            up_t.packed_ptrs,
            up_t.scale_ptrs,
            up_t.scale2_vals,
            expert_up_out,
            indices_dev,
            sh_gate_t,
            shared_gate_scratch,
            sh_up_t,
            shared_up_scratch,
            inter,
            h,
            top_k,
            stream,
        )?;
        ops::moe_expert_silu_down_shared_t_shape(
            ctx.gpu,
            silu_down_k,
            dn_out_block,
            dn_threads,
            expert_gate_out,
            expert_up_out,
            down_t.packed_ptrs,
            down_t.scale_ptrs,
            down_t.scale2_vals,
            expert_down_out,
            indices_dev,
            shared_gate_scratch,
            shared_up_scratch,
            sh_down_t,
            shared_out,
            h,
            inter,
            top_k,
            stream,
        )?;
        // `ATLAS_QWEN4_SHARED_BF16=1`: the fused kernels above write ZERO to
        // `shared_out` whenever the shared pointer table is null
        // (`moe_shared_expert_fused_t.cu:77-80,87-90,188-192`,
        // lanes variants `:411-413,:492-494`), so the BF16 arm must run
        // AFTER them, overwriting that zero with the real shared-expert
        // contribution, not before. `shared_bf16` no longer implies a null
        // `shared_expert` (see `qwen4_shared_bf16` module doc, C2 fix) —
        // `shared_gate_t`/`shared_up_t`/`shared_down_t` are built normally
        // from the resident NVFP4 copy and the fused ops above compute (and
        // discard) that NVFP4 result; this overwrite is what actually
        // selects BF16 for decode.
        if let Some(sh) = self.weights.shared_bf16.as_ref() {
            let shared_inter = ctx.config.shared_expert_intermediate_size as u32;
            self.shared_decode_bf16(
                sh,
                ctx,
                expert_input,
                shared_gate_scratch,
                shared_up_scratch,
                shared_out,
                h,
                shared_inter,
                stream,
            )?;
        } else if let Some(fp8) = shared_fp8 {
            let scratch = crate::layers::qwen4_dense_fp8::proj::SharedScratch {
                input: expert_input,
                gate: shared_gate_scratch,
                up: shared_up_scratch,
                out: shared_out,
            };
            fp8.forward(ctx.gpu, scratch, h, stream)?;
        }
        Ok(())
    }
}
