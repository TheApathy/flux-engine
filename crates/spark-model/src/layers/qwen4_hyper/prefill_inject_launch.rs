// SPDX-License-Identifier: AGPL-3.0-only

//! Launchers for the fused pending-injection preparations
//! (`ATLAS_QWEN4_PREFILL_MHC_FUSED`, see [`super::prefill_inject`]).

use std::sync::OnceLock;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::Qwen4HyperConnection;
use super::pending::PendingInject;
use super::prefill_inject::SELECTOR;

const INJECT_NORM_THREADS: u32 = 256;

fn inject_rstd_kernel(gpu: &dyn GpuBackend, combine: bool, gatepre: bool) -> Result<KernelHandle> {
    static PLAIN: OnceLock<Result<KernelHandle, String>> = OnceLock::new();
    static COMBINE: OnceLock<Result<KernelHandle, String>> = OnceLock::new();
    static COMBINE_GATEPRE: OnceLock<Result<KernelHandle, String>> = OnceLock::new();
    let (cell, name) = match (combine, gatepre) {
        (true, true) => (&COMBINE_GATEPRE, "qwen4_hc_combine_inject_rstd_gatepre"),
        (true, false) => (&COMBINE, "qwen4_hc_combine_inject_rstd"),
        (false, _) => (&PLAIN, "qwen4_hc_inject_rstd"),
    };
    cell.get_or_init(|| {
        gpu.kernel("qwen4_hc_inject", name)
            .map_err(|e| e.to_string())
    })
    .clone()
    .map_err(|e| anyhow::anyhow!("{SELECTOR}: {name} unavailable: {e}"))
}

fn combine_mx_inject_rstd_kernel(
    gpu: &dyn GpuBackend,
    corrected: bool,
    gatepre: bool,
) -> Result<KernelHandle> {
    static PLAIN: OnceLock<Result<KernelHandle, String>> = OnceLock::new();
    static PLAIN_GATEPRE: OnceLock<Result<KernelHandle, String>> = OnceLock::new();
    static R2: OnceLock<Result<KernelHandle, String>> = OnceLock::new();
    static R2_GATEPRE: OnceLock<Result<KernelHandle, String>> = OnceLock::new();
    let (cell, name) = match (corrected, gatepre) {
        (true, true) => (&R2_GATEPRE, "qwen4_hc_combine_mx_r2_inject_rstd_gatepre"),
        (true, false) => (&R2, "qwen4_hc_combine_mx_r2_inject_rstd"),
        (false, true) => (&PLAIN_GATEPRE, "qwen4_hc_combine_mx_inject_rstd_gatepre"),
        (false, false) => (&PLAIN, "qwen4_hc_combine_mx_inject_rstd"),
    };
    cell.get_or_init(|| {
        gpu.kernel("qwen4_hc_inject", name)
            .map_err(|e| e.to_string())
    })
    .clone()
    .map_err(|e| anyhow::anyhow!("{SELECTOR}: {name} unavailable: {e}"))
}

fn combine_inject_norm_kernel(gpu: &dyn GpuBackend, gatepre: bool) -> Result<KernelHandle> {
    static K: OnceLock<Result<KernelHandle, String>> = OnceLock::new();
    static K_GATEPRE: OnceLock<Result<KernelHandle, String>> = OnceLock::new();
    let (cell, name) = if gatepre {
        (&K_GATEPRE, "qwen4_hc_combine_inject_norm_gatepre")
    } else {
        (&K, "qwen4_hc_combine_inject_norm")
    };
    cell.get_or_init(|| {
        gpu.kernel("qwen4_hc_inject", name)
            .map_err(|e| e.to_string())
    })
    .clone()
    .map_err(|e| anyhow::anyhow!("{SELECTOR}: combine kernel unavailable: {e}"))
}

fn inject_norm_kernel(gpu: &dyn GpuBackend) -> Result<KernelHandle> {
    static K: OnceLock<Result<KernelHandle, String>> = OnceLock::new();
    K.get_or_init(|| {
        gpu.kernel("qwen4_hc_inject", "qwen4_hc_inject_norm")
            .map_err(|e| e.to_string())
    })
    .clone()
    .map_err(|e| anyhow::anyhow!("{SELECTOR}: inject-norm kernel unavailable: {e}"))
}

impl Qwen4HyperConnection {
    /// `hyper += core * saved_scale` then this hyperconnection's group RMS
    /// norm into `residual`, one launch (`qwen4_hc_inject_norm`).
    pub(crate) fn inject_norm_prefill(
        &self,
        p: &PendingInject,
        gpu: &dyn GpuBackend,
        eps: f32,
        stream: u64,
    ) -> Result<()> {
        if let Some(c) = &p.combine {
            let gatepre = !c.gate_scalars.is_null();
            let launch = KernelLaunch::new(gpu, combine_inject_norm_kernel(gpu, gatepre)?)
                .grid([p.rows as u32, 1, 1])
                .block([INJECT_NORM_THREADS, 1, 1])
                .arg_ptr(p.hyper)
                .arg_ptr(p.residual)
                .arg_ptr(self.norm.weight)
                .arg_ptr(c.expert_out)
                .arg_ptr(c.token_to_perm)
                .arg_ptr(c.topk_weights)
                .arg_ptr(c.shared_out);
            let launch = if gatepre {
                launch.arg_ptr(c.gate_scalars)
            } else {
                launch.arg_ptr(c.gate_input).arg_ptr(c.shared_gate)
            };
            launch
                .arg_u32(p.rows as u32)
                .arg_u32(self.hidden_size as u32)
                .arg_u32(c.topk)
                .arg_f32(eps)
                .launch(stream)?;
            static COMBINED: std::sync::Once = std::sync::Once::new();
            COMBINED.call_once(|| {
                tracing::info!(
                    "MHC_FUSED_COMBINE_ENGAGED selector={SELECTOR} rows={}",
                    p.rows
                );
            });
            return Ok(());
        }
        let kernel = inject_norm_kernel(gpu)?;
        KernelLaunch::new(gpu, kernel)
            .grid([p.rows as u32, 1, 1])
            .block([INJECT_NORM_THREADS, 1, 1])
            .arg_ptr(p.hyper)
            .arg_ptr(p.core)
            .arg_ptr(p.residual)
            .arg_ptr(self.norm.weight)
            .arg_u32(p.rows as u32)
            .arg_u32(self.hidden_size as u32)
            .arg_f32(eps)
            .launch(stream)?;
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                "MHC_FUSED_INJECT_NORM_ENGAGED selector={SELECTOR} rows={}",
                p.rows
            );
        });
        Ok(())
    }
    /// `hyper += core * saved_scale` (forming a deferred MoE output first)
    /// and the HC_FUSED preparation's per-(token, stream) rstd into `rstd`
    /// (f32 `[rows, 4]`), one launch (`qwen4_hc_[combine_]inject_rstd`).
    pub(crate) fn inject_rstd_prefill(
        &self,
        p: &PendingInject,
        rstd: DevicePtr,
        gpu: &dyn GpuBackend,
        eps: f32,
        stream: u64,
    ) -> Result<()> {
        if let Some(c) = p.combine.filter(|c| !c.slot_scales.is_null()) {
            let corrected = !c.slot_corrections.is_null();
            let gatepre = !c.gate_scalars.is_null();
            if corrected {
                crate::model::qwen4_prefill_engagement::preflight_slot_r2(
                    crate::model::qwen4_prefill_engagement::MoeSlotR2Consumer::FusedRstd,
                    p.rows,
                )?;
            }
            let launch =
                KernelLaunch::new(gpu, combine_mx_inject_rstd_kernel(gpu, corrected, gatepre)?)
                    .grid([p.rows as u32, 1, 1])
                    .block([INJECT_NORM_THREADS, 1, 1])
                    .arg_ptr(p.hyper)
                    .arg_ptr(p.residual)
                    .arg_ptr(rstd)
                    .arg_ptr(c.expert_out)
                    .arg_ptr(c.slot_scales);
            let launch = if corrected {
                launch.arg_ptr(c.slot_corrections)
            } else {
                launch
            };
            let launch = launch
                .arg_ptr(c.token_to_perm)
                .arg_ptr(c.topk_weights)
                .arg_ptr(c.shared_out);
            let launch = if gatepre {
                launch.arg_ptr(c.gate_scalars)
            } else {
                launch.arg_ptr(c.gate_input).arg_ptr(c.shared_gate)
            };
            launch
                .arg_u32(p.rows as u32)
                .arg_u32(self.hidden_size as u32)
                .arg_u32(c.topk)
                .arg_f32(eps)
                .launch(stream)?;
            if corrected {
                crate::model::qwen4_prefill_engagement::commit_slot_r2(
                    crate::model::qwen4_prefill_engagement::MoeSlotR2Consumer::FusedRstd,
                );
            }
            static MX: std::sync::Once = std::sync::Once::new();
            MX.call_once(|| {
                tracing::info!(
                    "MHC_FUSED_COMBINE_MX_RSTD_ENGAGED selector={SELECTOR} corrected={} rows={}",
                    corrected,
                    p.rows
                );
            });
            return Ok(());
        }
        if let Some(c) = &p.combine {
            let gatepre = !c.gate_scalars.is_null();
            let launch = KernelLaunch::new(gpu, inject_rstd_kernel(gpu, true, gatepre)?)
                .grid([p.rows as u32, 1, 1])
                .block([INJECT_NORM_THREADS, 1, 1])
                .arg_ptr(p.hyper)
                .arg_ptr(p.residual)
                .arg_ptr(rstd)
                .arg_ptr(c.expert_out)
                .arg_ptr(c.token_to_perm)
                .arg_ptr(c.topk_weights)
                .arg_ptr(c.shared_out);
            let launch = if gatepre {
                launch.arg_ptr(c.gate_scalars)
            } else {
                launch.arg_ptr(c.gate_input).arg_ptr(c.shared_gate)
            };
            launch
                .arg_u32(p.rows as u32)
                .arg_u32(self.hidden_size as u32)
                .arg_u32(c.topk)
                .arg_f32(eps)
                .launch(stream)?;
            static COMBINED: std::sync::Once = std::sync::Once::new();
            COMBINED.call_once(|| {
                tracing::info!(
                    "MHC_FUSED_COMBINE_RSTD_ENGAGED selector={SELECTOR} rows={}",
                    p.rows
                );
            });
            return Ok(());
        }
        KernelLaunch::new(gpu, inject_rstd_kernel(gpu, false, false)?)
            .grid([p.rows as u32, 1, 1])
            .block([INJECT_NORM_THREADS, 1, 1])
            .arg_ptr(p.hyper)
            .arg_ptr(p.core)
            .arg_ptr(p.residual)
            .arg_ptr(rstd)
            .arg_u32(p.rows as u32)
            .arg_u32(self.hidden_size as u32)
            .arg_f32(eps)
            .launch(stream)?;
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                "MHC_FUSED_INJECT_RSTD_ENGAGED selector={SELECTOR} rows={}",
                p.rows
            );
        });
        Ok(())
    }
}
