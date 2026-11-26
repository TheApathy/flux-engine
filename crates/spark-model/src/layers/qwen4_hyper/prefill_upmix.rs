// SPDX-License-Identifier: AGPL-3.0-only

//! Up projection with the hyperconnection mix in its epilogue
//! (`ATLAS_QWEN4_PREFILL_MHC_FUSED=upmix`, default off): one tensor-core
//! kernel (`qwen4_hc_upmix`) replaces the cuBLASLt up GEMM and
//! `qwen4_hc_mix` on the full-row fast HC path, so the 10240-wide mix logits
//! are never written to DRAM and read back.

use std::sync::OnceLock;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::Qwen4HyperConnection;
use super::prefill_inject::{SELECTOR, selection};
use crate::layers::qwen4_fast_proj;

const RANK: usize = 320;
const HIDDEN: usize = 2560;
const STREAMS: usize = 4;
const TILE_H: u32 = 32;
const TILE_ROWS: u32 = 64;
const THREADS: u32 = 256;

/// `qwen4_hc_upmix_hcfused`: HC_FUSED's up+mix (same arguments as its
/// `qwen4_hc_up_mix_mma`, byte-identical output) on this kernel's tiling.
pub(super) fn hcfused_kernel(gpu: &dyn GpuBackend) -> Result<KernelHandle> {
    static K: OnceLock<Result<KernelHandle, String>> = OnceLock::new();
    K.get_or_init(|| {
        gpu.kernel("qwen4_hc_upmix", "qwen4_hc_upmix_hcfused")
            .map_err(|e| e.to_string())
    })
    .clone()
    .map_err(|e| anyhow::anyhow!("{SELECTOR}: upmix_hcfused kernel unavailable: {e}"))
}

/// [`hcfused_kernel`] that also reduces the absmax of its output for the
/// FP8 projection reading it (`ATLAS_QWEN4_PREFILL_FP8_ACT_AMAX_FUSED`).
pub(super) fn hcfused_amax_kernel(gpu: &dyn GpuBackend) -> Result<KernelHandle> {
    static K: OnceLock<Result<KernelHandle, String>> = OnceLock::new();
    K.get_or_init(|| {
        gpu.kernel("qwen4_hc_upmix", "qwen4_hc_upmix_hcfused_amax")
            .map_err(|e| e.to_string())
    })
    .clone()
    .map_err(|e| anyhow::anyhow!("{SELECTOR}: upmix_hcfused_amax kernel unavailable: {e}"))
}

fn upmix_kernel(gpu: &dyn GpuBackend) -> Result<KernelHandle> {
    static K: OnceLock<Result<KernelHandle, String>> = OnceLock::new();
    K.get_or_init(|| {
        gpu.kernel("qwen4_hc_upmix", "qwen4_hc_upmix")
            .map_err(|e| e.to_string())
    })
    .clone()
    .map_err(|e| anyhow::anyhow!("{SELECTOR}: upmix kernel unavailable: {e}"))
}

impl Qwen4HyperConnection {
    /// The lever is selected and the canonical geometry / alignment holds
    /// (`act` rows feed 16-byte cp.async; `normed`/`mixed` bf16x2 accesses).
    pub(super) fn upmix_admitted(
        &self,
        act: DevicePtr,
        normed: DevicePtr,
        mixed: DevicePtr,
    ) -> Result<bool> {
        Ok(selection()?.upmix
            && (self.hidden_size, self.hc_count, self.rank) == (HIDDEN, STREAMS, RANK)
            && act.0.is_multiple_of(16)
            && normed.0.is_multiple_of(4)
            && mixed.0.is_multiple_of(4)
            && !act.is_null()
            && !normed.is_null()
            && !mixed.is_null())
    }

    /// `mixed = mix(up(act), normed)` and the injection-scale transform of
    /// `inject` in place, exactly as the up GEMM + `qwen4_hc_mix` pair.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn upmix_prefill(
        &self,
        act: DevicePtr,
        normed: DevicePtr,
        inject: DevicePtr,
        mixed: DevicePtr,
        num_tokens: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let rows = u32::try_from(num_tokens)?;
        let kernel = upmix_kernel(gpu)?;
        let (up, alpha) = qwen4_fast_proj::dequant_to_scratch(
            gpu,
            &self.up,
            self.residual_width(),
            self.rank,
            stream,
        )?;
        KernelLaunch::new(gpu, kernel)
            .grid([
                self.hidden_size as u32 / TILE_H,
                div_ceil(rows, TILE_ROWS),
                1,
            ])
            .block([THREADS, 1, 1])
            .arg_ptr(act)
            .arg_ptr(up)
            .arg_f32(alpha)
            .arg_ptr(normed)
            .arg_ptr(inject)
            .arg_ptr(mixed)
            .arg_u32(rows)
            .arg_u32(self.hidden_size as u32)
            .arg_u32(self.hc_count as u32)
            .launch(stream)?;
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!("MHC_FUSED_UPMIX_ENGAGED selector={SELECTOR} rows={num_tokens}");
        });
        Ok(())
    }
}
