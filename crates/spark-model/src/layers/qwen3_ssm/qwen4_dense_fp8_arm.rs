// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_QWEN4_DENSE_FP8=1` decode arm of the GDN layer: the qkvz and
//! out_proj GEMVs run on native MXFP8 checkpoint tensors inside the decode scope. Each
//! accessor returns `None` when the arm does not apply, so the existing
//! NVFP4 chain in `ssm_forward.rs` runs unchanged.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::Qwen3SsmLayer;
use crate::layers::qwen4_dense_fp8::proj::GdnFp8;

impl Qwen3SsmLayer {
    /// Install the decode copies. Only the sequential `[Q|K|V|Z]` qkvz
    /// layout (Qwen3.5/Qwen4) writes the GEMV output straight into the
    /// deinterleaved buffer, so any other layout is refused.
    pub(crate) fn set_qwen4_dense_fp8(&mut self, fp8: GdnFp8) -> Result<()> {
        ensure!(
            self.sequential_qkvz,
            "ATLAS_QWEN4_DENSE_FP8=1 needs the sequential [Q|K|V|Z] qkvz layout"
        );
        self.qwen4_dense_fp8 = Some(fp8);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn qwen4_fp8_qkvz(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        output: DevicePtr,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Option<Result<()>> {
        self.qwen4_dense_fp8
            .as_ref()?
            .qkvz(gpu, input, output, n, k, stream)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn qwen4_fp8_out_proj(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        output: DevicePtr,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Option<Result<()>> {
        self.qwen4_dense_fp8
            .as_ref()?
            .out_proj(gpu, input, output, n, k, stream)
    }
}
