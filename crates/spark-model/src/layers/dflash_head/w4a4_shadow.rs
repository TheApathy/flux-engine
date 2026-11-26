// SPDX-License-Identifier: AGPL-3.0-only

//! Default-off DFlash2 W4A4 execution using the established row-major M64 ABI.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use crate::device_allocations::DeviceAllocations;
use crate::layers::ops;
use crate::weight_map::QuantizedWeight;

const EXACT_ROWS: u32 = 16;

pub(super) struct DflashW4a4Shadow {
    quantize: KernelHandle,
    gemm_m64: KernelHandle,
    activation_packed: DevicePtr,
    activation_scales: DevicePtr,
    max_k: u32,
}

impl DflashW4a4Shadow {
    pub(super) fn new(gpu: &dyn GpuBackend, rows: usize, max_k: usize) -> Result<Self> {
        ensure!(
            rows == EXACT_ROWS as usize,
            "DFlash2 W4A4 shadow requires exactly {EXACT_ROWS} query rows"
        );
        ensure!(
            max_k > 0 && max_k.is_multiple_of(16),
            "DFlash2 W4A4 shadow max K must be a positive multiple of 16"
        );
        let packed_bytes = rows
            .checked_mul(max_k / 2)
            .context("DFlash2 W4A4 packed scratch extent overflow")?;
        let scale_bytes = rows
            .checked_mul(max_k / 16)
            .context("DFlash2 W4A4 scale scratch extent overflow")?;
        let max_k = u32::try_from(max_k).context("DFlash2 W4A4 max K exceeds u32")?;
        let quantize = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
        let gemm_m64 = gpu.kernel("nvfp4_cutlass", "nvfp4_nvfp4_gemm_t_m64")?;
        let mut owned = DeviceAllocations::new(gpu);
        let activation_packed = owned.alloc(packed_bytes)?;
        let activation_scales = owned.alloc(scale_bytes)?;
        owned.commit();
        Ok(Self {
            quantize,
            gemm_m64,
            activation_packed,
            activation_scales,
            max_k,
        })
    }

    pub(super) fn device_ptrs(&self) -> [DevicePtr; 2] {
        [self.activation_packed, self.activation_scales]
    }

    pub(super) fn quantize(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        input_scale: f32,
        rows: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        self.validate_activation(input, input_scale, rows, k)?;
        ops::quantize_bf16_to_nvfp4(
            gpu,
            self.quantize,
            input,
            self.activation_packed,
            self.activation_scales,
            input_scale,
            rows,
            k,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn project_prepared(
        &self,
        gpu: &dyn GpuBackend,
        input_scale: f32,
        weight: &QuantizedWeight,
        output: DevicePtr,
        rows: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        self.validate_activation(self.activation_packed, input_scale, rows, k)?;
        ensure!(
            n > 0 && n.is_multiple_of(8),
            "DFlash2 W4A4 projection N must be a positive multiple of 8"
        );
        ensure!(
            !weight.weight.is_null() && !weight.weight_scale.is_null() && !output.is_null(),
            "DFlash2 W4A4 projection received a null operand"
        );
        ensure!(
            weight.weight_scale_2.is_finite() && weight.weight_scale_2 > 0.0,
            "DFlash2 W4A4 weight_scale_2 must be finite and positive"
        );
        let scale2_ab = input_scale * weight.weight_scale_2;
        ensure!(
            scale2_ab.is_finite() && scale2_ab > 0.0,
            "DFlash2 W4A4 combined scale must be finite and positive"
        );
        static SEEN: std::sync::Once = std::sync::Once::new();
        SEEN.call_once(|| {
            tracing::info!(
                "DFLASH2_W4A4_EXEC: calibrated row-major M64 shadow engaged at M={rows}"
            );
        });
        ops::nvfp4_nvfp4_gemm(
            gpu,
            self.gemm_m64,
            self.activation_packed,
            self.activation_scales,
            weight.weight,
            weight.weight_scale,
            scale2_ab,
            output,
            rows,
            n,
            k,
            stream,
        )
    }

    fn validate_activation(
        &self,
        input: DevicePtr,
        input_scale: f32,
        rows: u32,
        k: u32,
    ) -> Result<()> {
        ensure!(
            rows == EXACT_ROWS,
            "DFlash2 W4A4 shadow admits exactly {EXACT_ROWS} rows; got {rows}"
        );
        ensure!(
            k > 0 && k <= self.max_k && k.is_multiple_of(16),
            "DFlash2 W4A4 K={k} exceeds admitted max {} or is misaligned",
            self.max_k
        );
        ensure!(!input.is_null(), "DFlash2 W4A4 activation is null");
        ensure!(
            input_scale.is_finite() && input_scale > 0.0,
            "DFlash2 W4A4 input scale must be finite and positive"
        );
        Ok(())
    }
}
