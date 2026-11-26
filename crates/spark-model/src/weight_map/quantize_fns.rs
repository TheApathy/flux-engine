// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `weight_map.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::*;

/// Quantize a BF16 dense weight to FP8 E4M3 on GPU.
///
/// Allocates FP8 weight buffer + per-row scale buffer, runs GPU quantization
/// kernel. Called once at model load time (not on the hot path).
pub fn quantize_to_fp8(
    bf16_weight: &DenseWeight,
    n: usize,
    k: usize,
    gpu: &dyn GpuBackend,
    quantize_kernel: spark_runtime::gpu::KernelHandle,
    stream: u64,
) -> Result<Fp8DenseWeight> {
    use spark_runtime::kernel_args::KernelLaunch;

    // Allocate FP8 weight buffer: N * K bytes
    let fp8_buf = gpu.alloc(n * k)?;
    // Allocate per-row scale buffer: N * 4 bytes (f32)
    let scale_buf = gpu.alloc(n * 4)?;

    // Launch quantization kernel: Grid=(N), Block=(256)
    KernelLaunch::new(gpu, quantize_kernel)
        .grid([n as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(bf16_weight.weight)
        .arg_ptr(fp8_buf)
        .arg_ptr(scale_buf)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(stream)?;

    // Synchronize to ensure quantization completes before using the buffers
    gpu.synchronize(stream)?;

    Ok(Fp8DenseWeight {
        weight: fp8_buf,
        row_scale: scale_buf,
    })
}
