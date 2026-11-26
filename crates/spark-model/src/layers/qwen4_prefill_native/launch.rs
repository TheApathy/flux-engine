// SPDX-License-Identifier: AGPL-3.0-only

//! The native launches behind the call-site hooks. Each one counts itself in
//! the per-call receipt only after its GEMM was enqueued.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::registry::NativeFp8;
use super::{Module, note_call};
use crate::weight_map::DenseWeight;

/// `out[m, n] (ldc) = a8[m, k] * a_scale @ (w8 * s)^T` on the cuBLASLt
/// SCALAR_32F FP8 GEMM, with the activation already quantized into `a8` and
/// its scale at `ws + 12` (the fast-projection scratch layout). Alpha is 1:
/// the native rows carry their whole scale in the device tensor scale.
#[allow(clippy::too_many_arguments)]
pub(crate) fn native_fp8_gemm(
    gpu: &dyn GpuBackend,
    native: &NativeFp8,
    out: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    ldc: usize,
    ws: DevicePtr,
    a8: DevicePtr,
    stream: u64,
) -> Result<()> {
    let a_scale = ws.offset(12);
    spark_runtime::cublaslt::fp8_gemm_device_scaled(
        a8.0,
        a_scale.0,
        native.w8.0,
        native.scale.0,
        out.0,
        m as u32,
        n as u32,
        k as u32,
        ldc as u32,
        1.0,
        false,
        stream,
    )?;
    crate::layers::qwen4_fp8_proj_dump::maybe_dump(
        gpu,
        a8,
        a_scale,
        native.w8,
        native.scale,
        1.0,
        out,
        (m, n, k, ldc),
        stream,
    )?;
    note_call(
        native.module,
        "fp8_e4m3_mxfp8_pow2_tensor",
        m,
        n,
        k,
        native.scale_value,
    );
    Ok(())
}

/// BF16 twin of the fast-projection dequant GEMM: the resident BF16 weight
/// goes straight to cuBLASLt (alpha 1) instead of a `LUT * fp8` scratch.
#[allow(clippy::too_many_arguments)]
pub(crate) fn bf16_dequant_gemm(
    act: DevicePtr,
    weight: DevicePtr,
    module: Module,
    out: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    ldc: usize,
    tuned: bool,
    stream: u64,
) -> Result<()> {
    let (m32, n32, k32, ldc32) = (m as u32, n as u32, k as u32, ldc as u32);
    if tuned {
        spark_runtime::cublaslt::bf16_gemm_act_weight_t_tuned_ex(
            act.0, weight.0, out.0, m32, n32, k32, ldc32, 1.0, stream,
        )?;
    } else {
        spark_runtime::cublaslt::bf16_gemm_act_weight_t_ldc(
            act.0, weight.0, out.0, m32, n32, k32, ldc32, 1.0, stream,
        )?;
    }
    note_call(module, "bf16_cublaslt", m, n, k, 1.0);
    Ok(())
}

/// Prefill router logits from the resident BF16 gate (`dense_gemm`), the arm
/// `ATLAS_QWEN4_ROUTER_BF16=1` also takes, without removing the NVFP4 copy
/// that decode keeps using.
#[allow(clippy::too_many_arguments)]
pub(crate) fn router_dense_gemm(
    gpu: &dyn GpuBackend,
    dense_gemm: KernelHandle,
    input: DevicePtr,
    gate: &DenseWeight,
    logits: DevicePtr,
    rows: u32,
    experts: u32,
    hidden: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        dense_gemm.0 != 0,
        "{}: router dense_gemm kernel is not loaded",
        super::SELECTOR
    );
    crate::layers::ops::dense_gemm(
        gpu, dense_gemm, input, gate, logits, rows, experts, hidden, stream,
    )?;
    note_call(
        Module::Router,
        "bf16_dense_gemm",
        rows as usize,
        experts as usize,
        hidden as usize,
        1.0,
    );
    Ok(())
}
