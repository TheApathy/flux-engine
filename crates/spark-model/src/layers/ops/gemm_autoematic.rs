// SPDX-License-Identifier: AGPL-3.0-only

//! Kernel dispatch operations for AutoEMatic AutoFP6 (`+0.4 bpw`) and
//! DualPlaneNvfp4 (`8.5 bpw`) Blackwell CUDA kernels.
//!
//! Covers all six entry points compiled from:
//!   - `kernels/gb10/common/autoematic_fp6_gemm.cu`
//!   - `kernels/gb10/common/dual_plane_nvfp4.cu`

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_loader::autoematic_sidecar::AutoEmaticProjectionSidecar;
use crate::weight_map::{DualPlaneNvfp4Weight, QuantizedWeight};

/// Maximum batch rows `M` accepted by the small-M fused GEMV kernels.
pub const AUTOEMATIC_GEMV_MAX_M: u32 = 8;

/// Explicit dispatch mode controlling whether AutoEMatic sparse refinement is
/// active on this layer execution (replaces raw boolean kernel flags).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AutoEmaticRefinementMode {
    /// Skip sparse refinement reads/MMAs (100% native NVFP4 fast path).
    BaseNvfp4Only,
    /// Apply `+0.4 bpw` sparse AutoFP6 refinement (`W_q + 0.25 * W_delta4`).
    AutoFp6Refined,
}

impl AutoEmaticRefinementMode {
    /// Convert to the `unsigned int enable_refine` ABI argument expected by
    /// `autoematic_fp6_gemm_sm120` and `autoematic_fp6_gemv_batch`.
    #[must_use]
    pub const fn as_kernel_flag(self) -> u32 {
        match self {
            Self::BaseNvfp4Only => 0,
            Self::AutoFp6Refined => 1,
        }
    }
}

fn validate_sm120_mma_dims(op_name: &str, m: u32, n: u32, k: u32) -> Result<()> {
    ensure!(
        m > 0 && n > 0 && k > 0,
        "{op_name}: dimensions must be positive (M={m}, N={n}, K={k})"
    );
    ensure!(
        k.is_multiple_of(64),
        "{op_name}: K={k} must be a multiple of 64 for Blackwell m16n8k64 FP4 MMA"
    );
    Ok(())
}

fn validate_small_m_gemv_dims(op_name: &str, m: u32, n: u32, k: u32) -> Result<()> {
    ensure!(
        (1..=AUTOEMATIC_GEMV_MAX_M).contains(&m),
        "{op_name}: M={m} must lie in 1..={AUTOEMATIC_GEMV_MAX_M}"
    );
    ensure!(
        n > 0 && k > 0 && k.is_multiple_of(16),
        "{op_name}: N={n} must be positive and K={k} must be a positive multiple of 16"
    );
    Ok(())
}

fn validate_repack_dims(op_name: &str, n: u32, k: u32, scale2: f32) -> Result<()> {
    ensure!(
        n > 0 && k > 0 && k.is_multiple_of(16),
        "{op_name}: N={n} must be positive and K={k} must be a positive multiple of 16"
    );
    ensure!(
        scale2.is_finite() && scale2 > 0.0,
        "{op_name}: scale2={scale2} must be finite and positive"
    );
    Ok(())
}

// ── AutoEMatic AutoFP6 dispatch ─────────────────────────────────────────────

/// Launch `autoematic_fp6_gemm_sm120`: Extended-K native Blackwell FP4 TC GEMM.
///
/// Grid: `(ceil(N/64), ceil(M/64), 1)`  Block: `(128, 1, 1)`
///
/// # Errors
/// Returns an error if `K % 64 != 0`, dimensions are zero, or CUDA launch fails.
pub fn autoematic_fp6_gemm_sm120(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a_packed: DevicePtr,
    a_scale: DevicePtr,
    weight: &QuantizedWeight,
    sidecar: Option<&AutoEmaticProjectionSidecar>,
    scale2_ab: f32,
    mode: AutoEmaticRefinementMode,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    validate_sm120_mma_dims("autoematic_fp6_gemm_sm120", m, n, k)?;
    let (codes, mask, prefix, num_sg, flag) = match (mode, sidecar) {
        (AutoEmaticRefinementMode::AutoFp6Refined, Some(sc)) if !sc.is_null() => (
            sc.refine_codes,
            sc.refine_mask,
            sc.group_prefix,
            sc.num_refined_sg,
            mode.as_kernel_flag(),
        ),
        _ => (DevicePtr::NULL, DevicePtr::NULL, DevicePtr::NULL, 0, 0),
    };
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 64), div_ceil(m, 64), 1])
        .block([128, 1, 1])
        .arg_ptr(a_packed)
        .arg_ptr(a_scale)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_ptr(codes)
        .arg_ptr(mask)
        .arg_ptr(prefix)
        .arg_u32(num_sg)
        .arg_f32(scale2_ab)
        .arg_u32(flag)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Launch `autoematic_fp6_gemv_batch`: small-M (`1..=8`) BF16-activation AutoFP6 GEMV.
///
/// Grid: `(ceil(N/4), 1, 1)`  Block: `(256, 1, 1)`
///
/// # Errors
/// Returns an error if `M` is outside `1..=8`, `K % 16 != 0`, or CUDA launch fails.
pub fn autoematic_fp6_gemv_batch(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input_bf16: DevicePtr,
    weight: &QuantizedWeight,
    sidecar: Option<&AutoEmaticProjectionSidecar>,
    mode: AutoEmaticRefinementMode,
    output_bf16: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    validate_small_m_gemv_dims("autoematic_fp6_gemv_batch", m, n, k)?;
    let (codes, mask, prefix, num_sg, flag) = match (mode, sidecar) {
        (AutoEmaticRefinementMode::AutoFp6Refined, Some(sc)) if !sc.is_null() => (
            sc.refine_codes,
            sc.refine_mask,
            sc.group_prefix,
            sc.num_refined_sg,
            mode.as_kernel_flag(),
        ),
        _ => (DevicePtr::NULL, DevicePtr::NULL, DevicePtr::NULL, 0, 0),
    };
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input_bf16)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_ptr(codes)
        .arg_ptr(mask)
        .arg_ptr(prefix)
        .arg_u32(num_sg)
        .arg_f32(weight.weight_scale_2)
        .arg_u32(flag)
        .arg_ptr(output_bf16)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

// ── DualPlaneNvfp4 repack & verify dispatch ─────────────────────────────────

/// Launch `repack_bf16_to_dual_plane_nvfp4`: on-the-fly BF16 `[N, K]` -> DualPlaneNvfp4.
///
/// Grid: `(N, 1, 1)`  Block: `(256, 1, 1)`
///
/// # Errors
/// Returns an error if `K % 16 != 0`, `scale2 <= 0`, or CUDA launch fails.
pub fn repack_bf16_to_dual_plane_nvfp4(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input_bf16: DevicePtr,
    dest: &DualPlaneNvfp4Weight,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    validate_repack_dims("repack_bf16_to_dual_plane_nvfp4", n, k, dest.weight_scale_2)?;
    KernelLaunch::new(gpu, kernel)
        .grid([n, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input_bf16)
        .arg_ptr(dest.w_hi)
        .arg_ptr(dest.w_lo)
        .arg_ptr(dest.weight_scale)
        .arg_f32(dest.weight_scale_2)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Launch `repack_fp8_blockscaled_to_dual_plane_nvfp4`: on-the-fly block-128 FP8 -> DualPlaneNvfp4.
///
/// Grid: `(N, 1, 1)`  Block: `(256, 1, 1)`
///
/// # Errors
/// Returns an error if `K % 16 != 0`, `scale2 <= 0`, or CUDA launch fails.
pub fn repack_fp8_blockscaled_to_dual_plane_nvfp4(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    fp8_in: DevicePtr,
    block_scale_bf16: DevicePtr,
    dest: &DualPlaneNvfp4Weight,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    validate_repack_dims(
        "repack_fp8_blockscaled_to_dual_plane_nvfp4",
        n,
        k,
        dest.weight_scale_2,
    )?;
    KernelLaunch::new(gpu, kernel)
        .grid([n, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(fp8_in)
        .arg_ptr(block_scale_bf16)
        .arg_ptr(dest.w_hi)
        .arg_ptr(dest.w_lo)
        .arg_ptr(dest.weight_scale)
        .arg_f32(dest.weight_scale_2)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Launch `dual_plane_nvfp4_verify_gemm_sm120`: Extended-K Blackwell FP4 TC verify GEMM.
///
/// Grid: `(ceil(N/64), ceil(M/64), 1)`  Block: `(128, 1, 1)`
///
/// # Errors
/// Returns an error if `K % 64 != 0`, dimensions are zero, or CUDA launch fails.
pub fn dual_plane_nvfp4_verify_gemm_sm120(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a_packed: DevicePtr,
    a_scale: DevicePtr,
    weight: &DualPlaneNvfp4Weight,
    scale2_ab: f32,
    output_bf16: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    validate_sm120_mma_dims("dual_plane_nvfp4_verify_gemm_sm120", m, n, k)?;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 64), div_ceil(m, 64), 1])
        .block([128, 1, 1])
        .arg_ptr(a_packed)
        .arg_ptr(a_scale)
        .arg_ptr(weight.w_hi)
        .arg_ptr(weight.w_lo)
        .arg_ptr(weight.weight_scale)
        .arg_f32(scale2_ab)
        .arg_ptr(output_bf16)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Launch `dual_plane_nvfp4_verify_gemv_batch`: small-M (`1..=8`) BF16 fused verify GEMV.
///
/// Grid: `(ceil(N/4), 1, 1)`  Block: `(256, 1, 1)`
///
/// # Errors
/// Returns an error if `M` is outside `1..=8`, `K % 16 != 0`, or CUDA launch fails.
pub fn dual_plane_nvfp4_verify_gemv_batch(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input_bf16: DevicePtr,
    weight: &DualPlaneNvfp4Weight,
    output_bf16: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    validate_small_m_gemv_dims("dual_plane_nvfp4_verify_gemv_batch", m, n, k)?;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input_bf16)
        .arg_ptr(weight.w_hi)
        .arg_ptr(weight.w_lo)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output_bf16)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refinement_mode_flags_and_dim_guards() {
        assert_eq!(AutoEmaticRefinementMode::BaseNvfp4Only.as_kernel_flag(), 0);
        assert_eq!(AutoEmaticRefinementMode::AutoFp6Refined.as_kernel_flag(), 1);

        assert!(validate_sm120_mma_dims("test", 16, 64, 64).is_ok());
        assert!(validate_sm120_mma_dims("test", 16, 64, 48).is_err());
        assert!(validate_sm120_mma_dims("test", 0, 64, 64).is_err());

        assert!(validate_small_m_gemv_dims("test", 1, 64, 16).is_ok());
        assert!(validate_small_m_gemv_dims("test", 8, 64, 128).is_ok());
        assert!(validate_small_m_gemv_dims("test", 0, 64, 16).is_err());
        assert!(validate_small_m_gemv_dims("test", 9, 64, 16).is_err());
        assert!(validate_small_m_gemv_dims("test", 4, 64, 24).is_err());

        assert!(validate_repack_dims("test", 64, 16, 0.5).is_ok());
        assert!(validate_repack_dims("test", 64, 16, 0.0).is_err());
        assert!(validate_repack_dims("test", 64, 12, 0.5).is_err());
    }
}
