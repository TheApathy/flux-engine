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

fn validate_finite_positive_scale(op_name: &str, label: &str, scale: f32) -> Result<()> {
    ensure!(
        scale.is_finite() && scale > 0.0,
        "{op_name}: {label}={scale} must be finite and positive"
    );
    Ok(())
}

fn validate_quantized_weight(op_name: &str, weight: &QuantizedWeight) -> Result<()> {
    ensure!(
        !weight.is_null() && weight.weight_scale != DevicePtr::NULL,
        "{op_name}: base NVFP4 weight or block scale pointer is NULL"
    );
    Ok(())
}

fn validate_dual_plane_weight(op_name: &str, weight: &DualPlaneNvfp4Weight) -> Result<()> {
    ensure!(
        !weight.is_null(),
        "{op_name}: DualPlaneNvfp4Weight has a NULL w_hi, w_lo, or weight_scale pointer"
    );
    Ok(())
}

fn resolve_autoematic_sidecar(
    op_name: &str,
    sidecar: Option<&AutoEmaticProjectionSidecar>,
    mode: AutoEmaticRefinementMode,
    n: u32,
    k: u32,
) -> Result<(DevicePtr, DevicePtr, DevicePtr, u32, u32)> {
    match (mode, sidecar) {
        (AutoEmaticRefinementMode::BaseNvfp4Only, _) | (_, None) => {
            Ok((DevicePtr::NULL, DevicePtr::NULL, DevicePtr::NULL, 0, 0))
        }
        (AutoEmaticRefinementMode::AutoFp6Refined, Some(sc)) => {
            ensure!(
                !sc.is_null(),
                "{op_name}: AutoFp6Refined requested with null or empty AutoEmaticProjectionSidecar"
            );
            ensure!(
                sc.n == n as usize && sc.k == k as usize,
                "{op_name}: AutoEMatic sidecar dims [{}x{}] do not match projection [{n}x{k}]",
                sc.n,
                sc.k
            );
            Ok((
                sc.refine_codes,
                sc.refine_mask,
                sc.group_prefix,
                sc.num_refined_sg,
                mode.as_kernel_flag(),
            ))
        }
    }
}

// ── AutoEMatic AutoFP6 dispatch ─────────────────────────────────────────────

/// Launch `autoematic_fp6_gemm_sm120`: Extended-K native Blackwell FP4 TC GEMM.
///
/// Grid: `(ceil(N/64), ceil(M/64), 1)`  Block: `(128, 1, 1)`
///
/// # Errors
/// Returns an error if `K % 64 != 0`, dimensions are zero, `scale2_ab` is
/// non-finite/non-positive, base/sidecar pointers are null, sidecar dims
/// disagree with `[n, k]`, or CUDA launch fails.
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
    validate_quantized_weight("autoematic_fp6_gemm_sm120", weight)?;
    validate_finite_positive_scale("autoematic_fp6_gemm_sm120", "scale2_ab", scale2_ab)?;
    let (codes, mask, prefix, num_sg, flag) =
        resolve_autoematic_sidecar("autoematic_fp6_gemm_sm120", sidecar, mode, n, k)?;
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
/// Returns an error if `M` is outside `1..=8`, `K % 16 != 0`, `weight_scale_2`
/// is non-finite/non-positive, base/sidecar pointers are null, sidecar dims
/// disagree with `[n, k]`, or CUDA launch fails.
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
    validate_quantized_weight("autoematic_fp6_gemv_batch", weight)?;
    validate_finite_positive_scale(
        "autoematic_fp6_gemv_batch",
        "weight_scale_2",
        weight.weight_scale_2,
    )?;
    let (codes, mask, prefix, num_sg, flag) =
        resolve_autoematic_sidecar("autoematic_fp6_gemv_batch", sidecar, mode, n, k)?;
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
/// Returns an error if `K % 16 != 0`, `dest` has null buffers, `scale2 <= 0`, or CUDA launch fails.
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
    validate_dual_plane_weight("repack_bf16_to_dual_plane_nvfp4", dest)?;
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
/// Returns an error if `K % 16 != 0`, `dest` has null buffers, `scale2 <= 0`, or CUDA launch fails.
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
    validate_dual_plane_weight("repack_fp8_blockscaled_to_dual_plane_nvfp4", dest)?;
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
/// Returns an error if `K % 64 != 0`, dimensions are zero, `weight` has null
/// buffers, `scale2_ab` is non-finite/non-positive, or CUDA launch fails.
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
    validate_dual_plane_weight("dual_plane_nvfp4_verify_gemm_sm120", weight)?;
    validate_finite_positive_scale("dual_plane_nvfp4_verify_gemm_sm120", "scale2_ab", scale2_ab)?;
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
/// Returns an error if `M` is outside `1..=8`, `K % 16 != 0`, `weight` has null
/// buffers, `weight_scale_2` is non-finite/non-positive, or CUDA launch fails.
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
    validate_dual_plane_weight("dual_plane_nvfp4_verify_gemv_batch", weight)?;
    validate_finite_positive_scale(
        "dual_plane_nvfp4_verify_gemv_batch",
        "weight_scale_2",
        weight.weight_scale_2,
    )?;
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

        assert!(validate_finite_positive_scale("test", "scale2_ab", 0.25).is_ok());
        assert!(validate_finite_positive_scale("test", "scale2_ab", f32::NAN).is_err());
        assert!(validate_finite_positive_scale("test", "scale2_ab", f32::INFINITY).is_err());
        assert!(validate_finite_positive_scale("test", "scale2_ab", 0.0).is_err());
    }

    #[test]
    fn sidecar_and_weight_bounds_fail_closed() {
        let valid_sidecar = AutoEmaticProjectionSidecar {
            refine_codes: DevicePtr(0x1000),
            refine_mask: DevicePtr(0x2000),
            group_prefix: DevicePtr(0x3000),
            num_refined_sg: 4,
            rho_actual: 0.25,
            refinement_scale: 0.25,
            n: 64,
            k: 64,
        };
        assert!(!valid_sidecar.is_null());
        let resolved = resolve_autoematic_sidecar(
            "test",
            Some(&valid_sidecar),
            AutoEmaticRefinementMode::AutoFp6Refined,
            64,
            64,
        )
        .unwrap();
        assert_eq!(resolved.3, 4);
        assert_eq!(resolved.4, 1);

        // Mismatched sidecar dims must fail closed before launch.
        assert!(
            resolve_autoematic_sidecar(
                "test",
                Some(&valid_sidecar),
                AutoEmaticRefinementMode::AutoFp6Refined,
                128,
                64,
            )
            .is_err()
        );
        assert!(
            resolve_autoematic_sidecar(
                "test",
                Some(&valid_sidecar),
                AutoEmaticRefinementMode::AutoFp6Refined,
                64,
                128,
            )
            .is_err()
        );

        // Partial null pointers or zero refined subgroups must fail closed.
        let mut missing_mask = valid_sidecar;
        missing_mask.refine_mask = DevicePtr::NULL;
        assert!(missing_mask.is_null());
        assert!(
            resolve_autoematic_sidecar(
                "test",
                Some(&missing_mask),
                AutoEmaticRefinementMode::AutoFp6Refined,
                64,
                64,
            )
            .is_err()
        );

        let mut zero_sg = valid_sidecar;
        zero_sg.num_refined_sg = 0;
        assert!(zero_sg.is_null());
        assert!(
            resolve_autoematic_sidecar(
                "test",
                Some(&zero_sg),
                AutoEmaticRefinementMode::AutoFp6Refined,
                64,
                64,
            )
            .is_err()
        );

        // DualPlaneNvfp4Weight requires w_hi, w_lo, and weight_scale all non-null.
        let mut dp = DualPlaneNvfp4Weight {
            w_hi: DevicePtr(0x1000),
            w_lo: DevicePtr(0x2000),
            weight_scale: DevicePtr(0x3000),
            weight_scale_2: 0.5,
            input_scale: DevicePtr::NULL,
        };
        assert!(validate_dual_plane_weight("test", &dp).is_ok());
        dp.weight_scale = DevicePtr::NULL;
        assert!(dp.is_null());
        assert!(validate_dual_plane_weight("test", &dp).is_err());
    }
}
