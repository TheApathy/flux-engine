// SPDX-License-Identifier: AGPL-3.0-only

//! DualPlaneNvfp4 (`8.5 bpw`) bit-shared weight representation and host mirror.
//!
//! Represents any upstream BF16 or block-128 FP8 E4M3 weight matrix as two
//! native NVFP4 E2M1 bit-shared planes under one shared per-16 UE4M3 group
//! scale and one shared per-tensor FP32 `scale2`:
//!
//! ```text
//! W_hat = (UE4M3(B_scale) * scale2) * (E2M1(W_hi) + 0.125 * E2M1(W_lo))
//! ```
//!
//! - **Draft view (`W_hi`, `4.5 bpw`)**: zero-copy [`QuantizedWeight`] alias
//!   consumed directly by Flux's existing NVFP4 GEMV/GEMM kernels (~20.5 dB
//!   SQNR, cosine similarity ~0.9955).
//! - **Verify view (`W_hi + 0.125 * W_lo`, `8.5 bpw`)**: executes natively on
//!   Blackwell FP4 Tensor Cores via Extended-K MMA (`dual_plane_nvfp4.cu`) at
//!   **~38.5 dB SQNR** vs FP32, beating block-128 FP8 E4M3 (~31.8 dB) by
//!   `+6.7 dB` while requiring zero external draft model weights.
//!
//! Opt-in only via `ATLAS_DUAL_PLANE_NVFP4` (default OFF).

use std::fmt;

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::QuantizedWeight;

/// Environment variable enabling load-time DualPlaneNvfp4 (`8.5 bpw`) repacking.
///
/// Unset or empty keeps the existing NVFP4/FP8 path untouched (default OFF).
pub const DUAL_PLANE_NVFP4_ENV: &str = "ATLAS_DUAL_PLANE_NVFP4";

/// Group size shared by both E2M1 planes (elements per UE4M3 scale byte).
pub const DUAL_PLANE_GROUP_SIZE: usize = 16;

/// Residual plane scale factor (`1/8 = 0.125`), matching `dual_plane_nvfp4.cu`.
pub const DUAL_PLANE_LO_SCALE: f32 = 0.125;

/// Total storage bits per weight (`4.0 W_hi + 4.0 W_lo + 0.5 shared UE4M3 scale`).
pub const DUAL_PLANE_BPW: f32 = 8.5;

/// Draft-pass bits per weight (`4.0 W_hi + 0.5 shared UE4M3 scale`).
pub const DUAL_PLANE_DRAFT_BPW: f32 = 4.5;

/// Standard E2M1 lookup table matching `DP_E2M1_LUT` in `dual_plane_nvfp4.cu`.
pub const DUAL_PLANE_E2M1_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Error returned when parsing `ATLAS_DUAL_PLANE_NVFP4`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DualPlaneConfigError {
    raw: String,
}

impl fmt::Display for DualPlaneConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{DUAL_PLANE_NVFP4_ENV} must be one of 0/1/false/true/off/on, got {:?}",
            self.raw
        )
    }
}

impl std::error::Error for DualPlaneConfigError {}

/// Load-time policy controlling DualPlaneNvfp4 repacking.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DualPlaneLoadPolicy {
    /// Disabled (default): baseline NVFP4/FP8 weight loading is unchanged.
    #[default]
    Disabled,
    /// Enabled: repack eligible BF16/FP8 matrices into shared-scale `W_hi + W_lo`.
    Enabled,
}

impl DualPlaneLoadPolicy {
    /// Read `ATLAS_DUAL_PLANE_NVFP4` from the process environment.
    ///
    /// Returns [`Self::Disabled`] when unset or empty.
    ///
    /// # Errors
    /// Fails closed on unrecognized non-empty strings.
    pub fn from_env() -> std::result::Result<Self, DualPlaneConfigError> {
        let raw = std::env::var(DUAL_PLANE_NVFP4_ENV).ok();
        Self::parse_optional(raw.as_deref())
    }

    /// Parse an optional policy value.
    ///
    /// # Errors
    /// Returns [`DualPlaneConfigError`] on unrecognized non-empty tokens.
    pub fn parse_optional(raw: Option<&str>) -> std::result::Result<Self, DualPlaneConfigError> {
        let Some(trimmed) = raw.map(str::trim).filter(|s| !s.is_empty()) else {
            return Ok(Self::Disabled);
        };
        match trimmed.to_ascii_lowercase().as_str() {
            "0" | "false" | "off" => Ok(Self::Disabled),
            "1" | "true" | "on" => Ok(Self::Enabled),
            _ => Err(DualPlaneConfigError {
                raw: trimmed.to_owned(),
            }),
        }
    }
}

/// Device-resident DualPlaneNvfp4 (`8.5 bpw`) weight matrix.
#[derive(Clone, Copy, Debug)]
pub struct DualPlaneNvfp4Weight {
    /// Coarse E2M1 MSB plane `[N, K/2]` (4.0 bpw).
    pub w_hi: DevicePtr,
    /// Residual E2M1 LSB plane `[N, K/2]` scaled by `0.125` (4.0 bpw).
    pub w_lo: DevicePtr,
    /// Shared per-16 UE4M3 block scales `[N, K/16]` (0.5 bpw).
    pub weight_scale: DevicePtr,
    /// Shared per-tensor FP32 `scale2`.
    pub weight_scale_2: f32,
    /// Optional activation scale pointer (NULL when dynamic quantization is used).
    pub input_scale: DevicePtr,
}

impl DualPlaneNvfp4Weight {
    /// Null placeholder weight.
    #[must_use]
    pub const fn null() -> Self {
        Self {
            w_hi: DevicePtr::NULL,
            w_lo: DevicePtr::NULL,
            weight_scale: DevicePtr::NULL,
            weight_scale_2: 0.0,
            input_scale: DevicePtr::NULL,
        }
    }

    /// Whether this weight holds null pointers.
    #[must_use]
    pub fn is_null(&self) -> bool {
        self.w_hi == DevicePtr::NULL || self.w_lo == DevicePtr::NULL
    }

    /// Zero-copy `4.5 bpw` draft view aliasing `w_hi` and the shared scales.
    #[must_use]
    pub const fn draft_view(&self) -> QuantizedWeight {
        QuantizedWeight {
            weight: self.w_hi,
            weight_scale: self.weight_scale,
            weight_scale_2: self.weight_scale_2,
            input_scale: self.input_scale,
        }
    }

    /// Exact device byte footprint (`8.5 bpw`) for an `[n, k]` weight matrix.
    ///
    /// Returns `None` if `k` is not divisible by 16 or on arithmetic overflow.
    #[must_use]
    pub const fn storage_bytes(n: usize, k: usize) -> Option<usize> {
        if !k.is_multiple_of(DUAL_PLANE_GROUP_SIZE) {
            return None;
        }
        let plane_bytes = match n.checked_mul(k / 2) {
            Some(v) => v,
            None => return None,
        };
        let scale_bytes = match n.checked_mul(k / DUAL_PLANE_GROUP_SIZE) {
            Some(v) => v,
            None => return None,
        };
        match plane_bytes.checked_mul(2) {
            Some(both_planes) => both_planes.checked_add(scale_bytes),
            None => None,
        }
    }
}

// ── Host-side bit-exact quantizer & SQNR mirror (matches dual_plane_nvfp4.cu) ─

/// Branchless FP32 -> E2M1 nibble matching `dp_float_to_e2m1` in `dual_plane_nvfp4.cu`.
#[must_use]
pub fn float_to_e2m1(x: f32) -> u8 {
    let u = x.to_bits();
    let sign = ((u >> 28) & 8) as u8;
    let abits = u & 0x7FFF_FFFF;
    let mag = u8::from(abits > 0x3E80_0000)
        + u8::from(abits >= 0x3F40_0000)
        + u8::from(abits > 0x3FA0_0000)
        + u8::from(abits >= 0x3FE0_0000)
        + u8::from(abits > 0x4020_0000)
        + u8::from(abits >= 0x4060_0000)
        + u8::from(abits > 0x40A0_0000);
    sign | mag
}

/// Positive FP32 -> UE4M3 scale byte matching `dp_float_to_ue4m3` in `dual_plane_nvfp4.cu`.
#[must_use]
pub fn float_to_ue4m3(v: f32) -> u8 {
    let mut absv = v.abs();
    if absv == 0.0 || !absv.is_finite() {
        return 0;
    }
    if absv > 448.0 {
        absv = 448.0;
    }
    let bits = absv.to_bits();
    let f32_exp = ((bits >> 23) & 0xFF) as i32 - 127;
    let f32_man = bits & 0x7F_FFFF;

    if f32_exp < -9 {
        return 0;
    }
    if f32_exp < -6 {
        let man = (absv * 512.0 + 0.5) as i32;
        return man.clamp(0, 7) as u8;
    }
    let mut e4m3_exp = f32_exp + 7;
    let mut e4m3_man = (f32_man + (1_u32 << 19)) >> 20;
    if e4m3_man > 7 {
        e4m3_man = 0;
        e4m3_exp += 1;
    }
    if e4m3_exp > 15 || (e4m3_exp == 15 && e4m3_man > 6) {
        e4m3_exp = 15;
        e4m3_man = 6;
    }
    if e4m3_exp < 1 {
        e4m3_exp = 1;
    }
    (((e4m3_exp as u32) << 3) | e4m3_man) as u8
}

/// Decode signed FP8 E4M3 byte to FP32 (matching `dp_e4m3_to_f32` in `dual_plane_nvfp4.cu`).
#[must_use]
pub fn e4m3_byte_to_f32(byte: u8) -> f32 {
    let sign = (byte >> 7) & 1;
    let exp = u32::from((byte >> 3) & 0x0F);
    let man = u32::from(byte & 0x07);
    let val = if exp == 0 {
        (man as f32) * 0.001_953_125_f32 // 2^-9
    } else if exp == 15 && man == 7 {
        0.0
    } else {
        f32::from_bits(((exp + 120) << 23) | (man << 20))
    };
    if sign != 0 { -val } else { val }
}

/// Encode signed FP32 to FP8 E4M3fn byte (saturation at +/-448.0).
#[must_use]
pub fn f32_to_signed_e4m3(v: f32) -> u8 {
    let mag = float_to_ue4m3(v);
    if v.is_sign_negative() && mag != 0 {
        mag | 0x80
    } else {
        mag
    }
}

/// Host-side packed DualPlaneNvfp4 matrix for offline verification and unit tests.
#[derive(Clone, Debug)]
pub struct HostDualPlaneMatrix {
    pub w_hi: Vec<u8>,
    pub w_lo: Vec<u8>,
    pub scales: Vec<u8>,
    pub scale2: f32,
    pub n: usize,
    pub k: usize,
}

/// Quantize an `[n, k]` FP32 matrix into shared-scale DualPlaneNvfp4 (`8.5 bpw`).
///
/// # Errors
/// Returns an error if dimensions are zero, `k` is not a multiple of 16, or
/// `weights.len() != n * k`.
pub fn quantize_f32_to_dual_plane_host(
    weights: &[f32],
    n: usize,
    k: usize,
) -> Result<HostDualPlaneMatrix> {
    ensure!(n > 0 && k > 0, "DualPlaneNvfp4 dimensions must be positive");
    ensure!(
        k.is_multiple_of(DUAL_PLANE_GROUP_SIZE),
        "DualPlaneNvfp4 K={k} must be a multiple of {DUAL_PLANE_GROUP_SIZE}"
    );
    ensure!(
        weights.len() == n * k,
        "DualPlaneNvfp4 slice len {} != n*k ({})",
        weights.len(),
        n * k
    );

    let global_max = weights.iter().fold(0.0_f32, |m, &v| m.max(v.abs()));
    let scale2 = if global_max > 0.0 {
        global_max / (6.0 * 448.0)
    } else {
        1.0
    };
    let inv_scale2 = 1.0 / scale2;
    let num_groups = k / DUAL_PLANE_GROUP_SIZE;

    let mut w_hi = vec![0_u8; n * (k / 2)];
    let mut w_lo = vec![0_u8; n * (k / 2)];
    let mut scales = vec![0_u8; n * num_groups];

    for row in 0..n {
        let row_in = &weights[row * k..(row + 1) * k];
        let row_hi = &mut w_hi[row * (k / 2)..(row + 1) * (k / 2)];
        let row_lo = &mut w_lo[row * (k / 2)..(row + 1) * (k / 2)];
        let row_sc = &mut scales[row * num_groups..(row + 1) * num_groups];

        for g in 0..num_groups {
            let base = g * DUAL_PLANE_GROUP_SIZE;
            let group = &row_in[base..base + DUAL_PLANE_GROUP_SIZE];
            let gmax = group.iter().fold(0.0_f32, |m, &v| m.max(v.abs()));
            let sc_byte = float_to_ue4m3(gmax * inv_scale2 * (1.0 / 6.0));
            row_sc[g] = sc_byte;

            let eff = e4m3_byte_to_f32(sc_byte) * scale2;
            let inv_eff = if eff > 0.0 { 1.0 / eff } else { 0.0 };

            for pair in 0..8 {
                let n0 = group[pair * 2] * inv_eff;
                let n1 = group[pair * 2 + 1] * inv_eff;
                let hi0 = float_to_e2m1(n0);
                let hi1 = float_to_e2m1(n1);
                let lo0 = float_to_e2m1(8.0 * (n0 - DUAL_PLANE_E2M1_LUT[hi0 as usize]));
                let lo1 = float_to_e2m1(8.0 * (n1 - DUAL_PLANE_E2M1_LUT[hi1 as usize]));
                row_hi[g * 8 + pair] = (hi1 << 4) | (hi0 & 0x0F);
                row_lo[g * 8 + pair] = (lo1 << 4) | (lo0 & 0x0F);
            }
        }
    }

    Ok(HostDualPlaneMatrix {
        w_hi,
        w_lo,
        scales,
        scale2,
        n,
        k,
    })
}

/// Reference block-128x128 FP8 E4M3 quantization (DeepSeek/Qwen FP8 layout).
///
/// # Errors
/// Returns an error if dimensions are zero or `weights.len() != n * k`.
pub fn quantize_fp8_block128_host(
    weights: &[f32],
    n: usize,
    k: usize,
) -> Result<(Vec<u8>, Vec<f32>)> {
    ensure!(n > 0 && k > 0, "FP8 block-128 dimensions must be positive");
    ensure!(
        weights.len() == n * k,
        "FP8 slice len {} != n*k ({})",
        weights.len(),
        n * k
    );
    let n_blocks = n.div_ceil(128);
    let k_blocks = k.div_ceil(128);
    let mut scales = vec![0.0_f32; n_blocks * k_blocks];

    for nb in 0..n_blocks {
        let r_end = ((nb + 1) * 128).min(n);
        for kb in 0..k_blocks {
            let c_end = ((kb + 1) * 128).min(k);
            let mut bmax = 0.0_f32;
            for r in (nb * 128)..r_end {
                for c in (kb * 128)..c_end {
                    bmax = bmax.max(weights[r * k + c].abs());
                }
            }
            scales[nb * k_blocks + kb] = if bmax > 0.0 { bmax / 448.0 } else { 1.0 };
        }
    }

    let mut fp8 = vec![0_u8; n * k];
    for r in 0..n {
        let nb = r / 128;
        for c in 0..k {
            let kb = c / 128;
            let s = scales[nb * k_blocks + kb];
            fp8[r * k + c] = f32_to_signed_e4m3(weights[r * k + c] / s);
        }
    }
    Ok((fp8, scales))
}

/// Dequantize a block-128 FP8 E4M3 matrix back to FP32.
///
/// # Errors
/// Returns an error if buffer extents do not match `[n, k]`.
pub fn dequantize_fp8_block128_host(
    fp8: &[u8],
    block_scales: &[f32],
    n: usize,
    k: usize,
) -> Result<Vec<f32>> {
    let n_blocks = n.div_ceil(128);
    let k_blocks = k.div_ceil(128);
    ensure!(fp8.len() == n * k, "FP8 payload extent mismatch");
    ensure!(
        block_scales.len() == n_blocks * k_blocks,
        "FP8 block scale extent mismatch"
    );
    let mut out = vec![0.0_f32; n * k];
    for r in 0..n {
        let nb = r / 128;
        for c in 0..k {
            let kb = c / 128;
            out[r * k + c] = e4m3_byte_to_f32(fp8[r * k + c]) * block_scales[nb * k_blocks + kb];
        }
    }
    Ok(out)
}

/// Repack a block-128 FP8 E4M3 matrix directly into DualPlaneNvfp4 (host mirror
/// of `repack_fp8_blockscaled_to_dual_plane_nvfp4`).
///
/// # Errors
/// Returns an error on invalid extents or `k % 16 != 0`.
pub fn repack_fp8_block128_to_dual_plane_host(
    fp8: &[u8],
    block_scales: &[f32],
    n: usize,
    k: usize,
) -> Result<HostDualPlaneMatrix> {
    let dequant = dequantize_fp8_block128_host(fp8, block_scales, n, k)?;
    quantize_f32_to_dual_plane_host(&dequant, n, k)
}

/// Dequantize ONLY the `W_hi` draft plane (`4.5 bpw`).
#[must_use]
pub fn dequantize_dual_plane_hi_host(mat: &HostDualPlaneMatrix) -> Vec<f32> {
    let num_groups = mat.k / DUAL_PLANE_GROUP_SIZE;
    let mut out = vec![0.0_f32; mat.n * mat.k];
    for row in 0..mat.n {
        for g in 0..num_groups {
            let eff = e4m3_byte_to_f32(mat.scales[row * num_groups + g]) * mat.scale2;
            let byte_base = row * (mat.k / 2) + g * 8;
            let elem_base = row * mat.k + g * DUAL_PLANE_GROUP_SIZE;
            for p in 0..8 {
                let bh = mat.w_hi[byte_base + p];
                out[elem_base + p * 2] = DUAL_PLANE_E2M1_LUT[(bh & 0x0F) as usize] * eff;
                out[elem_base + p * 2 + 1] = DUAL_PLANE_E2M1_LUT[(bh >> 4) as usize] * eff;
            }
        }
    }
    out
}

/// Dequantize the full verify planes `W_hi + 0.125 * W_lo` (`8.5 bpw`).
#[must_use]
pub fn dequantize_dual_plane_full_host(mat: &HostDualPlaneMatrix) -> Vec<f32> {
    let num_groups = mat.k / DUAL_PLANE_GROUP_SIZE;
    let mut out = vec![0.0_f32; mat.n * mat.k];
    for row in 0..mat.n {
        for g in 0..num_groups {
            let eff = e4m3_byte_to_f32(mat.scales[row * num_groups + g]) * mat.scale2;
            let byte_base = row * (mat.k / 2) + g * 8;
            let elem_base = row * mat.k + g * DUAL_PLANE_GROUP_SIZE;
            for p in 0..8 {
                let bh = mat.w_hi[byte_base + p];
                let bl = mat.w_lo[byte_base + p];
                let w0 = (DUAL_PLANE_E2M1_LUT[(bh & 0x0F) as usize]
                    + DUAL_PLANE_LO_SCALE * DUAL_PLANE_E2M1_LUT[(bl & 0x0F) as usize])
                    * eff;
                let w1 = (DUAL_PLANE_E2M1_LUT[(bh >> 4) as usize]
                    + DUAL_PLANE_LO_SCALE * DUAL_PLANE_E2M1_LUT[(bl >> 4) as usize])
                    * eff;
                out[elem_base + p * 2] = w0;
                out[elem_base + p * 2 + 1] = w1;
            }
        }
    }
    out
}

/// Signal-to-quantization-noise ratio in decibels: `10 * log10(||ref||^2 / ||ref - approx||^2)`.
#[must_use]
pub fn compute_sqnr_db(reference: &[f32], approx: &[f32]) -> f64 {
    let mut sig = 0.0_f64;
    let mut err = 0.0_f64;
    for (&r, &a) in reference.iter().zip(approx.iter()) {
        let rf = f64::from(r);
        let d = rf - f64::from(a);
        sig += rf * rf;
        err += d * d;
    }
    if err <= 0.0 {
        return f64::INFINITY;
    }
    10.0 * (sig / err).log10()
}

/// Cosine similarity between two equal-length vectors.
#[must_use]
pub fn compute_cosine_similarity(reference: &[f32], approx: &[f32]) -> f64 {
    let mut dot = 0.0_f64;
    let mut nr = 0.0_f64;
    let mut na = 0.0_f64;
    for (&r, &a) in reference.iter().zip(approx.iter()) {
        let rf = f64::from(r);
        let af = f64::from(a);
        dot += rf * af;
        nr += rf * rf;
        na += af * af;
    }
    let denom = (nr * na).sqrt();
    if denom <= 0.0 { 0.0 } else { dot / denom }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic standard-normal generator (Box-Muller over SplitMix64) so
    /// tests need zero external RNG crates and remain bit-reproducible.
    fn deterministic_gaussian(count: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        let mut next_u01 = || -> f64 {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            // Strictly in (0, 1) to avoid ln(0).
            ((z >> 11) as f64 + 0.5) * (1.0 / ((1_u64 << 53) as f64))
        };

        let mut out = Vec::with_capacity(count);
        while out.len() < count {
            let u1 = next_u01();
            let u2 = next_u01();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = std::f64::consts::TAU * u2;
            out.push((r * theta.cos()) as f32);
            if out.len() < count {
                out.push((r * theta.sin()) as f32);
            }
        }
        out
    }

    #[test]
    fn dual_plane_nvfp4_achieves_38_5_db_sqnr_and_beats_fp8_by_over_6_db() {
        let (n, k) = (64, 512);
        let weights = deterministic_gaussian(n * k, 0xA71A_5000_0001);

        let dual = quantize_f32_to_dual_plane_host(&weights, n, k).unwrap();
        let hi_only = dequantize_dual_plane_hi_host(&dual);
        let full = dequantize_dual_plane_full_host(&dual);

        let (fp8, fp8_scales) = quantize_fp8_block128_host(&weights, n, k).unwrap();
        let fp8_deq = dequantize_fp8_block128_host(&fp8, &fp8_scales, n, k).unwrap();

        let sqnr_hi = compute_sqnr_db(&weights, &hi_only);
        let sqnr_full = compute_sqnr_db(&weights, &full);
        let sqnr_fp8 = compute_sqnr_db(&weights, &fp8_deq);
        let cos_hi = compute_cosine_similarity(&weights, &hi_only);
        let cos_full = compute_cosine_similarity(&weights, &full);

        assert!(
            (37.5..=40.0).contains(&sqnr_full),
            "DualPlaneNvfp4 SQNR should be ~38.5 dB, got {sqnr_full:.2} dB"
        );
        assert!(
            (30.5..=33.5).contains(&sqnr_fp8),
            "Block-128 FP8 E4M3 SQNR should be ~31.8 dB, got {sqnr_fp8:.2} dB"
        );
        assert!(
            sqnr_full - sqnr_fp8 >= 6.0,
            "DualPlaneNvfp4 ({sqnr_full:.2} dB) must exceed FP8 E4M3 ({sqnr_fp8:.2} dB) by >= 6 dB"
        );
        assert!(
            (19.5..=22.0).contains(&sqnr_hi) && cos_hi >= 0.993,
            "W_hi draft plane expected ~20.5 dB / cos>=0.993, got {sqnr_hi:.2} dB / cos={cos_hi:.5}"
        );
        assert!(cos_full >= 0.9999);
    }

    #[test]
    fn fp8_block128_repack_to_dual_plane_preserves_fp8_beyond_37_db() {
        let (n, k) = (64, 512);
        let weights = deterministic_gaussian(n * k, 0xD0A1_B1A4_0002);
        let (fp8, fp8_scales) = quantize_fp8_block128_host(&weights, n, k).unwrap();
        let fp8_ref = dequantize_fp8_block128_host(&fp8, &fp8_scales, n, k).unwrap();

        let repacked = repack_fp8_block128_to_dual_plane_host(&fp8, &fp8_scales, n, k).unwrap();
        let full = dequantize_dual_plane_full_host(&repacked);
        let sqnr_vs_fp8 = compute_sqnr_db(&fp8_ref, &full);
        assert!(
            sqnr_vs_fp8 >= 37.5,
            "FP8 -> DualPlaneNvfp4 repack must match FP8 to >=37.5 dB, got {sqnr_vs_fp8:.2} dB"
        );
    }

    #[test]
    fn dual_plane_policy_and_zero_copy_draft_view() {
        assert_eq!(
            DualPlaneLoadPolicy::parse_optional(None).unwrap(),
            DualPlaneLoadPolicy::Disabled
        );
        assert_eq!(
            DualPlaneLoadPolicy::parse_optional(Some("   ")).unwrap(),
            DualPlaneLoadPolicy::Disabled
        );
        assert_eq!(
            DualPlaneLoadPolicy::parse_optional(Some("0")).unwrap(),
            DualPlaneLoadPolicy::Disabled
        );
        assert_eq!(
            DualPlaneLoadPolicy::parse_optional(Some("1")).unwrap(),
            DualPlaneLoadPolicy::Enabled
        );
        assert_eq!(
            DualPlaneLoadPolicy::parse_optional(Some("true")).unwrap(),
            DualPlaneLoadPolicy::Enabled
        );
        assert!(DualPlaneLoadPolicy::parse_optional(Some("maybe")).is_err());

        let w = DualPlaneNvfp4Weight {
            w_hi: DevicePtr(0x1000),
            w_lo: DevicePtr(0x2000),
            weight_scale: DevicePtr(0x3000),
            weight_scale_2: 0.25,
            input_scale: DevicePtr::NULL,
        };
        assert!(!w.is_null());
        let draft = w.draft_view();
        assert_eq!(draft.weight, DevicePtr(0x1000));
        assert_eq!(draft.weight_scale, DevicePtr(0x3000));
        assert_eq!(draft.weight_scale_2, 0.25);

        // 8.5 bpw for 64 x 256 = 64 * 256 * 8.5 / 8 = 17,408 bytes.
        assert_eq!(DualPlaneNvfp4Weight::storage_bytes(64, 256), Some(17_408));
        assert_eq!(DualPlaneNvfp4Weight::storage_bytes(64, 250), None);
    }
}
