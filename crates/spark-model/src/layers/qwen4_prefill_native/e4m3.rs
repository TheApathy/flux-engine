// SPDX-License-Identifier: AGPL-3.0-only

//! Host math for `ATLAS_QWEN4_PREFILL_NATIVE`: a BF16 tensor that is the exact
//! dequantization of a checkpoint MXFP8 matrix (block-32 E4M3 values times a
//! UE8M0 power-of-two scale) becomes E4M3 bytes plus ONE power-of-two tensor
//! scale, the operand layout of the existing cuBLASLt `SCALAR_32F` FP8 GEMM.
//!
//! Why a power of two: every MXFP8 value is `q * 2^e` with `q` on the E4M3
//! grid. Dividing by `s = 2^E` (the smallest power of two with
//! `amax / s <= 448`, which is the largest block scale) keeps each value on
//! the E4M3 grid unless its block sits so many binades below the largest
//! block that its low mantissa bits fall under the subnormal step. On the
//! Flash-Next checkpoint 99.77-99.91% of elements are reproduced bit-exactly
//! and the relative Frobenius error is 2.3e-6..8.6e-6, against 2.3e-2..3.0e-2
//! for an `amax / 448` tensor scale and 9.2e-2..9.7e-2 for the
//! NVFP4 -> E4M3 path.

use anyhow::{Result, bail, ensure};

pub(crate) const E4M3_MAX: f32 = 448.0;
const E4M3_MIN_NORMAL: f64 = 1.0 / 64.0; // 2^-6
const E4M3_SUBNORMAL_STEPS: f64 = 512.0; // 1 / 2^-9

pub(crate) fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// E4M3 (`e4m3fn`: no infinities, 0x7F/0xFF are NaN) to f32.
pub(crate) fn e4m3_decode(code: u8) -> f32 {
    let sign = if code & 0x80 != 0 { -1.0f32 } else { 1.0f32 };
    let exp = i32::from((code >> 3) & 0x0F);
    let man = f32::from(code & 0x07);
    if exp == 0x0F && man == 7.0 {
        return f32::NAN;
    }
    let mag = if exp == 0 {
        man * (2.0f32).powi(-9)
    } else {
        (1.0 + man / 8.0) * (2.0f32).powi(exp - 7)
    };
    sign * mag
}

/// f32 to E4M3 with round-to-nearest-even, saturating finite overflow to
/// +-448 (the `__NV_SATFINITE` conversion the device kernels use).
pub(crate) fn e4m3_encode(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7F;
    }
    let sign: u8 = if x.is_sign_negative() { 0x80 } else { 0x00 };
    let a = f64::from(x).abs();
    if a > f64::from(E4M3_MAX) {
        return sign | 0x7E;
    }
    if a < E4M3_MIN_NORMAL {
        // Subnormal grid 2^-9; rounding up to 8 steps lands on 0x08 (= 2^-6).
        let steps = (a * E4M3_SUBNORMAL_STEPS).round_ties_even() as u8;
        return sign | steps;
    }
    let mut exp = ((a.to_bits() >> 52) & 0x7FF) as i32 - 1023;
    let mut mant = (a / 2f64.powi(exp - 3)).round_ties_even(); // in [8, 16]
    if mant >= 16.0 {
        exp += 1;
        mant = 8.0;
    }
    if exp > 8 || (exp == 8 && mant > 14.0) {
        return sign | 0x7E;
    }
    sign | (((exp + 7) as u8) << 3) | ((mant as u8) - 8)
}

/// Smallest power of two `s` with `amax / s <= 448` (1.0 for an all-zero
/// tensor). A power of two keeps the MXFP8 grid; see the module docs.
pub(crate) fn pow2_tensor_scale(amax: f32) -> Result<f32> {
    ensure!(
        amax.is_finite() && amax >= 0.0,
        "native E4M3: non-finite tensor absmax {amax}"
    );
    if amax == 0.0 {
        return Ok(1.0);
    }
    let amax = f64::from(amax);
    let max = f64::from(E4M3_MAX);
    let mut exp = (amax / max).log2().ceil() as i32;
    while amax / 2f64.powi(exp) > max {
        exp += 1;
    }
    while amax / 2f64.powi(exp - 1) <= max {
        exp -= 1;
    }
    ensure!(
        (-126..=127).contains(&exp),
        "native E4M3: tensor scale 2^{exp} outside f32 normal range"
    );
    Ok(2f64.powi(exp) as f32)
}

/// The gated-Q weight-row permutation `qwen4_fp8_weight_rows_tensor` applies
/// (`qg = (heads, head_dim)`): source row `n` of the interleaved
/// `[q_h | gate_h]` blocks lands at this destination row (all q rows first).
/// Production uses the inverse ([`qg_src_row`]); tests pin it to this map.
#[cfg(test)]
pub(crate) fn qg_dest_row(n: usize, heads: usize, head_dim: usize) -> usize {
    let group = 2 * head_dim;
    let (h, idx) = (n / group, n % group);
    if idx < head_dim {
        h * head_dim + idx
    } else {
        heads * head_dim + h * head_dim + (idx - head_dim)
    }
}

/// Inverse of [`qg_dest_row`].
pub(crate) fn qg_src_row(dest: usize, heads: usize, head_dim: usize) -> usize {
    let q_rows = heads * head_dim;
    let (base, d) = if dest < q_rows {
        (0, dest)
    } else {
        (head_dim, dest - q_rows)
    };
    (d / head_dim) * 2 * head_dim + base + d % head_dim
}

/// Reconstruction quality versus the BF16 source, accumulated per tensor.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct ConversionStats {
    pub elements: usize,
    pub inexact: usize,
    pub err_sq: f64,
    pub ref_sq: f64,
}

impl ConversionStats {
    pub(crate) fn merge(self, other: Self) -> Self {
        Self {
            elements: self.elements + other.elements,
            inexact: self.inexact + other.inexact,
            err_sq: self.err_sq + other.err_sq,
            ref_sq: self.ref_sq + other.ref_sq,
        }
    }

    /// `||W8 * s - W||_F / ||W||_F` (0 for an all-zero tensor).
    pub(crate) fn rel_fro(&self) -> f64 {
        if self.ref_sq == 0.0 {
            0.0
        } else {
            (self.err_sq / self.ref_sq).sqrt()
        }
    }

    pub(crate) fn inexact_fraction(&self) -> f64 {
        if self.elements == 0 {
            0.0
        } else {
            self.inexact as f64 / self.elements as f64
        }
    }
}

/// E4M3 rows (destination order) plus the power-of-two tensor scale.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct NativeE4m3 {
    pub bytes: Vec<u8>,
    pub scale: f32,
    pub stats: ConversionStats,
}

fn tensor_absmax(src: &[u16]) -> Result<f32> {
    src.iter().try_fold(0.0f32, |m, &b| {
        let v = bf16_to_f32(b);
        if v.is_finite() {
            Ok(m.max(v.abs()))
        } else {
            bail!("native E4M3: non-finite BF16 weight element")
        }
    })
}

fn convert_row(src: &[u16], dst: &mut [u8], inv: f32, scale: f32) -> ConversionStats {
    src.iter()
        .zip(dst.iter_mut())
        .fold(ConversionStats::default(), |acc, (&b, out)| {
            let v = bf16_to_f32(b);
            let code = e4m3_encode(v * inv);
            *out = code;
            let back = e4m3_decode(code) * scale;
            let err = f64::from(back) - f64::from(v);
            ConversionStats {
                elements: acc.elements + 1,
                inexact: acc.inexact + usize::from(back != v),
                err_sq: acc.err_sq + err * err,
                ref_sq: acc.ref_sq + f64::from(v) * f64::from(v),
            }
        })
}

/// Quantize a row-major `[rows, k]` BF16 tensor to E4M3 with one power-of-two
/// scale; `qg` applies the gated-Q row permutation of the device converter.
pub(crate) fn quantize_pow2_tensor(
    src: &[u16],
    rows: usize,
    k: usize,
    qg: Option<(usize, usize)>,
) -> Result<NativeE4m3> {
    ensure!(
        rows > 0 && k > 0 && src.len() == rows * k,
        "native E4M3: shape [{rows}, {k}] does not match {} elements",
        src.len()
    );
    if let Some((heads, head_dim)) = qg {
        ensure!(
            heads > 0 && head_dim > 0 && 2 * heads * head_dim == rows,
            "native E4M3: gated-Q shape ({heads}, {head_dim}) does not cover {rows} rows"
        );
    }
    let scale = pow2_tensor_scale(tensor_absmax(src)?)?;
    let inv = 1.0 / scale; // exact: scale is a power of two
    let mut bytes = vec![0u8; rows * k];
    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .clamp(1, 8);
    let rows_per = rows.div_ceil(threads);
    let src_row = |dest: usize| qg.map_or(dest, |(h, d)| qg_src_row(dest, h, d));
    let stats = std::thread::scope(|scope| {
        let handles: Vec<_> = bytes
            .chunks_mut(rows_per * k)
            .enumerate()
            .map(|(chunk, out)| {
                scope.spawn(move || {
                    out.chunks_mut(k).enumerate().fold(
                        ConversionStats::default(),
                        |acc, (i, dst)| {
                            let s = src_row(chunk * rows_per + i);
                            acc.merge(convert_row(&src[s * k..(s + 1) * k], dst, inv, scale))
                        },
                    )
                })
            })
            .collect();
        handles
            .into_iter()
            .try_fold(ConversionStats::default(), |acc, h| {
                h.join()
                    .map(|s| acc.merge(s))
                    .map_err(|_| anyhow::anyhow!("native E4M3: conversion thread panicked"))
            })
    })?;
    Ok(NativeE4m3 {
        bytes,
        scale,
        stats,
    })
}

#[cfg(test)]
#[path = "e4m3_tests.rs"]
mod tests;
