// SPDX-License-Identifier: AGPL-3.0-only

//! Host model of the native MXFP8 decode GEMV (`dense_gemv_mxfp8.cu`).
//!
//! The checkpoint ships each dense matrix as E4M3 codes `[N, K]` plus UE8M0
//! block-32 exponents `[N, K/32]`. The kernel consumes both as shipped: no
//! requantization, so the weight error against the exact MXFP8 -> BF16
//! dequant (`dequant_mxfp8_bf16.cu`) is zero by construction. These
//! functions reproduce the kernel's per-weight value and its FP32 reduction
//! order bit for bit, so a GPU check can demand exact equality with this
//! model and a bounded difference against a dequantize-then-GEMV reference.

use anyhow::{Result, ensure};

/// Weights per UE8M0 exponent.
pub(crate) const MXFP8_BLOCK: usize = 32;
/// Weights per lane iteration (one `uint4`).
#[cfg(test)]
const LANE_VEC: usize = 16;
/// Threads cooperating on one output row.
#[cfg(test)]
const LANES_PER_ROW: usize = 64;
#[cfg(test)]
const WARP: usize = 32;

#[cfg(test)]
/// E4M3 decode as `dequant_mxfp8_bf16` does it: -0 and both NaN codes -> 0.
pub(crate) fn e4m3_to_f32(code: u8) -> f32 {
    let exponent = u32::from((code >> 3) & 0x0F);
    let mantissa = u32::from(code & 0x07);
    let value = if (exponent == 0 && mantissa == 0) || (exponent == 0x0F && mantissa == 0x07) {
        0.0
    } else if exponent == 0 {
        mantissa as f32 * (1.0 / 512.0)
    } else {
        f32::from_bits(((exponent + 120) << 23) | (mantissa << 20))
    };
    if code & 0x80 != 0 { -value } else { value }
}

#[cfg(test)]
/// UE8M0 exponent -> 2^(e-127); 255 (reserved NaN) -> 0, like the kernel.
pub(crate) fn e8m0_to_f32(exponent: u8) -> f32 {
    match exponent {
        255 => 0.0,
        0 => f32::from_bits(1 << 22),
        e => f32::from_bits(u32::from(e) << 23),
    }
}

#[cfg(test)]
/// The weight one element contributes: exact product, truncated to BF16.
pub(crate) fn weight(code: u8, exponent: u8) -> f32 {
    let product = e4m3_to_f32(code) * e8m0_to_f32(exponent);
    f32::from_bits(product.to_bits() & 0xFFFF_0000)
}

#[cfg(test)]
pub(crate) fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

#[cfg(test)]
/// Round-to-nearest-even f32 -> BF16 (`__float2bfloat16`), finite inputs.
pub(crate) fn f32_to_bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    let round = 0x7FFF + ((bits >> 16) & 1);
    (bits.wrapping_add(round) >> 16) as u16
}

/// Geometry the kernel requires; checked on the host before any launch.
pub(crate) fn validate_geometry(n: usize, k: usize, scale_cols: usize) -> Result<()> {
    ensure!(
        n > 0 && k > 0 && k.is_multiple_of(MXFP8_BLOCK),
        "MXFP8 GEMV needs n>0 and k a multiple of {MXFP8_BLOCK}, got [{n},{k}]"
    );
    ensure!(
        scale_cols == k / MXFP8_BLOCK,
        "MXFP8 scale grid has {scale_cols} columns, expected k/32 = {}",
        k / MXFP8_BLOCK
    );
    Ok(())
}

/// The reserved UE8M0 NaN exponent must never reach a decode GEMV.
pub(crate) fn validate_scales(prefix: &str, scales: &[u8]) -> Result<()> {
    ensure!(
        !scales.contains(&u8::MAX),
        "MXFP8 tensor {prefix} contains reserved E8M0 NaN scale 255"
    );
    Ok(())
}

#[cfg(test)]
/// One output row exactly as `dense_gemv_mxfp8` computes it.
fn kernel_row(a: &[u16], codes: &[u8], scales: &[u8], k: usize) -> u16 {
    let chunks = k / LANE_VEC;
    let mut lanes = [0f32; LANES_PER_ROW];
    for (lane, acc) in lanes.iter_mut().enumerate() {
        for kv in (lane..chunks).step_by(LANES_PER_ROW) {
            let exponent = scales[kv / 2];
            for i in 0..LANE_VEC {
                let col = kv * LANE_VEC + i;
                *acc += bf16_to_f32(a[col]) * weight(codes[col], exponent);
            }
        }
    }
    let warp_sum = |warp: &mut [f32]| {
        let mut offset = WARP / 2;
        while offset > 0 {
            for t in 0..WARP - offset {
                warp[t] += warp[t + offset];
            }
            offset /= 2;
        }
        warp[0]
    };
    let (w0, w1) = lanes.split_at_mut(WARP);
    f32_to_bf16(warp_sum(w0) + warp_sum(w1))
}

#[cfg(test)]
/// Host model of `dense_gemv_mxfp8` over row-major `[n, k]` codes.
pub(crate) fn kernel_gemv(
    a: &[u16],
    codes: &[u8],
    scales: &[u8],
    n: usize,
    k: usize,
) -> Result<Vec<u16>> {
    validate_geometry(n, k, scales.len() / n.max(1))?;
    ensure!(
        a.len() == k && codes.len() == n * k && scales.len() == n * (k / MXFP8_BLOCK),
        "MXFP8 GEMV buffers do not match [{n},{k}]"
    );
    let sc = k / MXFP8_BLOCK;
    Ok((0..n)
        .map(|r| kernel_row(a, &codes[r * k..][..k], &scales[r * sc..][..sc], k))
        .collect())
}

#[cfg(test)]
/// Exact MXFP8 -> BF16 dequant, the `dequant_mxfp8_bf16` output.
pub(crate) fn dequant_bf16(codes: &[u8], scales: &[u8], n: usize, k: usize) -> Vec<u16> {
    let sc = k / MXFP8_BLOCK;
    (0..n * k)
        .map(|i| {
            (weight(codes[i], scales[(i / k) * sc + (i % k) / MXFP8_BLOCK]).to_bits() >> 16) as u16
        })
        .collect()
}

#[cfg(test)]
#[path = "mxfp8_tests.rs"]
pub(crate) mod tests;
