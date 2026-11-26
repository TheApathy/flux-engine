// SPDX-License-Identifier: AGPL-3.0-only

use super::{
    bf16_to_f32, dequant_bf16, e4m3_to_f32, e8m0_to_f32, f32_to_bf16, kernel_gemv,
    validate_geometry, validate_scales, weight,
};

/// splitmix64: deterministic, dependency-free test data.
pub(crate) struct Rng(pub(crate) u64);

impl Rng {
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Synthetic MXFP8 matrix shaped like a checkpoint tensor: random E4M3
/// codes (NaN codes included, they decode to 0) and block exponents spread
/// over 2^-14..2^-3 so rows mix magnitudes. Activations are BF16 in [-2, 2).
pub(crate) fn synthetic(n: usize, k: usize, seed: u64) -> (Vec<u16>, Vec<u8>, Vec<u8>) {
    let mut rng = Rng(seed);
    let codes = (0..n * k).map(|_| rng.next_u64() as u8).collect();
    let scales = (0..n * k / 32)
        .map(|_| 113 + (rng.next_u64() % 12) as u8)
        .collect();
    let a = (0..k)
        .map(|_| f32_to_bf16(((rng.next_u64() >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0))
        .collect();
    (a, codes, scales)
}

/// Dequantize-then-GEMV reference: sequential FP64 over the exact BF16 weights.
fn dequant_gemv_f64(a: &[u16], w_bf16: &[u16], n: usize, k: usize) -> (Vec<f64>, Vec<f64>) {
    (0..n)
        .map(|r| {
            let row = &w_bf16[r * k..][..k];
            row.iter().zip(a).fold((0f64, 0f64), |(s, m), (&w, &x)| {
                let p = f64::from(bf16_to_f32(w)) * f64::from(bf16_to_f32(x));
                (s + p, m + p.abs())
            })
        })
        .unzip()
}

#[test]
fn every_code_and_exponent_decodes_like_the_dequant_kernel() {
    for code in 0u8..=255 {
        let v = e4m3_to_f32(code);
        if code & 0x7F == 0x7F || code & 0x7F == 0 {
            assert_eq!(v, 0.0, "code {code:#04x}");
        }
        for e in [0u8, 1, 100, 120, 127, 134] {
            let w = weight(code, e);
            let exact = f64::from(v) * 2f64.powi(i32::from(e) - 127);
            assert_eq!(w.to_bits() & 0xFFFF, 0, "weight is a BF16 value");
            if exact.abs() >= f64::from(f32::MIN_POSITIVE) {
                assert_eq!(
                    f64::from(w),
                    exact,
                    "code {code:#04x} e {e}: power-of-two scale is exact"
                );
            }
        }
    }
    assert_eq!(e8m0_to_f32(127), 1.0);
    assert_eq!(e8m0_to_f32(255), 0.0);
    assert_eq!(e4m3_to_f32(0x7E), 448.0);
    assert_eq!(e4m3_to_f32(0x01), 1.0 / 512.0);
}

#[test]
fn native_gemv_has_zero_weight_error_and_matches_dequant_gemv() {
    let (n, k) = (96usize, 2560usize);
    let (a, codes, scales) = synthetic(n, k, 20261030);
    let w_bf16 = dequant_bf16(&codes, &scales, n, k);
    // Zero weight error: the in-register weight IS the exact dequant.
    for (i, &bits) in w_bf16.iter().enumerate() {
        let in_kernel = weight(codes[i], scales[(i / k) * (k / 32) + (i % k) / 32]);
        assert_eq!(in_kernel.to_bits() >> 16, u32::from(bits));
    }
    let y = kernel_gemv(&a, &codes, &scales, n, k).unwrap();
    let (y_ref, mag) = dequant_gemv_f64(&a, &w_bf16, n, k);
    let mut worst = 0f64;
    for r in 0..n {
        let diff = (f64::from(bf16_to_f32(y[r])) - y_ref[r]).abs();
        // BF16 output rounding (2^-9 relative) plus FP32 accumulation-order noise.
        let bound = y_ref[r].abs() * 2f64.powi(-8) + mag[r] * 2f64.powi(-14);
        assert!(
            diff <= bound,
            "row {r}: {} vs {} (bound {bound})",
            bf16_to_f32(y[r]),
            y_ref[r]
        );
        worst = worst.max(diff / mag[r].max(f64::MIN_POSITIVE));
    }
    eprintln!(
        "QWEN4_DENSE_FP8 native MXFP8 GEMV [{n}x{k}]: weight error 0, worst |dy|/sum|a*w| = {worst:.3e}"
    );
}

#[test]
fn kernel_model_uses_one_exponent_per_32_columns() {
    // Row of ones, exponent 127 everywhere except block 1 (cols 32..64) at 128:
    // that block's weights double, so the dot doubles its contribution.
    let k = 128usize;
    let a = vec![f32_to_bf16(1.0); k];
    let codes = vec![0x38u8; k]; // E4M3 1.0
    let mut scales = vec![127u8; k / 32];
    let base = kernel_gemv(&a, &codes, &scales, 1, k).unwrap()[0];
    scales[1] = 128;
    let doubled = kernel_gemv(&a, &codes, &scales, 1, k).unwrap()[0];
    assert_eq!(bf16_to_f32(base), 128.0);
    assert_eq!(bf16_to_f32(doubled), 160.0);
}

#[test]
fn geometry_and_reserved_scales_are_refused() {
    assert!(validate_geometry(4, 64, 2).is_ok());
    assert!(
        validate_geometry(4, 48, 1).is_err(),
        "k must be a multiple of 32"
    );
    assert!(validate_geometry(4, 64, 3).is_err());
    assert!(validate_geometry(0, 64, 2).is_err());
    assert!(validate_scales("x", &[0, 127, 254]).is_ok());
    assert!(validate_scales("x", &[127, 255]).is_err());
    assert!(kernel_gemv(&[0; 64], &[0; 64 * 2], &[127; 3], 2, 64).is_err());
}
