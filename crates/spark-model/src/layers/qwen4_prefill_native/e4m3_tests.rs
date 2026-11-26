// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn f32_to_bf16(v: f32) -> u16 {
    // Test inputs are exactly representable in BF16 (MXFP8 values).
    let bits = v.to_bits();
    assert_eq!(bits & 0xFFFF, 0, "{v} is not exact in BF16");
    (bits >> 16) as u16
}

/// Deterministic LCG so the fixtures need no RNG crate.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
}

/// A synthetic MXFP8 matrix: random finite E4M3 codes times a per-32-block
/// UE8M0 scale spread over `spread` binades below `2^top` (matching typical
/// checkpoint block-scale distributions).
fn mxfp8_fixture(rows: usize, k: usize, top: i32, spread: u32, seed: u64) -> Vec<f32> {
    let mut rng = Lcg(seed);
    let mut out = Vec::with_capacity(rows * k);
    for _ in 0..rows * (k / 32) {
        let e = top - (rng.next() % (spread + 1)) as i32;
        for _ in 0..32 {
            let mut code = (rng.next() & 0xFF) as u8;
            if code & 0x7F >= 0x7E {
                // No NaN, and no exact 448: the tensor absmax then is not
                // 448 * 2^k, so `amax / 448` is not itself a power of two
                // (the typical case on real checkpoint tensors).
                code = (code & 0x80) | 0x7D;
            }
            out.push(e4m3_decode(code) * 2f32.powi(e));
        }
    }
    out
}

fn to_bf16(v: &[f32]) -> Vec<u16> {
    v.iter().map(|&x| f32_to_bf16(x)).collect()
}

fn rel_fro(approx: &[f32], exact: &[f32]) -> f64 {
    let (e, r) = approx
        .iter()
        .zip(exact)
        .fold((0.0f64, 0.0f64), |(e, r), (&a, &x)| {
            let d = f64::from(a) - f64::from(x);
            (e + d * d, r + f64::from(x) * f64::from(x))
        });
    (e / r).sqrt()
}

fn decode_all(q: &NativeE4m3) -> Vec<f32> {
    q.bytes.iter().map(|&c| e4m3_decode(c) * q.scale).collect()
}

/// Reference model of today's path: BF16 -> NVFP4 (g16, e4m3 block scale,
/// one scale2 = amax / 2688, as `quantize_to_nvfp4`) -> `amax / 448` tensor
/// E4M3 of the `LUT * fp8` values (`qwen4_fp8_weight_rows_tensor`), times
/// scale2 as the GEMM alpha.
fn nvfp4_then_tensor_e4m3(w: &[f32]) -> Vec<f32> {
    const LUT: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let e2m1 = |x: f32| -> f32 {
        let a = x.abs().min(6.0);
        let mut best = 0usize;
        for (i, &l) in LUT.iter().enumerate() {
            let (d, db) = ((a - l).abs(), (a - LUT[best]).abs());
            // nearest; exact ties go to the even code (lower mantissa bit)
            if d < db || (d == db && i % 2 == 0 && best % 2 == 1) {
                best = i;
            }
        }
        LUT[best].copysign(x)
    };
    let amax = w.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let scale2 = amax / (6.0 * 448.0);
    let lut_fp8: Vec<f32> = w
        .chunks(16)
        .flat_map(|g| {
            let gmax = g.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let bs = e4m3_decode(e4m3_encode(gmax / 6.0 / scale2)).max(2f32.powi(-9));
            g.iter()
                .map(move |&v| e2m1(v / (bs * scale2)) * bs)
                .collect::<Vec<_>>()
        })
        .collect();
    let lamax = lut_fp8.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let t = lamax / 448.0;
    lut_fp8
        .iter()
        .map(|&v| e4m3_decode(e4m3_encode(v / t)) * t * scale2)
        .collect()
}

#[test]
fn e4m3_codes_round_trip_and_known_values() {
    for code in 0u8..=255 {
        if code & 0x7F == 0x7F {
            assert!(e4m3_decode(code).is_nan());
            continue;
        }
        let v = e4m3_decode(code);
        let back = e4m3_encode(v);
        // +0/-0 keep their sign bit; everything else is an identity.
        assert_eq!(back, code, "code {code:#04x} value {v}");
    }
    assert_eq!(e4m3_encode(1.0), 0x38);
    assert_eq!(e4m3_encode(-1.0), 0xB8);
    assert_eq!(e4m3_encode(448.0), 0x7E);
    assert_eq!(e4m3_encode(1.0e6), 0x7E);
    assert_eq!(e4m3_encode(2f32.powi(-9)), 0x01);
    assert_eq!(e4m3_encode(2f32.powi(-11)), 0x00); // below half a subnormal step
    assert_eq!(e4m3_encode(f32::NAN), 0x7F);
}

#[test]
fn e4m3_encode_rounds_half_to_even() {
    // 1.0625 is halfway between 1.0 (m=0) and 1.125 (m=1): even -> 1.0.
    assert_eq!(e4m3_decode(e4m3_encode(1.0625)), 1.0);
    // 1.1875 is halfway between 1.125 (m=1) and 1.25 (m=2): even -> 1.25.
    assert_eq!(e4m3_decode(e4m3_encode(1.1875)), 1.25);
    // Subnormal tie 1.5 * 2^-9 -> 2 * 2^-9.
    assert_eq!(e4m3_encode(1.5 * 2f32.powi(-9)), 0x02);
    // Rounding up out of the subnormal range reaches the min normal 2^-6.
    assert_eq!(e4m3_encode(7.6 * 2f32.powi(-9)), 0x08);
    // Mantissa carry bumps the exponent: 1.97 -> 2.0.
    assert_eq!(e4m3_decode(e4m3_encode(1.97)), 2.0);
}

#[test]
fn pow2_scale_is_smallest_power_of_two_fitting_448() {
    assert_eq!(pow2_tensor_scale(0.0).unwrap(), 1.0);
    assert_eq!(pow2_tensor_scale(448.0).unwrap(), 1.0);
    assert_eq!(pow2_tensor_scale(448.5).unwrap(), 2.0);
    assert_eq!(pow2_tensor_scale(224.0).unwrap(), 0.5);
    assert_eq!(pow2_tensor_scale(0.8125).unwrap(), 2f32.powi(-9));
    assert!(pow2_tensor_scale(f32::INFINITY).is_err());
    assert!(pow2_tensor_scale(-1.0).is_err());
}

#[test]
fn qg_permutation_matches_device_map_and_inverts() {
    let (heads, hd) = (3usize, 4usize);
    let mut seen = vec![false; 2 * heads * hd];
    for n in 0..2 * heads * hd {
        let d = qg_dest_row(n, heads, hd);
        assert!(!seen[d]);
        seen[d] = true;
        assert_eq!(qg_src_row(d, heads, hd), n);
    }
    // Head 1, gate element 2: src row 1*8 + 4 + 2 = 14 -> 12 + 4 + 2 = 18.
    assert_eq!(qg_dest_row(14, heads, hd), 18);
}

#[test]
fn single_block_scale_tensor_is_bit_exact() {
    let w = mxfp8_fixture(16, 64, -9, 0, 7);
    let q = quantize_pow2_tensor(&to_bf16(&w), 16, 64, None).unwrap();
    assert_eq!(q.stats.inexact, 0);
    assert_eq!(q.stats.rel_fro(), 0.0);
    assert_eq!(decode_all(&q), w);
}

#[test]
fn gated_q_rows_are_permuted_like_the_device_converter() {
    let (heads, hd, k) = (2usize, 4usize, 32usize);
    let rows = 2 * heads * hd;
    let w = mxfp8_fixture(rows, k, -8, 0, 11);
    let q = quantize_pow2_tensor(&to_bf16(&w), rows, k, Some((heads, hd))).unwrap();
    let back = decode_all(&q);
    for n in 0..rows {
        let d = qg_dest_row(n, heads, hd);
        assert_eq!(&back[d * k..(d + 1) * k], &w[n * k..(n + 1) * k]);
    }
    assert!(quantize_pow2_tensor(&to_bf16(&w), rows, k, Some((3, hd))).is_err());
}

#[test]
fn native_error_is_orders_below_the_nvfp4_path() {
    // Four-binade block-scale spread, as on the real GDN/attention tensors.
    let (rows, k) = (64usize, 512usize);
    let w = mxfp8_fixture(rows, k, -8, 4, 2026);
    let native = quantize_pow2_tensor(&to_bf16(&w), rows, k, None).unwrap();
    let native_err = rel_fro(&decode_all(&native), &w);
    assert!((native_err - native.stats.rel_fro()).abs() < 1e-12);
    let amax = w.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let t = amax / 448.0;
    let amax448: Vec<f32> = w
        .iter()
        .map(|&v| e4m3_decode(e4m3_encode(v / t)) * t)
        .collect();
    let amax448_err = rel_fro(&amax448, &w);
    let nvfp4_err = rel_fro(&nvfp4_then_tensor_e4m3(&w), &w);
    eprintln!(
        "native_pow2={native_err:.3e} inexact={:.4} amax448={amax448_err:.3e} nvfp4_e4m3={nvfp4_err:.3e}",
        native.stats.inexact_fraction()
    );
    assert!(native_err < 1e-3, "native {native_err}");
    assert!(
        native_err * 50.0 < amax448_err,
        "native {native_err} vs amax/448 {amax448_err}"
    );
    assert!(
        native_err * 50.0 < nvfp4_err,
        "native {native_err} vs nvfp4 {nvfp4_err}"
    );
    assert!(
        nvfp4_err > 5e-2,
        "the NVFP4 reference model lost its expected error: {nvfp4_err}"
    );
}

#[test]
fn rejects_bad_shapes_and_non_finite_weights() {
    assert!(quantize_pow2_tensor(&[0u16; 8], 3, 3, None).is_err());
    let inf = f32_to_bf16(f32::INFINITY);
    assert!(quantize_pow2_tensor(&[inf, 0, 0, 0], 1, 4, None).is_err());
    let zeros = quantize_pow2_tensor(&[0u16; 8], 2, 4, None).unwrap();
    assert_eq!((zeros.scale, zeros.stats.rel_fro()), (1.0, 0.0));
}
