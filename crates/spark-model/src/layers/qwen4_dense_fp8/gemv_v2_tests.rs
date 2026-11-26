// SPDX-License-Identifier: AGPL-3.0-only

//! Host proofs for `fused_v2`. The GPU byte-equality check against the
//! reference kernel lives in `mxfp8_gpu_tests.rs` (ignored: needs a GB10).

use super::super::mxfp8::tests::{Rng, synthetic};
use super::super::mxfp8::{bf16_to_f32, e8m0_to_f32, f32_to_bf16, kernel_gemv, weight};
use super::{
    GemvVariant, TPO32_MAX_K, TpoPolicy, parse_tpo, parse_variant, receipt_line, threads_per_out,
};

/// What `cvt.rn.f16x2.e4m3x2` + `cvt.f32.f16` yield: the true E4M3FN value
/// (computed in f64 from the encoding, independent of the reference decode).
fn hw_e4m3(code: u8) -> f32 {
    let e = i32::from((code >> 3) & 0x0F);
    let m = f64::from(code & 0x07);
    let mag = if e == 0 {
        m / 8.0 * 2f64.powi(-6)
    } else {
        (1.0 + m / 8.0) * 2f64.powi(e - 7)
    };
    let v = mag as f32; // <= 4 significant bits: exact
    if code & 0x80 != 0 { -v } else { v }
}

fn is_nan_code(code: u8) -> bool {
    code & 0x7F == 0x7F
}

fn fast_chunk_ok(exponent: u8, codes: &[u8]) -> bool {
    (10..=246).contains(&exponent) && !codes.iter().any(|&c| is_nan_code(c))
}

/// The fused_v2 weight for one element of a chunk.
fn v2_weight(code: u8, exponent: u8, fast: bool) -> f32 {
    if fast {
        hw_e4m3(code) * f32::from_bits(u32::from(exponent) << 23)
    } else {
        weight(code, exponent)
    }
}

#[test]
fn fast_decode_equals_reference_weight_for_every_code_and_exponent_in_range() {
    let mut checked = 0u32;
    for exponent in 10u8..=246 {
        for code in 0u8..=255 {
            if is_nan_code(code) {
                continue;
            }
            let fast = v2_weight(code, exponent, true);
            let reference = weight(code, exponent);
            assert_eq!(
                fast.to_bits(),
                reference.to_bits(),
                "code {code:#04x} e {exponent}"
            );
            // The exactness argument: zero or a normal, finite product.
            assert!(fast == 0.0 || (fast.is_normal() && fast.is_finite()));
            checked += 1;
        }
    }
    assert_eq!(checked, 237 * 254);
}

#[test]
fn range_bounds_are_where_the_argument_stops() {
    // e = 9: the smallest subnormal code times 2^-118 is an f32 subnormal.
    assert!(!(hw_e4m3(0x01) * e8m0_to_f32(9)).is_normal());
    // e = 247: 448 * 2^120 overflows f32.
    assert!((hw_e4m3(0x7E) * e8m0_to_f32(247)).is_infinite());
    // NaN codes: the hardware cvt gives NaN, the reference gives +-0.
    assert_eq!(weight(0x7F, 127).to_bits(), 0);
    assert_eq!(weight(0xFF, 127).to_bits(), 0x8000_0000);
}

/// Host model of one `fused_v2` row: per-chunk fast/fallback decode, then
/// either the reference 64-lane mapping or the TPO32 dual-accumulator one.
fn v2_row(a: &[u16], codes: &[u8], scales: &[u8], k: usize, tpo: u32) -> u16 {
    const VEC: usize = 16;
    let chunks = k / VEC;
    let chunk = |acc: &mut f32, kv: usize| {
        let e = scales[kv / 2];
        let cs = &codes[kv * VEC..][..VEC];
        let fast = fast_chunk_ok(e, cs);
        for (i, &c) in cs.iter().enumerate() {
            *acc += bf16_to_f32(a[kv * VEC + i]) * v2_weight(c, e, fast);
        }
    };
    let tree = |lanes: &mut [f32]| {
        let mut offset = 16;
        while offset > 0 {
            for t in 0..32 - offset {
                lanes[t] += lanes[t + offset];
            }
            offset /= 2;
        }
        lanes[0]
    };
    let (mut lo, mut hi) = ([0f32; 32], [0f32; 32]);
    if tpo == 32 {
        for l in 0..32 {
            let mut kv = l;
            while kv < chunks {
                chunk(&mut lo[l], kv);
                if kv + 32 < chunks {
                    chunk(&mut hi[l], kv + 32);
                }
                kv += 64;
            }
        }
    } else {
        for vl in 0..64 {
            let acc = if vl < 32 {
                &mut lo[vl]
            } else {
                &mut hi[vl - 32]
            };
            for kv in (vl..chunks).step_by(64) {
                chunk(acc, kv);
            }
        }
    }
    f32_to_bf16(tree(&mut lo) + tree(&mut hi))
}

pub(crate) fn adversarial(n: usize, k: usize, seed: u64) -> (Vec<u16>, Vec<u8>, Vec<u8>) {
    let mut rng = Rng(seed);
    let codes = (0..n * k)
        .map(|i| match i % 7 {
            0 => [0x00, 0x80, 0x7F, 0xFF, 0x01, 0x81, 0x07, 0x7E][rng.next_u64() as usize % 8],
            _ => rng.next_u64() as u8,
        })
        .collect();
    // Every exponent the kernel can see, both range edges heavily.
    let scales = (0..n * k / 32)
        .map(|i| match i % 5 {
            0 => [0u8, 1, 9, 10, 246, 247, 254, 127][rng.next_u64() as usize % 8],
            _ => rng.next_u64() as u8 % 255,
        })
        .collect();
    // Activations: finite BF16 incl. subnormals and both zeros, moderate range
    // so sums stay finite (NaN payloads are not modelled on the host).
    let a = (0..k)
        .map(|i| match i % 9 {
            0 => 0x0001,
            1 => 0x8000,
            2 => 0x0000,
            _ => {
                let v = ((rng.next_u64() >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0;
                f32_to_bf16(v * 2f32.powi((rng.next_u64() % 40) as i32 - 30))
            }
        })
        .collect();
    (a, codes, scales)
}

fn check_shape(n: usize, k: usize, seed: u64) {
    for (a, codes, scales) in [synthetic(n, k, seed), adversarial(n, k, seed ^ 0xA5)] {
        let reference = kernel_gemv(&a, &codes, &scales, n, k).unwrap();
        let sc = k / 32;
        for tpo in [32, 64] {
            for r in 0..n {
                let got = v2_row(&a, &codes[r * k..][..k], &scales[r * sc..][..sc], k, tpo);
                let want = reference[r];
                // Compare as values when both are NaN (payload not modelled).
                let same = got == want || (bf16_to_f32(got).is_nan() && bf16_to_f32(want).is_nan());
                assert!(
                    same,
                    "[{n},{k}] tpo {tpo} row {r}: {got:#06x} vs {want:#06x}"
                );
            }
        }
    }
}

#[test]
fn v2_host_model_matches_reference_on_flash_next_shapes() {
    // k=2560: qkv|z, q, k|v, gate|up; k=6144: GDN out_proj, attn o; k=640: down.
    for (n, k) in [(48, 2560), (24, 6144), (64, 640)] {
        check_shape(n, k, 0x5EED ^ k as u64);
    }
}

#[test]
fn v2_host_model_matches_reference_for_every_small_k() {
    for k in (32..=2048).step_by(32) {
        check_shape(3, k, k as u64);
    }
}

#[test]
fn selectors_parse_strictly() {
    assert_eq!(parse_variant(None).unwrap(), GemvVariant::FusedV2);
    assert_eq!(
        parse_variant(Some("fused_v2")).unwrap(),
        GemvVariant::FusedV2
    );
    assert_eq!(parse_variant(Some("ref")).unwrap(), GemvVariant::Reference);
    assert!(parse_variant(Some("1")).is_err());
    assert_eq!(parse_tpo(None).unwrap(), TpoPolicy::Auto);
    assert_eq!(parse_tpo(Some("32")).unwrap(), TpoPolicy::Fixed(32));
    assert_eq!(parse_tpo(Some("64")).unwrap(), TpoPolicy::Fixed(64));
    assert!(parse_tpo(Some("128")).is_err());
}

#[test]
fn auto_tpo_follows_k() {
    assert_eq!(threads_per_out(TpoPolicy::Auto, 640), 32);
    assert_eq!(threads_per_out(TpoPolicy::Auto, 2560), 32);
    assert_eq!(threads_per_out(TpoPolicy::Auto, TPO32_MAX_K), 32);
    assert_eq!(threads_per_out(TpoPolicy::Auto, 6144), 64);
    assert_eq!(threads_per_out(TpoPolicy::Fixed(64), 640), 64);
}

#[test]
fn receipt_line_format() {
    assert_eq!(
        receipt_line(GemvVariant::FusedV2, Some(32), 2560),
        "DENSE_FP8_GEMV variant=fused_v2 threads_per_out=32 k=2560"
    );
    assert_eq!(
        receipt_line(GemvVariant::Reference, None, 6144),
        "DENSE_FP8_GEMV variant=ref threads_per_out=64 k=6144"
    );
}
