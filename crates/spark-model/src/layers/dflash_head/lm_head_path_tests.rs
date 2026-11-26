// SPDX-License-Identifier: AGPL-3.0-only

use super::{LmHeadPath, resolve_lm_head_path};

const CONSTRUCTOR_SOURCE: &str = include_str!("from_weights.rs");
const NOISE_PASS_SOURCE: &str = include_str!("noise_pass/eager.rs");

fn compact(source: &str) -> String {
    source.chars().filter(|ch| !ch.is_whitespace()).collect()
}

#[test]
fn projection_and_norm_are_resolved_as_one_path() {
    let cases = [
        (true, true, true, true, Ok(LmHeadPath::Nvfp4)),
        (false, true, true, true, Ok(LmHeadPath::Fp8)),
        (false, false, false, false, Ok(LmHeadPath::Bf16)),
    ];

    for (requested, weight, kernel, fp8, expected) in cases {
        assert_eq!(
            resolve_lm_head_path(requested, weight, kernel, fp8),
            expected
        );
    }
}

#[test]
fn requested_nvfp4_never_silently_falls_back() {
    let missing_weight = resolve_lm_head_path(true, false, true, true)
        .expect_err("a requested NVFP4 head needs its transposed weight");
    assert!(missing_weight.contains("transposed weight"));

    let missing_kernel = resolve_lm_head_path(true, true, false, true)
        .expect_err("a requested NVFP4 head needs its GEMM kernel");
    assert!(missing_kernel.contains("GEMM kernel"));

    assert!(resolve_lm_head_path(true, false, false, false).is_err());
}

#[test]
fn only_fp8_uses_compensated_norm_and_selector_restores_canonical_norm() {
    assert!(!LmHeadPath::Nvfp4.uses_fp8_input_norm());
    assert!(!LmHeadPath::Bf16.uses_fp8_input_norm());
    assert!(LmHeadPath::Fp8.uses_fp8_input_norm());

    assert!(!LmHeadPath::Nvfp4.restores_unscaled_norm_for_selector(true));
    assert!(!LmHeadPath::Bf16.restores_unscaled_norm_for_selector(true));
    assert!(LmHeadPath::Fp8.restores_unscaled_norm_for_selector(true));
    assert!(!LmHeadPath::Fp8.restores_unscaled_norm_for_selector(false));
}

#[test]
fn constructor_and_selector_keep_the_scale_boundary_visible() {
    let constructor = compact(CONSTRUCTOR_SOURCE);
    assert!(constructor.contains("letnorm=weights.norm;"));
    assert!(!constructor.contains("norm_after"));
    assert!(constructor.contains("lm_head_shared_fp8=Some(DflashFp8LmHead{"));
    assert!(constructor.contains("input_norm:DenseWeight{weight:norm_scaled"));
    assert!(constructor.contains("hidden_norm:weights.hidden_norm,norm,"));

    let noise = compact(NOISE_PASS_SOURCE);
    assert!(noise.contains("LmHeadPath::Nvfp4|LmHeadPath::Bf16=>&self.norm"));
    let fp8_projection = noise
        .find("draft_lm_head_fp8")
        .expect("FP8 projection dispatch");
    let restore = noise
        .find("iflm_head_path.restores_unscaled_norm_for_selector")
        .expect("unscaled selector restore");
    let selector = noise
        .find("&sel.hidden_projection")
        .expect("selector hidden projection");
    assert!(fp8_projection < restore);
    assert!(restore < selector);
}
