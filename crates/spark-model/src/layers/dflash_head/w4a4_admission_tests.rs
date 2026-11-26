// SPDX-License-Identifier: AGPL-3.0-only

use super::DflashQuantization;
use super::w4a4_admission::{admit_w4a4_shadow, parse_w4a4_setting, validate_complete_layers};

#[test]
fn w4a4_flag_is_strict_and_default_off() {
    assert!(!parse_w4a4_setting(None).unwrap());
    assert!(!parse_w4a4_setting(Some("0")).unwrap());
    assert!(parse_w4a4_setting(Some("1")).unwrap());
    for invalid in ["", "true", "01", "2"] {
        assert!(parse_w4a4_setting(Some(invalid)).is_err());
    }
}

#[test]
fn disabled_route_never_claims_checkpoint_or_width_requirements() {
    assert!(!admit_w4a4_shadow(false, false, DflashQuantization::Bf16, 1, 1, false).unwrap());
}

#[test]
fn exact_rtncal_gamma15_k16_noise_only_route_is_admitted() {
    assert!(admit_w4a4_shadow(true, true, DflashQuantization::Nvfp4, 15, 16, true).unwrap());
    validate_complete_layers(true, 5, 5).unwrap();
}

#[test]
fn every_exact_route_precondition_is_load_bearing() {
    assert!(admit_w4a4_shadow(true, false, DflashQuantization::Nvfp4, 15, 16, true).is_err());
    assert!(admit_w4a4_shadow(true, true, DflashQuantization::Bf16, 15, 16, true).is_err());
    assert!(admit_w4a4_shadow(true, true, DflashQuantization::Nvfp4, 16, 16, true).is_err());
    assert!(admit_w4a4_shadow(true, true, DflashQuantization::Nvfp4, 15, 17, true).is_err());
    assert!(admit_w4a4_shadow(true, true, DflashQuantization::Nvfp4, 15, 16, false).is_err());
    assert!(validate_complete_layers(true, 4, 5).is_err());
    validate_complete_layers(false, 0, 5).unwrap();
}

#[test]
fn forward_has_four_shared_quantizations_and_seven_w4a4_projections() {
    let source = include_str!("forward_block_layer.rs");
    let body = &source[source
        .find("    fn forward_block_layer_nvfp4(")
        .expect("NVFP4 forward")..];
    assert_eq!(body.matches("w4a4.quantize(").count(), 4);
    assert_eq!(body.matches("w4a4.project_prepared(").count(), 7);

    let context = &body[body.find("// 3b'. K/V projections").expect("context start")
        ..body
            .find("// 4. K/V for noise tokens")
            .expect("context end")];
    assert!(!context.contains("w4a4.quantize("));
    assert!(!context.contains("w4a4.project_prepared("));
    assert!(context.contains("ops::w4a16_gemm("));
}
