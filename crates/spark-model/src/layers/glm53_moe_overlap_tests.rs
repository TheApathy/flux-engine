// SPDX-License-Identifier: AGPL-3.0-only

use std::cell::Cell;
use std::ffi::OsStr;

use super::*;

const SERIAL_SOURCE: &str = include_str!("glm53_moe_serial.rs");
const DISPATCH_SOURCE: &str = include_str!("../model/glm53/dispatch.rs");
const MODEL_SOURCE: &str = include_str!("../model/glm53/target_model_exl3.rs");

#[test]
fn selector_is_default_off_and_strict() {
    assert_eq!(parse(None).unwrap(), Setting::Off);
    assert_eq!(parse(Some(OsStr::new("0"))).unwrap(), Setting::Off);
    assert_eq!(parse(Some(OsStr::new("1"))).unwrap(), Setting::On);
    assert!(parse(Some(OsStr::new("true"))).is_err());
    assert!(parse(Some(OsStr::new("2"))).is_err());
}

#[cfg(unix)]
#[test]
fn selector_rejects_non_utf8() {
    use std::os::unix::ffi::OsStrExt;

    assert!(parse(Some(OsStr::from_bytes(&[0xff]))).is_err());
}

fn admitted<'a>() -> Admission<'a> {
    Admission {
        request: Some(OsStr::new("1")),
        private_prefill: true,
        staged_loaded: true,
        reconstruct: Some(OsStr::new("1")),
        reconstruct_dtype: Some(OsStr::new("bf16")),
        reconstruct_overlap: None,
        chunk_rows: Some(OsStr::new("64k2")),
        direct_combine: Some(OsStr::new("1")),
        layer_major: Some(OsStr::new("1")),
        layer_major_rows: Some(OsStr::new("2048")),
    }
}

#[test]
fn admission_is_bound_to_the_current_exact_recipe() {
    assert_eq!(admit(admitted()).unwrap(), Setting::On);
    for mutate in [
        |a: &mut Admission<'_>| a.reconstruct_dtype = Some(OsStr::new("f16")),
        |a: &mut Admission<'_>| a.chunk_rows = Some(OsStr::new("64k2s8")),
        |a: &mut Admission<'_>| a.reconstruct_overlap = Some(OsStr::new("1")),
        |a: &mut Admission<'_>| a.private_prefill = false,
        |a: &mut Admission<'_>| a.staged_loaded = false,
        |a: &mut Admission<'_>| a.reconstruct = Some(OsStr::new("0")),
        |a: &mut Admission<'_>| a.direct_combine = Some(OsStr::new("0")),
        |a: &mut Admission<'_>| a.layer_major = Some(OsStr::new("0")),
        |a: &mut Admission<'_>| a.layer_major_rows = Some(OsStr::new("4096")),
    ] {
        let mut wrong = admitted();
        mutate(&mut wrong);
        assert!(admit(wrong).is_err());
    }
}

#[test]
fn disabled_admission_does_not_claim_recipe_ownership() {
    let mut input = admitted();
    input.request = None;
    input.reconstruct_dtype = Some(OsStr::new("invalid-but-owned-elsewhere"));
    assert_eq!(admit(input).unwrap(), Setting::Off);
}

#[test]
fn disabled_setting_allocates_no_owner_and_enabled_allocates_exactly_once() {
    for setting in [Setting::Off, Setting::Off] {
        let creates = Cell::new(0);
        let owner = initialize(setting, || {
            creates.set(creates.get() + 1);
            Ok::<_, anyhow::Error>(17)
        })
        .unwrap();
        assert!(owner.is_none());
        assert_eq!(creates.get(), 0);
    }

    let creates = Cell::new(0);
    let owner = initialize(Setting::On, || {
        creates.set(creates.get() + 1);
        Ok::<_, anyhow::Error>(17)
    })
    .unwrap();
    assert_eq!(owner, Some(17));
    for _ in 0..4 {
        assert_eq!(owner.as_ref(), Some(&17));
    }
    assert_eq!(creates.get(), 1);
}

#[test]
fn owner_presence_must_match_the_current_latched_setting() {
    assert!(validate_owner_state(Setting::Off, false).is_ok());
    assert!(validate_owner_state(Setting::On, true).is_ok());
    assert!(validate_owner_state(Setting::Off, true).is_err());
    assert!(validate_owner_state(Setting::On, false).is_err());
}

#[test]
fn overlap_owner_is_model_persistent_and_dispatchers_only_borrow_it() {
    assert_eq!(
        MODEL_SOURCE.matches("Glm53MoeOverlap::from_env(").count(),
        1
    );
    assert!(MODEL_SOURCE.contains("exl3_prefill_overlap: Option<Glm53MoeOverlap>"));
    assert!(MODEL_SOURCE.contains("self.exl3_prefill_overlap.as_ref(),"));

    assert!(DISPATCH_SOURCE.contains("exl3_prefill_overlap: Option<&'w Glm53MoeOverlap>"));
    assert!(DISPATCH_SOURCE.contains("exl3_prefill_overlap,"));
    assert!(DISPATCH_SOURCE.contains("Glm53MoeOverlap::validate_env("));
    assert!(!DISPATCH_SOURCE.contains("Glm53MoeOverlap::new("));
    assert!(!SERIAL_SOURCE.contains("exl3_prefill_overlap: Option<Glm53MoeOverlap>"));
    assert!(!SERIAL_SOURCE.contains("Glm53MoeOverlap::new(gpu)?"));
}

#[test]
fn receipt_and_scope_match_the_preregistered_walk() {
    assert!(receipt_due(1));
    assert!(!receipt_due(2));
    assert!(!receipt_due(41));
    assert!(receipt_due(42));
    assert!(receipt_due(84));
    assert!(!scope(1023));
    assert!(scope(1024));
    assert!(scope(2047));
    assert!(scope(2048));
    assert!(!scope(2049));
}

#[test]
fn production_receipt_is_after_join_observation_and_combine() {
    let run = SERIAL_SOURCE.find("overlap.run(").unwrap();
    let combine = SERIAL_SOURCE[run..]
        .find("fused.combine_shared(")
        .map(|offset| run + offset)
        .unwrap();
    let receipt = SERIAL_SOURCE[combine..]
        .find("joined.complete(rows)")
        .map(|offset| combine + offset)
        .unwrap();
    assert!(run < combine && combine < receipt);
    assert!(include_str!("glm53_moe_overlap.rs").contains(
        "if self.work_in_flight && !self.completed {\n            self.owner.quiesce(self.gpu, self.main_stream);"
    ));
}
