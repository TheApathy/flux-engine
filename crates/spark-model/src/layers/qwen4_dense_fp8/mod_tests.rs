// SPDX-License-Identifier: AGPL-3.0-only

use super::{
    BF16_ARM_SELECTORS, DecodeScope, DenseFp8Launch, LayerReceipt, Module, decide,
    first_set_bf16_arm, in_decode_scope, layer_engaged, module_dispatches, parse_selector,
};

fn qwen4() -> DenseFp8Launch {
    DenseFp8Launch {
        is_qwen4: true,
        ..DenseFp8Launch::default()
    }
}

#[test]
fn selector_parse_is_exact() {
    assert!(!parse_selector(None).unwrap());
    assert!(!parse_selector(Some("0")).unwrap());
    assert!(parse_selector(Some("1")).unwrap());
    for bad in ["", "true", "01", " 1", "yes"] {
        assert!(parse_selector(Some(bad)).is_err(), "{bad:?}");
    }
}

#[test]
fn default_is_off_for_every_launch() {
    let everything = DenseFp8Launch {
        is_qwen4: true,
        use_native_mtp: true,
        has_dflash: true,
        self_speculative: true,
        multi_batch: true,
        native_batch: true,
        multi_rank: true,
        bf16_arm: Some("ATLAS_QWEN4_GDN_BF16"),
    };
    for launch in [DenseFp8Launch::default(), qwen4(), everything] {
        assert!(!decide(false, launch).unwrap());
    }
    for (is_qwen4, mtp) in [(false, false), (true, true), (false, true)] {
        assert!(!layer_engaged(false, is_qwen4, mtp).unwrap());
    }
}

#[test]
fn plain_c1_qwen4_target_engages() {
    assert!(decide(true, qwen4()).unwrap());
    assert!(layer_engaged(true, true, false).unwrap());
}

#[test]
fn selector_refuses_other_models_and_nvfp4_only_routes() {
    assert!(decide(true, DenseFp8Launch::default()).is_err());
    assert!(layer_engaged(true, false, false).is_err());
    assert!(layer_engaged(true, true, true).is_err());
    let blocked = [
        DenseFp8Launch {
            use_native_mtp: true,
            ..qwen4()
        },
        DenseFp8Launch {
            has_dflash: true,
            ..qwen4()
        },
        DenseFp8Launch {
            self_speculative: true,
            ..qwen4()
        },
        DenseFp8Launch {
            multi_batch: true,
            ..qwen4()
        },
        DenseFp8Launch {
            native_batch: true,
            ..qwen4()
        },
        DenseFp8Launch {
            multi_rank: true,
            ..qwen4()
        },
    ];
    for launch in blocked {
        let err = decide(true, launch).unwrap_err().to_string();
        assert!(err.contains("refused"), "{launch:?}: {err}");
    }
}

#[test]
fn decode_scope_nests_and_closes() {
    assert!(!in_decode_scope());
    {
        let _outer = DecodeScope::enter();
        assert!(in_decode_scope());
        {
            let _inner = DecodeScope::enter();
            assert!(in_decode_scope());
        }
        assert!(
            in_decode_scope(),
            "inner drop must not close the outer scope"
        );
    }
    assert!(!in_decode_scope());
}

#[test]
fn decode_scope_is_per_thread() {
    let _scope = DecodeScope::enter();
    let other = std::thread::spawn(in_decode_scope).join().unwrap();
    assert!(in_decode_scope());
    assert!(
        !other,
        "a prefill worker thread must not inherit the decode scope"
    );
}

#[test]
fn layer_receipt_counts_each_dispatch_and_feeds_the_module_total() {
    let receipt = LayerReceipt::new(Module::Shared, 7);
    let before = module_dispatches(Module::Shared);
    assert_eq!(receipt.dispatches(), 0);
    receipt.record();
    receipt.record();
    assert_eq!(receipt.dispatches(), 2);
    assert!(module_dispatches(Module::Shared) >= before + 2);
    assert_eq!(Module::ALL.map(Module::name), ["gdn", "attn", "shared"]);
}

#[test]
fn bf16_reference_arms_are_mutually_exclusive_with_fp8() {
    for name in BF16_ARM_SELECTORS {
        let launch = DenseFp8Launch {
            bf16_arm: Some(name),
            ..qwen4()
        };
        let err = decide(true, launch).unwrap_err().to_string();
        assert!(
            err.contains(name) && err.contains("mutually exclusive"),
            "{err}"
        );
        // Default (FP8 unset) never refuses because of a BF16 arm.
        assert!(!decide(false, launch).unwrap());
    }
}

#[test]
fn bf16_arm_detection_reads_each_selector() {
    assert_eq!(first_set_bf16_arm(|_| None), None);
    assert_eq!(first_set_bf16_arm(|_| Some("0".into())), None);
    assert_eq!(first_set_bf16_arm(|_| Some(String::new())), None);
    for name in BF16_ARM_SELECTORS {
        let only = |n: &str| (n == name).then(|| "1".to_string());
        assert_eq!(first_set_bf16_arm(only), Some(name));
    }
}

#[test]
fn umbrella_implied_bf16_arm_refuses_fp8() {
    // ATLAS_QWEN4_DENSE_NATIVE=1 + ATLAS_QWEN4_DENSE_FP8=1 with no explicit
    // per-module BF16 env: the exclusion must still see the BF16 arms.
    let plan = crate::qwen4_native_selectors::plan_from(|name| {
        Ok((name == "ATLAS_QWEN4_DENSE_NATIVE").then(|| "1".to_string()))
    })
    .unwrap();
    let arm = first_set_bf16_arm(plan.reader());
    assert_eq!(arm, Some("ATLAS_QWEN4_GDN_BF16"));
    let err = decide(
        true,
        DenseFp8Launch {
            bf16_arm: arm,
            ..qwen4()
        },
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("mutually exclusive"), "{err}");
    // With every BF16 arm explicitly 0 under the umbrella, FP8 is admitted.
    let plan = crate::qwen4_native_selectors::plan_from(|name| {
        Ok(match name {
            "ATLAS_QWEN4_DENSE_NATIVE" => Some("1".to_string()),
            "ATLAS_QWEN4_GDN_BF16" | "ATLAS_QWEN4_ATTN_BF16" | "ATLAS_QWEN4_SHARED_BF16" => {
                Some("0".to_string())
            }
            _ => None,
        })
    })
    .unwrap();
    assert_eq!(first_set_bf16_arm(plan.reader()), None);
}
