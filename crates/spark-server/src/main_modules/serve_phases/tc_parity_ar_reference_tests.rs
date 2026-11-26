// SPDX-License-Identifier: AGPL-3.0-only

use std::collections::BTreeMap;

use super::*;

fn plain_qwen38() -> ReferenceContext<'static> {
    ReferenceContext {
        target_model: QUALIFIED_TARGET,
        model_type: "qwen3_5",
        hidden_size: 5120,
        num_experts: 0,
        speculative: false,
        dflash: false,
        self_speculative: false,
        ngram_speculative: false,
        mtp_gate: false,
    }
}

fn qualified_environment() -> BTreeMap<&'static str, String> {
    let mut environment = BTreeMap::from([(CONTROL_ENV, QUALIFIED_ROWS.to_string())]);
    environment.extend(
        REQUIRED_ROUTE_ENV
            .iter()
            .map(|&(name, value)| (name, value.to_string())),
    );
    environment
}

fn assess(
    context: ReferenceContext<'_>,
    environment: &BTreeMap<&str, String>,
) -> Result<Option<usize>> {
    admit_with(context, |name| Ok(environment.get(name).cloned()))
}

#[test]
fn absent_control_preserves_the_production_default() {
    assert_eq!(assess(plain_qwen38(), &BTreeMap::new()).unwrap(), None);
}

#[test]
fn unreadable_control_fails_closed() {
    let error = admit_with(plain_qwen38(), |name| {
        anyhow::bail!("{name} is not valid Unicode")
    })
    .unwrap_err();
    assert!(error.to_string().contains(CONTROL_ENV), "{error:#}");
    assert!(error.to_string().contains("not valid Unicode"), "{error:#}");
}

#[test]
fn exact_dense_q27_control_admits_sixteen_rows() {
    assert_eq!(
        assess(plain_qwen38(), &qualified_environment()).unwrap(),
        Some(16)
    );
}

#[test]
fn control_accepts_only_the_qualified_width() {
    for value in ["0", "15", "32", "sixteen"] {
        let mut environment = qualified_environment();
        environment.insert(CONTROL_ENV, value.to_string());
        let error = assess(plain_qwen38(), &environment).unwrap_err();
        assert!(error.to_string().contains(CONTROL_ENV), "{error:#}");
        assert!(error.to_string().contains("exactly 16"), "{error:#}");
    }
}

#[test]
fn target_and_dense_q27_identity_are_both_fail_closed() {
    let environment = qualified_environment();
    let mut wrong_target = plain_qwen38();
    wrong_target.target_model = "qwen3.5-27b";
    assert!(
        assess(wrong_target, &environment)
            .unwrap_err()
            .to_string()
            .contains(QUALIFIED_TARGET)
    );

    for mutate in [
        |context: &mut ReferenceContext<'_>| context.model_type = "qwen4_exp",
        |context: &mut ReferenceContext<'_>| context.hidden_size = 2560,
        |context: &mut ReferenceContext<'_>| context.num_experts = 256,
    ] {
        let mut context = plain_qwen38();
        mutate(&mut context);
        let error = assess(context, &environment).unwrap_err();
        assert!(error.to_string().contains("dense qwen3_5 27B"), "{error:#}");
    }
}

#[test]
fn every_non_plain_scheduler_mode_is_rejected() {
    let environment = qualified_environment();
    for mutate in [
        |context: &mut ReferenceContext<'_>| context.speculative = true,
        |context: &mut ReferenceContext<'_>| context.dflash = true,
        |context: &mut ReferenceContext<'_>| context.self_speculative = true,
        |context: &mut ReferenceContext<'_>| context.ngram_speculative = true,
        |context: &mut ReferenceContext<'_>| context.mtp_gate = true,
    ] {
        let mut context = plain_qwen38();
        mutate(&mut context);
        let error = assess(context, &environment).unwrap_err();
        assert!(
            error.to_string().contains("plain autoregressive"),
            "{error:#}"
        );
    }
}

#[test]
fn each_tensor_core_route_selector_is_required_exactly() {
    for &(name, expected) in REQUIRED_ROUTE_ENV {
        let mut missing = qualified_environment();
        missing.remove(name);
        let error = assess(plain_qwen38(), &missing).unwrap_err();
        assert!(error.to_string().contains(name), "{name}: {error:#}");

        let mut wrong = qualified_environment();
        wrong.insert(name, format!("not-{expected}"));
        let error = assess(plain_qwen38(), &wrong).unwrap_err();
        assert!(error.to_string().contains(name), "{name}: {error:#}");
    }
}

#[test]
fn kill_switch_and_truncated_vocab_are_rejected() {
    for (name, value) in [
        ("ATLAS_DISABLE_SPECULATION", "1"),
        ("ATLAS_DISABLE_SPECULATION", "invalid"),
        ("ATLAS_TARGET_LMHEAD_VOCAB", "96000"),
    ] {
        let mut environment = qualified_environment();
        environment.insert(name, value.to_string());
        let error = assess(plain_qwen38(), &environment).unwrap_err();
        assert!(error.to_string().contains(name), "{name}: {error:#}");
    }
}
