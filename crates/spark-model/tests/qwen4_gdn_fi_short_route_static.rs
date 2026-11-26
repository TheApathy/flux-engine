// SPDX-License-Identifier: AGPL-3.0-only

//! Source wiring guard for the Flash-Next short-prefill FI-GDN quarantine.

mod policy_under_test {
    include!("../src/layers/qwen3_ssm/qwen4_gdn_fi_policy.rs");
}

#[test]
fn short_route_boundary_executes_fail_closed() {
    assert!(!policy_under_test::use_fi(false, 8_192));
    assert!(!policy_under_test::use_fi(true, 64));
    assert!(policy_under_test::use_fi(true, 65));
}

#[test]
fn short_rows_route_around_fi_gdn_and_its_fused_norm() {
    let policy = include_str!("../src/layers/qwen3_ssm/qwen4_gdn_fi_policy.rs");
    for needle in [
        "pub(super) const SHORT_FALLBACK_MAX_ROWS: usize = 64",
        "requested && rows > SHORT_FALLBACK_MAX_ROWS",
        "assert!(!use_fi(true, 64))",
        "assert!(use_fi(true, 65))",
    ] {
        assert!(policy.contains(needle), "missing policy contract: {needle}");
    }
    assert!(!policy.contains("synchronize"));

    let forward = include_str!("../src/layers/qwen3_ssm/qwen4_prefill_exact_forward.rs");
    let decision = forward
        .find("let use_fi = super::qwen4_gdn_fi_policy::use_fi(")
        .expect("missing per-shape FI-GDN route");
    let fused_norm = forward[decision..]
        .find("let norm_fp8 = if use_fi")
        .map(|offset| decision + offset)
        .expect("fused norm must follow the route");
    let dispatch = forward[fused_norm..]
        .find("if use_fi")
        .map(|offset| fused_norm + offset)
        .expect("FI-GDN dispatch must use the same route");
    let exact = forward[dispatch..]
        .find("ops::gdn_decode_f32_sequence(")
        .map(|offset| dispatch + offset)
        .expect("exact recurrence fallback");
    assert!(decision < fused_norm && fused_norm < dispatch && dispatch < exact);
    assert!(!forward[decision..exact].contains("synchronize_device"));
    for needle in [
        "SSM_PREFILL_GDN_FI_SHORT_FALLBACK_ENGAGED",
        "reason=async_fi_gdn_state",
        "static LOGGED: std::sync::Once",
    ] {
        assert!(
            forward[decision..exact].contains(needle),
            "missing fallback receipt: {needle}"
        );
    }
}
