// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

pub(super) fn flash_geometry() -> Geometry {
    Geometry {
        hidden: 2560,
        layers: 48,
        attention_layers: 12,
        ssm_layers: 36,
        experts: 512,
        top_k: 10,
        routed_intermediate: 640,
        shared_intermediate: 640,
        q_heads: 24,
        kv_heads: 2,
        head_dim: 256,
        key_heads: 16,
        key_dim: 128,
        value_heads: 48,
        value_dim: 128,
        conv_dim: 4,
        qkvz: 16384,
    }
}

#[test]
fn exact_m2013_attention_receipt_binds_selector_census_and_geometry() {
    let selectors = Selectors::from_values(Some("1"), Some("0"));
    let mut capture = Capture::new(selectors, 2013, flash_geometry()).unwrap();
    for _ in 0..12 {
        capture.engage(PrefillPath::Attention, 2013).unwrap();
    }
    assert_eq!(
        capture.finish().unwrap(),
        vec![
            "QWEN4_PREFILL_SELECTOR_RECEIPT family=attention \
selector=ATLAS_QWEN4_ATTN_PREFILL_BATCH value=1 M=2013 attention_selector=1 ssm_selector=0 \
path_success=enqueued serialized_fallback=false expected_layers=12 engaged_layers=12 \
H=2560 L=48 E=512 TOPK=10 I=640 SI=640 Q=24 KV=2 HD=256"
                .to_string(),
        ]
    );
}

#[test]
fn exact_m2013_ssm_receipt_binds_selector_census_and_geometry() {
    let selectors = Selectors::from_values(Some("0"), Some("1"));
    let mut capture = Capture::new(selectors, 2013, flash_geometry()).unwrap();
    for _ in 0..36 {
        capture.engage(PrefillPath::Ssm, 2013).unwrap();
    }
    let lines = capture.finish().unwrap();
    assert_eq!(lines.len(), 1);
    assert!(lines[0].contains("family=ssm selector=ATLAS_QWEN4_SSM_PREFILL_BATCH"));
    assert!(lines[0].contains("expected_layers=36 engaged_layers=36"));
    assert!(lines[0].contains("NK=16 KD=128 NV=48 VD=128 D=4 QKVZ=16384"));
}

#[test]
fn malformed_and_absent_selectors_preserve_default_off_state() {
    assert_eq!(Selectors::from_values(None, None), Selectors::default());
    assert_eq!(
        Selectors::from_values(Some("true"), Some("2")),
        Selectors::default()
    );
    assert!(!Selectors::from_values(Some("0"), Some("0")).any());
}

#[test]
fn r2_receipt_is_disabled_when_the_fast_family_is_request_suppressed() {
    let suppressed = Selectors::from_values(None, None)
        .with_moe_only(false)
        .with_slot_mxfp8_r2(true);
    assert!(!suppressed.slot_mxfp8_r2);
    assert!(!suppressed.any());

    let admitted = Selectors::from_values(None, None)
        .with_moe_only(true)
        .with_slot_mxfp8_r2(true);
    assert!(admitted.slot_mxfp8_r2);
    assert!(admitted.any());
}

#[test]
fn receipt_drop_releases_capture_before_discarding_all_deferred_state() {
    let source = include_str!("../qwen4_prefill_engagement.rs");
    let drop_start = source.find("impl Drop for ReceiptGuard").unwrap();
    let drop_body = &source[drop_start
        ..source[drop_start..]
            .find("pub(crate) fn begin")
            .map_or(source.len(), |offset| drop_start + offset)];
    let release = drop_body.find("receipt mutex").unwrap();
    let pending = drop_body.find("pending::discard_stale()").unwrap();
    let status = drop_body.find("qwen4_moe_status::discard_stale()").unwrap();
    let slot = drop_body.find("discard_stale_slot_storage()").unwrap();
    assert!(release < pending && pending < status && status < slot);
}

#[test]
fn hostile_m_geometry_family_and_census_drift_fail_closed() {
    let attention = Selectors::from_values(Some("1"), Some("0"));
    assert!(Capture::new(attention, 1, flash_geometry()).is_err());
    let mut bad_geometry = flash_geometry();
    bad_geometry.top_k = 513;
    assert!(Capture::new(attention, 2013, bad_geometry).is_err());

    let mut capture = Capture::new(attention, 2013, flash_geometry()).unwrap();
    assert!(capture.engage(PrefillPath::Attention, 2012).is_err());
    assert!(capture.engage(PrefillPath::Ssm, 2013).is_err());
    capture.engage(PrefillPath::Attention, 2013).unwrap();
    assert!(capture.finish().is_err());
}
