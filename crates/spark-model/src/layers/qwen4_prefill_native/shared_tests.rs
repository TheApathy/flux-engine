// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use spark_runtime::weights::WeightDtype as D;

/// `e4m3fn` decode (no infinities; 0x7F/0xFF are NaN).
fn e4m3_decode(code: u8) -> f32 {
    let sign = if code & 0x80 != 0 { -1.0f32 } else { 1.0f32 };
    let exp = i32::from((code >> 3) & 0x0F);
    let man = f32::from(code & 0x07);
    if exp == 0x0F && man == 7.0 {
        return f32::NAN;
    }
    let mag = if exp == 0 {
        man * 2f32.powi(-9)
    } else {
        (1.0 + man / 8.0) * 2f32.powi(exp - 7)
    };
    sign * mag
}

/// Round-to-nearest-even f32 -> BF16 -> f32 (what the dequant kernel stores).
fn through_bf16(v: f32) -> f32 {
    let bits = v.to_bits();
    let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1)) & 0xFFFF_0000;
    f32::from_bits(rounded)
}

#[test]
fn flash_next_shapes() {
    assert_eq!(shapes(2560, 640), [(640, 2560), (640, 2560), (2560, 640)]);
}

#[test]
fn shipped_check_needs_mxfp8_with_the_declared_shape() {
    let ok = Shipped {
        weight: Some(D::FP8E4M3),
        scale: Some(D::UInt8),
        shape: Some(vec![640, 2560]),
    };
    assert!(check_shipped("gate_proj", &ok, (640, 2560)).is_ok());
    let block128 = Shipped {
        scale: Some(D::BF16),
        ..ok.clone()
    };
    assert!(
        check_shipped("gate_proj", &block128, (640, 2560)).is_err(),
        "block-128 FP8 is not MXFP8"
    );
    let bf16 = Shipped {
        weight: Some(D::BF16),
        scale: None,
        ..ok.clone()
    };
    assert!(check_shipped("gate_proj", &bf16, (640, 2560)).is_err());
    let missing = Shipped {
        weight: None,
        scale: None,
        shape: None,
    };
    assert!(check_shipped("gate_proj", &missing, (640, 2560)).is_err());
    assert!(
        check_shipped("down_proj", &ok, (2560, 640)).is_err(),
        "transposed shape"
    );
}

/// The native operand is the BF16 dequantization of MXFP8. Every finite E4M3
/// code times every UE8M0 block scale the BF16 normal range can hold is a
/// BF16 value, so the weight error versus exact MXFP8 is exactly zero (the
/// FP8 alternatives measured 1.7e-6..7.7e-6 on the checkpoint, NVFP4 9e-2).
#[test]
fn bf16_operand_reproduces_every_mxfp8_value_exactly() {
    let mut checked = 0usize;
    for e in -117..=119i32 {
        let scale = 2f32.powi(e);
        for code in 0u8..=255 {
            let q = e4m3_decode(code);
            if q.is_nan() {
                continue;
            }
            let exact = q * scale;
            assert_eq!(
                through_bf16(exact).to_bits(),
                exact.to_bits(),
                "code {code:#04x} e {e}"
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 237 * 254);
    assert_eq!(weight_rel_err_vs_mxfp8(), 0.0);
}

#[test]
fn registry_pairs_layers_by_shared_gate_and_refuses_duplicates() {
    let mut reg = Registry::default();
    let entry = SharedBf16 {
        gate: DevicePtr(0x1000),
        up: DevicePtr(0x2000),
        down: DevicePtr(0x3000),
        hidden: 2560,
        inter: 640,
        source: Source::Owned,
    };
    reg.insert(DevicePtr(0x9000), entry).unwrap();
    assert_eq!(reg.get(DevicePtr(0x9000)), Some(entry));
    assert_eq!(reg.get(DevicePtr(0x9100)), None);
    assert!(
        reg.insert(DevicePtr(0x9000), entry).is_err(),
        "duplicate layer"
    );
    assert!(reg.insert(DevicePtr::NULL, entry).is_err(), "null key");
    let null_down = SharedBf16 {
        down: DevicePtr::NULL,
        ..entry
    };
    assert!(reg.insert(DevicePtr(0x9200), null_down).is_err());
    let alias = SharedBf16 {
        source: Source::Alias,
        ..entry
    };
    reg.insert(DevicePtr(0x9300), alias).unwrap();
    assert_eq!(reg.layers(), 2);
    assert_eq!(reg.owned_bytes, layer_bytes(2560, 640));
    assert_eq!(reg.alias_bytes, layer_bytes(2560, 640));
}

#[test]
fn flash_next_resident_bytes() {
    assert_eq!(layer_bytes(2560, 640), 9_830_400);
    assert_eq!(48 * layer_bytes(2560, 640), 471_859_200);
}

#[test]
fn lookup_is_inert_while_the_selector_is_off() {
    // The test process never sets the selector.
    if selected().unwrap_or(true) {
        return;
    }
    assert_eq!(lookup(DevicePtr(0x9000)), None);
}

#[test]
fn receipt_requires_every_target_layer() {
    assert!(check_complete(48, 48).is_ok());
    assert!(check_complete(47, 48).is_err());
    assert!(
        check_complete(0, 0).is_err(),
        "planned but nothing registered"
    );
}

#[test]
fn fakequant_shared_v7_conflict_is_refused() {
    assert!(refuse_conflict(false).is_ok());
    assert!(refuse_conflict(true).is_err());
}
