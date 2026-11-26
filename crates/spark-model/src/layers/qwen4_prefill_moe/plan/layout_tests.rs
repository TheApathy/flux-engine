// SPDX-License-Identifier: AGPL-3.0-only

use super::{Limits, Plan, parse_selector, validate_conflict};

fn limits() -> Limits {
    Limits {
        capacity: 2048,
        qkv: 2048 * 32768,
        norm: 2048 * 5120,
        hidden: 2048 * 20480,
        residual: 2048 * 20480,
        output: 2048 * 5120,
    }
}

#[test]
fn explicit_selector_is_default_off_and_rejects_typos() {
    assert_eq!(parse_selector(None), Ok(false));
    assert_eq!(parse_selector(Some("0")), Ok(false));
    assert_eq!(parse_selector(Some("1")), Ok(true));
    for bad in ["", "true", "2", "01", " 1", "1\n"] {
        assert!(parse_selector(Some(bad)).is_err(), "{bad:?}");
    }
}

#[test]
fn isolated_experiment_rejects_active_or_malformed_competing_flags() {
    for value in [None, Some("0")] {
        assert!(validate_conflict(value, false).is_ok());
    }
    for bad in ["1", "2", "true", "", " 0"] {
        assert!(validate_conflict(Some(bad), false).is_err());
    }
    assert!(validate_conflict(None, true).is_ok());
    assert!(validate_conflict(Some("1"), true).is_ok());
    for bad in ["0", "2", "true", ""] {
        assert!(validate_conflict(Some(bad), true).is_err());
    }
}

#[test]
fn staging_starts_after_decode_hyper_projection_and_fits_exactly() {
    let p = Plan::new(2048, 0, false, 2560, 10240, limits()).unwrap();
    assert_eq!(p.row_bytes, 20480);
    assert_eq!(p.core_bytes, 5120);
    assert_eq!(p.staging_offset, 20480);
    assert_eq!(p.input_bytes, 10485760);
    let mut tight = limits();
    tight.qkv = p.staging_offset + p.input_bytes;
    assert!(Plan::new(2048, 0, false, 2560, 10240, tight).is_ok());
    tight.qkv -= 1;
    assert!(Plan::new(2048, 0, false, 2560, 10240, tight).is_err());
}

#[test]
fn row_count_and_dense_window_are_bounded_without_overflow() {
    for rows in [0, 1, 2049, usize::MAX] {
        assert!(Plan::new(rows, 0, false, 2560, 10240, limits()).is_err());
    }
    assert!(Plan::new(2, 2046, false, 2560, 10240, limits()).is_ok());
    assert!(Plan::new(2, 2047, false, 2560, 10240, limits()).is_err());
    assert!(Plan::new(2, usize::MAX, false, 2560, 10240, limits()).is_err());
    assert!(Plan::new(2, 0, false, usize::MAX, 10240, limits()).is_err());
    assert!(Plan::new(2, 0, false, 2560, usize::MAX, limits()).is_err());
    assert!(Plan::new(2, 0, false, 0, 10240, limits()).is_err());
    assert!(Plan::new(2, 0, false, 2560, 0, limits()).is_err());
}

#[test]
fn every_live_arena_is_checked_before_staging() {
    let base = limits();
    for short in [
        Limits {
            capacity: 1,
            ..base
        },
        Limits {
            norm: 10239,
            ..base
        },
        Limits {
            output: 10239,
            ..base
        },
        Limits {
            hidden: 40959,
            ..base
        },
        Limits {
            residual: 40959,
            ..base
        },
    ] {
        assert!(Plan::new(2, 0, false, 2560, 10240, short).is_err());
    }
}
