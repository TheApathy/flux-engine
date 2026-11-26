// SPDX-License-Identifier: AGPL-3.0-only

use super::{Plan, grid32_partition, parse_selector, validate_regions, validate_request};
use std::ffi::OsStr;

#[test]
fn selector_is_explicit_and_rejects_malformed_values() {
    assert_eq!(parse_selector(None), Ok(false));
    assert_eq!(parse_selector(Some(OsStr::new("0"))), Ok(false));
    assert_eq!(parse_selector(Some(OsStr::new("1"))), Ok(true));
    for value in ["", "true", "01", " 1", "1 "] {
        assert!(parse_selector(Some(OsStr::new(value))).is_err());
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        assert!(parse_selector(Some(OsStr::from_bytes(&[255]))).is_err());
    }
}

#[test]
fn request_bounds_admit_singleton_but_staging_requires_multiple_rows() {
    assert!(validate_request(1, 0, false).is_ok());
    assert!(validate_request(1, 2047, false).is_ok());
    for (rows, start) in [(0, 0), (2049, 0), (2, 2047), (1, usize::MAX)] {
        assert!(validate_request(rows, start, false).is_err());
    }
    assert!(Plan::new(1, 2048, [usize::MAX; 6]).is_err());
    assert!(Plan::new(2048, 2047, [usize::MAX; 6]).is_err());
}

#[test]
fn tile_tail_coverage_and_all_arenas_are_exact() {
    for rows in [2, 4, 5, 8, 9, 17, 18, 32, 33, 63, 64, 65, 2048] {
        let p = Plan::new(rows, rows, [usize::MAX; 6]).unwrap();
        let tiles: Vec<_> = p.tiles().collect();
        assert_eq!(tiles.iter().map(|(_, count)| count).sum::<usize>(), rows);
        assert_eq!(tiles[0].0, 0);
        assert!(tiles.iter().all(|(_, count)| (1..=32).contains(count)));
        assert_eq!(tiles.last().unwrap().0 + tiles.last().unwrap().1, rows);
        assert!(Plan::new(rows, rows, p.bytes).is_ok());
        for index in 0..6 {
            let mut limits = p.bytes;
            limits[index] -= 1;
            assert!(Plan::new(rows, rows, limits).is_err());
        }
    }
    let p = Plan::new(2048, 2048, [usize::MAX; 6]).unwrap();
    assert_eq!(
        p.bytes,
        [
            10_485_760,
            67_108_864,
            134_217_728,
            786_432,
            25_165_824,
            10_485_760
        ]
    );
    assert_eq!(p.residual_bytes, 41_943_040);
    assert_eq!(Plan::H_STATE_BYTES, 3_145_728);
    assert_eq!(Plan::CONV_STATE_BYTES, 163_840);
    assert_eq!(Plan::CONV_DIM + Plan::VALUE_DIM, Plan::QKVZ);
}

#[test]
fn grid32_partition_covers_only_complete_tiles_and_one_incumbent_tail() {
    for rows in [2, 31, 32, 33, 63, 64, 65, 2047, 2048] {
        let (full_rows, tail) = grid32_partition(rows);
        assert_eq!(full_rows + tail, rows);
        assert_eq!(full_rows % 32, 0);
        assert!(tail < 32);
    }
    assert_eq!(grid32_partition(2048), (2048, 0));
    assert_eq!(grid32_partition(2047), (2016, 31));
}

#[test]
fn typed_regions_reject_aliasing_alignment_null_and_overflow() {
    assert!(validate_regions(&[(1024, 32, 2), (1056, 16, 4)]).is_ok());
    for regions in [
        vec![(0, 32, 2)],
        vec![(1024, 0, 2)],
        vec![(1025, 16, 2)],
        vec![(1026, 16, 4)],
        vec![(1024, 32, 2), (1054, 16, 2)],
        vec![(u64::MAX - 3, 8, 4)],
    ] {
        assert!(validate_regions(&regions).is_err());
    }
}

#[test]
fn tp2_8_24_head_geometry_scales_conv_and_value_arenas_proportionally() {
    let p_tp1 = Plan::new_for_heads(2048, 2048, [usize::MAX; 6], 16, 48).unwrap();
    let p_tp2 = Plan::new_for_heads(2048, 2048, [usize::MAX; 6], 8, 24).unwrap();
    assert_eq!(p_tp2.shape.nk, 8);
    assert_eq!(p_tp2.shape.nv, 24);
    assert_eq!(p_tp2.shape.conv_dim, 5_120);
    assert_eq!(p_tp2.shape.value_dim, 3_072);
    assert_eq!(p_tp2.shape.qkvz, 8_192);
    assert_eq!(p_tp2.shape.gates, 48);
    assert_eq!(p_tp2.shape.h_state_bytes, p_tp1.shape.h_state_bytes / 2);
    assert_eq!(
        p_tp2.shape.conv_state_bytes,
        p_tp1.shape.conv_state_bytes / 2
    );
    // Input/output hidden_size [rows, 2560] stay full width; QKVZ/BA/out_sub halve.
    assert_eq!(p_tp2.bytes[0], p_tp1.bytes[0]);
    assert_eq!(p_tp2.bytes[5], p_tp1.bytes[5]);
    assert_eq!(p_tp2.bytes[1], p_tp1.bytes[1] / 2);
    assert_eq!(p_tp2.bytes[2], p_tp1.bytes[2] / 2);
    assert_eq!(p_tp2.bytes[3], p_tp1.bytes[3] / 2);
    assert_eq!(p_tp2.bytes[4], p_tp1.bytes[4] / 2);
}
