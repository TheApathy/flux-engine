// SPDX-License-Identifier: AGPL-3.0-only

use super::{
    has_no_swap_lineage, native_qsa_routes, parse_native_batch_selector,
    select_native_target_width, validate_active_rows, validate_kv_block_ownership,
    validate_kv_rows_before_allocation,
};
use crate::layers::qwen4_qsa::native_batch::{
    NativeAttnRoute, NativeQsaMode, dense_policy_seq_lens,
};

#[test]
fn selector_is_exact_and_default_on() {
    assert!(parse_native_batch_selector(None).unwrap());
    assert!(!parse_native_batch_selector(Some("0")).unwrap());
    assert!(parse_native_batch_selector(Some("1")).unwrap());
    for hostile in ["", "true", "01", "2", " 1"] {
        assert!(parse_native_batch_selector(Some(hostile)).is_err());
    }
}

// Source-only (no cargo build at 8b4a4d11).
#[test]
fn typed_route_agrees_with_the_dispatch_row_validator() {
    use crate::qwen4_batch_policy::{
        FrameRoute, NativeBatchSelector, ProcessFacts, RowFacts, resolve_plan, route_frame,
    };
    let facts = ProcessFacts {
        profile: false,
        moe_exact_rows: true,
        ssm_h_fp16: false,
        lm_head_tc: false,
        lm_head_nvfp4: true,
        lm_head_full_vocab: true,
        qsa_index: true,
    };
    let plan = resolve_plan(NativeBatchSelector::Auto, &facts, false).unwrap();
    for (rows, seq_len) in [
        (2, 17),
        (9, 17),
        (32, 17),
        (2, 2_046),
        (2, 2_047),
        (4, usize::MAX),
    ] {
        let seq_lens = vec![seq_len; rows];
        let qsa = vec![false; rows];
        let slots: Vec<_> = (0..rows).collect();
        let facts: Vec<_> = seq_lens
            .iter()
            .map(|&seq_len| RowFacts {
                qsa_required: false,
                seq_len,
            })
            .collect();
        let native = matches!(route_frame(&plan, &facts), Ok(FrameRoute::Native));
        let valid = validate_active_rows(rows, &seq_lens, &slots, &qsa).is_ok();
        assert_eq!(native, valid, "C{rows} at seq_len {seq_len}");
    }
}

#[test]
fn selected_widths_admit_c2_c256_and_refuse_wider_frames() {
    assert!(!select_native_target_width(false, 128).unwrap());
    assert!(!select_native_target_width(true, 1).unwrap());
    for rows in 2..=crate::layers::QWEN4_NATIVE_MAX_ROWS {
        assert!(select_native_target_width(true, rows).unwrap(), "C{rows}");
    }
    let over_max = crate::layers::QWEN4_NATIVE_MAX_ROWS + 1;
    for rows in [over_max, over_max * 2, usize::MAX] {
        assert!(select_native_target_width(true, rows).is_err(), "C{rows}");
    }
}

#[test]
fn profiled_requests_are_never_native_runtime_eligible() {
    assert!(super::qwen4_native_batch_runtime_eligible(false));
    assert!(!super::qwen4_native_batch_runtime_eligible(true));
}

#[test]
fn active_row_policy_admits_every_live_c2_c256_width() {
    for rows in 2..=crate::layers::QWEN4_NATIVE_MAX_ROWS {
        let seq_lens = vec![17; rows];
        let slot_ids: Vec<_> = (0..rows).collect();
        assert!(
            validate_active_rows(rows, &seq_lens, &slot_ids, &vec![false; rows]).is_ok(),
            "C{rows}"
        );
    }
}

#[test]
fn active_row_policy_rejects_padding_qsa_boundary_and_aliases() {
    assert!(validate_active_rows(1, &[1], &[0], &[false]).is_err());
    let over_max = crate::layers::QWEN4_NATIVE_MAX_ROWS + 1;
    assert!(
        validate_active_rows(
            over_max,
            &vec![1; over_max],
            &(0..over_max).collect::<Vec<_>>(),
            &vec![false; over_max]
        )
        .is_err()
    );
    assert!(validate_active_rows(4, &[1; 4], &[0, 1, 2, 3], &[false; 3]).is_err());
    assert!(validate_active_rows(2, &[1, 2_047], &[0, 1], &[false; 2]).is_err());
    assert!(validate_active_rows(2, &[1, 2], &[0, 1], &[false, true]).is_err());
    assert!(validate_active_rows(2, &[1, 2], &[4, 4], &[false; 2]).is_err());
}

#[test]
fn no_swap_lineage_accepts_only_the_allocator_zero_cursor_shape() {
    assert!(has_no_swap_lineage(&[], &[0; 12], 12));
    assert!(!has_no_swap_lineage(&[], &[], 12));
    assert!(!has_no_swap_lineage(&[], &[0; 11], 12));
    assert!(!has_no_swap_lineage(&[], &[0; 13], 12));

    let mut advanced = [0; 12];
    advanced[7] = 1;
    assert!(!has_no_swap_lineage(&[], &advanced, 12));
    assert!(!has_no_swap_lineage(&[3], &[0; 12], 12));
}

#[test]
fn kv_preflight_counts_only_the_blocks_needed_for_this_decode() {
    assert_eq!(
        validate_kv_rows_before_allocation(&[0, 0], &[vec![], vec![]], 16, 8, 64),
        Ok(2)
    );
    assert_eq!(
        validate_kv_rows_before_allocation(&[15, 16], &[vec![3], vec![7]], 16, 8, 64),
        Ok(1)
    );
}

#[test]
fn kv_preflight_rejects_aliases_gaps_and_capacity_errors_before_growth() {
    assert!(
        validate_kv_rows_before_allocation(&[1, 1], &[vec![3], vec![3]], 16, 8, 64)
            .unwrap_err()
            .contains("alias")
    );
    assert!(
        validate_kv_rows_before_allocation(&[1, 1], &[vec![3, 3], vec![4]], 16, 8, 64)
            .unwrap_err()
            .contains("alias")
    );
    assert!(
        validate_kv_rows_before_allocation(&[1, 1], &[vec![], vec![4]], 16, 8, 64)
            .unwrap_err()
            .contains("existing tokens")
    );
    assert!(
        validate_kv_rows_before_allocation(&[1, 1], &[vec![64], vec![4]], 16, 8, 64)
            .unwrap_err()
            .contains("out-of-range")
    );
    assert!(
        validate_kv_rows_before_allocation(&[32, 1], &[vec![1, 2], vec![4]], 16, 2, 64)
            .unwrap_err()
            .contains("row capacity")
    );
    assert!(validate_kv_rows_before_allocation(&[0, 0], &[vec![], vec![]], 0, 8, 64).is_err());
    assert!(validate_kv_rows_before_allocation(&[0, 0], &[vec![], vec![]], 16, 0, 64).is_err());
}

#[test]
fn kv_preflight_rejects_dummy_freed_and_shared_blocks() {
    assert!(validate_kv_block_ownership(&[vec![3], vec![4]], 0, |_| 1).is_ok());
    assert!(
        validate_kv_block_ownership(&[vec![0], vec![4]], 0, |_| 1)
            .unwrap_err()
            .contains("dummy")
    );
    assert!(
        validate_kv_block_ownership(&[vec![3], vec![4]], 0, |block| {
            if block == 3 { 0 } else { 1 }
        })
        .unwrap_err()
        .contains("exclusively")
    );
    assert!(
        validate_kv_block_ownership(&[vec![3], vec![4]], 0, |block| {
            if block == 4 { 2 } else { 1 }
        })
        .unwrap_err()
        .contains("exclusively")
    );
}

#[test]
fn qsa_rows_stay_out_of_the_native_frame_with_the_selector_off() {
    assert_eq!(
        native_qsa_routes(NativeQsaMode::Off, &[false, true], &[1, 9_000]),
        Ok(None)
    );
    assert!(validate_active_rows(2, &[1, 9_000], &[0, 1], &[false, true]).is_err());
    assert_eq!(
        native_qsa_routes(NativeQsaMode::PerRow, &[false; 2], &[1, 2]),
        Ok(None)
    );
}

#[test]
fn qsa_row_frames_route_per_row_and_pass_the_dense_lane_policy() {
    let seq_lens = [17, 2_046, 2_047, 2_048, 2_051, 32_767];
    let required = [false, true, true, true, true, true];
    let routes = native_qsa_routes(NativeQsaMode::PerRow, &required, &seq_lens)
        .unwrap()
        .expect("QSA rows present");
    assert_eq!(
        routes,
        [
            NativeAttnRoute::Dense,
            NativeAttnRoute::IndexedDense,
            NativeAttnRoute::IndexedDense,
            NativeAttnRoute::Sparse,
            NativeAttnRoute::Sparse,
            NativeAttnRoute::Sparse,
        ]
    );
    let policy = dense_policy_seq_lens(&routes, &seq_lens).unwrap();
    let slots: Vec<_> = (0..seq_lens.len()).collect();
    assert!(validate_active_rows(6, &policy, &slots, &[false; 6]).is_ok());
    // A row without QSA still may not reach the sparse boundary.
    assert!(native_qsa_routes(NativeQsaMode::PerRow, &[false, true], &[2_047, 2_047]).is_err());
    // Slot aliasing is still rejected for QSA-row frames.
    assert!(validate_active_rows(2, &[0, 0], &[3, 3], &[false; 2]).is_err());
}
