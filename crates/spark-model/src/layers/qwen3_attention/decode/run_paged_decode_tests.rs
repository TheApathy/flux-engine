// SPDX-License-Identifier: AGPL-3.0-only

use super::{
    Gqa12RaggedMode, bf16_gqa4_geometry, compute_num_splits_for,
    flat_window_crosses_k1_split_boundary_for, flat_window_min_seq_len_host, gqa12_ragged_eligible,
    gqa12_ragged_grid_and_workspace_bytes, gqa12_ragged_mode_from,
};

#[test]
fn bf16_gqa4_geometry_matches_the_runtime_route_and_rejects_zero_kv_heads() {
    assert!(bf16_gqa4_geometry(2, 0, 256, 24, 2));
    assert!(!bf16_gqa4_geometry(1, 0, 256, 24, 2));
    assert!(!bf16_gqa4_geometry(2, 1, 256, 24, 2));
    assert!(!bf16_gqa4_geometry(2, 0, 128, 24, 2));
    assert!(!bf16_gqa4_geometry(2, 0, 256, 24, 0));
    assert!(!bf16_gqa4_geometry(2, 0, 256, 28, 6));
}

#[test]
fn default_workspace_regime_keeps_occupancy_policy() {
    assert_eq!(compute_num_splits_for(24, 1, 94, false), 2);
    assert_eq!(compute_num_splits_for(24, 5, 98, false), 1);
    assert_eq!(compute_num_splits_for(4, 3, 98, false), 4);
}

#[test]
fn aggressive_regime_matches_independent_k1_geometry() {
    for seq_len in [1, 94, 511, 512, 513, 1536, 32_768] {
        let k1 = compute_num_splits_for(24, 1, seq_len, true);
        for rows in [2, 3, 5, 17, 32] {
            assert_eq!(
                compute_num_splits_for(24, rows, seq_len, true),
                k1,
                "rows={rows} seq_len={seq_len}"
            );
        }
    }
}

#[test]
fn aggressive_regime_retains_sequence_and_cap_bounds() {
    assert_eq!(compute_num_splits_for(24, 5, 98, true), 2);
    assert_eq!(compute_num_splits_for(24, 5, 1536, true), 3);
    assert_eq!(compute_num_splits_for(24, 5, u32::MAX, true), 64);
    assert_eq!(compute_num_splits_for(4, 32, u32::MAX, true), 64);
    assert_eq!(compute_num_splits_for(16, 1, 136_000, true), 16);
}

#[test]
fn aggressive_regime_does_not_overcommit_small_head_arena() {
    assert_eq!(compute_num_splits_for(4, 32, 98, true), 1);
    assert_ne!(compute_num_splits_for(4, 32, 98, true), 12);
}

#[test]
fn qwen_flat_window_detects_sequence_split_boundaries() {
    assert_eq!(flat_window_min_seq_len_host(5, 1025), 1021);
    assert!(!flat_window_crosses_k1_split_boundary_for(
        24, 5, 1024, true
    ));
    assert!(flat_window_crosses_k1_split_boundary_for(24, 5, 1025, true));
    assert!(!flat_window_crosses_k1_split_boundary_for(
        24, 5, 1029, true
    ));
    assert!(!flat_window_crosses_k1_split_boundary_for(
        24, 5, 1025, false
    ));
    assert!(!flat_window_crosses_k1_split_boundary_for(
        4, 32, 1025, true
    ));
}

#[test]
fn gqa12_ragged_policy_engages_for_c8_to_c64_and_rejects_tree_or_bad_shapes() {
    assert_eq!(gqa12_ragged_mode_from(None), Gqa12RaggedMode::Auto);
    assert_eq!(gqa12_ragged_mode_from(Some("1")), Gqa12RaggedMode::ForceOn);
    assert_eq!(
        gqa12_ragged_mode_from(Some("true")),
        Gqa12RaggedMode::ForceOn
    );
    assert_eq!(gqa12_ragged_mode_from(Some("0")), Gqa12RaggedMode::ForceOff);

    // 12:1 Flash-Next (24 Q / 2 KV, HDIM=256): auto-on for C=8..256, off for C<8 or C>256
    for c in [8, 16, 32, 64, 128, 256] {
        assert!(gqa12_ragged_eligible(
            Gqa12RaggedMode::Auto,
            c,
            256,
            24,
            2,
            false
        ));
        assert!(gqa12_ragged_eligible(
            Gqa12RaggedMode::Auto,
            c,
            256,
            16,
            2,
            false
        ));
    }
    for c in [0, 1, 4, 5, 7, 257] {
        assert!(!gqa12_ragged_eligible(
            Gqa12RaggedMode::Auto,
            c,
            256,
            24,
            2,
            false
        ));
    }
    // 6:1 Qwen3.8-27B stays on legacy verify in Auto mode (protecting M=17 verify parity),
    // but engages when explicitly requested via ForceOn.
    assert!(!gqa12_ragged_eligible(
        Gqa12RaggedMode::Auto,
        17,
        256,
        24,
        4,
        false
    ));
    assert!(gqa12_ragged_eligible(
        Gqa12RaggedMode::ForceOn,
        16,
        256,
        24,
        4,
        false
    ));
    assert!(!gqa12_ragged_eligible(
        Gqa12RaggedMode::ForceOff,
        32,
        256,
        24,
        2,
        false
    ));

    // Tree indirection, zero KV heads, non-256 HDIM, and indivisible head counts reject.
    assert!(!gqa12_ragged_eligible(
        Gqa12RaggedMode::ForceOn,
        16,
        256,
        24,
        2,
        true
    ));
    assert!(!gqa12_ragged_eligible(
        Gqa12RaggedMode::ForceOn,
        16,
        128,
        24,
        2,
        false
    ));
    assert!(!gqa12_ragged_eligible(
        Gqa12RaggedMode::ForceOn,
        16,
        256,
        24,
        0,
        false
    ));
    assert!(!gqa12_ragged_eligible(
        Gqa12RaggedMode::ForceOn,
        16,
        256,
        24,
        5,
        false
    ));
}

#[test]
fn gqa12_ragged_grid_and_workspace_bounds_fit_shared_scratch() {
    // 24 Q / 2 KV -> 2 groups, 48 partitions -> 96 CTAs (2/SM on GB10), 2,377,728 bytes
    let (parts, groups, bytes) = gqa12_ragged_grid_and_workspace_bytes(24, 2, 256).unwrap();
    assert_eq!((parts, groups), (48, 2));
    assert_eq!(bytes, 2 * 96 * 12 * 258 * 4);
    assert!(bytes < 3 * 1024 * 1024);

    // 24 Q / 4 KV -> 4 groups, 24 partitions -> 96 CTAs, identical 2,377,728 bytes
    let (parts6, groups6, bytes6) = gqa12_ragged_grid_and_workspace_bytes(24, 4, 256).unwrap();
    assert_eq!((parts6, groups6), (24, 4));
    assert_eq!(bytes6, bytes);

    assert!(gqa12_ragged_grid_and_workspace_bytes(24, 0, 256).is_none());
    assert!(gqa12_ragged_grid_and_workspace_bytes(24, 5, 256).is_none());
}
