// SPDX-License-Identifier: AGPL-3.0-only

use super::{
    Qwen4DeviceAttentionRoute, Qwen4K5DeviceAttentionPlan, Qwen4K5DeviceAttentionTopology,
    VERIFY_MAX_ROWS,
};
use anyhow::Result;
use std::collections::HashSet;

#[test]
fn k5_k7_k8_k9_plans_classify_dense_boundary_and_sparse_seq_lens() -> Result<()> {
    for k in [5usize, 7, 8, 9] {
        for seq_len in [64usize, 2_045, 2_052, 8_192] {
            let plan = Qwen4K5DeviceAttentionPlan::from_validated_rows(k, seq_len)?;
            assert_eq!(plan.num_rows, k);
            assert!(plan.row(k).is_err(), "K={k} seq_len={seq_len}");
            assert!(plan.row(VERIFY_MAX_ROWS).is_err());

            let mut expected_mask = 0u16;
            for r in 0..k {
                let row = plan.row(r)?;
                let pos = seq_len + r;
                let cur_len = pos + 1;
                let vis = cur_len / 4;
                let expected_grid = if vis > 512 {
                    u32::try_from(vis.div_ceil(8))?.next_power_of_two()
                } else {
                    0
                };
                let expected_route = if pos >= 2_048 {
                    Qwen4DeviceAttentionRoute::SparseQsa
                } else {
                    Qwen4DeviceAttentionRoute::DensePaged { num_splits: 2 }
                };
                assert_eq!(row.qsa_pool_endpoint, cur_len.is_multiple_of(4));
                assert_eq!(row.qsa_visible_groups, u32::try_from(vis)?);
                assert_eq!(row.qsa_score_grid_x, expected_grid);
                assert_eq!(row.route, expected_route);
                if row.qsa_pool_endpoint {
                    expected_mask |= 1u16 << r;
                }
            }
            assert_eq!(plan.max_visible_groups(), ((seq_len + k) / 4) as u32);

            let topo = plan.topology();
            assert_eq!(topo.num_rows, k as u8);
            assert_eq!(topo.pool_endpoint_mask, expected_mask);
            assert_eq!(topo.pool_endpoint_mask >> k, 0);
            assert!(topo.score_grids[k..].iter().all(|&g| g == 0));
            assert!(
                topo.routes[k..]
                    .iter()
                    .all(|&r| r == Qwen4DeviceAttentionRoute::SparseQsa)
            );
        }
    }

    // Spot-check the 2048 sparse transition and score-grid boundary at seq_len=2045:
    // rows 0..=2 (pos 2045..=2047) are DensePaged, rows 3..k (pos 2048..) are SparseQsa,
    // and row 6 (pos 2051, seq_len=2052, vis=513) is the first row with score_grid=128.
    let k5_boundary = Qwen4K5DeviceAttentionPlan::from_validated_rows(5, 2_045)?;
    assert_eq!(
        k5_boundary.row(2)?.route,
        Qwen4DeviceAttentionRoute::DensePaged { num_splits: 2 }
    );
    assert_eq!(
        k5_boundary.row(3)?.route,
        Qwen4DeviceAttentionRoute::SparseQsa
    );
    assert_eq!(k5_boundary.row(4)?.qsa_score_grid_x, 0);

    let k7_boundary = Qwen4K5DeviceAttentionPlan::from_validated_rows(7, 2_045)?;
    assert_eq!(k7_boundary.row(5)?.qsa_score_grid_x, 0);
    assert_eq!(k7_boundary.row(6)?.qsa_score_grid_x, 128);

    // Spot-check seq_len=8192 score-grid power-of-two jump across rows 2 -> 3:
    // row 2 (pos 8194, cur_len 8195, vis 2048 -> 256) vs row 3 (pos 8195, cur_len 8196, vis 2049 -> 512).
    let k9_long = Qwen4K5DeviceAttentionPlan::from_validated_rows(9, 8_192)?;
    assert_eq!(k9_long.row(2)?.qsa_score_grid_x, 256);
    assert_eq!(k9_long.row(3)?.qsa_score_grid_x, 512);
    assert_eq!(k9_long.row(8)?.qsa_score_grid_x, 512);
    Ok(())
}

#[test]
fn topology_key_distinguishes_row_counts_u16_bit8_and_replay_boundaries() -> Result<()> {
    for seq_len in [64usize, 2_045, 2_052, 8_192] {
        let mut seen: HashSet<Qwen4K5DeviceAttentionTopology> = HashSet::new();
        for k in [5usize, 7, 8, 9] {
            let topo = Qwen4K5DeviceAttentionPlan::from_validated_rows(k, seq_len)?.topology();
            assert!(
                seen.insert(topo),
                "duplicate topology at seq_len={seq_len} for K={k}"
            );
        }
        assert_eq!(seen.len(), 4);
    }

    // K=9 requires u16 pool_endpoint_mask so bit 8 (row 8) does not overflow u8.
    // At seq_len=67, cur_len = 68..=76 => rows 0, 4, 8 are multiples of 4.
    let k9_bit8 = Qwen4K5DeviceAttentionPlan::from_validated_rows(9, 67)?.topology();
    assert_eq!(k9_bit8.pool_endpoint_mask, 1u16 | (1u16 << 4) | (1u16 << 8));

    // Replay stability across 4-token group strides and invalidation on phase/grid changes.
    for k in [5usize, 7, 8, 9] {
        let t_2992 = Qwen4K5DeviceAttentionPlan::from_validated_rows(k, 2_992)?.topology();
        let t_2996 = Qwen4K5DeviceAttentionPlan::from_validated_rows(k, 2_996)?.topology();
        let t_3000 = Qwen4K5DeviceAttentionPlan::from_validated_rows(k, 3_000)?.topology();
        let t_2993 = Qwen4K5DeviceAttentionPlan::from_validated_rows(k, 2_993)?.topology();
        assert_eq!(t_2992, t_2996, "K={k}");
        assert_eq!(t_2996, t_3000, "K={k}");
        assert_ne!(t_2992, t_2993, "K={k}");

        // Crossing into the 2048 sparse regime changes topology.
        let before = Qwen4K5DeviceAttentionPlan::from_validated_rows(k, 2_048 - k)?.topology();
        let crossing = Qwen4K5DeviceAttentionPlan::from_validated_rows(k, 2_049 - k)?.topology();
        assert_ne!(before, crossing, "K={k}");
    }
    Ok(())
}

#[test]
fn plan_rejects_out_of_range_widths_and_position_overflow() -> Result<()> {
    for bad_k in [0usize, 1, VERIFY_MAX_ROWS + 1, 16] {
        assert!(Qwen4K5DeviceAttentionPlan::from_validated_rows(bad_k, 64).is_err());
    }
    for good_k in 2..=VERIFY_MAX_ROWS {
        let plan = Qwen4K5DeviceAttentionPlan::from_validated_rows(good_k, 64)?;
        assert_eq!(plan.num_rows, good_k);
    }
    assert!(Qwen4K5DeviceAttentionPlan::from_validated_rows(9, usize::MAX - 2).is_err());
    Ok(())
}
