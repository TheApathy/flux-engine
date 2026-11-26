// SPDX-License-Identifier: AGPL-3.0-only

use super::DraftRowLayout;
use crate::weight_loader::DrafterCheckpointFamily;

#[test]
fn generic_dflash_keeps_anchor_plus_gamma_masks_and_drops_anchor_output() {
    let layout = DraftRowLayout::for_family(DrafterCheckpointFamily::Dflash, 15);
    assert_eq!(layout.query_rows, 16);
    assert_eq!(layout.output_start, 1);
    assert_eq!(layout.feedback_rows(), 15);
}

#[test]
fn dspark_uses_gamma_total_rows_and_includes_anchor_output() {
    let layout = DraftRowLayout::for_family(DrafterCheckpointFamily::Dspark, 7);
    assert_eq!(layout.query_rows, 7);
    assert_eq!(layout.output_start, 0);
    assert_eq!(layout.feedback_rows(), 6);
}

#[test]
fn dspark_k14_is_one_anchor_plus_thirteen_masks_not_two_blocks() {
    let layout = DraftRowLayout::for_family(DrafterCheckpointFamily::Dspark, 14);
    assert_eq!(layout.query_rows, 14);
    assert_eq!(layout.output_start, 0);
    assert_eq!(layout.feedback_rows(), 13);
}

#[test]
fn c1_stage2_bundle_packs_meta_gumbel_q_base_and_ring_into_one_h2d_and_stage1_flag() {
    use super::{DflashNoiseGraphMeta, NoiseGraphHeadState, parse_dflash_stage1_graph_enabled};
    use spark_runtime::gpu::mock::MockGpuBackend;
    use spark_runtime::gpu::{DevicePtr, GpuBackend};
    use spark_runtime::sampler::gumbel::GumbelDraftSpec;

    assert!(parse_dflash_stage1_graph_enabled(None));
    assert!(parse_dflash_stage1_graph_enabled(Some("1")));
    assert!(!parse_dflash_stage1_graph_enabled(Some("0")));
    assert!(!parse_dflash_stage1_graph_enabled(Some("off")));
    assert!(!parse_dflash_stage1_graph_enabled(Some("FALSE")));

    let gpu = MockGpuBackend::new();
    let mut state = NoiseGraphHeadState::new();
    let meta = DflashNoiseGraphMeta::try_new(128, 42, &[128, 129, 130, 131], 4096).unwrap();
    let k_ptrs: Vec<DevicePtr> = (0..5).map(|i| DevicePtr(0x10_0000 + i * 0x1000)).collect();
    let v_ptrs: Vec<DevicePtr> = (0..5).map(|i| DevicePtr(0x20_0000 + i * 0x1000)).collect();
    let spec = GumbelDraftSpec {
        seed: 0x0102_0304_0506_0708,
        step0_abs_pos: 128,
        temperature: 0.7,
        top_k: 32,
        top_p: 0.95,
        min_p: 0.05,
    };

    // Warm mask upload once; subsequent steady-state bundle upload must issue 1 H2D.
    let _ = state
        .upload_c1_stage2_bundle(
            &gpu,
            &meta,
            999,
            Some((64, &k_ptrs, &v_ptrs)),
            Some(spec),
            17,
            5,
            0,
        )
        .unwrap();
    let ptrs = state
        .upload_c1_stage2_bundle(
            &gpu,
            &meta,
            999,
            Some((72, &k_ptrs, &v_ptrs)),
            Some(spec),
            23,
            5,
            0,
        )
        .unwrap();

    let mut u32_buf = [0u8; 4];
    let mut u64_buf = [0u8; 8];
    gpu.copy_d2h(
        ptrs.meta_dev.offset(DflashNoiseGraphMeta::EFF_CTX_OFFSET),
        &mut u32_buf,
    )
    .unwrap();
    assert_eq!(u32::from_le_bytes(u32_buf), 128);
    gpu.copy_d2h(
        ptrs.meta_dev
            .offset(DflashNoiseGraphMeta::LAST_TOKEN_OFFSET),
        &mut u32_buf,
    )
    .unwrap();
    assert_eq!(u32::from_le_bytes(u32_buf), 42);
    gpu.copy_d2h(ptrs.gumbel_ptrs.seeds, &mut u64_buf).unwrap();
    assert_eq!(u64::from_le_bytes(u64_buf), spec.seed);
    gpu.copy_d2h(ptrs.gumbel_ptrs.step0_pos, &mut u64_buf)
        .unwrap();
    assert_eq!(u64::from_le_bytes(u64_buf), 128);
    gpu.copy_d2h(ptrs.gumbel_ptrs.temps, &mut u32_buf).unwrap();
    assert_eq!(f32::from_le_bytes(u32_buf).to_bits(), 0.7f32.to_bits());
    gpu.copy_d2h(ptrs.gumbel_ptrs.top_ks, &mut u32_buf).unwrap();
    assert_eq!(i32::from_le_bytes(u32_buf), 32);
    gpu.copy_d2h(ptrs.q_row_base_dev, &mut u32_buf).unwrap();
    assert_eq!(u32::from_le_bytes(u32_buf), 23);
    let ring = ptrs.ring_meta_ptrs.expect("ring_meta_ptrs present");
    gpu.copy_d2h(ring.ring_start_dev, &mut u32_buf).unwrap();
    assert_eq!(u32::from_le_bytes(u32_buf), 72);
    gpu.copy_d2h(ring.layer_k_ptr(4), &mut u64_buf).unwrap();
    assert_eq!(u64::from_le_bytes(u64_buf), k_ptrs[4].0);
    gpu.copy_d2h(ring.layer_v_ptr(4), &mut u64_buf).unwrap();
    assert_eq!(u64::from_le_bytes(u64_buf), v_ptrs[4].0);
}
