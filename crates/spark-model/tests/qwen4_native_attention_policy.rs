// SPDX-License-Identifier: AGPL-3.0-only

#[path = "../src/layers/qwen3_attention/trait_impl/multi_seq/qwen4_native_policy.rs"]
mod policy;

use policy::{
    Admission, QWEN4_HC_COUNT, QWEN4_NATIVE_MAX_ROWS, QWEN4_QSA_SPARSE_START, RowLayout, validate,
};
use spark_model::layers::ops::{
    ExactLmHeadRoute, ExactLmHeadTier, W4a16ExactAttentionKernels, W4a16ExactLmHeadKernels,
};
use spark_runtime::gpu::KernelHandle;

fn admitted<'a>(rows: usize, seq_lens: &'a [usize]) -> Admission<'a> {
    Admission {
        rows,
        metadata_rows: rows,
        seq_lens,
        block_table_rows: rows,
        qsa_required: false,
        is_qwen4: true,
        bf16_kv: true,
        bf16_residual: true,
        tp_world_size: 1,
        ep_world_size: 1,
        comm_present: false,
        graph_capture: false,
        tree_aware: false,
        ddtree_parent_ids: false,
        self_spec_sparse: false,
        ffn_defer: false,
        mrope_interleaved: true,
        qkv_mega: false,
    }
}

#[test]
fn native_lane_admits_every_live_width_c2_through_c32() {
    for rows in 2..=QWEN4_NATIVE_MAX_ROWS {
        validate(admitted(rows, &vec![1; rows])).unwrap();
    }
    for rows in [0, 1, QWEN4_NATIVE_MAX_ROWS + 1, QWEN4_NATIVE_MAX_ROWS * 2] {
        assert!(validate(admitted(rows, &vec![1; rows])).is_err(), "C{rows}");
    }
}

#[test]
fn every_c2_c32_projection_width_has_a_c1_order_exact_tier() {
    let qkv = W4a16ExactAttentionKernels::new(KernelHandle(3), KernelHandle(4))
        .with_m4(KernelHandle(1), KernelHandle(2))
        .with_m32(KernelHandle(5), KernelHandle(6));
    let projections = W4a16ExactLmHeadKernels::new(
        KernelHandle(7),
        KernelHandle(8),
        KernelHandle(9),
        KernelHandle(10),
    );
    for rows in 2..=QWEN4_NATIVE_MAX_ROWS {
        let mut rem = rows;
        while rem > 0 {
            let chunk = if rem == 33 { 17 } else { rem.min(32) };
            let (qg, dual_kv) = match chunk {
                2..=4 => (1, 2),
                5..=17 => (3, 4),
                _ => (5, 6),
            };
            assert_eq!(qkv.qg_for_rows(chunk).0, qg, "QG C{rows} chunk={chunk}");
            assert_eq!(
                qkv.dual_kv_for_rows(chunk).0,
                dual_kv,
                "KV C{rows} chunk={chunk}"
            );
            rem -= chunk;
        }
        if rows > 32 {
            assert_eq!(
                qkv.qg_for_rows(rows).0,
                0,
                "untiled QG C{rows} must fail closed"
            );
            assert_eq!(
                qkv.dual_kv_for_rows(rows).0,
                0,
                "untiled KV C{rows} must fail closed"
            );
        }
        let tier = match rows {
            2..=4 => ExactLmHeadTier::M4,
            5..=8 => ExactLmHeadTier::M8,
            9..=17 => ExactLmHeadTier::M17,
            _ => ExactLmHeadTier::M32,
        };
        assert_eq!(
            projections.route_for_rows(rows as u32),
            Some(ExactLmHeadRoute::Exact(tier)),
            "projection C{rows}"
        );
    }
}

#[test]
fn qsa_horizon_and_physical_boundary_fail_closed() {
    let mut request_can_reach_qsa = admitted(2, &[10, 20]);
    request_can_reach_qsa.qsa_required = true;
    assert!(
        validate(request_can_reach_qsa)
            .unwrap_err()
            .contains("per-row QSA")
    );

    validate(admitted(2, &[QWEN4_QSA_SPARSE_START - 2, 7])).unwrap();
    assert!(
        validate(admitted(2, &[QWEN4_QSA_SPARSE_START - 1, 7]))
            .unwrap_err()
            .contains("sparse-QSA boundary")
    );
    assert!(validate(admitted(2, &[usize::MAX, 7])).is_err());
}

#[test]
fn topology_dtype_and_target_only_controls_fail_closed() {
    let mut a = admitted(2, &[1, 1]);
    a.bf16_kv = false;
    assert!(validate(a).is_err());

    let mut a = admitted(2, &[1, 1]);
    a.tp_world_size = 2;
    assert!(validate(a).is_err());

    let mut a = admitted(2, &[1, 1]);
    a.ep_world_size = 2;
    assert!(validate(a).is_err());

    let mut a = admitted(2, &[1, 1]);
    a.self_spec_sparse = true;
    assert!(validate(a).unwrap_err().contains("target-only"));

    let mut a = admitted(2, &[1, 1]);
    a.tree_aware = true;
    assert!(validate(a).is_err());
}

#[test]
fn qkv_mega_is_rejected_for_canonical_qwen4_mrope() {
    let mut a = admitted(2, &[1, 1]);
    a.qkv_mega = true;
    assert!(validate(a).unwrap_err().contains("ATLAS_ATTN_QKV_MEGA"));

    let mut scalar_rope = admitted(2, &[1, 1]);
    scalar_rope.mrope_interleaved = false;
    scalar_rope.qkv_mega = true;
    validate(scalar_rope).unwrap();
}

#[test]
fn row_metadata_must_be_exact_not_padded_or_truncated() {
    let mut a = admitted(4, &[1, 2, 3, 4]);
    a.metadata_rows = 8;
    assert!(validate(a).is_err());

    let mut a = admitted(4, &[1, 2, 3, 4]);
    a.block_table_rows = 3;
    assert!(validate(a).is_err());
}

#[test]
fn c2_through_c32_row_geometry_uses_independent_per_sequence_offsets() {
    let max_blocks = 4_097;
    let layout = RowLayout::new(2_560, QWEN4_HC_COUNT, max_blocks).unwrap();
    assert_eq!(layout.hyper_row_bytes, 20_480);
    assert_eq!(layout.core_row_bytes, 5_120);
    assert_eq!(layout.block_table_row_bytes, 16_388);

    assert_eq!(
        layout.offsets(0).unwrap(),
        policy::RowOffsets {
            hyper: 0,
            core: 0,
            position: 0,
            slot: 0,
            seq_len: 0,
            block_table: 0,
        }
    );
    assert_eq!(
        layout.offsets(1).unwrap(),
        policy::RowOffsets {
            hyper: 20_480,
            core: 5_120,
            position: 4,
            slot: 8,
            seq_len: 4,
            block_table: 16_388,
        }
    );
    assert_eq!(layout.extents(2).unwrap().hyper, 40_960);
    assert_eq!(layout.extents(2).unwrap().core, 10_240);

    let row3 = layout.offsets(3).unwrap();
    assert_eq!(row3.hyper, 61_440);
    assert_eq!(row3.core, 15_360);
    assert_eq!(row3.position, 12);
    assert_eq!(row3.slot, 24);
    assert_eq!(row3.seq_len, 12);
    assert_eq!(row3.block_table, 49_164);
    let c4 = layout.extents(4).unwrap();
    assert_eq!(c4.hyper, 81_920);
    assert_eq!(c4.core, 20_480);
    assert_eq!(c4.positions, 16);
    assert_eq!(c4.slots, 32);
    assert_eq!(c4.seq_lens, 16);
    assert_eq!(c4.block_table, 65_552);

    let c8 = layout.extents(8).unwrap();
    assert_eq!(c8.hyper, 163_840);
    assert_eq!(c8.core, 40_960);
    assert_eq!(c8.positions, 32);
    assert_eq!(c8.slots, 64);
    assert_eq!(c8.seq_lens, 32);
    assert_eq!(c8.block_table, 131_104);

    let c32 = layout.extents(32).unwrap();
    assert_eq!(c32.hyper, 655_360);
    assert_eq!(c32.core, 163_840);
    assert_eq!(c32.positions, 128);
    assert_eq!(c32.slots, 256);
    assert_eq!(c32.seq_lens, 128);
    assert_eq!(c32.block_table, 524_416);
}

#[test]
fn row_geometry_fails_closed_on_wrong_hc_or_overflow() {
    assert!(RowLayout::new(2_560, 1, 16).is_err());
    assert!(RowLayout::new(0, QWEN4_HC_COUNT, 16).is_err());
    assert!(RowLayout::new(2_560, QWEN4_HC_COUNT, 0).is_err());
    assert!(RowLayout::new(usize::MAX, QWEN4_HC_COUNT, 16).is_err());
    assert!(RowLayout::new(2_560, QWEN4_HC_COUNT, usize::MAX).is_err());

    let layout = RowLayout::new(2_560, QWEN4_HC_COUNT, 16).unwrap();
    assert!(layout.offsets(usize::MAX).is_err());
    assert!(layout.extents(usize::MAX).is_err());
}

#[test]
fn serving_route_dispatches_native_batch_and_retains_serial_fallback() {
    let dispatch = include_str!("../src/model/trait_impl/decode_a2.rs");
    assert!(dispatch.contains("return self.decode_qwen4_native_batch_dispatch(tokens, seqs);"));
    assert!(dispatch.contains("self.config.hc_mult > 0"));
    assert!(dispatch.contains("return self.decode_batch_serial_rows(tokens, seqs);"));
}

#[test]
fn attention_seam_orders_exact_projection_before_cache_and_injection() {
    let source = include_str!("../src/layers/qwen3_attention/trait_impl/multi_seq/qwen4_native.rs");
    let prepare = source.find("prepare_batched_into(").unwrap();
    let qkv = source.find("ms_phase_qkv_qwen4_native_exact(&c)").unwrap();
    let cache = source.find("ms_phase_cache_write_qwen4_native(&c").unwrap();
    let paged = source.find("ms_phase_paged_decode(&c").unwrap();
    let inject = source.find("inject_saved_batched(").unwrap();
    assert!(prepare < qkv && qkv < cache && cache < paged && paged < inject);
}

#[test]
fn all_admission_and_extent_rejections_precede_first_mutation() {
    let source = include_str!("../src/layers/qwen3_attention/trait_impl/multi_seq/qwen4_native.rs");
    let mutation = source.find("prepare_batched_into(").unwrap();
    for preflight in [
        "validate(Admission {",
        "RowLayout::new(",
        "row_layout.extents(num_seqs)",
        "row_extents.hyper <= arena.hidden_states",
        "native Qwen4 attention metadata pointers must be non-null",
        "row_extents.hyper <= arena.qkv_output",
        "native Qwen4 attention HC scratch overflow",
        "table.len() <= metadata.max_blocks_per_seq",
        "self.validate_yarn_multi_seq_controls()?",
        "self.preflight_qwen4_native_tail_selectors(num_seqs)?",
        "self.validate_qwen4_native_exact_qkv(num_seqs)?",
        "ATLAS_ATTN_QKV_MEGA is incompatible with native Qwen4 interleaved MRoPE",
        "ordinary_nvfp4_o",
        "seq_len / block_size_host < table.len()",
        "native Qwen4 attention sequence extent overflow",
        "native Qwen4 attention QKV extent overflow",
        "native Qwen4 attention query extent overflow",
        "row_extents.core <= arena.moe_output",
    ] {
        let at = source
            .find(preflight)
            .unwrap_or_else(|| panic!("missing preflight marker: {preflight}"));
        assert!(at < mutation, "preflight ran after mutation: {preflight}");
    }
    for gpu_work in [
        "ms_phase_qkv_qwen4_native_exact(&c)",
        "ms_phase_rope(&c",
        "ms_phase_cache_write_qwen4_native(&c",
        "ms_phase_paged_decode(&c",
        "ms_phase_o_proj(&c",
        "inject_saved_batched(",
    ] {
        assert!(
            source.find(gpu_work).unwrap() > mutation,
            "GPU work escaped preflight: {gpu_work}"
        );
    }
}
