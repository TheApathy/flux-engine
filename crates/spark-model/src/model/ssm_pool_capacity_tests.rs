// SPDX-License-Identifier: AGPL-3.0-only

use super::super::ssm_pool_geometry::*;
use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;
fn input() -> SsmPoolGeometryInput {
    SsmPoolGeometryInput {
        max_slots: 8,
        num_ssm_layers: 2,
        h_bytes: 8,
        conv_bytes: 4,
        has_mtp: false,
        compact_lazy_h: false,
        num_intermediates: 0,
        lazy_commit: false,
        num_key_heads: 0,
        key_head_dim: 0,
        num_value_heads: 0,
        value_head_dim: 0,
    }
}

fn expect_error(input: SsmPoolGeometryInput, expected: &str) {
    let error = checked_ssm_pool_geometry(input).unwrap_err();
    assert!(
        error.to_string().contains(expected),
        "wrong error for {input:?}: {error:#}"
    );
}
fn tiny_config(num_layers: usize, enabled_bytes: bool) -> ModelConfig {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.layer_types = if num_layers == 0 {
        vec![LayerType::FullAttention]
    } else {
        vec![LayerType::LinearAttention; num_layers]
    };
    config.linear_num_key_heads = usize::from(enabled_bytes);
    config.linear_key_head_dim = usize::from(enabled_bytes);
    config.linear_num_value_heads = usize::from(enabled_bytes);
    config.linear_value_head_dim = usize::from(enabled_bytes);
    config.linear_conv_kernel_dim = usize::from(enabled_bytes);
    config
}

#[test]
fn canonical_plain_geometry_is_dummy_inclusive_and_constructed_exactly() {
    let geometry = checked_ssm_pool_geometry(input()).unwrap();
    assert_eq!(geometry.total_slots, 9);
    assert_eq!(geometry.state_copies, 1);
    assert_eq!(geometry.total_bytes, 216);
    assert_eq!(geometry.kv_retain_bytes + geometry.gate_retain_bytes, 0);

    let gpu = MockGpuBackend::new();
    let config = tiny_config(2, true);
    let expected = checked_ssm_pool_geometry(
        SsmPoolGeometryInput::from_config(&config, 1, false, 0, false).unwrap(),
    )
    .unwrap();
    let pool = SsmStatePool::new(&config, 1, false, 0, &gpu).unwrap();
    assert_eq!(gpu.alloc_count(), 4);
    assert_eq!(
        gpu.read_alloc(pool.h_state_pools[0]).unwrap().len(),
        expected.h_state_allocation_bytes
    );
    assert_eq!(
        gpu.read_alloc(pool.conv_state_pools[0]).unwrap().len(),
        expected.conv_state_allocation_bytes
    );
}

#[test]
fn flat_tree_and_lazy_shapes_use_the_exact_shared_formula() {
    let flat = checked_ssm_speculative_geometry_with_uncap(true, true, 15, None, false).unwrap();
    assert_eq!(flat.dflash_verify_width, 16);
    assert_eq!(flat.ddtree_capacity, 16);
    assert_eq!(flat.num_intermediates, 17);

    // A requested wide DDTree does not inflate the intermediate pool while
    // `uncap_tree` (`ATLAS_DDTREE_UNCAP=1`) is false.
    //
    // `ddtree_capacity` still becomes 31 — the tree's own budget is unchanged —
    // but `num_intermediates` stays at the flat 17, because `draft_budget`
    // caps `tree_nodes` at `flat` unless `ATLAS_DDTREE_UNCAP=1`, so no node can
    // ever be placed in the extra slots.
    let wide =
        checked_ssm_speculative_geometry_with_uncap(true, true, 15, Some(31), false).unwrap();
    assert_eq!(
        wide.ddtree_capacity, 31,
        "the tree's own budget is untouched"
    );
    assert_eq!(
        wide.num_intermediates, 17,
        "but the pool is not sized for it"
    );

    let mut flat_input = input();
    flat_input.has_mtp = true;
    flat_input.num_intermediates = flat.num_intermediates;
    assert_eq!(
        checked_ssm_pool_geometry(flat_input).unwrap().total_bytes,
        4104
    );

    // Same bytes as flat now, which is the whole point: the wide request costs
    // nothing until the tree can use it.
    let mut wide_input = input();
    wide_input.has_mtp = true;
    wide_input.num_intermediates = wide.num_intermediates;
    assert_eq!(
        checked_ssm_pool_geometry(wide_input).unwrap().total_bytes,
        4104
    );

    let mut lazy = input();
    lazy.max_slots = 1;
    lazy.has_mtp = true;
    lazy.num_intermediates = 4;
    lazy.lazy_commit = true;
    lazy.num_key_heads = 1;
    lazy.key_head_dim = 2;
    lazy.num_value_heads = 1;
    lazy.value_head_dim = 2;
    let lazy = checked_ssm_pool_geometry(lazy).unwrap();
    assert_eq!(lazy.kv_retain_bytes, 48);
    assert_eq!(lazy.gate_retain_bytes, 32);
    assert_eq!(lazy.total_bytes, 608);
}

#[test]
fn disabled_and_enabled_zero_byte_shapes_are_distinct() {
    let mut disabled = input();
    disabled.max_slots = usize::MAX;
    disabled.num_ssm_layers = 0;
    disabled.h_bytes = 0;
    disabled.conv_bytes = 0;
    assert_eq!(
        checked_ssm_pool_geometry(disabled).unwrap(),
        SsmPoolGeometry::default()
    );

    let gpu = MockGpuBackend::new();
    let mut config = tiny_config(0, false);
    config.mamba_num_heads = usize::MAX;
    config.mamba_head_dim = usize::MAX;
    config.ssm_state_size = usize::MAX;
    let pool = SsmStatePool::new(&config, 4, true, usize::MAX, &gpu).unwrap();
    assert_eq!(pool.num_ssm_layers, 0);
    assert_eq!(gpu.alloc_count(), 0);

    let mut zero_h = input();
    zero_h.h_bytes = 0;
    expect_error(zero_h, "h state bytes must be positive");
    let mut zero_conv = input();
    zero_conv.conv_bytes = 0;
    expect_error(zero_conv, "conv state bytes must be positive");
    let mut zero_intermediates = input();
    zero_intermediates.has_mtp = true;
    expect_error(zero_intermediates, "intermediate count must be positive");

    let error = match SsmStatePool::new(&tiny_config(2, false), 1, false, 0, &gpu) {
        Ok(_) => panic!("enabled zero-byte state unexpectedly allocated"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("state bytes must be positive"));
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn state_and_speculative_overflow_boundaries_fail_closed() {
    let mut slots = input();
    slots.max_slots = usize::MAX;
    expect_error(slots, "dummy-inclusive slots");

    let mut sum = input();
    sum.h_bytes = usize::MAX;
    sum.conv_bytes = 1;
    expect_error(sum, "state bytes per copy");

    let mut copies = input();
    copies.has_mtp = true;
    copies.num_intermediates = usize::MAX;
    expect_error(copies, "state copies");

    let mut layer = input();
    layer.max_slots = 0;
    layer.num_ssm_layers = 2;
    layer.h_bytes = usize::MAX / 2;
    layer.conv_bytes = 2;
    expect_error(layer, "all SSM layers");

    let mut per_layer = input();
    per_layer.max_slots = 1;
    per_layer.num_ssm_layers = 1;
    per_layer.h_bytes = usize::MAX / 2;
    per_layer.conv_bytes = 1;
    // Still fails closed at the same dummy-inclusive product (2 slots x 2^63 bytes). The E2 memory diet
    // (fc02ee695) split bytes-per-layer into a base term over every dummy-inclusive slot plus a speculative
    // term, so the checked_mul that overflows here is now labelled "dummy-inclusive base bytes per layer".
    expect_error(per_layer, "dummy-inclusive base bytes per layer");

    assert!(
        checked_ssm_speculative_geometry(true, false, usize::MAX - 1, None)
            .unwrap_err()
            .to_string()
            .contains("drafts + 2")
    );
    assert!(
        checked_ssm_speculative_geometry(true, true, usize::MAX, None)
            .unwrap_err()
            .to_string()
            .contains("DFlash verify width")
    );
    assert!(
        checked_ssm_speculative_geometry(true, true, 32, Some(32))
            .unwrap_err()
            .to_string()
            .contains("kernel maximum")
    );
}

#[test]
fn every_lazy_retention_boundary_is_checked() {
    let lazy = |nk, kd, nv, vd, ni| SsmPoolGeometryInput {
        has_mtp: true,
        num_intermediates: ni,
        lazy_commit: true,
        num_key_heads: nk,
        key_head_dim: kd,
        num_value_heads: nv,
        value_head_dim: vd,
        ..input()
    };

    for (shape, expected) in [
        (lazy(2, usize::MAX, 1, 1, 1), "key width"),
        (lazy(1, usize::MAX / 2 + 1, 1, 1, 1), "doubled key width"),
        (lazy(1, 1, 2, usize::MAX, 1), "value width"),
        (lazy(1, usize::MAX / 2, 1, 2, 1), "retention conv width"),
        (
            lazy(0, 0, 1, usize::MAX / 2 + 1, 2),
            "KV retention elements",
        ),
        (lazy(0, 0, 1, usize::MAX / 2 + 1, 1), "KV retention bytes"),
        (lazy(1, 1, usize::MAX, 0, 1), "gate width"),
        (
            lazy(1, 1, usize::MAX / 4 + 1, 0, 2),
            "gate retention elements",
        ),
        (lazy(1, 1, usize::MAX / 8 + 1, 0, 1), "gate retention bytes"),
        (
            lazy(1, usize::MAX / 8, usize::MAX / 16 + 2, 0, 1),
            "retention bytes per slot",
        ),
    ] {
        expect_error(shape, expected);
    }
}

#[test]
fn raw_gdn_wrap_to_four_rejects_before_allocation() {
    let mut config = tiny_config(1, true);
    config.linear_num_key_heads = 0;
    config.linear_num_value_heads = usize::MAX / 4 + 2;
    let gpu = MockGpuBackend::new();
    let error = match SsmStatePool::new(&config, 1, false, 0, &gpu) {
        Ok(_) => panic!("wrapped GDN state geometry unexpectedly allocated"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("GDN h FP32 bytes"), "{error:#}");
    assert_eq!(gpu.alloc_count(), 0);
}

/// The gate is real in BOTH directions: with `uncap_tree = true`
/// (`ATLAS_DDTREE_UNCAP=1`), the pool pays for the wide tree again because the
/// tree can then actually place nodes in those slots.
#[test]
fn uncapping_the_tree_restores_the_intermediates_it_needs() {
    let capped =
        checked_ssm_speculative_geometry_with_uncap(true, true, 15, Some(31), false).unwrap();
    assert_eq!(
        capped.num_intermediates, 17,
        "capped, the pool stays sized for the flat path"
    );

    let wide = checked_ssm_speculative_geometry_with_uncap(true, true, 15, Some(31), true).unwrap();
    assert_eq!(
        wide.num_intermediates, 32,
        "uncapped, the pool is sized for the tree again"
    );
}

#[test]
fn attention_only_pool_admits_sequences_without_recurrent_state() {
    // Attention-only models (Gemma) have no SSM layers but still claim one
    // slot index per sequence; an empty free list refused every request.
    let gpu = MockGpuBackend::new();
    let pool = SsmStatePool::new(&tiny_config(0, false), 4, false, 0, &gpu).unwrap();
    assert_eq!(pool.num_ssm_layers, 0);
    assert_eq!(gpu.alloc_count(), 0);
    let claimed: Vec<usize> = (0..4).map(|_| pool.claim_slot().unwrap()).collect();
    assert!(pool.claim_slot().is_err());
    pool.release_slot(claimed[0]);
    assert_eq!(pool.claim_slot().unwrap(), claimed[0]);
    assert_eq!(pool.claimed_slots(), None);
}

#[test]
fn qwen4_compact_lazy_h_cuts_c64_mtp_pool_by_over_47_gib() {
    let mut bad = input();
    bad.compact_lazy_h = true;
    expect_error(bad, "compact lazy H requires speculative state");

    let h_bytes = 48 * 128 * 128 * 4; // 3 MiB
    let conv_bytes = (2 * 16 * 128 + 48 * 128) * 4 * 4; // 160 KiB
    let full = checked_ssm_pool_geometry(SsmPoolGeometryInput {
        max_slots: 64,
        num_ssm_layers: 36,
        h_bytes,
        conv_bytes,
        has_mtp: true,
        compact_lazy_h: false,
        num_intermediates: 6,
        lazy_commit: false,
        num_key_heads: 16,
        key_head_dim: 128,
        num_value_heads: 48,
        value_head_dim: 128,
    })
    .unwrap();
    let compact = checked_ssm_pool_geometry(SsmPoolGeometryInput {
        compact_lazy_h: true,
        ..SsmPoolGeometryInput {
            max_slots: 64,
            num_ssm_layers: 36,
            h_bytes,
            conv_bytes,
            has_mtp: true,
            compact_lazy_h: false,
            num_intermediates: 6,
            lazy_commit: false,
            num_key_heads: 16,
            key_head_dim: 128,
            num_value_heads: 48,
            value_head_dim: 128,
        }
    })
    .unwrap();
    assert_eq!(compact.spec_slots, 64);
    assert_eq!(compact.h_intermediate_allocation_bytes, 0);
    assert_eq!(compact.h_checkpoint_allocation_bytes, 0);
    assert_eq!(compact.h_verify_scratch_bytes, 64 * h_bytes);
    // Full C64 ni=6 pool = 61,955,112,960 B (~57.70 GiB); compact = 10,588,127,232 B (~9.86 GiB).
    assert_eq!(full.total_bytes, 61_955_112_960);
    assert_eq!(compact.total_bytes, 10_588_127_232);
    assert!(full.total_bytes - compact.total_bytes > 47 * (1 << 30));
}

#[test]
fn ssm_decode_ring_policy_omits_qwen4_multi_seq_unless_explicitly_enabled() {
    use crate::model::ssm_pool_geometry::ssm_decode_ring_enabled_for_env;

    assert!(ssm_decode_ring_enabled_for_env(None, false, 64).unwrap());
    assert!(ssm_decode_ring_enabled_for_env(None, true, 1).unwrap());
    assert!(!ssm_decode_ring_enabled_for_env(None, true, 2).unwrap());
    assert!(!ssm_decode_ring_enabled_for_env(None, true, 64).unwrap());
    assert!(!ssm_decode_ring_enabled_for_env(None, true, 128).unwrap());
    assert!(ssm_decode_ring_enabled_for_env(Some("1"), true, 64).unwrap());
    assert!(!ssm_decode_ring_enabled_for_env(Some("0"), false, 1).unwrap());
    for bad in ["", "2", "yes", "auto"] {
        assert!(ssm_decode_ring_enabled_for_env(Some(bad), true, 64).is_err());
    }
}
