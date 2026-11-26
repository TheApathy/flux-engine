// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use spark_runtime::gpu::carveout::CarveoutArena;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;

const MIB: usize = 1 << 20;
/// The GB10 DISPLAY_FRM carveout as the driver reports it (~2,042–2,046 MiB).
const GB10: usize = 2042 * MIB;

fn config(
    layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    dtype: KvCacheDtype,
) -> KvCacheConfig {
    KvCacheConfig {
        block_size: 16,
        num_kv_heads,
        head_dim,
        num_layers: layers,
        dtype,
        layer_dtypes: vec![],
        layer_dims: vec![],
        cache_blocks_per_seq: None,
    }
}

fn carveout_footprint(shape: KvShape, blocks: usize, placement: &KvPlacement) -> usize {
    shape
        .buffers(blocks)
        .iter()
        .filter(|&&(buffer, _)| placement.contains(buffer))
        .map(|&(_, bytes)| CarveoutArena::footprint(bytes))
        .sum()
}

#[test]
fn qwen38_flash_next_gains_blocks_at_zero_system_memory_cost() {
    // Qwen3.8-Flash-Next: 12 full-attention layers, 2 KV heads × 256 dim, NVFP4 (~9 KiB/side/block = 216 KiB/block).
    let cfg = config(12, 2, 256, KvCacheDtype::Nvfp4);
    let shape = KvShape {
        config: &cfg,
        v_aliases_k: false,
        auxiliary_block_bytes: 0,
    };
    let budgeted = 67_618;
    let (n, placement) = plan(shape, budgeted, GB10);
    // 2042 MiB over 221,184 B/block (~216 KiB) is ~9,679 blocks; whole ~660 MiB
    // pools move into the carveout, adding ~8,000+ blocks (+~128K KV tokens) at 0 system RAM cost.
    assert!(n >= budgeted + 7_500, "{n}");
    assert!(
        shape.system_bytes(n, &placement) <= shape.system_bytes(budgeted, &KvPlacement::default())
    );
    assert!(carveout_footprint(shape, n, &placement) <= GB10);

    let over = KvPlacement::plan(&shape.buffers(n + 1), GB10);
    assert!(
        shape.system_bytes(n + 1, &over) > shape.system_bytes(budgeted, &KvPlacement::default())
    );
}

#[test]
fn mla_aliased_v_pools_gain_about_a_fifth_more_blocks_at_no_system_memory_cost() {
    let cfg = config(11, 1, 512, KvCacheDtype::Fp8);
    let shape = KvShape {
        config: &cfg,
        v_aliases_k: true,
        auxiliary_block_bytes: 0,
    };
    let budgeted = 95_535;
    let (n, placement) = plan(shape, budgeted, GB10);
    assert!(n >= budgeted + 18_000, "{n}");
    assert!(
        shape.system_bytes(n, &placement) <= shape.system_bytes(budgeted, &KvPlacement::default())
    );
    assert!(carveout_footprint(shape, n, &placement) <= GB10);
}

#[test]
fn any_agreed_count_up_to_the_planned_one_still_fits() {
    let cfg = config(12, 2, 256, KvCacheDtype::Fp8);
    let shape = KvShape {
        config: &cfg,
        v_aliases_k: false,
        auxiliary_block_bytes: 0,
    };
    let budgeted = 60_000;
    let (n, placement) = plan(shape, budgeted, GB10);
    let system = shape.system_bytes(budgeted, &KvPlacement::default());
    for agreed in [budgeted - 5_000, budgeted, (budgeted + n) / 2, n - 1, n] {
        assert!(shape.system_bytes(agreed, &placement) <= system, "{agreed}");
        assert!(
            carveout_footprint(shape, agreed, &placement) <= GB10,
            "{agreed}"
        );
    }
}

#[test]
fn without_a_carveout_the_budgeted_pool_is_unchanged() {
    let cfg = config(12, 2, 256, KvCacheDtype::Fp8);
    let shape = KvShape {
        config: &cfg,
        v_aliases_k: false,
        auxiliary_block_bytes: 0,
    };
    let (n, placement) = plan(shape, 67_618, 0);
    assert_eq!(n, 67_618);
    assert!(placement.is_empty());

    let (zero_n, zero_placement) = plan(shape, 0, GB10);
    assert_eq!(zero_n, 0);
    assert!(zero_placement.is_empty());
}

#[test]
fn extend_queries_backend_carveout_and_allocates_cleanly() {
    let cfg = config(4, 2, 256, KvCacheDtype::Bf16);
    let shape = KvShape {
        config: &cfg,
        v_aliases_k: false,
        auxiliary_block_bytes: 0,
    };
    let no_carveout = MockGpuBackend::new();
    assert_eq!(
        extend(&no_carveout, shape, 500),
        (500, KvPlacement::default())
    );

    let with_carveout = MockGpuBackend::with_carveout(64 * MIB);
    let (n, placement) = extend(&with_carveout, shape, 500);
    assert!(n > 500);
    assert!(!placement.is_empty());
}

#[test]
fn qwen_carveout_growth_reserves_indexer_state_for_every_added_token() {
    let cfg = config(12, 1, 256, KvCacheDtype::Bf16);
    let bare = KvShape {
        config: &cfg,
        v_aliases_k: false,
        auxiliary_block_bytes: 0,
    };
    let shape = bare.with_per_token_aux(768).unwrap();
    let budgeted = 208_676;
    let (n, placement) = plan(shape, budgeted, GB10);
    assert!(n > budgeted);
    let actual = |blocks, placement: &KvPlacement| {
        bare.system_bytes(blocks, placement) + blocks * cfg.block_size * 768
    };
    let budget = actual(budgeted, &KvPlacement::default());
    assert!(actual(n, &placement) <= budget);
    assert!(carveout_footprint(shape, n, &placement) <= GB10);
    let over = KvPlacement::plan(&shape.buffers(n + 1), GB10);
    assert!(actual(n + 1, &over) > budget);
    let (old_n, old_placement) = plan(bare, budgeted, GB10);
    assert!(actual(old_n, &old_placement) > budget);
    assert!(bare.with_per_token_aux(usize::MAX).is_err());
    assert!(shape.with_per_block_aux(usize::MAX).is_err());
}

#[test]
fn carveout_growth_is_unchanged_without_per_token_aux() {
    let cfg = config(12, 1, 256, KvCacheDtype::Bf16);
    let bare = KvShape {
        config: &cfg,
        v_aliases_k: false,
        auxiliary_block_bytes: 0,
    };
    let budgeted = 208_676;
    assert_eq!(
        plan(bare, budgeted, GB10),
        plan(bare.with_per_token_aux(0).unwrap(), budgeted, GB10),
    );
}
