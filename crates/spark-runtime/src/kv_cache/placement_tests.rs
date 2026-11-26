// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::gpu::carveout::CarveoutArena;
use crate::gpu::mock::MockGpuBackend;
use crate::kv_cache::KvCacheDtype;

const MIB: usize = 1 << 20;

/// Qwen4/hybrid-shaped KV config: 2 KV heads × 256 dim, mixed or uniform dtype.
fn kv_cfg(layers: usize, dtype: KvCacheDtype) -> KvCacheConfig {
    KvCacheConfig {
        block_size: 16,
        num_kv_heads: 2,
        head_dim: 256,
        num_layers: layers,
        dtype,
        layer_dtypes: vec![],
        layer_dims: vec![],
        cache_blocks_per_seq: None,
    }
}

/// Synthetic buffer set mixing K/V pools with auxiliary index buffers so
/// `CarveoutOrder::Size` and `CarveoutOrder::Latent` can be compared directly.
fn synthetic_glm_buffers() -> Vec<(KvBuffer, usize)> {
    vec![
        (KvBuffer::K(0), 844_800),
        (KvBuffer::K(1), 844_800),
        (KvBuffer::K(2), 844_800),
        (KvBuffer::TailMap, 400),
        (KvBuffer::IndexValues(0), 102_400),
        (KvBuffer::IndexScales(0), 0),
        (KvBuffer::IndexTail(0), 819_200),
        (KvBuffer::IndexValues(1), 102_400),
        (KvBuffer::IndexScales(1), 0),
        (KvBuffer::IndexTail(1), 819_200),
        (KvBuffer::IndexValues(2), 102_400),
        (KvBuffer::IndexScales(2), 0),
        (KvBuffer::IndexTail(2), 819_200),
    ]
}

#[test]
fn plan_takes_the_largest_buffers_that_fit() {
    let sizes = synthetic_glm_buffers();
    // K 844800 B and tails 819200 B occupy 13 × 64 KiB each, values 2 × 64 KiB, TailMap 1 × 64 KiB.
    let placement = KvPlacement::plan_ordered(&sizes, 2 * MIB, CarveoutOrder::Size);
    let expected: BTreeSet<_> = [
        KvBuffer::K(0),
        KvBuffer::K(1),
        KvBuffer::IndexValues(0),
        KvBuffer::IndexValues(1),
        KvBuffer::IndexValues(2),
    ]
    .into();
    assert_eq!(placement.carveout, expected);
    assert_eq!(placement.carved_bytes(&sizes), 2 * 844_800 + 3 * 102_400);
    // Zero-byte buffers are never placed.
    assert!(!placement.contains(KvBuffer::IndexScales(0)));
}

#[test]
fn latent_order_places_only_kv_pools() {
    let sizes = synthetic_glm_buffers();
    let placement = KvPlacement::plan_ordered(&sizes, 2 * MIB, CarveoutOrder::Latent);
    let expected: BTreeSet<_> = [KvBuffer::K(0), KvBuffer::K(1)].into();
    assert_eq!(placement.carveout, expected);

    // Room for everything still places no index buffer under `Latent`.
    let all = KvPlacement::plan_ordered(&sizes, 64 * MIB, CarveoutOrder::Latent);
    let expected: BTreeSet<_> = [KvBuffer::K(0), KvBuffer::K(1), KvBuffer::K(2)].into();
    assert_eq!(all.carveout, expected);
}

#[test]
fn buffer_sizes_match_what_the_constructors_allocate_including_asymmetric_dtypes() {
    let mut cfg = kv_cfg(3, KvCacheDtype::Bf16KTurbo4V);
    cfg.layer_dtypes = vec![
        KvCacheDtype::Bf16,
        KvCacheDtype::Bf16KTurbo4V,
        KvCacheDtype::Fp8,
    ];
    let sizes = PagedKvCache::buffer_sizes(&cfg, 100, false);
    assert_eq!(sizes.len(), 6);

    let gpu = MockGpuBackend::with_carveout(64 * MIB);
    let placement = KvPlacement::plan_ordered(&sizes, 64 * MIB, CarveoutOrder::Latent);
    let _cache = PagedKvCache::new_placed(cfg, 100, &gpu, false, placement).unwrap();

    let mut listed: Vec<usize> = sizes.iter().map(|&(_, n)| n).filter(|&n| n > 0).collect();
    let mut allocated = gpu.carveout_alloc_sizes();
    listed.sort_unstable();
    allocated.sort_unstable();
    assert_eq!(allocated, listed);
}

#[test]
fn v_aliased_pools_allocate_once_per_layer_and_read_write_roundtrip() {
    let cfg = kv_cfg(2, KvCacheDtype::Fp8);
    let sizes = PagedKvCache::buffer_sizes(&cfg, 32, true);
    assert_eq!(sizes.len(), 2);

    let gpu = MockGpuBackend::with_carveout(16 * MIB);
    let placement = KvPlacement::plan_ordered(&sizes, 16 * MIB, CarveoutOrder::Latent);
    let mut cache = PagedKvCache::new_placed(cfg, 32, &gpu, true, placement).unwrap();
    assert_eq!(cache.k_pool_ptr(0), cache.v_pool_ptr(0));
    assert_eq!(gpu.carveout_alloc_sizes().len(), 2);

    let blk = cache.alloc_block().unwrap();
    cache.zero_block(blk, &gpu, 0).unwrap();
    cache.release(&gpu).unwrap();
    assert_eq!(gpu.carveout_used(), 0);
    assert_eq!(gpu.invalid_free_count(), 0);
}

#[test]
fn only_placed_buffers_use_the_carveout() {
    let cfg = kv_cfg(3, KvCacheDtype::Bf16);
    let sizes = PagedKvCache::buffer_sizes(&cfg, 64, false);
    // Each BF16 pool is 64 * 16 * 2 * 256 * 2 = 1 MiB. A 2 MiB carveout holds 2 of 6 pools.
    let placement = KvPlacement::plan_ordered(&sizes, 2 * MIB, CarveoutOrder::Latent);
    assert_eq!(placement.len(), 2);

    let gpu = MockGpuBackend::with_carveout(2 * MIB);
    let _cache = PagedKvCache::new_placed(cfg, 64, &gpu, false, placement.clone()).unwrap();
    let footprints: usize = sizes
        .iter()
        .filter(|(b, _)| placement.carveout.contains(b))
        .map(|&(_, n)| CarveoutArena::footprint(n))
        .sum();
    assert_eq!(gpu.carveout_used(), footprints);
    assert_eq!(gpu.carveout_alloc_sizes().len(), placement.len());
    assert_eq!(gpu.alloc_count(), 6);
}

#[test]
fn a_placement_planned_at_more_blocks_fits_fewer() {
    let cfg = kv_cfg(3, KvCacheDtype::Bf16);
    let planned = PagedKvCache::buffer_sizes(&cfg, 64, false);
    let placement = KvPlacement::plan_ordered(&planned, 2 * MIB, CarveoutOrder::Latent);
    let gpu = MockGpuBackend::with_carveout(2 * MIB);
    let _cache = PagedKvCache::new_placed(cfg.clone(), 40, &gpu, false, placement.clone()).unwrap();
    let fewer = PagedKvCache::buffer_sizes(&cfg, 40, false);
    assert!(placement.carved_bytes(&fewer) < placement.carved_bytes(&planned));
}

#[test]
fn release_returns_carveout_memory_and_is_idempotent() {
    let cfg = kv_cfg(3, KvCacheDtype::Bf16);
    let sizes = PagedKvCache::buffer_sizes(&cfg, 64, false);
    let gpu = MockGpuBackend::with_carveout(2 * MIB);
    let mut cache = PagedKvCache::new_placed(
        cfg,
        64,
        &gpu,
        false,
        KvPlacement::plan_ordered(&sizes, 2 * MIB, CarveoutOrder::Latent),
    )
    .unwrap();
    assert!(gpu.carveout_used() > 0);
    cache.release(&gpu).unwrap();
    assert_eq!(gpu.carveout_used(), 0);
    assert_eq!(gpu.alloc_count(), 0);
    cache.release(&gpu).unwrap();
    assert_eq!(gpu.invalid_free_count(), 0);
}

#[test]
fn without_a_carveout_nothing_is_placed() {
    let cfg = kv_cfg(2, KvCacheDtype::Bf16);
    let sizes = PagedKvCache::buffer_sizes(&cfg, 64, false);
    assert!(KvPlacement::plan_ordered(&sizes, 0, CarveoutOrder::Size).is_empty());
    let gpu = MockGpuBackend::new();
    let _cache = PagedKvCache::new_placed(cfg, 64, &gpu, false, KvPlacement::default()).unwrap();
    assert!(gpu.carveout_alloc_sizes().is_empty());
}

#[test]
fn failed_placed_pool_allocation_releases_both_arenas_and_can_retry() {
    let cfg = kv_cfg(3, KvCacheDtype::Bf16);
    let sizes = PagedKvCache::buffer_sizes(&cfg, 64, false);
    // One K pool fits in the carveout; the remaining 5 K/V pools use system memory.
    let capacity = CarveoutArena::footprint(sizes[0].1);
    let placement = KvPlacement::plan_ordered(&sizes, capacity, CarveoutOrder::Latent);
    assert_eq!(placement.len(), 1);

    for fail_after in 0..(cfg.num_layers * 2) {
        let gpu = MockGpuBackend::with_carveout(capacity);
        gpu.fail_alloc_after(fail_after);
        assert!(
            PagedKvCache::new_placed(cfg.clone(), 64, &gpu, false, placement.clone()).is_err(),
            "fail_after={fail_after}"
        );
        assert_eq!(gpu.alloc_count(), 0, "fail_after={fail_after}");
        assert_eq!(gpu.carveout_used(), 0, "fail_after={fail_after}");
        assert_eq!(gpu.invalid_free_count(), 0, "fail_after={fail_after}");

        let mut cache =
            PagedKvCache::new_placed(cfg.clone(), 64, &gpu, false, placement.clone()).unwrap();
        cache.release(&gpu).unwrap();
        assert_eq!(gpu.alloc_count(), 0);
        assert_eq!(gpu.carveout_used(), 0);
        assert_eq!(gpu.invalid_free_count(), 0);
    }
}
