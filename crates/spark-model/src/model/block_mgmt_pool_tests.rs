// SPDX-License-Identifier: AGPL-3.0-only

//! Shared-pool allocator behaviour behind `--kv-pool-tokens`: one row runs
//! out alone, released blocks are reused, and admit/finish cycles never leak.

use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype};
use spark_runtime::prefix_cache::NoPrefixCaching;

const BS: usize = 16;

fn pool(gpu: &MockGpuBackend, blocks: usize) -> PagedKvCache {
    let config = KvCacheConfig {
        block_size: BS,
        num_kv_heads: 1,
        head_dim: 4,
        num_layers: 1,
        dtype: KvCacheDtype::Bf16,
        layer_dtypes: vec![],
        layer_dims: vec![],
        cache_blocks_per_seq: None,
    };
    PagedKvCache::new(config, blocks, gpu).unwrap()
}

fn seq(seq_len: usize) -> SequenceState {
    SequenceState {
        tokens: Vec::new(),
        block_table: Vec::new(),
        seq_len,
        rotary_positions: crate::traits::RotaryPositions::identity(),
        qwen4_qsa_required: false,
        layer_states: Vec::new(),
        proposer_state: None,
        proposer_state_alt: None,
        slot_idx: 0,
        marconi_skip_to: 0,
        session_hash: 0,
        verify_frame: None,
        chunked_prefill_meta: None,
        qwen4_prefill_lineage: crate::traits::Qwen4PrefillLineage::Unset,
        cached_prefix_tokens: 0,
        kv_valid_tokens: 0,
        prompt_len: 0,
        disk_block_ids: Vec::new(),
        mtp_lastk_host_buf: Vec::new(),
        mtp_lastk_host_filled: 0,
        mtp_lastk_end_abs: 0,
        disk_last_offloaded_per_layer: Vec::new(),
    }
}

/// Grow `s` by one decode position, reserving `rows` ahead like the scheduler.
fn step(
    s: &mut SequenceState,
    rows: usize,
    kv: &mut PagedKvCache,
    gpu: &MockGpuBackend,
) -> Result<()> {
    reserve_decode_rows(s, rows, kv, &NoPrefixCaching, gpu, 0)?;
    s.seq_len += 1;
    Ok(())
}

#[test]
fn reserve_allocates_exactly_the_blocks_a_verify_frame_writes() {
    let gpu = MockGpuBackend::new();
    let mut kv = pool(&gpu, 8);
    let mut s = seq(14);
    // K5 frame at seq_len 14 writes rows 14..=18: blocks 0 and 1.
    reserve_decode_rows(&mut s, 5, &mut kv, &NoPrefixCaching, &gpu, 0).unwrap();
    assert_eq!(s.block_table.len(), 2);
    // Idempotent: the step's own ensure calls (or a second reserve) take nothing.
    reserve_decode_rows(&mut s, 5, &mut kv, &NoPrefixCaching, &gpu, 0).unwrap();
    assert_eq!(s.block_table.len(), 2);
    assert_eq!(kv.num_free_blocks(), 6);
    // Zero rows is a no-op.
    let mut t = seq(0);
    reserve_decode_rows(&mut t, 0, &mut kv, &NoPrefixCaching, &gpu, 0).unwrap();
    assert!(t.block_table.is_empty());
}

#[test]
fn exhaustion_fails_only_the_row_that_cannot_grow() {
    let gpu = MockGpuBackend::new();
    // 4 blocks = 64 tokens shared by two rows.
    let mut kv = pool(&gpu, 4);
    let mut deep = seq(0);
    let mut shallow = seq(0);
    // Deep row takes 3 blocks (48 tokens), shallow row takes the last one.
    for _ in 0..48 {
        step(&mut deep, 1, &mut kv, &gpu).unwrap();
    }
    for _ in 0..16 {
        step(&mut shallow, 1, &mut kv, &gpu).unwrap();
    }
    assert_eq!(kv.num_free_blocks(), 0);
    // Deep row needs block 3: refused, and it is the only failure.
    let err = step(&mut deep, 1, &mut kv, &gpu).unwrap_err().to_string();
    assert!(err.contains("alloc failed"), "{err}");
    assert_eq!(deep.block_table.len(), 3);
    assert_eq!(deep.seq_len, 48);
    // The failed row is retired: its blocks return to the pool.
    kv.free_blocks(&deep.block_table);
    deep.block_table.clear();
    assert_eq!(kv.num_free_blocks(), 3);
    // The surviving row keeps decoding into the freed blocks.
    for _ in 0..40 {
        step(&mut shallow, 1, &mut kv, &gpu).unwrap();
    }
    assert_eq!(shallow.block_table.len(), 4);
    assert_eq!(kv.num_free_blocks(), 0);
}

#[test]
fn released_blocks_are_reused_by_the_next_request() {
    let gpu = MockGpuBackend::new();
    let mut kv = pool(&gpu, 6);
    let mut a = seq(0);
    reserve_decode_rows(&mut a, 6 * BS, &mut kv, &NoPrefixCaching, &gpu, 0).unwrap();
    assert_eq!(kv.num_free_blocks(), 0);
    let mut b = seq(0);
    assert!(reserve_decode_rows(&mut b, 1, &mut kv, &NoPrefixCaching, &gpu, 0).is_err());
    kv.free_blocks(&a.block_table);
    let first_owner: std::collections::BTreeSet<u32> = a.block_table.drain(..).collect();
    reserve_decode_rows(&mut b, 6 * BS, &mut kv, &NoPrefixCaching, &gpu, 0).unwrap();
    let second_owner: std::collections::BTreeSet<u32> = b.block_table.iter().copied().collect();
    assert_eq!(first_owner, second_owner);
}

#[test]
fn admit_finish_cycles_do_not_leak_blocks() {
    let gpu = MockGpuBackend::new();
    let total = 24;
    let mut kv = pool(&gpu, total);
    let mut live: Vec<SequenceState> = Vec::new();
    let mut failed = 0usize;
    for cycle in 0..2_000usize {
        // Admit a request with a cycle-dependent depth, finish the oldest
        // when four are live, and fail (retire) a row the pool refuses.
        let mut s = seq(0);
        let depth = 1 + (cycle * 37) % (12 * BS);
        match reserve_decode_rows(&mut s, depth, &mut kv, &NoPrefixCaching, &gpu, 0) {
            Ok(()) => {
                s.seq_len = depth;
                live.push(s);
            }
            Err(_) => {
                failed += 1;
                kv.free_blocks(&s.block_table);
            }
        }
        if live.len() == 4 {
            let done = live.remove(0);
            kv.free_blocks(&done.block_table);
        }
        let owned: usize = live.iter().map(|s| s.block_table.len()).sum();
        assert_eq!(kv.num_free_blocks() + owned, total, "cycle {cycle}");
    }
    assert!(failed > 0, "the workload must exercise exhaustion");
    for done in live.drain(..) {
        kv.free_blocks(&done.block_table);
    }
    assert_eq!(kv.num_free_blocks(), total);
}

#[test]
fn high_speed_swap_window_is_left_alone() {
    let gpu = MockGpuBackend::new();
    let config = KvCacheConfig {
        block_size: BS,
        num_kv_heads: 1,
        head_dim: 4,
        num_layers: 1,
        dtype: KvCacheDtype::Bf16,
        layer_dtypes: vec![],
        layer_dims: vec![],
        cache_blocks_per_seq: Some(2),
    };
    let mut kv = PagedKvCache::new(config, 4, &gpu).unwrap();
    let mut s = seq(0);
    reserve_decode_rows(&mut s, 64, &mut kv, &NoPrefixCaching, &gpu, 0).unwrap();
    assert!(s.block_table.is_empty());
    assert_eq!(kv.num_free_blocks(), 4);
}
