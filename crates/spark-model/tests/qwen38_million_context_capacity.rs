// SPDX-License-Identifier: AGPL-3.0-only

use spark_model::factory::kv_pool::reachable_blocks;
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype};

const MAX_SEQ_LEN: usize = 1_000_000;
const BLOCK_SIZE: usize = 16;
const MAX_BATCH_SIZE: usize = 1;

#[test]
fn exact_million_c1_uses_62502_blocks() {
    let active_blocks = MAX_SEQ_LEN.div_ceil(BLOCK_SIZE);
    assert_eq!(active_blocks, 62_500);
    assert_eq!(
        reachable_blocks(MAX_SEQ_LEN, MAX_BATCH_SIZE, BLOCK_SIZE),
        62_502
    );

    let source = include_str!("../src/factory/build.rs");
    assert!(source.contains("max_seq_len.div_ceil(kv_block_size)"));
    assert!(source.contains(".saturating_mul(blocks_per_seq.saturating_add(1))"));
    assert!(source.contains("let n = budget_blocks.min(reachable);"));
    assert!(source.contains("let index_max_blocks = (i32::MAX as usize) / block_size;"));
}

#[test]
fn synthetic_1048576_boundary_fits_index_widths() {
    let allocated_blocks = reachable_blocks(1_048_576, MAX_BATCH_SIZE, BLOCK_SIZE);
    assert_eq!(allocated_blocks, 65_538);

    let index_max_blocks = (i32::MAX as usize) / BLOCK_SIZE;
    assert!(allocated_blocks <= index_max_blocks);

    let highest_physical_block = allocated_blocks - 1;
    let highest_slot = allocated_blocks * BLOCK_SIZE - 1;
    assert!(i32::try_from(highest_physical_block).is_ok());
    assert!(i32::try_from(highest_slot).is_ok());

    let kv_config = KvCacheConfig {
        block_size: BLOCK_SIZE,
        num_kv_heads: 4,
        head_dim: 256,
        num_layers: 16,
        dtype: KvCacheDtype::Nvfp4,
        layer_dtypes: Vec::new(),
        layer_dims: Vec::new(),
        cache_blocks_per_seq: None,
    };
    let nvfp4_bytes_per_side_block = u64::try_from(kv_config.block_bytes_for_layer(0)).unwrap();
    assert_eq!(nvfp4_bytes_per_side_block, 9_216);
    let final_side_byte = (allocated_blocks as u64)
        .checked_mul(nvfp4_bytes_per_side_block)
        .and_then(|extent| extent.checked_sub(1))
        .unwrap();
    assert_eq!(final_side_byte, 603_998_207);
}
