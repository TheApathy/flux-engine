// SPDX-License-Identifier: AGPL-3.0-only

use super::{
    ReserveBreakdown, StartupProjection, kv_block_bytes_all_layers, kv_cache_addressable_bytes,
    kv_cache_projected_bytes, qwen4_construction_transient_bytes,
};
use atlas_core::config::ModelConfig;

const MIB: usize = 1024 * 1024;
const GIB: usize = 1024 * MIB;
/// Flash-Next: 12 attention layers x (K + V) x 16 tokens x 2 heads x 256 dims
/// x 2 B = 32,768 B per layer per block (report section 2, "8,450 blk x 12
/// layers x 32,768 B").
const FN_BLOCK_BYTES_ALL_LAYERS: usize = 12 * 32_768;

fn flash_next_config() -> ModelConfig {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.model_type = "qwen4_exp".to_string();
    config
}

#[test]
fn addressable_kv_matches_the_measured_batch_one_pool() {
    let bytes = kv_cache_addressable_bytes(1, 135_168, 16, FN_BLOCK_BYTES_ALL_LAYERS).unwrap();
    assert_eq!(bytes, 8_450 * FN_BLOCK_BYTES_ALL_LAYERS);
    assert_eq!(bytes / MIB, 3_168, "3.1 GiB on the r8 receipt");
}

#[test]
fn addressable_kv_is_only_an_upper_bound_above_batch_one() {
    let batch8 = kv_cache_addressable_bytes(8, 135_168, 16, FN_BLOCK_BYTES_ALL_LAYERS).unwrap();
    assert_eq!(batch8 / GIB, 24, "~25 GiB at batch 8 / 135168");
    let c128 = kv_cache_addressable_bytes(128, 262_144, 16, FN_BLOCK_BYTES_ALL_LAYERS).unwrap();
    assert_eq!(c128 / GIB, 768, "~800 GiB at C128 / 262144");
}

#[test]
fn projected_kv_mirrors_the_factory_budget_cap() {
    // r8-like host state: 20 GiB MemAvailable after the store, 11.3 GiB reserve.
    let (mem_available, reserve) = (20 * GIB, 11_300 * MIB);
    let budget = ((20 * GIB - 11_300 * MIB) as f64 * 0.85) as usize;
    let batch1 = kv_cache_addressable_bytes(1, 135_168, 16, FN_BLOCK_BYTES_ALL_LAYERS).unwrap();
    assert_eq!(
        kv_cache_projected_bytes(batch1, mem_available, reserve, 0.85),
        batch1,
        "batch 1 is capped by the addressable size, as the factory logs"
    );
    let batch8 = kv_cache_addressable_bytes(8, 135_168, 16, FN_BLOCK_BYTES_ALL_LAYERS).unwrap();
    assert_eq!(
        kv_cache_projected_bytes(batch8, mem_available, reserve, 0.85),
        budget,
        "batch 8 / 135168 is capped to the factory budget, not 25 GiB"
    );
    let c128 = kv_cache_addressable_bytes(128, 262_144, 16, FN_BLOCK_BYTES_ALL_LAYERS).unwrap();
    assert_eq!(
        kv_cache_projected_bytes(c128, mem_available, reserve, 0.85),
        budget,
        "C128 / 262144 is capped to the factory budget, not 800 GiB"
    );
    assert_eq!(
        kv_cache_projected_bytes(c128, reserve, reserve + 1, 0.85),
        0
    );
}

#[test]
fn block_bytes_follow_the_config_geometry() {
    let config = flash_next_config();
    let bytes = kv_block_bytes_all_layers(&config, 16).unwrap();
    assert_eq!(
        bytes,
        2 * config.num_key_value_heads * config.head_dim * 16 * 2 * config.num_attention_layers()
    );
    let mut mla = flash_next_config();
    mla.kv_lora_rank = 512;
    mla.qk_rope_head_dim = 64;
    assert_eq!(
        kv_block_bytes_all_layers(&mla, 16).unwrap(),
        2 * 576 * 16 * 2 * mla.num_attention_layers()
    );
}

#[test]
fn qwen4_transient_counts_slab_mtp_and_lm_head_once() {
    let config = flash_next_config();
    let lm_head = config.vocab_size * config.hidden_size * 9 / 16;
    let plain = qwen4_construction_transient_bytes(&config, false).unwrap();
    assert_eq!(plain, 1_415_578_112 + lm_head);
    let native = qwen4_construction_transient_bytes(&config, true).unwrap();
    assert_eq!(native, plain + 1_200 * MIB);
    let mut other = flash_next_config();
    other.model_type = "qwen3_next".to_string();
    assert_eq!(qwen4_construction_transient_bytes(&other, true).unwrap(), 0);
}

#[test]
fn projection_sums_modeled_terms_including_eager_qsa_shadows() {
    // r8 095613Z W16K: 19.5 GiB after the store, 2.5 GiB target construction,
    // 1.35 + 1.2 + 0.36 transient, 1.33 pools, 3.55 arena, 3.1 KV.
    let projection = StartupProjection {
        mem_available_bytes: 19_968 * MIB,
        target_construction_bytes: 2_560 * MIB,
        qwen4_transient_bytes: 2_980 * MIB,
        pools_bytes: 1_362 * MIB,
        buffer_arena_bytes: 3_640 * MIB,
        kv_projected_bytes: 3_168 * MIB,
        qsa_fp8_shadow_bytes: 1_661_337_600,
        floor_bytes: 6_656 * MIB,
        is_discrete: false,
    };
    assert_eq!(projection.need_bytes(), 13_710 * MIB + 1_661_337_600);
    assert_eq!(
        projection.projected_post_start_bytes(),
        6_258 * MIB - 1_661_337_600
    );
    // Below the floor on paper while the measured post-start value was
    // 13.4 GiB: the transpose reclaim is not a pre-construction term, which
    // is why this projection informs and the end-of-load table enforces.
    assert!(projection.projected_post_start_bytes() < projection.floor_bytes);
    let summary = projection.summary();
    assert!(summary.contains("modeled need 15294 MiB"), "{summary}");
    assert!(
        summary.contains("projected post-start MemAvailable 4673 MiB"),
        "{summary}"
    );
    assert!(summary.contains("floor 6656 MiB"), "{summary}");
    assert!(
        summary.contains("eager QSA shadows (addressable upper bound) 1584"),
        "{summary}"
    );
    assert!(
        summary.contains("partial estimate, not a peak bound"),
        "{summary}"
    );
    assert!(summary.contains("additional prefill scratch"), "{summary}");
    assert!(summary.contains("lazy request caches"), "{summary}");
}

#[test]
fn projection_carries_the_eager_shadow_reserve_from_preflight() {
    let source = include_str!("preflight.rs");
    assert!(source.contains("qsa_fp8_shadow_bytes: breakdown.qsa_fp8_shadow_bytes"));
    assert!(source.contains(".saturating_add(self.qsa_fp8_shadow_bytes)"));
}

#[test]
fn reserve_breakdown_defaults_to_zero_terms() {
    let breakdown = ReserveBreakdown::default();
    assert_eq!(breakdown.ssm_pool_bytes + breakdown.cuda_headroom_bytes, 0);
}
