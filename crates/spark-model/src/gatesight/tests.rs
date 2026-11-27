// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for GateSight static slow-head ranking, L2 ring sizing, Path-A block-16 scoring,
//! Path-B authenticated MLP sidecar, 2-zone prefix protection, streaming reservoir parity,
//! cross-layer budget allocation, and Flash-Next QSA safety floor clamping.

use super::*;

#[test]
fn static_slow_head_rank_ordering_matches_half_life_formula() {
    // Higher A_log or larger dt_bias -> faster decay -> shorter half-life.
    // Lower A_log / very negative dt_bias -> slower decay -> longer half-life (slow head).
    let a_log = [-2.0f32, 0.0, -4.0, 1.5, -3.5];
    let dt_bias = [-5.0f32, 0.0, -6.0, 2.0, -4.0];

    let cls = GdnLayerSlowHeads::classify(0, &a_log, &dt_bias, 1000.0)
        .expect("classify static slow heads");

    // Verify exact monotonic descending ordering by tau_static.
    let ranked = cls.ranked_heads_descending();
    assert_eq!(ranked, vec![2, 4, 0, 1, 3]);
    for pair in ranked.windows(2) {
        assert!(
            cls.tau_static[pair[0]] > cls.tau_static[pair[1]],
            "head {} ({}) should have longer half-life than head {} ({})",
            pair[0],
            cls.tau_static[pair[0]],
            pair[1],
            cls.tau_static[pair[1]]
        );
    }
    // Head 2 (a_log=-4, dt_bias=-6) has tau_static ~ 15,297 >> 1000 tokens.
    assert!(cls.slow_heads.contains(&2));
    assert!(!cls.slow_heads.contains(&3));
    assert!(cls.tau_static[2] > 10_000.0);
    assert!(cls.tau_static[3] < 1.0);
}

#[test]
fn rolling_ring_fits_gb10_32mib_l2_and_rotates_chronologically() {
    let mut ring = GdnSignalRingDescriptor::new(6, 8192, 48).expect("valid Qwen3.8 ring");
    // 6 * 8192 * 48 * 3 * 2 B = 14,155,776 B = 13.5 MiB.
    assert_eq!(ring.total_bytes(), 14_155_776);
    assert!(ring.fits_in_l2(GB10_L2_CACHE_BYTES));
    assert!(ring.gb10_l2_fraction() < 0.45);

    // Push 8 GDN layers (0..8): ring must retain the 6 most recent [2,3,4,5,6,7] oldest-to-newest.
    for layer_idx in 0..8 {
        let (slot, byte_off) = ring.record_gdn_layer_write(layer_idx);
        assert_eq!(slot, layer_idx % 6);
        assert_eq!(byte_off, slot * ring.slot_bytes());
    }
    let ordered = ring.ordered_active_slots();
    assert_eq!(
        ordered,
        vec![(2, 2), (3, 3), (4, 4), (5, 5), (0, 6), (1, 7)]
    );
}

#[test]
fn path_a_zero_training_block16_max_pooling_preserves_needle_block() {
    let desc = GdnSignalRingDescriptor::new(2, 64, 4).expect("ring");
    let mut host_ring = GdnSignalRingHost::new(desc);

    // Layer 0: head 1 is slow; Layer 1: head 3 is slow.
    let l0 = GdnLayerSlowHeads::classify(0, &[1.0, -5.0, 1.0, 1.0], &[1.0, -6.0, 1.0, 1.0], 1000.0)
        .expect("l0");
    let l1 = GdnLayerSlowHeads::classify(1, &[1.0, 1.0, 1.0, -5.0], &[1.0, 1.0, 1.0, -6.0], 1000.0)
        .expect("l1");
    let table = ModelSlowHeadTable::new(vec![l0, l1]);

    // 4 blocks of 16 tokens = 64 tokens. Inject one high-salience needle at token 37 (block 2).
    let mut s0 = vec![GdnSignalSample::new(0.1, -0.01, 1.0); 64 * 4];
    let mut s1 = vec![GdnSignalSample::new(0.1, -0.01, 1.0); 64 * 4];
    s0[37 * 4 + 1] = GdnSignalSample::new(0.95, -0.5, 12.0);
    s1[37 * 4 + 3] = GdnSignalSample::new(0.90, -0.4, 10.0);
    host_ring.push_layer_samples(0, 64, &s0).expect("push l0");
    host_ring.push_layer_samples(1, 64, &s1).expect("push l1");

    let scorer = ZeroTrainingBlockScorer::new(
        GATESIGHT_BLOCK_SIZE,
        DecayWeighting::ForgettingBoost { lambda: 0.5 },
    )
    .expect("scorer");
    let block_scores = scorer
        .score_blocks(&host_ring, &table, 64)
        .expect("score blocks");
    assert_eq!(block_scores.len(), 4);
    assert!(
        block_scores[2] > block_scores[0] * 10.0 && block_scores[2] > block_scores[1] * 10.0,
        "needle block 2 ({}) must strongly dominate background blocks ({:?})",
        block_scores[2],
        block_scores
    );
}

#[test]
fn two_zone_prefix_protection_never_evicts_zone1_or_sinks() {
    // 160 tokens = 10 blocks of 16. Shared prefix = 50 tokens -> Zone 1 rounds up to 4 blocks (64 toks).
    // Give Zone-1 blocks terrible scores (-100.0) and verify they stay 100% protected.
    let mut block_scores = vec![1.0f32; 10];
    block_scores[0] = -100.0;
    block_scores[1] = -100.0;
    block_scores[2] = -100.0;
    block_scores[3] = -100.0;
    block_scores[5] = 9.0; // high-salience Zone-2 block
    block_scores[7] = 8.0; // high-salience Zone-2 block

    let plan = plan_zone2_compaction(&CompactionRequest {
        mode: GateSightMode::PathABlock16,
        policy: ModelFamilyPolicy::Qwen38Dense27B,
        total_tokens: 160,
        shared_prefix_len: 50,
        block_size: GATESIGHT_BLOCK_SIZE,
        sink_blocks: 1,
        recent_window_tokens: 16, // protects block 9
        target_retention: 0.50, // Zone 2 has 6 blocks -> keep 3 (block 9 tail + top 2: blocks 5, 7)
        scores: &block_scores,
    })
    .expect("plan");

    match plan {
        CompactionPlan::Block16PathA {
            zone1_protected_blocks,
            kept_zone2_blocks,
            evicted_zone2_blocks,
            effective_retention,
            was_clamped,
        } => {
            assert_eq!(zone1_protected_blocks, 4);
            assert!(!was_clamped);
            assert!((effective_retention - 0.50).abs() < 1e-6);
            assert_eq!(kept_zone2_blocks, vec![5, 7, 9]);
            assert_eq!(evicted_zone2_blocks, vec![4, 6, 8]);
            assert!(
                evicted_zone2_blocks
                    .iter()
                    .all(|&b| b >= zone1_protected_blocks),
                "Zone-1 blocks must never be evicted"
            );
        }
        other => panic!("expected Block16PathA, got {other:?}"),
    }
}

#[test]
fn flash_next_qsa_clamps_aggressive_retention_to_0_60_safety_floor() {
    let token_scores: Vec<f32> = (0..160).map(|i| (i as f32) * 0.01).collect();
    let plan = plan_zone2_compaction(&CompactionRequest {
        mode: GateSightMode::PathBMlp,
        policy: ModelFamilyPolicy::Qwen38FlashNextQsa,
        total_tokens: 160,
        shared_prefix_len: 32,
        block_size: GATESIGHT_BLOCK_SIZE,
        sink_blocks: 1,
        recent_window_tokens: 16,
        target_retention: 0.10, // aggressive 10x request -> must clamp to 0.60!
        scores: &token_scores,
    })
    .expect("plan");

    match plan {
        CompactionPlan::TokenLevelPathB {
            zone1_protected_tokens,
            kept_zone2_tokens,
            evicted_zone2_tokens,
            requires_v_transpose_repack,
            effective_retention,
            was_clamped,
        } => {
            assert_eq!(zone1_protected_tokens, 32);
            assert!(was_clamped, "Flash-Next QSA must clamp p=0.10 to 0.60");
            assert!((effective_retention - FLASH_NEXT_QSA_MIN_RETENTION).abs() < 1e-6);
            // Zone 2 = 128 tokens, ceil(128 * 0.60) = 77 tokens kept.
            assert_eq!(kept_zone2_tokens.len(), 77);
            assert_eq!(kept_zone2_tokens.len() + evicted_zone2_tokens.len(), 128);
            assert!(requires_v_transpose_repack);
            assert!(is_bitfp4_block_aligned(zone1_protected_tokens));
        }
        other => panic!("expected TokenLevelPathB, got {other:?}"),
    }
}

#[test]
fn streaming_reservoir_topk_matches_global_topk_across_arbitrary_chunk_sizes() {
    // Deterministic pseudo-random scores including exact score ties.
    let items: Vec<ScoredItem> = (0..512)
        .map(|i| {
            let bucket = ((i * 73 + 19) % 37) as f32;
            ScoredItem::new(i, bucket)
        })
        .collect();
    let k = 96usize;

    // Global reference Top-K with identical tie-breaker: higher score, then higher index.
    let mut sorted = items.clone();
    sorted.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| b.index.cmp(&a.index))
    });
    let mut expected_kept: Vec<usize> = sorted[..k].iter().map(|s| s.index).collect();
    let mut expected_evicted: Vec<usize> = sorted[k..].iter().map(|s| s.index).collect();
    expected_kept.sort_unstable();
    expected_evicted.sort_unstable();

    for chunk_size in [1usize, 7, 16, 64, 128, 512] {
        let mut reservoir = StreamingTopKReservoir::new(k);
        for chunk in items.chunks(chunk_size) {
            let _ = reservoir.push_chunk(chunk.iter().copied());
        }
        let (kept, evicted) = reservoir.finish();
        assert_eq!(
            kept, expected_kept,
            "streaming kept mismatch for chunk_size={chunk_size}"
        );
        assert_eq!(
            evicted, expected_evicted,
            "streaming evicted mismatch for chunk_size={chunk_size}"
        );
    }
}

#[test]
fn path_b_mlp_sidecar_roundtrip_auth_and_scoring() {
    let window_gdn_layers = 6;
    let num_v_heads = 4;
    let hidden_dim = 8;
    let input_dim = window_gdn_layers * num_v_heads * 3 + 3; // 75 (same formula as 6*48*3+3 = 867)

    let mut w1 = vec![0.0f32; hidden_dim * input_dim];
    // Weight unit 0 strongly on signal 0 (beta of layer 0, head 0) and last age feature.
    w1[0] = 2.5;
    w1[input_dim - 1] = 0.5;
    let mut w2 = vec![0.0f32; hidden_dim];
    w2[0] = 1.5;

    let head = HeadMlpWeights {
        w1,
        b1: vec![0.0; hidden_dim],
        w2,
        b2: -0.5,
    };
    let sidecar = GateSightMlpSidecar {
        num_attn_layers: 2,
        num_kv_heads: 2,
        window_gdn_layers,
        num_v_heads,
        hidden_dim,
        activation: MlpActivation::Relu,
        layer_gate_weights: vec![0.05, 0.45],
        heads: vec![head; 4],
    };

    let bytes = sidecar.to_bytes().expect("serialize sidecar");
    let decoded = GateSightMlpSidecar::from_bytes(&bytes).expect("verify + decode sidecar");
    assert_eq!(decoded, sidecar);

    // Tampering with a single payload byte must fail SHA-256 verification.
    let mut tampered = bytes.clone();
    let last = tampered.len() - 1;
    tampered[last] ^= 0x01;
    assert!(matches!(
        GateSightMlpSidecar::from_bytes(&tampered),
        Err(GateSightError::InvalidSidecar(_))
    ));

    // Tampering with header byte 32 (`activation` tag: Relu 0 -> Silu 1) must also fail SHA-256.
    let mut header_tampered = bytes.clone();
    header_tampered[32] = 1;
    assert!(matches!(
        GateSightMlpSidecar::from_bytes(&header_tampered),
        Err(GateSightError::InvalidSidecar(_))
    ));

    // Score two tokens: token 1 has much higher beta on (layer 0, head 0) -> strictly higher score.
    let desc = GdnSignalRingDescriptor::new(window_gdn_layers, 16, num_v_heads).expect("ring");
    let mut ring = GdnSignalRingHost::new(desc);
    for gdn_layer in 0..window_gdn_layers {
        let mut samples = vec![GdnSignalSample::new(0.1, -0.05, 1.0); 2 * num_v_heads];
        if gdn_layer == 0 {
            samples[num_v_heads] = GdnSignalSample::new(0.95, -0.05, 1.0); // token 1, head 0
        }
        ring.push_layer_samples(gdn_layer, 2, &samples)
            .expect("push samples");
    }

    let scorer = GateSightMlpScorer::new(decoded).expect("mlp scorer");
    let scores = scorer
        .score_layer_tokens(1, &ring, 0, 2, 100)
        .expect("score layer");
    assert_eq!(scores.len(), 2);
    assert!(scores[0] > 0.0 && scores[0] < 1.0);
    assert!(
        scores[1] > scores[0],
        "expected score[1] > score[0]: {scores:?}"
    );
}

#[test]
fn cross_layer_gate_budget_allocation_preserves_exact_sum_and_order() {
    let weights = [0.022f32, 0.10, 0.25, 0.459];
    let budgets = allocate_layer_budgets(&weights, 400, 16, 256).expect("allocate");
    let total: usize = budgets.iter().map(|b| b.kept_items).sum();
    assert_eq!(total, 400);
    assert!(budgets[3].kept_items > budgets[2].kept_items);
    assert!(budgets[2].kept_items > budgets[1].kept_items);
    assert!(budgets[1].kept_items >= budgets[0].kept_items);
    assert!(
        budgets
            .iter()
            .all(|b| b.kept_items >= 16 && b.kept_items <= 256)
    );
}

#[test]
fn env_config_defaults_off_and_rejects_invalid_values() {
    let default_cfg = GateSightConfig::from_env_values(None, None, None).expect("default");
    assert_eq!(default_cfg.mode, GateSightMode::Disabled);
    assert!(!default_cfg.mode.is_enabled());

    let path_a =
        GateSightConfig::from_env_values(Some("path_a"), Some("0.5"), None).expect("path_a");
    assert_eq!(path_a.mode, GateSightMode::PathABlock16);
    assert!((path_a.target_retention - 0.5).abs() < 1e-6);

    // Path B fails closed without a sidecar path and succeeds when one is provided.
    assert!(GateSightConfig::from_env_values(Some("path_b"), Some("0.25"), None).is_err());
    let path_b =
        GateSightConfig::from_env_values(Some("path_b"), Some("0.25"), Some("/tmp/gatesight.bin"))
            .expect("path_b with sidecar");
    assert_eq!(path_b.mode, GateSightMode::PathBMlp);
    assert!(path_b.sidecar_path.is_some());

    assert!(GateSightConfig::from_env_values(Some("bogus"), None, None).is_err());
    assert!(GateSightConfig::from_env_values(Some("path_a"), Some("1.5"), None).is_err());
    assert!(GateSightConfig::from_env_values(Some("path_a"), Some("0.0"), None).is_err());
}

#[test]
fn scorer_and_allocator_reject_nan_shape_mismatch_and_overflow_without_panicking() {
    // Mismatched w2 length in score_feature must return Err, not panic.
    let bad_w2 = HeadMlpWeights {
        w1: vec![0.0; 4],
        b1: vec![0.0; 2],
        w2: vec![0.0; 1], // shorter than hidden_dim=2
        b2: 0.0,
    };
    assert!(matches!(
        bad_w2.score_feature(&[1.0, 2.0], MlpActivation::Relu),
        Err(GateSightError::DimensionMismatch { .. })
    ));

    // Non-finite weights must fail validation.
    let nan_head = HeadMlpWeights {
        w1: vec![f32::NAN; 4],
        b1: vec![0.0; 2],
        w2: vec![0.0; 2],
        b2: 0.0,
    };
    assert!(matches!(
        nan_head.validate(2, 2),
        Err(GateSightError::InvalidSidecar(_))
    ));

    // Overflowing min_per_layer * n returns InvalidConfig; usize::MAX max_per_layer saturates cleanly.
    assert!(matches!(
        allocate_layer_budgets(&[0.5, 0.5], 10, usize::MAX, usize::MAX),
        Err(GateSightError::InvalidConfig(_))
    ));
    let sat = allocate_layer_budgets(&[0.25, 0.75], 100, 0, usize::MAX)
        .expect("saturating max_per_layer");
    assert_eq!(sat[0].kept_items + sat[1].kept_items, 100);
}
