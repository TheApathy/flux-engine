// SPDX-License-Identifier: AGPL-3.0-only

use std::collections::HashSet;

use super::{
    CtxCacheInvalidationReason, CtxKvCacheRepr, DflashLayerQuantMode, DflashNoiseGraphKey,
    DflashNoiseGraphMeta, noise_graph_stochastic_draft_sampling_active,
    parse_dflash_noise_graph_env, plan_ctx_kv_cache_transition,
};

#[test]
fn noise_graph_meta_abi_layout_alignment_and_roundtrip() {
    assert_eq!(std::mem::size_of::<DflashNoiseGraphMeta>(), 80);
    assert_eq!(std::mem::align_of::<DflashNoiseGraphMeta>(), 16);
    assert_eq!(
        std::mem::offset_of!(DflashNoiseGraphMeta, eff_ctx),
        DflashNoiseGraphMeta::EFF_CTX_OFFSET
    );
    assert_eq!(
        std::mem::offset_of!(DflashNoiseGraphMeta, last_token),
        DflashNoiseGraphMeta::LAST_TOKEN_OFFSET
    );
    assert_eq!(
        std::mem::offset_of!(DflashNoiseGraphMeta, noise_len),
        DflashNoiseGraphMeta::NOISE_LEN_OFFSET
    );
    assert_eq!(std::mem::offset_of!(DflashNoiseGraphMeta, _pad), 12);
    assert_eq!(
        std::mem::offset_of!(DflashNoiseGraphMeta, noise_positions),
        DflashNoiseGraphMeta::NOISE_POSITIONS_OFFSET
    );

    let positions = [128u32, 129, 130, 131, 132, 133, 134, 135];
    let meta = DflashNoiseGraphMeta::try_new(128, 42_001, &positions, 512).unwrap();
    assert_eq!(meta.eff_ctx, 128);
    assert_eq!(meta.last_token, 42_001);
    assert_eq!(meta.noise_len, 8);
    assert_eq!(meta._pad, 0);
    assert_eq!(&meta.noise_positions[..8], &positions);
    assert_eq!(&meta.noise_positions[8..], &[0u32; 8]);

    let bytes = meta.to_le_bytes();
    assert_eq!(bytes.len(), 80);
    assert_eq!(u32::from_le_bytes(bytes[0..4].try_into().unwrap()), 128);
    assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 42_001);
    assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 8);
    assert_eq!(u32::from_le_bytes(bytes[12..16].try_into().unwrap()), 0);
    assert_eq!(u32::from_le_bytes(bytes[16..20].try_into().unwrap()), 128);
    assert_eq!(u32::from_le_bytes(bytes[44..48].try_into().unwrap()), 135);
    assert_eq!(DflashNoiseGraphMeta::from_le_bytes(&bytes).unwrap(), meta);

    let mut staging = [0xAAu8; 96];
    assert_eq!(meta.write_le_bytes(&mut staging).unwrap(), 80);
    assert_eq!(&staging[..80], &bytes);
    assert!(meta.write_le_bytes(&mut staging[..79]).is_err());
    assert!(DflashNoiseGraphMeta::from_le_bytes(&bytes[..79]).is_err());
}

#[test]
fn noise_graph_meta_rejects_out_of_bounds_eff_ctx_and_noise_positions() {
    // eff_ctx > ctx_window
    assert!(DflashNoiseGraphMeta::try_new(513, 1, &[513, 514], 512).is_err());
    // empty noise_positions
    assert!(DflashNoiseGraphMeta::try_new(16, 1, &[], 512).is_err());
    // > 16 noise_positions
    let too_many: Vec<u32> = (16..33).collect();
    assert!(DflashNoiseGraphMeta::try_new(16, 1, &too_many, 512).is_err());
    // first noise position < eff_ctx
    assert!(DflashNoiseGraphMeta::try_new(16, 1, &[15, 16], 512).is_err());
    // non-increasing noise positions
    assert!(DflashNoiseGraphMeta::try_new(16, 1, &[16, 16, 17], 512).is_err());
    assert!(DflashNoiseGraphMeta::try_new(16, 1, &[16, 18, 17], 512).is_err());
    // max valid boundary: 16 positions at eff_ctx = ctx_window
    let exact_max: Vec<u32> = (512..528).collect();
    assert!(DflashNoiseGraphMeta::try_new(512, 7, &exact_max, 512).is_ok());
}

#[test]
fn ring_wrap_around_and_repr_transitions_invalidate_cached_ctx_kv() {
    // Cold start into PostRope: no invalidation needed.
    assert_eq!(
        plan_ctx_kv_cache_transition(None, true, 0, 0),
        (CtxKvCacheRepr::PostRope, CtxCacheInvalidationReason::None)
    );
    // Steady linear growth in PostRope (needed_start == 0): keeps cache hot.
    assert_eq!(
        plan_ctx_kv_cache_transition(Some(CtxKvCacheRepr::PostRope), true, 0, 256),
        (CtxKvCacheRepr::PostRope, CtxCacheInvalidationReason::None)
    );
    // Ring-native PostRope wrap-around (needed_start > 0, want_post_rope == true):
    // keeps circular PostRope cache hot across the ctx_window boundary.
    assert_eq!(
        plan_ctx_kv_cache_transition(Some(CtxKvCacheRepr::PostRope), true, 1, 4096),
        (CtxKvCacheRepr::PostRope, CtxCacheInvalidationReason::None)
    );
    // Ring wrap-around fallback to eager (needed_start > 0, want_post_rope == false)
    // while holding PostRope cache: MUST invalidate before eager PreRope runs.
    assert_eq!(
        plan_ctx_kv_cache_transition(Some(CtxKvCacheRepr::PostRope), false, 1, 512),
        (
            CtxKvCacheRepr::PreRope,
            CtxCacheInvalidationReason::RingWrapAround
        )
    );
    // Subsequent wrapped eager steps stay in PreRope without repeated clears.
    assert_eq!(
        plan_ctx_kv_cache_transition(Some(CtxKvCacheRepr::PreRope), false, 8, 520),
        (CtxKvCacheRepr::PreRope, CtxCacheInvalidationReason::None)
    );
    // Transition from wrapped PreRope -> ring PostRope invalidates cached PreRope K.
    assert_eq!(
        plan_ctx_kv_cache_transition(Some(CtxKvCacheRepr::PreRope), true, 8, 520),
        (
            CtxKvCacheRepr::PostRope,
            CtxCacheInvalidationReason::ReprTransition {
                from: CtxKvCacheRepr::PreRope,
                to: CtxKvCacheRepr::PostRope,
            }
        )
    );
    // Non-wrapped fallback from PostRope -> PreRope (e.g. sampled request)
    // invalidates cached PostRope K so eager won't double-RoPE it.
    assert_eq!(
        plan_ctx_kv_cache_transition(Some(CtxKvCacheRepr::PostRope), false, 0, 64),
        (
            CtxKvCacheRepr::PreRope,
            CtxCacheInvalidationReason::ReprTransition {
                from: CtxKvCacheRepr::PostRope,
                to: CtxKvCacheRepr::PreRope,
            }
        )
    );
    // Transition from PreRope -> PostRope with non-empty cache invalidates
    // so Stage 1 recomputes clean PostRope K.
    assert_eq!(
        plan_ctx_kv_cache_transition(Some(CtxKvCacheRepr::PreRope), true, 0, 64),
        (
            CtxKvCacheRepr::PostRope,
            CtxCacheInvalidationReason::ReprTransition {
                from: CtxKvCacheRepr::PreRope,
                to: CtxKvCacheRepr::PostRope,
            }
        )
    );
    // Empty cache transitions need no invalidation.
    assert_eq!(
        plan_ctx_kv_cache_transition(Some(CtxKvCacheRepr::PostRope), false, 0, 0),
        (CtxKvCacheRepr::PreRope, CtxCacheInvalidationReason::None)
    );
}

#[test]
fn noise_graph_key_separates_gamma_quant_and_markov_and_env_fails_closed() {
    let mut keys = HashSet::new();
    for gamma_eff in [7usize, 8, 15] {
        for quant_mode in [
            DflashLayerQuantMode::Bf16,
            DflashLayerQuantMode::Fp8,
            DflashLayerQuantMode::Nvfp4,
            DflashLayerQuantMode::Mixed,
        ] {
            for markov_active in [false, true] {
                assert!(
                    keys.insert(DflashNoiseGraphKey {
                        gamma_eff,
                        quant_mode,
                        markov_active,
                    }),
                    "duplicate noise graph key for ({gamma_eff}, {quant_mode:?}, {markov_active})"
                );
            }
        }
    }
    assert_eq!(keys.len(), 3 * 4 * 2);

    assert!(!parse_dflash_noise_graph_env(None).unwrap());
    assert!(!parse_dflash_noise_graph_env(Some("0")).unwrap());
    assert!(parse_dflash_noise_graph_env(Some("1")).unwrap());
    assert!(parse_dflash_noise_graph_env(Some(" 1 ")).unwrap());
    for bad in ["", "2", "true", "yes", "-1"] {
        assert!(parse_dflash_noise_graph_env(Some(bad)).is_err());
    }

    // SM-aware auto-enable: <= 64 SMs (GB10) stays off when unset; > 64 SMs
    // (SM120/SM103) auto-enables when unset, while "0" explicitly opts out.
    for sm in [0u32, 48, 64] {
        assert!(
            !super::parse_dflash_noise_graph_env_for_sm(None, sm).unwrap(),
            "sm={sm} must stay off when unset"
        );
        assert!(
            !super::parse_dflash_noise_graph_env_for_sm(Some("0"), sm).unwrap(),
            "sm={sm}"
        );
        assert!(
            super::parse_dflash_noise_graph_env_for_sm(Some("1"), sm).unwrap(),
            "sm={sm}"
        );
    }
    for sm in [65u32, 110, 170, 188] {
        assert!(
            super::parse_dflash_noise_graph_env_for_sm(None, sm).unwrap(),
            "sm={sm} must auto-enable when unset"
        );
        assert!(
            !super::parse_dflash_noise_graph_env_for_sm(Some("0"), sm).unwrap(),
            "sm={sm} must honor explicit opt-out"
        );
        assert!(
            super::parse_dflash_noise_graph_env_for_sm(Some("1"), sm).unwrap(),
            "sm={sm}"
        );
        for bad in ["", "2", "true", "off"] {
            assert!(
                super::parse_dflash_noise_graph_env_for_sm(Some(bad), sm).is_err(),
                "sm={sm} bad={bad:?}"
            );
        }
    }
}

#[test]
fn noise_graph_allows_greedy_draft_with_positive_target_temperature() {
    use crate::dflash_sampling::{DflashDraftSampleMethod, DflashDraftSamplingRequest};
    use spark_runtime::sampler::SamplingParams;

    let mut sampled_params = SamplingParams::greedy(64);
    sampled_params.temperature = 0.7;
    sampled_params.seed = Some(1337);
    let sampled_req =
        DflashDraftSamplingRequest::from_seeded_sampling_params(&sampled_params, &[], 0, 15)
            .unwrap();

    let mut zero_temp_params = SamplingParams::greedy(64);
    zero_temp_params.seed = Some(1337);
    assert!(
        DflashDraftSamplingRequest::from_seeded_sampling_params(&zero_temp_params, &[], 0, 15)
            .is_err(),
        "zero-temperature request must not construct a stochastic draft sampling request"
    );

    // Greedy drafting paired with target Gumbel verify (temperature > 0.0)
    // keeps the noise pass deterministic and must allow Stage-2 CUDA graph replay.
    assert!(!noise_graph_stochastic_draft_sampling_active(
        DflashDraftSampleMethod::Greedy,
        None,
    ));
    assert!(!noise_graph_stochastic_draft_sampling_active(
        DflashDraftSampleMethod::Greedy,
        Some(&sampled_req),
    ));
    assert!(!noise_graph_stochastic_draft_sampling_active(
        DflashDraftSampleMethod::Probabilistic,
        None,
    ));
    // Stochastic draft sampling inside the noise pass must still bypass the graph.
    assert!(noise_graph_stochastic_draft_sampling_active(
        DflashDraftSampleMethod::Probabilistic,
        Some(&sampled_req),
    ));
}

#[test]
fn c1_gumbel_meta_upload_encodes_active_greedy_and_rotating_slots() {
    use super::{GumbelScratchPtrs, NoiseGraphHeadState, gumbel_spec_fields};
    use spark_runtime::gpu::GpuBackend;
    use spark_runtime::gpu::mock::MockGpuBackend;
    use spark_runtime::sampler::gumbel::GumbelDraftSpec;

    let gpu = MockGpuBackend::new();
    let ptrs = GumbelScratchPtrs {
        seeds: gpu.alloc(64 * 8).unwrap(),
        step0_pos: gpu.alloc(64 * 8).unwrap(),
        temps: gpu.alloc(64 * 4).unwrap(),
        top_ks: gpu.alloc(64 * 4).unwrap(),
        top_ps: gpu.alloc(64 * 4).unwrap(),
        min_ps: gpu.alloc(64 * 4).unwrap(),
    };
    let anchor_dev = gpu.alloc(64 * 4).unwrap();
    let mut state = NoiseGraphHeadState::new();

    let active = GumbelDraftSpec {
        seed: 0xDEAD_BEEF_CAFE_0001,
        temperature: 0.75,
        top_k: 20,
        top_p: 0.95,
        min_p: 0.05,
        step0_abs_pos: 1234,
    };
    assert_eq!(
        gumbel_spec_fields(Some(active)),
        (0xDEAD_BEEF_CAFE_0001, 1234, 0.75, 20, 0.95, 0.05)
    );
    assert_eq!(gumbel_spec_fields(None), (0, 0, 0.0, 0, 1.0, 0.0));
    assert_eq!(
        gumbel_spec_fields(Some(GumbelDraftSpec {
            temperature: 0.0,
            ..active
        })),
        (0, 0, 0.0, 0, 1.0, 0.0)
    );

    // Rotate past NOISE_META_PINNED_SLOTS to verify slot wraparound.
    for step in 0..6u64 {
        let spec = GumbelDraftSpec {
            seed: active.seed + step,
            step0_abs_pos: active.step0_abs_pos + step,
            ..active
        };
        state
            .upload_gumbel_meta_c1(
                &gpu,
                Some(spec),
                ptrs,
                Some((700 + step as u32, anchor_dev)),
                0,
            )
            .unwrap();
    }
    assert_eq!(
        u64::from_le_bytes(gpu.read_alloc(ptrs.seeds).unwrap()[..8].try_into().unwrap()),
        active.seed + 5
    );
    assert_eq!(
        u64::from_le_bytes(
            gpu.read_alloc(ptrs.step0_pos).unwrap()[..8]
                .try_into()
                .unwrap()
        ),
        active.step0_abs_pos + 5
    );
    assert_eq!(
        f32::from_le_bytes(gpu.read_alloc(ptrs.temps).unwrap()[..4].try_into().unwrap()),
        0.75
    );
    assert_eq!(
        u32::from_le_bytes(
            gpu.read_alloc(ptrs.top_ks).unwrap()[..4]
                .try_into()
                .unwrap()
        ),
        20
    );
    assert_eq!(
        f32::from_le_bytes(
            gpu.read_alloc(ptrs.top_ps).unwrap()[..4]
                .try_into()
                .unwrap()
        ),
        0.95
    );
    assert_eq!(
        f32::from_le_bytes(
            gpu.read_alloc(ptrs.min_ps).unwrap()[..4]
                .try_into()
                .unwrap()
        ),
        0.05
    );
    assert_eq!(
        u32::from_le_bytes(gpu.read_alloc(anchor_dev).unwrap()[..4].try_into().unwrap()),
        705
    );

    // Switching back to greedy resets temp to 0.0 without touching anchor_dev when None.
    state
        .upload_gumbel_meta_c1(&gpu, None, ptrs, None, 0)
        .unwrap();
    assert_eq!(
        f32::from_le_bytes(gpu.read_alloc(ptrs.temps).unwrap()[..4].try_into().unwrap()),
        0.0
    );
    assert_eq!(
        u32::from_le_bytes(gpu.read_alloc(anchor_dev).unwrap()[..4].try_into().unwrap()),
        705
    );

    // Rejection staging scales draft temp, validates slot*8+gamma bounds, and uploads q_row_base.
    let q_row_base_dev = gpu.alloc(64 * 4).unwrap();
    let (scaled, base_row, rec) =
        super::resolve_rejection_seq_staging(Some(active), Some(5), 4, 512, 0.5);
    assert_eq!(base_row, 40);
    assert_eq!(rec, Some((5, active.step0_abs_pos, 4)));
    assert!((scaled.unwrap().temperature - 0.375).abs() < 1.0e-6);
    // Out-of-bounds slot or missing slot degrades to greedy + None record.
    assert_eq!(
        super::resolve_rejection_seq_staging(Some(active), Some(64), 4, 512, 1.0),
        (None, 0, None)
    );
    assert_eq!(
        super::resolve_rejection_seq_staging(Some(active), None, 4, 512, 1.0),
        (None, 0, None)
    );
    state
        .upload_gumbel_meta_c1_with_q_base(
            &gpu,
            scaled,
            ptrs,
            None,
            Some((base_row, q_row_base_dev)),
            0,
        )
        .unwrap();
    assert_eq!(
        u32::from_le_bytes(
            gpu.read_alloc(q_row_base_dev).unwrap()[..4]
                .try_into()
                .unwrap()
        ),
        40
    );
}

#[test]
fn c1_stage2_graph_key_isolates_shortlist_k_and_tc_lm_head_modes() {
    use super::{
        LmHeadNvfp4Mode, NoiseGraphRuntimeKey, resolve_c1_lm_head_nvfp4_mode_for_sm,
        resolve_dflash2_selector_topk, should_use_coalesced_nvfp4_lm_head,
        validate_selector_shortlist,
    };

    // GB10 (<= 64 SMs) preserves Exact in greedy mode and CoalescedTc in sampled mode;
    // SM120/SM103 (> 64 SMs) uses CoalescedTc even in C=1 greedy mode (lossless under
    // target verification).
    for gb10_sms in [0u32, 48, 64] {
        assert_eq!(
            resolve_c1_lm_head_nvfp4_mode_for_sm(false, gb10_sms),
            LmHeadNvfp4Mode::Exact,
            "GB10 sm_count={gb10_sms} greedy must stay Exact"
        );
        assert_eq!(
            resolve_c1_lm_head_nvfp4_mode_for_sm(true, gb10_sms),
            LmHeadNvfp4Mode::CoalescedTc,
            "GB10 sm_count={gb10_sms} sampled must use CoalescedTc"
        );
    }
    for high_sms in [65u32, 110, 170, 188] {
        assert_eq!(
            resolve_c1_lm_head_nvfp4_mode_for_sm(false, high_sms),
            LmHeadNvfp4Mode::CoalescedTc,
            "SM120/SM103 sm_count={high_sms} greedy must use CoalescedTc"
        );
        assert_eq!(
            resolve_c1_lm_head_nvfp4_mode_for_sm(true, high_sms),
            LmHeadNvfp4Mode::CoalescedTc,
            "SM120/SM103 sm_count={high_sms} sampled must use CoalescedTc"
        );
    }

    // C=1 T=0 greedy stays on exact scalar LM head for gamma_eff <= 32,
    // while rows > 32 or CoalescedTc (rows >= 2) route to coalesced NVFP4 TC.
    assert!(!should_use_coalesced_nvfp4_lm_head(
        LmHeadNvfp4Mode::Exact,
        1
    ));
    assert!(!should_use_coalesced_nvfp4_lm_head(
        LmHeadNvfp4Mode::Exact,
        7
    ));
    assert!(!should_use_coalesced_nvfp4_lm_head(
        LmHeadNvfp4Mode::Exact,
        15
    ));
    assert!(!should_use_coalesced_nvfp4_lm_head(
        LmHeadNvfp4Mode::Exact,
        32
    ));
    assert!(should_use_coalesced_nvfp4_lm_head(
        LmHeadNvfp4Mode::Exact,
        48
    ));
    assert!(!should_use_coalesced_nvfp4_lm_head(
        LmHeadNvfp4Mode::CoalescedTc,
        1
    ));
    assert!(should_use_coalesced_nvfp4_lm_head(
        LmHeadNvfp4Mode::CoalescedTc,
        2
    ));
    assert!(should_use_coalesced_nvfp4_lm_head(
        LmHeadNvfp4Mode::CoalescedTc,
        7
    ));
    assert!(should_use_coalesced_nvfp4_lm_head(
        LmHeadNvfp4Mode::CoalescedTc,
        48
    ));

    // Keeps checkpoint-trained selector_top_k (16) while enabling fast radix top-k + out_probs.
    assert_eq!(
        resolve_dflash2_selector_topk(16, 248_320, true, true, true),
        (16, true, true)
    );
    assert_eq!(
        resolve_dflash2_selector_topk(16, 248_320, false, true, true),
        (16, true, true)
    );
    assert_eq!(
        resolve_dflash2_selector_topk(16, 248_320, true, false, true),
        (16, false, true)
    );
    assert_eq!(
        resolve_dflash2_selector_topk(16, 248_320, true, true, false),
        (16, true, false)
    );
    // Clamps to small vocab cleanly.
    assert_eq!(
        resolve_dflash2_selector_topk(16, 12, true, true, true),
        (12, true, true)
    );

    assert_eq!(
        validate_selector_shortlist(7, 15, 248_320, 32, 32).unwrap(),
        (7, 248_320, 32)
    );
    assert!(validate_selector_shortlist(7, 15, 248_320, 33, 32).is_err());

    let base = NoiseGraphRuntimeKey {
        logical: DflashNoiseGraphKey {
            gamma_eff: 7,
            quant_mode: DflashLayerQuantMode::Nvfp4,
            markov_active: false,
        },
        attn_mode: super::Stage2CtxAttnMode::Linear,
        split_kv_splits: 1,
        lm_head_path_tag: 2,
        prefer_tc_lm_head: false,
        rejection_mode: false,
        ngram_mode: false,
        shortlist_k: 16,
        mask_id: 248_077,
        stream: 7,
        ctx_kv_ptrs: vec![(0x1000, 0x2000)],
    };
    let split_kv_mode = NoiseGraphRuntimeKey {
        split_kv_splits: 6,
        ..base.clone()
    };
    let ring_mode = NoiseGraphRuntimeKey {
        attn_mode: super::Stage2CtxAttnMode::CircularRingH128,
        ..base.clone()
    };
    let sampled = NoiseGraphRuntimeKey {
        prefer_tc_lm_head: true,
        shortlist_k: 32,
        ..base.clone()
    };
    let tc_only = NoiseGraphRuntimeKey {
        prefer_tc_lm_head: true,
        ..base.clone()
    };
    let k32_only = NoiseGraphRuntimeKey {
        shortlist_k: 32,
        ..base.clone()
    };
    let rejection_only = NoiseGraphRuntimeKey {
        rejection_mode: true,
        shortlist_k: 20,
        ..base.clone()
    };
    let ngram_only = NoiseGraphRuntimeKey {
        ngram_mode: true,
        ..base.clone()
    };
    let set: HashSet<NoiseGraphRuntimeKey> = [
        base,
        split_kv_mode,
        ring_mode,
        sampled,
        tc_only,
        k32_only,
        rejection_only,
        ngram_only,
    ]
    .into_iter()
    .collect();
    assert_eq!(set.len(), 8);
}

#[test]
fn dflash_split_kv_policy_saturates_wide_sm_and_preserves_gb10_and_short_ctx() {
    use super::{
        DFLASH_SPLIT_KV_H128_MAX_SPLITS, DflashNoiseGraphMeta, NoiseGraphHeadState,
        dflash_split_kv_h128_workspace_bytes, resolve_dflash_split_kv_splits_for_sm,
    };
    use spark_runtime::gpu::mock::MockGpuBackend;

    // GB10 (<= 64 SMs), short context (< 256), and non-H128 stay single-split (1).
    for sm in [0u32, 48, 64] {
        for eff_ctx in [0usize, 255, 256, 2048, 4096] {
            assert_eq!(
                resolve_dflash_split_kv_splits_for_sm(128, eff_ctx, 32, 1, sm),
                1
            );
        }
    }
    assert_eq!(
        resolve_dflash_split_kv_splits_for_sm(128, 255, 32, 1, 188),
        1
    );
    assert_eq!(
        resolve_dflash_split_kv_splits_for_sm(256, 2048, 32, 1, 188),
        1
    );
    assert_eq!(
        resolve_dflash_split_kv_splits_for_sm(128, 2048, 32, 6, 188),
        1
    );

    // Wide-SM (> 64 SMs, e.g. 188-SM SM120) lifts C=1..4 from 32..128 CTAs to 192..512 CTAs.
    let s_c1 = resolve_dflash_split_kv_splits_for_sm(128, 256, 32, 1, 188);
    assert_eq!(s_c1, 6);
    assert_eq!(32 * s_c1, 192);
    let s_c2 = resolve_dflash_split_kv_splits_for_sm(128, 2048, 32, 2, 188);
    assert_eq!(s_c2, 4);
    assert_eq!(32 * 2 * s_c2, 256);
    let s_c4 = resolve_dflash_split_kv_splits_for_sm(128, 2048, 32, 4, 188);
    assert_eq!(s_c4, 4);
    assert_eq!(32 * 4 * s_c4, 512);
    // Smaller head counts clamp cleanly at 8 splits.
    assert_eq!(
        resolve_dflash_split_kv_splits_for_sm(128, 2048, 16, 1, 188),
        8
    );

    // Workspace allocation is fixed-pointer capture-safe and fails closed on oversize demand.
    let alloc_bytes = dflash_split_kv_h128_workspace_bytes(
        1,
        DFLASH_SPLIT_KV_H128_MAX_SPLITS,
        32,
        DflashNoiseGraphMeta::MAX_NOISE_POSITIONS,
    )
    .expect("workspace bytes");
    assert_eq!(alloc_bytes, 8 * 32 * 16 * 130 * 4);
    let needed_bytes = dflash_split_kv_h128_workspace_bytes(1, 6, 32, 8).expect("needed");
    let gpu = MockGpuBackend::new();
    let mut state = NoiseGraphHeadState::new();
    let ptr1 = state
        .ensure_split_kv_workspace(&gpu, alloc_bytes, needed_bytes)
        .expect("first alloc")
        .expect("workspace ptr");
    assert_eq!(gpu.alloc_count(), 1);
    let ptr2 = state
        .ensure_split_kv_workspace(&gpu, alloc_bytes, alloc_bytes)
        .expect("reuse alloc")
        .expect("same ptr");
    assert_eq!(ptr1, ptr2);
    assert_eq!(gpu.alloc_count(), 1);
    assert_eq!(
        state
            .ensure_split_kv_workspace(&gpu, alloc_bytes, alloc_bytes + 4)
            .expect("oversize fail-closed"),
        None
    );
}

#[test]
fn stage1_ring_append_plan_covers_linear_wrapped_two_span_prefill_and_rollback() {
    use super::{Stage1RingSpan, all_layers_cover_old_fc_prefix, plan_stage1_ring_append};

    // 1. needed_start == 0 cold start and steady incremental append.
    let cold = plan_stage1_ring_append(0, 128, 4096, 0, 0, 0, 0).unwrap();
    assert_eq!(cold.old_ctx_count, 0);
    assert_eq!(cold.new_abs_start, 0);
    assert_eq!(cold.new_ctx_count, 128);
    assert_eq!(
        cold.spans(),
        &[Stage1RingSpan {
            linear_row: 0,
            physical_slot: 0,
            count: 128,
        }]
    );

    let linear_step = plan_stage1_ring_append(0, 134, 4096, 0, 128, 0, 128).unwrap();
    assert_eq!(linear_step.old_ctx_count, 128);
    assert_eq!(linear_step.new_abs_start, 128);
    assert_eq!(linear_step.new_ctx_count, 6);
    assert_eq!(
        linear_step.spans(),
        &[Stage1RingSpan {
            linear_row: 128,
            physical_slot: 128,
            count: 6,
        }]
    );

    // 2. Initial post-prefill fill at 136.6K (new_count == window, 2 spans).
    //    needed=[132574..136670), 132574 % 4096 == 1502 -> [1502..4096) + [0..1502).
    let prefill_136k = plan_stage1_ring_append(132_574, 4096, 4096, 0, 0, 0, 0).unwrap();
    assert_eq!(prefill_136k.old_ctx_count, 0);
    assert_eq!(prefill_136k.new_abs_start, 132_574);
    assert_eq!(prefill_136k.new_ctx_count, 4096);
    assert_eq!(
        prefill_136k.spans(),
        &[
            Stage1RingSpan {
                linear_row: 0,
                physical_slot: 1502,
                count: 2594,
            },
            Stage1RingSpan {
                linear_row: 2594,
                physical_slot: 0,
                count: 1502,
            },
        ]
    );

    // 3. Steady decode at 136.6K (needed_start > 0, single physical span).
    let decode_136k =
        plan_stage1_ring_append(132_579, 4096, 4096, 132_574, 136_670, 132_574, 136_670).unwrap();
    assert_eq!(decode_136k.old_ctx_count, 4091);
    assert_eq!(decode_136k.new_abs_start, 136_670);
    assert_eq!(decode_136k.new_ctx_count, 5);
    assert_eq!(
        decode_136k.spans(),
        &[Stage1RingSpan {
            linear_row: 4091,
            physical_slot: 1502,
            count: 5,
        }]
    );

    // 4. Crossing the 4096 boundary from 4095 -> 4099 (2-span wrap-around).
    let wrap_4k = plan_stage1_ring_append(3, 4096, 4096, 0, 4095, 0, 4095).unwrap();
    assert_eq!(wrap_4k.old_ctx_count, 4092);
    assert_eq!(wrap_4k.new_abs_start, 4095);
    assert_eq!(wrap_4k.new_ctx_count, 4);
    assert_eq!(
        wrap_4k.spans(),
        &[
            Stage1RingSpan {
                linear_row: 4092,
                physical_slot: 4095,
                count: 1,
            },
            Stage1RingSpan {
                linear_row: 4093,
                physical_slot: 0,
                count: 3,
            },
        ]
    );

    // 5. High-context 2-span wrap crossing 4096 * 34 = 139264.
    let wrap_139k =
        plan_stage1_ring_append(135_170, 4096, 4096, 135_166, 139_262, 135_166, 139_262).unwrap();
    assert_eq!(wrap_139k.old_ctx_count, 4092);
    assert_eq!(wrap_139k.new_abs_start, 139_262);
    assert_eq!(wrap_139k.new_ctx_count, 4);
    assert_eq!(
        wrap_139k.spans(),
        &[
            Stage1RingSpan {
                linear_row: 4092,
                physical_slot: 4094,
                count: 2,
            },
            Stage1RingSpan {
                linear_row: 4094,
                physical_slot: 0,
                count: 2,
            },
        ]
    );

    // 6. Rollback invalidation (cache_end > needed_end) forces full recompute.
    let rollback_wrapped =
        plan_stage1_ring_append(132_570, 4096, 4096, 132_574, 136_670, 132_574, 136_670).unwrap();
    assert_eq!(rollback_wrapped.old_ctx_count, 0);
    assert_eq!(rollback_wrapped.new_ctx_count, 4096);

    let rollback_linear = plan_stage1_ring_append(0, 400, 4096, 0, 500, 0, 500).unwrap();
    assert_eq!(rollback_linear.old_ctx_count, 0);
    assert_eq!(rollback_linear.new_ctx_count, 400);

    // 7. Forward jump exceeding window forces full recompute.
    let jump = plan_stage1_ring_append(10_000, 4096, 4096, 0, 4096, 0, 4096).unwrap();
    assert_eq!(jump.old_ctx_count, 0);
    assert_eq!(jump.new_ctx_count, 4096);

    // 8. Exact cache hit emits zero spans.
    let hit =
        plan_stage1_ring_append(132_574, 4096, 4096, 132_574, 136_670, 132_574, 136_670).unwrap();
    assert_eq!(hit.old_ctx_count, 4096);
    assert_eq!(hit.new_ctx_count, 0);
    assert!(hit.spans().is_empty());

    // 9. Multi-layer `skip_old_fc_copy` coverage predicate across wrapped and rollback states.
    assert!(all_layers_cover_old_fc_prefix(
        132_578,
        4096,
        4092,
        4096,
        &[132_574; 8],
        &[136_670; 8],
        &[132_574; 8],
        &[136_670; 8],
    ));
    // One lagging layer must refuse skip.
    let mut lagging_v_end = [136_670; 8];
    lagging_v_end[5] = 136_669;
    assert!(!all_layers_cover_old_fc_prefix(
        132_578,
        4096,
        4092,
        4096,
        &[132_574; 8],
        &[136_670; 8],
        &[132_574; 8],
        &lagging_v_end,
    ));
    // Rollback in any layer must refuse skip.
    assert!(!all_layers_cover_old_fc_prefix(
        132_574,
        4096,
        4092,
        4096,
        &[132_574; 8],
        &[136_675; 8],
        &[132_574; 8],
        &[136_675; 8],
    ));
}

#[test]
fn stage1_ring_simulation_preserves_exact_chronological_slots_across_wraps() {
    use super::plan_stage1_ring_append;

    const WINDOW: usize = 64;
    let mut ring_slots = vec![usize::MAX; WINDOW];
    let mut cache_start = 0usize;
    let mut cache_end = 0usize;
    let mut pos = 0usize;

    // Walk from cold start through prefill + dozens of variable-width decode
    // steps (1..=8 tokens) well past multiple ring wraps.
    let deltas = [
        20usize, 5, 7, 3, 8, 6, 8, 5, 4, 3, 1, 8, 7, 6, 5, 8, 2, 4, 7, 8, 3, 6, 8, 7,
    ];
    for &delta in &deltas {
        pos += delta;
        let eff_ctx = pos.min(WINDOW);
        let needed_start = pos - eff_ctx;
        let plan = plan_stage1_ring_append(
            needed_start,
            eff_ctx,
            WINDOW,
            cache_start,
            cache_end,
            cache_start,
            cache_end,
        )
        .unwrap();
        for span in plan.spans() {
            for j in 0..span.count {
                let abs_pos = needed_start + span.linear_row + j;
                let phys = span.physical_slot + j;
                assert_eq!(phys, abs_pos % WINDOW);
                ring_slots[phys] = abs_pos;
            }
        }
        cache_start = needed_start;
        cache_end = pos;

        // Verify Stage-2 circular attention indexing `(ring_start + k_row) % WINDOW`
        // recovers exact chronological tokens `[needed_start..pos)`.
        let ring_start = needed_start % WINDOW;
        for k_row in 0..eff_ctx {
            let mut phys = ring_start + k_row;
            if phys >= WINDOW {
                phys -= WINDOW;
            }
            assert_eq!(
                ring_slots[phys],
                needed_start + k_row,
                "pos={pos} needed_start={needed_start} k_row={k_row}"
            );
        }
    }
}

#[test]
fn stage2_ctx_attn_mode_preserves_linear_under_window_and_engages_ring_past_window() {
    use super::{Stage2CtxAttnMode, resolve_stage2_ctx_attn_mode};

    // needed_start == 0 stays Linear (0-ULP parity for seq_len <= 4096) even when ring kernel exists.
    assert_eq!(
        resolve_stage2_ctx_attn_mode(128, 0, 256, 4096, true, true, true),
        Some(Stage2CtxAttnMode::Linear)
    );
    assert_eq!(
        resolve_stage2_ctx_attn_mode(128, 0, 4096, 4096, true, true, true),
        Some(Stage2CtxAttnMode::Linear)
    );
    // needed_start > 0 engages CircularRingH128 for HD=128 with SWA, full ring, or partial ring.
    assert_eq!(
        resolve_stage2_ctx_attn_mode(128, 1, 4096, 4096, true, true, true),
        Some(Stage2CtxAttnMode::CircularRingH128)
    );
    assert_eq!(
        resolve_stage2_ctx_attn_mode(128, 132_574, 4096, 4096, false, true, true),
        Some(Stage2CtxAttnMode::CircularRingH128)
    );
    // Partial ring (e.g. prefix-cache skip with eff_ctx < window) works with CircularRingH128.
    assert_eq!(
        resolve_stage2_ctx_attn_mode(128, 10_000, 50, 4096, true, true, true),
        Some(Stage2CtxAttnMode::CircularRingH128)
    );
    // HD=256 (no ring kernel): non-SWA full ring allowed on Linear; SWA or partial misaligned ring fails closed.
    assert_eq!(
        resolve_stage2_ctx_attn_mode(256, 132_574, 4096, 4096, true, false, false),
        Some(Stage2CtxAttnMode::Linear)
    );
    assert_eq!(
        resolve_stage2_ctx_attn_mode(256, 132_574, 4096, 4096, true, false, true),
        None
    );
    assert_eq!(
        resolve_stage2_ctx_attn_mode(256, 10_000, 50, 4096, true, false, false),
        None
    );
    // Aligned partial ring on non-SWA Linear is valid because slots [0..eff_ctx) match.
    assert_eq!(
        resolve_stage2_ctx_attn_mode(256, 8192, 50, 4096, true, false, false),
        Some(Stage2CtxAttnMode::Linear)
    );
    // Invalid geometry rejected.
    assert_eq!(
        resolve_stage2_ctx_attn_mode(128, 0, 4097, 4096, true, true, false),
        None
    );
    assert_eq!(
        resolve_stage2_ctx_attn_mode(64, 0, 128, 4096, true, true, false),
        None
    );
    // High-SM GPUs (SM120, >64 SMs) prefer pointer-agnostic CircularRingH128
    // even at needed_start == 0 so Stage-2 CUDA graphs reuse cleanly across
    // requests; GB10 (<=64 SMs) preserves Linear bit-for-bit.
    assert_eq!(
        super::resolve_stage2_ctx_attn_mode_for_sm(128, 0, 256, 4096, true, true, true, 188),
        Some(Stage2CtxAttnMode::CircularRingH128)
    );
    assert_eq!(
        super::resolve_stage2_ctx_attn_mode_for_sm(128, 0, 256, 4096, true, true, true, 48),
        Some(Stage2CtxAttnMode::Linear)
    );
    assert_eq!(
        super::resolve_stage2_ctx_attn_mode_for_sm(256, 0, 256, 4096, true, false, true, 188),
        Some(Stage2CtxAttnMode::Linear)
    );
}

#[test]
fn c1_ring_meta_upload_encodes_aligned_ring_start_and_per_layer_kv_ptrs() {
    use super::{NoiseGraphHeadState, RingMetaLayout};
    use spark_runtime::gpu::DevicePtr;
    use spark_runtime::gpu::mock::MockGpuBackend;

    let gpu = MockGpuBackend::new();
    let mut state = NoiseGraphHeadState::new();
    let k_ptrs = [
        DevicePtr(0x10_0000),
        DevicePtr(0x11_0000),
        DevicePtr(0x12_0000),
        DevicePtr(0x13_0000),
    ];
    let v_ptrs = [
        DevicePtr(0x20_0000),
        DevicePtr(0x21_0000),
        DevicePtr(0x22_0000),
        DevicePtr(0x23_0000),
    ];
    let layout = RingMetaLayout::new(k_ptrs.len()).unwrap();
    assert_eq!(layout.ring_start_off, 0);
    assert_eq!(layout.ctx_k_ptrs_off, 8);
    assert_eq!(layout.ctx_v_ptrs_off, 8 + 4 * 8);
    assert_eq!(layout.total_bytes, 8 + 8 * 8);

    // Rotate past NOISE_META_PINNED_SLOTS to verify slot wraparound and fixed device ptr.
    let mut first_base = DevicePtr::NULL;
    for step in 0..6u32 {
        let ptrs = state
            .upload_ring_meta(&gpu, 1502 + step, &k_ptrs, &v_ptrs, 0)
            .unwrap();
        if step == 0 {
            first_base = ptrs.base;
        } else {
            assert_eq!(ptrs.base, first_base);
        }
        assert_eq!(ptrs.ring_start_dev, ptrs.base);
        assert_eq!(ptrs.layer_k_ptr(2).0, ptrs.base.0 + 8 + 16);
        assert_eq!(ptrs.layer_v_ptr(3).0, ptrs.base.0 + 8 + 32 + 24);
    }

    let dev_bytes = gpu.read_alloc(first_base).unwrap();
    assert_eq!(
        u32::from_le_bytes(dev_bytes[0..4].try_into().unwrap()),
        1507
    );
    assert_eq!(u32::from_le_bytes(dev_bytes[4..8].try_into().unwrap()), 0);
    for (l, (&k_ptr, &v_ptr)) in k_ptrs.iter().zip(v_ptrs.iter()).enumerate() {
        let k_off = layout.ctx_k_ptrs_off + l * 8;
        let v_off = layout.ctx_v_ptrs_off + l * 8;
        assert_eq!(
            u64::from_le_bytes(dev_bytes[k_off..k_off + 8].try_into().unwrap()),
            k_ptr.0
        );
        assert_eq!(
            u64::from_le_bytes(dev_bytes[v_off..v_off + 8].try_into().unwrap()),
            v_ptr.0
        );
    }
    // Layer count mismatch fails closed instead of overflowing the fixed slab.
    assert!(
        state
            .upload_ring_meta(&gpu, 0, &k_ptrs[..2], &v_ptrs[..2], 0)
            .is_err()
    );
}

#[test]
fn bf16_proj_cublas_alignment_accepts_conv_selector_and_rejects_odd_or_misaligned() {
    use super::bf16_proj_cublas_aligned;
    use spark_runtime::gpu::DevicePtr;

    let p = DevicePtr(0x1_0000);
    // DFlash2 conv kernel_projection ([M, 2560] x [640, 2560]^T) and
    // selector hidden_projection ([M, 2560] x [256, 2560]^T) across M=1..128.
    for m in [1u32, 3, 4, 8, 16, 48, 64, 128] {
        assert!(bf16_proj_cublas_aligned(p, p, p, m, 640, 2560));
        assert!(bf16_proj_cublas_aligned(p, p, p, m, 256, 2560));
    }
    // Odd vocab (248_077), zero dims, null pointers, and unaligned pointers
    // fail closed to scalar dense_gemm.
    assert!(!bf16_proj_cublas_aligned(p, p, p, 4, 248_077, 2560));
    assert!(!bf16_proj_cublas_aligned(p, p, p, 0, 640, 2560));
    assert!(!bf16_proj_cublas_aligned(p, p, p, 4, 0, 2560));
    assert!(!bf16_proj_cublas_aligned(p, p, p, 4, 640, 0));
    assert!(!bf16_proj_cublas_aligned(p, p, p, 4, 640, 2564));
    assert!(!bf16_proj_cublas_aligned(
        DevicePtr::NULL,
        p,
        p,
        4,
        640,
        2560
    ));
    assert!(!bf16_proj_cublas_aligned(
        p,
        DevicePtr::NULL,
        p,
        4,
        640,
        2560
    ));
    assert!(!bf16_proj_cublas_aligned(
        p,
        p,
        DevicePtr::NULL,
        4,
        640,
        2560
    ));
    assert!(!bf16_proj_cublas_aligned(
        DevicePtr(0x1_0008),
        p,
        p,
        4,
        640,
        2560
    ));
}
