// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::gpu::mock::MockGpuBackend;

#[test]
fn test_buffer_sizes_qwen3() {
    let cfg = ModelConfig::qwen3_next_80b_nvfp4();
    let sizes = BufferSizes::from_config(&cfg, 1, 4096, 16, 1).unwrap();

    // hidden_states: 1 * 2048 * 2 = 4096 (BF16, 2 bytes/elem).
    // (Was FP32 = 8192 in earlier prototypes; NVFP4 path keeps the
    // residual stream in BF16, halving the buffer size.)
    assert_eq!(sizes.hidden_states, 4096);
    // qkv: 1 * (16*2 + 2*2) * 256 * 2 = 1 * 36 * 256 * 2 = 18432
    // Q+gate: 16*2*256, K: 2*256, V: 2*256
    assert_eq!(sizes.qkv_output, 18432);
    // attn: 1 * 16 * 256 * 2 = 8192
    assert_eq!(sizes.attn_output, 8192);
    // gate: 1 * 512 * 2 = 1024
    assert_eq!(sizes.gate_logits, 1024);
    // logits: 1 * 151936 * 2 = 303872
    assert_eq!(sizes.logits, 303872);
    // ssm_qkvz: 1 * 12288 * 2 = 24576
    // Q(16*128) + K(16*128) + V(32*128) + Z(32*128) = 12288
    assert_eq!(sizes.ssm_qkvz, 24576);
    // ssm_ba: max(1 * 64 * 2, 256) = 256 (minimum allocation)
    assert_eq!(sizes.ssm_ba, 256);
    // ssm_deinterleaved: same as ssm_qkvz = 24576
    assert_eq!(sizes.ssm_deinterleaved, 24576);
    // ssm_gates: 1 * 32 * 2 * 4 = 256 (FP32 gate + beta, scaled by M)
    assert_eq!(sizes.ssm_gates, 256);
}

#[test]
fn test_buffer_arena_alloc() {
    let cfg = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let arena = BufferArena::new(&cfg, 128, 4096, 16, 128, &gpu).unwrap();

    assert!(!arena.hidden_states().is_null());
    assert!(!arena.logits().is_null());
    assert!(!arena.moe_worklist().is_null());
    assert!(!arena.moe_worklist_total().is_null());
    assert_eq!(arena.max_batch_tokens(), 128);
    assert_eq!(gpu.alloc_count(), 21);
}

/// The Flash-Next gatepre gate scalars are an arena region sized at load (one
/// f32 per batch token), so prefill never allocates them mid-request; every
/// other model gets no buffer and retains the baseline 21-allocation count.
#[test]
fn gatepre_gate_scalars_are_preallocated_only_for_qwen4() {
    let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
    let plain = MockGpuBackend::new();
    let arena = BufferArena::new(&cfg, 128, 4096, 16, 128, &plain).unwrap();
    assert!(arena.moe_gate_scalars().is_null());
    assert_eq!(arena.sizes().moe_gate_scalars, 0);
    assert_eq!(plain.alloc_count(), 21);

    cfg.model_type = "qwen4_exp".to_string();
    assert!(cfg.is_qwen4_exp());
    let gpu = MockGpuBackend::new();
    let arena = BufferArena::new(&cfg, 256, 4096, 16, 4, &gpu).unwrap();
    assert!(!arena.moe_gate_scalars().is_null());
    assert_eq!(arena.sizes().moe_gate_scalars, 256 * 4);
}

#[test]
fn test_buffer_sizes_scale_with_batch() {
    let cfg = ModelConfig::qwen3_next_80b_nvfp4();
    let s1 = BufferSizes::from_config(&cfg, 1, 4096, 16, 1).unwrap();
    let s128 = BufferSizes::from_config(&cfg, 128, 4096, 16, 128).unwrap();
    assert_eq!(s128.hidden_states, s1.hidden_states * 128);
    assert_eq!(s1.logits, cfg.vocab_size * 2);
    assert_eq!(s128.logits, 128 * cfg.vocab_size * 2);
}

#[test]
fn qwen4_exp_expands_only_the_persistent_residual_buffers() {
    let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
    cfg.hidden_size = 2560;
    cfg.hc_count = 4;
    let sizes = BufferSizes::from_config(&cfg, 3, 4096, 16, 1).unwrap();

    assert_eq!(sizes.hidden_states, 3 * 4 * 2560 * 2);
    assert_eq!(sizes.residual, 3 * 4 * 2560 * 2);
    assert_eq!(sizes.norm_output, 3 * 2560 * 2);
    assert_eq!(sizes.moe_output, 3 * 2560 * 2);
}

/// `ssm_qkvz` is the Q staging buffer for the multi-seq attention QKV routes,
/// at the full gated projection width. On Qwen3.8-27B that width (2*24*256 =
/// 12288) happens to equal `ssm_qkvz_size()`, so the fit was a coincidence of
/// this architecture rather than something the sizing guaranteed. Pin the
/// invariant directly so a future model with a wider gated q_proj_dim fails
/// here instead of overrunning the arena at runtime.
#[test]
fn ssm_qkvz_covers_gated_attention_q_staging() {
    for m in [1usize, 17, 128] {
        let cfg = ModelConfig::qwen3_next_80b_nvfp4();
        let sizes = BufferSizes::from_config(&cfg, m, 4096, 16, 1).unwrap();
        let q_proj_mul = if cfg.attn_gated { 2 } else { 1 };
        let q_staging = m * cfg.num_attention_heads * q_proj_mul * cfg.head_dim * 2;
        assert!(
            sizes.ssm_qkvz >= q_staging,
            "m={m}: ssm_qkvz={} < gated Q staging={q_staging}",
            sizes.ssm_qkvz
        );
        // The same buffer still has to hold the prefill K+V staging.
        let kv_staging = m * 2 * cfg.num_key_value_heads * cfg.head_dim * 2;
        assert!(
            sizes.ssm_qkvz >= kv_staging,
            "m={m}: ssm_qkvz < K+V staging"
        );
    }
}

#[test]
fn merged_flashinfer_ffn_arena_is_exact_shared_and_capped() {
    let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
    cfg.model_type = "qwen3_5".to_string();
    cfg.num_experts = 0;
    cfg.hidden_size = 5_120;
    cfg.intermediate_size = 17_408;
    cfg.tp_world_size = 1;

    assert_eq!(
        super::sizes::qwen38_flashinfer_merged_output_bytes(&cfg, 2_079, true).unwrap(),
        144_764_928
    );
    assert_eq!(
        super::sizes::qwen38_flashinfer_merged_output_bytes(&cfg, 8_192, true).unwrap(),
        570_425_344
    );
    assert_eq!(
        super::sizes::qwen38_flashinfer_merged_output_bytes(&cfg, 32_800, true).unwrap(),
        570_425_344,
        "only admitted M<=8192 needs the shared arena"
    );
    assert_eq!(
        super::sizes::qwen38_flashinfer_merged_output_bytes(&cfg, 8_192, false).unwrap(),
        256
    );

    cfg.intermediate_size -= 1;
    assert_eq!(
        super::sizes::qwen38_flashinfer_merged_output_bytes(&cfg, 8_192, true).unwrap(),
        256,
        "non-Qwen3.8 geometry must not pay the opt-in arena cost"
    );
}

#[test]
fn arena_rejects_oversized_rows_before_any_mock_allocation() {
    let cfg = ModelConfig::qwen3_next_80b_nvfp4();
    for rows in [usize::MAX - 1, usize::MAX] {
        let gpu = MockGpuBackend::new();
        assert!(BufferArena::new(&cfg, rows, 4096, 16, 1, &gpu).is_err());
        assert_eq!(gpu.alloc_count(), 0);
    }
}

#[test]
fn arena_rejects_block_table_and_decode_metadata_overflow() {
    let cfg = ModelConfig::qwen3_next_80b_nvfp4();
    for (seq_len, block_size, batch_size) in [
        (usize::MAX, 1, 1),
        (usize::MAX - 1, 1, 1),
        (usize::MAX, 16, 128),
        (4096, 16, usize::MAX),
    ] {
        let gpu = MockGpuBackend::new();
        assert!(BufferArena::new(&cfg, 1, seq_len, block_size, batch_size, &gpu).is_err());
        assert_eq!(gpu.alloc_count(), 0);
    }
}

#[test]
fn zero_kv_block_size_is_an_error_before_any_mock_allocation() {
    let cfg = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let error = BufferArena::new(&cfg, 1, 4096, 0, 1, &gpu).err().unwrap();
    assert!(error.to_string().contains("KV block size must be positive"));
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn arena_rejects_malformed_derived_widths_before_any_mock_allocation() {
    let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
    cfg.hidden_size = usize::MAX;
    cfg.hc_count = 2;
    let gpu = MockGpuBackend::new();
    assert!(BufferArena::new(&cfg, 1, 4096, 16, 1, &gpu).is_err());
    assert_eq!(gpu.alloc_count(), 0);

    cfg.hidden_size = 1;
    cfg.hc_count = 1;
    cfg.linear_num_key_heads = usize::MAX;
    cfg.linear_key_head_dim = 2;
    assert!(BufferArena::new(&cfg, 1, 4096, 16, 1, &gpu).is_err());
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn arena_rejects_aggregate_overflow_even_when_each_field_fits() {
    // Keep every individual tensor and the 12-byte/row metadata representable,
    // while their combined arena exceeds usize capacity.
    let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
    cfg.model_type = "checked_arena_fixture".to_string();
    cfg.hidden_size = 1;
    cfg.hc_count = 1;
    cfg.hc_lowrank = 0;
    cfg.intermediate_size = 1;
    cfg.vocab_size = 1;
    cfg.num_attention_heads = 0;
    cfg.num_key_value_heads = 0;
    cfg.head_dim = 0;
    cfg.num_experts = 0;
    cfg.num_experts_per_tok = 0;
    cfg.shared_expert_intermediate_size = 0;
    cfg.moe_latent_size = 0;
    cfg.kv_lora_rank = 0;
    cfg.linear_num_key_heads = 0;
    cfg.linear_key_head_dim = 0;
    cfg.linear_num_value_heads = 0;
    cfg.linear_value_head_dim = 0;
    cfg.mamba_num_heads = 0;
    cfg.mamba_head_dim = 0;
    cfg.n_groups = 0;
    cfg.ssm_state_size = 0;
    cfg.mrope_interleaved = false;

    let gpu = MockGpuBackend::new();
    let error = BufferArena::new(&cfg, usize::MAX / 16, 4096, 16, 1, &gpu)
        .err()
        .unwrap();
    assert!(error.to_string().contains("total bytes"), "{error}");
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn arena_total_bytes_rejects_overflow_after_public_field_changes() {
    let cfg = ModelConfig::qwen3_next_80b_nvfp4();
    let mut sizes = BufferSizes::from_config(&cfg, 1, 4096, 16, 1).unwrap();
    sizes.hidden_states = usize::MAX;
    sizes.residual = 1;
    assert!(sizes.total_bytes().is_err());
}

#[test]
fn checked_arena_metadata_preserves_c128_padding_and_mrope_capacity() {
    let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
    cfg.mrope_interleaved = true;
    let m = 129;
    let max_blocks = 4096 / 16 + 1;
    for batch in [1, 32, 33, 64, 65, 96, 97, 128] {
        let layout = DecodeMetaLayout::for_max_batch_size(batch);
        let sizes = BufferSizes::from_config(&cfg, m, 4096, 16, batch).unwrap();
        let routing = 2 * m * cfg.num_experts_per_tok * 4;
        let positions_end = m * 4 * 3;
        let slots_end = ((positions_end + 7) & !7) + m * 8;
        let prefill_end = ((slots_end + 3) & !3) + max_blocks * 4 + 4;
        assert!(sizes.scratch >= routing + prefill_end);
        assert!(sizes.scratch >= 32768 + layout.meta_bytes(max_blocks));
        assert_eq!(
            sizes.logits,
            m.min(32.max(layout.rows() + 1)) * cfg.vocab_size * 2
        );
        assert!(sizes.total_bytes().is_ok());
    }
}
