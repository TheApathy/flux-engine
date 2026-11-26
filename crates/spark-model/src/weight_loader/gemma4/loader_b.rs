// SPDX-License-Identifier: AGPL-3.0-only

//! Gemma-4 weight loader: per-layer helpers + auxiliary methods (embedding,
//! norm, lm_head, mtp, kv_layer_dims).

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

use crate::layers::FfnComponent;
use crate::weight_map::{DenseWeight, MtpWeights, QuantizeCtx, dense};

/// Build the Gemma-4 26B MoE FFN component for one layer (None for dense).
pub(super) fn build_moe_ffn(
    store: &WeightStore,
    lp: &str,
    i: usize,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    variant: crate::weight_map::Nvfp4Variant,
    qctx: QuantizeCtx,
    h: usize,
    absmax_k: spark_runtime::gpu::KernelHandle,
    quantize_k: spark_runtime::gpu::KernelHandle,
    stream: u64,
) -> Result<Option<(FfnComponent, DenseWeight, DenseWeight, DenseWeight)>> {
    if config.num_experts == 0 {
        return Ok(None);
    }
    use crate::weight_map::load_moe_gemma4;
    tracing::info!(
        "L{i}: loading MoE ({} experts, top-{})...",
        config.num_experts,
        config.num_experts_per_tok
    );
    let moe_weights = load_moe_gemma4(store, lp, config.num_experts, gpu, config, variant, qctx)?;
    gpu.synchronize(stream)?;
    let gate_nvfp4 = crate::weight_map::quantize_to_nvfp4(
        &moe_weights.gate,
        config.num_experts,
        h,
        gpu,
        absmax_k,
        quantize_k,
        stream,
    )?;
    let mut moe_layer = crate::layers::MoeLayer::new(
        moe_weights,
        config.num_experts,
        Some(gate_nvfp4),
        gpu,
        config,
    )?;
    moe_layer.set_gelu_activation(gpu)?;
    // Set pre-expert norm: router sees raw input, experts see normed input.
    let pre_expert_norm = dense(store, &format!("{lp}.pre_feedforward_layernorm_2.weight"))?;
    moe_layer.set_pre_expert_norm(pre_expert_norm);
    gpu.synchronize(stream)?;
    tracing::info!("L{i}: MoE layer built (GeGLU activation, pre-expert norm)");

    // Load extra norms for dual FFN
    let pre_moe_norm = dense(store, &format!("{lp}.pre_feedforward_layernorm_2.weight"))?;
    let post_moe_out_norm = dense(store, &format!("{lp}.post_feedforward_layernorm_2.weight"))?;
    let post_dense_ffn_norm = dense(store, &format!("{lp}.post_feedforward_layernorm_1.weight"))?;
    Ok(Some((
        FfnComponent::Moe(moe_layer),
        pre_moe_norm,
        post_moe_out_norm,
        post_dense_ffn_norm,
    )))
}

/// Allocate a BF16 ones-filled buffer of length `head_dim` for v_norm
/// (Gemma-4 v_norm is identity-shaped per HF reference).
pub(super) fn make_v_norm_ones_bf16(gpu: &dyn GpuBackend, head_dim: usize) -> Result<DenseWeight> {
    let bytes = head_dim * 2; // BF16
    let ptr = gpu.alloc(bytes)?;
    // BF16 1.0 = 0x3F80 little-endian → bytes 0x80, 0x3F
    let ones_host: Vec<u8> = std::iter::repeat_with(|| [0x80u8, 0x3Fu8])
        .take(head_dim)
        .flatten()
        .collect();
    gpu.copy_h2d(&ones_host, ptr)?;
    Ok(DenseWeight { weight: ptr })
}

/// Optional BF16 dequant of MLP gate/up/down (Gemma-4 dense, NVFP4 on disk).
pub(super) fn build_bf16_mlp(
    store: &WeightStore,
    lp: &str,
    bf16_mlp: bool,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    h: usize,
) -> Result<Option<(DenseWeight, DenseWeight, DenseWeight)>> {
    let mlp_is_nvfp4 = store.contains(&format!("{lp}.mlp.gate_proj.weight_scale"));
    if !(bf16_mlp && mlp_is_nvfp4) {
        return Ok(None);
    }
    use crate::weight_map::dequant_nvfp4_to_bf16;
    let gate_bf16 = dequant_nvfp4_to_bf16(
        store,
        &format!("{lp}.mlp.gate_proj"),
        config.intermediate_size,
        h,
        gpu,
    )?;
    let up_bf16 = dequant_nvfp4_to_bf16(
        store,
        &format!("{lp}.mlp.up_proj"),
        config.intermediate_size,
        h,
        gpu,
    )?;
    let down_bf16 = dequant_nvfp4_to_bf16(
        store,
        &format!("{lp}.mlp.down_proj"),
        h,
        config.intermediate_size,
        gpu,
    )?;
    Ok(Some((gate_bf16, up_bf16, down_bf16)))
}

pub(super) fn load_embedding_impl(
    store: &WeightStore,
    config: &ModelConfig,
) -> Result<DenseWeight> {
    let prefix = &config.weight_prefix;
    dense(store, &format!("{prefix}.embed_tokens.weight"))
}

pub(super) fn load_final_norm_impl(
    store: &WeightStore,
    config: &ModelConfig,
) -> Result<DenseWeight> {
    // Gemma-4's model-specific rms_norm kernel uses the absolute
    // convention (`out = x * rms * weight`), so norms load as-is.
    let prefix = &config.weight_prefix;
    dense(store, &format!("{prefix}.norm.weight"))
}

pub(super) fn load_lm_head_impl(store: &WeightStore, config: &ModelConfig) -> Result<DenseWeight> {
    // Gemma-4 uses tied embeddings (no separate lm_head tensor).
    // Check for explicit lm_head first, fall back to embed_tokens.
    for pattern in &[
        "lm_head.weight",
        "language_model.lm_head.weight",
        "model.lm_head.weight",
    ] {
        if store.contains(pattern) {
            return dense(store, pattern);
        }
    }
    load_embedding_impl(store, config)
}

pub(super) fn load_mtp_weights_impl(
    _store: &WeightStore,
    _config: &ModelConfig,
    _gpu: &dyn GpuBackend,
) -> Result<Option<MtpWeights>> {
    // Gemma-4 has no MTP head.
    Ok(None)
}

/// Per-layer KV-head TP sharding/replication plan shared by layer loading and
/// KV cache sizing.
///
/// When `global_kv_heads >= tp_size` and divisible, each TP rank owns
/// `global_kv_heads / tp_size` distinct KV heads (`kv_tp_size = tp_size`,
/// `kv_tp_rank = tp_rank`). When `tp_size > global_kv_heads` and divisible,
/// consecutive groups of `tp_size / global_kv_heads` TP ranks replicate one
/// KV head each (`kv_tp_size = global_kv_heads`,
/// `kv_tp_rank = tp_rank / replicate`, `local_kv_heads = 1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct KvTpPlan {
    pub kv_tp_rank: usize,
    pub kv_tp_size: usize,
    pub local_kv_heads: usize,
}

pub(super) fn plan_kv_tp(
    global_kv_heads: usize,
    tp_rank: usize,
    tp_size: usize,
) -> Result<KvTpPlan> {
    let tp_size = tp_size.max(1);
    anyhow::ensure!(
        global_kv_heads > 0,
        "Gemma-4 KV head count must be non-zero"
    );
    anyhow::ensure!(
        tp_rank < tp_size,
        "Gemma-4 tp_rank ({tp_rank}) must be < tp_size ({tp_size})"
    );
    if tp_size == 1 {
        return Ok(KvTpPlan {
            kv_tp_rank: 0,
            kv_tp_size: 1,
            local_kv_heads: global_kv_heads,
        });
    }
    if global_kv_heads.is_multiple_of(tp_size) {
        Ok(KvTpPlan {
            kv_tp_rank: tp_rank,
            kv_tp_size: tp_size,
            local_kv_heads: global_kv_heads / tp_size,
        })
    } else if tp_size > global_kv_heads && tp_size.is_multiple_of(global_kv_heads) {
        let replicate = tp_size / global_kv_heads;
        Ok(KvTpPlan {
            kv_tp_rank: tp_rank / replicate,
            kv_tp_size: global_kv_heads,
            local_kv_heads: 1,
        })
    } else {
        anyhow::bail!(
            "Gemma-4 TP requires layer global_kv_heads ({global_kv_heads}) divisible by \
             tp_size ({tp_size}) or tp_size divisible by global_kv_heads for KV replication"
        );
    }
}

/// Infer the global full-attention KV head count from existing `ModelConfig`
/// fields without requiring a separate parser field:
/// - Gemma-4 26B MoE (`num_experts > 0`, global sliding KV heads = 8): 2
/// - Gemma-4 31B Dense (`num_experts == 0`, global sliding KV heads = 16): 4
fn infer_global_full_kv_heads(config: &ModelConfig) -> usize {
    let global_sliding_nkv = config.num_key_value_heads * config.kv_tp_size();
    if config.num_experts > 0 && global_sliding_nkv == 8 {
        2
    } else if config.num_experts == 0 && global_sliding_nkv == 16 {
        4
    } else if global_sliding_nkv >= 4 && global_sliding_nkv.is_multiple_of(4) {
        global_sliding_nkv / 4
    } else {
        global_sliding_nkv.max(1)
    }
}

pub(super) fn kv_layer_dims_impl(config: &ModelConfig) -> Vec<(usize, usize)> {
    // Gemma-4 uses a 5:1 sliding→full pattern (layer is full when
    // `(global_layer + 1) % 6 == 0`):
    //   26B MoE: sliding global_nkv=8 (hd=256), full global_nkv=2 (hd=512)
    //   31B:     sliding global_nkv=16 (hd=256), full global_nkv=4 (hd=512)
    // Both layer kinds use the exact same KV-head TP sharding/replication plan
    // as `loader_a::load_layers_impl`.
    let tp_size = config.tp_world_size.max(1);
    let tp_rank = config.tp_rank.min(tp_size - 1);
    let global_sliding_nkv = config.num_key_value_heads * config.kv_tp_size();
    let sliding_nkv = plan_kv_tp(global_sliding_nkv, tp_rank, tp_size)
        .map(|plan| plan.local_kv_heads)
        .unwrap_or(config.num_key_value_heads);
    let full_hd = if config.head_dim > 0 {
        config.head_dim
    } else {
        512
    };
    let sliding_hd = 256usize.min(full_hd);
    let global_full_nkv = infer_global_full_kv_heads(config);
    let full_nkv = plan_kv_tp(global_full_nkv, tp_rank, tp_size)
        .map(|plan| plan.local_kv_heads)
        .unwrap_or_else(|_| (global_full_nkv / tp_size).max(1));
    let mut dims = Vec::with_capacity(config.num_hidden_layers);
    for i in 0..config.num_hidden_layers {
        let global_layer = config.pp_layer_offset + i;
        if (global_layer + 1).is_multiple_of(6) {
            dims.push((full_nkv, full_hd));
        } else {
            dims.push((sliding_nkv, sliding_hd));
        }
    }
    dims
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gemma4_config(num_experts: usize, global_sliding_nkv: usize, tp_size: usize) -> ModelConfig {
        let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
        cfg.model_type = "gemma4".to_string();
        cfg.num_hidden_layers = 6;
        cfg.num_experts = num_experts;
        cfg.head_dim = 512;
        cfg.tp_world_size = tp_size;
        cfg.tp_rank = 0;
        if tp_size <= 1 {
            cfg.tp_kv_replicate = 1;
            cfg.num_key_value_heads = global_sliding_nkv;
        } else if global_sliding_nkv.is_multiple_of(tp_size) {
            cfg.tp_kv_replicate = 1;
            cfg.num_key_value_heads = global_sliding_nkv / tp_size;
        } else {
            cfg.tp_kv_replicate = tp_size / global_sliding_nkv;
            cfg.num_key_value_heads = 1;
        }
        cfg
    }

    #[test]
    fn kv_layer_dims_matches_26b_moe_and_31b_dense_across_tp_degrees() {
        // 26B MoE: global sliding=8, full=2
        for (tp, exp_sliding, exp_full) in [(1, 8, 2), (2, 4, 1), (4, 2, 1), (8, 1, 1), (16, 1, 1)]
        {
            let cfg = gemma4_config(128, 8, tp);
            let dims = kv_layer_dims_impl(&cfg);
            assert_eq!(dims.len(), 6);
            assert_eq!(dims[0], (exp_sliding, 256), "26B sliding tp={tp}");
            assert_eq!(dims[5], (exp_full, 512), "26B full tp={tp}");
        }

        // 31B Dense: global sliding=16, full=4
        for (tp, exp_sliding, exp_full) in [
            (1, 16, 4),
            (2, 8, 2),
            (4, 4, 1),
            (8, 2, 1),
            (16, 1, 1),
            (32, 1, 1),
        ] {
            let cfg = gemma4_config(0, 16, tp);
            let dims = kv_layer_dims_impl(&cfg);
            assert_eq!(dims.len(), 6);
            assert_eq!(dims[0], (exp_sliding, 256), "31B sliding tp={tp}");
            assert_eq!(dims[5], (exp_full, 512), "31B full tp={tp}");
        }
    }

    #[test]
    fn plan_kv_tp_replicates_heads_and_rejects_non_divisible_counts() {
        // 26B full layer (2 KV heads) at TP=4: ranks 0,1 share head 0; ranks 2,3 share head 1.
        assert_eq!(
            plan_kv_tp(2, 0, 4).unwrap(),
            KvTpPlan {
                kv_tp_rank: 0,
                kv_tp_size: 2,
                local_kv_heads: 1,
            }
        );
        assert_eq!(
            plan_kv_tp(2, 1, 4).unwrap(),
            KvTpPlan {
                kv_tp_rank: 0,
                kv_tp_size: 2,
                local_kv_heads: 1,
            }
        );
        assert_eq!(
            plan_kv_tp(2, 2, 4).unwrap(),
            KvTpPlan {
                kv_tp_rank: 1,
                kv_tp_size: 2,
                local_kv_heads: 1,
            }
        );
        assert_eq!(
            plan_kv_tp(2, 3, 4).unwrap(),
            KvTpPlan {
                kv_tp_rank: 1,
                kv_tp_size: 2,
                local_kv_heads: 1,
            }
        );
        assert!(plan_kv_tp(3, 0, 2).is_err());
        assert!(plan_kv_tp(0, 0, 1).is_err());
    }
}
