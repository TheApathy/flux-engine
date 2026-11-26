// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use super::{ModelWeightLoader, QuantFormat};
use crate::layer::TransformerLayer;
use crate::layers::{FfnComponent, MoeLayer, Qwen3AttentionLayer, Qwen3SsmLayer};
use crate::tp_shard::{
    TpAttentionDims, TpShardKind, load_qkvo_tp, shard_dense_bf16, shard_fp8_block_scaled,
    shard_quantized_nvfp4,
};
use crate::weight_map::{
    AttentionWeights, DenseWeight, Fp8ExpertWeight, MtpWeights, Nvfp4Variant, QuantizeCtx,
    QuantizedWeight, dense, detect_nvfp4_variant, load_attention,
    load_fp8_block_scaled_as_fp8weight, load_kv_scales, load_moe, load_moe_qwen35_fp8_experts,
    load_moe_skip_experts, load_mtp, load_ssm, quantize_to_nvfp4,
};

pub struct Qwen3WeightLoader;

/// Release a replaced tensor buffer after TP sharding: tombstone and free via
/// [`WeightStore::release_consumed_alias`] if `ptr` aliases a store entry, or
/// free directly via `gpu.free` if it was freshly allocated (e.g. a dequant or
/// runtime-quantized intermediate).
fn release_ptr_or_store_alias(
    store: &WeightStore,
    candidate_keys: &[String],
    ptr: DevicePtr,
    gpu: &dyn GpuBackend,
) -> Result<()> {
    if ptr.is_null() {
        return Ok(());
    }
    for key in candidate_keys {
        if let Ok(tensor) = store.get(key)
            && tensor.ptr == ptr
        {
            let dtype = tensor.dtype;
            store.release_consumed_alias(key, ptr, dtype, gpu)?;
            return Ok(());
        }
    }
    gpu.free(ptr)
}

impl ModelWeightLoader for Qwen3WeightLoader {
    fn supports_tp(&self) -> bool {
        // Qwen3-Next FullAttention layers (gated) are TP-sharded across
        // both quant paths (FP8 native, BF16 → NVFP4). LinearAttention
        // (GDN SSM) layers run full-replica per rank — same trade-off as
        // qwen35.rs: SSM weight memory not recovered, but functionally
        // correct and the bulk of compute (attention + MoE) is sharded.
        true
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        let layer_types = if config.layer_types.is_empty() {
            (0..config.num_hidden_layers)
                .map(|i| config.layer_type(i))
                .collect::<Vec<_>>()
        } else {
            config.layer_types.clone()
        };

        let mut layers: Vec<Box<dyn TransformerLayer>> =
            Vec::with_capacity(config.num_hidden_layers);
        let mut attn_idx = 0usize;

        // Kernels + stream for BF16→NVFP4 runtime quantization of dense weights
        let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
        let quantize_k = gpu
            .kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4_weight_rne")
            .or_else(|_| gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4"))?;
        let stream = gpu.default_stream();
        let qctx = QuantizeCtx {
            absmax_k,
            quantize_k,
            stream,
        };

        // Detect weight format variant (Standard NVFP4, CompressedTensors, or FP8 block-scaled).
        let variant = detect_nvfp4_variant(store, config);
        let quant_format = if variant == Nvfp4Variant::Fp8Dequanted {
            QuantFormat::Fp8
        } else {
            QuantFormat::Nvfp4
        };
        let native_fp8 = quant_format == QuantFormat::Fp8;
        tracing::info!(
            "Qwen3 weight variant: {:?}, native_fp8: {}",
            variant,
            native_fp8
        );

        let h = config.hidden_size;

        // Estimate MoE transpose memory — skip if GPU can't hold all layers' transposed copies.
        let skip_moe_transpose = {
            let inter = config.moe_intermediate_size;
            let h = config.hidden_size;
            let group_size = 16usize;
            let gu_bytes = inter * h / 2 + inter * h / group_size;
            let d_bytes = h * inter / 2 + h * inter / group_size;
            let per_layer = config.num_experts * (2 * gu_bytes + d_bytes);
            let total = per_layer * config.num_hidden_layers;
            let available = gpu.free_memory().unwrap_or(0);
            let headroom = 2 * 1024 * 1024 * 1024;
            let skip = total > available.saturating_sub(headroom);
            if skip {
                tracing::warn!(
                    "Skipping MoE weight transposition ({:.1} GB needed, {:.1} GB available). \
                     Prefill will use fallback grouped GEMM.",
                    total as f64 / (1024.0 * 1024.0 * 1024.0),
                    available as f64 / (1024.0 * 1024.0 * 1024.0),
                );
            }
            skip
        };

        for (i, lt) in layer_types.iter().enumerate() {
            let lp = config.layer_prefix(i);
            let input_norm = dense(store, &format!("{lp}.input_layernorm.weight"))?;
            let post_attn_norm = dense(store, &format!("{lp}.post_attention_layernorm.weight"))?;

            // ── MoE weights ──
            let moe_weights = if native_fp8 {
                load_moe_skip_experts(store, &lp, config.num_experts, gpu, config, variant, qctx)?
            } else {
                load_moe(store, &lp, config.num_experts, gpu, config, variant, qctx)?
            };
            let gate_nvfp4 = quantize_to_nvfp4(
                &moe_weights.gate,
                config.num_experts,
                h,
                gpu,
                absmax_k,
                quantize_k,
                stream,
            )?;
            let mut moe_layer = MoeLayer::new(
                moe_weights,
                config.num_experts,
                Some(gate_nvfp4),
                gpu,
                config,
            )?;
            if !native_fp8 && !skip_moe_transpose {
                moe_layer.transpose_for_prefill(gpu, config)?;
            }
            if !native_fp8 {
                moe_layer.predequant_for_prefill(gpu, config, stream)?;
            }

            // Native FP8 MoE: load FP8 expert weights for fused batch dispatch.
            // Because `load_moe_skip_experts` left routed experts NULL above,
            // any failure here leaves no fallback and must propagate.
            if native_fp8 {
                let fp8_experts =
                    load_moe_qwen35_fp8_experts(store, &lp, config.num_experts, gpu, config)
                        .with_context(|| {
                            format!("Layer {i}: failed to load native FP8 MoE experts")
                        })?;
                let sp = format!("{lp}.mlp.shared_expert");
                let shared_fp8 = Fp8ExpertWeight {
                    gate_proj: load_fp8_block_scaled_as_fp8weight(
                        store,
                        &format!("{sp}.gate_proj"),
                        gpu,
                    )
                    .with_context(|| {
                        format!("Layer {i}: failed to load shared expert FP8 gate_proj")
                    })?,
                    up_proj: load_fp8_block_scaled_as_fp8weight(
                        store,
                        &format!("{sp}.up_proj"),
                        gpu,
                    )
                    .with_context(|| {
                        format!("Layer {i}: failed to load shared expert FP8 up_proj")
                    })?,
                    down_proj: load_fp8_block_scaled_as_fp8weight(
                        store,
                        &format!("{sp}.down_proj"),
                        gpu,
                    )
                    .with_context(|| {
                        format!("Layer {i}: failed to load shared expert FP8 down_proj")
                    })?,
                };
                moe_layer
                    .set_fp8_experts(&fp8_experts, shared_fp8, gpu)
                    .with_context(|| format!("Layer {i}: FP8 expert tables failed"))?;
                tracing::info!("Layer {i}: MoE experts loaded as native FP8");
            }

            let ffn = FfnComponent::Moe(moe_layer);

            match lt {
                // ── Native FP8 Attention ──
                LayerType::FullAttention if native_fp8 => {
                    let p = format!("{lp}.self_attn");
                    let tp_rank = config.tp_rank;
                    let tp_size = config.tp_world_size.max(1);
                    let kv_tp_rank = config.kv_tp_rank();
                    let kv_tp_size = config.kv_tp_size();
                    let block_size = 128usize;
                    let load_fp8 = |name: &str,
                                    _full_n: usize,
                                    _full_k: usize,
                                    kind: TpShardKind|
                     -> Result<crate::weight_map::Fp8Weight> {
                        let prefix = format!("{p}.{name}");
                        let src = load_fp8_block_scaled_as_fp8weight(store, &prefix, gpu)?;
                        let (eff_rank, eff_size) = if matches!(name, "k_proj" | "v_proj") {
                            (kv_tp_rank, kv_tp_size)
                        } else {
                            (tp_rank, tp_size)
                        };
                        if eff_size == 1 {
                            return Ok(src);
                        }
                        let weight_key = format!("{prefix}.weight");
                        let scale_key = format!("{prefix}.weight_scale_inv");
                        let weight_dtype = store.get(&weight_key)?.dtype;
                        let scale_dtype = store.get(&scale_key)?.dtype;
                        let sharded = shard_fp8_block_scaled(
                            &src, kind, eff_rank, eff_size, block_size, gpu,
                        )?;
                        store.release_consumed_alias(&weight_key, src.weight, weight_dtype, gpu)?;
                        store.release_consumed_alias(
                            &scale_key,
                            src.row_scale,
                            scale_dtype,
                            gpu,
                        )?;
                        Ok(sharded)
                    };
                    let [q_fp8, k_fp8, v_fp8, o_fp8] = load_qkvo_tp(config, load_fp8)?;

                    let (k_scale, v_scale) = load_kv_scales(store, &p, gpu);
                    let dummy = DenseWeight {
                        weight: DevicePtr::NULL,
                    };
                    let attn = AttentionWeights {
                        q_proj: dummy,
                        k_proj: dummy,
                        v_proj: dummy,
                        o_proj: QuantizedWeight::null(),
                        q_norm: dense(store, &format!("{p}.q_norm.weight"))?,
                        k_norm: dense(store, &format!("{p}.k_norm.weight"))?,
                        q_norm_full: None,
                        k_norm_full: None,
                        k_scale,
                        v_scale,
                    };

                    let layer_kv_dtype = layer_kv_dtypes[attn_idx];
                    let mut layer = Qwen3AttentionLayer::new(
                        input_norm,
                        attn,
                        post_attn_norm,
                        ffn,
                        attn_idx,
                        None,
                        None,
                        None, // No NVFP4 — w8a16 handles everything
                        gpu,
                        layer_kv_dtype,
                        config.fp8_kv_calibration_tokens,
                        config,
                    )?;
                    layer.set_fp8_weights(Some(q_fp8), Some(k_fp8), Some(v_fp8), Some(o_fp8));
                    if let Err(e) = layer.transpose_fp8_for_prefill(gpu, stream) {
                        tracing::warn!("Layer {i}: FP8 transpose failed: {e}");
                    }
                    layers.push(Box::new(layer));
                    attn_idx += 1;
                }
                // ── NVFP4 Attention (original path) ──
                LayerType::FullAttention => {
                    let mut attn = load_attention(store, &lp, gpu, variant, qctx, config)?;
                    let p = format!("{lp}.self_attn");
                    let dims = TpAttentionDims::from_config(config);
                    // TP shard Q/K/V BF16 projections BEFORE quantization (using
                    // kv_tp_rank/kv_tp_size for K/V) and shard the pre-quantized
                    // NVFP4 o_proj via shard_quantized_nvfp4. After sharding,
                    // dims are TP-LOCAL and config head counts (already divided
                    // in main.rs) match the post-shard quantize calls below.
                    if dims.tp_size > 1 {
                        let (qp, _, _) = shard_dense_bf16(
                            attn.q_proj.weight,
                            dims.full_q_n,
                            dims.h,
                            TpShardKind::ColumnParallel,
                            dims.tp_rank,
                            dims.tp_size,
                            gpu,
                        )?;
                        if qp != attn.q_proj.weight {
                            release_ptr_or_store_alias(
                                store,
                                &[format!("{p}.q_proj.weight")],
                                attn.q_proj.weight,
                                gpu,
                            )?;
                        }
                        attn.q_proj.weight = qp;
                        let (kp, _, _) = shard_dense_bf16(
                            attn.k_proj.weight,
                            dims.full_kv_n,
                            dims.h,
                            TpShardKind::ColumnParallel,
                            dims.kv_tp_rank,
                            dims.kv_tp_size,
                            gpu,
                        )?;
                        if kp != attn.k_proj.weight {
                            release_ptr_or_store_alias(
                                store,
                                &[format!("{p}.k_proj.weight")],
                                attn.k_proj.weight,
                                gpu,
                            )?;
                        }
                        attn.k_proj.weight = kp;
                        let (vp, _, _) = shard_dense_bf16(
                            attn.v_proj.weight,
                            dims.full_kv_n,
                            dims.h,
                            TpShardKind::ColumnParallel,
                            dims.kv_tp_rank,
                            dims.kv_tp_size,
                            gpu,
                        )?;
                        if vp != attn.v_proj.weight {
                            release_ptr_or_store_alias(
                                store,
                                &[format!("{p}.v_proj.weight")],
                                attn.v_proj.weight,
                                gpu,
                            )?;
                        }
                        attn.v_proj.weight = vp;
                        let old_o = attn.o_proj;
                        let sharded_o = shard_quantized_nvfp4(
                            &old_o,
                            dims.h,
                            dims.full_o_in,
                            TpShardKind::RowParallel,
                            dims.tp_rank,
                            dims.tp_size,
                            16,
                            gpu,
                        )?;
                        if sharded_o.weight != old_o.weight {
                            release_ptr_or_store_alias(
                                store,
                                &[
                                    format!("{p}.o_proj.weight_packed"),
                                    format!("{p}.o_proj.weight"),
                                ],
                                old_o.weight,
                                gpu,
                            )?;
                        }
                        if sharded_o.weight_scale != old_o.weight_scale {
                            release_ptr_or_store_alias(
                                store,
                                &[format!("{p}.o_proj.weight_scale")],
                                old_o.weight_scale,
                                gpu,
                            )?;
                        }
                        attn.o_proj = sharded_o;
                    }
                    let q_nvfp4 = quantize_to_nvfp4(
                        &attn.q_proj,
                        config.num_attention_heads * config.head_dim * 2,
                        h,
                        gpu,
                        absmax_k,
                        quantize_k,
                        stream,
                    )?;
                    let k_nvfp4 = quantize_to_nvfp4(
                        &attn.k_proj,
                        config.num_key_value_heads * config.head_dim,
                        h,
                        gpu,
                        absmax_k,
                        quantize_k,
                        stream,
                    )?;
                    let v_nvfp4 = quantize_to_nvfp4(
                        &attn.v_proj,
                        config.num_key_value_heads * config.head_dim,
                        h,
                        gpu,
                        absmax_k,
                        quantize_k,
                        stream,
                    )?;
                    let layer_kv_dtype = layer_kv_dtypes[attn_idx];
                    let mut layer = Qwen3AttentionLayer::new(
                        input_norm,
                        attn,
                        post_attn_norm,
                        ffn,
                        attn_idx,
                        Some(q_nvfp4),
                        Some(k_nvfp4),
                        Some(v_nvfp4),
                        gpu,
                        layer_kv_dtype,
                        config.fp8_kv_calibration_tokens,
                        config,
                    )?;
                    let qt = q_nvfp4.transpose_for_gemm(
                        gpu,
                        config.num_attention_heads * config.head_dim * 2,
                        h,
                    )?;
                    let kt = k_nvfp4.transpose_for_gemm(
                        gpu,
                        config.num_key_value_heads * config.head_dim,
                        h,
                    )?;
                    let vt = v_nvfp4.transpose_for_gemm(
                        gpu,
                        config.num_key_value_heads * config.head_dim,
                        h,
                    )?;
                    let ot = layer.attn.o_proj.transpose_for_gemm(
                        gpu,
                        h,
                        config.num_attention_heads * config.head_dim,
                    )?;
                    layer.set_prefill_weights(Some(qt), Some(kt), Some(vt), Some(ot));
                    layer.predequant_for_prefill(gpu, config, stream)?;
                    layers.push(Box::new(layer));
                    attn_idx += 1;
                }
                // ── SSM (FP8→BF16→NVFP4 conversion, same path for native_fp8 and non-native) ──
                // Native FP8 SSM decode is disabled upstream (Qwen35 `&& false`) due to
                // block-scale → per-row-scale precision loss. Instead, dequant FP8→BF16
                // then quantize BF16→NVFP4. Only qkvz + out_proj need conversion (tiny).
                //
                LayerType::LinearAttention => {
                    let ssm = load_ssm(store, &lp, gpu, variant, qctx, config)?;
                    let qkvz_nvfp4 = quantize_to_nvfp4(
                        &ssm.in_proj_qkvz,
                        config.ssm_qkvz_size(),
                        h,
                        gpu,
                        absmax_k,
                        quantize_k,
                        stream,
                    )?;
                    layers.push(Box::new(Qwen3SsmLayer::new(
                        input_norm,
                        ssm,
                        post_attn_norm,
                        ffn,
                        Some(qkvz_nvfp4),
                        config,
                        gpu,
                    )?));
                }
                LayerType::Moe => unreachable!("Qwen3 has no standalone MoE layers"),
            }

            if (i + 1) % 12 == 0 {
                tracing::info!("Loaded layers 0..{}", i + 1);
            }
        }

        tracing::info!(
            "Weight loader: {} layers ({} attention, {} SSM)",
            layers.len(),
            attn_idx,
            layers.len() - attn_idx,
        );

        Ok(layers)
    }

    fn load_embedding(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, "model.embed_tokens.weight")
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, "model.norm.weight")
    }

    fn load_lm_head(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        if store.contains("lm_head.weight") {
            dense(store, "lm_head.weight")
        } else {
            self.load_embedding(store, config, _gpu)
        }
    }

    fn load_mtp_weights(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        if !store.contains("mtp.fc.weight") {
            tracing::info!("No MTP weights found — speculative decoding disabled");
            return Ok(None);
        }
        let variant = detect_nvfp4_variant(store, config);
        tracing::info!("Loading MTP weights (variant={:?})...", variant);
        let mtp = load_mtp(store, config.num_experts, gpu, variant)?;
        tracing::info!(
            "MTP weights loaded: fc=[{},{}], {} experts, attn layer",
            config.hidden_size,
            config.hidden_size * 2,
            mtp.experts.len(),
        );
        Ok(Some(mtp))
    }
}
