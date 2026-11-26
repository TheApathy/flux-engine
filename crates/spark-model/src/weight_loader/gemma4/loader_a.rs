// SPDX-License-Identifier: AGPL-3.0-only

//! Gemma-4 weight loader: per-layer construction (`load_layers`).

use anyhow::{Context, Result};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::ffn_prefill::{install_ffn_prefill_transpose, plan_ffn_prefill_transpose};
use super::loader_b::{build_bf16_mlp, build_moe_ffn, plan_kv_tp};
use crate::layer::TransformerLayer;
use crate::layers::dense_ffn::DenseFfnWeights;
use crate::layers::{DenseFfnLayer, FfnActivation, FfnComponent, Qwen3AttentionLayer};
use crate::tp_shard::{TpShardKind, shard_dense_bf16};
use crate::weight_map::{
    AttentionWeights, QuantizeCtx, dense, dense_auto, dequant_nvfp4_to_bf16, detect_nvfp4_variant,
    load_kv_scales, quantize_to_nvfp4, quantized_any,
};

/// Read the global output dimension (`shape[0]`) of a projection tensor from
/// either BF16/ModelOpt `.weight` or compressed-tensors `.weight_packed`.
fn proj_out_dim(store: &WeightStore, proj_prefix: &str) -> Result<usize> {
    let wt = store
        .get(&format!("{proj_prefix}.weight"))
        .or_else(|_| store.get(&format!("{proj_prefix}.weight_packed")))
        .with_context(|| {
            format!("missing both '{proj_prefix}.weight' and '{proj_prefix}.weight_packed'")
        })?;
    wt.shape
        .first()
        .copied()
        .with_context(|| format!("projection '{proj_prefix}' has empty shape"))
}

/// Release a dense attention buffer replaced by a TP shard: tombstone+free via
/// `WeightStore` when `old_ptr` aliases a store-owned tensor, or `gpu.free`
/// directly when it is a freshly allocated dequant buffer.
fn release_replaced_dense(
    store: &WeightStore,
    key: &str,
    old_ptr: DevicePtr,
    gpu: &dyn GpuBackend,
) -> Result<()> {
    if let Ok(wt) = store.get(key)
        && wt.ptr == old_ptr
    {
        store.release_consumed_alias(key, old_ptr, wt.dtype, gpu)?;
        return Ok(());
    }
    gpu.free(old_ptr)?;
    Ok(())
}

pub(super) fn load_layers_impl(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    layer_kv_dtypes: &[KvCacheDtype],
) -> Result<Vec<Box<dyn TransformerLayer>>> {
    let mut layers: Vec<Box<dyn TransformerLayer>> = Vec::with_capacity(config.num_hidden_layers);

    let variant = detect_nvfp4_variant(store, config);
    tracing::info!("Gemma-4 NVFP4 variant: {:?}", variant);

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
    let h = config.hidden_size;
    let tp_rank = config.tp_rank;
    let tp_size = config.tp_world_size.max(1);
    let global_q_heads = config.num_attention_heads * tp_size;

    // Attention quantization choice for Gemma-4. Keeping q/k/v/o_proj at BF16
    // avoids per-tensor absmax precision loss from Gemma-4's high-magnitude
    // calibration channels. Override via `ATLAS_GEMMA4_BF16_ATTN=0/1`.
    let bf16_attn_default = true;
    let bf16_attn = match std::env::var("ATLAS_GEMMA4_BF16_ATTN").ok().as_deref() {
        Some("0") => false,
        Some("1") => true,
        _ => bf16_attn_default,
    };
    // BF16 MLP (gate/up/down) — disabled by default (`ATLAS_GEMMA4_BF16_MLP=1`
    // opts in). Parsed once before `plan_ffn_prefill_transpose` so transpose
    // planning skips memory probing when BF16 MLP supersedes NVFP4 prefill.
    let bf16_mlp_default = false;
    let bf16_mlp = match std::env::var("ATLAS_GEMMA4_BF16_MLP").ok().as_deref() {
        Some("0") => false,
        Some("1") => true,
        _ => bf16_mlp_default,
    };
    // Decided once, before any layer allocates — see `ffn_prefill`.
    let ffn_prefill_transpose = plan_ffn_prefill_transpose(config, gpu, bf16_mlp)?;

    for i in 0..config.num_hidden_layers {
        let lp = config.layer_prefix(i);

        // ── Layer norms ──
        // Gemma-4 has 4 norms: input, post_attn, pre_ffn, post_ffn.
        // The existing Qwen3AttentionLayer uses:
        //   input_norm  = pre-attention norm
        //   post_attn_norm = pre-FFN norm (applied before FFN input)
        // Map: input_layernorm → input_norm, pre_feedforward_layernorm → post_attn_norm.
        // Gemma-4's model-specific rms_norm kernel uses the absolute
        // convention `out = x * rms * weight` (see
        // `kernels/gb10/gemma-4-*/nvfp4/rms_norm.cu`), so norm weights
        // are loaded as-is — no convention shift needed.
        let input_norm = dense(store, &format!("{lp}.input_layernorm.weight"))?;
        let pre_ffn_norm = dense(store, &format!("{lp}.pre_feedforward_layernorm.weight"))?;

        // Post-sub-layer norms: applied to attn/FFN output before residual add.
        let post_attn_out_norm = dense(store, &format!("{lp}.post_attention_layernorm.weight"))?;
        let post_ffn_norm_w = dense(store, &format!("{lp}.post_feedforward_layernorm.weight"))?;

        // Read layer_scalar ([1] BF16 or FP32 tensor).
        // Applied at the END of the layer forward pass to the ENTIRE hidden_states
        // (including residual), NOT fused into post-norm weights.
        // Reference: hidden_states = (residual + post_attn_norm(attn) + post_ffn_norm(ffn)) * layer_scalar
        let layer_scalar_key = format!("{lp}.layer_scalar");
        let layer_scalar_val = if store.contains(&layer_scalar_key) {
            let wt = store.get(&layer_scalar_key)?;
            anyhow::ensure!(
                wt.num_elements() == 1,
                "L{i}: expected scalar (1 element) for '{layer_scalar_key}', got shape {:?}",
                wt.shape
            );
            let scalar_f32 = match wt.dtype {
                WeightDtype::BF16 => {
                    let mut scalar_buf = [0u8; 2];
                    gpu.copy_d2h(wt.ptr, &mut scalar_buf)?;
                    let scalar_bf16 = u16::from_le_bytes(scalar_buf);
                    f32::from_bits(u32::from(scalar_bf16) << 16)
                }
                WeightDtype::FP32 => {
                    let mut scalar_buf = [0u8; 4];
                    gpu.copy_d2h(wt.ptr, &mut scalar_buf)?;
                    f32::from_le_bytes(scalar_buf)
                }
                other => {
                    anyhow::bail!(
                        "L{i}: unsupported dtype {other:?} for '{layer_scalar_key}' \
                         (expected BF16 or FP32)"
                    );
                }
            };
            tracing::info!("L{i}: layer_scalar = {scalar_f32:.6}");
            Some(scalar_f32)
        } else {
            None
        };

        // ── Detect layer type from global on-disk weight shapes ──
        let p = format!("{lp}.self_attn");
        let sliding_head_dim: usize = 256;
        let q_out_dim = proj_out_dim(store, &format!("{p}.q_proj"))?;
        let kv_out_dim = proj_out_dim(store, &format!("{p}.k_proj"))?;
        let is_full_attn = q_out_dim != global_q_heads * sliding_head_dim;

        // ── Attention weights ──
        // 31B: BF16 on disk (dense_auto). 26B MoE: NVFP4 on disk (dequant to BF16).
        let attn_is_nvfp4 = store.contains(&format!("{p}.q_proj.weight_scale"));
        let q_key = format!("{p}.q_proj.weight");
        let k_key = format!("{p}.k_proj.weight");
        let v_key = format!("{p}.v_proj.weight");
        let o_key = format!("{p}.o_proj.weight");
        let q_dense = if attn_is_nvfp4 {
            dequant_nvfp4_to_bf16(store, &format!("{p}.q_proj"), q_out_dim, h, gpu)?
        } else {
            dense_auto(store, &q_key, gpu)?
        };
        let k_dense = if attn_is_nvfp4 {
            dequant_nvfp4_to_bf16(store, &format!("{p}.k_proj"), kv_out_dim, h, gpu)?
        } else {
            dense_auto(store, &k_key, gpu)?
        };
        let has_v_proj = store.contains(&v_key)
            || store.contains(&format!("{p}.v_proj.weight_packed"))
            || store.contains(&format!("{p}.v_proj.weight_scale"));
        let v_dense = if has_v_proj {
            if attn_is_nvfp4 {
                dequant_nvfp4_to_bf16(store, &format!("{p}.v_proj"), kv_out_dim, h, gpu)?
            } else {
                dense_auto(store, &v_key, gpu)?
            }
        } else {
            k_dense // K=V: alias K as V
        };
        let o_dense = if attn_is_nvfp4 {
            dequant_nvfp4_to_bf16(store, &format!("{p}.o_proj"), h, q_out_dim, gpu)?
        } else {
            dense_auto(store, &o_key, gpu)?
        };
        let v_aliases_k = !has_v_proj;
        if is_full_attn {
            tracing::info!(
                "L{i}: full attention (Q_dim={q_out_dim}, K_dim={kv_out_dim}, K=V={v_aliases_k})"
            );
        } else {
            tracing::debug!("L{i}: sliding attention (Q_dim={q_out_dim}, K_dim={kv_out_dim})");
        }

        // ── TP shard q/k/v/o BF16 BEFORE quantize ──
        // Per-layer dim overrides: q_out_dim and kv_out_dim are read from
        // the on-disk shape, varying per layer (sliding=256 head_dim,
        // full=512 head_dim). Under TP, Q and O shard across `tp_size`; K and
        // V follow `plan_kv_tp`, replicating a single KV head per rank group
        // when `tp_size > global_kv_heads` (e.g. TP=4 on 26B full layers with
        // 2 KV heads, or TP=8 on 31B full layers with 4 KV heads).
        //
        // K=V aliasing: when v_dense aliases k_dense (full-attention layers),
        // sharding k_dense in place makes the alias stale. We re-alias
        // after the slice so both still point at the shared sharded
        // weight.
        anyhow::ensure!(
            global_q_heads > 0 && q_out_dim.is_multiple_of(global_q_heads),
            "Gemma-4 L{i}: q_proj out_dim ({q_out_dim}) is not divisible by \
             global_q_heads ({global_q_heads})"
        );
        let actual_head_dim = q_out_dim / global_q_heads; // 256 or 512
        anyhow::ensure!(
            actual_head_dim > 0 && kv_out_dim.is_multiple_of(actual_head_dim),
            "Gemma-4 L{i}: k_proj out_dim ({kv_out_dim}) is not divisible by \
             actual_head_dim ({actual_head_dim})"
        );
        let global_kv_heads = kv_out_dim / actual_head_dim;
        let kv_plan = plan_kv_tp(global_kv_heads, tp_rank, tp_size)?;
        let (mut q_dense, mut k_dense, mut v_dense, mut o_dense) =
            (q_dense, k_dense, v_dense, o_dense);
        let local_q_out = q_out_dim / tp_size;
        let local_kv_out = kv_plan.local_kv_heads * actual_head_dim;
        if tp_size > 1 {
            let (qp, _, _) = shard_dense_bf16(
                q_dense.weight,
                q_out_dim,
                h,
                TpShardKind::ColumnParallel,
                tp_rank,
                tp_size,
                gpu,
            )?;
            if qp != q_dense.weight {
                release_replaced_dense(store, &q_key, q_dense.weight, gpu)?;
            }
            q_dense.weight = qp;
            if kv_plan.kv_tp_size > 1 {
                let (kp, _, _) = shard_dense_bf16(
                    k_dense.weight,
                    kv_out_dim,
                    h,
                    TpShardKind::ColumnParallel,
                    kv_plan.kv_tp_rank,
                    kv_plan.kv_tp_size,
                    gpu,
                )?;
                if kp != k_dense.weight {
                    release_replaced_dense(store, &k_key, k_dense.weight, gpu)?;
                }
                k_dense.weight = kp;
            }
            if v_aliases_k {
                // Re-alias V to the freshly sharded (or replicated) K.
                v_dense.weight = k_dense.weight;
            } else if kv_plan.kv_tp_size > 1 {
                let (vp, _, _) = shard_dense_bf16(
                    v_dense.weight,
                    kv_out_dim,
                    h,
                    TpShardKind::ColumnParallel,
                    kv_plan.kv_tp_rank,
                    kv_plan.kv_tp_size,
                    gpu,
                )?;
                if vp != v_dense.weight {
                    release_replaced_dense(store, &v_key, v_dense.weight, gpu)?;
                }
                v_dense.weight = vp;
            }
            let (op, _, _) = shard_dense_bf16(
                o_dense.weight,
                h,
                q_out_dim,
                TpShardKind::RowParallel,
                tp_rank,
                tp_size,
                gpu,
            )?;
            if op != o_dense.weight {
                release_replaced_dense(store, &o_key, o_dense.weight, gpu)?;
            }
            o_dense.weight = op;
        }
        let q_out_dim = local_q_out;
        let kv_out_dim = local_kv_out;
        let (q_nvfp4_opt, k_nvfp4_opt, v_nvfp4_opt) = if bf16_attn {
            tracing::info!(
                "L{i}: BF16 attention (dense_gemv path) — skip NVFP4 q/k/v quant for Gemma-4 precision"
            );
            (None, None, None)
        } else {
            let q = quantize_to_nvfp4(&q_dense, q_out_dim, h, gpu, absmax_k, quantize_k, stream)?;
            let k = quantize_to_nvfp4(&k_dense, kv_out_dim, h, gpu, absmax_k, quantize_k, stream)?;
            let v = quantize_to_nvfp4(&v_dense, kv_out_dim, h, gpu, absmax_k, quantize_k, stream)?;
            (Some(q), Some(k), Some(v))
        };
        // Honor Nvidia ModelOpt's official ignore list for Gemma-4:
        // ALL self_attn projections (q/k/v/o) stay BF16. Atlas
        // previously quantized o_proj unconditionally, losing ~7 bits
        // per layer to per-tensor absmax across 60 layers — a major
        // contributor to the creative-collapse drift. When bf16_attn
        // is on, we still emit a placeholder NVFP4 o_proj (the
        // AttentionWeights struct currently requires it) but the
        // layer dispatch below installs `o_dense_bf16` which the
        // decode/prefill paths prefer over the NVFP4 path.
        let o_nvfp4 = quantize_to_nvfp4(&o_dense, h, q_out_dim, gpu, absmax_k, quantize_k, stream)?;

        let (k_scale, v_scale) = load_kv_scales(store, &p, gpu);

        let attn = AttentionWeights {
            q_proj: q_dense,
            k_proj: k_dense,
            v_proj: v_dense,
            o_proj: o_nvfp4,
            q_norm: dense(store, &format!("{p}.q_norm.weight"))?,
            k_norm: dense(store, &format!("{p}.k_norm.weight"))?,
            q_norm_full: None,
            k_norm_full: None,
            k_scale,
            v_scale,
        };

        // ── MLP weights (NVFP4 on disk, Standard triple-scale format) ──
        // For Gemma-4 dense with bf16_mlp=true we ALSO dequantize the
        // NVFP4 weights to BF16 dense buffers and install them on the
        // FfnComponent::Dense via `set_bf16_weights`. The decode/prefill
        // dispatch in DenseFfnLayer prefers BF16 when set, falls back
        // to NVFP4 otherwise (matches the o_proj BF16 pattern). NVFP4
        // weights are kept loaded as a fallback / for non-MLP code that
        // still uses them.
        let gate_proj = quantized_any(
            store,
            &format!("{lp}.mlp.gate_proj"),
            config.intermediate_size,
            h,
            gpu,
            variant,
            qctx,
        )?;
        let up_proj = quantized_any(
            store,
            &format!("{lp}.mlp.up_proj"),
            config.intermediate_size,
            h,
            gpu,
            variant,
            qctx,
        )?;
        let down_proj = quantized_any(
            store,
            &format!("{lp}.mlp.down_proj"),
            h,
            config.intermediate_size,
            gpu,
            variant,
            qctx,
        )?;
        let ffn_weights = DenseFfnWeights {
            gate_proj,
            up_proj,
            down_proj,
        };
        let bf16_mlp_weights = build_bf16_mlp(store, &lp, bf16_mlp, config, gpu, h)?;
        gpu.synchronize(stream)?;
        tracing::info!(
            "L{i}: FFN weights loaded (bf16_mlp={bf16_mlp}), building DenseFfnLayer (GELU)..."
        );
        let mut ffn_layer =
            DenseFfnLayer::new_with_activation(ffn_weights, FfnActivation::GeLU, gpu)?;
        if ffn_prefill_transpose && bf16_mlp_weights.is_none() {
            install_ffn_prefill_transpose(&mut ffn_layer, gpu, h, config.intermediate_size)?;
        }
        if let Some((g, u, d)) = bf16_mlp_weights {
            ffn_layer.set_bf16_weights(g, u, d);
            tracing::info!("L{i}: BF16 MLP weights installed");
        }
        let ffn = FfnComponent::Dense(ffn_layer);
        gpu.synchronize(stream)?;
        tracing::info!("L{i}: DenseFfnLayer built");

        // ── MoE experts (Gemma-4 26B) — extracted to loader_b ──
        let moe_ffn = build_moe_ffn(
            store, &lp, i, config, gpu, variant, qctx, h, absmax_k, quantize_k, stream,
        )?;

        tracing::info!("L{i}: building attention layer...");

        // ── Construct attention layer (ungated Q, GQA with Q/K norms) ──
        // Full-attention layers keep the same Q head count (32 on 31B, 16 on
        // 26B) with larger head_dim=512 and fewer KV heads (4 on 31B, 2 on 26B).
        // Create a per-layer config override so the attention kernels use correct dimensions.
        let layer_kv_dtype = layer_kv_dtypes[i];
        // Construct with global config (for buffer sizing / kernel selection).
        // Then set per-layer overrides for heterogeneous layers.
        let mut layer = Qwen3AttentionLayer::new_ungated(
            input_norm,
            attn,
            pre_ffn_norm,
            ffn,
            i,
            q_nvfp4_opt,
            k_nvfp4_opt,
            v_nvfp4_opt,
            gpu,
            layer_kv_dtype,
            config.fp8_kv_calibration_tokens,
            config,
        )?;
        // ALL layers get dimension + RoPE overrides because config uses max values
        // for buffer sizing but layers have different actual dimensions.
        //
        // Gemma-4 heterogeneous attention:
        //   31B Sliding: 32 Q × 256 hd, 16 KV × 256 hd  (q_out=8192, kv_out=4096)
        //   31B Full:    32 Q × 512 hd,  4 KV × 512 hd  (q_out=16384, kv_out=2048)
        //   26B Sliding: 16 Q × 256 hd,  8 KV × 256 hd  (q_out=4096, kv_out=2048)
        //   26B Full:    16 Q × 512 hd,  2 KV × 512 hd  (q_out=8192, kv_out=1024)
        let actual_kv_heads = kv_plan.local_kv_heads;
        layer.set_dimension_overrides(actual_head_dim, config.num_attention_heads, actual_kv_heads);
        // Gemma-4: QK-norm handles scaling, so attention scale = 1.0
        layer.set_attn_scale_override(1.0);
        // Install BF16 dense fallback for o_proj — matches Nvidia
        // ModelOpt's official ignore list (`*.self_attn*` covers q/k/v
        // AND o; o was previously runtime-quantized to NVFP4 which
        // accumulated ~7-bit precision loss per layer over 60 layers).
        // The decode/prefill dispatch checks `o_dense_bf16` first and
        // falls through to the NVFP4 path when None.
        if bf16_attn {
            layer.set_o_dense_bf16(o_dense);
        }
        // Set post-sublayer norms (raw; layer_scalar is applied at end of layer)
        layer.set_post_sublayer_norms(post_attn_out_norm, post_ffn_norm_w);
        // Layer scalar: applied to ENTIRE hidden_states at end of layer
        if let Some(scalar) = layer_scalar_val {
            layer.set_layer_scalar(scalar);
        }
        // Gemma-4 v_norm — applied at EVERY layer per HF reference
        // (modeling_gemma4.py:1170 declares
        // `Gemma4RMSNorm(head_dim, with_scale=False)` for v_norm AND
        // line 1220 applies `value_states = self.v_norm(value_states)`
        // unconditionally on every layer that owns its own KV state).
        //
        // K=V detection (full-attention layers where v_proj weight is
        // missing) is preserved separately — it controls weight
        // ALIASING (V buffer points at K projection output), not the
        // application of v_norm. Sliding layers get v_norm without
        // K=V aliasing.
        //
        // Gemma-4's model-specific rms_norm kernel uses the absolute
        // convention `out = x * rms * weight` (see
        // `kernels/gb10/gemma-4-*/nvfp4/rms_norm.cu:100`). To get
        // pure-RMS behavior we use a ONES-filled weight buffer (NOT
        // zeros — zeros would multiply V by zero and wipe the
        // attention contribution).
        let is_k_eq_v = v_aliases_k;
        let v_norm_w = super::loader_b::make_v_norm_ones_bf16(gpu, actual_head_dim)?;
        if is_k_eq_v {
            layer.set_k_eq_v(v_norm_w);
        } else {
            layer.set_v_norm(v_norm_w);
        }
        if is_full_attn {
            // Full attention: head_dim=512, theta=1M, rope_type="proportional"
            // (HF's `_compute_default_rope_parameters` for partial_rotary_factor=0.25):
            //   rope_angles = int(0.25 * head_dim / 2) = 64
            //   pairs are (i, i + head_dim/2), freq denom = head_dim
            // The proportional kernel reads `rotary_dim_override` as rope_angles.
            let rope_angles = ((actual_head_dim as f32 * 0.25) / 2.0) as u32;
            layer.set_rope_overrides(1_000_000.0, rope_angles);
            layer.set_rope_proportional(true);
            // Full layers attend to the entire KV cache — window=None → 0 in kernel.
            layer.set_sliding_window(None);
            tracing::info!(
                "L{i}: full attn: hd={actual_head_dim}, nq={}, nkv={actual_kv_heads}, rope=1M/proportional/angles={rope_angles}, K=V={is_k_eq_v}",
                config.num_attention_heads
            );
        } else {
            // Sliding attention: hd=256, theta=10K, full rotation (rotary_dim = head_dim = 256)
            layer.set_rope_overrides(10_000.0, actual_head_dim as u32);
            // Sliding layers only attend to the last `window_size` tokens.
            if config.sliding_window > 0 {
                layer.set_sliding_window(Some(config.sliding_window));
            }
            if i < 3 {
                tracing::info!(
                    "L{i}: sliding attn: hd={actual_head_dim}, nkv={actual_kv_heads}, window={:?}",
                    config.sliding_window
                );
            }
        }
        // Set dual FFN (MoE alongside dense) if present
        if let Some((moe_comp, pre_norm, post_norm, dense_norm)) = moe_ffn {
            layer.set_moe_ffn(moe_comp, pre_norm, post_norm, dense_norm);
        }

        gpu.synchronize(stream)?;
        tracing::info!("L{i}: attention layer built OK");

        layers.push(Box::new(layer));

        if (i + 1) % 10 == 0 {
            tracing::info!("Loaded layers 0..{}", i + 1);
        }
    }

    tracing::info!(
        "Gemma-4 weight loader: {} layers (all attention, ungated, dense FFN)",
        layers.len(),
    );

    Ok(layers)
}
