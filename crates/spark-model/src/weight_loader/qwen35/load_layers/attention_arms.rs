// SPDX-License-Identifier: AGPL-3.0-only
//
// Helper for the non-FP8 FullAttention arm of `load_layers`. Handles the
// CompressedTensors NVFP4 / Standard / Fp8Dequanted / Bf16Raw variants
// — anything but the native-FP8 (block-scaled-on-disk) path which stays
// inline because it owns enough closures to fight extraction.

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::{WeightDtype, WeightStore};

use crate::layer::TransformerLayer;
use crate::layers::{FfnComponent, Qwen3AttentionLayer};
use crate::tp_shard::{TpShardKind, load_qkvo_tp, shard_dense_bf16, shard_quantized_nvfp4};
use crate::weight_map::{
    AttentionWeights, DenseWeight, Nvfp4Variant, dense_auto, dense_f32_safe,
    dense_modelopt_mixed_or_fp8_or_bf16, load_kv_scales, quantize_to_nvfp4,
    quantized_modelopt_mixed,
};

/// Release a replaced buffer after TP sharding: tombstone and free via
/// [`WeightStore::release_consumed_alias`] if `ptr` aliases a store entry, or
/// free directly via `gpu.free` if it was freshly allocated (e.g. from the
/// `Fp8ViaBf16`/`Mxfp8ViaBf16` runtime quantization route).
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

#[allow(clippy::too_many_arguments)]
pub(super) fn build_full_attention_nvfp4(
    layer_idx: usize,
    store: &WeightStore,
    lp: &str,
    gpu: &dyn GpuBackend,
    variant: Nvfp4Variant,
    config: &ModelConfig,
    h: usize,
    absmax_k: spark_runtime::gpu::KernelHandle,
    quantize_k: spark_runtime::gpu::KernelHandle,
    stream: u64,
    layer_kv_dtype: KvCacheDtype,
    attn_idx: usize,
    input_norm: DenseWeight,
    post_attn_norm: DenseWeight,
    ffn: FfnComponent,
    release_mxfp8_sources: bool,
) -> Result<Box<dyn TransformerLayer>> {
    let p = format!("{lp}.self_attn");
    // Quantization recipes can exclude attention while quantizing the routed
    // experts. Flash-Next NVFP4 is such a mixed checkpoint, so select the
    // attention arm from the projection itself rather than the model-wide
    // variant inferred from expert tensors.
    let variant = store
        .get(&format!("{p}.q_proj.weight"))
        .map(|weight| {
            if weight.dtype == WeightDtype::BF16 {
                Nvfp4Variant::Bf16Raw
            } else {
                variant
            }
        })
        .unwrap_or(variant);
    let tp_rank = config.tp_rank;
    let tp_size = config.tp_world_size.max(1);
    let kv_tp_rank = config.kv_tp_rank();
    let kv_tp_size = config.kv_tp_size();
    let i = layer_idx;

    // `ATLAS_QWEN4_ATTN_BF16=1` decode-only retained copies, filled only
    // by the `CompressedTensors | Standard` arm below when admitted.
    let mut attn_bf16_decode: Option<(DenseWeight, DenseWeight, DenseWeight, DenseWeight)> = None;

    // BF16 dequant of o_proj from the Fp8Dequanted / Bf16Raw arm: no struct
    // keeps it (o runs on its NVFP4 copy), so dense dedup releases it.
    let mut o_bf16_source: Option<DenseWeight> = None;
    let (attn, q_nvfp4, k_nvfp4, v_nvfp4) = match variant {
        // NVFP4 already packed in the checkpoint (compressed-tensors:
        // `.weight_packed` + `weight_global_scale`; modelopt: uint8
        // `.weight` + `weight_scale_2`). `quantized_auto` reads either
        // schema and returns the packed pointer without going through
        // BF16, so we shard the packed bytes directly. Treating
        // modelopt as Bf16Raw blew up here previously: `dense_auto`
        // returned a uint8 ptr aliased as BF16, then quantize_to_nvfp4
        // launched an absmax kernel that read 2× the allocation and
        // hit ILLEGAL_ADDRESS at the first full_attention layer.
        Nvfp4Variant::CompressedTensors | Nvfp4Variant::Standard => {
            anyhow::ensure!(
                !super::tq_plus_weight_rotation::weight_rotation_enabled(),
                "TQ_PLUS_WEIGHT_ROTATION=1 is not supported for pre-packed {variant:?} attention weights"
            );
            let group_size = 16usize;
            let load_nvfp4 = |name: &str,
                              full_n: usize,
                              full_k: usize,
                              kind: TpShardKind|
             -> Result<crate::weight_map::QuantizedWeight> {
                let prefix = format!("{p}.{name}");
                let src = quantized_modelopt_mixed(
                    store, &prefix, full_n, full_k, config, variant, gpu, absmax_k, quantize_k,
                    stream,
                )?;
                let (eff_rank, eff_size) = if matches!(name, "k_proj" | "v_proj") {
                    (kv_tp_rank, kv_tp_size)
                } else {
                    (tp_rank, tp_size)
                };
                if eff_size == 1 {
                    return Ok(src);
                }
                let sharded = shard_quantized_nvfp4(
                    &src, full_n, full_k, kind, eff_rank, eff_size, group_size, gpu,
                )?;
                if sharded.weight != src.weight {
                    release_ptr_or_store_alias(
                        store,
                        &[
                            format!("{prefix}.weight_packed"),
                            format!("{prefix}.weight"),
                        ],
                        src.weight,
                        gpu,
                    )?;
                }
                if sharded.weight_scale != src.weight_scale {
                    release_ptr_or_store_alias(
                        store,
                        &[format!("{prefix}.weight_scale")],
                        src.weight_scale,
                        gpu,
                    )?;
                }
                Ok(sharded)
            };
            let [q, k, v, o] = load_qkvo_tp(config, load_nvfp4)?;
            let dummy = DenseWeight {
                weight: DevicePtr::NULL,
            };
            let (k_scale, v_scale) = load_kv_scales(store, &p, gpu);
            let attn = AttentionWeights {
                q_proj: dummy,
                k_proj: dummy,
                v_proj: dummy,
                o_proj: o,
                q_norm: dense_f32_safe(store, &format!("{p}.q_norm.weight"), gpu)?,
                k_norm: dense_f32_safe(store, &format!("{p}.k_norm.weight"), gpu)?,
                q_norm_full: None,
                k_norm_full: None,
                k_scale,
                v_scale,
            };
            // `ATLAS_QWEN4_ATTN_BF16=1` (default off, decode-only; refused
            // at launch for anything but C1/batch=1/TP1/EP1 qwen4_exp —
            // see `qwen4_decode_bf16_arms`): keep the checkpoint's exact
            // MXFP8->BF16 dequant of q/k/v/o alongside the NVFP4 copies
            // above, and route ONLY decode through it via the
            // decode-only fields. `q_nvfp4`/`k_nvfp4`/`v_nvfp4`/`o` (in
            // `attn.o_proj`) above are untouched, so prefill is
            // unchanged. `tp_size == 1` is guaranteed by the admission
            // refusal of multi-rank launches.
            if config.is_qwen4_exp() && tp_size == 1 {
                let attn_bf16_engaged =
                    crate::layers::qwen4_decode_bf16_arms::attn_bf16_selected()?;
                if attn_bf16_engaged {
                    let dims = crate::tp_shard::TpAttentionDims::from_config(config);
                    let q_bf16 = crate::weight_map::modelopt_mixed_bf16_retained(
                        store,
                        &format!("{p}.q_proj"),
                        dims.full_q_n,
                        dims.h,
                        config,
                        gpu,
                    )?;
                    let k_bf16 = crate::weight_map::modelopt_mixed_bf16_retained(
                        store,
                        &format!("{p}.k_proj"),
                        dims.full_kv_n,
                        dims.h,
                        config,
                        gpu,
                    )?;
                    let v_bf16 = crate::weight_map::modelopt_mixed_bf16_retained(
                        store,
                        &format!("{p}.v_proj"),
                        dims.full_kv_n,
                        dims.h,
                        config,
                        gpu,
                    )?;
                    let o_bf16 = crate::weight_map::modelopt_mixed_bf16_retained(
                        store,
                        &format!("{p}.o_proj"),
                        dims.h,
                        dims.full_o_in,
                        config,
                        gpu,
                    )?;
                    match (q_bf16, k_bf16, v_bf16, o_bf16) {
                        (Some(q_bf16), Some(k_bf16), Some(v_bf16), Some(o_bf16)) => {
                            static LOG_ATTN_QKVO: std::sync::Once = std::sync::Once::new();
                            LOG_ATTN_QKVO.call_once(|| {
                                tracing::info!(
                                    "QWEN4_ATTN_BF16 q/k/v/o: decode-only BF16 dense_gemv, prefill unchanged (extra_resident_bytes_per_layer={})",
                                    (dims.full_q_n * dims.h
                                        + 2 * dims.full_kv_n * dims.h
                                        + dims.h * dims.full_o_in)
                                        * 2
                                );
                            });
                            attn_bf16_decode = Some((q_bf16, k_bf16, v_bf16, o_bf16));
                        }
                        _ => anyhow::bail!(
                            "{}=1 refused: layer {i} attention projections are not MXFP8/FP8 \
                             ModelOpt-mixed on disk, nothing to retain in BF16",
                            crate::layers::qwen4_decode_bf16_arms::QWEN4_ATTN_BF16_ENV
                        ),
                    }
                }
            }
            (attn, Some(q), Some(k), Some(v))
        }
        Nvfp4Variant::Fp8Dequanted | Nvfp4Variant::Bf16Raw => {
            tracing::info!("Layer {i}: loading attention projections ({variant:?})");
            let load_bf16_then_nvfp4 = |name: &str,
                                        full_n: usize,
                                        full_k: usize,
                                        kind: TpShardKind|
             -> Result<(
                DenseWeight,
                crate::weight_map::QuantizedWeight,
            )> {
                let src = dense_auto(store, &format!("{p}.{name}.weight"), gpu)?;
                let (eff_rank, eff_size) = if matches!(name, "k_proj" | "v_proj") {
                    (kv_tp_rank, kv_tp_size)
                } else {
                    (tp_rank, tp_size)
                };
                let (mut sharded_ptr, local_n, local_k) =
                    shard_dense_bf16(src.weight, full_n, full_k, kind, eff_rank, eff_size, gpu)?;
                // TQ+ weight pre-rotation: apply canonical two-sided Rademacher
                // rotation S2·H·S1 along each head's output axis of Q/K/V
                // BEFORE quantization (all three vendored head dims 128/256/512
                // carry TQ+ sign masks in `tq_plus_signs.cuh`). O-projection
                // inverse rotation (`wht_bf16_inplace_inv`) is not yet wired.
                // Never mutate a `WeightStore` alias in place: allocate an
                // owned rotated scratch buffer and free it after quantization.
                if super::tq_plus_weight_rotation::weight_rotation_enabled()
                    && matches!(name, "q_proj" | "k_proj" | "v_proj")
                {
                    let layout = if name == "q_proj" && config.attn_gated {
                        super::tq_plus_weight_rotation::ProjHeadLayout::GatedInterleaved
                    } else {
                        super::tq_plus_weight_rotation::ProjHeadLayout::Standard
                    };
                    let geom = super::tq_plus_weight_rotation::RowMajorProjGeometry {
                        out_dim: local_n,
                        in_dim: local_k,
                        head_dim: config.head_dim,
                        layout,
                    };
                    let owned_rotated =
                        match super::tq_plus_weight_rotation::rotate_row_major_qkv_bf16(
                            gpu,
                            sharded_ptr,
                            geom,
                            stream,
                        ) {
                            Ok(ptr) => ptr,
                            Err(err) => {
                                if sharded_ptr != src.weight {
                                    let _ = gpu.free(sharded_ptr);
                                }
                                return Err(err);
                            }
                        };
                    if sharded_ptr != src.weight {
                        gpu.free(sharded_ptr)?;
                    }
                    sharded_ptr = owned_rotated;
                    let n_heads = layout.num_rotated_heads(local_n, config.head_dim)?;
                    tracing::info!(
                        "TQ+ weight rotation applied: {name} ({n_heads}h × {}hd, layout={layout:?})",
                        config.head_dim
                    );
                }
                let sharded = DenseWeight {
                    weight: sharded_ptr,
                };
                let q = quantize_to_nvfp4(
                    &sharded, local_n, local_k, gpu, absmax_k, quantize_k, stream,
                );
                if sharded_ptr != src.weight {
                    gpu.free(sharded_ptr)?;
                }
                let q = q?;
                Ok((src, q))
            };
            tracing::info!("Layer {i}: BF16 → NVFP4 (TP-aware)");
            let [
                (q_dense, q_nvfp4),
                (k_dense, k_nvfp4),
                (v_dense, v_nvfp4),
                (o_dense, o_nvfp4),
            ] = load_qkvo_tp(config, load_bf16_then_nvfp4)?;
            o_bf16_source = Some(o_dense);
            tracing::info!(
                "Layer {i}: Q/K/V/O quantized, {:.1} GB free",
                gpu.free_memory()? as f64 / (1024.0 * 1024.0 * 1024.0)
            );

            let (k_scale, v_scale) = load_kv_scales(store, &p, gpu);

            let attn = AttentionWeights {
                q_proj: q_dense,
                k_proj: k_dense,
                v_proj: v_dense,
                o_proj: o_nvfp4,
                q_norm: dense_f32_safe(store, &format!("{p}.q_norm.weight"), gpu)?,
                k_norm: dense_f32_safe(store, &format!("{p}.k_norm.weight"), gpu)?,
                q_norm_full: None,
                k_norm_full: None,
                k_scale,
                v_scale,
            };
            (attn, Some(q_nvfp4), Some(k_nvfp4), Some(v_nvfp4))
        }
    };

    let mut layer = Qwen3AttentionLayer::new(
        input_norm,
        attn,
        post_attn_norm,
        ffn,
        attn_idx,
        q_nvfp4,
        k_nvfp4,
        v_nvfp4,
        gpu,
        layer_kv_dtype,
        config.fp8_kv_calibration_tokens,
        config,
    )?;

    if let Some((q_bf16, k_bf16, v_bf16, o_bf16)) = attn_bf16_decode {
        layer.set_qkvo_dense_bf16_decode_only(q_bf16, k_bf16, v_bf16, o_bf16);
    } else if config.is_qwen4_exp() && crate::layers::qwen4_decode_bf16_arms::attn_bf16_selected()?
    {
        // `ATLAS_QWEN4_ATTN_BF16=1` only engages inside the
        // `CompressedTensors | Standard` arm above (the census row-#5
        // MXFP8 path). Any other variant (native-FP8 block-scaled,
        // `Fp8Dequanted`, `Bf16Raw`) would silently leave this layer on
        // NVFP4 decode while the control claims to be engaged elsewhere
        // — refuse loudly instead of silently changing nothing.
        anyhow::bail!(
            "{}=1 refused: layer {i} attention variant {variant:?} has no MXFP8/FP8 \
             ModelOpt-mixed source to retain in BF16 (only CompressedTensors/Standard \
             checkpoints are qualified)",
            crate::layers::qwen4_decode_bf16_arms::QWEN4_ATTN_BF16_ENV
        );
    }

    if config.is_qwen4_exp() && config.qwen4_qsa {
        anyhow::ensure!(
            config.indexer_n_heads == 4
                && config.indexer_kv_heads == 1
                && config.indexer_head_dim == 128
                && config.indexer_budget == 2048
                && config.indexer_compress_ratio == 4,
            "unsupported Qwen4 QSA geometry"
        );
        layer.set_qwen4_qsa(crate::layers::Qwen4QsaIndexer::new(
            dense_modelopt_mixed_or_fp8_or_bf16(
                store,
                &format!("{p}.indexer.index_qk_proj"),
                config,
                gpu,
            )?,
            dense_f32_safe(store, &format!("{p}.indexer.q_layernorm.weight"), gpu)?,
            dense_f32_safe(store, &format!("{p}.indexer.k_layernorm.weight"), gpu)?,
            gpu,
            config,
        )?);
    }

    let skip_qwen4_legacy_prefill = config.is_qwen4_exp()
        && std::env::var("ATLAS_GDN_BF16_WEIGHTS").ok().as_deref() != Some("1")
        && std::env::var("ATLAS_QWEN4_RELEASE_BF16_QKVZ")
            .ok()
            .as_deref()
            == Some("1");
    if !skip_qwen4_legacy_prefill {
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let gated = config.attn_gated;
        let q_proj_n = if gated {
            num_heads * head_dim * 2
        } else {
            num_heads * head_dim
        };
        if let Some(ref qw) = q_nvfp4 {
            let qt = qw.transpose_for_gemm_cached(
                gpu,
                q_proj_n,
                h,
                &format!("L{i}.self_attn.q_proj.t"),
            )?;
            let kt = k_nvfp4.as_ref().unwrap().transpose_for_gemm_cached(
                gpu,
                num_kv_heads * head_dim,
                h,
                &format!("L{i}.self_attn.k_proj.t"),
            )?;
            let vt = v_nvfp4.as_ref().unwrap().transpose_for_gemm_cached(
                gpu,
                num_kv_heads * head_dim,
                h,
                &format!("L{i}.self_attn.v_proj.t"),
            )?;
            let ot = layer.attn.o_proj.transpose_for_gemm_cached(
                gpu,
                h,
                num_heads * head_dim,
                &format!("L{i}.self_attn.o_proj.t"),
            )?;
            layer.set_prefill_weights(Some(qt), Some(kt), Some(vt), Some(ot));
        }
        layer.predequant_for_prefill(gpu, config, stream)?;
    }

    // Lever 5: opt-in sliding-window cap on full-attention layers
    // (ATLAS_ATTN_SLIDING_WINDOW). Default None = global attention →
    // byte-identical to the un-gated build.
    if let Some(w) = super::attn_sliding_window::sliding_window_override() {
        layer.set_sliding_window(Some(w));
    }
    if super::dense_fp8::engaged(config)? {
        layer.set_qwen4_dense_fp8(super::dense_fp8::load_attention(store, lp, config, gpu)?);
    }

    // ATLAS_QWEN4_DENSE_DEDUP=1: with the legacy prefill copies skipped, the
    // NVFP4 q/k/v/o are the only operands any qwen4 attention route reads.
    // Target layers only: the MTP layer's projections are also read by the
    // native MTP proposer construction that runs after this loader.
    if skip_qwen4_legacy_prefill
        && super::dense_dedup::selected()
        && super::dense_dedup::is_target_layer_prefix(lp)
        && let Some(o_dense) = o_bf16_source
    {
        let bytes = release_attention_dense_copies(store, &p, &mut layer, o_dense, gpu, stream)?;
        super::dense_dedup::log_layer(&p, "attention", bytes);
    }

    // ATLAS_QWEN4_RELEASE_MXFP8_SOURCES: release q/k/v/o MXFP8 store
    // originals only after every reader above has run (dense dedup's
    // `release_attention_dense_copies` looks the same names up). Only the
    // Standard/CompressedTensors arm read them via `quantized_modelopt_mixed`.
    // Target layers only: the MTP layer's projections are read later.
    if release_mxfp8_sources
        && matches!(
            variant,
            Nvfp4Variant::CompressedTensors | Nvfp4Variant::Standard
        )
        && super::dense_dedup::is_target_layer_prefix(lp)
    {
        gpu.synchronize(stream)?;
        let mut bytes = 0usize;
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            bytes += super::mxfp8_release::release_attention_proj(store, &p, name, gpu)?;
        }
        super::mxfp8_release::log_layer(&p, "attention", bytes);
    }

    Ok(Box::new(layer))
}

/// Release the attention projections' BF16 dequants (q/k/v fallbacks and the
/// unreferenced o) and their FP8 store originals. The q/k/v fallbacks are
/// nulled only when the layer's decode/prefill weights are NVFP4, the state
/// every qwen4 route dispatches on before it would reach the BF16 fallback.
fn release_attention_dense_copies(
    store: &WeightStore,
    p: &str,
    layer: &mut Qwen3AttentionLayer,
    o_dense: DenseWeight,
    gpu: &dyn GpuBackend,
    stream: u64,
) -> Result<super::dense_dedup::DedupBytes> {
    let bf16_bytes = |name: &str| -> Result<usize> {
        Ok(store.get(&format!("{p}.{name}.weight"))?.num_elements() * 2)
    };
    // A reader that pins attention (ATTN_BF16, DENSE_FP8, PREFILL_NATIVE)
    // keeps the BF16 fallbacks, the o dequant and the store originals. Report
    // all four projections with their actual logical BF16 byte sizes.
    let mut pinned = false;
    for name in ["q_proj", "k_proj", "v_proj", "o_proj"] {
        let logical = bf16_bytes(name)?;
        if crate::source_pins::skip_if_pinned(
            &format!("{p}.{name}"),
            logical,
            crate::source_pins::PinKind::Bf16Alias,
        )?
        .is_some()
        {
            pinned = true;
        }
    }
    if pinned {
        return Ok(super::dense_dedup::DedupBytes::default());
    }
    anyhow::ensure!(
        layer.has_nvfp4_qkv_weights(),
        "dense dedup: {p} q/k/v are not NVFP4; refusing to drop their BF16 fallback"
    );
    gpu.synchronize(stream)?;
    let mut bytes = super::dense_dedup::DedupBytes::default();
    let fallbacks = [
        ("q_proj", layer.attn.q_proj),
        ("k_proj", layer.attn.k_proj),
        ("v_proj", layer.attn.v_proj),
        ("o_proj", o_dense),
    ];
    for (name, dense) in fallbacks {
        let logical = bf16_bytes(name)?;
        bytes.bf16_copies += super::linear_attn_arms::release_replaced_bf16_source(
            store,
            &format!("{p}.{name}.weight"),
            dense,
            logical,
            gpu,
        )?;
    }
    let null = DenseWeight {
        weight: spark_runtime::gpu::DevicePtr::NULL,
    };
    layer.attn.q_proj = null;
    layer.attn.k_proj = null;
    layer.attn.v_proj = null;
    for name in ["q_proj", "k_proj", "v_proj", "o_proj"] {
        bytes.store_originals +=
            super::dense_dedup::release_fp8_store_source(store, &format!("{p}.{name}"), gpu)?;
    }
    Ok(bytes)
}
