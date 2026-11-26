// SPDX-License-Identifier: AGPL-3.0-only

mod load_layers;

use anyhow::{Context, Result};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use super::ModelWeightLoader;
use crate::layer::TransformerLayer;
use crate::weight_map::{
    DenseWeight, MtpWeights, dense, dense_auto, dense_f32_safe, detect_nvfp4_variant, load_mtp,
};

pub struct Qwen35WeightLoader;

pub(crate) fn qwen4_mtp_config(target: &ModelConfig) -> ModelConfig {
    let mut config = target.clone();
    let tp = config.tp_world_size.max(1);
    let kv_tp = config.kv_tp_size();
    config.num_attention_heads *= tp;
    config.num_key_value_heads *= kv_tp;
    config.linear_num_key_heads *= tp;
    config.linear_num_value_heads *= tp;
    config.tp_rank = 0;
    config.tp_world_size = 1;
    config.tp_kv_replicate = 1;
    config.ep_rank = 0;
    config.ep_world_size = 1;
    config.pp_rank = 0;
    config.pp_world_size = 1;
    config.pp_layer_offset = 0;
    config.num_hidden_layers = 1;
    config.layer_types = vec![atlas_core::config::LayerType::FullAttention];
    config.full_attention_interval = 1;
    config.weight_prefix = "mtp".to_string();
    config.dflash_capture_layers.clear();
    config.vision = None;
    config
}

/// `ATLAS_QWEN4_MTP_UNIFIED_T=1` (default off): build the MTP layer's routed
/// experts through the phased unified-layout transpose, which frees the
/// untransposed copies, instead of the load-time `transpose_for_prefill`,
/// which keeps them (~1.3 GiB on Flash-Next). Under `ATLAS_UNIFIED_MOE_LAYOUT=1`
/// without the hybrid layout the MTP head's one-row MoE already dispatches the
/// transposed copy (`MoeLayer::use_t_layout_for_decode`), so the originals are
/// never read, and both paths build that copy with the same
/// `transpose_for_gemm` calls.
fn qwen4_mtp_unified_t_selected() -> Result<bool> {
    let layout = |name: &str| {
        std::env::var(name)
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    };
    if std::env::var("ATLAS_QWEN4_MTP_UNIFIED_T").ok().as_deref() != Some("1") {
        return Ok(false);
    }
    anyhow::ensure!(
        layout("ATLAS_UNIFIED_MOE_LAYOUT") && !layout("ATLAS_HYBRID_MOE_LAYOUT"),
        "ATLAS_QWEN4_MTP_UNIFIED_T=1 requires ATLAS_UNIFIED_MOE_LAYOUT=1 \
         without ATLAS_HYBRID_MOE_LAYOUT=1"
    );
    Ok(true)
}

pub(crate) fn load_qwen4_mtp_layer(
    store: &WeightStore,
    target: &ModelConfig,
    gpu: &dyn GpuBackend,
    kv_dtype: KvCacheDtype,
) -> Result<Box<dyn TransformerLayer>> {
    let config = qwen4_mtp_config(target);
    let unified_t = qwen4_mtp_unified_t_selected()?;
    let mut layers = {
        let _defer = unified_t.then(load_layers::DeferMoeTranspose::enter);
        load_layers::load_layers(&Qwen35WeightLoader, store, &config, gpu, &[kv_dtype])?
    };
    anyhow::ensure!(layers.len() == 1, "Qwen4 MTP must load exactly one layer");
    let mut layer = layers.remove(0);
    if unified_t {
        let free_before = gpu.free_memory().unwrap_or(0);
        layer.transpose_moe_for_prefill_unified(gpu, &config)?;
        let free_after = gpu.free_memory().unwrap_or(0);
        // Near zero when the originals were freed as the transposed copy was
        // built; the load-time path keeps both (~+1350 MiB on Flash-Next).
        let net_alloc_mib = (free_before as i64 - free_after as i64) / (1024 * 1024);
        tracing::info!(
            "QWEN4_MTP_UNIFIED_T_ENGAGED selector=ATLAS_QWEN4_MTP_UNIFIED_T \
             originals_freed=true net_alloc_mib={net_alloc_mib}"
        );
    }
    Ok(layer)
}

impl ModelWeightLoader for Qwen35WeightLoader {
    fn supports_tp(&self) -> bool {
        // Both FullAttention layers (across all 3 quant paths: FP8 native,
        // NVFP4-from-disk, BF16 → NVFP4) and LinearAttention (GDN SSM) layers
        // (HeadParallel sharding in `weight_map::load_ssm_qwen35`) are
        // TP-sharded across ranks. Routed MoE experts are EP-sharded.
        true
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        load_layers::load_layers(self, store, config, gpu, layer_kv_dtypes)
    }

    fn load_embedding(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        let prefix = &config.weight_prefix;
        dense(store, &format!("{prefix}.embed_tokens.weight"))
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        let prefix = &config.weight_prefix;
        if config.is_qwen4_exp() {
            // Qwen4 has no terminal RMSNorm; the model-level hyperconnection
            // mixer is installed separately. Return a real tensor to satisfy
            // the architecture-neutral constructor without allocating a
            // dummy. `final_norm_rows` treats it as an identity only after the
            // terminal mixer has produced the H-wide LM-head input.
            return dense_f32_safe(
                store,
                &format!("{prefix}.hyper_connection_mixer.hc_norm.weight"),
                gpu,
            );
        }
        dense_f32_safe(store, &format!("{prefix}.norm.weight"), gpu)
    }

    fn load_lm_head(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        super::qwen35_mixed_precision::load_lm_head(store, config, gpu)
    }

    fn load_qwen4_final_mixer(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<crate::layers::Qwen4HyperConnection>> {
        if !config.is_qwen4_exp() {
            return Ok(None);
        }
        let p = format!("{}.hyper_connection_mixer", config.weight_prefix);
        let norm = dense_f32_safe(store, &format!("{p}.hc_norm.weight"), gpu)?;
        let down_key = format!("{p}.input_mix_weight_down.weight");
        let up_key = format!("{p}.input_mix_weight_up.weight");
        // ATLAS_QWEN4_HC_BF16=1 (default off): the control is only
        // meaningful for a tensor that is BF16 in the checkpoint.
        let keep_bf16 = crate::layers::qwen4_hc_bf16::selected()?;
        if keep_bf16 {
            let r = config.residual_width();
            let rank = config.hc_lowrank;
            let check = |key: &str, expected_shape: &[usize]| -> Result<()> {
                let stored = store.get(key)?;
                crate::layers::qwen4_hc_bf16::validate_stored_bf16_tensor(
                    key,
                    crate::layers::qwen4_hc_bf16::StoredHcTensor {
                        dtype: stored.dtype,
                        shape: &stored.shape,
                    },
                    expected_shape,
                )
            };
            check(&down_key, &[rank, r])?;
            check(&up_key, &[r, rank])?;
        }
        let down_dense = dense_auto(store, &down_key, gpu)?;
        let up_dense = dense_auto(store, &up_key, gpu)?;
        let absmax = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
        let quantize = gpu
            .kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4_weight_rne")
            .or_else(|_| gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4"))?;
        let stream = gpu.default_stream();
        let r = config.residual_width();
        let down = crate::weight_map::quantize_to_nvfp4_cached(
            &down_dense,
            config.hc_lowrank,
            r,
            gpu,
            absmax,
            quantize,
            stream,
            "qwen4.final_mixer.down.nvfp4",
        )?;
        let up = crate::weight_map::quantize_to_nvfp4_cached(
            &up_dense,
            r,
            config.hc_lowrank,
            gpu,
            absmax,
            quantize,
            stream,
            "qwen4.final_mixer.up.nvfp4",
        )?;
        {
            // ATLAS_QWEN4_PREFILL_NATIVE=1: resident BF16 sources of the final mixer.
            use crate::layers::qwen4_prefill_native::{Module, register_bf16, store_retains_bf16};
            let kept =
                |name: &str, ptr| store_retains_bf16(store, &format!("{p}.{name}.weight"), ptr);
            let (dk, uk) = (
                kept("input_mix_weight_down", down_dense.weight),
                kept("input_mix_weight_up", up_dense.weight),
            );
            register_bf16(
                config,
                Module::FinalDown,
                &down,
                down_dense.weight,
                config.hc_lowrank,
                r,
                dk,
            )?;
            register_bf16(
                config,
                Module::FinalUp,
                &up,
                up_dense.weight,
                r,
                config.hc_lowrank,
                uk,
            )?;
        }
        let (down_bf16, up_bf16, _) = crate::layers::qwen4_hc_bf16::resident_copies(
            keep_bf16,
            down_dense,
            up_dense,
            None::<DenseWeight>,
        );
        // ATLAS_QWEN4_DENSE_DEDUP=1: release the BF16 sources unless a reader
        // above (HC_BF16, PREFILL_NATIVE) pinned them (`source_pins`).
        load_layers::release_final_mixer_sources(store, &p, down_dense, up_dense, gpu, stream)?;
        Ok(Some(crate::layers::Qwen4HyperConnection::new_with_bf16(
            norm,
            down,
            up,
            None,
            down_bf16,
            up_bf16,
            None,
            config.hidden_size,
            config.hc_count,
            config.hc_lowrank,
            gpu,
        )?))
    }

    fn load_mtp_weights(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        if !store.contains("mtp.fc.weight") {
            let qwen4_mtp_tensors =
                config.is_qwen4_exp() && store.names().any(|name| name.starts_with("mtp."));
            if qwen4_mtp_tensors {
                tracing::info!(
                    "Qwen4 MTP-prefixed tensors detected; exact packed/numbered schema admission is deferred to the factory"
                );
            } else {
                tracing::info!("No MTP weights found — speculative decoding disabled");
            }
            return Ok(None);
        }
        let variant = detect_nvfp4_variant(store, config);
        tracing::info!(
            "Loading MTP weights ({} experts, variant={:?})...",
            config.num_experts,
            variant
        );
        let mtp = load_mtp(store, config.num_experts, gpu, variant)?;
        let fc_shape = store
            .get("mtp.fc.weight")
            .ok()
            .filter(|t| t.shape.len() == 2)
            .map(|t| (t.shape[0], t.shape[1]))
            .unwrap_or((config.hidden_size, config.hidden_size * 2));
        tracing::info!(
            "MTP weights loaded: fc=[{},{}], {} experts, attn layer",
            fc_shape.0,
            fc_shape.1,
            mtp.experts.len(),
        );
        Ok(Some(mtp))
    }

    /// Load the Qwen3.5/3.6/3.8 ViT tower. Returns `None` when `config.vision`
    /// is `None` (text-only checkpoints) or vision tensors are absent. Probes
    /// both `model.visual.*` and `model.language_model.visual.*`, loads
    /// `vcfg.depth` blocks plus optional deepstack mergers and the final
    /// merger, and auto-dequants any FP8/MXFP8/NVFP4 linear projections to BF16
    /// across all blocks and mergers.
    fn load_vision_encoder(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<crate::layers::VisionEncoder>> {
        use crate::weight_map::dense_modelopt_mixed_or_fp8_or_bf16;
        let vcfg = match &config.vision {
            Some(v) => v.clone(),
            None => return Ok(None),
        };
        // AEON-7's v2 NVFP4 re-quant (and other multimodal-preserved
        // checkpoints quantized via AutoModelForImageTextToText) keeps
        // the canonical nested layout `model.language_model.visual.*`
        // instead of the flat `model.visual.*` form. Probe the canonical
        // tensor under both prefixes; first hit wins.
        let vp = if store.contains("model.visual.patch_embed.proj.weight") {
            "model.visual"
        } else if store.contains("model.language_model.visual.patch_embed.proj.weight") {
            "model.language_model.visual"
        } else {
            tracing::warn!(
                "Vision encoder tensors absent under both `model.visual.*` and \
                 `model.language_model.visual.*`; skipping vision tower (text-only mode)"
            );
            return Ok(None);
        };

        // Patch embed + position embed are always BF16.
        let patch_embed_w = dense(store, &format!("{vp}.patch_embed.proj.weight"))?;
        let patch_embed_b = dense(store, &format!("{vp}.patch_embed.proj.bias"))?;
        let pos_embed = dense(store, &format!("{vp}.pos_embed.weight"))?;
        let pos_embed_shape = store.get(&format!("{vp}.pos_embed.weight"))?.shape.clone();
        let num_position_embeddings = pos_embed_shape
            .first()
            .copied()
            .context("pos_embed shape missing rows")?;

        let mut blocks = Vec::with_capacity(vcfg.depth);
        for i in 0..vcfg.depth {
            let bp = format!("{vp}.blocks.{i}");
            blocks.push(crate::layers::ViTBlock {
                norm1_w: dense(store, &format!("{bp}.norm1.weight"))?.weight,
                norm1_b: dense(store, &format!("{bp}.norm1.bias"))?.weight,
                qkv_w: dense_modelopt_mixed_or_fp8_or_bf16(
                    store,
                    &format!("{bp}.attn.qkv"),
                    config,
                    gpu,
                )?
                .weight,
                qkv_b: dense(store, &format!("{bp}.attn.qkv.bias"))?.weight,
                proj_w: dense_modelopt_mixed_or_fp8_or_bf16(
                    store,
                    &format!("{bp}.attn.proj"),
                    config,
                    gpu,
                )?
                .weight,
                proj_b: dense(store, &format!("{bp}.attn.proj.bias"))?.weight,
                norm2_w: dense(store, &format!("{bp}.norm2.weight"))?.weight,
                norm2_b: dense(store, &format!("{bp}.norm2.bias"))?.weight,
                fc1_w: dense_modelopt_mixed_or_fp8_or_bf16(
                    store,
                    &format!("{bp}.mlp.linear_fc1"),
                    config,
                    gpu,
                )?
                .weight,
                fc1_b: dense(store, &format!("{bp}.mlp.linear_fc1.bias"))?.weight,
                fc2_w: dense_modelopt_mixed_or_fp8_or_bf16(
                    store,
                    &format!("{bp}.mlp.linear_fc2"),
                    config,
                    gpu,
                )?
                .weight,
                fc2_b: dense(store, &format!("{bp}.mlp.linear_fc2.bias"))?.weight,
            });
        }

        let mut deepstack = Vec::with_capacity(vcfg.deepstack_visual_indexes.len());
        for i in 0..vcfg.deepstack_visual_indexes.len() {
            let mp = format!("{vp}.deepstack_merger_list.{i}");
            deepstack.push(crate::layers::MergerLayer {
                norm_w: dense(store, &format!("{mp}.norm.weight"))?.weight,
                norm_b: dense(store, &format!("{mp}.norm.bias"))?.weight,
                fc1_w: dense_modelopt_mixed_or_fp8_or_bf16(
                    store,
                    &format!("{mp}.linear_fc1"),
                    config,
                    gpu,
                )?
                .weight,
                fc1_b: dense(store, &format!("{mp}.linear_fc1.bias"))?.weight,
                fc2_w: dense_modelopt_mixed_or_fp8_or_bf16(
                    store,
                    &format!("{mp}.linear_fc2"),
                    config,
                    gpu,
                )?
                .weight,
                fc2_b: dense(store, &format!("{mp}.linear_fc2.bias"))?.weight,
            });
        }

        let mp = format!("{vp}.merger");
        let merger = crate::layers::MergerLayer {
            norm_w: dense(store, &format!("{mp}.norm.weight"))?.weight,
            norm_b: dense(store, &format!("{mp}.norm.bias"))?.weight,
            fc1_w: dense_modelopt_mixed_or_fp8_or_bf16(
                store,
                &format!("{mp}.linear_fc1"),
                config,
                gpu,
            )?
            .weight,
            fc1_b: dense(store, &format!("{mp}.linear_fc1.bias"))?.weight,
            fc2_w: dense_modelopt_mixed_or_fp8_or_bf16(
                store,
                &format!("{mp}.linear_fc2"),
                config,
                gpu,
            )?
            .weight,
            fc2_b: dense(store, &format!("{mp}.linear_fc2.bias"))?.weight,
        };

        let deepstack_indexes = vcfg.deepstack_visual_indexes.clone();
        let ve = crate::layers::VisionEncoder::new(
            patch_embed_w.weight,
            patch_embed_b.weight,
            pos_embed.weight,
            num_position_embeddings,
            blocks,
            deepstack,
            deepstack_indexes,
            merger,
            vcfg.hidden_size,
            vcfg.num_heads,
            vcfg.spatial_merge_size,
            vcfg.out_hidden_size,
            vcfg.intermediate_size,
            gpu,
        )?;
        tracing::info!(
            "Qwen3.5/3.6 vision encoder loaded: prefix={}, depth={}, hidden={}, heads={}",
            vp,
            vcfg.depth,
            vcfg.hidden_size,
            vcfg.num_heads,
        );
        Ok(Some(ve))
    }
}
