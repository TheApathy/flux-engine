// SPDX-License-Identifier: AGPL-3.0-only

//! Weight loader for Gemma-4 models.
//!
//! Gemma-4 is a pure-attention model (no SSM) with:
//! - 5:1 sliding + full attention pattern (dual RoPE theta)
//! - Heterogeneous head dimensions: `head_dim=256` (sliding) / `512` (full)
//! - GeGLU activation (gate + up projection fused)
//! - Tied word embeddings
//! - 4 layer norms per layer on 31B dense (`input`, `post_attn`, `pre_ffn`,
//!   `post_ffn`); 7 norms + 128-expert dual-FFN MoE on 26B MoE
//! - Per-layer scalar (`layer_scalar`)
//! - Attention weights: BF16 on disk (31B) or NVFP4 dequantized to BF16
//!   (26B MoE); NVFP4 MLP weights (standard triple-scale or compressed-tensors)
//!
//! Weight prefix: `model.language_model.` (auto-detected from safetensors keys
//! by the main loader when `nested_config = true`).

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use super::ModelWeightLoader;
use crate::layer::TransformerLayer;
use crate::weight_map::{DenseWeight, MtpWeights};

mod ffn_prefill;
#[cfg(test)]
mod ffn_prefill_tests;
mod loader_a;
mod loader_b;

pub struct Gemma4WeightLoader;

impl ModelWeightLoader for Gemma4WeightLoader {
    fn supports_tp(&self) -> bool {
        // Gemma-4 has per-layer dim overrides: sliding layers use
        // head_dim=256 explicitly, full-attention layers use whatever
        // the on-disk weight's [out, in] shape declares. We slice
        // each layer's q/k/v BF16 weights with the layer-specific
        // full dims (read from store) and replicate K/V heads per rank
        // group when `tp_size` exceeds a full layer's KV head count,
        // then pass the per-rank-local dims to quantize_to_nvfp4. K=V
        // aliasing in full-attention layers works under TP because both
        // K and V share the same per-rank slice pointer (the alias
        // survives sharding).
        // MLP weights stay full-replica per rank — Gemma-4 31B is
        // dense; sharding the GeGLU MLP would double the TP win but
        // requires NVFP4 byte-slicing in `quantized_any`-loaded paths
        // (deferred — leaves Gemma-4 functionally correct under TP
        // with extra MLP memory per rank).
        true
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        loader_a::load_layers_impl(store, config, gpu, layer_kv_dtypes)
    }

    fn load_embedding(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        loader_b::load_embedding_impl(store, config)
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        loader_b::load_final_norm_impl(store, config)
    }

    fn load_lm_head(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        loader_b::load_lm_head_impl(store, config)
    }

    fn load_mtp_weights(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        loader_b::load_mtp_weights_impl(store, config, gpu)
    }

    fn kv_layer_dims(&self, config: &ModelConfig) -> Vec<(usize, usize)> {
        loader_b::kv_layer_dims_impl(config)
    }
}
