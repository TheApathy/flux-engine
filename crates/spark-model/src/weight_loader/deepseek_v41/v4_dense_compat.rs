// SPDX-License-Identifier: AGPL-3.0-only

//! Standalone embedding, final-norm, and `lm_head` loaders for DeepSeek-V4.1.
//!
//! [`super::DeepSeekV41WeightLoader`] delegates its three non-expert dense weights here.
//! Each loader probes common flattened and standard Hugging Face tensor spellings and
//! handles tied embeddings.

use anyhow::{Context, Result};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

use crate::weight_map::{DenseWeight, dense, dense_auto};

pub(crate) fn load_embedding(
    store: &WeightStore,
    _config: &ModelConfig,
    _gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    // Flattened naming first, then standard HF names.
    if let Ok(w) = dense(store, "embed.weight") {
        return Ok(w);
    }
    if let Ok(w) = dense(store, "model.embed_tokens.weight") {
        return Ok(w);
    }
    dense(store, "embed_tokens.weight")
        .context("DeepSeek-V4.1: no embedding tensor found (tried embed.weight, model.embed_tokens.weight, embed_tokens.weight)")
}

pub(crate) fn load_final_norm(
    store: &WeightStore,
    _config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    // DeepSeek-V4.1 ships vanilla RMSNorm weights (scale = weight). Load them
    // verbatim for `dsv41_fwd::dsv41_rmsnorm`.
    if let Ok(w) = dense_auto(store, "norm.weight", gpu) {
        return Ok(w);
    }
    if let Ok(w) = dense_auto(store, "model.norm.weight", gpu) {
        return Ok(w);
    }
    dense_auto(store, "final_norm.weight", gpu)
        .context("DeepSeek-V4.1: no final norm tensor found (tried norm.weight, model.norm.weight, final_norm.weight)")
}

pub(crate) fn load_lm_head(
    store: &WeightStore,
    config: &ModelConfig,
    _gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    // Try standard HF name first
    if store.contains("lm_head.weight") {
        return dense(store, "lm_head.weight");
    }
    // Flattened / consolidated checkpoints
    if store.contains("output.weight") {
        return dense(store, "output.weight");
    }
    if store.contains("head.weight") {
        return dense(store, "head.weight");
    }
    // Tied embeddings: either config says so, or no separate head exists
    if config.tie_word_embeddings
        || store.contains("embed.weight")
        || store.contains("model.embed_tokens.weight")
        || store.contains("embed_tokens.weight")
    {
        // Tied: reuse the embedding tensor via the same lookup order as `load_embedding`.
        if let Ok(w) = dense(store, "embed.weight") {
            return Ok(w);
        }
        if let Ok(w) = dense(store, "model.embed_tokens.weight") {
            return Ok(w);
        }
        return dense(store, "embed_tokens.weight")
            .context("DeepSeek-V4.1: tied lm_head — no embedding tensor found");
    }
    anyhow::bail!(
        "DeepSeek-V4.1: lm_head not found (tried lm_head.weight, output.weight, head.weight, and tied embeddings)"
    )
}
