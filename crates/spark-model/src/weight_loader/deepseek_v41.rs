// SPDX-License-Identifier: AGPL-3.0-only

//! Weight loader for **DeepSeek-V4.1-Flash-Next**.
//!
//! This loader deliberately **fails loudly for unimplemented [`ModelWeightLoader`] entry
//! points** rather than falling back to an EXL3-based loader. DeepSeek-V4.1's routed
//! experts use **CB3 3-bit** packing (not EXL3 2-bit), so decoding CB3 bytes as EXL3
//! would silently produce plausible garbage.
//!
//! [`v4_dense_compat`] carries the dense non-expert loaders (`load_embedding`,
//! `load_final_norm`, `load_lm_head`), which probe common flattened and standard
//! Hugging Face tensor spellings.
//!
//! ## What IS implemented and tested
//! - Embedding, final norm, and `lm_head`: `embed.weight` and `norm.weight` are the first
//!   spellings probed by [`v4_dense_compat`], while `head.weight` is probed third (after
//!   `lm_head.weight` and `output.weight`, both verified absent in V4.1).
//! - [`v41_key`], the weight-key join, which handles V4.1's empty prefix (see below).
//! - The expert mapping, in [`atlas_core::config::ExpertPack`].
//! - Reading CB3 expert payload, in `spark_runtime::weights::deepseek_v41_pack`, and GPU
//!   residency in [`cb3_arena`].
//! - End-to-end serving via the standalone `crate::model::dsv41::Dsv41Model`
//!   ([`forward::V41Forward`]), including sparse attention, engram gather, CB3 routed MoE,
//!   and DSpark MTP drafting ([`mtp::DsparkWeights`] / [`dspark::Dspark`]).
//!
//! ## Legacy [`ModelWeightLoader`] / [`TransformerLayer`] status
//! - [`ModelWeightLoader::load_layers`] builds all 40 [`layer::DeepSeekV41Layer`] stubs
//!   and makes the CB3 expert pack resident, but those stubs refuse `TransformerLayer`
//!   forward calls because serving runs through standalone `crate::model::dsv41::Dsv41Model`.
//! - [`ModelWeightLoader::load_mtp_weights`] still refuses loudly; DSpark loads the three
//!   `mtp.{0,1,2}` draft blocks directly via [`mtp::DsparkWeights::load`].

pub mod attn_block;
pub mod cb3_arena;
pub mod device_allocs;
pub mod forward;
pub mod fwd;
pub mod image_splice;
pub mod indexer;
pub mod layer;
mod load_layers;
pub use load_layers::resolve_model_dir;
pub mod dspark;
pub mod dspark_adapt;
pub mod dspark_sampling;
pub mod moe;
pub mod moe_decode;
pub mod moe_forward;
pub mod mtp;
pub mod mtp_moe_decode;
pub mod ops;
pub mod routing;
pub mod seams;
mod v4_dense_compat;

use anyhow::{Result, bail};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;

use super::{DenseWeight, ModelWeightLoader, MtpWeights, WeightStore};
use crate::layer::TransformerLayer;
use spark_runtime::kv_cache::KvCacheDtype;

/// Join a weight-key prefix to a suffix, tolerating an EMPTY prefix.
///
/// **V4.1's prefix is legitimately empty.** Zero of its 3925 index keys carry the `model.`
/// prefix used by standard HF layouts — they are bare (`embed.weight`,
/// `layers.0.attn_norm.weight`). The naive `format!("{prefix}.{suffix}")` therefore yields
/// a LEADING DOT and every lookup misses.
///
/// Unlike loaders that substitute a default prefix when the configured one is empty,
/// here empty is the correct and final answer, so we omit the separator rather than
/// inventing a prefix.
pub fn v41_key(prefix: &str, suffix: &str) -> String {
    if prefix.is_empty() {
        suffix.to_string()
    } else {
        format!("{prefix}.{suffix}")
    }
}

/// Tensors the SERVING weight store must never load for DeepSeek-V4.1.
///
/// - `layers.{1,14}.engram.embed.*`: two ~95 GB tables. They cannot fit next to the expert
///   arena on a 119.7 GB box and are row-gathered from NVMe by the engram lane. Uploading them
///   would over-allocate unified memory, which takes the HOST down, not the process.
/// - `mtp.*`: the three DSpark drafter blocks (~7.9 GB incl. 7.2 GB of FP4 experts), loaded ONLY
///   when `ATLAS_DSV41_DSPARK=1` (see `mtp::DsparkWeights`), so the drafter costs nothing when
///   speculation is off.
pub fn skip_tensor_for_serving(name: &str) -> bool {
    name.contains(".engram.embed.") || (name.starts_with("mtp.") && !dspark_enabled())
}

/// `ATLAS_DSV41_DSPARK=1`: load the DSpark drafter (speculation stays OFF by default until its
/// end-to-end gate passes).
pub fn dspark_enabled() -> bool {
    std::env::var("ATLAS_DSV41_DSPARK").as_deref() == Ok("1")
}

/// DeepSeek-V4.1-Flash-Next weight loader.
pub struct DeepSeekV41WeightLoader;

/// One message, for the [`ModelWeightLoader`] entry points that are intentionally unimplemented.
///
/// Since `load_layers` landed this is only [`ModelWeightLoader::load_mtp_weights`], which
/// stays unimplemented on the generic trait because DSpark loads the three `mtp.{0,1,2}`
/// draft modules directly via [`mtp::DsparkWeights::load`] in `crate::model::dsv41::Dsv41Model`.
fn v41_unimplemented(what: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "DeepSeek-V4.1 {what} is not implemented. What DOES work: the config parses, the \
         384 -> 154 -> 124 expert mapping resolves, CB3 bytes are read and decoded \
         (kernels/gb10/deepseek-v4.1/cb3/, bit-exact against an independent reference), the \
         expert pack is made RESIDENT, and `load_layers` builds all 40 layers. This is a \
         deliberate hard stop, NOT a fallback to an EXL3-based loader that would decode \
         V4.1's CB3 bytes into plausible garbage."
    )
}

impl ModelWeightLoader for DeepSeekV41WeightLoader {
    /// Declared explicitly, as the trait requires. V4.1 uses `num_key_value_heads = 1`
    /// (MQA), so head-parallel sharding is impossible: 1 is not divisible by a tp size
    /// greater than 1. This is a property of the architecture, not an unfinished feature.
    fn supports_tp(&self) -> bool {
        false
    }

    /// Build all 40 layers and make the CB3 expert pack RESIDENT.
    ///
    /// The pack is parsed, residency-checked against the real budget and uploaded once
    /// (71.7 GB decimal / 66.8 GiB at the served `packed_keep = 124`), the dense per-layer
    /// tensors resolve, and the [`layer::DeepSeekV41Layer`] stubs are constructed.
    ///
    /// The returned stubs refuse [`TransformerLayer`] forward calls; end-to-end serving
    /// executes through standalone `crate::model::dsv41::Dsv41Model` ([`forward::V41Forward`]).
    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        load_layers::load_all_layers(store, config, gpu, layer_kv_dtypes)
    }

    // ---------------------------------------------------------------------------------
    // Dense non-expert tensors delegate to `v4_dense_compat`.
    //
    // `embed.weight` and `norm.weight` are the first spellings probed, while
    // `head.weight` is probed third (after `lm_head.weight` and `output.weight`, both
    // verified absent in V4.1). Verified against the shipped index: `embed.weight`,
    // `norm.weight`, and `head.weight` are all present, while `lm_head.weight`,
    // `output.weight`, and `model.embed_tokens.weight` are absent.
    // ---------------------------------------------------------------------------------

    fn load_embedding(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        v4_dense_compat::load_embedding(store, config, gpu)
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        // V4.1 ships vanilla RMSNorm weights (scale = weight); `V41Forward` /
        // `dsv41_fwd::dsv41_rmsnorm` consumes them directly, so loading them verbatim
        // via `v4_dense_compat::load_final_norm` is exact.
        v4_dense_compat::load_final_norm(store, config, gpu)
    }

    fn load_lm_head(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        v4_dense_compat::load_lm_head(store, config, gpu)
    }

    fn load_mtp_weights(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        // NOT `Ok(None)`. None means "this checkpoint has no MTP", which is FALSE here:
        // V4.1 declares num_nextn_predict_layers = 3 and ships 2401 `mtp.*` tensors.
        // Returning None would silently disable speculation and look like a config choice.
        bail!(v41_unimplemented(
            "ModelWeightLoader::load_mtp_weights (use mtp::DsparkWeights via Dsv41Model)"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The empty prefix must not produce a leading dot. This is the trap that would have
    /// made every V4.1 weight lookup miss.
    #[test]
    fn v41_key_omits_the_separator_for_an_empty_prefix() {
        assert_eq!(v41_key("", "embed.weight"), "embed.weight");
        assert_eq!(
            v41_key("", "layers.0.attn_norm.weight"),
            "layers.0.attn_norm.weight"
        );
        assert!(!v41_key("", "norm.weight").starts_with('.'));
    }

    /// And a non-empty prefix still joins normally.
    #[test]
    fn v41_key_joins_a_non_empty_prefix_normally() {
        assert_eq!(v41_key("model", "embed.weight"), "model.embed.weight");
    }

    /// Keys built for a real checkpoint must exist in its index when
    /// `FLUX_TEST_DSV41_MODEL_DIR` is set. Skips when unset.
    #[test]
    fn v41_keys_resolve_against_the_shipped_weight_index() {
        let Some(model_dir) = std::env::var_os("FLUX_TEST_DSV41_MODEL_DIR") else {
            return;
        };
        let index_path = std::path::PathBuf::from(model_dir).join("model.safetensors.index.json");
        let Ok(raw) = std::fs::read_to_string(&index_path) else {
            eprintln!("skipping: {} not present", index_path.display());
            return;
        };
        let index: serde_json::Value = serde_json::from_str(&raw).expect("index is JSON");
        let map = index["weight_map"].as_object().expect("weight_map");

        // V4.1's prefix, as parse_deepseek_v41 sets it.
        for suffix in [
            "embed.weight",
            "head.weight",
            "norm.weight",
            "layers.0.attn_norm.weight",
            "layers.39.ffn_norm.weight",
        ] {
            let key = v41_key("", suffix);
            assert!(
                map.contains_key(&key),
                "key {key} is not in the shipped weight index"
            );
        }

        // The `model.` prefixed spelling must NOT resolve in V4.1.
        assert!(
            !map.contains_key("model.embed.weight"),
            "V4.1 must not carry the `model.` prefix"
        );

        // PINS THE DELEGATION. `load_embedding`, `load_final_norm`, and `load_lm_head`
        // delegate to `v4_dense_compat`, which probes several spellings (`embed.weight`
        // and `norm.weight` first; `head.weight` third after `lm_head.weight` and
        // `output.weight`). Assert both halves: the names V4.1 uses are present, and the
        // alternatives those probes would otherwise hit or fall through to are absent.
        for present in ["embed.weight", "norm.weight", "head.weight"] {
            assert!(map.contains_key(present), "{present} must be present");
        }
        for absent in [
            "lm_head.weight",
            "model.embed_tokens.weight",
            "embed_tokens.weight",
            "model.norm.weight",
            "final_norm.weight",
            "output.weight",
        ] {
            assert!(
                !map.contains_key(absent),
                "{absent} is absent in V4.1; if it appears, the probe ORDER decides which \
                 tensor wins and the delegation must be re-examined"
            );
        }
    }

    /// The dispatch must hand V4.1 its OWN loader, not an EXL3-based one.
    ///
    /// Asserted behaviourally rather than by reading the match arm: build a real V4.1
    /// config, run the factory, and invoke `load_mtp_weights` to check that the loader
    /// refuses with a message naming CB3.
    #[test]
    fn the_factory_gives_v41_its_own_loader() {
        let Some(model_dir) = std::env::var_os("FLUX_TEST_DSV41_MODEL_DIR") else {
            return;
        };
        let config_path = std::path::PathBuf::from(model_dir).join("config.json");
        let Ok(json) = std::fs::read_to_string(&config_path) else {
            eprintln!("skipping: {} not present", config_path.display());
            return;
        };
        let config = atlas_core::config::parse_config(&json).expect("V4.1 config must parse");
        assert_eq!(config.model_type, "deepseek_v41");

        let loader = crate::factory::loader_for_config(&config)
            .expect("deepseek_v41 must dispatch to a loader, not fall through to bail");
        assert!(
            !loader.supports_tp(),
            "V4.1 is MQA (num_key_value_heads = 1); head-parallel TP is impossible"
        );
        let empty_store = WeightStore::empty();
        let mock_gpu = spark_runtime::gpu::mock::MockGpuBackend::new();
        let Err(err) = loader.load_mtp_weights(&empty_store, &config, &mock_gpu) else {
            panic!("V4.1 ModelWeightLoader::load_mtp_weights must refuse loudly");
        };
        let msg = err.to_string();
        assert!(
            msg.contains("CB3"),
            "refusal should identify the DeepSeek-V4.1 CB3 loader, got: {msg}"
        );
    }

    /// Every entry point fails LOUDLY rather than returning an empty success.
    ///
    /// `load_mtp_weights` is the one that matters: `Ok(None)` is a legal value meaning
    /// "no MTP in this checkpoint", which is false for V4.1 (3 modules, 2401 mtp.* tensors)
    /// and would silently disable speculation while looking deliberate.
    #[test]
    fn unimplemented_entry_points_name_the_seam() {
        let message = v41_unimplemented("MTP loading").to_string();
        assert!(message.contains("deliberate hard stop"), "{message}");
        assert!(message.contains("plausible garbage"), "{message}");
        // It must say what DOES work, so the reader knows where the boundary is — and
        // it must NOT still claim residency is missing, which was true and no longer is.
        assert!(message.contains("RESIDENT"), "{message}");
        assert!(
            !message.contains("cannot yet make the pack resident"),
            "the message is stale: residency landed. {message}"
        );
    }
}
