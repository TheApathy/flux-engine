// SPDX-License-Identifier: AGPL-3.0-only

//! Load-time sizing of the Qwen4 prefill scratch that used to grow (and free
//! its old block) inside the first request large enough to need it: the
//! FlashInfer GDN bridge buffers, the QSA sparse-core scratch and the E4M3 QSA
//! query scratch (`ATLAS_QWEN4_PREFILL_QSA_FP8`, no-op when off). A free there
//! runs mid-request while earlier kernels may still be queued (the GB10
//! free-under-traffic class); all three now retire instead of freeing, and this
//! reserve makes the engine's largest chunk fit from the start. Runs before
//! the KV cache is sized from free memory, so the pool accounts for it.

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::PagedKvCache;

pub(super) fn maybe_reserve_qwen4_prefill_scratch(
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    max_batch_tokens: usize,
    max_seq_len: usize,
) -> Result<()> {
    if !config.is_qwen4_exp() || max_batch_tokens == 0 {
        return Ok(());
    }
    crate::layers::qwen3_ssm::reserve_qwen4_gdn_fi_bridge(max_batch_tokens)?;
    crate::layers::qwen4_qsa::prefill_sparse::reserve(gpu, max_batch_tokens, max_seq_len)?;
    crate::layers::qwen4_qsa::prefill_fp8::reserve_q8(gpu, max_batch_tokens)?;
    Ok(())
}

/// Materialize the selected E4M3 QSA shadow against the target cache only.
///
/// This must be called after the target [`PagedKvCache`] is constructed. The
/// native-MTP proposer owns a distinct one-layer cache constructed earlier;
/// passing the target attention-layer count makes accidental MTP use fail
/// closed in `reserve_target_shadows`.
pub(super) fn maybe_reserve_qwen4_target_shadows(
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    kv_cache: &PagedKvCache,
) -> Result<()> {
    if !config.is_qwen4_exp() || !config.qwen4_qsa {
        return Ok(());
    }
    crate::layers::qwen4_qsa::prefill_fp8::reserve_target_shadows(
        gpu,
        kv_cache,
        config.num_attention_layers(),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use atlas_core::config::ModelConfig;
    use spark_runtime::gpu::mock::MockGpuBackend;
    use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};

    use super::maybe_reserve_qwen4_target_shadows;

    #[test]
    fn target_shadow_reserve_follows_mtp_and_target_cache_but_precedes_model() {
        let source = include_str!("build.rs");
        let mtp = source
            .find("Qwen4MtpHead::new")
            .expect("native-MTP construction site");
        let target_cache = source
            .find("|| PagedKvCache::new(kv_config, num_kv_blocks, gpu.as_ref())")
            .expect("target paged-KV construction site");
        let reserve = source
            .find("maybe_reserve_qwen4_target_shadows(&config, gpu.as_ref(), &kv_cache)")
            .expect("target QSA shadow reserve site");
        let model = source
            .find("let mut model = TransformerModel::new(")
            .expect("target model construction site");

        assert!(mtp < target_cache, "native MTP must be constructed first");
        assert!(
            target_cache < reserve,
            "reserve needs target cache geometry"
        );
        assert!(reserve < model, "shadows must materialize before readiness");
        assert_eq!(
            source
                .matches("maybe_reserve_qwen4_target_shadows(")
                .count(),
            1,
            "only the target-cache call site may reserve QSA shadows"
        );

        for auxiliary in [
            include_str!("../layers/qwen4_mtp.rs"),
            include_str!("../layers/qwen4_mtp/forward.rs"),
            include_str!("../layers/qwen4_mtp/graph_cache.rs"),
            include_str!("../layers/qwen4_mtp/graphed_chain.rs"),
            include_str!("../layers/qwen4_mtp/gumbel_draft.rs"),
            include_str!("../layers/qwen4_mtp/init.rs"),
            include_str!("../layers/qwen4_mtp/policy.rs"),
            include_str!("../layers/qwen4_mtp/proposer_impl.rs"),
            include_str!("../layers/qwen4_mtp/ragged_propose.rs"),
            include_str!("../layers/qwen4_mtp/state.rs"),
            include_str!("../layers/mtp_head/new.rs"),
        ] {
            assert!(
                !auxiliary.contains("reserve_target_shadows"),
                "an auxiliary proposer must never reserve target QSA shadows"
            );
        }
    }

    #[test]
    fn qwen4_without_qsa_never_reserves_fp8_shadows() {
        let gpu = MockGpuBackend::new();
        let cache = PagedKvCache::new(
            KvCacheConfig {
                block_size: 16,
                num_kv_heads: 2,
                head_dim: 256,
                num_layers: 12,
                dtype: KvCacheDtype::Bf16,
                layer_dtypes: vec![],
                layer_dims: vec![],
                cache_blocks_per_seq: None,
            },
            2,
            &gpu,
        )
        .unwrap();
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "qwen4_exp".to_string();
        config.qwen4_qsa = false;
        let allocations = gpu.alloc_count();
        let memsets = gpu.memset_count();

        maybe_reserve_qwen4_target_shadows(&config, &gpu, &cache).unwrap();

        assert_eq!(gpu.alloc_count(), allocations);
        assert_eq!(gpu.memset_count(), memsets);
    }
}
