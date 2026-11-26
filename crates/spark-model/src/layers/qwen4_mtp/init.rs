// SPDX-License-Identifier: AGPL-3.0-only

//! `Qwen4MtpHead` struct definition and constructor.

use anyhow::{Result, ensure};
use parking_lot::Mutex;
use spark_runtime::gpu::{GpuBackend, KernelHandle};
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};
use spark_runtime::weights::WeightStore;

use super::graph_cache::{MTP_PINNED_H2D_RING_BYTES, MtpGraphCache, MtpPinnedH2dRing};
use crate::layer::TransformerLayer;
use crate::layers::{Qwen4HyperConnection, ops, qwen4_mtp_gate};
use crate::weight_map::{
    DenseWeight, QuantizedWeight, dense_auto, dense_f32_safe, quantize_to_nvfp4_cached,
};

pub struct Qwen4MtpHead {
    pub(super) pre_fc_norm_embedding: DenseWeight,
    pub(super) pre_fc_norm_hidden: DenseWeight,
    pub(super) fc_embedding: QuantizedWeight,
    pub(super) fc_hidden: QuantizedWeight,
    pub(super) layer: Box<dyn TransformerLayer>,
    pub(super) final_mixer: Qwen4HyperConnection,
    pub(super) embed_tokens: DenseWeight,
    pub(super) lm_head_nvfp4: QuantizedWeight,
    pub(super) mtp_vocab_size: u32,
    pub(super) kv_cache: Mutex<PagedKvCache>,
    /// Reusable host staging buffer for `propose_batch` and `propose_chain`
    /// metadata packing, eliminating `0.8–2.0 MiB/step` heap churn.
    pub(super) packed_meta_scratch: Mutex<Vec<u8>>,
    /// Dedicated page-locked ring for async stream-ordered metadata H2D copies.
    pub(super) pinned_h2d_ring: Mutex<Option<MtpPinnedH2dRing>>,
    /// Per-`(batch_size, step_count, want_prob, ...)` CUDA graph cache for
    /// uniform batched MTP proposer chains (`ATLAS_QWEN4_MTP_GRAPH`).
    pub(super) graph_cache: Mutex<MtpGraphCache>,
    pub(super) rms_norm_k: KernelHandle,
    pub(super) pre_fc_dual_norm_k: KernelHandle,
    pub(super) embed_dual_rms_norm_fused_k: KernelHandle,
    pub(super) embed_dual_rms_norm_rows_packed_k: KernelHandle,
    pub(super) fc_dual_broadcast_rows_k: KernelHandle,
    pub(super) w4a16_gemv_k: KernelHandle,
    pub(super) w4a16_gemv_sw_k: KernelHandle,
    pub(super) gemv_sw: bool,
    pub(super) w4a16_gemv_exact_m4_k: KernelHandle,
    pub(super) w4a16_exact_fc_kernels: ops::W4a16ExactLmHeadKernels,
    /// Opt-in (`ATLAS_QWEN4_MTP_LM_HEAD_TC=1`) `w4a16_gemm_pipe` kernel for A/B
    /// testing MTP LM head at `b >= 8` (default off in favor of coalesced exact GEMV).
    pub(super) w4a16_gemm_pipe_k: KernelHandle,
    pub(super) residual_add_k: KernelHandle,
    pub(super) residual_add_broadcast_hc_k: KernelHandle,
    pub(super) argmax_k: KernelHandle,
    pub(super) argmax_rows_k: KernelHandle,
    pub(super) argmax_prob_k: KernelHandle,
    pub(super) argmax_prob_batched_k: KernelHandle,
    pub(super) mtp_gumbel_argmax_k: KernelHandle,
    pub(super) embed_from_argmax_k: KernelHandle,
    /// The exact non-YaRN MRoPE symbol used by the inner attention layer.
    /// That layer probes it as optional, so image admission must preserve and
    /// check availability before any proposer device write.
    pub(super) mrope_kernel_available: bool,
}

impl Qwen4MtpHead {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        store: &WeightStore,
        config: &atlas_core::config::ModelConfig,
        layer: Box<dyn TransformerLayer>,
        final_mixer: Qwen4HyperConnection,
        embed_tokens: DenseWeight,
        lm_head_nvfp4: QuantizedWeight,
        gpu: &dyn GpuBackend,
        mtp_vocab_size: u32,
        max_seq_len: usize,
        max_batch_size: usize,
        // Token capacity of a shared target KV pool (`--kv-pool-tokens`);
        // `None` sizes the MTP cache per slot at max_seq_len as before.
        shared_target_pool_tokens: Option<usize>,
    ) -> Result<Self> {
        ensure!(
            config.is_qwen4_exp(),
            "Qwen4MtpHead requires qwen4_exp target"
        );
        let h = config.hidden_size;
        let stream = gpu.default_stream();
        let absmax = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
        let quantize = gpu
            .kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4_weight_rne")
            .or_else(|_| gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4"))?;
        let mrope_kernel_available = gpu
            .kernel("rope_mrope_interleaved", "rope_forward_mrope_interleaved")
            .is_ok();
        let fc_embedding_dense = dense_auto(store, "mtp.fc_embedding.weight", gpu)?;
        let fc_hidden_dense = dense_auto(store, "mtp.fc_hidden.weight", gpu)?;
        let fc_embedding = quantize_to_nvfp4_cached(
            &fc_embedding_dense,
            h,
            h,
            gpu,
            absmax,
            quantize,
            stream,
            "mtp.fc_embedding.nvfp4",
        )?;
        let fc_hidden = quantize_to_nvfp4_cached(
            &fc_hidden_dense,
            h,
            h,
            gpu,
            absmax,
            quantize,
            stream,
            "mtp.fc_hidden.nvfp4",
        )?;
        let kv_config = KvCacheConfig {
            block_size: 16,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            num_layers: 1,
            dtype: KvCacheDtype::Bf16,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
        };
        let num_blocks = crate::factory::kv_pool::mtp_kv_pool_blocks(
            max_seq_len,
            max_batch_size,
            kv_config.block_size,
            shared_target_pool_tokens,
        );
        let kv_cache = PagedKvCache::new(kv_config, num_blocks, gpu)?;
        let effective_vocab = if qwen4_mtp_gate::mtp_vocab_prune_selected() {
            (qwen4_mtp_gate::PRUNED_VOCAB_SIZE as usize).min(config.vocab_size)
        } else if mtp_vocab_size == 0 {
            config.vocab_size
        } else {
            (mtp_vocab_size as usize).min(config.vocab_size)
        };
        tracing::info!(
            hidden = h,
            residual = config.residual_width(),
            experts = config.num_experts,
            vocab = effective_vocab,
            "Qwen4 native MTP head constructed"
        );
        let w4a16_gemv_exact_m4_k = gpu.kernel("w4a16_gemv", "w4a16_gemv_batch_logits_exact_m4")?;
        let w4a16_exact_fc_kernels = ops::W4a16ExactLmHeadKernels::new(
            w4a16_gemv_exact_m4_k,
            crate::layers::try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_batch_logits_exact_m8"),
            crate::layers::try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_batch_logits_exact_m17"),
            crate::layers::try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_batch_logits_exact_m32"),
        )
        .with_rt2(
            crate::layers::try_kernel(gpu, "w4a16_gemv_rt", ops::ExactLmHeadTier::M4.symbol_rt2()),
            crate::layers::try_kernel(gpu, "w4a16_gemv_rt", ops::ExactLmHeadTier::M8.symbol_rt2()),
            crate::layers::try_kernel(gpu, "w4a16_gemv_rt", ops::ExactLmHeadTier::M17.symbol_rt2()),
            crate::layers::try_kernel(gpu, "w4a16_gemv_rt", ops::ExactLmHeadTier::M32.symbol_rt2()),
        )
        .with_rowpar(crate::layers::try_kernel(
            gpu,
            "w4a16_gemv",
            "w4a16_gemv_rows",
        ))
        .with_small_k_sw(
            crate::layers::try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_batch_logits_exact_sw_m4_o2"),
            crate::layers::try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_batch_logits_exact_sw_m8_o1"),
        )
        .with_small_k_sw_m10(crate::layers::try_kernel(
            gpu,
            "w4a16_gemv",
            "w4a16_gemv_batch_logits_exact_sw_m10_o1",
        ))
        .with_wide_n_sw4(crate::layers::try_kernel(
            gpu,
            "w4a16_gemv",
            "w4a16_gemv_batch_logits_exact_sw_m8_o4w4",
        ))
        .with_wide_n_sw4_m10(crate::layers::try_kernel(
            gpu,
            "w4a16_gemv",
            "w4a16_gemv_batch_logits_exact_sw_m10_o4w4",
        ))
        .with_wide_n_sw4_m15_m25(
            crate::layers::try_kernel(
                gpu,
                "w4a16_gemv",
                "w4a16_gemv_batch_logits_exact_sw_m15_o2w4",
            ),
            crate::layers::try_kernel(
                gpu,
                "w4a16_gemv",
                "w4a16_gemv_batch_logits_exact_sw_m25_o1w4",
            ),
        )
        .with_single_sweep_m64_m80_m128(
            crate::layers::try_kernel(gpu, "w4a16_gemv", ops::ExactSingleSweepTier::M64.symbol()),
            crate::layers::try_kernel(gpu, "w4a16_gemv", ops::ExactSingleSweepTier::M80.symbol()),
            crate::layers::try_kernel(gpu, "w4a16_gemv", ops::ExactSingleSweepTier::M128.symbol()),
        );
        Ok(Self {
            pre_fc_norm_embedding: dense_f32_safe(store, "mtp.pre_fc_norm_embedding.weight", gpu)?,
            pre_fc_norm_hidden: dense_f32_safe(store, "mtp.pre_fc_norm_hidden.weight", gpu)?,
            fc_embedding,
            fc_hidden,
            layer,
            final_mixer,
            embed_tokens,
            lm_head_nvfp4,
            mtp_vocab_size,
            kv_cache: Mutex::new(kv_cache),
            packed_meta_scratch: Mutex::new(Vec::new()),
            pinned_h2d_ring: Mutex::new(MtpPinnedH2dRing::new(gpu, MTP_PINNED_H2D_RING_BYTES)),
            graph_cache: Mutex::new(MtpGraphCache::default()),
            rms_norm_k: gpu.kernel("norm", "rms_norm")?,
            pre_fc_dual_norm_k: crate::layers::try_kernel(
                gpu,
                "norm",
                "qwen4_mtp_pre_fc_dual_rms_norm_rows",
            ),
            embed_dual_rms_norm_fused_k: crate::layers::try_kernel(
                gpu,
                "qwen4_mtp_wave2",
                "qwen4_mtp_embed_dual_rms_norm_fused",
            ),
            embed_dual_rms_norm_rows_packed_k: crate::layers::try_kernel(
                gpu,
                "qwen4_mtp_wave2",
                "qwen4_mtp_embed_dual_rms_norm_rows_packed",
            ),
            fc_dual_broadcast_rows_k: crate::layers::try_kernel(
                gpu,
                "qwen4_mtp_wave2",
                "qwen4_mtp_fc_dual_broadcast_rows",
            ),
            w4a16_gemv_k: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
            w4a16_gemv_sw_k: crate::layers::try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_sw"),
            gemv_sw: ops::gemv_sw_enabled(),
            w4a16_gemv_exact_m4_k,
            w4a16_exact_fc_kernels,
            w4a16_gemm_pipe_k: crate::layers::try_kernel(gpu, "w4a16", "w4a16_gemm_pipe"),
            residual_add_k: gpu.kernel("residual_add", "bf16_residual_add")?,
            residual_add_broadcast_hc_k: crate::layers::try_kernel(
                gpu,
                "residual_add",
                "bf16_residual_add_broadcast_hc",
            ),
            argmax_k: gpu.kernel("argmax", "argmax_bf16")?,
            argmax_rows_k: crate::layers::try_kernel(gpu, "argmax", "argmax_bf16_rows"),
            argmax_prob_k: crate::layers::try_kernel(gpu, "argmax", "argmax_prob_bf16"),
            argmax_prob_batched_k: crate::layers::try_kernel(
                gpu,
                "argmax",
                "argmax_prob_batched_bf16",
            ),
            mtp_gumbel_argmax_k: crate::layers::try_kernel(
                gpu,
                "sampler_topk_candidates",
                "mtp_gumbel_argmax_bf16",
            ),
            embed_from_argmax_k: gpu
                .kernel("embed_from_argmax", "embed_from_argmax")
                .unwrap_or(KernelHandle(0)),
            mrope_kernel_available,
        })
    }
}
