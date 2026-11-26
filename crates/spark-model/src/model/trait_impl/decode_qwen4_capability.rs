// SPDX-License-Identifier: AGPL-3.0-only

//! Native Qwen4 frame capability per width.
//!
//! Every state-independent check of the native preflight
//! (`decode_qwen4_native_batch_dispatch` and each layer's
//! `preflight_qwen4_native_target`) is evaluated here, before routing, and
//! reported as a typed [`NativeGap`]. Under the default plan a width with a
//! gap serializes with that reason; with `ATLAS_QWEN4_NATIVE_BATCH=1` the load
//! refuses it. After this gate the dispatch can fail only on an invariant
//! (aliased slots, state or KV blocks, poisoned PLE, malformed tokens), on KV
//! exhaustion, or on a GPU error, none of which serial decode would survive.
//!
//! The facts are fixed for the process, so each width is computed once.

use atlas_core::config::LayerType;
use spark_runtime::kv_cache::KvCacheDtype;

use super::super::types::TransformerModel;
use crate::layer::ForwardContext;
use crate::qwen4_batch_policy::{NativeGap, SerialReason as R};

type Capability = std::result::Result<(), NativeGap>;

fn require(ok: bool, reason: R, detail: &str) -> Capability {
    if ok {
        Ok(())
    } else {
        Err(NativeGap::new(reason, detail))
    }
}

impl TransformerModel {
    /// Whether the native frame can run C`rows` in this process, cached per
    /// width. Widths outside C2-C256 (`QWEN4_NATIVE_MAX_ROWS`) never reach it:
    /// load admission refuses them.
    pub(super) fn qwen4_native_capability(&self, rows: usize) -> Capability {
        match self.qwen4_native_capability_cache.get(rows) {
            Some(cell) if rows >= 2 => cell
                .get_or_init(|| self.qwen4_native_capability_uncached(rows))
                .clone(),
            _ => Err(NativeGap::new(
                R::ArenaCapacity,
                format!(
                    "native Qwen4 batching supports C2-C{} only, got C{rows}",
                    crate::layers::QWEN4_NATIVE_MAX_ROWS
                ),
            )),
        }
    }

    /// Load-time report: every width in C2..=min(`max_batch_size`, `QWEN4_NATIVE_MAX_ROWS`)
    /// the native frame cannot run, with its gap. Empty when every width runs.
    pub(super) fn qwen4_native_capability_report_dispatch(
        &self,
        max_batch_size: usize,
    ) -> Vec<(usize, NativeGap)> {
        (2..=max_batch_size.min(crate::layers::QWEN4_NATIVE_MAX_ROWS))
            .filter_map(|rows| {
                self.qwen4_native_capability(rows)
                    .err()
                    .map(|gap| (rows, gap))
            })
            .collect()
    }

    fn qwen4_native_capability_uncached(&self, rows: usize) -> Capability {
        let config = &self.config;
        // Model-level checks, in the dispatch's order.
        require(
            config.is_qwen4_exp()
                && config.hc_count == 4
                && config.hidden_size == 2_560
                && config.residual_width() == 10_240
                && !config.use_fp32_residual(),
            R::ModelGeometry,
            "native Qwen4 decode requires the canonical BF16 Flash-Next geometry",
        )?;
        let topology_ok = (config.tp_world_size == 1
            && config.ep_world_size == 1
            && ((config.pp_world_size == 1 && self.comm.is_none())
                || (config.pp_world_size >= 2 && self.comm.is_some())))
            || ((2..=4).contains(&config.tp_world_size)
                && (config.ep_world_size <= 1 || config.ep_world_size == config.tp_world_size)
                && config.pp_world_size == 1
                && self.comm.is_some());
        require(
            topology_ok,
            R::Topology,
            "native Qwen4 decode requires TP1..TP4 (+ matching EP) with PP1, or PP>=2 with TP1/EP1",
        )?;
        let target_only =
            !self.self_speculative && (!self.any_proposer() || self.speculation_forced_disabled);
        let native_mtp_bootstrap = self.qwen4_native_mtp_bootstrap_admitted();
        require(
            target_only || native_mtp_bootstrap,
            R::TargetOnly,
            "native Qwen4 decode is target-only; disable speculative decoding",
        )?;
        require(
            !crate::layers::qwen3_ssm::ssm_h_fp16_enabled(),
            R::SsmHFp16,
            "native Qwen4 decode requires FP32 recurrent state; unset ATLAS_SSM_H_FP16",
        )?;
        // Read-only historical prefix-cache sharing is safe on resident non-HSS
        // KV: dispatch verifies every writable decode block is exclusively owned.
        let prefix_cache_admitted = !self.prefix_cache.is_active()
            || self.kv_cache.lock().config().cache_blocks_per_seq.is_none();
        require(
            prefix_cache_admitted,
            R::PrefixCache,
            "native Qwen4 decode does not admit prefix-cache ownership",
        )?;
        require(
            self.layers.len() == config.num_hidden_layers
                && (0..self.layers.len()).all(|index| {
                    matches!(
                        config.layer_type(index),
                        LayerType::FullAttention | LayerType::LinearAttention
                    )
                }),
            R::ModelGeometry,
            "native Qwen4 decode found an unsupported or incomplete layer schedule",
        )?;
        require(
            self.qwen4_ple.is_none()
                || (self.layers.len() > 1 && config.ple_layer_ids.as_slice() == [2]),
            R::PleRows,
            "native Qwen4 PLE requires the canonical layer-2 injection boundary",
        )?;
        require(
            !self.use_fp32_logits,
            R::Fp32Logits,
            "native Qwen4 decode requires full-vocabulary BF16 row-major logits",
        )?;
        require(
            self.verify_lmhead_vocab() as usize == config.vocab_size,
            R::LmHeadVocabCap,
            "native Qwen4 decode requires full-vocabulary logits; unset ATLAS_TARGET_LMHEAD_VOCAB",
        )?;
        require(
            self.lm_head_nvfp4.is_some(),
            R::LmHeadNotNvfp4,
            "native Qwen4 decode requires an ordinary NVFP4 LM head",
        )?;
        require(
            !(super::super::impl_a3::lm_head_tc_enabled()
                && self.lm_head_nvfp4_t.is_some()
                && self.w4a16_gemm_t_m32_n64_kernel.0 != 0),
            R::LmHeadTensorCore,
            "native Qwen4 decode requires the exact LM-head route; unset ATLAS_LM_HEAD_TC",
        )?;
        match self.w4a16_exact_lm_head_kernels.route_for_rows(rows as u32) {
            None => {
                return Err(NativeGap::new(
                    R::LmHeadRowTier,
                    format!("native Qwen4 decode has no exact/K1 LM-head row tier for C{rows}"),
                ));
            }
            Some(crate::layers::ops::ExactLmHeadRoute::SerialK1(_)) => require(
                crate::layers::ops::use_gemv_sw(self.gemv_sw, self.w4a16_gemv_sw_kernel)
                    || self.w4a16_gemv_kernel.0 != 0,
                R::LmHeadRowTier,
                "native Qwen4 decode is missing the K1 LM-head fallback kernel",
            )?,
            Some(_) => {}
        }
        let final_mixer = self.qwen4_final_mixer.as_ref().ok_or_else(|| {
            NativeGap::new(
                R::FinalMixer,
                "native Qwen4 decode is missing the final hyperconnection mixer",
            )
        })?;
        require(
            final_mixer.residual_width() == config.residual_width() && final_mixer.inject.is_none(),
            R::FinalMixer,
            "native Qwen4 final mixer geometry is not target-decode compatible",
        )?;
        final_mixer
            .preflight_qwen4_native_batched(rows, false)
            .map_err(|error| NativeGap::new(R::FinalMixer, error))?;
        self.qwen4_native_arena_capability(rows)?;
        self.preflight_embed_qwen4_batch(rows)
            .map_err(|error| NativeGap::new(R::EmbedBatch, error))?;
        if let Some(ple) = &self.qwen4_ple {
            ple.native_decode_capability(rows, config.residual_width() * 2)
                .map_err(|error| NativeGap::new(R::PleRows, error))?;
        }
        {
            let kv_cache = self.kv_cache.lock();
            require(
                kv_cache.config().cache_blocks_per_seq.is_none()
                    && (0..kv_cache.num_layers()).all(|layer| {
                        matches!(
                            kv_cache.dtype_for_layer(layer),
                            KvCacheDtype::Bf16 | KvCacheDtype::Nvfp4
                        )
                    }),
                R::KvLayout,
                "native Qwen4 decode requires resident BF16 KV (or NVFP4) without high-speed swap",
            )?;
        }
        // Layer-level checks: the same context the dispatch's preflight uses.
        let ctx = ForwardContext {
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            attn_metadata: None,
            profile: false,
            comm: None,
            graph_capture: false,
            ddtree_parent_ids_dev: None,
            tree_aware_attn: None,
            ssm_multi_seq_ptr_table_override: None,
            self_spec_sparse_draft: None,
            ffn_defer: None,
        };
        for (index, layer) in self.layers.iter().enumerate() {
            layer
                .qwen4_native_capability(rows, &ctx)
                .map_err(|gap| gap.at_layer(index))?;
        }
        Ok(())
    }

    /// The dispatch's model-arena extents for C`rows`.
    fn qwen4_native_arena_capability(&self, rows: usize) -> Capability {
        let overflow = |detail: &str| NativeGap::new(R::ArenaCapacity, detail);
        let config = &self.config;
        let hyper_bytes = rows
            .checked_mul(config.residual_width())
            .and_then(|items| items.checked_mul(2))
            .ok_or_else(|| overflow("native Qwen4 hyper-row extent overflow"))?;
        let core_bytes = rows
            .checked_mul(config.hidden_size)
            .and_then(|items| items.checked_mul(2))
            .ok_or_else(|| overflow("native Qwen4 core-row extent overflow"))?;
        let logits_bytes = rows
            .checked_mul(config.vocab_size)
            .and_then(|items| items.checked_mul(2))
            .ok_or_else(|| overflow("native Qwen4 logits extent overflow"))?;
        let hyper_scratch = rows
            .checked_mul(config.hc_lowrank)
            .and_then(|elements| elements.checked_mul(2))
            .and_then(|down_bytes| {
                rows.checked_mul(config.hc_count)
                    .and_then(|elements| elements.checked_mul(2))
                    .and_then(|inject_bytes| down_bytes.checked_add(inject_bytes))
            })
            .ok_or_else(|| overflow("native Qwen4 hyper scratch extent overflow"))?;
        let arenas = self.buffers.sizes();
        let metadata_bytes = crate::layer::STEP_META_SCRATCH_OFFSET
            .checked_add(
                self.buffers
                    .decode_meta()
                    .meta_bytes(self.max_blocks_per_seq as usize),
            )
            .ok_or_else(|| overflow("native Qwen4 metadata scratch extent overflow"))?;
        require(
            rows <= self.buffers.decode_meta().rows()
                && rows <= self.buffers.max_batch_tokens()
                && hyper_bytes <= arenas.hidden_states
                && hyper_bytes <= arenas.residual
                && hyper_bytes <= arenas.qkv_output
                && core_bytes <= arenas.norm_output
                && core_bytes <= arenas.moe_output
                && logits_bytes <= arenas.logits
                && hyper_scratch <= arenas.ssm_ba
                && metadata_bytes <= arenas.scratch,
            R::ArenaCapacity,
            "native Qwen4 active rows exceed the serving arenas",
        )
    }

    /// Load-time probe of lane 03's per-row QSA path for a native plan with
    /// QSA rows: every full-attention layer must admit one indexed row. Only
    /// the state-independent checks can fail on this probe (indexer present,
    /// BF16 resident KV, side-cache geometry, graph-safe kernels).
    pub(super) fn qwen4_native_qsa_capability(&self) -> Capability {
        use crate::layers::qwen4_qsa::native_batch::NativeAttnRoute;
        let kv_cache = self.kv_cache.lock();
        for (index, layer) in self.layers.iter().enumerate() {
            if matches!(self.config.layer_type(index), LayerType::FullAttention) {
                layer
                    .preflight_qwen4_native_target_qsa(
                        &[NativeAttnRoute::IndexedDense],
                        &[0],
                        &kv_cache,
                    )
                    .map_err(|error| NativeGap::new(R::QsaRequired, error).at_layer(index))?;
            }
        }
        Ok(())
    }
}
