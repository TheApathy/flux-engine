// SPDX-License-Identifier: AGPL-3.0-only

//! Single-sequence Qwen4 decode with an eager PLE boundary and a graphed
//! layers-1..tail suffix. PLE performs host/NVMe work and therefore cannot be
//! captured, but all device addresses after that boundary are stable for a
//! fixed sequence slot.

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::PagedKvCache;

use super::super::types::TransformerModel;
use super::capture_guard::CaptureGuard;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::traits::SequenceState;

impl TransformerModel {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_qwen4_ple_suffix_graph(
        &self,
        token: u32,
        hidden: DevicePtr,
        residual: DevicePtr,
        seq: &mut SequenceState,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        anyhow::ensure!(
            super::decode_qwen4_graph_policy::plain_qsa_graph_capture_allowed(
                self.config.is_qwen4_exp(),
                seq.qwen4_qsa_required,
                seq.seq_len,
                crate::layers::qwen4_qsa::graph_safe_enabled(),
            ),
            "Qwen4 PLE suffix graph requires graph-safe QSA and sparse topology for QSA requests"
        );
        anyhow::ensure!(
            self.layers.len() > 1,
            "Qwen4 PLE suffix graph requires at least two layers"
        );
        let ple = self
            .qwen4_ple
            .as_ref()
            .context("Qwen4 PLE suffix graph requested without PLE")?;

        let gap_start = if spark_runtime::cublaslt::stage_timer::gap_enabled() {
            spark_runtime::cublaslt::stage_timer::record_raw(stream)
        } else {
            0
        };
        // Layer 0 consumes the fresh embedding before PLE chooses and injects
        // its sparse row at the layer-1 boundary.
        self.layers[0].decode(
            hidden,
            residual,
            seq.layer_states[0].as_mut(),
            kv_cache,
            seq.seq_len,
            &mut seq.block_table,
            &mut seq.disk_block_ids,
            &mut seq.disk_last_offloaded_per_layer,
            ctx,
            stream,
        )?;
        ple.forward_token(
            token,
            &seq.tokens,
            hidden,
            seq.slot_idx,
            seq.seq_len == 0,
            self.gpu.as_ref(),
            stream,
        )?;

        // The dense/sparse attention route is chosen on the host at 2048 and
        // is the one frame-varying choice left in the graph under
        // ATLAS_QWEN4_QSA_GRAPH_SAFE; recapture once when a sequence crosses it
        // (including multi-token speculative jumps such as 2045 -> 2050).
        let cached_sparse = self
            .decode_graph_sparse_regime
            .lock()
            .get(&seq.slot_idx)
            .copied();
        if super::decode_qwen4_graph_policy::ple_suffix_graph_needs_recapture(
            seq.qwen4_qsa_required && crate::layers::qwen4_qsa::graph_safe_enabled(),
            cached_sparse,
            seq.seq_len,
        ) {
            self.invalidate_rotary_graphs(seq.slot_idx)?;
        }
        if let Some(graph) = self.decode_graph.lock().get(&seq.slot_idx).copied() {
            if graph.0 != 0 {
                self.gpu.launch_graph(graph, stream)?;
            }
            if gap_start != 0 {
                let end = spark_runtime::cublaslt::stage_timer::record_raw(stream);
                crate::model::decode_stages::gap_step(gap_start, end, seq.seq_len);
            }
            seq.tokens.push(token);
            seq.seq_len += 1;
            return Ok(self.decode_logits_ptr());
        }

        let graph_ctx = ForwardContext {
            graph_capture: true,
            ..*ctx
        };
        let capture = CaptureGuard::begin(self.gpu.as_ref(), stream, "qwen4_ple_suffix")?;
        for i in 1..self.layers.len() {
            self.layers[i].decode(
                hidden,
                residual,
                seq.layer_states[i].as_mut(),
                kv_cache,
                seq.seq_len,
                &mut seq.block_table,
                &mut seq.disk_block_ids,
                &mut seq.disk_last_offloaded_per_layer,
                &graph_ctx,
                stream,
            )?;
        }

        let normed = self.buffers.norm_output();
        let final_input = self
            .qwen4_final_hidden(hidden, residual, stream)?
            .context("Qwen4 PLE suffix graph is missing the terminal mixer")?;
        self.final_norm_rows(final_input, normed, 1, stream)?;
        self.lm_head(normed, stream)?;

        // Success path: disarms the guard and returns the real handle. Any
        // `?` above this point dropped `capture` while still armed, which
        // closed the capture via `Drop` instead (see `CaptureGuard`).
        let graph = capture.end()?;
        if graph.0 != 0 {
            tracing::info!(
                "Qwen4 PLE segmented graph captured for slot={} (layers 1..{} + head)",
                seq.slot_idx,
                self.layers.len()
            );
            self.decode_graph.lock().insert(seq.slot_idx, graph);
            self.decode_graph_sparse_regime
                .lock()
                .insert(seq.slot_idx, seq.seq_len >= 2_048);
            self.gpu.launch_graph(graph, stream)?;
        } else {
            tracing::warn!("Qwen4 PLE segmented graph capture returned a null handle");
        }

        seq.tokens.push(token);
        seq.seq_len += 1;
        Ok(self.decode_logits_ptr())
    }
}
