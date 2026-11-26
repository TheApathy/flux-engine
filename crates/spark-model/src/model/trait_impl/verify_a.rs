// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, HostToDeviceCopy, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use super::super::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use super::super::ssm_pool::SsmStatePool;
use super::super::ssm_snapshot::SsmSnapshotPool;
use super::super::types::{PinnedMetaStaging, TransformerModel};
use crate::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use crate::layers::ops;
use crate::speculative::DraftProposer;
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use crate::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

impl TransformerModel {
    pub(super) fn decode_verify_dispatch(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<Vec<u32>> {
        let k = tokens.len();
        if k == 0 {
            return Ok(Vec::new());
        }
        if k == 1 {
            let logits = self.decode(tokens[0], seq, stream)?;
            let tok = self.argmax_on_device(logits, stream)?;
            return Ok(vec![tok]);
        }

        // GEMM-batched verification: process all K tokens per-layer,
        // using GEMM for weight-heavy projections to amortize bandwidth.
        // Honor caller's stream when non-zero; fall back to backend default stream when 0.
        let stream = if stream == 0 {
            self.gpu.default_stream()
        } else {
            stream
        };
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let fp32 = if self.config.use_fp32_residual() {
            4usize
        } else {
            2usize
        };

        let hidden = self.buffers.hidden_states(); // [K, H]
        let residual = self.buffers.residual(); // [K, H]

        let mut kv_cache = self.kv_cache.lock();

        // ── Embed all K tokens into hidden[K, H] ──
        for (t, &token) in tokens.iter().enumerate() {
            let h_t = hidden.offset(t * h * fp32);
            self.embed(token, h_t, stream)?;
        }

        // ── Per-layer processing ──
        for (i, layer) in self.layers.iter().enumerate() {
            let layer_type = self.config.layer_type(i);

            if layer_type == atlas_core::config::LayerType::FullAttention {
                // Attention layers: sequential per-token (need per-token metadata)
                for t in 0..k {
                    let pos = seq.seq_len + t;
                    let bs = kv_cache.block_size();
                    let blocks_needed = (pos / bs) + 1;
                    ensure_blocks_through_decode(
                        seq,
                        blocks_needed - 1,
                        &mut kv_cache,
                        self.prefix_cache.as_ref(),
                        self.gpu.as_ref(),
                        stream,
                    )?;

                    // Upload per-token attention metadata
                    let meta_base = self
                        .buffers
                        .scratch()
                        .offset(crate::layer::STEP_META_SCRATCH_OFFSET);
                    let max_blocks = seq.block_table.len() as u32;
                    let pos_val = pos as u32;
                    let pos_bytes = pos_val.to_le_bytes();
                    let block_idx = seq
                        .physical_block_for(pos / bs)
                        .unwrap_or(self.dummy_kv_block);
                    let global_slot = (block_idx as i64) * (bs as i64) + ((pos % bs) as i64);
                    let slot_bytes = global_slot.to_le_bytes();
                    let actual_seq_len = (pos + 1) as i32;
                    let seq_len_bytes = actual_seq_len.to_le_bytes();
                    let bt_i32: Vec<i32> = seq.block_table.iter().map(|&b| b as i32).collect();
                    // SAFETY: `bt_i32` is a live initialized `Vec<i32>`; `u8` has
                    // alignment 1 and `size_of_val` matches its exact initialized
                    // byte length for `copy_h2d_group_fast`.
                    let bt_bytes: &[u8] = unsafe {
                        std::slice::from_raw_parts(
                            bt_i32.as_ptr() as *const u8,
                            std::mem::size_of_val(bt_i32.as_slice()),
                        )
                    };
                    let copies = [
                        HostToDeviceCopy::new(&pos_bytes, meta_base),
                        HostToDeviceCopy::new(&slot_bytes, meta_base.offset(8)),
                        HostToDeviceCopy::new(&seq_len_bytes, meta_base.offset(16)),
                        HostToDeviceCopy::new(bt_bytes, meta_base.offset(256)),
                    ];
                    self.copy_h2d_group_fast(&copies, stream)?;

                    let attn_metadata = AttnMetadataDev {
                        qwen4_qsa_required: seq.qwen4_qsa_required,
                        positions: meta_base,
                        positions_h: meta_base,
                        positions_w: meta_base,
                        slot: meta_base.offset(8),
                        seq_len: meta_base.offset(16),
                        block_table: meta_base.offset(256),
                        max_blocks_per_seq: max_blocks,
                        num_seqs: 1,
                    };

                    let ctx = ForwardContext {
                        buffers: &self.buffers,
                        gpu: self.gpu.as_ref(),
                        config: &self.config,
                        attn_metadata: Some(attn_metadata),
                        profile: false,
                        comm: self.comm_ref(),
                        graph_capture: false,
                        ddtree_parent_ids_dev: None,
                        tree_aware_attn: None,
                        ssm_multi_seq_ptr_table_override: None,
                        self_spec_sparse_draft: None,
                        ffn_defer: None,
                    };

                    let h_t = hidden.offset(t * h * fp32);
                    let r_t = residual.offset(t * h * fp32);
                    layer.decode(
                        h_t,
                        r_t,
                        seq.layer_states[i].as_mut(),
                        &mut kv_cache,
                        pos,
                        &mut seq.block_table,
                        &mut seq.disk_block_ids,
                        &mut seq.disk_last_offloaded_per_layer,
                        &ctx,
                        stream,
                    )?;
                }
            } else {
                // SSM layers: GEMM-batched via decode_batched override
                let ctx = ForwardContext {
                    buffers: &self.buffers,
                    gpu: self.gpu.as_ref(),
                    config: &self.config,
                    attn_metadata: None,
                    profile: false,
                    comm: self.comm_ref(),
                    graph_capture: false,
                    ddtree_parent_ids_dev: None,
                    tree_aware_attn: None,
                    ssm_multi_seq_ptr_table_override: None,
                    self_spec_sparse_draft: None,
                    ffn_defer: None,
                };

                layer.decode_batched(
                    hidden,
                    residual,
                    k,
                    seq.layer_states[i].as_mut(),
                    &mut kv_cache,
                    seq.seq_len,
                    &mut seq.block_table,
                    &mut seq.disk_block_ids,
                    &mut seq.disk_last_offloaded_per_layer,
                    &ctx,
                    stream,
                )?;
            }
        }

        // ── Final norm for K tokens ──
        let normed = self.buffers.norm_output();
        self.final_norm_rows(hidden, normed, k as u32, stream)?;

        // ── LM head for K tokens → logits[K, vocab] ──
        self.lm_head_batched(normed, k as u32, stream)?;

        // ── Argmax per token ──
        // Verify-side (possibly truncated) vocab must match the stride/range
        // `lm_head_batched` wrote — full vocab when truncation is off.
        let vocab = self.verify_lmhead_vocab() as usize;
        let mut results = Vec::with_capacity(k);
        for t in 0..k {
            let logits_t = self.buffers.logits().offset(t * vocab * bf16);
            let out_ptr = self.buffers.scratch().offset(t * 4);
            ops::argmax_bf16(
                self.gpu.as_ref(),
                self.argmax_kernel,
                logits_t,
                out_ptr,
                vocab as u32,
                stream,
            )?;
        }
        // D2H: copy all K argmax results at once on the active stream.
        let mut buf = vec![0u8; k * 4];
        self.gpu
            .copy_d2h_on_stream(self.buffers.scratch(), &mut buf, stream)?;
        for t in 0..k {
            let tok =
                u32::from_le_bytes([buf[t * 4], buf[t * 4 + 1], buf[t * 4 + 2], buf[t * 4 + 3]]);
            results.push(tok);
        }

        // Push ALL tokens and advance seq_len by K. See `decode_verify_graphed`
        // for rationale — the prior `seq_len += k-1 / skip tokens[0]` semantic
        // had an off-by-one that misaligned positions across iterations and
        // broke 80B-nvfp4-mtp fib on the argmax-edge final token.
        for &token in tokens {
            seq.tokens.push(token);
        }
        seq.seq_len += k;

        Ok(results)
    }

    pub(super) fn checkpoint_ssm_states_dispatch(&self, seq: &mut SequenceState) -> Result<()> {
        use crate::layer::SsmLayerState;

        let stream = self.gpu.default_stream();
        for (i, layer_state) in seq.layer_states.iter_mut().enumerate() {
            if self.config.layer_type(i) == atlas_core::config::LayerType::LinearAttention {
                let ssm = layer_state
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {i}"))?;

                // Use pool geometry (handles both GDN and Mamba-2, and FP16 h_state).
                let h_bytes = if ssm.h_is_f16 {
                    self.ssm_pool.h_bytes / 2
                } else {
                    self.ssm_pool.h_bytes
                };
                let conv_bytes = self.ssm_pool.conv_bytes;

                // All SSM buffers belong to `ssm_pool` (`free_sequence_dispatch` does
                // not `gpu.free()` them). Refuse to allocate unowned GPU memory when
                // the pool did not provision checkpoint buffers for this slot.
                let h_ckpt = ssm.h_state_checkpoint.ok_or_else(|| {
                    anyhow::anyhow!(
                        "checkpoint_ssm_states: layer {i} (slot {}) has no pool-owned \
                         h_state_checkpoint",
                        seq.slot_idx
                    )
                })?;
                let conv_ckpt = ssm.conv_state_checkpoint.ok_or_else(|| {
                    anyhow::anyhow!(
                        "checkpoint_ssm_states: layer {i} (slot {}) has no pool-owned \
                         conv_state_checkpoint",
                        seq.slot_idx
                    )
                })?;

                // D2D copy: state → checkpoint
                self.gpu
                    .copy_d2d_async(ssm.h_state, h_ckpt, h_bytes, stream)?;
                self.gpu
                    .copy_d2d_async(ssm.conv_state, conv_ckpt, conv_bytes, stream)?;
            }
        }
        self.gpu.synchronize(stream)?;
        Ok(())
    }

    pub(super) fn rollback_ssm_states_dispatch(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
    ) -> Result<()> {
        // GDN LAZY guard: under ATLAS_SSM_GDN_LAZY the ExactSequence kernel
        // writes only inter[k-1]; every other slot this function would copy is
        // GARBAGE. Our serving path commits through commit_verify_state_async
        // (which replays instead) — reaching this sync rollback under the lazy
        // flag means an unsupported route is live. Refuse loudly rather than
        // restore garbage into h_state.
        if crate::layers::qwen3_ssm::gdn_seq_lazy_enabled() {
            anyhow::bail!(
                "rollback_ssm_states: sync SSM rollback is unsupported under \
                 ATLAS_SSM_GDN_LAZY=1 (sparse intermediates). This serving route \
                 must commit via commit_verify_state_async."
            );
        }
        use crate::layer::SsmLayerState;

        let stream = self.gpu.default_stream();
        for (i, layer_state) in seq.layer_states.iter_mut().enumerate() {
            if self.config.layer_type(i) == atlas_core::config::LayerType::LinearAttention {
                let ssm = layer_state
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {i}"))?;

                let h_bytes = if ssm.h_is_f16 {
                    self.ssm_pool.h_bytes / 2
                } else {
                    self.ssm_pool.h_bytes
                };
                let conv_bytes = self.ssm_pool.conv_bytes;

                if num_accepted == 0 {
                    // Restore to pre-verification checkpoint
                    if let Some(ckpt) = ssm.h_state_checkpoint {
                        self.gpu
                            .copy_d2d_async(ckpt, ssm.h_state, h_bytes, stream)?;
                    }
                    if let Some(ckpt) = ssm.conv_state_checkpoint {
                        self.gpu
                            .copy_d2d_async(ckpt, ssm.conv_state, conv_bytes, stream)?;
                    }
                } else if num_accepted <= ssm.h_state_intermediates.len() {
                    // Restore to intermediate checkpoint after the last accepted token
                    let idx = num_accepted - 1;
                    self.gpu.copy_d2d_async(
                        ssm.h_state_intermediates[idx],
                        ssm.h_state,
                        h_bytes,
                        stream,
                    )?;
                    self.gpu.copy_d2d_async(
                        ssm.conv_state_intermediates[idx],
                        ssm.conv_state,
                        conv_bytes,
                        stream,
                    )?;
                } else if ssm.h_state_intermediates.is_empty() {
                    // No intermediates available (self-spec / ngram path) and
                    // the caller asked for a partial rollback. Without
                    // intermediates we cannot reach the post-N-token state
                    // by replay; silently skipping would leave SSM state
                    // advanced past the accepted boundary, corrupting
                    // every subsequent decode. Fail fast so the operator
                    // sees the misconfiguration instead of silent gibberish.
                    anyhow::bail!(
                        "rollback_ssm_states: cannot restore SSM to N={num_accepted} \
                         without per-token intermediates (layer {i}). \
                         self-speculative / ngram with SSM models needs MTP \
                         intermediates support; use --speculative (MTP) or \
                         --num-drafts 1 for SSM models."
                    );
                }
                // If num_accepted == num_tokens, SSM state is already correct
            }
        }
        // No synchronize needed: rollback copies and subsequent operations
        // are on the same CUDA stream, so ordering is guaranteed.
        Ok(())
    }

    /// Phase-C decode-time boundary snapshot save.
    ///
    /// Copies the sequence's live SSM state (the active `SsmStatePool`
    /// slot `seq.slot_idx`) into the decode-rollback ring slot
    /// `(seq.slot_idx, ring_slot)` of [`SsmSnapshotPool`]. Reuses the
    /// same D2D copy primitive Marconi and MTP verify use (SSOT). The
    /// copies run on the default stream so they are ordered after the
    /// decode that produced this boundary token and before any later
    /// decode that would overwrite the pool slot.
    pub(super) fn save_decode_ssm_snapshot_dispatch(
        &self,
        seq: &SequenceState,
        ring_slot: usize,
    ) -> Result<()> {
        if !self.ssm_snapshots.decode_rollback_enabled() {
            anyhow::bail!("save_decode_ssm_snapshot: decode-rollback region not allocated");
        }
        let stream = self.gpu.default_stream();
        self.ssm_snapshots.save_decode(
            seq.slot_idx,
            ring_slot,
            &self.ssm_pool,
            self.gpu.as_ref(),
            stream,
        )
    }

    /// Phase-C decode-time boundary snapshot restore.
    ///
    /// Inverse of [`Self::save_decode_ssm_snapshot_dispatch`]: copies the
    /// ring snapshot `(seq.slot_idx, ring_slot)` back into the live
    /// `SsmStatePool` slot, undoing every recurrent update the dropped
    /// degenerate tail applied.
    pub(super) fn restore_decode_ssm_snapshot_dispatch(
        &self,
        seq: &SequenceState,
        ring_slot: usize,
    ) -> Result<()> {
        if !self.ssm_snapshots.decode_rollback_enabled() {
            anyhow::bail!("restore_decode_ssm_snapshot: decode-rollback region not allocated");
        }
        let stream = self.gpu.default_stream();
        self.ssm_snapshots.restore_decode(
            seq.slot_idx,
            ring_slot,
            &self.ssm_pool,
            self.gpu.as_ref(),
            stream,
        )
    }
}
