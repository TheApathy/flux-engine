// SPDX-License-Identifier: AGPL-3.0-only

//! Phase 1+1b: embed chunk tokens to hidden buffer + overlay vision-pad
//! positions with pre-computed vision encoder embeddings.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use spark_runtime::gpu::HostToDeviceCopy;

use super::super::super::types::TransformerModel;
use crate::layers::ops;

impl TransformerModel {
    pub(super) fn prefill_b_embed_chunk(
        &self,
        tokens: &[u32],
        chunk_start: usize,
        chunk_len: usize,
        stream: u64,
    ) -> Result<()> {
        // Single-stream entry point: write to the arena's hidden buffer at offset 0.
        let hidden = self.buffers.hidden_states();
        self.prefill_b_embed_chunk_at(tokens, chunk_start, chunk_len, hidden, stream)
    }

    /// Embed `chunk_len` tokens into `hidden_dst` starting at position 0
    /// of the destination, then apply embedding scale + vision-pad overlay.
    /// Used by both the single-stream entry point above (writing into the
    /// arena's `hidden_states()`) and by Q12 batched prefill (writing into
    /// per-stream offsets of a shared stacked-streams buffer).
    pub(in crate::model) fn prefill_b_embed_chunk_at(
        &self,
        tokens: &[u32],
        chunk_start: usize,
        chunk_len: usize,
        hidden_dst: spark_runtime::gpu::DevicePtr,
        stream: u64,
    ) -> Result<()> {
        self.validate_vision_prompt(tokens, chunk_start, chunk_len)?;
        if self.config.pp_rank > 0 {
            // Non-zero PP stages receive hidden activations from `pp_rank - 1`
            // in `pp_stage_enter`; skipping embedding avoids both redundant work
            // and any non-blocking-stream overlap with `hidden_dst`.
            return Ok(());
        }
        let h = self.config.hidden_size;

        // ── 1. Embed chunk tokens ──
        // Qwen4 carries hc_count residual streams, so each token occupies one
        // residual_width row with its embedding repeated across all streams.
        if self.config.is_qwen4_exp() {
            let row_bytes = self.config.residual_width() * 2;
            let chunk_tokens = &tokens[chunk_start..chunk_start + chunk_len];
            if Self::qwen4_embed_batch_selected() {
                self.embed_qwen4_batch(chunk_tokens, hidden_dst, stream)?;
            } else {
                for (row, &token) in chunk_tokens.iter().enumerate() {
                    self.embed(token, hidden_dst.offset(row * row_bytes), stream)?;
                }
            }
        } else {
            // Upload token IDs to device and do a single batched embed kernel launch
            // instead of chunk_len individual D2D copies.
            {
                let chunk_tokens = &tokens[chunk_start..chunk_start + chunk_len];
                // SAFETY: `chunk_tokens` is a valid initialized `&[u32]` slice;
                // `u8` has alignment 1 and `size_of_val` matches its exact
                // initialized byte length for `copy_h2d_group_fast`.
                let token_ids_bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(
                        chunk_tokens.as_ptr() as *const u8,
                        std::mem::size_of_val(chunk_tokens),
                    )
                };
                let token_ids_dev = self.buffers.scratch(); // temporary, overwritten by MoE later
                self.copy_h2d_group_fast(
                    &[HostToDeviceCopy::new(token_ids_bytes, token_ids_dev)],
                    stream,
                )?;
                ops::batched_embed(
                    self.gpu.as_ref(),
                    self.batched_embed_kernel,
                    token_ids_dev,
                    self.embed_tokens.weight,
                    hidden_dst,
                    chunk_len as u32,
                    h as u32,
                    stream,
                )?;
                let dump_embed =
                    chunk_len > 0 && std::env::var("ATLAS_DUMP_EMBED").ok().as_deref() == Some("1");
                if dump_embed {
                    self.gpu.synchronize(stream)?;
                    let rw = self.config.residual_width();
                    let elem_bytes = if self.config.use_fp32_residual() {
                        4usize
                    } else {
                        2usize
                    };
                    let offset = (chunk_len - 1) * rw * elem_bytes;
                    let (v, n) = if elem_bytes == 4 {
                        self.readback_f32(hidden_dst.offset(offset), rw)?
                    } else {
                        self.readback_bf16(hidden_dst.offset(offset), rw)?
                    };
                    tracing::info!(
                        "ATLAS_EMBED post-batched_embed (chunk_start={}, last_tok_id={}): |x|={:.4} first5={:?}",
                        chunk_start,
                        tokens[chunk_start + chunk_len - 1],
                        n,
                        &v[..5.min(v.len())]
                    );
                }
                self.scale_embeddings(hidden_dst, chunk_len, stream)?;
                if dump_embed {
                    self.gpu.synchronize(stream)?;
                    let rw = self.config.residual_width();
                    let elem_bytes = if self.config.use_fp32_residual() {
                        4usize
                    } else {
                        2usize
                    };
                    let offset = (chunk_len - 1) * rw * elem_bytes;
                    let (v, n) = if elem_bytes == 4 {
                        self.readback_f32(hidden_dst.offset(offset), rw)?
                    } else {
                        self.readback_bf16(hidden_dst.offset(offset), rw)?
                    };
                    tracing::info!(
                        "ATLAS_EMBED post-scale_embeddings: |x|={:.4} first5={:?}",
                        n,
                        &v[..5.min(v.len())]
                    );
                }
            }
        }

        self.splice_vision_embeddings(tokens, chunk_start, chunk_len, hidden_dst, stream)?;

        Ok(())
    }
}
