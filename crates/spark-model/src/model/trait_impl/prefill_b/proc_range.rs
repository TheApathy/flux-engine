// SPDX-License-Identifier: AGPL-3.0-only

//! Phase 2b: compute the effective processing range within this chunk
//! after Marconi/prefix-cache skip. May early-return when the entire
//! chunk is covered by cache and is_last_chunk == false.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, HostToDeviceCopy};

use super::super::super::types::TransformerModel;
use crate::layers::ops;
use crate::traits::SequenceState;

pub(in crate::model) enum ProcRange {
    /// Process this many tokens; phase 3+ run normally.
    Compute {
        proc_start: usize,
        proc_count: usize,
        effective_seq_len_start: usize,
    },
    /// Whole chunk cached and not last — caller returns immediately.
    EarlyReturn(DevicePtr),
}

impl TransformerModel {
    pub(in crate::model) fn prefill_b_proc_range(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_start: usize,
        chunk_len: usize,
        is_last_chunk: bool,
        kv_write_start: usize,
        marconi_skip: bool,
        stream: u64,
    ) -> Result<ProcRange> {
        let h = self.config.hidden_size;
        let hidden = self.buffers.hidden_states();

        if marconi_skip && kv_write_start > chunk_start {
            // Skip cached tokens within this chunk
            let skip_in_chunk = (kv_write_start - chunk_start).min(chunk_len);
            if skip_in_chunk >= chunk_len {
                // Entire chunk is cached. Non-last chunks append their tokens
                // below before returning `EarlyReturn`; last chunks fall through
                // to Phase 5 (`prefill_b.rs`) which appends the chunk's tokens once.
                seq.seq_len = chunk_start + chunk_len;
                if is_last_chunk {
                    // Need to process at least the last token for logits.
                    // Advance DFlash past the skipped prefix up to the recomputed
                    // final token so capture lands cleanly at `chunk_start + chunk_len - 1`.
                    self.advance_dflash_marconi_skip(seq, chunk_start + chunk_len - 1)?;
                    // Re-embed just the last token into hidden[0].
                    if self.config.pp_rank > 0 {
                        return Ok(ProcRange::Compute {
                            proc_start: chunk_start + chunk_len - 1,
                            proc_count: 1,
                            effective_seq_len_start: chunk_start + chunk_len - 1,
                        });
                    }
                    let last_tok = tokens[chunk_start + chunk_len - 1];
                    if self.config.is_qwen4_exp() {
                        self.embed(last_tok, hidden, stream)?;
                        return Ok(ProcRange::Compute {
                            proc_start: chunk_start + chunk_len - 1,
                            proc_count: 1,
                            effective_seq_len_start: chunk_start + chunk_len - 1,
                        });
                    }
                    let last_tok_bytes = last_tok.to_le_bytes();
                    let token_id_dev = self.buffers.scratch();
                    self.copy_h2d_group_fast(
                        &[HostToDeviceCopy::new(&last_tok_bytes, token_id_dev)],
                        stream,
                    )?;
                    ops::batched_embed(
                        self.gpu.as_ref(),
                        self.batched_embed_kernel,
                        token_id_dev,
                        self.embed_tokens.weight,
                        hidden,
                        1,
                        h as u32,
                        stream,
                    )?;
                    self.scale_embeddings(hidden, 1usize, stream)?;
                    Ok(ProcRange::Compute {
                        proc_start: chunk_start + chunk_len - 1,
                        proc_count: 1,
                        effective_seq_len_start: chunk_start + chunk_len - 1,
                    })
                } else {
                    self.advance_dflash_marconi_skip(seq, chunk_start + chunk_len)?;
                    seq.tokens
                        .extend_from_slice(&tokens[chunk_start..chunk_start + chunk_len]);
                    Ok(ProcRange::EarlyReturn(DevicePtr::NULL))
                }
            } else {
                // Re-embed only uncached portion
                let uncached_start = chunk_start + skip_in_chunk;
                let uncached_count = chunk_len - skip_in_chunk;
                self.advance_dflash_marconi_skip(seq, uncached_start)?;
                if self.config.pp_rank > 0 {
                    return Ok(ProcRange::Compute {
                        proc_start: uncached_start,
                        proc_count: uncached_count,
                        effective_seq_len_start: uncached_start,
                    });
                }
                if self.config.is_qwen4_exp() {
                    let row_bytes = self.config.residual_width() * 2;
                    let chunk_tokens = &tokens[uncached_start..uncached_start + uncached_count];
                    if Self::qwen4_embed_batch_selected() {
                        self.embed_qwen4_batch(chunk_tokens, hidden, stream)?;
                    } else {
                        for (row, &token) in chunk_tokens.iter().enumerate() {
                            self.embed(token, hidden.offset(row * row_bytes), stream)?;
                        }
                    }
                    return Ok(ProcRange::Compute {
                        proc_start: uncached_start,
                        proc_count: uncached_count,
                        effective_seq_len_start: uncached_start,
                    });
                }
                let uncached_tokens = &tokens[uncached_start..uncached_start + uncached_count];
                // SAFETY: `uncached_tokens` is a valid initialized `&[u32]` slice;
                // `u8` has alignment 1 and `size_of_val` matches its exact initialized
                // byte length for `copy_h2d_group_fast`.
                let token_ids_bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(
                        uncached_tokens.as_ptr() as *const u8,
                        std::mem::size_of_val(uncached_tokens),
                    )
                };
                let token_ids_dev = self.buffers.scratch();
                self.copy_h2d_group_fast(
                    &[HostToDeviceCopy::new(token_ids_bytes, token_ids_dev)],
                    stream,
                )?;
                ops::batched_embed(
                    self.gpu.as_ref(),
                    self.batched_embed_kernel,
                    token_ids_dev,
                    self.embed_tokens.weight,
                    hidden,
                    uncached_count as u32,
                    h as u32,
                    stream,
                )?;
                self.scale_embeddings(hidden, uncached_count, stream)?;
                Ok(ProcRange::Compute {
                    proc_start: uncached_start,
                    proc_count: uncached_count,
                    effective_seq_len_start: uncached_start,
                })
            }
        } else {
            Ok(ProcRange::Compute {
                proc_start: chunk_start,
                proc_count: chunk_len,
                effective_seq_len_start: chunk_start,
            })
        }
    }
}
