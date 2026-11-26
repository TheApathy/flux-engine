// SPDX-License-Identifier: AGPL-3.0-only

//! Phase 5+6+7+8 — last-chunk finalization:
//!   • final RMS-norm on the last token's hidden state
//!   • LM head → logits buffer
//!   • diagnostic dumps (long-context / Gemma4 paths)
//!   • prefix-cache insert + Marconi snapshot save (with reclaim retry)
//!   • DFlash ctx-len bookkeeping

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::super::types::TransformerModel;
use crate::layers::ops;
use crate::traits::SequenceState;

impl TransformerModel {
    pub(in crate::model) fn prefill_b_finalize_last(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        kv_cache: &mut PagedKvCache,
        chunk_start: usize,
        chunk_len: usize,
        proc_count: usize,
        stream: u64,
    ) -> Result<DevicePtr> {
        self.prefill_b_finalize_last_at(
            tokens,
            seq,
            kv_cache,
            chunk_start,
            chunk_len,
            proc_count,
            0,
            stream,
        )
    }

    /// Q12 Path B: stream-offset-aware finalize for the kernel-batched
    /// orchestrator. `hidden_stream_offset_tokens` is `b * chunk_len`
    /// where `b` is the stream's index in the batched dispatch.
    /// All other args identical to `prefill_b_finalize_last`.
    pub(in crate::model) fn prefill_b_finalize_last_at(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        kv_cache: &mut PagedKvCache,
        chunk_start: usize,
        chunk_len: usize,
        proc_count: usize,
        hidden_stream_offset_tokens: usize,
        stream: u64,
    ) -> Result<DevicePtr> {
        let h = self.config.hidden_size;
        let fp32 = if self.config.use_fp32_residual() {
            4usize
        } else {
            2usize
        };
        let hidden = self.buffers.hidden_states();
        let bs = kv_cache.block_size();

        // Numerics gate: dump the whole residual stream of this chunk
        // (`proc_count` rows x hidden, pre-final-norm, native residual dtype)
        // when ATLAS_PREFILL_HIDDEN_DUMP=<dir>. Diagnostic only.
        if let Ok(dir) = std::env::var("ATLAS_PREFILL_HIDDEN_DUMP")
            && !dir.is_empty()
        {
            self.gpu.synchronize(stream)?;
            let rw = self.config.residual_width();
            let bytes_len = proc_count * rw * fp32;
            let mut buf = vec![0u8; bytes_len];
            self.gpu.copy_d2h(
                hidden.offset(hidden_stream_offset_tokens * rw * fp32),
                &mut buf,
            )?;
            std::fs::create_dir_all(&dir).ok();
            let name = format!("hidden_start{chunk_start}_rows{proc_count}_h{rw}_elem{fp32}.bin");
            std::fs::write(std::path::Path::new(&dir).join(&name), &buf).ok();
            tracing::info!("ATLAS_PREFILL_HIDDEN_DUMP: wrote {name} ({bytes_len} bytes)");
        }

        if self.config.pp_rank == 0 {
            // ── 6. Final norm on LAST token only ──
            let last_token_offset = hidden_stream_offset_tokens + proc_count - 1;
            let persistent_width = self.config.residual_width();
            let last_hidden = hidden.offset(last_token_offset * persistent_width * fp32);
            let last_residual = self
                .buffers
                .residual()
                .offset(last_token_offset * persistent_width * fp32);
            let normed = self.buffers.norm_output();
            let final_input = self
                .qwen4_final_hidden_nvfp4(last_hidden, last_residual, stream)?
                .unwrap_or(last_hidden);
            self.final_norm_rows(final_input, normed, 1, stream)?;

            // ATLAS_PREFILL_LEAN_FINALIZE (K8 item A): the two implicit diagnostics below
            // run on every prompt longer than 16384 tokens (a stream sync, a 16-value
            // readback, a full-vocab logits D2H and a host sort of every logit). They read
            // device memory and log only, so skipping them changes no device state. An
            // explicit diag request (diag_gemma4_enabled) still runs them.
            let implicit_diag = (chunk_start + chunk_len) > 16384;
            let lean =
                crate::model::host_bucket::armed(crate::model::host_bucket::Lever::LeanFinalize)
                    .map_err(anyhow::Error::msg)?
                    && implicit_diag;
            if lean && crate::model::host_bucket::first_time("PREFILL_LEAN_FINALIZE_ENGAGED") {
                tracing::info!(
                    "PREFILL_LEAN_FINALIZE_ENGAGED selector={} tokens={} skipped=diag_post_norm,diag_logits",
                    crate::model::host_bucket::Lever::LeanFinalize.selector(),
                    chunk_start + chunk_len,
                );
            }
            let run_diag =
                (implicit_diag && !lean) || crate::model::env_diag::diag_gemma4_enabled();

            // Diagnostic: post-norm hidden state
            if run_diag {
                self.gpu.synchronize(stream)?;
                let (vals, norm) = self.readback_bf16(normed, h.min(16))?;
                tracing::warn!(
                    "DIAG post-norm: norm={norm:.4} first2={:.4?}",
                    &vals[..2.min(vals.len())]
                );
            }

            // Per-layer divergence dump: final-norm output (input to lm_head).
            if let Ok(dir) = std::env::var("ATLAS_NEMO_DUMP")
                && !dir.is_empty()
            {
                self.gpu.synchronize(stream)?;
                let (vals, _) = self.readback_bf16(normed, h)?;
                let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::create_dir_all(&dir).ok();
                std::fs::write(
                    std::path::Path::new(&dir).join("atlas_final_norm.bin"),
                    &bytes,
                )
                .ok();
            }

            // ── 7. LM head on last token → logits ──
            self.lm_head(normed, stream)?;

            // Per-layer divergence dump: full logits vector + top-10 token IDs.
            if let Ok(dir) = std::env::var("ATLAS_NEMO_DUMP")
                && !dir.is_empty()
            {
                self.gpu.synchronize(stream)?;
                let n_logits = self.config.vocab_size;
                let logits_ptr = self.decode_logits_ptr();
                let logit_vals: Vec<f32> = if self.use_fp32_logits {
                    let mut buf = vec![0u8; n_logits * 4];
                    self.gpu.copy_d2h(logits_ptr, &mut buf)?;
                    buf.chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect()
                } else {
                    let mut buf = vec![0u8; n_logits * 2];
                    self.gpu.copy_d2h(logits_ptr, &mut buf)?;
                    buf.chunks_exact(2)
                        .map(|c| {
                            let bits = u16::from_le_bytes([c[0], c[1]]);
                            f32::from_bits((bits as u32) << 16)
                        })
                        .collect()
                };
                let lbytes: Vec<u8> = logit_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::create_dir_all(&dir).ok();
                std::fs::write(std::path::Path::new(&dir).join("atlas_logits.bin"), &lbytes).ok();
                let mut idx: Vec<usize> = (0..logit_vals.len()).collect();
                idx.sort_by(|&a, &b| logit_vals[b].total_cmp(&logit_vals[a]));
                let top: Vec<(usize, f32)> =
                    idx.iter().take(10).map(|&i| (i, logit_vals[i])).collect();
                tracing::info!("ATLAS_NEMO_DUMP: top-10 logits = {top:?}");
            }

            // Diagnostic: logits stats (matches `self.lm_head` output buffer & dtype)
            if run_diag {
                self.gpu.synchronize(stream)?;
                let logits_ptr = self.decode_logits_ptr();
                let n_logits = self.config.vocab_size;
                let logit_vals: Vec<f32> = if self.use_fp32_logits {
                    let mut buf = vec![0u8; n_logits * 4];
                    self.gpu.copy_d2h(logits_ptr, &mut buf)?;
                    buf.chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect()
                } else {
                    let mut buf = vec![0u8; n_logits * 2];
                    self.gpu.copy_d2h(logits_ptr, &mut buf)?;
                    buf.chunks_exact(2)
                        .map(|c| {
                            let bits = u16::from_le_bytes([c[0], c[1]]);
                            f32::from_bits((bits as u32) << 16)
                        })
                        .collect()
                };
                let max = logit_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let min = logit_vals.iter().cloned().fold(f32::INFINITY, f32::min);
                let nan_count = logit_vals.iter().filter(|v| v.is_nan()).count();
                let mut idx: Vec<usize> = (0..logit_vals.len()).collect();
                idx.sort_by(|&a, &b| logit_vals[b].total_cmp(&logit_vals[a]));
                let top5: Vec<(usize, f32)> =
                    idx.iter().take(5).map(|&i| (i, logit_vals[i])).collect();
                tracing::warn!(
                    "DIAG logits[0..{}]: max={max:.4} min={min:.4} nan={nan_count} top5={top5:?}",
                    n_logits,
                );
            }
        }

        // ── 8. Insert into prefix cache + Marconi snapshot ──
        // Hybrid `lookup_paired` / `lookup_aligned` only restores snapshots at
        // exact `block_size` boundaries (the KV radix tree stores only full
        // blocks). Saving an SSM snapshot at an unaligned `tokens.len()` can
        // never be restored and would evict valid block-aligned checkpoints.
        let noncausal_spans = seq.rotary_positions.noncausal_spans();
        let has_vision_pad = self.tokens_have_vision_pad(tokens);
        let save_aligned_ssm =
            !has_vision_pad && bs > 0 && tokens.len() >= bs && tokens.len().is_multiple_of(bs);
        let mut snap_saved = None;
        if self.prefix_cache.is_active() && self.ssm_snapshots.is_enabled() && save_aligned_ssm {
            let snap_result = match self.ssm_snapshots.save(
                seq.slot_idx,
                seq.session_hash,
                Self::seq_h_is_f16(seq),
                &self.ssm_pool,
                self.gpu.as_ref(),
                stream,
            ) {
                Ok(Some(id)) => Some(id),
                Ok(None) => {
                    tracing::debug!("Snapshot pool full, reclaiming...");
                    if self
                        .ssm_snapshots
                        .reclaim_from_cache(self.prefix_cache.as_ref(), kv_cache)
                    {
                        self.ssm_snapshots
                            .save(
                                seq.slot_idx,
                                seq.session_hash,
                                Self::seq_h_is_f16(seq),
                                &self.ssm_pool,
                                self.gpu.as_ref(),
                                stream,
                            )
                            .ok()
                            .flatten()
                    } else {
                        tracing::debug!("Reclaim failed — no evictable snapshots");
                        None
                    }
                }
                Err(e) => {
                    tracing::warn!("SSM snapshot save error: {e}");
                    None
                }
            };
            snap_saved = snap_result;
        }
        let qsa_populated =
            self.config.is_qwen4_exp() && self.config.qwen4_qsa && seq.qwen4_qsa_required;
        if let Some(snap_id) = snap_saved {
            if self.tokens_have_vision_pad(tokens) {
                self.ssm_snapshots.free(snap_id);
            } else {
                tracing::info!(
                    "Saved SSM snapshot {} for {} tokens ({} blocks) [chunk]",
                    snap_id,
                    tokens.len(),
                    seq.block_table.len(),
                );
                let (displaced, acquired) =
                    self.prefix_cache.insert_with_snapshot_and_noncausal_spans(
                        tokens,
                        &seq.block_table,
                        &seq.disk_block_ids,
                        bs,
                        snap_id,
                        seq.session_hash,
                        seq.cached_prefix_tokens,
                        &noncausal_spans,
                    );
                if qsa_populated {
                    self.prefix_cache
                        .mark_blocks_qsa_populated(tokens, &seq.block_table, bs);
                }
                super::super::super::block_mgmt::cache_acquires_disk_refs(&acquired);
                if let Some(old) = displaced {
                    self.ssm_snapshots.free(old);
                }
            }
        } else if !self.tokens_have_vision_pad(tokens) {
            let acquired = self.prefix_cache.insert_with_noncausal_spans(
                tokens,
                &seq.block_table,
                &seq.disk_block_ids,
                bs,
                seq.cached_prefix_tokens,
                &noncausal_spans,
            );
            if qsa_populated {
                self.prefix_cache
                    .mark_blocks_qsa_populated(tokens, &seq.block_table, bs);
            }
            super::super::super::block_mgmt::cache_acquires_disk_refs(&acquired);
        }

        // DFlash: advance ctx_len after the LAST chunk of chunked prefill.
        self.update_dflash_ctx_len_after_prefill(seq, chunk_start, chunk_len)?;

        // DFlash drafter-retrain teacher-forced hidden capture (default-OFF).
        // No-op unless ATLAS_DUMP_CTX_HIDDEN is set. This is the LAST-chunk
        // finalizer of the CHUNKED prefill path (the scheduler's default path
        // when max_prefill_tokens>0), so ctx_hidden_acc is fully populated for
        // all prompt positions here. `tokens` is the full prompt.
        self.dump_ctx_hidden_after_prefill(seq, tokens, stream)?;

        // MTP last-K prefill: replay the last K prompt-tail target hidden
        // states through the MTP head so its self-attention sees recent
        // prompt context before the first decode. Gated by
        // `ATLAS_MTP_LASTK_PREFILL=N` (capacity captured at model init).
        // No-op when capacity==0, no proposer is wired, or the proposer is
        // not MTP (DFlash has its own prefill path via dflash_capture_layers).
        self.mtp_lastk_prefill_after_finalize(
            tokens,
            seq,
            chunk_start,
            chunk_len,
            proc_count,
            stream,
        )?;

        Ok(self.decode_logits_ptr())
    }
}
