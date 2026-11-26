// SPDX-License-Identifier: AGPL-3.0-only

//! Phase 4: forward through every transformer layer (decode-path on
//! single-token chunks, prefill-path otherwise) plus DFlash capture
//! and per-layer profiling/diagnostics.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::super::types::TransformerModel;
use crate::layer::{AttnMetadataDev, ForwardContext};
use crate::traits::SequenceState;

impl TransformerModel {
    pub(super) fn prefill_b_forward_layers(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        kv_cache: &mut PagedKvCache,
        chunk_start: usize,
        chunk_len: usize,
        is_last_chunk: bool,
        proc_count: usize,
        effective_seq_len_start: usize,
        _kv_write_start: usize,
        _marconi_skip: bool,
        meta_base: DevicePtr,
        slot_offset: usize,
        pos_stream_bytes: usize,
        use_mrope: bool,
        needs_paged: bool,
        stream: u64,
        ple_prefetch: Option<(usize, usize, crate::layers::qwen4_ple::PlePrefillRead)>,
        cross_chunk_ple_prefetch_admitted: bool,
    ) -> Result<()> {
        let fp32 = if self.config.use_fp32_residual() {
            4usize
        } else {
            2usize
        };
        let hidden = self.buffers.hidden_states();
        let residual = self.buffers.residual();

        let (block_table_dev, seq_len_dev) = if needs_paged {
            let page_meta = seq.chunked_prefill_meta.as_ref().unwrap();
            (page_meta.block_table, page_meta.seq_len)
        } else {
            (DevicePtr::NULL, DevicePtr::NULL)
        };

        let (positions_h_dev, positions_w_dev) = if use_mrope {
            (
                meta_base.offset(pos_stream_bytes),
                meta_base.offset(pos_stream_bytes * 2),
            )
        } else {
            (meta_base, meta_base)
        };
        let attn_metadata = AttnMetadataDev {
            qwen4_qsa_required: seq.qwen4_qsa_required,
            positions: meta_base,
            positions_h: positions_h_dev,
            positions_w: positions_w_dev,
            slot: meta_base.offset(slot_offset),
            seq_len: seq_len_dev,
            block_table: block_table_dev,
            max_blocks_per_seq: seq.block_table.len() as u32,
            num_seqs: 1,
        };

        // Consume the one-shot ATLAS_PROFILE_FIRST flag (additive).
        let profile_now = self.profile
            || self
                .profile_first_pending
                .swap(false, std::sync::atomic::Ordering::Relaxed);

        let ctx = ForwardContext {
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            attn_metadata: Some(attn_metadata),
            profile: profile_now,
            comm: self.comm_ref(),
            graph_capture: false,
            ddtree_parent_ids_dev: None,
            tree_aware_attn: None,
            ssm_multi_seq_ptr_table_override: None,
            self_spec_sparse_draft: None,
            ffn_defer: None,
        };

        // When proc_count == 1 (warm prefix cache hit), use the decode layer path
        // instead of the prefill path. Decode uses GEMV kernels optimized for M=1
        // and the decode MoE path, which is ~7x faster per layer than the prefill
        // GEMM path for a single token (0.7ms/layer vs 5ms/layer).
        let use_decode_path = proc_count == 1 && effective_seq_len_start > 0;
        let prefill_receipt = if use_decode_path {
            None
        } else {
            crate::model::qwen4_prefill_engagement::begin(&self.config, proc_count)?
        };
        let layer_kv_write_start = super::cached_kv_rows_in_slice(
            seq.cached_prefix_tokens,
            effective_seq_len_start,
            proc_count,
        );
        let prefill_t0 = if profile_now {
            self.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut layer_times: Vec<u128> = Vec::new();
        // A PLE row read started before the embedding (level 2) is only valid
        // for the exact token range this chunk ends up processing.
        let mut ple_prefetch = ple_prefetch.and_then(|(start, len, read)| {
            (start == effective_seq_len_start && len == proc_count).then_some(read)
        });
        // Expert-major super-chunk (ATLAS_QWEN4_PREFILL_MOE_EXPERT_MAJOR=1): mixers per slice,
        // one MoE pass per layer over all proc_count rows.
        let mixer_rows = crate::layers::qwen4_prefill_moe::mixer_rows();
        let expert_major = !use_decode_path
            && crate::layers::qwen4_prefill_moe::expert_major_selected()
            && proc_count > mixer_rows;
        if expert_major {
            anyhow::ensure!(
                self.dflash_capture_layers.is_empty() && !profile_now && !use_mrope,
                "expert-major prefill does not support DFlash capture, ATLAS_PROFILE or MRoPE"
            );
            tracing::info!(
                "QWEN4_PREFILL_EXPERT_MAJOR_ENGAGED start={effective_seq_len_start} rows={proc_count} mixer_rows={mixer_rows}"
            );
        }
        use spark_runtime::cublaslt::stage_timer;
        let stages = !use_decode_path && stage_timer::enabled();
        if stages {
            stage_timer::begin(stream);
        }
        // Deferred hyperconnection injections (ATLAS_QWEN4_PREFILL_MHC_FUSED)
        // never outlive this loop. One left by a failed chunk is dropped.
        if crate::layers::qwen4_hyper::pending::discard_stale() {
            tracing::warn!("dropped a stale pending HC injection from a failed prefill chunk");
        }
        if chunk_start == 0 && crate::layers::moe::qwen4_moe_status::discard_stale() {
            tracing::warn!("dropped deferred compact-MoE status words from a failed prefill chunk");
        }
        // ATLAS_QWEN4_PREFILL_OPROJ_AMAX_FUSED: count this chunk's o_proj hand-offs (receipt below).
        crate::layers::qwen4_fp8_act_amax::oproj_chunk_begin();
        let readers_every_layer =
            profile_now || std::env::var("ATLAS_NEMO_DUMP").is_ok_and(|d| !d.is_empty());
        let last_layer_crop = self.prefill_b_last_layer_crop(
            is_last_chunk,
            proc_count,
            effective_seq_len_start,
            tokens.len().saturating_sub(chunk_start + chunk_len),
            use_decode_path,
            expert_major,
            profile_now,
        )?;
        let ple_prefill_batch = std::env::var("ATLAS_QWEN4_PLE_PREFILL_BATCH")
            .ok()
            .as_deref()
            == Some("1");
        self.pp_stage_enter(hidden, proc_count, stream)?;
        for (i, layer) in self.layers.iter().enumerate() {
            if i == 1
                && let Some(ple) = &self.qwen4_ple
            {
                crate::layers::qwen4_hyper::pending::flush(self.gpu.as_ref(), stream)?;
                let row_bytes = self.config.residual_width() * 2;
                let end = effective_seq_len_start
                    .saturating_add(proc_count)
                    .min(tokens.len());
                let prior = &tokens[..effective_seq_len_start.min(tokens.len())];
                let current = &tokens[effective_seq_len_start..end];
                anyhow::ensure!(
                    current.len() == proc_count,
                    "Qwen4 PLE chunk extent {} != proc_count {proc_count}",
                    current.len()
                );
                if ple_prefill_batch {
                    ple.forward_prefill(
                        current,
                        prior,
                        hidden,
                        seq.slot_idx,
                        effective_seq_len_start == 0,
                        self.gpu.as_ref(),
                        stream,
                        ple_prefetch.take(),
                    )?;
                } else {
                    let mut history = prior.to_vec();
                    for (row, &token) in current.iter().enumerate() {
                        ple.forward_token(
                            token,
                            &history,
                            hidden.offset(row * row_bytes),
                            seq.slot_idx,
                            effective_seq_len_start == 0 && row == 0,
                            self.gpu.as_ref(),
                            stream,
                        )?;
                        history.push(token);
                    }
                }
                ple.capture_post_prefill_parity(
                    hidden,
                    tokens,
                    prior,
                    current,
                    effective_seq_len_start,
                    seq.slot_idx,
                    effective_seq_len_start == 0,
                    self.gpu.as_ref(),
                    stream,
                )?;
            }
            // Layer 1 has already consumed this chunk's PLE read; launch next
            // chunk's cross-chunk PLE prefetch at layer 2 so all remaining 46
            // layers (~2.2-2.6s) hide NVMe I/O variance instead of only 8.
            let prefetch_layer = if self.layers.len() > 2 { 2 } else { 0 };
            if i == prefetch_layer
                && ple_prefill_batch
                && let Some(ple) = &self.qwen4_ple
            {
                self.begin_cross_chunk_ple_prefetch(
                    ple,
                    tokens,
                    seq.slot_idx,
                    chunk_start,
                    chunk_len,
                    cross_chunk_ple_prefetch_admitted,
                )?;
            }
            let lt0 = if profile_now {
                self.gpu.synchronize(stream)?;
                Some(std::time::Instant::now())
            } else {
                None
            };
            if stages {
                stage_timer::set_group("");
                stage_timer::mark(if i == 1 { "ple" } else { "between" }, stream);
                stage_timer::set_group(match self.config.layer_type(i) {
                    atlas_core::config::LayerType::FullAttention => "attn",
                    _ => "ssm",
                });
            }
            if use_decode_path {
                layer
                    .decode(
                        hidden,
                        residual,
                        seq.layer_states[i].as_mut(),
                        kv_cache,
                        effective_seq_len_start,
                        &mut seq.block_table,
                        &mut seq.disk_block_ids,
                        &mut seq.disk_last_offloaded_per_layer,
                        &ctx,
                        stream,
                    )
                    .map_err(|e| anyhow::anyhow!("Prefill-as-decode layer {i} failed: {e}"))?;
            } else if expert_major {
                let row_bytes = self.config.residual_width() * fp32;
                let mut s = 0usize;
                while s < proc_count {
                    let n = mixer_rows.min(proc_count - s);
                    let slice_meta = AttnMetadataDev {
                        positions: meta_base.offset(s * 4),
                        positions_h: meta_base.offset(s * 4),
                        positions_w: meta_base.offset(s * 4),
                        slot: meta_base.offset(slot_offset + s * 8),
                        ..attn_metadata
                    };
                    let slice_ctx = ForwardContext {
                        attn_metadata: Some(slice_meta),
                        ..ctx
                    };
                    let slice_kv_write_start = super::cached_kv_rows_in_slice(
                        seq.cached_prefix_tokens,
                        effective_seq_len_start + s,
                        n,
                    );
                    {
                        let _slice = crate::layers::qwen4_prefill_moe::ExpertMajorSliceGuard::enter(
                            mixer_rows,
                        );
                        layer
                            .prefill(
                                hidden.offset(s * row_bytes),
                                residual.offset(s * row_bytes),
                                n,
                                seq.layer_states[i].as_mut(),
                                kv_cache,
                                effective_seq_len_start + s,
                                &mut seq.block_table,
                                &mut seq.disk_block_ids,
                                &mut seq.disk_last_offloaded_per_layer,
                                slice_kv_write_start,
                                &slice_ctx,
                                stream,
                            )
                            .map_err(|e| {
                                anyhow::anyhow!(
                                    "Expert-major layer {i} mixer slice {s}+{n} failed: {e}"
                                )
                            })?;
                    }
                    // Pending HC injections cover one slice's rows: never let them cross.
                    crate::layers::qwen4_hyper::pending::flush(self.gpu.as_ref(), stream)?;
                    s += n;
                }
                layer
                    .qwen4_prefill_finish_moe(hidden, residual, proc_count, &ctx, stream)
                    .map_err(|e| anyhow::anyhow!("Expert-major layer {i} MoE failed: {e}"))?;
                crate::layers::qwen4_hyper::pending::flush(self.gpu.as_ref(), stream)?;
            } else {
                use crate::layers::qwen4_prefill_moe::last_layer_crop::CropGuard;
                let _crop = last_layer_crop
                    .filter(|_| i + 1 == self.layers.len())
                    .map(CropGuard::enter);
                layer
                    .prefill(
                        hidden,
                        residual,
                        proc_count,
                        seq.layer_states[i].as_mut(),
                        kv_cache,
                        effective_seq_len_start,
                        &mut seq.block_table,
                        &mut seq.disk_block_ids,
                        &mut seq.disk_last_offloaded_per_layer,
                        layer_kv_write_start,
                        &ctx,
                        stream,
                    )
                    .map_err(|e| anyhow::anyhow!("Prefill chunk layer {i} failed: {e}"))?;
            }
            if stages {
                stage_timer::mark("tail", stream);
            }
            if super::prefill_sync_diag() >= 2 {
                self.gpu.synchronize_device()?;
            }
            if readers_every_layer || self.dflash_capture_layers.contains(&i) {
                crate::layers::qwen4_hyper::pending::flush(self.gpu.as_ref(), stream)?;
            }
            // DFlash chunked-prefill capture.
            self.try_dflash_prefill_capture_layer(
                seq,
                i,
                effective_seq_len_start,
                proc_count,
                stream,
            )?;
            if let Some(lt0) = lt0 {
                self.gpu.synchronize(stream)?;
                layer_times.push(lt0.elapsed().as_micros());
            }
            // MLA diagnostic: per-layer hidden norm for Mistral (once per session)
            static CHUNK_DIAG_DONE: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if profile_now
                && self.config.model_type == "mistral"
                && !CHUNK_DIAG_DONE.load(std::sync::atomic::Ordering::Relaxed)
            {
                self.gpu.synchronize(stream)?;
                let rw = self.config.residual_width();
                let last_offset = (proc_count - 1) * rw * fp32;
                let norm_res = if fp32 == 4 {
                    self.readback_f32(hidden.offset(last_offset), rw)
                        .map(|(_, norm)| norm)
                } else {
                    self.readback_bf16(hidden.offset(last_offset), rw)
                        .map(|(_, norm)| norm)
                };
                if let Ok(norm) = norm_res {
                    tracing::info!(
                        "LAYER_NORM L{i}/{}: hidden_norm={norm:.4}",
                        self.layers.len()
                    );
                    if i == self.layers.len() - 1 {
                        CHUNK_DIAG_DONE.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                }
            }
            // Diagnostic: dump hidden state norm after first 4 and last 4 layers
            if profile_now && (i < 4 || i >= self.layers.len().saturating_sub(4)) {
                self.gpu.synchronize(stream)?;
                let rw = self.config.residual_width();
                let (_, norm) = if fp32 == 4 {
                    self.readback_f32(hidden, rw.min(64))?
                } else {
                    self.readback_bf16(hidden, rw.min(64))?
                };
                tracing::info!("L{i} hidden[0] norm={norm:.4}");
            }
            // Per-layer numerical-divergence dump (env-gated, zero overhead when
            // unset). `ATLAS_NEMO_DUMP=<dir>` writes the LAST token's full
            // post-layer residual-stream hidden vector for every layer as
            // headerless little-endian f32: `<dir>/atlas_L{i}.bin`. Overwrites
            // on every call so the final chunk's last token wins (methodology
            // §3 gotcha #5). Compared 1:1 against the HF CPU/GPU oracle.
            if is_last_chunk
                && let Ok(dir) = std::env::var("ATLAS_NEMO_DUMP")
                && !dir.is_empty()
            {
                self.gpu.synchronize(stream)?;
                // Rows of the residual stream are residual_width() wide (4*h for
                // the Qwen4 hyperconnection models); dump the whole last row.
                let rw = self.config.residual_width();
                let last_start = (proc_count - 1) * rw;
                let (vals, _) = if self.config.use_fp32_residual() {
                    self.readback_f32(hidden.offset(last_start * fp32), rw)?
                } else {
                    self.readback_bf16(hidden.offset(last_start * fp32), rw)?
                };
                let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::create_dir_all(&dir).ok();
                let path = std::path::Path::new(&dir).join(format!("atlas_L{i}.bin"));
                std::fs::write(&path, &bytes).ok();
                if i == self.layers.len() - 1 {
                    tracing::info!(
                        "ATLAS_NEMO_DUMP: wrote {} per-layer hidden \
                         vectors ({rw} f32 each) to {dir}",
                        self.layers.len()
                    );
                }
            }
            // Last-chunk diagnostic: log LAST token's hidden norm at every layer.
            if profile_now && is_last_chunk && proc_count > 1 && (chunk_start + chunk_len) > 16384 {
                self.gpu.synchronize(stream)?;
                let rw = self.config.residual_width();
                let last_start = (proc_count - 1) * rw;
                let (vals, norm) = if fp32 == 4 {
                    self.readback_f32(hidden.offset(last_start * fp32), rw.min(16))?
                } else {
                    self.readback_bf16(hidden.offset(last_start * fp32), rw.min(16))?
                };
                let lt = self.config.layer_type(i);
                tracing::warn!(
                    "DIAG L{i} ({lt:?}) last_tok_norm={norm:.4} first2={:.4?}",
                    &vals[..2.min(vals.len())]
                );
            }
        }
        crate::layers::qwen4_hyper::pending::flush(self.gpu.as_ref(), stream)?;
        // Deferred compact-MoE status words (ATLAS_QWEN4_PREFILL_MOE_STATUS_ASYNC): fail closed
        // before anything consumes this chunk's output (defer() auto-flushes if SLOTS fills up).
        crate::layers::moe::qwen4_moe_status::flush(self.gpu.as_ref(), stream)?;
        self.pp_stage_exit(hidden, proc_count, stream)?;
        // One OPROJ_AMAX_FUSED_CHUNK line per chunk while that selector is on (host counts only).
        crate::layers::qwen4_fp8_act_amax::oproj_chunk_receipt(effective_seq_len_start, proc_count);
        if stages {
            stage_timer::mark("final_flush", stream);
        }
        if stages && let Some(summary) = stage_timer::finish() {
            tracing::info!(
                "QWEN4_PREFILL_STAGES start={effective_seq_len_start} rows={proc_count} {summary}"
            );
        }
        if let Some(t0) = prefill_t0 {
            self.gpu.synchronize(stream)?;
            let total_us = t0.elapsed().as_micros();
            let mut indexed: Vec<(usize, u128)> = layer_times.iter().copied().enumerate().collect();
            indexed.sort_by_key(|x| std::cmp::Reverse(x.1));
            let top5: Vec<String> = indexed
                .iter()
                .take(5)
                .map(|(i, us)| format!("L{}={:.2}ms", i, *us as f64 / 1000.0))
                .collect();
            let path_label = if use_decode_path { "decode" } else { "prefill" };
            tracing::info!(
                "Prefill chunk {} tok (proc {}, {}): {:.1}ms total, top5: {}",
                chunk_len,
                proc_count,
                path_label,
                total_us as f64 / 1000.0,
                top5.join(", "),
            );
        }
        if let Some(receipt) = prefill_receipt {
            receipt.finish()?;
        }
        // Every layer of this chunk has captured; advance the DFlash ring so
        // the next chunk's capture starts at the right absolute cursor.
        self.update_dflash_ctx_len_after_prefill(seq, effective_seq_len_start, proc_count)?;
        Ok(())
    }

    /// `ATLAS_QWEN4_PREFILL_LAST_LAYER_CROP`: the crop the final layer of this
    /// chunk may apply, refused (full layer) while anything reads other rows
    /// of the final layer's output.
    fn prefill_b_last_layer_crop(
        &self,
        is_last_chunk: bool,
        proc_count: usize,
        start: usize,
        rows_after: usize,
        use_decode_path: bool,
        expert_major: bool,
        profile_now: bool,
    ) -> Result<Option<crate::layers::qwen4_prefill_moe::last_layer_crop::Crop>> {
        use crate::layers::qwen4_prefill_moe::last_layer_crop::{self, Decision, Request};
        // A crop left noted by a failed chunk must not suppress this chunk's readers.
        last_layer_crop::take_engaged();
        if !last_layer_crop::armed()? || self.config.pp_world_size > 1 {
            return Ok(None);
        }
        let Some(last) = self.layers.len().checked_sub(1) else {
            return Ok(None);
        };
        let env_set = |key: &str| std::env::var(key).is_ok_and(|v| !v.is_empty());
        let reader = [
            (expert_major, "expert_major"),
            (
                dflash_reads_final_layer(&self.dflash_capture_layers, last),
                "dflash_capture",
            ),
            (profile_now, "profile"),
            (env_set("ATLAS_PREFILL_HIDDEN_DUMP"), "hidden_dump"),
            (super::qgate_dump::qgate_dump_armed(), "qgate_dump"),
            (
                crate::model::env_diag::dump_ctx_hidden_path().is_some(),
                "ctx_hidden_dump",
            ),
        ]
        .into_iter()
        .find_map(|(reads, reason)| reads.then_some(reason));
        let request = Request {
            final_attention_layer: self.config.is_qwen4_exp()
                && self.config.layer_type(last) == atlas_core::config::LayerType::FullAttention,
            decode_path: use_decode_path,
            rows: proc_count,
            last_chunk: is_last_chunk,
            // The MTP last-K capture reads the prompt's last K final-layer rows.
            rows_after,
            lastk: if self.mtp_lastk_capture_active() {
                self.mtp_lastk_capacity
            } else {
                0
            },
            reader,
            tail_prepare_exact: crate::layers::qwen4_hyper::prefill_fused::level() == 2,
        };
        Ok(match last_layer_crop::decide(true, request) {
            Decision::Engage(crop) => Some(crop),
            Decision::Refused(reason) => {
                last_layer_crop::log_refusal(reason, proc_count, start);
                None
            }
            Decision::Off => None,
        })
    }
}

/// Whether DFlash prefill capture reads rows from the model's final layer
/// (`last_layer`), in which case `ATLAS_QWEN4_PREFILL_LAST_LAYER_CROP` must
/// leave the final layer uncropped.
fn dflash_reads_final_layer(dflash_capture_layers: &[usize], last_layer: usize) -> bool {
    dflash_capture_layers.contains(&last_layer)
}

#[cfg(test)]
mod last_layer_crop_dflash_tests {
    use super::dflash_reads_final_layer;
    use crate::layers::qwen4_prefill_moe::last_layer_crop::{
        Crop, Decision, HC_ROWS, MOE_ROWS, Request, decide,
    };

    #[test]
    fn dflash_capture_blocks_last_layer_crop_only_when_final_layer_is_captured() {
        let last = 47usize;
        // DFlash2 captures [5, 15, 25, 35, 45]; layer 47 is not captured.
        assert!(!dflash_reads_final_layer(&[5, 15, 25, 35, 45], last));
        assert!(!dflash_reads_final_layer(&[], last));
        assert!(dflash_reads_final_layer(&[5, 15, 25, 35, 47], last));

        let req_for = |taps: &[usize]| Request {
            final_attention_layer: true,
            decode_path: false,
            rows: 16_384,
            last_chunk: true,
            rows_after: 0,
            lastk: 0,
            reader: dflash_reads_final_layer(taps, last).then_some("dflash_capture"),
            tail_prepare_exact: true,
        };
        assert_eq!(
            decide(true, req_for(&[5, 15, 25, 35, 45])),
            Decision::Engage(Crop::Tail {
                hc_rows: HC_ROWS,
                moe_rows: MOE_ROWS,
                lastk: 0,
                rows_after: 0,
            })
        );
        assert_eq!(
            decide(true, req_for(&[5, 15, 25, 35, 47])),
            Decision::Refused("dflash_capture")
        );
    }
}
