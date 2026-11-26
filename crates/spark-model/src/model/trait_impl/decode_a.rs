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
    pub(crate) fn decode_dispatch(
        &self,
        token: u32,
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<DevicePtr> {
        crate::model::verify_failure::check()?;
        self.ensure_verify_sequence_idle(seq)?;
        seq.retire_committed_verify_handoff();
        // ATLAS_QWEN4_DENSE_FP8=1: the native MXFP8 decode arms only run
        // inside this single-token decode (inert when no layer has them).
        let _dense_fp8_scope = crate::layers::qwen4_dense_fp8::DecodeScope::enter();
        let rotary =
            crate::model::rotary_meta::SingleRotary::new(&seq.rotary_positions, seq.seq_len)?;
        if self.config.is_qwen4_exp() && seq.seq_len >= 2048 && !self.config.qwen4_qsa {
            anyhow::bail!(
                "Qwen4 QSA is disabled: exact full-attention fallback is limited to 2048 tokens"
            );
        }
        // ATLAS_SSM_H_FP16 stage 2: flip this sequence's h-state to FP16 before
        // any graph capture/replay below. Must be here and not in a layer — see
        // `ssm_h_to_f16_dispatch`. No-op unless the flag is set.
        self.ssm_h_to_f16_dispatch(seq)?;
        // Use backend's own stream (non-default, required for CUDA graph capture).
        let stream = self.gpu.default_stream();
        let hidden = self.buffers.hidden_states();
        let residual = self.buffers.residual();

        let mut kv_cache = self.kv_cache.lock();

        // ── Phase 1: Operations OUTSIDE graph (vary per token) ──

        // MLA models: zero buffers reused for Q_absorbed computation.
        // Without this, stale prefill data in expert_up_out / ssm_conv_out_f32 /
        // ssm_ba contaminates the absorbed attention → generic/wrong output.
        if self.config.kv_lora_rank > 0 {
            self.buffers.zero_all(self.gpu.as_ref(), stream)?;
        }

        // 1. Embedding lookup (PP stages > 0 receive hidden states via pp_stage_enter)
        if self.config.pp_rank == 0 {
            self.embed(token, hidden, stream)?;
        }
        let hidden_elem_bytes = if self.config.use_fp32_residual() {
            4
        } else {
            2
        };
        let persistent_row_bytes = self.config.residual_width() * hidden_elem_bytes;
        self.capture_k1_stage("embed", hidden, 1, persistent_row_bytes, stream)?;

        // 2. Pre-allocate KV cache blocks + upload attention metadata
        let bs = kv_cache.block_size();
        let blocks_needed = (seq.seq_len / bs) + 1;
        ensure_blocks_through_decode(
            seq,
            blocks_needed - 1,
            &mut kv_cache,
            self.prefix_cache.as_ref(),
            self.gpu.as_ref(),
            stream,
        )?;

        let meta_base = self
            .buffers
            .scratch()
            .offset(crate::layer::STEP_META_SCRATCH_OFFSET);
        let max_blocks = seq.block_table.len() as u32;

        let pos_val = rotary.position;
        let pos_bytes = pos_val.to_le_bytes();

        let block_idx = seq
            .physical_block_for(seq.seq_len / bs)
            .unwrap_or(self.dummy_kv_block);
        let global_slot = (block_idx as i64) * (bs as i64) + ((seq.seq_len % bs) as i64);
        let slot_bytes = global_slot.to_le_bytes();

        let actual_seq_len = (seq.seq_len + 1) as i32;
        let seq_len_bytes = actual_seq_len.to_le_bytes();

        let bt_i32: Vec<i32> = seq.block_table.iter().map(|&b| b as i32).collect();
        // SAFETY: `bt_i32` is a live initialized `Vec<i32>` whose byte length is
        // `size_of_val(bt_i32.as_slice())`; `u8` has alignment 1 and the slice
        // borrows `bt_i32` immutably only across `copy_h2d_group_fast`.
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

        let (positions_h, positions_w) =
            rotary.upload_axes(self.gpu.as_ref(), meta_base, stream)?;
        let attn_metadata = AttnMetadataDev {
            qwen4_qsa_required: seq.qwen4_qsa_required,
            positions: meta_base,
            positions_h,
            positions_w,
            slot: meta_base.offset(8),
            seq_len: meta_base.offset(16),
            block_table: meta_base.offset(256),
            max_blocks_per_seq: max_blocks,
            num_seqs: 1,
        };

        // CUDA graphs cannot capture NCCL all-reduce (it runs on a separate
        // stream) or cuStreamSynchronize calls. Suppress for EP and profile.
        // Re-enable graphs once FP8 calibration is frozen.
        //
        // Single atomic load below, reused for both the freeze decision and
        // the final use_graphs flag — earlier code did two Relaxed loads
        // for the same value (the second one read whatever the conditional
        // store wrote, but a local mirror gives the same outcome at half
        // the load cost).
        let mut suppress_graphs = self
            .suppress_graphs
            .load(std::sync::atomic::Ordering::Relaxed);
        if self.config.fp8_kv_calibration_tokens > 0
            && suppress_graphs
            && seq.seq_len > self.config.fp8_kv_calibration_tokens + 10
            && !crate::model::env_diag::diag_gemma4_enabled()
            // ATLAS_DUMP_HIDDEN: keep eager mode so CPU sync dump stays safe.
            && std::env::var("ATLAS_DUMP_HIDDEN").is_err()
            && !self.serial_batch_suppress_active()
        {
            self.suppress_graphs
                .store(false, std::sync::atomic::Ordering::Relaxed);
            suppress_graphs = false;
            tracing::info!("FP8 calibration frozen — re-enabling CUDA graphs");
        }
        // Phase 6.2.c — `--high-speed-swap` paths do host-side D2H + dequant
        // + per-step disk I/O which is illegal under CUDA graph capture
        // (cuStreamSynchronize fails with status 900 = CAPTURE_UNSUPPORTED).
        // Capture isn't a useful win for HSS anyway: per-layer launch overhead
        // is small relative to the per-step disk I/O on the critical path.
        let hss_engaged = kv_cache.config().cache_blocks_per_seq.is_some();
        let serial_debug_dump_all =
            std::env::var("ATLAS_SERIAL_DEBUG_DUMP_ALL").ok().as_deref() == Some("1");
        let serial_debug_dump = std::env::var("ATLAS_SERIAL_DEBUG_DUMP").is_ok()
            && (serial_debug_dump_all
                || std::env::var("ATLAS_SERIAL_DEBUG_SEQ_LEN")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
                    .is_none_or(|wanted| wanted == seq.seq_len));
        let decode_stages = crate::model::decode_stages::enabled();
        let qsa_graph_capture_allowed =
            super::decode_qwen4_graph_policy::plain_qsa_graph_capture_allowed(
                self.config.is_qwen4_exp(),
                seq.qwen4_qsa_required,
                seq.seq_len,
                crate::layers::qwen4_qsa::graph_safe_enabled(),
            );
        let layer_fnv = layer_fnv_enabled();
        let use_graphs = self.comm.is_none()
            && qsa_graph_capture_allowed
            && !decode_stages
            && !self.profile
            && !suppress_graphs
            && !hss_engaged
            && self.qwen4_ple.is_none()
            && !serial_debug_dump
            && !layer_fnv
            && crate::model::k1_stage_diag::graph_capture_allowed();

        // PLE performs host/NVMe work between layers 0 and 1, so it cannot
        // live inside a monolithic CUDA graph. The remaining layers are
        // address-stable for a fixed sequence slot, however. The segmented
        // path is qualified and default-on; setting the environment variable
        // to zero retains an explicit eager diagnostic control.
        let use_qwen4_ple_suffix_graph = self.config.is_qwen4_exp()
            && qsa_graph_capture_allowed
            && self.qwen4_ple.is_some()
            && self.comm.is_none()
            && !self.profile
            && !suppress_graphs
            && !hss_engaged
            && !serial_debug_dump
            && crate::model::k1_stage_diag::graph_capture_allowed()
            && !decode_stages
            && !layer_fnv
            && super::decode_qwen4_graph_policy::ple_suffix_graph_proposer_allowed(
                self.proposer_arm_kind(0),
                self.proposer_arm_kind(1),
                self.dflash_hidden_save.is_some(),
            )
            && CrossSeqStagingMode::from_staging(self.qwen4_cross_seq_staging.as_ref())
                .allows_ple_suffix_graph(seq.slot_idx)
            && std::env::var("ATLAS_QWEN4_PLE_SEGMENTED_GRAPHS")
                .ok()
                .as_deref()
                != Some("0");

        let ctx = ForwardContext {
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            attn_metadata: Some(attn_metadata),
            profile: self.profile,
            comm: self.comm_ref(),
            graph_capture: use_graphs,
            ddtree_parent_ids_dev: None,
            tree_aware_attn: None,
            ssm_multi_seq_ptr_table_override: None,
            self_spec_sparse_draft: None,
            ffn_defer: None,
        };

        // Materialize any deferred KEEP_LATE DeltaNet updates on this slot
        // BEFORE entering CUDA graph capture or replay (no-op single atomic
        // load when no rows are pending).
        self.flush_gdn_keep_late_slot(seq, false, stream)?;

        if use_qwen4_ple_suffix_graph {
            return self.decode_qwen4_ple_suffix_graph(
                token,
                hidden,
                residual,
                seq,
                &mut kv_cache,
                &ctx,
                stream,
            );
        }

        // Profile mode: use per-layer sync decode for timing breakdown.
        if self.profile && self.qwen4_ple.is_none() {
            return self.decode_profiled(token, hidden, residual, seq, &mut kv_cache, &ctx, stream);
        }

        // ── Phase 2: Try CUDA graph replay ──

        let mut graph_cache = if use_graphs {
            Some(self.decode_graph.lock())
        } else {
            None
        };

        // For batch=1, the captured graph works for any max_blocks because
        // max_blocks_per_seq is only used as block_table stride (seq_idx * stride),
        // and seq_idx=0 makes the stride irrelevant. All dynamic data (seq_len,
        // block_table, positions, slots) is read from device memory uploaded
        // before each graph replay.
        // SLOT-KEYED LOOKUP: only replay if this seq's slot matches a captured graph.
        if let Some(ref cache) = graph_cache
            && let Some(graph) = cache.get(&seq.slot_idx)
            && graph.0 != 0
        {
            self.gpu.launch_graph(*graph, stream)?;
            self.maybe_periodic_normalize_mamba_ssm(seq, stream);
            seq.tokens.push(token);
            seq.seq_len += 1;
            return Ok(self.decode_logits_ptr());
        }

        // ── Phase 3: Capture new CUDA graph (or run eagerly for EP) ──

        // `capture` is `Some` for exactly the span `begin_capture`..`end_capture`
        // below, whenever `use_graphs` is true (the only condition that begins
        // one). Every fallible call in between (the layer loop, final norm, LM
        // head) now drops this guard on an early `?` return instead of leaving
        // the stream permanently reported as capturing — see
        // `trait_impl::capture_guard` / `DEBUG_SYNC_AUDIT.md` (this was
        // `decode_a.rs:269,460`'s bare `begin_capture(stream)?` .. `end_capture(stream)?`
        // before this fix).
        let mut capture = if use_graphs {
            tracing::info!(
                "CUDA graph capture: starting for {} layers",
                self.layers.len()
            );
            Some(super::capture_guard::CaptureGuard::begin(
                self.gpu.as_ref(),
                stream,
                "decode_a_default",
            )?)
        } else {
            None
        };

        if decode_stages {
            spark_runtime::cublaslt::stage_timer::begin(stream);
        }
        self.pp_stage_enter(hidden, 1, stream)?;
        for (i, layer) in self.layers.iter().enumerate() {
            if i == 1
                && let Some(ple) = &self.qwen4_ple
            {
                ple.forward_token(
                    token,
                    &seq.tokens,
                    hidden,
                    seq.slot_idx,
                    seq.seq_len == 0,
                    self.gpu.as_ref(),
                    stream,
                )?;
                spark_runtime::cublaslt::stage_timer::set_group("");
                spark_runtime::cublaslt::stage_timer::mark("ple", stream);
            }
            if decode_stages {
                spark_runtime::cublaslt::stage_timer::set_group(if i == 0 {
                    "l0"
                } else {
                    match self.config.layer_type(i) {
                        atlas_core::config::LayerType::FullAttention => "attn",
                        _ => "ssm",
                    }
                });
            }
            layer.decode(
                hidden,
                residual,
                seq.layer_states[i].as_mut(),
                &mut kv_cache,
                seq.seq_len,
                &mut seq.block_table,
                &mut seq.disk_block_ids,
                &mut seq.disk_last_offloaded_per_layer,
                &ctx,
                stream,
            )?;
            self.capture_k1_stage(
                &format!("layer_{i:02}"),
                hidden,
                1,
                persistent_row_bytes,
                stream,
            )?;
            if !use_graphs {
                self.log_layer_fnv(
                    seq.seq_len,
                    i,
                    hidden,
                    residual,
                    persistent_row_bytes,
                    stream,
                )?;
            }
            // Offline K=gamma parity probe: capture the single-token hidden
            // at one requested absolute position so it can be diffed against
            // `ATLAS_KGAMMA_DEBUG_DUMP` row-for-row.  The seq-len selector is
            // important: without it, the first decode would win the one-shot
            // filenames and hide the actual divergent position.
            if serial_debug_dump {
                let h_bytes = self.config.hidden_size * 2;
                let path = if serial_debug_dump_all {
                    format!("/tmp/atlas_serial_seqlen{}_layer{i}.bin", seq.seq_len)
                } else {
                    format!("/tmp/atlas_serial_layer{i}.bin")
                };
                if !std::path::Path::new(&path).exists() {
                    let mut buf = vec![0u8; h_bytes];
                    self.gpu.synchronize(stream)?;
                    self.gpu.copy_d2h(hidden, &mut buf)?;
                    std::fs::write(path, buf)?;
                }
            }
            // DFlash 5-layer hidden capture (no-op when proposer is not DFlash).
            // Single-token decode: row 0 of `hidden_states()` holds the post-layer
            // activation. Cheap d2d when the layer index matches; otherwise a
            // hashmap-free position() probe over a 5-element vec.
            self.try_dflash_capture(i, 0, stream)?;
        }
        self.pp_stage_exit(hidden, 1, stream)?;
        if self.config.pp_rank > 0 {
            seq.tokens.push(token);
            seq.seq_len += 1;
            return Ok(self.decode_logits_ptr());
        }
        // MLA absorbed attention: defensive sync before final norm in eager
        // mode. Skipped under graph capture because cuStreamSynchronize is
        // illegal inside a capture region (CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED,
        // status 900). The sync is redundant when all kernels run on the same
        // stream — they are already sequenced — so the removal is safe for
        // both eager (retains sync as paranoia) and graph mode.
        if self.config.kv_lora_rank > 0 && !use_graphs {
            self.gpu.synchronize(stream)?;
        }

        let normed = self.buffers.norm_output();
        let final_input = self
            .qwen4_final_hidden(hidden, residual, stream)?
            .unwrap_or(hidden);
        self.final_norm_rows(final_input, normed, 1, stream)?;
        self.capture_k1_stage("final_norm", normed, 1, self.config.hidden_size * 2, stream)?;

        if serial_debug_dump {
            let h_bytes = self.config.hidden_size * 2;
            let path = if serial_debug_dump_all {
                format!("/tmp/atlas_serial_seqlen{}_final_norm.bin", seq.seq_len)
            } else {
                "/tmp/atlas_serial_final_norm.bin".to_string()
            };
            if !std::path::Path::new(&path).exists() {
                let mut buf = vec![0u8; h_bytes];
                self.gpu.synchronize(stream)?;
                self.gpu.copy_d2h(normed, &mut buf)?;
                std::fs::write(path, buf)?;
            }
        }

        // LM head reads from normed directly (no D2D copy needed)
        self.lm_head(normed, stream)?;
        if decode_stages {
            spark_runtime::cublaslt::stage_timer::set_group("");
            spark_runtime::cublaslt::stage_timer::mark("final_head", stream);
            crate::model::decode_stages::token_done(seq.seq_len);
        }
        let logits_elem_bytes = if self.use_fp32_logits { 4 } else { 2 };
        self.capture_k1_stage(
            "logits",
            self.decode_logits_ptr(),
            1,
            self.config.vocab_size * logits_elem_bytes,
            stream,
        )?;

        // Decode-step diagnostic for Gemma-4 degeneration analysis. Only fires
        // when ATLAS_DIAG_GEMMA4=1 (which also disables CUDA graphs upstream,
        // so the d2h sync below is safe). Reads top-5 tokens by logit so we
        // can see whether the LM head produced a near-tie or a confident bad
        // pick. (B4 — Creative haiku degeneration loop diagnostic.)
        if crate::model::env_diag::diag_gemma4_enabled() {
            self.gpu.synchronize(stream)?;
            let n_logits = self.config.vocab_size;
            // Read the buffer the LM head actually wrote to. With Gemma-4
            // dense the single-token decode lm_head produces FP32 in
            // `logits_fp32_buf`; the BF16 buffer would be all zeros there.
            let logit_vals: Vec<f32> = if self.use_fp32_logits {
                let mut buf = vec![0u8; n_logits * 4];
                if let Err(e) = self.gpu.copy_d2h(self.logits_fp32_buf, &mut buf) {
                    tracing::error!("ATLAS_DIAG_GEMMA4: copy_d2h(logits_fp32_buf): {e:#}");
                }
                buf.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            } else {
                let mut buf = vec![0u8; n_logits * 2];
                if let Err(e) = self.gpu.copy_d2h(self.buffers.logits(), &mut buf) {
                    tracing::error!("ATLAS_DIAG_GEMMA4: copy_d2h(logits BF16): {e:#}");
                }
                buf.chunks_exact(2)
                    .map(|c| {
                        let bits = u16::from_le_bytes([c[0], c[1]]);
                        f32::from_bits((bits as u32) << 16)
                    })
                    .collect()
            };
            let max = logit_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let min = logit_vals.iter().cloned().fold(f32::INFINITY, f32::min);
            let mut idx: Vec<usize> = (0..logit_vals.len()).collect();
            idx.sort_by(|&a, &b| {
                logit_vals[b]
                    .partial_cmp(&logit_vals[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let top5: Vec<(usize, f32)> = idx.iter().take(5).map(|&i| (i, logit_vals[i])).collect();
            tracing::warn!(
                "DIAG decode logits: max={max:.4} min={min:.4} prev_token={token} top5={top5:?}",
            );
        }

        if use_graphs {
            // `capture` is `Some` here: it was set exactly when `use_graphs`
            // was set above, and nothing in between took or replaced it.
            let graph = capture
                .take()
                .expect("use_graphs implies begin_capture ran above")
                .end()?;
            if graph.0 != 0 {
                tracing::info!(
                    "CUDA graph captured successfully for slot={} (handle={:?})",
                    seq.slot_idx,
                    graph.0
                );
                if let Some(ref mut cache) = graph_cache {
                    cache.insert(seq.slot_idx, graph);
                }
                self.decode_graph_sparse_regime
                    .lock()
                    .insert(seq.slot_idx, seq.seq_len >= 2_048);
                self.gpu.launch_graph(graph, stream)?;
            } else {
                tracing::warn!("CUDA graph capture returned null handle — running eagerly");
            }
            // If graph.0 == 0 (mock): operations already executed during capture
        }

        // Run periodic Mamba-2 SSM state normalization outside CUDA graph
        // capture so the `seq_len % 64 == 0` branch is never baked into (or
        // permanently omitted from) a captured graph.
        self.maybe_periodic_normalize_mamba_ssm(seq, stream);

        seq.tokens.push(token);
        seq.seq_len += 1;

        Ok(self.decode_logits_ptr())
    }

    fn maybe_periodic_normalize_mamba_ssm(&self, seq: &SequenceState, stream: u64) {
        // Mamba-2 has no per-token gate clamping (unlike GDN), so state can drift
        // from accumulated BF16 input truncation. Normalize every 64 tokens.
        if self.config.mamba_num_heads > 0
            && seq.seq_len > 0
            && seq.seq_len.is_multiple_of(64)
            && let Err(e) = self.normalize_ssm_states(seq, stream)
        {
            tracing::warn!("Periodic SSM state normalization failed: {e:#}");
        }
    }
}

#[inline]
fn layer_fnv_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_LAYER_FNV").ok().as_deref() == Some("1"))
}

impl TransformerModel {
    /// `ATLAS_LAYER_FNV=1` (eager decode only): FNV-1a of the hidden and
    /// residual rows after every layer, keyed by sequence position. Two runs of
    /// one prompt agree up to the first (position, layer) whose hash differs,
    /// which names the layer where a nondeterministic value first appears.
    fn log_layer_fnv(
        &self,
        seq_len: usize,
        layer: usize,
        hidden: DevicePtr,
        residual: DevicePtr,
        row_bytes: usize,
        stream: u64,
    ) -> Result<()> {
        if !layer_fnv_enabled() {
            return Ok(());
        }
        self.gpu.synchronize(stream)?;
        let fnv = |ptr: DevicePtr| -> Result<u64> {
            let mut buf = vec![0u8; row_bytes];
            self.gpu.copy_d2h(ptr, &mut buf)?;
            let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
            for &byte in &buf {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0100_0000_01b3);
            }
            Ok(hash)
        };
        let h = fnv(hidden)?;
        let r = fnv(residual)?;
        tracing::info!("LAYER_FNV pos={seq_len} layer={layer} hidden={h:016x} residual={r:016x}");
        Ok(())
    }
}

/// Whether cross-sequence concurrency staging (`C > 1`) is provisioned on this
/// server, gating which slots may capture the Qwen4 PLE segmented suffix graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CrossSeqStagingMode {
    SingleSeqOnly,
    MultiSeqActive,
}

impl CrossSeqStagingMode {
    #[inline]
    #[must_use]
    const fn from_staging<T>(staging: Option<&T>) -> Self {
        if staging.is_some() {
            Self::MultiSeqActive
        } else {
            Self::SingleSeqOnly
        }
    }

    /// Admit the Qwen4 PLE segmented suffix graph for `slot_idx`.
    ///
    /// At `C == 1` (`SingleSeqOnly`), all slots are admitted. On `C > 1`
    /// servers (`MultiSeqActive`), only `slot_idx == 0` is admitted so at most
    /// one suffix graph (~150 MiB) is ever resident — restoring graphed `K=1`
    /// bootstrap/fallback latency for solo `c=1` traffic without re-introducing
    /// the 16× (~2.4 GiB) multi-slot UMA graph-memory regression.
    #[inline]
    #[must_use]
    const fn allows_ple_suffix_graph(self, slot_idx: usize) -> bool {
        match self {
            Self::SingleSeqOnly => true,
            Self::MultiSeqActive => slot_idx == 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CrossSeqStagingMode;

    #[test]
    fn ple_suffix_graph_allows_all_slots_at_c1_and_only_slot_zero_at_multi_seq() {
        let single = CrossSeqStagingMode::from_staging(Option::<&()>::None);
        assert_eq!(single, CrossSeqStagingMode::SingleSeqOnly);
        for slot in [0, 1, 2, 7, 15, 63] {
            assert!(
                single.allows_ple_suffix_graph(slot),
                "C=1 must allow PLE suffix graph for slot={slot}"
            );
        }

        let multi = CrossSeqStagingMode::from_staging(Some(&()));
        assert_eq!(multi, CrossSeqStagingMode::MultiSeqActive);
        assert!(
            multi.allows_ple_suffix_graph(0),
            "C>1 must allow PLE suffix graph for slot 0 to preserve solo c=1 speed"
        );
        for slot in [1, 2, 3, 7, 15, 63] {
            assert!(
                !multi.allows_ple_suffix_graph(slot),
                "C>1 must refuse PLE suffix graph for slot={slot} to cap UMA graph memory to 1 slot"
            );
        }
    }
}
