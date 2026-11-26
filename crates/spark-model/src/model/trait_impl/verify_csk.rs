// SPDX-License-Identifier: AGPL-3.0-only

//! K=3 verify, c-batched per-step (Path B).
//!
//! Replaces c independent single-sequence K=3 verify forwards (each at
//! M=3 = c × 3 = 12 forwards per layer per step at c=4) with K=3
//! SEQUENTIAL c-batched single-step decodes (each at M=c = 4 = 3 batched
//! forwards per layer per step at c=4, K=3). Trades 12 → 3 FFN/SSM/attn
//! forward calls per layer per step, keeping the same M=c×K total work
//! but folding c-dimensional weight reuse onto the LPDDR5X-bandwidth-
//! bound GEMMs (which dominate the per-step profile per `forward_kgamma`
//! traces).
//!
//! ## Why per-step instead of a single M=c*K forward
//!
//! A true M=c*K verify (Path A) requires a multi-seq multi-token forward
//! that cross-cuts: the graph cache (currently single-seq slot-keyed), the
//! SSM state pool (per-seq h_state pointers plus per-step intermediates for
//! partial rollback), KV slot computation (per-seq block tables), the MTP
//! proposer state, EP broadcast format, grammar matcher state, and
//! `save_hidden_for_mtp` slot indices.
//!
//! Path B uses ONLY existing primitives:
//! - `decode_batch_dispatch` for the c-batched single-step decode (it
//!   already calls `decode_multi_seq` per layer which uses the multi-seq
//!   SSM kernels when `ATLAS_SSM_MULTI_SEQ_BATCHED=1` +
//!   `ATLAS_SSM_MULTI_SEQ_KERNEL=1`).
//! - Manual D2D copy from `ssm.h_state` → `ssm_pool.h_intermediate(layer,
//!   slot, k)` after each step k, so the existing
//!   `commit_verify_state_async(seq, num_accepted, K)` partial-rollback
//!   path (which reads `h_intermediate(layer, slot, num_accepted - 1)`)
//!   works unchanged.
//!
//! ## Per-step intermediate snapshot cost
//!
//! After each of K=3 steps, we copy `h_state` + `conv_state` to a pool
//! slot for every (seq, ssm_layer). For Qwen3.6-27B with 48 SSM layers
//! and c=4 seqs, that's 4 × 48 × 2 = 384 D2D copies per K-step:
//! `h_state = nv * vd * kd * 4 = 32 * 128 * 128 * 4 = 2 MiB`, while
//! `conv_state = 8192 * 4 * 4 = 128 KiB` (16× smaller). Total per step is
//! `4 × 48 × 2.125 MiB = 408 MiB/step` (~1.5 ms/step at 273 GB/s LPDDR5X;
//! ~3.0 ms across the 2 snapshotted steps of K=3).
//!
//! Future optimization: launch a single multi-layer-multi-seq h_state
//! snapshot kernel (one launch per step instead of 384) for ~10× lower
//! launch overhead.
//!
//! ## Current Status
//!
//! Opt-in experimental c-batched K=2/K=3 verify path. Multi-sequence decode
//! graph capture now exists in both forms:
//! - **Piecewise indirection-table graphs** (`decode_a2_piecewise.rs`,
//!   `ATLAS_MULTISEQ_GRAPHS=1`): refreshes device-resident SSM pointer tables
//!   pre-replay and captures slot-agnostic SSM/FFN segments.
//! - **Per-slot-tuple graphs** (`decode_a2.rs` `batch_decode_graphs`,
//!   `ATLAS_SSM_MULTI_SEQ_GRAPH=1`): keys captured graphs by `(sorted_slot_ids,
//!   padded_n)`.
//!
//! CSK remains opt-in alongside single-sequence graphed verify and native
//! batched verify (`verify_qwen4_batched.rs`).

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::{Result, bail};
use atlas_core::config::LayerType;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::types::TransformerModel;
use crate::layer::{LayerState, SsmLayerState};
use crate::layers::ops;
use crate::traits::{Model, SequenceState};

impl TransformerModel {
    /// K=3 verify across c sequences via sequential c-batched K-loop.
    ///
    /// See module doc + trait method doc on `Model::decode_verify_k3_batched_csk`
    /// for the full contract and side-effect semantics.
    pub(super) fn decode_verify_k3_batched_csk_dispatch(
        &self,
        tokens_per_seq: &[[u32; 3]],
        seqs: &mut [&mut SequenceState],
    ) -> Result<Vec<[u32; 3]>> {
        let c = seqs.len();
        let k_steps = 3usize;
        if tokens_per_seq.len() != c {
            bail!(
                "decode_verify_k3_batched_csk: tokens_per_seq.len()={} != seqs.len()={}",
                tokens_per_seq.len(),
                c
            );
        }
        if c < 2 {
            // Fall back to single-seq K=3 graphed verify (no batching to do).
            let mut out = Vec::with_capacity(c);
            for (i, seq) in seqs.iter_mut().enumerate() {
                let r = self.decode_verify_graphed_k3_dispatch(&tokens_per_seq[i], seq, 0)?;
                out.push(r);
            }
            return Ok(out);
        }

        // F62 (2026-04-27) SpecMamba dual-buffer pre-verify copy: copy
        // checkpoint → live h_state so the verify kernels can scratch-
        // write the canonical state. Must run BEFORE the first
        // intermediate snapshot.
        for seq in seqs.iter_mut() {
            self.pre_verify_copy_async(seq)?;
        }

        let stream = self.gpu.default_stream();
        let bf16 = 2usize;
        let vocab = self.config.vocab_size;

        // Out-of-band record of original seq_len so the scheduler can
        // truncate per-seq to `original_seq_len + accept_count[i] + 1`
        // after K steps. We don't return it because the scheduler holds
        // it directly; the contract on the trait method is that the
        // caller knows `original_seq_len[i] = seq.seq_len - K` after this
        // returns (since every seq was advanced exactly K positions).

        let mut results: Vec<[u32; 3]> = vec![[0u32; 3]; c];

        // ── Per-step c-batched K-loop ──
        //
        // Read back each step's argmax results from `buffers.scratch()[0..c*4)`
        // BEFORE the next iteration's `decode_batch_dispatch` overwrites
        // `buffers.scratch()` with token IDs and MoE routing metadata.
        let argmax_out = self.buffers.scratch();
        let mut step_argmax_buf = vec![0u8; c * 4];
        for k in 0..k_steps {
            // Gather the k-th token from each seq.
            let step_tokens: Vec<u32> = (0..c).map(|i| tokens_per_seq[i][k]).collect();

            // c-batched single-step decode. This advances each
            // seq.seq_len by 1 and pushes the token. Uses
            // `decode_multi_seq` per layer (batched projections + per-seq
            // SSM state advance, possibly via multi-seq kernels when
            // ATLAS_SSM_MULTI_SEQ_BATCHED=1 + ATLAS_SSM_MULTI_SEQ_KERNEL=1).
            //
            // Returns the BF16 logits buffer pointing at
            // self.buffers.logits()[0..c, vocab]. We need argmax per row.
            let logits_ptr = self.decode_batch_dispatch(&step_tokens, seqs, stream)?;

            for i in 0..c {
                let logits_i = logits_ptr.offset(i * vocab * bf16);
                let out_i = argmax_out.offset(i * 4);
                ops::argmax_bf16(
                    self.gpu.as_ref(),
                    self.argmax_kernel,
                    logits_i,
                    out_i,
                    vocab as u32,
                    stream,
                )?;
            }
            self.gpu
                .copy_d2h_on_stream(argmax_out, &mut step_argmax_buf, stream)?;
            for i in 0..c {
                let off = i * 4;
                results[i][k] = u32::from_le_bytes([
                    step_argmax_buf[off],
                    step_argmax_buf[off + 1],
                    step_argmax_buf[off + 2],
                    step_argmax_buf[off + 3],
                ]);
            }

            // ── Snapshot per-seq SSM state into intermediate pool slot k ──
            // `intermediates[t]` holds the recurrent state after processing
            // token `t` (0-indexed). Saving step `k`'s post-decode state into
            // `intermediate[k]` ensures `commit_verify_state_async` for
            // `num_accepted = m` (`1 <= m < K`) restores `intermediate[m - 1]`,
            // which is the exact post-step-`(m - 1)` state (e.g., `num_accepted=1`
            // restores `intermediate[0]` = post-step-0 state).
            //
            // `num_accepted == K` (full accept) consumes live `ssm.h_state`
            // directly, so step `k = K - 1` does not need an intermediate copy.
            if k + 1 < k_steps {
                self.snapshot_ssm_intermediates_per_seq(seqs, k, stream)?;
            }
        }

        Ok(results)
    }

    /// K=2 verify across c sequences via sequential c-batched K-loop.
    /// K=2 sibling of `decode_verify_k3_batched_csk_dispatch`.
    ///
    /// See `Model::decode_verify_k2_batched_csk` trait doc for the contract.
    pub(super) fn decode_verify_k2_batched_csk_dispatch(
        &self,
        tokens_per_seq: &[[u32; 2]],
        seqs: &mut [&mut SequenceState],
    ) -> Result<Vec<[u32; 2]>> {
        let c = seqs.len();
        let k_steps = 2usize;
        if tokens_per_seq.len() != c {
            bail!(
                "decode_verify_k2_batched_csk: tokens_per_seq.len()={} != seqs.len()={}",
                tokens_per_seq.len(),
                c
            );
        }
        if c < 2 {
            // Fall back to single-seq K=2 graphed verify (no batching to do).
            let mut out = Vec::with_capacity(c);
            for (i, seq) in seqs.iter_mut().enumerate() {
                let r = self.decode_verify_graphed_dispatch(&tokens_per_seq[i], seq, 0)?;
                out.push(r);
            }
            return Ok(out);
        }

        // F62 (2026-04-27) SpecMamba dual-buffer pre-verify copy: copy
        // checkpoint → live h_state so the verify kernels can scratch-
        // write the canonical state. Must run BEFORE the first
        // intermediate snapshot.
        for seq in seqs.iter_mut() {
            self.pre_verify_copy_async(seq)?;
        }

        let stream = self.gpu.default_stream();
        let bf16 = 2usize;
        let vocab = self.config.vocab_size;

        let mut results: Vec<[u32; 2]> = vec![[0u32; 2]; c];

        // ── Per-step c-batched K-loop ──
        for k in 0..k_steps {
            let step_tokens: Vec<u32> = (0..c).map(|i| tokens_per_seq[i][k]).collect();

            // c-batched single-step decode (see K=3 sibling for details).
            let logits_ptr = self.decode_batch_dispatch(&step_tokens, seqs, stream)?;

            // Argmax per seq.
            let argmax_out = self.buffers.scratch();
            for i in 0..c {
                let logits_i = logits_ptr.offset(i * vocab * bf16);
                let out_i = argmax_out.offset(i * 4);
                ops::argmax_bf16(
                    self.gpu.as_ref(),
                    self.argmax_kernel,
                    logits_i,
                    out_i,
                    vocab as u32,
                    stream,
                )?;
            }
            self.gpu.synchronize(stream)?;
            let mut argmax_buf = vec![0u8; c * 4];
            self.gpu.copy_d2h(argmax_out, &mut argmax_buf)?;
            for i in 0..c {
                let off = i * 4;
                let v = u32::from_le_bytes([
                    argmax_buf[off],
                    argmax_buf[off + 1],
                    argmax_buf[off + 2],
                    argmax_buf[off + 3],
                ]);
                results[i][k] = v;
            }

            // Snapshot per-seq SSM state into intermediate pool slot k for
            // partial-rollback semantics (matches K=3 logic). Skip on last
            // step k=K-1 because full-accept (num_accepted=K) reads
            // ssm.h_state directly.
            if k + 1 < k_steps {
                self.snapshot_ssm_intermediates_per_seq(seqs, k, stream)?;
            }
        }

        Ok(results)
    }

    /// D2D copy each seq's live `ssm.h_state` and `ssm.conv_state` into
    /// the corresponding `ssm_pool.h_intermediate(layer, slot, k)` /
    /// `conv_intermediate(layer, slot, k)`. One launch per (seq, layer)
    /// × 2 (h + conv) = 2 × c × num_ssm_layers launches per step.
    ///
    /// Future opt: collapse into a single multi-seq-multi-layer kernel.
    fn snapshot_ssm_intermediates_per_seq(
        &self,
        seqs: &mut [&mut SequenceState],
        inter_slot: usize,
        stream: u64,
    ) -> Result<()> {
        let nv = self.config.linear_num_value_heads;
        let vd = self.config.linear_value_head_dim;
        let kd = self.config.linear_key_head_dim;
        let nk = self.config.linear_num_key_heads;
        let h_bytes = nv * vd * kd * 4; // FP32
        let conv_dim = nk * kd * 2 + nv * vd;
        let d_conv = self.config.linear_conv_kernel_dim;
        let conv_bytes = conv_dim * d_conv * 4; // FP32

        for seq in seqs.iter_mut() {
            let slot = seq.slot_idx;
            let mut ssm_layer_idx = 0usize;
            for (layer_i, layer_state) in seq.layer_states.iter_mut().enumerate() {
                if self.config.layer_type(layer_i) != LayerType::LinearAttention {
                    continue;
                }
                let ssm = layer_state
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {layer_i}"))?;

                let h_inter = self
                    .ssm_pool
                    .h_intermediate(ssm_layer_idx, slot, inter_slot);
                let conv_inter = self
                    .ssm_pool
                    .conv_intermediate(ssm_layer_idx, slot, inter_slot);
                self.gpu
                    .copy_d2d_async(ssm.h_state, h_inter, h_bytes, stream)?;
                self.gpu
                    .copy_d2d_async(ssm.conv_state, conv_inter, conv_bytes, stream)?;

                ssm_layer_idx += 1;
            }
        }
        Ok(())
    }
}
