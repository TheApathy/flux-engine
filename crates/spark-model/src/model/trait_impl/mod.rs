// SPDX-License-Identifier: AGPL-3.0-only

//! `impl Model for TransformerModel` — trait impl delegating phase dispatch to
//! `<method>_dispatch` helpers across sibling files while implementing small
//! accessors and DDTree/verify coordination inline.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, HostToDeviceCopy};
use spark_runtime::kv_cache::PagedKvCache;

use super::types::{PinnedMetaStaging, TransformerModel};
use crate::layer::{AttnMetadataDev, LayerState};
use crate::speculative::DraftProposer;
use crate::traits::{ChunkedPrefillPageMetadata, Model, PrefillSlice, SequenceState};
use crate::weight_map::{DenseWeight, MtpWeights};

mod async_chkpt;
pub(crate) mod capture_guard;
pub(crate) mod commit_cache;
pub(crate) mod commit_plan;
mod decode_a;
mod decode_a2;
mod decode_a2_piecewise;
mod decode_b;
mod decode_b2;
mod decode_qwen4_capability;
mod decode_qwen4_graph_catalog;
mod decode_qwen4_graph_policy;
mod decode_qwen4_native_batch;
mod decode_qwen4_ple_graph;
mod decode_qwen4_route;
mod decode_serial_batch;
pub(crate) mod device_topk;
mod ep_misc;
mod k1_stage_capture;
mod meta;
pub(crate) mod mixed_qwen4_piggyback;
mod mixed_scratch;
mod ple_prefetch_slots;
mod prefill_a;
mod prefill_b;
mod prefill_c;
mod prefill_d;
mod prefix_hit;
mod sequence;
mod sequence_graph_cleanup;
mod speculative;
mod verify_a;
mod verify_b;
mod verify_c;
mod verify_c2;
mod verify_csk;
mod verify_d;
mod verify_dflash_batched;
mod verify_dflash_output;
mod verify_graph_receipts;
pub(in crate::model) mod verify_qwen4_batched;

impl Model for TransformerModel {
    fn prepare_vision_embed(&self, images: &[(Vec<f32>, usize, usize)]) -> Result<()> {
        crate::model::verify_failure::check()?;
        self.ensure_verify_healthy()?;
        self.prepare_vision_embed_dispatch(images)
    }
    fn prefill(&self, tokens: &[u32], seq: &mut SequenceState, stream: u64) -> Result<DevicePtr> {
        crate::model::verify_failure::check()?;
        self.ensure_verify_healthy()?;
        self.ep_head_sync_for_prefill(seq, 0)?;
        self.prefill_dispatch(tokens, seq, stream)
    }
    fn prefill_chunk(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_start: usize,
        chunk_len: usize,
        is_last_chunk: bool,
        stream: u64,
    ) -> Result<DevicePtr> {
        crate::model::verify_failure::check()?;
        self.ensure_verify_healthy()?;
        self.ep_head_sync_for_prefill(seq, chunk_start)?;
        self.prefill_chunk_dispatch(tokens, seq, chunk_start, chunk_len, is_last_chunk, stream)
    }
    fn supports_prefill_twophase(&self) -> bool {
        // Qwen4's fast-prefill lineage is request scoped and must stay on the
        // guarded chunked path. Gate it before the scheduler enters a method
        // whose post-effect errors cannot safely be replayed.
        !self.config.is_qwen4_exp()
    }
    fn prefill_twophase(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_size: usize,
        stream: u64,
    ) -> Result<DevicePtr> {
        crate::model::verify_failure::check()?;
        self.ensure_verify_healthy()?;
        self.prefill_twophase_dispatch(tokens, seq, chunk_size, stream)
    }
    fn resume_conversation(&self, seq: &mut SequenceState, prompt: &[u32]) -> Result<usize> {
        self.qwen4_seqstate_cache_resume(prompt, seq)
    }
    fn decode(&self, token: u32, seq: &mut SequenceState, _stream: u64) -> Result<DevicePtr> {
        crate::model::verify_failure::check()?;
        self.ensure_verify_healthy()?;
        self.qwen4_seqstate_cache_save_if_due(seq);
        self.qwen4_seqstate_roundtrip_if_due(seq)?;
        self.ep_head_sync_for_decode(seq)?;
        self.decode_dispatch(token, seq, _stream)
    }
    fn decode_batch(
        &self,
        tokens: &[u32],
        seqs: &mut [&mut SequenceState],
        stream: u64,
    ) -> Result<DevicePtr> {
        crate::model::verify_failure::check()?;
        self.ensure_verify_healthy()?;
        for seq in seqs.iter_mut() {
            self.qwen4_seqstate_cache_save_if_due(seq);
            self.qwen4_seqstate_roundtrip_if_due(seq)?;
        }
        self.decode_batch_dispatch(tokens, seqs, stream)
    }
    fn mixed_forward(
        &self,
        decode_tokens: &[u32],
        decode_seqs: &mut [&mut SequenceState],
        prefill_tokens: &[u32],
        prefill_seq: &mut SequenceState,
        prefill_chunk_start: usize,
        prefill_chunk_len: usize,
        prefill_is_last: bool,
        stream: u64,
    ) -> Result<crate::traits::MixedForwardResult> {
        crate::model::verify_failure::check()?;
        self.ensure_verify_healthy()?;
        self.mixed_forward_dispatch(
            decode_tokens,
            decode_seqs,
            prefill_tokens,
            prefill_seq,
            prefill_chunk_start,
            prefill_chunk_len,
            prefill_is_last,
            stream,
        )
    }

    /// Model-level batched prefill. The concrete dispatch owns every safe
    /// pre-effect sequential fallback; once it begins batched work, errors
    /// must propagate so a partially-mutated frame is never replayed.
    fn prefill_batch_chunk(
        &self,
        streams: &mut [PrefillSlice<'_>],
        stream: u64,
    ) -> Result<Vec<DevicePtr>> {
        crate::model::verify_failure::check()?;
        self.ensure_verify_healthy()?;
        self.prefill_batch_chunk_dispatch(streams, stream)
    }
    fn mixed_forward_batch(
        &self,
        decode_tokens: &[u32],
        decode_seqs: &mut [&mut SequenceState],
        prefill_streams: &mut [PrefillSlice<'_>],
        stream: u64,
    ) -> Result<crate::traits::MixedBatchResult> {
        crate::model::verify_failure::check()?;
        self.ensure_verify_healthy()?;
        // The default trait path mutates decode state before it reaches the
        // guarded prefill entry. Multi-chunk is a C1-only experiment, so keep
        // the whole family suppressed across both halves of an N>1 mixed
        // dispatch and retain the incumbent decode-then-prefill ordering.
        let multi_chunk = crate::layers::qwen4_prefill_moe::multi_chunk_selected()?;
        if multi_chunk {
            for slice in prefill_streams.iter_mut() {
                crate::layers::qwen4_prefill_moe::attn16::mark_ordinary_for_selected_route(
                    &mut slice.seq.qwen4_prefill_lineage,
                    slice.prompt_tokens.len(),
                );
            }
        }
        let _multi_chunk_suppressed =
            multi_chunk.then(crate::layers::qwen4_prefill_moe::attn16::suppress_family);
        let decode_logits = if decode_tokens.is_empty() {
            DevicePtr::NULL
        } else {
            self.decode_batch_dispatch(decode_tokens, decode_seqs, stream)?
        };
        let prefill_logits = self.prefill_batch_chunk_dispatch(prefill_streams, stream)?;
        Ok(crate::traits::MixedBatchResult {
            decode_logits,
            prefill_logits,
        })
    }
    fn vocab_size(&self) -> usize {
        self.vocab_size_dispatch()
    }
    fn high_speed_swap_dims(&self) -> Option<spark_storage::ModelDims> {
        self.high_speed_swap_dims_dispatch()
    }
    fn normalize_ssm_states(&self, seq: &SequenceState, stream: u64) -> Result<()> {
        crate::model::verify_failure::check()?;
        self.normalize_ssm_states_dispatch(seq, stream)
    }
    fn finish_deferred_prefill_tail(
        &self,
        seq: &mut SequenceState,
        device_first_token: Option<u32>,
        suppress_ids: &[u32],
    ) -> Result<()> {
        crate::model::verify_failure::check()?;
        self.k8a_finish_deferred_lastk(seq, device_first_token, suppress_ids)
    }
    fn bind_gpu_to_thread(&self) -> Result<()> {
        self.bind_gpu_to_thread_dispatch()
    }
    fn alloc_sequence(&self) -> Result<SequenceState> {
        crate::model::verify_failure::check()?;
        self.ensure_verify_healthy()?;
        self.alloc_sequence_dispatch()
    }
    fn begin_prefill_admission(&self, prompt_tokens: &[u32], first_chunk_len: usize) {
        if crate::model::verify_failure::check().is_err() {
            return;
        }
        self.begin_ple_admission_prefetch(prompt_tokens, first_chunk_len);
    }
    fn begin_prefill_queue_prefetch(&self, prompt_tokens: &[u32], first_chunk_len: usize) {
        if crate::model::verify_failure::check().is_err() {
            return;
        }
        self.begin_ple_queue_prefetch(prompt_tokens, first_chunk_len);
    }
    fn warm_ple_draft_ahead(&self, tokens: &[u32]) {
        if crate::model::verify_failure::check().is_err() {
            return;
        }
        self.warm_ple_draft_ahead_dispatch(tokens);
    }
    fn warm_ple_draft_ahead_seqs(&self, seqs: &[(&[u32], &[u32])]) {
        if crate::model::verify_failure::check().is_err() {
            return;
        }
        self.warm_ple_draft_ahead_seqs_dispatch(seqs);
    }
    fn alloc_sequence_on(&self, stream: u64) -> Result<SequenceState> {
        crate::model::verify_failure::check()?;
        self.ensure_verify_healthy()?;
        self.alloc_sequence_on_dispatch(stream)
    }
    fn copy_logits_to_host(&self, logits_ptr: DevicePtr, dst: &mut [u8]) -> Result<()> {
        crate::model::verify_failure::check()?;
        self.copy_logits_to_host_dispatch(logits_ptr, dst)
    }
    fn logits_ptr_is_fp32(&self, logits_ptr: DevicePtr) -> bool {
        self.logits_ptr_is_fp32_dispatch(logits_ptr)
    }
    fn logits_buffer_ptr(&self) -> DevicePtr {
        self.logits_buffer_ptr_dispatch()
    }
    fn argmax_on_device(&self, logits_ptr: DevicePtr, _stream: u64) -> Result<u32> {
        crate::model::verify_failure::check()?;
        self.argmax_on_device_dispatch(logits_ptr, _stream)
    }
    fn argmax_batch(&self, logits_ptr: DevicePtr, n: usize, _stream: u64) -> Result<Vec<u32>> {
        crate::model::verify_failure::check()?;
        self.argmax_batch_dispatch(logits_ptr, n, _stream)
    }
    fn device_topk_supported(&self) -> bool {
        self.topk_candidates_kernel.0 != 0
            && self.f2_sum_kernel.0 != 0
            && self.topk_candidates_buf.0 != 0
    }
    fn topk_candidates(
        &self,
        logits_row: DevicePtr,
        temperature: f32,
        m: usize,
    ) -> Result<Vec<(u32, f32)>> {
        crate::model::verify_failure::check()?;
        self.topk_candidates_dispatch(logits_row, temperature, m)
    }
    fn edited_sum_exp(
        &self,
        logits_row: DevicePtr,
        edits: &[(u32, spark_runtime::sampler::f2_certify::PreF2Edit)],
    ) -> Result<spark_runtime::sampler::f2_certify::SumExpStats> {
        crate::model::verify_failure::check()?;
        self.edited_sum_exp_dispatch(logits_row, edits)
    }
    fn sample_gumbel_verify_rows_batched(
        &self,
        logits_base: DevicePtr,
        vocab_stride: usize,
        reqs: &[spark_runtime::sampler::device_topk::GumbelVerifyRowSpec],
    ) -> Result<Option<Vec<u32>>> {
        crate::model::verify_failure::check()?;
        self.sample_gumbel_verify_rows_batched_dispatch(logits_base, vocab_stride, reqs)
    }
    fn stage_next_verify_gumbel_rows(
        &self,
        reqs: &[spark_runtime::sampler::device_topk::GumbelVerifyRowSpec],
    ) {
        *self.staged_verify_gumbel_rows.lock() = if reqs.is_empty() {
            None
        } else {
            Some(reqs.to_vec())
        };
        *self.in_graph_gumbel_verify_tokens.lock() = None;
    }
    fn take_in_graph_gumbel_verify_tokens(&self) -> Option<Vec<u32>> {
        self.in_graph_gumbel_verify_tokens.lock().take()
    }
    fn gpu_rejection_verify_available(&self) -> bool {
        self.target_rejection_verify_kernel.0 != 0
            && !self.target_rejection_verify_buf.is_null()
            && !self.target_rejection_q_buf.is_null()
    }
    fn sample_rejection_verify_rows_batched(
        &self,
        logits_base: DevicePtr,
        vocab_stride: usize,
        reqs: &[crate::model::trait_impl::device_topk::RejectionVerifyRowSpec],
    ) -> Result<Option<Vec<crate::model::trait_impl::device_topk::PackedRejectionWord>>> {
        crate::model::verify_failure::check()?;
        self.sample_rejection_verify_rows_batched_dispatch(logits_base, vocab_stride, reqs)
    }
    fn stage_next_verify_rejection_rows(
        &self,
        reqs: &[crate::model::trait_impl::device_topk::RejectionVerifyRowSpec],
    ) {
        *self.staged_verify_rejection_rows.lock() = if reqs.is_empty() {
            None
        } else {
            Some(reqs.to_vec())
        };
        *self.in_graph_rejection_verify_words.lock() = None;
    }
    fn take_in_graph_rejection_verify_words(
        &self,
    ) -> Option<Vec<crate::model::trait_impl::device_topk::PackedRejectionWord>> {
        self.in_graph_rejection_verify_words.lock().take()
    }
    fn hidden_after_norm(&self) -> DevicePtr {
        self.hidden_after_norm_dispatch()
    }
    fn decode_verify(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<Vec<u32>> {
        crate::model::verify_failure::check()?;
        self.begin_verify_frame(seq, tokens.len())?;
        self.decode_verify_dispatch(tokens, seq, stream)
    }
    fn checkpoint_ssm_states(&self, seq: &mut SequenceState) -> Result<()> {
        crate::model::verify_failure::check()?;
        self.checkpoint_ssm_states_dispatch(seq)
    }
    fn rollback_ssm_states(&self, seq: &mut SequenceState, num_accepted: usize) -> Result<()> {
        crate::model::verify_failure::check()?;
        self.rollback_ssm_states_dispatch(seq, num_accepted)
    }
    fn has_ssm_layers(&self) -> bool {
        self.ssm_pool.num_ssm_layers > 0
    }
    fn decode_rollback_ring_slots(&self) -> usize {
        if self.ssm_snapshots.decode_rollback_enabled() {
            self.ssm_snapshots.decode_ring_slots
        } else {
            0
        }
    }
    fn save_decode_ssm_snapshot(&self, seq: &SequenceState, ring_slot: usize) -> Result<()> {
        crate::model::verify_failure::check()?;
        self.save_decode_ssm_snapshot_dispatch(seq, ring_slot)
    }
    fn restore_decode_ssm_snapshot(&self, seq: &SequenceState, ring_slot: usize) -> Result<()> {
        crate::model::verify_failure::check()?;
        self.restore_decode_ssm_snapshot_dispatch(seq, ring_slot)
    }
    fn generate_speculative(
        &self,
        prompt_tokens: &[u32],
        params: &spark_runtime::sampler::SamplingParams,
        num_drafts: usize,
    ) -> Result<crate::engine::GenerateResult> {
        crate::model::verify_failure::check()?;
        self.ensure_verify_healthy()?;
        self.generate_speculative_dispatch(prompt_tokens, params, num_drafts)
    }
    fn has_proposer(&self) -> bool {
        self.has_proposer_dispatch()
    }
    fn speculation_forced_disabled(&self) -> bool {
        self.speculation_forced_disabled
    }
    fn proposer_is_dflash(&self) -> bool {
        self.active_proposer().is_some_and(|p| p.is_dflash())
    }
    fn proposer_target_token_authority(&self) -> crate::speculative::TargetTokenAuthority {
        self.active_proposer().map_or(
            crate::speculative::TargetTokenAuthority::ServingPolicy,
            |p| p.target_token_authority(),
        )
    }
    fn dflash_verify_capacity_k(&self) -> Option<usize> {
        self.active_proposer()
            .filter(|proposer| proposer.is_dflash())
            .map(|_| self.ddtree_parent_ids_capacity)
    }
    fn dflash_sampled_proposal_width(&self, requested: usize) -> Result<usize> {
        self.dflash_sampled_proposal_width_dispatch(requested)
    }
    fn dflash_draft_sample_method(&self) -> crate::dflash_sampling::DflashDraftSampleMethod {
        self.active_proposer().map_or(
            crate::dflash_sampling::DflashDraftSampleMethod::Greedy,
            |proposer| proposer.dflash_draft_sample_method(),
        )
    }
    fn proposer_arm_count(&self) -> usize {
        TransformerModel::proposer_arm_count(self)
    }
    fn proposer_arm_kind(&self, arm: u8) -> Option<crate::speculative::DraftProposerKind> {
        TransformerModel::proposer_arm_kind(self, arm)
    }
    fn proposer_arm(&self) -> u8 {
        TransformerModel::proposer_arm(self)
    }
    fn set_proposer_arm(&self, arm: u8) -> u8 {
        TransformerModel::set_proposer_arm(self, arm)
    }
    fn has_self_speculative(&self) -> bool {
        self.has_self_speculative_dispatch()
    }
    fn supports_ngram_speculative(&self) -> bool {
        !self.config.is_qwen4_exp()
    }
    fn decode_draft(&self, token: u32, seq: &mut SequenceState, stream: u64) -> Result<DevicePtr> {
        crate::model::verify_failure::check()?;
        self.ensure_verify_healthy()?;
        self.decode_draft_dispatch(token, seq, stream)
    }
    fn decode_draft_sparse(
        &self,
        token: u32,
        thresh_frac: f32,
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<DevicePtr> {
        crate::model::verify_failure::check()?;
        self.ensure_verify_healthy()?;
        self.decode_draft_sparse_dispatch(token, thresh_frac, seq, stream)
    }
    fn cache_sequence(&self, seq: &SequenceState) {
        if crate::model::verify_failure::check().is_err() {
            return;
        }
        if let Err(error) = self.ensure_verify_sequence_idle(seq) {
            tracing::error!("cache_sequence refused unresolved verify ownership: {error:#}");
            return;
        }
        self.cache_sequence_dispatch(seq)
    }
    fn free_sequence(&self, seq: &mut SequenceState) -> Result<()> {
        self.free_owned_sequence(seq)
    }
    fn qwen4_target_state_receipt(
        &self,
        seq: &SequenceState,
    ) -> Result<Option<crate::model::qwen4_target_state_receipt::Qwen4TargetStateReceipt>> {
        crate::model::qwen4_target_state_receipt::capture(self, seq)
    }
    fn decode_verify_graphed(
        &self,
        tokens: &[u32; 2],
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<[u32; 2]> {
        crate::model::verify_failure::check()?;
        self.ep_head_sync_for_verify(seq)?;
        self.begin_verify_frame(seq, tokens.len())?;
        self.decode_verify_graphed_dispatch(tokens, seq, _stream)
    }
    fn decode_verify_graphed_k3(
        &self,
        tokens: &[u32; 3],
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<[u32; 3]> {
        crate::model::verify_failure::check()?;
        self.ep_head_sync_for_verify(seq)?;
        self.begin_verify_frame(seq, tokens.len())?;
        self.decode_verify_graphed_k3_dispatch(tokens, seq, _stream)
    }
    fn decode_verify_k3_batched_csk(
        &self,
        tokens_per_seq: &[[u32; 3]],
        seqs: &mut [&mut SequenceState],
        _stream: u64,
    ) -> Result<Vec<[u32; 3]>> {
        crate::model::verify_failure::check()?;
        if self.config.is_qwen4_exp() {
            anyhow::ensure!(
                tokens_per_seq.len() == seqs.len() && seqs.len() <= 1,
                "Qwen4 multi-sequence speculative verify is not qualified"
            );
            return match seqs.first_mut() {
                Some(seq) => Ok(vec![self.decode_verify_graphed_k3(
                    &tokens_per_seq[0],
                    seq,
                    _stream,
                )?]),
                None => Ok(Vec::new()),
            };
        }
        self.decode_verify_k3_batched_csk_dispatch(tokens_per_seq, seqs)
    }
    fn decode_verify_k2_batched_csk(
        &self,
        tokens_per_seq: &[[u32; 2]],
        seqs: &mut [&mut SequenceState],
        _stream: u64,
    ) -> Result<Vec<[u32; 2]>> {
        crate::model::verify_failure::check()?;
        if self.config.is_qwen4_exp() {
            anyhow::ensure!(
                tokens_per_seq.len() == seqs.len() && seqs.len() <= 1,
                "Qwen4 multi-sequence speculative verify is not qualified"
            );
            return match seqs.first_mut() {
                Some(seq) => Ok(vec![self.decode_verify_graphed(
                    &tokens_per_seq[0],
                    seq,
                    _stream,
                )?]),
                None => Ok(Vec::new()),
            };
        }
        self.decode_verify_k2_batched_csk_dispatch(tokens_per_seq, seqs)
    }
    fn decode_verify_graphed_k4(
        &self,
        tokens: &[u32; 4],
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<[u32; 4]> {
        crate::model::verify_failure::check()?;
        self.ep_head_sync_for_verify(seq)?;
        self.begin_verify_frame(seq, tokens.len())?;
        self.decode_verify_graphed_k4_dispatch(tokens, seq, _stream)
    }
    fn decode_verify_dflash_batched(
        &self,
        tokens_per_seq: &[Vec<u32>],
        seqs: &mut [&mut SequenceState],
        _stream: u64,
    ) -> Result<Vec<Vec<u32>>> {
        crate::model::verify_failure::check()?;
        if self.config.is_qwen4_exp() {
            anyhow::ensure!(
                tokens_per_seq.len() == seqs.len() && seqs.len() <= 1,
                "Qwen4 multi-sequence speculative verify is not qualified"
            );
            return match seqs.first_mut() {
                Some(seq) => Ok(vec![self.decode_verify(
                    &tokens_per_seq[0],
                    seq,
                    _stream,
                )?]),
                None => Ok(Vec::new()),
            };
        }
        self.decode_verify_dflash_batched_dispatch(tokens_per_seq, seqs)
    }
    fn decode_verify_qwen4_batched(
        &self,
        tokens_per_seq: &[Vec<u32>],
        seqs: &mut [&mut SequenceState],
    ) -> Result<Vec<Vec<u32>>> {
        self.decode_verify_qwen4_batched_dispatch(tokens_per_seq, seqs)
    }
    fn begin_verify_qwen4_batch(&self, seqs: &mut [&mut SequenceState], k: usize) -> Result<()> {
        self.begin_owned_verify_batch(seqs, k)
    }
    fn end_verify_qwen4_batch_frame(&self, seqs: &[&SequenceState]) -> Result<()> {
        self.end_owned_verify_batch_frame(seqs)
    }
    fn finish_verify_qwen4_batch(&self, seqs: &[&SequenceState]) -> Result<()> {
        self.finish_owned_verify_batch(seqs)
    }
    fn abort_verify_qwen4_batch(&self, seqs: &mut [&mut SequenceState]) -> Result<()> {
        self.abort_owned_verify_batch(seqs)
    }
    fn retire_verify_qwen4_batch_terminal(&self, seq: &SequenceState) -> Result<()> {
        self.retire_owned_batch_terminal(seq)
    }
    fn preflight_verify_qwen4_batched(
        &self,
        tokens_per_seq: &[Vec<u32>],
        seqs: &[&SequenceState],
    ) -> Result<()> {
        self.qwen4_cross_seq_admission(tokens_per_seq, seqs)
            .map(|_| ())
    }
    fn qwen4_batched_k5_armed(&self) -> bool {
        self.qwen4_cross_seq_staging.is_some()
    }
    fn save_hidden_for_mtp_sequence_in_frame(
        &self,
        row_base: usize,
        token_idx: usize,
        seq: &SequenceState,
        stream: u64,
    ) -> Result<()> {
        self.save_hidden_for_mtp_sequence_in_frame_dispatch(row_base, token_idx, seq, stream)
    }
    fn retire_dflash_frame_captures(
        &self,
        row_base: usize,
        num_accepted: usize,
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<()> {
        self.retire_dflash_frame_captures_impl(row_base, num_accepted, seq, stream)
    }
    fn retire_dflash_flat_batch_captures(
        &self,
        row_base: usize,
        num_accepted: usize,
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<()> {
        self.retire_dflash_frame_captures_from_save(row_base, num_accepted, seq, stream)
    }
    fn decode_verify_graphed_kgamma(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<Vec<u32>> {
        crate::model::verify_failure::check()?;
        self.ep_head_sync_for_verify(seq)?;
        self.begin_verify_frame(seq, tokens.len())?;
        self.decode_verify_graphed_kgamma_dispatch(tokens, seq, _stream)
    }
    fn k1_stage_diag_requested(&self, pre_verify_len: usize, tokens: &[u32]) -> Result<bool> {
        crate::model::k1_stage_diag::requested_at(pre_verify_len, tokens)
    }
    fn k1_stage_diag_begin(&self, pre_verify_len: usize, tokens: &[u32]) -> Result<()> {
        crate::model::k1_stage_diag::begin_serial(pre_verify_len, tokens)
    }
    fn k1_stage_diag_finish_serial_row(&self) -> Result<()> {
        crate::model::k1_stage_diag::finish_serial_row()
    }
    fn k1_stage_diag_finish_serial(&self) -> Result<()> {
        crate::model::k1_stage_diag::finish_serial()
    }
    fn k1_stage_diag_abort(&self) {
        crate::model::k1_stage_diag::abort();
    }
    fn k16_fixture_evidence_begin(&self, pre_verify_len: usize, tokens: &[u32]) -> Result<()> {
        crate::model::k1_stage_diag::arm_fixture_receipt(pre_verify_len, tokens)?;
        if let Err(error) = crate::model::k16_route_receipt::begin(pre_verify_len, tokens) {
            crate::model::k1_stage_diag::abort();
            return Err(error);
        }
        Ok(())
    }
    fn k16_fixture_evidence_finish(&self, pre_verify_len: usize, tokens: &[u32]) -> Result<()> {
        let routes = crate::model::k16_route_receipt::take_exact(pre_verify_len, tokens)?;
        let stages = crate::model::k1_stage_diag::take_fixture_report(pre_verify_len, tokens)?;
        tracing::info!(
            run_id = stages.manifest.run_id.as_str(),
            verify_step = stages.manifest.verify_step,
            pre_verify_len,
            inputs = ?tokens,
            route_pre_verify_len = routes.pre_verify_len,
            route_inputs = ?routes.tokens,
            stages = stages.stages,
            terminal_stage = stages.terminal_stage.as_str(),
            logits_compared = stages.logits_compared,
            batched_entries = routes.batched_entries,
            qkv_layers = routes.qkv_layers,
            ssm_layers = routes.ssm_layers,
            "DFLASH_K16_FIXTURE_EVIDENCE exact_stage_and_route_match"
        );
        Ok(())
    }
    fn k16_fixture_evidence_abort(&self) {
        crate::model::k16_route_receipt::abort();
        crate::model::k1_stage_diag::abort();
    }
    fn k16_commit_parity_requested(&self, pre_verify_len: usize, tokens: &[u32]) -> Result<bool> {
        crate::model::k16_commit_parity::requested_at(self, pre_verify_len, tokens)
    }
    fn k16_commit_parity_capture_live(
        &self,
        seq: &SequenceState,
    ) -> Result<crate::model::k16_commit_parity::CanonicalStateSnapshot> {
        crate::model::k16_commit_parity::capture_live(self, seq)
    }
    fn k16_commit_parity_restore_live(
        &self,
        seq: &SequenceState,
        snapshot: &crate::model::k16_commit_parity::CanonicalStateSnapshot,
    ) -> Result<()> {
        crate::model::k16_commit_parity::restore_live(self, seq, snapshot)
    }
    fn k16_commit_parity_capture_sources(
        &self,
        seq: &SequenceState,
        k: usize,
        last_inter_slot: usize,
    ) -> Result<crate::model::k16_commit_parity::CommitSourceSnapshot> {
        crate::model::k16_commit_parity::capture_commit_sources(self, seq, k, last_inter_slot)
    }
    fn k16_commit_parity_restore_sources(
        &self,
        seq: &SequenceState,
        snapshot: &crate::model::k16_commit_parity::CommitSourceSnapshot,
    ) -> Result<()> {
        crate::model::k16_commit_parity::restore_commit_sources(self, seq, snapshot)
    }
    fn k16_commit_parity_drain_default(&self) -> Result<()> {
        crate::model::k16_commit_parity::drain_default(self)
    }
    fn k16_commit_parity_compare_committed(
        &self,
        seq: &SequenceState,
        expected: &crate::model::k16_commit_parity::CanonicalStateSnapshot,
        pre_verify_len: usize,
        tokens: &[u32],
        total_accepted: usize,
        k: usize,
        last_inter_slot: usize,
    ) -> Result<()> {
        crate::model::k16_commit_parity::compare_committed(
            self,
            seq,
            expected,
            pre_verify_len,
            tokens,
            total_accepted,
            k,
            last_inter_slot,
        )
    }
    fn save_hidden_for_mtp(&self, token_idx: usize, _stream: u64) -> Result<()> {
        crate::model::verify_failure::check()?;
        self.save_hidden_for_mtp_dispatch(token_idx, _stream)
    }
    fn save_hidden_for_mtp_sequence(
        &self,
        token_idx: usize,
        seq: &SequenceState,
        stream: u64,
    ) -> Result<()> {
        crate::model::verify_failure::check()?;
        self.save_hidden_for_mtp_sequence_dispatch(token_idx, seq, stream)
    }
    fn save_hidden_for_dflash(
        &self,
        token: u32,
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<()> {
        crate::model::verify_failure::check()?;
        self.save_hidden_for_dflash_dispatch(token, seq, _stream)
    }
    fn dflash_capture_thinking(&self, seq: &mut SequenceState, stream: u64) -> Result<()> {
        crate::model::verify_failure::check()?;
        self.dflash_capture_thinking_dispatch(seq, stream)
    }
    fn reconcile_dflash_context_before_bootstrap(
        &self,
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<usize> {
        crate::model::verify_failure::check()?;
        self.reconcile_dflash_context_before_bootstrap_dispatch(seq, stream)
    }
    fn run_mtp_propose(
        &self,
        token: u32,
        position: usize,
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<Option<u32>> {
        crate::model::verify_failure::check()?;
        self.run_mtp_propose_dispatch(token, position, seq, _stream)
    }
    fn run_mtp_propose_multi(
        &self,
        token: u32,
        position: usize,
        num_drafts: usize,
        seq: &mut SequenceState,
        _stream: u64,
        grammar_bitmask: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        crate::model::verify_failure::check()?;
        self.run_mtp_propose_multi_dispatch(
            token,
            position,
            num_drafts,
            seq,
            _stream,
            grammar_bitmask,
        )
    }
    fn run_mtp_propose_multi_batched(
        &self,
        tokens: &[u32],
        positions: &[usize],
        num_drafts: usize,
        seqs: &mut [&mut SequenceState],
        _stream: u64,
    ) -> Result<Vec<Vec<u32>>> {
        crate::model::verify_failure::check()?;
        self.run_mtp_propose_multi_batched_dispatch(tokens, positions, num_drafts, seqs, _stream)
    }
    fn stage_dflash_sampling_request(
        &self,
        seq: &mut SequenceState,
        request: crate::dflash_sampling::DflashDraftSamplingRequest,
    ) -> Result<()> {
        crate::model::verify_failure::check()?;
        self.stage_dflash_sampling_request_dispatch(seq, request)
    }
    fn take_pending_dflash_sampled_proposal(
        &self,
        seq: &mut SequenceState,
    ) -> Option<crate::dflash_sampling::DflashSampledProposalFrame> {
        self.take_pending_dflash_sampled_proposal_dispatch(seq)
    }
    fn read_deferred_draft_token(&self) -> Result<u32> {
        crate::model::verify_failure::check()?;
        self.read_deferred_draft_token_dispatch()
    }
    fn take_pending_tree_payload(
        &self,
        seq: &mut SequenceState,
    ) -> Option<crate::layers::DDTreePayload> {
        let proposer = self.active_proposer()?;
        let state = seq.proposer_state.as_mut()?;
        proposer.take_pending_tree_payload(state.as_mut())
    }

    /// Upload `payload.parent_indices` to the per-model DDTree scratch slot.
    ///
    /// Writes into the persistent `ddtree_parent_ids_persistent` device
    /// buffer (allocated once at init) so the device address stays stable
    /// across CUDA graph capture and replay. Older revisions of this
    /// function allocated a fresh scratch buffer per call, which left the
    /// captured graph reading a freed/stale pointer on replay.
    fn set_ddtree_parent_ids(&self, payload: &crate::layers::DDTreePayload) -> Result<bool> {
        crate::model::verify_failure::check()?;
        use crate::layers::dflash_head::ddtree_gdn_dispatch::requires_tree_kernel;
        static SETDBG: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = SETDBG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let needs_tree = requires_tree_kernel(Some(payload));
        // ATLAS_FORCE_TREEWY=1: stash even for flat chain payloads so the
        // tree-aware WY GDN kernel fires on them (used for A/B precision diff
        // vs wy17; without the bypass, flat chains dispatch wy17).
        let force_treewy = std::env::var("ATLAS_FORCE_TREEWY").ok().as_deref() == Some("1");
        if n < 3 {
            tracing::info!(
                "DDTree set_ddtree_parent_ids #{n}: payload.len={} needs_tree={} force_treewy={} parents={:?}",
                payload.tree_token_ids.len(),
                needs_tree,
                force_treewy,
                payload.parent_indices.iter().take(8).collect::<Vec<_>>()
            );
        }
        if !needs_tree && !force_treewy {
            // Flat chain → kernel falls through to wy_k. Nothing to stash.
            return Ok(false);
        }
        // Convert payload (compact-tree frame) to kernel frame.
        //
        // The K=γ verify kernel processes T = γ+1 tokens in order:
        //   token 0    = bonus (= last verified token from previous step)
        //   token i+1  = drafts[i] for i in 0..γ
        //
        // Each kernel token needs parent_ids[i]:
        //   -1  → load state from h_state (the pre-tree commit point)
        //   k≥0 → load state from h_state_inter[k]   (must satisfy k < i)
        //
        // payload.parent_indices[j] is in COMPACT-TREE frame for draft j:
        //   -1  → draft j attaches to the synthetic root (a.k.a. the bonus)
        //   k≥0 → draft j's parent is draft k
        //
        // Mapping compact-tree → kernel frame:
        //   kernel_parents[0]     = -1                                 (bonus loads from h_state)
        //   kernel_parents[j+1]   = if payload.parent_indices[j] < 0:
        //                              0                               (root-children load post-bonus state)
        //                          else:
        //                              payload.parent_indices[j] + 1   (+1 skips bonus slot)
        let stream = self.gpu.default_stream();
        let mut kernel_parents: Vec<i32> = Vec::with_capacity(1 + payload.parent_indices.len());
        kernel_parents.push(-1i32);
        for &p in &payload.parent_indices {
            kernel_parents.push(if p < 0 { 0 } else { p + 1 });
        }
        // Refuse to overrun the persistent buffer (defensive — payload
        // should never exceed dflash_kgamma).
        if self.ddtree_parent_ids_capacity == 0 {
            anyhow::bail!(
                "set_ddtree_parent_ids: persistent buffer not allocated (dflash_kgamma=0)"
            );
        }
        if kernel_parents.len() > self.ddtree_parent_ids_capacity {
            anyhow::bail!(
                "set_ddtree_parent_ids: payload has {} tokens > capacity {}",
                kernel_parents.len(),
                self.ddtree_parent_ids_capacity
            );
        }
        let bytes: Vec<u8> = kernel_parents
            .iter()
            .flat_map(|p| p.to_le_bytes())
            .collect();
        // Write into the persistent buffer (NOT a fresh alloc). The device
        // pointer never changes — CUDA graph replays see the new payload.
        self.copy_h2d_group_fast(
            &[HostToDeviceCopy::new(
                &bytes,
                self.ddtree_parent_ids_persistent,
            )],
            stream,
        )?;
        *self.ddtree_parent_ids_dev.lock() = Some(self.ddtree_parent_ids_persistent);
        *self.ddtree_num_tree_tokens.lock() = kernel_parents.len();
        // Stash the host-side mirror so verify_d.rs can derive per-token
        // depths for tree-aware RoPE/seq_lens without a D2H copy.
        *self.ddtree_parent_ids_host.lock() = kernel_parents;
        Ok(true)
    }

    fn clear_ddtree_parent_ids(&self) {
        // Also disarm the flat-chain injection flag. Every caller of this
        // method is either an error path that skips the commit (the commit's
        // swap-to-false is what normally consumes the flag — an armed flag
        // would otherwise leak into an unrelated later commit) or runs after
        // the commit already consumed it (reset is a no-op).
        self.dflash_flat_tree_route
            .store(false, std::sync::atomic::Ordering::Release);
        // Restore the persistent buffer to the linear-chain default so the
        // graph-safe verify path (which keeps `Some(persistent)` to always
        // hit tree_wy) reads bit-equivalent-to-wy17 parents on the next
        // non-tree call. We don't release the buffer — its address must
        // remain stable for captured graphs.
        if self.ddtree_parent_ids_capacity > 0 {
            let cap = self.ddtree_parent_ids_capacity;
            let mut chain = Vec::<i32>::with_capacity(cap);
            chain.push(-1);
            for i in 1..cap {
                chain.push((i - 1) as i32);
            }
            let bytes: Vec<u8> = chain.iter().flat_map(|p| p.to_le_bytes()).collect();
            let stream = self.gpu.default_stream();
            // Best-effort restamp. A failed copy here is non-fatal: the next
            // set_ddtree_parent_ids call will overwrite, and the graph-safe
            // path always re-stamps before replay-only invocations.
            if let Err(e) = self.copy_h2d_group_fast(
                &[HostToDeviceCopy::new(
                    &bytes,
                    self.ddtree_parent_ids_persistent,
                )],
                stream,
            ) {
                tracing::warn!("clear_ddtree_parent_ids: failed to restamp linear chain: {e:#}");
            }
        }
        *self.ddtree_parent_ids_dev.lock() = None;
        *self.ddtree_num_tree_tokens.lock() = 0;
        self.ddtree_parent_ids_host.lock().clear();
        // Do NOT clear `ddtree_dfs_inv_perm` here. The scheduler calls
        // `clear_ddtree_parent_ids` (`verify_dflash_step.rs`) BEFORE the KV
        // gather (`compact_verify_kv`) and the ctx-hidden row stamp
        // (`set_dflash_accepted_compact`), both of which must translate the
        // committed compact indices into the kernel (DFS-reordered) frame via
        // this permutation. Clearing it here would leave those consumers with
        // an empty (identity) map on a genuine fork. The permutation is
        // authoritatively (re)set or cleared at the start of every
        // `decode_verify_graphed_kgamma_dispatch` (`verify_d.rs`) based on
        // that step's DFS-active state, so it never leaks across verifies.
    }

    /// See the trait doc: ancestor-exactness of the LAST K=γ verify's
    /// attention. Written by `decode_verify_graphed_kgamma_dispatch`
    /// (verify_d.rs) on every call; the scheduler consults it right after
    /// verify to gate the full tree-commit walker. Release/Acquire pairs
    /// with the store for the same reason as `dflash_flat_tree_route`.
    fn dflash_tree_ancestor_attn_exact(&self) -> bool {
        self.dflash_tree_ancestor_attn
            .load(std::sync::atomic::Ordering::Acquire)
    }

    fn dflash_tree_verify_capable(&self) -> bool {
        let tree_metadata_enabled = std::env::var("ATLAS_DDTREE_TREE_AWARE_VERIFY")
            .ok()
            .as_deref()
            == Some("1")
            && std::env::var("ATLAS_TREE_AWARE_ATTN").ok().as_deref() == Some("1");
        let tree_conv_enabled = std::env::var("ATLAS_DDTREE_TREE_CONV_EXACT")
            .ok()
            .as_deref()
            == Some("1");
        // One-shot diagnostic. This predicate is the single gate between a
        // built tree payload and a widened verify, and when it returns false
        // the payload is discarded with no indication of WHICH of its four
        // conjuncts failed — which is why DDTree read as an unexplained no-op
        // on Qwen3.8 even after the proposer was fixed and was demonstrably
        // emitting 15-node trees (`DFlash FREE_SLOTS #0: ... nodes=15`).
        let attn_ok = self
            .layers
            .iter()
            .all(|layer| layer.ddtree_ancestor_attention_exact());
        let conv_ok = self
            .layers
            .iter()
            .all(|layer| layer.ddtree_conv_state_exact());
        static CAP_DBG: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        if CAP_DBG.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 2 {
            tracing::info!(
                "DDTREE_CAPABLE: metadata(TREE_AWARE_VERIFY&&TREE_AWARE_ATTN)={} \
                 conv_env(TREE_CONV_EXACT)={} all_layers_ancestor_attn_exact={} \
                 all_layers_conv_state_exact={} -> capable={}",
                tree_metadata_enabled,
                tree_conv_enabled,
                attn_ok,
                conv_ok,
                tree_metadata_enabled && tree_conv_enabled && attn_ok && conv_ok
            );
            if !attn_ok {
                let bad: Vec<usize> = self
                    .layers
                    .iter()
                    .enumerate()
                    .filter(|(_, l)| !l.ddtree_ancestor_attention_exact())
                    .map(|(i, _)| i)
                    .take(8)
                    .collect();
                tracing::info!("DDTREE_CAPABLE: first layers failing ancestor_attn_exact: {bad:?}");
            }
            if !conv_ok {
                let bad: Vec<usize> = self
                    .layers
                    .iter()
                    .enumerate()
                    .filter(|(_, l)| !l.ddtree_conv_state_exact())
                    .map(|(i, _)| i)
                    .take(8)
                    .collect();
                tracing::info!("DDTREE_CAPABLE: first layers failing conv_state_exact: {bad:?}");
            }
        }
        tree_metadata_enabled && tree_conv_enabled && attn_ok && conv_ok
    }

    fn trim_proposer_state(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
        _stream: u64,
    ) -> Result<()> {
        crate::model::verify_failure::check()?;
        self.trim_proposer_state_dispatch(seq, num_accepted, _stream)
    }
    fn dflash_collect_async_drafts(&self, seq: &mut SequenceState) -> Result<Option<Vec<u32>>> {
        crate::model::verify_failure::check()?;
        self.dflash_collect_async_drafts_dispatch(seq)
    }

    fn dflash_arm_propose_overlap(&self) -> Result<()> {
        crate::model::verify_failure::check()?;
        let Some(proposer) = self.active_proposer() else {
            return Ok(());
        };
        proposer.arm_propose_overlap(self.gpu.as_ref(), self.gpu.default_stream())
    }

    fn dflash_seq_has_async_placeholder(&self, seq: &SequenceState) -> bool {
        seq.proposer_state
            .as_ref()
            .and_then(|s| {
                s.as_any()
                    .downcast_ref::<crate::layers::DflashProposerState>()
            })
            .is_some_and(|ds| ds.async_placeholder)
    }

    fn mtp_dloop_max_verify_k(&self) -> usize {
        if self.config.num_experts > 1 {
            return 0;
        }
        if self.ssm_pool.num_ssm_layers > 0 {
            self.ssm_pool.num_intermediates.saturating_sub(1)
        } else {
            self.buffers.max_batch_tokens()
        }
    }

    fn dflash_stash_recycle(
        &self,
        seq: &mut SequenceState,
        drafts: &[u32],
        num_accepted: usize,
        corrected_token: u32,
    ) -> Result<()> {
        crate::model::verify_failure::check()?;
        // Stash `drafts[num_accepted+1 ..]` (the tail discarded purely because
        // it sat downstream of the first content miss) keyed by the corrected
        // token the target committed at that miss. The next propose re-offers
        // this tail when its last_token == corrected_token. LOSSLESS: only
        // affects what is PROPOSED, never what is committed.
        let Some(ps) = seq.proposer_state.as_mut() else {
            return Ok(());
        };
        let Some(ds) = ps
            .as_any_mut()
            .downcast_mut::<crate::layers::DflashProposerState>()
        else {
            return Ok(());
        };
        // The tail starts AFTER the mismatched draft at index `num_accepted`
        // (that draft was wrong → replaced by the bonus). `num_accepted+1` is
        // the first downstream-discarded draft. Full-accept (num_accepted ==
        // drafts.len()) leaves an empty tail → nothing to recycle.
        let tail_start = num_accepted.saturating_add(1);
        if tail_start < drafts.len() {
            ds.recycle_tail.clear();
            ds.recycle_tail.extend_from_slice(&drafts[tail_start..]);
            ds.recycle_key = corrected_token;
            ds.recycle_valid = true;
        } else {
            // No usable tail this step — invalidate any prior stash so it is
            // never re-offered staler than one step.
            ds.recycle_tail.clear();
            ds.recycle_valid = false;
        }
        Ok(())
    }
    fn dflash_stash_echo(
        &self,
        seq: &mut SequenceState,
        verified: &[u32],
        num_accepted: usize,
        bonus_token: u32,
    ) -> Result<()> {
        crate::model::verify_failure::check()?;
        // Stash `verified[num_accepted+1 ..]` — the target's own argmaxes at
        // the positions AFTER the bonus — keyed by the bonus. The next propose
        // offers this tail as the draft chain and skips the drafter forward.
        // LOSSLESS: only affects what is PROPOSED, never what is committed.
        use crate::layers::dflash_head::echo;
        let Some(cfg) = echo::EchoConfig::from_env() else {
            return Ok(());
        };
        let Some(ps) = seq.proposer_state.as_mut() else {
            return Ok(());
        };
        let Some(ds) = ps
            .as_any_mut()
            .downcast_mut::<crate::layers::DflashProposerState>()
        else {
            return Ok(());
        };
        let tail = echo::echo_tail(verified, num_accepted);
        if echo::should_stash(&cfg, num_accepted, tail.len()) {
            ds.echo_tail.clear();
            ds.echo_tail.extend_from_slice(tail);
            ds.echo_key = bonus_token;
            ds.echo_valid = true;
        } else {
            // Gated out (full accept, weak prefix, or short tail) — never
            // re-offer a stash staler than one step.
            ds.echo_tail.clear();
            ds.echo_valid = false;
        }
        Ok(())
    }
    fn compact_sequence(&self, seq: &mut SequenceState, new_slot: usize) -> Result<bool> {
        crate::model::verify_failure::check()?;
        self.ensure_verify_sequence_idle(seq)?;
        self.compact_sequence_dispatch(seq, new_slot)
    }
    fn save_sequence_state(
        &self,
        seq: &SequenceState,
        writer: &mut dyn std::io::Write,
    ) -> Result<()> {
        crate::model::verify_failure::check()?;
        self.ensure_verify_sequence_idle(seq)?;
        self.save_sequence_state_dispatch(seq, writer)
    }
    fn restore_sequence_state(
        &self,
        seq: &mut SequenceState,
        num_blocks: usize,
        reader: &mut dyn std::io::Read,
    ) -> Result<()> {
        crate::model::verify_failure::check()?;
        self.ensure_verify_sequence_idle(seq)?;
        seq.verify_frame = None;
        self.discard_native_verify_frame(seq);
        self.restore_sequence_state_dispatch(seq, num_blocks, reader)
    }
    fn num_free_blocks(&self) -> usize {
        self.num_free_blocks_dispatch()
    }
    fn num_total_blocks(&self) -> usize {
        self.num_total_blocks_dispatch()
    }
    fn num_admission_blocks(&self) -> usize {
        // Model construction permanently allocates one zeroed dummy KV
        // block for padded decode rows. It is part of the occupancy gauge,
        // but no request can ever allocate it.
        self.num_total_blocks_dispatch().saturating_sub(1)
    }
    fn reserve_decode_kv(&self, seq: &mut SequenceState, rows: usize) -> Result<()> {
        let mut kv_cache = self.kv_cache.lock();
        super::block_mgmt::reserve_decode_rows(
            seq,
            rows,
            &mut kv_cache,
            self.prefix_cache.as_ref(),
            self.gpu.as_ref(),
            self.gpu.default_stream(),
        )
    }
    fn start_checkpoint_async(&self, seq: &mut SequenceState) -> Result<()> {
        crate::model::verify_failure::check()?;
        self.ensure_verify_healthy()?;
        self.start_checkpoint_async_dispatch(seq)
    }
    fn start_rollback_and_checkpoint_async(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
    ) -> Result<()> {
        crate::model::verify_failure::check()?;
        self.ensure_verify_healthy()?;
        self.start_rollback_and_checkpoint_async_dispatch(seq, num_accepted)
    }
    fn sync_secondary(&self) -> Result<()> {
        self.sync_secondary_dispatch()
    }
    fn pre_verify_copy_async(&self, seq: &mut SequenceState) -> Result<()> {
        crate::model::verify_failure::check()?;
        self.ensure_verify_healthy()?;
        self.pre_verify_copy_async_dispatch(seq)
    }
    fn commit_qwen4_qsa_k5_only(&self, num_accepted: usize, k: usize) -> Result<()> {
        crate::model::verify_failure::check()?;
        anyhow::ensure!(
            !self.config.is_qwen4_exp() || k <= 1,
            "Qwen4 QSA restoration requires a sequence-owned frame"
        );
        self.commit_qwen4_qsa_k5_only_dispatch(num_accepted, k)
    }
    fn commit_qwen4_qsa_for_sequence(
        &self,
        seq: &SequenceState,
        accepted: usize,
        rows: usize,
    ) -> Result<()> {
        crate::model::verify_failure::check()?;
        self.restore_owned_qsa_prefix(seq, accepted, rows)
    }
    fn finish_verify_diagnostic(&self, seq: &mut SequenceState, rows: usize) -> Result<()> {
        crate::model::verify_failure::check()?;
        TransformerModel::finish_verify_diagnostic(self, seq, rows)
    }
    fn finish_verify_rebuild(
        &self,
        seq: &mut SequenceState,
        accepted: usize,
        rows: usize,
    ) -> Result<()> {
        crate::model::verify_failure::check()?;
        TransformerModel::finish_verify_rebuild(self, seq, accepted, rows)
    }
    fn abort_verify_state(&self, seq: &mut SequenceState, k: usize) -> Result<()> {
        self.abort_owned_verify_state(seq, k)
    }
    fn commit_verify_state_async(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
        k: usize,
    ) -> Result<()> {
        crate::model::verify_failure::check()?;
        // Legacy callers (K=2/K=3/K=4 chain verifies) → derive the kernel
        // slot as `num_accepted - 1` (chain-contiguous). DDTree callers
        // must use `commit_verify_state_async_with_slot` to pass the
        // explicit non-contiguous slot.
        let last_inter_slot = num_accepted.saturating_sub(1);
        self.commit_owned_verify_state(seq, num_accepted, k, last_inter_slot)
    }
    fn commit_verify_state_async_with_slot(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
        k: usize,
        last_inter_slot: usize,
    ) -> Result<()> {
        crate::model::verify_failure::check()?;
        // DFS reorder: when the K=γ verify ran with `ATLAS_DDTREE_DFS_REORDER=1`, the SSM kernel wrote h_state_inter in
        // DFS slot order — not original-compact order. greedy_sample_ddtree
        // returns indices in the ORIGINAL compact frame, so we must map
        // them via dfs_inv_perm[orig] = dfs_slot before reading the inter
        // pool. Mapping is a no-op when the buffer is empty (chain mode or
        // DFS-disabled).
        let inv_perm = self.ddtree_dfs_inv_perm.lock();
        let mapped_slot = if !inv_perm.is_empty() && last_inter_slot < inv_perm.len() {
            inv_perm[last_inter_slot]
        } else {
            last_inter_slot
        };
        drop(inv_perm);
        self.commit_owned_verify_state(seq, num_accepted, k, mapped_slot)
    }
    fn supports_batched_verify_commit(&self) -> bool {
        let tp_broadcast = self.qwen4_tp_batched_verify_supported_dispatch()
            && self
                .comm
                .as_ref()
                .is_some_and(|c| c.world_size() > 1 && c.rank() == 0);
        self.supports_verify_batch_cached() && !tp_broadcast
    }
    fn commit_verify_batch_cached(
        &self,
        entries: &mut [(&mut SequenceState, usize, usize)],
        stream: u64,
    ) -> Result<()> {
        self.commit_owned_verify_batch_cached(entries, stream)
    }
    fn compact_verify_kv(
        &self,
        seq: &SequenceState,
        accepted_compact: &[usize],
        pre_verify_len: usize,
    ) -> Result<()> {
        crate::model::verify_failure::check()?;
        self.compact_verify_kv_dispatch(seq, accepted_compact, pre_verify_len)
    }
    fn set_dflash_accepted_compact(&self, seq: &mut SequenceState, accepted_compact: &[usize]) {
        self.set_dflash_accepted_compact_dispatch(seq, accepted_compact);
    }
    fn ep_worker_step(&self, seq: &mut SequenceState) -> Result<bool> {
        crate::model::verify_failure::check()?;
        self.ep_worker_step_dispatch(seq)
    }
    fn is_ep(&self) -> bool {
        self.is_ep_dispatch()
    }
    fn qwen4_tp_batched_verify_supported(&self) -> bool {
        self.qwen4_tp_batched_verify_supported_dispatch()
    }
    fn hc_mult(&self) -> usize {
        // Qwen4 names its highway width `hc_count`, while the older GLM
        // configuration uses `hc_mult`.  The scheduler consumes this method
        // as a capability guard for mixed prefill, slot compaction, and other
        // paths that still assume one ordinary residual stream.  Returning
        // only `hc_mult` made Qwen4 look like an ordinary model even though it
        // owns four request-local HC/PLE streams.
        if self.config.is_qwen4_exp() {
            self.config.hc_count
        } else {
            self.config.hc_mult
        }
    }
    fn is_mla(&self) -> bool {
        self.is_mla_dispatch()
    }
    fn qwen4_batch_plan(&self) -> Result<Option<crate::qwen4_batch_policy::Qwen4BatchPlan>> {
        if self.config.is_qwen4_exp() {
            self.qwen4_batch_plan_dispatch().map(Some)
        } else {
            Ok(None)
        }
    }
    fn qwen4_decode_route(
        &self,
        seqs: &[&mut SequenceState],
    ) -> Result<Option<crate::qwen4_batch_policy::FrameRoute>> {
        if self.config.is_qwen4_exp() && seqs.len() > 1 {
            self.qwen4_frame_route(seqs).map(Some)
        } else {
            Ok(None)
        }
    }
    fn qwen4_native_capability_report(
        &self,
        max_batch_size: usize,
    ) -> Result<crate::qwen4_batch_policy::NativeCapabilityReport> {
        let Some(plan) = self.qwen4_batch_plan()? else {
            return Ok(Default::default());
        };
        if !plan.native() {
            return Ok(Default::default());
        }
        if plan.qsa_rows
            && let Err(gap) = self.qwen4_native_qsa_capability()
        {
            anyhow::bail!(
                "ATLAS_QWEN4_NATIVE_BATCH_QSA=1 refused: native QSA rows are unavailable: {}",
                gap.detail
            );
        }
        Ok(crate::qwen4_batch_policy::NativeCapabilityReport {
            gaps: self.qwen4_native_capability_report_dispatch(max_batch_size),
        })
    }
    fn fast_prefill_chunk_rows(&self) -> Option<usize> {
        // Only the Qwen4 fast-prefill family locks its admission to the chunk
        // geometry (`qwen4_prefill_moe::ChunkGeometry::chunk_rows_at`). A
        // malformed selector is reported by the family itself at admission.
        let armed = self.config.is_qwen4_exp()
            && crate::layers::qwen4_prefill_moe::attn16::selected().unwrap_or(false);
        armed.then(crate::layers::qwen4_prefill_moe::max_chunk_rows)
    }
    fn fast_prefill_chunk_widths(&self) -> Vec<usize> {
        if self.fast_prefill_chunk_rows().is_none() {
            return Vec::new();
        }
        crate::layers::qwen4_prefill_moe::interleave_widths()
    }
    fn decode_logits_fp32(&self) -> bool {
        self.decode_logits_fp32_dispatch()
    }
    fn decode_logits_ptr(&self) -> DevicePtr {
        self.decode_logits_ptr_dispatch()
    }
    fn ep_broadcast_cmd(&self, cmd: u32) -> Result<()> {
        self.ep_broadcast_cmd_dispatch(cmd)
    }
    fn ep_broadcast_tokens(&self, tokens: &[u32]) -> Result<Vec<u32>> {
        self.ep_broadcast_tokens_dispatch(tokens)
    }
    fn synchronize_stream(&self, stream: u64) -> Result<()> {
        self.synchronize_stream_dispatch(stream)
    }
    fn default_stream(&self) -> u64 {
        self.default_stream_dispatch()
    }
    fn create_stream(&self) -> Result<u64> {
        self.create_stream_dispatch()
    }
    fn create_event(&self) -> Result<u64> {
        self.create_event_dispatch()
    }
    fn record_event(&self, event: u64, stream: u64) -> Result<()> {
        self.record_event_dispatch(event, stream)
    }
    fn stream_wait_event(&self, stream: u64, event: u64) -> Result<()> {
        self.stream_wait_event_dispatch(stream, event)
    }
}
