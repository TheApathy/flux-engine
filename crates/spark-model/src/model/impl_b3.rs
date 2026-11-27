// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use atlas_core::config::{DflashCaptureMode, LayerType, ModelConfig};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, HostToDeviceCopy, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use super::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use super::ssm_pool::SsmStatePool;
use super::ssm_snapshot::SsmSnapshotPool;
use super::types::{PinnedMetaStaging, TransformerModel};
use crate::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use crate::layers::ops;
use crate::speculative::{CanonicalCommittedPrefix, DraftProposer, TargetTokenAuthority};
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use crate::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

impl TransformerModel {
    pub(super) fn run_mtp_propose_inner(
        &self,
        token: u32,
        position: usize,
        num_drafts: usize,
        seq: &mut SequenceState,
        grammar_bitmask: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        crate::model::verify_failure::check()?;
        self.ensure_verify_sequence_idle(seq)?;
        let result =
            self.run_mtp_propose_owned_inner(token, position, num_drafts, seq, grammar_bitmask);
        // Deferred replay owns its own copied ticket. Sequence metadata is
        // only the commit-to-propose handoff, even when proposal fails early.
        seq.verify_frame = None;
        self.discard_native_verify_frame(seq);
        result
    }

    fn run_mtp_propose_owned_inner(
        &self,
        token: u32,
        position: usize,
        num_drafts: usize,
        seq: &mut SequenceState,
        grammar_bitmask: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        super::final_norm::check_partial_head_admission(
            &self.config,
            super::final_norm::PartialHead::EarlyExit,
            Self::early_exit_enabled(),
        )?;
        let proposer = match self.active_proposer() {
            Some(p) => p.as_ref(),
            None => return Ok(Vec::new()),
        };
        // Consume a sequence/cursor-bound handoff before replay, rotary, or
        // proposer mutation. Shared target scratch may now belong to another
        // request, so the native head must never fall back to its singleton.
        let native_hidden =
            if proposer.kind() == crate::speculative::DraftProposerKind::Qwen4NativeMtp {
                let (storage, frames) = self.qwen4_mtp_hidden_frames.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("native MTP has no sequence-owned target-hidden storage")
                })?;
                let retained = frames
                    .lock()
                    .take(seq.slot_idx, seq.session_hash, position)
                    .map_err(anyhow::Error::msg)?;
                Some((
                    storage.offset(retained.selected.offset),
                    storage.offset(retained.prefix.offset),
                ))
            } else {
                None
            };
        // ATLAS_MTP_LASTK_DEFER fallback: never propose from an unreplayed state.
        self.k8a_late_replay_if_pending(seq)?;
        self.sync_proposer_rotary(seq)?;
        if !seq.rotary_positions.is_identity() {
            anyhow::ensure!(
                position == seq.seq_len,
                "proposer physical position disagrees with target-fed length"
            );
            seq.rotary_positions.tail_scalar(position)?;
        }
        // ATLAS_DFLASH_DEBUG_DUMP_FULL=1: emit the full token sequence
        // ONCE so a Python reference can run the SAME tokens through HF
        // transformers and dump matching hidden-state captures.
        static TOKENS_DUMPED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        if !TOKENS_DUMPED.load(std::sync::atomic::Ordering::Relaxed)
            && std::env::var("ATLAS_DFLASH_DEBUG_DUMP_FULL")
                .ok()
                .as_deref()
                == Some("1")
            // ATLAS_DFLASH_DUMP_MIN_POS=N defers the one-shot dump until the
            // first propose at position >= N (mid-generation ctx capture).
            && position
                >= std::env::var("ATLAS_DFLASH_DUMP_MIN_POS")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(0)
        {
            let tokens_json = serde_json::json!({
                "prompt_len": seq.prompt_len,
                "position": position,
                "last_token": token,
                "all_tokens": seq.tokens.clone(),
                "generated_tokens": seq.tokens.iter().skip(seq.prompt_len).copied().collect::<Vec<u32>>(),
            });
            if let Err(e) = std::fs::write(
                "/tmp/atlas_tokens.json",
                serde_json::to_string_pretty(&tokens_json).unwrap_or_default(),
            ) {
                tracing::warn!("DFLASH DUMP_FULL: tokens write failed: {e}");
            } else {
                tracing::info!(
                    "DFLASH DUMP_FULL: wrote /tmp/atlas_tokens.json (position={}, all_tokens.len()={}, prompt_len={})",
                    position,
                    seq.tokens.len(),
                    seq.prompt_len,
                );
            }
            TOKENS_DUMPED.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        let stream = self.gpu.default_stream();
        let draft_embed_target = None;
        // MTP loads ALL experts on every rank (no EP filtering), so its MoE
        // output is already complete — no all_reduce needed. Passing comm: None
        // prevents MoeLayer::forward() from doubling the output via SUM.
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
        // ── ATLAS_DFLASH_EARLY_EXIT=1: target early-exit drafter ──
        //
        // Replace the tiny z-lab drafter's propose with the TARGET'S OWN first
        // N layers + lm_head as the draft source (see `early_exit.rs`). The
        // drafts flow into the unchanged verify path; the full 64-layer target
        // still commits only its greedy token, so output stays byte-identical
        // (LOSSLESS — verify is the oracle). This is the in-distribution
        // predictor that beats the tiny drafter on NOVEL coding tokens.
        //
        // The DFlash proposer state's ctx accumulator append (done inside
        // `propose_drafts`) is intentionally skipped here: when early-exit is
        // the draft source the neural drafter never runs, so its ctx is never
        // consumed. The verify path's SSM checkpoint/rollback is independent of
        // this and unchanged.
        let target_token_authority = proposer.target_token_authority();
        if Self::early_exit_enabled() {
            anyhow::ensure!(
                target_token_authority == TargetTokenAuthority::ServingPolicy,
                "target early-exit cannot bypass exact-authority proposal admission"
            );
            return self.early_exit_propose(token, num_drafts, seq);
        }
        let canonical_prefix = if target_token_authority == TargetTokenAuthority::ExactRawArgmax {
            let prefix = CanonicalCommittedPrefix {
                fed_tokens: &seq.tokens,
                pending_token: token,
                position,
                prompt_len: seq.prompt_len,
            };
            prefix.validate()?;
            Some(prefix)
        } else {
            None
        };
        let prop_state = seq
            .proposer_state
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("No proposer state for sequence"))?;
        if let Some(prefix) = canonical_prefix {
            proposer.sync_committed_prefix(prefix, prop_state.as_mut(), &ctx, stream)?;
        }
        // Refresh the host token mirror used by both prompt-lookup drafting
        // (ATLAS_DFLASH_PLD) and the generalized retrieval-augmented drafter
        // (ATLAS_DFLASH_RETRIEVAL). `seq.tokens` is the FULL committed
        // target-fed prefix; `token` above is the just-emitted pending token.
        // The retrieval haystack includes reference code in the prompt for
        // free. Only populated when one of the flags is on (default off ⇒ no
        // extra copy, legacy behavior byte-for-byte).
        let want_token_mirror = std::env::var("ATLAS_DFLASH_PLD").ok().as_deref() == Some("1")
            || std::env::var("ATLAS_DFLASH_RETRIEVAL").ok().as_deref() == Some("1")
            || std::env::var("ATLAS_DFLASH_SAM").ok().as_deref() == Some("1")
            || crate::layers::dflash_head::ngram_bonus::ngram_bonus_enabled();
        if want_token_mirror
            && let Some(ds) = prop_state
                .as_any_mut()
                .downcast_mut::<crate::layers::DflashProposerState>()
        {
            ds.pld_tokens.clear();
            ds.pld_tokens.extend_from_slice(&seq.tokens);
        }
        // ATLAS_QWEN4_MTP_GATE_QSA_MIN_WIDTH: the MTP gate needs to know
        // whether this request reaches QSA context.
        let _qsa_context =
            crate::layers::qwen4_mtp_gate::QsaContextGuard::new(seq.qwen4_qsa_required);
        if grammar_bitmask.is_none()
            && num_drafts > 0
            && proposer.kind() == crate::speculative::DraftProposerKind::Qwen4NativeMtp
            && let Some(drafts) =
                crate::layers::qwen4_mtp::try_ngram_propose(&seq.tokens, token, num_drafts)
        {
            if let Some(mtp_state) = prop_state
                .as_any_mut()
                .downcast_mut::<crate::layers::qwen4_mtp::Qwen4MtpState>()
            {
                mtp_state.notify_external_draft_with_tokens(&drafts);
            }
            return Ok(drafts);
        }
        // K7 committed-row rewrite: the native Qwen4 MTP head reads the verify
        // hidden rows staged in the dedicated recondition tail slice by `save_hidden_for_mtp`.
        let hidden_stack = if crate::layers::qwen4_mtp_rows::recondition_engaged()
            && proposer.kind() == crate::speculative::DraftProposerKind::Qwen4NativeMtp
        {
            native_hidden.map(|(_, prefix)| prefix)
        } else {
            self.dflash_hidden_save
        };
        proposer.propose(
            token,
            native_hidden.map_or(self.mtp_hidden_save, |(selected, _)| selected),
            position,
            num_drafts,
            prop_state.as_mut(),
            &ctx,
            stream,
            draft_embed_target,
            grammar_bitmask,
            hidden_stack,
        )
    }

    pub(super) fn run_mtp_propose_multi_batched_inner(
        &self,
        tokens: &[u32],
        positions: &[usize],
        num_drafts: usize,
        seqs: &mut [&mut SequenceState],
    ) -> Result<Vec<Vec<u32>>> {
        anyhow::ensure!(
            tokens.len() == positions.len() && tokens.len() == seqs.len(),
            "run_mtp_propose_multi_batched extent mismatch"
        );
        let b = seqs.len();
        if b > 1
            && !Self::early_exit_enabled()
            && self.active_proposer().is_some_and(|p| {
                p.kind() == crate::speculative::DraftProposerKind::DFlash
                    && p.target_token_authority() == TargetTokenAuthority::ServingPolicy
                    && p.dflash_draft_sample_method()
                        != crate::dflash_sampling::DflashDraftSampleMethod::Probabilistic
            })
        {
            crate::model::verify_failure::check()?;
            for seq in seqs.iter() {
                self.ensure_verify_sequence_idle(seq)?;
            }
            let batch_res = self
                .run_dflash_propose_multi_batched_owned_inner(tokens, positions, num_drafts, seqs);
            match batch_res {
                Ok(Some(drafts)) => {
                    for seq in seqs.iter_mut() {
                        seq.verify_frame = None;
                        self.discard_native_verify_frame(seq);
                    }
                    return Ok(drafts);
                }
                Ok(None) => {
                    // Proposer declined cross-sequence batching without mutating state;
                    // fall through to the per-sequence propose loop below.
                }
                Err(err) => {
                    for seq in seqs.iter_mut() {
                        seq.verify_frame = None;
                        self.discard_native_verify_frame(seq);
                    }
                    return Err(err);
                }
            }
        }
        if b <= 1
            || Self::early_exit_enabled()
            || !self.active_proposer().is_some_and(|p| {
                p.kind() == crate::speculative::DraftProposerKind::Qwen4NativeMtp
                    && p.target_token_authority() == TargetTokenAuthority::ServingPolicy
            })
        {
            let mut out = Vec::with_capacity(b);
            for i in 0..b {
                out.push(self.run_mtp_propose_inner(
                    tokens[i],
                    positions[i],
                    num_drafts,
                    seqs[i],
                    None,
                )?);
            }
            return Ok(out);
        }

        crate::model::verify_failure::check()?;
        for seq in seqs.iter() {
            self.ensure_verify_sequence_idle(seq)?;
        }
        let result =
            self.run_mtp_propose_multi_batched_owned_inner(tokens, positions, num_drafts, seqs);
        for seq in seqs.iter_mut() {
            seq.verify_frame = None;
            self.discard_native_verify_frame(seq);
        }
        result
    }

    fn run_dflash_propose_multi_batched_owned_inner(
        &self,
        tokens: &[u32],
        positions: &[usize],
        num_drafts: usize,
        seqs: &mut [&mut SequenceState],
    ) -> Result<Option<Vec<Vec<u32>>>> {
        let b = seqs.len();
        super::final_norm::check_partial_head_admission(
            &self.config,
            super::final_norm::PartialHead::EarlyExit,
            Self::early_exit_enabled(),
        )?;
        let Some(proposer) = self.active_proposer() else {
            return Ok(Some(vec![Vec::new(); b]));
        };
        // Batched DFlash proposal passes `target_hidden_stacks = &[None; b]` and
        // reads each sequence's already-retired `ctx_hidden_acc` ring. If any
        // sequence still has unretired single-seq capture debt, fall back cleanly
        // to per-sequence proposal before mutating any state.
        for (seq, &position) in seqs.iter().zip(positions.iter()) {
            let Some(dstate) = seq.proposer_state.as_deref().and_then(|ps| {
                ps.as_any()
                    .downcast_ref::<crate::layers::DflashProposerState>()
            }) else {
                return Ok(None);
            };
            if dstate.ctx_len != position {
                return Ok(None);
            }
        }
        for (seq, &position) in seqs.iter_mut().zip(positions.iter()) {
            self.k8a_late_replay_if_pending(seq)?;
            self.sync_proposer_rotary(seq)?;
            if !seq.rotary_positions.is_identity() {
                anyhow::ensure!(
                    position == seq.seq_len,
                    "proposer physical position disagrees with target-fed length"
                );
                seq.rotary_positions.tail_scalar(position)?;
            }
        }
        let want_token_mirror = std::env::var("ATLAS_DFLASH_PLD").ok().as_deref() == Some("1")
            || std::env::var("ATLAS_DFLASH_RETRIEVAL").ok().as_deref() == Some("1")
            || std::env::var("ATLAS_DFLASH_SAM").ok().as_deref() == Some("1")
            || crate::layers::dflash_head::ngram_bonus::ngram_bonus_enabled();
        if want_token_mirror {
            for seq in seqs.iter_mut() {
                if let Some(ds) = seq.proposer_state.as_mut().and_then(|ps| {
                    ps.as_any_mut()
                        .downcast_mut::<crate::layers::DflashProposerState>()
                }) {
                    ds.pld_tokens.clear();
                    ds.pld_tokens.extend_from_slice(&seq.tokens);
                }
            }
        }
        let stream = self.gpu.default_stream();
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
        let null_hiddens = vec![DevicePtr::NULL; b];
        let no_stacks = vec![None; b];
        let mut state_refs: Vec<&mut dyn crate::speculative::ProposerState> = Vec::with_capacity(b);
        for seq in seqs.iter_mut() {
            let state = seq
                .proposer_state
                .as_deref_mut()
                .ok_or_else(|| anyhow::anyhow!("No proposer state for sequence"))?;
            state_refs.push(state);
        }
        proposer.propose_batch(
            tokens,
            &null_hiddens,
            positions,
            num_drafts,
            &mut state_refs,
            &ctx,
            stream,
            &no_stacks,
        )
    }

    fn run_mtp_propose_multi_batched_owned_inner(
        &self,
        tokens: &[u32],
        positions: &[usize],
        num_drafts: usize,
        seqs: &mut [&mut SequenceState],
    ) -> Result<Vec<Vec<u32>>> {
        let b = seqs.len();
        super::final_norm::check_partial_head_admission(
            &self.config,
            super::final_norm::PartialHead::EarlyExit,
            Self::early_exit_enabled(),
        )?;
        let proposer = match self.active_proposer() {
            Some(p) => p.as_ref(),
            None => return Ok(vec![Vec::new(); b]),
        };
        let (storage, frames) = self.qwen4_mtp_hidden_frames.as_ref().ok_or_else(|| {
            anyhow::anyhow!("native MTP has no sequence-owned target-hidden storage")
        })?;
        let mut target_hiddens = Vec::with_capacity(b);
        let mut target_hidden_stacks = Vec::with_capacity(b);
        let recondition = crate::layers::qwen4_mtp_rows::recondition_engaged();
        {
            let mut locked_frames = frames.lock();
            for (seq, &position) in seqs.iter().zip(positions.iter()) {
                let retained = locked_frames
                    .take(seq.slot_idx, seq.session_hash, position)
                    .map_err(anyhow::Error::msg)?;
                target_hiddens.push(storage.offset(retained.selected.offset));
                target_hidden_stacks
                    .push(recondition.then(|| storage.offset(retained.prefix.offset)));
            }
        }
        let mut same_qsa = true;
        let first_qsa = seqs[0].qwen4_qsa_required;
        for (seq, &position) in seqs.iter_mut().zip(positions.iter()) {
            self.k8a_late_replay_if_pending(seq)?;
            self.sync_proposer_rotary(seq)?;
            if !seq.rotary_positions.is_identity() {
                anyhow::ensure!(
                    position == seq.seq_len,
                    "proposer physical position disagrees with target-fed length"
                );
                seq.rotary_positions.tail_scalar(position)?;
            }
            if seq.qwen4_qsa_required != first_qsa {
                same_qsa = false;
            }
        }

        let stream = self.gpu.default_stream();
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

        let ragged_mtp = crate::layers::qwen4_mtp::batched_ragged_mtp_enabled();
        let mut unique_leaders: Vec<usize> = Vec::with_capacity(b);
        let mut leader_of: Vec<usize> = Vec::with_capacity(b);
        for s in 0..b {
            let slot_s = seqs[s].slot_idx;
            let mut matched_u = None;
            if !ragged_mtp
                && let Some(leader_slot) =
                    super::trait_impl::verify_qwen4_batched::dedup_leader_of_slot(slot_s)
            {
                for (u, &l_idx) in unique_leaders.iter().enumerate() {
                    if seqs[l_idx].slot_idx == leader_slot
                        && tokens[s] == tokens[l_idx]
                        && positions[s] == positions[l_idx]
                        && seqs[s].seq_len == seqs[l_idx].seq_len
                        && seqs[s].qwen4_qsa_required == seqs[l_idx].qwen4_qsa_required
                        && target_hidden_stacks[s].is_none()
                        && target_hidden_stacks[l_idx].is_none()
                    {
                        matched_u = Some(u);
                        break;
                    }
                }
            }
            if let Some(u) = matched_u {
                leader_of.push(u);
            } else {
                let u = unique_leaders.len();
                unique_leaders.push(s);
                leader_of.push(u);
            }
        }
        let u_count = unique_leaders.len();
        let qsa_batch_compat = same_qsa || crate::layers::qwen4_mtp_gate::qsa_min_width().is_none();
        if qsa_batch_compat && u_count == b && (2..=64).contains(&u_count) {
            let _qsa_context = crate::layers::qwen4_mtp_gate::QsaContextGuard::new(first_qsa);
            let u_tokens: Vec<u32> = unique_leaders.iter().map(|&s| tokens[s]).collect();
            let u_hiddens: Vec<DevicePtr> =
                unique_leaders.iter().map(|&s| target_hiddens[s]).collect();
            let u_positions: Vec<usize> = unique_leaders.iter().map(|&s| positions[s]).collect();
            let u_stacks: Vec<Option<DevicePtr>> = unique_leaders
                .iter()
                .map(|&s| target_hidden_stacks[s])
                .collect();
            let batch_opt = {
                let mut prop_states: Vec<&mut dyn crate::speculative::ProposerState> =
                    Vec::with_capacity(u_count);
                for (s, seq) in seqs.iter_mut().enumerate() {
                    if unique_leaders[leader_of[s]] == s {
                        let state = seq
                            .proposer_state
                            .as_deref_mut()
                            .ok_or_else(|| anyhow::anyhow!("No proposer state for sequence"))?;
                        prop_states.push(state);
                    }
                }
                proposer.propose_batch(
                    &u_tokens,
                    &u_hiddens,
                    &u_positions,
                    num_drafts,
                    &mut prop_states,
                    &ctx,
                    stream,
                    &u_stacks,
                )?
            };
            if let Some(u_drafts) = batch_opt {
                return Ok(u_drafts);
            }
        }

        let mut u_out = Vec::with_capacity(u_count);
        let mut snapshots = Vec::with_capacity(u_count);
        for &i in &unique_leaders {
            let _qsa_context =
                crate::layers::qwen4_mtp_gate::QsaContextGuard::new(seqs[i].qwen4_qsa_required);
            let prop_state = seqs[i]
                .proposer_state
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("No proposer state for sequence"))?;
            let drafts = proposer.propose(
                tokens[i],
                target_hiddens[i],
                positions[i],
                num_drafts,
                prop_state.as_mut(),
                &ctx,
                stream,
                None,
                None,
                target_hidden_stacks[i],
            )?;
            let snap = prop_state
                .as_any()
                .downcast_ref::<crate::layers::qwen4_mtp::Qwen4MtpState>()
                .map(|st| st.snapshot_dedup_draft());
            snapshots.push(snap);
            u_out.push(drafts);
        }
        if u_count < b {
            for (s, seq) in seqs.iter_mut().enumerate() {
                let u = leader_of[s];
                if unique_leaders[u] != s
                    && let (Some((sl, lnd, ld)), Some(st)) = (
                        snapshots[u].as_ref(),
                        seq.proposer_state.as_deref_mut().and_then(|s| {
                            s.as_any_mut()
                                .downcast_mut::<crate::layers::qwen4_mtp::Qwen4MtpState>()
                        }),
                    )
                {
                    st.apply_dedup_draft(*sl, *lnd, ld);
                }
            }
        }
        Ok((0..b).map(|s| u_out[leader_of[s]].clone()).collect())
    }

    /// Dedicated 9-row staging slice for K7 committed-row rewrite
    /// (`ATLAS_QWEN4_MTP_RECONDITION`), placed past the last-K capture rows
    /// and saved-logits slot so it never aliases `ATLAS_MTP_LASTK_DEFER=1`.
    pub(super) fn mtp_recondition_staging_buf(&self) -> Option<DevicePtr> {
        let buf = self.mtp_lastk_buf?;
        if !self.config.is_qwen4_exp() {
            return None;
        }
        let row_bytes = self.config.residual_width() * 2;
        let lastk_bytes = if self.mtp_lastk_capacity > 0 {
            self.mtp_lastk_capacity * row_bytes + self.config.vocab_size * 4
        } else {
            0
        };
        Some(buf.offset(lastk_bytes))
    }

    /// Borrow the GPU backend for post-construction wiring (e.g. installing
    /// a DFlash proposer that needs to allocate paged KV caches against the
    /// same GPU the target uses).
    pub fn gpu_backend(&self) -> &dyn GpuBackend {
        self.gpu.as_ref()
    }

    /// The target's transposed NVFP4 lm_head (`ATLAS_LM_HEAD_T=1` builds it)
    /// plus its padded row stride, for post-construction sharing with the
    /// DFlash drafter's propose lm_head. The T layout is `[K/2, N_pad]` with
    /// `N_pad = ceil(vocab/64)*64` (see `QuantizedWeight::transpose_for_gemm`),
    /// so a vocab-prefix slice (`--mtp-vocab`) is a column sub-range read with
    /// `ldb = N_pad` — no extra memory.
    pub fn dflash_lm_head_t(&self) -> Option<(crate::weight_map::QuantizedWeight, u32)> {
        self.lm_head_nvfp4_t
            .map(|w| (w, (self.config.vocab_size.div_ceil(64) * 64) as u32))
    }

    /// The target's already-resident untransposed NVFP4 `lm_head` and exact
    /// multi-row NVFP4 LM-head kernel bundle, for zero-allocation sharing with
    /// the DFlash drafter's propose LM head.
    pub fn dflash_lm_head_nvfp4_exact(
        &self,
    ) -> Option<(
        crate::weight_map::QuantizedWeight,
        crate::layers::ops::W4a16ExactLmHeadKernels,
    )> {
        self.lm_head_nvfp4
            .map(|w| (w, self.w4a16_exact_lm_head_kernels))
    }

    /// Install a DFlash drafter as the PRIMARY proposer (arm 0), DEMOTING
    /// whatever MTP proposer `TransformerModel::new` built to arm 1 rather
    /// than dropping it. The target's hidden-state capture buffer is already
    /// allocated when the config's `dflash_capture_layers` is non-empty
    /// (factory.rs populates it before construction), so this method only
    /// arranges the proposer slots.
    ///
    /// Previously this OVERWROTE `self.proposer`, so the native MTP head was
    /// built, logged as ENABLED, and immediately deallocated on every
    /// `--dflash` run. Demoting it costs ~0.78 GB resident at the default
    /// `--mtp-quantization bf16` (0.42 B params on Qwen3.8-27B) and is what
    /// lets the scheduler's speculation gate arbitrate between the external
    /// drafter and the native head at runtime.
    ///
    /// The `--dflash` / `--speculative` CLI exclusion that used to make the
    /// demotion unreachable has been lifted; passing both is now the
    /// documented way to ask for a two-arm gate. Passing only one still
    /// yields exactly one proposer and `proposer_alt == None`.
    pub fn set_dflash_proposer(
        &mut self,
        mut proposer: std::sync::Arc<dyn DraftProposer>,
    ) -> Result<()> {
        crate::speculative::validate_dflash_install_capacity(
            proposer.is_dflash(),
            proposer.physical_verify_k(),
            self.ddtree_parent_ids_capacity,
        )?;
        if self.config.is_qwen4_exp() && self.config.yarn_factor > 0.0 {
            let inner = std::sync::Arc::get_mut(&mut proposer).ok_or_else(|| {
                anyhow::anyhow!(
                    "set_dflash_proposer: shared Arc prevents syncing Qwen4 YaRN target config"
                )
            })?;
            inner.sync_target_config(&self.config, self.gpu.as_ref())?;
            if !self.target_rejection_q_buf.is_null() {
                inner.bind_rejection_q_scratch(
                    self.target_rejection_q_buf,
                    self.target_rejection_q_buf
                        .offset(crate::model::trait_impl::device_topk::REJECTION_Q_PROBS_OFFSET),
                    crate::model::trait_impl::device_topk::REJECTION_Q_ROW_CAP,
                    crate::model::trait_impl::device_topk::REJECTION_Q_STRIDE,
                );
            }
        } else if let Some(inner) = std::sync::Arc::get_mut(&mut proposer) {
            inner.sync_target_config(&self.config, self.gpu.as_ref())?;
            if !self.target_rejection_q_buf.is_null() {
                inner.bind_rejection_q_scratch(
                    self.target_rejection_q_buf,
                    self.target_rejection_q_buf
                        .offset(crate::model::trait_impl::device_topk::REJECTION_Q_PROBS_OFFSET),
                    crate::model::trait_impl::device_topk::REJECTION_Q_ROW_CAP,
                    crate::model::trait_impl::device_topk::REJECTION_Q_STRIDE,
                );
            }
        }
        if let Some(mut prev) = self.proposer.take() {
            if !self.target_rejection_q_buf.is_null()
                && let Some(prev_inner) = std::sync::Arc::get_mut(&mut prev)
            {
                prev_inner.bind_rejection_q_scratch(
                    self.target_rejection_q_buf,
                    self.target_rejection_q_buf
                        .offset(crate::model::trait_impl::device_topk::REJECTION_Q_PROBS_OFFSET),
                    crate::model::trait_impl::device_topk::REJECTION_Q_ROW_CAP,
                    crate::model::trait_impl::device_topk::REJECTION_Q_STRIDE,
                );
            }
            tracing::info!(
                "DFlash: BlockDiffusionDraftHead installed as proposer arm 0; \
                 existing MTP proposer DEMOTED to arm 1 (gate-selectable)"
            );
            self.proposer_alt = Some(prev);
        }
        self.proposer = Some(proposer);
        self.active_proposer_arm
            .store(0, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// The proposer for the arm the speculation gate currently has selected.
    ///
    /// Falls back to arm 0 whenever arm 1 is empty, so every single-proposer
    /// build behaves exactly as it did when `self.proposer` was read directly.
    pub(super) fn active_proposer(&self) -> Option<&std::sync::Arc<dyn DraftProposer>> {
        if self
            .active_proposer_arm
            .load(std::sync::atomic::Ordering::Relaxed)
            == 1
            && let Some(alt) = self.proposer_alt.as_ref()
        {
            return Some(alt);
        }
        self.proposer.as_ref()
    }

    /// The proposer for the arm that is NOT live — the one whose per-sequence
    /// state is parked in `SequenceState::proposer_state_alt`.
    ///
    /// Pairing matters: `proposer_state` belongs to the ACTIVE arm, so a
    /// teardown that frees it must do so through `active_proposer`, and free
    /// `proposer_state_alt` through this one. Pairing by field order instead
    /// would hand one arm's state to the other arm's allocator whenever the
    /// gate happened to be sitting on arm 1.
    pub(super) fn inactive_proposer(&self) -> Option<&std::sync::Arc<dyn DraftProposer>> {
        if self
            .active_proposer_arm
            .load(std::sync::atomic::Ordering::Relaxed)
            == 1
            && self.proposer_alt.is_some()
        {
            self.proposer.as_ref()
        } else {
            self.proposer_alt.as_ref()
        }
    }

    /// Is ANY proposer installed? The capability question, as distinct from
    /// "which arm is live" — used by `has_proposer` and by buffer-allocation
    /// gates that must provision for either arm.
    pub(super) fn any_proposer(&self) -> bool {
        self.proposer.is_some() || self.proposer_alt.is_some()
    }

    /// How many proposer arms this build can offer the gate (0, 1 or 2).
    pub(super) fn proposer_arm_count(&self) -> usize {
        self.proposer.iter().count() + self.proposer_alt.iter().count()
    }

    /// Construction identity for a physical arm, independent of which arm
    /// the runtime gate currently has selected.
    pub(super) fn proposer_arm_kind(
        &self,
        arm: u8,
    ) -> Option<crate::speculative::DraftProposerKind> {
        match arm {
            0 => self.proposer.as_ref(),
            1 => self.proposer_alt.as_ref(),
            _ => None,
        }
        .map(|proposer| proposer.kind())
    }

    /// Point the gate at a proposer arm. Returns the arm actually selected,
    /// which is 0 when the requested arm does not exist on this build.
    ///
    /// Callers MUST pair this with `swap_proposer_state` on every live
    /// sequence — see that method for why.
    pub(super) fn set_proposer_arm(&self, arm: u8) -> u8 {
        let effective = if arm == 1 && self.proposer_alt.is_some() {
            1
        } else {
            0
        };
        self.active_proposer_arm
            .store(effective, std::sync::atomic::Ordering::Relaxed);
        effective
    }

    pub(super) fn proposer_arm(&self) -> u8 {
        self.active_proposer_arm
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// DFlash prefill capture: copy `proc_count` tokens × hidden_size BF16
    /// from `self.buffers.hidden_states()` (filled by the just-completed
    /// prefill layer) into the per-sequence DFlash accumulator. Called
    /// inside the prefill layer loop after each layer. No-op when:
    ///   - DFlash is disabled (capture_layers empty)
    ///   - `layer_idx` is not in `dflash_capture_layers`
    ///   - The seq has no `DflashProposerState`
    ///   - Rank > 0 under EP/TP (drafter is rank-0 only)
    ///
    /// Layout: writes `hidden[t]` BF16 into physical accumulator slot
    /// `(chunk_start + t) % ctx_capacity`, capture slice `slot_idx`.
    /// Per-layer call performs `proc_count` strided d2d_async copies —
    /// at typical prefill of 128–4096 tokens × 5 capture layers, total
    /// 640–20480 launches per prefill. Acceptable launch overhead for
    /// first land; replace with a strided-scatter kernel if profiling
    /// shows it's a bottleneck.
    pub(super) fn try_dflash_prefill_capture_layer(
        &self,
        seq: &mut crate::traits::SequenceState,
        layer_idx: usize,
        chunk_start: usize,
        proc_count: usize,
        stream: u64,
    ) -> Result<()> {
        if self.dflash_capture_layers.is_empty() {
            return Ok(());
        }
        let slot_idx = match self
            .dflash_capture_layers
            .iter()
            .position(|&l| l == layer_idx)
        {
            Some(s) => s,
            None => return Ok(()),
        };
        if let Some(ref c) = self.comm
            && c.rank() != 0
        {
            return Ok(());
        }
        let dstate = match seq.proposer_state.as_mut() {
            Some(ps) => match ps
                .as_any_mut()
                .downcast_mut::<crate::layers::DflashProposerState>()
            {
                Some(s) => s,
                None => return Ok(()),
            },
            None => return Ok(()),
        };
        let h = self.dflash_capture_width;
        // Each ring slot holds one BF16 row per capture layer: `n_capture * h * 2` bytes.
        let n_capture = self.dflash_capture_layers.len();
        let bf16 = 2usize;
        let acc_base = dstate.ctx_hidden_acc;
        if proc_count == 0 {
            return Ok(());
        }
        anyhow::ensure!(
            dstate.ctx_slot_bytes == n_capture * h * bf16,
            "DFlash capture slot geometry does not match proposer ring"
        );
        // Defensive Marconi prefix-hit catch-up (also covers non-chunked prefill):
        // when zero real rows have been captured yet (`ctx_len == ctx_valid_from_abs`)
        // and the prefix cache covered `[0..chunk_start)`, advance the ring cursor
        // with `ctx_valid_from_abs = chunk_start` so `plan_append_at` aligns without
        // exposing uninitialized ring slots `< chunk_start`.
        if dstate.ctx_len < chunk_start
            && dstate.ctx_len == dstate.ctx_valid_from_abs
            && seq.cached_prefix_tokens >= chunk_start
        {
            dstate.advance_marconi_skip(chunk_start)?;
        }
        let append = dstate
            .ctx_ring_state()?
            .plan_append_at(chunk_start, proc_count)?;
        let capture_count = proc_count;
        let source_stride;
        let mut src_linear_base = 0usize;
        let src_base = match self.dflash_capture_mode {
            DflashCaptureMode::ResidualSlice => {
                source_stride = self.config.residual_width();
                self.buffers
                    .hidden_states()
                    .offset(self.dflash_capture_offset * bf16)
            }
            DflashCaptureMode::Qwen4HyperProjected => {
                let mixer = self.qwen4_final_mixer.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("projected DFlash capture missing Qwen4 hyper mixer")
                })?;
                source_stride = h;
                if std::env::var("ATLAS_QWEN4_HYPER_PREFILL_GEMM")
                    .ok()
                    .as_deref()
                    == Some("1")
                {
                    mixer.prepare_prefill(
                        self.buffers.hidden_states(),
                        self.buffers.residual(),
                        capture_count,
                        &self.buffers,
                        self.gpu.as_ref(),
                        self.config.rms_norm_eps as f32,
                        stream,
                    )?
                } else {
                    // Match the target layer's default Qwen4 prefill path.
                    // Switching only the capture projection to the faster
                    // tensor-core GEMM changes BF16 features between prompt
                    // and decode even when the target itself stays exact.
                    mixer.prepare_prefill_exact(
                        self.buffers.hidden_states(),
                        self.buffers.residual(),
                        capture_count,
                        &self.buffers,
                        self.gpu.as_ref(),
                        self.config.rms_norm_eps as f32,
                        stream,
                    )?
                }
            }
            DflashCaptureMode::Qwen4LayerAttnHc => {
                let next_layer = layer_idx + 1;
                let mixer = self
                    .layers
                    .get(next_layer)
                    .and_then(|layer| layer.qwen4_attn_hyper())
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "Qwen4LayerAttnHc capture missing layers[{next_layer}].qwen4_attn_hyper()"
                        )
                    })?;
                source_stride = h;
                mixer.mix_only_rows_nvfp4(
                    self.buffers.hidden_states(),
                    capture_count,
                    self.buffers.norm_output(),
                    &self.buffers,
                    self.gpu.as_ref(),
                    self.config.rms_norm_eps as f32,
                    stream,
                )?
            }
            DflashCaptureMode::Qwen4HcMean => {
                source_stride = h;
                anyhow::ensure!(
                    self.config.residual_width() == 4 * h,
                    "Qwen4HcMean capture expects residual_width={} == 4 * h ({})",
                    self.config.residual_width(),
                    4 * h
                );
                let out = self.buffers.norm_output();
                let hyper_row_bytes = self
                    .config
                    .residual_width()
                    .checked_mul(bf16)
                    .ok_or_else(|| anyhow::anyhow!("Qwen4HcMean hyper row bytes overflow"))?;
                let out_row_bytes = h
                    .checked_mul(bf16)
                    .ok_or_else(|| anyhow::anyhow!("Qwen4HcMean output row bytes overflow"))?;
                let (linear_base, slices) = plan_hc_mean_slices(append.write.spans())?;
                src_linear_base = linear_base;
                for slice in slices {
                    let slice_end = slice
                        .src_linear_row
                        .checked_add(slice.row_count)
                        .ok_or_else(|| anyhow::anyhow!("Qwen4HcMean slice end overflow"))?;
                    anyhow::ensure!(
                        slice_end <= proc_count,
                        "Qwen4HcMean slice [{}..{slice_end}) exceeds proc_count={proc_count}",
                        slice.src_linear_row
                    );
                    let hyper_off = slice
                        .src_linear_row
                        .checked_mul(hyper_row_bytes)
                        .ok_or_else(|| anyhow::anyhow!("Qwen4HcMean hyper offset overflow"))?;
                    let dst_off = slice
                        .dst_row
                        .checked_mul(out_row_bytes)
                        .ok_or_else(|| anyhow::anyhow!("Qwen4HcMean dst offset overflow"))?;
                    self.launch_qwen4_hc_mean_rows(
                        self.buffers.hidden_states().offset(hyper_off),
                        out.offset(dst_off),
                        slice.row_count,
                        h,
                        stream,
                    )?;
                }
                out
            }
        };
        let strided =
            crate::layers::dflash_capture_strided_enabled() && self.strided_copy_rows_kernel.0 != 0;
        for span in append.write.spans() {
            let src_linear_offset =
                span.linear_slot
                    .checked_sub(src_linear_base)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "DFlash capture span linear_slot={} precedes reduced slice base={src_linear_base}",
                            span.linear_slot
                        )
                    })?;
            if strided {
                // One strided copy per span: rows are consecutive physical
                // slots, so the destination stride is exactly one slot. The
                // per-row offset check below is kept for the span's ends,
                // which bound every row in between.
                let first_abs = chunk_start + span.linear_slot;
                let last_abs = first_abs + span.slot_count - 1;
                let offset_of = |abs| {
                    crate::layers::dflash_head::ring_window::accumulator_capture_offset(
                        abs,
                        slot_idx,
                        dstate.ctx_capacity,
                        dstate.ctx_slot_bytes,
                        h * bf16,
                    )
                };
                let dst_offset = offset_of(first_abs)?;
                anyhow::ensure!(
                    dst_offset / dstate.ctx_slot_bytes == span.physical_slot
                        && offset_of(last_abs)? / dstate.ctx_slot_bytes
                            == span.physical_slot + span.slot_count - 1,
                    "DFlash capture plan and byte offset disagree"
                );
                crate::layers::ops::strided_copy_rows(
                    self.gpu.as_ref(),
                    self.strided_copy_rows_kernel,
                    src_base.offset(src_linear_offset * source_stride * bf16),
                    acc_base.offset(dst_offset),
                    span.slot_count as u32,
                    (h * bf16) as u32,
                    (source_stride * bf16) as u64,
                    dstate.ctx_slot_bytes as u64,
                    stream,
                )?;
                continue;
            }
            for local in 0..span.slot_count {
                let source_row = span.linear_slot + local;
                let physical_slot = span.physical_slot + local;
                let src = src_base.offset((src_linear_offset + local) * source_stride * bf16);
                let absolute_position = chunk_start + source_row;
                let dst_offset =
                    crate::layers::dflash_head::ring_window::accumulator_capture_offset(
                        absolute_position,
                        slot_idx,
                        dstate.ctx_capacity,
                        dstate.ctx_slot_bytes,
                        h * bf16,
                    )?;
                anyhow::ensure!(
                    dst_offset / dstate.ctx_slot_bytes == physical_slot,
                    "DFlash capture plan and byte offset disagree"
                );
                self.gpu
                    .copy_d2d_async(src, acc_base.offset(dst_offset), h * bf16, stream)?;
            }
        }
        Ok(())
    }

    /// Whether [`Self::mtp_lastk_capture_chunk`] reads each prefill chunk's
    /// tail rows. The last-layer crop (`ATLAS_QWEN4_PREFILL_LAST_LAYER_CROP`)
    /// refuses exactly when this holds.
    pub(super) fn mtp_lastk_capture_active(&self) -> bool {
        mtp_lastk_capture_runs(self.mtp_lastk_capacity, self.any_proposer())
    }

    /// Per-chunk MTP last-K capture (called from `prefill_chunk_dispatch` after
    /// `forward_layers`). Copies the just-computed chunk's hidden states into
    /// the per-sequence host ring buffer `seq.mtp_lastk_host_buf`, shifting
    /// older rows out when the ring is full. The actual H2D into the shared
    /// `mtp_lastk_buf` device buffer + proposer `prefill_last_k` happens at
    /// `mtp_lastk_prefill_after_finalize` on the last chunk only.
    ///
    /// No-op when MTP last-K prefill is disabled (capacity==0), no proposer
    /// is wired, or rank > 0 under EP/TP.
    ///
    /// `proc_count` is the number of rows in `hidden_states` for this chunk.
    /// `chunk_start + chunk_len - 1` is the absolute position of the last
    /// captured row.
    pub(super) fn mtp_lastk_capture_chunk(
        &self,
        seq: &mut crate::traits::SequenceState,
        chunk_start: usize,
        chunk_len: usize,
        proc_count: usize,
        total_tokens: usize,
        stream: u64,
    ) -> Result<()> {
        super::speculation_control::run_lastk_if_enabled(self.speculation_forced_disabled, || {
            self.mtp_lastk_capture_chunk_enabled(
                seq,
                chunk_start,
                chunk_len,
                proc_count,
                total_tokens,
                stream,
            )
        })?;
        Ok(())
    }

    fn mtp_lastk_capture_chunk_enabled(
        &self,
        seq: &mut crate::traits::SequenceState,
        chunk_start: usize,
        chunk_len: usize,
        proc_count: usize,
        total_tokens: usize,
        stream: u64,
    ) -> Result<()> {
        let capacity = self.mtp_lastk_capacity;
        if !self.mtp_lastk_capture_active() {
            return Ok(());
        }
        if let Some(ref c) = self.comm
            && c.rank() != 0
        {
            return Ok(());
        }
        if proc_count == 0 {
            return Ok(());
        }
        // Any chunk ending at or before `total_tokens - capacity` whose successor
        // chunks lie beyond the cached prefix will have all `capacity` rows of
        // `mtp_lastk_host_buf` completely overwritten by those later chunks.
        // Enabled by default; `ATLAS_QWEN4_MTP_LASTK_SKIP_INTERMEDIATE=0` opts out.
        let window_start = total_tokens.saturating_sub(capacity);
        if mtp_lastk_skip_intermediate_enabled()
            && chunk_start + chunk_len <= window_start
            && seq.cached_prefix_tokens <= window_start
        {
            return Ok(());
        }
        let row_bytes = if self.config.is_qwen4_exp() {
            self.config.residual_width() * 2
        } else {
            let element_bytes = if self.config.use_fp32_residual() {
                4
            } else {
                2
            };
            self.config.hidden_size * element_bytes
        };
        let ring_capacity_bytes = capacity * row_bytes;

        // Lazy-size the host ring buffer.
        if seq.mtp_lastk_host_buf.len() != ring_capacity_bytes {
            seq.mtp_lastk_host_buf = vec![0u8; ring_capacity_bytes];
            seq.mtp_lastk_host_filled = 0;
            seq.mtp_lastk_end_abs = 0;
        }

        // Number of rows to ingest from this chunk: at most `capacity` (the
        // last `capacity` rows of the chunk fully replace the ring), at most
        // `proc_count` rows present in `hidden_states`.
        let ingest = capacity.min(proc_count);
        if ingest == 0 {
            return Ok(());
        }

        // Source row range in `hidden_states`: the trailing `ingest` rows of
        // the chunk's proc_count outputs.
        let src_first_row = proc_count - ingest;
        let src_bytes = ingest * row_bytes;
        let src_ptr = self
            .buffers
            .hidden_states()
            .offset(src_first_row * row_bytes);

        if ingest >= capacity {
            // Chunk produced ≥ K rows — wipe ring and copy the last K rows.
            // D2H directly into the host ring buf.
            // Invariant: copy_d2h requires `&mut [u8]` of exact size; the ring is
            // sized to `ring_capacity_bytes` and `src_bytes == ring_capacity_bytes`.
            self.gpu.synchronize(stream)?;
            self.gpu
                .copy_d2h(src_ptr, &mut seq.mtp_lastk_host_buf[..src_bytes])?;
            seq.mtp_lastk_host_filled = capacity;
        } else {
            // Shift the existing ring left by `ingest` rows (drop oldest),
            // then append the new rows at the tail.
            let shift_bytes = ingest * row_bytes;
            if seq.mtp_lastk_host_filled == capacity {
                // Ring is full: drop oldest `ingest` rows.
                seq.mtp_lastk_host_buf
                    .copy_within(shift_bytes..ring_capacity_bytes, 0);
                let tail_start = ring_capacity_bytes - shift_bytes;
                self.gpu.synchronize(stream)?;
                self.gpu.copy_d2h(
                    src_ptr,
                    &mut seq.mtp_lastk_host_buf[tail_start..tail_start + shift_bytes],
                )?;
            } else if seq.mtp_lastk_host_filled + ingest <= capacity {
                // Ring has room: append at `filled`.
                let tail_start = seq.mtp_lastk_host_filled * row_bytes;
                self.gpu.synchronize(stream)?;
                self.gpu.copy_d2h(
                    src_ptr,
                    &mut seq.mtp_lastk_host_buf[tail_start..tail_start + shift_bytes],
                )?;
                seq.mtp_lastk_host_filled += ingest;
            } else {
                // Partial overlap: existing rows shift out the front, new
                // rows fill the tail. New filled count is `capacity`.
                let drop = seq.mtp_lastk_host_filled + ingest - capacity;
                let drop_bytes = drop * row_bytes;
                let keep_bytes = seq.mtp_lastk_host_filled * row_bytes - drop_bytes;
                seq.mtp_lastk_host_buf
                    .copy_within(drop_bytes..drop_bytes + keep_bytes, 0);
                let tail_start = keep_bytes;
                self.gpu.synchronize(stream)?;
                self.gpu.copy_d2h(
                    src_ptr,
                    &mut seq.mtp_lastk_host_buf[tail_start..tail_start + shift_bytes],
                )?;
                seq.mtp_lastk_host_filled = capacity;
            }
        }
        seq.mtp_lastk_end_abs = chunk_start + chunk_len - 1;

        if std::env::var("ATLAS_MTP_DEBUG").ok().as_deref() == Some("1") {
            tracing::info!(
                "ATLAS_MTP_DEBUG mtp_lastk_capture_chunk: \
                 chunk=[{chunk_start},{}] proc_count={proc_count} \
                 ingest={ingest} filled={}/{capacity} end_abs={}",
                chunk_start + chunk_len,
                seq.mtp_lastk_host_filled,
                seq.mtp_lastk_end_abs,
            );
        }
        static CAPTURED: std::sync::Once = std::sync::Once::new();
        let filled = seq.mtp_lastk_host_filled;
        CAPTURED.call_once(|| {
            tracing::info!(
                "MTP_LASTK_CAPTURE_ENGAGED capacity={capacity} ingest={ingest} proc_count={proc_count} filled={filled}"
            );
        });
        Ok(())
    }

    /// MTP last-K prefill (called from `prefill_b_finalize_last` on the last
    /// chunk). H2Ds the per-sequence cross-chunk host ring buffer
    /// (`seq.mtp_lastk_host_buf`, populated by `mtp_lastk_capture_chunk` on
    /// every chunk) into the shared `self.mtp_lastk_buf`, then asks the
    /// proposer to replay the rows so its self-attention KV cache covers the
    /// prompt tail before the first decode.
    ///
    /// No-op when:
    ///   - `mtp_lastk_capacity == 0` (env-gate disabled)
    ///   - No proposer wired (e.g. non-speculative serve)
    ///   - No `mtp_lastk_buf` allocated (defensive)
    ///   - Rank > 0 under EP/TP (MTP runs on rank 0 only)
    ///   - `seq.mtp_lastk_host_filled == 0` (capture never fired)
    ///
    /// `proc_count` and `chunk_len` are retained as args for backward
    /// compatibility with the call site but are no longer used to size the
    /// captured window — the host ring already holds the cross-chunk tail.
    pub(super) fn mtp_lastk_prefill_after_finalize(
        &self,
        tokens: &[u32],
        seq: &mut crate::traits::SequenceState,
        _chunk_start: usize,
        _chunk_len: usize,
        _proc_count: usize,
        stream: u64,
    ) -> Result<()> {
        if self.k8a_defer_lastk(tokens, seq, stream)? {
            return Ok(());
        }
        super::speculation_control::run_lastk_if_enabled(self.speculation_forced_disabled, || {
            self.mtp_lastk_prefill_after_finalize_enabled(
                tokens,
                seq,
                _chunk_start,
                _chunk_len,
                _proc_count,
                stream,
            )
        })?;
        Ok(())
    }

    pub(super) fn mtp_lastk_prefill_after_finalize_enabled(
        &self,
        tokens: &[u32],
        seq: &mut crate::traits::SequenceState,
        _chunk_start: usize,
        _chunk_len: usize,
        _proc_count: usize,
        stream: u64,
    ) -> Result<()> {
        let capacity = self.mtp_lastk_capacity;
        if capacity == 0 {
            return Ok(());
        }
        let proposer = match self.active_proposer() {
            Some(p) => p.clone(),
            None => return Ok(()),
        };
        self.sync_proposer_rotary(seq)?;
        let lastk_buf = match self.mtp_lastk_buf {
            Some(p) => p,
            None => return Ok(()),
        };
        if let Some(ref c) = self.comm
            && c.rank() != 0
        {
            return Ok(());
        }

        // Read the per-seq cross-chunk capture state populated by
        // `mtp_lastk_capture_chunk` at every prefill chunk.
        let filled = seq.mtp_lastk_host_filled;
        if filled == 0 || seq.mtp_lastk_host_buf.is_empty() {
            // Capture never fired (e.g. all chunks early-returned via cache).
            // Nothing to prefill into MTP.
            tracing::warn!(
                "MTP last-K prefill: skipped — host ring empty (filled={filled}, \
                 buf_bytes={}). Likely all chunks were prefix-cache early-returns.",
                seq.mtp_lastk_host_buf.len(),
            );
            return Ok(());
        }
        let end_abs = seq.mtp_lastk_end_abs;
        let row_bytes = if self.config.is_qwen4_exp() {
            self.config.residual_width() * 2
        } else {
            let element_bytes = if self.config.use_fp32_residual() {
                4
            } else {
                2
            };
            self.config.hidden_size * element_bytes
        };
        let used_bytes = filled * row_bytes;
        let start_abs = end_abs
            .checked_add(1)
            .and_then(|end| end.checked_sub(filled))
            .ok_or_else(|| anyhow::anyhow!("MTP last-K captured span overflow"))?;
        anyhow::ensure!(
            start_abs <= end_abs
                && end_abs < tokens.len()
                && filled <= capacity
                && used_bytes <= seq.mtp_lastk_host_buf.len(),
            "MTP last-K captured span exceeds token/hidden history"
        );
        let predict_into = start_abs
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("MTP last-K predict-into overflow"))?;
        seq.rotary_positions.range(predict_into, filled)?;

        // Live rows are always stored contiguously in `[0 .. filled * row_bytes]` (oldest first).
        let src_slice = &seq.mtp_lastk_host_buf[..used_bytes];
        maybe_dump_lastk_ring(src_slice, end_abs, filled);
        self.copy_h2d_group_fast(&[HostToDeviceCopy::new(src_slice, lastk_buf)], stream)?;

        // The captured tokens span `[end_abs - filled + 1 ..= end_abs]`.
        let captured_tokens: Vec<u32> = tokens[start_abs..=end_abs].to_vec();

        // Build a ForwardContext mirroring `run_mtp_propose_inner`. MTP
        // loads ALL experts on every rank, so MoE output is already complete
        // → no all_reduce needed (comm: None).
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

        let prop_state = match seq.proposer_state.as_mut() {
            Some(ps) => ps,
            None => return Ok(()),
        };

        // The replay runs the MTP layer over the shared activation buffers
        // (its MoE and head use `buffers.logits()` as scratch), and the
        // scheduler samples the first output token from the target's logits
        // after this returns. Park them in the capture buffer's tail and
        // restore them after the replay.
        let logits = self.decode_logits_ptr();
        let logits_bytes = self.config.vocab_size * if self.use_fp32_logits { 4 } else { 2 };
        let saved_logits = lastk_buf.offset(capacity * row_bytes);
        self.gpu
            .copy_d2d_async(logits, saved_logits, logits_bytes, stream)?;

        let t0 = std::time::Instant::now();
        proposer.prefill_last_k(
            &captured_tokens,
            lastk_buf,
            end_abs,
            prop_state.as_mut(),
            &ctx,
            stream,
        )?;
        self.gpu
            .copy_d2d_async(saved_logits, logits, logits_bytes, stream)?;
        let dt_ms = t0.elapsed().as_secs_f64() * 1000.0;
        tracing::info!(
            "MTP last-K prefill: filled={filled}/{capacity} rows from cross-chunk ring, \
             end_abs={end_abs} (span=[{start_abs},{end_abs}]), took {dt_ms:.1}ms",
        );
        Ok(())
    }

    /// Advance DFlash's absolute cursor past a Marconi prefix-cache hit
    /// `[0..skip_to)` while recording `ctx_valid_from_abs = skip_to` so
    /// cross-attention never reads uninitialized target hidden slots `< skip_to`.
    pub(in crate::model) fn advance_dflash_marconi_skip(
        &self,
        seq: &mut crate::traits::SequenceState,
        skip_to: usize,
    ) -> Result<()> {
        if self.dflash_capture_layers.is_empty() {
            return Ok(());
        }
        if let Some(ref c) = self.comm
            && c.rank() != 0
        {
            return Ok(());
        }
        if let Some(ps) = seq.proposer_state.as_mut()
            && let Some(dstate) = ps
                .as_any_mut()
                .downcast_mut::<crate::layers::DflashProposerState>()
        {
            dstate.advance_marconi_skip(skip_to)?;
        }
        Ok(())
    }

    /// After a prefill chunk, advance the seq's DFlash `ctx_len` to
    /// `chunk_start + proc_count`: the next chunk's capture plans its ring
    /// writes against this cursor, and the drafter sees all captured prompt
    /// positions on the first propose() call.
    pub(super) fn update_dflash_ctx_len_after_prefill(
        &self,
        seq: &mut crate::traits::SequenceState,
        chunk_start: usize,
        proc_count: usize,
    ) -> Result<()> {
        if self.dflash_capture_layers.is_empty() {
            return Ok(());
        }
        if let Some(ref c) = self.comm
            && c.rank() != 0
        {
            return Ok(());
        }
        if let Some(ps) = seq.proposer_state.as_mut()
            && let Some(dstate) = ps
                .as_any_mut()
                .downcast_mut::<crate::layers::DflashProposerState>()
        {
            if proc_count == 0 {
                return Ok(());
            }
            if let Some(next) = dstate
                .ctx_ring_state()?
                .advanced_after_chunk(chunk_start, proc_count)?
            {
                dstate.apply_ctx_ring_state(next)?;
            }
        }
        Ok(())
    }

    /// DFlash drafter-retrain teacher-forced capture. When
    /// `ATLAS_DUMP_CTX_HIDDEN=<path>` is set, dump the per-sequence
    /// chronological resident tail of `ctx_hidden_acc` (at most 4096 positions
    /// × capture layers, computed on the serving path) to the append-only file.
    /// One record per request. Long prompts therefore require training-token
    /// truncation to the same absolute tail; the runtime logs that contract.
    /// No-op when the env var is unset, DFlash capture is inactive, or rank > 0
    /// under EP/TP.
    ///
    /// The physical device layout is `[ring_slot, capture, hidden]` BF16. The
    /// dump gathers it chronologically into the SpecForge offline `[T, L*H]`
    /// tensor with L = capture layers in
    /// `dflash_capture_layers` order (e.g. [1,16,31,46,61]). The Python
    /// harness pads to SPECFORGE_PAD_TO and saves `{md5(padded ids)}.pt`.
    ///
    /// Record format (little-endian):
    /// ```text
    /// u32 magic        = 0xC7D5_1DEE
    /// u32 seq_len      = number of resident positions dumped (= ring.resident_len)
    /// u32 n_capture    = number of capture layers (`self.dflash_capture_layers.len()`)
    /// u32 hidden_dim   = capture width (`self.dflash_capture_width`)
    /// u64 tok_fnv      = FNV-1a of the request's prompt-token bytes (pairing)
    /// bf16 payload[seq_len * n_capture * hidden_dim]  (layout [pos, slot, h])
    /// ```
    /// Must run in eager mode (synchronous `copy_d2h`); prefill is already
    /// eager (no CUDA-graph capture on the prefill path).
    pub(super) fn dump_ctx_hidden_after_prefill(
        &self,
        seq: &mut crate::traits::SequenceState,
        tokens: &[u32],
        stream: u64,
    ) -> Result<()> {
        let path = match crate::model::env_diag::dump_ctx_hidden_path() {
            Some(p) => p,
            None => return Ok(()),
        };
        if self.dflash_capture_layers.is_empty() {
            return Ok(());
        }
        if let Some(ref c) = self.comm
            && c.rank() != 0
        {
            return Ok(());
        }
        let h = self.dflash_capture_width;
        let bf16 = 2usize;
        let n_capture = self.dflash_capture_layers.len();

        let (acc_base, ring) = match seq.proposer_state.as_mut() {
            Some(ps) => match ps
                .as_any_mut()
                .downcast_mut::<crate::layers::DflashProposerState>()
            {
                Some(dstate) => (dstate.ctx_hidden_acc, dstate.ctx_ring_state()?),
                None => return Ok(()),
            },
            None => return Ok(()),
        };
        let n = ring.resident_len;
        if acc_base.0 == 0 || n == 0 {
            tracing::warn!(
                "ATLAS_DUMP_CTX_HIDDEN: no ctx_hidden_acc/ctx_len (acc={:#x}, n={}) — skipping dump",
                acc_base.0,
                n
            );
            return Ok(());
        }

        // The capture copies run on the prefill stream; `copy_d2h` only
        // drains the default stream, so without this the dump can read
        // slots whose copies are still queued.
        self.gpu.synchronize(stream)?;
        let resident_start = ring.resident_start();
        let gather = ring.plan_gather(resident_start, n)?;
        let slot_bytes = n_capture
            .checked_mul(h)
            .and_then(|elements| elements.checked_mul(bf16))
            .ok_or_else(|| anyhow::anyhow!("DFlash dump slot size overflowed"))?;
        let total_bytes = n
            .checked_mul(slot_bytes)
            .ok_or_else(|| anyhow::anyhow!("DFlash dump payload size overflowed"))?;
        let mut host_buf = vec![0u8; total_bytes];
        for span in gather.spans() {
            let byte_count = span.slot_count * slot_bytes;
            let dst_start = span.dst_slot * slot_bytes;
            self.gpu.copy_d2h(
                acc_base.offset(span.src_slot * slot_bytes),
                &mut host_buf[dst_start..dst_start + byte_count],
            )?;
        }

        // FNV-1a over the prompt token bytes — lets the Python harness assert
        // it paired the right record with the right sample.
        const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const FNV_PRIME: u64 = 0x100_0000_01b3;
        let mut fnv: u64 = FNV_OFFSET;
        for &t in tokens {
            for &b in t.to_le_bytes().iter() {
                fnv ^= b as u64;
                fnv = fnv.wrapping_mul(FNV_PRIME);
            }
        }

        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| anyhow::anyhow!("ATLAS_DUMP_CTX_HIDDEN open {path}: {e}"))?;
        const CTX_HIDDEN_MAGIC: u32 = 0xC7D5_1DEE;
        f.write_all(&CTX_HIDDEN_MAGIC.to_le_bytes())?;
        f.write_all(&(n as u32).to_le_bytes())?;
        f.write_all(&(n_capture as u32).to_le_bytes())?;
        f.write_all(&(h as u32).to_le_bytes())?;
        f.write_all(&fnv.to_le_bytes())?;
        f.write_all(&host_buf)?;
        if ring.absolute_len > ring.resident_len {
            tracing::warn!(
                "ATLAS_DUMP_CTX_HIDDEN: bounded-ring payload contains only chronological absolute positions [{}..{}); retraining must truncate tokens to the same {}-slot window and requalify acceptance",
                resident_start,
                ring.absolute_len,
                ring.capacity,
            );
        }
        tracing::info!(
            "ATLAS_DUMP_CTX_HIDDEN: wrote resident seq_len={n} absolute_len={} absolute_start={resident_start} n_capture={n_capture} h={h} \
             ({total_bytes} bytes payload) fnv={fnv:#018x} → {path}",
            ring.absolute_len,
        );
        Ok(())
    }

    /// Capture one serial-decode token's target hidden row (`dflash_hidden_save[0]`)
    /// into the sequence's bounded DFlash context ring and advance `ctx_len` in
    /// lockstep with `seq.seq_len`.
    ///
    /// Runs unconditionally whenever DFlash is the active proposer so plain/serial
    /// decode fallbacks (thinking spans, budget nudges, grammar-gated steps) never
    /// drift `ctx_len` behind `seq.seq_len` and trigger `DFLASH_CURSOR_RECONCILE_FAILED`
    /// when speculation resumes. Must be called AFTER `decode` has incremented
    /// `seq.seq_len`, so the just-decoded token sits at `seq.seq_len - 1`.
    /// Idempotent when `dstate.ctx_len == seq.seq_len`.
    ///
    /// Drafter-conditioning ONLY: the target's verify path is untouched, so
    /// committed tokens remain byte-identical.
    pub(super) fn capture_dflash_serial_decode_row(
        &self,
        seq: &mut crate::traits::SequenceState,
        stream: u64,
    ) -> Result<()> {
        if !self.active_proposer().is_some_and(|p| p.is_dflash()) {
            return Ok(());
        }
        if self.dflash_capture_layers.is_empty() {
            return Ok(());
        }
        let Some(src_base) = self.dflash_hidden_save.filter(|p| !p.is_null()) else {
            return Ok(());
        };
        if let Some(ref c) = self.comm
            && c.rank() != 0
        {
            return Ok(());
        }
        let Some(dstate) = seq.proposer_state.as_mut().and_then(|ps| {
            ps.as_any_mut()
                .downcast_mut::<crate::layers::DflashProposerState>()
        }) else {
            return Ok(());
        };
        let Some(abs_pos) = seq.seq_len.checked_sub(1) else {
            return Ok(());
        };
        if dstate.ctx_len == seq.seq_len {
            return Ok(());
        }
        let append = dstate.ctx_ring_state()?.plan_append_at(abs_pos, 1)?;
        let copy_stream = if stream == 0 {
            self.gpu.default_stream()
        } else {
            stream
        };
        let captured = dstate.commit_serial_decode_capture(
            self.gpu.as_ref(),
            src_base,
            seq.seq_len,
            append,
            copy_stream,
        )?;
        if captured && std::env::var("ATLAS_PROPOSE_PROBE").ok().as_deref() == Some("1") {
            tracing::info!(
                "capture_dflash_serial_decode_row: abs_pos={} ctx_len={} slot_bytes={}",
                abs_pos,
                dstate.ctx_len,
                dstate.ctx_slot_bytes,
            );
        }
        Ok(())
    }

    /// Capture one batched K=1 decode token's target hidden row
    /// (`dflash_hidden_save[row]`) into `seq`'s bounded DFlash context ring and
    /// advance `ctx_len` in lockstep with `seq.seq_len`.
    ///
    /// Mirror of [`Self::capture_dflash_serial_decode_row`] for
    /// `decode_qwen4_native_batch.rs`, called AFTER `seq.seq_len += 1` so the
    /// just-decoded token sits at `seq.seq_len - 1`. Idempotent when
    /// `dstate.ctx_len == seq.seq_len`.
    pub(crate) fn capture_dflash_batched_decode_row(
        &self,
        seq: &mut SequenceState,
        row: usize,
        stream: u64,
    ) -> Result<()> {
        if !self.active_proposer().is_some_and(|p| p.is_dflash()) {
            return Ok(());
        }
        if self.dflash_capture_layers.is_empty() {
            return Ok(());
        }
        let Some(src_base) = self.dflash_hidden_save.filter(|p| !p.is_null()) else {
            return Ok(());
        };
        if self.comm.as_ref().is_some_and(|c| c.rank() != 0) {
            return Ok(());
        }
        anyhow::ensure!(
            row < self.dflash_hidden_save_rows,
            "capture_dflash_batched_decode_row: row={row} exceeds dflash_hidden_save_rows={}",
            self.dflash_hidden_save_rows,
        );
        let Some(dstate) = seq.proposer_state.as_mut().and_then(|ps| {
            ps.as_any_mut()
                .downcast_mut::<crate::layers::DflashProposerState>()
        }) else {
            return Ok(());
        };
        let Some(abs_pos) = seq.seq_len.checked_sub(1) else {
            return Ok(());
        };
        if dstate.ctx_len == seq.seq_len {
            return Ok(());
        }
        let expected_slot_bytes = self
            .dflash_capture_layers
            .len()
            .checked_mul(self.dflash_capture_width)
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(|| anyhow::anyhow!("DFlash batched decode capture slot size overflow"))?;
        anyhow::ensure!(
            dstate.ctx_slot_bytes == expected_slot_bytes && expected_slot_bytes > 0,
            "DFlash batched decode capture slot geometry ({}) disagrees with model capture ({expected_slot_bytes})",
            dstate.ctx_slot_bytes,
        );
        let src_off = row
            .checked_mul(dstate.ctx_slot_bytes)
            .ok_or_else(|| anyhow::anyhow!("DFlash batched decode capture offset overflow"))?;
        let append = dstate.ctx_ring_state()?.plan_append_at(abs_pos, 1)?;
        let copy_stream = if stream == 0 {
            self.gpu.default_stream()
        } else {
            stream
        };
        let captured = dstate.commit_serial_decode_capture(
            self.gpu.as_ref(),
            src_base.offset(src_off),
            seq.seq_len,
            append,
            copy_stream,
        )?;
        if captured && std::env::var("ATLAS_PROPOSE_PROBE").ok().as_deref() == Some("1") {
            tracing::info!(
                "capture_dflash_batched_decode_row: row={} abs_pos={} ctx_len={} slot_bytes={}",
                row,
                abs_pos,
                dstate.ctx_len,
                dstate.ctx_slot_bytes,
            );
        }
        Ok(())
    }

    /// Retire `num_accepted + 1` accepted batched-verify DFlash hidden rows
    /// (`dflash_hidden_save[row_base..=row_base + num_accepted]`) into `seq`'s
    /// private `DflashProposerState` ring, advancing `dstate.ctx_len` to match
    /// the post-commit `seq.seq_len`.
    ///
    /// Called from `retire_dflash_frame_captures_impl` after `accept_and_emit_row`
    /// has already rolled `seq.seq_len` back to `pre_verify_len + num_accepted + 1`
    /// and before the next subframe or proposer call overwrites `dflash_hidden_save`.
    pub(crate) fn retire_dflash_frame_captures_from_save(
        &self,
        row_base: usize,
        num_accepted: usize,
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<()> {
        if self.comm.as_ref().is_some_and(|c| c.rank() != 0) {
            return Ok(());
        }
        let save_buf = self
            .dflash_hidden_save
            .filter(|p| !p.is_null())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "retire_dflash_frame_captures_from_save: dflash_hidden_save is not allocated"
                )
            })?;
        let n_copy = num_accepted
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("DFlash frame capture count overflow"))?;
        let end_row = row_base
            .checked_add(n_copy)
            .ok_or_else(|| anyhow::anyhow!("DFlash frame capture row range overflow"))?;
        anyhow::ensure!(
            end_row <= self.dflash_hidden_save_rows,
            "DFlash frame capture rows {row_base}..{end_row} exceed dflash_hidden_save_rows={}",
            self.dflash_hidden_save_rows,
        );
        let dstate = seq
            .proposer_state
            .as_mut()
            .and_then(|ps| {
                ps.as_any_mut()
                    .downcast_mut::<crate::layers::DflashProposerState>()
            })
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "retire_dflash_frame_captures_from_save: sequence has no DflashProposerState"
                )
            })?;
        let expected_slot_bytes = self
            .dflash_capture_layers
            .len()
            .checked_mul(self.dflash_capture_width)
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(|| anyhow::anyhow!("DFlash frame capture slot size overflow"))?;
        let slot_bytes = dstate.ctx_slot_bytes;
        anyhow::ensure!(
            slot_bytes == expected_slot_bytes && expected_slot_bytes > 0,
            "DFlash frame capture slot geometry ({slot_bytes}) disagrees with model capture ({expected_slot_bytes})",
        );
        let ring = dstate.ctx_ring_state()?;
        let expected_seq_len = ring
            .absolute_len
            .checked_add(n_copy)
            .ok_or_else(|| anyhow::anyhow!("DFlash frame capture cursor overflow"))?;
        anyhow::ensure!(
            expected_seq_len == seq.seq_len,
            "DFlash frame capture cursor mismatch: ctx_len={} + n_copy={n_copy} != seq_len={}",
            ring.absolute_len,
            seq.seq_len,
        );
        let append = ring.plan_append_at(ring.absolute_len, n_copy)?;
        let mut copy_ops: [(usize, usize, usize); 2] = [(0, 0, 0); 2];
        let mut copy_count = 0usize;
        for span in append.write.spans() {
            anyhow::ensure!(
                append.write.physical_slot_for(span.linear_slot)? == span.physical_slot,
                "DFlash frame capture ring plan and physical slot disagree"
            );
            let src_row = row_base
                .checked_add(span.linear_slot)
                .ok_or_else(|| anyhow::anyhow!("DFlash frame capture source row overflow"))?;
            let src_off = src_row
                .checked_mul(slot_bytes)
                .ok_or_else(|| anyhow::anyhow!("DFlash frame capture source offset overflow"))?;
            let dst_off = span
                .physical_slot
                .checked_mul(slot_bytes)
                .ok_or_else(|| anyhow::anyhow!("DFlash frame capture dst offset overflow"))?;
            let copy_bytes = span
                .slot_count
                .checked_mul(slot_bytes)
                .ok_or_else(|| anyhow::anyhow!("DFlash frame capture copy bytes overflow"))?;
            anyhow::ensure!(
                !dstate.ctx_hidden_acc.is_null()
                    && dst_off
                        .checked_add(copy_bytes)
                        .is_some_and(|end| end <= dstate.ctx_allocation_bytes),
                "DFlash frame capture dst [{dst_off}..+{copy_bytes}) exceeds ctx_allocation_bytes={}",
                dstate.ctx_allocation_bytes,
            );
            copy_ops[copy_count] = (src_off, dst_off, copy_bytes);
            copy_count += 1;
        }
        let copy_stream = if stream == 0 {
            self.gpu.default_stream()
        } else {
            stream
        };
        for &(src_off, dst_off, copy_bytes) in &copy_ops[..copy_count] {
            self.gpu.copy_d2d_async(
                save_buf.offset(src_off),
                dstate.ctx_hidden_acc.offset(dst_off),
                copy_bytes,
                copy_stream,
            )?;
        }
        dstate.apply_ctx_ring_state(append.next)?;
        dstate.last_capture_idx = 0;
        dstate.last_num_accepted = 0;
        dstate.last_accepted_compact.clear();
        Ok(())
    }

    /// Thinking-phase ctx capture entry point. Delegates directly to
    /// [`Self::capture_dflash_serial_decode_row`] (idempotent when
    /// `decode_batch` already captured this token).
    pub(super) fn dflash_capture_thinking_dispatch(
        &self,
        seq: &mut crate::traits::SequenceState,
        stream: u64,
    ) -> Result<()> {
        self.capture_dflash_serial_decode_row(seq, stream)
    }

    /// Bring the per-sequence DFlash context ring to the target's current
    /// absolute fed-token cursor before an ordinary bootstrap decode.
    ///
    /// This is deliberately separate from `dflash_capture_thinking`: the rows
    /// already exist in the multi-row verify capture buffer and must be copied
    /// *before* decode reuses row zero.  The state helper validates that the
    /// cursor debt is exactly root + accepted drafts and refuses every other
    /// gap.
    pub(super) fn reconcile_dflash_context_before_bootstrap_dispatch(
        &self,
        seq: &mut crate::traits::SequenceState,
        stream: u64,
    ) -> Result<usize> {
        let Some(proposer) = self.active_proposer() else {
            return Ok(0);
        };
        if !proposer.is_dflash() {
            return Ok(0);
        }
        if let Some(ref c) = self.comm
            && c.rank() != 0
        {
            return Ok(0);
        }
        let physical_verify_k = proposer.physical_verify_k().ok_or_else(|| {
            anyhow::anyhow!("active DFlash proposer omitted its physical verify capacity")
        })?;
        let captured = self.dflash_hidden_save.ok_or_else(|| {
            anyhow::anyhow!("active DFlash proposer has no target-hidden capture buffer")
        })?;
        let position = seq.seq_len;
        let state = seq
            .proposer_state
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("active DFlash sequence has no proposer state"))?;
        let dstate = state
            .as_any_mut()
            .downcast_mut::<crate::layers::DflashProposerState>()
            .ok_or_else(|| {
                anyhow::anyhow!("active DFlash sequence has the wrong proposer state")
            })?;
        let copy_stream = if stream == 0 {
            self.gpu.default_stream()
        } else {
            stream
        };
        let appended = dstate.append_pending_ctx_captures(
            self.gpu.as_ref(),
            captured,
            position,
            physical_verify_k,
            copy_stream,
        )?;
        // Bootstrap owns the next capture cycle.  Once the absolute cursor
        // proves the old rows are resident (including the idempotent already-
        // appended case), they must never be selected again if an unrelated
        // bootstrap-side operation fails before the normal trim(0).
        dstate.last_capture_idx = 0;
        dstate.last_num_accepted = 0;
        dstate.last_accepted_compact.clear();
        Ok(appended)
    }

    /// No-op. The original "bonus-decode" approach was based on the
    /// misunderstanding that the drafter needs the bonus's hidden in
    /// ctx — actually the bonus is at position seq_len (not yet in
    /// KV), and ctx should contain hiddens for positions [0..seq_len-1]
    /// which is exactly what dflash_hidden_save[0..N+1] provides
    /// after a verify (last_token at position P, drafts at P+1..P+N).
    /// The bonus appears as the FIRST noise embedding (Q-side input)
    /// at position P+N+1.
    pub(super) fn save_hidden_for_dflash_dispatch(
        &self,
        _token: u32,
        _seq: &mut crate::traits::SequenceState,
        _stream: u64,
    ) -> Result<()> {
        Ok(())
    }

    /// DFlash 5-layer hidden capture. Called inside each per-layer loop after
    /// `layer.decode(...)` returns. No-op when DFlash is disabled (the buffer
    /// is `None`) or when `layer_idx` is not in `dflash_capture_layers`.
    ///
    /// Writes hidden state for input row `token_idx` into the
    /// `dflash_hidden_save` buffer at `[token_idx, slot, ..]` where `slot`
    /// is this layer's index in `dflash_capture_layers`. The buffer is laid
    /// out as `[k_max, n_capture, hidden]` BF16 so the downstream propose
    /// can read whole-token rows of stride `n_capture * hidden * bf16`.
    ///
    /// Pre-2026-05-18 bug: the per-token offset was missing — every token
    /// position wrote to slot 0 of the buffer, leaving only the last
    /// captured token's hiddens visible to the drafter and corrupting the
    /// ctx_hidden_acc on K>=2 verify paths. K=2/3/4 partly hid the bug
    /// (small γ, low ctx mismatch), but K=γ=16 collapsed accept rate to
    /// ~2% because every ctx slot got seeded with the last-position hidden.
    ///
    /// Under EP/TP world > 1: only rank 0 owns the drafter (replicated, not
    /// sharded — same pattern as MTP under EP),
    /// so non-rank-0 ranks skip the capture. The captured hiddens are
    /// post-TP-allreduce so semantically correct on rank 0.
    pub(super) fn try_dflash_capture(
        &self,
        layer_idx: usize,
        token_idx: usize,
        stream: u64,
    ) -> Result<()> {
        // DEBUG: bypass capture to test if dflash_hidden_save writes corrupt
        // target output. Set ATLAS_DFLASH_NO_CAPTURE=1 to skip the d2d.
        if std::env::var("ATLAS_DFLASH_NO_CAPTURE").ok().as_deref() == Some("1") {
            return Ok(());
        }
        let dst = match self.dflash_hidden_save {
            Some(p) => p,
            None => return Ok(()),
        };
        // Rank-0 gate (mirrors save_hidden_for_mtp's effective behavior).
        if let Some(ref c) = self.comm
            && c.rank() != 0
        {
            return Ok(());
        }
        let slot = match self
            .dflash_capture_layers
            .iter()
            .position(|&l| l == layer_idx)
        {
            Some(s) => s,
            None => return Ok(()),
        };
        let h = self.dflash_capture_width;
        let bf16 = 2usize;
        // Release builds too: an FP32 residual row copied as BF16 is silent
        // garbage for the drafter. Pairing refuses it up front
        // (`validate_dflash_capture_residual`); this catches a residual mode
        // flipped after construction.
        anyhow::ensure!(
            !self.config.use_fp32_residual(),
            "DFlash hidden capture assumes a BF16 residual; FP32-residual models need a downcast capture mode"
        );
        let n_capture = self.dflash_capture_layers.len();
        let raw = self
            .buffers
            .hidden_states()
            .offset(token_idx * self.config.residual_width() * bf16);
        let src = match self.dflash_capture_mode {
            DflashCaptureMode::ResidualSlice => raw.offset(self.dflash_capture_offset * bf16),
            DflashCaptureMode::Qwen4HyperProjected => {
                let mixer = self.qwen4_final_mixer.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("projected DFlash capture missing Qwen4 hyper mixer")
                })?;
                let residual = self
                    .buffers
                    .residual()
                    .offset(token_idx * self.config.residual_width() * bf16);
                // DFlash decode/verify capture forces `prepare_decode_nvfp4`
                // explicitly so capture arithmetic stays consistent across
                // single-token and multi-token rows regardless of
                // `ATLAS_QWEN4_HC_BF16` (`has_dflash` is also refused by
                // `qwen4_hc_bf16::admit_launch`).
                mixer
                    .prepare_decode_nvfp4(
                        raw,
                        residual,
                        &self.buffers,
                        self.gpu.as_ref(),
                        self.config.rms_norm_eps as f32,
                        stream,
                    )?
                    .0
            }
            DflashCaptureMode::Qwen4LayerAttnHc => {
                let next_layer = layer_idx + 1;
                let mixer = self
                    .layers
                    .get(next_layer)
                    .and_then(|layer| layer.qwen4_attn_hyper())
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "Qwen4LayerAttnHc capture missing layers[{next_layer}].qwen4_attn_hyper()"
                        )
                    })?;
                mixer.mix_only_decode_nvfp4(
                    raw,
                    self.buffers.norm_output(),
                    &self.buffers,
                    self.gpu.as_ref(),
                    self.config.rms_norm_eps as f32,
                    stream,
                )?
            }
            DflashCaptureMode::Qwen4HcMean => {
                let out = self.buffers.norm_output();
                self.launch_qwen4_hc_mean_rows(raw, out, 1, h, stream)?;
                out
            }
        };
        // [token_idx, slot, hidden] BF16 layout. Per-token stride =
        // n_capture * h * bf16; per-slot stride = h * bf16.
        let dst_off = token_idx * n_capture * h * bf16 + slot * h * bf16;
        let dst_slot = dst.offset(dst_off);
        self.gpu.copy_d2d_async(src, dst_slot, h * bf16, stream)?;
        if std::env::var("ATLAS_PROPOSE_PROBE").ok().as_deref() == Some("1") {
            tracing::info!(
                "try_dflash_capture: layer_idx={} token_idx={} slot={} dst_off={} h={}",
                layer_idx,
                token_idx,
                slot,
                dst_off,
                h,
            );
        }
        Ok(())
    }

    fn launch_qwen4_hc_mean_rows(
        &self,
        hyper_4h: DevicePtr,
        out_h: DevicePtr,
        rows: usize,
        h: usize,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            self.qwen4_hc_mean_rows_kernel.0 != 0,
            "Qwen4HcMean capture requires qwen4_hyper::qwen4_hc_mean_rows kernel"
        );
        let rows_i32 =
            i32::try_from(rows).map_err(|_| anyhow::anyhow!("qwen4_hc_mean_rows rows overflow"))?;
        let h_i32 =
            i32::try_from(h).map_err(|_| anyhow::anyhow!("qwen4_hc_mean_rows hidden overflow"))?;
        let grid_x = spark_runtime::kernel_args::div_ceil(h.div_ceil(2) as u32, 256);
        spark_runtime::kernel_args::KernelLaunch::new(
            self.gpu.as_ref(),
            self.qwen4_hc_mean_rows_kernel,
        )
        .grid([grid_x, rows as u32, 1])
        .block([256, 1, 1])
        .arg_ptr(hyper_4h)
        .arg_ptr(out_h)
        .arg_i32(rows_i32)
        .arg_i32(h_i32)
        .launch(stream)
    }

    /// Batched DFlash capture for all `k` rows of a verify step in one
    /// transformation pass + strided scatter into `dflash_hidden_save`.
    /// Replaces `for t in 0..k { self.try_dflash_capture(layer_idx, t, stream)?; }`
    /// with a single batched HC contraction (or HC mean / slice) and single
    /// strided copy, CUDA-graph-safe and row-exact.
    pub(super) fn try_dflash_capture_rows(
        &self,
        layer_idx: usize,
        k: usize,
        stream: u64,
    ) -> Result<()> {
        if k == 0 {
            return Ok(());
        }
        let Some(dst) = self.dflash_hidden_save else {
            return Ok(());
        };
        if self.comm.as_ref().is_some_and(|c| c.rank() != 0) {
            return Ok(());
        }
        let Some(slot) = self
            .dflash_capture_layers
            .iter()
            .position(|&l| l == layer_idx)
        else {
            return Ok(());
        };
        if std::env::var("ATLAS_DFLASH_NO_CAPTURE").ok().as_deref() == Some("1") {
            return Ok(());
        }
        anyhow::ensure!(
            k <= self.dflash_hidden_save_rows,
            "try_dflash_capture_rows: k={k} exceeds dflash_hidden_save_rows={}",
            self.dflash_hidden_save_rows,
        );
        if k == 1 {
            return self.try_dflash_capture(layer_idx, 0, stream);
        }
        anyhow::ensure!(
            !self.config.use_fp32_residual(),
            "DFlash hidden capture assumes a BF16 residual; FP32-residual models need a downcast capture mode"
        );
        let h = self.dflash_capture_width;
        let bf16 = 2usize;
        let n_capture = self.dflash_capture_layers.len();
        let (src_base, src_stride) = match self.dflash_capture_mode {
            DflashCaptureMode::ResidualSlice => (
                self.buffers
                    .hidden_states()
                    .offset(self.dflash_capture_offset * bf16),
                self.config.residual_width(),
            ),
            DflashCaptureMode::Qwen4HyperProjected => {
                let mixer = self.qwen4_final_mixer.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("projected DFlash capture missing Qwen4 hyper mixer")
                })?;
                let out = mixer.mix_only_rows_nvfp4(
                    self.buffers.hidden_states(),
                    k,
                    self.buffers.norm_output(),
                    &self.buffers,
                    self.gpu.as_ref(),
                    self.config.rms_norm_eps as f32,
                    stream,
                )?;
                (out, h)
            }
            DflashCaptureMode::Qwen4LayerAttnHc => {
                let next_layer = layer_idx + 1;
                let mixer = self
                    .layers
                    .get(next_layer)
                    .and_then(|layer| layer.qwen4_attn_hyper())
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "Qwen4LayerAttnHc capture missing layers[{next_layer}].qwen4_attn_hyper()"
                        )
                    })?;
                let out = mixer.mix_only_rows_nvfp4(
                    self.buffers.hidden_states(),
                    k,
                    self.buffers.norm_output(),
                    &self.buffers,
                    self.gpu.as_ref(),
                    self.config.rms_norm_eps as f32,
                    stream,
                )?;
                (out, h)
            }
            DflashCaptureMode::Qwen4HcMean => {
                let out = self.buffers.norm_output();
                self.launch_qwen4_hc_mean_rows(self.buffers.hidden_states(), out, k, h, stream)?;
                (out, h)
            }
        };
        let row_bytes = h * bf16;
        let dst_base = dst.offset(slot * row_bytes);
        let src_stride_bytes = src_stride * bf16;
        let dst_stride_bytes = n_capture * row_bytes;
        if crate::layers::dflash_capture_strided_enabled() && self.strided_copy_rows_kernel.0 != 0 {
            crate::layers::ops::strided_copy_rows(
                self.gpu.as_ref(),
                self.strided_copy_rows_kernel,
                src_base,
                dst_base,
                k as u32,
                row_bytes as u32,
                src_stride_bytes as u64,
                dst_stride_bytes as u64,
                stream,
            )?;
        } else {
            for t in 0..k {
                self.gpu.copy_d2d_async(
                    src_base.offset(t * src_stride_bytes),
                    dst_base.offset(t * dst_stride_bytes),
                    row_bytes,
                    stream,
                )?;
            }
        }
        Ok(())
    }

    /// Flush captured hidden states from `dflash_hidden_save` (GPU) to the
    /// path in `$ATLAS_DUMP_HIDDEN` (host file). No-op when:
    ///   - env var is unset (production hot path — zero cost)
    ///   - `dflash_hidden_save` was never allocated
    ///   - `dflash_capture_layers` is empty
    ///   - this is a non-rank-0 worker under EP/TP (drafter is replicated)
    ///
    /// Must be called from a verify path that runs in EAGER mode (no CUDA
    /// graph capture) — `copy_d2h` is a synchronous device→host transfer
    /// and would fail with `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED (900)`
    /// inside a captured region. `verify_b.rs` already forces eager mode
    /// when `ATLAS_DUMP_HIDDEN` is set (see the env-var check at the
    /// `suppress_graphs` re-enable site), so unconditional calls from the
    /// post-layer-loop site are safe.
    ///
    /// Record format (little-endian, 16-byte header + bf16 payload):
    /// ```text
    /// u32 magic         = 0xA71A5DEE   // distinct from TOKEN_DUMP_MAGIC (0xA71B5DEE)
    /// u32 layer_idx                    // ABSOLUTE layer index (one of dflash_capture_layers)
    /// u32 token_idx                    // position within the K-step verify
    /// u32 hidden_dim                   // model hidden_size (5120 on AEON-27B, 2048 on 35B-A3B-abl)
    /// bf16 hidden[hidden_dim]
    /// ```
    /// Bytes per record = 16 + hidden_dim * 2. The dump is APPEND-ONLY
    /// across the run; downstream training code splits records on the
    /// magic header.
    ///
    /// Pairs with the emitted-token records written by `meta.rs::argmax*`
    /// (magic 0xA71B5DEE). Together they let an offline trainer pair
    /// each emitted token with the 5 hidden states that preceded it.
    ///
    /// `k` is the verify width (number of `token_idx` slots populated by
    /// the just-completed layer loop — K=2 for K=2 verify, K=3 for K=3
    /// MTP verify, K=γ+1 for DFlash γ verify).
    pub(super) fn flush_hidden_dump(&self, k: usize) -> Result<()> {
        // Hot-path early exit: env var check is the cheapest possible.
        let path = match std::env::var("ATLAS_DUMP_HIDDEN") {
            Ok(p) => p,
            Err(_) => return Ok(()),
        };
        if self.dflash_capture_layers.is_empty() {
            return Ok(());
        }
        let dst = match self.dflash_hidden_save {
            Some(p) => p,
            None => return Ok(()),
        };
        // Rank-0 gate (mirrors try_dflash_capture's behavior). Replicated
        // drafter under EP/TP — only rank 0 owns it.
        if let Some(ref c) = self.comm
            && c.rank() != 0
        {
            return Ok(());
        }

        let h = self.dflash_capture_width;
        let bf16 = 2usize;
        let n_capture = self.dflash_capture_layers.len();
        let bytes_per_slot = h * bf16;
        let bytes_per_token = n_capture * bytes_per_slot;
        let total_bytes = k * bytes_per_token;

        // copy_d2h is synchronous on the default stream — graph-capture
        // unsafe but the caller guarantees eager mode (see method doc).
        let mut host_buf = vec![0u8; total_bytes];
        self.gpu.copy_d2h(dst, &mut host_buf)?;

        // Open file in append mode. If open or any write fails we surface
        // the error rather than silently swallowing (caller is in eager
        // mode by design — we want loud failures so missing dumps are
        // visible in logs).
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| anyhow::anyhow!("ATLAS_DUMP_HIDDEN open {path}: {e}"))?;

        const HIDDEN_DUMP_MAGIC: u32 = 0xA71A_5DEE;
        for t in 0..k {
            for (slot, &layer_idx) in self.dflash_capture_layers.iter().enumerate() {
                let off = t * bytes_per_token + slot * bytes_per_slot;
                let hidden_bytes = &host_buf[off..off + bytes_per_slot];
                f.write_all(&HIDDEN_DUMP_MAGIC.to_le_bytes())?;
                f.write_all(&(layer_idx as u32).to_le_bytes())?;
                f.write_all(&(t as u32).to_le_bytes())?;
                f.write_all(&(h as u32).to_le_bytes())?;
                f.write_all(hidden_bytes)?;
            }
        }
        // No fsync — append-only writes to a single file are fine across
        // open/close cycles, and per-step fsync would tank throughput.
        Ok(())
    }
}

/// Whether intermediate prefill chunks outside the trailing `capacity` window
/// skip MTP last-K host-ring capture (default ON; set
/// `ATLAS_QWEN4_MTP_LASTK_SKIP_INTERMEDIATE=0` to disable).
fn mtp_lastk_skip_intermediate_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("ATLAS_QWEN4_MTP_LASTK_SKIP_INTERMEDIATE")
            .ok()
            .as_deref()
            != Some("0")
    })
}

/// The MTP last-K prompt capture runs only with a capacity
/// (`ATLAS_MTP_LASTK_PREFILL`, default 64 on Qwen4 even without speculation)
/// AND an installed proposer (`--speculative` MTP or DFlash, fixed at load).
pub(crate) fn mtp_lastk_capture_runs(capacity: usize, proposer_installed: bool) -> bool {
    let runs = capacity > 0 && proposer_installed;
    if runs {
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            if std::env::var("ATLAS_QWEN4_MTP_LASTK_SKIP_INTERMEDIATE")
                .ok()
                .as_deref()
                == Some("1")
            {
                tracing::info!(
                    "QWEN4_MTP_LASTK_SKIP_INTERMEDIATE_ENGAGED selector=ATLAS_QWEN4_MTP_LASTK_SKIP_INTERMEDIATE capacity={capacity}"
                );
            }
        });
    }
    runs
}

/// `ATLAS_MTP_LASTK_DUMP=<dir>` (default off): write the last-K ring uploaded
/// to the proposer, one file per prefill in request order, for byte-identity
/// gates (the last-layer crop's K-row tail must leave it unchanged).
fn maybe_dump_lastk_ring(rows: &[u8], end_abs: usize, filled: usize) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static DIR: std::sync::OnceLock<Option<std::path::PathBuf>> = std::sync::OnceLock::new();
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let Some(dir) = DIR.get_or_init(|| {
        std::env::var("ATLAS_MTP_LASTK_DUMP")
            .ok()
            .filter(|d| !d.is_empty())
            .map(std::path::PathBuf::from)
    }) else {
        return;
    };
    let path = dir.join(lastk_dump_name(
        NEXT.fetch_add(1, Ordering::Relaxed),
        end_abs,
        filled,
    ));
    match std::fs::create_dir_all(dir).and_then(|()| std::fs::write(&path, rows)) {
        Ok(()) => tracing::info!(
            "MTP_LASTK_DUMP wrote {} ({} bytes)",
            path.display(),
            rows.len()
        ),
        Err(e) => tracing::warn!("ATLAS_MTP_LASTK_DUMP: cannot write {}: {e}", path.display()),
    }
}

/// Contiguous row range of `hidden_states` to reduce with `Qwen4HcMean` into
/// `norm_output` starting at `dst_row`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct HcMeanSlice {
    src_linear_row: usize,
    dst_row: usize,
    row_count: usize,
}

/// Plan `Qwen4HcMean` row reductions covering only the rows written by
/// `spans`, returning `(src_linear_base, slices)`.
///
/// Adjacent spans (including wrap-around 2-span ring writes where span 1
/// immediately follows span 0 in chunk linear order) coalesce into one launch
/// writing compactly from `dst_row = 0`. Every copied ring row `r` is then
/// read from `out[r - src_linear_base]`.
fn plan_hc_mean_slices(
    spans: &[crate::layers::dflash_head::ring_window::RingSpan],
) -> Result<(usize, Vec<HcMeanSlice>)> {
    let Some(min_linear) = spans
        .iter()
        .filter(|s| s.slot_count > 0)
        .map(|s| s.linear_slot)
        .min()
    else {
        return Ok((0, Vec::new()));
    };
    let mut slices: Vec<HcMeanSlice> = Vec::with_capacity(spans.len());
    for span in spans.iter().copied().filter(|s| s.slot_count > 0) {
        let dst_row = span
            .linear_slot
            .checked_sub(min_linear)
            .ok_or_else(|| anyhow::anyhow!("Qwen4HcMean span linear_slot underflow"))?;
        if let Some(prev) = slices.last_mut()
            && prev.src_linear_row.checked_add(prev.row_count) == Some(span.linear_slot)
            && prev.dst_row.checked_add(prev.row_count) == Some(dst_row)
        {
            prev.row_count = prev
                .row_count
                .checked_add(span.slot_count)
                .ok_or_else(|| anyhow::anyhow!("Qwen4HcMean coalesced row count overflow"))?;
        } else {
            slices.push(HcMeanSlice {
                src_linear_row: span.linear_slot,
                dst_row,
                row_count: span.slot_count,
            });
        }
    }
    Ok((min_linear, slices))
}

fn lastk_dump_name(index: u64, end_abs: usize, filled: usize) -> String {
    format!("mtp_lastk_{index:04}_end{end_abs}_rows{filled}.bin")
}

#[cfg(test)]
mod lastk_dump_tests {
    use super::{HcMeanSlice, plan_hc_mean_slices};
    use crate::layers::dflash_head::ring_window::{RingSpan, RingState};

    #[test]
    fn dump_names_sort_in_request_order() {
        assert_eq!(
            super::lastk_dump_name(3, 8191, 64),
            "mtp_lastk_0003_end8191_rows64.bin"
        );
        assert!(super::lastk_dump_name(9, 1, 1) < super::lastk_dump_name(10, 0, 0));
    }

    #[test]
    fn hc_mean_slices_cover_single_span_trailing_ring_rows_only() {
        // 16K prefill into a 2048-slot ring writes one contiguous trailing span [14336..16384).
        let ring = RingState::new(32_768, 2048).expect("ring");
        let append = ring.plan_append_at(0, 16_384).expect("plan");
        let spans = append.write.spans();
        assert_eq!(spans.len(), 1);
        let (base, slices) = plan_hc_mean_slices(spans).expect("slices");
        assert_eq!(base, 14_336);
        assert_eq!(
            slices,
            vec![HcMeanSlice {
                src_linear_row: 14_336,
                dst_row: 0,
                row_count: 2048,
            }]
        );
        assert_eq!(spans[0].linear_slot - base, 0);
    }

    #[test]
    fn hc_mean_slices_coalesce_wrap_around_two_span_ring_writes() {
        // DFlash2 W=8 16K prefill into a 2047-slot ring wraps around physical slot 0:
        // span 0: linear [14337..16376) -> phys [8..2047), span 1: linear [16376..16384) -> phys [0..8).
        let ring = RingState::new(32_768, 2047).expect("ring");
        let append = ring.plan_append_at(0, 16_384).expect("plan");
        let spans = append.write.spans();
        assert_eq!(spans.len(), 2);
        let (base, slices) = plan_hc_mean_slices(spans).expect("slices");
        assert_eq!(base, 14_337);
        assert_eq!(
            slices,
            vec![HcMeanSlice {
                src_linear_row: 14_337,
                dst_row: 0,
                row_count: 2047,
            }]
        );
        // Verify every copied row maps to the exact reduced output row for both spans.
        let mut simulated_out = vec![usize::MAX; 2047];
        for s in &slices {
            for i in 0..s.row_count {
                simulated_out[s.dst_row + i] = s.src_linear_row + i;
            }
        }
        for span in spans {
            let src_off = span.linear_slot - base;
            for local in 0..span.slot_count {
                assert_eq!(simulated_out[src_off + local], span.linear_slot + local);
            }
        }

        // Small continuation chunk wrapping around capacity boundary [2000..2100).
        let cont = RingState::from_lengths(32_768, 2047, 2000, 2000).expect("cont ring");
        let wrap = cont.plan_append_at(2000, 100).expect("wrap plan");
        assert_eq!(wrap.write.spans().len(), 2);
        let (wrap_base, wrap_slices) = plan_hc_mean_slices(wrap.write.spans()).expect("wrap");
        assert_eq!(wrap_base, 0);
        assert_eq!(
            wrap_slices,
            vec![HcMeanSlice {
                src_linear_row: 0,
                dst_row: 0,
                row_count: 100,
            }]
        );
    }

    #[test]
    fn hc_mean_slices_handle_empty_and_disjoint_spans() {
        assert_eq!(plan_hc_mean_slices(&[]).expect("empty"), (0, Vec::new()));

        let disjoint = [
            RingSpan {
                physical_slot: 10,
                linear_slot: 100,
                slot_count: 32,
            },
            RingSpan {
                physical_slot: 0,
                linear_slot: 200,
                slot_count: 16,
            },
        ];
        let (base, slices) = plan_hc_mean_slices(&disjoint).expect("disjoint");
        assert_eq!(base, 100);
        assert_eq!(
            slices,
            vec![
                HcMeanSlice {
                    src_linear_row: 100,
                    dst_row: 0,
                    row_count: 32,
                },
                HcMeanSlice {
                    src_linear_row: 200,
                    dst_row: 100,
                    row_count: 16,
                },
            ]
        );
        let mut simulated_out = vec![usize::MAX; 116];
        for s in &slices {
            for i in 0..s.row_count {
                simulated_out[s.dst_row + i] = s.src_linear_row + i;
            }
        }
        for span in disjoint {
            let src_off = span.linear_slot - base;
            for local in 0..span.slot_count {
                assert_eq!(simulated_out[src_off + local], span.linear_slot + local);
            }
        }
    }
}
