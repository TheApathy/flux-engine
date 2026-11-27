// SPDX-License-Identifier: AGPL-3.0-only

//! `Model` trait — the interface the scheduler talks to.
//!
//! ## Dispatch contract
//!
//! Per request, the scheduler invokes:
//!
//! 1. [`Model::prefill`] (or [`Model::prefill_chunk`] for chunked prefill)
//!    once per sequence. Returns logits at the last prompt position;
//!    populates the sequence's KV cache and SSM state.
//! 2. [`Model::decode`] once per emitted token. Returns next-token logits;
//!    extends KV/SSM state by one position. May be replaced by
//!    [`Model::decode_batch`] when multiple sequences are co-scheduled.
//! 3. Optional speculative-decode verify path: [`Model::decode_verify_graphed`]
//!    (K=2), [`Model::decode_verify_graphed_k3`] (K=3),
//!    [`Model::decode_verify_graphed_k4`] (K=4), or
//!    [`Model::decode_verify_graphed_kgamma`] (DFlash γ-token).
//!    These take [last_token, draft0, ..] and return per-position logits;
//!    the scheduler picks accept/reject and rolls back state on reject.
//! 4. [`Model::mixed_forward`] fuses one decode step + one prefill chunk
//!    through a single weight load; used by the scheduler to amortize
//!    weight-streaming cost when both phases are pending.
//!
//! Implementors live under `crates/spark-model/src/model/trait_impl/`,
//! split per phase (prefill_a/b/c/d, decode_a/b, verify_a/b/c/d) per
//! ADR-0006's multi-file module idiom.
//!
//! ## Concurrency
//!
//! `Model: Send + Sync` — a single instance handles all sequences
//! concurrently. Per-sequence state lives in [`SequenceState`].

use anyhow::{Result, bail};
use spark_runtime::gpu::DevicePtr;

/// EP worker opcode for variable-width speculative verification. The frame is
/// `opcode, k, k bulk-broadcast tokens, accepted_drafts`; the result may be
/// [`EP_VERIFY_KGAMMA_ABORT`] when rank 0 cannot commit the verify.
pub const EP_CMD_VERIFY_KGAMMA: u32 = 0xFFFFFFF5;

/// Multi-rank worker opcode for cross-sequence Qwen4 batched verification (`TP=2` or `PP>=2`).
pub const EP_CMD_VERIFY_QWEN4_BATCHED: u32 = 0xFFFFFFF6;

/// Multi-rank worker opcode for committing one slot's accepted prefix after batched verify.
pub const EP_CMD_COMMIT_VERIFY_SLOT: u32 = 0xFFFFFFF7;

/// Multi-rank worker opcode for aborting one slot's in-flight batched verify lane.
pub const EP_CMD_ABORT_VERIFY_SLOT: u32 = 0xFFFFFFF8;

/// Multi-rank worker opcode for native Qwen4 batched decode (`C=2..=256`, `PP>=2`).
pub const EP_CMD_DECODE_QWEN4_BATCH: u32 = 0xFFFFFFF9;

/// Result sentinel releasing an EP worker from a variable-width verify without
/// committing any of its over-extended target state.
pub const EP_VERIFY_KGAMMA_ABORT: u32 = u32::MAX;

use super::{MixedBatchResult, MixedForwardResult, PrefillSlice, SequenceState};
#[allow(unused_imports)]
pub use crate::model::trait_impl::device_topk::{
    HOST_POLICY_FALLBACK_TOKEN, PackedRejectionWord, REJECTION_BONUS_Q_ROW_IDX, REJECTION_MAX_ROWS,
    REJECTION_NO_MASK_TOKEN, REJECTION_Q_ROW_CAP, REJECTION_Q_STRIDE, RejectionVerifyRowSpec,
};
pub use spark_runtime::buffers::padded_decode_rows as padded_batch_n;

/// Sampled internal speculation (see [`Model::spec_verify`]).
#[derive(Clone, Debug, PartialEq)]
pub struct SpecSampling {
    /// The request temperature (> 0) the draft distributions use: q = softmax(logits / T).
    pub temperature: f32,
    /// One uniform in [0,1) per draft, from the request RNG.
    pub draft_uniforms: Vec<f32>,
}

/// High-level model trait for forward pass execution.
///
/// Implementations: `TransformerModel`, `Glm53Model`, `Glm53Exl3Model`, and
/// `Dsv41Model`.
///
/// # Safety
///
/// `Send + Sync` is required by `Box<dyn Model>` usage patterns.
/// `Sync` safety: the model is exclusively accessed from the scheduler
/// thread. The `unsafe impl Sync` on `TransformerModel` documents this
/// single-thread invariant — do NOT share `&dyn Model` across threads.
pub trait Model: Send + Sync {
    /// Run prefill: process all prompt tokens through the model.
    ///
    /// Returns logits DevicePtr for the last token position.
    /// Updates KV cache and SSM states for the sequence.
    fn prefill(&self, tokens: &[u32], seq: &mut SequenceState, stream: u64) -> Result<DevicePtr>;

    /// Process `chunk_len` tokens starting at `chunk_start` in the prompt.
    /// `is_last_chunk` runs final norm + LM head; intermediate chunks return
    /// `DevicePtr::NULL`. KV blocks alloc incrementally; SSM state carries
    /// across chunks; attention uses FA on chunk 0, paged decode after.
    fn prefill_chunk(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_start: usize,
        chunk_len: usize,
        is_last_chunk: bool,
        stream: u64,
    ) -> Result<DevicePtr>;

    /// Before the first prefill chunk: restore a saved state of a prefix of
    /// `prompt` into the fresh `seq` and return its length, so the caller
    /// starts prefilling there. 0 = nothing restored (the default).
    fn resume_conversation(&self, seq: &mut SequenceState, prompt: &[u32]) -> Result<usize> {
        let _ = (seq, prompt);
        Ok(0)
    }

    /// Run one decode step: process a single new token.
    ///
    /// Returns logits DevicePtr for the new token.
    /// Updates KV cache and SSM states.
    fn decode(&self, token: u32, seq: &mut SequenceState, stream: u64) -> Result<DevicePtr>;

    /// Run batched decode: process one token per sequence.
    ///
    /// Returns logits DevicePtr for [batch_size, vocab_size].
    fn decode_batch(
        &self,
        tokens: &[u32],
        seqs: &mut [&mut SequenceState],
        stream: u64,
    ) -> Result<DevicePtr>;

    /// Process N decode tokens + an M-token prefill chunk in one pass through
    /// the same weight loads. Returns decode logits `[N, vocab]` and prefill
    /// logits `[1, vocab]` (when `is_last`). Default: serial decode + prefill.
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
    ) -> Result<MixedForwardResult> {
        // Default: serial execution (no weight sharing)
        let decode_logits = if !decode_tokens.is_empty() {
            self.decode_batch(decode_tokens, decode_seqs, stream)?
        } else {
            spark_runtime::gpu::DevicePtr::NULL
        };
        let prefill_logits = self.prefill_chunk(
            prefill_tokens,
            prefill_seq,
            prefill_chunk_start,
            prefill_chunk_len,
            prefill_is_last,
            stream,
        )?;
        Ok(MixedForwardResult {
            decode_logits,
            prefill_logits,
        })
    }

    /// Process N concurrent prefill chunks in one forward pass (same weight
    /// load amortised across N streams). The default implementation falls
    /// back to a per-stream loop calling `prefill_chunk` — implementors that
    /// support kernel-level batched prefill should override this.
    ///
    /// Returns a `Vec<DevicePtr>` parallel to `streams`: each entry is the
    /// last-token logits pointer for that stream when its chunk is
    /// `is_last_chunk`, or `DevicePtr::NULL` otherwise.
    fn prefill_batch_chunk(
        &self,
        streams: &mut [PrefillSlice<'_>],
        stream: u64,
    ) -> Result<Vec<DevicePtr>> {
        // Default: serialized per-stream prefill_chunk. This preserves
        // current behavior for any model that doesn't override; only the
        // weight-streaming amortisation is lost vs a true batched path.
        let mut out = Vec::with_capacity(streams.len());
        for slice in streams.iter_mut() {
            let logits = self.prefill_chunk(
                slice.prompt_tokens,
                slice.seq,
                slice.chunk_start,
                slice.chunk_len,
                slice.is_last_chunk,
                stream,
            )?;
            out.push(logits);
        }
        Ok(out)
    }

    /// Generalised mixed forward: M decode tokens + N concurrent prefill
    /// chunks fused into one forward pass. Default: delegates to
    /// `decode_batch` + `prefill_batch_chunk` serially. Models that
    /// implement true mixed batching should override.
    fn mixed_forward_batch(
        &self,
        decode_tokens: &[u32],
        decode_seqs: &mut [&mut SequenceState],
        prefill_streams: &mut [PrefillSlice<'_>],
        stream: u64,
    ) -> Result<MixedBatchResult> {
        // Default: serial execution.
        let decode_logits = if !decode_tokens.is_empty() {
            self.decode_batch(decode_tokens, decode_seqs, stream)?
        } else {
            spark_runtime::gpu::DevicePtr::NULL
        };
        let prefill_logits = self.prefill_batch_chunk(prefill_streams, stream)?;
        Ok(MixedBatchResult {
            decode_logits,
            prefill_logits,
        })
    }

    /// Normalize SSM h_state norms to prevent catastrophic state explosion
    /// during long chunked prefill. Called between chunks by the scheduler.
    /// Default: no-op (models without SSM layers don't need normalization).
    fn normalize_ssm_states(&self, _seq: &SequenceState, _stream: u64) -> Result<()> {
        Ok(())
    }

    /// Deferred last-K prefill tail (`ATLAS_MTP_LASTK_DEFER`): run the part of the last prefill chunk
    /// that was deferred past the first token (the MTP last-K prompt replay). The
    /// scheduler calls this after emitting the first token of a sequence that keeps
    /// decoding, before any other model call for it. `device_first_token` is the
    /// first token when [`Model::argmax_on_device`] sampled it (its scratch write is
    /// redone after the replay), `None` when it was sampled on the host.
    /// `suppress_ids` are the ids the first-token sample masked (the stop ids): when
    /// the raw device argmax was one of them, the host fallback picked the token.
    /// Default: nothing is ever deferred.
    fn finish_deferred_prefill_tail(
        &self,
        _seq: &mut SequenceState,
        _device_first_token: Option<u32>,
        _suppress_ids: &[u32],
    ) -> Result<()> {
        Ok(())
    }

    /// Whether the scheduler may enter [`Model::prefill_twophase`]. This is
    /// checked before any model or GPU effect; an error returned after entry
    /// remains request-fatal because recurrent/KV state may already be dirty.
    /// Models that need to retain the historical scheduler behavior can use
    /// the default. Architectures with a separately guarded chunked path must
    /// override this capability instead of relying on error replay.
    fn supports_prefill_twophase(&self) -> bool {
        true
    }

    /// Per-layer chunked prefill: SSM layers use three phases (proj →
    /// single-launch GDN → post) so the recurrence sees the full sequence
    /// in one launch; attention layers use standard chunked prefill.
    /// Returns last-token logits. Default: single-chunk prefill (no SSM).
    fn prefill_twophase(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        _chunk_size: usize,
        stream: u64,
    ) -> Result<DevicePtr> {
        // Default: single-chunk prefill (no two-phase benefit without SSM)
        self.prefill_chunk(tokens, seq, 0, tokens.len(), true, stream)
    }

    /// Vocab size (for sampler allocation).
    fn vocab_size(&self) -> usize;

    /// Dims for the `--high-speed-swap` orchestrator (installed thread-local
    /// after `bind_gpu_to_thread`). `None` for legacy/non-attention models.
    fn high_speed_swap_dims(&self) -> Option<spark_storage::ModelDims> {
        None
    }

    /// Bind the GPU context to the current thread.
    /// Must be called from any thread other than the one that created the model.
    fn bind_gpu_to_thread(&self) -> Result<()>;

    /// Allocate a new SequenceState with SSM states.
    fn alloc_sequence(&self) -> Result<SequenceState>;

    /// Called by the scheduler once a prompt is admitted, before
    /// [`Self::alloc_sequence`], with the length of the first prefill chunk.
    /// Flash-Next (`ATLAS_QWEN4_PLE_EARLY_ISSUE=1`, default off) starts that
    /// chunk's PLE row read here so it overlaps slot allocation; the prefill
    /// of exactly those tokens consumes it. Default: nothing.
    fn begin_prefill_admission(&self, _prompt_tokens: &[u32], _first_chunk_len: usize) {}

    /// The pending-queue head's tokens, handed over while another request
    /// runs (`ATLAS_QWEN4_PLE_QUEUE_PREFETCH`). A model may start chunk 0's
    /// host-side reads now; its later `begin_prefill_admission` for the same
    /// tokens must adopt them. Default: no-op.
    fn begin_prefill_queue_prefetch(&self, _prompt_tokens: &[u32], _first_chunk_len: usize) {}

    /// Best-effort asynchronous pre-warming of offloaded PLE rows for newly
    /// drafted tokens so NVMe reads overlap with MTP drafting / scheduling.
    fn warm_ple_draft_ahead(&self, _tokens: &[u32]) {}

    /// Best-effort asynchronous pre-warming of offloaded PLE rows for
    /// per-sequence `(window_tokens, prior_tokens)` pairs so bigram and
    /// trigram PLE hashes match cross-sequence verification identically.
    fn warm_ple_draft_ahead_seqs(&self, _seqs: &[(&[u32], &[u32])]) {}

    /// [`Self::alloc_sequence`] for a sequence whose first device work runs on
    /// `stream` (the scheduler's prefill stream). A model may then order that
    /// stream after the new state's initialisation with an event instead of a
    /// host sync (`ATLAS_ALLOC_SEQ_ASYNC`). Default: exactly
    /// [`Self::alloc_sequence`].
    fn alloc_sequence_on(&self, stream: u64) -> Result<SequenceState> {
        let _ = stream;
        self.alloc_sequence()
    }

    /// Copy logits from device to host buffer (for CPU-side sampling).
    ///
    /// `logits_ptr` points to `[vocab_size]` BF16 values on device.
    /// `dst` must be at least `vocab_size * 2` bytes.
    fn copy_logits_to_host(&self, logits_ptr: DevicePtr, dst: &mut [u8]) -> Result<()>;

    /// FP32 logits flag (host buffer needs `vocab*4` bytes, reinterpret `&[f32]`).
    /// True only for Gemma-4 dense single-token decode `lm_head`; default false.
    fn logits_ptr_is_fp32(&self, _logits_ptr: DevicePtr) -> bool {
        false
    }

    /// Base pointer of the on-device logits buffer (`[k, vocab]` BF16 after
    /// `decode_verify_graphed`). Lets the scheduler read logits for temp
    /// sampling even though graphs bake in argmax.
    fn logits_buffer_ptr(&self) -> DevicePtr;

    /// GPU argmax: 4-byte D2H copy vs 304KB BF16 D2H + CPU argmax.
    fn argmax_on_device(&self, logits_ptr: DevicePtr, stream: u64) -> Result<u32>;

    /// GPU batched argmax over `[N, vocab]` BF16; returns N token IDs.
    fn argmax_batch(&self, logits_ptr: DevicePtr, n: usize, stream: u64) -> Result<Vec<u32>>;

    /// Whether [`Model::topk_candidates`] can run (`ATLAS_SAMPLER_DEVICE_TOPK=1`
    /// loaded its kernel and buffer). Default: no.
    fn device_topk_supported(&self) -> bool {
        false
    }

    /// Device top-`m` candidates of one BF16 logits row for stochastic
    /// sampling: `(id, raw f32 value)` of the best finite logits by
    /// `(v / temperature desc, id asc)`, in that order; every finite entry
    /// when there are at most `m`. Spec:
    /// `spark_runtime::sampler::device_topk::topk_candidates_reference`.
    fn topk_candidates(
        &self,
        _logits_row: DevicePtr,
        _temperature: f32,
        _m: usize,
    ) -> Result<Vec<(u32, f32)>> {
        anyhow::bail!("device top-k sampling is not supported by this model")
    }

    /// Certified-F2 statistics of one BF16 row after `edits`
    /// (`edited_sum_exp_bf16`). Spec:
    /// `spark_runtime::sampler::f2_certify::edited_sum_exp_reference`.
    fn edited_sum_exp(
        &self,
        _logits_row: DevicePtr,
        _edits: &[(u32, spark_runtime::sampler::f2_certify::PreF2Edit)],
    ) -> Result<spark_runtime::sampler::f2_certify::SumExpStats> {
        anyhow::bail!("certified F2 is not supported by this model")
    }

    /// Batched 1-launch GPU target Gumbel-Max verify over BF16 verify logits
    /// rows (`sampler_topk_candidates::target_gumbel_verify_rows_bf16`).
    ///
    /// Returns `Ok(None)` when this model/target does not carry the kernel or
    /// when `logits_base` is FP32 so callers fall back cleanly to their per-row
    /// sampler. When supported, returns `Ok(Some(tokens))` with
    /// `tokens.len() == reqs.len()` matching `reqs` order.
    fn sample_gumbel_verify_rows_batched(
        &self,
        _logits_base: DevicePtr,
        _vocab_stride: usize,
        _reqs: &[spark_runtime::sampler::device_topk::GumbelVerifyRowSpec],
    ) -> Result<Option<Vec<u32>>> {
        Ok(None)
    }

    /// Stage per-row Gumbel-Max verify specs for the immediately following verify
    /// forward so the verify CUDA graph can fold `target_gumbel_verify_rows_bf16`
    /// directly into its tail (replacing the discarded `argmax_bf16_rows` and
    /// collapsing two GPU drains into one). Default is a safe no-op.
    fn stage_next_verify_gumbel_rows(
        &self,
        _reqs: &[spark_runtime::sampler::device_topk::GumbelVerifyRowSpec],
    ) {
    }

    /// Take the tokens produced in-graph by the immediately preceding verify
    /// forward when `stage_next_verify_gumbel_rows` was consumed in-graph.
    /// Returns `None` if verify ran eager/argmax instead, in which case the
    /// caller falls back to `sample_gumbel_verify_rows_batched`.
    fn take_in_graph_gumbel_verify_tokens(&self) -> Option<Vec<u32>> {
        None
    }

    /// Returns `true` when this model carries the GPU maximal-coupling
    /// rejection verify kernel (`target_rejection_verify_rows_bf16`) and its
    /// pre-allocated staging + shared `(q_cand_ids, q_probs)` scratch buffers.
    fn gpu_rejection_verify_available(&self) -> bool {
        false
    }

    /// Batched GPU maximal-coupling rejection verify over BF16 verify logits
    /// rows (`sampler_topk_candidates::target_rejection_verify_rows_bf16`).
    ///
    /// Returns `Ok(None)` when this model/target does not carry the kernel or
    /// when `logits_base` is FP32 so callers fall back cleanly.
    fn sample_rejection_verify_rows_batched(
        &self,
        _logits_base: DevicePtr,
        _vocab_stride: usize,
        _reqs: &[RejectionVerifyRowSpec],
    ) -> Result<Option<Vec<PackedRejectionWord>>> {
        Ok(None)
    }

    /// Stage per-row maximal-coupling rejection verify specs for the
    /// immediately following verify forward so the verify CUDA graph can fold
    /// `target_rejection_verify_rows_bf16` directly into its tail.
    fn stage_next_verify_rejection_rows(&self, _reqs: &[RejectionVerifyRowSpec]) {}

    /// Take the packed words produced in-graph by the immediately preceding
    /// verify forward when `stage_next_verify_rejection_rows` was consumed
    /// in-graph.
    fn take_in_graph_rejection_verify_words(&self) -> Option<Vec<PackedRejectionWord>> {
        None
    }

    /// Return the hidden state after final norm from the last decode step.
    ///
    /// Used by MTP speculative decoding: the MTP head takes the target model's
    /// post-norm hidden states as input alongside the token embedding.
    fn hidden_after_norm(&self) -> DevicePtr;

    /// L2-resident multi-token verification: per-position argmax token IDs;
    /// each token advances KV/SSM state. All tokens go through each layer
    /// before moving on so weights stay in L2.
    fn decode_verify(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<Vec<u32>>;

    /// This model must choose request-policy tokens before persistent commit.
    /// Schedulers must route all speculative widths (and ordinary replay)
    /// through the policy-aware entry when this capability is true.
    fn requires_verify_policy(&self) -> bool {
        false
    }

    fn bind_verify_policy_request(
        &self,
        _tokens: &[u32],
        _seq: &SequenceState,
        _stream: u64,
    ) -> Result<crate::model::glm53::verify_policy_binding::BoundVerifyRequest> {
        anyhow::bail!("model does not support bound policy verification")
    }

    fn decode_verify_with_policy(
        &self,
        _request: crate::model::glm53::verify_policy_binding::BoundVerifyRequest,
        _policy: &mut dyn crate::model::glm53::verify_policy_transaction::VerifyPolicy,
    ) -> Result<crate::model::glm53::verify_policy_transaction::VerifyOutcome> {
        anyhow::bail!("model does not support policy-before-commit verification")
    }

    /// Publication failure follows an irreversible device commit. The active
    /// sequence must not be reused without a successful model reset.
    fn poison_verify_policy(&self, _stream: u64) {}

    /// Checkpoint SSM states before speculative verification.
    fn checkpoint_ssm_states(&self, seq: &mut SequenceState) -> Result<()>;

    /// Rollback SSM states after partial acceptance.
    fn rollback_ssm_states(&self, seq: &mut SequenceState, num_accepted: usize) -> Result<()>;

    /// True when this model has recurrent SSM / Mamba layers whose
    /// `h_state` + `conv_state` are advanced in-place every decoded
    /// token.
    ///
    /// Pure-attention models return `false` (the default): their only
    /// per-token state is the paged KV cache, which the Phase-C
    /// boundary rollback rewinds by lowering `seq_len`. Hybrid models
    /// (Qwen3.6-A3B, MiniMax, Nemotron-nano) return `true` — for those
    /// the scheduler MUST also restore the SSM state from a decode-time
    /// snapshot, because the recurrent state cannot be undone by
    /// lowering a cursor.
    fn has_ssm_layers(&self) -> bool {
        false
    }

    /// Number of decode-rollback SSM snapshot slots reserved **per
    /// active sequence** (Phase-C). The scheduler's per-sequence
    /// snapshot ring is sized from this. `0` (the default) means the
    /// model keeps no decode-rollback snapshots — appropriate for
    /// pure-attention models and for SSM models when the snapshot pool
    /// has no capacity reserved. SSM models with a populated pool
    /// override to `ROLLBACK_RESTEER_CAP + 1`.
    fn decode_rollback_ring_slots(&self) -> usize {
        0
    }

    /// Save `seq`'s live SSM `h_state` + `conv_state` (all SSM layers)
    /// into the decode-rollback snapshot slot `ring_slot`.
    ///
    /// `ring_slot` is a per-sequence ring index in
    /// `[0, decode_rollback_ring_slots())`; the model maps it to a
    /// concrete snapshot-pool slot keyed by `seq.slot_idx`. Reuses the
    /// same `SsmSnapshotPool` D2D copy primitive as Marconi prefix
    /// caching and MTP verify (SSOT — one snapshot mechanism).
    ///
    /// Default: no-op `Ok(())` for pure-attention models, which have no
    /// SSM state to snapshot.
    fn save_decode_ssm_snapshot(&self, _seq: &SequenceState, _ring_slot: usize) -> Result<()> {
        Ok(())
    }

    /// Restore `seq`'s SSM `h_state` + `conv_state` (all SSM layers)
    /// from the decode-rollback snapshot slot `ring_slot` previously
    /// written by [`Self::save_decode_ssm_snapshot`].
    ///
    /// Default: no-op `Ok(())` for pure-attention models.
    fn restore_decode_ssm_snapshot(&self, _seq: &SequenceState, _ring_slot: usize) -> Result<()> {
        Ok(())
    }

    /// Speculative decoding via the model's internal MTP proposer; falls
    /// back to regular decode when no proposer is wired up.
    fn generate_speculative(
        &self,
        prompt_tokens: &[u32],
        params: &spark_runtime::sampler::SamplingParams,
        num_drafts: usize,
    ) -> Result<crate::engine::GenerateResult>;

    /// Check if speculative decoding is available (MTP or self-speculative).
    fn has_proposer(&self) -> bool;

    /// Immutable process-start qualification control. Implementations that do
    /// not support a loaded proposer/serial control retain the false default.
    fn speculation_forced_disabled(&self) -> bool {
        false
    }

    /// True when the installed draft proposer is the DFlash γ-block drafter
    /// (see [`crate::speculative::DraftProposer::is_dflash`]). DFlash drafts
    /// bypass the grammar bitmask, so grammar-constrained sequences need the
    /// scheduler-side gate / verify-side masking keyed off this flag.
    fn proposer_is_dflash(&self) -> bool {
        false
    }

    /// Token authority required by the currently active proposer. Generic
    /// models and all existing serving-policy proposers retain the default;
    /// an exact-authority scheduler must query this value rather than infer it
    /// from the proposer kind or runtime flags.
    fn proposer_target_token_authority(&self) -> crate::speculative::TargetTokenAuthority {
        crate::speculative::TargetTokenAuthority::ServingPolicy
    }

    /// Exact verify-side token capacity for the installed DFlash proposer.
    ///
    /// Includes the bonus/root row. The universal default is `None`; the
    /// TransformerModel implementation must forward its actual persistent
    /// parent/state allocation rather than recomputing an environment value.
    fn dflash_verify_capacity_k(&self) -> Option<usize> {
        None
    }

    /// Resolve the exact neural flat width consumed by a sampled DFlash
    /// selector request.  Default rejects so no scheduler may infer this from
    /// public draft count, trained gamma, or physical verify capacity.
    fn dflash_sampled_proposal_width(&self, _requested: usize) -> Result<usize> {
        anyhow::bail!("model does not implement sampled DFlash width planning")
    }

    /// Configured DFlash draft selector contract.  The scheduler uses this
    /// only to select the fail-closed sampled request/verify transaction.
    fn dflash_draft_sample_method(&self) -> crate::dflash_sampling::DflashDraftSampleMethod {
        crate::dflash_sampling::DflashDraftSampleMethod::Greedy
    }

    // ── Speculation arms ────────────────────────────────────────────────
    //
    // A build can carry more than one draft proposer: `--dflash --speculative`
    // installs the external DFlash/DDTree drafter as arm 0 and DEMOTES the
    // in-checkpoint MTP head to arm 1 instead of dropping it. The scheduler's
    // throughput gate picks between them per step, because their strengths are
    // opposite — the external drafter wins on code, the native head on prose.
    //
    // The defaults below describe a single-arm model, so implementations that
    // do not support arms keep working unchanged.

    /// How many proposer arms this model can offer (0, 1 or 2).
    /// A gate is only worth arming when this is ≥ 2.
    fn proposer_arm_count(&self) -> usize {
        if self.has_proposer() { 1 } else { 0 }
    }

    /// Construction identity for a physical proposer arm. Unknown model
    /// implementations return `None`; admission must treat that as unproven.
    fn proposer_arm_kind(&self, _arm: u8) -> Option<crate::speculative::DraftProposerKind> {
        None
    }

    /// Which proposer arm is currently live.
    fn proposer_arm(&self) -> u8 {
        0
    }

    /// Point the model at a proposer arm; returns the arm actually selected,
    /// which is 0 when the requested arm does not exist on this build.
    ///
    /// MUST be paired with [`Self::swap_proposer_state`] on every live
    /// sequence whenever the returned arm differs from the previous one —
    /// each proposer owns a differently-typed per-sequence state, and readers
    /// downcast `SequenceState::proposer_state` to the type they expect.
    fn set_proposer_arm(&self, _arm: u8) -> u8 {
        0
    }

    /// Exchange a sequence's active and parked proposer states, preserving the
    /// invariant that `proposer_state` always belongs to the live arm.
    ///
    /// A free function on the trait rather than on the model: it touches only
    /// the sequence, and every implementation would write the same two lines.
    fn swap_proposer_state(&self, seq: &mut SequenceState) {
        std::mem::swap(&mut seq.proposer_state, &mut seq.proposer_state_alt);
    }

    /// Check if self-speculative decoding is enabled.
    fn has_self_speculative(&self) -> bool;

    /// Whether this backend implements the scheduler's lossless K=2
    /// prompt-lookup verify/checkpoint contract. Backends fail closed unless
    /// they opt in explicitly; a generic `decode_verify_graphed` method alone
    /// is not proof that recurrent rollback and policy-bound logits are safe.
    fn supports_ngram_speculative(&self) -> bool {
        false
    }

    /// The model speculates with its own drafter: the scheduler drives
    /// [`Model::spec_verify`] then [`Model::spec_commit`] for eligible steps. Default: no.
    fn has_internal_spec(&self) -> bool {
        false
    }

    /// Feed `token` at `seq.seq_len`, draft with the model's own drafter and run ONE verify
    /// pass over [token, drafts..]. Returns (drafts, argmax of every verify row); the verify
    /// logits `[drafts.len() + 1, vocab]` (bf16) are at `logits_buffer_ptr()`, row i being the
    /// distribution after [token, d1..di]. The model holds the verify positions until
    /// [`Model::spec_commit`]; `seq` itself is not advanced here. `None`: the model cannot
    /// speculate at this position (e.g. too close to max_seq); decode plainly instead.
    ///
    /// `sampling`: None = greedy drafts (the argmax chain). Some = SAMPLED drafts for T > 0: draft
    /// i is drawn from the model's draft distribution q_i at `draft_uniforms[i]` (one uniform in
    /// [0,1) per draft, from the request RNG), and q_i is kept for [`Model::spec_draft_probs`].
    ///
    /// `verify_drafts`: verify only the first k drafts (k + 1 rows; the returned drafts are those
    /// k). None = all of them. Output is exact for every k; it is a speed choice (DSV4.1:
    /// `dspark_adapt::AdaptiveK`).
    fn spec_verify(
        &self,
        _token: u32,
        _seq: &mut SequenceState,
        _sampling: Option<&SpecSampling>,
        _verify_drafts: Option<usize>,
        _stream: u64,
    ) -> Result<Option<(Vec<u32>, Vec<u32>)>> {
        anyhow::bail!("spec_verify: this model has no internal speculation")
    }

    /// After a SAMPLED [`Model::spec_verify`]: the draft distributions, fp32 `[drafts, vocab]` on
    /// the device (row i = q_i, the distribution draft i was drawn from). None if the last
    /// spec_verify was greedy or the model has no internal speculation.
    fn spec_draft_probs(&self) -> Option<DevicePtr> {
        None
    }

    /// Keep `token` and the first `accepted` drafts of the last [`Model::spec_verify`] (the
    /// rest is rolled back) and advance `seq` by `accepted + 1`.
    fn spec_commit(&self, _seq: &mut SequenceState, _accepted: usize, _stream: u64) -> Result<()> {
        anyhow::bail!("spec_commit: this model has no internal speculation")
    }

    /// Eager decode skipping SSM layers. Used by self-speculative drafting.
    /// Returns logits pointer for argmax. Advances seq_len by 1.
    fn decode_draft(&self, token: u32, seq: &mut SequenceState, stream: u64) -> Result<DevicePtr>;

    /// SPARSE self-speculative draft (sparsity-drafted self-speculation).
    /// Same layer-skip shape as `decode_draft` but routes the dense FFN
    /// through the column-sparse draft GEMV at the given keep threshold
    /// (`thresh_frac`, fraction of per-row max-abs). Selected by
    /// `step_self_spec` when `ATLAS_SELF_SPEC_SPARSE=1`. The draft need not be
    /// bit-exact — the dense verify is the lossless oracle. Default impl falls
    /// back to `decode_draft` (ignoring `thresh_frac`) so non-TransformerModel
    /// impls keep working.
    fn decode_draft_sparse(
        &self,
        token: u32,
        thresh_frac: f32,
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<DevicePtr> {
        let _ = thresh_frac;
        self.decode_draft(token, seq, stream)
    }

    /// Insert the full token sequence (prompt + generated) into the prefix
    /// cache. Call BEFORE `free_sequence()` (block indices must still be
    /// valid). Benefits multi-turn agentic sessions that resend full history.
    fn cache_sequence(&self, seq: &SequenceState);

    /// Free all GPU resources associated with a sequence.
    ///
    /// Releases KV cache blocks and returns SSM state pool slot.
    /// Must be called when a sequence is no longer needed.
    fn free_sequence(&self, seq: &mut SequenceState) -> Result<()>;

    /// Qualification-only exact digest of live Qwen4 recurrent state and the
    /// logically committed KV prefix.  The universal default is inert; the
    /// Qwen4 transformer backend fails closed when its receipt env is armed.
    fn qwen4_target_state_receipt(
        &self,
        _seq: &SequenceState,
    ) -> Result<Option<crate::model::qwen4_target_state_receipt::Qwen4TargetStateReceipt>> {
        Ok(None)
    }

    /// Move a sequence's SSM states to a different pool slot.
    ///
    /// Copies h_state and conv_state across all SSM layers from the current
    /// slot to `new_slot`. Used by the scheduler for slot compaction after
    /// swap_remove to keep the pool dense.
    ///
    /// `new_slot` must be FREE: the implementation claims it from its slot
    /// pool before copying anything. Returns `Ok(true)` when the sequence now
    /// sits on `new_slot`, and `Ok(false)` — sequence untouched, nothing
    /// copied — when `new_slot` is owned by another live sequence (e.g. one
    /// still prefilling, which the scheduler's active list does not see).
    /// Callers must treat `false` as "leave it in place"; contiguity is a
    /// throughput preference, never a correctness need.
    ///
    /// On `Err`, the sequence's slot index, pool-backed pointers, and source
    /// slot ownership must remain unchanged so the caller may retire the
    /// destination owner without freeing survivor state.
    fn compact_sequence(&self, seq: &mut SequenceState, new_slot: usize) -> Result<bool>;

    /// CUDA-graphed K=2 verify: 2 tokens, capture-then-replay. Returns
    /// `[verified_0, verified_1]` argmax IDs. SSM intermediates saved for
    /// partial rollback via `rollback_ssm_states`.
    fn decode_verify_graphed(
        &self,
        tokens: &[u32; 2],
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<[u32; 2]>;

    /// CUDA-graphed K=3 verify (1 verified + 2 drafts). Returns 3 argmax IDs.
    /// SSM intermediates `[0]` and `[1]` are saved for partial rollback.
    fn decode_verify_graphed_k3(
        &self,
        tokens: &[u32; 3],
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<[u32; 3]>;

    /// K=3 verify, c-batched per-step (Path B). Runs K=3 SEQUENTIAL c-batched
    /// single-step decodes (using `decode_batch` + `decode_multi_seq`) instead
    /// of c independent K=3 verify forwards — folds c × 3 = 12 forwards per
    /// layer per step (c=4 case) into 3 batched forwards per layer per step
    /// for a 4× reduction in FFN/SSM/attn kernel launches.
    ///
    /// `tokens_per_seq[i]` = `[last_token, draft0, draft1]` for sequence `i`.
    /// `seqs[i]` mirrors `tokens_per_seq[i]` and must NOT be cross-aliased.
    ///
    /// Returns `Vec<[u32; 3]>` where element `i` is the argmax outputs for
    /// sequence `i`'s three verify positions, identical layout to
    /// `decode_verify_graphed_k3`'s `[u32; 3]` return value.
    ///
    /// **Side effects per seq:** each seq's `seq_len` is incremented by K=3
    /// and the 3 tokens are pushed to `seq.tokens`. Callers must roll back
    /// to `original_seq_len + accept_count[i] + 1` based on the per-seq
    /// argmax / draft comparison AND call
    /// `commit_verify_state_async(seq, accept_count[i] + 1, 3)` to restore
    /// SSM state from the per-step intermediate snapshots saved internally.
    ///
    /// Default impl falls back to looping single-seq `decode_verify_graphed_k3`
    /// (functionally equivalent but no batching win); TransformerModel
    /// overrides with the real batched K-loop.
    fn decode_verify_k3_batched_csk(
        &self,
        tokens_per_seq: &[[u32; 3]],
        seqs: &mut [&mut SequenceState],
        _stream: u64,
    ) -> Result<Vec<[u32; 3]>> {
        // Default: per-seq fallback (no batching win). Real implementation
        // lives on TransformerModel.
        let mut out = Vec::with_capacity(seqs.len());
        for (i, seq) in seqs.iter_mut().enumerate() {
            let r = self.decode_verify_graphed_k3(&tokens_per_seq[i], seq, 0)?;
            out.push(r);
        }
        Ok(out)
    }

    /// K=2 verify, c-batched per-step. K=2 sibling of
    /// `decode_verify_k3_batched_csk`. Runs K=2 SEQUENTIAL c-batched
    /// single-step decodes (using `decode_batch` + `decode_multi_seq`)
    /// instead of c independent K=2 verify forwards — folds c × 2 = 2c
    /// forwards per layer per step into 2 batched forwards per layer per
    /// step.
    ///
    /// `tokens_per_seq[i]` = `[last_token, draft0]` for sequence `i`.
    /// `seqs[i]` mirrors `tokens_per_seq[i]` and must NOT be cross-aliased.
    ///
    /// Returns `Vec<[u32; 2]>` where element `i` is the argmax outputs
    /// for sequence `i`'s two verify positions, identical layout to
    /// `decode_verify_graphed`'s `[u32; 2]` return value.
    ///
    /// **Side effects per seq:** each seq's `seq_len` is incremented by
    /// K=2 and the 2 tokens are pushed to `seq.tokens`. Callers must
    /// roll back to `original_seq_len + accept_count[i] + 1` based on
    /// per-seq argmax / draft comparison AND call
    /// `commit_verify_state_async(seq, accept_count[i] + 1, 2)` to
    /// restore SSM state from the per-step intermediate snapshots saved
    /// internally.
    ///
    /// Default impl falls back to looping single-seq
    /// `decode_verify_graphed` (functionally equivalent but no batching
    /// win); TransformerModel overrides with the real batched K-loop.
    fn decode_verify_k2_batched_csk(
        &self,
        tokens_per_seq: &[[u32; 2]],
        seqs: &mut [&mut SequenceState],
        _stream: u64,
    ) -> Result<Vec<[u32; 2]>> {
        let mut out = Vec::with_capacity(seqs.len());
        for (i, seq) in seqs.iter_mut().enumerate() {
            let r = self.decode_verify_graphed(&tokens_per_seq[i], seq, 0)?;
            out.push(r);
        }
        Ok(out)
    }

    /// Cross-sequence batched DFlash verify: verify `c` sequences' K=γ+1
    /// draft windows in ONE forward whose FFN GEMMs read the ~14 GB of NVFP4
    /// FFN weights ONCE for all `c×K` rows (instead of `c` separate per-seq
    /// verify forwards). `tokens_per_seq[s]` = seq `s`'s `[last_token,
    /// draft0, .., draft_{γ-1}]`; all seqs MUST share the same `K` (scheduler
    /// aligns the batch).
    ///
    /// Returns `Vec<Vec<u32>>` where element `s` is seq `s`'s per-position
    /// argmax (length `K`), identical layout to `decode_verify` per seq.
    ///
    /// **Side effects per seq (identical to `decode_verify`):** `seq_len += K`
    /// and all `K` tokens pushed to `seq.tokens`. The caller rolls back per-seq
    /// and calls `commit_verify_state_async(seq, num_accepted+1, K)` per-seq
    /// exactly as the single-seq DFlash verify path does — the per-sequence
    /// accept/commit/propose loops are UNCHANGED, so output stays lossless.
    ///
    /// Default impl loops the single-seq `decode_verify` (functionally
    /// identical, no batching win); TransformerModel overrides with the real
    /// batched forward.
    fn decode_verify_dflash_batched(
        &self,
        tokens_per_seq: &[Vec<u32>],
        seqs: &mut [&mut SequenceState],
        _stream: u64,
    ) -> Result<Vec<Vec<u32>>> {
        let mut out = Vec::with_capacity(seqs.len());
        for (i, seq) in seqs.iter_mut().enumerate() {
            let r = self.decode_verify(&tokens_per_seq[i], seq, 0)?;
            out.push(r);
        }
        Ok(out)
    }

    /// Cross-sequence native-MTP K-row verify (`ATLAS_QWEN4_BATCHED_K5=1`).
    ///
    /// `tokens_per_seq[s]` is sequence `s`'s `[last_token, drafts..]`; every
    /// window has the same width K. Returns each sequence's per-row greedy
    /// argmax, identical in layout and value to its own single-sequence
    /// `decode_verify_dflash`. Side effects per sequence are those of the
    /// single-sequence verify (`seq_len += K`, the K inputs pushed) and every
    /// sequence's QSA journal / GDN retain transaction is left in the lane of
    /// its slot, so each commit must run under that slot's
    /// `qwen4_verify_lanes::LaneGuard`. Implementations without the
    /// cross-sequence frame fail closed.
    fn decode_verify_qwen4_batched(
        &self,
        _tokens_per_seq: &[Vec<u32>],
        _seqs: &mut [&mut SequenceState],
    ) -> Result<Vec<Vec<u32>>> {
        anyhow::bail!("this model has no cross-sequence Qwen4 verify")
    }

    /// Reserve shared target scratch once for the entire cohort, every
    /// sequence verifying `k` rows (K5; K1..=K8 with
    /// `ATLAS_QWEN4_BATCHED_ADAPTIVE_K=1`). All read-only subframe preflights
    /// must succeed before this call.
    fn begin_verify_qwen4_batch(&self, _seqs: &mut [&mut SequenceState], _k: usize) -> Result<()> {
        anyhow::bail!("this model has no owned native K5 batch cohort")
    }

    /// Settle one physical target subframe while retaining the cohort owner.
    fn end_verify_qwen4_batch_frame(&self, _seqs: &[&SequenceState]) -> Result<()> {
        anyhow::bail!("this model has no owned native K5 batch cohort")
    }

    /// Release scratch only after all cohort subframes and hidden saves settle.
    fn finish_verify_qwen4_batch(&self, _seqs: &[&SequenceState]) -> Result<()> {
        anyhow::bail!("this model has no owned native K5 batch cohort")
    }

    /// Attempt every lane's rightful cleanup. Committed or published prefixes
    /// drain without preframe rollback; poisoned ownership remains fatal.
    fn abort_verify_qwen4_batch(&self, _seqs: &mut [&mut SequenceState]) -> Result<()> {
        anyhow::bail!("this model has no owned native K5 batch cohort")
    }

    /// A positive terminal commit retires with no future hidden consumer.
    /// A zero-prefix terminal abort is already settled by ordinary abort.
    fn retire_verify_qwen4_batch_terminal(&self, _seq: &SequenceState) -> Result<()> {
        anyhow::bail!("this model has no owned native K5 batch cohort")
    }

    /// Read-only admission of a cross-sequence native-MTP verify frame. `Ok`
    /// means `decode_verify_qwen4_batched` will not refuse these windows
    /// before its first effect; `Err` leaves every sequence untouched so the
    /// caller can verify them one at a time instead.
    fn preflight_verify_qwen4_batched(
        &self,
        _tokens_per_seq: &[Vec<u32>],
        _seqs: &[&SequenceState],
    ) -> Result<()> {
        anyhow::bail!("this model has no cross-sequence Qwen4 verify")
    }

    /// Whether the cross-sequence native-MTP verify is armed: the selector is
    /// set and per-slot verify lanes were provisioned at construction.
    fn qwen4_batched_k5_armed(&self) -> bool {
        false
    }

    /// Sequence-owned native-MTP hidden handoff out of a cross-sequence
    /// verify frame: like `save_hidden_for_mtp_sequence(token_idx, ..)` with
    /// the sequence's rows starting at `row_base` of the target hidden arena.
    /// Every sequence of a frame must save before any proposer runs, because
    /// proposals reuse the shared target scratch.
    fn save_hidden_for_mtp_sequence_in_frame(
        &self,
        _row_base: usize,
        _token_idx: usize,
        _seq: &SequenceState,
        _stream: u64,
    ) -> Result<()> {
        anyhow::bail!("this model has no cross-sequence native-MTP hidden handoff")
    }

    /// Retire accepted batched-verify DFlash hidden captures from `dflash_hidden_save`
    /// at `row_base..=row_base + num_accepted` into `seq`'s private `DflashProposerState`
    /// ring and close the verify batch transaction's hidden-publication lane.
    fn retire_dflash_frame_captures(
        &self,
        _row_base: usize,
        _num_accepted: usize,
        _seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<()> {
        Ok(())
    }

    /// Retire accepted cross-sequence flat-DFlash hidden captures (`ATLAS_DFLASH_BATCHED_VERIFY=1`)
    /// from `dflash_hidden_save[row_base..=row_base + num_accepted]` directly into `seq`'s
    /// private `DflashProposerState` ring so subsequent `run_mtp_propose_multi_batched` sees
    /// fresh 5-layer target hidden conditioning with `dstate.ctx_len == seq.seq_len`.
    fn retire_dflash_flat_batch_captures(
        &self,
        _row_base: usize,
        _num_accepted: usize,
        _seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<()> {
        Ok(())
    }

    /// CUDA-graphed K=4 verify (1 verified + 3 drafts). Returns 4 argmax IDs.
    /// SSM intermediates [0..3] saved for partial rollback.
    fn decode_verify_graphed_k4(
        &self,
        tokens: &[u32; 4],
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<[u32; 4]>;

    /// DFlash K=γ graphed verify (γ+1 tokens). Specialization of the K=2/3/4
    /// pattern for arbitrary K. Default impl falls back to eager
    /// `decode_verify`. Models can override for CUDA-graph speedup keyed by
    /// `(slot_idx, K)`.
    fn decode_verify_graphed_kgamma(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<Vec<u32>> {
        self.decode_verify(tokens, seq, stream)
    }

    /// True while the one-shot C=1 stage comparator still needs a replay.
    fn k1_stage_diag_requested(&self, _pre_verify_len: usize, _tokens: &[u32]) -> Result<bool> {
        Ok(false)
    }

    /// Begin/advance/end the ordinary-decode side of the stage comparator.
    fn k1_stage_diag_begin(&self, _pre_verify_len: usize, _tokens: &[u32]) -> Result<()> {
        Ok(())
    }
    fn k1_stage_diag_finish_serial_row(&self) -> Result<()> {
        Ok(())
    }
    fn k1_stage_diag_finish_serial(&self) -> Result<()> {
        Ok(())
    }
    fn k1_stage_diag_abort(&self) {}

    /// Arm/query the qualification-only native-K16 stage and route receipts.
    /// Backends are fail-closed unless they implement the exact Qwen4 frame.
    fn k16_fixture_evidence_begin(&self, _pre_verify_len: usize, _tokens: &[u32]) -> Result<()> {
        bail!("K16 fixture evidence is unsupported by this model backend")
    }
    fn k16_fixture_evidence_finish(&self, _pre_verify_len: usize, _tokens: &[u32]) -> Result<()> {
        bail!("K16 fixture evidence is unsupported by this model backend")
    }
    fn k16_fixture_evidence_abort(&self) {}

    /// One-shot native-Qwen4 K16 committed recurrent-state diagnostic.
    /// Defaults remain inert for every backend that does not explicitly
    /// implement the exact Qwen4 state contract.
    fn k16_commit_parity_requested(&self, _pre_verify_len: usize, _tokens: &[u32]) -> Result<bool> {
        Ok(false)
    }
    fn k16_commit_parity_capture_live(
        &self,
        _seq: &SequenceState,
    ) -> Result<crate::model::k16_commit_parity::CanonicalStateSnapshot> {
        bail!("K16 commit parity is unsupported by this model backend")
    }
    fn k16_commit_parity_restore_live(
        &self,
        _seq: &SequenceState,
        _snapshot: &crate::model::k16_commit_parity::CanonicalStateSnapshot,
    ) -> Result<()> {
        bail!("K16 commit parity is unsupported by this model backend")
    }
    fn k16_commit_parity_capture_sources(
        &self,
        _seq: &SequenceState,
        _k: usize,
        _last_inter_slot: usize,
    ) -> Result<crate::model::k16_commit_parity::CommitSourceSnapshot> {
        bail!("commit-parity source capture is unsupported by this model backend")
    }
    fn k16_commit_parity_restore_sources(
        &self,
        _seq: &SequenceState,
        _snapshot: &crate::model::k16_commit_parity::CommitSourceSnapshot,
    ) -> Result<()> {
        bail!("commit-parity source restore is unsupported by this model backend")
    }
    fn k16_commit_parity_drain_default(&self) -> Result<()> {
        bail!("K16 commit parity is unsupported by this model backend")
    }
    fn k16_commit_parity_compare_committed(
        &self,
        _seq: &SequenceState,
        _expected: &crate::model::k16_commit_parity::CanonicalStateSnapshot,
        _pre_verify_len: usize,
        _tokens: &[u32],
        _total_accepted: usize,
        _k: usize,
        _last_inter_slot: usize,
    ) -> Result<()> {
        bail!("K16 commit parity is unsupported by this model backend")
    }

    /// DFlash γ-token verification: 1 verified + γ drafts → per-position
    /// argmax. Variable-length γ (vs fixed K=2/3/4) because it's a drafter
    /// config field. CUDA-graph capture keyed by `(slot_idx, tokens.len())`.
    /// Default routes to `decode_verify_graphed_kgamma`.
    fn decode_verify_dflash(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<Vec<u32>> {
        // DEBUG: route to the K=2 graphed path (used by MTP, known-working).
        // ATLAS_DFLASH_VERIFY_VIA_K2=1 forces this; otherwise use kgamma.
        if std::env::var("ATLAS_DFLASH_VERIFY_VIA_K2").ok().as_deref() == Some("1")
            && tokens.len() == 2
        {
            let pair: &[u32; 2] = tokens.try_into().expect("len==2 guarantees array");
            let result = self.decode_verify_graphed(pair, seq, stream)?;
            return Ok(result.to_vec());
        }
        self.decode_verify_graphed_kgamma(tokens, seq, stream)
    }

    /// Save the raw, pre-final-norm target hidden state at `token_idx` to a
    /// dedicated legacy MTP input buffer. Must precede `run_mtp_propose` — MTP
    /// overwrites shared buffers including `norm_output`.
    ///
    /// Migration: external native Qwen4 MTP callers must instead call
    /// [`Self::save_hidden_for_mtp_sequence`] with the sequence subsequently
    /// passed to `run_mtp_propose`/`run_mtp_propose_multi`. This legacy save
    /// does not publish a sequence-owned frame, so native proposal rejects it
    /// rather than consuming the unbound shared buffer.
    fn save_hidden_for_mtp(&self, token_idx: usize, stream: u64) -> Result<()>;

    /// Sequence-bound raw target-hidden handoff. Native MTP implementations
    /// may retain accepted prefixes independently for each physical slot.
    /// The default preserves other model implementations' existing handoff.
    /// Save after committing the accepted target prefix, at its current
    /// `seq_len`, and before any other request overwrites target scratch. The
    /// native handoff is consumed once by proposal at that same sequence cursor.
    fn save_hidden_for_mtp_sequence(
        &self,
        token_idx: usize,
        _seq: &SequenceState,
        stream: u64,
    ) -> Result<()> {
        self.save_hidden_for_mtp(token_idx, stream)
    }

    /// Run the bonus token through the target model and capture its hidden
    /// states into the DFlash `dflash_hidden_save` buffer at index 0.
    ///
    /// This is needed because the verify forward pass does not include the
    /// bonus token, so DFlash's next propose() step would otherwise draft
    /// from an outdated prefix.
    ///
    /// Default: no-op (DFlash not active).
    fn save_hidden_for_dflash(
        &self,
        _token: u32,
        _seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<()> {
        Ok(())
    }

    /// THINKING-PHASE ctx capture (default-OFF, `ATLAS_DFLASH_CAPTURE_THINKING=1`).
    ///
    /// Append the just-decoded thinking token's captured 5-layer target
    /// hidden (in `dflash_hidden_save[0]`) into the per-seq `ctx_hidden_acc`
    /// accumulator at its absolute slot, keeping `ctx_len` in lockstep with
    /// `seq.seq_len`. Must be called from the plain-decode (thinking) path
    /// AFTER `seq.seq_len` has been incremented. Fills the ctx region that
    /// the thinking span would otherwise leave ZERO, so the answer-phase
    /// drafter conditions on real reasoning context.
    ///
    /// Drafter-conditioning only — target verify untouched, committed tokens
    /// byte-identical. Default: no-op (DFlash not active / flag off).
    fn dflash_capture_thinking(&self, _seq: &mut SequenceState, _stream: u64) -> Result<()> {
        Ok(())
    }

    /// Retire any verifier-capture rows that precede a plain bootstrap decode.
    ///
    /// DFlash normally appends these rows at proposal entry.  When a policy or
    /// structural boundary intentionally suppresses that proposal, bootstrap
    /// must append them before decode overwrites capture row zero.  Returns the
    /// exact number of rows appended; zero is the normal no-debt case.
    /// Implementations must fail rather than advancing over a cursor hole.
    fn reconcile_dflash_context_before_bootstrap(
        &self,
        _seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<usize> {
        Ok(0)
    }

    /// Run the MTP proposer for one draft token off the saved hidden state.
    /// `None` when no proposer is wired.
    fn run_mtp_propose(
        &self,
        token: u32,
        position: usize,
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<Option<u32>>;

    /// Run the MTP proposer to generate multiple draft tokens.
    ///
    /// Uses the hidden state previously saved via `save_hidden_for_mtp`.
    /// Returns empty vec if no MTP proposer is available.
    ///
    /// `grammar_bitmask`: when `Some`, drafts are constrained to the allowed
    /// token set of an XGrammar matcher at its current position. Format is
    /// `ceil(vocab_size / 32)` i32 words; bit `tok` set ⇒ allowed. `None`
    /// preserves the unconstrained GPU-argmax fast path.
    fn run_mtp_propose_multi(
        &self,
        token: u32,
        position: usize,
        num_drafts: usize,
        seq: &mut SequenceState,
        stream: u64,
        grammar_bitmask: Option<&[i32]>,
    ) -> Result<Vec<u32>>;

    /// Cross-sequence batched MTP proposal for `tokens.len()` sequences sharing
    /// the same requested `num_drafts` and unconstrained sampling policy.
    /// Default loops [`Self::run_mtp_propose_multi`] per sequence.
    fn run_mtp_propose_multi_batched(
        &self,
        tokens: &[u32],
        positions: &[usize],
        num_drafts: usize,
        seqs: &mut [&mut SequenceState],
        stream: u64,
    ) -> Result<Vec<Vec<u32>>> {
        anyhow::ensure!(
            tokens.len() == positions.len() && tokens.len() == seqs.len(),
            "run_mtp_propose_multi_batched extent mismatch"
        );
        let mut out = Vec::with_capacity(seqs.len());
        for ((&token, &position), seq) in tokens.iter().zip(positions).zip(seqs.iter_mut()) {
            out.push(self.run_mtp_propose_multi(token, position, num_drafts, seq, stream, None)?);
        }
        Ok(out)
    }

    /// Stage one deterministic sampled-DFlash selector request on `seq`.
    /// Default rejects; only a proposer with the typed sparse-q ABI may
    /// accept it.
    fn stage_dflash_sampling_request(
        &self,
        _seq: &mut SequenceState,
        _request: crate::dflash_sampling::DflashDraftSamplingRequest,
    ) -> Result<()> {
        anyhow::bail!("model does not implement sampled DFlash requests")
    }

    /// Drain the sampled proposal paired with the last proposer result.
    fn take_pending_dflash_sampled_proposal(
        &self,
        _seq: &mut SequenceState,
    ) -> Option<crate::dflash_sampling::DflashSampledProposalFrame> {
        None
    }

    /// Read the draft token ID stored on GPU by the last `run_mtp_propose_multi`
    /// call (which used `embed_from_argmax` to write the draft embedding and
    /// token ID directly on GPU). Returns 0 if no proposer is available.
    fn read_deferred_draft_token(&self) -> Result<u32> {
        Ok(0)
    }

    /// Upload a DDTree payload's `parent_indices` to device and stash
    /// the pointer so the next K=γ verify call routes through the tree-aware
    /// GDN kernel. Caller is responsible for clearing via
    /// `clear_ddtree_parent_ids` after verify. No-op for models without an
    /// SSM path. Returns Ok(true) when stashed, Ok(false) when payload is
    /// flat (kernel falls through to wy_k).
    fn set_ddtree_parent_ids(&self, _payload: &crate::layers::DDTreePayload) -> Result<bool> {
        Ok(false)
    }

    /// Clear the DDTree `parent_ids` stash after verify. Default no-op.
    fn clear_ddtree_parent_ids(&self) {}

    /// Did the LAST K=γ verify give every tree node ancestor-exact attention
    /// (pre-context + true ancestors + itself)?
    ///
    /// `true` for flat/chain payloads and for the per-row KV indirection
    /// verify path (`ATLAS_TREE_AWARE_ATTN` +
    /// `ATLAS_DDTREE_TREE_AWARE_VERIFY`). `false` when a NON-flat tree
    /// payload was verified with prefix-read metadata only (incl.
    /// `ATLAS_DDTREE_DFS_REORDER=1`, whose depth-prefix reads are exact only
    /// for the leftmost spine — branch rows read a spine sibling and miss
    /// their own key). The scheduler MUST NOT run the full tree-commit
    /// walker (`greedy_sample_ddtree_full`) when this is `false`: branch
    /// rows' argmaxes are mis-conditioned and committing through them
    /// diverges from the greedy oracle (the 2026-07-08 deep-branch-tail
    /// corruption). Default `true` (models without a DFlash tree verify
    /// never produce mis-conditioned rows).
    fn dflash_tree_ancestor_attn_exact(&self) -> bool {
        true
    }

    /// Whether this model can run a non-flat DDTree verify with exact
    /// attention and recurrent state for every layer.
    ///
    /// The scheduler consults this before widening a flat DFlash verify to
    /// the full tree. Returning false discards the tree payload and preserves
    /// the ordinary flat verifier, avoiding expensive work whose branch rows
    /// could not be committed safely. Models must fail closed.
    fn dflash_tree_verify_capable(&self) -> bool {
        false
    }

    /// Drain the per-seq DDTree payload stashed by the last propose
    /// call. Default returns `None` (flat / MTP / ngram drafters). DDTree-
    /// capable drafters override to surface their last-step tree topology so
    /// the scheduler can assign to `ActiveSeq.pending_tree_payload` for the
    /// next-step verifier.
    fn take_pending_tree_payload(
        &self,
        _seq: &mut SequenceState,
    ) -> Option<crate::layers::DDTreePayload> {
        None
    }

    /// Encode images through the vision encoder and store embeddings for the next prefill.
    ///
    /// Each tuple is `(pixels: Vec<f32>, grid_h: usize, grid_w: usize)`.
    /// Pixels are laid out [P, C×T×Hp×Wp] matching `vision_preprocess::preprocess_image`.
    /// Must be called before `prefill_chunk` when the prompt contains `<|image_pad|>` tokens.
    ///
    /// Default: no-op (text-only models).
    fn prepare_vision_embed(&self, _images: &[(Vec<f32>, usize, usize)]) -> Result<()> {
        Ok(())
    }

    /// EP worker step: receive a command from rank 0 and execute it.
    ///
    /// Returns false when the worker should shut down.
    /// Only valid on rank > 0 with EP enabled.
    fn ep_worker_step(&self, _seq: &mut SequenceState) -> Result<bool> {
        Ok(true) // no-op for non-EP models
    }

    /// Check whether expert parallelism (EP) is enabled (multi-GPU MoE).
    ///
    /// When true, the scheduler must use separate decode + prefill commands
    /// with explicit EP broadcasts rather than mixed_forward (which has no
    /// EP broadcast protocol defined).
    fn is_ep(&self) -> bool {
        false
    }

    /// True when cross-sequence Qwen4 batched verification (`verify_qwen4_batched`)
    /// is supported across TP worker ranks (`TP=2, EP=1, PP=1`).
    fn qwen4_tp_batched_verify_supported(&self) -> bool {
        false
    }

    /// True when single-token decode `lm_head` writes FP32 logits to a
    /// dedicated FP32 scratch buffer (rather than the shared BF16 logits
    /// buffer). Callers that consume those logits must read from
    /// [`Self::decode_logits_ptr`] using 4 bytes/element. Defaults false;
    /// only Gemma-4 dense overrides today (gated by
    /// `ATLAS_GEMMA4_FP32_LMHEAD=1`).
    fn decode_logits_fp32(&self) -> bool {
        false
    }

    /// Buffer pointer the single-token decode `lm_head` last wrote to. The
    /// returned dtype is FP32 when [`Self::decode_logits_fp32`] is true,
    /// BF16 otherwise. The default impl returns the shared BF16 logits
    /// buffer ([`Self::logits_buffer_ptr`]) used by every existing model.
    /// Override alongside [`Self::decode_logits_fp32`] on models that route
    /// the lm_head output through an FP32 scratch (Gemma-4 + softcap).
    fn decode_logits_ptr(&self) -> DevicePtr {
        self.logits_buffer_ptr()
    }

    /// Multi-head Latent Attention guard. When true, chunked prefill MUST run
    /// as a single chunk — Flux Engine has no paged-MLA prefill kernel and
    /// multi-chunk MLA silently corrupts attention output.
    fn is_mla(&self) -> bool {
        false
    }

    /// Multi-head hyperconnection stream count (`0` for ordinary residual
    /// models).  Until a model implements its highway semantics in every
    /// batched phase, the scheduler uses this capability to serialize mixed
    /// prefill and the model assembles concurrent decode rows through C1.
    fn hc_mult(&self) -> usize {
        0
    }

    /// Qwen4 batch-versus-serial decode plan (`crate::qwen4_batch_policy`).
    /// `None` for every model without a native Qwen4 batch route. The
    /// scheduler uses it to log and count routes; the model's own
    /// `decode_batch` applies the same plan, so this never changes a route.
    fn qwen4_batch_plan(&self) -> Result<Option<crate::qwen4_batch_policy::Qwen4BatchPlan>> {
        Ok(None)
    }

    /// The route `decode_batch` will take for exactly these rows: the plan,
    /// the row policy and the per-width native capability. `None` for C1 and
    /// for every model without a native Qwen4 route. Read-only.
    fn qwen4_decode_route(
        &self,
        _seqs: &[&mut SequenceState],
    ) -> Result<Option<crate::qwen4_batch_policy::FrameRoute>> {
        Ok(None)
    }

    /// Load-time native capability of a native Qwen4 plan for every width in
    /// `C2..=min(max_batch_size, QWEN4_NATIVE_MAX_ROWS)` (256-row cap). Errors
    /// when QSA rows were requested but cannot run. Empty for serial plans and
    /// other models.
    fn qwen4_native_capability_report(
        &self,
        _max_batch_size: usize,
    ) -> Result<crate::qwen4_batch_policy::NativeCapabilityReport> {
        Ok(Default::default())
    }

    /// Row width of the model's fast-prefill chunk geometry when a fast
    /// prefill family is armed that admits only chunks of exactly this many
    /// rows at aligned offsets (Qwen4 `ATLAS_QWEN4_PREFILL_MOE_BATCH` with
    /// `ATLAS_QWEN4_PREFILL_WIDE`). The scheduler warns when its
    /// active-decode chunk cap is below this width, because such chunks are
    /// refused by the family and the request runs on the ordinary path.
    /// `None` when no chunk-geometry-locked prefill route exists.
    fn fast_prefill_chunk_rows(&self) -> Option<usize> {
        None
    }

    /// Prepared chunk widths (ascending) that the armed fast-prefill family
    /// admits at any offset aligned to the width, for bounded prefill chunks
    /// interleaved with decode (Qwen4 wide geometry: powers of two from 512
    /// up to `ATLAS_QWEN4_PREFILL_WIDE`). Empty when no such family is armed;
    /// the scheduler then keeps its historical chunk sizing.
    fn fast_prefill_chunk_widths(&self) -> Vec<usize> {
        Vec::new()
    }

    /// EP broadcast: send a command (u32) to all worker ranks.
    ///
    /// Called by rank 0 before each model operation to synchronize workers.
    /// Only valid when EP is enabled.
    fn ep_broadcast_cmd(&self, _cmd: u32) -> Result<()> {
        Ok(()) // no-op for non-EP models
    }

    /// EP bulk broadcast: send an array of u32 tokens to all worker ranks.
    /// Uses a single NCCL broadcast instead of per-token broadcasts.
    fn ep_broadcast_tokens(&self, _tokens: &[u32]) -> Result<Vec<u32>> {
        Ok(Vec::new()) // no-op for non-EP models
    }

    /// Trim the MTP proposer's KV cache after verification.
    ///
    /// Called on rejection to discard the rejected draft's MTP KV entry.
    fn trim_proposer_state(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
        stream: u64,
    ) -> Result<()>;

    /// `ATLAS_DFLASH_ASYNC`: collect the real drafts of an async
    /// (second-stream) DFlash propose previously launched for this sequence.
    ///
    /// The scheduler calls this at the top of each step for sequences with
    /// pending drafts; `Some(drafts)` replaces the placeholder chain the
    /// launch returned. `Some(vec![])` signals an orphaned placeholder —
    /// the caller clears its pending drafts and bootstraps (lossless).
    /// Default `None`: nothing pending / feature unsupported.
    fn dflash_collect_async_drafts(&self, seq: &mut SequenceState) -> Result<Option<Vec<u32>>> {
        let _ = seq;
        Ok(None)
    }

    /// ATLAS_DFLASH_FUSED=1: record the propose-ordering event immediately
    /// after verify returns, before commit kernels are enqueued. This lets the
    /// drafter's secondary CUDA stream run in parallel with SSM commit + KV
    /// reshape (~10ms), instead of waiting for them.
    ///
    /// Must be called right after `decode_verify_dflash` returns, before any
    /// commit call. No-op when ATLAS_DFLASH_ASYNC or ATLAS_DFLASH_FUSED unset.
    fn dflash_arm_propose_overlap(&self) -> Result<()> {
        Ok(())
    }

    /// True when `seq` currently holds an uncollected placeholder chain from an
    /// in-flight asynchronous DFlash propose (`ATLAS_DFLASH_ASYNC=1`).
    fn dflash_seq_has_async_placeholder(&self, seq: &SequenceState) -> bool {
        let _ = seq;
        false
    }

    /// Maximum physical verify width `K` available for zero-training native-MTP
    /// DLoop (`ATLAS_MTP_DLOOP=1`). Returns `0` (fail-closed) by default.
    fn mtp_dloop_max_verify_k(&self) -> usize {
        0
    }

    /// ATLAS_DFLASH_RECYCLE=1: stash the discarded draft tail on the DFlash
    /// proposer state so the next propose can re-offer it (lossless).
    ///
    /// `drafts` is the FULL set of drafts that was just verified, and
    /// `num_accepted` is the matching-prefix length. The discarded tail is
    /// `drafts[num_accepted+1 .. ]` — the draft at index `num_accepted` was the
    /// mismatch (replaced by the target's correction), and everything after it
    /// was thrown away purely because it sat downstream of that miss.
    /// `corrected_token` is the bonus the target committed at the mismatch
    /// (= the next step's `last_token`); it becomes the recycle key.
    ///
    /// Default no-op for non-DFlash / non-recycle backends.
    fn dflash_stash_recycle(
        &self,
        _seq: &mut SequenceState,
        _drafts: &[u32],
        _num_accepted: usize,
        _corrected_token: u32,
    ) -> Result<()> {
        Ok(())
    }

    /// ATLAS_DFLASH_ECHO=1 (echo-drafting / Jacobi salvage): stash the
    /// TARGET'S OWN verify argmaxes downstream of the bonus on the DFlash
    /// proposer state so the next propose can offer them as a
    /// target-authored draft chain at zero propose cost (lossless).
    ///
    /// `verified` is the FULL flat-chain verify output
    /// (`[argmax_after_last_token, argmax_after_draft_0, ...]`, length γ+1)
    /// and `num_accepted` the matching-prefix length; the salvage is
    /// `verified[num_accepted+1 ..]` — the tokens AFTER the bonus.
    /// `bonus_token` (= the next step's `last_token`) becomes the echo key.
    /// The min-accept / min-tail gates (`echo::should_stash`) are applied
    /// here; a gated-out step invalidates any prior stash.
    ///
    /// Default no-op for non-DFlash / non-echo backends.
    fn dflash_stash_echo(
        &self,
        _seq: &mut SequenceState,
        _verified: &[u32],
        _num_accepted: usize,
        _bonus_token: u32,
    ) -> Result<()> {
        Ok(())
    }

    /// Launch SSM state checkpoint D2D copies on a secondary CUDA stream.
    ///
    /// Non-blocking: returns immediately. The copies can overlap with MTP
    /// propose on the default stream since they access disjoint memory.
    /// Call `sync_secondary` before the next verify to ensure completion.
    fn start_checkpoint_async(&self, seq: &mut SequenceState) -> Result<()> {
        // Default: fall back to synchronous checkpoint.
        self.checkpoint_ssm_states(seq)
    }

    /// Launch SSM state rollback + checkpoint on the secondary stream.
    ///
    /// Used on the reject path: rollback to `intermediate[0]`, then checkpoint
    /// the rolled-back state for the next verify iteration.
    fn start_rollback_and_checkpoint_async(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
    ) -> Result<()> {
        // Default: fall back to synchronous operations.
        self.rollback_ssm_states(seq, num_accepted)?;
        self.checkpoint_ssm_states(seq)
    }

    /// Wait for all work on the secondary stream to complete.
    fn sync_secondary(&self) -> Result<()> {
        Ok(()) // No-op if no secondary stream.
    }

    /// Copy canonical SSM state from `*_checkpoint` into
    /// `*_state` BEFORE verify so the kernel can scratch-write it. Runs on
    /// default_stream (FIFO ordering with the next kernel). No-op default
    /// for non-MTP backends.
    fn pre_verify_copy_async(&self, _seq: &mut SequenceState) -> Result<()> {
        Ok(())
    }

    /// Finalize only the Qwen4 QSA side-cache transaction for an exact K5
    /// frame. Diagnostics use this before replaying ordinary K1 rows so they
    /// do not consume or overwrite recurrent/PLE commit sources.
    fn commit_qwen4_qsa_k5_only(&self, _num_accepted: usize, _k: usize) -> Result<()> {
        Ok(())
    }

    /// Sequence-bound QSA restoration for diagnostics. The frame remains owned.
    fn commit_qwen4_qsa_for_sequence(
        &self,
        _seq: &SequenceState,
        accepted: usize,
        rows: usize,
    ) -> Result<()> {
        self.commit_qwen4_qsa_k5_only(accepted, rows)
    }

    /// Release a diagnostic frame only after its QSA and recurrent restoration succeeded.
    fn finish_verify_diagnostic(&self, _seq: &mut SequenceState, _rows: usize) -> Result<()> {
        Ok(())
    }

    /// Finish a serial K1 rebuild using the original verifier's owner ticket.
    fn finish_verify_rebuild(
        &self,
        _seq: &mut SequenceState,
        _accepted: usize,
        _rows: usize,
    ) -> Result<()> {
        Ok(())
    }

    /// Abort a failed verify frame and restore every mutable cache owner.
    /// Implementations with asynchronous commit streams must drain them before
    /// restoring the default-stream QSA and recurrent/PLE state.
    fn abort_verify_state(&self, seq: &mut SequenceState, k: usize) -> Result<()> {
        let qsa = self.commit_qwen4_qsa_for_sequence(seq, 0, k);
        let recurrent = self.pre_verify_copy_async(seq);
        match (qsa, recurrent) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(error), Ok(())) | (Ok(()), Err(error)) => Err(error),
            (Err(qsa_error), Err(recurrent_error)) => Err(anyhow::anyhow!(
                "QSA rollback failed: {qsa_error:#}; recurrent rollback failed: {recurrent_error:#}"
            )),
        }
    }

    /// Commit a verify pass to the canonical SSM state.
    /// `num_accepted ∈ [0, k]`: full accept → copy `h_state` → checkpoint;
    /// partial → copy `h_state_intermediates[num_accepted-1]`; full reject →
    /// no-op. Runs on secondary_stream; pair with `sync_secondary`.
    fn commit_verify_state_async(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
        k: usize,
    ) -> Result<()> {
        // Default: fall back to synchronous behavior compatible with the
        // legacy NGram path. Backends without dual-buffer support use the
        // pre-existing checkpoint/rollback machinery.
        if num_accepted == k {
            self.checkpoint_ssm_states(seq)
        } else if num_accepted > 0 {
            self.rollback_ssm_states(seq, num_accepted)
        } else {
            Ok(())
        }
    }

    /// DDTree-aware variant of `commit_verify_state_async`.
    ///
    /// `last_inter_slot` is the kernel slot index of the *last accepted*
    /// SSM state in `h_state_intermediates`. For chain-mode verify this
    /// equals `num_accepted - 1` and the behavior is identical to
    /// `commit_verify_state_async`. For DDTree (tree-aware) verify the
    /// accepted compact indices may be non-contiguous — e.g.
    /// `[1, 4, 7]` — and the kernel writes the post-acceptance state at
    /// slot `7`, NOT slot `2 = len - 1`. Callers must pass the explicit
    /// slot derived from the tree walk (`accepted_compact_indices.last()`
    /// or `0` when no drafts were accepted).
    ///
    /// Default impl forwards to the legacy `commit_verify_state_async`,
    /// which only sees `num_accepted` and `k` — equivalent to assuming
    /// `last_inter_slot == num_accepted - 1`. Backends that distinguish
    /// chain vs tree (TransformerModel) should override.
    fn commit_verify_state_async_with_slot(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
        k: usize,
        _last_inter_slot: usize,
    ) -> Result<()> {
        self.commit_verify_state_async(seq, num_accepted, k)
    }

    /// Whether [`Self::commit_verify_batch_cached`] can collapse a multi-sequence
    /// verify commit into a single cached-table GPU launch instead of per-sequence
    /// [`Self::commit_verify_state_async`] calls. Default: `false`.
    fn supports_batched_verify_commit(&self) -> bool {
        false
    }

    /// Commit multiple verified sequences `(seq, num_accepted, k)` in a single
    /// cached-pointer-table GPU launch when supported, falling back cleanly to
    /// [`Self::commit_verify_state_async`] per sequence otherwise.
    fn commit_verify_batch_cached(
        &self,
        entries: &mut [(&mut SequenceState, usize, usize)],
        _stream: u64,
    ) -> Result<()> {
        for (seq, num_accepted, k) in entries.iter_mut() {
            self.commit_verify_state_async(seq, *num_accepted, *k)?;
        }
        Ok(())
    }

    /// Compact a tree-fork accept's scattered attention KV down to a
    /// contiguous run (mirrors the flat-chain KV layout) so the next decode
    /// reads correct contiguous KV.
    ///
    /// `accepted_compact` is the full accepted path (compact indices, walk
    /// order, possibly non-contiguous). `pre_verify_len` is the sequence
    /// length before this verify advanced it. No-op when the path is already
    /// contiguous (every flat-chain accept). LOSSLESS: relocates committed
    /// K/V bytes, never alters a value.
    ///
    /// Default impl is a no-op (non-hybrid / no-tree backends never produce
    /// a non-contiguous accept). `TransformerModel` overrides.
    fn compact_verify_kv(
        &self,
        _seq: &SequenceState,
        _accepted_compact: &[usize],
        _pre_verify_len: usize,
    ) -> Result<()> {
        Ok(())
    }

    /// `ATLAS_DFLASH_TREE_COMMIT`: stamp the accepted-path compact
    /// indices onto the proposer so its next ctx-hidden append reads the
    /// scattered fork-capture rows. Call AFTER `trim_proposer_state`.
    /// Default no-op (non-DFlash / non-tree backends).
    fn set_dflash_accepted_compact(&self, _seq: &mut SequenceState, _accepted_compact: &[usize]) {}

    /// Save KV blocks + SSM state to writer. Does NOT free resources.
    ///
    /// Format: `[KV layers × blocks × (K + V)]` then `[SSM layers × (h + conv)]`.
    /// The model owns the serialization format.
    fn save_sequence_state(
        &self,
        _seq: &SequenceState,
        _writer: &mut dyn std::io::Write,
    ) -> Result<()> {
        bail!("swap not supported by this model")
    }

    /// Restore KV blocks + SSM state from reader into an allocated sequence.
    ///
    /// Allocates `num_blocks` new KV blocks, fills from reader, restores SSM.
    fn restore_sequence_state(
        &self,
        _seq: &mut SequenceState,
        _num_blocks: usize,
        _reader: &mut dyn std::io::Read,
    ) -> Result<()> {
        bail!("swap not supported by this model")
    }

    /// Number of free KV cache blocks available for allocation.
    fn num_free_blocks(&self) -> usize {
        0
    }

    /// Total KV blocks in the paged cache (denominator for occupancy
    /// gauges). Default 0 for backends without a paged cache.
    fn num_total_blocks(&self) -> usize {
        0
    }

    /// KV blocks available to request admission after permanent model-owned
    /// reservations. Backends without reserved paged-cache blocks inherit
    /// the raw total.
    fn num_admission_blocks(&self) -> usize {
        self.num_total_blocks()
    }

    /// Allocate `seq`'s KV blocks for its next `rows` positions now, as the
    /// next decode (`rows = 1`) or verify frame (`num_drafts + 1`) would.
    /// Under a shared pool (`--kv-pool-tokens`) the scheduler calls this per
    /// row before a batched step, so a row the pool cannot grow fails alone.
    /// Default: nothing; the step allocates for itself.
    fn reserve_decode_kv(&self, seq: &mut SequenceState, rows: usize) -> Result<()> {
        let _ = (seq, rows);
        Ok(())
    }

    /// Block the host until every operation already submitted to `stream`
    /// has completed.
    ///
    /// This has no default implementation: scheduler error paths use it as a
    /// lifetime barrier before model-owned device buffers may be released.
    /// A silent no-op would turn a recoverable request failure into use-after-
    /// free on the next request.
    fn synchronize_stream(&self, stream: u64) -> Result<()>;

    /// Return the default CUDA stream handle.
    fn default_stream(&self) -> u64 {
        0
    }

    /// Create a new CUDA stream (for overlapping prefill with decode).
    fn create_stream(&self) -> Result<u64> {
        Ok(0)
    }

    /// Create a CUDA event (for inter-stream synchronization).
    fn create_event(&self) -> Result<u64> {
        Ok(0)
    }

    /// Record an event on a stream (marks a point in the stream's work).
    fn record_event(&self, _event: u64, _stream: u64) -> Result<()> {
        Ok(())
    }

    /// Make a stream wait for an event (GPU-side sync, CPU does not block).
    fn stream_wait_event(&self, _stream: u64, _event: u64) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod dflash_capacity_abi_tests {
    use super::Model;

    #[test]
    fn capacity_accessor_is_object_safe() {
        fn read_capacity(model: &dyn Model) -> Option<usize> {
            model.dflash_verify_capacity_k()
        }
        let accessor: fn(&dyn Model) -> Option<usize> = read_capacity;
        let _ = accessor;
    }

    #[test]
    fn ple_draft_ahead_methods_are_object_safe() {
        fn warm_flat(model: &dyn Model, tokens: &[u32]) {
            model.warm_ple_draft_ahead(tokens);
        }
        fn warm_seqs(model: &dyn Model, seqs: &[(&[u32], &[u32])]) {
            model.warm_ple_draft_ahead_seqs(seqs);
        }
        let _: fn(&dyn Model, &[u32]) = warm_flat;
        let _: fn(&dyn Model, &[(&[u32], &[u32])]) = warm_seqs;
    }
}

#[cfg(test)]
mod padded_batch_n_tests {
    use super::padded_batch_n;

    #[test]
    fn covers_the_upstream_ladder_through_c128() {
        for (n, want) in [
            (1usize, 2usize),
            (2, 2),
            (3, 4),
            (4, 4),
            (5, 8),
            (8, 8),
            (9, 12),
            (12, 12),
            (13, 16),
            (16, 16),
            (17, 24),
            (24, 24),
            (25, 32),
            (32, 32),
            (33, 48),
            (48, 48),
            (49, 64),
            (64, 64),
            (65, 96),
            (96, 96),
            (97, 128),
            (128, 128),
        ] {
            assert_eq!(padded_batch_n(n), want, "n={n}");
        }
    }

    #[test]
    fn values_above_the_qualified_ladder_fall_through() {
        assert_eq!(padded_batch_n(129), 192);
        assert_eq!(padded_batch_n(193), 256);
        assert_eq!(padded_batch_n(257), 257);
    }
}
