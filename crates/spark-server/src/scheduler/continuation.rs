// SPDX-License-Identifier: AGPL-3.0-only

//! ContinuationRecord: facade over decode sequence state without moving memory.
//!
//! - Wraps `SequenceState`, `SsmDecodeRing`, `session_hash`, `prompt_prefix_hash`,
//!   and deterministic incremental `prefix_hash`.
//! - Computes `prefix_hash` alongside `session_hash`.
//! - Retains `session_hash` for routing and snapshot lookups to preserve
//!   byte-identical default behavior.
//! - Routes sequence claim, prefill cancel/disconnect/error abort release, swap-out/in,
//!   rollback, boundary snapshotting, and sequence completion release through the record
//!   facade while preserving fixed CUDA memory addresses and CUDA graph stability.
//!
//! ## Reset-Proof Scope (`scope=cpu-metadata`)
//! - Enforces CPU-metadata consistency on sequence claim (`claim_metadata_consistent`),
//!   rollback snapshot restore (`token_position == keep_len` and
//!   `prefix_hash == expected_prefix_hash`), and sequence release
//!   (`release_metadata_consistent` set only after `Model::free_sequence` returns `Ok(())`).
//! - When enabled via `debug_assertions` or `FLUX_CONTINUATION_RESET_PROOF=1` in release
//!   builds, dirty claims and inconsistent swap restores are refused at runtime (not
//!   compiled out via `debug_assert!`). Externally observable device-hygiene proof is
//!   provided by the soak contamination canaries (`back-to-back` and `slot-reuse`).

use super::rollback::RollbackFallback;
use super::ssm_decode_ring::SsmDecodeRing;
use spark_model::traits::{Model, SequenceState};

/// Canonical fixed seed for deterministic 64-bit prefix hashing.
/// Unlike `DefaultHasher`, this seed is identical across runs and processes.
pub const CANONICAL_PREFIX_SEED: u64 = 0x517c_c1b7_2722_0a95;

/// FNV-1a 64-bit prime constant.
pub const FNV_PRIME_64: u64 = 0x0100_0000_01b3;

/// Reset-proof verification scope identifier (CPU metadata + snapshot hash/position ledger).
pub const PROOF_SCOPE_CPU_METADATA: &str = "cpu-metadata";

/// Reserved scope identifier for future device-side tensor digest verification.
pub const PROOF_SCOPE_DEVICE_TENSOR_HASH: &str = "device-tensor-hash";

/// Update a running prefix hash with a newly committed token ID.
#[inline]
pub fn hash_token_incremental(prev_hash: u64, token: u32) -> u64 {
    (prev_hash ^ (token as u64)).wrapping_mul(FNV_PRIME_64)
}

/// Compute a deterministic 64-bit prefix hash from tokens and optional image digest.
pub fn compute_prefix_hash(prompt_tokens: &[u32], image_digest: Option<&[u8]>) -> u64 {
    let mut h = CANONICAL_PREFIX_SEED;
    if let Some(digest) = image_digest
        && !digest.is_empty()
    {
        for &byte in digest {
            h = (h ^ (byte as u64)).wrapping_mul(FNV_PRIME_64);
        }
    }
    for &tok in prompt_tokens {
        h = hash_token_incremental(h, tok);
    }
    h
}

/// Compute a deterministic prefix hash folding multimodal pixel tensors into the seed.
pub fn compute_prefix_hash_with_images(
    prompt_tokens: &[u32],
    image_pixels: &[(Vec<f32>, usize, usize)],
) -> u64 {
    if image_pixels.is_empty() {
        compute_prefix_hash(prompt_tokens, None)
    } else {
        let mut ctx = ring::digest::Context::new(&ring::digest::SHA256);
        for (pixels, w, h) in image_pixels {
            ctx.update(&w.to_le_bytes());
            ctx.update(&h.to_le_bytes());
            for &val in pixels {
                ctx.update(&val.to_bits().to_le_bytes());
            }
        }
        let digest = ctx.finish();
        compute_prefix_hash(prompt_tokens, Some(digest.as_ref()))
    }
}

/// Recompute the deterministic prefix hash from the initial prompt prefix hash
/// (which already includes any multimodal image digest) and the committed output tokens.
#[inline]
pub fn recompute_prefix_hash_from_prompt(
    prompt_prefix_hash: u64,
    committed_output_tokens: &[u32],
) -> u64 {
    let mut h = prompt_prefix_hash;
    for &tok in committed_output_tokens {
        h = hash_token_incremental(h, tok);
    }
    h
}

/// Lifecycle state of a continuation record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RecordState {
    #[default]
    Fresh,
    Live,
    Snapshotted,
    Released,
}

/// Verification entry for one snapshot in the reset proof ledger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnapshotProofEntry {
    pub token_position: usize,
    pub prefix_hash: u64,
    pub snapshot_slot: usize,
}

/// Canary/debug reset proof verifying per-request state hygiene.
///
/// Under `scope = "cpu-metadata"`, `claim_metadata_consistent` verifies CPU-side
/// `SequenceState` reset invariants on sequence claim or swap restore, `snapshots` verifies
/// `(token_position, prefix_hash)` on SSM rollback restore, and `release_metadata_consistent`
/// is set only after `Model::free_sequence` returns `Ok(())`.
#[derive(Debug, Clone)]
pub struct ResetProofPayload {
    pub scope: &'static str,
    pub claimed_slot_idx: usize,
    pub initial_prefix_hash: u64,
    pub snapshots: Vec<SnapshotProofEntry>,
    pub claim_metadata_consistent: bool,
    pub release_metadata_consistent: bool,
}

impl ResetProofPayload {
    /// Construct a reset proof payload for a slot index and initial prefix hash.
    pub fn new(slot_idx: usize, initial_prefix_hash: u64) -> Self {
        let claim_metadata_consistent = slot_idx != usize::MAX;
        Self {
            scope: PROOF_SCOPE_CPU_METADATA,
            claimed_slot_idx: slot_idx,
            initial_prefix_hash,
            snapshots: Vec::new(),
            claim_metadata_consistent,
            release_metadata_consistent: false,
        }
    }

    /// Construct a reset proof payload from a newly claimed `SequenceState`,
    /// verifying that the claimed sequence CPU metadata is clean (`slot_idx != usize::MAX`,
    /// `seq_len == 0`, empty `block_table`, empty `tokens`, and no residual verify/chunk metadata).
    pub fn from_claimed_seq(seq: &SequenceState, initial_prefix_hash: u64) -> Self {
        let claim_metadata_consistent = seq.slot_idx != usize::MAX
            && seq.seq_len == 0
            && seq.block_table.is_empty()
            && seq.tokens.is_empty()
            && seq.verify_frame.is_none()
            && seq.chunked_prefill_meta.is_none();
        Self {
            scope: PROOF_SCOPE_CPU_METADATA,
            claimed_slot_idx: seq.slot_idx,
            initial_prefix_hash,
            snapshots: Vec::new(),
            claim_metadata_consistent,
            release_metadata_consistent: false,
        }
    }

    /// Construct a reset proof payload for a sequence restored from disk swap,
    /// recording its actual `seq.slot_idx` and verifying restored CPU state consistency.
    pub fn from_restored_seq(seq: &SequenceState, prefix_hash: u64) -> Self {
        let claim_metadata_consistent = seq.slot_idx != usize::MAX
            && seq.seq_len == seq.tokens.len()
            && seq.verify_frame.is_none();
        Self {
            scope: PROOF_SCOPE_CPU_METADATA,
            claimed_slot_idx: seq.slot_idx,
            initial_prefix_hash: prefix_hash,
            snapshots: Vec::new(),
            claim_metadata_consistent,
            release_metadata_consistent: false,
        }
    }

    /// Runtime enforcement of `claim_metadata_consistent`.
    /// Works identically in debug builds and release builds when `FLUX_CONTINUATION_RESET_PROOF=1`.
    pub fn verify_claim(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.claim_metadata_consistent,
            "ContinuationRecord reset proof (scope={}) failed on sequence claim: slot_idx={}, claim_metadata_consistent=false",
            self.scope,
            self.claimed_slot_idx,
        );
        Ok(())
    }

    /// Format a canonical receipt line reporting the proof scope and consistency state.
    pub fn receipt(&self) -> String {
        format!(
            "scope={} slot_idx={} initial_prefix_hash=0x{:016x} claim_metadata_consistent={} release_metadata_consistent={} snapshots={}",
            self.scope,
            self.claimed_slot_idx,
            self.initial_prefix_hash,
            self.claim_metadata_consistent,
            self.release_metadata_consistent,
            self.snapshots.len(),
        )
    }

    pub fn record_snapshot(
        &mut self,
        token_position: usize,
        prefix_hash: u64,
        snapshot_slot: usize,
    ) {
        self.snapshots.push(SnapshotProofEntry {
            token_position,
            prefix_hash,
            snapshot_slot,
        });
    }

    pub fn truncate_after(&mut self, keep_len: usize) {
        self.snapshots
            .retain(|entry| entry.token_position <= keep_len);
    }

    pub fn verify_restore(&self, snapshot_slot: usize, expected_prefix_hash: u64) -> bool {
        if let Some(entry) = self
            .snapshots
            .iter()
            .rfind(|e| e.snapshot_slot == snapshot_slot)
        {
            entry.prefix_hash == expected_prefix_hash
        } else {
            false
        }
    }

    /// Verify that `snapshot_slot` matches both `expected_position` (`keep_len`)
    /// and `expected_prefix_hash`.
    pub fn verify_restore_at(
        &self,
        snapshot_slot: usize,
        expected_position: usize,
        expected_prefix_hash: u64,
    ) -> bool {
        if let Some(entry) = self
            .snapshots
            .iter()
            .rfind(|e| e.snapshot_slot == snapshot_slot)
        {
            entry.token_position == expected_position && entry.prefix_hash == expected_prefix_hash
        } else {
            false
        }
    }

    /// Retrieve the recorded prefix hash for a given snapshot slot.
    pub fn snapshot_prefix_hash(&self, snapshot_slot: usize) -> Option<u64> {
        self.snapshots
            .iter()
            .rfind(|e| e.snapshot_slot == snapshot_slot)
            .map(|e| e.prefix_hash)
    }

    /// Verify position and return the recorded prefix hash for a given snapshot slot.
    pub fn verify_and_get_restore_hash(
        &self,
        snapshot_slot: usize,
        expected_position: usize,
    ) -> Option<u64> {
        let entry = self
            .snapshots
            .iter()
            .rfind(|e| e.snapshot_slot == snapshot_slot)?;
        if entry.token_position == expected_position {
            Some(entry.prefix_hash)
        } else {
            None
        }
    }
}

/// Check if reset proof verification is active (`cfg(debug_assertions)` or
/// `FLUX_CONTINUATION_RESET_PROOF=1` in release builds).
#[inline]
pub fn is_reset_proof_enabled() -> bool {
    #[cfg(debug_assertions)]
    {
        true
    }
    #[cfg(not(debug_assertions))]
    {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var("FLUX_CONTINUATION_RESET_PROOF")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false)
        })
    }
}

/// Ownership-preserving quarantine record for a `SequenceState` whose reclamation
/// (`model.free_sequence`) failed after a dirty-claim or swap-resume failure.
///
/// Retaining the `SequenceState` in process quarantine prevents dropping an unreclaimed
/// or partially-freed GPU slot on the stack, while activating a permanent model-scoped
/// fail-stop gate that refuses blind sequence re-allocation for the remainder of the
/// process lifetime. Production recovery from a quarantined sequence is **process restart**
/// (never retrying a potentially partial `free_sequence` in-process).
pub struct QuarantinedSequence {
    pub model_addr: usize,
    pub slot_idx: usize,
    pub seq: SequenceState,
    pub reason: String,
}

static QUARANTINED_SEQUENCES: std::sync::Mutex<Vec<QuarantinedSequence>> =
    std::sync::Mutex::new(Vec::new());

#[inline]
fn model_key(model: &dyn Model) -> usize {
    (model as *const dyn Model as *const ()) as usize
}

/// Move an unreclaimed or partially-freed `SequenceState` into the process quarantine
/// registry so its ownership is preserved (never dropped on error) and all future sequence
/// claims on `model` permanently fail-stop until process restart.
pub fn quarantine_unreclaimed_sequence(
    model: &dyn Model,
    seq: SequenceState,
    reason: String,
) -> usize {
    let model_addr = model_key(model);
    let slot_idx = seq.slot_idx;
    let mut guard = QUARANTINED_SEQUENCES
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    guard.push(QuarantinedSequence {
        model_addr,
        slot_idx,
        seq,
        reason,
    });
    guard.iter().filter(|q| q.model_addr == model_addr).count()
}

/// Return true if `model` currently has any unreclaimed `SequenceState` in quarantine
/// (permanent fail-stop active; process restart required for production recovery).
pub fn is_model_quarantined(model: &dyn Model) -> bool {
    quarantined_count_for_model(model) > 0
}

/// Return the number of unreclaimed `SequenceState` entries currently quarantined for `model`.
pub fn quarantined_count_for_model(model: &dyn Model) -> usize {
    let model_addr = model_key(model);
    let guard = QUARANTINED_SEQUENCES
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    guard.iter().filter(|q| q.model_addr == model_addr).count()
}

/// Return `(slot_idx, reason)` summaries for all sequences currently quarantined for `model`.
pub fn quarantined_summaries_for_model(model: &dyn Model) -> Vec<(usize, String)> {
    let model_addr = model_key(model);
    let guard = QUARANTINED_SEQUENCES
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    guard
        .iter()
        .filter(|q| q.model_addr == model_addr)
        .map(|q| (q.slot_idx, q.reason.clone()))
        .collect()
}

/// Permanent fail-stop gate: refuses new sequence claims on `model` while any unreclaimed or
/// partially-freed `SequenceState` is in quarantine, preventing blind retry or in-process
/// re-free of a potentially partially freed slot. Production recovery is process restart.
pub fn ensure_not_quarantined(model: &dyn Model) -> anyhow::Result<()> {
    let model_addr = model_key(model);
    let guard = QUARANTINED_SEQUENCES
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut count = 0usize;
    let mut first_slot = 0usize;
    let mut first_reason = String::new();
    for q in guard.iter().filter(|q| q.model_addr == model_addr) {
        if count == 0 {
            first_slot = q.slot_idx;
            first_reason.clone_from(&q.reason);
        }
        count += 1;
    }
    if count > 0 {
        anyhow::bail!(
            "ContinuationRecord fail-stop active: {count} unreclaimed sequence(s) quarantined (first slot {first_slot}: {first_reason}); refusing blind sequence allocation (process restart required for production recovery)"
        );
    }
    Ok(())
}

/// Attempt to reclaim `seq` via `model.free_sequence` after `primary_err`; if `free_sequence`
/// also fails, move `seq` into the ownership-preserving quarantine registry (never dropping it)
/// and activate permanent fail-stop for `model` (process restart required for production recovery).
pub fn reclaim_or_quarantine_sequence(
    model: &dyn Model,
    mut seq: SequenceState,
    primary_err: anyhow::Error,
    context: &str,
) -> anyhow::Error {
    let slot_idx = seq.slot_idx;
    if let Err(free_err) = model.free_sequence(&mut seq) {
        let reason = format!(
            "{context}: {primary_err:#}; additionally failed to reclaim dirty sequence slot {slot_idx}: {free_err:#}"
        );
        let quarantined_total = quarantine_unreclaimed_sequence(model, seq, reason.clone());
        return anyhow::anyhow!(
            "{reason} (SequenceState quarantined; fail-stop active with {quarantined_total} quarantined sequence(s); process restart required)"
        );
    }
    primary_err
}

/// ContinuationRecord step 1 facade wrapping `SequenceState`, `SsmDecodeRing`,
/// `session_hash`, `prompt_prefix_hash`, and deterministic `prefix_hash`.
pub struct ContinuationRecord {
    pub id: u64,
    pub seq: SequenceState,
    pub ssm_rollback_ring: SsmDecodeRing,
    pub session_hash: u64,
    pub prompt_prefix_hash: u64,
    pub prefix_hash: u64,
    pub state: RecordState,
    pub reset_proof: Option<ResetProofPayload>,
}

impl ContinuationRecord {
    /// Claim a fresh sequence slot from `model` on the default stream and arm its reset proof.
    /// When reset proof is enabled (`debug_assertions` or `FLUX_CONTINUATION_RESET_PROOF=1` in release),
    /// a dirty `SequenceState` is refused at runtime after freeing the sequence slot.
    pub fn claim_sequence(
        model: &dyn Model,
        initial_prefix_hash: u64,
    ) -> anyhow::Result<(SequenceState, Option<ResetProofPayload>)> {
        Self::claim_sequence_with_proof_mode(model, initial_prefix_hash, is_reset_proof_enabled())
    }

    /// Claim a fresh sequence slot from `model` on the default stream with an explicit proof toggle.
    /// Enforces `claim_metadata_consistent` at runtime when `enable_proof` is true, freeing the
    /// sequence before returning `Err` if dirty, or quarantining the `SequenceState` and activating
    /// fail-stop if `free_sequence` fails.
    pub fn claim_sequence_with_proof_mode(
        model: &dyn Model,
        initial_prefix_hash: u64,
        enable_proof: bool,
    ) -> anyhow::Result<(SequenceState, Option<ResetProofPayload>)> {
        ensure_not_quarantined(model)?;
        let seq = model.alloc_sequence()?;
        let proof = if enable_proof {
            let p = ResetProofPayload::from_claimed_seq(&seq, initial_prefix_hash);
            if let Err(err) = p.verify_claim() {
                return Err(reclaim_or_quarantine_sequence(
                    model,
                    seq,
                    err,
                    "ContinuationRecord::claim_sequence",
                ));
            }
            Some(p)
        } else {
            None
        };
        Ok((seq, proof))
    }

    /// Claim a fresh sequence slot from `model` on `stream` and arm its reset proof.
    /// When reset proof is enabled (`debug_assertions` or `FLUX_CONTINUATION_RESET_PROOF=1` in release),
    /// a dirty `SequenceState` is refused at runtime after freeing the sequence slot.
    pub fn claim_sequence_on(
        model: &dyn Model,
        stream: u64,
        initial_prefix_hash: u64,
    ) -> anyhow::Result<(SequenceState, Option<ResetProofPayload>)> {
        Self::claim_sequence_on_with_proof_mode(
            model,
            stream,
            initial_prefix_hash,
            is_reset_proof_enabled(),
        )
    }

    /// Claim a fresh sequence slot from `model` on `stream` with an explicit proof toggle.
    /// Enforces `claim_metadata_consistent` at runtime when `enable_proof` is true, freeing the
    /// sequence before returning `Err` if dirty, or quarantining the `SequenceState` and activating
    /// fail-stop if `free_sequence` fails.
    pub fn claim_sequence_on_with_proof_mode(
        model: &dyn Model,
        stream: u64,
        initial_prefix_hash: u64,
        enable_proof: bool,
    ) -> anyhow::Result<(SequenceState, Option<ResetProofPayload>)> {
        ensure_not_quarantined(model)?;
        let seq = model.alloc_sequence_on(stream)?;
        let proof = if enable_proof {
            let p = ResetProofPayload::from_claimed_seq(&seq, initial_prefix_hash);
            if let Err(err) = p.verify_claim() {
                return Err(reclaim_or_quarantine_sequence(
                    model,
                    seq,
                    err,
                    "ContinuationRecord::claim_sequence_on",
                ));
            }
            Some(p)
        } else {
            None
        };
        Ok((seq, proof))
    }

    /// Abort and release a prefill sequence through the record facade, marking
    /// `release_metadata_consistent = true` only after `model.free_sequence` succeeds.
    pub fn abort_prefill_sequence(
        model: &dyn Model,
        seq: &mut SequenceState,
        reset_proof: &mut Option<ResetProofPayload>,
    ) -> anyhow::Result<()> {
        let res = model.free_sequence(seq);
        if res.is_ok()
            && let Some(proof) = reset_proof.as_mut()
        {
            proof.release_metadata_consistent = true;
        }
        res
    }

    /// Release a raw `SequenceState` through the record facade.
    pub fn release_raw_sequence(model: &dyn Model, seq: &mut SequenceState) -> anyhow::Result<()> {
        model.free_sequence(seq)
    }

    /// Fallible constructor that enforces `claim_metadata_consistent` at runtime when proof is enabled,
    /// returning ownership of `seq` alongside the error on failure so the caller can release it via `model`.
    #[expect(
        clippy::result_large_err,
        reason = "returns SequenceState by value so caller can release GPU slots on failure"
    )]
    pub fn try_new(
        id: u64,
        seq: SequenceState,
        ssm_rollback_ring: SsmDecodeRing,
        session_hash: u64,
        prefix_hash: u64,
    ) -> Result<Self, (SequenceState, anyhow::Error)> {
        let reset_proof = if is_reset_proof_enabled() {
            let proof = ResetProofPayload::from_claimed_seq(&seq, prefix_hash);
            if let Err(err) = proof.verify_claim() {
                return Err((seq, err));
            }
            Some(proof)
        } else {
            None
        };
        Ok(Self {
            id,
            seq,
            ssm_rollback_ring,
            session_hash,
            prompt_prefix_hash: prefix_hash,
            prefix_hash,
            state: RecordState::Live,
            reset_proof,
        })
    }

    /// Construct a new `ContinuationRecord` wrapping the given components without dropping `seq`.
    /// Call [`Self::verify_claim`] (or use [`Self::try_new`] / [`Self::claim_from_prompt`]) to enforce
    /// `claim_metadata_consistent` while retaining the ability to release `self.seq` via `model`.
    pub fn new(
        id: u64,
        seq: SequenceState,
        ssm_rollback_ring: SsmDecodeRing,
        session_hash: u64,
        prefix_hash: u64,
    ) -> Self {
        let reset_proof = if is_reset_proof_enabled() {
            Some(ResetProofPayload::from_claimed_seq(&seq, prefix_hash))
        } else {
            None
        };
        Self {
            id,
            seq,
            ssm_rollback_ring,
            session_hash,
            prompt_prefix_hash: prefix_hash,
            prefix_hash,
            state: RecordState::Live,
            reset_proof,
        }
    }

    /// Fallible constructor from initial prompt tokens and allocation parameters, returning
    /// ownership of `seq` alongside the error if `verify_claim()` fails.
    #[expect(
        clippy::result_large_err,
        reason = "returns SequenceState by value so caller can release GPU slots on failure"
    )]
    pub fn try_from_prompt(
        id: u64,
        seq: SequenceState,
        ring_capacity: usize,
        prompt_tokens: &[u32],
        image_digest: Option<&[u8]>,
    ) -> Result<Self, (SequenceState, anyhow::Error)> {
        let session_hash = seq.session_hash;
        let prefix_hash = compute_prefix_hash(prompt_tokens, image_digest);
        let ssm_rollback_ring = SsmDecodeRing::new(ring_capacity);
        Self::try_new(id, seq, ssm_rollback_ring, session_hash, prefix_hash)
    }

    /// Claim a fresh sequence from `model` and construct a `ContinuationRecord` from prompt tokens,
    /// reclaiming the sequence via `model.free_sequence` if `verify_claim()` fails.
    pub fn claim_from_prompt(
        id: u64,
        model: &dyn Model,
        ring_capacity: usize,
        prompt_tokens: &[u32],
        image_digest: Option<&[u8]>,
    ) -> anyhow::Result<Self> {
        let prefix_hash = compute_prefix_hash(prompt_tokens, image_digest);
        let (seq, reset_proof) = Self::claim_sequence(model, prefix_hash)?;
        let session_hash = seq.session_hash;
        let ssm_rollback_ring = SsmDecodeRing::new(ring_capacity);
        Ok(Self {
            id,
            seq,
            ssm_rollback_ring,
            session_hash,
            prompt_prefix_hash: prefix_hash,
            prefix_hash,
            state: RecordState::Live,
            reset_proof,
        })
    }

    /// Construct a `ContinuationRecord` from initial prompt tokens and allocation parameters
    /// without dropping `seq`.
    pub fn from_prompt(
        id: u64,
        seq: SequenceState,
        ring_capacity: usize,
        prompt_tokens: &[u32],
        image_digest: Option<&[u8]>,
    ) -> Self {
        let session_hash = seq.session_hash;
        let prefix_hash = compute_prefix_hash(prompt_tokens, image_digest);
        let ssm_rollback_ring = SsmDecodeRing::new(ring_capacity);
        Self::new(id, seq, ssm_rollback_ring, session_hash, prefix_hash)
    }

    /// Verify that `claim_metadata_consistent` holds on this record when reset proof is armed.
    pub fn verify_claim(&self) -> anyhow::Result<()> {
        if let Some(ref proof) = self.reset_proof {
            proof.verify_claim()?;
        }
        Ok(())
    }

    /// Update the deterministic incremental prefix hash with a newly committed token.
    #[inline]
    pub fn update_prefix_hash(&mut self, token: u32) {
        self.prefix_hash = hash_token_incremental(self.prefix_hash, token);
    }

    #[inline]
    pub fn session_hash(&self) -> u64 {
        self.session_hash
    }

    #[inline]
    pub fn prompt_prefix_hash(&self) -> u64 {
        self.prompt_prefix_hash
    }

    #[inline]
    pub fn prefix_hash(&self) -> u64 {
        self.prefix_hash
    }

    #[inline]
    pub fn slot_idx(&self) -> usize {
        self.seq.slot_idx
    }

    #[inline]
    pub fn state(&self) -> RecordState {
        self.state
    }

    #[inline]
    pub fn set_state(&mut self, state: RecordState) {
        self.state = state;
    }

    /// Record an SSM boundary snapshot if the model has SSM layers and the decode ring is enabled.
    pub fn snapshot_boundary_if_ssm(
        &mut self,
        output_tokens: &[u32],
        model: &dyn Model,
    ) -> Option<usize> {
        if !model.has_ssm_layers() || !self.ssm_rollback_ring.is_enabled() {
            return None;
        }
        let mask = super::helpers::boundary_token_mask()?;
        let &last = output_tokens.last()?;
        let id = last as usize;
        if id >= mask.len() || !mask[id] {
            return None;
        }
        let token_position = output_tokens.len();
        self.record_and_save_boundary_snapshot(token_position, model)
    }

    /// Record an SSM boundary snapshot at token_position and save via model.
    pub fn record_and_save_boundary_snapshot(
        &mut self,
        token_position: usize,
        model: &dyn Model,
    ) -> Option<usize> {
        if !model.has_ssm_layers() || !self.ssm_rollback_ring.is_enabled() {
            return None;
        }
        let slot = self.ssm_rollback_ring.record(token_position)?;
        if let Err(e) = model.save_decode_ssm_snapshot(&self.seq, slot) {
            tracing::warn!(
                error = %e,
                ring_slot = slot,
                token_position,
                "SSM decode-snapshot save failed; dropping ring entry"
            );
            self.ssm_rollback_ring
                .truncate_after(token_position.saturating_sub(1));
            return None;
        }
        if let Some(proof) = &mut self.reset_proof {
            proof.record_snapshot(token_position, self.prefix_hash, slot);
        }
        self.state = RecordState::Snapshotted;
        Some(slot)
    }

    /// Restore SSM state for rollback if hybrid model, verifying reset proof against
    /// `expected_prefix_hash` and `keep_len` when armed, and restoring `prefix_hash`.
    pub fn restore_for_rollback(
        &mut self,
        ssm_slot: Option<usize>,
        keep_len: usize,
        expected_prefix_hash: u64,
        model: &dyn Model,
    ) -> Result<(), RollbackFallback> {
        if let Some(slot) = ssm_slot {
            if let Some(proof) = self.reset_proof.as_ref()
                && (!proof.verify_restore(slot, expected_prefix_hash)
                    || !proof.verify_restore_at(slot, keep_len, expected_prefix_hash))
            {
                tracing::error!(
                    ring_slot = slot,
                    keep_len,
                    expected_prefix_hash,
                    "Reset proof: snapshot verification failed on rollback restore; declining rollback"
                );
                return Err(RollbackFallback::NoSsmSnapshot);
            }
            if let Err(e) = model.restore_decode_ssm_snapshot(&self.seq, slot) {
                tracing::error!(
                    error = %e,
                    ring_slot = slot,
                    keep_len,
                    "SSM decode-snapshot restore failed; declining rollback"
                );
                return Err(RollbackFallback::NoSsmSnapshot);
            }
            if let Some(proof) = &mut self.reset_proof {
                proof.truncate_after(keep_len);
            }
            self.ssm_rollback_ring.truncate_after(keep_len);
        }
        self.prefix_hash = expected_prefix_hash;
        Ok(())
    }

    /// Complete sequence and release GPU resources through the record facade.
    /// Marks `RecordState::Released` and `release_metadata_consistent = true` only if `free_sequence` succeeds.
    pub fn finish_or_release(
        &mut self,
        model: &dyn Model,
        cache_sequence: bool,
    ) -> anyhow::Result<()> {
        if cache_sequence {
            model.cache_sequence(&self.seq);
        }
        let res = model.free_sequence(&mut self.seq);
        if res.is_ok() {
            self.state = RecordState::Released;
            if let Some(proof) = &mut self.reset_proof {
                proof.release_metadata_consistent = true;
            }
        }
        res
    }

    /// Alias for [`Self::finish_or_release`].
    #[inline]
    pub fn release(&mut self, model: &dyn Model, cache_sequence: bool) -> anyhow::Result<()> {
        self.finish_or_release(model, cache_sequence)
    }
}

/// Borrowed reference view of a continuation record for delegation from `ActiveSeq`.
pub struct ContinuationRecordRef<'a> {
    pub seq: &'a SequenceState,
    pub ssm_rollback_ring: &'a SsmDecodeRing,
    pub session_hash: u64,
    pub prompt_prefix_hash: u64,
    pub prefix_hash: u64,
    pub state: RecordState,
    pub reset_proof: Option<&'a ResetProofPayload>,
}

/// Mutable borrowed view of a continuation record for delegation from `ActiveSeq`.
pub struct ContinuationRecordMut<'a> {
    pub seq: &'a mut SequenceState,
    pub ssm_rollback_ring: &'a mut SsmDecodeRing,
    pub session_hash: u64,
    pub prompt_prefix_hash: u64,
    pub prefix_hash: &'a mut u64,
    pub state: &'a mut RecordState,
    pub reset_proof: &'a mut Option<ResetProofPayload>,
}

impl<'a> ContinuationRecordMut<'a> {
    #[inline]
    pub fn update_prefix_hash(&mut self, token: u32) {
        *self.prefix_hash = hash_token_incremental(*self.prefix_hash, token);
    }

    pub fn snapshot_boundary_if_ssm(
        &mut self,
        output_tokens: &[u32],
        model: &dyn Model,
    ) -> Option<usize> {
        if !model.has_ssm_layers() || !self.ssm_rollback_ring.is_enabled() {
            return None;
        }
        let mask = super::helpers::boundary_token_mask()?;
        let &last = output_tokens.last()?;
        let id = last as usize;
        if id >= mask.len() || !mask[id] {
            return None;
        }
        let token_position = output_tokens.len();
        self.record_and_save_boundary_snapshot(token_position, model)
    }

    /// Record an SSM boundary snapshot at token_position and save via model.
    pub fn record_and_save_boundary_snapshot(
        &mut self,
        token_position: usize,
        model: &dyn Model,
    ) -> Option<usize> {
        if !model.has_ssm_layers() || !self.ssm_rollback_ring.is_enabled() {
            return None;
        }
        let slot = self.ssm_rollback_ring.record(token_position)?;
        if let Err(e) = model.save_decode_ssm_snapshot(self.seq, slot) {
            tracing::warn!(
                error = %e,
                ring_slot = slot,
                token_position,
                "SSM decode-snapshot save failed; dropping ring entry"
            );
            self.ssm_rollback_ring
                .truncate_after(token_position.saturating_sub(1));
            return None;
        }
        if let Some(proof) = self.reset_proof.as_mut() {
            proof.record_snapshot(token_position, *self.prefix_hash, slot);
        }
        *self.state = RecordState::Snapshotted;
        Some(slot)
    }

    /// Restore SSM state for rollback if hybrid model, verifying reset proof against
    /// `expected_prefix_hash` and `keep_len` when armed, and restoring `prefix_hash`.
    pub fn restore_for_rollback(
        &mut self,
        ssm_slot: Option<usize>,
        keep_len: usize,
        expected_prefix_hash: u64,
        model: &dyn Model,
    ) -> Result<(), RollbackFallback> {
        if let Some(slot) = ssm_slot {
            if let Some(proof) = self.reset_proof.as_ref()
                && (!proof.verify_restore(slot, expected_prefix_hash)
                    || !proof.verify_restore_at(slot, keep_len, expected_prefix_hash))
            {
                tracing::error!(
                    ring_slot = slot,
                    keep_len,
                    expected_prefix_hash,
                    "Reset proof: snapshot verification failed on rollback restore; declining rollback"
                );
                return Err(RollbackFallback::NoSsmSnapshot);
            }
            if let Err(e) = model.restore_decode_ssm_snapshot(self.seq, slot) {
                tracing::error!(
                    error = %e,
                    ring_slot = slot,
                    keep_len,
                    "SSM decode-snapshot restore failed; declining rollback"
                );
                return Err(RollbackFallback::NoSsmSnapshot);
            }
            if let Some(proof) = self.reset_proof.as_mut() {
                proof.truncate_after(keep_len);
            }
            self.ssm_rollback_ring.truncate_after(keep_len);
        }
        *self.prefix_hash = expected_prefix_hash;
        Ok(())
    }

    /// Complete sequence and release GPU resources through the record facade.
    /// Marks `RecordState::Released` and `release_metadata_consistent = true` only if `free_sequence` succeeds.
    pub fn finish_or_release(
        &mut self,
        model: &dyn Model,
        cache_sequence: bool,
    ) -> anyhow::Result<()> {
        if cache_sequence {
            model.cache_sequence(self.seq);
        }
        let res = model.free_sequence(self.seq);
        if res.is_ok() {
            *self.state = RecordState::Released;
            if let Some(proof) = self.reset_proof.as_mut() {
                proof.release_metadata_consistent = true;
            }
        }
        res
    }

    /// Alias for [`Self::finish_or_release`].
    #[inline]
    pub fn release(&mut self, model: &dyn Model, cache_sequence: bool) -> anyhow::Result<()> {
        self.finish_or_release(model, cache_sequence)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::DevicePtr;
    use std::time::Instant;

    #[test]
    fn test_deterministic_prefix_hash() {
        let tokens = [101u32, 2054, 2003, 1037, 3231, 102];
        let h1 = compute_prefix_hash(&tokens, None);
        let h2 = compute_prefix_hash(&tokens, None);
        assert_eq!(
            h1, h2,
            "Prefix hash must be strictly deterministic across calls"
        );

        // Incremental equivalence
        let mut running = CANONICAL_PREFIX_SEED;
        for &tok in &tokens {
            running = hash_token_incremental(running, tok);
        }
        assert_eq!(
            h1, running,
            "Batch prefix hash must match incremental token folding"
        );
    }

    #[test]
    fn test_prefix_hash_multimodal_differentiation() {
        let tokens = [1u32, 2, 3];
        let h_text = compute_prefix_hash(&tokens, None);
        let pixels_a = [10u8, 20, 30, 40];
        let pixels_b = [10u8, 20, 30, 41];
        let h_img_a = compute_prefix_hash(&tokens, Some(&pixels_a));
        let h_img_b = compute_prefix_hash(&tokens, Some(&pixels_b));

        assert_ne!(h_text, h_img_a);
        assert_ne!(h_img_a, h_img_b);
    }

    #[test]
    fn test_reset_proof_snapshot_and_verify() {
        let mut proof = ResetProofPayload::new(3, 0x1234);
        assert_eq!(proof.scope, PROOF_SCOPE_CPU_METADATA);
        assert_eq!(proof.claimed_slot_idx, 3);
        assert!(proof.claim_metadata_consistent);
        assert!(!proof.release_metadata_consistent);
        assert!(proof.verify_claim().is_ok());

        proof.record_snapshot(10, 0xAAAA, 1);
        proof.record_snapshot(20, 0xBBBB, 2);
        proof.record_snapshot(30, 0xCCCC, 0);

        assert!(proof.verify_restore(1, 0xAAAA));
        assert!(proof.verify_restore_at(1, 10, 0xAAAA));
        assert!(!proof.verify_restore_at(1, 11, 0xAAAA));
        assert!(proof.verify_restore(2, 0xBBBB));
        assert!(!proof.verify_restore(2, 0x9999));
        assert!(!proof.verify_restore(5, 0xAAAA));

        proof.truncate_after(15);
        assert_eq!(proof.snapshots.len(), 1);
        assert!(proof.verify_restore(1, 0xAAAA));
        assert!(!proof.verify_restore(2, 0xBBBB));
    }

    #[test]
    fn test_record_state_lifecycle() {
        assert_eq!(RecordState::default(), RecordState::Fresh);
        let mut s = RecordState::Fresh;
        assert_eq!(s, RecordState::Fresh);
        s = RecordState::Live;
        assert_eq!(s, RecordState::Live);
        s = RecordState::Snapshotted;
        assert_eq!(s, RecordState::Snapshotted);
        s = RecordState::Released;
        assert_eq!(s, RecordState::Released);
    }

    #[test]
    fn test_continuation_record_rollback_restores_prefix_hash() {
        let model = crate::scheduler::test_fixtures::model::recording();
        let prompt = [101u32, 102, 103];
        let seq = model.alloc_sequence().expect("clean alloc_sequence");
        let mut record = ContinuationRecord::from_prompt(1, seq, 4, &prompt, None);
        let prompt_hash = compute_prefix_hash(&prompt, None);
        assert_eq!(record.prefix_hash(), prompt_hash);
        assert_eq!(record.prompt_prefix_hash(), prompt_hash);

        // Advance 5 tokens to a boundary
        let mut output_tokens = Vec::new();
        for tok in 201..=205 {
            output_tokens.push(tok);
            record.update_prefix_hash(tok);
        }
        let boundary_pos = output_tokens.len();
        let boundary_hash = record.prefix_hash();
        assert_eq!(
            boundary_hash,
            recompute_prefix_hash_from_prompt(record.prompt_prefix_hash(), &output_tokens)
        );

        // Record snapshot at boundary
        let slot = record.ssm_rollback_ring.record(boundary_pos).expect("slot");
        if let Some(proof) = record.reset_proof.as_mut() {
            proof.record_snapshot(boundary_pos, boundary_hash, slot);
        }

        // Generate 3 more tokens (dropped tokens on rollback)
        for tok in 301..=303 {
            output_tokens.push(tok);
            record.update_prefix_hash(tok);
        }
        assert_ne!(
            record.prefix_hash(),
            boundary_hash,
            "Hash advanced past boundary"
        );

        // Roll back to boundary: keep_len = 5, dropped = 3
        let target_hash = recompute_prefix_hash_from_prompt(
            record.prompt_prefix_hash(),
            &output_tokens[..boundary_pos],
        );
        let res = record.restore_for_rollback(Some(slot), boundary_pos, target_hash, &model);
        assert!(res.is_ok(), "Rollback restore must succeed");
        assert_eq!(
            record.prefix_hash(),
            boundary_hash,
            "Rollback must restore prefix_hash to boundary snapshot prefix hash"
        );

        // Subsequent token generation increments from the restored boundary prefix hash
        record.update_prefix_hash(401);
        assert_eq!(
            record.prefix_hash(),
            hash_token_incremental(boundary_hash, 401),
            "Next token must fold into the restored boundary hash"
        );
    }

    #[test]
    fn test_continuation_record_mut_rollback_restores_prefix_hash() {
        let model = crate::scheduler::test_fixtures::model::recording();
        let sink = crate::scheduler::types::ResponseSink::Blocking(None);
        let mut active = crate::scheduler::test_fixtures::active_seq(sink, 2);
        active.ssm_rollback_ring = SsmDecodeRing::new(4);
        active.prompt_prefix_hash = 0x1234_5678;
        active.prefix_hash = hash_token_incremental(active.prompt_prefix_hash, 31);
        active.reset_proof = Some(ResetProofPayload::new(2, active.prompt_prefix_hash));

        // Advance to boundary (output_tokens starts with [31], add 10 and 20 -> len 3)
        active.output_tokens.push(10);
        active.update_prefix_hash(10);
        active.output_tokens.push(20);
        active.update_prefix_hash(20);
        let keep_len = active.output_tokens.len();
        let boundary_hash = active.prefix_hash();
        let slot = active.ssm_rollback_ring.record(keep_len).expect("slot");
        if let Some(proof) = active.reset_proof.as_mut() {
            proof.record_snapshot(keep_len, boundary_hash, slot);
        }

        // Emit more tokens
        active.output_tokens.push(30);
        active.update_prefix_hash(30);
        active.output_tokens.push(40);
        active.update_prefix_hash(40);
        assert_ne!(active.prefix_hash, boundary_hash);

        // Roll back via ContinuationRecordMut
        let target_hash = recompute_prefix_hash_from_prompt(
            active.prompt_prefix_hash,
            &active.output_tokens[..keep_len],
        );
        let res = active.continuation_record_mut().restore_for_rollback(
            Some(slot),
            keep_len,
            target_hash,
            &model,
        );
        assert!(res.is_ok());
        assert_eq!(
            active.prefix_hash, boundary_hash,
            "ActiveSeq prefix_hash must be restored to snapshot boundary hash through ContinuationRecordMut"
        );
    }

    #[test]
    fn test_serial_vs_speculative_eos_pop_prefix_hash_parity() {
        let m = crate::scheduler::test_fixtures::model::recording();
        let prompt_hash = compute_prefix_hash(&[10, 20, 30], None);
        let initial_hash = hash_token_incremental(prompt_hash, 31);

        // Case 1: Suppressed EOS (require_tool_call = true) followed by committed content token.
        let mut spec_seq = crate::scheduler::test_fixtures::active_seq(
            crate::scheduler::types::ResponseSink::Blocking(None),
            0,
        );
        spec_seq.prompt_prefix_hash = prompt_hash;
        spec_seq.prefix_hash = initial_hash;
        spec_seq.require_tool_call = true;

        let mut serial_seq = crate::scheduler::test_fixtures::active_seq(
            crate::scheduler::types::ResponseSink::Blocking(None),
            0,
        );
        serial_seq.prompt_prefix_hash = prompt_hash;
        serial_seq.prefix_hash = initial_hash;
        serial_seq.require_tool_call = true;
        let mut serial_active = vec![serial_seq];

        // Sample EOS (token 17) while suppressed: speculative stages then pops, serial skips before push.
        crate::scheduler::emit_step::emit_token_with_eos_policy(&mut spec_seq, 17, None, false);
        let mut logits = vec![-10.0f32; 32];
        logits[17] = 10.0;
        *m.host_logits.lock().unwrap() = logits;
        crate::scheduler::decode_logits_step::process_decode_logits(
            &m,
            &mut serial_active,
            DevicePtr::NULL,
            Instant::now(),
            None,
            None,
            None,
            None,
            None,
            &[],
            false,
            false,
        );

        assert_eq!(spec_seq.output_tokens, serial_active[0].output_tokens);
        assert_eq!(spec_seq.output_tokens, vec![31]);
        assert_eq!(
            spec_seq.prefix_hash, serial_active[0].prefix_hash,
            "Suppressed EOS must leave speculative and serial prefix_hash identical"
        );
        assert_eq!(spec_seq.prefix_hash, initial_hash);

        // Next sample is a committed content token (25): both paths must fold onto the exact same hash.
        crate::scheduler::emit_step::emit_token_with_eos_policy(&mut spec_seq, 25, None, false);
        let mut logits = vec![-10.0f32; 32];
        logits[25] = 10.0;
        *m.host_logits.lock().unwrap() = logits;
        crate::scheduler::decode_logits_step::process_decode_logits(
            &m,
            &mut serial_active,
            DevicePtr::NULL,
            Instant::now(),
            None,
            None,
            None,
            None,
            None,
            &[],
            false,
            false,
        );
        assert_eq!(spec_seq.output_tokens, serial_active[0].output_tokens);
        assert_eq!(spec_seq.output_tokens, vec![31, 25]);
        assert_eq!(spec_seq.prefix_hash, serial_active[0].prefix_hash);
        assert_eq!(
            spec_seq.prefix_hash,
            recompute_prefix_hash_from_prompt(prompt_hash, &[31, 25])
        );

        // Case 2: Implicit-close EOS inside thinking (honor_eos_inside_thinking = true) followed by content + terminal EOS.
        let mut spec_think = crate::scheduler::test_fixtures::active_seq(
            crate::scheduler::types::ResponseSink::Blocking(None),
            1,
        );
        spec_think.prompt_prefix_hash = prompt_hash;
        spec_think.prefix_hash = initial_hash;
        spec_think.inside_thinking = true;
        spec_think.enable_thinking = true;
        spec_think.think_ended = false;

        let mut serial_think = crate::scheduler::test_fixtures::active_seq(
            crate::scheduler::types::ResponseSink::Blocking(None),
            1,
        );
        serial_think.prompt_prefix_hash = prompt_hash;
        serial_think.prefix_hash = initial_hash;
        serial_think.inside_thinking = true;
        serial_think.enable_thinking = true;
        serial_think.think_ended = false;
        let mut serial_think_active = vec![serial_think];

        // First EOS (17) implicitly closes thinking and is popped/discarded.
        crate::scheduler::emit_step::emit_token_with_eos_policy(&mut spec_think, 17, None, true);
        let mut logits = vec![-10.0f32; 32];
        logits[17] = 10.0;
        *m.host_logits.lock().unwrap() = logits;
        crate::scheduler::decode_logits_step::process_decode_logits(
            &m,
            &mut serial_think_active,
            DevicePtr::NULL,
            Instant::now(),
            None,
            None,
            None,
            None,
            None,
            &[],
            false,
            true,
        );
        assert!(!spec_think.inside_thinking && !serial_think_active[0].inside_thinking);
        assert_eq!(
            spec_think.output_tokens,
            serial_think_active[0].output_tokens
        );
        assert_eq!(
            spec_think.prefix_hash, serial_think_active[0].prefix_hash,
            "Implicit-close EOS in thinking must leave speculative and serial prefix_hash identical"
        );
        assert_eq!(spec_think.prefix_hash, initial_hash);

        // Second EOS (17) after thinking closed is a terminal EOS and is committed on both paths.
        crate::scheduler::emit_step::emit_token_with_eos_policy(&mut spec_think, 17, None, true);
        crate::scheduler::decode_logits_step::process_decode_logits(
            &m,
            &mut serial_think_active,
            DevicePtr::NULL,
            Instant::now(),
            None,
            None,
            None,
            None,
            None,
            &[],
            false,
            true,
        );
        assert!(spec_think.finished && serial_think_active[0].finished);
        assert_eq!(
            spec_think.output_tokens,
            serial_think_active[0].output_tokens
        );
        assert_eq!(spec_think.output_tokens, vec![31, 17]);
        assert_eq!(spec_think.prefix_hash, serial_think_active[0].prefix_hash);
        assert_eq!(
            spec_think.prefix_hash,
            recompute_prefix_hash_from_prompt(prompt_hash, &[31, 17])
        );
    }

    #[test]
    fn test_multimodal_rollback_preserves_image_digest_across_proof_and_ring_modes() {
        let model = crate::scheduler::test_fixtures::model::recording();
        let prompt = vec![101u32, 102, 103, 104];
        let image_pixels = vec![(vec![0.125f32, 0.375, 0.625, 0.875], 2usize, 2usize)];
        let mm_prompt_hash = compute_prefix_hash_with_images(&prompt, &image_pixels);
        let text_only_prompt_hash = compute_prefix_hash(&prompt, None);
        assert_ne!(
            mm_prompt_hash, text_only_prompt_hash,
            "Multimodal prompt hash must differ from text-only prompt hash"
        );

        let keep_tokens = [201u32, 202, 203];
        let dropped_tokens = [301u32, 302];
        let keep_len = keep_tokens.len();
        let expected_boundary_hash =
            recompute_prefix_hash_from_prompt(mm_prompt_hash, &keep_tokens);

        for (proof_enabled, ring_enabled) in
            [(true, true), (false, true), (true, false), (false, false)]
        {
            let sink = crate::scheduler::types::ResponseSink::Blocking(None);
            let mut active = crate::scheduler::test_fixtures::active_seq(sink, 1);
            active.seq.tokens = prompt.clone();
            active.seq.seq_len = prompt.len();
            active.prompt_prefix_hash = mm_prompt_hash;
            active.prefix_hash = mm_prompt_hash;
            active.output_tokens.clear();
            active.ssm_rollback_ring = SsmDecodeRing::new(if ring_enabled { 4 } else { 0 });
            active.reset_proof = proof_enabled.then(|| ResetProofPayload::new(1, mm_prompt_hash));

            for &tok in &keep_tokens {
                active.seq.tokens.push(tok);
                active.seq.seq_len += 1;
                active.output_tokens.push(tok);
                active.update_prefix_hash(tok);
            }
            assert_eq!(active.prefix_hash, expected_boundary_hash);

            let ssm_slot = if ring_enabled {
                let slot = active
                    .ssm_rollback_ring
                    .record(keep_len)
                    .expect("ring slot");
                if let Some(proof) = active.reset_proof.as_mut() {
                    proof.record_snapshot(keep_len, active.prefix_hash, slot);
                }
                Some(slot)
            } else {
                None
            };

            for &tok in &dropped_tokens {
                active.seq.tokens.push(tok);
                active.seq.seq_len += 1;
                active.output_tokens.push(tok);
                active.update_prefix_hash(tok);
            }
            assert_ne!(active.prefix_hash, expected_boundary_hash);

            let target_hash = recompute_prefix_hash_from_prompt(
                active.prompt_prefix_hash,
                &active.output_tokens[..keep_len],
            );
            active
                .continuation_record_mut()
                .restore_for_rollback(ssm_slot, keep_len, target_hash, &model)
                .expect("restore_for_rollback must succeed");
            crate::scheduler::rollback::apply_rollback(&mut active, keep_len, dropped_tokens.len());

            assert_eq!(
                active.prefix_hash, expected_boundary_hash,
                "Multimodal prefix_hash must be preserved after rollback (proof={proof_enabled}, ring={ring_enabled})"
            );
            assert_ne!(
                active.prefix_hash,
                compute_prefix_hash(&active.seq.tokens, None),
                "Rollback must never drop the multimodal image digest (proof={proof_enabled}, ring={ring_enabled})"
            );
        }
    }

    #[test]
    fn test_verify_restore_rejects_corrupted_hash_and_position_mismatch_on_rollback() {
        let model = crate::scheduler::test_fixtures::model::recording();
        let prompt = [1u32, 2, 3];
        let seq = model.alloc_sequence().expect("clean alloc_sequence");
        let mut record = ContinuationRecord::from_prompt(7, seq, 4, &prompt, None);
        for tok in [10u32, 20, 30] {
            record.update_prefix_hash(tok);
        }
        let boundary_hash = record.prefix_hash();
        let slot = record.ssm_rollback_ring.record(3).expect("slot");
        record
            .reset_proof
            .as_mut()
            .expect("reset_proof armed")
            .record_snapshot(3, boundary_hash, slot);

        record.update_prefix_hash(40);
        let post_boundary_hash = record.prefix_hash();

        // Corrupted target hash must be rejected by restore_for_rollback
        let bad_hash = boundary_hash ^ 0xDEAD_BEEF;
        let err = record
            .restore_for_rollback(Some(slot), 3, bad_hash, &model)
            .expect_err("corrupted target prefix_hash must be rejected");
        assert_eq!(err, RollbackFallback::NoSsmSnapshot);
        assert_eq!(
            record.prefix_hash(),
            post_boundary_hash,
            "Failed restore must not overwrite prefix_hash with corrupted target"
        );

        // Mismatched position (keep_len = 2 instead of 3) must also be rejected
        let pos_err = record
            .restore_for_rollback(Some(slot), 2, boundary_hash, &model)
            .expect_err("position mismatch must be rejected");
        assert_eq!(pos_err, RollbackFallback::NoSsmSnapshot);

        // Matching position and hash succeeds
        record
            .restore_for_rollback(Some(slot), 3, boundary_hash, &model)
            .expect("matching snapshot must succeed");
        assert_eq!(record.prefix_hash(), boundary_hash);
    }

    #[test]
    fn test_release_does_not_mark_released_or_verified_when_free_sequence_fails() {
        let model = crate::scheduler::test_fixtures::model::recording();
        *model.free_error.lock().unwrap() = Some("injected free_sequence error".to_string());

        let seq = model.alloc_sequence().expect("clean alloc_sequence");
        let mut record = ContinuationRecord::from_prompt(9, seq, 2, &[1, 2, 3], None);
        assert_eq!(record.state(), RecordState::Live);
        assert!(
            !record
                .reset_proof
                .as_ref()
                .expect("proof armed")
                .release_metadata_consistent
        );

        // Failing free_sequence on ContinuationRecord must leave state Live and release_metadata_consistent false
        let err = record.release(&model, false);
        assert!(err.is_err(), "release must propagate free_sequence error");
        assert_eq!(
            record.state(),
            RecordState::Live,
            "RecordState must not transition to Released when free_sequence fails"
        );
        assert!(
            !record
                .reset_proof
                .as_ref()
                .expect("proof armed")
                .release_metadata_consistent,
            "release_metadata_consistent must remain false when free_sequence fails"
        );

        // Failing free_sequence on ContinuationRecordMut must also leave state Live and release_metadata_consistent false
        let sink = crate::scheduler::types::ResponseSink::Blocking(None);
        let mut active = crate::scheduler::test_fixtures::active_seq(sink, 3);
        active.reset_proof = Some(ResetProofPayload::new(3, 0xABCD));
        let mut_err = active.continuation_record_mut().release(&model, false);
        assert!(mut_err.is_err());
        assert_eq!(active.record_state, RecordState::Live);
        assert!(
            !active
                .reset_proof
                .as_ref()
                .unwrap()
                .release_metadata_consistent
        );

        // Clearing free_error allows release to succeed and flip state + release_metadata_consistent
        *model.free_error.lock().unwrap() = None;
        record.release(&model, true).expect("release succeeds");
        assert_eq!(record.state(), RecordState::Released);
        assert!(
            record
                .reset_proof
                .as_ref()
                .unwrap()
                .release_metadata_consistent
        );

        active
            .continuation_record_mut()
            .release(&model, false)
            .expect("mut release succeeds");
        assert_eq!(active.record_state, RecordState::Released);
        assert!(
            active
                .reset_proof
                .as_ref()
                .unwrap()
                .release_metadata_consistent
        );
    }

    #[test]
    fn test_reset_proof_claim_and_restored_sequence_verification() {
        let model = crate::scheduler::test_fixtures::model::recording();
        let (mut clean_seq, mut proof) =
            ContinuationRecord::claim_sequence(&model, 0x5555).expect("claim_sequence");
        let proof_ref = proof.as_ref().expect("proof armed");
        assert!(
            proof_ref.claim_metadata_consistent,
            "Freshly allocated sequence must verify clean"
        );
        assert_eq!(proof_ref.claimed_slot_idx, clean_seq.slot_idx);
        assert!(!proof_ref.release_metadata_consistent);
        assert!(proof_ref.verify_claim().is_ok());

        // Dirty sequence (non-zero seq_len or non-empty block_table) is flagged by from_claimed_seq
        let mut dirty_seq = crate::scheduler::test_fixtures::model::seq_state();
        dirty_seq.seq_len = 9;
        let dirty_proof = ResetProofPayload::from_claimed_seq(&dirty_seq, 0x5555);
        assert!(
            !dirty_proof.claim_metadata_consistent,
            "Dirty SequenceState with non-zero seq_len must set claim_metadata_consistent = false"
        );
        assert!(dirty_proof.verify_claim().is_err());

        // Restored sequence from swap preserves actual slot_idx (not hardcoded 0)
        let mut restored_seq = crate::scheduler::test_fixtures::model::seq_state();
        restored_seq.slot_idx = 6;
        restored_seq.tokens = vec![10, 20, 30];
        restored_seq.seq_len = 3;
        let restored_proof = ResetProofPayload::from_restored_seq(&restored_seq, 0x7777);
        assert!(restored_proof.claim_metadata_consistent);
        assert!(restored_proof.verify_claim().is_ok());
        assert_eq!(
            restored_proof.claimed_slot_idx, 6,
            "Restored sequence proof must record actual slot_idx, not hardcoded 0"
        );

        // abort_prefill_sequence respects free_error and only sets release_metadata_consistent on Ok
        *model.free_error.lock().unwrap() = Some("abort free failed".to_string());
        assert!(
            ContinuationRecord::abort_prefill_sequence(&model, &mut clean_seq, &mut proof).is_err()
        );
        assert!(!proof.as_ref().unwrap().release_metadata_consistent);

        *model.free_error.lock().unwrap() = None;
        ContinuationRecord::abort_prefill_sequence(&model, &mut clean_seq, &mut proof)
            .expect("abort_prefill_sequence succeeds");
        assert!(proof.as_ref().unwrap().release_metadata_consistent);
    }

    #[test]
    fn test_release_canary_runtime_refusal_on_dirty_claim_and_restored_seq() {
        use std::sync::atomic::Ordering;
        let model = crate::scheduler::test_fixtures::model::recording();

        // Inject a dirty sequence on alloc_sequence (seq_len = 9).
        model.dirty_alloc_seq_len.store(9, Ordering::Relaxed);

        // With proof enabled (simulating FLUX_CONTINUATION_RESET_PROOF=1 in release),
        // claim_sequence_with_proof_mode and claim_sequence_on_with_proof_mode MUST refuse
        // at runtime (not via debug_assert!) and free the dirty slot so it is not leaked.
        let err = ContinuationRecord::claim_sequence_with_proof_mode(&model, 0x1111, true)
            .err()
            .expect("dirty SequenceState must be refused at runtime when proof is enabled");
        assert!(
            err.to_string().contains("scope=cpu-metadata")
                && err.to_string().contains("claim_metadata_consistent=false"),
            "unexpected error: {err:#}"
        );
        assert_eq!(
            *model.frees.lock().unwrap(),
            vec![0],
            "dirty sequence slot must be freed before returning Err"
        );

        let err_on = ContinuationRecord::claim_sequence_on_with_proof_mode(&model, 7, 0x2222, true)
            .err()
            .expect("dirty SequenceState on stream must be refused at runtime");
        assert!(err_on.to_string().contains("scope=cpu-metadata"));
        assert_eq!(
            *model.frees.lock().unwrap(),
            vec![0, 0],
            "dirty sequence slot on stream must also be freed before returning Err"
        );

        // With proof disabled (default release mode without FLUX_CONTINUATION_RESET_PROOF),
        // claim_sequence_with_proof_mode succeeds without allocating a proof payload.
        let (seq_no_proof, proof_none) =
            ContinuationRecord::claim_sequence_with_proof_mode(&model, 0x3333, false)
                .expect("proof-disabled mode does not allocate ResetProofPayload");
        assert!(proof_none.is_none());
        assert_eq!(seq_no_proof.seq_len, 9);

        // Restore clean allocation and verify try_from_prompt returns ownership of dirty SequenceState
        // alongside the error so the caller can reclaim it without leaking.
        model.dirty_alloc_seq_len.store(0, Ordering::Relaxed);
        let mut dirty_seq = model.alloc_sequence().expect("clean alloc");
        dirty_seq.slot_idx = 5;
        dirty_seq.block_table.push(42);
        let (mut returned_seq, try_err) =
            match ContinuationRecord::try_from_prompt(11, dirty_seq, 2, &[1, 2], None) {
                Ok(_) => panic!("try_from_prompt must refuse non-empty block_table at runtime"),
                Err(pair) => pair,
            };
        assert!(
            try_err
                .to_string()
                .contains("claim_metadata_consistent=false")
        );
        assert_eq!(returned_seq.slot_idx, 5);
        assert_eq!(returned_seq.block_table, vec![42]);
        ContinuationRecord::release_raw_sequence(&model, &mut returned_seq)
            .expect("caller reclaims returned SequenceState");
        assert_eq!(*model.frees.lock().unwrap(), vec![0, 0, 5]);

        // Restored sequence with invalid slot_idx (usize::MAX) or mismatched seq_len is refused by verify_claim().
        let mut bad_restored = model.alloc_sequence().expect("clean alloc");
        bad_restored.slot_idx = usize::MAX;
        let bad_slot_proof = ResetProofPayload::from_restored_seq(&bad_restored, 0x4444);
        assert!(!bad_slot_proof.claim_metadata_consistent);
        assert!(bad_slot_proof.verify_claim().is_err());

        let mut mismatch_restored = model.alloc_sequence().expect("clean alloc");
        mismatch_restored.slot_idx = 4;
        mismatch_restored.tokens = vec![1, 2, 3];
        mismatch_restored.seq_len = 2;
        let mismatch_proof = ResetProofPayload::from_restored_seq(&mismatch_restored, 0x4444);
        assert!(!mismatch_proof.claim_metadata_consistent);
        assert!(mismatch_proof.verify_claim().is_err());

        // When BOTH dirty-claim verification fails AND model.free_sequence fails,
        // the SequenceState must be retained in ownership-preserving quarantine (never dropped)
        // and permanent fail-stop must block any subsequent claims on that model instance
        // (production recovery is process restart; no in-process retry of a potentially partial free).
        let q_model: &'static _ =
            Box::leak(Box::new(crate::scheduler::test_fixtures::model::recording()));
        q_model.dirty_alloc_seq_len.store(9, Ordering::Relaxed);
        *q_model.free_error.lock().unwrap() = Some("injected reclaim free failure".to_string());
        assert!(!is_model_quarantined(q_model));

        let combined_err =
            ContinuationRecord::claim_sequence_with_proof_mode(q_model, 0x1111, true)
                .err()
                .expect("dirty claim + free error must fail and quarantine");
        assert!(
            combined_err
                .to_string()
                .contains("claim_metadata_consistent=false")
                && combined_err
                    .to_string()
                    .contains("additionally failed to reclaim dirty sequence slot 0")
                && combined_err
                    .to_string()
                    .contains("injected reclaim free failure")
                && combined_err
                    .to_string()
                    .contains("SequenceState quarantined; fail-stop active"),
            "unexpected combined error: {combined_err:#}"
        );
        assert!(is_model_quarantined(q_model));
        assert_eq!(quarantined_count_for_model(q_model), 1);
        assert_eq!(quarantined_summaries_for_model(q_model)[0].0, 0);
        assert!(
            q_model.frees.lock().unwrap().is_empty(),
            "failed free_sequence must not record slot as reclaimed"
        );

        // Even if the underlying model free_error is later cleared, the model remains permanently
        // fail-stopped (process restart required) and refuses any subsequent claim before allocating.
        *q_model.free_error.lock().unwrap() = None;
        let fail_stop_err =
            ContinuationRecord::claim_sequence_on_with_proof_mode(q_model, 7, 0x2222, true)
                .err()
                .expect("subsequent claim while quarantined must permanently fail-stop");
        assert!(
            fail_stop_err.to_string().contains(
                "ContinuationRecord fail-stop active: 1 unreclaimed sequence(s) quarantined"
            ) && fail_stop_err
                .to_string()
                .contains("process restart required for production recovery"),
            "unexpected fail-stop error: {fail_stop_err:#}"
        );
        assert_eq!(
            quarantined_count_for_model(q_model),
            1,
            "fail-stop refusal must not allocate or quarantine a second sequence"
        );

        // Also verify stream claim_sequence_on_with_proof_mode quarantine path on a fresh model.
        let q_stream_model: &'static _ =
            Box::leak(Box::new(crate::scheduler::test_fixtures::model::recording()));
        q_stream_model
            .dirty_alloc_seq_len
            .store(9, Ordering::Relaxed);
        *q_stream_model.free_error.lock().unwrap() =
            Some("stream reclaim free failure".to_string());
        let combined_err_on =
            ContinuationRecord::claim_sequence_on_with_proof_mode(q_stream_model, 7, 0x2222, true)
                .err()
                .expect("dirty claim_on + free error must fail and quarantine");
        assert!(
            combined_err_on
                .to_string()
                .contains("additionally failed to reclaim dirty sequence slot 0")
                && combined_err_on
                    .to_string()
                    .contains("SequenceState quarantined; fail-stop active"),
            "unexpected combined error on stream: {combined_err_on:#}"
        );
        assert!(is_model_quarantined(q_stream_model));
        assert_eq!(quarantined_count_for_model(q_stream_model), 1);
    }

    #[test]
    fn test_swap_resume_enforces_restored_proof_and_surfaces_reclaim_failure() {
        let m: &'static _ =
            Box::leak(Box::new(crate::scheduler::test_fixtures::model::recording()));
        let dir = tempfile::tempdir().expect("spill tempdir");
        let mut spill = crate::scheduler::KvSpillManager::new(dir.path().join("swap"), 1 << 20)
            .expect("spill manager");

        // 1. Clean swap_out_sequence -> resume_swapped_seq round-trip.
        let mut clean_active = crate::scheduler::test_fixtures::active_seq(
            crate::scheduler::types::ResponseSink::Blocking(None),
            2,
        );
        clean_active.seq.tokens = vec![3, 5, 8];
        clean_active.seq.seq_len = 3;
        let mut active = vec![clean_active];
        let swapped = crate::scheduler::lifecycle::swap_out_sequence(m, &mut active, 0, &mut spill)
            .expect("swap out clean sequence");
        assert_eq!(*m.frees.lock().unwrap(), vec![2]);

        let resumed =
            crate::scheduler::lifecycle::resume_swapped_seq(None, None, m, swapped, &mut spill)
                .expect("clean swap resume must succeed");
        let resumed_proof = resumed.reset_proof.as_ref().expect("proof armed on resume");
        assert!(resumed_proof.claim_metadata_consistent);
        assert_eq!(resumed_proof.claimed_slot_idx, resumed.seq.slot_idx);

        // 2. Inconsistent SwappedSeq (seq_len != tokens.len()) is refused by resume_swapped_seq
        // and its newly claimed slot is reclaimed via finish_or_release.
        m.frees.lock().unwrap().clear();
        let mut active_bad = vec![crate::scheduler::test_fixtures::active_seq(
            crate::scheduler::types::ResponseSink::Blocking(None),
            4,
        )];
        let mut swapped_bad =
            crate::scheduler::lifecycle::swap_out_sequence(m, &mut active_bad, 0, &mut spill)
                .expect("swap out second sequence");
        assert_eq!(*m.frees.lock().unwrap(), vec![4]);
        m.frees.lock().unwrap().clear();

        // Corrupt swapped metadata so seq_len != tokens.len().
        swapped_bad.seq_len = swapped_bad.tokens.len() + 5;
        let resume_err =
            crate::scheduler::lifecycle::resume_swapped_seq(None, None, m, swapped_bad, &mut spill)
                .err()
                .expect("inconsistent SwappedSeq must be refused on resume_swapped_seq");
        assert!(
            resume_err
                .to_string()
                .contains("claim_metadata_consistent=false"),
            "unexpected resume error: {resume_err:#}"
        );
        assert_eq!(
            *m.frees.lock().unwrap(),
            vec![0],
            "refused restored sequence must reclaim its allocated slot"
        );

        // 3. Inconsistent SwappedSeq when free_sequence ALSO fails:
        //    - Delivers terminal error to the request sink so caller does not hang after mod.rs removes SwappedSeq,
        //    - Retains ownership of the unreclaimed SequenceState in quarantine (never dropped), and
        //    - Permanently activates fail-stop against blind retry (process restart required in production).
        let (tx, mut rx) = tokio::sync::oneshot::channel();
        let mut active_fail_free = vec![crate::scheduler::test_fixtures::active_seq(
            crate::scheduler::types::ResponseSink::Blocking(Some(tx)),
            3,
        )];
        let mut swapped_fail_free =
            crate::scheduler::lifecycle::swap_out_sequence(m, &mut active_fail_free, 0, &mut spill)
                .expect("swap out third sequence");
        m.frees.lock().unwrap().clear();
        swapped_fail_free.seq_len = swapped_fail_free.tokens.len() + 3;
        *m.free_error.lock().unwrap() = Some("swap reclaim free failed".to_string());

        let resume_reclaim_err = crate::scheduler::lifecycle::resume_swapped_seq(
            None,
            None,
            m,
            swapped_fail_free,
            &mut spill,
        )
        .err()
        .expect("inconsistent SwappedSeq + free_error must fail and quarantine");
        assert!(
            resume_reclaim_err
                .to_string()
                .contains("claim_metadata_consistent=false")
                && resume_reclaim_err
                    .to_string()
                    .contains("additionally failed to reclaim restored sequence slot 0")
                && resume_reclaim_err
                    .to_string()
                    .contains("swap reclaim free failed")
                && resume_reclaim_err
                    .to_string()
                    .contains("SequenceState quarantined; fail-stop active"),
            "unexpected combined swap-resume error: {resume_reclaim_err:#}"
        );
        assert!(
            m.frees.lock().unwrap().is_empty(),
            "failed reclaim must not record slot as freed"
        );
        // Client sink must receive terminal error so removed swapped request does not hang.
        let sink_res = match rx.try_recv() {
            Ok(res) => res,
            Err(_) => panic!("swapped request sink must receive terminal error on swap-in failure"),
        };
        let sink_err = sink_res.err().expect("sink response must be Err");
        assert!(
            sink_err.to_string().contains("Swap-in failed")
                && sink_err
                    .to_string()
                    .contains("additionally failed to reclaim restored sequence slot 0"),
            "unexpected sink error: {sink_err:#}"
        );

        // Ownership is permanently preserved in quarantine and fail-stop remains active on `m`
        // even if `free_error` is cleared (process restart required for production recovery).
        *m.free_error.lock().unwrap() = None;
        assert!(is_model_quarantined(m));
        assert_eq!(quarantined_count_for_model(m), 1);
        assert_eq!(quarantined_summaries_for_model(m)[0].0, 0);

        let blind_retry_err = ContinuationRecord::claim_sequence(m, 0x9999)
            .err()
            .expect("blind retry while quarantined must permanently fail-stop");
        assert!(
            blind_retry_err
                .to_string()
                .contains("ContinuationRecord fail-stop active")
                && blind_retry_err
                    .to_string()
                    .contains("process restart required for production recovery")
        );
    }

    #[test]
    fn test_prefill_cancel_and_disconnect_retire_through_facade_with_barriers() {
        use std::sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        };
        let m = crate::scheduler::test_fixtures::model::recording();

        // 1. Single-shot / first-chunk allocated prefill cancellation via stop_allocated_prefill:
        // verifies stream-drain barriers precede free_sequence and reset_proof marks release_metadata_consistent.
        let (mut seq, mut proof) =
            ContinuationRecord::claim_sequence_on(&m, 5, 0xABCD).expect("claim_sequence_on");
        assert!(!proof.as_ref().unwrap().release_metadata_consistent);
        let cancel_flag = Arc::new(AtomicBool::new(true));
        let mut sink = crate::scheduler::types::ResponseSink::Blocking(None);

        let stopped = crate::scheduler::prefill_lifecycle::stop_allocated_prefill(
            &m,
            5,
            &mut seq,
            &mut proof,
            &mut sink,
            Some(&cancel_flag),
            None,
            "first-chunk",
        );
        assert!(stopped, "cancelled allocated prefill must stop and retire");
        assert!(
            proof.as_ref().unwrap().release_metadata_consistent,
            "stop_allocated_prefill must mark release_metadata_consistent = true through facade"
        );
        assert_eq!(
            *m.lifecycle_events.lock().unwrap(),
            ["sync:5", "sync:0", "free:0"],
            "stream-drain barriers must precede sequence release on stop_allocated_prefill"
        );

        // 2. Chunked prefill disconnect retirement via retire_stopped_prefills:
        // verifies stream-drain barriers precede free_sequence and facade updates proof.
        m.lifecycle_events.lock().unwrap().clear();
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        drop(rx);
        let mut prefill = crate::scheduler::test_fixtures::prefill_in_progress(
            crate::scheduler::types::ResponseSink::Streaming(tx),
        );
        prefill.reset_proof = Some(ResetProofPayload::new(0, prefill.prompt_prefix_hash));
        let mut prefilling = vec![prefill];

        crate::scheduler::prefill_lifecycle::retire_stopped_prefills(&m, &mut prefilling, 7);
        assert!(prefilling.is_empty());
        assert_eq!(
            *m.lifecycle_events.lock().unwrap(),
            ["sync:7", "sync:0", "free:0"],
            "stream-drain barriers must precede sequence release on retire_stopped_prefills"
        );

        // 3. If free_sequence fails during stop_allocated_prefill, release_metadata_consistent stays false.
        *m.free_error.lock().unwrap() = Some("free failed on cancel".to_string());
        let (mut seq_err, mut proof_err) =
            ContinuationRecord::claim_sequence(&m, 0xEEEE).expect("claim_sequence");
        cancel_flag.store(true, Ordering::Release);
        let stopped_err = crate::scheduler::prefill_lifecycle::stop_allocated_prefill(
            &m,
            3,
            &mut seq_err,
            &mut proof_err,
            &mut sink,
            Some(&cancel_flag),
            None,
            "first-chunk",
        );
        assert!(stopped_err);
        assert!(
            !proof_err.as_ref().unwrap().release_metadata_consistent,
            "release_metadata_consistent must remain false when free_sequence fails during cancel retirement"
        );
    }

    #[test]
    fn test_reset_proof_receipt_and_step2_todo_hook() {
        let model = crate::scheduler::test_fixtures::model::recording();
        let (mut seq, mut proof) =
            ContinuationRecord::claim_sequence(&model, 0x0123_4567_89ab_cdef)
                .expect("claim_sequence");
        let p = proof.as_mut().expect("proof armed");
        p.record_snapshot(4, 0x9999, 1);

        let before_release = p.receipt();
        assert!(before_release.contains("scope=cpu-metadata"));
        assert!(before_release.contains("slot_idx=0"));
        assert!(before_release.contains("initial_prefix_hash=0x0123456789abcdef"));
        assert!(before_release.contains("claim_metadata_consistent=true"));
        assert!(before_release.contains("release_metadata_consistent=false"));
        assert!(before_release.contains("snapshots=1"));

        ContinuationRecord::abort_prefill_sequence(&model, &mut seq, &mut proof)
            .expect("abort_prefill_sequence");
        let after_release = proof.as_ref().unwrap().receipt();
        assert!(after_release.contains("scope=cpu-metadata"));
        assert!(after_release.contains("release_metadata_consistent=true"));
    }
}
