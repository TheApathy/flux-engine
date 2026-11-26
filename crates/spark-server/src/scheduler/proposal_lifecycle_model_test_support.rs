// SPDX-License-Identifier: AGPL-3.0-only

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

use anyhow::Result;
use spark_model::traits::SequenceState;
use spark_runtime::gpu::DevicePtr;

use super::super::*;

pub(crate) fn seq_state() -> SequenceState {
    SequenceState {
        tokens: Vec::new(),
        block_table: Vec::new(),
        seq_len: 9,
        qwen4_qsa_required: false,
        layer_states: Vec::new(),
        rotary_positions: spark_model::traits::RotaryPositions::identity(),
        proposer_state: None,
        proposer_state_alt: None,
        slot_idx: 0,
        marconi_skip_to: 0,
        session_hash: 0,
        verify_frame: None,
        chunked_prefill_meta: None,
        qwen4_prefill_lineage: spark_model::traits::Qwen4PrefillLineage::Unset,
        cached_prefix_tokens: 0,
        kv_valid_tokens: 0,
        prompt_len: 0,
        disk_block_ids: Vec::new(),
        mtp_lastk_host_buf: Vec::new(),
        mtp_lastk_host_filled: 0,
        mtp_lastk_end_abs: 0,
        disk_last_offloaded_per_layer: Vec::new(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TestBatchPhase {
    Reserved,
    InFlight,
    Committed,
    Published,
    Retired,
    Aborted,
}

pub(crate) struct TestModel {
    proposal: Mutex<Option<Result<Vec<u32>>>>,
    tree: Mutex<Option<DDTreePayload>>,
    pub(crate) proposals: AtomicUsize,
    pub(crate) takes: AtomicUsize,
    /// Tokens `argmax_batch` returns; empty = the call is unexpected.
    pub(crate) argmax: Mutex<Vec<u32>>,
    /// One logits row `copy_logits_to_host` serves (as bf16); its length is
    /// the vocab size. Empty = the call is unexpected.
    pub(crate) host_logits: Mutex<Vec<f32>>,
    cancel_during_propose: Mutex<Option<Arc<AtomicBool>>>,
    /// `(from_slot, to_slot)` for every `compact_sequence` call.
    pub(crate) compacts: Mutex<Vec<(usize, usize)>>,
    /// `slot_idx` of every sequence handed to `free_sequence` (the real
    /// pool skips `usize::MAX`; recording it here lets a test see which
    /// slot the scheduler believed it was releasing).
    pub(crate) frees: Mutex<Vec<usize>>,
    /// When set, `compact_sequence` fails without moving anything.
    pub(crate) fail_compact: AtomicBool,
    /// Slots owned by a sequence the active list cannot see (a PREFILLING
    /// one): `compact_sequence` refuses them with `Ok(false)` like the pool.
    pub(crate) held_slots: Mutex<Vec<usize>>,
    /// When set, `alloc_sequence` fails with this message.
    pub(crate) alloc_error: Mutex<Option<String>>,
    /// When non-zero, `alloc_sequence` returns a dirty `SequenceState` with this `seq_len`.
    pub(crate) dirty_alloc_seq_len: AtomicUsize,
    /// When set, `free_sequence` fails with this message.
    pub(crate) free_error: Mutex<Option<String>>,
    /// Ordered lifetime-barrier trace used by hostile retirement tests.
    pub(crate) lifecycle_events: Mutex<Vec<String>>,
    /// Enables actual bootstrap dispatch with an injected failing stage.
    /// None retains the mock's original unexpected-decode tripwire.
    pub(crate) bootstrap_failure: Mutex<Option<&'static str>>,
    pub(crate) bootstrap_calls: Mutex<Vec<&'static str>>,
    /// Enables actual verifier dispatch and injects one mandatory failing stage.
    pub(crate) verify_failure: Mutex<Option<&'static str>>,
    pub(crate) verify_targets: Mutex<Vec<u32>>,
    pub(crate) verify_calls: Mutex<Vec<&'static str>>,
    pub(crate) verify_commits: Mutex<Vec<(usize, usize, usize)>>,
    pub(crate) verify_aborts: Mutex<Vec<(usize, usize, usize)>>,
    pub(crate) verify_abort_error: AtomicBool,
    pub(crate) verify_is_ep: AtomicBool,
    /// Arms the cross-sequence batched native-MTP verifier
    /// (`ATLAS_QWEN4_BATCHED_K5`) and reports a native MTP proposer.
    pub(crate) batched_k5: AtomicBool,
    /// Makes `preflight_verify_qwen4_batched` refuse (e.g. a poisoned
    /// pointer table), so the scheduler must verify per sequence.
    pub(crate) batched_refused: AtomicBool,
    pub(crate) batched_frames: AtomicUsize,
    pub(crate) batched_preflights: AtomicUsize,
    pub(crate) batched_refuse_preflight: AtomicUsize,
    pub(crate) batched_frame_shapes: Mutex<Vec<(usize, usize, usize)>>,
    pub(crate) batched_fail_frame: AtomicUsize,
    pub(crate) batched_fail_slot: AtomicUsize,
    pub(crate) batched_owner_active: AtomicBool,
    pub(crate) batched_lanes: Mutex<std::collections::BTreeMap<usize, (usize, TestBatchPhase)>>,
    pub(crate) batched_abort_attempts: Mutex<Vec<usize>>,
    batched_current_frame: Mutex<Vec<usize>>,
    verify_ep_width_pending: AtomicBool,
    pub(crate) caches: AtomicUsize,
    /// `(offset, len)` of every `prefill_chunk`; `None` keeps the call an
    /// unexpected tripwire.
    pub(crate) prefill_chunks: Mutex<Option<Vec<(usize, usize)>>>,
    /// After this many chunks, drop the held value (a response receiver), so
    /// the request's sink reads as disconnected while the prefill runs.
    pub(crate) disconnect_after_chunks: Mutex<Option<(usize, Box<dyn std::any::Any + Send>)>>,
    /// Phase-C decode-ring SSM mock: `has_ssm_layers()` return value.
    /// `false` (the default) keeps every other test on the pure-attention
    /// path untouched.
    pub(crate) ssm_layers: AtomicBool,
    /// `decode_rollback_ring_slots()` return value.
    pub(crate) ssm_ring_slots: AtomicUsize,
    /// `ring_slot` of every `save_decode_ssm_snapshot` call, in order.
    pub(crate) ssm_save_calls: Mutex<Vec<usize>>,
    /// `ring_slot` of every `restore_decode_ssm_snapshot` call, in order.
    pub(crate) ssm_restore_calls: Mutex<Vec<usize>>,
    /// When set, `restore_decode_ssm_snapshot` fails with this message
    /// instead of succeeding — models a corrupt/evicted GPU snapshot.
    pub(crate) ssm_restore_fail: Mutex<Option<String>>,
}

impl TestModel {
    pub(crate) fn new(proposal: Result<Vec<u32>>, tree: Option<DDTreePayload>) -> Self {
        Self {
            proposal: Mutex::new(Some(proposal)),
            tree: Mutex::new(tree),
            proposals: AtomicUsize::new(0),
            takes: AtomicUsize::new(0),
            argmax: Mutex::new(Vec::new()),
            host_logits: Mutex::new(Vec::new()),
            cancel_during_propose: Mutex::new(None),
            compacts: Mutex::new(Vec::new()),
            frees: Mutex::new(Vec::new()),
            fail_compact: AtomicBool::new(false),
            held_slots: Mutex::new(Vec::new()),
            alloc_error: Mutex::new(None),
            dirty_alloc_seq_len: AtomicUsize::new(0),
            free_error: Mutex::new(None),
            lifecycle_events: Mutex::new(Vec::new()),
            bootstrap_failure: Mutex::new(None),
            bootstrap_calls: Mutex::new(Vec::new()),
            verify_failure: Mutex::new(None),
            verify_targets: Mutex::new(Vec::new()),
            verify_calls: Mutex::new(Vec::new()),
            verify_commits: Mutex::new(Vec::new()),
            verify_aborts: Mutex::new(Vec::new()),
            verify_abort_error: AtomicBool::new(false),
            verify_is_ep: AtomicBool::new(false),
            batched_k5: AtomicBool::new(false),
            batched_refused: AtomicBool::new(false),
            batched_frames: AtomicUsize::new(0),
            batched_preflights: AtomicUsize::new(0),
            batched_refuse_preflight: AtomicUsize::new(usize::MAX),
            batched_frame_shapes: Mutex::new(Vec::new()),
            batched_fail_frame: AtomicUsize::new(usize::MAX),
            batched_fail_slot: AtomicUsize::new(usize::MAX),
            batched_owner_active: AtomicBool::new(false),
            batched_lanes: Mutex::new(std::collections::BTreeMap::new()),
            batched_abort_attempts: Mutex::new(Vec::new()),
            batched_current_frame: Mutex::new(Vec::new()),
            verify_ep_width_pending: AtomicBool::new(false),
            caches: AtomicUsize::new(0),
            prefill_chunks: Mutex::new(None),
            disconnect_after_chunks: Mutex::new(None),
            ssm_layers: AtomicBool::new(false),
            ssm_ring_slots: AtomicUsize::new(0),
            ssm_save_calls: Mutex::new(Vec::new()),
            ssm_restore_calls: Mutex::new(Vec::new()),
            ssm_restore_fail: Mutex::new(None),
        }
    }

    /// Mark this model hybrid (SSM layers) with `ring_slots` decode-rollback
    /// snapshot slots reserved per sequence.
    pub(crate) fn with_ssm_layers(self, ring_slots: usize) -> Self {
        self.ssm_layers.store(true, Ordering::SeqCst);
        self.ssm_ring_slots.store(ring_slots, Ordering::SeqCst);
        self
    }

    /// Make every `restore_decode_ssm_snapshot` call fail with `msg`,
    /// modeling a corrupt or evicted GPU snapshot.
    pub(crate) fn failing_ssm_restore(self, msg: impl Into<String>) -> Self {
        *self.ssm_restore_fail.lock().unwrap() = Some(msg.into());
        self
    }

    pub(crate) fn cancelling_during_propose(self, flag: Arc<AtomicBool>) -> Self {
        *self.cancel_during_propose.lock().unwrap() = Some(flag);
        self
    }

    fn bootstrap_stage(&self, stage: &'static str) -> Result<bool> {
        let failure = *self.bootstrap_failure.lock().unwrap();
        let Some(failure) = failure else {
            return Ok(false);
        };
        self.bootstrap_calls.lock().unwrap().push(stage);
        anyhow::ensure!(failure != stage, "injected bootstrap {stage} failure");
        Ok(true)
    }

    fn verify_stage(&self, stage: &'static str) -> Result<bool> {
        let Some(failure) = *self.verify_failure.lock().unwrap() else {
            return Ok(false);
        };
        self.verify_calls.lock().unwrap().push(stage);
        anyhow::ensure!(failure != stage, "injected verifier {stage} failure");
        Ok(true)
    }

    fn verify_slot_stage(&self, stage: &'static str, slot: usize) -> Result<bool> {
        let failing_slot = self.batched_fail_slot.load(Ordering::SeqCst);
        if self.batched_owner_active.load(Ordering::SeqCst)
            && failing_slot != usize::MAX
            && failing_slot != slot
            && *self.verify_failure.lock().unwrap() == Some(stage)
        {
            self.verify_calls.lock().unwrap().push(stage);
            return Ok(true);
        }
        self.verify_stage(stage)
    }

    fn verify_reply(&self, tokens: &[u32], seq: &mut SequenceState) -> Result<Vec<u32>> {
        assert!(self.verify_failure.lock().unwrap().is_some());
        // Model a forward that can fail after partially advancing state.
        seq.seq_len += tokens.len();
        seq.tokens.extend_from_slice(tokens);
        self.verify_stage("decode_verify")?;
        Ok(self.verify_targets.lock().unwrap().clone())
    }
}

pub(crate) struct Frame {
    pub(crate) state: SequenceState,
    pub(crate) drafts: Vec<u32>,
    pub(crate) tree: Option<DDTreePayload>,
    pub(crate) sampled: Option<spark_model::dflash_sampling::DflashSampledProposalFrame>,
}

impl SchedulerProposalFrame for Frame {
    type State = SequenceState;
    type Tree = DDTreePayload;
    type Sampled = spark_model::dflash_sampling::DflashSampledProposalFrame;

    fn position(&self) -> usize {
        self.state.seq_len
    }
    fn parts(
        &mut self,
    ) -> (
        &mut SequenceState,
        &mut Vec<u32>,
        &mut Option<DDTreePayload>,
        &mut Option<Self::Sampled>,
    ) {
        (
            &mut self.state,
            &mut self.drafts,
            &mut self.tree,
            &mut self.sampled,
        )
    }
}

pub(crate) fn tree(token: u32) -> DDTreePayload {
    DDTreePayload {
        tree_token_ids: vec![token],
        parent_indices: vec![-1],
    }
}

impl Model for TestModel {
    fn synchronize_stream(&self, stream: u64) -> Result<()> {
        self.lifecycle_events
            .lock()
            .unwrap()
            .push(format!("sync:{stream}"));
        Ok(())
    }

    fn prefill(&self, _: &[u32], _: &mut SequenceState, _: u64) -> Result<DevicePtr> {
        unreachable!()
    }
    fn prefill_chunk(
        &self,
        _: &[u32],
        _: &mut SequenceState,
        offset: usize,
        len: usize,
        _: bool,
        _: u64,
    ) -> Result<DevicePtr> {
        let mut chunks = self.prefill_chunks.lock().unwrap();
        let chunks = chunks.as_mut().expect("unexpected prefill_chunk");
        chunks.push((offset, len));
        let mut disconnect = self.disconnect_after_chunks.lock().unwrap();
        if disconnect
            .as_ref()
            .is_some_and(|(after, _)| chunks.len() >= *after)
        {
            disconnect.take();
        }
        Ok(DevicePtr::NULL)
    }
    fn decode(&self, _: u32, seq: &mut SequenceState, _: u64) -> Result<DevicePtr> {
        let bootstrap_enabled = self.bootstrap_failure.lock().unwrap().is_some();
        if bootstrap_enabled {
            // Model a forward that advances state before returning an error.
            seq.seq_len += 1;
            self.bootstrap_stage("decode")?;
            return Ok(DevicePtr::NULL);
        }
        unreachable!()
    }
    fn decode_batch(&self, _: &[u32], _: &mut [&mut SequenceState], _: u64) -> Result<DevicePtr> {
        unreachable!()
    }
    fn vocab_size(&self) -> usize {
        self.host_logits.lock().unwrap().len()
    }
    fn bind_gpu_to_thread(&self) -> Result<()> {
        Ok(())
    }
    fn alloc_sequence(&self) -> Result<SequenceState> {
        if let Some(msg) = self.alloc_error.lock().unwrap().as_deref() {
            anyhow::bail!("{msg}");
        }
        let mut s = seq_state();
        s.seq_len = self.dirty_alloc_seq_len.load(Ordering::Relaxed);
        Ok(s)
    }
    fn copy_logits_to_host(&self, _: DevicePtr, out: &mut [u8]) -> Result<()> {
        self.bootstrap_stage("logits_download")?;
        self.verify_stage("logits_download")?;
        let row = self.host_logits.lock().unwrap();
        assert!(!row.is_empty(), "unexpected copy_logits_to_host");
        for (i, v) in row.iter().enumerate() {
            let bits = (v.to_bits() >> 16) as u16; // exact for bf16-representable test values
            out[i * 2..i * 2 + 2].copy_from_slice(&bits.to_le_bytes());
        }
        Ok(())
    }
    fn logits_buffer_ptr(&self) -> DevicePtr {
        DevicePtr::NULL
    }
    fn argmax_on_device(&self, _: DevicePtr, _: u64) -> Result<u32> {
        if self.bootstrap_stage("sample")? {
            return Ok(17);
        }
        unreachable!()
    }
    fn argmax_batch(&self, _: DevicePtr, _: usize, _: u64) -> Result<Vec<u32>> {
        let tokens = self.argmax.lock().unwrap().clone();
        assert!(!tokens.is_empty(), "unexpected argmax_batch");
        Ok(tokens)
    }
    fn hidden_after_norm(&self) -> DevicePtr {
        DevicePtr::NULL
    }
    fn decode_verify(&self, tokens: &[u32], seq: &mut SequenceState, _: u64) -> Result<Vec<u32>> {
        self.verify_reply(tokens, seq)
    }
    fn checkpoint_ssm_states(&self, _: &mut SequenceState) -> Result<()> {
        unreachable!()
    }
    fn rollback_ssm_states(&self, _: &mut SequenceState, _: usize) -> Result<()> {
        unreachable!()
    }
    fn has_ssm_layers(&self) -> bool {
        self.ssm_layers.load(Ordering::SeqCst)
    }
    fn decode_rollback_ring_slots(&self) -> usize {
        self.ssm_ring_slots.load(Ordering::SeqCst)
    }
    fn save_decode_ssm_snapshot(&self, _seq: &SequenceState, ring_slot: usize) -> Result<()> {
        self.ssm_save_calls.lock().unwrap().push(ring_slot);
        Ok(())
    }
    fn restore_decode_ssm_snapshot(&self, _seq: &SequenceState, ring_slot: usize) -> Result<()> {
        self.ssm_restore_calls.lock().unwrap().push(ring_slot);
        if let Some(msg) = self.ssm_restore_fail.lock().unwrap().clone() {
            anyhow::bail!(msg);
        }
        Ok(())
    }
    fn generate_speculative(
        &self,
        _: &[u32],
        _: &spark_runtime::sampler::SamplingParams,
        _: usize,
    ) -> Result<spark_model::engine::GenerateResult> {
        unreachable!()
    }
    fn has_proposer(&self) -> bool {
        true
    }
    fn has_self_speculative(&self) -> bool {
        false
    }
    fn decode_draft(&self, _: u32, _: &mut SequenceState, _: u64) -> Result<DevicePtr> {
        unreachable!()
    }
    fn cache_sequence(&self, _: &SequenceState) {
        self.caches.fetch_add(1, Ordering::SeqCst);
    }
    fn sync_secondary(&self) -> Result<()> {
        self.bootstrap_stage("sync_secondary")?;
        self.verify_stage("sync_secondary")?;
        Ok(())
    }
    fn reconcile_dflash_context_before_bootstrap(
        &self,
        _: &mut SequenceState,
        _: u64,
    ) -> Result<usize> {
        self.bootstrap_stage("reconcile_context")?;
        Ok(0)
    }
    fn ep_broadcast_cmd(&self, cmd: u32) -> Result<()> {
        if cmd != 0xFFFFFFF1 {
            self.bootstrap_stage("EP broadcast")?;
            let stage = if cmd == spark_model::traits::EP_CMD_VERIFY_KGAMMA {
                self.verify_ep_width_pending.store(true, Ordering::SeqCst);
                "EP command"
            } else if self.verify_ep_width_pending.swap(false, Ordering::SeqCst) {
                "EP width"
            } else if cmd >= 0xFFFFFF00 {
                "EP command"
            } else if cmd >= 31 {
                "EP token"
            } else {
                "EP result"
            };
            self.verify_stage(stage)?;
        }
        Ok(())
    }
    fn is_ep(&self) -> bool {
        self.verify_is_ep.load(Ordering::SeqCst)
    }
    fn start_checkpoint_async(&self, _: &mut SequenceState) -> Result<()> {
        assert!(self.bootstrap_stage("start_checkpoint_async")?);
        Ok(())
    }
    fn commit_verify_state_async(
        &self,
        seq: &mut SequenceState,
        accepted: usize,
        k: usize,
    ) -> Result<()> {
        self.verify_commits
            .lock()
            .unwrap()
            .push((seq.seq_len, accepted, k));
        assert!(self.verify_slot_stage("commit", seq.slot_idx)?);
        if self.batched_owner_active.load(Ordering::SeqCst) {
            let mut lanes = self.batched_lanes.lock().unwrap();
            let (_, phase) = lanes
                .get_mut(&seq.slot_idx)
                .expect("reserved physical lane");
            anyhow::ensure!(
                *phase == TestBatchPhase::InFlight,
                "commit requires an in-flight lane"
            );
            *phase = TestBatchPhase::Committed;
        }
        Ok(())
    }
    fn abort_verify_state(&self, seq: &mut SequenceState, k: usize) -> Result<()> {
        self.verify_aborts
            .lock()
            .unwrap()
            .push((seq.seq_len, seq.tokens.len(), k));
        anyhow::ensure!(
            !self.verify_abort_error.load(Ordering::SeqCst),
            "injected verify rollback failure"
        );
        if self.batched_owner_active.load(Ordering::SeqCst)
            && let Some((cursor, phase)) = self.batched_lanes.lock().unwrap().get_mut(&seq.slot_idx)
        {
            anyhow::ensure!(
                *phase == TestBatchPhase::InFlight,
                "zero abort requires an in-flight lane"
            );
            seq.seq_len = *cursor;
            seq.tokens.truncate(*cursor);
            *phase = TestBatchPhase::Aborted;
        }
        Ok(())
    }
    fn free_sequence(&self, seq: &mut SequenceState) -> Result<()> {
        if let Some(msg) = self.free_error.lock().unwrap().as_deref() {
            anyhow::bail!("{msg}");
        }
        self.frees.lock().unwrap().push(seq.slot_idx);
        self.lifecycle_events
            .lock()
            .unwrap()
            .push(format!("free:{}", seq.slot_idx));
        Ok(())
    }
    fn compact_sequence(&self, seq: &mut SequenceState, slot: usize) -> Result<bool> {
        self.compacts.lock().unwrap().push((seq.slot_idx, slot));
        if self.fail_compact.load(Ordering::SeqCst) {
            anyhow::bail!("injected compact failure");
        }
        // Mirrors the pool contract: a slot listed in `held_slots` is owned
        // by a sequence the active list cannot see and is refused.
        if self.held_slots.lock().unwrap().contains(&slot) {
            return Ok(false);
        }
        seq.slot_idx = slot;
        Ok(true)
    }
    fn save_sequence_state(&self, _: &SequenceState, _: &mut dyn std::io::Write) -> Result<()> {
        Ok(())
    }
    fn restore_sequence_state(
        &self,
        _: &mut SequenceState,
        _: usize,
        _: &mut dyn std::io::Read,
    ) -> Result<()> {
        Ok(())
    }
    fn decode_verify_graphed(
        &self,
        tokens: &[u32; 2],
        seq: &mut SequenceState,
        _: u64,
    ) -> Result<[u32; 2]> {
        Ok(self.verify_reply(tokens, seq)?.try_into().expect("K2 rows"))
    }
    fn decode_verify_graphed_k3(
        &self,
        tokens: &[u32; 3],
        seq: &mut SequenceState,
        _: u64,
    ) -> Result<[u32; 3]> {
        Ok(self.verify_reply(tokens, seq)?.try_into().expect("K3 rows"))
    }
    fn decode_verify_graphed_k4(
        &self,
        tokens: &[u32; 4],
        seq: &mut SequenceState,
        _: u64,
    ) -> Result<[u32; 4]> {
        Ok(self.verify_reply(tokens, seq)?.try_into().expect("K4 rows"))
    }
    fn decode_verify_dflash(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        _: u64,
    ) -> Result<Vec<u32>> {
        self.verify_reply(tokens, seq)
    }
    fn decode_verify_dflash_batched(
        &self,
        tokens_per_seq: &[Vec<u32>],
        seqs: &mut [&mut SequenceState],
        _: u64,
    ) -> Result<Vec<Vec<u32>>> {
        let mut results = Vec::with_capacity(seqs.len());
        for (tokens, seq) in tokens_per_seq.iter().zip(seqs.iter_mut()) {
            results.push(self.verify_reply(tokens, seq)?);
        }
        if *self.verify_failure.lock().unwrap() == Some("outer rows") {
            results.pop();
        }
        Ok(results)
    }
    fn save_hidden_for_mtp(&self, _: usize, _: u64) -> Result<()> {
        self.verify_stage("save hidden")?;
        if self.bootstrap_failure.lock().unwrap().is_some()
            || self.verify_failure.lock().unwrap().is_some()
        {
            return Ok(());
        }
        unreachable!()
    }
    fn run_mtp_propose(
        &self,
        _: u32,
        _: usize,
        _: &mut SequenceState,
        _: u64,
    ) -> Result<Option<u32>> {
        unreachable!()
    }
    fn run_mtp_propose_multi(
        &self,
        _: u32,
        _: usize,
        _: usize,
        seq: &mut SequenceState,
        _: u64,
        _: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        self.verify_stage("propose")?;
        anyhow::ensure!(
            !self.batched_owner_active.load(Ordering::SeqCst),
            "proposer entered while batch cohort still owns scratch"
        );
        if self.batched_k5.load(Ordering::SeqCst) {
            self.lifecycle_events
                .lock()
                .unwrap()
                .push(format!("propose:{}", seq.slot_idx));
        }
        self.proposals.fetch_add(1, Ordering::SeqCst);
        if let Some(flag) = self.cancel_during_propose.lock().unwrap().as_ref() {
            flag.store(true, Ordering::Release);
        }
        let proposal = self.proposal.lock().unwrap().take();
        if self.batched_k5.load(Ordering::SeqCst) {
            proposal.unwrap_or_else(|| Ok(Vec::new()))
        } else {
            proposal.expect("one proposal")
        }
    }
    fn take_pending_tree_payload(&self, _: &mut SequenceState) -> Option<DDTreePayload> {
        self.takes.fetch_add(1, Ordering::SeqCst);
        self.tree.lock().unwrap().take()
    }
    fn trim_proposer_state(&self, seq: &mut SequenceState, _: usize, _: u64) -> Result<()> {
        self.verify_slot_stage("trim", seq.slot_idx)?;
        Ok(())
    }
    fn qwen4_batched_k5_armed(&self) -> bool {
        self.batched_k5.load(Ordering::SeqCst)
    }
    fn proposer_arm_kind(&self, _: u8) -> Option<spark_model::speculative::DraftProposerKind> {
        self.batched_k5
            .load(Ordering::SeqCst)
            .then_some(spark_model::speculative::DraftProposerKind::Qwen4NativeMtp)
    }
    fn preflight_verify_qwen4_batched(&self, _: &[Vec<u32>], _: &[&SequenceState]) -> Result<()> {
        let preflight = self.batched_preflights.fetch_add(1, Ordering::SeqCst);
        assert!(self.verify_stage("batched preflight")?);
        anyhow::ensure!(
            !self.batched_refused.load(Ordering::SeqCst)
                && self.batched_refuse_preflight.load(Ordering::SeqCst) != preflight,
            "injected batched admission refusal"
        );
        Ok(())
    }
    fn begin_verify_qwen4_batch(&self, seqs: &mut [&mut SequenceState], _k: usize) -> Result<()> {
        self.verify_stage("batch begin")?;
        anyhow::ensure!(
            !self.batched_owner_active.swap(true, Ordering::SeqCst),
            "batch cohort already active"
        );
        let mut lanes = self.batched_lanes.lock().unwrap();
        lanes.clear();
        for seq in seqs {
            anyhow::ensure!(seq.tokens.len() == seq.seq_len, "begin host extent differs");
            anyhow::ensure!(
                lanes
                    .insert(seq.slot_idx, (seq.seq_len, TestBatchPhase::Reserved))
                    .is_none(),
                "duplicate physical slot in batch begin"
            );
        }
        self.lifecycle_events
            .lock()
            .unwrap()
            .push(format!("batch-begin:{}", lanes.len()));
        Ok(())
    }
    fn end_verify_qwen4_batch_frame(&self, seqs: &[&SequenceState]) -> Result<()> {
        self.verify_stage("batch end")?;
        let lanes = self.batched_lanes.lock().unwrap();
        let slots: Vec<_> = seqs.iter().map(|seq| seq.slot_idx).collect();
        let mut frame = self.batched_current_frame.lock().unwrap();
        anyhow::ensure!(
            *frame == slots,
            "frame end did not match the active physical slots"
        );
        anyhow::ensure!(
            slots.iter().all(|slot| matches!(
                lanes.get(slot),
                Some((
                    _,
                    TestBatchPhase::Published | TestBatchPhase::Retired | TestBatchPhase::Aborted
                ))
            )),
            "frame end has an unsettled lane"
        );
        frame.clear();
        Ok(())
    }
    fn finish_verify_qwen4_batch(&self, seqs: &[&SequenceState]) -> Result<()> {
        self.verify_stage("batch finish")?;
        anyhow::ensure!(
            self.batched_current_frame.lock().unwrap().is_empty(),
            "frame remains active"
        );
        let lanes = self.batched_lanes.lock().unwrap();
        anyhow::ensure!(
            seqs.len() == lanes.len()
                && seqs.iter().all(|seq| matches!(
                    lanes.get(&seq.slot_idx),
                    Some((
                        _,
                        TestBatchPhase::Published
                            | TestBatchPhase::Retired
                            | TestBatchPhase::Aborted
                    ))
                )),
            "cohort finish has an unsettled lane"
        );
        self.batched_owner_active.store(false, Ordering::SeqCst);
        self.lifecycle_events
            .lock()
            .unwrap()
            .push("batch-finish".into());
        Ok(())
    }
    fn abort_verify_qwen4_batch(&self, seqs: &mut [&mut SequenceState]) -> Result<()> {
        self.verify_calls.lock().unwrap().push("batch abort");
        let mut lanes = self.batched_lanes.lock().unwrap();
        for seq in seqs {
            self.batched_abort_attempts
                .lock()
                .unwrap()
                .push(seq.slot_idx);
            if let Some((cursor, phase)) = lanes.get_mut(&seq.slot_idx) {
                match phase {
                    TestBatchPhase::Reserved | TestBatchPhase::InFlight => {
                        seq.seq_len = *cursor;
                        seq.tokens.truncate(*cursor);
                        *phase = TestBatchPhase::Aborted;
                    }
                    TestBatchPhase::Committed
                    | TestBatchPhase::Published
                    | TestBatchPhase::Retired
                    | TestBatchPhase::Aborted => {}
                }
            }
        }
        anyhow::ensure!(
            !self.verify_abort_error.load(Ordering::SeqCst),
            "injected cohort abort failure"
        );
        self.batched_current_frame.lock().unwrap().clear();
        self.batched_owner_active.store(false, Ordering::SeqCst);
        Ok(())
    }
    fn retire_verify_qwen4_batch_terminal(&self, seq: &SequenceState) -> Result<()> {
        self.verify_slot_stage("batch terminal retire", seq.slot_idx)?;
        let mut lanes = self.batched_lanes.lock().unwrap();
        let (_, phase) = lanes
            .get_mut(&seq.slot_idx)
            .expect("reserved physical lane");
        anyhow::ensure!(
            *phase == TestBatchPhase::Committed,
            "terminal retire requires a positive commit"
        );
        *phase = TestBatchPhase::Retired;
        Ok(())
    }
    fn decode_verify_qwen4_batched(
        &self,
        tokens_per_seq: &[Vec<u32>],
        seqs: &mut [&mut SequenceState],
    ) -> Result<Vec<Vec<u32>>> {
        let frame_number = self.batched_frames.fetch_add(1, Ordering::SeqCst);
        anyhow::ensure!(
            self.batched_owner_active.load(Ordering::SeqCst),
            "frame lacks a cohort owner"
        );
        {
            let mut frame = self.batched_current_frame.lock().unwrap();
            anyhow::ensure!(frame.is_empty(), "prior frame is not settled");
            let mut lanes = self.batched_lanes.lock().unwrap();
            for seq in seqs.iter() {
                let (_, phase) = lanes
                    .get_mut(&seq.slot_idx)
                    .expect("reserved physical lane");
                anyhow::ensure!(
                    *phase == TestBatchPhase::Reserved,
                    "lane reused within cohort"
                );
                *phase = TestBatchPhase::InFlight;
                frame.push(seq.slot_idx);
            }
        }
        self.batched_frame_shapes.lock().unwrap().push((
            seqs.len(),
            tokens_per_seq.iter().map(Vec::len).sum(),
            tokens_per_seq[0].len(),
        ));
        // Model a frame that advances every sequence before it can fail.
        for (tokens, seq) in tokens_per_seq.iter().zip(seqs.iter_mut()) {
            seq.seq_len += tokens.len();
            seq.tokens.extend_from_slice(tokens);
        }
        self.verify_stage("batched verify")?;
        anyhow::ensure!(
            self.batched_fail_frame.load(Ordering::SeqCst) != frame_number,
            "injected later frame failure"
        );
        let targets = self.verify_targets.lock().unwrap().clone();
        let mut results = vec![targets; seqs.len()];
        match *self.verify_failure.lock().unwrap() {
            Some("batched rows") => {
                results.pop();
            }
            Some("batched short row") => {
                results.last_mut().expect("a frame has sequences").pop();
            }
            _ => {}
        }
        Ok(results)
    }
    fn save_hidden_for_mtp_sequence_in_frame(
        &self,
        _: usize,
        _: usize,
        seq: &SequenceState,
        _: u64,
    ) -> Result<()> {
        self.verify_slot_stage("save hidden", seq.slot_idx)?;
        let mut lanes = self.batched_lanes.lock().unwrap();
        let (_, phase) = lanes
            .get_mut(&seq.slot_idx)
            .expect("reserved physical lane");
        anyhow::ensure!(
            *phase == TestBatchPhase::Committed,
            "hidden publication requires a commit"
        );
        *phase = TestBatchPhase::Published;
        Ok(())
    }
}
