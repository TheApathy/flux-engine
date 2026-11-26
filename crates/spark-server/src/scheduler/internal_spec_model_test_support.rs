// SPDX-License-Identifier: AGPL-3.0-only

use std::sync::{
    Mutex,
    atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering},
};

use anyhow::{Result, bail};
use spark_model::traits::{Model, SequenceState, SpecSampling};
use spark_runtime::gpu::DevicePtr;

use super::fixture::seq_state;

pub(super) const VOCAB: usize = 128;
pub(super) const LOGITS: DevicePtr = DevicePtr(0x1000);

pub(super) enum VerifyReply {
    Frame(Vec<u32>, Vec<u32>),
    NearEnd,
    Error,
}

pub(super) struct TestModel {
    reply: Mutex<Option<VerifyReply>>,
    pending: Mutex<Option<(u32, Vec<u32>)>>,
    pub(super) verify_calls: AtomicUsize,
    pub(super) decode_calls: AtomicUsize,
    pub(super) free_calls: AtomicUsize,
    pub(super) commits: Mutex<Vec<usize>>,
    pub(super) commit_error: AtomicBool,
    pub(super) argmax_error: AtomicBool,
    pub(super) next_token: AtomicU32,
    pub(super) argmax_ptr: AtomicU64,
}

impl TestModel {
    pub(super) fn new(reply: VerifyReply, next_token: u32) -> Self {
        Self {
            reply: Mutex::new(Some(reply)),
            pending: Mutex::new(None),
            verify_calls: AtomicUsize::new(0),
            decode_calls: AtomicUsize::new(0),
            free_calls: AtomicUsize::new(0),
            commits: Mutex::new(Vec::new()),
            commit_error: AtomicBool::new(false),
            argmax_error: AtomicBool::new(false),
            next_token: AtomicU32::new(next_token),
            argmax_ptr: AtomicU64::new(0),
        }
    }
}

impl Model for TestModel {
    fn synchronize_stream(&self, _: u64) -> Result<()> {
        Ok(())
    }

    fn prefill(&self, _: &[u32], _: &mut SequenceState, _: u64) -> Result<DevicePtr> {
        unreachable!()
    }
    fn prefill_chunk(
        &self,
        _: &[u32],
        _: &mut SequenceState,
        _: usize,
        _: usize,
        _: bool,
        _: u64,
    ) -> Result<DevicePtr> {
        unreachable!()
    }
    fn decode(&self, _: u32, _: &mut SequenceState, _: u64) -> Result<DevicePtr> {
        unreachable!()
    }
    fn decode_batch(
        &self,
        tokens: &[u32],
        seqs: &mut [&mut SequenceState],
        _: u64,
    ) -> Result<DevicePtr> {
        self.decode_calls.fetch_add(1, Ordering::SeqCst);
        for (token, seq) in tokens.iter().zip(seqs.iter_mut()) {
            seq.tokens.push(*token);
            seq.seq_len += 1;
        }
        Ok(LOGITS)
    }
    fn vocab_size(&self) -> usize {
        VOCAB
    }
    fn bind_gpu_to_thread(&self) -> Result<()> {
        Ok(())
    }
    fn alloc_sequence(&self) -> Result<SequenceState> {
        Ok(seq_state())
    }
    fn copy_logits_to_host(&self, _: DevicePtr, _: &mut [u8]) -> Result<()> {
        unreachable!()
    }
    fn logits_buffer_ptr(&self) -> DevicePtr {
        LOGITS
    }
    fn argmax_on_device(&self, _: DevicePtr, _: u64) -> Result<u32> {
        unreachable!()
    }
    fn argmax_batch(&self, ptr: DevicePtr, _: usize, _: u64) -> Result<Vec<u32>> {
        self.argmax_ptr.store(ptr.0, Ordering::SeqCst);
        if self.argmax_error.load(Ordering::SeqCst) {
            bail!("injected argmax failure");
        }
        Ok(vec![self.next_token.load(Ordering::SeqCst)])
    }
    fn hidden_after_norm(&self) -> DevicePtr {
        DevicePtr::NULL
    }
    fn decode_verify(&self, _: &[u32], _: &mut SequenceState, _: u64) -> Result<Vec<u32>> {
        unreachable!()
    }
    fn checkpoint_ssm_states(&self, _: &mut SequenceState) -> Result<()> {
        unreachable!()
    }
    fn rollback_ssm_states(&self, _: &mut SequenceState, _: usize) -> Result<()> {
        unreachable!()
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
        false
    }
    fn has_self_speculative(&self) -> bool {
        false
    }
    fn has_internal_spec(&self) -> bool {
        true
    }
    fn spec_verify(
        &self,
        token: u32,
        _: &mut SequenceState,
        _: Option<&SpecSampling>,
        _: Option<usize>,
        _: u64,
    ) -> Result<Option<(Vec<u32>, Vec<u32>)>> {
        self.verify_calls.fetch_add(1, Ordering::SeqCst);
        match self.reply.lock().unwrap().take().expect("one verify reply") {
            VerifyReply::Frame(drafts, targets) => {
                *self.pending.lock().unwrap() = Some((token, drafts.clone()));
                Ok(Some((drafts, targets)))
            }
            VerifyReply::NearEnd => Ok(None),
            VerifyReply::Error => bail!("injected verify failure"),
        }
    }
    fn spec_commit(&self, seq: &mut SequenceState, accepted: usize, _: u64) -> Result<()> {
        let (token, drafts) = self.pending.lock().unwrap().take().expect("pending verify");
        if self.commit_error.load(Ordering::SeqCst) {
            bail!("injected commit failure");
        }
        self.commits.lock().unwrap().push(accepted);
        seq.tokens.push(token);
        seq.tokens.extend_from_slice(&drafts[..accepted]);
        seq.seq_len += accepted + 1;
        Ok(())
    }
    fn decode_draft(&self, _: u32, _: &mut SequenceState, _: u64) -> Result<DevicePtr> {
        unreachable!()
    }
    fn cache_sequence(&self, _: &SequenceState) {}
    fn free_sequence(&self, _: &mut SequenceState) -> Result<()> {
        self.free_calls.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    fn compact_sequence(&self, seq: &mut SequenceState, slot: usize) -> Result<bool> {
        seq.slot_idx = slot;
        Ok(true)
    }
    fn decode_verify_graphed(
        &self,
        _: &[u32; 2],
        _: &mut SequenceState,
        _: u64,
    ) -> Result<[u32; 2]> {
        unreachable!()
    }
    fn decode_verify_graphed_k3(
        &self,
        _: &[u32; 3],
        _: &mut SequenceState,
        _: u64,
    ) -> Result<[u32; 3]> {
        unreachable!()
    }
    fn decode_verify_graphed_k4(
        &self,
        _: &[u32; 4],
        _: &mut SequenceState,
        _: u64,
    ) -> Result<[u32; 4]> {
        unreachable!()
    }
    fn save_hidden_for_mtp(&self, _: usize, _: u64) -> Result<()> {
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
        _: &mut SequenceState,
        _: u64,
        _: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        unreachable!()
    }
    fn trim_proposer_state(&self, _: &mut SequenceState, _: usize, _: u64) -> Result<()> {
        Ok(())
    }
}
