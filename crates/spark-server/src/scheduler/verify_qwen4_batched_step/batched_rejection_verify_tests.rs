// SPDX-License-Identifier: AGPL-3.0-only

use super::{
    batched_verify_proposer_eligible_with_policy, clamp_drafts_to_max_seq_len_count,
    clamp_pending_drafts_for_max_seq_len, clamp_pending_drafts_to_model_max_seq_len,
    presample_batched_rejection_frame_ragged_with_policy, resolve_dflash_gumbel_num_drafts,
    stage_batched_rejection_frame_if_eligible_with_policy,
};
use crate::scheduler::stochastic_verify_walk::SpecVerifyPolicy;
use crate::scheduler::{ActiveSeq, ResponseSink, ThinkSpecCtx};
use anyhow::Result;
use spark_model::dflash_sampling::DflashDraftSampleMethod;
use spark_model::speculative::DraftProposerKind;
use spark_model::traits::{Model, SequenceState};
use spark_model::{
    HOST_POLICY_FALLBACK_TOKEN, PackedRejectionWord, REJECTION_BONUS_Q_ROW_IDX,
    REJECTION_NO_MASK_TOKEN, RejectionVerifyRowSpec,
};
use spark_runtime::gpu::DevicePtr;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

const VOCAB: usize = 128;
const THINK_START: u32 = 4;
const THINK_END: u32 = 5;
const TOOL_CALL: u32 = 6;
const EOS_ID: u32 = 127;
const LOGITS_BASE: u64 = 0x30_0000;

fn think_ctx(enabled: bool) -> ThinkSpecCtx<'static> {
    ThinkSpecCtx {
        enabled,
        code_fence_token: None,
        reflection_suppress_ids: &[],
        adaptive_sampling: false,
        honor_eos_inside_thinking: Some(false),
    }
}

fn make_rejection_seq(slot: usize, inside_thinking: bool, seed: u64) -> ActiveSeq {
    let mut a = crate::scheduler::test_fixtures::active_seq(ResponseSink::Blocking(None), slot);
    a.seq.prompt_len = 8;
    a.seq.seq_len = 8;
    a.output_tokens.clear();
    a.temperature = 0.8;
    a.top_k = 20;
    a.top_p = 0.95;
    a.min_p = 0.05;
    a.top_n_sigma = 0.0;
    a.repetition_penalty = 1.0;
    a.presence_penalty = 0.0;
    a.frequency_penalty = 0.0;
    a.lz_penalty = 0.0;
    a.dry_multiplier = 0.0;
    a.logit_bias.clear();
    a.seed = Some(seed);
    a.max_output_tokens = 200;
    a.remaining = 200;
    a.eos_tokens = vec![EOS_ID];
    a.inside_thinking = inside_thinking;
    a.enable_thinking = inside_thinking;
    a.think_ended = !inside_thinking;
    a.think_just_ended = false;
    a.think_start_token = Some(THINK_START);
    a.think_end_token = Some(THINK_END);
    a.tool_call_start_token = Some(TOOL_CALL);
    a
}

struct MockRejectionModel {
    inner: Box<dyn Model>,
    rejection_available: bool,
    sample_method: DflashDraftSampleMethod,
    return_none: bool,
    swap_dims: Option<spark_storage::ModelDims>,
    rejection_calls: AtomicUsize,
    recorded_reqs: Mutex<Vec<Vec<RejectionVerifyRowSpec>>>,
    staged_reqs: Mutex<Vec<Vec<RejectionVerifyRowSpec>>>,
    in_graph_words: Mutex<Option<Vec<PackedRejectionWord>>>,
}

impl MockRejectionModel {
    fn new(rejection_available: bool, return_none: bool) -> Self {
        Self {
            inner: Box::new(crate::scheduler::test_fixtures::model::recording()),
            rejection_available,
            sample_method: DflashDraftSampleMethod::Greedy,
            return_none,
            swap_dims: None,
            rejection_calls: AtomicUsize::new(0),
            recorded_reqs: Mutex::new(Vec::new()),
            staged_reqs: Mutex::new(Vec::new()),
            in_graph_words: Mutex::new(None),
        }
    }
}

impl Model for MockRejectionModel {
    fn prefill(&self, t: &[u32], s: &mut SequenceState, st: u64) -> Result<DevicePtr> {
        self.inner.prefill(t, s, st)
    }
    fn prefill_chunk(
        &self,
        t: &[u32],
        s: &mut SequenceState,
        o: usize,
        l: usize,
        last: bool,
        st: u64,
    ) -> Result<DevicePtr> {
        self.inner.prefill_chunk(t, s, o, l, last, st)
    }
    fn decode(&self, t: u32, s: &mut SequenceState, st: u64) -> Result<DevicePtr> {
        self.inner.decode(t, s, st)
    }
    fn decode_batch(&self, t: &[u32], s: &mut [&mut SequenceState], st: u64) -> Result<DevicePtr> {
        self.inner.decode_batch(t, s, st)
    }
    fn vocab_size(&self) -> usize {
        VOCAB
    }
    fn high_speed_swap_dims(&self) -> Option<spark_storage::ModelDims> {
        self.swap_dims
    }
    fn bind_gpu_to_thread(&self) -> Result<()> {
        Ok(())
    }
    fn synchronize_stream(&self, _: u64) -> Result<()> {
        Ok(())
    }
    fn alloc_sequence(&self) -> Result<SequenceState> {
        self.inner.alloc_sequence()
    }
    fn free_sequence(&self, s: &mut SequenceState) -> Result<()> {
        self.inner.free_sequence(s)
    }
    fn compact_sequence(&self, s: &mut SequenceState, k: usize) -> Result<bool> {
        self.inner.compact_sequence(s, k)
    }
    fn cache_sequence(&self, s: &SequenceState) {
        self.inner.cache_sequence(s)
    }
    fn logits_buffer_ptr(&self) -> DevicePtr {
        DevicePtr(LOGITS_BASE)
    }
    fn copy_logits_to_host(&self, p: DevicePtr, out: &mut [u8]) -> Result<()> {
        self.inner.copy_logits_to_host(p, out)
    }
    fn argmax_on_device(&self, p: DevicePtr, st: u64) -> Result<u32> {
        self.inner.argmax_on_device(p, st)
    }
    fn argmax_batch(&self, p: DevicePtr, n: usize, st: u64) -> Result<Vec<u32>> {
        self.inner.argmax_batch(p, n, st)
    }
    fn proposer_is_dflash(&self) -> bool {
        true
    }
    fn proposer_arm(&self) -> u8 {
        0
    }
    fn proposer_arm_kind(&self, arm: u8) -> Option<DraftProposerKind> {
        (arm == 0).then_some(DraftProposerKind::DFlash)
    }
    fn dflash_draft_sample_method(&self) -> DflashDraftSampleMethod {
        self.sample_method
    }
    fn gpu_rejection_verify_available(&self) -> bool {
        self.rejection_available
    }
    fn sample_rejection_verify_rows_batched(
        &self,
        logits_base: DevicePtr,
        vocab_stride: usize,
        reqs: &[RejectionVerifyRowSpec],
    ) -> Result<Option<Vec<PackedRejectionWord>>> {
        assert_eq!(logits_base, DevicePtr(LOGITS_BASE));
        assert_eq!(vocab_stride, VOCAB);
        self.rejection_calls.fetch_add(1, Ordering::SeqCst);
        self.recorded_reqs.lock().unwrap().push(reqs.to_vec());
        if self.return_none {
            return Ok(None);
        }
        Ok(Some(
            reqs.iter()
                .map(|r| PackedRejectionWord(20 + (r.row_idx as u32 % 50)))
                .collect(),
        ))
    }
    fn stage_next_verify_rejection_rows(&self, reqs: &[RejectionVerifyRowSpec]) {
        self.staged_reqs.lock().unwrap().push(reqs.to_vec());
    }
    fn take_in_graph_rejection_verify_words(&self) -> Option<Vec<PackedRejectionWord>> {
        self.in_graph_words.lock().unwrap().take()
    }
    fn hidden_after_norm(&self) -> DevicePtr {
        DevicePtr::NULL
    }
    fn decode_verify(&self, t: &[u32], s: &mut SequenceState, st: u64) -> Result<Vec<u32>> {
        self.inner.decode_verify(t, s, st)
    }
    fn run_mtp_propose_multi(
        &self,
        tok: u32,
        pos: usize,
        k: usize,
        s: &mut SequenceState,
        st: u64,
        mask: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        Model::run_mtp_propose_multi(&*self.inner, tok, pos, k, s, st, mask)
    }
    fn trim_proposer_state(&self, s: &mut SequenceState, n: usize, st: u64) -> Result<()> {
        self.inner.trim_proposer_state(s, n, st)
    }
    fn decode_verify_graphed(
        &self,
        t: &[u32; 2],
        s: &mut SequenceState,
        st: u64,
    ) -> Result<[u32; 2]> {
        self.inner.decode_verify_graphed(t, s, st)
    }
    fn decode_verify_graphed_k3(
        &self,
        t: &[u32; 3],
        s: &mut SequenceState,
        st: u64,
    ) -> Result<[u32; 3]> {
        self.inner.decode_verify_graphed_k3(t, s, st)
    }
    fn decode_verify_graphed_k4(
        &self,
        t: &[u32; 4],
        s: &mut SequenceState,
        st: u64,
    ) -> Result<[u32; 4]> {
        self.inner.decode_verify_graphed_k4(t, s, st)
    }
    fn checkpoint_ssm_states(&self, s: &mut SequenceState) -> Result<()> {
        self.inner.checkpoint_ssm_states(s)
    }
    fn rollback_ssm_states(&self, s: &mut SequenceState, n: usize) -> Result<()> {
        self.inner.rollback_ssm_states(s, n)
    }
    fn has_ssm_layers(&self) -> bool {
        false
    }
    fn generate_speculative(
        &self,
        p: &[u32],
        sp: &spark_runtime::sampler::SamplingParams,
        nd: usize,
    ) -> Result<spark_model::engine::GenerateResult> {
        self.inner.generate_speculative(p, sp, nd)
    }
    fn has_proposer(&self) -> bool {
        true
    }
    fn has_self_speculative(&self) -> bool {
        false
    }
    fn decode_draft(&self, t: u32, s: &mut SequenceState, st: u64) -> Result<DevicePtr> {
        self.inner.decode_draft(t, s, st)
    }
    fn save_hidden_for_mtp(&self, r: usize, st: u64) -> Result<()> {
        self.inner.save_hidden_for_mtp(r, st)
    }
    fn run_mtp_propose(
        &self,
        t: u32,
        p: usize,
        s: &mut SequenceState,
        st: u64,
    ) -> Result<Option<u32>> {
        self.inner.run_mtp_propose(t, p, s, st)
    }
}

#[test]
fn rejection_policy_eligibility_and_draft_width_follow_kernel_availability() {
    let mut m = MockRejectionModel::new(true, false);
    m.sample_method = DflashDraftSampleMethod::Probabilistic;
    assert!(batched_verify_proposer_eligible_with_policy(
        &m,
        SpecVerifyPolicy::Rejection
    ));
    assert!(!batched_verify_proposer_eligible_with_policy(
        &m,
        SpecVerifyPolicy::Gumbel
    ));
    m.rejection_available = false;
    assert!(!batched_verify_proposer_eligible_with_policy(
        &m,
        SpecVerifyPolicy::Rejection
    ));

    // Non-greedy DFlash under Rejection policy keeps gamma=7 for C<=8 and
    // narrows C>8 (C=16) to gamma=5 to eliminate wasted drafter rows.
    for (live, want_non_greedy) in [(1usize, 7usize), (4, 7), (8, 7), (16, 5)] {
        assert_eq!(
            resolve_dflash_gumbel_num_drafts(
                true,
                SpecVerifyPolicy::Rejection,
                true,
                live,
                7,
                None
            ),
            want_non_greedy,
            "non-greedy Rejection at live={live} must resolve to gamma={want_non_greedy}"
        );
        assert_eq!(
            resolve_dflash_gumbel_num_drafts(
                true,
                SpecVerifyPolicy::Rejection,
                false,
                live,
                7,
                None
            ),
            7,
            "greedy cohort at live={live} keeps gamma=7"
        );
    }
}

#[test]
fn max_seq_len_clamp_trims_only_overshooting_flat_drafts() {
    assert_eq!(clamp_drafts_to_max_seq_len_count(100, 7, 1024), 7);
    assert_eq!(clamp_drafts_to_max_seq_len_count(1020, 7, 1024), 3);
    assert_eq!(clamp_drafts_to_max_seq_len_count(1023, 7, 1024), 0);
    assert_eq!(clamp_drafts_to_max_seq_len_count(1024, 7, 1024), 0);
    assert_eq!(clamp_drafts_to_max_seq_len_count(1020, 7, 0), 7);

    let mut s0 = make_rejection_seq(0, false, 10);
    s0.seq.seq_len = 252;
    s0.pending_drafts = vec![10, 11, 12, 13, 14, 15]; // 252 + 1 + 6 = 259 > 256 -> keep 3
    let mut s1 = make_rejection_seq(1, false, 11);
    s1.seq.seq_len = 64;
    s1.pending_drafts = vec![20, 21, 22, 23, 24, 25]; // fits easily
    let mut s2 = make_rejection_seq(2, false, 12);
    s2.seq.seq_len = 255;
    s2.pending_drafts = vec![30, 31]; // 255 + 1 + 2 = 258 > 256 -> clamp to 0 (K=1)
    let mut active = [s0, s1, s2];

    clamp_pending_drafts_for_max_seq_len(&mut active, &[0, 1, 2], 256);
    assert_eq!(active[0].pending_drafts, vec![10, 11, 12]);
    assert_eq!(active[1].pending_drafts, vec![20, 21, 22, 23, 24, 25]);
    assert!(active[2].pending_drafts.is_empty());

    // Model-driven helper derives ceiling from `high_speed_swap_dims()`.
    let mut model = MockRejectionModel::new(true, false);
    model.swap_dims = Some(spark_storage::ModelDims {
        num_layers: 4,
        max_blocks_per_layer: 16,
        num_q_heads: 4,
        num_kv_heads: 2,
        head_dim: 64,
        block_size: 16,
    }); // ceiling = 256
    active[0].pending_drafts = vec![10, 11, 12, 13, 14];
    clamp_pending_drafts_to_model_max_seq_len(&model, &mut active, &[0, 1]);
    assert_eq!(active[0].pending_drafts, vec![10, 11, 12]);
}

#[test]
fn batched_rejection_staging_in_graph_consumption_and_eager_fallback() {
    let ctx = think_ctx(true);

    // 1. Mixed cohort (thinking Rejection + forced-end HostPolicySkip + greedy + post-think content Rejection)
    let s0 = make_rejection_seq(0, true, 111); // K=4, row_base=0 (thinking, no mask)
    let mut s1 = make_rejection_seq(1, true, 222);
    s1.force_end_thinking = true; // K=2, row_base=4 (HostPolicySkip -> temp=-1.0)
    let mut s2 = make_rejection_seq(2, false, 333);
    s2.temperature = 0.0; // K=3, row_base=6 (GreedyArgmax -> temp=0.0)
    let s3 = make_rejection_seq(3, false, 444); // K=2, row_base=9 (post-think -> mask think_start/end)
    let active = [s0, s1, s2, s3];
    let idxs = [0usize, 1, 2, 3];
    let drafts_per_seq = vec![vec![10, 11, 12], vec![20], vec![30, 31], vec![40]];
    let k_per_seq = [4usize, 2, 3, 2];
    let row_bases = [0usize, 4, 6, 9];
    let model = MockRejectionModel::new(true, false);

    let staged_mask = stage_batched_rejection_frame_if_eligible_with_policy(
        &model,
        &active,
        &idxs,
        &drafts_per_seq,
        &k_per_seq,
        &row_bases,
        11,
        &ctx,
        SpecVerifyPolicy::Rejection,
    );
    assert_eq!(staged_mask, Some(vec![true, false, false, true]));
    {
        let staged_all = model.staged_reqs.lock().unwrap();
        assert_eq!(staged_all.len(), 1);
        let staged = &staged_all[0];
        assert_eq!(staged.len(), 11);
        assert_eq!(
            staged.iter().map(|r| r.row_idx).collect::<Vec<_>>(),
            (0..11).collect::<Vec<_>>()
        );
        // Seq 0 (thinking): temp > 0, no think-tag mask.
        assert!(staged[0..4].iter().all(|r| r.temperature > 0.0
            && r.mask_tok0 == REJECTION_NO_MASK_TOKEN
            && r.mask_tok1 == REJECTION_NO_MASK_TOKEN));
        // Seq 1 (force_end_thinking): temp == -1.0, bonus q_row_idx.
        assert!(
            staged[4..6]
                .iter()
                .all(|r| r.temperature == -1.0 && r.q_row_idx == REJECTION_BONUS_Q_ROW_IDX)
        );
        // Seq 2 (greedy): temp == 0.0, draft rows carry slot*8+r, bonus row carries 0xFFFF.
        assert!(staged[6..9].iter().all(|r| r.temperature == 0.0));
        assert_eq!(staged[6].q_row_idx, 2 * 8);
        assert_eq!(staged[6].draft_token, 30);
        assert_eq!(staged[7].q_row_idx, 2 * 8 + 1);
        assert_eq!(staged[7].draft_token, 31);
        assert_eq!(staged[8].q_row_idx, REJECTION_BONUS_Q_ROW_IDX);
        // Seq 3 (post-think content): temp > 0 and masks (THINK_START, THINK_END).
        assert!(staged[9..11].iter().all(|r| r.temperature > 0.0
            && r.mask_tok0 == THINK_START
            && r.mask_tok1 == THINK_END));
    }

    // In-graph consumption: 0 eager GPU launches.
    *model.in_graph_words.lock().unwrap() = Some(vec![
        PackedRejectionWord((1 << 31) | 10),
        PackedRejectionWord((1 << 31) | 11),
        PackedRejectionWord(19),
        PackedRejectionWord(21),
        PackedRejectionWord(HOST_POLICY_FALLBACK_TOKEN),
        PackedRejectionWord(HOST_POLICY_FALLBACK_TOKEN),
        PackedRejectionWord((1 << 31) | 30),
        PackedRejectionWord(39),
        PackedRejectionWord(41),
        PackedRejectionWord((1 << 31) | 40),
        PackedRejectionWord(45),
    ]);
    let frame = presample_batched_rejection_frame_ragged_with_policy(
        &model,
        &active,
        &idxs,
        &drafts_per_seq,
        &k_per_seq,
        &row_bases,
        &ctx,
        staged_mask.as_deref(),
        SpecVerifyPolicy::Rejection,
    )
    .unwrap()
    .expect("mixed cohort must consume in-graph rejection words");
    assert_eq!(model.rejection_calls.load(Ordering::SeqCst), 0);
    assert_eq!(frame.seq_offsets, vec![Some(0), None, None, Some(9)]);
    assert_eq!(frame.words[0], PackedRejectionWord((1 << 31) | 10));
    assert_eq!(frame.words[9], PackedRejectionWord((1 << 31) | 40));

    // 2. Eager fallback (in-graph words None): calls sample_rejection_verify_rows_batched
    //    once over ONLY eligible sequences (seq 0 rows 0..4 and seq 3 rows 9..11).
    let fallback = presample_batched_rejection_frame_ragged_with_policy(
        &model,
        &active,
        &idxs,
        &drafts_per_seq,
        &k_per_seq,
        &row_bases,
        &ctx,
        staged_mask.as_deref(),
        SpecVerifyPolicy::Rejection,
    )
    .unwrap()
    .expect("eager fallback must launch batched rejection verify");
    assert_eq!(model.rejection_calls.load(Ordering::SeqCst), 1);
    assert_eq!(fallback.seq_offsets, vec![Some(0), None, None, Some(4)]);
    assert_eq!(fallback.words.len(), 6);
    {
        let recorded = model.recorded_reqs.lock().unwrap();
        assert_eq!(recorded.len(), 1);
        let row_idxs: Vec<usize> = recorded[0].iter().map(|r| r.row_idx).collect();
        assert_eq!(row_idxs, vec![0, 1, 2, 3, 9, 10]);
        assert_eq!(recorded[0][4].mask_tok0, THINK_START);
        assert_eq!(recorded[0][4].mask_tok1, THINK_END);
    }

    // 3. Non-Rejection policy or unavailable kernel clears staging with `&[]` and returns None.
    let cleared = stage_batched_rejection_frame_if_eligible_with_policy(
        &model,
        &active,
        &idxs,
        &drafts_per_seq,
        &k_per_seq,
        &row_bases,
        11,
        &ctx,
        SpecVerifyPolicy::Gumbel,
    );
    assert_eq!(cleared, None);
    assert!(model.staged_reqs.lock().unwrap().last().unwrap().is_empty());

    let unavailable = MockRejectionModel::new(false, false);
    assert_eq!(
        stage_batched_rejection_frame_if_eligible_with_policy(
            &unavailable,
            &active,
            &idxs,
            &drafts_per_seq,
            &k_per_seq,
            &row_bases,
            11,
            &ctx,
            SpecVerifyPolicy::Rejection,
        ),
        None
    );
    assert!(
        unavailable
            .staged_reqs
            .lock()
            .unwrap()
            .last()
            .unwrap()
            .is_empty()
    );
    assert!(
        presample_batched_rejection_frame_ragged_with_policy(
            &unavailable,
            &active,
            &idxs,
            &drafts_per_seq,
            &k_per_seq,
            &row_bases,
            &ctx,
            None,
            SpecVerifyPolicy::Rejection,
        )
        .unwrap()
        .is_none()
    );
}
