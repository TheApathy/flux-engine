// SPDX-License-Identifier: AGPL-3.0-only

use super::{
    presample_batched_gumbel_frame_ragged_with_policy, presample_batched_gumbel_frame_with_policy,
    stage_batched_gumbel_frame_if_eligible_with_policy,
};
use crate::scheduler::spec_policy_accept::{
    batched_gumbel_seq_eligible, presample_single_seq_gumbel_with_policy,
    presampled_content_row_ok, run_dflash_content_accept_at_row_with_presampled,
    stage_single_seq_gumbel_if_eligible_with_policy,
};
use crate::scheduler::stochastic_verify_walk::{
    SpecVerifyPolicy, run_gumbel_thinking_accept_at_row,
};
use crate::scheduler::think_spec_accept::{
    presampled_thinking_row_ok, run_dflash_thinking_accept_at_row_with_presampled,
};
use crate::scheduler::{ActiveSeq, ResponseSink, ThinkSpecCtx};
use anyhow::Result;
use spark_model::traits::{Model, SequenceState};
use spark_runtime::gpu::DevicePtr;
use spark_runtime::sampler::device_topk::GumbelVerifyRowSpec;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

const VOCAB: usize = 128;
const THINK_START: u32 = 4;
const THINK_END: u32 = 5;
const TOOL_CALL: u32 = 6;
const EOS_ID: u32 = 127;
const LOGITS_BASE: u64 = 0x20_0000;

fn think_ctx<'a>(enabled: bool, suppress: &'a [u32], adaptive: bool) -> ThinkSpecCtx<'a> {
    ThinkSpecCtx {
        enabled,
        code_fence_token: None,
        reflection_suppress_ids: suppress,
        adaptive_sampling: adaptive,
        honor_eos_inside_thinking: Some(false),
    }
}

fn make_gumbel_seq(slot: usize, inside_thinking: bool, seed: u64) -> ActiveSeq {
    let mut a = crate::scheduler::test_fixtures::active_seq(ResponseSink::Blocking(None), slot);
    a.seq.prompt_len = 8;
    a.seq.seq_len = 8;
    a.output_tokens.clear();
    a.temperature = 0.8;
    a.top_k = 20;
    a.top_p = 0.95;
    a.min_p = 0.0;
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

fn bf16_row_with_masked_leader(masked: u32, fallback_tok: u32) -> Vec<u8> {
    let mut f32s = vec![-8.0f32; VOCAB];
    f32s[masked as usize] = 50.0;
    f32s[fallback_tok as usize] = 40.0;
    f32s.into_iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

struct MockGumbelModel {
    inner: Box<dyn Model>,
    rows_bf16: Vec<Vec<u8>>,
    gumbel_calls: AtomicUsize,
    recorded_reqs: Mutex<Vec<Vec<GumbelVerifyRowSpec>>>,
    staged_reqs: Mutex<Vec<Vec<GumbelVerifyRowSpec>>>,
    in_graph_tokens: Mutex<Option<Vec<u32>>>,
    d2h_calls: AtomicUsize,
    return_none: bool,
}

impl MockGumbelModel {
    fn new(rows_bf16: Vec<Vec<u8>>, return_none: bool) -> Self {
        Self {
            inner: Box::new(crate::scheduler::test_fixtures::model::recording()),
            rows_bf16,
            gumbel_calls: AtomicUsize::new(0),
            recorded_reqs: Mutex::new(Vec::new()),
            staged_reqs: Mutex::new(Vec::new()),
            in_graph_tokens: Mutex::new(None),
            d2h_calls: AtomicUsize::new(0),
            return_none,
        }
    }
}

impl Model for MockGumbelModel {
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
    fn copy_logits_to_host(&self, ptr: DevicePtr, out: &mut [u8]) -> Result<()> {
        self.d2h_calls.fetch_add(1, Ordering::SeqCst);
        let row_bytes = VOCAB * 2;
        let byte_off = (ptr.0 - LOGITS_BASE) as usize;
        let start_row = byte_off / row_bytes;
        let count_rows = out.len() / row_bytes;
        for r in 0..count_rows {
            let src = &self.rows_bf16[start_row + r];
            out[r * row_bytes..(r + 1) * row_bytes].copy_from_slice(src);
        }
        Ok(())
    }
    fn argmax_on_device(&self, p: DevicePtr, st: u64) -> Result<u32> {
        self.inner.argmax_on_device(p, st)
    }
    fn argmax_batch(&self, p: DevicePtr, n: usize, st: u64) -> Result<Vec<u32>> {
        self.inner.argmax_batch(p, n, st)
    }
    fn sample_gumbel_verify_rows_batched(
        &self,
        logits_base: DevicePtr,
        vocab_stride: usize,
        reqs: &[GumbelVerifyRowSpec],
    ) -> Result<Option<Vec<u32>>> {
        assert_eq!(logits_base, DevicePtr(LOGITS_BASE));
        assert_eq!(vocab_stride, VOCAB);
        self.gumbel_calls.fetch_add(1, Ordering::SeqCst);
        self.recorded_reqs.lock().unwrap().push(reqs.to_vec());
        if self.return_none {
            return Ok(None);
        }
        Ok(Some(
            reqs.iter().map(|r| 20 + (r.row_idx as u32 % 50)).collect(),
        ))
    }
    fn stage_next_verify_gumbel_rows(&self, reqs: &[GumbelVerifyRowSpec]) {
        self.staged_reqs.lock().unwrap().push(reqs.to_vec());
    }
    fn take_in_graph_gumbel_verify_tokens(&self) -> Option<Vec<u32>> {
        self.in_graph_tokens.lock().unwrap().take()
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
fn batched_gumbel_eligibility_enforces_every_lossless_guard() {
    let ctx = think_ctx(true, &[], false);
    let thinking = make_gumbel_seq(0, true, 100);
    let content = make_gumbel_seq(1, false, 200);
    assert!(batched_gumbel_seq_eligible(&thinking, 5, &ctx));
    assert!(batched_gumbel_seq_eligible(&content, 5, &ctx));

    // Full-vocab with top_p >= 1.0 is allowed; top_k==0 with top_p < 1.0 is not.
    let mut full_vocab = make_gumbel_seq(1, false, 200);
    full_vocab.top_k = 0;
    full_vocab.top_p = 1.0;
    assert!(batched_gumbel_seq_eligible(&full_vocab, 5, &ctx));
    full_vocab.top_p = 0.95;
    assert!(!batched_gumbel_seq_eligible(&full_vocab, 5, &ctx));

    // Every single lossless exclusion must reject in both content and thinking.
    let check_ineligible = |mutate: fn(&mut ActiveSeq), label: &str| {
        let mut sc = make_gumbel_seq(1, false, 200);
        mutate(&mut sc);
        assert!(
            !batched_gumbel_seq_eligible(&sc, 5, &ctx),
            "expected ineligible in content: {label}"
        );
        let mut st = make_gumbel_seq(0, true, 100);
        mutate(&mut st);
        assert!(
            !batched_gumbel_seq_eligible(&st, 5, &ctx),
            "expected ineligible in thinking: {label}"
        );
    };
    check_ineligible(|s| s.temperature = 0.0, "greedy");
    check_ineligible(|s| s.temperature = f32::NAN, "nan_temp");
    check_ineligible(|s| s.min_p = 0.05, "min_p");
    check_ineligible(|s| s.top_n_sigma = 1.5, "top_n_sigma");
    check_ineligible(|s| s.presence_penalty = 0.2, "presence_penalty");
    check_ineligible(|s| s.frequency_penalty = 0.1, "frequency_penalty");
    check_ineligible(|s| s.repetition_penalty = 1.1, "repetition_penalty");
    check_ineligible(|s| s.lz_penalty = 0.3, "lz_penalty");
    check_ineligible(|s| s.dry_multiplier = 0.5, "dry_multiplier");
    check_ineligible(|s| s.logit_bias = vec![(10, -2.0)], "logit_bias");
    check_ineligible(|s| s.top_logprobs = Some(2), "logprobs");
    check_ineligible(|s| s.top_p = 0.0, "top_p_zero");
    check_ineligible(|s| s.top_k = 257, "top_k_over_256");

    // Context-level guards.
    assert!(!batched_gumbel_seq_eligible(
        &thinking,
        5,
        &think_ctx(true, &[9], false)
    ));
    assert!(!batched_gumbel_seq_eligible(
        &thinking,
        5,
        &think_ctx(true, &[], true)
    ));
    assert!(!batched_gumbel_seq_eligible(
        &thinking,
        5,
        &think_ctx(false, &[], false)
    ));

    // Thinking budget exact boundary: `thinking_tokens + k_verify <= budget` is ok; `>` is not.
    let mut budgeted = make_gumbel_seq(0, true, 100);
    budgeted.thinking_budget = Some(100);
    budgeted.thinking_tokens = 95;
    assert!(batched_gumbel_seq_eligible(&budgeted, 5, &ctx));
    budgeted.thinking_tokens = 96;
    assert!(!batched_gumbel_seq_eligible(&budgeted, 5, &ctx));

    // Thinking answer-runway exact boundary (`max_output_tokens = 200` -> runway 100):
    // frames approaching cutoff stay eligible (mid-frame crossing is caught row-by-row
    // by `emit_step` + `presampled_thinking_row_ok`), while row 0 at/past cutoff rejects.
    let mut runway = make_gumbel_seq(0, true, 100);
    runway.thinking_budget = None;
    runway.max_output_tokens = 200;
    runway.output_tokens = vec![10; 99];
    assert!(batched_gumbel_seq_eligible(&runway, 5, &ctx));
    runway.output_tokens = vec![10; 100];
    assert!(!batched_gumbel_seq_eligible(&runway, 5, &ctx));

    // Content phase-specific guards.
    let mut c = make_gumbel_seq(1, false, 200);
    c.think_just_ended = true;
    assert!(!batched_gumbel_seq_eligible(&c, 5, &ctx));
    c.think_just_ended = false;
    c.suppress_tool_call = true;
    assert!(!batched_gumbel_seq_eligible(&c, 5, &ctx));
    c.suppress_tool_call = false;
    c.require_tool_call = true;
    assert!(!batched_gumbel_seq_eligible(&c, 5, &ctx));
    c.require_tool_call = false;
    c.min_tokens = 10;
    assert!(!batched_gumbel_seq_eligible(&c, 5, &ctx));
}

#[test]
fn presampled_dynamic_guards_invalidate_on_state_transition_and_masked_tokens() {
    let ctx = think_ctx(true, &[], false);
    let mut t = make_gumbel_seq(0, true, 10);
    let pos = crate::scheduler::stochastic_verify_walk::seq_abs_pos(&t);
    assert!(presampled_thinking_row_ok(&t, 12, VOCAB, pos, &ctx));
    assert!(!presampled_thinking_row_ok(&t, TOOL_CALL, VOCAB, pos, &ctx));
    assert!(!presampled_thinking_row_ok(
        &t,
        VOCAB as u32,
        VOCAB,
        pos,
        &ctx
    ));
    assert!(!presampled_thinking_row_ok(&t, 12, VOCAB, pos + 1, &ctx));
    t.force_end_thinking = true;
    assert!(!presampled_thinking_row_ok(&t, 12, VOCAB, pos, &ctx));
    t.force_end_thinking = false;
    t.inside_thinking = false;
    assert!(!presampled_thinking_row_ok(&t, 12, VOCAB, pos, &ctx));

    let mut c = make_gumbel_seq(1, false, 20);
    let cpos = crate::scheduler::stochastic_verify_walk::seq_abs_pos(&c);
    assert!(presampled_content_row_ok(&c, 12, VOCAB, cpos, &ctx));
    assert!(!presampled_content_row_ok(
        &c,
        THINK_START,
        VOCAB,
        cpos,
        &ctx
    ));
    assert!(!presampled_content_row_ok(&c, THINK_END, VOCAB, cpos, &ctx));
    assert!(!presampled_content_row_ok(
        &c,
        VOCAB as u32,
        VOCAB,
        cpos,
        &ctx
    ));
    assert!(!presampled_content_row_ok(&c, 12, VOCAB, cpos + 1, &ctx));
    c.suppress_tool_call = true;
    assert!(!presampled_content_row_ok(&c, 12, VOCAB, cpos, &ctx));
}

#[test]
fn presample_batched_gumbel_frame_collects_eligible_rows_in_one_call() {
    let ctx = think_ctx(true, &[], false);
    let s0 = make_gumbel_seq(0, true, 111);
    let mut s1 = make_gumbel_seq(1, false, 222);
    s1.presence_penalty = 0.5; // ineligible
    let s2 = make_gumbel_seq(2, false, 333);
    let active = [s0, s1, s2];
    let idxs = [0usize, 1, 2];
    let model = MockGumbelModel::new(Vec::new(), false);

    // Non-Gumbel policy does zero GPU calls.
    assert!(
        presample_batched_gumbel_frame_with_policy(
            &model,
            &active,
            &idxs,
            5,
            &ctx,
            SpecVerifyPolicy::Argmax,
        )
        .unwrap()
        .is_none()
    );
    assert_eq!(model.gumbel_calls.load(Ordering::SeqCst), 0);

    // Gumbel policy launches ONE call with 10 rows: seq 0 (`0..5`) + seq 2 (`10..15`).
    let frame = presample_batched_gumbel_frame_with_policy(
        &model,
        &active,
        &idxs,
        5,
        &ctx,
        SpecVerifyPolicy::Gumbel,
    )
    .unwrap()
    .expect("batched gumbel frame must engage");
    assert_eq!(model.gumbel_calls.load(Ordering::SeqCst), 1);
    assert_eq!(frame.seq_offsets, vec![Some(0), None, Some(5)]);
    assert_eq!(frame.tokens.len(), 10);
    let recorded = model.recorded_reqs.lock().unwrap();
    let row_idxs: Vec<usize> = recorded[0].iter().map(|r| r.row_idx).collect();
    assert_eq!(row_idxs, vec![0, 1, 2, 3, 4, 10, 11, 12, 13, 14]);
    assert_eq!(recorded[0][0].abs_pos, 8);
    assert_eq!(recorded[0][4].abs_pos, 12);
    assert_eq!(recorded[0][5].abs_pos, 8);
    assert_eq!(recorded[0][9].abs_pos, 12);

    // Unsupported model (`Ok(None)`) returns `Ok(None)` cleanly.
    let unsupported = MockGumbelModel::new(Vec::new(), true);
    assert!(
        presample_batched_gumbel_frame_with_policy(
            &unsupported,
            &active,
            &idxs,
            5,
            &ctx,
            SpecVerifyPolicy::Gumbel,
        )
        .unwrap()
        .is_none()
    );
}

#[test]
fn presample_batched_gumbel_frame_ragged_indexes_cumulative_row_bases_and_k1() {
    let ctx = think_ctx(true, &[], false);
    let s0 = make_gumbel_seq(0, true, 111); // K=7, row_base=0 (eligible)
    let mut s1 = make_gumbel_seq(1, false, 222);
    s1.presence_penalty = 0.5; // K=3, row_base=7 (ineligible -> skipped)
    let s2 = make_gumbel_seq(2, false, 333); // K=1, row_base=10 (anchor-only eligible)
    let s3 = make_gumbel_seq(3, false, 444); // K=5, row_base=11 (eligible)
    let active = [s0, s1, s2, s3];
    let idxs = [0usize, 1, 2, 3];
    let k_per_seq = [7usize, 3, 1, 5];
    let row_bases = [0usize, 7, 10, 11];
    let model = MockGumbelModel::new(Vec::new(), false);

    let frame = presample_batched_gumbel_frame_ragged_with_policy(
        &model,
        &active,
        &idxs,
        &k_per_seq,
        &row_bases,
        &ctx,
        None,
        SpecVerifyPolicy::Gumbel,
    )
    .unwrap()
    .expect("ragged batched gumbel frame must engage");

    assert_eq!(model.gumbel_calls.load(Ordering::SeqCst), 1);
    assert_eq!(frame.seq_offsets, vec![Some(0), None, Some(7), Some(8)]);
    assert_eq!(frame.tokens.len(), 7 + 1 + 5);
    let recorded = model.recorded_reqs.lock().unwrap();
    let row_idxs: Vec<usize> = recorded[0].iter().map(|r| r.row_idx).collect();
    assert_eq!(row_idxs, vec![0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15]);
    // Verify abs_pos resets cleanly for each sequence (K=1 has single abs_pos=8).
    assert_eq!(recorded[0][0].abs_pos, 8);
    assert_eq!(recorded[0][6].abs_pos, 14);
    assert_eq!(recorded[0][7].abs_pos, 8);
    assert_eq!(recorded[0][8].abs_pos, 8);
    assert_eq!(recorded[0][12].abs_pos, 12);
}

#[test]
fn presampled_walk_uses_zero_d2h_and_falls_back_cleanly_on_boundary_or_mask() {
    let ctx = think_ctx(true, &[], false);

    // A. Normal thinking walk accepting 2 drafts + bonus -> 0 logits D2H!
    {
        let model = MockGumbelModel::new(Vec::new(), false);
        let mut a = make_gumbel_seq(0, true, 10);
        let out = run_dflash_thinking_accept_at_row_with_presampled(
            &model,
            &mut a,
            0,
            &[10, 11],
            &[99, 99, 99],
            &ctx,
            Some(&[10, 11, 12]),
        )
        .unwrap();
        assert_eq!(model.d2h_calls.load(Ordering::SeqCst), 0);
        assert_eq!(
            (out.num_accepted, out.bonus, out.consumed_rows),
            (2, Some(12), 3)
        );
        assert_eq!(a.output_tokens, vec![10, 11, 12]);
        assert!(a.inside_thinking);
    }

    // B. `</think>` boundary ends thinking cleanly on that row with 0 logits D2H.
    {
        let model = MockGumbelModel::new(Vec::new(), false);
        let mut a = make_gumbel_seq(0, true, 11);
        let out = run_dflash_thinking_accept_at_row_with_presampled(
            &model,
            &mut a,
            0,
            &[10, THINK_END],
            &[99, 99, 99],
            &ctx,
            Some(&[10, THINK_END, 25]),
        )
        .unwrap();
        assert_eq!(model.d2h_calls.load(Ordering::SeqCst), 0);
        assert_eq!(
            (out.num_accepted, out.bonus, out.consumed_rows),
            (1, Some(THINK_END), 2)
        );
        assert!(!a.inside_thinking);
        assert!(a.think_ended);
        assert_eq!(a.output_tokens, vec![10, THINK_END]);
    }

    // C. Masked `<tool_call>` in pre-sampled row 1 invalidates and falls back to D2H.
    {
        let rows = vec![
            bf16_row_with_masked_leader(TOOL_CALL, 31),
            bf16_row_with_masked_leader(TOOL_CALL, 33),
            bf16_row_with_masked_leader(TOOL_CALL, 35),
        ];
        let model = MockGumbelModel::new(rows, false);
        let mut a = make_gumbel_seq(0, true, 12);
        let out = run_dflash_thinking_accept_at_row_with_presampled(
            &model,
            &mut a,
            0,
            &[10, 15],
            &[99, 99, 99],
            &ctx,
            Some(&[10, TOOL_CALL, 20]),
        )
        .unwrap();
        assert_eq!(model.d2h_calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            (out.num_accepted, out.bonus, out.consumed_rows),
            (1, Some(33), 2)
        );
        assert_eq!(a.output_tokens, vec![10, 33]);
    }

    // D. Masked `</think>` in post-think content row 0 invalidates and falls back to D2H.
    {
        let rows = vec![
            bf16_row_with_masked_leader(THINK_END, 44),
            bf16_row_with_masked_leader(THINK_END, 45),
        ];
        let model = MockGumbelModel::new(rows, false);
        let mut a = make_gumbel_seq(1, false, 13);
        let out = run_dflash_content_accept_at_row_with_presampled(
            &model,
            &mut a,
            0,
            &[25],
            &[99, 99],
            &ctx,
            Some(&[THINK_END, 25]),
        )
        .unwrap();
        assert_eq!(model.d2h_calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            (out.num_accepted, out.bonus, out.consumed_rows),
            (0, Some(44), 1)
        );
        assert_eq!(a.output_tokens, vec![44]);
    }
}

#[test]
fn in_graph_gumbel_staging_consumes_tokens_with_zero_extra_launch_and_falls_back_cleanly() {
    let ctx = think_ctx(true, &[], false);

    // 1. Batched ragged cohort: all eligible -> stages once, consumes in-graph with 0 extra launches.
    {
        let s0 = make_gumbel_seq(0, true, 111);
        let s1 = make_gumbel_seq(1, false, 222);
        let active = [s0, s1];
        let idxs = [0usize, 1];
        let k_per_seq = [4usize, 2];
        let row_bases = [0usize, 4];
        let model = MockGumbelModel::new(Vec::new(), false);

        let staged_mask = stage_batched_gumbel_frame_if_eligible_with_policy(
            &model,
            &active,
            &idxs,
            &k_per_seq,
            &row_bases,
            6,
            &ctx,
            SpecVerifyPolicy::Gumbel,
        );
        assert_eq!(staged_mask, Some(vec![true, true]));
        {
            let staged_all = model.staged_reqs.lock().unwrap();
            assert_eq!(staged_all.len(), 1);
            let staged = &staged_all[0];
            assert_eq!(staged.len(), 6);
            assert_eq!(
                staged.iter().map(|r| r.row_idx).collect::<Vec<_>>(),
                vec![0, 1, 2, 3, 4, 5]
            );
        }

        *model.in_graph_tokens.lock().unwrap() = Some(vec![70, 71, 72, 73, 80, 81]);
        let frame = presample_batched_gumbel_frame_ragged_with_policy(
            &model,
            &active,
            &idxs,
            &k_per_seq,
            &row_bases,
            &ctx,
            staged_mask.as_deref(),
            SpecVerifyPolicy::Gumbel,
        )
        .unwrap()
        .expect("in-graph tokens must be consumed");
        assert_eq!(model.gumbel_calls.load(Ordering::SeqCst), 0);
        assert_eq!(frame.tokens, vec![70, 71, 72, 73, 80, 81]);
        assert_eq!(frame.seq_offsets, vec![Some(0), Some(4)]);
    }

    // 2. Mixed cohort (Gumbel + host-policy/budget_nudge skip + greedy T==0):
    //    stages per-seq `temp > 0`, `temp == -1.0`, `temp == 0.0` in ONE unified
    //    frame, consuming in-graph tokens for eligible seqs with 0 extra GPU launches!
    {
        let s0 = make_gumbel_seq(0, true, 111); // K=4, row_base=0 (Gumbel)
        let mut s1 = make_gumbel_seq(1, true, 222);
        s1.force_end_thinking = true; // K=2, row_base=4 (HostPolicySkip -> temp=-1.0)
        let mut s2 = make_gumbel_seq(2, false, 333);
        s2.temperature = 0.0; // K=3, row_base=6 (GreedyArgmax -> temp=0.0)
        let s3 = make_gumbel_seq(3, false, 444); // K=2, row_base=9 (Gumbel)
        let active = [s0, s1, s2, s3];
        let idxs = [0usize, 1, 2, 3];
        let k_per_seq = [4usize, 2, 3, 2];
        let row_bases = [0usize, 4, 6, 9];
        let model = MockGumbelModel::new(Vec::new(), false);

        let staged_mask = stage_batched_gumbel_frame_if_eligible_with_policy(
            &model,
            &active,
            &idxs,
            &k_per_seq,
            &row_bases,
            11,
            &ctx,
            SpecVerifyPolicy::Gumbel,
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
            assert!(staged[0..4].iter().all(|r| r.temperature > 0.0));
            assert!(staged[4..6].iter().all(|r| r.temperature == -1.0));
            assert!(staged[6..9].iter().all(|r| r.temperature == 0.0));
            assert!(staged[9..11].iter().all(|r| r.temperature > 0.0));
        }

        // Simulate unified GPU kernel output: valid tokens for Gumbel & greedy rows,
        // `u32::MAX` sentinel for the skipped `force_end_thinking` rows (4..6).
        *model.in_graph_tokens.lock().unwrap() =
            Some(vec![10, 11, 12, 13, u32::MAX, u32::MAX, 30, 31, 32, 40, 41]);
        let frame = presample_batched_gumbel_frame_ragged_with_policy(
            &model,
            &active,
            &idxs,
            &k_per_seq,
            &row_bases,
            &ctx,
            staged_mask.as_deref(),
            SpecVerifyPolicy::Gumbel,
        )
        .unwrap()
        .expect("mixed cohort must consume in-graph Gumbel tokens for eligible seqs");
        assert_eq!(
            model.gumbel_calls.load(Ordering::SeqCst),
            0,
            "mixed cohort must consume in-graph tokens with zero extra GPU launches"
        );
        assert_eq!(frame.seq_offsets, vec![Some(0), None, None, Some(9)]);
        assert_eq!(&frame.tokens[0..4], &[10, 11, 12, 13]);
        assert_eq!(&frame.tokens[9..11], &[40, 41]);
    }

    // 2b. Pure greedy cohort -> skips heap staging (verify_qwen4_batched auto-stages temp=0.0).
    {
        let mut g0 = make_gumbel_seq(0, false, 111);
        g0.temperature = 0.0;
        let mut g1 = make_gumbel_seq(1, false, 222);
        g1.temperature = 0.0;
        let active = [g0, g1];
        let model = MockGumbelModel::new(Vec::new(), false);
        let staged_mask = stage_batched_gumbel_frame_if_eligible_with_policy(
            &model,
            &active,
            &[0, 1],
            &[4, 2],
            &[0, 4],
            6,
            &ctx,
            SpecVerifyPolicy::Gumbel,
        );
        assert_eq!(staged_mask, None);
        assert!(model.staged_reqs.lock().unwrap().is_empty());
    }

    // 3. C=1 single-sequence staging (eligible Gumbel + budget-nudge skip) &
    //    zero-extra-launch consumption (row_base == 0 only).
    {
        let a = make_gumbel_seq(0, false, 333);
        let model = MockGumbelModel::new(Vec::new(), false);
        stage_single_seq_gumbel_if_eligible_with_policy(
            &model,
            &a,
            5,
            &ctx,
            SpecVerifyPolicy::Gumbel,
        );
        assert_eq!(model.staged_reqs.lock().unwrap().len(), 1);
        assert_eq!(model.staged_reqs.lock().unwrap()[0].len(), 5);
        assert!(
            model.staged_reqs.lock().unwrap()[0]
                .iter()
                .all(|r| r.temperature > 0.0)
        );

        // Ineligible T>0 C=1 (e.g. force_end_thinking) stages temp=-1.0 skip so the
        // C=1 CUDA graph key stays constant across token 128 budget nudge.
        let mut nudged = make_gumbel_seq(0, true, 333);
        nudged.force_end_thinking = true;
        stage_single_seq_gumbel_if_eligible_with_policy(
            &model,
            &nudged,
            5,
            &ctx,
            SpecVerifyPolicy::Gumbel,
        );
        {
            let staged_all = model.staged_reqs.lock().unwrap();
            assert_eq!(staged_all.len(), 2);
            assert_eq!(staged_all[1].len(), 5);
            assert!(staged_all[1].iter().all(|r| r.temperature == -1.0));
        }

        *model.in_graph_tokens.lock().unwrap() = Some(vec![41, 42, 43, 44, 45]);
        let tokens = presample_single_seq_gumbel_with_policy(
            &model,
            &a,
            0,
            5,
            &ctx,
            SpecVerifyPolicy::Gumbel,
        )
        .unwrap()
        .expect("C=1 in-graph tokens must be consumed");
        assert_eq!(model.gumbel_calls.load(Ordering::SeqCst), 0);
        assert_eq!(tokens, vec![41, 42, 43, 44, 45]);

        // When eager fallback yields None from take_in_graph_gumbel_verify_tokens,
        // presample_single_seq_gumbel falls back to sample_gumbel_verify_rows_batched.
        let fallback = presample_single_seq_gumbel_with_policy(
            &model,
            &a,
            0,
            5,
            &ctx,
            SpecVerifyPolicy::Gumbel,
        )
        .unwrap()
        .expect("fallback batched launch must succeed");
        assert_eq!(model.gumbel_calls.load(Ordering::SeqCst), 1);
        assert_eq!(fallback.len(), 5);
    }
}

#[test]
fn run_gumbel_thinking_accept_at_row_performs_lazy_per_row_d2h() {
    let ctx = think_ctx(true, &[], false);

    // A. Forced budget nudge + armed `</think>` across 5 verify rows: ZERO D2H calls!
    {
        let model = MockGumbelModel::new(Vec::new(), false);
        let mut a = make_gumbel_seq(0, true, 55);
        a.max_output_tokens = 64;
        a.remaining = 64;
        a.force_end_thinking = true;
        a.budget_nudge = crate::scheduler::think_budget_nudge::BudgetNudge::plan(
            Some(std::sync::Arc::from([30u32, 31, 32])),
            0,
            64,
        );
        let out =
            run_gumbel_thinking_accept_at_row(&model, &mut a, 0, &[30, 31, 32, THINK_END], 5, &ctx);
        assert_eq!(model.d2h_calls.load(Ordering::SeqCst), 0);
        assert_eq!(
            (out.num_accepted, out.bonus, out.consumed_rows),
            (3, Some(THINK_END), 4)
        );
        assert_eq!(a.output_tokens, vec![30, 31, 32, THINK_END]);
    }

    // B. Unforced row 0 mismatch over a 5-row verify frame: copies ONLY row 0 (1 D2H).
    //    `rows_bf16` only supplies row 0, proving rows 1..5 are never downloaded.
    {
        let rows = vec![bf16_row_with_masked_leader(TOOL_CALL, 31)];
        let model = MockGumbelModel::new(rows, false);
        let mut a = make_gumbel_seq(0, true, 56);
        a.max_output_tokens = 64;
        a.remaining = 64;
        let out = run_gumbel_thinking_accept_at_row(&model, &mut a, 0, &[99, 98, 97, 96], 5, &ctx);
        assert_eq!(model.d2h_calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            (out.num_accepted, out.bonus, out.consumed_rows),
            (0, Some(31), 1)
        );
        assert_eq!(a.output_tokens, vec![31]);
    }
}
