// SPDX-License-Identifier: AGPL-3.0-only

use super::{
    BatchedK5Plan, batched_k5_policy_required, plan_for_cohort, step_verify_qwen4_batched,
    verify_frame, warm_ple_draft_ahead_for_seq, warm_ple_draft_ahead_for_seqs,
    warm_ple_verify_prefetch_for_seqs,
};
use crate::grammar::{GrammarEngine, GrammarState};
use crate::scheduler::{ActiveSeq, ResponseSink, ThinkSpecCtx};
use anyhow::Result;
use spark_model::traits::{Model, SequenceState};
use spark_runtime::gpu::DevicePtr;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

const VOCAB: usize = 129;
const EOS_ID: u32 = 128;
const ILLEGAL: u32 = 1; // ASCII SOH — never allowed outside string or at JSON start
const LOGITS_BASE: u64 = 0x10_0000;
const SCHEMA: &str = r#"{"type":"object","additionalProperties":false,"properties":{"bridge":{"type":"string"}},"required":["bridge"]}"#;

fn strict_json_grammar() -> GrammarState {
    let mut vocab: Vec<String> = (0u8..128).map(|b| String::from(b as char)).collect();
    vocab.push("<eos>".to_string());
    let mut engine = GrammarEngine::new(&vocab, &[EOS_ID as i32]).unwrap();
    let compiled = engine.compile_json_schema(SCHEMA).unwrap();
    GrammarState::new(&compiled, engine.vocab_size()).unwrap()
}

fn ascii_tokens(s: &str) -> Vec<u32> {
    s.bytes().map(u32::from).collect()
}

fn bf16_row(legal_top: u32, illegal_outscores: bool) -> Vec<u8> {
    let mut f32s = vec![-8.0f32; VOCAB];
    f32s[legal_top as usize] = 40.0;
    if illegal_outscores {
        f32s[ILLEGAL as usize] = 50.0;
    }
    f32s.into_iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

fn raw_argmax_of_bf16_row(row: &[u8]) -> u32 {
    row.chunks_exact(2)
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            let fa = f32::from_bits((u16::from_le_bytes([a[0], a[1]]) as u32) << 16);
            let fb = f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16);
            fa.total_cmp(&fb)
        })
        .map_or(0, |(id, _)| id as u32)
}

fn default_think_ctx() -> ThinkSpecCtx<'static> {
    ThinkSpecCtx {
        enabled: false,
        code_fence_token: None,
        reflection_suppress_ids: &[],
        adaptive_sampling: false,
        honor_eos_inside_thinking: Some(false),
    }
}

fn active_with_drafts(
    slot: usize,
    last_token: u32,
    drafts: Vec<u32>,
    grammar: Option<GrammarState>,
) -> ActiveSeq {
    let mut a = crate::scheduler::test_fixtures::active_seq(ResponseSink::Blocking(None), slot);
    a.seq.prompt_len = 2;
    a.seq.tokens = vec![65, last_token];
    a.seq.seq_len = 2;
    a.last_token = last_token;
    a.output_tokens = vec![last_token];
    a.max_output_tokens = 64;
    a.remaining = 63;
    a.eos_tokens = vec![EOS_ID];
    a.inside_thinking = false;
    a.think_ended = true;
    a.think_just_ended = false;
    a.repetition_penalty = 1.0;
    a.presence_penalty = 0.0;
    a.frequency_penalty = 0.0;
    a.dry_multiplier = 0.0;
    a.lz_penalty = 0.0;
    a.pending_drafts = drafts;
    a.grammar_state = grammar;
    a
}

struct BatchedGrammarMockModel {
    inner: Box<dyn Model>,
    dflash: bool,
    dflash_sample_method: spark_model::dflash_sampling::DflashDraftSampleMethod,
    frame_rows_bf16: Vec<Vec<u8>>,
    next_proposals: Vec<Vec<u32>>,
    batched_verify_calls: AtomicUsize,
    batched_propose_calls: AtomicUsize,
    committed: Mutex<Vec<(usize, usize, usize)>>,
    saved_hiddens: Mutex<Vec<(usize, usize, usize)>>,
    retired_dflash: Mutex<Vec<(usize, usize, usize, usize)>>,
    trimmed: Mutex<Vec<(usize, usize)>>,
    ple_warmed_seqs: Mutex<Vec<Vec<(Vec<u32>, Vec<u32>)>>>,
}

impl BatchedGrammarMockModel {
    fn new(frame_rows_bf16: Vec<Vec<u8>>, next_proposals: Vec<Vec<u32>>) -> Self {
        Self::with_proposer(
            false,
            spark_model::dflash_sampling::DflashDraftSampleMethod::Greedy,
            frame_rows_bf16,
            next_proposals,
        )
    }

    fn new_dflash(
        frame_rows_bf16: Vec<Vec<u8>>,
        next_proposals: Vec<Vec<u32>>,
        dflash_sample_method: spark_model::dflash_sampling::DflashDraftSampleMethod,
    ) -> Self {
        Self::with_proposer(true, dflash_sample_method, frame_rows_bf16, next_proposals)
    }

    fn with_proposer(
        dflash: bool,
        dflash_sample_method: spark_model::dflash_sampling::DflashDraftSampleMethod,
        frame_rows_bf16: Vec<Vec<u8>>,
        next_proposals: Vec<Vec<u32>>,
    ) -> Self {
        let inner = crate::scheduler::test_fixtures::model::recording();
        inner.batched_k5.store(true, Ordering::SeqCst);
        Self {
            inner: Box::new(inner),
            dflash,
            dflash_sample_method,
            frame_rows_bf16,
            next_proposals,
            batched_verify_calls: AtomicUsize::new(0),
            batched_propose_calls: AtomicUsize::new(0),
            committed: Mutex::new(Vec::new()),
            saved_hiddens: Mutex::new(Vec::new()),
            retired_dflash: Mutex::new(Vec::new()),
            trimmed: Mutex::new(Vec::new()),
            ple_warmed_seqs: Mutex::new(Vec::new()),
        }
    }
}

impl Model for BatchedGrammarMockModel {
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
        let row_bytes = VOCAB * 2;
        let byte_off = (ptr.0 - LOGITS_BASE) as usize;
        assert_eq!(byte_off % row_bytes, 0);
        assert_eq!(out.len() % row_bytes, 0);
        let start_row = byte_off / row_bytes;
        let count_rows = out.len() / row_bytes;
        for r in 0..count_rows {
            let src = &self.frame_rows_bf16[start_row + r];
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
    fn hidden_after_norm(&self) -> DevicePtr {
        DevicePtr::NULL
    }
    fn decode_verify(&self, t: &[u32], s: &mut SequenceState, st: u64) -> Result<Vec<u32>> {
        self.inner.decode_verify(t, s, st)
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
        true
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
    fn qwen4_batched_k5_armed(&self) -> bool {
        true
    }
    fn proposer_is_dflash(&self) -> bool {
        self.dflash
    }
    fn dflash_draft_sample_method(&self) -> spark_model::dflash_sampling::DflashDraftSampleMethod {
        self.dflash_sample_method
    }
    fn proposer_arm_kind(&self, _: u8) -> Option<spark_model::speculative::DraftProposerKind> {
        if self.dflash {
            Some(spark_model::speculative::DraftProposerKind::DFlash)
        } else {
            Some(spark_model::speculative::DraftProposerKind::Qwen4NativeMtp)
        }
    }
    fn sync_secondary(&self) -> Result<()> {
        Ok(())
    }
    fn preflight_verify_qwen4_batched(&self, _: &[Vec<u32>], _: &[&SequenceState]) -> Result<()> {
        Ok(())
    }
    fn begin_verify_qwen4_batch(&self, _: &mut [&mut SequenceState], _: usize) -> Result<()> {
        Ok(())
    }
    fn end_verify_qwen4_batch_frame(&self, _: &[&SequenceState]) -> Result<()> {
        Ok(())
    }
    fn finish_verify_qwen4_batch(&self, _: &[&SequenceState]) -> Result<()> {
        Ok(())
    }
    fn abort_verify_qwen4_batch(&self, _: &mut [&mut SequenceState]) -> Result<()> {
        Ok(())
    }
    fn retire_verify_qwen4_batch_terminal(&self, _: &SequenceState) -> Result<()> {
        Ok(())
    }
    fn decode_verify_qwen4_batched(
        &self,
        tokens_per_seq: &[Vec<u32>],
        seqs: &mut [&mut SequenceState],
    ) -> Result<Vec<Vec<u32>>> {
        self.batched_verify_calls.fetch_add(1, Ordering::SeqCst);
        let mut row_cursor = 0usize;
        let mut out = Vec::with_capacity(seqs.len());
        for (window, seq) in tokens_per_seq.iter().zip(seqs.iter_mut()) {
            seq.tokens.extend_from_slice(window);
            seq.seq_len += window.len();
            let verified: Vec<u32> = (0..window.len())
                .map(|i| raw_argmax_of_bf16_row(&self.frame_rows_bf16[row_cursor + i]))
                .collect();
            row_cursor += window.len();
            out.push(verified);
        }
        Ok(out)
    }
    fn commit_verify_state_async(
        &self,
        seq: &mut SequenceState,
        accepted: usize,
        k: usize,
    ) -> Result<()> {
        self.committed
            .lock()
            .unwrap()
            .push((seq.slot_idx, accepted, k));
        Ok(())
    }
    fn save_hidden_for_mtp_sequence_in_frame(
        &self,
        row_base: usize,
        num_accepted: usize,
        seq: &SequenceState,
        _: u64,
    ) -> Result<()> {
        self.saved_hiddens
            .lock()
            .unwrap()
            .push((seq.slot_idx, row_base, num_accepted));
        Ok(())
    }
    fn retire_dflash_frame_captures(
        &self,
        row_base: usize,
        num_accepted: usize,
        seq: &mut SequenceState,
        _: u64,
    ) -> Result<()> {
        self.retired_dflash.lock().unwrap().push((
            seq.slot_idx,
            row_base,
            num_accepted,
            seq.seq_len,
        ));
        Ok(())
    }
    fn trim_proposer_state(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
        _: u64,
    ) -> Result<()> {
        self.trimmed
            .lock()
            .unwrap()
            .push((seq.slot_idx, num_accepted));
        Ok(())
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
        Ok(self
            .next_proposals
            .get(seq.slot_idx)
            .cloned()
            .unwrap_or_default())
    }
    fn run_mtp_propose_multi_batched(
        &self,
        _: &[u32],
        _: &[usize],
        _: usize,
        seqs: &mut [&mut SequenceState],
        _: u64,
    ) -> Result<Vec<Vec<u32>>> {
        self.batched_propose_calls.fetch_add(1, Ordering::SeqCst);
        Ok(seqs
            .iter()
            .map(|s| {
                self.next_proposals
                    .get(s.slot_idx)
                    .cloned()
                    .unwrap_or_default()
            })
            .collect())
    }
    fn warm_ple_draft_ahead_seqs(&self, seqs: &[(&[u32], &[u32])]) {
        self.ple_warmed_seqs.lock().unwrap().push(
            seqs.iter()
                .map(|(window, prior)| (window.to_vec(), prior.to_vec()))
                .collect(),
        );
    }
}

#[test]
fn grammar_and_mixed_cohorts_batch_together_in_plan_for_cohort() {
    let think = default_think_ctx();
    let model = BatchedGrammarMockModel::new(Vec::new(), Vec::new());

    // Pure grammar cohort (C = 2, K = 5)
    let pure_grammar = vec![
        active_with_drafts(0, 65, ascii_tokens(r#"{"br"#), Some(strict_json_grammar())),
        active_with_drafts(1, 66, ascii_tokens(r#"{"ok"#), Some(strict_json_grammar())),
    ];
    for a in &pure_grammar {
        assert!(!batched_k5_policy_required(a, &think));
    }
    let plan = plan_for_cohort(&model, &pure_grammar, &[0, 1], 4, &think);
    assert_eq!(
        plan,
        BatchedK5Plan {
            frames: vec![vec![0, 1]],
            leftover: vec![],
        }
    );

    // Mixed grammar + plain cohort (C = 4, K = 5)
    let mixed = vec![
        active_with_drafts(0, 65, ascii_tokens(r#"{"br"#), Some(strict_json_grammar())),
        active_with_drafts(1, 66, vec![70, 71, 72, 73], None),
        active_with_drafts(2, 67, ascii_tokens(r#"{"hi"#), Some(strict_json_grammar())),
        active_with_drafts(3, 68, vec![80, 81, 82, 83], None),
    ];
    let mixed_plan = plan_for_cohort(&model, &mixed, &[0, 1, 2, 3], 4, &think);
    assert_eq!(
        mixed_plan,
        BatchedK5Plan {
            frames: vec![vec![0, 1, 2, 3]],
            leftover: vec![],
        }
    );
}

#[test]
fn batched_verify_advances_strict_grammar_only_on_accepted_prefix_and_masked_bonus() {
    // Seq 0 (K = 4, 3 legal drafts `{"b` + legal bonus `r`): accepts all 3 + bonus -> `{"br`.
    // Seq 1 (K = 4, `temperature = 0.8`, drafts = `[b'{', ILLEGAL, b'Z']`):
    //   - row 0: legal `{` matches draft[0] -> accepted!
    //   - row 1: raw argmax is ILLEGAL (50.0 > 40.0), legal top is `"`; grammar bitmask
    //     suppresses ILLEGAL so target is `"`, rejecting draft[1]=ILLEGAL and stopping!
    //   - row 2..3 (`b'Z'`): unaccepted tail must NEVER advance `grammar_state`!
    //   - next batched proposal for Seq 1 returns `[ILLEGAL, ILLEGAL, ILLEGAL]`:
    //     `splice_grammar_drafts` coalesces forced `"bridge":` prefix into `pending_drafts`
    //     while leaving `grammar_state` rolled back to exact post-`{"` state!
    let mut frame_rows = Vec::new();
    for ch in ascii_tokens(r#"{"br"#) {
        frame_rows.push(bf16_row(ch, false));
    }
    frame_rows.push(bf16_row(u32::from(b'{'), false));
    frame_rows.push(bf16_row(u32::from(b'"'), true)); // raw argmax is ILLEGAL!
    frame_rows.push(bf16_row(u32::from(b'Z'), true));
    frame_rows.push(bf16_row(u32::from(b'Z'), true));

    let next_proposals = vec![ascii_tokens("idge"), vec![ILLEGAL, ILLEGAL, ILLEGAL]];
    let model = BatchedGrammarMockModel::new(frame_rows, next_proposals);
    let mut active = vec![
        active_with_drafts(0, 65, ascii_tokens(r#"{"b"#), Some(strict_json_grammar())),
        active_with_drafts(
            1,
            66,
            vec![u32::from(b'{'), ILLEGAL, u32::from(b'Z')],
            Some(strict_json_grammar()),
        ),
    ];
    active[1].temperature = 0.8;
    active[1].seed = Some(0xCAFE_BABE);

    let think = default_think_ctx();
    let plan = plan_for_cohort(&model, &active, &[0, 1], 3, &think);
    assert_eq!(plan.frames, vec![vec![0, 1]]);
    assert!(step_verify_qwen4_batched(
        &model,
        &mut active,
        &plan.frames,
        3,
        &think
    ));

    assert_eq!(model.batched_verify_calls.load(Ordering::SeqCst), 1);
    assert_eq!(model.batched_propose_calls.load(Ordering::SeqCst), 1);
    assert_eq!(active[0].output_tokens[1..], ascii_tokens(r#"{"br"#));
    assert_eq!(active[1].output_tokens[1..], ascii_tokens(r#"{""#));
    assert_eq!(
        *model.committed.lock().unwrap(),
        vec![(0, 4, 4), (1, 2, 4)],
        "seq 0 commits 3+1=4 inputs; seq 1 commits 1+1=2 inputs"
    );
    assert_eq!(*model.trimmed.lock().unwrap(), vec![(0, 3), (1, 1)]);
    // Host sequence extents rolled back to pre_verify_len (2) + num_accepted + 1.
    assert_eq!(active[0].seq.seq_len, 6);
    assert_eq!(active[0].seq.tokens.len(), 6);
    assert_eq!(active[1].seq.seq_len, 4);
    assert_eq!(active[1].seq.tokens.len(), 4);

    // Seq 1's illegal proposal `[ILLEGAL, ILLEGAL, ILLEGAL]` after `{"` was coalesced
    // to the forced schema key `"bridge"` prefix (`b`, `r`, `i`) without mutating gs!
    assert_eq!(active[1].pending_drafts, ascii_tokens("bri"));

    // Batched cohorts (`propose_idxs.len() >= 2`) immediately submit an
    // exact-order reusable PLE verify prefetch for the proposed cohort so
    // NVMe reads overlap with tick-end host work, KV alloc, and Layer 0,
    // while solo `C=1` (`propose_idxs.len() == 1`) continues warming PLE.
    {
        let warmed = model.ple_warmed_seqs.lock().unwrap();
        assert_eq!(
            warmed.len(),
            1,
            "batched cohorts (propose_idxs.len() >= 2) must submit PLE verify prefetch"
        );
        let mut expected_win_0 = vec![active[0].last_token];
        expected_win_0.extend_from_slice(&active[0].pending_drafts);
        let mut expected_win_1 = vec![active[1].last_token];
        expected_win_1.extend_from_slice(&active[1].pending_drafts);
        assert_eq!(
            warmed[0],
            vec![
                (expected_win_0, active[0].seq.tokens.clone()),
                (expected_win_1, active[1].seq.tokens.clone()),
            ]
        );
    }
    super::propose_next_batch(&model, &mut active, &[0], 3);
    {
        let warmed = model.ple_warmed_seqs.lock().unwrap();
        assert_eq!(
            warmed.len(),
            2,
            "solo C=1 propose_next_batch must still warm PLE draft-ahead"
        );
        let mut expected_win_0 = vec![active[0].last_token];
        expected_win_0.extend_from_slice(&active[0].pending_drafts);
        assert_eq!(
            warmed[1],
            vec![(expected_win_0, active[0].seq.tokens.clone())]
        );
    }

    // Verify each sequence's GrammarState advanced ONLY by its emitted prefix:
    // Seq 0 emitted `{"br` -> `i` is legal next, `Z` is illegal!
    let gs0 = active[0].grammar_state.as_mut().unwrap();
    assert!(!gs0.accept_token(u32::from(b'Z')));
    assert!(gs0.accept_token(u32::from(b'i')));

    // Seq 1 emitted `{"` -> `b` is legal next, `Z` is illegal!
    let gs1 = active[1].grammar_state.as_mut().unwrap();
    assert!(!gs1.accept_token(u32::from(b'Z')));
    assert!(gs1.accept_token(u32::from(b'b')));
}

#[test]
fn mixed_grammar_and_plain_cohort_shares_one_batched_verify_forward() {
    // Seq 0: strict JSON grammar, drafts `[b'{', ILLEGAL, b'"']` -> accepts `{`, masked bonus `"`.
    // Seq 1: grammarless neutral greedy, drafts `[70, 71, 99]`, verified `[70, 71, 72, 73]` -> accepts 2 + bonus 72.
    let frame_rows = vec![
        bf16_row(u32::from(b'{'), false),
        bf16_row(u32::from(b'"'), true),
        bf16_row(u32::from(b'b'), false),
        bf16_row(u32::from(b'r'), false),
        bf16_row(70, false),
        bf16_row(71, false),
        bf16_row(72, false),
        bf16_row(73, false),
    ];
    let model =
        BatchedGrammarMockModel::new(frame_rows, vec![ascii_tokens("bri"), vec![73, 74, 75]]);
    let mut active = vec![
        active_with_drafts(
            0,
            65,
            vec![u32::from(b'{'), ILLEGAL, u32::from(b'"')],
            Some(strict_json_grammar()),
        ),
        active_with_drafts(1, 66, vec![70, 71, 99], None),
    ];
    // Seq 1 is pure neutral non-post-thinking greedy (fast non-policy branch in finalize_commit).
    active[1].think_ended = false;

    let think = default_think_ctx();
    let plan = plan_for_cohort(&model, &active, &[0, 1], 3, &think);
    assert_eq!(plan.frames, vec![vec![0, 1]]);
    assert!(step_verify_qwen4_batched(
        &model,
        &mut active,
        &plan.frames,
        3,
        &think
    ));

    assert_eq!(model.batched_verify_calls.load(Ordering::SeqCst), 1);
    assert_eq!(active[0].output_tokens[1..], ascii_tokens(r#"{""#));
    assert_eq!(active[1].output_tokens[1..], [70, 71, 72]);
    assert_eq!(active[1].last_token, 72);
    let gs0 = active[0].grammar_state.as_mut().unwrap();
    assert!(gs0.accept_token(u32::from(b'b')));
}

#[test]
fn ragged_verify_frame_commits_mixed_k7_k3_k1_k5_with_exact_row_bases() {
    // 4 sequences in ONE ragged frame:
    //   seq 0 (K=7, row_base=0):  6 drafts [10..=15], target agrees on all 6 + bonus 16 -> accept 6
    //   seq 1 (K=3, row_base=7):  2 drafts [20, 21],  target agrees on [20], rejects 21->29 -> accept 1
    //   seq 2 (K=1, row_base=10): 0 drafts [] (anchor-only!), target emits bonus 30 -> accept 0
    //   seq 3 (K=5, row_base=11): 4 drafts [40..=43], target agrees on [40, 41], bonus 49 -> accept 2
    let target_tokens: Vec<u32> = vec![
        10, 11, 12, 13, 14, 15, 16, // seq 0: rows 0..7
        20, 29, 99, // seq 1: rows 7..10
        30, // seq 2: row 10 (K=1 anchor-only)
        40, 41, 49, 99, 99, // seq 3: rows 11..16
    ];
    let frame_rows: Vec<Vec<u8>> = target_tokens
        .iter()
        .map(|&tok| bf16_row(tok, false))
        .collect();
    let model = BatchedGrammarMockModel::new(frame_rows, Vec::new());

    let mut active = vec![
        active_with_drafts(0, 9, vec![10, 11, 12, 13, 14, 15], None),
        active_with_drafts(1, 19, vec![20, 21], None),
        active_with_drafts(2, 29, Vec::new(), None),
        active_with_drafts(3, 39, vec![40, 41, 42, 43], None),
    ];
    for a in &mut active {
        a.think_ended = false;
        a.tool_call_end_token = None;
    }
    let tokens_per_seq: Vec<Vec<u32>> = active
        .iter()
        .map(|a| {
            let mut w = Vec::with_capacity(1 + a.pending_drafts.len());
            w.push(a.last_token);
            w.extend_from_slice(&a.pending_drafts);
            w
        })
        .collect();
    let think = default_think_ctx();
    let propose_ready = verify_frame(
        &model,
        &mut active,
        &[0, 1, 2, 3],
        &tokens_per_seq,
        4,
        &think,
    )
    .expect("ragged verify_frame succeeds");

    assert_eq!(propose_ready, vec![0, 1, 2, 3]);
    assert_eq!(model.batched_verify_calls.load(Ordering::SeqCst), 1);
    assert_eq!(active[0].output_tokens[1..], [10, 11, 12, 13, 14, 15, 16]);
    assert_eq!(active[1].output_tokens[1..], [20, 29]);
    assert_eq!(active[2].output_tokens[1..], [30]);
    assert_eq!(active[3].output_tokens[1..], [40, 41, 49]);
    assert_eq!(active[2].last_token, 30);

    // Commit records (slot, accepted_inputs = num_accepted + 1, k_verify).
    assert_eq!(
        *model.committed.lock().unwrap(),
        vec![(0, 7, 7), (1, 2, 3), (2, 1, 1), (3, 3, 5)]
    );
    // Hidden handoffs MUST use cumulative row bases (0, 7, 10, 11), NOT s * k!
    assert_eq!(
        *model.saved_hiddens.lock().unwrap(),
        vec![(0, 0, 6), (1, 7, 1), (2, 10, 0), (3, 11, 2)]
    );
    assert_eq!(
        *model.trimmed.lock().unwrap(),
        vec![(0, 6), (1, 1), (2, 0), (3, 2)]
    );
    // Sequence extents rolled back to pre_verify_len (2) + num_accepted + 1.
    assert_eq!(active[0].seq.seq_len, 9);
    assert_eq!(active[1].seq.seq_len, 4);
    assert_eq!(active[2].seq.seq_len, 3);
    assert_eq!(active[3].seq.seq_len, 5);
}

#[test]
fn warm_ple_draft_ahead_packs_dedups_and_skips_finished_or_empty_seqs() {
    let model = BatchedGrammarMockModel::new(Vec::new(), Vec::new());
    let mut s0 = active_with_drafts(0, 70, vec![71, 72, 73], None);
    s0.seq.tokens = vec![10, 20, 30];
    // Identical 2-token prior tail `[20, 30]`, last_token `70`, and drafts -> deduplicated.
    let mut s1 = active_with_drafts(1, 70, vec![71, 72, 73], None);
    s1.seq.tokens = vec![99, 20, 30];
    // Distinct prior tail `[21, 30]` -> kept because trigram/bigram hashes differ.
    let mut s2 = active_with_drafts(2, 70, vec![71, 72, 73], None);
    s2.seq.tokens = vec![21, 30];
    // Finished or empty pending_drafts -> skipped.
    let mut s3 = active_with_drafts(3, 80, vec![81, 82], None);
    s3.finished = true;
    let s4 = active_with_drafts(4, 90, Vec::new(), None);
    let active = [s0, s1, s2, s3, s4];

    warm_ple_draft_ahead_for_seqs(&model, &active, &[0, 1, 2, 3, 4]);
    {
        let warmed = model.ple_warmed_seqs.lock().unwrap();
        assert_eq!(warmed.len(), 1);
        assert_eq!(
            warmed[0],
            vec![
                (vec![70, 71, 72, 73], vec![10, 20, 30]),
                (vec![70, 71, 72, 73], vec![21, 30]),
            ]
        );
    }

    // Single-sequence C=1 helper skips finished/empty and submits exact `(window, prior)`.
    warm_ple_draft_ahead_for_seq(&model, &active[3]);
    warm_ple_draft_ahead_for_seq(&model, &active[4]);
    assert_eq!(model.ple_warmed_seqs.lock().unwrap().len(), 1);

    warm_ple_draft_ahead_for_seq(&model, &active[2]);
    {
        let warmed = model.ple_warmed_seqs.lock().unwrap();
        assert_eq!(warmed.len(), 2);
        assert_eq!(warmed[1], vec![(vec![70, 71, 72, 73], vec![21, 30])]);
    }

    // Cross-sequence verify prefetch preserves exact cohort order and duplicates
    // (so un-deduplicated DFlash2 verify frames hit exact key match), skips finished
    // sequences, and admits K=1 anchor-only lanes only when `allow_k1` is enabled.
    warm_ple_verify_prefetch_for_seqs(&model, &active, &[0, 1, 2, 3, 4], false);
    warm_ple_verify_prefetch_for_seqs(&model, &active, &[0, 1, 2, 3, 4], true);
    let warmed = model.ple_warmed_seqs.lock().unwrap();
    assert_eq!(warmed.len(), 4);
    assert_eq!(
        warmed[2],
        vec![
            (vec![70, 71, 72, 73], vec![10, 20, 30]),
            (vec![70, 71, 72, 73], vec![99, 20, 30]),
            (vec![70, 71, 72, 73], vec![21, 30]),
        ]
    );
    assert_eq!(
        warmed[3],
        vec![
            (vec![70, 71, 72, 73], vec![10, 20, 30]),
            (vec![70, 71, 72, 73], vec![99, 20, 30]),
            (vec![70, 71, 72, 73], vec![21, 30]),
            (vec![90], vec![65, 90]),
        ]
    );
}

#[test]
fn dflash_batched_admits_greedy_k8_uniform_refuses_probabilistic_and_retires_captures() {
    use spark_model::dflash_sampling::DflashDraftSampleMethod;

    let think = default_think_ctx();
    // Seq 0 (K=8, row_base=0): drafts [10..=16], target agrees on 3 drafts + bonus 90 -> accept 3, seq_len = 2 + 4 = 6.
    // Seq 1 (K=8, row_base=8): drafts [20..=26], target agrees on all 7 drafts + bonus 27 -> accept 7, seq_len = 2 + 8 = 10.
    let target_tokens: Vec<u32> = vec![
        10, 11, 12, 90, 99, 99, 99, 99, // seq 0: rows 0..8
        20, 21, 22, 23, 24, 25, 26, 27, // seq 1: rows 8..16
    ];
    let frame_rows: Vec<Vec<u8>> = target_tokens
        .iter()
        .map(|&tok| bf16_row(tok, false))
        .collect();
    let next_proposals = vec![
        vec![30, 31, 32, 33, 34, 35, 36],
        vec![40, 41, 42, 43, 44, 45, 46],
    ];
    let model = BatchedGrammarMockModel::new_dflash(
        frame_rows,
        next_proposals,
        DflashDraftSampleMethod::Greedy,
    );
    let mut active = vec![
        active_with_drafts(0, 9, vec![10, 11, 12, 13, 14, 15, 16], None),
        active_with_drafts(1, 19, vec![20, 21, 22, 23, 24, 25, 26], None),
    ];
    for a in &mut active {
        a.think_ended = false;
        a.tool_call_end_token = None;
    }
    let plan = plan_for_cohort(&model, &active, &[0, 1], 7, &think);
    assert_eq!(
        plan,
        BatchedK5Plan {
            frames: vec![vec![0, 1]],
            leftover: Vec::new(),
        }
    );
    assert!(step_verify_qwen4_batched(
        &model,
        &mut active,
        &plan.frames,
        7,
        &think
    ));
    assert_eq!(model.batched_verify_calls.load(Ordering::SeqCst), 1);
    assert_eq!(model.batched_propose_calls.load(Ordering::SeqCst), 1);
    assert!(
        model.saved_hiddens.lock().unwrap().is_empty(),
        "DFlash2 must retire DFlash captures rather than saving native-MTP hiddens"
    );
    assert_eq!(
        *model.retired_dflash.lock().unwrap(),
        vec![(0, 0, 3, 6), (1, 8, 7, 10)],
        "retire_dflash_frame_captures sees exact (slot, row_base, num_accepted, seq_len = pre_verify_len + num_accepted + 1)"
    );
    assert_eq!(*model.trimmed.lock().unwrap(), vec![(0, 3), (1, 7)]);

    // Probabilistic DFlash stays excluded from batched Qwen4 verify.
    let prob_model = BatchedGrammarMockModel::new_dflash(
        Vec::new(),
        Vec::new(),
        DflashDraftSampleMethod::Probabilistic,
    );
    let prob_plan = plan_for_cohort(&prob_model, &active, &[0, 1], 7, &think);
    assert!(prob_plan.frames.is_empty());
    assert_eq!(prob_plan.leftover, vec![0, 1]);
    assert!(!step_verify_qwen4_batched(
        &prob_model,
        &mut active,
        &[vec![0, 1]],
        7,
        &think
    ));

    // Bypassing CohortAcceptEwma (cohort_adaptive_k = false) preserves full gamma=7 (K=8)
    // across all C=2..=16 even after sustained zero-accept frames.
    let ctrl = super::CohortAcceptEwma::new();
    for _ in 0..16 {
        ctrl.observe_frame_seqs(8, 0, 56, 8);
        let _ = ctrl.select_k_deterministic(8, 8, false);
    }
    for live in 2..=16usize {
        assert_eq!(
            super::resolve_cohort_width_with_controller(true, true, false, None, live, 7, &ctrl),
            7,
            "DFlash2 must keep full gamma=7 (K=8) at C={live}"
        );
    }
}
