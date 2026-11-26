// SPDX-License-Identifier: AGPL-3.0-only

//! Cross-request 65,536-token validated history ring and suffix grafting.
//!
//! Combines a cross-request circular buffer of target-validated generated tokens
//! (`RING_CAP = 65_536`) indexed by 2-way set-associative 64-bit 8-gram and
//! 4-gram hash tables (`NgramIndex<8>` and `NgramIndex<4>`) with per-request
//! prompt+output suffix lookup and partial-chain suffix grafting.
//!
//! Every entry in the ring carries a monotonic `span_id` per sequence/rollback
//! epoch so suffix matches and continuations never bleed across unrelated
//! requests. Per-sequence auto-silence suspends copy proposals for 16 steps
//! after 4 consecutive 0-hit proposals so novel text never wastes verify rows.

use std::collections::HashMap;
use std::sync::OnceLock;

use parking_lot::Mutex;
use spark_runtime::sampler::gumbel::splitmix64_mix;

/// Ring capacity in validated tokens (power of two).
pub(crate) const RING_CAP: usize = 65_536;
const RING_MASK: usize = RING_CAP - 1;

/// Hash table bucket count (power of two: 16,384 buckets * 2 ways = 32,768 slots).
const INDEX_BUCKETS: usize = 16_384;
const INDEX_MASK: usize = INDEX_BUCKETS - 1;

/// Consecutive 0-hit copy proposals before silencing a sequence.
pub(crate) const SILENCE_AFTER_MISSES: u8 = 4;
/// Steps a silenced sequence skips copy proposals before re-probing.
pub(crate) const SILENCE_STEPS: u8 = 16;

/// Maximum tracked concurrent/recent sequence trackers.
const MAX_SEQ_TRACKERS: usize = 256;

#[derive(Clone, Copy, Default)]
struct IndexSlot {
    tag: u32,
    /// Monotonic cursor `pos` of the LAST token of the indexed `N`-gram (`pos + 1` is first continuation).
    end_pos: u64,
}

struct NgramIndex<const N: usize> {
    buckets: Box<[[IndexSlot; 2]; INDEX_BUCKETS]>,
}

impl<const N: usize> NgramIndex<N> {
    fn new() -> Self {
        let vec = vec![[IndexSlot::default(); 2]; INDEX_BUCKETS];
        let buckets = vec
            .into_boxed_slice()
            .try_into()
            .unwrap_or_else(|_| unreachable!("exact bucket allocation"));
        Self { buckets }
    }

    #[inline]
    fn hash_ngram(gram: &[u32; N]) -> (usize, u32) {
        let mut h = 0x9E37_79B9_7F4A_7C15u64 ^ (N as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
        for &tok in gram {
            h = splitmix64_mix(h ^ u64::from(tok));
        }
        let bucket = (h as usize) & INDEX_MASK;
        let tag = ((h >> 32) as u32) | 1; // non-zero tag distinguishes populated slots
        (bucket, tag)
    }

    #[inline]
    fn insert(&mut self, gram: &[u32; N], end_pos: u64) {
        let (bucket, tag) = Self::hash_ngram(gram);
        let pair = &mut self.buckets[bucket];
        if pair[0].tag == tag && pair[0].end_pos == end_pos {
            return;
        }
        pair[1] = pair[0];
        pair[0] = IndexSlot { tag, end_pos };
    }

    #[inline]
    fn candidates(&self, gram: &[u32; N]) -> [Option<u64>; 2] {
        let (bucket, tag) = Self::hash_ngram(gram);
        let pair = &self.buckets[bucket];
        [
            (pair[0].tag == tag).then_some(pair[0].end_pos),
            (pair[1].tag == tag).then_some(pair[1].end_pos),
        ]
    }
}

/// Per-sequence copy-draft state (ingestion watermark + 4-miss/16-step auto-silence).
#[derive(Clone, Copy, Debug)]
pub(crate) struct SeqHistoryTracker {
    span_id: u32,
    ingested_len: usize,
    zero_hit_streak: u8,
    silence_remaining: u8,
    /// `(output_len_at_graft_or_propose, first_copy_token)` awaiting next verify check.
    pending_probe: Option<(usize, u32)>,
    last_tick: u64,
}

impl SeqHistoryTracker {
    fn new(span_id: u32, tick: u64) -> Self {
        Self {
            span_id,
            ingested_len: 0,
            zero_hit_streak: 0,
            silence_remaining: 0,
            pending_probe: None,
            last_tick: tick,
        }
    }

    /// Record whether the previous copy proposal got at least 1 token accepted.
    pub(crate) fn record_outcome(&mut self, accepted_any: bool) {
        if accepted_any {
            self.zero_hit_streak = 0;
        } else {
            self.zero_hit_streak = self.zero_hit_streak.saturating_add(1);
            if self.zero_hit_streak >= SILENCE_AFTER_MISSES {
                self.silence_remaining = SILENCE_STEPS;
                self.zero_hit_streak = 0;
            }
        }
    }

    /// Check whether this sequence may attempt a copy proposal on this step,
    /// advancing any active silence countdown.
    pub(crate) fn check_and_step_silence(&mut self) -> bool {
        if self.silence_remaining > 0 {
            self.silence_remaining -= 1;
            return false;
        }
        true
    }

    #[cfg(test)]
    pub(crate) fn silence_remaining(&self) -> u8 {
        self.silence_remaining
    }
}

/// Cross-request 65,536-token validated history ring + `Index<8>` / `Index<4>`.
pub(crate) struct HistoryRing {
    tokens: Box<[u32; RING_CAP]>,
    spans: Box<[u32; RING_CAP]>,
    /// Next monotonic write index (`0` on empty ring; valid written indices are `< write_cursor`).
    write_cursor: u64,
    next_span_id: u32,
    index8: NgramIndex<8>,
    index4: NgramIndex<4>,
    trackers: HashMap<u64, SeqHistoryTracker>,
    tick: u64,
}

impl HistoryRing {
    pub(crate) fn new() -> Self {
        let tokens = vec![0u32; RING_CAP]
            .into_boxed_slice()
            .try_into()
            .unwrap_or_else(|_| unreachable!("exact ring allocation"));
        let spans = vec![0u32; RING_CAP]
            .into_boxed_slice()
            .try_into()
            .unwrap_or_else(|_| unreachable!("exact span allocation"));
        Self {
            tokens,
            spans,
            write_cursor: 0,
            next_span_id: 1,
            index8: NgramIndex::new(),
            index4: NgramIndex::new(),
            trackers: HashMap::new(),
            tick: 0,
        }
    }

    fn alloc_span_id(&mut self) -> u32 {
        let id = self.next_span_id;
        self.next_span_id = self.next_span_id.wrapping_add(1).max(1);
        id
    }

    fn tracker_mut(&mut self, session_hash: u64) -> &mut SeqHistoryTracker {
        self.tick = self.tick.wrapping_add(1);
        let now = self.tick;
        if !self.trackers.contains_key(&session_hash)
            && self.trackers.len() >= MAX_SEQ_TRACKERS
            && let Some((&oldest_key, _)) = self.trackers.iter().min_by_key(|(_, t)| t.last_tick)
        {
            self.trackers.remove(&oldest_key);
        }
        if !self.trackers.contains_key(&session_hash) {
            let span_id = self.alloc_span_id();
            self.trackers
                .insert(session_hash, SeqHistoryTracker::new(span_id, now));
        }
        let tracker = self
            .trackers
            .get_mut(&session_hash)
            .expect("inserted above");
        tracker.last_tick = now;
        tracker
    }

    /// Append a contiguous slice of validated tokens belonging to `span_id`,
    /// using `context_tail` (up to 7 immediately preceding tokens of the same span)
    /// so cross-step 4-grams and 8-grams are indexed without gaps.
    pub(crate) fn append_span_tokens(
        &mut self,
        span_id: u32,
        context_tail: &[u32],
        new_tokens: &[u32],
    ) {
        if new_tokens.is_empty() {
            return;
        }
        let tail_keep = context_tail.len().min(7);
        let tail = &context_tail[context_tail.len() - tail_keep..];

        // Rolling 8-token window seeded from `tail` when contiguous in the ring
        // or directly from `tail` for exact per-span indexing.
        let mut win: [u32; 8] = [0; 8];
        let mut win_len = 0usize;
        for &t in tail {
            win[win_len] = t;
            win_len += 1;
        }

        for &tok in new_tokens {
            let pos = self.write_cursor;
            let slot = (pos as usize) & RING_MASK;
            self.tokens[slot] = tok;
            self.spans[slot] = span_id;
            self.write_cursor = pos + 1;

            if win_len < 8 {
                win[win_len] = tok;
                win_len += 1;
            } else {
                win.copy_within(1..8, 0);
                win[7] = tok;
            }

            // Only index n-grams whose entire N-token body resides physically in the ring
            // under `span_id` so lookup verification against `self.tokens`/`self.spans` holds.
            if win_len >= 4 && self.span_contiguous_at(pos, 4, span_id) {
                let gram4: [u32; 4] = [
                    win[win_len - 4],
                    win[win_len - 3],
                    win[win_len - 2],
                    win[win_len - 1],
                ];
                self.index4.insert(&gram4, pos);
            }
            if win_len == 8 && self.span_contiguous_at(pos, 8, span_id) {
                self.index8.insert(&win, pos);
            }
        }
    }

    #[inline]
    fn span_contiguous_at(&self, end_pos: u64, len: usize, span_id: u32) -> bool {
        if end_pos + 1 < len as u64
            || self.write_cursor - (end_pos + 1 - len as u64) > RING_CAP as u64
        {
            return false;
        }
        let start = end_pos + 1 - len as u64;
        for p in start..=end_pos {
            if self.spans[(p as usize) & RING_MASK] != span_id {
                return false;
            }
        }
        true
    }

    /// Match an `N`-gram at `end_pos` in the ring and extract `need` continuation tokens
    /// from the SAME `span_id`.
    fn verify_and_extract<const N: usize>(
        &self,
        gram: &[u32; N],
        end_pos: u64,
        need: usize,
    ) -> Option<Vec<u32>> {
        if need == 0 || end_pos + 1 < N as u64 {
            return None;
        }
        let gram_start = end_pos + 1 - N as u64;
        let cont_end = end_pos.checked_add(need as u64)?;
        // Must still reside inside the 65,536-token window and before `write_cursor`.
        if cont_end >= self.write_cursor || self.write_cursor - gram_start > (RING_CAP - 8) as u64 {
            return None;
        }
        let span = self.spans[(gram_start as usize) & RING_MASK];
        if span == 0 {
            return None;
        }
        for (i, &expected) in gram.iter().enumerate() {
            let p = gram_start + i as u64;
            let idx = (p as usize) & RING_MASK;
            if self.spans[idx] != span || self.tokens[idx] != expected {
                return None;
            }
        }
        let mut out = Vec::with_capacity(need);
        for step in 1..=need {
            let p = end_pos + step as u64;
            let idx = (p as usize) & RING_MASK;
            if self.spans[idx] != span {
                return None;
            }
            out.push(self.tokens[idx]);
        }
        Some(out)
    }

    /// Query global `Index<8>` then (when `min_n <= 4`) `Index<4>` for `need` continuation tokens.
    pub(crate) fn lookup_global(&self, suffix: &[u32], need: usize) -> Option<Vec<u32>> {
        self.lookup_global_min_n(suffix, need, 4)
    }

    pub(crate) fn lookup_global_min_n(
        &self,
        suffix: &[u32],
        need: usize,
        min_n: usize,
    ) -> Option<Vec<u32>> {
        if need == 0 {
            return None;
        }
        if suffix.len() >= 8 {
            let gram8: [u32; 8] = suffix[suffix.len() - 8..]
                .try_into()
                .expect("slice len == 8");
            for cand in self.index8.candidates(&gram8).into_iter().flatten() {
                if let Some(cont) = self.verify_and_extract(&gram8, cand, need) {
                    return Some(cont);
                }
            }
        }
        if min_n <= 4 && suffix.len() >= 4 {
            let gram4: [u32; 4] = suffix[suffix.len() - 4..]
                .try_into()
                .expect("slice len == 4");
            for cand in self.index4.candidates(&gram4).into_iter().flatten() {
                if let Some(cont) = self.verify_and_extract(&gram4, cand, need) {
                    return Some(cont);
                }
            }
        }
        None
    }

    /// Sync newly validated `output_tokens` of `session_hash` into the ring and
    /// settle any pending 0-hit / hit outcome from the previous step.
    pub(crate) fn sync_sequence(&mut self, session_hash: u64, output_tokens: &[u32]) {
        let (span_id, prev_len) = {
            let tracker = self.tracker_mut(session_hash);
            if let Some((probe_len, first_tok)) = tracker.pending_probe
                && output_tokens.len() > tracker.ingested_len
            {
                if output_tokens.len() > probe_len {
                    let hit = output_tokens.get(probe_len) == Some(&first_tok);
                    tracker.record_outcome(hit);
                }
                tracker.pending_probe = None;
            }
            if output_tokens.len() < tracker.ingested_len {
                // Sequence rolled back: start a fresh span so discarded tokens never match.
                tracker.ingested_len = output_tokens.len();
                tracker.pending_probe = None;
                let fresh = self.alloc_span_id();
                let tracker = self.tracker_mut(session_hash);
                tracker.span_id = fresh;
                return;
            }
            (tracker.span_id, tracker.ingested_len)
        };

        if output_tokens.len() > prev_len {
            let ctx_start = prev_len.saturating_sub(7);
            let context_tail = &output_tokens[ctx_start..prev_len];
            let new_tokens = &output_tokens[prev_len..];
            self.append_span_tokens(span_id, context_tail, new_tokens);
            self.tracker_mut(session_hash).ingested_len = output_tokens.len();
        }
    }

    /// Propose or suffix-graft `need` tokens for `session_hash` from `(seq_tokens, tail_tokens)`
    /// (`tail_tokens` = `[last_token]` or `[last_token, d0, d1, ...]`).
    ///
    /// Cascade order:
    ///   1. Per-request prompt+output 8-gram suffix match (8K window)
    ///   2. Cross-request `HistoryRing` `Index<8>` match (64K ring)
    ///   3. Per-request prompt+output 4-gram suffix match (4K window)
    ///   4. Cross-request `HistoryRing` `Index<4>` match (64K ring)
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn propose_continuation(
        &mut self,
        session_hash: u64,
        seq_tokens: &[u32],
        tail_tokens: &[u32],
        output_len_at_graft: usize,
        need: usize,
    ) -> Option<Vec<u32>> {
        self.propose_continuation_min_n(
            session_hash,
            seq_tokens,
            tail_tokens,
            output_len_at_graft,
            need,
            4,
        )
    }

    pub(crate) fn propose_continuation_min_n(
        &mut self,
        session_hash: u64,
        seq_tokens: &[u32],
        tail_tokens: &[u32],
        output_len_at_graft: usize,
        need: usize,
        min_n: usize,
    ) -> Option<Vec<u32>> {
        if need == 0 || tail_tokens.is_empty() {
            return None;
        }
        if !self.tracker_mut(session_hash).check_and_step_silence() {
            return None;
        }

        // Assemble up to 8 suffix tokens from `seq_tokens ++ tail_tokens` on the stack.
        let mut suffix_buf = [0u32; 8];
        let total = seq_tokens.len() + tail_tokens.len();
        let suffix_len = total.min(8);
        for i in 0..suffix_len {
            let global_idx = total - suffix_len + i;
            suffix_buf[i] = if global_idx < seq_tokens.len() {
                seq_tokens[global_idx]
            } else {
                tail_tokens[global_idx - seq_tokens.len()]
            };
        }
        let suffix = &suffix_buf[..suffix_len];

        let found = local_suffix_lookup(seq_tokens, tail_tokens, suffix, 8, 8192, need)
            .or_else(|| {
                (suffix.len() >= 8)
                    .then(|| self.lookup_global_min_n(suffix, need, 8))
                    .flatten()
            })
            .or_else(|| {
                (min_n <= 4)
                    .then(|| local_suffix_lookup(seq_tokens, tail_tokens, suffix, 4, 4096, need))
                    .flatten()
            })
            .or_else(|| {
                (min_n <= 4)
                    .then(|| self.lookup_global_min_n(suffix, need, 4))
                    .flatten()
            });

        if let Some(ref tokens) = found
            && let Some(&first) = tokens.first()
        {
            self.tracker_mut(session_hash).pending_probe = Some((output_len_at_graft, first));
        }
        found
    }
}

/// Search `seq_tokens ++ tail_tokens` backward over `window` tokens for an exact
/// `min_n`-gram match of `suffix` with `need` valid continuation tokens.
fn local_suffix_lookup(
    seq_tokens: &[u32],
    tail_tokens: &[u32],
    suffix: &[u32],
    min_n: usize,
    window: usize,
    need: usize,
) -> Option<Vec<u32>> {
    if suffix.len() < min_n || need == 0 {
        return None;
    }
    let pat = &suffix[suffix.len() - min_n..];
    let total = seq_tokens.len() + tail_tokens.len();
    // Candidate match ends at `end_idx`, and continuation occupies `end_idx + 1 ..= end_idx + need`,
    // which must lie strictly before the suffix's own current end (`total - 1`).
    if total <= min_n + need {
        return None;
    }
    let at = |idx: usize| -> u32 {
        if idx < seq_tokens.len() {
            seq_tokens[idx]
        } else {
            tail_tokens[idx - seq_tokens.len()]
        }
    };
    let max_end = total - 1 - need;
    let min_end = total.saturating_sub(window).max(min_n - 1);
    let last_pat = pat[min_n - 1];
    let first_pat = pat[0];
    for end_idx in (min_end..=max_end).rev() {
        if at(end_idx) != last_pat || at(end_idx + 1 - min_n) != first_pat {
            continue;
        }
        let mut matched = true;
        for k in 1..min_n - 1 {
            if at(end_idx + 1 - min_n + k) != pat[k] {
                matched = false;
                break;
            }
        }
        if matched {
            return Some((1..=need).map(|step| at(end_idx + step)).collect());
        }
    }
    None
}

// ── Global singleton + scheduler helpers ────────────────────────────────────

/// Maximum draft chain width produced by unscored N-gram copy/graft (`4` drafts
/// = `K=5` `r5` MoE tier), preventing speculative tier inflation into `r8` (`K=6..8`).
const MAX_HISTORY_CHAIN_DRAFTS: usize = 4;

static GLOBAL_RING: OnceLock<Mutex<HistoryRing>> = OnceLock::new();

fn global_ring() -> &'static Mutex<HistoryRing> {
    GLOBAL_RING.get_or_init(|| Mutex::new(HistoryRing::new()))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HistoryDraftMode {
    Off,
    /// Default fallback: fills empty/suppressed frames at 0 GPU cost (does not graft onto non-empty neural drafts).
    FallbackOnly,
    /// `FLUX_HISTORY_DRAFTS=1`: pre-empts after 0-accept neural misses + 8-gram suffix grafting.
    Explicit,
}

pub(crate) fn parse_history_draft_mode(value: Option<&str>) -> HistoryDraftMode {
    match value.map(str::trim) {
        Some("0") | Some("off") | Some("false") => HistoryDraftMode::Off,
        Some("1") | Some("on") | Some("true") => HistoryDraftMode::Explicit,
        _ => HistoryDraftMode::FallbackOnly,
    }
}

pub(crate) fn history_draft_mode() -> HistoryDraftMode {
    static MODE: OnceLock<HistoryDraftMode> = OnceLock::new();
    *MODE.get_or_init(|| {
        let raw = std::env::var("FLUX_HISTORY_DRAFTS")
            .or_else(|_| std::env::var("ATLAS_HISTORY_DRAFTS"))
            .ok();
        let mode = parse_history_draft_mode(raw.as_deref());
        if mode == HistoryDraftMode::Explicit {
            tracing::info!(
                "FLUX_HISTORY_DRAFTS=1 engaged (64K cross-request ring + suffix grafting)"
            );
        }
        mode
    })
}

/// Sync a single sequence's newly validated `output_tokens` into the global 65,536-token ring.
pub(crate) fn sync_one_validated_tokens(session_hash: u64, output_tokens: &[u32]) {
    if history_draft_mode() == HistoryDraftMode::Off {
        return;
    }
    global_ring()
        .lock()
        .sync_sequence(session_hash, output_tokens);
}

/// Sync all active sequences' newly validated `output_tokens` into the global 65,536-token ring.
pub(crate) fn sync_active_validated_tokens(active: &[super::ActiveSeq]) {
    if history_draft_mode() == HistoryDraftMode::Off || active.is_empty() {
        return;
    }
    let mut ring = global_ring().lock();
    for a in active {
        ring.sync_sequence(a.session_hash, &a.output_tokens);
    }
}

/// Attempt an explicit pre-emptive 8-gram copy proposal when `FLUX_HISTORY_DRAFTS=1`
/// and the neural drafter had zero accepted tokens on the preceding frame (or at bootstrap).
pub(crate) fn try_explicit_preempt(
    session_hash: u64,
    seq_tokens: &[u32],
    last_token: u32,
    output_len: usize,
    last_verify_accepted: u16,
    num_drafts: usize,
) -> Option<Vec<u32>> {
    let capped_drafts = num_drafts.min(MAX_HISTORY_CHAIN_DRAFTS);
    if history_draft_mode() != HistoryDraftMode::Explicit
        || capped_drafts == 0
        || last_verify_accepted > 0
    {
        return None;
    }
    global_ring().lock().propose_continuation_min_n(
        session_hash,
        seq_tokens,
        &[last_token],
        output_len,
        capped_drafts,
        8,
    )
}

/// Fill or suffix-graft remaining draft slots up to `target_drafts.min(4)` when the neural
/// drafter was suppressed (depth 0), truncated early by the confidence gate, or
/// returned empty (`FLUX_HISTORY_DRAFTS=1` or default zero-GPU-cost fallback).
pub(crate) fn fill_or_graft_drafts(
    session_hash: u64,
    seq_tokens: &[u32],
    last_token: u32,
    existing_drafts: &[u32],
    output_len: usize,
    target_drafts: usize,
) -> Option<Vec<u32>> {
    let mode = history_draft_mode();
    let effective_target = target_drafts.min(MAX_HISTORY_CHAIN_DRAFTS);
    if mode == HistoryDraftMode::Off || existing_drafts.len() >= effective_target {
        return None;
    }
    // In default FallbackOnly mode, fill empty proposals (suppressed / depth-0 /
    // zero-draft confidence stop); suffix grafting onto non-empty neural prefixes runs
    // only on exact 8-gram matches when FLUX_HISTORY_DRAFTS=1 is explicitly enabled.
    if !existing_drafts.is_empty() && mode != HistoryDraftMode::Explicit {
        return None;
    }
    let min_n = if existing_drafts.is_empty() { 4 } else { 8 };
    let need = effective_target - existing_drafts.len();
    let mut tail = Vec::with_capacity(1 + existing_drafts.len());
    tail.push(last_token);
    tail.extend_from_slice(existing_drafts);
    let graft_output_len = output_len + existing_drafts.len();
    global_ring().lock().propose_continuation_min_n(
        session_hash,
        seq_tokens,
        &tail,
        graft_output_len,
        need,
        min_n,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cross_request_8gram_and_4gram_lookup_recalls_validated_continuations() {
        let mut ring = HistoryRing::new();
        // Request 1 generates a typical Rust snippet.
        let req1_out = vec![10, 11, 12, 13, 14, 15, 16, 17, 100, 101, 102, 103];
        ring.sync_sequence(0x1111, &req1_out);

        // Request 2 hits the exact 8-gram [10..=17] and recalls [100, 101, 102, 103]!
        let got8 = ring
            .propose_continuation(0x2222, &[99, 10, 11, 12, 13, 14, 15, 16], &[17], 1, 4)
            .unwrap();
        assert_eq!(got8, vec![100, 101, 102, 103]);

        // Request 3 only shares the 4-gram [14, 15, 16, 17] and still recalls via Index<4>!
        let got4 = ring
            .propose_continuation(0x3333, &[777, 14, 15, 16], &[17], 1, 3)
            .unwrap();
        assert_eq!(got4, vec![100, 101, 102]);
    }

    #[test]
    fn span_boundaries_prevent_cross_request_token_bleed() {
        let mut ring = HistoryRing::new();
        // Request A ends right after [1, 2, 3, 4] with no continuation of its own.
        ring.sync_sequence(0xAAAA, &[1, 2, 3, 4]);
        // Request B starts immediately after in the circular buffer with [900, 901, 902, 903].
        ring.sync_sequence(0xBBBB, &[900, 901, 902, 903]);

        // Querying [1, 2, 3, 4] must NEVER return Request B's [900, 901]!
        assert_eq!(
            ring.propose_continuation(0xCCCC, &[1, 2, 3], &[4], 1, 2),
            None
        );
    }

    #[test]
    fn wraparound_past_65536_tokens_evicts_overwritten_entries_cleanly() {
        let mut ring = HistoryRing::new();
        ring.sync_sequence(0x1, &[50, 51, 52, 53, 60, 61]);
        assert_eq!(ring.lookup_global(&[50, 51, 52, 53], 2), Some(vec![60, 61]));

        // Flood > 65,536 tokens so the early entry is overwritten.
        let flood: Vec<u32> = (0..(RING_CAP as u32 + 128))
            .map(|i| 10_000 + (i % 997))
            .collect();
        ring.sync_sequence(0x2, &flood);
        assert_eq!(ring.lookup_global(&[50, 51, 52, 53], 2), None);
    }

    #[test]
    fn local_prompt_and_output_suffix_grafting_extends_partial_chains() {
        let mut ring = HistoryRing::new();
        // Prompt has `[40, 41, 42, 43, 44, 45, 46, 47]`; neural drafter gave 2 tokens `[42, 43]`
        // after `last_token = 41`. Suffix `[..., 40, 41, 42, 43]` grafts `[44, 45]`!
        let seq_tokens = vec![1, 2, 40, 41, 42, 43, 44, 45, 46, 47, 9, 40];
        let tail = [41, 42, 43];
        let grafted = ring
            .propose_continuation(0x77, &seq_tokens, &tail, 3, 2)
            .unwrap();
        assert_eq!(grafted, vec![44, 45]);
    }

    #[test]
    fn four_consecutive_zero_hit_proposals_auto_silence_for_16_steps_then_recover() {
        let mut ring = HistoryRing::new();
        ring.sync_sequence(0xFEED, &[1, 2, 3, 4, 99, 100]);

        let mut out = vec![10u32];
        for miss in 0..4 {
            let prop = ring
                .propose_continuation(0xBEEF, &[1, 2, 3], &[4], out.len(), 2)
                .expect("should propose before 4th miss settles");
            assert_eq!(prop, vec![99, 100], "miss={miss}");
            // Target rejected 99 and committed 555 instead -> 0-hit!
            out.push(555 + miss);
            ring.sync_sequence(0xBEEF, &out);
        }

        assert_eq!(
            ring.tracker_mut(0xBEEF).silence_remaining(),
            SILENCE_STEPS,
            "must arm 16-step silence after 4 consecutive 0-hit proposals"
        );

        // Next 16 attempts return None without wasting verify rows.
        for step in 0..SILENCE_STEPS {
            assert_eq!(
                ring.propose_continuation(0xBEEF, &[1, 2, 3], &[4], out.len(), 2),
                None,
                "must stay silenced at step {step}"
            );
        }

        // On step 17, copy drafting automatically re-awakens!
        assert_eq!(
            ring.propose_continuation(0xBEEF, &[1, 2, 3], &[4], out.len(), 2),
            Some(vec![99, 100])
        );
    }

    #[test]
    fn min_n_8_graft_guard_rejects_spurious_4gram_coincidences() {
        let mut ring = HistoryRing::new();
        ring.sync_sequence(0xABCD, &[10, 11, 12, 13, 14, 15, 16, 17, 200, 201, 202]);

        // Only 4 tokens [14, 15, 16, 17] match: `min_n=8` rejects while `min_n=4` matches.
        assert_eq!(
            ring.propose_continuation_min_n(0x9999, &[90, 91, 92, 93, 14, 15, 16], &[17], 4, 3, 8),
            None
        );
        assert_eq!(
            ring.propose_continuation_min_n(0x9999, &[90, 91, 92, 93, 14, 15, 16], &[17], 4, 3, 4),
            Some(vec![200, 201, 202])
        );

        // Full 8-gram [10..=17] matches under `min_n=8`.
        assert_eq!(
            ring.propose_continuation_min_n(0x8888, &[10, 11, 12, 13, 14, 15, 16], &[17], 4, 2, 8),
            Some(vec![200, 201])
        );
    }
}
