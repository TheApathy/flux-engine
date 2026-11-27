// SPDX-License-Identifier: AGPL-3.0-only

//! 4-gate safe in-context n-gram bonus for the DFlash2 candidate selector
//! (`ATLAS_DFLASH_NGRAM_BONUS=1`, default OFF).
//!
//! Designed specifically for extended reasoning (`<think>` blocks) where exact
//! repeated formulas, variable names, and step headers recur beyond the local
//! window while keeping greedy decoding 100% lossless and preventing
//! loop amplification or stochastic distribution drift:
//!
//! - **Gate 0 (`T=0` ONLY):** strictly bypassed whenever `temperature > 0.0`
//!   (both on host dispatch and inside `dflash2_selector_walk_ngram_batched`).
//! - **Gate 1 (Uncertainty):** fires ONLY when `top1 - top2 < margin_thresh`
//!   (default `1.0` logit). Confident drafter tokens are never perturbed.
//! - **Gate 2 (Top-4 verbatim 3/4-gram):** matches exact 4-gram first, then
//!   3-gram, and boosts ONLY if the matched continuation token is ALREADY in the
//!   selector's top-4 candidates (`0..min(4, top_k_shortlist)`).
//! - **Gate 3 (Hard logit cap):** bonus strictly capped at `<= 0.75` (`0.50`
//!   for 3-gram, `0.75` for 4-gram).
//! - **Gate 4 (Anti-loop `<= 64` window):** the per-sequence table builder
//!   excludes all matches whose occurrence lies inside the trailing 64 tokens
//!   AND blocks any 3/4-gram whose trailing 3-gram already occurs inside the
//!   trailing 64 tokens (preventing short-period repetition loops).

use std::collections::HashSet;
use std::sync::OnceLock;

use anyhow::{Result, ensure};

/// Environment variable gating the 4-gate safe in-context n-gram selector bonus.
pub const NGRAM_BONUS_ENV: &str = "ATLAS_DFLASH_NGRAM_BONUS";

/// Power-of-two slot count per sequence hash table (matches `DF2_NGRAM_TABLE_SLOTS`).
pub const NGRAM_TABLE_SLOTS: usize = 512;

/// Bitmask for open-addressed probing into [`NGRAM_TABLE_SLOTS`].
const NGRAM_TABLE_MASK: usize = NGRAM_TABLE_SLOTS - 1;

/// Maximum linear probe length (matches `DF2_NGRAM_MAX_PROBE`).
pub const NGRAM_MAX_PROBE: usize = 8;

/// Top-rank cap inside which a matched continuation may be boosted (`DF2_NGRAM_TOP_RANK`).
pub const NGRAM_TOP_RANK_CAP: usize = 4;

/// Hard upper bound on any logit bonus (`DF2_NGRAM_HARD_CAP`).
pub const NGRAM_HARD_CAP: f32 = 0.75;

/// Default uncertainty margin threshold (`top1 - top2 < margin_thresh`).
pub const NGRAM_DEFAULT_MARGIN: f32 = 1.0;

/// Default 3-gram bonus (clamped to `min(max_bonus, 0.50)`).
pub const NGRAM_TRIGRAM_CAP: f32 = 0.50;

/// Minimum trailing exclusion distance in tokens (Gate 4 anti-loop guard).
pub const NGRAM_ANTI_LOOP_WINDOW: usize = 64;

/// Bounded suffix scan window (tokens) so host table build stays O(1) (`~5 us`).
pub const NGRAM_SCAN_WINDOW: usize = 4096;

/// SplitMix64 domain salt for 3-gram keys (matches `DF2_NGRAM_SALT3`).
pub const NGRAM_SALT3: u64 = 0x9E37_79B9_7F4A_7C15;

/// SplitMix64 domain salt for 4-gram keys (matches `DF2_NGRAM_SALT4`).
pub const NGRAM_SALT4: u64 = 0xBF58_476D_1CE4_E5B9;

/// Sentinel for an absent token in `prev3_tokens` or empty hash probe (`DF2_NGRAM_EMPTY_TOK`).
pub const NGRAM_EMPTY_TOK: u32 = 0xFFFF_FFFF;

/// Exact byte size of one `NgramTableEntry` (`16 B`, matches CUDA `struct Df2NgramEntry`).
pub const NGRAM_ENTRY_BYTES: usize = 16;

/// Exact byte size of one sequence's 512-slot hash table (`8,192 B`).
pub const NGRAM_TABLE_BYTES: usize = NGRAM_TABLE_SLOTS * NGRAM_ENTRY_BYTES;

/// Exact byte size of one sequence's `prev3_tokens` triple (`12 B`).
pub const NGRAM_PREV3_BYTES: usize = 3 * std::mem::size_of::<u32>();

/// Cached process-wide gate for `ATLAS_DFLASH_NGRAM_BONUS=1` (default OFF).
#[must_use]
pub fn ngram_bonus_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let on = parse_ngram_bonus_env(std::env::var(NGRAM_BONUS_ENV).ok().as_deref());
        if on {
            tracing::info!(
                "DFlash2 4-gate safe in-context n-gram bonus ENABLED ({NGRAM_BONUS_ENV}=1, \
                 T=0 only, margin<{NGRAM_DEFAULT_MARGIN}, top-{NGRAM_TOP_RANK_CAP}, \
                 cap<={NGRAM_HARD_CAP}, anti-loop>={NGRAM_ANTI_LOOP_WINDOW})"
            );
        }
        on
    })
}

/// Pure env parser for [`NGRAM_BONUS_ENV`]: only `"1"` enables the feature.
#[inline]
#[must_use]
pub fn parse_ngram_bonus_env(raw: Option<&str>) -> bool {
    raw.map(str::trim) == Some("1")
}

/// Hyperparameters for the 4-gate in-context n-gram bonus.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NgramBonusConfig {
    /// Gate 1: fire only when `top1 - top2 < margin_thresh`.
    pub margin_thresh: f32,
    /// Gate 3: maximum bonus (always hard-clamped to `<= 0.75`).
    pub max_bonus: f32,
    /// Gate 4: exclude occurrences within the trailing `anti_loop_window` tokens
    /// (enforced `>= 64`).
    pub anti_loop_window: usize,
    /// Maximum historical tokens scanned when populating the sequence table.
    pub scan_window: usize,
}

impl Default for NgramBonusConfig {
    fn default() -> Self {
        Self {
            margin_thresh: NGRAM_DEFAULT_MARGIN,
            max_bonus: NGRAM_HARD_CAP,
            anti_loop_window: NGRAM_ANTI_LOOP_WINDOW,
            scan_window: NGRAM_SCAN_WINDOW,
        }
    }
}

impl NgramBonusConfig {
    /// Effective clamped bonus strictly respecting Gate 3 (`0.0..=0.75`).
    #[inline]
    #[must_use]
    pub fn effective_max_bonus(self) -> f32 {
        if self.max_bonus.is_finite() && self.max_bonus > 0.0 {
            self.max_bonus.min(NGRAM_HARD_CAP)
        } else {
            0.0
        }
    }

    /// Effective anti-loop window strictly respecting Gate 4 (`>= 64`).
    #[inline]
    #[must_use]
    pub fn effective_anti_loop_window(self) -> usize {
        self.anti_loop_window.max(NGRAM_ANTI_LOOP_WINDOW)
    }
}

/// 16-byte device-wire table slot matching CUDA `struct Df2NgramEntry`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NgramTableEntry {
    pub key: u64,
    pub next_token: u32,
    pub match_len: u32,
}

const _: () = {
    assert!(std::mem::size_of::<NgramTableEntry>() == NGRAM_ENTRY_BYTES);
    assert!(std::mem::align_of::<NgramTableEntry>() == 8);
};

/// Inputs deciding whether C=1 DFlash2 selector dispatch may engage the
/// 4-gate in-context n-gram walk.
#[derive(Debug, Clone, Copy)]
pub(super) struct C1NgramAdmission {
    pub ngram_enabled: bool,
    pub ngram_ready: bool,
    pub use_rejection_selector: bool,
    pub is_probabilistic_sampled: bool,
    pub markov_active: bool,
    pub probabilistic_method: bool,
    pub raw_gumbel_spec: Option<spark_runtime::sampler::gumbel::GumbelDraftSpec>,
}

/// Fail-closed Gate-0 admission check for C=1 eager and Stage-1/Stage-2 graph
/// selector routing.
///
/// Inspects the **raw** `raw_gumbel_spec` BEFORE any rejection-fallback
/// erasure (`staged_spec` / `effective_gumbel_spec`), guaranteeing that a
/// stochastic `temperature > 0.0` request NEVER falls into the greedy n-gram
/// walk even when rejection sampling declines rejection mode
/// (e.g. missing slot or unready rejection scratch).
#[inline]
#[must_use]
pub(super) fn admit_c1_ngram_selector(input: C1NgramAdmission) -> bool {
    input.ngram_enabled
        && input.ngram_ready
        && !input.use_rejection_selector
        && !input.is_probabilistic_sampled
        && !input.markov_active
        && !input.probabilistic_method
        && !input.raw_gumbel_spec.is_some_and(|spec| spec.is_active())
}

/// Open-addressed 512-slot per-sequence n-gram lookup table (`8,192 B`).
#[derive(Debug, Clone)]
pub struct NgramSequenceTable {
    entries: [NgramTableEntry; NGRAM_TABLE_SLOTS],
}

impl Default for NgramSequenceTable {
    fn default() -> Self {
        Self::empty()
    }
}

impl NgramSequenceTable {
    /// Create an all-zero empty table (every probe misses).
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            entries: [NgramTableEntry {
                key: 0,
                next_token: 0,
                match_len: 0,
            }; NGRAM_TABLE_SLOTS],
        }
    }

    /// Read-only view of all 512 slots.
    #[inline]
    #[must_use]
    pub fn entries(&self) -> &[NgramTableEntry; NGRAM_TABLE_SLOTS] {
        &self.entries
    }

    /// Insert or update `(key, match_len) -> next_token` with bounded linear probing.
    ///
    /// More recent valid matches (outside the anti-loop window) overwrite older
    /// entries for the exact same `(key, match_len)`. Returns `false` if `key == 0`
    /// (reserved empty-slot sentinel), `match_len == 0`, or the probe window is full.
    pub fn insert(&mut self, key: u64, next_token: u32, match_len: u32) -> bool {
        if key == 0 || match_len == 0 {
            return false;
        }
        let mut slot = (key as usize) & NGRAM_TABLE_MASK;
        for _ in 0..NGRAM_MAX_PROBE {
            let entry = &mut self.entries[slot];
            if entry.key == 0 || (entry.key == key && entry.match_len == match_len) {
                *entry = NgramTableEntry {
                    key,
                    next_token,
                    match_len,
                };
                return true;
            }
            slot = (slot + 1) & NGRAM_TABLE_MASK;
        }
        false
    }

    /// Probe `(key, expected_len)` using the exact same loop as CUDA `df2_ngram_probe`.
    #[must_use]
    pub fn probe(&self, key: u64, expected_len: u32) -> u32 {
        let mut slot = (key as usize) & NGRAM_TABLE_MASK;
        for _ in 0..NGRAM_MAX_PROBE {
            let entry = self.entries[slot];
            if entry.key == 0 {
                return NGRAM_EMPTY_TOK;
            }
            if entry.key == key && entry.match_len == expected_len {
                return entry.next_token;
            }
            slot = (slot + 1) & NGRAM_TABLE_MASK;
        }
        NGRAM_EMPTY_TOK
    }

    /// Serialize this 512-slot table into exact little-endian CUDA wire bytes (`8,192 B`).
    ///
    /// # Errors
    ///
    /// Returns an error if `dst.len() < NGRAM_TABLE_BYTES`.
    pub fn write_le_bytes_into(&self, dst: &mut [u8]) -> Result<usize> {
        ensure!(
            dst.len() >= NGRAM_TABLE_BYTES,
            "ngram table buffer too small: {} < {NGRAM_TABLE_BYTES}",
            dst.len(),
        );
        for (idx, entry) in self.entries.iter().enumerate() {
            let off = idx * NGRAM_ENTRY_BYTES;
            dst[off..off + 8].copy_from_slice(&entry.key.to_le_bytes());
            dst[off + 8..off + 12].copy_from_slice(&entry.next_token.to_le_bytes());
            dst[off + 12..off + 16].copy_from_slice(&entry.match_len.to_le_bytes());
        }
        Ok(NGRAM_TABLE_BYTES)
    }
}

/// Bit-exact SplitMix64 finalizer matching CUDA `df2_splitmix64_mix`.
#[inline]
#[must_use]
pub const fn splitmix64_mix(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Bit-exact 3-gram hash key matching CUDA `df2_ngram_key3`.
#[inline]
#[must_use]
pub const fn ngram_key3(a: u32, b: u32, c: u32) -> u64 {
    let mut h = splitmix64_mix((a as u64) ^ NGRAM_SALT3);
    h = splitmix64_mix(h ^ (b as u64));
    h = splitmix64_mix(h ^ (c as u64));
    if h == 0 { 1 } else { h }
}

/// Bit-exact 4-gram hash key matching CUDA `df2_ngram_key4`.
#[inline]
#[must_use]
pub const fn ngram_key4(a: u32, b: u32, c: u32, d: u32) -> u64 {
    let mut h = splitmix64_mix((a as u64) ^ NGRAM_SALT4);
    h = splitmix64_mix(h ^ (b as u64));
    h = splitmix64_mix(h ^ (c as u64));
    h = splitmix64_mix(h ^ (d as u64));
    if h == 0 { 1 } else { h }
}

/// Extract `[t-3, t-2, t-1]` preceding `anchor_id` from committed `haystack`.
///
/// Missing entries are filled with [`NGRAM_EMPTY_TOK`]. When `haystack` already
/// ends with `anchor_id`, `prev3` reads the three tokens immediately preceding
/// that final occurrence; otherwise it reads the last three tokens of `haystack`.
#[must_use]
pub fn extract_prev3_tokens(haystack: &[u32], anchor_id: u32) -> [u32; 3] {
    let prefix = if haystack.last().copied() == Some(anchor_id) {
        &haystack[..haystack.len() - 1]
    } else {
        haystack
    };
    let n = prefix.len();
    [
        if n >= 3 {
            prefix[n - 3]
        } else {
            NGRAM_EMPTY_TOK
        },
        if n >= 2 {
            prefix[n - 2]
        } else {
            NGRAM_EMPTY_TOK
        },
        if n >= 1 {
            prefix[n - 1]
        } else {
            NGRAM_EMPTY_TOK
        },
    ]
}

/// Build a Gate-4 anti-loop protected sequence n-gram table and `[t-3, t-2, t-1]`
/// prefix from committed `haystack` plus current `anchor_id`.
///
/// Gate 4 guarantees:
/// 1. Every indexed 3-gram or 4-gram occurrence ends strictly more than
///    `anti_loop_window` (`>= 64`) tokens before the current anchor position.
/// 2. Any 3-gram `(a, b, c)` appearing anywhere inside the trailing
///    `anti_loop_window` tokens is blacklisted, so short-period loops
///    (e.g. `A B C A B C ...`) can NEVER be reinforced even if the same pattern
///    also occurred >64 tokens earlier.
#[must_use]
pub fn build_sequence_ngram_table(
    haystack: &[u32],
    anchor_id: u32,
    cfg: NgramBonusConfig,
) -> (NgramSequenceTable, [u32; 3]) {
    let prev3 = extract_prev3_tokens(haystack, anchor_id);
    let mut table = NgramSequenceTable::empty();

    let anti_loop = cfg.effective_anti_loop_window();
    // Effective stream up to and including `anchor_id`:
    // if `haystack` already ends with `anchor_id`, its length is `haystack.len()`,
    // otherwise the conceptual stream has length `haystack.len() + 1`.
    let trailing_matches_anchor = haystack.last().copied() == Some(anchor_id);
    let full_len = if trailing_matches_anchor {
        haystack.len()
    } else {
        haystack.len() + 1
    };

    // Need at least 4 historical tokens strictly before the anti-loop window.
    if full_len <= anti_loop + 3 {
        return (table, prev3);
    }

    // Collect every *completed* 3-gram occurrence whose continuation lies inside
    // the trailing `anti_loop` window (`next_idx in full_len - anti_loop .. full_len`).
    // Notice this intentionally excludes the *current* query 3-gram ending at
    // `full_len - 1` (which has no continuation yet), while blacklisting any
    // 3-gram that already repeated within the last 64 tokens.
    let recent_start = full_len.saturating_sub(anti_loop);
    let mut recent_3grams: HashSet<u64> = HashSet::with_capacity(anti_loop);
    let token_at = |idx: usize| -> u32 {
        if idx < haystack.len() {
            haystack[idx]
        } else {
            anchor_id
        }
    };
    for next_idx in recent_start.max(3)..full_len {
        recent_3grams.insert(ngram_key3(
            token_at(next_idx - 3),
            token_at(next_idx - 2),
            token_at(next_idx - 1),
        ));
    }

    // Eligible continuation index `next_idx` must satisfy `next_idx < full_len - anti_loop`,
    // i.e. distance from current position `(full_len - 1)` is `>= anti_loop`.
    let eligible_end = full_len - anti_loop;
    let scan_start = eligible_end.saturating_sub(cfg.scan_window);

    // Populate in forward chronological order so the most recent valid
    // occurrence (outside the anti-loop window) wins on key updates.
    for next_idx in scan_start..eligible_end {
        let next_tok = token_at(next_idx);
        if next_idx >= 3 {
            let t0 = token_at(next_idx - 3);
            let t1 = token_at(next_idx - 2);
            let t2 = token_at(next_idx - 1);
            let key3 = ngram_key3(t0, t1, t2);
            if !recent_3grams.contains(&key3) {
                let _ = table.insert(key3, next_tok, 3);
                if next_idx >= 4 {
                    let t_prev = token_at(next_idx - 4);
                    let key4 = ngram_key4(t_prev, t0, t1, t2);
                    let _ = table.insert(key4, next_tok, 4);
                }
            }
        }
    }

    (table, prev3)
}

/// CPU reference implementation mirroring CUDA `dflash2_selector_walk_ngram_batched`.
///
/// Computes exact `s_score[k] = unary[t, k] + sum_r pred[prev, r] * hidden[t, r] * succ[cand, r]`,
/// applies Gates 0..3 against `table` (which enforces Gate 4), and returns
/// `(chosen_tokens, top1_softmax_probs)` for all `gamma` positions.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors dflash2_selector_walk_ngram_batched ABI verbatim for exact CPU/GPU parity tests"
)]
#[must_use]
pub fn dflash2_selector_walk_ngram_cpu(
    unary: &[f32],
    candidates: &[u32],
    hidden: &[f32],
    pred: &[f32],
    succ: &[f32],
    anchor_id: u32,
    prev3: [u32; 3],
    table: Option<&NgramSequenceTable>,
    temperature: f32,
    cfg: NgramBonusConfig,
    top_k_shortlist: usize,
    gamma: usize,
    rank: usize,
) -> (Vec<u32>, Vec<f32>) {
    if top_k_shortlist == 0 || top_k_shortlist > 32 || gamma == 0 {
        return (Vec::new(), Vec::new());
    }
    let gate0_greedy_ok = temperature.is_finite() && temperature <= 0.0;
    let eff_max_bonus = cfg.effective_max_bonus();
    let ngram_active = gate0_greedy_ok
        && table.is_some()
        && cfg.margin_thresh.is_finite()
        && cfg.margin_thresh > 0.0
        && eff_max_bonus > 0.0;

    let mut hist = [prev3[0], prev3[1], prev3[2], anchor_id];
    let mut out_tokens = Vec::with_capacity(gamma);
    let mut out_probs = Vec::with_capacity(gamma);
    let mut scores = vec![0.0f32; top_k_shortlist];
    let mut cands = vec![0u32; top_k_shortlist];

    for t in 0..gamma {
        let prev_tok = hist[3] as usize;
        let pr = &pred[prev_tok * rank..(prev_tok + 1) * rank];
        let hr = &hidden[t * rank..(t + 1) * rank];

        for k in 0..top_k_shortlist {
            let cand = candidates[t * top_k_shortlist + k];
            let sr = &succ[cand as usize * rank..(cand as usize + 1) * rank];
            let mut acc = 0.0f32;
            for r in 0..rank {
                acc += pr[r] * hr[r] * sr[r];
            }
            scores[k] = unary[t * top_k_shortlist + k] + acc;
            cands[k] = cand;
        }

        // Top-4 slots descending by (score, ~cand_id).
        let rank_cap = top_k_shortlist.min(NGRAM_TOP_RANK_CAP);
        let mut top_idx = [0usize; NGRAM_TOP_RANK_CAP];
        let mut used_mask = 0u32;
        for slot_out in top_idx.iter_mut().take(rank_cap) {
            let mut best: Option<usize> = None;
            for j in 0..top_k_shortlist {
                if (used_mask & (1u32 << j)) != 0 {
                    continue;
                }
                let take = match best {
                    None => true,
                    Some(b) => {
                        scores[j] > scores[b] || (scores[j] == scores[b] && cands[j] < cands[b])
                    }
                };
                if take {
                    best = Some(j);
                }
            }
            let chosen_rank = best.unwrap_or(0);
            *slot_out = chosen_rank;
            used_mask |= 1u32 << chosen_rank;
        }

        if ngram_active
            && rank_cap >= 2
            && let Some(tbl) = table
        {
            let margin = scores[top_idx[0]] - scores[top_idx[1]];
            if margin.is_finite() && margin < cfg.margin_thresh {
                let mut matched_tok = NGRAM_EMPTY_TOK;
                let mut bonus = 0.0f32;
                if hist[0] != NGRAM_EMPTY_TOK
                    && hist[1] != NGRAM_EMPTY_TOK
                    && hist[2] != NGRAM_EMPTY_TOK
                {
                    matched_tok = tbl.probe(ngram_key4(hist[0], hist[1], hist[2], hist[3]), 4);
                    if matched_tok != NGRAM_EMPTY_TOK {
                        bonus = eff_max_bonus;
                    }
                }
                if matched_tok == NGRAM_EMPTY_TOK
                    && hist[1] != NGRAM_EMPTY_TOK
                    && hist[2] != NGRAM_EMPTY_TOK
                {
                    matched_tok = tbl.probe(ngram_key3(hist[1], hist[2], hist[3]), 3);
                    if matched_tok != NGRAM_EMPTY_TOK {
                        bonus = eff_max_bonus.min(NGRAM_TRIGRAM_CAP);
                    }
                }
                if matched_tok != NGRAM_EMPTY_TOK && bonus > 0.0 {
                    for &slot in top_idx.iter().take(rank_cap) {
                        if cands[slot] == matched_tok {
                            scores[slot] += bonus;
                            break;
                        }
                    }
                }
            }
        }

        let mut best = scores[0];
        let mut best_k = 0usize;
        for (j, &sc) in scores.iter().enumerate().skip(1) {
            if sc > best {
                best = sc;
                best_k = j;
            }
        }
        let chosen = cands[best_k];
        let mut denom = 0.0f32;
        for &sc in &scores {
            let term = (sc - best).exp();
            if term.is_finite() {
                denom += term;
            }
        }
        let prob = if denom > 0.0 && denom.is_finite() {
            1.0 / denom
        } else {
            1.0
        };
        out_tokens.push(chosen);
        out_probs.push(prob);
        hist = [hist[1], hist[2], hist[3], chosen];
    }

    (out_tokens, out_probs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zero_codebooks(vocab: usize, rank: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        (
            vec![0.0; rank],
            vec![0.0; vocab * rank],
            vec![0.0; vocab * rank],
        )
    }

    /// Build a synthetic history with an old verbatim 4-gram `(10, 11, 12, 13) -> 42`
    /// followed by `gap` distinct filler tokens (avoiding any recent repetition),
    /// ending at `(10, 11, 12)` with anchor `13`.
    fn history_with_old_4gram(gap: usize) -> Vec<u32> {
        let mut toks = vec![10, 11, 12, 13, 42];
        for i in 0..gap {
            toks.push(1000 + i as u32);
        }
        toks.extend_from_slice(&[10, 11, 12]);
        toks
    }

    #[test]
    fn env_parser_defaults_off_and_accepts_only_one() {
        assert!(!parse_ngram_bonus_env(None));
        assert!(!parse_ngram_bonus_env(Some("")));
        assert!(!parse_ngram_bonus_env(Some("0")));
        assert!(!parse_ngram_bonus_env(Some("true")));
        assert!(!parse_ngram_bonus_env(Some("2")));
        assert!(parse_ngram_bonus_env(Some("1")));
        assert!(parse_ngram_bonus_env(Some(" 1 ")));
    }

    #[test]
    fn gate0_never_fires_when_temperature_is_positive() {
        let history = history_with_old_4gram(80);
        let cfg = NgramBonusConfig::default();
        let (table, prev3) = build_sequence_ngram_table(&history, 13, cfg);
        let (hidden, pred, succ) = zero_codebooks(100, 4);
        // Tight margin (0.40 < 1.0): candidate 42 is rank-1 (index 1), would win at T=0.
        let unary = [5.0, 4.6, 1.0, 0.5, 0.2, 0.1];
        let cands = [7, 42, 8, 9, 10, 11];

        let (greedy_toks, _) = dflash2_selector_walk_ngram_cpu(
            &unary,
            &cands,
            &hidden,
            &pred,
            &succ,
            13,
            prev3,
            Some(&table),
            0.0,
            cfg,
            6,
            1,
            4,
        );
        assert_eq!(
            greedy_toks,
            vec![42],
            "T=0 must promote top-4 verbatim 4-gram"
        );

        for temp in [
            1e-4_f32,
            0.3,
            0.6,
            1.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ] {
            let (sampled_toks, _) = dflash2_selector_walk_ngram_cpu(
                &unary,
                &cands,
                &hidden,
                &pred,
                &succ,
                13,
                prev3,
                Some(&table),
                temp,
                cfg,
                6,
                1,
                4,
            );
            assert_eq!(
                sampled_toks,
                vec![7],
                "Gate 0 violated: n-gram fired at temperature={temp}"
            );
        }
    }

    #[test]
    fn gate0_c1_admission_refuses_active_gumbel_spec_even_under_rejection_fallback() {
        let active_spec = spark_runtime::sampler::gumbel::GumbelDraftSpec {
            seed: 42,
            step0_abs_pos: 100,
            temperature: 0.6,
            top_k: 20,
            top_p: 0.95,
            min_p: 0.0,
        };
        let greedy_spec = spark_runtime::sampler::gumbel::GumbelDraftSpec {
            temperature: 0.0,
            ..active_spec
        };
        let base = C1NgramAdmission {
            ngram_enabled: true,
            ngram_ready: true,
            use_rejection_selector: false,
            is_probabilistic_sampled: false,
            markov_active: false,
            probabilistic_method: false,
            raw_gumbel_spec: None,
        };
        assert!(admit_c1_ngram_selector(base));
        assert!(admit_c1_ngram_selector(C1NgramAdmission {
            raw_gumbel_spec: Some(greedy_spec),
            ..base
        }));
        // Even when rejection fallback sets `use_rejection_selector = false` and
        // erases `staged_spec`, an active T>0 raw_gumbel_spec MUST stay blocked.
        assert!(!admit_c1_ngram_selector(C1NgramAdmission {
            raw_gumbel_spec: Some(active_spec),
            ..base
        }));
    }

    #[test]
    fn table_insert_rejects_sentinel_zero_and_cpu_walk_handles_edge_shortlists() {
        let mut table = NgramSequenceTable::empty();
        assert!(!table.insert(0, 42, 4));
        assert!(!table.insert(ngram_key4(1, 2, 3, 4), 42, 0));
        let k = ngram_key4(1, 2, 3, 4);
        assert!(table.insert(k, 99, 4));
        assert_eq!(table.probe(k, 4), 99);

        let cfg = NgramBonusConfig::default();
        assert_eq!(
            dflash2_selector_walk_ngram_cpu(
                &[],
                &[],
                &[],
                &[],
                &[],
                0,
                [NGRAM_EMPTY_TOK; 3],
                Some(&table),
                0.0,
                cfg,
                0,
                1,
                4,
            ),
            (Vec::new(), Vec::new())
        );
        assert_eq!(
            dflash2_selector_walk_ngram_cpu(
                &[],
                &[],
                &[],
                &[],
                &[],
                0,
                [NGRAM_EMPTY_TOK; 3],
                Some(&table),
                0.0,
                cfg,
                33,
                1,
                4,
            ),
            (Vec::new(), Vec::new())
        );
    }

    #[test]
    fn gate1_never_fires_on_confident_tokens() {
        let history = history_with_old_4gram(80);
        let cfg = NgramBonusConfig {
            margin_thresh: 1.0,
            ..NgramBonusConfig::default()
        };
        let (table, prev3) = build_sequence_ngram_table(&history, 13, cfg);
        let (hidden, pred, succ) = zero_codebooks(100, 4);
        // Margin is 1.01 >= 1.0: confident token must never be flipped.
        let unary = [6.01, 5.00, 1.0, 0.5];
        let cands = [7, 42, 8, 9];
        let (toks, _) = dflash2_selector_walk_ngram_cpu(
            &unary,
            &cands,
            &hidden,
            &pred,
            &succ,
            13,
            prev3,
            Some(&table),
            0.0,
            cfg,
            4,
            1,
            4,
        );
        assert_eq!(toks, vec![7]);
    }

    #[test]
    fn gate2_never_fires_outside_top_four_candidates() {
        let history = history_with_old_4gram(80);
        let cfg = NgramBonusConfig::default();
        let (table, prev3) = build_sequence_ngram_table(&history, 13, cfg);
        let (hidden, pred, succ) = zero_codebooks(100, 4);
        // Margin is tiny (0.10 < 1.0), and candidate 42 is within 0.40 of top-1,
        // BUT candidate 42 is ranked 5th (rank index 4 >= 4) so Gate 2 MUST reject it.
        let unary = [5.00, 4.90, 4.80, 4.70, 4.60, 1.00];
        let cands = [1, 2, 3, 4, 42, 6];
        let (toks, _) = dflash2_selector_walk_ngram_cpu(
            &unary,
            &cands,
            &hidden,
            &pred,
            &succ,
            13,
            prev3,
            Some(&table),
            0.0,
            cfg,
            6,
            1,
            4,
        );
        assert_eq!(
            toks,
            vec![1],
            "Gate 2 violated: rank-5 candidate was promoted"
        );
    }

    #[test]
    fn gate3_hard_caps_bonus_at_0_75_and_trigram_at_0_50() {
        let history = history_with_old_4gram(80);
        // Even if caller requests max_bonus = 5.0, Gate 3 clamps strictly to 0.75.
        let cfg = NgramBonusConfig {
            margin_thresh: 2.0,
            max_bonus: 5.0,
            ..NgramBonusConfig::default()
        };
        assert!((cfg.effective_max_bonus() - 0.75).abs() < 1e-6);

        let (table, prev3) = build_sequence_ngram_table(&history, 13, cfg);
        let (hidden, pred, succ) = zero_codebooks(100, 4);

        // Gap = 0.80 > 0.75 cap -> 4-gram bonus CANNOT flip top-1.
        let unary_too_far = [5.00, 4.20, 1.0, 0.5];
        let cands = [7, 42, 8, 9];
        let (toks_far, _) = dflash2_selector_walk_ngram_cpu(
            &unary_too_far,
            &cands,
            &hidden,
            &pred,
            &succ,
            13,
            prev3,
            Some(&table),
            0.0,
            cfg,
            4,
            1,
            4,
        );
        assert_eq!(toks_far, vec![7]);

        // 3-gram only (mismatch on oldest token prev3[0]): capped at 0.50,
        // so gap 0.60 cannot flip, while gap 0.40 does flip.
        let prev3_only_trigram = [999, 11, 12];
        let (toks_tri_blocked, _) = dflash2_selector_walk_ngram_cpu(
            &[5.00, 4.40, 1.0, 0.5],
            &cands,
            &hidden,
            &pred,
            &succ,
            13,
            prev3_only_trigram,
            Some(&table),
            0.0,
            cfg,
            4,
            1,
            4,
        );
        assert_eq!(toks_tri_blocked, vec![7]);

        let (toks_tri_ok, _) = dflash2_selector_walk_ngram_cpu(
            &[5.00, 4.60, 1.0, 0.5],
            &cands,
            &hidden,
            &pred,
            &succ,
            13,
            prev3_only_trigram,
            Some(&table),
            0.0,
            cfg,
            4,
            1,
            4,
        );
        assert_eq!(toks_tri_ok, vec![42]);
    }

    #[test]
    fn gate4_blocks_matches_inside_anti_loop_window_and_recent_cycles() {
        let cfg = NgramBonusConfig::default();

        // Occurrence only 32 tokens ago (inside 64-token anti-loop window): MUST NOT index.
        let recent_only = history_with_old_4gram(32);
        let (tbl_recent, _) = build_sequence_ngram_table(&recent_only, 13, cfg);
        assert_eq!(
            tbl_recent.probe(ngram_key4(10, 11, 12, 13), 4),
            NGRAM_EMPTY_TOK
        );
        assert_eq!(tbl_recent.probe(ngram_key3(11, 12, 13), 3), NGRAM_EMPTY_TOK);

        // Short-period repetition loop: pattern occurred 100 tokens ago AND ALSO
        // occurred 10 tokens ago -> recent_3grams blacklist MUST block amplification!
        let mut looped = history_with_old_4gram(90);
        looped.extend_from_slice(&[13, 42, 500, 501, 502, 10, 11, 12]);
        let (tbl_loop, _) = build_sequence_ngram_table(&looped, 13, cfg);
        assert_eq!(
            tbl_loop.probe(ngram_key4(10, 11, 12, 13), 4),
            NGRAM_EMPTY_TOK,
            "Gate 4 violated: recent repeating 3-gram loop was indexed"
        );
        assert_eq!(
            tbl_loop.probe(ngram_key3(11, 12, 13), 3),
            NGRAM_EMPTY_TOK,
            "Gate 4 violated: recent repeating 3-gram loop was indexed"
        );

        // Legitimate long-range verbatim recall (> 64 tokens ago, not repeating locally):
        // MUST index and recover exact continuation 42.
        let clean_old = history_with_old_4gram(80);
        let (tbl_clean, prev3) = build_sequence_ngram_table(&clean_old, 13, cfg);
        assert_eq!(prev3, [10, 11, 12]);
        assert_eq!(tbl_clean.probe(ngram_key4(10, 11, 12, 13), 4), 42);
        assert_eq!(tbl_clean.probe(ngram_key3(11, 12, 13), 3), 42);
    }

    #[test]
    fn wire_serialization_matches_cuda_df2_ngram_entry_layout() {
        let mut table = NgramSequenceTable::empty();
        let k3 = ngram_key3(11, 12, 13);
        let k4 = ngram_key4(10, 11, 12, 13);
        assert!(table.insert(k3, 42, 3));
        assert!(table.insert(k4, 42, 4));
        let mut buf = vec![0u8; NGRAM_TABLE_BYTES];
        let written = table
            .write_le_bytes_into(&mut buf)
            .expect("serialize table");
        assert_eq!(written, NGRAM_TABLE_BYTES);

        for (key, expected_len) in [(k3, 3u32), (k4, 4u32)] {
            let mut slot = (key as usize) & NGRAM_TABLE_MASK;
            let mut found = false;
            for _ in 0..NGRAM_MAX_PROBE {
                let off = slot * NGRAM_ENTRY_BYTES;
                let wire_key = u64::from_le_bytes(buf[off..off + 8].try_into().unwrap());
                let wire_tok = u32::from_le_bytes(buf[off + 8..off + 12].try_into().unwrap());
                let wire_len = u32::from_le_bytes(buf[off + 12..off + 16].try_into().unwrap());
                if wire_key == key && wire_len == expected_len {
                    assert_eq!(wire_tok, 42);
                    found = true;
                    break;
                }
                slot = (slot + 1) & NGRAM_TABLE_MASK;
            }
            assert!(
                found,
                "expected key {key:#x} len={expected_len} in wire bytes"
            );
        }
    }
}
