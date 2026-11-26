// SPDX-License-Identifier: AGPL-3.0-only

//! Position- and token-ID-keyed stateless Gumbel-Max sampling and CRN draft coupling.
//!
//! Every `(seed, abs_pos, token_id)` triple maps deterministically via two-round
//! SplitMix64 to an independent `Gumbel(0, 1)` variate without sequential RNG state.
//! By the Gumbel-Max theorem, `argmax_{j in kept} (logit_j / T + G(seed, pos, id_j))`
//! draws from the exact categorical softmax over `kept`, while making speculative
//! verification at any `T > 0` a pure deterministic equality check (`draft == target`)
//! that is 100% bit-identical to `K=1` serial decoding.

use std::cell::Cell;
use std::sync::OnceLock;

use parking_lot::Mutex;

use super::candidate_order::{CandidateOrder, order_candidates};

/// Largest IEEE-754 `f64` strictly below `1.0` (`1.0 - 2^-53`).
const MAX_UNIT_F64: f64 = f64::from_bits(0x3FEF_FFFF_FFFF_FFFF);

/// Largest IEEE-754 `f32` strictly below `1.0` (`1.0 - 2^-24`).
const MAX_UNIT_F32: f32 = f32::from_bits(0x3F7F_FFFF);

/// Minimum effective temperature divisor for Gumbel-Max logit scaling.
pub const GUMBEL_MIN_TEMPERATURE: f32 = 1.0e-6;

/// Mask keeping non-negative 63-bit seeds (`0 ..= i64::MAX as u64`).
pub const PROMPT_SEED_MASK: u64 = 0x7FFF_FFFF_FFFF_FFFF;

/// Finalizer of Stafford's Mix13 (`SplitMix64`).
#[inline]
#[must_use]
pub const fn splitmix64_mix(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Position-keyed base word shared across all vocabulary IDs at `(seed, abs_pos)`.
#[inline]
#[must_use]
pub const fn position_base(seed: u64, abs_pos: u64) -> u64 {
    splitmix64_mix(
        splitmix64_mix(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
            ^ abs_pos.wrapping_mul(0xD1B5_4A32_D192_ED03),
    )
}

/// Stateless 64-bit random bits for `(seed, abs_pos, token_id)`.
#[inline]
#[must_use]
pub const fn key_bits(seed: u64, abs_pos: u64, token_id: u32) -> u64 {
    splitmix64_mix(position_base(seed, abs_pos) ^ (token_id as u64))
}

/// Stateless uniform variate in `(0, 1)` with 53-bit mantissa resolution.
#[inline]
#[must_use]
pub fn uniform_f64(seed: u64, abs_pos: u64, token_id: u32) -> f64 {
    let bits = key_bits(seed, abs_pos, token_id);
    let u53 = ((bits >> 11) as f64) * 2f64.powi(-53) + 2f64.powi(-54);
    u53.min(MAX_UNIT_F64)
}

/// Stateless `Gumbel(0, 1)` variate in `f64` for `(seed, abs_pos, token_id)`.
#[inline]
#[must_use]
pub fn gumbel_f64(seed: u64, abs_pos: u64, token_id: u32) -> f64 {
    let u53 = uniform_f64(seed, abs_pos, token_id);
    -((-u53.ln()).ln())
}

/// GPU-compatible `f32` uniform variate in `(0, 1)` using the top 24 bits of `key_bits`.
///
/// Uses `bits >> 40`, which is the upper 24-bit prefix of `bits >> 11` (`u53`), so
/// `gumbel_f32` on the MTP drafter and `gumbel_f64` on the 248K target head share
/// the exact same underlying uniform draw (Common Random Numbers).
#[inline]
#[must_use]
pub fn uniform_f32(seed: u64, abs_pos: u64, token_id: u32) -> f32 {
    let bits = key_bits(seed, abs_pos, token_id);
    let u24 = ((bits >> 40) as f32) * 2f32.powi(-24) + 2f32.powi(-25);
    u24.min(MAX_UNIT_F32)
}

/// GPU-compatible `Gumbel(0, 1)` variate in `f32` matching `sampler_topk_candidates.cu`.
#[inline]
#[must_use]
pub fn gumbel_f32(seed: u64, abs_pos: u64, token_id: u32) -> f32 {
    let u24 = uniform_f32(seed, abs_pos, token_id);
    -((-u24.ln()).ln())
}

/// Derive a deterministic 63-bit seed from `tokens` via SHA-256 + SplitMix64.
#[must_use]
pub fn seed_from_prompt_tokens(tokens: &[u32]) -> u64 {
    let mut ctx = ring::digest::Context::new(&ring::digest::SHA256);
    ctx.update(&(tokens.len() as u64).to_le_bytes());
    for &tok in tokens {
        ctx.update(&tok.to_le_bytes());
    }
    let digest = ctx.finish();
    let bytes = digest.as_ref();
    let raw = u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]);
    splitmix64_mix(raw) & PROMPT_SEED_MASK
}

/// Whether `FLUX_PROMPT_SEED=1` (or `ATLAS_PROMPT_SEED=1`) is enabled.
#[must_use]
pub fn prompt_seed_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("FLUX_PROMPT_SEED")
            .or_else(|_| std::env::var("ATLAS_PROMPT_SEED"))
            .ok()
            .as_deref()
            .map(str::trim)
            == Some("1")
    })
}

/// Resolve `seed`, falling back to [`seed_from_prompt_tokens`] when `FLUX_PROMPT_SEED=1`.
#[must_use]
pub fn resolve_seed_with_prompt(seed: Option<u64>, prompt_tokens: &[u32]) -> Option<u64> {
    seed.or_else(|| prompt_seed_enabled().then(|| seed_from_prompt_tokens(prompt_tokens)))
}

/// Pure parser for whether `FLUX_SPEC_VERIFY` selects Gumbel-Max sampling.
#[must_use]
pub fn is_gumbel_verify_env(value: Option<&str>) -> bool {
    matches!(value.map(str::trim), Some("gumbel"))
}

/// Pure parser for whether `FLUX_SPEC_VERIFY` selects GPU maximal-coupling rejection sampling.
#[must_use]
pub fn is_rejection_verify_env(value: Option<&str>) -> bool {
    matches!(value.map(str::trim), Some("rejection"))
}

/// Whether `FLUX_SPEC_VERIFY=gumbel` (or `ATLAS_SPEC_VERIFY=gumbel`) is active.
#[must_use]
pub fn gumbel_sampling_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let raw = std::env::var("FLUX_SPEC_VERIFY")
            .or_else(|_| std::env::var("ATLAS_SPEC_VERIFY"))
            .ok();
        is_gumbel_verify_env(raw.as_deref())
    })
}

/// Whether `FLUX_SPEC_VERIFY=rejection` (or `ATLAS_SPEC_VERIFY=rejection`) is active.
#[must_use]
pub fn rejection_sampling_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let raw = std::env::var("FLUX_SPEC_VERIFY")
            .or_else(|_| std::env::var("ATLAS_SPEC_VERIFY"))
            .ok();
        is_rejection_verify_env(raw.as_deref())
    })
}

/// Pure parser for `ATLAS_QWEN4_DFLASH_REJECT_TEMP_SCALE` (default `1.0`, valid finite `> 0.0`).
#[must_use]
pub fn parse_rejection_draft_temp_scale(raw: Option<&str>) -> f32 {
    raw.map(str::trim)
        .and_then(|s| s.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(1.0)
}

/// Draft-temperature multiplier `T_draft / T_target` for GPU maximal-coupling
/// rejection sampling (`ATLAS_QWEN4_DFLASH_REJECT_TEMP_SCALE`, default `1.0`).
#[must_use]
pub fn rejection_draft_temp_scale() -> f32 {
    static SCALE: OnceLock<f32> = OnceLock::new();
    *SCALE.get_or_init(|| {
        parse_rejection_draft_temp_scale(
            std::env::var("ATLAS_QWEN4_DFLASH_REJECT_TEMP_SCALE")
                .ok()
                .as_deref(),
        )
    })
}

/// Pure parser for `ATLAS_QWEN4_DFLASH_REJECT_SHORTLIST_K` (default `20`, clamped `1..=20`).
#[must_use]
pub fn parse_rejection_shortlist_k(raw: Option<&str>) -> usize {
    raw.map(str::trim)
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&k| k >= 1)
        .map(|k| k.min(20))
        .unwrap_or(20)
}

/// Shortlist width `K` (`1..=20`, default `20`) for GPU maximal-coupling draft
/// distribution capture (`ATLAS_QWEN4_DFLASH_REJECT_SHORTLIST_K`).
#[must_use]
pub fn rejection_shortlist_k() -> usize {
    static K: OnceLock<usize> = OnceLock::new();
    *K.get_or_init(|| {
        parse_rejection_shortlist_k(
            std::env::var("ATLAS_QWEN4_DFLASH_REJECT_SHORTLIST_K")
                .ok()
                .as_deref(),
        )
    })
}

thread_local! {
    /// Optional `(base_seed, abs_pos)` override for `sample_with_params_history` callers
    /// that already hold `ActiveSeq` (`abs_pos = prompt_len + output_tokens.len()`).
    static GUMBEL_POS_CTX: Cell<Option<(u64, u64)>> = const { Cell::new(None) };
}

/// Scoped RAII guard binding `(base_seed, abs_pos)` for host sampling calls on this thread.
pub struct GumbelPosGuard {
    prev: Option<(u64, u64)>,
}

impl GumbelPosGuard {
    #[must_use]
    pub fn new(seed: u64, abs_pos: u64) -> Self {
        let prev = GUMBEL_POS_CTX.with(|c| c.replace(Some((seed, abs_pos))));
        Self { prev }
    }
}

impl Drop for GumbelPosGuard {
    fn drop(&mut self) {
        GUMBEL_POS_CTX.with(|c| c.set(self.prev));
    }
}

/// Read the active thread-local `(base_seed, abs_pos)` override, if any.
#[inline]
#[must_use]
pub fn current_gumbel_pos() -> Option<(u64, u64)> {
    GUMBEL_POS_CTX.with(Cell::get)
}

/// Sampling parameters forwarded to the MTP drafter for CRN noise-coupled drafting.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GumbelDraftSpec {
    pub seed: u64,
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub min_p: f32,
    /// Expected `abs_pos` of `draft[0]` (`position + 1 == prompt_len + output_tokens.len()`).
    pub step0_abs_pos: u64,
}

impl GumbelDraftSpec {
    /// True when stochastic Gumbel coupling should perturb draft selection.
    #[inline]
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.temperature > 0.0 && self.temperature.is_finite()
    }
}

/// Bounded ring of recent sequence draft specs keyed by `(slot_or_row, propose_position)`
/// so both single-sequence (`C=1`) and cross-sequence (`C=2..8`) MTP proposers resolve
/// the exact request `(seed, T, top_k, top_p, min_p)` without cross-slot collisions or
/// stale carryover into subsequent greedy (`temperature == 0.0`) requests.
const DRAFT_SPEC_CAP: usize = 64;
const UNSLOTTED_DRAFT_SPEC: usize = usize::MAX;

struct DraftSpecRing {
    entries: [Option<(usize, usize, GumbelDraftSpec)>; DRAFT_SPEC_CAP],
    cursor: usize,
}

static DRAFT_SPECS: Mutex<DraftSpecRing> = Mutex::new(DraftSpecRing {
    entries: [None; DRAFT_SPEC_CAP],
    cursor: 0,
});

/// Clear all pending CRN Gumbel draft specs across every slot and position.
pub fn clear_gumbel_draft_specs() {
    let mut ring = DRAFT_SPECS.lock();
    ring.entries.fill(None);
    ring.cursor = 0;
}

/// Invalidate any pending CRN Gumbel draft spec for `slot_or_row` or unslotted `propose_position`.
pub fn invalidate_gumbel_draft_spec(slot_or_row: usize, propose_position: usize) {
    let mut ring = DRAFT_SPECS.lock();
    for entry in &mut ring.entries {
        if let Some((slot, pos, _)) = entry
            && (*slot == slot_or_row || (*slot == UNSLOTTED_DRAFT_SPEC && *pos == propose_position))
        {
            *entry = None;
        }
    }
}

/// Publish a sequence's CRN Gumbel draft spec scoped to `slot_or_row` and `propose_position`.
pub fn register_gumbel_draft_spec_for_slot(
    slot_or_row: usize,
    propose_position: usize,
    spec: GumbelDraftSpec,
) {
    if !spec.is_active() {
        invalidate_gumbel_draft_spec(slot_or_row, propose_position);
        return;
    }
    let mut ring = DRAFT_SPECS.lock();
    for entry in &mut ring.entries {
        if let Some((slot, pos, existing)) = entry
            && *slot == slot_or_row
            && *pos == propose_position
        {
            *existing = spec;
            return;
        }
    }
    let idx = ring.cursor % DRAFT_SPEC_CAP;
    ring.entries[idx] = Some((slot_or_row, propose_position, spec));
    ring.cursor = ring.cursor.wrapping_add(1);
}

/// Publish a sequence's CRN Gumbel draft spec before its next MTP proposal.
///
/// `propose_position` is the `position` argument passed to `DraftProposer::propose`
/// (`seq.seq_len`), so `draft[step]` predicts `abs_pos = propose_position + 1 + step`.
pub fn register_gumbel_draft_spec(propose_position: usize, spec: GumbelDraftSpec) {
    register_gumbel_draft_spec_for_slot(UNSLOTTED_DRAFT_SPEC, propose_position, spec);
}

/// Consume and remove the most recently registered CRN Gumbel draft spec for
/// `(slot_or_row, propose_position)` (falling back to unslotted `propose_position`).
#[must_use]
pub fn take_gumbel_draft_spec_for_slot(
    slot_or_row: usize,
    propose_position: usize,
) -> Option<GumbelDraftSpec> {
    let mut ring = DRAFT_SPECS.lock();
    for offset in 0..DRAFT_SPEC_CAP {
        let idx = ring.cursor.wrapping_sub(1).wrapping_sub(offset) % DRAFT_SPEC_CAP;
        if let Some((slot, pos, spec)) = ring.entries[idx]
            && slot == slot_or_row
            && pos == propose_position
            && spec.is_active()
        {
            ring.entries[idx] = None;
            return Some(spec);
        }
    }
    for offset in 0..DRAFT_SPEC_CAP {
        let idx = ring.cursor.wrapping_sub(1).wrapping_sub(offset) % DRAFT_SPEC_CAP;
        if let Some((slot, pos, spec)) = ring.entries[idx]
            && slot == UNSLOTTED_DRAFT_SPEC
            && pos == propose_position
            && spec.is_active()
        {
            ring.entries[idx] = None;
            return Some(spec);
        }
    }
    None
}

/// Consume and remove the most recently registered CRN Gumbel draft spec for `propose_position`.
#[must_use]
pub fn take_gumbel_draft_spec(propose_position: usize) -> Option<GumbelDraftSpec> {
    let mut ring = DRAFT_SPECS.lock();
    for offset in 0..DRAFT_SPEC_CAP {
        let idx = ring.cursor.wrapping_sub(1).wrapping_sub(offset) % DRAFT_SPEC_CAP;
        if let Some((_, pos, spec)) = ring.entries[idx]
            && pos == propose_position
            && spec.is_active()
        {
            ring.entries[idx] = None;
            return Some(spec);
        }
    }
    None
}

/// Look up and consume the most recently registered CRN Gumbel draft spec for `propose_position`.
#[must_use]
pub fn lookup_gumbel_draft_spec(propose_position: usize) -> Option<GumbelDraftSpec> {
    take_gumbel_draft_spec(propose_position)
}

/// Select a noise-coupled Gumbel draft token from a BF16 shortlist row on host,
/// returning `(global_token_id, chosen_token_prob)` where `chosen_token_prob` is
/// the post-temperature softmax probability `p_T(chosen_tok) in (0, 1]`.
///
/// `shortlist_ids`: optional compact->global token ID table (`None` = identity `0..V`).
#[must_use]
pub fn select_gumbel_shortlist_bf16(
    logits_bf16: &[u8],
    shortlist_ids: Option<&[u32]>,
    spec: GumbelDraftSpec,
    abs_pos: u64,
) -> Option<(u32, f32)> {
    let vocab = logits_bf16.len() / 2;
    if vocab == 0 {
        return None;
    }
    if let Some(ids) = shortlist_ids
        && ids.len() < vocab
    {
        return None;
    }
    let eff_temp = spec.temperature.max(GUMBEL_MIN_TEMPERATURE);
    let inv_t = 1.0f32 / eff_temp;
    let mut max_raw = f32::NEG_INFINITY;
    let mut candidates: Vec<(u32, f32)> = Vec::with_capacity(vocab);
    for (c, pair) in logits_bf16.chunks_exact(2).enumerate() {
        let bits = (u32::from(pair[0]) | (u32::from(pair[1]) << 8)) << 16;
        let raw = f32::from_bits(bits);
        if !raw.is_finite() {
            continue;
        }
        if raw > max_raw {
            max_raw = raw;
        }
        let global_id = shortlist_ids.map_or(c as u32, |ids| ids[c]);
        candidates.push((global_id, raw * inv_t));
    }
    if candidates.is_empty() {
        return None;
    }
    // Compute post-temperature softmax denominator over all finite logits to
    // match `mtp_gumbel_argmax_bf16` and report true `p_T(chosen_tok)`.
    let mut sum_exp = 0.0f64;
    for pair in logits_bf16.chunks_exact(2) {
        let bits = (u32::from(pair[0]) | (u32::from(pair[1]) << 8)) << 16;
        let raw = f32::from_bits(bits);
        if raw.is_finite() {
            sum_exp += f64::from(((raw - max_raw) / eff_temp).exp());
        }
    }

    order_candidates(&mut candidates, spec.top_k as usize, CandidateOrder::Total);
    let max_scaled = candidates[0].1;
    let mut kept: Vec<(u32, f32, f32)> = candidates
        .iter()
        .map(|&(id, scaled)| (id, scaled, (scaled - max_scaled).exp()))
        .collect();
    if spec.min_p > 0.0 {
        let thresh = spec.min_p * kept[0].2;
        kept.retain(|entry| entry.2 >= thresh);
    }
    if spec.top_p < 1.0 && !kept.is_empty() {
        let sum: f32 = kept.iter().map(|entry| entry.2).sum();
        if sum > 0.0 {
            let mut cum = 0.0f32;
            let mut cutoff = kept.len();
            for (i, entry) in kept.iter().enumerate() {
                cum += entry.2 / sum;
                if cum >= spec.top_p {
                    cutoff = i + 1;
                    break;
                }
            }
            kept.truncate(cutoff);
        }
    }
    let mut best_id = candidates[0].0;
    let mut best_weight = 1.0f32;
    let mut best_score = f64::NEG_INFINITY;
    for &(id, scaled, weight) in &kept {
        let score = f64::from(scaled) + f64::from(gumbel_f32(spec.seed, abs_pos, id));
        if score > best_score || (score == best_score && id < best_id) {
            best_score = score;
            best_id = id;
            best_weight = weight;
        }
    }
    let chosen_prob = if sum_exp > 0.0 && sum_exp.is_finite() && best_weight.is_finite() {
        (f64::from(best_weight) / sum_exp).clamp(0.0, 1.0) as f32
    } else {
        0.0
    };
    Some((best_id, chosen_prob))
}

#[cfg(test)]
#[path = "gumbel_tests.rs"]
mod tests;

#[cfg(test)]
mod draft_spec_ring_tests {
    use super::*;

    #[test]
    fn greedy_step_or_consumption_clears_prior_gumbel_spec_and_isolates_slots() {
        clear_gumbel_draft_specs();
        let spec_a = GumbelDraftSpec {
            seed: 101,
            temperature: 0.8,
            top_k: 20,
            top_p: 0.95,
            min_p: 0.0,
            step0_abs_pos: 43,
        };
        let spec_b = GumbelDraftSpec {
            seed: 202,
            temperature: 0.6,
            ..spec_a
        };
        // Slot 0 and slot 1 at identical propose_position=42 do not collide.
        register_gumbel_draft_spec_for_slot(0, 42, spec_a);
        register_gumbel_draft_spec_for_slot(1, 42, spec_b);
        assert_eq!(take_gumbel_draft_spec_for_slot(0, 42), Some(spec_a));
        assert_eq!(take_gumbel_draft_spec_for_slot(1, 42), Some(spec_b));
        // Consumed entries are gone on subsequent reads.
        assert_eq!(lookup_gumbel_draft_spec(42), None);

        // Greedy (T=0) invalidation after a Gumbel step at the same position leaves None.
        register_gumbel_draft_spec_for_slot(3, 42, spec_a);
        register_gumbel_draft_spec_for_slot(
            3,
            42,
            GumbelDraftSpec {
                temperature: 0.0,
                ..spec_a
            },
        );
        assert_eq!(take_gumbel_draft_spec_for_slot(3, 42), None);
        assert_eq!(lookup_gumbel_draft_spec(42), None);
    }

    #[test]
    fn shortlist_prob_reports_post_temperature_probability_of_chosen_token() {
        let logits = [4.0f32, 2.0, 0.0];
        let bf16: Vec<u8> = logits
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let temp = 2.0f32;
        let spec = GumbelDraftSpec {
            seed: 0x5150,
            temperature: temp,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            step0_abs_pos: 0,
        };
        let denom: f64 = logits
            .iter()
            .map(|&v| f64::from(((v - 4.0) / temp).exp()))
            .sum();
        let expected_probs: Vec<f32> = logits
            .iter()
            .map(|&v| (f64::from(((v - 4.0) / temp).exp()) / denom) as f32)
            .collect();
        let mut saw_argmax = false;
        let mut saw_non_argmax = false;
        for pos in 0..128u64 {
            let (tok, prob) =
                select_gumbel_shortlist_bf16(&bf16, None, spec, pos).expect("chosen token");
            let want = expected_probs[tok as usize];
            assert!(
                (prob - want).abs() < 1e-5,
                "pos={pos} tok={tok} got={prob} want={want}"
            );
            if tok == 0 {
                saw_argmax = true;
            } else {
                saw_non_argmax = true;
            }
        }
        assert!(
            saw_argmax && saw_non_argmax,
            "expected both argmax and non-argmax picks"
        );
    }
}
