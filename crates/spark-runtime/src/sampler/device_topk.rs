// SPDX-License-Identifier: AGPL-3.0-only
//
// Device top-k candidate route for stochastic sampling (SPEED_PLAN W1).
//
// Default: `DeviceTopkMode::Auto` on wide-SM GPUs (`sm_count > 64`) and `Off`
// on compact GPUs (`sm_count <= 64`), overridable via
// `ATLAS_SAMPLER_DEVICE_TOPK=0|1|auto`. When enabled, the scheduler replaces
// the full-vocabulary logits D2H with a device selection of the `m` best
// candidates. The host then writes those candidates into a sparse or
// vocab-length row whose other entries are -inf and runs the UNCHANGED host
// interventions and sampler on it.
//
// Why the result is identical by construction:
//
// * Selection key. The device ranks finite raw logits `v` by
//   `(v / temperature desc, id asc)`: the same f32 division the host sampler
//   does (`v / temperature`, IEEE round-to-nearest on both sides), the same
//   finiteness filter (on the raw value), and the canonical tie order below.
// * Canonical tie order. The host defaults to `CandidateOrder::Total`
//   (`(logit desc, id asc)`), which makes the sorted list unique; the device
//   route also implies [`canonical_ties_enabled`], forcing `Total` even when
//   `FLUX_SAMPLER_TIE_ORDER=legacy` is set. For A/B control arms that want
//   canonical host ties without the device route, set
//   `ATLAS_SAMPLER_CANONICAL_TIES=1` with `ATLAS_SAMPLER_DEVICE_TOPK=0`.
// * Edit budget. Every host intervention the route admits only LOWERS
//   logits (penalties >= neutral, masks, fixed negative offsets) or masks all
//   but one id. If at most `E` ids are lowered, the true top-`top_k` after
//   the edits lies inside the device top-`(top_k + E)` under the same key:
//   at least `top_k` unedited candidates still beat every outsider. Anything
//   that could raise a logit, or that reads the whole row (top-n-sigma,
//   F2's full softmax, logprobs, row hashes), is routed to the full host
//   path instead; see [`candidate_budget`] and the scheduler's route.

use std::cmp::Ordering;
use std::sync::OnceLock;

use super::SamplingParams;

/// Largest candidate count the device kernel returns per row
/// (`top_k + E`; E bounds the ids the host may lower).
pub const DEVICE_TOPK_M_MAX: usize = 512;

/// Largest request `top_k` the route admits (`admit`).
pub const DEVICE_TOPK_K_MAX: usize = 256;

/// Columns scanned per CTA in `topk_slice_stage1_bf16` (31 CTAs for 248,320 vocab).
pub const DEVICE_TOPK_SLICE_WIDTH: usize = 8192;

/// Maximum candidate budget `m` handled by the 2-stage multi-CTA sliced kernel.
pub const DEVICE_TOPK_SLICE_M_MAX: usize = 64;

/// Maximum slice CTAs provisioned for the 2-stage sliced kernel (`64 * 8192 = 524,288` vocab).
pub const DEVICE_TOPK_SLICE_MAX_CTAS: usize = 64;

/// Intermediate device workspace bytes required by `topk_slice_stage1_bf16` + `stage2_merge`:
/// `[MAX_CTAS * SLICE_M_MAX]` `u64` keys + `f32` raw values + `[MAX_CTAS]` `u32` counts.
pub const DEVICE_TOPK_SLICE_WORKSPACE_BYTES: usize =
    DEVICE_TOPK_SLICE_MAX_CTAS * DEVICE_TOPK_SLICE_M_MAX * (8 + 4) + DEVICE_TOPK_SLICE_MAX_CTAS * 4;

/// Maximum rows staged per batched GPU target Gumbel-Max verify launch
/// (`sampler_topk_candidates::target_gumbel_verify_rows_bf16`).
/// Sized to hold a full 256-row verify frame (`8 KiB` staging buffer total).
pub const GUMBEL_VERIFY_MAX_ROWS: usize = 256;

/// Per-row sampling request for batched GPU target Gumbel-Max verify
/// (`sampler_topk_candidates::target_gumbel_verify_rows_bf16`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GumbelVerifyRowSpec {
    /// Row index in the `[num_frame_rows, vocab_stride]` BF16 verify logits buffer.
    pub row_idx: usize,
    /// Sequence base Gumbel seed.
    pub seed: u64,
    /// Absolute token position (`seq_abs_pos` for this verify row).
    pub abs_pos: u64,
    /// Sampling temperature (`> 0.0`, finite).
    pub temperature: f32,
    /// Request `top_k` (`0` = full vocabulary, else `1..=DEVICE_TOPK_K_MAX`).
    pub top_k: u32,
    /// Request `top_p` (`(0.0, 1.0]` finite; `>= 1.0` disables nucleus truncation).
    pub top_p: f32,
}

/// The sampler's LZ penalty window (`apply_lz_penalty`'s `LZ_WINDOW`).
const LZ_WINDOW: usize = 256;

/// Operating mode configured via `ATLAS_SAMPLER_DEVICE_TOPK`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceTopkMode {
    /// Device top-k candidate selection is disabled; all rows sample on host
    /// (default on compact GPUs with `sm_count <= 64`, or when
    /// `ATLAS_SAMPLER_DEVICE_TOPK=0`).
    Off,
    /// `ATLAS_SAMPLER_DEVICE_TOPK=1`: device top-k is active and admission rejects
    /// stochastic requests whose `top_k` is `0` or above [`DEVICE_TOPK_K_MAX`].
    Strict,
    /// `ATLAS_SAMPLER_DEVICE_TOPK=auto` (default on wide-SM GPUs with
    /// `sm_count > 64`): device top-k is active for eligible requests while
    /// unsupported requests (`top_k == 0`, `top_k > DEVICE_TOPK_K_MAX`, grammar,
    /// logprobs, etc.) are admitted and transparently fall back to the host path.
    Auto,
}

impl DeviceTopkMode {
    /// Whether the device candidate route is enabled (`Strict` or `Auto`).
    #[must_use]
    pub const fn is_enabled(self) -> bool {
        matches!(self, Self::Strict | Self::Auto)
    }

    /// Whether admission strictly refuses unsupported stochastic `top_k` values (`Strict` only).
    #[must_use]
    pub const fn is_strict(self) -> bool {
        matches!(self, Self::Strict)
    }
}

/// Parse an `ATLAS_SAMPLER_DEVICE_TOPK` environment value into a [`DeviceTopkMode`].
#[must_use]
pub fn parse_device_topk_mode(raw: Option<&str>) -> DeviceTopkMode {
    match raw.map(str::trim) {
        Some("1") => DeviceTopkMode::Strict,
        Some(v) if v.eq_ignore_ascii_case("auto") => DeviceTopkMode::Auto,
        _ => DeviceTopkMode::Off,
    }
}

/// Resolve `ATLAS_SAMPLER_DEVICE_TOPK` for a GPU with `sm_count` SMs.
///
/// When the environment variable is unset:
/// - Wide-SM GPUs (`sm_count > 64`, e.g., SM120 / SM103) default to
///   [`DeviceTopkMode::Auto`], avoiding full-vocabulary D2H on eligible batches
///   while transparently falling back to host for unsupported requests.
/// - Compact GPUs (`sm_count <= 64`, e.g., GB10) preserve [`DeviceTopkMode::Off`].
///
/// Any explicit value (`"0"`, `"off"`, `"1"`, `"auto"`) always overrides the
/// hardware default.
#[must_use]
pub fn resolve_device_topk_mode(raw: Option<&str>, sm_count: u32) -> DeviceTopkMode {
    match raw {
        None if sm_count > 64 => DeviceTopkMode::Auto,
        other => parse_device_topk_mode(other),
    }
}

/// Process-wide `ATLAS_SAMPLER_DEVICE_TOPK` mode (`Off`, `Strict` for `=1`, `Auto` for `=auto`).
#[must_use]
pub fn device_topk_mode() -> DeviceTopkMode {
    static RAW: OnceLock<Option<String>> = OnceLock::new();
    let raw = RAW.get_or_init(|| std::env::var("ATLAS_SAMPLER_DEVICE_TOPK").ok());
    resolve_device_topk_mode(raw.as_deref(), atlas_core::device::active_sm_count())
}

/// `ATLAS_SAMPLER_DEVICE_TOPK=1` or `=auto`: the device candidate route is enabled.
pub fn device_topk_enabled() -> bool {
    device_topk_mode().is_enabled()
}

/// `ATLAS_SAMPLER_DEVICE_TOPK=1`: strict admission refuses unsupported `top_k`.
#[must_use]
pub fn device_topk_strict() -> bool {
    device_topk_mode().is_strict()
}

/// Whether the host sampler breaks equal-value ties by lower token id.
/// `ATLAS_SAMPLER_CANONICAL_TIES=1`, implied by the device route (`=1` or `=auto`).
pub fn canonical_ties_enabled() -> bool {
    static EXPLICIT: OnceLock<bool> = OnceLock::new();
    *EXPLICIT.get_or_init(|| env_is_one("ATLAS_SAMPLER_CANONICAL_TIES")) || device_topk_enabled()
}

fn env_is_one(name: &str) -> bool {
    std::env::var(name).ok().as_deref() == Some("1")
}

/// The host sampler's descending comparator. `canonical` adds the id key.
#[cfg_attr(not(test), allow(dead_code))]
#[inline]
pub(super) fn cmp_desc(a: &(u32, f32), b: &(u32, f32), canonical: bool) -> Ordering {
    let by_value = b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal);
    if canonical {
        by_value.then(a.0.cmp(&b.0))
    } else {
        by_value
    }
}

/// The device kernel's 64-bit selection key for one raw logit, or `None` when
/// the raw value is not finite (the host sampler drops it). Larger is better.
/// High half: an order-preserving map of `v / temperature` with -0 folded
/// onto +0 (the host compares them equal). Low half: `!id`, so a lower id
/// wins an equal value.
pub fn selection_key(id: u32, raw: f32, temperature: f32) -> Option<u64> {
    if !raw.is_finite() {
        return None;
    }
    let key = raw / temperature + 0.0;
    let bits = key.to_bits();
    let ord = if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000
    };
    Some((u64::from(ord) << 32) | u64::from(!id))
}

/// CPU specification of the device kernel: the `m` best finite raw logits of a
/// little-endian BF16 row by [`selection_key`], as `(id, raw value)` in key
/// order. Returns every finite entry when there are at most `m`.
pub fn topk_candidates_reference(row_bf16: &[u8], temperature: f32, m: usize) -> Vec<(u32, f32)> {
    let mut keyed: Vec<(u64, u32, f32)> = row_bf16
        .chunks_exact(2)
        .enumerate()
        .filter_map(|(id, b)| {
            let raw = f32::from_bits(((b[0] as u32) | ((b[1] as u32) << 8)) << 16);
            selection_key(id as u32, raw, temperature).map(|k| (k, id as u32, raw))
        })
        .collect();
    keyed.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    keyed.truncate(m);
    keyed.into_iter().map(|(_, id, raw)| (id, raw)).collect()
}

/// Whether `(vocab, m)` should dispatch to the 2-stage multi-CTA sliced kernel
/// (`topk_slice_stage1_bf16` + `topk_slice_stage2_merge`) instead of single-CTA
/// 10-pass `topk_candidates_bf16`.
#[must_use]
pub const fn use_sliced_topk(vocab: usize, m: usize) -> bool {
    m >= 1
        && m <= DEVICE_TOPK_SLICE_M_MAX
        && vocab > DEVICE_TOPK_SLICE_WIDTH
        && vocab.div_ceil(DEVICE_TOPK_SLICE_WIDTH) <= DEVICE_TOPK_SLICE_MAX_CTAS
}

/// Grid X dimension (`num_slices`) for `topk_slice_stage1_bf16` over `vocab` columns.
#[must_use]
pub const fn topk_slice_grid(vocab: usize) -> u32 {
    vocab.div_ceil(DEVICE_TOPK_SLICE_WIDTH) as u32
}

/// CPU specification of the 2-stage multi-CTA sliced device kernel
/// (`topk_slice_stage1_bf16` across `8192`-column slices + `topk_slice_stage2_merge`).
///
/// Each `8192`-element slice independently selects at most `m` (`<= 64`) top
/// candidates by [`selection_key`], then Stage 2 merges all slice winners and
/// keeps the global top `m` in descending key order.
#[must_use]
pub fn topk_sliced_candidates_reference(
    row_bf16: &[u8],
    temperature: f32,
    m: usize,
) -> Vec<(u32, f32)> {
    let m = m.min(DEVICE_TOPK_SLICE_M_MAX);
    if m == 0 {
        return Vec::new();
    }
    let slice_bytes = DEVICE_TOPK_SLICE_WIDTH * 2;
    let mut merged: Vec<(u64, u32, f32)> = Vec::new();
    for (cta, slice) in row_bf16.chunks(slice_bytes).enumerate() {
        let base_id = (cta * DEVICE_TOPK_SLICE_WIDTH) as u32;
        let mut local: Vec<(u64, u32, f32)> = slice
            .chunks_exact(2)
            .enumerate()
            .filter_map(|(j, b)| {
                let id = base_id + j as u32;
                let raw = f32::from_bits(((u32::from(b[0])) | (u32::from(b[1]) << 8)) << 16);
                selection_key(id, raw, temperature).map(|k| (k, id, raw))
            })
            .collect();
        if local.len() > m {
            local.select_nth_unstable_by(m - 1, |a, b| b.0.cmp(&a.0));
            local.truncate(m);
        }
        merged.extend(local);
    }
    if merged.len() > m {
        merged.select_nth_unstable_by(m - 1, |a, b| b.0.cmp(&a.0));
        merged.truncate(m);
    }
    merged.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    merged.into_iter().map(|(_, id, raw)| (id, raw)).collect()
}

/// A vocab-length f32 row holding `candidates` and -inf everywhere else.
/// Ids at or past `vocab` are an error: the device returned garbage.
pub fn sparse_row_from_candidates(vocab: usize, candidates: &[(u32, f32)]) -> Option<Vec<f32>> {
    let mut row = vec![f32::NEG_INFINITY; vocab];
    for &(id, raw) in candidates {
        *row.get_mut(id as usize)? = raw;
    }
    Some(row)
}

/// Compact O(m) sparse representation of a vocab-length logit row whose absent
/// entries are implicitly `f32::NEG_INFINITY`.
///
/// Entries are kept in strictly ascending token-ID order so penalties,
/// interventions, top-n-sigma, sorting, softmax, Gumbel-Max, and categorical
/// sampling execute in O(m) without materializing or scanning a full-vocab
/// `Vec<f32>` (`~1 MB` for `vocab = 248,320`), while remaining bit-identical to
/// expanding via [`sparse_row_from_candidates`] and running the dense sampler.
#[derive(Clone, Debug, PartialEq)]
pub struct SparseLogitRow {
    vocab_size: usize,
    entries: Vec<(u32, f32)>,
}

impl SparseLogitRow {
    /// Build a compact sparse row from device top-`m` `(token_id, raw_logit)`
    /// candidates. Returns `None` if any `id >= vocab_size`. Duplicate IDs keep
    /// the last entry, matching [`sparse_row_from_candidates`] overwrite order.
    #[must_use]
    pub fn from_candidates(vocab_size: usize, candidates: &[(u32, f32)]) -> Option<Self> {
        let mut entries: Vec<(u32, f32)> = Vec::with_capacity(candidates.len());
        for &(id, raw) in candidates {
            if (id as usize) >= vocab_size {
                return None;
            }
            entries.push((id, raw));
        }
        entries.sort_by_key(|&(id, _)| id);
        if entries.windows(2).any(|w| w[0].0 == w[1].0) {
            let mut deduped: Vec<(u32, f32)> = Vec::with_capacity(entries.len());
            for (id, raw) in entries {
                if let Some(last) = deduped.last_mut()
                    && last.0 == id
                {
                    last.1 = raw;
                } else {
                    deduped.push((id, raw));
                }
            }
            entries = deduped;
        }
        Some(Self {
            vocab_size,
            entries,
        })
    }

    /// Build a sparse row with a single populated `(id, value)` entry and `-inf`
    /// everywhere else. Returns `None` if `id >= vocab_size`.
    #[must_use]
    pub fn from_single(vocab_size: usize, id: u32, value: f32) -> Option<Self> {
        if (id as usize) >= vocab_size {
            return None;
        }
        Some(Self {
            vocab_size,
            entries: vec![(id, value)],
        })
    }

    /// Logical vocabulary length represented by this row.
    #[must_use]
    pub const fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Populated `(token_id, logit)` pairs in strictly ascending token-ID order.
    #[must_use]
    pub fn entries(&self) -> &[(u32, f32)] {
        &self.entries
    }

    /// Number of explicitly stored candidate entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no candidate entries are stored (all vocabulary logits are `-inf`).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Read the logit at `id`, returning `f32::NEG_INFINITY` when absent.
    #[must_use]
    pub fn get(&self, id: u32) -> f32 {
        match self.entries.binary_search_by_key(&id, |&(tid, _)| tid) {
            Ok(idx) => self.entries[idx].1,
            Err(_) => f32::NEG_INFINITY,
        }
    }

    /// Mask token `id` to `f32::NEG_INFINITY` (no-op if already absent).
    pub fn set_neg_inf(&mut self, id: u32) {
        if let Ok(idx) = self.entries.binary_search_by_key(&id, |&(tid, _)| tid) {
            self.entries[idx].1 = f32::NEG_INFINITY;
        }
    }

    /// Subtract a finite `delta` from `id`'s logit (no-op if absent, since
    /// `-inf - delta == -inf` for finite `delta`).
    pub fn sub_logit(&mut self, id: u32, delta: f32) {
        if let Ok(idx) = self.entries.binary_search_by_key(&id, |&(tid, _)| tid) {
            self.entries[idx].1 -= delta;
        }
    }

    /// Add a finite `delta` to `id`'s logit (no-op if absent, since
    /// `-inf + delta == -inf` for finite `delta`).
    pub fn add_logit(&mut self, id: u32, delta: f32) {
        if let Ok(idx) = self.entries.binary_search_by_key(&id, |&(tid, _)| tid) {
            self.entries[idx].1 += delta;
        }
    }

    /// Apply one [`super::f2_certify::PreF2Edit`] directly to `id` in place.
    pub fn apply_pre_f2_edit(&mut self, id: u32, edit: super::f2_certify::PreF2Edit) {
        if let Ok(idx) = self.entries.binary_search_by_key(&id, |&(tid, _)| tid) {
            edit.apply(&mut self.entries[idx].1);
        }
    }

    /// Mask every vocabulary entry to `-inf` and set `id` to `value` (used by
    /// forced `</think>` and forced `<tool_call>` pinning). Returns `false` if
    /// `id >= vocab_size`, leaving the row unchanged.
    pub fn pin_token(&mut self, id: u32, value: f32) -> bool {
        if (id as usize) >= self.vocab_size {
            return false;
        }
        self.entries.clear();
        self.entries.push((id, value));
        true
    }

    /// Expand this sparse row into a full `vocab_size`-element dense `Vec<f32>`
    /// with `-inf` at every absent position (slow fallback / test helper).
    #[must_use]
    pub fn to_dense(&self) -> Vec<f32> {
        let mut row = vec![f32::NEG_INFINITY; self.vocab_size];
        for &(id, val) in &self.entries {
            if let Some(slot) = row.get_mut(id as usize) {
                *slot = val;
            }
        }
        row
    }

    /// Direct O(m) Gumbel-Max sampling over this sparse row (bit-identical to
    /// expanding via [`Self::to_dense`] and calling [`super::choose_gumbel`]).
    #[must_use]
    pub fn choose_gumbel(
        &self,
        params: &SamplingParams,
        token_history: &[u32],
        seed: u64,
        abs_pos: u64,
    ) -> u32 {
        let mut entries = self.entries.clone();
        super::sample_impl::choose_gumbel_sparse_impl(
            self.vocab_size,
            &mut entries,
            params,
            token_history,
            seed,
            abs_pos,
            false,
        )
    }

    /// Zero-clone in-place variant of [`Self::choose_gumbel`] (mutates entries
    /// with penalties / top-n-sigma in place).
    pub fn choose_gumbel_in_place(
        &mut self,
        params: &SamplingParams,
        token_history: &[u32],
        seed: u64,
        abs_pos: u64,
    ) -> u32 {
        super::sample_impl::choose_gumbel_sparse_impl(
            self.vocab_size,
            &mut self.entries,
            params,
            token_history,
            seed,
            abs_pos,
            false,
        )
    }

    /// Direct O(m) counterpart to [`super::sample_with_params_history`].
    #[must_use]
    pub fn sample_with_params_history(
        &self,
        params: &SamplingParams,
        token_history: &[u32],
    ) -> u32 {
        self.sample_with_params_seeded(params, token_history, params.seed)
    }

    /// Zero-clone in-place counterpart to [`super::sample_with_params_history`].
    pub fn sample_with_params_history_in_place(
        &mut self,
        params: &SamplingParams,
        token_history: &[u32],
    ) -> u32 {
        self.sample_with_params_seeded_in_place(params, token_history, params.seed)
    }

    /// Direct O(m) counterpart to [`super::sample_with_params_seeded`].
    #[must_use]
    pub fn sample_with_params_seeded(
        &self,
        params: &SamplingParams,
        token_history: &[u32],
        seed: Option<u64>,
    ) -> u32 {
        let mut entries = self.entries.clone();
        super::sample_impl::sample_sparse_seeded_impl(
            self.vocab_size,
            &mut entries,
            params,
            token_history,
            seed,
            canonical_ties_enabled(),
        )
    }

    /// Zero-clone in-place counterpart to [`super::sample_with_params_seeded`].
    pub fn sample_with_params_seeded_in_place(
        &mut self,
        params: &SamplingParams,
        token_history: &[u32],
        seed: Option<u64>,
    ) -> u32 {
        super::sample_impl::sample_sparse_seeded_impl(
            self.vocab_size,
            &mut self.entries,
            params,
            token_history,
            seed,
            canonical_ties_enabled(),
        )
    }

    /// Direct O(m) counterpart to [`super::prepare_sampling_distribution_with_history`].
    #[must_use]
    pub fn prepare_sampling_distribution_with_history(
        &self,
        params: &SamplingParams,
        token_history: &[u32],
    ) -> super::PreparedSamplingDistribution {
        let mut entries = self.entries.clone();
        let order = if canonical_ties_enabled() {
            super::candidate_order::CandidateOrder::Total
        } else {
            super::candidate_order::candidate_order()
        };
        super::sample_impl::prepare_sparse_sampling_distribution_ordered(
            self.vocab_size,
            &mut entries,
            params,
            token_history,
            false,
            order,
        )
    }
}

/// Whether `params` satisfies the static request-level preconditions for the
/// device top-k route (`temperature > 0.0`, `1 <= top_k <= DEVICE_TOPK_K_MAX`,
/// finite parameters, no `top_n_sigma`, `logit_bias`, or logit-raising penalties).
///
/// Full per-row eligibility (including token-history edit budget and host-only
/// features) is checked by [`candidate_budget`] at decode/verify time.
#[must_use]
pub fn can_use_device_topk(params: &SamplingParams) -> bool {
    candidate_budget(params, &[], 0).is_ok()
}

/// Request-level admission check parameterized by `mode`.
///
/// Under [`DeviceTopkMode::Strict`] (`ATLAS_SAMPLER_DEVICE_TOPK=1`), a stochastic
/// request whose `top_k` is disabled (0) or above [`DEVICE_TOPK_K_MAX`] is
/// rejected rather than silently served by a different path. Under
/// [`DeviceTopkMode::Auto`] (`ATLAS_SAMPLER_DEVICE_TOPK=auto`) or
/// [`DeviceTopkMode::Off`], all requests are admitted and ineligible rows fall
/// back to the host sampler path at runtime.
pub fn admit_with_mode(params: &SamplingParams, mode: DeviceTopkMode) -> Result<(), String> {
    if !mode.is_strict() || params.temperature <= 0.0 {
        return Ok(());
    }
    admit_top_k(params.top_k)
}

/// Request-level refusal: under `ATLAS_SAMPLER_DEVICE_TOPK=1` (`Strict`), a
/// stochastic request whose `top_k` is disabled (0) or above
/// [`DEVICE_TOPK_K_MAX`] is rejected rather than silently served by a different
/// path. Under `Off` or `ATLAS_SAMPLER_DEVICE_TOPK=auto`, every request is
/// admitted.
pub fn admit(params: &SamplingParams) -> Result<(), String> {
    admit_with_mode(params, device_topk_mode())
}

pub(super) fn admit_top_k(top_k: u32) -> Result<(), String> {
    if top_k == 0 || top_k as usize > DEVICE_TOPK_K_MAX {
        return Err(format!(
            "ATLAS_SAMPLER_DEVICE_TOPK=1 serves stochastic requests with 1 <= top_k <= {DEVICE_TOPK_K_MAX}; got top_k={top_k}"
        ));
    }
    Ok(())
}

/// The candidate count `m = top_k + E` for one row, where `E` bounds the ids
/// the sampler's own penalty stage may lower, plus `extra_lowered` ids the
/// caller's interventions may lower. `Err(reason)` routes the row to the full
/// host path: something in `params` can raise a logit or read the whole row.
///
/// E is the exact set of ids the penalty stage can touch, with each penalty's
/// own window (`apply_penalties_and_bias` / `prepare_sampling_distribution`):
/// repetition / presence / frequency act on the last
/// `repetition_penalty_window` tokens (all when 0 or longer than the history),
/// LZ on the last 256, DRY on the whole history. Tokens outside those windows
/// are not penalised by the host either, so the decrease-only argument holds
/// over exactly this set.
pub fn candidate_budget(
    params: &SamplingParams,
    token_history: &[u32],
    extra_lowered: usize,
) -> Result<usize, &'static str> {
    let floats = [
        params.temperature,
        params.top_n_sigma,
        params.repetition_penalty,
        params.presence_penalty,
        params.frequency_penalty,
        params.lz_penalty,
        params.dry_multiplier,
        params.dry_base,
    ];
    if floats.iter().any(|v| v.is_nan()) {
        return Err("nan_param");
    }
    if params.temperature <= 0.0 {
        return Err("greedy");
    }
    if params.top_k == 0 || params.top_k as usize > DEVICE_TOPK_K_MAX {
        return Err("top_k");
    }
    if params.top_n_sigma > 0.0 {
        return Err("top_n_sigma");
    }
    if !params.logit_bias.is_empty() {
        return Err("logit_bias");
    }
    if floats.iter().any(|v| v.is_infinite()) {
        return Err("infinite_param");
    }
    let rep = params.repetition_penalty;
    let rep_active = rep != 1.0 && rep > 0.0;
    if rep_active && rep < 1.0 {
        return Err("repetition_penalty_below_one");
    }
    let additive_active = params.frequency_penalty != 0.0 || params.presence_penalty != 0.0;
    if additive_active && (params.frequency_penalty < 0.0 || params.presence_penalty < 0.0) {
        return Err("negative_additive_penalty");
    }
    // LZ / DRY only subtract when their multiplier is positive; a negative
    // multiplier is inert in the sampler (gated on > 0), so it edits nothing.
    // DRY subtracts multiplier * base^k: a negative base can make that
    // negative for odd k, i.e. RAISE a logit.
    let lz_active = params.lz_penalty > 0.0 && token_history.len() >= 4;
    let dry_active = params.dry_multiplier > 0.0 && token_history.len() >= 3;
    if dry_active && params.dry_base < 0.0 {
        return Err("negative_dry_base");
    }
    let tail = |window: usize| {
        if window > 0 && window < token_history.len() {
            &token_history[token_history.len() - window..]
        } else {
            token_history
        }
    };
    let mut lowered: Vec<u32> = Vec::new();
    if dry_active {
        lowered.extend_from_slice(token_history);
    } else {
        if rep_active || additive_active {
            lowered.extend_from_slice(tail(params.repetition_penalty_window as usize));
        }
        if lz_active {
            lowered.extend_from_slice(tail(LZ_WINDOW));
        }
    }
    lowered.sort_unstable();
    lowered.dedup();
    let m = params.top_k as usize + lowered.len() + extra_lowered;
    if m > DEVICE_TOPK_M_MAX {
        return Err("edit_budget");
    }
    Ok(m)
}

#[cfg(test)]
#[path = "device_topk_tests.rs"]
mod tests;

#[cfg(test)]
mod sparse_fast_path_tests {
    use super::super::candidate_order::CandidateOrder;
    use super::super::f2_certify::PreF2Edit;
    use super::super::sample_impl::{
        prepare_sampling_distribution_ordered, prepare_sparse_sampling_distribution_ordered,
        sample_seeded_impl, sample_sparse_seeded_impl,
    };
    use super::super::{SamplingParams, choose_gumbel};
    use super::{SparseLogitRow, sparse_row_from_candidates};

    fn dense_bytes(row: &[f32]) -> Vec<u8> {
        row.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// Deterministic PRNG (SplitMix64) so tests have zero external randomness.
    struct Rng(u64);
    impl Rng {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        fn f32_range(&mut self, lo: f32, hi: f32) -> f32 {
            let u = ((self.next_u64() >> 40) as f32) / ((1u32 << 24) as f32);
            lo + (hi - lo) * u
        }
    }

    fn base_params(temp: f32, top_k: u32, top_p: f32, min_p: f32) -> SamplingParams {
        let mut p = SamplingParams::greedy(32);
        p.temperature = temp;
        p.top_k = top_k;
        p.top_p = top_p;
        p.min_p = min_p;
        p
    }

    fn assert_sparse_matches_dense(
        vocab: usize,
        candidates: &[(u32, f32)],
        params: &SamplingParams,
        history: &[u32],
        seed: u64,
        abs_pos: u64,
    ) {
        let dense = sparse_row_from_candidates(vocab, candidates).unwrap();
        let bytes = dense_bytes(&dense);
        let sparse = SparseLogitRow::from_candidates(vocab, candidates).unwrap();
        let sparse_dense = sparse.to_dense();
        assert_eq!(sparse_dense.len(), dense.len());
        assert!(
            sparse_dense
                .iter()
                .zip(&dense)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "dense bit expansion mismatch"
        );

        // 1. Gumbel-Max exact parity (non-mutating and in-place).
        let dense_gumbel = choose_gumbel(&bytes, params, history, seed, abs_pos);
        let sparse_gumbel = sparse.choose_gumbel(params, history, seed, abs_pos);
        let mut sparse_mut = sparse.clone();
        let sparse_gumbel_in_place =
            sparse_mut.choose_gumbel_in_place(params, history, seed, abs_pos);
        assert_eq!(sparse_gumbel, dense_gumbel, "gumbel mismatch");
        assert_eq!(
            sparse_gumbel_in_place, dense_gumbel,
            "gumbel in-place mismatch"
        );

        // 2. Prepared distribution exact parity under BOTH Total and Legacy tie orders.
        for order in [CandidateOrder::Total, CandidateOrder::Legacy] {
            let dense_dist =
                prepare_sampling_distribution_ordered(&bytes, params, history, false, order);
            let mut entries = sparse.entries().to_vec();
            let sparse_dist = prepare_sparse_sampling_distribution_ordered(
                vocab,
                &mut entries,
                params,
                history,
                false,
                order,
            );
            assert_eq!(sparse_dist, dense_dist, "distribution mismatch ({order:?})");
        }

        // 3. Seeded categorical sample exact parity under BOTH canonical_ties settings.
        for canonical in [true, false] {
            let dense_tok = sample_seeded_impl(&bytes, params, history, Some(seed), canonical);
            let mut entries = sparse.entries().to_vec();
            let sparse_tok = sample_sparse_seeded_impl(
                vocab,
                &mut entries,
                params,
                history,
                Some(seed),
                canonical,
            );
            assert_eq!(
                sparse_tok, dense_tok,
                "seeded sample mismatch (canonical={canonical})"
            );
        }
    }

    #[test]
    fn sparse_matches_dense_across_topk_topp_minp_temperature_and_gumbel() {
        let vocab = 1024usize;
        let mut rng = Rng(0xA17A_5EED_0001);
        let temps = [0.0f32, 1e-7, 0.25, 0.7, 1.0, 2.2];
        let top_ks = [0u32, 1, 4, 16, 24, 64];
        let top_ps = [0.0f32, 0.2, 0.85, 0.95, 1.0];
        let min_ps = [0.0f32, 0.05, 0.4];

        for case in 0..48u64 {
            let m = 24usize;
            let mut cands = Vec::with_capacity(m);
            for _ in 0..m {
                let id = (rng.next_u64() as usize % vocab) as u32;
                let val = rng.f32_range(-8.0, 14.0);
                cands.push((id, val));
            }
            let temp = temps[(case as usize) % temps.len()];
            let top_k = top_ks[((case / 2) as usize) % top_ks.len()];
            let top_p = top_ps[((case / 3) as usize) % top_ps.len()];
            let min_p = min_ps[((case / 5) as usize) % min_ps.len()];
            let params = base_params(temp, top_k, top_p, min_p);
            assert_sparse_matches_dense(
                vocab,
                &cands,
                &params,
                &[],
                0xC0FF_EE00 + case,
                100 + case,
            );
        }
    }

    #[test]
    fn sparse_matches_dense_with_exact_ties_and_signed_zeros() {
        let vocab = 256usize;
        let candidates = vec![
            (50, 2.0f32),
            (12, 2.0f32),
            (99, 2.0f32),
            (3, -0.0f32),
            (7, 0.0f32),
            (200, -0.0f32),
            (1, 2.0f32),
            (40, 1.5f32),
        ];
        for (temp, top_k, top_p) in [
            (0.0, 0, 1.0),
            (0.8, 3, 1.0),
            (0.8, 5, 0.7),
            (1.0, 0, 0.9),
            (1.5, 4, 0.5),
        ] {
            let params = base_params(temp, top_k, top_p, 0.0);
            for pos in 0..8u64 {
                assert_sparse_matches_dense(vocab, &candidates, &params, &[], 0x71E5, pos);
            }
        }
    }

    #[test]
    fn sparse_matches_dense_across_all_penalties_and_top_n_sigma() {
        let vocab = 512usize;
        let candidates = vec![
            (10, 4.5f32),
            (20, -1.25f32),
            (30, 4.2f32),
            (40, 3.8f32),
            (50, 0.0f32),
            (60, 4.1f32),
            (70, -3.0f32),
            (80, 2.9f32),
            (90, 4.4f32),
        ];
        // History with repeated n-grams (triggers repetition, presence, frequency, LZ, DRY).
        let history = vec![
            10, 20, 30, 40, 10, 20, 30, 50, 99, 10, 20, 30, 60, 10, 20, 30,
        ];

        let mut params = base_params(0.75, 6, 0.92, 0.02);
        params.repetition_penalty = 1.25;
        params.repetition_penalty_window = 10;
        params.presence_penalty = 0.4;
        params.frequency_penalty = 0.35;
        params.lz_penalty = 0.6;
        params.dry_multiplier = 0.8;
        params.dry_base = 1.75;
        params.dry_allowed_length = 2;
        params.dry_sequence_breakers = vec![99];
        params.logit_bias = vec![(10, -0.7), (20, 1.5), (10, -0.3), (333, 4.0)];
        params.top_n_sigma = 1.2;

        for temp in [0.0f32, 0.65, 1.1] {
            params.temperature = temp;
            assert_sparse_matches_dense(vocab, &candidates, &params, &history, 0xDEAD_BEEF, 42);
        }

        // Also test unwindowed repetition/additive penalties (`window == 0`).
        params.repetition_penalty_window = 0;
        params.top_n_sigma = 0.0;
        assert_sparse_matches_dense(vocab, &candidates, &params, &history, 0xBEEF_CAFE, 7);
    }

    #[test]
    fn sparse_edits_pin_duplicates_and_nonfinite_fallbacks_match_dense() {
        let vocab = 128usize;
        // Out-of-range id is rejected just like `sparse_row_from_candidates`.
        assert!(SparseLogitRow::from_candidates(vocab, &[(128, 1.0)]).is_none());
        assert!(SparseLogitRow::from_single(vocab, 128, 0.0).is_none());

        // Duplicate IDs keep last write.
        let dup_cands = [(15, 1.0f32), (4, 3.0), (15, 5.0), (4, -2.0)];
        let mut sparse = SparseLogitRow::from_candidates(vocab, &dup_cands).unwrap();
        let mut dense = sparse_row_from_candidates(vocab, &dup_cands).unwrap();
        assert_eq!(sparse.to_dense(), dense);
        assert_eq!(sparse.get(15), 5.0);
        assert_eq!(sparse.get(4), -2.0);
        assert_eq!(sparse.get(77), f32::NEG_INFINITY);

        // Apply PreF2Edits + sub/add/set_neg_inf on both present and absent IDs.
        for (id, edit) in [
            (15, PreF2Edit::Sub(10.0)),
            (4, PreF2Edit::NegInf),
            (77, PreF2Edit::Sub(10.0)),
            (88, PreF2Edit::NegInf),
        ] {
            sparse.apply_pre_f2_edit(id, edit);
            edit.apply(&mut dense[id as usize]);
        }
        sparse.sub_logit(15, 12.0);
        dense[15] -= 12.0;
        sparse.add_logit(15, 0.5);
        dense[15] += 0.5;
        assert_eq!(sparse.to_dense(), dense);

        let params = base_params(0.8, 8, 0.9, 0.0);
        let bytes = dense_bytes(&dense);
        assert_eq!(
            sparse.choose_gumbel(&params, &[], 11, 22),
            choose_gumbel(&bytes, &params, &[], 11, 22)
        );

        // Pin token matches filling whole dense row with -inf except pinned ID.
        assert!(sparse.pin_token(63, 0.0));
        dense.fill(f32::NEG_INFINITY);
        dense[63] = 0.0;
        assert_eq!(sparse.to_dense(), dense);
        assert_eq!(sparse.choose_gumbel(&params, &[], 11, 22), 63);
        assert!(!sparse.pin_token(128, 0.0));

        // All -inf and leading-NaN fallback cases match dense `first_max_index`.
        assert_sparse_matches_dense(vocab, &[], &params, &[], 1, 2);
        assert_sparse_matches_dense(
            vocab,
            &[(10, f32::NEG_INFINITY), (20, f32::NEG_INFINITY)],
            &params,
            &[],
            1,
            2,
        );
        assert_sparse_matches_dense(
            vocab,
            &[(0, f32::NAN), (1, f32::NAN), (5, f32::NEG_INFINITY)],
            &params,
            &[],
            1,
            2,
        );
        assert_sparse_matches_dense(2, &[(0, f32::NAN), (1, f32::NAN)], &params, &[], 1, 2);
        assert_sparse_matches_dense(0, &[], &params, &[], 1, 2);
    }

    #[test]
    fn resolve_device_topk_mode_auto_enables_only_above_64_sms() {
        use super::{DeviceTopkMode, resolve_device_topk_mode};
        assert_eq!(resolve_device_topk_mode(None, 48), DeviceTopkMode::Off);
        assert_eq!(resolve_device_topk_mode(None, 64), DeviceTopkMode::Off);
        assert_eq!(resolve_device_topk_mode(None, 148), DeviceTopkMode::Auto);
        assert_eq!(resolve_device_topk_mode(None, 188), DeviceTopkMode::Auto);
        assert_eq!(
            resolve_device_topk_mode(Some("0"), 188),
            DeviceTopkMode::Off
        );
        assert_eq!(
            resolve_device_topk_mode(Some("off"), 188),
            DeviceTopkMode::Off
        );
        assert_eq!(
            resolve_device_topk_mode(Some("1"), 48),
            DeviceTopkMode::Strict
        );
        assert_eq!(
            resolve_device_topk_mode(Some("auto"), 48),
            DeviceTopkMode::Auto
        );
    }
}
