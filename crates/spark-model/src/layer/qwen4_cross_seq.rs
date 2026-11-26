// SPDX-License-Identifier: AGPL-3.0-only

//! Frame types for the cross-sequence Qwen4 K-row verify
//! (`ATLAS_QWEN4_BATCHED_K5`).
//!
//! One frame holds `seqs` sequences of `k` verify rows each, packed
//! sequence-major: sequence `s` owns rows `[s*k, (s+1)*k)` of the shared
//! hidden/residual arenas. Row-parallel work (hyperconnections, projections,
//! MoE, final mixer, LM head) runs once over all rows on the exact multi-row
//! tiers; recurrent and attention cores run per sequence against that
//! sequence's own state, KV block table and transaction lane.

use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::{AttnMetadataDev, LayerState};

/// Widest frame: the exact projection, hyperconnection, logits and metadata
/// tiers the frame reuses are provisioned for 256 rows
/// (`layers::QWEN4_NATIVE_MAX_ROWS`).
pub const QWEN4_CROSS_SEQ_MAX_ROWS: usize = 256;

/// Maximum number of concurrent sequences packed into one cross-sequence frame.
pub const QWEN4_CROSS_SEQ_MAX_SEQS: usize = 64;

/// Narrowest and widest per-sequence verify width for uniform frames (the exact
/// hybrid K-row family, `layers::QWEN4_HYBRID_MAX_ROWS`).
pub const QWEN4_CROSS_SEQ_MIN_K: usize = 2;
pub const QWEN4_CROSS_SEQ_MAX_K: usize = 9;

/// Narrowest per-sequence verify width admitted in a ragged mixed-`K` frame
/// (`K_s == 1` is anchor-only decode with zero drafts).
pub const QWEN4_CROSS_SEQ_RAGGED_MIN_K: usize = 1;

/// Widest per-sequence verify width packed by [`plan_ragged_frames`].
pub const QWEN4_CROSS_SEQ_RAGGED_PLAN_MAX_K: usize = 8;

/// Shape of one cross-sequence verify frame (uniform or ragged mixed-`K`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen4CrossSeqFrame {
    seqs: usize,
    seq_ks: [u8; QWEN4_CROSS_SEQ_MAX_SEQS],
    row_bases: [u16; QWEN4_CROSS_SEQ_MAX_SEQS],
    total_rows: u16,
    min_k: u8,
    max_k: u8,
}

impl Qwen4CrossSeqFrame {
    /// Narrowest per-sequence verify width admitted in a ragged frame (`K=1` anchor-only).
    pub const RAGGED_MIN_K: usize = QWEN4_CROSS_SEQ_RAGGED_MIN_K;

    /// Widest per-sequence verify width packed by [`Self::plan_ragged_frames`] (`K=8`).
    pub const RAGGED_PLAN_MAX_K: usize = QWEN4_CROSS_SEQ_RAGGED_PLAN_MAX_K;

    /// Admit `seqs` uniform sequences of `k` rows. At least two sequences (C1 keeps
    /// the proven single-sequence verifier), `k` in the exact hybrid family
    /// (`2..=9`), and at most [`QWEN4_CROSS_SEQ_MAX_ROWS`] rows.
    pub fn new(seqs: usize, k: usize) -> std::result::Result<Self, &'static str> {
        if seqs < 2 {
            return Err("cross-sequence verify needs at least two sequences");
        }
        Self::new_active(seqs, k)
    }

    /// Construct an active uniform execution sub-frame of `1..=64` unique leader
    /// sequences of width `k in 2..=9`.
    pub(crate) fn new_active(seqs: usize, k: usize) -> std::result::Result<Self, &'static str> {
        if seqs < 1 {
            return Err("cross-sequence verify needs at least one sequence");
        }
        if seqs > QWEN4_CROSS_SEQ_MAX_SEQS {
            return Err("cross-sequence verify exceeds 64 sequences");
        }
        if !(QWEN4_CROSS_SEQ_MIN_K..=QWEN4_CROSS_SEQ_MAX_K).contains(&k) {
            return Err("cross-sequence verify width must be K2-K9");
        }
        let rows = seqs
            .checked_mul(k)
            .ok_or("cross-sequence verify row count overflow")?;
        if rows > QWEN4_CROSS_SEQ_MAX_ROWS {
            return Err("cross-sequence verify frame exceeds 256 rows");
        }
        let mut seq_ks = [0u8; QWEN4_CROSS_SEQ_MAX_SEQS];
        let mut row_bases = [0u16; QWEN4_CROSS_SEQ_MAX_SEQS];
        for s in 0..seqs {
            seq_ks[s] = k as u8;
            row_bases[s] = (s * k) as u16;
        }
        Ok(Self {
            seqs,
            seq_ks,
            row_bases,
            total_rows: rows as u16,
            min_k: k as u8,
            max_k: k as u8,
        })
    }

    /// Construct a cross-sequence frame with per-sequence widths `ks`
    /// (`seqs in 1..=64`, each `k_s in 1..=9`, `sum(k_s) <= 256`).
    pub fn new_ragged(ks: &[usize]) -> std::result::Result<Self, &'static str> {
        let seqs = ks.len();
        if seqs < 1 {
            return Err("cross-sequence verify needs at least one sequence");
        }
        if seqs > QWEN4_CROSS_SEQ_MAX_SEQS {
            return Err("cross-sequence verify exceeds 64 sequences");
        }
        let mut seq_ks = [0u8; QWEN4_CROSS_SEQ_MAX_SEQS];
        let mut row_bases = [0u16; QWEN4_CROSS_SEQ_MAX_SEQS];
        let mut total_rows = 0usize;
        let mut min_k = usize::MAX;
        let mut max_k = 0usize;
        for (s, &k_s) in ks.iter().enumerate() {
            if !(QWEN4_CROSS_SEQ_RAGGED_MIN_K..=QWEN4_CROSS_SEQ_MAX_K).contains(&k_s) {
                return Err("ragged cross-sequence verify width must be K1-K9");
            }
            let next_total = total_rows
                .checked_add(k_s)
                .ok_or("cross-sequence verify row count overflow")?;
            if next_total > QWEN4_CROSS_SEQ_MAX_ROWS {
                return Err("cross-sequence verify frame exceeds 256 rows");
            }
            seq_ks[s] = k_s as u8;
            row_bases[s] = total_rows as u16;
            total_rows = next_total;
            min_k = min_k.min(k_s);
            max_k = max_k.max(k_s);
        }
        Ok(Self {
            seqs,
            seq_ks,
            row_bases,
            total_rows: total_rows as u16,
            min_k: min_k as u8,
            max_k: max_k as u8,
        })
    }

    /// Greedy First-Fit Decreasing bin-packer grouping heterogeneous per-sequence
    /// widths `k_s in 1..=8` into minimal frames with
    /// `sum(k_s) <= QWEN4_CROSS_SEQ_MAX_ROWS` and `seqs in 2..=64`.
    /// Delegates to [`plan_ragged_frames`].
    #[must_use]
    pub fn plan_ragged_frames(ks: &[usize]) -> Vec<Vec<usize>> {
        plan_ragged_frames(ks)
    }

    /// Greedy First-Fit Decreasing bin-packer with an explicit per-frame row
    /// ceiling `max_rows` (clamped to [`QWEN4_CROSS_SEQ_MAX_ROWS`]).
    #[must_use]
    pub fn plan_ragged_frames_with_max_rows(ks: &[usize], max_rows: usize) -> Vec<Vec<usize>> {
        plan_ragged_frames_with_max_rows(ks, max_rows)
    }

    /// Split `n` uniform `k`-row sequences into balanced non-singleton frames
    /// bounded by both `max_rows` (clamped to [`QWEN4_CROSS_SEQ_MAX_ROWS`]) and
    /// [`QWEN4_CROSS_SEQ_MAX_SEQS`]. Delegates to [`plan_frames_with_max_rows`].
    #[must_use]
    pub fn plan_frames_with_max_rows(n: usize, k: usize, max_rows: usize) -> Vec<usize> {
        plan_frames_with_max_rows(n, k, max_rows)
    }

    pub const fn seqs(self) -> usize {
        self.seqs
    }

    /// Uniform width `K` when `!self.is_ragged()`, or `self.max_k()` when ragged.
    /// Per-sequence offsets MUST use [`Self::row_base`] and [`Self::seq_k`].
    pub const fn k(self) -> usize {
        self.max_k as usize
    }

    pub const fn rows(self) -> usize {
        self.total_rows as usize
    }

    /// First row owned by sequence `s` (or `self.rows()` when `s >= self.seqs()`).
    pub const fn row_base(self, s: usize) -> usize {
        if s < self.seqs {
            self.row_bases[s] as usize
        } else {
            self.total_rows as usize
        }
    }

    /// Number of verify rows owned by sequence `u` (`0` if `u >= self.seqs()`).
    #[must_use]
    pub const fn seq_k(self, u: usize) -> usize {
        if u < self.seqs {
            self.seq_ks[u] as usize
        } else {
            0
        }
    }

    /// Narrowest per-sequence verify width `K` across this frame.
    #[must_use]
    pub const fn min_k(self) -> usize {
        self.min_k as usize
    }

    /// Widest per-sequence verify width `K` across this frame.
    #[must_use]
    pub const fn max_k(self) -> usize {
        self.max_k as usize
    }

    /// Whether sequences in this frame carry distinct per-sequence widths `K`.
    #[must_use]
    pub const fn is_ragged(self) -> bool {
        self.min_k != self.max_k
    }
}

/// Greedy First-Fit Decreasing bin-packer for heterogeneous ragged verify widths
/// `k_s in 1..=8`.
///
/// Entries outside `1..=8` are ignored. Returns per-frame index lists into `ks`
/// with each frame satisfying `2 <= frame.len() <= 64` and
/// `sum(k_s) <= QWEN4_CROSS_SEQ_MAX_ROWS` (`256`).
/// When all eligible entries fit in one frame, original order is preserved.
#[must_use]
pub fn plan_ragged_frames(ks: &[usize]) -> Vec<Vec<usize>> {
    plan_ragged_frames_with_max_rows(ks, QWEN4_CROSS_SEQ_MAX_ROWS)
}

/// Greedy First-Fit Decreasing bin-packer with an explicit per-frame row
/// ceiling `max_rows` (clamped to [`QWEN4_CROSS_SEQ_MAX_ROWS`]).
#[must_use]
pub fn plan_ragged_frames_with_max_rows(ks: &[usize], max_rows: usize) -> Vec<Vec<usize>> {
    let row_cap = max_rows.min(QWEN4_CROSS_SEQ_MAX_ROWS);
    let mut eligible: Vec<usize> = ks
        .iter()
        .enumerate()
        .filter_map(|(idx, &k)| {
            (QWEN4_CROSS_SEQ_RAGGED_MIN_K..=QWEN4_CROSS_SEQ_RAGGED_PLAN_MAX_K)
                .contains(&k)
                .then_some(idx)
        })
        .collect();
    if eligible.len() < 2 || row_cap < 2 {
        return Vec::new();
    }
    let total_rows: usize = eligible.iter().map(|&idx| ks[idx]).sum();
    if eligible.len() <= QWEN4_CROSS_SEQ_MAX_SEQS && total_rows <= row_cap {
        return vec![eligible];
    }

    // First-Fit Decreasing by width, tie-broken by original index.
    eligible.sort_by(|&a, &b| ks[b].cmp(&ks[a]).then_with(|| a.cmp(&b)));
    let mut bins: Vec<(Vec<usize>, usize)> = Vec::new();
    for idx in eligible {
        let k = ks[idx];
        if k > row_cap {
            continue;
        }
        if let Some((members, rows)) = bins.iter_mut().find(|(members, rows)| {
            members.len() < QWEN4_CROSS_SEQ_MAX_SEQS && *rows + k <= row_cap
        }) {
            members.push(idx);
            *rows += k;
        } else {
            bins.push((vec![idx], k));
        }
    }

    // Eliminate singleton bins by borrowing one member from the fullest bin of
    // size >= 3 whose smallest member fits within `row_cap` rows.
    for singleton_idx in 0..bins.len() {
        if bins[singleton_idx].0.len() != 1 {
            continue;
        }
        let s_rows = bins[singleton_idx].1;
        let donor = bins
            .iter()
            .enumerate()
            .filter(|&(j, (members, _))| j != singleton_idx && members.len() >= 3)
            .filter_map(|(j, (members, _))| {
                members
                    .iter()
                    .enumerate()
                    .rev()
                    .find(|&(_, &m_idx)| s_rows + ks[m_idx] <= row_cap)
                    .map(|(pos, _)| (j, pos, members.len()))
            })
            .max_by_key(|&(_, _, len)| len);
        if let Some((donor_bin, donor_pos, _)) = donor {
            let moved = bins[donor_bin].0.remove(donor_pos);
            let moved_k = ks[moved];
            bins[donor_bin].1 -= moved_k;
            bins[singleton_idx].0.push(moved);
            bins[singleton_idx].1 += moved_k;
        }
    }

    let mut out: Vec<Vec<usize>> = bins
        .into_iter()
        .filter(|(members, _)| members.len() >= 2)
        .map(|(mut members, _)| {
            members.sort_unstable();
            members
        })
        .collect();
    out.sort_by_key(|members| members[0]);
    out
}

/// Largest number of `k`-row sequences one frame of at most `max_rows` rows
/// can hold, bounded by [`QWEN4_CROSS_SEQ_MAX_SEQS`] (`64`).
#[must_use]
pub const fn max_seqs_per_frame_with_max_rows(k: usize, max_rows: usize) -> usize {
    if k == 0 {
        return 0;
    }
    let row_cap = if max_rows > QWEN4_CROSS_SEQ_MAX_ROWS {
        QWEN4_CROSS_SEQ_MAX_ROWS
    } else {
        max_rows
    };
    let by_rows = row_cap / k;
    if by_rows > QWEN4_CROSS_SEQ_MAX_SEQS {
        QWEN4_CROSS_SEQ_MAX_SEQS
    } else {
        by_rows
    }
}

/// Largest number of `k`-row sequences one frame can hold (bounded by both
/// [`QWEN4_CROSS_SEQ_MAX_ROWS`] and [`QWEN4_CROSS_SEQ_MAX_SEQS`]).
pub const fn max_seqs_per_frame(k: usize) -> usize {
    max_seqs_per_frame_with_max_rows(k, QWEN4_CROSS_SEQ_MAX_ROWS)
}

/// Split `n` eligible sequences into frames of at most
/// [`max_seqs_per_frame_with_max_rows`]`(k, max_rows)` sequences, never
/// leaving a singleton frame when it can be avoided.
#[must_use]
pub fn plan_frames_with_max_rows(n: usize, k: usize, max_rows: usize) -> Vec<usize> {
    let cap = max_seqs_per_frame_with_max_rows(k, max_rows);
    if cap < 2 || n < 2 {
        return Vec::new();
    }
    let frames = n.div_ceil(cap);
    // Balanced split: sizes differ by at most one, so no frame is a
    // singleton whenever n >= 2 * frames.
    let base = n / frames;
    let extra = n % frames;
    let sizes: Vec<usize> = (0..frames).map(|i| base + usize::from(i < extra)).collect();
    sizes.into_iter().filter(|&size| size >= 2).collect()
}

/// Split `n` eligible sequences into frames of at most
/// [`max_seqs_per_frame`]`(k)` sequences, never leaving a singleton frame
/// when it can be avoided (a singleton goes back to the C1 verifier).
/// Returns the per-frame sequence counts in order.
pub fn plan_frames(n: usize, k: usize) -> Vec<usize> {
    plan_frames_with_max_rows(n, k, QWEN4_CROSS_SEQ_MAX_ROWS)
}

/// One sequence's view of a cross-sequence frame for a single layer.
pub struct Qwen4CrossSeqRows<'a> {
    pub state: &'a mut (dyn LayerState + 'static),
    pub seq_len: usize,
    pub block_table: &'a mut Vec<u32>,
    pub disk_block_ids: &'a mut Vec<u32>,
    pub disk_last_offloaded_per_layer: &'a mut Vec<u32>,
    /// This sequence's recurrent intermediates for this SSM layer
    /// (`SsmStatePool::{h,conv}_intermediate(layer, slot, 0)`); NULL for
    /// attention layers.
    pub h_intermediate: DevicePtr,
    pub conv_intermediate: DevicePtr,
    pub h_intermediate_stride: usize,
    pub conv_intermediate_stride: usize,
    /// Attention metadata for exactly this sequence's `k` rows.
    pub metadata: AttnMetadataDev,
    /// Physical SSM slot; selects the QSA-journal / GDN-retain lane.
    pub slot_idx: usize,
}

/// Verifier-owned staging for row-parallel inputs/outputs that per-sequence
/// attention cores must not overlap. Each attention core runs against the
/// exact single-sequence buffer footprint (`gate_logits` rows `0..k`,
/// `ssm_qkvz` rows `0..k`), and the frame copies rows in and out of here.
#[derive(Clone, Copy, Debug)]
pub struct Qwen4CrossSeqStaging {
    /// Mixed attention-hyperconnection rows, `[rows, hidden]` BF16.
    pub attn_inputs: DevicePtr,
    pub attn_inputs_bytes: usize,
    /// Gated attention-core rows awaiting the batched O projection,
    /// `[rows, q_heads * head_dim]` BF16.
    pub raw_rows: DevicePtr,
    pub raw_rows_bytes: usize,
    /// Dedicated device buffer for `QsaBatchSeqDesc[0..C]` uploaded once per
    /// verify step inside `qwen4_cross_seq_upload_metadata` (`64 * 32 = 2048 B`).
    pub verify_meta_dev: DevicePtr,
    pub verify_meta_bytes: usize,
    /// Model-wide shared scratch arena for cross-sequence batched QSA verify
    /// (`~57.2 MiB`), allocated once when `config.qwen4_qsa` is enabled.
    pub qsa_scratch: crate::layers::qwen4_qsa::Qwen4CrossSeqQsaScratch,
}

impl Qwen4CrossSeqStaging {
    /// Exact byte size of `verify_meta_dev` (`64 × 32 B = 2048 B`).
    pub const VERIFY_META_BYTES: usize = crate::layers::qwen4_qsa::QSA_CROSS_SEQ_MAX_SEQS
        * crate::layers::qwen4_qsa::QsaBatchSeqDesc::SIZE_BYTES;

    /// Device bytes of `(attn_inputs, raw_rows)` for a full 256-row frame.
    pub fn bytes_for(config: &ModelConfig) -> anyhow::Result<(usize, usize)> {
        staging_bytes(
            config.hidden_size,
            config.num_attention_heads,
            config.head_dim,
        )
    }

    /// Total device bytes of [`Self`] including `verify_meta_dev` and the
    /// model-wide QSA batch scratch when `config.qwen4_qsa` is enabled (startup ledger).
    pub fn total_bytes_for(config: &ModelConfig) -> anyhow::Result<usize> {
        let (inputs, raw) = Self::bytes_for(config)?;
        let qsa_bytes = if config.qwen4_qsa {
            crate::layers::qwen4_qsa::Qwen4CrossSeqQsaScratch::total_bytes()?
        } else {
            0
        };
        inputs
            .checked_add(raw)
            .and_then(|total| total.checked_add(Self::VERIFY_META_BYTES))
            .and_then(|total| total.checked_add(qsa_bytes))
            .ok_or_else(|| anyhow::anyhow!("cross-sequence staging total overflow"))
    }

    /// Allocate one model-wide cross-sequence verify staging bundle.
    pub fn allocate(gpu: &dyn GpuBackend, config: &ModelConfig) -> anyhow::Result<Self> {
        let (attn_inputs_bytes, raw_rows_bytes) = Self::bytes_for(config)?;
        let verify_meta_bytes = Self::VERIFY_META_BYTES;
        let qsa_scratch = if config.qwen4_qsa {
            crate::layers::qwen4_qsa::Qwen4CrossSeqQsaScratch::allocate(gpu)?
        } else {
            crate::layers::qwen4_qsa::Qwen4CrossSeqQsaScratch::NULL
        };
        Ok(Self {
            attn_inputs: gpu.alloc(attn_inputs_bytes)?,
            attn_inputs_bytes,
            raw_rows: gpu.alloc(raw_rows_bytes)?,
            raw_rows_bytes,
            verify_meta_dev: gpu.alloc(verify_meta_bytes)?,
            verify_meta_bytes,
            qsa_scratch,
        })
    }
}

fn staging_bytes(
    hidden_size: usize,
    q_heads: usize,
    head_dim: usize,
) -> anyhow::Result<(usize, usize)> {
    let row = |width: usize| {
        width
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_mul(QWEN4_CROSS_SEQ_MAX_ROWS))
    };
    let inputs = row(hidden_size);
    let raw = q_heads.checked_mul(head_dim).and_then(row);
    match (inputs, raw) {
        (Some(inputs), Some(raw)) if inputs > 0 && raw > 0 => Ok((inputs, raw)),
        _ => anyhow::bail!("cross-sequence staging geometry is empty or overflows"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn staging_covers_256_flash_next_rows() {
        // Flash-Next: hidden 2560, 24 query heads x 256.
        assert_eq!(
            staging_bytes(2_560, 24, 256).unwrap(),
            (256 * 2_560 * 2, 256 * 24 * 256 * 2)
        );
        assert!(staging_bytes(0, 24, 256).is_err());
        assert!(staging_bytes(2_560, usize::MAX, 2).is_err());
    }

    #[test]
    fn frame_admits_k5_up_to_fifty_one_sequences() {
        for seqs in 2..=51 {
            let frame = Qwen4CrossSeqFrame::new(seqs, 5).unwrap();
            assert_eq!(frame.rows(), seqs * 5);
            assert_eq!(frame.row_base(seqs - 1), (seqs - 1) * 5);
        }
        assert!(Qwen4CrossSeqFrame::new(52, 5).is_err(), "260 rows > 256");
        // C=64 x K=4 = 256 rows and C=32 x K=8 = 256 rows both fit in one frame.
        assert_eq!(Qwen4CrossSeqFrame::new(64, 4).unwrap().rows(), 256);
        assert_eq!(Qwen4CrossSeqFrame::new(32, 8).unwrap().rows(), 256);
        assert!(Qwen4CrossSeqFrame::new(33, 8).is_err(), "264 rows > 256");
    }

    #[test]
    fn frame_rejects_c1_and_unqualified_widths() {
        assert!(Qwen4CrossSeqFrame::new(1, 5).is_err());
        assert!(Qwen4CrossSeqFrame::new(0, 5).is_err());
        assert!(Qwen4CrossSeqFrame::new(2, 1).is_err());
        assert!(Qwen4CrossSeqFrame::new(2, 10).is_err());
        assert!(Qwen4CrossSeqFrame::new(2, 16).is_err());
        assert!(Qwen4CrossSeqFrame::new(usize::MAX, 9).is_err());
        assert!(Qwen4CrossSeqFrame::new(64, 4).is_ok());
        assert!(Qwen4CrossSeqFrame::new(65, 2).is_err());
    }

    #[test]
    fn per_frame_capacity_follows_the_256_row_and_64_seq_ceilings() {
        assert_eq!(max_seqs_per_frame(8), 32);
        assert_eq!(max_seqs_per_frame(5), 51);
        assert_eq!(max_seqs_per_frame(4), 64);
        // K=3 and K=2 clamp at QWEN4_CROSS_SEQ_MAX_SEQS (64).
        assert_eq!(max_seqs_per_frame(3), 64);
        assert_eq!(max_seqs_per_frame(2), 64);
        assert_eq!(max_seqs_per_frame(9), 28);
        assert_eq!(max_seqs_per_frame(0), 0);
        // Explicit 128-row GB10 ceiling remains exact.
        assert_eq!(max_seqs_per_frame_with_max_rows(8, 128), 16);
        assert_eq!(max_seqs_per_frame_with_max_rows(5, 128), 25);
        assert_eq!(max_seqs_per_frame_with_max_rows(4, 128), 32);
        assert_eq!(max_seqs_per_frame_with_max_rows(3, 128), 42);
        assert_eq!(max_seqs_per_frame_with_max_rows(2, 128), 64);
    }

    #[test]
    fn plans_cover_every_sequence_without_singletons() {
        for max_rows in [128usize, 256usize] {
            for k in 2..=9 {
                for n in 2..=128 {
                    let plan = plan_frames_with_max_rows(n, k, max_rows);
                    let cap = max_seqs_per_frame_with_max_rows(k, max_rows);
                    assert!(
                        plan.iter().all(|&s| (2..=cap).contains(&s)),
                        "max_rows={max_rows} n={n} k={k} {plan:?}"
                    );
                    assert_eq!(
                        plan.iter().sum::<usize>(),
                        n,
                        "max_rows={max_rows} n={n} k={k} {plan:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn k5_plans_match_the_documented_concurrency_ladder() {
        assert_eq!(plan_frames(2, 5), vec![2]);
        assert_eq!(plan_frames(8, 5), vec![8]);
        assert_eq!(plan_frames(51, 5), vec![51]);
        assert_eq!(plan_frames(52, 5), vec![26, 26]);
        assert_eq!(plan_frames(64, 4), vec![64]);
        assert_eq!(plan_frames(32, 8), vec![32]);
        assert_eq!(plan_frames_with_max_rows(25, 5, 128), vec![25]);
        assert_eq!(plan_frames_with_max_rows(26, 5, 128), vec![13, 13]);
        assert_eq!(plan_frames_with_max_rows(32, 5, 128), vec![16, 16]);
        assert!(plan_frames(1, 5).is_empty());
        assert!(
            plan_frames(8, 129).is_empty(),
            "K129 cannot pair within 256 rows"
        );
    }

    #[test]
    fn ragged_ready_accessors_report_uniform_frame_geometry() {
        let frame = Qwen4CrossSeqFrame::new(4, 5).unwrap();
        assert!(!frame.is_ragged());
        assert_eq!(frame.min_k(), 5);
        assert_eq!(frame.max_k(), 5);
        for u in 0..4 {
            assert_eq!(frame.seq_k(u), 5);
            assert_eq!(frame.row_base(u), u * 5);
        }
        assert_eq!(frame.seq_k(4), 0);
        assert_eq!(frame.row_base(4), 20);
    }

    #[test]
    fn new_ragged_supports_mixed_k1_through_k9_and_cumulative_row_bases() {
        let ks = [7, 3, 1, 5, 8, 2, 9, 4];
        let frame = Qwen4CrossSeqFrame::new_ragged(&ks).unwrap();
        assert!(frame.is_ragged());
        assert_eq!(frame.seqs(), ks.len());
        assert_eq!(frame.min_k(), 1);
        assert_eq!(frame.max_k(), 9);
        assert_eq!(frame.k(), 9);
        assert_eq!(frame.rows(), ks.iter().sum::<usize>());
        let mut expected_base = 0usize;
        for (s, &k_s) in ks.iter().enumerate() {
            assert_eq!(frame.seq_k(s), k_s);
            assert_eq!(frame.row_base(s), expected_base);
            expected_base += k_s;
        }
        assert_eq!(frame.seq_k(ks.len()), 0);
        assert_eq!(frame.row_base(ks.len()), expected_base);

        // Anchor-only uniform K=1 frame is valid via new_ragged and reports !is_ragged().
        let anchor_only = Qwen4CrossSeqFrame::new_ragged(&[1, 1, 1]).unwrap();
        assert!(!anchor_only.is_ragged());
        assert_eq!(anchor_only.min_k(), 1);
        assert_eq!(anchor_only.max_k(), 1);
        assert_eq!(anchor_only.rows(), 3);
        assert_eq!(anchor_only.row_base(2), 2);

        // Single-sequence leader sub-frame is valid via new_ragged.
        let single = Qwen4CrossSeqFrame::new_ragged(&[7]).unwrap();
        assert_eq!(single.seqs(), 1);
        assert_eq!(single.rows(), 7);
        assert!(!single.is_ragged());
    }

    #[test]
    fn new_ragged_enforces_bounds() {
        assert!(Qwen4CrossSeqFrame::new_ragged(&[]).is_err());
        assert!(Qwen4CrossSeqFrame::new_ragged(&[0, 5]).is_err());
        assert!(Qwen4CrossSeqFrame::new_ragged(&[5, 10]).is_err());
        assert!(Qwen4CrossSeqFrame::new_ragged(&[1; 64]).is_ok());
        assert!(Qwen4CrossSeqFrame::new_ragged(&[1; 65]).is_err());
        // Exactly 256 rows fits; 257 rows rejects.
        assert!(Qwen4CrossSeqFrame::new_ragged(&[8; 32]).is_ok());
        let mut over = vec![8usize; 32];
        over.push(1);
        assert!(Qwen4CrossSeqFrame::new_ragged(&over).is_err());
    }

    #[test]
    fn plan_ragged_frames_packs_heterogeneous_widths_without_singletons() {
        // Single-frame fast path preserves original order.
        let ks = [7, 1, 5, 3, 8, 2];
        assert_eq!(plan_ragged_frames(&ks), vec![vec![0, 1, 2, 3, 4, 5]]);

        // Entries outside 1..=8 are ignored cleanly.
        assert_eq!(
            plan_ragged_frames(&[0, 7, 9, 1, 12, 5]),
            vec![vec![1, 3, 5]]
        );
        assert!(plan_ragged_frames(&[0, 7, 9]).is_empty());

        // 32 K=8 sequences (256 rows) fit in 1 frame under 256-row cap;
        // 33 K=8 sequences (264 > 256 rows) pack into 2 frames with zero singletons.
        let thirty_two_k8 = vec![8usize; 32];
        assert_eq!(
            Qwen4CrossSeqFrame::plan_ragged_frames(&thirty_two_k8),
            vec![(0..32).collect::<Vec<_>>()]
        );
        let thirty_three_k8 = vec![8usize; 33];
        let bins = Qwen4CrossSeqFrame::plan_ragged_frames(&thirty_three_k8);
        assert_eq!(bins.len(), 2);
        let mut covered = Vec::new();
        for bin in &bins {
            assert!((2..=64).contains(&bin.len()), "bin={bin:?}");
            let rows: usize = bin.iter().map(|&i| thirty_three_k8[i]).sum();
            assert!(rows <= 256, "rows={rows}");
            covered.extend_from_slice(bin);
        }
        covered.sort_unstable();
        assert_eq!(covered, (0..33).collect::<Vec<_>>());

        // Explicit 128-row cap splits 17 K=8 sequences (136 > 128) into 2 frames.
        let seventeen_k8 = vec![8usize; 17];
        let bins_128 = Qwen4CrossSeqFrame::plan_ragged_frames_with_max_rows(&seventeen_k8, 128);
        assert_eq!(bins_128.len(), 2);
        assert!(bins_128.iter().all(|b| b.len() >= 2 && b.len() * 8 <= 128));

        // 65 anchor-only K=1 sequences exceed 64 seqs -> 2 frames (63 + 2), zero singletons.
        let sixty_five_k1 = vec![1usize; 65];
        let k1_bins = plan_ragged_frames(&sixty_five_k1);
        assert_eq!(k1_bins.len(), 2);
        assert!(k1_bins.iter().all(|b| (2..=64).contains(&b.len())));
        assert_eq!(k1_bins.iter().map(Vec::len).sum::<usize>(), 65);

        // Large heterogeneous cohort of 40 sequences across K=1..=8 (sum = 180):
        // 1 bin under 256-row cap, 2 bins under 128-row cap.
        let mixed: Vec<usize> = (0..40).map(|i| (i % 8) + 1).collect();
        assert_eq!(plan_ragged_frames(&mixed).len(), 1);
        let mixed_bins_128 = plan_ragged_frames_with_max_rows(&mixed, 128);
        assert_eq!(mixed_bins_128.len(), 2);
        let mut seen = Vec::new();
        for bin in &mixed_bins_128 {
            assert!((2..=64).contains(&bin.len()));
            let rows: usize = bin.iter().map(|&i| mixed[i]).sum();
            assert!(rows <= 128);
            seen.extend_from_slice(bin);
        }
        seen.sort_unstable();
        assert_eq!(seen, (0..40).collect::<Vec<_>>());
    }
}
