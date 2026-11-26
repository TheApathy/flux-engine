// SPDX-License-Identifier: AGPL-3.0-only

//! Exact fused score -> top-k for the QSA-sparse prefill core
//! (`ATLAS_QWEN4_PREFILL_QSA_FUSED_TOPK=1`, default off; K8 item E, phase 1).
//!
//! `qwen4_qsa_prefill_score_gmax` writes the same fp32 scores as
//! `qwen4_qsa_prefill_score` plus, per row, the largest key of every 8-block
//! group. `qwen4_qsa_prefill_topk_fused` then selects tau0 = the 512th-largest
//! group maximum, keeps the groups whose maximum is >= tau0 and runs today's
//! radix select (same passes, same lower-block tie rule) over their scores
//! only: about 4,100 keys per row instead of n (32,768 at 128K). The group
//! maxima are real scores, so tau0 <= tau (the row's 512th-largest score) and
//! the candidates hold every key >= tau: the mask is byte-identical. Rows whose
//! candidates exceed the shared-memory cap fall back to the full-row select.
//!
//! This file is std-only so `rustc --test` runs its tests without the crate:
//! the selector, the launch geometry, the per-batch engagement rule, and a CPU
//! model of both select algorithms checked against each other.

use std::sync::OnceLock;

pub const SELECTOR: &str = "ATLAS_QWEN4_PREFILL_QSA_FUSED_TOPK";
pub const FUSED_RADIX_ALIAS: &str = "ATLAS_QWEN4_QSA_FUSED_RADIX";
/// Optional: the smallest n_p (complete blocks of a batch's first row) that
/// takes the fused path. Below it the candidates are a large share of the row.
pub const MIN_BLOCKS_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_QSA_FUSED_TOPK_MIN_BLOCKS";
/// Blocks per group: one m16n8 n-tile of the score kernel.
pub const GROUP: usize = 8;
pub const BLOCK_TOPK: usize = 512;
const COMPRESS_RATIO: usize = 4;
/// Default MIN_BLOCKS: 10752 blocks = 43,008 tokens, the measured crossover
/// (micro-queue 112 QSA_FUSED_SWEEP, fused/unfused per row at the engine's V2
/// batch: 1.29 at n = 6144, 1.17 at 8064, 0.90 at 12208, 0.81 at 16320, 0.86 at
/// 32640; linear crossover ~10,700). Below it the 512 candidate groups are too
/// large a share of the row for the two-stage select to pay for itself.
pub const DEFAULT_MIN_BLOCKS: usize = 10_752;
/// When `score_v2` / `c3_v2` is active, rows up to `n <= 8,192` blocks (`32K` prefill) stay in
/// `qwen4_qsa_prefill_score` (`5 CTAs/SM`, zero `gmax` overhead) + `topk_v2`'s hybrid `rk[16]` +
/// tail `skeys` path (`<= 16 KB` dyn SMEM = `3 CTAs/SM`), and batches with `n_p >= 8,192` blocks
/// (`32,768` tokens, `64K` and `128K` prefill) engage `qwen4_qsa_prefill_topk_fused_c3_v2`.
pub const SCORE_V2_DEFAULT_MIN_BLOCKS: usize = 8_192;
/// Static shared memory of `qwen4_qsa_prefill_topk_fused`: hist[2048] +
/// s_warp[8] + s_sc[4] = 8,240 B; 8,256 leaves alignment slack.
pub const STATIC_SMEM: usize = 8_256;
/// Dynamic shared memory that keeps two 256-thread CTAs per SM (100 KB per SM,
/// 1 KB reserved per CTA): 2 * (8,256 + 40,960 + 1,024) = 100,480 B.
pub const TWO_CTA_DYN_SMEM: usize = 40 * 1024;
/// The opt-in ceiling requested at load (same as the staged top-k's).
pub const MAX_DYN_SMEM: usize = 88 * 1024;
/// GB10 (sm_121) MaxSharedMemoryPerBlockOptin.
const SMEM_OPTIN_MAX: usize = 101_376;
const _: () = assert!(STATIC_SMEM + MAX_DYN_SMEM <= SMEM_OPTIN_MAX);
/// Candidate groups per row: 512 plus ties among group maxima at tau0.
pub const MIN_CAP_GROUPS: usize = 576;
pub const MAX_CAP_GROUPS: usize = 1024;
/// Shared words per candidate group: 8 keys + its index.
const WORDS_PER_GROUP: usize = GROUP + 1;

pub fn parse_selected_for(value: Option<&str>, var_name: &str) -> Result<bool, String> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => Err(format!("{var_name} must be absent, 0, or 1; got {other:?}")),
    }
}

pub fn parse_selected(value: Option<&str>) -> Result<bool, String> {
    parse_selected_for(value, SELECTOR)
}

pub fn parse_min_blocks(value: Option<&str>) -> Result<usize, String> {
    let Some(text) = value else {
        return Ok(DEFAULT_MIN_BLOCKS);
    };
    let n: usize = text
        .parse()
        .map_err(|_| format!("{MIN_BLOCKS_SELECTOR} must be an integer; got {text:?}"))?;
    if n <= BLOCK_TOPK {
        return Err(format!(
            "{MIN_BLOCKS_SELECTOR} must exceed {BLOCK_TOPK} (rows with n_p <= 512 select every block); got {n}"
        ));
    }
    Ok(n)
}

pub fn parse_min_blocks_for(value: Option<&str>, score_v2: bool) -> Result<usize, String> {
    if value.is_none() {
        return Ok(if score_v2 {
            SCORE_V2_DEFAULT_MIN_BLOCKS
        } else {
            DEFAULT_MIN_BLOCKS
        });
    }
    parse_min_blocks(value)
}

fn selected_source() -> Result<Option<&'static str>, String> {
    static S: OnceLock<Result<Option<&'static str>, String>> = OnceLock::new();
    S.get_or_init(|| {
        let alias = if let Ok(v) = std::env::var(FUSED_RADIX_ALIAS) {
            parse_selected_for(Some(v.as_str()), FUSED_RADIX_ALIAS)?
        } else {
            false
        };
        let primary = parse_selected_for(std::env::var(SELECTOR).ok().as_deref(), SELECTOR)?;
        let active_selector = if alias {
            Some(FUSED_RADIX_ALIAS)
        } else if primary {
            Some(SELECTOR)
        } else {
            None
        };
        if let Some(sel) = active_selector {
            tracing::info!("QSA_FUSED_RADIX_ENGAGED selector={sel}");
        }
        Ok(active_selector)
    })
    .clone()
}

/// The activating env var name (`ATLAS_QWEN4_QSA_FUSED_RADIX` if set via the
/// alias, else `ATLAS_QWEN4_PREFILL_QSA_FUSED_TOPK`).
pub fn active_selector() -> &'static str {
    selected_source().ok().flatten().unwrap_or(SELECTOR)
}

/// Whether the fused path is selected (read once per process).
/// Supports either `ATLAS_QWEN4_PREFILL_QSA_FUSED_TOPK` or `ATLAS_QWEN4_QSA_FUSED_RADIX`.
pub fn selected() -> Result<bool, String> {
    selected_source().map(|src| src.is_some())
}

/// The engagement threshold in blocks (read once per process).
pub fn min_blocks() -> Result<usize, String> {
    static S: OnceLock<Result<usize, String>> = OnceLock::new();
    S.get_or_init(|| parse_min_blocks(std::env::var(MIN_BLOCKS_SELECTOR).ok().as_deref()))
        .clone()
}

/// The engagement threshold in blocks for `score_v2` on/off (read once per process).
pub fn min_blocks_for(score_v2: bool) -> Result<usize, String> {
    if !score_v2 {
        return min_blocks();
    }
    static S_V2: OnceLock<Result<usize, String>> = OnceLock::new();
    S_V2.get_or_init(|| {
        parse_min_blocks_for(std::env::var(MIN_BLOCKS_SELECTOR).ok().as_deref(), true)
    })
    .clone()
}

/// Row stride of the group-maxima region for a score row stride `ld` (a
/// multiple of 64, so `ld / 8` is a multiple of 8).
pub fn gmax_ld(ld: usize) -> usize {
    ld / GROUP
}

/// Shared-memory layout of one `qwen4_qsa_prefill_topk_fused` launch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FusedLaunch {
    /// Words of the maxima / mask-word region (>= ld/8 and >= mask_words; % 4 == 0).
    pub a_words: usize,
    pub cap_groups: usize,
    pub dyn_smem: usize,
}

/// Geometry for a chunk with score stride `ld` and `mask_words` mask words:
/// the largest candidate cap that keeps two CTAs per SM, else the largest the
/// opt-in ceiling allows; `None` when even that holds fewer than
/// [`MIN_CAP_GROUPS`] (the chunk then keeps today's select).
pub fn launch_geometry(ld: usize, mask_words: usize) -> Option<FusedLaunch> {
    let a_words = gmax_ld(ld).max(mask_words).next_multiple_of(4);
    for budget in [TWO_CTA_DYN_SMEM, MAX_DYN_SMEM] {
        let Some(free) = budget.checked_sub(a_words * 4) else {
            continue;
        };
        let cap_groups = (free / (WORDS_PER_GROUP * 4)).min(MAX_CAP_GROUPS);
        if cap_groups >= MIN_CAP_GROUPS {
            return Some(FusedLaunch {
                a_words,
                cap_groups,
                dyn_smem: (a_words + WORDS_PER_GROUP * cap_groups) * 4,
            });
        }
    }
    None
}

/// Dynamic shared memory that keeps three CTAs per SM: 3 * (8,256 + 24,576 + 1,024) = 101,568 B.
pub const THREE_CTA_DYN_SMEM: usize = 24 * 1024;
/// The compact select keeps the mask words in its 2048-word radix histogram.
pub const COMPACT_MAX_MASK_WORDS: usize = 2048;

/// Geometry of `qwen4_qsa_prefill_topk_fused_c3`: the candidates' keys overwrite the staged
/// maxima (region A = max(ld/8, 8 * cap) words) and the mask words live in the static histogram,
/// so the dynamic part is (region A + cap) words. The largest cap that keeps three CTAs per SM
/// (one wave for a 128-row V2 batch), else two, else the opt-in ceiling; `None` past 2048 mask
/// words (262K tokens) or when no cap of at least [`MIN_CAP_GROUPS`] fits.
pub fn launch_geometry_compact(ld: usize, mask_words: usize) -> Option<FusedLaunch> {
    if mask_words > COMPACT_MAX_MASK_WORDS {
        return None;
    }
    for budget in [THREE_CTA_DYN_SMEM, TWO_CTA_DYN_SMEM, MAX_DYN_SMEM] {
        for cap_groups in (MIN_CAP_GROUPS..=MAX_CAP_GROUPS).rev() {
            let a_words = gmax_ld(ld).max(GROUP * cap_groups).next_multiple_of(4);
            if (a_words + cap_groups) * 4 <= budget {
                return Some(FusedLaunch {
                    a_words,
                    cap_groups,
                    dyn_smem: (a_words + cap_groups) * 4,
                });
            }
        }
    }
    None
}

/// Whether a score batch whose first row is at absolute position `first_pos`
/// takes the fused path: every row of the batch has at least `min_blocks`
/// complete blocks (rows only grow within a batch).
pub fn batch_engages(first_pos: usize, min_blocks: usize) -> bool {
    (first_pos + 1) / COMPRESS_RATIO >= min_blocks.max(BLOCK_TOPK + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selector_is_strict() {
        assert_eq!(parse_selected(None), Ok(false));
        assert_eq!(parse_selected(Some("0")), Ok(false));
        assert_eq!(parse_selected(Some("1")), Ok(true));
        assert!(parse_selected(Some("true")).is_err());
        assert!(parse_selected(Some("2")).is_err());
        let alias_err = parse_selected_for(Some("bad"), FUSED_RADIX_ALIAS).unwrap_err();
        assert!(alias_err.contains(FUSED_RADIX_ALIAS), "{alias_err}");
        assert!(!alias_err.contains(SELECTOR), "{alias_err}");
        assert_eq!(parse_min_blocks(None), Ok(DEFAULT_MIN_BLOCKS));
        assert_eq!(parse_min_blocks(Some("4096")), Ok(4096));
        assert!(parse_min_blocks(Some("512")).is_err());
        assert!(parse_min_blocks(Some("x")).is_err());
    }

    /// The serve geometry (V2 score batches, --max-seq-len 135168): every chunk
    /// keeps two CTAs per SM with room for 512 groups plus ties.
    #[test]
    fn launch_geometry_keeps_two_ctas_per_sm_through_the_context_limit() {
        for end in [16_384usize, 32_768, 65_536, 98_304, 131_072, 135_168] {
            let blocks = end / COMPRESS_RATIO;
            let ld = blocks.next_multiple_of(64);
            let mask_words = (blocks + 1).div_ceil(32);
            let g = launch_geometry(ld, mask_words).expect("geometry");
            assert!(g.dyn_smem <= TWO_CTA_DYN_SMEM, "end={end}: {g:?}");
            assert!(
                g.cap_groups >= MIN_CAP_GROUPS && g.a_words.is_multiple_of(4),
                "end={end}"
            );
            assert!(g.a_words >= ld / GROUP && g.a_words >= mask_words);
            assert!(2 * (STATIC_SMEM + g.dyn_smem + 1024) <= 102_400);
        }
        // 128K: 4,096 maxima, 682 candidate groups.
        let g = launch_geometry(32_768, 1025).unwrap();
        assert_eq!((g.a_words, g.cap_groups, g.dyn_smem), (4096, 682, 40_936));
        // 256K needs the one-CTA budget; 1M has no room for the maxima.
        let g = launch_geometry(65_536, 2049).unwrap();
        assert!(g.dyn_smem > TWO_CTA_DYN_SMEM && g.dyn_smem <= MAX_DYN_SMEM);
        assert_eq!(launch_geometry(262_144, 8193), None);
    }

    #[test]
    fn default_threshold_is_the_measured_crossover() {
        // The 32K request (rows up to n = 8192) keeps the unfused select; 64K's
        // last chunk (n >= 12288) and every later chunk take the fused one.
        assert_eq!(parse_min_blocks(None), Ok(10_752));
        assert!(!batch_engages(32_767, DEFAULT_MIN_BLOCKS));
        assert!(!batch_engages(43_006, DEFAULT_MIN_BLOCKS));
        assert!(batch_engages(43_007, DEFAULT_MIN_BLOCKS));
        assert!(batch_engages(49_152, DEFAULT_MIN_BLOCKS));
    }

    #[test]
    fn compact_geometry_keeps_three_ctas_per_sm_through_the_context_limit() {
        for end in [16_384usize, 32_768, 65_536, 98_304, 131_072, 135_168] {
            let blocks = end / COMPRESS_RATIO;
            let ld = blocks.next_multiple_of(64);
            let mask_words = (blocks + 1).div_ceil(32);
            let g = launch_geometry_compact(ld, mask_words).expect("geometry");
            assert!(g.dyn_smem <= THREE_CTA_DYN_SMEM, "end={end}: {g:?}");
            assert!(
                3 * (STATIC_SMEM + g.dyn_smem + 1024) <= 102_400,
                "end={end}"
            );
            assert!(g.cap_groups >= MIN_CAP_GROUPS && g.a_words.is_multiple_of(4));
            // Region A holds the maxima, then the candidates' keys.
            assert!(g.a_words >= ld / GROUP && g.a_words >= GROUP * g.cap_groups);
            assert!(mask_words <= COMPACT_MAX_MASK_WORDS);
        }
        // 128K: the same 682-group cap as the staged select, in 24,552 B instead of 40,936.
        let g = launch_geometry_compact(32_768, 1025).unwrap();
        assert_eq!((g.a_words, g.cap_groups, g.dyn_smem), (5456, 682, 24_552));
        // Past 262K tokens the mask words no longer fit the histogram.
        assert!(launch_geometry_compact(65_536, 2049).is_none());
    }

    #[test]
    fn compact_geometry_covers_the_native_262k_boundary() {
        // 262,080 is the retrieval prompt and 262,143 is the largest admitted
        // prompt when one output token must fit under max_seq_len=262,144.
        // Both cross c3_v2's ng > 4096 shared-maxima branch and sit exactly at
        // the compact selector's 2,048-word mask ceiling.
        for end in [262_080usize, 262_143] {
            let blocks = end / COMPRESS_RATIO;
            let ld = blocks.next_multiple_of(64);
            let mask_words = (blocks + 1).div_ceil(32);
            let ng = blocks.div_ceil(GROUP);
            let g = launch_geometry_compact(ld, mask_words).expect("native 262K geometry");

            assert!(ng > 4096, "end={end}: ng={ng}");
            assert_eq!(ld, 65_536, "end={end}");
            assert_eq!(mask_words, COMPACT_MAX_MASK_WORDS, "end={end}");
            assert_eq!(
                (g.a_words, g.cap_groups, g.dyn_smem),
                (8192, 1024, 36_864),
                "end={end}"
            );
            assert!(g.a_words >= ng, "end={end}: {g:?}");
            assert!(g.a_words >= GROUP * g.cap_groups, "end={end}: {g:?}");
            assert_eq!(g.dyn_smem, (g.a_words + g.cap_groups) * 4);
            assert!(g.dyn_smem <= TWO_CTA_DYN_SMEM, "end={end}: {g:?}");
            assert!(2 * (STATIC_SMEM + g.dyn_smem + 1024) <= 102_400);
        }

        // At the exact 262,144-token endpoint the plan's sentinel word raises
        // mask_words to 2,049. Compact selection must refuse it, while the
        // incumbent fused selector remains a safe one-CTA option and the caller
        // can otherwise fall back to the unfused global selector.
        let blocks = 262_144usize / COMPRESS_RATIO;
        let ld = blocks.next_multiple_of(64);
        let mask_words = (blocks + 1).div_ceil(32);
        assert_eq!((blocks, ld, mask_words), (65_536, 65_536, 2_049));
        assert!(launch_geometry_compact(ld, mask_words).is_none());
        let staged = launch_geometry(ld, mask_words).expect("native endpoint fallback geometry");
        assert_eq!(
            (staged.a_words, staged.cap_groups, staged.dyn_smem),
            (8192, 1024, 69_632)
        );
        assert!(STATIC_SMEM + staged.dyn_smem <= SMEM_OPTIN_MAX);
    }

    #[test]
    fn batches_engage_from_the_threshold() {
        // n_p = (p + 1) / 4 of the batch's first row.
        assert!(!batch_engages(24_570, 6144));
        assert!(batch_engages(24_575, 6144));
        assert!(batch_engages(122_880, 6144));
        // Never inside the dense window, whatever the override.
        assert!(!batch_engages(2050, 513));
        assert!(batch_engages(2051, 513));
    }

    // ---- CPU model of the two select algorithms (kernel order, kernel passes) ----

    struct Kth {
        thr: u32,
        k_left: u32,
        eq_total: u32,
    }

    /// topk_row / radix_kth: three MSB-first passes, bins counted from the top,
    /// over the keys >= floor (at least k of them).
    fn radix_kth(keys: &[u32], k: u32, floor: u32) -> Kth {
        assert!(keys.iter().filter(|&&x| x >= floor).count() >= k as usize && k >= 1);
        let (mut prefix, mut pmask, mut k) = (0u32, 0u32, k);
        let mut eq_total = 0;
        for pass in 0..3 {
            let shift = [20u32, 10, 0][pass];
            let bins = if pass == 0 { 2048u32 } else { 1024 };
            let mut hist = vec![0u32; bins as usize];
            for &key in keys {
                assert_eq!(key >> 31, 0);
                if key >= floor && key & pmask == prefix {
                    hist[((key >> shift) & (bins - 1)) as usize] += 1;
                }
            }
            let mut above = 0u32;
            let mut bin = bins;
            while bin > 0 {
                bin -= 1;
                let c = hist[bin as usize];
                if above + c >= k {
                    break;
                }
                above += c;
            }
            prefix |= bin << shift;
            pmask |= (bins - 1) << shift;
            k -= above;
            eq_total = hist[bin as usize];
        }
        Kth {
            thr: prefix,
            k_left: k,
            eq_total,
        }
    }

    /// The mask the kernels write for `blocks` (block index, key) in ascending
    /// block order: keys > thr, plus the first k_left keys == thr.
    fn mask_from(blocks: &[(usize, u32)], s: &Kth, mask_words: usize) -> Vec<u32> {
        let mut out = vec![0u32; mask_words];
        let mut left = s.k_left;
        let take_all_eq = s.k_left >= s.eq_total;
        for &(b, key) in blocks {
            let sel = key > s.thr
                || (key == s.thr
                    && (take_all_eq || {
                        let t = left > 0;
                        left = left.saturating_sub(1);
                        t
                    }));
            if sel {
                out[b / 32] |= 1 << (b % 32);
            }
        }
        out
    }

    fn dense_mask(n: usize, mask_words: usize) -> Vec<u32> {
        (0..mask_words)
            .map(|w| {
                let lo = w * 32;
                if lo + 32 <= n {
                    u32::MAX
                } else if lo < n {
                    (1u32 << (n - lo)) - 1
                } else {
                    0
                }
            })
            .collect()
    }

    /// qwen4_qsa_prefill_topk[_global] for one row of n keys.
    fn old_select(keys: &[u32], mask_words: usize) -> Vec<u32> {
        let n = keys.len();
        if n <= BLOCK_TOPK {
            return dense_mask(n, mask_words);
        }
        let s = radix_kth(keys, BLOCK_TOPK as u32, 0);
        let blocks: Vec<(usize, u32)> = keys.iter().copied().enumerate().collect();
        mask_from(&blocks, &s, mask_words)
    }

    /// qwen4_qsa_prefill_score_gmax's maxima (keys below n only).
    fn group_maxima(keys: &[u32]) -> Vec<u32> {
        keys.chunks(GROUP)
            .map(|g| g.iter().copied().max().unwrap())
            .collect()
    }

    /// qwen4_qsa_prefill_topk_fused for one row: returns the mask and whether
    /// the row fell back to the full-row select.
    fn fused_select(keys: &[u32], mask_words: usize, cap_groups: usize) -> (Vec<u32>, bool) {
        let n = keys.len();
        if n <= BLOCK_TOPK {
            return (dense_mask(n, mask_words), false);
        }
        let gmax = group_maxima(keys);
        let ng = gmax.len();
        let (tau0, ncand) = if ng > BLOCK_TOPK {
            let g = radix_kth(&gmax, BLOCK_TOPK as u32, 0);
            (g.thr, (BLOCK_TOPK as u32 - g.k_left + g.eq_total) as usize)
        } else {
            (0, ng)
        };
        if ncand > cap_groups {
            return (old_select(keys, mask_words), true);
        }
        let cand: Vec<usize> = (0..ng).filter(|&g| gmax[g] >= tau0).collect();
        assert_eq!(cand.len(), ncand, "compaction count");
        // Candidate keys in ascending block order; the last group may be partial.
        let blocks: Vec<(usize, u32)> = cand
            .iter()
            .flat_map(|&g| (g * GROUP..((g + 1) * GROUP).min(n)).map(|b| (b, keys[b])))
            .collect();
        let m = ncand * GROUP
            - if cand[ncand - 1] == ng - 1 {
                ng * GROUP - n
            } else {
                0
            };
        assert_eq!(blocks.len(), m);
        let cand_keys: Vec<u32> = blocks.iter().map(|&(_, k)| k).collect();
        // The kernel counts only candidate keys >= tau0; the unfiltered select agrees.
        let s = radix_kth(&cand_keys, BLOCK_TOPK as u32, tau0);
        let u = radix_kth(&cand_keys, BLOCK_TOPK as u32, 0);
        assert_eq!((s.thr, s.k_left, s.eq_total), (u.thr, u.k_left, u.eq_total));
        (mask_from(&blocks, &s, mask_words), false)
    }

    /// Sort-based reference of the selection rule (score desc, then lower block).
    fn reference_select(keys: &[u32], mask_words: usize) -> Vec<u32> {
        let n = keys.len();
        if n <= BLOCK_TOPK {
            return dense_mask(n, mask_words);
        }
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| keys[b].cmp(&keys[a]).then(a.cmp(&b)));
        let mut out = vec![0u32; mask_words];
        for &b in &order[..BLOCK_TOPK] {
            out[b / 32] |= 1 << (b % 32);
        }
        out
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            // splitmix64
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    /// Positive fp32 score bits (what fabsf(sum of relus * scale) produces).
    fn score_key(x: f32) -> u32 {
        x.abs().to_bits() & 0x7fff_ffff
    }

    #[derive(Clone, Copy, Debug)]
    enum Kind {
        Continuous,
        FewValues,
        AllEqual,
        MostlyZero,
        Replicated,
        RecencyBiased,
    }

    fn row(rng: &mut Rng, kind: Kind, n: usize) -> Vec<u32> {
        match kind {
            Kind::Continuous => (0..n)
                .map(|_| score_key((rng.below(1 << 24) as f32) / (1u32 << 20) as f32))
                .collect(),
            Kind::FewValues => (0..n)
                .map(|_| score_key(rng.below(40) as f32 * 0.25))
                .collect(),
            Kind::AllEqual => vec![score_key(1.5); n],
            Kind::MostlyZero => (0..n)
                .map(|_| {
                    if rng.below(100) < 97 {
                        0
                    } else {
                        score_key(rng.below(1000) as f32 / 7.0)
                    }
                })
                .collect(),
            Kind::Replicated => {
                // A 1,000-block motif repeated: every score appears n/1000 times.
                let motif: Vec<u32> = (0..1000)
                    .map(|_| score_key(rng.below(1 << 20) as f32 / 4096.0))
                    .collect();
                (0..n).map(|b| motif[b % 1000]).collect()
            }
            Kind::RecencyBiased => (0..n)
                .map(|b| {
                    let base = rng.below(1 << 16) as f32 / 65536.0;
                    let near = if b + 2048 >= n || b < 4 { 2.0 } else { 0.0 };
                    score_key(base + near)
                })
                .collect(),
        }
    }

    #[test]
    fn radix_model_matches_the_sorted_reference() {
        let mut rng = Rng(7);
        for kind in [
            Kind::Continuous,
            Kind::FewValues,
            Kind::AllEqual,
            Kind::MostlyZero,
        ] {
            for n in [513usize, 777, 4096, 9001] {
                let keys = row(&mut rng, kind, n);
                let mw = (n + 1).div_ceil(32) + 3;
                assert_eq!(
                    old_select(&keys, mw),
                    reference_select(&keys, mw),
                    "{kind:?} n={n}"
                );
            }
        }
    }

    #[test]
    fn fused_select_is_byte_identical_to_the_full_row_select() {
        let mut rng = Rng(0x5eed);
        let kinds = [
            Kind::Continuous,
            Kind::FewValues,
            Kind::AllEqual,
            Kind::MostlyZero,
            Kind::Replicated,
            Kind::RecencyBiased,
        ];
        let mut fused_rows = 0;
        let mut fallback_rows = 0;
        for kind in kinds {
            for n in [
                513usize, 520, 4095, 4096, 4097, 6143, 8192, 12_345, 22_529, 32_768, 32_769,
                33_791, 33_792, 33_793, 65_520, 65_535,
            ] {
                for cap in [MIN_CAP_GROUPS, 682, MAX_CAP_GROUPS, 0] {
                    let keys = row(&mut rng, kind, n);
                    let mw = (n + 1).div_ceil(32) + 2;
                    let (fused, fell_back) = fused_select(&keys, mw, cap);
                    assert_eq!(fused, old_select(&keys, mw), "{kind:?} n={n} cap={cap}");
                    assert_eq!(
                        fused.iter().map(|w| w.count_ones()).sum::<u32>(),
                        n.min(BLOCK_TOPK) as u32
                    );
                    if fell_back {
                        fallback_rows += 1;
                    } else {
                        fused_rows += 1;
                    }
                }
            }
        }
        // Both paths are exercised: ties/zeros overflow the cap, continuous rows do not.
        assert!(
            fused_rows > 100 && fallback_rows > 50,
            "{fused_rows}/{fallback_rows}"
        );
    }

    #[test]
    fn continuous_rows_need_512_candidate_groups() {
        // Without ties among group maxima the candidates are exactly 512 groups.
        let mut rng = Rng(42);
        for n in [8192usize, 32_768] {
            let keys = row(&mut rng, Kind::Continuous, n);
            let g = radix_kth(&group_maxima(&keys), BLOCK_TOPK as u32, 0);
            assert_eq!(BLOCK_TOPK as u32 - g.k_left + g.eq_total, 512, "n={n}");
            let (fused, fell_back) = fused_select(&keys, n / 32 + 1, 682);
            assert!(!fell_back);
            assert_eq!(fused, reference_select(&keys, n / 32 + 1));
        }
    }
}
