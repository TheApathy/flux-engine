// SPDX-License-Identifier: AGPL-3.0-only

//! Fused Flash-Next PLE prefill join, `ATLAS_QWEN4_PLE_PREFILL_FUSED`
//! (default on; set to `0` to disable; read once when prefill scratch is allocated).
//!
//! Off, the join after the two cuBLASLt projections is
//! `qwen4_ple_prepare_prefill` (writes `gated_value` and `conv_input`, two
//! fp32 `[M, 4H]` matrices), `qwen4_ple_conv_inject_prefill` (reads them
//! back) and `qwen4_ple_commit_prefill_state`; the block pipeline adds a
//! history commit per block. On, it is three kernels that never materialise
//! the fp32 matrices (`kernels/gb10/common/qwen4_hyper.cu`):
//!
//! - `qwen4_ple_gate_prefill`: prepare's two per-(token, stream) scalars,
//!   the sigmoid gate `g` and the conv-norm `r`, with prepare's exact fp32
//!   reduction tree (a 256-thread block emulates the 1024-thread one; the
//!   tests below check the emulation map); 8 bytes per token and stream.
//! - `qwen4_ple_conv_inject_fused_prefill`: recomputes `g * value` and
//!   `g * value * r * (1 + norm_conv)` (the stored fp32 values, bit for bit)
//!   and does the causal conv + injection.
//! - `qwen4_ple_commit_fused_prefill_state`: the chunk-end state commit from
//!   the recomputed conv inputs.
//!
//! The cuBLASLt projections are untouched (same call, algorithm and operand
//! layout). The scratch loses `gated_value` and `conv_input`
//! (81,920 B per token: 1,280 MiB at 16,384 rows) and gains the gate scalars
//! (32 B per token).
//!
//! This file is dependency-free (std only) so `rustc --test` runs its tests.

use std::sync::OnceLock;

pub(crate) const SELECTOR: &str = "ATLAS_QWEN4_PLE_PREFILL_FUSED";
pub(crate) const MODULE: &str = "qwen4_hyper";
pub(crate) const GATE_KERNEL: &str = "qwen4_ple_gate_prefill";
pub(crate) const INJECT_KERNEL: &str = "qwen4_ple_conv_inject_fused_prefill";
pub(crate) const COMMIT_KERNEL: &str = "qwen4_ple_commit_fused_prefill_state";
pub(crate) const KERNELS: [&str; 3] = [GATE_KERNEL, INJECT_KERNEL, COMMIT_KERNEL];

/// Block of `qwen4_ple_gate_prefill` (`QWEN4_PLE_GATE_THREADS`).
pub(crate) const GATE_THREADS: u32 = 256;
/// Block of `qwen4_ple_prepare_prefill`, whose reduction tree the gate kernel
/// emulates (`QWEN4_PLE_PREPARE_THREADS`).
#[cfg(test)]
const PREPARE_THREADS: u32 = 1024;
/// Virtual threads per physical gate thread.
#[cfg(test)]
const VIRTUAL_PER_THREAD: u32 = PREPARE_THREADS / GATE_THREADS;
/// Block of the inject and commit kernels.
pub(crate) const INJECT_THREADS: u32 = 256;
/// Tokens per inject thread run (`QWEN4_PLE_INJECT_RUN` in the kernel).
pub(crate) const INJECT_RUN: u32 = 64;
/// Hyperconnection streams the kernels hard-code.
pub(crate) const STREAMS: usize = 4;
/// Gate scalars per (token, stream): g, r.
pub(crate) const GATE_FLOATS: usize = 2;
/// Conv taps (`PLE_CONV_HISTORY`).
pub(crate) const HISTORY: usize = 9;
/// Largest grid.y of a launch.
const MAX_GRID_Y: u32 = 65_535;

/// Strict parse: absent, empty or `1` = on (default on), `0`/`false`/`no` = off,
/// anything else refuses the load (a typo must not silently run the other path).
pub(crate) fn parse_selector(value: Option<&str>) -> Result<bool, String> {
    match value {
        None | Some("") | Some("1") => Ok(true),
        Some("0") | Some("false") | Some("no") => Ok(false),
        Some(other) => Err(format!("{SELECTOR} must be absent, 0 or 1; got {other:?}")),
    }
}

/// [`SELECTOR`], read once per process.
pub(crate) fn selected() -> Result<bool, String> {
    static CELL: OnceLock<Result<bool, String>> = OnceLock::new();
    CELL.get_or_init(|| parse_selector(std::env::var(SELECTOR).ok().as_deref()))
        .clone()
}

/// The kernels hard-code four streams and index channels in `u32`.
pub(crate) fn check_geometry(hidden_size: usize, hc_count: usize) -> Result<(), String> {
    if hc_count != STREAMS {
        return Err(format!(
            "hc_count={hc_count} (fused kernels need {STREAMS})"
        ));
    }
    if hidden_size == 0 {
        return Err("hidden_size=0".into());
    }
    let residual = hidden_size
        .checked_mul(STREAMS)
        .and_then(|r| r.checked_mul(HISTORY))
        .ok_or_else(|| format!("hidden_size={hidden_size} overflows"))?;
    if residual > u32::MAX as usize {
        return Err(format!(
            "hidden_size={hidden_size} exceeds the u32 channel index"
        ));
    }
    Ok(())
}

/// One buffer of the PLE prefill scratch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Buffer {
    Records,
    Scales,
    Embedding,
    Key,
    Value,
    GatedValue,
    ConvInput,
    Gate,
}

/// The prefill scratch, in allocation order, with its byte extents.
/// `record_bytes` per head record, `heads` records per token, `embed` the
/// PLE embedding width (= H), `residual_width` = 4H.
pub(crate) fn scratch_extents(
    max_tokens: usize,
    heads: usize,
    record_bytes: usize,
    embed: usize,
    residual_width: usize,
    fused: bool,
) -> Result<Vec<(Buffer, usize)>, String> {
    let bytes = |width: usize, element: usize| -> Result<usize, String> {
        max_tokens
            .checked_mul(width)
            .and_then(|value| value.checked_mul(element))
            .ok_or_else(|| "Qwen4 PLE prefill scratch extent overflow".to_string())
    };
    let f32_bytes = std::mem::size_of::<f32>();
    let mut extents = vec![
        (Buffer::Records, bytes(heads * record_bytes, 1)?),
        (Buffer::Scales, bytes(heads, f32_bytes)?),
        (Buffer::Embedding, bytes(embed, 2)?),
        (Buffer::Key, bytes(residual_width, 2)?),
        (Buffer::Value, bytes(embed, 2)?),
    ];
    if fused {
        extents.push((Buffer::Gate, bytes(STREAMS * GATE_FLOATS, f32_bytes)?));
    } else {
        extents.push((Buffer::GatedValue, bytes(residual_width, f32_bytes)?));
        extents.push((Buffer::ConvInput, bytes(residual_width, f32_bytes)?));
    }
    Ok(extents)
}

/// Total bytes of an extent list.
pub(crate) fn total_bytes(extents: &[(Buffer, usize)]) -> usize {
    extents.iter().map(|(_, bytes)| bytes).sum()
}

/// `qwen4_ple_gate_prefill` grid: one block per (token, stream).
pub(crate) fn gate_grid(rows: u32) -> Result<[u32; 3], String> {
    let blocks = rows
        .checked_mul(STREAMS as u32)
        .ok_or_else(|| format!("gate grid overflows at {rows} rows"))?;
    Ok([blocks, 1, 1])
}

/// `qwen4_ple_conv_inject_fused_prefill` grid: channel blocks x token runs.
pub(crate) fn inject_grid(rows: u32, residual_width: u32) -> Result<[u32; 3], String> {
    let runs = rows.div_ceil(INJECT_RUN);
    if runs > MAX_GRID_Y {
        return Err(format!(
            "inject grid.y {runs} exceeds {MAX_GRID_Y} at {rows} rows"
        ));
    }
    Ok([residual_width.div_ceil(INJECT_THREADS), runs, 1])
}

/// `qwen4_ple_commit_fused_prefill_state` grid: one thread per channel.
pub(crate) fn commit_grid(residual_width: u32) -> [u32; 3] {
    [residual_width.div_ceil(INJECT_THREADS), 1, 1]
}

/// Prepare's virtual thread that gate thread `tid` plays in slice `r`.
#[cfg(test)]
fn virtual_thread(tid: u32, r: u32) -> u32 {
    tid + r * GATE_THREADS
}

/// Load receipt when the fused join is armed.
pub(crate) fn engaged_line(max_tokens: usize, scratch_bytes: usize, saved_bytes: usize) -> String {
    format!(
        "QWEN4_PLE_PREFILL_FUSED_ENGAGED selector={SELECTOR} kernels={} max_tokens={max_tokens} \
         scratch_bytes={scratch_bytes} saved_bytes={saved_bytes} join=gate+conv_inject+commit \
         projections=unchanged",
        KERNELS.join(",")
    )
}

/// Load receipt when the selector is set but the old join runs.
pub(crate) fn refused_line(reason: &str) -> String {
    format!("QWEN4_PLE_PREFILL_FUSED_REFUSED selector={SELECTOR} reason={reason} join=prepare_conv")
}

/// First-use receipt per execution path (`sync`, `async`, `async_blocks`).
pub(crate) fn ran_line(path: &str) -> String {
    format!("QWEN4_PLE_PREFILL_FUSED_RAN selector={SELECTOR} path={path}")
}

#[cfg(test)]
mod tests {
    use super::*;

    const H: usize = 2560;

    #[test]
    fn selector_parses_strictly() {
        assert_eq!(parse_selector(None), Ok(true));
        assert_eq!(parse_selector(Some("")), Ok(true));
        assert_eq!(parse_selector(Some("0")), Ok(false));
        assert_eq!(parse_selector(Some("false")), Ok(false));
        assert_eq!(parse_selector(Some("no")), Ok(false));
        assert_eq!(parse_selector(Some("1")), Ok(true));
        for bad in ["2", "on", "true", " 1", "1 "] {
            assert!(parse_selector(Some(bad)).is_err(), "{bad:?} must refuse");
        }
    }

    #[test]
    fn geometry_needs_four_streams() {
        assert!(check_geometry(H, 4).is_ok());
        assert!(check_geometry(H, 2).is_err());
        assert!(check_geometry(0, 4).is_err());
        assert!(check_geometry(usize::MAX / 2, 4).is_err());
    }

    #[test]
    fn fused_scratch_drops_both_fp32_matrices() {
        let rows = 16_384;
        let old = scratch_extents(rows, 16, 90, H, 4 * H, false).unwrap();
        let new = scratch_extents(rows, 16, 90, H, 4 * H, true).unwrap();
        // Unfused scratch totals 114,144 B per token across all 7 buffers.
        assert_eq!(total_bytes(&old), rows * 114_144);
        assert_eq!(total_bytes(&new), rows * (114_144 - 81_920 + 32));
        assert_eq!(total_bytes(&old) - total_bytes(&new), 1_341_652_992);
        assert!(
            new.iter()
                .all(|(b, _)| *b != Buffer::GatedValue && *b != Buffer::ConvInput)
        );
        assert_eq!(new.last(), Some(&(Buffer::Gate, rows * 32)));
        // The five shared buffers keep their order and size.
        assert_eq!(old[..5], new[..5]);
        assert!(scratch_extents(usize::MAX / 8, 16, 90, H, 4 * H, true).is_err());
    }

    #[test]
    fn grids_cover_every_token_and_channel_once() {
        assert_eq!(gate_grid(16_384).unwrap(), [65_536, 1, 1]);
        assert!(gate_grid(u32::MAX).is_err());
        let residual = (4 * H) as u32;
        for rows in [1u32, 5, 9, 63, 64, 65, 1024, 8192, 16_384, 32_768] {
            let [x, y, z] = inject_grid(rows, residual).unwrap();
            assert_eq!(z, 1);
            assert!(x * INJECT_THREADS >= residual && (x - 1) * INJECT_THREADS < residual);
            // Runs [run*RUN, min(run*RUN+RUN, rows)) partition 0..rows.
            let mut seen = vec![0u8; rows as usize];
            for run in 0..y {
                let first = run * INJECT_RUN;
                assert!(first < rows, "an empty run would be launched");
                for t in first..(first + INJECT_RUN).min(rows) {
                    seen[t as usize] += 1;
                }
            }
            assert!(seen.iter().all(|&n| n == 1), "rows={rows}");
        }
        assert!(inject_grid(65_536 * INJECT_RUN, residual).is_err());
        assert_eq!(commit_grid(residual), [40, 1, 1]);
        // Each 256-channel inject block stays inside one stream (gate loads broadcast).
        assert_eq!(H % INJECT_THREADS as usize, 0);
    }

    /// Prepare's thread v owns h = v, v + 1024, v + 2048 (< H), in that order;
    /// the gate kernel's (tid, r) plays v = tid + 256 r with the same lane.
    #[test]
    fn virtual_threads_are_a_lane_preserving_bijection() {
        let mut owner = vec![None; PREPARE_THREADS as usize];
        for tid in 0..GATE_THREADS {
            for r in 0..VIRTUAL_PER_THREAD {
                let v = virtual_thread(tid, r);
                assert!(owner[v as usize].is_none());
                owner[v as usize] = Some((tid, r));
                assert_eq!(v % 32, tid % 32, "lane");
                assert_eq!(v / 32, tid / 32 + 8 * r, "virtual warp");
            }
        }
        assert!(owner.iter().all(Option::is_some));
        let mut covered = vec![0u8; H];
        for v in 0..PREPARE_THREADS as usize {
            let mut prev = None;
            let mut h = v;
            while h < H {
                covered[h] += 1;
                assert!(prev.is_none_or(|p| p < h));
                prev = Some(h);
                h += PREPARE_THREADS as usize;
            }
        }
        assert!(covered.iter().all(|&n| n == 1));
    }

    fn warp_sum(lanes: &mut [f32; 32]) {
        let mut d = 16;
        while d > 0 {
            let snapshot = *lanes;
            for (lane, value) in lanes.iter_mut().enumerate() {
                *value += snapshot[lane ^ d];
            }
            d >>= 1;
        }
    }

    /// prepare's tree: 1024 strided partials, 32 warp butterflies, warp 0's butterfly.
    fn prepare_tree(term: &dyn Fn(usize) -> f32, hidden: usize) -> f32 {
        let mut partial = vec![0.0f32; PREPARE_THREADS as usize];
        for (v, p) in partial.iter_mut().enumerate() {
            let mut h = v;
            while h < hidden {
                *p += term(h);
                h += PREPARE_THREADS as usize;
            }
        }
        let mut warp_parts = [0.0f32; 32];
        for (w, part) in warp_parts.iter_mut().enumerate() {
            let mut lanes = [0.0f32; 32];
            lanes.copy_from_slice(&partial[w * 32..w * 32 + 32]);
            warp_sum(&mut lanes);
            *part = lanes[0];
        }
        warp_sum(&mut warp_parts);
        warp_parts[0]
    }

    /// The gate kernel's tree: per (physical warp, r) butterflies stored at
    /// q_parts[warp + 8 r], then warp 0's butterfly over q_parts.
    fn gate_tree(term: &dyn Fn(usize) -> f32, hidden: usize) -> f32 {
        let warps = (GATE_THREADS / 32) as usize;
        let mut q_parts = [0.0f32; 32];
        for w in 0..warps {
            for r in 0..VIRTUAL_PER_THREAD {
                let mut lanes = [0.0f32; 32];
                for (lane, value) in lanes.iter_mut().enumerate() {
                    let tid = (w * 32 + lane) as u32;
                    let mut h = virtual_thread(tid, r) as usize;
                    while h < hidden {
                        *value += term(h);
                        h += PREPARE_THREADS as usize;
                    }
                }
                warp_sum(&mut lanes);
                q_parts[w + 8 * r as usize] = lanes[0];
            }
        }
        warp_sum(&mut q_parts);
        q_parts[0]
    }

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = (*seed >> 40) as u32;
        // Spread magnitudes over ~2^-20..2^4 and both signs: order-sensitive fp32 sums.
        let mantissa = (bits & 0xffff) as f32 / 65_536.0 + 0.5;
        let exponent = ((bits >> 16) & 0x1f) as i32 - 20;
        let sign = if bits & 0x200000 != 0 { -1.0 } else { 1.0 };
        sign * mantissa * 2f32.powi(exponent)
    }

    /// Rust f32 arithmetic is IEEE with no contraction, the fmad=false build's
    /// arithmetic: the two trees must give identical bits on order-sensitive data.
    #[test]
    fn gate_tree_equals_prepare_tree_bit_for_bit() {
        for (seed0, hidden) in [(1u64, H), (7, H), (42, 1000), (99, 1024), (5, 3000)] {
            let mut seed = seed0;
            let data: Vec<f32> = (0..hidden).map(|_| lcg(&mut seed)).collect();
            for term in [
                Box::new(|h: usize| data[h] * data[h]) as Box<dyn Fn(usize) -> f32>,
                Box::new(|h: usize| data[h] * data[(h * 7 + 3) % hidden]),
                Box::new(|h: usize| data[h]),
            ] {
                let a = prepare_tree(&*term, hidden);
                let b = gate_tree(&*term, hidden);
                assert_eq!(a.to_bits(), b.to_bits(), "seed={seed0} hidden={hidden}");
            }
        }
        // A sequential sum differs on this data (the test can tell trees apart).
        let mut seed = 1u64;
        let data: Vec<f32> = (0..H).map(|_| lcg(&mut seed)).collect();
        let serial: f32 = data.iter().fold(0.0, |acc, x| acc + x);
        assert_ne!(serial.to_bits(), prepare_tree(&|h| data[h], H).to_bits());
    }

    /// Where a conv tap of chunk token t (dilation d in {9, 6, 3}) comes from.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Tap {
        /// BF16 of the prepared input of chunk token k.
        Chunk(usize),
        /// Entry state slot j of the chunk's slot.
        Entry(usize),
        Zero,
    }

    /// qwen4_ple_conv_inject_prefill over the whole chunk.
    fn old_whole(t: usize, d: usize, reset: bool) -> Tap {
        if t >= d {
            Tap::Chunk(t - d)
        } else if reset {
            Tap::Zero
        } else {
            Tap::Entry(t + HISTORY - d)
        }
    }

    /// The pipeline's level-1 block path: block start > 0 reads a history the
    /// commit kernel wrote at num_tokens = start, reset 0.
    fn old_block(t: usize, d: usize, reset: bool, block: usize) -> Tap {
        let start = t / block * block;
        if start == 0 {
            return old_whole(t, d, reset);
        }
        let local = t - start;
        if local >= d {
            return Tap::Chunk(t - d);
        }
        // history[j] = bf16(conv_input[start - 9 + j]) when start - 9 + j >= 0.
        let j = local + HISTORY - d;
        let source = start as isize - HISTORY as isize + j as isize;
        assert!(source >= 0, "blocks are >= 9 tokens");
        Tap::Chunk(source as usize)
    }

    /// The fused inject's ring for a run starting at `first`.
    fn fused(t: usize, d: usize, reset: bool, first: usize) -> Tap {
        assert!(t >= first);
        let tap = t as isize - d as isize;
        if tap >= 0 {
            Tap::Chunk(tap as usize)
        } else if reset {
            Tap::Zero
        } else {
            Tap::Entry((tap + HISTORY as isize) as usize)
        }
    }

    #[test]
    fn fused_taps_equal_whole_and_block_taps() {
        for rows in [1usize, 5, 8, 9, 10, 63, 64, 65, 1023, 1024, 2048, 2500] {
            for reset in [false, true] {
                for t in 0..rows {
                    let run_first = t / INJECT_RUN as usize * INJECT_RUN as usize;
                    for d in [9usize, 6, 3] {
                        let whole = old_whole(t, d, reset);
                        assert_eq!(fused(t, d, reset, run_first), whole, "t={t} d={d}");
                        if rows >= 1024 {
                            assert_eq!(old_block(t, d, reset, 1024), whole, "block t={t} d={d}");
                        }
                    }
                }
            }
        }
    }

    /// The commit of chunk-end history: the last nine inputs, else the tail of
    /// the old history (or zeros under reset); both commit kernels share it.
    #[test]
    fn commit_sources_match_the_serial_history() {
        for rows in [1usize, 3, 8, 9, 10, 100] {
            for reset in [false, true] {
                let entry: Vec<i64> = (0..HISTORY as i64).map(|j| -(j + 1)).collect();
                let inputs: Vec<i64> = (0..rows as i64).map(|t| t + 1).collect();
                let mut serial = if reset {
                    vec![0; HISTORY]
                } else {
                    entry.clone()
                };
                for x in &inputs {
                    serial.remove(0);
                    serial.push(*x);
                }
                let committed: Vec<i64> = (0..HISTORY)
                    .map(|j| {
                        let source = rows as isize - HISTORY as isize + j as isize;
                        if source >= 0 {
                            inputs[source as usize]
                        } else if reset {
                            0
                        } else {
                            entry[rows + j]
                        }
                    })
                    .collect();
                assert_eq!(committed, serial, "rows={rows} reset={reset}");
            }
        }
    }

    #[test]
    fn receipts_name_the_selector_and_kernels() {
        let line = engaged_line(16_384, 524_288, 1_341_652_992);
        assert!(line.starts_with(
            "QWEN4_PLE_PREFILL_FUSED_ENGAGED selector=ATLAS_QWEN4_PLE_PREFILL_FUSED "
        ));
        for kernel in KERNELS {
            assert!(line.contains(kernel));
        }
        assert!(line.contains("saved_bytes=1341652992"));
        assert!(refused_line("kernel_missing").starts_with(
            "QWEN4_PLE_PREFILL_FUSED_REFUSED selector=ATLAS_QWEN4_PLE_PREFILL_FUSED reason=kernel_missing"
        ));
        assert_eq!(
            ran_line("async"),
            "QWEN4_PLE_PREFILL_FUSED_RAN selector=ATLAS_QWEN4_PLE_PREFILL_FUSED path=async"
        );
    }
}
