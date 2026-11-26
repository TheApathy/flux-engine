// SPDX-License-Identifier: AGPL-3.0-only

//! Per-row KV headroom under a shared pool (`--kv-pool-tokens`).
//!
//! A shared pool can be smaller than `max_batch_size x max_seq_len`. Depth
//! aware admission keeps the in-flight reservations inside it, but an
//! operator watermark (`ATLAS_KV_ADMIT_WATERMARK`) may overcommit on purpose,
//! and then a row can find the pool empty mid-decode. Inside a batched step
//! that error fails every row. Before each decode tick the scheduler instead
//! reserves each row's next KV positions itself (exactly what the step would
//! allocate). When the pool cannot grow a row, it evicts the least-progress
//! *other* active sequence first (fewest emitted tokens, tie-breaking by
//! `seq_len` descending, then newest) with `finish_reason: length` and retries
//! growing the failing row; only if no other active sequence remains does the
//! failing row itself finish with `finish_reason: length`.
//!
//! Without `--kv-pool-tokens` the tick is never touched (`row_headroom` is
//! `None`), whatever the pool's shape.

use super::*;

/// Set at model load from `--kv-pool-tokens`. The per-row reservation runs
/// only when the operator asked for a shared pool, so every per-slot and
/// high-speed-swap serve keeps the rc7 tick even if its pool shape matches.
static SHARED_POOL_REQUESTED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub(crate) fn configure(kv_pool_tokens: Option<usize>) {
    SHARED_POOL_REQUESTED.store(
        kv_pool_tokens.is_some(),
        std::sync::atomic::Ordering::SeqCst,
    );
}

/// The tick's per-row reservation width, or `None` to leave the tick
/// unchanged: only with `--kv-pool-tokens` and a pool that is actually
/// smaller than every slot at `max_seq_len`.
pub(super) fn row_headroom(
    total_blocks: usize,
    max_batch_size: usize,
    max_seq_len: usize,
    block_size: usize,
    num_drafts: usize,
) -> Option<usize> {
    if total_blocks > 0 && block_size > 0 {
        crate::metrics::compat_metrics::set_cache_config(
            block_size,
            total_blocks,
            total_blocks.saturating_mul(block_size),
        );
        crate::metrics::compat_metrics::set_kv_blocks_free(total_blocks);
    }
    row_headroom_with(
        SHARED_POOL_REQUESTED.load(std::sync::atomic::Ordering::SeqCst),
        total_blocks,
        max_batch_size,
        max_seq_len,
        block_size,
        num_drafts,
    )
}

fn row_headroom_with(
    requested: bool,
    total_blocks: usize,
    max_batch_size: usize,
    max_seq_len: usize,
    block_size: usize,
    num_drafts: usize,
) -> Option<usize> {
    (requested && shared_pool_active(total_blocks, max_batch_size, max_seq_len, block_size))
        .then(|| rows_ahead(num_drafts))
}

/// True when the pool cannot hold every slot at `max_seq_len`, which only a
/// shared pool (`--kv-pool-tokens`) or a high-speed-swap window allows. The
/// per-slot pool refuses to start in that shape, so this is false for it.
pub(super) fn shared_pool_active(
    total_blocks: usize,
    max_batch_size: usize,
    max_seq_len: usize,
    block_size: usize,
) -> bool {
    total_blocks > 0
        && max_seq_len > 0
        && total_blocks < max_batch_size.saturating_mul(max_seq_len.div_ceil(block_size.max(1)))
}

/// KV positions one tick may write per row: the decode row plus the drafts a
/// verify frame writes before rollback.
pub(super) fn rows_ahead(num_drafts: usize) -> usize {
    num_drafts.saturating_add(1)
}

/// Remove every row `reserve` refuses, keeping the survivors' order. Each
/// refused row goes to `on_refused` at once, before the next row reserves,
/// so blocks it releases are available to the rows after it: when several
/// rows need the last free block in the same tick, only the first finishes.
#[cfg(test)]
pub(super) fn retain_rows_with_headroom<T>(
    rows: &mut Vec<T>,
    mut reserve: impl FnMut(&mut T) -> Result<()>,
    mut on_refused: impl FnMut(T, anyhow::Error),
) {
    let mut i = 0;
    while i < rows.len() {
        match reserve(&mut rows[i]) {
            Ok(()) => i += 1,
            Err(e) => on_refused(rows.remove(i), e),
        }
    }
}

/// Reserve headroom for every row, evicting the least-progress victim first
/// (minimum `victim_key(row)`, tie-breaking on newest / highest index) when a
/// row's reservation fails so the oldest sequence closest to completion is
/// preserved instead of killed first. Only if no other row remains does the
/// failing row itself get evicted.
pub(super) fn retain_rows_with_victim_headroom<T, K: Ord>(
    rows: &mut Vec<T>,
    mut reserve: impl FnMut(&mut T) -> Result<()>,
    mut victim_key: impl FnMut(&T) -> K,
    mut on_evicted: impl FnMut(T, anyhow::Error),
) {
    let mut i = 0;
    while i < rows.len() {
        match reserve(&mut rows[i]) {
            Ok(()) => i += 1,
            Err(e) => {
                // Pick a victim != i with minimum victim_key (tie-break: highest index).
                let victim = (0..rows.len())
                    .filter(|&idx| idx != i)
                    .min_by_key(|&idx| (victim_key(&rows[idx]), std::cmp::Reverse(idx)));
                if let Some(v) = victim {
                    let evicted = rows.remove(v);
                    if v < i {
                        i -= 1;
                    }
                    on_evicted(evicted, e);
                    // Retry rows[i] now that the victim's blocks were released.
                } else {
                    let evicted = rows.remove(i);
                    on_evicted(evicted, e);
                }
            }
        }
    }
}

/// Reserve `rows` KV positions for every active row. When the shared pool
/// cannot grow a row, evict the least-progress other active sequence first
/// (fewest emitted tokens, tie-breaking by `seq_len` descending to free the
/// most blocks, then newest) and retry so the oldest request closest to
/// completion finishes cleanly; only if no other sequence remains does the
/// failing row itself finish with `length`.
pub(super) fn finish_rows_without_kv_headroom(
    model: &dyn Model,
    active: &mut Vec<ActiveSeq>,
    rows: usize,
) {
    retain_rows_with_victim_headroom(
        active,
        |a| model.reserve_decode_kv(&mut a.seq, rows),
        |a| (a.output_tokens.len(), std::cmp::Reverse(a.seq.seq_len)),
        |mut a, e| {
            crate::metrics::compat_metrics::inc_preemptions();
            tracing::warn!(
                "KV_POOL_EXHAUSTED x_flux_kv_exhausted=true finish_reason=length seq_len={} \
                 blocks={} output_tokens={} rows_ahead={rows} free_blocks={}: the shared \
                 --kv-pool-tokens pool cannot grow an active request; evicting least-progress \
                 sequence: {e:#}",
                a.seq.seq_len,
                a.seq.block_table.len(),
                a.output_tokens.len(),
                model.num_free_blocks(),
            );
            // Not a reusable prefix: the request stopped at a capacity boundary.
            finish_sequence_with_cache(model, &mut a, false);
        },
    );
    crate::metrics::compat_metrics::set_kv_blocks_free(model.num_free_blocks());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn per_slot_pools_are_never_shared() {
        // rc7 256k C1: 16,386 blocks for 16,384 per sequence.
        assert!(!shared_pool_active(16_386, 1, 262_144, 16));
        // Per-slot C4 at 512K: 4 x 32,769 + 1.
        assert!(!shared_pool_active(4 * 32_769 + 1, 4, 524_288, 16));
        // Backends with no paged pool.
        assert!(!shared_pool_active(0, 4, 524_288, 16));
        assert!(!shared_pool_active(100, 4, 0, 16));
    }

    #[test]
    fn a_pool_for_two_deep_sessions_at_c4_is_shared() {
        let blocks = 2 * 524_288 / 16 + 4 + 1;
        assert!(shared_pool_active(blocks, 4, 524_288, 16));
    }

    #[test]
    fn reservation_needs_the_operator_flag_not_just_a_small_pool() {
        let shared = 2 * 524_288 / 16 + 4 + 1;
        assert_eq!(row_headroom_with(true, shared, 4, 524_288, 16, 4), Some(5));
        // A high-speed-swap or any other small pool without --kv-pool-tokens.
        assert_eq!(row_headroom_with(false, shared, 4, 524_288, 16, 4), None);
        assert_eq!(
            row_headroom_with(false, 4 * 65 + 1, 4, 524_288, 16, 0),
            None
        );
        // The flag with a pool that holds every slot changes nothing either.
        assert_eq!(row_headroom_with(true, 16_386, 1, 262_144, 16, 4), None);
    }

    #[test]
    fn rows_ahead_covers_the_verify_frame() {
        assert_eq!(rows_ahead(0), 1);
        assert_eq!(rows_ahead(4), 5);
    }

    #[test]
    fn only_refused_rows_are_removed_in_order() {
        let mut rows = vec![10, 11, 12, 13, 14];
        let mut failed = Vec::new();
        retain_rows_with_headroom(
            &mut rows,
            |r| {
                if *r % 2 == 1 {
                    anyhow::bail!("row {r} out of blocks")
                }
                *r += 100;
                Ok(())
            },
            |r, e| failed.push((r, e.to_string())),
        );
        assert_eq!(rows, vec![110, 112, 114]);
        assert_eq!(
            failed.iter().map(|(r, _)| *r).collect::<Vec<_>>(),
            vec![11, 13]
        );
        assert!(failed[0].1.contains("row 11"));
    }

    #[test]
    fn a_refused_row_releases_its_blocks_before_the_next_row_reserves() {
        // Two rows each need the pool's last free block in the same tick:
        // the first is refused and frees its 3 blocks, so the second fits.
        let free = std::cell::Cell::new(0usize);
        let mut rows = vec![("a", 3usize), ("b", 3usize)];
        let mut refused = Vec::new();
        retain_rows_with_headroom(
            &mut rows,
            |r| {
                if free.get() == 0 {
                    anyhow::bail!("{} out of blocks", r.0)
                }
                free.set(free.get() - 1);
                r.1 += 1;
                Ok(())
            },
            |r, _| {
                free.set(free.get() + r.1);
                refused.push(r.0);
            },
        );
        assert_eq!(refused, vec!["a"]);
        assert_eq!(rows, vec![("b", 4)]);
        assert_eq!(free.get(), 2);
    }

    #[test]
    fn no_failure_leaves_the_batch_untouched() {
        let mut rows = vec![1, 2, 3];
        let mut failed = 0;
        retain_rows_with_headroom(&mut rows, |_| Ok(()), |_, _| failed += 1);
        assert_eq!((failed, rows.clone()), (0, vec![1, 2, 3]));
        retain_rows_with_headroom(
            &mut rows,
            |_| anyhow::bail!("empty pool"),
            |_, _| failed += 1,
        );
        assert_eq!(failed, 3);
        assert!(rows.is_empty());
    }

    #[test]
    fn victim_headroom_evicts_least_progress_newest_row_and_retries_oldest() {
        // Row "old" has 1000 emitted tokens and crosses a block boundary when
        // free == 0; row "mid" has 500 tokens; row "new" has 50 tokens and is
        // mid-block. Evicting "new" frees its blocks so "old" succeeds on retry.
        let free = std::cell::Cell::new(0usize);
        let mut rows = vec![("old", 1000usize, 4usize), ("mid", 500, 3), ("new", 50, 2)];
        let mut evicted = Vec::new();
        retain_rows_with_victim_headroom(
            &mut rows,
            |r| {
                if r.0 == "old" {
                    if free.get() == 0 {
                        anyhow::bail!("old out of blocks")
                    }
                    free.set(free.get() - 1);
                    r.2 += 1;
                }
                Ok(())
            },
            |r| r.1,
            |r, _| {
                free.set(free.get() + r.2);
                evicted.push(r.0);
            },
        );
        assert_eq!(evicted, vec!["new"]);
        assert_eq!(rows, vec![("old", 1000, 5), ("mid", 500, 3)]);
        assert_eq!(free.get(), 1);
    }

    #[test]
    fn victim_headroom_tiebreaks_newest_first_and_evicts_sole_row_when_alone() {
        let free = std::cell::Cell::new(0usize);
        let mut rows = vec![("first", 10usize, 1usize), ("second", 10, 1)];
        let mut evicted = Vec::new();
        retain_rows_with_victim_headroom(
            &mut rows,
            |r| {
                if free.get() == 0 {
                    anyhow::bail!("{} out of blocks", r.0)
                }
                free.set(free.get() - 1);
                Ok(())
            },
            |r| r.1,
            |r, _| {
                free.set(free.get() + r.2);
                evicted.push(r.0);
            },
        );
        assert_eq!(evicted, vec!["second"]);
        assert_eq!(rows, vec![("first", 10, 1)]);

        // Sole remaining row failing with no other victim gets evicted itself.
        let mut sole = vec![("only", 100usize, 2usize)];
        let mut sole_evicted = Vec::new();
        retain_rows_with_victim_headroom(
            &mut sole,
            |_| anyhow::bail!("empty"),
            |r| r.1,
            |r, _| sole_evicted.push(r.0),
        );
        assert_eq!(sole_evicted, vec!["only"]);
        assert!(sole.is_empty());
    }

    #[test]
    fn victim_headroom_tiebreaks_equal_output_by_seq_len_descending() {
        let free = std::cell::Cell::new(0usize);
        // (name, output_tokens, seq_len, blocks)
        let mut rows = vec![
            ("leader", 500usize, 4_000usize, 250usize),
            ("small_tie", 10, 1_000, 64),
            ("large_tie", 10, 8_000, 500),
        ];
        let mut evicted = Vec::new();
        retain_rows_with_victim_headroom(
            &mut rows,
            |r| {
                if r.0 == "leader" {
                    if free.get() == 0 {
                        anyhow::bail!("leader out of blocks")
                    }
                    free.set(free.get() - 1);
                }
                Ok(())
            },
            |r| (r.1, std::cmp::Reverse(r.2)),
            |r, _| {
                free.set(free.get() + r.3);
                evicted.push(r.0);
            },
        );
        assert_eq!(
            evicted,
            vec!["large_tie"],
            "equal output progress must evict the longer seq_len first to free the most KV blocks"
        );
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].0, "leader");
        assert_eq!(rows[1].0, "small_tie");
    }
}
