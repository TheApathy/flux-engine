// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Result, bail};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::PagedKvCache;

use crate::layer::LayerState;
use crate::traits::SequenceState;

/// Apply an `EvictedBlocks` result to the production cache and the HSS
/// orchestrator. Physical blocks return to the free list; disk-block IDs get
/// `dec_disk_ref`'d (Phase 6.1.e). When HSS isn't engaged the disk vec is
/// empty and this becomes a thin loop over the physical blocks.
pub(crate) fn apply_evicted_blocks(
    evicted: spark_runtime::prefix_cache::EvictedBlocks,
    kv_cache: &mut PagedKvCache,
) {
    for block in &evicted.physical {
        kv_cache.return_evicted_block(*block);
    }
    if !evicted.disk_block_ids.is_empty()
        && let Some(res) = spark_storage::with_local(|hss| {
            for id in &evicted.disk_block_ids {
                // dec_disk_ref returns the new refcount; discarded here.
                let _new_refcount = hss.dec_disk_ref(*id);
            }
            Ok(())
        })
        && let Err(e) = res
    {
        // Errors here are advisory — orchestrator absent shouldn't block
        // the cache eviction path. Log and continue.
        tracing::debug!("apply_evicted_blocks: spark_storage::with_local closure: {e:#}");
    }
}

/// Apply the disk-ref obligation reported by a `prefix_cache.insert*` call.
/// The cache returns the disk_block_ids on which it newly took an ownership
/// ref (a node was created OR an existing node had its disk_block_id
/// populated for the first time). The caller `inc_disk_ref`s each one so
/// the swap allocator's refcount matches the cache's reachability.
///
/// Without this, the cache stores disk_block_ids whose only live ref is
/// the sequence's; when `free_sequence` decs, the ID is reclaimed by the
/// swap allocator while the cache still references it — the next prefix
/// hit then trips `inc_disk_ref` on a freed ID and panics the scheduler
/// thread (Issue #17, panic at `high_speed_swap.rs:167`).
pub(crate) fn cache_acquires_disk_refs(newly_acquired: &[u32]) {
    if newly_acquired.is_empty() {
        return;
    }
    if let Some(res) = spark_storage::with_local(|hss| {
        for &id in newly_acquired {
            if id != u32::MAX {
                hss.inc_disk_ref(id);
            }
        }
        Ok(())
    }) && let Err(e) = res
    {
        tracing::debug!("cache_acquires_disk_refs: spark_storage::with_local: {e:#}");
    }
}

/// Phase 6.1.e: bump disk-side refcounts for blocks reused from a prefix-cache
/// hit, and push the disk_block_ids onto the sequence's history. The cache's
/// own ref keeps these slots alive across eviction; we add the seq's ref so
/// `free_sequence` can dec_disk_ref it on exit.
///
/// `matched_disk_block_ids` parallels `matched_blocks` when the entries were
/// inserted under HSS (every entry is a live disk_id). When HSS wasn't
/// engaged at insert time the slice is empty — the per-layer offload helper
/// will alloc fresh disk_ids and stream the data to disk on the first decode
/// step that touches each block.
pub(crate) fn reuse_prefix_match_disk_ids(
    matched_disk_block_ids: &[u32],
    seq_disk_block_ids: &mut Vec<u32>,
) {
    if matched_disk_block_ids.is_empty() {
        return;
    }
    if let Some(res) = spark_storage::with_local(|hss| {
        for &id in matched_disk_block_ids {
            if id == u32::MAX {
                // Mixed-mode entry — skip; the catch-up offload will populate.
                continue;
            }
            hss.inc_disk_ref(id);
            seq_disk_block_ids.push(id);
        }
        Ok(())
    }) && let Err(e) = res
    {
        tracing::debug!("reuse_prefix_match_disk_ids: spark_storage::with_local: {e:#}");
    }
}

/// Issue #31: validate that the block being evicted at logical position
/// `evict_pos` has been offloaded by every attention layer. The slide
/// is safe iff `disk_last_offloaded[L] > evict_pos` for all L (strictly
/// greater because `disk_last_offloaded[L]` is the count of offloaded
/// blocks, and a block at position N is "offloaded" iff the count is at
/// least N+1, i.e., > N).
///
/// Returns `Err` describing the first lagging layer if any layer hasn't
/// caught up, else `Ok(())`. Pure function — no side effects.
pub(crate) fn check_safe_to_evict(
    disk_last_offloaded_per_layer: &[u32],
    evict_pos: usize,
) -> Result<()> {
    for (layer_idx, &cursor) in disk_last_offloaded_per_layer.iter().enumerate() {
        if (cursor as usize) <= evict_pos {
            bail!(
                "high-speed-swap: attempting to evict block at logical position {} \
                 from HBM, but attention layer {} only offloaded up to position {}. \
                 Eviction would lose K/V data. Per-layer cursors: {:?}",
                evict_pos,
                layer_idx,
                cursor,
                disk_last_offloaded_per_layer,
            );
        }
    }
    Ok(())
}

/// Issue #31: after a successful slide advances `window_start` to
/// `new_window_start`, advance every attention layer's offload cursor
/// to keep pace. Layers whose cursor was already ≥ `new_window_start`
/// (e.g. they offloaded more recently in this chunk) are left alone.
/// Pure mutation on a `&mut [u32]`.
pub(crate) fn advance_layer_cursors_after_slide(
    disk_last_offloaded_per_layer: &mut [u32],
    new_window_start: usize,
) {
    let new_ws = new_window_start as u32;
    for cursor in disk_last_offloaded_per_layer.iter_mut() {
        if *cursor < new_ws {
            *cursor = new_ws;
        }
    }
}

/// Zero newly allocated KV blocks on `stream`, coalescing contiguous physical
/// block runs via [`PagedKvCache::zero_blocks`] when multiple blocks were
/// allocated and calling [`PagedKvCache::zero_block`] directly for a single
/// block (zero heap allocation).
fn zero_fresh_blocks(
    fresh: &[u32],
    kv_cache: &PagedKvCache,
    gpu: &dyn GpuBackend,
    stream: u64,
) -> Result<()> {
    match fresh {
        [] => Ok(()),
        [blk] => kv_cache.zero_block(*blk, gpu, stream),
        blocks => {
            let _ = kv_cache.zero_blocks(blocks, gpu, stream)?;
            Ok(())
        }
    }
}

/// Allocate a parallel HSS disk-block ID when `hss_active`, rolling back
/// `kv_cache.free_block(blk)` if disk-ID allocation fails.
fn alloc_hss_disk_id_or_rollback(
    hss_active: bool,
    blk: u32,
    kv_cache: &mut PagedKvCache,
) -> Result<Option<u32>> {
    if !hss_active {
        return Ok(None);
    }
    let res = spark_storage::with_local(|hss| {
        hss.alloc_disk_block_id().ok_or_else(|| {
            anyhow::anyhow!(
                "high-speed-swap: disk-block-id pool exhausted; \
                 increase --high-speed-swap-bytes or shorten --max-seq-len"
            )
        })
    })
    .ok_or_else(|| {
        anyhow::anyhow!(
            "high-speed-swap: orchestrator not installed but cache_blocks_per_seq is set"
        )
    });
    match res.and_then(|inner| inner) {
        Ok(id) => Ok(Some(id)),
        Err(e) => {
            kv_cache.free_block(blk);
            Err(e)
        }
    }
}

/// Zero `blk` (or record it into `fresh` for coalesced zeroing). On failure,
/// roll back both the physical block and any newly allocated HSS disk-block ID.
fn zero_or_stage_fresh_block(
    blk: u32,
    disk_id: Option<u32>,
    fresh: Option<&mut Vec<u32>>,
    kv_cache: &mut PagedKvCache,
    gpu: &dyn GpuBackend,
    stream: u64,
) -> Result<()> {
    match fresh {
        Some(list) => {
            list.push(blk);
            Ok(())
        }
        None => {
            if let Err(e) = kv_cache.zero_block(blk, gpu, stream) {
                kv_cache.free_block(blk);
                if let Some(id) = disk_id {
                    let _ = spark_storage::with_local(|hss| {
                        let _ = hss.dec_disk_ref(id);
                        Ok(())
                    });
                }
                return Err(e);
            }
            Ok(())
        }
    }
}

/// Sliding-window allocation helper (decode path).
///
/// Ensures `seq.physical_block_for(abs_block_idx)` is `Some` after this call
/// returns. With HSS off, this allocates any missing physical blocks and zeroes
/// them via [`zero_fresh_blocks`] (coalescing contiguous runs into one
/// `cuMemsetD8Async` per `(layer, side, run)`).
/// With HSS on (`cache_blocks_per_seq` set), it slides the rolling window
/// when at cap, allocates the new physical block (recycling the freed one),
/// zeroes it, and pushes a parallel disk_block_id onto `seq.disk_block_ids`.
///
/// Pre-condition (enforced at runtime via [`check_safe_to_evict`] before each
/// slide): every attention layer must have offloaded the evicted logical slot
/// `ws` (`disk_last_offloaded_per_layer[L] > ws` for all `L`), returning `Err`
/// if any layer lags.
pub(crate) fn ensure_blocks_through_decode(
    seq: &mut SequenceState,
    abs_block_idx: usize,
    kv_cache: &mut PagedKvCache,
    prefix_cache: &dyn spark_runtime::prefix_cache::PrefixCache,
    gpu: &dyn GpuBackend,
    stream: u64,
) -> Result<()> {
    if kv_cache.config().cache_blocks_per_seq.is_some() {
        return ensure_blocks_through_decode_inner(
            seq,
            abs_block_idx,
            kv_cache,
            prefix_cache,
            gpu,
            stream,
            None,
        );
    }
    let mut fresh: Vec<u32> = Vec::new();
    let grown = ensure_blocks_through_decode_inner(
        seq,
        abs_block_idx,
        kv_cache,
        prefix_cache,
        gpu,
        stream,
        Some(&mut fresh),
    );
    zero_fresh_blocks(&fresh, kv_cache, gpu, stream)?;
    grown
}

/// Block-table growth loop for [`ensure_blocks_through_decode`] and
/// [`reserve_decode_rows`]. `fresh = None` zeroes each block immediately via
/// [`PagedKvCache::zero_block`]; `Some(list)` records newly allocated physical
/// block IDs so the caller can issue one coalesced [`zero_fresh_blocks`] call.
fn ensure_blocks_through_decode_inner(
    seq: &mut SequenceState,
    abs_block_idx: usize,
    kv_cache: &mut PagedKvCache,
    prefix_cache: &dyn spark_runtime::prefix_cache::PrefixCache,
    gpu: &dyn GpuBackend,
    stream: u64,
    mut fresh: Option<&mut Vec<u32>>,
) -> Result<()> {
    let cap = kv_cache.config().cache_blocks_per_seq.map(|c| c as usize);
    // Loop invariant: each iter either slides (frees a block) or grows
    // block_table by one. Terminates when the highest needed logical block
    // is in-window.
    let mut slide_count = 0usize;
    let mut alloc_count = 0usize;
    loop {
        let ws = seq.hss_window_start();
        let bt_len = seq.block_table.len();
        // Highest in-window logical block (inclusive). If empty, treat as
        // "none yet" — keep the loop going until we've allocated.
        let in_window = bt_len > 0 && abs_block_idx < ws + bt_len;
        if in_window {
            if slide_count > 0 || alloc_count > 0 {
                tracing::trace!(
                    "ensure_blocks_through_decode: abs={} ws={} bt_len={} slid={} alloc'd={}",
                    abs_block_idx,
                    ws,
                    bt_len,
                    slide_count,
                    alloc_count
                );
            }
            return Ok(());
        }
        // We need to extend block_table by at least one. If at cap, slide.
        if let Some(c) = cap
            && bt_len >= c
        {
            // Issue #31: the safety condition for evicting block_table[0]
            // (logical position `ws`) is that EVERY attention layer has
            // already offloaded that block. The previous code asserted
            // the stricter (and overly-strict) condition
            // `disk_last_offloaded[L] == disk_block_ids.len()`, which
            // fails immediately after the first alloc within this loop
            // bumps `disk_block_ids.len()` past the cursors. The
            // assertion was `debug_assert!` only — release builds slid
            // past anyway, then `offload_layer_kv` bailed downstream.
            check_safe_to_evict(&seq.disk_last_offloaded_per_layer, ws).map_err(|e| {
                anyhow::anyhow!(
                    "{e} (decode path; disk_block_ids.len()={}, block_table.len()={}, \
                     slid={}, alloc'd={})",
                    seq.disk_block_ids.len(),
                    seq.block_table.len(),
                    slide_count,
                    alloc_count,
                )
            })?;
            let evicted = seq.block_table.remove(0);
            kv_cache.free_block(evicted);
            // After the slide, advance every layer cursor so the next
            // `offload_layer_kv` doesn't see `start < window_start`.
            // The blocks now outside the window were already on disk
            // (the safety check above guaranteed it).
            advance_layer_cursors_after_slide(&mut seq.disk_last_offloaded_per_layer, ws + 1);
            slide_count += 1;
            continue;
        }
        // F77 (2026-04-30): same try_alloc → evict prefix cache → retry
        // pattern as ensure_blocks_through_prefill. Without the
        // eviction fallback, multi-turn opencode sessions exhaust the
        // KV pool because every completed turn leaves prefix-cached
        // blocks alive — error observed live:
        // "alloc failed in ensure_blocks_through_decode: abs=590 ...
        //  free_blocks=0". The prefill helper already had this; the
        // decode helper diverged.
        let blk = match kv_cache.try_alloc_block() {
            Some(b) => b,
            None => {
                let evicted = prefix_cache.evict(1);
                apply_evicted_blocks(evicted, kv_cache);
                kv_cache.alloc_block().map_err(|e| {
                    anyhow::anyhow!(
                        "alloc failed in ensure_blocks_through_decode: abs={} ws={} bt_len={} \
                         cap={:?} free_blocks={} slid={} alloc'd={}: {}",
                        abs_block_idx,
                        ws,
                        bt_len,
                        cap,
                        kv_cache.num_free_blocks(),
                        slide_count,
                        alloc_count,
                        e
                    )
                })?
            }
        };
        let disk_id = alloc_hss_disk_id_or_rollback(cap.is_some(), blk, kv_cache)?;
        zero_or_stage_fresh_block(blk, disk_id, fresh.as_deref_mut(), kv_cache, gpu, stream)?;
        seq.block_table.push(blk);
        alloc_count += 1;
        if let Some(id) = disk_id {
            seq.disk_block_ids.push(id);
        }
    }
}

/// `--kv-pool-tokens` per-row headroom: make `seq` own KV blocks for positions
/// `seq_len .. seq_len + rows` before a batched step. These are exactly the
/// `ensure_blocks_through_decode` calls the step makes itself (one row for
/// decode, `num_drafts + 1` for a verify frame), only earlier and one row at a
/// time, so a row the shared pool cannot grow fails alone instead of failing
/// every row of the batch inside the forward. A row that fails keeps the
/// blocks it already owns; they return to the pool when the sequence is
/// freed. The pool itself never grows, so exhaustion cannot reach host memory.
/// No-op under high-speed swap, whose window owns its own sliding allocation.
pub(crate) fn reserve_decode_rows(
    seq: &mut SequenceState,
    rows: usize,
    kv_cache: &mut PagedKvCache,
    prefix_cache: &dyn spark_runtime::prefix_cache::PrefixCache,
    gpu: &dyn GpuBackend,
    stream: u64,
) -> Result<()> {
    if rows == 0 || kv_cache.config().cache_blocks_per_seq.is_some() {
        return Ok(());
    }
    let block_size = kv_cache.block_size().max(1);
    let last = seq
        .seq_len
        .checked_add(rows - 1)
        .ok_or_else(|| anyhow::anyhow!("KV headroom position overflow"))?;
    let mut fresh: Vec<u32> = Vec::new();
    let mut grown = Ok(());
    for block in seq.seq_len / block_size..=last / block_size {
        if let Err(err) = ensure_blocks_through_decode_inner(
            seq,
            block,
            kv_cache,
            prefix_cache,
            gpu,
            stream,
            Some(&mut fresh),
        ) {
            grown = Err(err);
            break;
        }
    }
    zero_fresh_blocks(&fresh, kv_cache, gpu, stream)?;
    grown
}

/// Monotonic (non-sliding) KV block allocation helper for the prefill path.
///
/// Grows `seq.block_table` (and parallel `seq.disk_block_ids` under HSS) up to
/// `abs_block_idx` without sliding the HBM window, with LRU prefix-cache
/// eviction fallback on pool exhaustion and optional
/// `ATLAS_PREFILL_KV_ZERO_COALESCE` coalesced zeroing.
///
/// Issue #31 (2026-05-08): the previous version of this helper slid the
/// HBM window forward when `block_table.len() >= cap` to make room for
/// new prefill allocs. That was wrong — slid-out blocks have their K/V
/// on disk via the per-layer offload, but the attention KERNEL during
/// prefill reads K/V from `block_table[bt_idx]` only (HBM), and the
/// orchestrator-fed disk-read path is wired up for DECODE attention
/// only (Phase 6.2.a), not for prefill (Phase 6.2.b deferred). So a
/// prefill that slid produced silently-wrong attention output for any
/// position outside the post-slide window. The original author's design
/// comment in `qwen3_attention/trait_impl/prefill_inner.rs:138-145`
/// states the intended invariant: "Sliding-window eviction is NOT
/// triggered here — prefill grows HBM monotonically; the cap kicks in
/// during decode." This helper now enforces that invariant by never
/// sliding during prefill. Cap-bound HBM is restored once the first
/// `ensure_blocks_through_decode` call hits the `bt_len >= cap` branch
/// (where attention reads through the orchestrator's tiled path, so
/// slides are correctness-safe).
pub(crate) fn ensure_blocks_through_prefill(
    seq: &mut SequenceState,
    abs_block_idx: usize,
    kv_cache: &mut PagedKvCache,
    prefix_cache: &dyn spark_runtime::prefix_cache::PrefixCache,
    gpu: &dyn GpuBackend,
    stream: u64,
) -> Result<()> {
    use super::host_bucket::{self, Lever};
    let coalesce = host_bucket::armed(Lever::KvZeroCoalesce).map_err(anyhow::Error::msg)?;
    if !coalesce {
        return grow_prefill_block_table(
            seq,
            abs_block_idx,
            kv_cache,
            prefix_cache,
            gpu,
            stream,
            None,
        );
    }
    // ATLAS_PREFILL_KV_ZERO_COALESCE (K8 item A): collect the fresh blocks, then zero
    // them with one memset per (layer, side, run). Same bytes, same value, same
    // stream, issued before this function returns, i.e. before the chunk's metadata
    // uploads and layers -- the device state the forward sees is unchanged. A
    // part-way failure still zeroes what it allocated, as the per-block path had.
    let mut fresh: Vec<u32> = Vec::new();
    let grown = grow_prefill_block_table(
        seq,
        abs_block_idx,
        kv_cache,
        prefix_cache,
        gpu,
        stream,
        Some(&mut fresh),
    );
    if !fresh.is_empty() {
        let (runs, memsets) = kv_cache.zero_blocks(&fresh, gpu, stream)?;
        if host_bucket::first_time("KV_ZERO_COALESCE_ENGAGED") {
            tracing::info!(
                "KV_ZERO_COALESCE_ENGAGED selector={} blocks={} runs={runs} memsets={memsets} per_block_memsets={}",
                Lever::KvZeroCoalesce.selector(),
                fresh.len(),
                2 * kv_cache.num_layers() * fresh.len(),
            );
        }
    }
    grown
}

/// The block-table growth loop of [`ensure_blocks_through_prefill`]. `fresh = None`
/// zeroes each new block on allocation (`zero_block`, two memsets per layer);
/// `Some(list)` records it for one coalesced `zero_blocks` call by the caller.
fn grow_prefill_block_table(
    seq: &mut SequenceState,
    abs_block_idx: usize,
    kv_cache: &mut PagedKvCache,
    prefix_cache: &dyn spark_runtime::prefix_cache::PrefixCache,
    gpu: &dyn GpuBackend,
    stream: u64,
    mut fresh: Option<&mut Vec<u32>>,
) -> Result<()> {
    let cap = kv_cache.config().cache_blocks_per_seq.map(|c| c as usize);
    loop {
        let ws = seq.hss_window_start();
        let bt_len = seq.block_table.len();
        let in_window = bt_len > 0 && abs_block_idx < ws + bt_len;
        if in_window {
            return Ok(());
        }
        // Issue #31: NEVER slide during prefill. block_table grows
        // monotonically until the chunk's full token range is in-window.
        // HBM headroom is preserved by the existing prefix-cache eviction
        // fallback below when `try_alloc_block` returns None. The cap
        // (when HSS is engaged) is enforced lazily on the first decode
        // step, where attention reads through the orchestrator's tiled
        // path and slides are correctness-safe.

        // Try alloc; on failure, evict prefix-cache entries and retry once.
        let blk = match kv_cache.try_alloc_block() {
            Some(b) => b,
            None => {
                // Ask the prefix cache to free a block via LRU eviction.
                let evicted = prefix_cache.evict(1);
                apply_evicted_blocks(evicted, kv_cache);
                kv_cache.alloc_block()?
            }
        };
        let disk_id = alloc_hss_disk_id_or_rollback(cap.is_some(), blk, kv_cache)?;
        zero_or_stage_fresh_block(blk, disk_id, fresh.as_deref_mut(), kv_cache, gpu, stream)?;
        seq.block_table.push(blk);
        if let Some(id) = disk_id {
            seq.disk_block_ids.push(id);
        }
    }
}

/// Extract mutable references to a single layer's state across N sequences.
///
/// The explicit lifetime `'a` ties the returned refs to the borrow of `all`,
/// so the compiler knows the borrow is released when the returned Vec is dropped.
/// Uses a for loop instead of iter_mut().map() because FnMut closures cannot
/// express that returned references outlive the closure invocation.
pub(crate) fn extract_layer_refs<'a>(
    all: &'a mut [Vec<Box<dyn LayerState>>],
    layer_idx: usize,
) -> Vec<&'a mut (dyn LayerState + 'static)> {
    let mut refs = Vec::with_capacity(all.len());
    for seq_states in all.iter_mut() {
        refs.push(seq_states[layer_idx].as_mut());
    }
    refs
}

#[cfg(test)]
#[path = "block_mgmt_pool_tests.rs"]
mod pool_tests;

#[cfg(test)]
mod tests {
    use super::*;

    // Issue #31 — `check_safe_to_evict` enforces the slide invariant.

    #[test]
    fn safe_to_evict_when_all_layers_caught_up() {
        // Every layer has offloaded position 5 (cursor=6 means 0..6 are on disk),
        // evicting position 5 is safe because cursor > 5 for every layer.
        let cursors = vec![6, 6, 6];
        assert!(check_safe_to_evict(&cursors, 5).is_ok());
    }

    #[test]
    fn unsafe_to_evict_when_a_layer_lags() {
        // Layer 1 has only offloaded up to position 4 (cursor=5 = positions 0..5
        // means position 4 is the last offloaded; cursor=5 means the LIMIT is 5,
        // i.e. cursor > 5 means position 5 is offloaded). Strict-greater comparison.
        let cursors = vec![10, 5, 10];
        let err = check_safe_to_evict(&cursors, 5).unwrap_err().to_string();
        assert!(err.contains("attention layer 1"), "got: {err}");
        assert!(err.contains("position 5"), "got: {err}");
    }

    #[test]
    fn unsafe_to_evict_when_a_layer_never_offloaded() {
        // Cursor=0 means the layer has offloaded NOTHING. Evicting any
        // position fails the check.
        let cursors = vec![10, 10, 0];
        let err = check_safe_to_evict(&cursors, 0).unwrap_err().to_string();
        assert!(err.contains("attention layer 2"), "got: {err}");
    }

    #[test]
    fn safe_to_evict_with_empty_cursor_vec_is_vacuously_true() {
        // A sequence whose `disk_last_offloaded_per_layer` hasn't been
        // populated yet (e.g. fresh sequence with no attn layers run)
        // can't have un-offloaded blocks because no layer has run. The
        // production `meta.rs:180` initializes this vec to `vec![0; n_attn]`
        // so this case shouldn't fire in real workloads, but the helper
        // should be vacuously correct.
        let cursors: Vec<u32> = vec![];
        assert!(check_safe_to_evict(&cursors, 100).is_ok());
    }

    // Issue #31 — `advance_layer_cursors_after_slide` keeps cursors ≥ window_start.

    #[test]
    fn advance_after_slide_promotes_lagging_cursors() {
        let mut cursors = vec![10, 5, 8];
        advance_layer_cursors_after_slide(&mut cursors, 9);
        // Layer 0 was already at 10 ≥ 9, unchanged. Layer 1 was at 5 < 9, bumped.
        // Layer 2 was at 8 < 9, bumped.
        assert_eq!(cursors, vec![10, 9, 9]);
    }

    #[test]
    fn advance_after_slide_never_moves_cursor_backward() {
        let mut cursors = vec![100, 100, 100];
        advance_layer_cursors_after_slide(&mut cursors, 50);
        // All cursors ≥ 50 → no change.
        assert_eq!(cursors, vec![100, 100, 100]);
    }

    #[test]
    fn advance_after_slide_idempotent() {
        let mut cursors = vec![5, 5, 5];
        advance_layer_cursors_after_slide(&mut cursors, 10);
        advance_layer_cursors_after_slide(&mut cursors, 10);
        assert_eq!(cursors, vec![10, 10, 10]);
    }

    // Round-trip: a slide loop pattern — for each slide, check then advance.
    // Models the cap=4 / chunk crossing case described in issue #31.

    #[test]
    fn slide_loop_round_trip_chunk_transition() {
        // After chunk N, all 3 attn layers have offloaded blocks 0..64.
        let mut cursors = vec![64u32, 64, 64];

        // Chunk N+1's bulk alloc loop: simulate 64 slides + 64 allocs with
        // window_start advancing one step per slide. cap = 64.
        let cap = 64;
        for slide_idx in 0..cap {
            let ws_before = slide_idx; // prior to this slide, ws = slide_idx
            // Safety check: every cursor > ws_before? Initial cursors are 64.
            // All slides up to slide_idx=63 have cursors > slide_idx → safe.
            assert!(
                check_safe_to_evict(&cursors, ws_before).is_ok(),
                "slide {slide_idx} should be safe with cursors {cursors:?}"
            );
            // Advance after the slide.
            advance_layer_cursors_after_slide(&mut cursors, ws_before + 1);
        }

        // After 64 slides (ws now 64), cursors should still be [64; 3] because
        // none of the advances moved them past 64 (each step advanced ws by 1
        // up to 64, and cursors started at 64 ≥ each new_ws).
        assert_eq!(cursors, vec![64, 64, 64]);
    }

    fn test_seq(seq_len: usize) -> SequenceState {
        SequenceState {
            tokens: Vec::new(),
            block_table: Vec::new(),
            seq_len,
            rotary_positions: crate::traits::RotaryPositions::identity(),
            qwen4_qsa_required: false,
            layer_states: Vec::new(),
            proposer_state: None,
            proposer_state_alt: None,
            slot_idx: 0,
            marconi_skip_to: 0,
            session_hash: 0,
            verify_frame: None,
            chunked_prefill_meta: None,
            qwen4_prefill_lineage: crate::traits::Qwen4PrefillLineage::Unset,
            cached_prefix_tokens: 0,
            kv_valid_tokens: 0,
            prompt_len: 0,
            disk_block_ids: Vec::new(),
            mtp_lastk_host_buf: Vec::new(),
            mtp_lastk_host_filled: 0,
            mtp_lastk_end_abs: 0,
            disk_last_offloaded_per_layer: Vec::new(),
        }
    }

    #[test]
    fn reserve_decode_rows_coalesces_contiguous_runs_and_isolates_non_contiguous_blocks() {
        use spark_runtime::gpu::mock::MockGpuBackend;
        use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype};
        use spark_runtime::prefix_cache::NoPrefixCaching;

        const BLOCK_SIZE: usize = 16;
        const NUM_LAYERS: usize = 2;
        const NUM_BLOCKS: usize = 8;

        let gpu = MockGpuBackend::new();
        let config = KvCacheConfig {
            block_size: BLOCK_SIZE,
            num_kv_heads: 2,
            head_dim: 4,
            num_layers: NUM_LAYERS,
            dtype: KvCacheDtype::Bf16,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
        };
        let block_bytes = config.block_bytes();
        let mut kv = PagedKvCache::new(config, NUM_BLOCKS, &gpu).expect("create mock KV cache");

        let fill_block = |kv: &PagedKvCache, blk: u32, byte: u8| {
            let buf = vec![byte; block_bytes];
            for layer in 0..NUM_LAYERS {
                kv.write_block(layer, blk, &buf, &buf, &gpu)
                    .expect("write sentinel");
            }
        };
        let assert_block_byte = |kv: &PagedKvCache, blk: u32, expected: u8| {
            for layer in 0..NUM_LAYERS {
                let (k, v) = kv.read_block(layer, blk, &gpu).expect("read block");
                assert!(
                    k.iter().all(|&b| b == expected),
                    "layer {layer} K block {blk} expected 0x{expected:02X}"
                );
                assert!(
                    v.iter().all(|&b| b == expected),
                    "layer {layer} V block {blk} expected 0x{expected:02X}"
                );
            }
        };

        for blk in 0..NUM_BLOCKS as u32 {
            fill_block(&kv, blk, 0xA5);
        }

        // 1. Four contiguous blocks [0, 1, 2, 3] coalesce to 1 run per (layer, side) = 4 memsets
        //    (instead of 4 blocks * 2 layers * 2 sides = 16).
        let mut seq_a = test_seq(0);
        let memsets_before = gpu.memset_count();
        reserve_decode_rows(
            &mut seq_a,
            4 * BLOCK_SIZE,
            &mut kv,
            &NoPrefixCaching,
            &gpu,
            0,
        )
        .expect("contiguous reserve");
        assert_eq!(seq_a.block_table, vec![0, 1, 2, 3]);
        assert_eq!(gpu.memset_count() - memsets_before, NUM_LAYERS * 2);
        for blk in 0..4u32 {
            assert_block_byte(&kv, blk, 0x00);
        }
        for blk in 4..NUM_BLOCKS as u32 {
            assert_block_byte(&kv, blk, 0xA5);
        }

        // 2. Fragment free list so next 4 blocks are [5, 4, 1, 0] (two runs: [0..2) and [4..6)),
        //    leaving pinned blocks [2, 3] between them untouched with sentinel 0x5A.
        let b4 = kv.alloc_block().expect("alloc 4");
        let b5 = kv.alloc_block().expect("alloc 5");
        assert_eq!((b4, b5), (4, 5));
        for blk in [2u32, 3] {
            fill_block(&kv, blk, 0x5A);
        }
        for blk in [0u32, 1, 4, 5] {
            fill_block(&kv, blk, 0xA5);
        }
        kv.free_blocks(&[0, 1, 4, 5]);

        let mut seq_b = test_seq(0);
        let memsets_before = gpu.memset_count();
        reserve_decode_rows(
            &mut seq_b,
            4 * BLOCK_SIZE,
            &mut kv,
            &NoPrefixCaching,
            &gpu,
            0,
        )
        .expect("non-contiguous reserve");
        assert_eq!(seq_b.block_table, vec![5, 4, 1, 0]);
        assert_eq!(gpu.memset_count() - memsets_before, NUM_LAYERS * 2 * 2);
        for blk in [0u32, 1, 4, 5] {
            assert_block_byte(&kv, blk, 0x00);
        }
        for blk in [2u32, 3] {
            assert_block_byte(&kv, blk, 0x5A);
        }
        for blk in [6u32, 7] {
            assert_block_byte(&kv, blk, 0xA5);
        }

        // 3. Part-way exhaustion still zeroes the 2 contiguous blocks [6, 7] allocated
        //    before failing (4 memsets) and preserves the alloc error message.
        let mut seq_c = test_seq(0);
        let memsets_before = gpu.memset_count();
        let err = reserve_decode_rows(
            &mut seq_c,
            3 * BLOCK_SIZE,
            &mut kv,
            &NoPrefixCaching,
            &gpu,
            0,
        )
        .expect_err("exhaustion must fail");
        assert!(err.to_string().contains("alloc failed"), "got: {err}");
        assert_eq!(seq_c.block_table, vec![6, 7]);
        assert_eq!(gpu.memset_count() - memsets_before, NUM_LAYERS * 2);
        for blk in [6u32, 7] {
            assert_block_byte(&kv, blk, 0x00);
        }
        for blk in [2u32, 3] {
            assert_block_byte(&kv, blk, 0x5A);
        }

        // 4. Single-block allocation takes the direct zero_block path (4 memsets).
        kv.free_block(7);
        fill_block(&kv, 7, 0xA5);
        let mut seq_d = test_seq(0);
        let memsets_before = gpu.memset_count();
        reserve_decode_rows(&mut seq_d, BLOCK_SIZE, &mut kv, &NoPrefixCaching, &gpu, 0)
            .expect("single-block reserve");
        assert_eq!(seq_d.block_table, vec![7]);
        assert_eq!(gpu.memset_count() - memsets_before, NUM_LAYERS * 2);
        assert_block_byte(&kv, 7, 0x00);
    }
}
