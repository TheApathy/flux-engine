// SPDX-License-Identifier: AGPL-3.0-only

//! Radix tree prefix cache for KV block reuse.
//!
//! Token sequences are chunked at `block_size` granularity. Each node in
//! the tree corresponds to one KV cache block. Lookup walks the tree
//! matching block-aligned chunks, returning cached physical block indices.
//!
//! Thread-safe via `Mutex<RadixTreeInner>`.

use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::Mutex;

use crate::byte_resume::{ByteResumeMatch, heal_bpe_boundary_drift};
use crate::prefix_cache::{
    EvictedBlocks, EvictedSnapshot, NonCausalSpan, PrefixCache, PrefixMatch,
    is_interior_of_noncausal_span,
};

mod inner;
mod snapshot;
mod snapshot_ladder;

#[cfg(test)]
mod tests;

use inner::{QsaFilter, RadixTreeInner};
use snapshot::SsmSnapshotIndex;
pub use snapshot_ladder::ssm_snap_ladder_base;

/// Maximum number of recent diverging branches probed during Byte Resume healing.
const MAX_BYTE_RESUME_CANDIDATES: usize = 8;

pub(crate) const fn prefix_hash_seed() -> u64 {
    0xcbf29ce484222325 // FNV-1a basis
}

pub(crate) const fn prefix_hash_push(mut h: u64, token: u32) -> u64 {
    h ^= token as u64;
    h.wrapping_mul(0x100000001b3)
}

/// FNV-1a-ish stable hash for the first `count` tokens — used to key
/// snapshots independently of the radix tree (allows the same prefix
/// hash to be reproduced across requests).
pub(crate) fn hash_token_prefix(tokens: &[u32], count: usize) -> u64 {
    let mut h = prefix_hash_seed();
    for &t in &tokens[..count] {
        h = prefix_hash_push(h, t);
    }
    h
}

/// Thread-safe radix tree prefix cache.
///
/// SSM snapshots are stored in a separate `SsmSnapshotIndex`, decoupled from
/// tree node lifetime. This ensures snapshots survive KV cache eviction.
/// Lock ordering: acquire `inner` first (then release), then `snapshot_index`.
pub struct RadixTree {
    inner: Mutex<RadixTreeInner>,
    snapshot_index: Mutex<SsmSnapshotIndex>,
    paired_truncate_to_snapshot_floor: AtomicBool,
}

impl Default for RadixTree {
    fn default() -> Self {
        Self::new()
    }
}

impl RadixTree {
    pub fn new() -> Self {
        let strict_floor_env = std::env::var("FLUX_PC_STRICT_SNAPSHOT_FLOOR").as_deref() == Ok("1");
        Self {
            inner: Mutex::new(RadixTreeInner::new()),
            snapshot_index: Mutex::new(SsmSnapshotIndex::new()),
            paired_truncate_to_snapshot_floor: AtomicBool::new(strict_floor_env),
        }
    }

    /// KV walk plus snapshot-index lookup with hit/miss telemetry.
    fn lookup_filtered(
        &self,
        tokens: &[u32],
        block_size: usize,
        session_hash: u64,
        qsa_filter: QsaFilter,
    ) -> PrefixMatch {
        if block_size == 0 {
            crate::prefix_cache::record_cache_miss();
            return PrefixMatch::empty();
        }
        let (matched_blocks, matched_disk_block_ids, matched_tokens) = self
            .inner
            .lock()
            .walk_and_pin(tokens, block_size, qsa_filter);
        let (ssm_snapshot, ssm_snapshot_tokens) = if matched_tokens > 0 {
            let mut idx = self.snapshot_index.lock();
            match idx.lookup(tokens, matched_tokens, session_hash) {
                Some((snap_id, tok_count)) => (Some(snap_id), tok_count),
                None => (None, 0),
            }
        } else {
            (None, 0)
        };
        let matched_disk_block_ids = if matched_disk_block_ids.iter().all(|&id| id == u32::MAX) {
            Vec::new()
        } else {
            matched_disk_block_ids
        };
        let prefix_match = PrefixMatch {
            matched_blocks,
            matched_disk_block_ids,
            matched_tokens,
            ssm_snapshot,
            ssm_snapshot_tokens,
        };
        if prefix_match.matched_tokens > 0 {
            crate::prefix_cache::record_cache_hit(prefix_match.matched_tokens);
        } else {
            crate::prefix_cache::record_cache_miss();
        }
        prefix_match
    }

    fn lookup_paired_filtered(
        &self,
        tokens: &[u32],
        block_size: usize,
        session_hash: u64,
        qsa_filter: QsaFilter,
    ) -> PrefixMatch {
        if block_size == 0 {
            crate::prefix_cache::record_cache_miss();
            return PrefixMatch::empty();
        }
        let (matched_blocks, mut matched_disk_block_ids, full_matched_tokens, full_blocks) = {
            let mut inner = self.inner.lock();
            let (mut blocks, mut disk, raw_matched) = inner.walk(tokens, block_size, qsa_filter);
            let full_blocks = raw_matched / block_size;
            let full_matched = full_blocks * block_size;
            // Strip any unaligned sub-block partial suffix: hybrid prefill/decode
            // must never reuse a partially-filled shared cached block in place.
            blocks.truncate(full_blocks);
            disk.truncate(full_blocks);
            if full_matched > 0 {
                inner.inc_refs(tokens, block_size, full_matched);
            }
            (blocks, disk, full_matched, full_blocks)
        };

        let (ssm_snapshot, ssm_snapshot_tokens) = if full_matched_tokens > 0 {
            let mut idx = self.snapshot_index.lock();
            match idx.lookup_aligned(tokens, full_matched_tokens, block_size, session_hash) {
                Some((snap_id, tok_count)) => (Some(snap_id), tok_count),
                None => (None, 0),
            }
        } else {
            (None, 0)
        };

        if matched_disk_block_ids.iter().all(|&id| id == u32::MAX) {
            matched_disk_block_ids.clear();
        }

        let mut prefix_match = PrefixMatch {
            matched_blocks,
            matched_disk_block_ids,
            matched_tokens: full_matched_tokens,
            ssm_snapshot,
            ssm_snapshot_tokens,
        };

        if let Some(snap_tok) = prefix_match.paired_ssm_tokens() {
            if self
                .paired_truncate_to_snapshot_floor
                .load(Ordering::Acquire)
                && prefix_match.matched_tokens > snap_tok
            {
                // PC_WRITE_FLOOR: release tree refs past `snap_blocks` and truncate
                // `matched_blocks` so recomputing `[snap_tok..]` allocates private
                // blocks rather than overwriting shared cached KV/QSA pages.
                let snap_blocks = snap_tok / block_size;
                self.inner
                    .lock()
                    .dec_refs_range(tokens, block_size, snap_blocks, full_blocks);
                prefix_match.matched_blocks.truncate(snap_blocks);
                if !prefix_match.matched_disk_block_ids.is_empty() {
                    prefix_match.matched_disk_block_ids.truncate(snap_blocks);
                }
                prefix_match.matched_tokens = snap_tok;
            }
            crate::prefix_cache::record_cache_hit(prefix_match.matched_tokens);
            prefix_match
        } else {
            if full_matched_tokens > 0 {
                tracing::info!(
                    "Prefix cache miss: {} KV tokens ({} blocks) without a restorable SSM snapshot — full prefill",
                    full_matched_tokens,
                    prefix_match.matched_blocks.len(),
                );
                self.inner
                    .lock()
                    .dec_refs_matched(tokens, block_size, full_matched_tokens);
            }
            crate::prefix_cache::record_cache_miss();
            PrefixMatch::empty()
        }
    }

    /// Probe cached branches at the first mismatch boundary to heal BPE
    /// token-boundary drift (`"\n"+"n"` vs `"\n\n"`, `</think>` whitespace splits)
    /// while preserving exact ID matching for special/control tokens.
    fn find_byte_resume_rewrite<'a>(
        &self,
        tokens: &[u32],
        block_size: usize,
        token_bytes: impl Fn(u32) -> Option<&'a [u8]>,
        is_special_or_added: impl Fn(u32) -> bool,
    ) -> Option<ByteResumeMatch> {
        if block_size == 0 || tokens.len() < block_size {
            return None;
        }
        let (raw_prefix_tokens, candidates) = {
            let inner = self.inner.lock();
            inner.candidate_cached_sequences(tokens, block_size, MAX_BYTE_RESUME_CANDIDATES)
        };
        let mut best: Option<ByteResumeMatch> = None;
        for cand in &candidates {
            if let Some(healed) =
                heal_bpe_boundary_drift(cand, tokens, &token_bytes, &is_special_or_added)
            {
                let aligned_gain = healed.block_aligned_cached_tokens(block_size);
                if aligned_gain <= raw_prefix_tokens {
                    continue;
                }
                let replace = match &best {
                    None => true,
                    Some(prev) => {
                        (aligned_gain, healed.cached_end)
                            > (
                                prev.block_aligned_cached_tokens(block_size),
                                prev.cached_end,
                            )
                    }
                };
                if replace {
                    best = Some(healed);
                }
            }
        }
        best
    }

    /// Attention-only lookup with automatic ExpertAtlas Byte Resume fallback.
    ///
    /// When BPE token-boundary drift is healed across at least one additional
    /// full KV block, returns the hit against `healed.rewritten_tokens` along
    /// with `Some(healed)` (safely backed off to full KV blocks when the seam
    /// sits mid-block).
    pub fn lookup_with_byte_resume<'a>(
        &self,
        tokens: &[u32],
        block_size: usize,
        session_hash: u64,
        token_bytes: impl Fn(u32) -> Option<&'a [u8]>,
        is_special_or_added: impl Fn(u32) -> bool,
    ) -> (PrefixMatch, Option<ByteResumeMatch>) {
        if let Some(healed) =
            self.find_byte_resume_rewrite(tokens, block_size, token_bytes, is_special_or_added)
        {
            let mut pm = self.lookup(&healed.rewritten_tokens, block_size, session_hash);
            let safe_cap = healed.block_aligned_cached_tokens(block_size);
            if !healed.is_block_aligned(block_size) && pm.matched_tokens > safe_cap {
                let safe_blocks = safe_cap / block_size;
                self.inner.lock().dec_refs_tail(
                    &healed.rewritten_tokens,
                    block_size,
                    safe_blocks,
                    pm.matched_tokens,
                );
                pm.matched_blocks.truncate(safe_blocks);
                if !pm.matched_disk_block_ids.is_empty() {
                    pm.matched_disk_block_ids.truncate(safe_blocks);
                }
                pm.matched_tokens = safe_cap;
            }
            if pm.matched_tokens > 0 {
                return (pm, Some(healed));
            }
        }
        (self.lookup(tokens, block_size, session_hash), None)
    }

    /// Hybrid (KV + SSM/PLE snapshot) lookup with automatic ExpertAtlas Byte Resume.
    pub fn lookup_paired_with_byte_resume<'a>(
        &self,
        tokens: &[u32],
        block_size: usize,
        session_hash: u64,
        token_bytes: impl Fn(u32) -> Option<&'a [u8]>,
        is_special_or_added: impl Fn(u32) -> bool,
    ) -> (PrefixMatch, Option<ByteResumeMatch>) {
        if let Some(healed) =
            self.find_byte_resume_rewrite(tokens, block_size, token_bytes, is_special_or_added)
        {
            let pm = self.lookup_paired(&healed.rewritten_tokens, block_size, session_hash);
            if pm.paired_ssm_tokens().is_some() {
                return (pm, Some(healed));
            }
        }
        (self.lookup_paired(tokens, block_size, session_hash), None)
    }
}

impl PrefixCache for RadixTree {
    fn lookup(&self, tokens: &[u32], block_size: usize, session_hash: u64) -> PrefixMatch {
        self.lookup_filtered(tokens, block_size, session_hash, QsaFilter::Any)
    }

    fn lookup_qsa(&self, tokens: &[u32], block_size: usize, session_hash: u64) -> PrefixMatch {
        self.lookup_filtered(
            tokens,
            block_size,
            session_hash,
            QsaFilter::RequirePopulated,
        )
    }

    fn lookup_paired(&self, tokens: &[u32], block_size: usize, session_hash: u64) -> PrefixMatch {
        self.lookup_paired_filtered(tokens, block_size, session_hash, QsaFilter::Any)
    }

    fn lookup_paired_qsa(
        &self,
        tokens: &[u32],
        block_size: usize,
        session_hash: u64,
    ) -> PrefixMatch {
        self.lookup_paired_filtered(
            tokens,
            block_size,
            session_hash,
            QsaFilter::RequirePopulated,
        )
    }

    fn mark_blocks_qsa_populated(&self, tokens: &[u32], block_table: &[u32], block_size: usize) {
        self.inner
            .lock()
            .mark_blocks_qsa_populated(tokens, block_table, block_size);
    }

    fn insert(
        &self,
        tokens: &[u32],
        block_table: &[u32],
        disk_block_ids: &[u32],
        block_size: usize,
        matched_tokens: usize,
    ) -> Vec<u32> {
        self.inner.lock().insert(
            tokens,
            block_table,
            disk_block_ids,
            block_size,
            matched_tokens,
        )
    }

    fn insert_with_snapshot(
        &self,
        tokens: &[u32],
        block_table: &[u32],
        disk_block_ids: &[u32],
        block_size: usize,
        snapshot_id: usize,
        session_hash: u64,
        matched_tokens: usize,
    ) -> (Option<usize>, Vec<u32>) {
        // Phase 1: insert tree nodes (lock inner, then release)
        let newly_acquired = self.inner.lock().insert(
            tokens,
            block_table,
            disk_block_ids,
            block_size,
            matched_tokens,
        );
        // Phase 2: register snapshot in index (lock snapshot_index, inner NOT held)
        let prefix_hash = hash_token_prefix(tokens, tokens.len());
        let mut idx = self.snapshot_index.lock();
        let displaced = idx.insert(prefix_hash, snapshot_id, session_hash, tokens.len());
        idx.stamp_ladder(tokens, session_hash);
        (displaced, newly_acquired)
    }

    fn insert_intermediate_snapshot(
        &self,
        tokens: &[u32],
        _block_table: &[u32],
        _disk_block_ids: &[u32],
        _block_size: usize,
        snapshot_id: usize,
        session_hash: u64,
        _matched_tokens: usize,
    ) -> Option<usize> {
        // Intermediate snapshots go directly into the index with the correct
        // token boundary (tokens.len()). Tree nodes are already inserted by
        // a prior `insert()` call, which handled the ref_count bookkeeping.
        let prefix_hash = hash_token_prefix(tokens, tokens.len());
        let mut idx = self.snapshot_index.lock();
        let displaced = idx.insert(prefix_hash, snapshot_id, session_hash, tokens.len());
        idx.stamp_ladder(tokens, session_hash);
        displaced
    }

    fn release(&self, tokens: &[u32], block_size: usize) {
        self.inner.lock().dec_refs(tokens, block_size);
    }

    fn release_matched(&self, tokens: &[u32], block_size: usize, matched_tokens: usize) {
        self.inner
            .lock()
            .dec_refs_matched(tokens, block_size, matched_tokens);
    }

    fn evict(&self, num_blocks: usize) -> EvictedBlocks {
        let (physical, disk) = self.inner.lock().evict(num_blocks);
        // Filter MAX sentinels out — the caller only needs disk_block_ids to
        // dec_disk_ref on, and MAX entries don't correspond to a live HSS ref.
        let disk_block_ids: Vec<u32> = disk.into_iter().filter(|&id| id != u32::MAX).collect();
        EvictedBlocks {
            physical,
            disk_block_ids,
        }
    }

    fn drain_displaced_partials(&self) -> EvictedBlocks {
        let (physical, disk) = self.inner.lock().drain_displaced_partials();
        let disk_block_ids: Vec<u32> = disk.into_iter().filter(|&id| id != u32::MAX).collect();
        EvictedBlocks {
            physical,
            disk_block_ids,
        }
    }

    fn evict_snapshot_lru(&self) -> Option<usize> {
        self.snapshot_index.lock().evict_lru()
    }

    fn evict_snapshot_lru_entry(&self) -> Option<EvictedSnapshot> {
        self.snapshot_index.lock().evict_lru_entry()
    }

    fn set_paired_truncate_to_snapshot_floor(&self, enabled: bool) {
        self.paired_truncate_to_snapshot_floor
            .store(enabled, Ordering::Release);
    }

    fn insert_with_noncausal_spans(
        &self,
        tokens: &[u32],
        block_table: &[u32],
        disk_block_ids: &[u32],
        block_size: usize,
        matched_tokens: usize,
        noncausal_spans: &[NonCausalSpan],
    ) -> Vec<u32> {
        self.inner.lock().insert_with_spans(
            tokens,
            block_table,
            disk_block_ids,
            block_size,
            matched_tokens,
            noncausal_spans,
        )
    }

    fn insert_with_snapshot_and_noncausal_spans(
        &self,
        tokens: &[u32],
        block_table: &[u32],
        disk_block_ids: &[u32],
        block_size: usize,
        snapshot_id: usize,
        session_hash: u64,
        matched_tokens: usize,
        noncausal_spans: &[NonCausalSpan],
    ) -> (Option<usize>, Vec<u32>) {
        let newly_acquired = self.inner.lock().insert_with_spans(
            tokens,
            block_table,
            disk_block_ids,
            block_size,
            matched_tokens,
            noncausal_spans,
        );
        if is_interior_of_noncausal_span(tokens.len(), noncausal_spans) {
            return (Some(snapshot_id), newly_acquired);
        }
        let prefix_hash = hash_token_prefix(tokens, tokens.len());
        let mut idx = self.snapshot_index.lock();
        let displaced = idx.insert(prefix_hash, snapshot_id, session_hash, tokens.len());
        idx.stamp_ladder_with_spans(tokens, session_hash, noncausal_spans);
        (displaced, newly_acquired)
    }

    fn insert_intermediate_snapshot_with_noncausal_spans(
        &self,
        tokens: &[u32],
        _block_table: &[u32],
        _disk_block_ids: &[u32],
        _block_size: usize,
        snapshot_id: usize,
        session_hash: u64,
        _matched_tokens: usize,
        noncausal_spans: &[NonCausalSpan],
    ) -> Option<usize> {
        if is_interior_of_noncausal_span(tokens.len(), noncausal_spans) {
            return Some(snapshot_id);
        }
        let prefix_hash = hash_token_prefix(tokens, tokens.len());
        let mut idx = self.snapshot_index.lock();
        let displaced = idx.insert(prefix_hash, snapshot_id, session_hash, tokens.len());
        idx.stamp_ladder_with_spans(tokens, session_hash, noncausal_spans);
        displaced
    }

    fn snapshot_count(&self) -> usize {
        self.snapshot_index.lock().len()
    }

    fn stats(&self) -> (usize, usize) {
        let inner = self.inner.lock();
        let entries = inner.num_entries();
        (entries, entries)
    }
}
