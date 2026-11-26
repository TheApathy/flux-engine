// SPDX-License-Identifier: AGPL-3.0-only

//! SSM snapshot LRU index — independent of the token-radix structure.
//!
//! Snapshots are keyed by (session_hash, token_count, prefix_hash) so the
//! same prompt across requests can hit a cached SSM state without going
//! through the radix tree.

use crate::prefix_cache::EvictedSnapshot;

use super::hash_token_prefix;

pub(super) struct SnapshotEntry {
    snapshot_id: usize,
    session_hash: u64,
    pub(super) token_count: usize,
    pub(super) prefix_hash: u64,
    last_access: u64,
    /// Cumulative hits over the entry's lifetime — combined with
    /// `last_access` in eviction to approximate the forecast-based
    /// policy from the Marconi paper §4 (B.4, 2026-04-25). Hot
    /// prefixes (high hit count) survive longer than cold ones at
    /// the same age.
    hit_count: u32,
}

pub(super) struct SsmSnapshotIndex {
    pub(super) entries: Vec<SnapshotEntry>,
    pub(super) access_counter: u64,
    /// Log-spaced restore points kept behind each live conversation's
    /// newest save (`snapshot_ladder`).
    pub(super) ladder: super::snapshot_ladder::Ladder,
}

impl SsmSnapshotIndex {
    pub(super) fn new() -> Self {
        Self {
            entries: Vec::new(),
            access_counter: 0,
            ladder: Default::default(),
        }
    }

    pub(super) fn insert(
        &mut self,
        prefix_hash: u64,
        snapshot_id: usize,
        session_hash: u64,
        token_count: usize,
    ) -> Option<usize> {
        for entry in &mut self.entries {
            if entry.prefix_hash == prefix_hash
                && entry.session_hash == session_hash
                && entry.token_count == token_count
            {
                let old = entry.snapshot_id;
                entry.snapshot_id = snapshot_id;
                self.access_counter += 1;
                entry.last_access = self.access_counter;
                return Some(old);
            }
        }
        self.access_counter += 1;
        self.entries.push(SnapshotEntry {
            snapshot_id,
            session_hash,
            token_count,
            prefix_hash,
            last_access: self.access_counter,
            hit_count: 0,
        });
        None
    }

    /// Find deepest snapshot matching session within matched_tokens range.
    pub(super) fn lookup(
        &mut self,
        tokens: &[u32],
        matched_tokens: usize,
        session_hash: u64,
    ) -> Option<(usize, usize)> {
        self.lookup_aligned(tokens, matched_tokens, 1, session_hash)
    }

    /// Find deepest snapshot matching `session_hash` within `matched_tokens`
    /// whose token boundary is an exact multiple of `alignment` (`>= 1`).
    pub(super) fn lookup_aligned(
        &mut self,
        tokens: &[u32],
        matched_tokens: usize,
        alignment: usize,
        session_hash: u64,
    ) -> Option<(usize, usize)> {
        let align = alignment.max(1);
        let mut best_idx: Option<(usize, usize, usize)> = None; // (entry_idx, snapshot_id, token_count)
        for (idx, entry) in self.entries.iter().enumerate() {
            if entry.token_count == 0
                || entry.token_count > matched_tokens
                || !entry.token_count.is_multiple_of(align)
            {
                continue;
            }
            if session_hash != 0 && entry.session_hash != 0 && entry.session_hash != session_hash {
                continue;
            }
            let h = hash_token_prefix(tokens, entry.token_count);
            if h != entry.prefix_hash {
                continue;
            }
            if best_idx.is_none_or(|(_, _, best_tok)| entry.token_count > best_tok) {
                best_idx = Some((idx, entry.snapshot_id, entry.token_count));
            }
        }
        if let Some((idx, snap_id, tok_count)) = best_idx {
            self.access_counter += 1;
            let entry = &mut self.entries[idx];
            entry.last_access = self.access_counter;
            entry.hit_count = entry.hit_count.saturating_add(1);
            Some((snap_id, tok_count))
        } else {
            None
        }
    }

    pub(super) fn evict_lru(&mut self) -> Option<usize> {
        self.evict_lru_entry().map(|e| e.snapshot_id)
    }

    pub(super) fn evict_lru_entry(&mut self) -> Option<EvictedSnapshot> {
        if self.entries.is_empty() {
            return None;
        }
        // Protection class, lowest evicted first: ordinary entries (0), then
        // restore-ladder rungs (1, `snapshot_ladder`). Rungs are spared only
        // while a non-rung candidate exists, so a pool of only-protected
        // entries still yields a victim and `save`/reclaim always make progress.
        let class = |entry: &SnapshotEntry| u8::from(self.ladder.holds(entry.prefix_hash));
        let victim_class = self.entries.iter().map(class).min()?;
        // Forecast-based policy (B.4, 2026-04-25, Marconi paper §4):
        // score = last_access.saturating_mul(1 + hit_count). Higher score =
        // more recent and/or hotter; lower score = older and/or less-frequently
        // used → evict first. Pure LRU (`last_access` only) discarded hot
        // prefixes that just happened to be re-accessed less recently than a
        // one-shot entry; weighting by `(1 + hit_count)` keeps recurrent
        // prefixes (system prompts, tool descriptions in agentic sessions)
        // resident longer.
        let mut best: Option<(usize, u64)> = None;
        for (i, entry) in self.entries.iter().enumerate() {
            if class(entry) > victim_class {
                continue; // a less valuable restore point is still resident
            }
            // Saturating math: hit_count fits u32 → +1 fits u64.
            let score = entry
                .last_access
                .saturating_mul(1 + u64::from(entry.hit_count));
            if best.is_none_or(|(_, victim_score)| score < victim_score) {
                best = Some((i, score));
            }
        }
        let (victim_idx, _) = best?;
        let entry = self.entries.swap_remove(victim_idx);
        Some(EvictedSnapshot {
            snapshot_id: entry.snapshot_id,
            session_hash: entry.session_hash,
            token_count: entry.token_count,
            prefix_hash: entry.prefix_hash,
        })
    }

    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }
}
