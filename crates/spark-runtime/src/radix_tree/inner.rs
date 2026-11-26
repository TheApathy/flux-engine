// SPDX-License-Identifier: AGPL-3.0-only

//! Single-threaded radix-tree state — wrapped behind a `Mutex` in the
//! [`super::RadixTree`] public API. Token sequences are chunked at
//! `block_size` granularity; each node holds one physical KV block index.

use std::collections::HashMap;

type NodeId = usize;

/// Whether a radix-tree walk requires populated Qwen4 QSA side-cache pages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum QsaFilter {
    /// Reuse any live cached block regardless of QSA side-cache state.
    Any,
    /// Stop before any block whose QSA side-cache pages were not populated.
    RequirePopulated,
}

pub(super) struct RadixNode {
    /// Children keyed by the token chunk (block_size tokens).
    children: HashMap<Vec<u32>, NodeId>,
    /// Physical KV cache block index stored at this node.
    block_idx: u32,
    /// `--high-speed-swap` disk-block ID (Phase 6.1.e). `u32::MAX` when HSS
    /// is not in use. The cache holds a refcount on this disk_id (bumped
    /// at insert time, dropped at evict time) parallel to its physical-
    /// block ref bookkeeping.
    disk_block_id: u32,
    /// Context hash: FNV-1a chain of this node's tokens + parent's context_hash.
    /// Ensures a block is only reused when the ENTIRE causal prefix matches,
    /// preventing cross-request KV contamination (vLLM-style parent-hash chain).
    context_hash: u64,
    /// Number of active sequences referencing this node.
    ref_count: u32,
    /// Monotonic access counter for LRU eviction.
    last_access: u64,
    /// Parent node (None for root).
    parent: Option<NodeId>,
    /// Token chunk that leads from parent to this node (for cleanup).
    parent_key: Option<Vec<u32>>,
    /// Partial (sub-block) suffix: the last incomplete block in a cached
    /// sequence. Stores `(partial_tokens, block_idx, disk_block_id)`
    /// where `partial_tokens.len() < block_size`. `disk_block_id == u32::MAX`
    /// when HSS isn't in use. Matched after all full blocks to cover the
    /// trailing remainder.
    partial_suffix: Option<(Vec<u32>, u32, u32)>,
    /// Number of active sequences borrowing a sub-block match on `partial_suffix`.
    /// Prevents eviction and defers reclaiming displaced partial blocks until
    /// all active borrowers release.
    partial_suffix_refs: u32,
    /// Displaced `(physical_block, disk_block_id)` pairs deferred while
    /// `partial_suffix_refs > 0`; flushed to `orphaned_*` once refs reach 0.
    pending_displaced_partials: Vec<(Option<u32>, Option<u32>)>,
    /// Minimum matched full-block depth required for this node to be a valid
    /// causal prefix match. Equals `depth` (`i + 1`) for ordinary causal blocks,
    /// and `span.end.div_ceil(block_size)` (or `usize::MAX` if `span.end` exceeds
    /// the inserted full blocks) when this block's token range overlaps a
    /// non-causal/bidirectional span `(span.start, span.end)` that has not yet
    /// closed at `(i + 1) * block_size`.
    min_commit_blocks: usize,
    /// True when any non-causal span `(span.start, span.end)` overlaps this
    /// block's token range, forbidding sub-block prefix reuse (`< block_size`).
    has_noncausal: bool,
    /// True iff Qwen4 QSA side-cache pages (`raw_ring` + `compressed_keys`)
    /// were populated for this node's physical `block_idx`. Defaults to
    /// `false` on insert and is set only when the owner of `block_idx` calls
    /// [`RadixTreeInner::mark_blocks_qsa_populated`].
    qsa_populated: bool,
}

fn block_overlaps_noncausal_span(
    block_idx: usize,
    block_size: usize,
    spans: &[crate::prefix_cache::NonCausalSpan],
) -> bool {
    let block_start = block_idx * block_size;
    let block_end = block_start + block_size;
    spans
        .iter()
        .any(|s| s.end > s.start && block_end > s.start && block_start < s.end)
}

/// Minimum full-block match depth required to commit block `block_idx` (`0`-indexed,
/// covering `[block_start, block_end)` where `block_end = (block_idx + 1) * block_size`).
fn required_commit_blocks_for_spans(
    block_idx: usize,
    block_size: usize,
    inserted_full_blocks: usize,
    spans: &[crate::prefix_cache::NonCausalSpan],
) -> usize {
    let block_start = block_idx * block_size;
    let block_end = block_start + block_size;
    let mut min_blocks = block_idx + 1;
    for span in spans {
        if span.end <= span.start {
            continue;
        }
        // If this block contains any token from `[span.start, span.end)` AND
        // the span has not finished by `block_end` (`block_end < span.end`), OR
        // `span.end` is unaligned inside a later block, require matching through
        // `span.end.div_ceil(block_size)` full blocks (or `usize::MAX` if the
        // inserted sequence never completed that full block).
        let overlaps_span = block_end > span.start && block_start < span.end;
        if overlaps_span {
            let close_blocks = span.end.div_ceil(block_size);
            if close_blocks > inserted_full_blocks {
                return usize::MAX;
            }
            if close_blocks > min_blocks {
                min_blocks = close_blocks;
            }
        }
    }
    min_blocks
}

/// Combine a parent context hash with a token chunk to produce a child context hash.
/// Uses FNV-1a for speed (~ns per block). The chain guarantees that two nodes with
/// identical token chunks but different prefixes get different context_hashes.
fn context_hash_combine(parent_hash: u64, tokens: &[u32]) -> u64 {
    let mut h = parent_hash;
    for &t in tokens {
        h ^= t as u64;
        h = h.wrapping_mul(0x100000001b3); // FNV-1a prime
    }
    h
}

/// Inner state protected by mutex.
pub(super) struct RadixTreeInner {
    nodes: Vec<RadixNode>,
    /// Indices of deleted nodes available for reuse.
    free_nodes: Vec<NodeId>,
    root: NodeId,
    access_counter: u64,
    /// Active sub-block child pins acquired by `walk_and_pin`, keyed by
    /// `(parent_node_id, suffix_tokens)` so `release_matched`/`release` unpins
    /// the exact child node that was pinned.
    subblock_pins: HashMap<(NodeId, Vec<u32>), Vec<NodeId>>,
    /// Orphaned physical KV blocks displaced when a `partial_suffix` was
    /// cleared or replaced; drained by `drain_displaced_partials` or `evict`.
    orphaned_phys: Vec<u32>,
    /// Orphaned HSS disk-block IDs displaced when a `partial_suffix` was
    /// cleared or replaced; drained by `drain_displaced_partials` or `evict`.
    orphaned_disk: Vec<u32>,
}

impl RadixTreeInner {
    pub(super) fn new() -> Self {
        let root = RadixNode {
            children: HashMap::new(),
            block_idx: u32::MAX,     // sentinel — root has no block
            disk_block_id: u32::MAX, // sentinel — root has no disk slot either
            context_hash: 0,         // root context hash is 0
            ref_count: 0,
            last_access: 0,
            parent: None,
            parent_key: None,
            partial_suffix: None,
            partial_suffix_refs: 0,
            pending_displaced_partials: Vec::new(),
            min_commit_blocks: 0,
            has_noncausal: false,
            qsa_populated: true,
        };
        Self {
            nodes: vec![root],
            free_nodes: Vec::new(),
            root: 0,
            access_counter: 0,
            subblock_pins: HashMap::new(),
            orphaned_phys: Vec::new(),
            orphaned_disk: Vec::new(),
        }
    }

    pub(super) fn next_access(&mut self) -> u64 {
        self.access_counter += 1;
        self.access_counter
    }

    pub(super) fn alloc_node(&mut self, node: RadixNode) -> NodeId {
        if let Some(id) = self.free_nodes.pop() {
            self.nodes[id] = node;
            id
        } else {
            let id = self.nodes.len();
            self.nodes.push(node);
            id
        }
    }

    /// Walk the tree matching block-aligned token chunks, optionally stopping
    /// before any node whose `qsa_populated` flag is `false`
    /// (`QsaFilter::RequirePopulated`).
    /// Returns (matched_blocks, matched_disk_block_ids, matched_tokens). The
    /// disk-id vec parallels matched_blocks; entries are `u32::MAX` on nodes
    /// that were inserted before HSS engaged (caller's responsibility to
    /// filter / treat MAX as "no disk-side ref to bump"). Snapshot lookup is
    /// now handled by the separate SsmSnapshotIndex after the walk completes.
    pub(super) fn walk(
        &self,
        tokens: &[u32],
        block_size: usize,
        qsa_filter: QsaFilter,
    ) -> (Vec<u32>, Vec<u32>, usize) {
        let require_qsa = matches!(qsa_filter, QsaFilter::RequirePopulated);
        let mut current = self.root;
        let mut matched_blocks = Vec::new();
        let mut matched_disk = Vec::new();
        let mut matched_min_commit = Vec::new();
        let mut matched_tokens = 0;
        let mut parent_ctx_hash: u64 = 0; // root context hash

        let num_full_blocks = tokens.len() / block_size;
        for i in 0..num_full_blocks {
            let chunk = &tokens[i * block_size..(i + 1) * block_size];
            let expected_hash = context_hash_combine(parent_ctx_hash, chunk);
            match self.nodes[current].children.get(chunk) {
                Some(&child)
                    if self.nodes[child].context_hash == expected_hash
                        && self.nodes[child].ref_count > 0
                        && (!require_qsa || self.nodes[child].qsa_populated) =>
                {
                    // Context chain matches AND block is still live — tentatively record.
                    matched_blocks.push(self.nodes[child].block_idx);
                    matched_disk.push(self.nodes[child].disk_block_id);
                    matched_min_commit.push(self.nodes[child].min_commit_blocks);
                    matched_tokens += block_size;
                    parent_ctx_hash = expected_hash;
                    current = child;
                }
                _ => break, // Token match but context mismatch, freed block, or unpopulated QSA — stop.
            }
        }

        // Enforce non-causal/bidirectional span closure: any block whose
        // `min_commit_blocks` exceeds the matched full-block depth lies inside
        // an unclosed non-causal span `(span.start, span.end)` and must roll
        // back to the deepest causal checkpoint `<= span.start`.
        let raw_full_blocks = matched_blocks.len();
        let mut valid_full_blocks = raw_full_blocks;
        while valid_full_blocks > 0 {
            if let Some(violating_idx) =
                (0..valid_full_blocks).rfind(|&idx| matched_min_commit[idx] > valid_full_blocks)
            {
                valid_full_blocks = violating_idx;
            } else {
                break;
            }
        }
        if valid_full_blocks < raw_full_blocks {
            matched_blocks.truncate(valid_full_blocks);
            matched_disk.truncate(valid_full_blocks);
            matched_tokens = valid_full_blocks * block_size;
            return (matched_blocks, matched_disk, matched_tokens);
        }

        // Sub-block matching: when remaining tokens don't fill a full block,
        // check two locations for a match:
        // 1. A child node whose key starts with the remaining tokens (the child
        //    covers a full block but we only need a prefix of it — safe ONLY
        //    when that child node is purely causal, i.e., `min_commit_blocks <= num_full_blocks + 1`,
        //    actually strictly causal `min_commit_blocks == num_full_blocks + 1` and not non-causal!).
        // 2. A partial suffix stored on the current node.
        let remainder = tokens.len() - matched_tokens;
        if remainder > 0 && remainder < block_size && matched_tokens == num_full_blocks * block_size
        {
            let suffix = &tokens[matched_tokens..];
            let mut found = false;
            for (key, &child_id) in &self.nodes[current].children {
                if key.len() >= suffix.len() && &key[..suffix.len()] == suffix {
                    let expected = context_hash_combine(parent_ctx_hash, key);
                    if self.nodes[child_id].context_hash == expected
                        && self.nodes[child_id].ref_count > 0
                        && (!require_qsa || self.nodes[child_id].qsa_populated)
                        && !self.nodes[child_id].has_noncausal
                        && self.nodes[child_id].min_commit_blocks == num_full_blocks + 1
                    {
                        matched_blocks.push(self.nodes[child_id].block_idx);
                        matched_disk.push(self.nodes[child_id].disk_block_id);
                        matched_tokens += remainder;
                        found = true;
                        break;
                    }
                }
            }
            if !found
                && !require_qsa
                && let Some((ref partial_toks, partial_block, partial_disk)) =
                    self.nodes[current].partial_suffix
                && partial_toks.len() >= suffix.len()
                && &partial_toks[..suffix.len()] == suffix
            {
                matched_blocks.push(partial_block);
                matched_disk.push(partial_disk);
                matched_tokens += remainder;
            }
        }

        (matched_blocks, matched_disk, matched_tokens)
    }

    /// Walk the tree and atomically pin every matched node — including any
    /// sub-block child match or `partial_suffix` match — so concurrent `evict`
    /// cannot free any physical KV block returned in the match before the
    /// caller calls `release` / `release_matched`.
    pub(super) fn walk_and_pin(
        &mut self,
        tokens: &[u32],
        block_size: usize,
        qsa_filter: QsaFilter,
    ) -> (Vec<u32>, Vec<u32>, usize) {
        let require_qsa = matches!(qsa_filter, QsaFilter::RequirePopulated);
        let mut current = self.root;
        let mut matched_nodes = Vec::new();
        let mut matched_blocks = Vec::new();
        let mut matched_disk = Vec::new();
        let mut matched_min_commit = Vec::new();
        let mut matched_tokens = 0;
        let mut parent_ctx_hash: u64 = 0;

        let num_full_blocks = tokens.len() / block_size;
        for i in 0..num_full_blocks {
            let chunk = &tokens[i * block_size..(i + 1) * block_size];
            let expected_hash = context_hash_combine(parent_ctx_hash, chunk);
            match self.nodes[current].children.get(chunk) {
                Some(&child)
                    if self.nodes[child].context_hash == expected_hash
                        && self.nodes[child].ref_count > 0
                        && (!require_qsa || self.nodes[child].qsa_populated) =>
                {
                    matched_nodes.push(child);
                    matched_blocks.push(self.nodes[child].block_idx);
                    matched_disk.push(self.nodes[child].disk_block_id);
                    matched_min_commit.push(self.nodes[child].min_commit_blocks);
                    matched_tokens += block_size;
                    parent_ctx_hash = expected_hash;
                    current = child;
                }
                _ => break,
            }
        }

        let raw_full_blocks = matched_blocks.len();
        let mut valid_full_blocks = raw_full_blocks;
        while valid_full_blocks > 0 {
            if let Some(violating_idx) =
                (0..valid_full_blocks).rfind(|&idx| matched_min_commit[idx] > valid_full_blocks)
            {
                valid_full_blocks = violating_idx;
            } else {
                break;
            }
        }
        if valid_full_blocks < raw_full_blocks {
            matched_nodes.truncate(valid_full_blocks);
            matched_blocks.truncate(valid_full_blocks);
            matched_disk.truncate(valid_full_blocks);
            matched_tokens = valid_full_blocks * block_size;
            if !matched_nodes.is_empty() {
                let access = self.next_access();
                for node_id in matched_nodes {
                    self.nodes[node_id].ref_count += 1;
                    self.nodes[node_id].last_access = access;
                }
            }
            return (matched_blocks, matched_disk, matched_tokens);
        }

        let mut subblock_child: Option<(NodeId, Vec<u32>)> = None;
        let mut matched_partial = false;
        let remainder = tokens.len() - matched_tokens;
        if remainder > 0 && remainder < block_size && matched_tokens == num_full_blocks * block_size
        {
            let suffix = &tokens[matched_tokens..];
            for (key, &child_id) in &self.nodes[current].children {
                if key.len() >= suffix.len() && &key[..suffix.len()] == suffix {
                    let expected = context_hash_combine(parent_ctx_hash, key);
                    if self.nodes[child_id].context_hash == expected
                        && self.nodes[child_id].ref_count > 0
                        && (!require_qsa || self.nodes[child_id].qsa_populated)
                        && !self.nodes[child_id].has_noncausal
                        && self.nodes[child_id].min_commit_blocks == num_full_blocks + 1
                    {
                        matched_blocks.push(self.nodes[child_id].block_idx);
                        matched_disk.push(self.nodes[child_id].disk_block_id);
                        matched_tokens += remainder;
                        subblock_child = Some((child_id, suffix.to_vec()));
                        break;
                    }
                }
            }
            if subblock_child.is_none()
                && !require_qsa
                && let Some((ref partial_toks, partial_block, partial_disk)) =
                    self.nodes[current].partial_suffix
                && partial_toks.len() >= suffix.len()
                && &partial_toks[..suffix.len()] == suffix
            {
                matched_blocks.push(partial_block);
                matched_disk.push(partial_disk);
                matched_tokens += remainder;
                matched_partial = true;
            }
        }

        if matched_tokens > 0 {
            let access = self.next_access();
            for node_id in matched_nodes {
                self.nodes[node_id].ref_count += 1;
                self.nodes[node_id].last_access = access;
            }
            if let Some((child_id, suffix_key)) = subblock_child {
                self.nodes[child_id].ref_count += 1;
                self.nodes[child_id].last_access = access;
                self.subblock_pins
                    .entry((current, suffix_key))
                    .or_default()
                    .push(child_id);
            } else if matched_partial {
                self.nodes[current].partial_suffix_refs += 1;
            }
        }

        (matched_blocks, matched_disk, matched_tokens)
    }

    /// Mark full-block nodes along `tokens` as QSA-populated only where the
    /// stored physical `block_idx` equals `block_table[i]`.
    pub(super) fn mark_blocks_qsa_populated(
        &mut self,
        tokens: &[u32],
        block_table: &[u32],
        block_size: usize,
    ) {
        if block_size == 0 {
            return;
        }
        let mut current = self.root;
        let mut parent_ctx_hash: u64 = 0;
        let num_blocks = (tokens.len() / block_size).min(block_table.len());
        for i in 0..num_blocks {
            let chunk = &tokens[i * block_size..(i + 1) * block_size];
            let expected_hash = context_hash_combine(parent_ctx_hash, chunk);
            match self.nodes[current].children.get(chunk) {
                Some(&child) if self.nodes[child].context_hash == expected_hash => {
                    if self.nodes[child].block_idx == block_table[i] {
                        self.nodes[child].qsa_populated = true;
                    }
                    parent_ctx_hash = expected_hash;
                    current = child;
                }
                _ => break,
            }
        }
    }

    /// Increment ref_count on all nodes along the matched path.
    pub(super) fn inc_refs(&mut self, tokens: &[u32], block_size: usize, num_matched: usize) {
        let access = self.next_access();
        let mut current = self.root;
        let num_blocks = num_matched / block_size;

        for i in 0..num_blocks {
            let chunk = &tokens[i * block_size..(i + 1) * block_size];
            if let Some(&child) = self.nodes[current].children.get(chunk) {
                self.nodes[child].ref_count += 1;
                self.nodes[child].last_access = access;
                current = child;
            } else {
                break;
            }
        }
    }

    /// Decrement ref_count on all nodes along the matched path (including any
    /// sub-block child or `partial_suffix` pin acquired by `walk_and_pin`).
    pub(super) fn dec_refs(&mut self, tokens: &[u32], block_size: usize) {
        self.dec_refs_matched(tokens, block_size, tokens.len());
    }

    /// Decrement the nodes acquired by a lookup of `num_matched` tokens,
    /// including any sub-block child or `partial_suffix` pin when
    /// `num_matched % block_size != 0`.
    pub(super) fn dec_refs_matched(
        &mut self,
        tokens: &[u32],
        block_size: usize,
        num_matched: usize,
    ) {
        if block_size == 0 {
            return;
        }
        self.dec_refs_tail(tokens, block_size, 0, num_matched);
    }

    /// Decrement `ref_count` on full-block nodes in `[keep_blocks, prev_full_blocks)`
    /// AND unpin any sub-block tail at `prev_matched_tokens` (when
    /// `prev_matched_tokens % block_size != 0` and `prev_full_blocks >= keep_blocks`).
    pub(super) fn dec_refs_tail(
        &mut self,
        tokens: &[u32],
        block_size: usize,
        keep_blocks: usize,
        prev_matched_tokens: usize,
    ) {
        if block_size == 0 {
            return;
        }
        let prev = prev_matched_tokens.min(tokens.len());
        let prev_full_blocks = prev / block_size;
        let has_subblock = !prev.is_multiple_of(block_size);
        let mut current = self.root;
        let mut reached_full = 0usize;

        for i in 0..prev_full_blocks {
            let chunk = &tokens[i * block_size..(i + 1) * block_size];
            if let Some(&child) = self.nodes[current].children.get(chunk) {
                if i >= keep_blocks && self.nodes[child].ref_count > 0 {
                    self.nodes[child].ref_count -= 1;
                }
                current = child;
                reached_full = i + 1;
            } else {
                break;
            }
        }

        if has_subblock && reached_full == prev_full_blocks && prev_full_blocks >= keep_blocks {
            let suffix = tokens[prev_full_blocks * block_size..prev].to_vec();
            let pin_key = (current, suffix);
            let pinned_child = if let Some(stack) = self.subblock_pins.get_mut(&pin_key) {
                let id = stack.pop();
                if stack.is_empty() {
                    self.subblock_pins.remove(&pin_key);
                }
                id
            } else {
                None
            };
            if let Some(child_id) = pinned_child {
                if self.nodes[child_id].ref_count > 0 {
                    self.nodes[child_id].ref_count -= 1;
                }
            } else if self.nodes[current].partial_suffix_refs > 0 {
                self.nodes[current].partial_suffix_refs -= 1;
                if self.nodes[current].partial_suffix_refs == 0 {
                    let deferred =
                        std::mem::take(&mut self.nodes[current].pending_displaced_partials);
                    for (phys, disk) in deferred {
                        if let Some(b) = phys {
                            self.orphaned_phys.push(b);
                        }
                        if let Some(d) = disk {
                            self.orphaned_disk.push(d);
                        }
                    }
                }
            }
        }
    }

    /// Decrement `ref_count` on full-block nodes in `[start_block, end_block)`
    /// along `tokens`. Used when paired hybrid lookup trims deeper KV blocks
    /// down to `PC_WRITE_FLOOR` (`ssm_snapshot_tokens / block_size`).
    pub(super) fn dec_refs_range(
        &mut self,
        tokens: &[u32],
        block_size: usize,
        start_block: usize,
        end_block: usize,
    ) {
        if block_size == 0 || start_block >= end_block {
            return;
        }
        self.dec_refs_tail(tokens, block_size, start_block, end_block * block_size);
    }

    fn record_displaced_partial(
        &mut self,
        node_id: NodeId,
        old_phys: Option<u32>,
        old_disk: Option<u32>,
    ) {
        if old_phys.is_none() && old_disk.is_none() {
            return;
        }
        if self.nodes[node_id].partial_suffix_refs > 0 {
            self.nodes[node_id]
                .pending_displaced_partials
                .push((old_phys, old_disk));
        } else {
            if let Some(b) = old_phys {
                self.orphaned_phys.push(b);
            }
            if let Some(d) = old_disk {
                self.orphaned_disk.push(d);
            }
        }
    }

    pub(super) fn drain_displaced_partials(&mut self) -> (Vec<u32>, Vec<u32>) {
        (
            std::mem::take(&mut self.orphaned_phys),
            std::mem::take(&mut self.orphaned_disk),
        )
    }

    /// Reconstruct up to `max_candidates` live cached token sequences diverging
    /// at the deepest full-block match of `tokens`, ordered by `last_access`
    /// descending so recent multi-turn sessions are probed first.
    pub(super) fn candidate_cached_sequences(
        &self,
        tokens: &[u32],
        block_size: usize,
        max_candidates: usize,
    ) -> (usize, Vec<Vec<u32>>) {
        if block_size == 0 || max_candidates == 0 {
            return (0, Vec::new());
        }
        let mut current = self.root;
        let mut parent_ctx_hash: u64 = 0;
        let mut matched_full_blocks = 0usize;
        let num_full_blocks = tokens.len() / block_size;

        for i in 0..num_full_blocks {
            let chunk = &tokens[i * block_size..(i + 1) * block_size];
            let expected_hash = context_hash_combine(parent_ctx_hash, chunk);
            match self.nodes[current].children.get(chunk) {
                Some(&child)
                    if self.nodes[child].context_hash == expected_hash
                        && self.nodes[child].ref_count > 0 =>
                {
                    matched_full_blocks += 1;
                    parent_ctx_hash = expected_hash;
                    current = child;
                }
                _ => break,
            }
        }

        let matched_prefix_len = matched_full_blocks * block_size;
        let base_prefix = &tokens[..matched_prefix_len];

        let mut child_branches: Vec<(&[u32], NodeId, u64)> = self.nodes[current]
            .children
            .iter()
            .filter_map(|(chunk, &child_id)| {
                let expected = context_hash_combine(parent_ctx_hash, chunk);
                let node = &self.nodes[child_id];
                (node.context_hash == expected && node.ref_count > 0).then_some((
                    chunk.as_slice(),
                    child_id,
                    node.last_access,
                ))
            })
            .collect();
        child_branches.sort_by(|a, b| b.2.cmp(&a.2));
        child_branches.truncate(max_candidates);

        let mut out = Vec::with_capacity(child_branches.len() + 1);
        for (first_chunk, start_id, _) in child_branches {
            let mut seq = Vec::with_capacity(base_prefix.len() + block_size * 4);
            seq.extend_from_slice(base_prefix);
            seq.extend_from_slice(first_chunk);
            let mut cursor = start_id;
            let mut cursor_hash = self.nodes[cursor].context_hash;
            loop {
                let next_best = self.nodes[cursor]
                    .children
                    .iter()
                    .filter_map(|(chunk, &cid)| {
                        let expected = context_hash_combine(cursor_hash, chunk);
                        let n = &self.nodes[cid];
                        (n.context_hash == expected && n.ref_count > 0).then_some((
                            chunk.as_slice(),
                            cid,
                            n.last_access,
                        ))
                    })
                    .max_by_key(|&(_, _, access)| access);
                if let Some((chunk, cid, _)) = next_best {
                    seq.extend_from_slice(chunk);
                    cursor = cid;
                    cursor_hash = self.nodes[cursor].context_hash;
                } else {
                    if let Some((ref partial_toks, _, _)) = self.nodes[cursor].partial_suffix {
                        seq.extend_from_slice(partial_toks);
                    }
                    break;
                }
            }
            out.push(seq);
        }

        if let Some((ref partial_toks, _, _)) = self.nodes[current].partial_suffix
            && matched_prefix_len > 0
            && out.len() < max_candidates
        {
            let mut seq = Vec::with_capacity(base_prefix.len() + partial_toks.len());
            seq.extend_from_slice(base_prefix);
            seq.extend_from_slice(partial_toks);
            out.push(seq);
        }

        (matched_prefix_len, out)
    }

    /// Insert blocks into the tree. Skips blocks that already exist.
    /// Snapshot storage is handled externally by SsmSnapshotIndex.
    ///
    /// `matched_tokens` is the prefix length the inserting sequence already
    /// acquired via `lookup()`'s `inc_refs` (those nodes' ref_count was
    /// bumped by walk and will be balanced by the eventual `release`). For
    /// blocks past that offset the inserting sequence holds no walk-ref but
    /// `release` will still dec_ref them — so we pre-bump here to keep the
    /// cache's baseline ref (=1) alive across the release. Without this the
    /// very first sequence to cache a prompt immediately renders its own
    /// cache entry un-findable on exit (`walk` requires `ref_count > 0`).
    pub(super) fn insert(
        &mut self,
        tokens: &[u32],
        block_table: &[u32],
        disk_block_ids: &[u32],
        block_size: usize,
        matched_tokens: usize,
    ) -> Vec<u32> {
        self.insert_with_spans(
            tokens,
            block_table,
            disk_block_ids,
            block_size,
            matched_tokens,
            &[],
        )
    }

    pub(super) fn insert_with_spans(
        &mut self,
        tokens: &[u32],
        block_table: &[u32],
        disk_block_ids: &[u32],
        block_size: usize,
        matched_tokens: usize,
        noncausal_spans: &[crate::prefix_cache::NonCausalSpan],
    ) -> Vec<u32> {
        let access = self.next_access();
        let mut current = self.root;
        let mut parent_ctx_hash: u64 = 0; // root context hash
        let num_full_blocks = tokens.len() / block_size;
        let num_blocks = num_full_blocks.min(block_table.len());
        let hss_active = !disk_block_ids.is_empty();
        debug_assert!(
            !hss_active || disk_block_ids.len() == block_table.len(),
            "disk_block_ids length mismatch: {} vs {}",
            disk_block_ids.len(),
            block_table.len(),
        );

        let mut newly_acquired: Vec<u32> = Vec::new();

        for i in 0..num_blocks {
            let chunk = &tokens[i * block_size..(i + 1) * block_size];
            let ctx_hash = context_hash_combine(parent_ctx_hash, chunk);
            let token_start = i * block_size;
            let is_seq_owned = token_start >= matched_tokens;
            let min_commit =
                required_commit_blocks_for_spans(i, block_size, num_blocks, noncausal_spans);
            let has_noncausal = block_overlaps_noncausal_span(i, block_size, noncausal_spans);
            let disk_id = if hss_active {
                disk_block_ids[i]
            } else {
                u32::MAX
            };

            if let Some(&child) = self.nodes[current].children.get(chunk) {
                self.nodes[child].last_access = access;
                self.nodes[child].context_hash = ctx_hash;
                if min_commit > self.nodes[child].min_commit_blocks {
                    self.nodes[child].min_commit_blocks = min_commit;
                }
                self.nodes[child].has_noncausal |= has_noncausal;
                if self.nodes[child].ref_count == 0 {
                    self.nodes[child].ref_count = 1; // restore cache's ref
                }
                if is_seq_owned {
                    self.nodes[child].ref_count += 1;
                }
                if hss_active && self.nodes[child].disk_block_id == u32::MAX && disk_id != u32::MAX
                {
                    self.nodes[child].disk_block_id = disk_id;
                    newly_acquired.push(disk_id);
                }
                parent_ctx_hash = ctx_hash;
                current = child;
            } else {
                if let Some((_, old_block, old_disk)) = self.nodes[current].partial_suffix.take() {
                    let disp_phys = (old_block != block_table[i]).then_some(old_block);
                    let disp_disk =
                        (old_disk != u32::MAX && old_disk != disk_id).then_some(old_disk);
                    self.record_displaced_partial(current, disp_phys, disp_disk);
                }
                let node = RadixNode {
                    children: HashMap::new(),
                    block_idx: block_table[i],
                    disk_block_id: disk_id,
                    context_hash: ctx_hash,
                    ref_count: if is_seq_owned { 2 } else { 1 },
                    last_access: access,
                    parent: Some(current),
                    parent_key: Some(chunk.to_vec()),
                    partial_suffix: None,
                    partial_suffix_refs: 0,
                    pending_displaced_partials: Vec::new(),
                    min_commit_blocks: min_commit,
                    has_noncausal,
                    qsa_populated: false,
                };
                let child_id = self.alloc_node(node);
                self.nodes[current]
                    .children
                    .insert(chunk.to_vec(), child_id);
                if hss_active && disk_id != u32::MAX {
                    newly_acquired.push(disk_id);
                }
                parent_ctx_hash = ctx_hash;
                current = child_id;
            }
        }

        let remainder = tokens.len() % block_size;
        let partial_start = num_full_blocks * block_size;
        let partial_touches_noncausal = noncausal_spans
            .iter()
            .any(|s| s.end > s.start && tokens.len() > s.start && partial_start < s.end);
        if remainder > 0
            && !partial_touches_noncausal
            && block_table.len() > num_full_blocks
            && current != self.root
        {
            let partial_toks = tokens[partial_start..].to_vec();
            let partial_block = block_table[num_full_blocks];
            let partial_disk = if hss_active && disk_block_ids.len() > num_full_blocks {
                disk_block_ids[num_full_blocks]
            } else {
                u32::MAX
            };
            let prior = self.nodes[current].partial_suffix.replace((
                partial_toks,
                partial_block,
                partial_disk,
            ));
            let prior_disk = prior.as_ref().map(|p| p.2);
            if let Some((_, old_block, old_disk)) = prior {
                let disp_phys = (old_block != partial_block).then_some(old_block);
                let disp_disk =
                    (old_disk != u32::MAX && old_disk != partial_disk).then_some(old_disk);
                self.record_displaced_partial(current, disp_phys, disp_disk);
            }
            if hss_active && partial_disk != u32::MAX && prior_disk != Some(partial_disk) {
                newly_acquired.push(partial_disk);
            }
        }

        newly_acquired
    }

    /// Evict up to `num_blocks` cached blocks (reclaiming orphaned displaced
    /// partial-suffix blocks first, then LRU leaf nodes at baseline refcount
    /// `ref_count <= 1` with no active borrowers).
    ///
    /// Evicting a leaf that also owns a `partial_suffix` frees both its full
    /// block and its partial block (so up to `num_blocks + 1` physical blocks
    /// may be returned). Returns physical block indices that were freed plus
    /// disk-block IDs (Phase 6.1.e). When HSS isn't in use, `u32::MAX`
    /// sentinels are filtered out by the public-trait wrapper into
    /// `EvictedBlocks::disk_block_ids`.
    pub(super) fn evict(&mut self, num_blocks: usize) -> (Vec<u32>, Vec<u32>) {
        let mut freed_phys = Vec::new();
        let mut freed_disk = Vec::new();
        if num_blocks == 0 {
            return (freed_phys, freed_disk);
        }

        // First reclaim any orphaned blocks/disk refs displaced when a
        // `partial_suffix` was cleared or overwritten on a prior insert.
        while freed_phys.len() < num_blocks {
            let Some(phys) = self.orphaned_phys.pop() else {
                break;
            };
            freed_phys.push(phys);
        }
        if !self.orphaned_disk.is_empty() {
            freed_disk.append(&mut self.orphaned_disk);
        }

        loop {
            if freed_phys.len() >= num_blocks {
                break;
            }

            let mut best: Option<(NodeId, u64)> = None;
            for (id, node) in self.nodes.iter().enumerate() {
                if id == self.root {
                    continue;
                }
                if node.ref_count <= 1
                    && node.partial_suffix_refs == 0
                    && node.children.is_empty()
                    && node.block_idx != u32::MAX
                {
                    match best {
                        None => best = Some((id, node.last_access)),
                        Some((_, best_access)) if node.last_access < best_access => {
                            best = Some((id, node.last_access));
                        }
                        _ => {}
                    }
                }
            }

            match best {
                Some((node_id, _)) => {
                    let block = self.nodes[node_id].block_idx;
                    let disk = self.nodes[node_id].disk_block_id;
                    freed_phys.push(block);
                    freed_disk.push(disk);

                    if let Some((_, partial_block, partial_disk)) =
                        self.nodes[node_id].partial_suffix.take()
                    {
                        freed_phys.push(partial_block);
                        freed_disk.push(partial_disk);
                    }
                    let deferred =
                        std::mem::take(&mut self.nodes[node_id].pending_displaced_partials);
                    for (phys, d) in deferred {
                        if let Some(b) = phys {
                            freed_phys.push(b);
                        }
                        if let Some(did) = d {
                            freed_disk.push(did);
                        }
                    }

                    if let Some(parent_id) = self.nodes[node_id].parent
                        && let Some(key) = self.nodes[node_id].parent_key.clone()
                    {
                        self.nodes[parent_id].children.remove(&key);
                    }

                    self.nodes[node_id].block_idx = u32::MAX;
                    self.nodes[node_id].disk_block_id = u32::MAX;
                    self.nodes[node_id].children.clear();
                    self.nodes[node_id].parent = None;
                    self.nodes[node_id].parent_key = None;
                    self.nodes[node_id].partial_suffix = None;
                    self.nodes[node_id].partial_suffix_refs = 0;
                    self.nodes[node_id].qsa_populated = false;
                    self.free_nodes.push(node_id);
                }
                None => break,
            }
        }

        (freed_phys, freed_disk)
    }

    pub(super) fn num_entries(&self) -> usize {
        // Count non-root, non-deleted nodes.
        self.nodes
            .iter()
            .enumerate()
            .filter(|(id, n)| *id != self.root && n.block_idx != u32::MAX)
            .count()
    }
}
