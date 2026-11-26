// SPDX-License-Identifier: AGPL-3.0-only

//! Restore-ladder tests: a long agentic conversation in a small snapshot
//! pool, then a client rewrite a few thousand tokens behind the tail.

use crate::prefix_cache::PrefixCache;
use crate::radix_tree::RadixTree;

use super::super::snapshot::SsmSnapshotIndex;
use super::super::snapshot_ladder::{LADDER_PATHS, parse_ladder_base, ssm_snap_ladder_base};

const SESSION: u64 = 0xc27f;
const POOL: usize = 16;

/// Deterministic token stream; `salt` changes every token from `from` on.
fn tokens(len: usize, from: usize, salt: u32) -> Vec<u32> {
    (0..len)
        .map(|i| {
            let base = (i as u32).wrapping_mul(2_654_435_761) >> 7;
            if i >= from { base ^ salt } else { base }
        })
        .collect()
}

/// Save a snapshot at `tokens.len()` the way a warm turn does, evicting
/// with the serving victim choice whenever the pool is full.
fn save(idx: &mut SsmSnapshotIndex, toks: &[u32], next_id: &mut usize) {
    while idx.entries.len() >= POOL {
        idx.evict_lru().expect("a full pool always yields a victim");
    }
    let hash = hash_token_prefix(toks);
    idx.insert(hash, *next_id, SESSION, toks.len());
    idx.stamp_ladder_with_base(toks, SESSION, 1024);
    *next_id += 1;
}

fn hash_token_prefix(toks: &[u32]) -> u64 {
    super::super::hash_token_prefix(toks, toks.len())
}

/// Grow one conversation from 8K to 178K tokens, ~608 tokens (38 blocks of 16)
/// per turn — the OMP shape that pushed every restore point into the last few turns.
fn grown_conversation() -> (SsmSnapshotIndex, Vec<u32>, usize) {
    let mut idx = SsmSnapshotIndex::new();
    let mut id = 0;
    let full = tokens(178_000, usize::MAX, 0);
    let (mut len, mut tail) = (8_000, 0);
    while len <= full.len() {
        save(&mut idx, &full[..len], &mut id);
        tail = len;
        len += 608;
    }
    (idx, full, tail)
}

/// The production failure: the client rewrote history ~8K tokens behind the
/// tail. The KV radix still matches up to the rewrite, so the deepest
/// resident snapshot at or below it must exist and be close — not a fall
/// back to recomputing all 170K tokens.
#[test]
fn rewrite_behind_tail_finds_a_nearby_anchor() {
    let (mut idx, _, tail) = grown_conversation();
    let base = 1024usize;
    for back in [1_000usize, 3_000, 8_300, 20_000, 60_000] {
        let diverge = tail - back;
        let rewritten = tokens(tail + 300, diverge, 0x5a5a);
        let (_, hit_tokens) = idx
            .lookup(&rewritten, diverge, SESSION)
            .unwrap_or_else(|| panic!("no anchor at all {back} tokens behind the tail"));
        let replay = diverge - hit_tokens;
        // Bucket k holds a rung somewhere in [b·2^k, b·2^(k+1)) unless it
        // is refilling, so the replay is a small multiple of the rewrite.
        assert!(
            replay <= 4 * back.max(base) + 608,
            "rewrite {back} back: anchor {hit_tokens} replays {replay} tokens",
        );
    }
}

/// The ladder stays a handful of slots: most of the pool still turns over
/// by recency, so the newest restore points are always resident.
#[test]
fn ladder_leaves_recent_turns_resident() {
    let (idx, _, tail) = grown_conversation();
    let depths: Vec<usize> = idx.entries.iter().map(|e| e.token_count).collect();
    assert!(
        depths.contains(&tail),
        "the newest save must be resident: {depths:?}"
    );
    assert!(
        depths.contains(&(tail - 608)),
        "the previous turn's save must be resident: {depths:?}"
    );
    let protected = idx
        .entries
        .iter()
        .filter(|e| idx.ladder.holds(e.prefix_hash))
        .count();
    assert!(
        protected < POOL,
        "ladder must not pin the whole pool ({protected}/{POOL})"
    );
}

/// Rungs are chosen on the live path only: entries of a dead branch (an
/// earlier rewrite) are not protected, so they age out by recency.
#[test]
fn dead_branch_is_not_a_rung() {
    let mut idx = SsmSnapshotIndex::new();
    let mut id = 0;
    let live = tokens(40_000, usize::MAX, 0);
    let dead = tokens(40_000, 20_000, 0x77);
    save(&mut idx, &dead[..30_000], &mut id);
    save(&mut idx, &live[..30_000], &mut id);
    save(&mut idx, &live[..40_000], &mut id);
    let dead_hash = super::super::hash_token_prefix(&dead, 30_000);
    let live_hash = super::super::hash_token_prefix(&live, 30_000);
    assert!(
        !idx.ladder.holds(dead_hash),
        "dead branch entry must not be a rung"
    );
    assert!(
        idx.ladder.holds(live_hash),
        "live path entry 10K back must be a rung"
    );
}

/// End-to-end compatibility with `RadixTree::lookup_paired` and `PC_WRITE_FLOOR`:
/// a long agentic session with block-aligned turns keeps a nearby block-aligned
/// rung when history is rewritten 8,320 tokens behind the tail, and paired
/// lookup truncates `matched_blocks` cleanly to `snap_tok / block_size`.
#[test]
fn paired_lookup_with_pc_write_floor_uses_ladder_anchor() {
    if ssm_snap_ladder_base().is_none() {
        return; // ladder disabled by env in this process
    }
    let tree = RadixTree::new();
    tree.set_paired_truncate_to_snapshot_floor(true);
    let bs = 16usize;
    let full = tokens(65_536, usize::MAX, 0);
    let all_blocks: Vec<u32> = (0..(full.len() / bs) as u32).collect();

    let mut snap_id = 1usize;
    let mut len = 4_096usize;
    let mut tail = 0usize;
    while len <= full.len() {
        while tree.snapshot_count() >= POOL {
            assert!(tree.evict_snapshot_lru().is_some());
        }
        let n_blocks = len / bs;
        tree.insert_with_snapshot(
            &full[..len],
            &all_blocks[..n_blocks],
            &[],
            bs,
            snap_id,
            SESSION,
            0,
        );
        tree.release(&full[..len], bs);
        tail = len;
        snap_id += 1;
        len += 512;
    }

    let diverge = tail - 8_320;
    let rewritten = tokens(tail + 256, diverge, 0x33cc);
    let hit = tree.lookup_paired(&rewritten, bs, SESSION);
    assert!(
        hit.ssm_snapshot.is_some(),
        "ladder must preserve a paired anchor behind the rewrite"
    );
    assert_eq!(hit.matched_tokens, hit.ssm_snapshot_tokens);
    assert_eq!(hit.matched_blocks.len(), hit.ssm_snapshot_tokens / bs);
    let replay = diverge - hit.ssm_snapshot_tokens;
    assert!(
        replay <= 4 * 8_320 + 512,
        "paired lookup replayed too far: {replay} tokens"
    );
    tree.release_matched(&rewritten, bs, hit.matched_tokens);
}

#[test]
fn ladder_env_parsing_and_path_cap_work() {
    assert_eq!(parse_ladder_base(None), Some(1024));
    assert_eq!(parse_ladder_base(Some("2048")), Some(2048));
    assert_eq!(parse_ladder_base(Some("0")), None);
    assert_eq!(parse_ladder_base(Some("off")), None);
    assert_eq!(parse_ladder_base(Some("false")), None);
    assert_eq!(parse_ladder_base(Some("invalid")), Some(1024));

    // Only LADDER_PATHS most recent sessions keep their rungs.
    let mut idx = SsmSnapshotIndex::new();
    for s in 1..=(LADDER_PATHS as u64 + 2) {
        let stream = tokens(8_000, 0, s as u32 * 17);
        let h_early = super::super::hash_token_prefix(&stream, 2_000);
        let h_late = super::super::hash_token_prefix(&stream, 8_000);
        idx.insert(h_early, (s * 2) as usize, s, 2_000);
        idx.insert(h_late, (s * 2 + 1) as usize, s, 8_000);
        idx.stamp_ladder_with_base(&stream, s, 1024);
    }
    let first_stream = tokens(8_000, 0, 17);
    let newest_stream = tokens(8_000, 0, (LADDER_PATHS as u32 + 2) * 17);
    assert!(
        !idx.ladder
            .holds(super::super::hash_token_prefix(&first_stream, 2_000))
    );
    assert!(
        idx.ladder
            .holds(super::super::hash_token_prefix(&newest_stream, 2_000))
    );
}
