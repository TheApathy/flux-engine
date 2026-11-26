// SPDX-License-Identifier: AGPL-3.0-only

//! Non-causal / bidirectional span `(span.start, span.end)` interior exclusion
//! tests for `RadixTree` and `SsmSnapshotIndex`.

use crate::prefix_cache::{NonCausalSpan, PrefixCache};
use crate::radix_tree::RadixTree;

#[test]
fn noncausal_span_excludes_interior_blocks_and_commits_boundaries() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..64).collect();
    // Bidirectional span [16, 48): blocks 1 ([16..32)) and 2 ([32..48)) are non-causal.
    let spans = [NonCausalSpan::new(16, 48)];
    tree.insert_with_noncausal_spans(&tokens, &[10, 20, 30, 40], &[], 16, 0, &spans);

    // 1. Exact match at `span.start` (16 tokens) hits causal block 0.
    let at_start = tree.lookup(&tokens[..16], 16, 0);
    assert_eq!(at_start.matched_tokens, 16);
    assert_eq!(at_start.matched_blocks, vec![10]);
    tree.release_matched(&tokens[..16], 16, at_start.matched_tokens);

    // 2. Diverging inside `(16, 48)` at 32 tokens MUST roll back to `16` (`span.start`),
    //    never exposing block 1 (`[16..32)`) whose KV attended forward to `[32..48)`.
    let mut diverge_32 = tokens[..32].to_vec();
    diverge_32.extend(9000..9032);
    let m32 = tree.lookup(&diverge_32, 16, 0);
    assert_eq!(m32.matched_tokens, 16);
    assert_eq!(m32.matched_blocks, vec![10]);
    tree.release_matched(&diverge_32, 16, m32.matched_tokens);

    // 3. Sub-block query ending at 24 tokens (inside `(16, 48)`) also rolls back to 16.
    let m24 = tree.lookup(&tokens[..24], 16, 0);
    assert_eq!(m24.matched_tokens, 16);
    assert_eq!(m24.matched_blocks, vec![10]);
    tree.release_matched(&tokens[..24], 16, m24.matched_tokens);

    // 4. Matching through `span.end` (48 tokens) commits all 3 blocks.
    let mut at_end_diverge = tokens[..48].to_vec();
    at_end_diverge.extend(8000..8016);
    let m48 = tree.lookup(&at_end_diverge, 16, 0);
    assert_eq!(m48.matched_tokens, 48);
    assert_eq!(m48.matched_blocks, vec![10, 20, 30]);
    tree.release_matched(&at_end_diverge, 16, m48.matched_tokens);

    // 5. Full 64-token match commits all 4 blocks.
    let m64 = tree.lookup(&tokens, 16, 0);
    assert_eq!(m64.matched_tokens, 64);
    assert_eq!(m64.matched_blocks, vec![10, 20, 30, 40]);
    tree.release_matched(&tokens, 16, m64.matched_tokens);
}

#[test]
fn noncausal_single_block_forbids_subblock_prefix_but_allows_full_block() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..16).collect();
    let spans = [NonCausalSpan::new(0, 16)];
    tree.insert_with_noncausal_spans(&tokens, &[77], &[], 16, 0, &spans);

    // 8 tokens lies strictly inside `(0, 16)` -> must NOT match sub-block prefix!
    let sub = tree.lookup(&tokens[..8], 16, 0);
    assert_eq!(sub.matched_tokens, 0);
    assert!(sub.matched_blocks.is_empty());

    // 16 tokens closes `span.end == 16` -> full block matches!
    let full = tree.lookup(&tokens, 16, 0);
    assert_eq!(full.matched_tokens, 16);
    assert_eq!(full.matched_blocks, vec![77]);
    tree.release_matched(&tokens, 16, full.matched_tokens);
}

#[test]
fn noncausal_unaligned_span_requires_closing_full_block() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..64).collect();
    // Unaligned span [20, 44) overlaps block 1 ([16..32)) and block 2 ([32..48)).
    let spans = [NonCausalSpan::new(20, 44)];
    tree.insert_with_noncausal_spans(&tokens, &[1, 2, 3, 4], &[], 16, 0, &spans);

    // Diverging at 32 tokens (before block 2 closes 44) rolls back to block 0 (16 tokens).
    let mut div32 = tokens[..32].to_vec();
    div32.extend(500..516);
    let m32 = tree.lookup(&div32, 16, 0);
    assert_eq!(m32.matched_tokens, 16);
    assert_eq!(m32.matched_blocks, vec![1]);
    tree.release_matched(&div32, 16, m32.matched_tokens);

    // Matching 48 tokens (>= 44) closes block 2 and hits 48 tokens.
    let m48 = tree.lookup(&tokens[..48], 16, 0);
    assert_eq!(m48.matched_tokens, 48);
    assert_eq!(m48.matched_blocks, vec![1, 2, 3]);
    tree.release_matched(&tokens[..48], 16, m48.matched_tokens);
}

#[test]
fn noncausal_span_suppresses_interior_snapshots_while_keeping_endpoints() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..64).collect();
    let spans = [NonCausalSpan::new(16, 48)];

    // Insert leaf at 64 with snapshot 103 (outside span -> retained).
    let (reclaim_leaf, _) = tree.insert_with_snapshot_and_noncausal_spans(
        &tokens,
        &[10, 20, 30, 40],
        &[],
        16,
        103,
        9,
        0,
        &spans,
    );
    assert_eq!(reclaim_leaf, None);

    // Checkpoint at `span.start` (16) -> retained!
    assert_eq!(
        tree.insert_intermediate_snapshot_with_noncausal_spans(
            &tokens[..16],
            &[10],
            &[],
            16,
            100,
            9,
            0,
            &spans,
        ),
        None
    );

    // Checkpoint at 32 (strictly inside `(16, 48)`) -> suppressed & returned for reclaim!
    assert_eq!(
        tree.insert_intermediate_snapshot_with_noncausal_spans(
            &tokens[..32],
            &[10, 20],
            &[],
            16,
            101,
            9,
            0,
            &spans,
        ),
        Some(101)
    );

    // Checkpoint at `span.end` (48) -> retained!
    assert_eq!(
        tree.insert_intermediate_snapshot_with_noncausal_spans(
            &tokens[..48],
            &[10, 20, 30],
            &[],
            16,
            102,
            9,
            0,
            &spans,
        ),
        None
    );

    assert_eq!(tree.snapshot_count(), 3);

    // Diverging at 32 tokens restores paired checkpoint 100 at 16 tokens (`span.start`).
    let mut div32 = tokens[..32].to_vec();
    div32.extend(7000..7016);
    let p_div = tree.lookup_paired(&div32, 16, 9);
    assert_eq!(p_div.matched_tokens, 16);
    assert_eq!(p_div.ssm_snapshot, Some(100));
    assert_eq!(p_div.ssm_snapshot_tokens, 16);
    tree.release_matched(&div32, 16, p_div.matched_tokens);

    // Diverging after `span.end` (at 48 tokens) restores paired checkpoint 102 at 48 tokens.
    let mut div48 = tokens[..48].to_vec();
    div48.extend(7100..7116);
    let p_end = tree.lookup_paired(&div48, 16, 9);
    assert_eq!(p_end.matched_tokens, 48);
    assert_eq!(p_end.ssm_snapshot, Some(102));
    assert_eq!(p_end.ssm_snapshot_tokens, 48);
    tree.release_matched(&div48, 16, p_end.matched_tokens);
}
