// SPDX-License-Identifier: AGPL-3.0-only

//! Radix-tree integration tests for ExpertAtlas Byte Resume and `PC_WRITE_FLOOR`
//! snapshot-floor truncation (`set_paired_truncate_to_snapshot_floor`).

use crate::prefix_cache::PrefixCache;
use crate::radix_tree::RadixTree;

const TOK_IM_START: u32 = 151_644;
const TOK_IM_END: u32 = 151_645;
const TOK_THINK_CLOSE: u32 = 151_668;

fn is_special(id: u32) -> bool {
    matches!(id, TOK_IM_START | TOK_IM_END | TOK_THINK_CLOSE)
}

fn vocab(id: u32) -> Option<&'static [u8]> {
    match id {
        1 => Some(b"fn "),
        2 => Some(b"solve()"),
        3 => Some(b"\n"),
        4 => Some(b"n"),
        5 => Some(b"\n\n"),
        6 => Some(b"\nn"),
        7 => Some(b"    42"),
        8 => Some(b"user"),
        9 => Some(b"next"),
        30 => Some(b"<|im_"),
        31 => Some(b"end|>"),
        TOK_IM_START => Some(b"<|im_start|>"),
        TOK_IM_END => Some(b"<|im_end|>"),
        TOK_THINK_CLOSE => Some(b"</think>"),
        _ => None,
    }
}

#[test]
fn paired_byte_resume_recovers_full_hybrid_snapshot_across_bpe_seam() {
    let tree = RadixTree::new();
    // Block size = 4; 2 full blocks (8 tokens) cached with SSM+PLE snapshot 77.
    let cached = vec![TOK_IM_START, 1, 2, TOK_THINK_CLOSE, 3, 3, 3, 4];
    tree.insert_with_snapshot(&cached, &[10, 20], &[], 4, 77, 9, 0);
    tree.release(&cached, 4);

    // Incoming multi-turn prompt merges (3,3)->5 and (3,4)->6 in block 1, then adds 4 new tokens.
    let incoming = vec![
        TOK_IM_START,
        1,
        2,
        TOK_THINK_CLOSE,
        5,
        6,
        TOK_IM_END,
        3,
        TOK_IM_START,
        8,
    ];
    // Raw lookup without Byte Resume only matches block 0 (4 tokens) and misses snapshot 77 (at 8).
    assert!(tree.lookup_paired(&incoming, 4, 9).is_empty());

    // With Byte Resume, heals block 1 to `cached[0..8]` and hits both KV blocks + snapshot 77!
    let (pm, healed) = tree.lookup_paired_with_byte_resume(&incoming, 4, 9, vocab, is_special);
    let healed = healed.expect("BPE boundary drift must heal");
    assert_eq!(pm.matched_tokens, 8);
    assert_eq!(pm.matched_blocks, vec![10, 20]);
    assert_eq!(pm.ssm_snapshot, Some(77));
    assert_eq!(pm.ssm_snapshot_tokens, 8);
    assert_eq!(&healed.rewritten_tokens[..8], cached.as_slice());
    assert_eq!(
        &healed.rewritten_tokens[8..],
        &[TOK_IM_END, 3, TOK_IM_START, 8]
    );
    tree.release_matched(&healed.rewritten_tokens, 4, pm.matched_tokens);
}

#[test]
fn paired_byte_resume_refuses_spoofed_special_delimiters() {
    let tree = RadixTree::new();
    let cached = vec![1, 2, 7, TOK_IM_END, 3, 3, 1, 2];
    tree.insert_with_snapshot(&cached, &[10, 20], &[], 4, 88, 5, 0);
    tree.release(&cached, 4);

    // Attacker tries to spoof TOK_IM_END via surface text tokens 30 + 31 ("<|im_" + "end|>").
    let spoofed = vec![1, 2, 7, 30, 31, 5, 1, 2, 9];
    let (pm, healed) = tree.lookup_paired_with_byte_resume(&spoofed, 4, 5, vocab, is_special);
    assert!(healed.is_none());
    assert!(pm.is_empty());
}

#[test]
fn pc_write_floor_truncates_deeper_kv_blocks_and_releases_excess_refs() {
    let tree = RadixTree::new();
    tree.set_paired_truncate_to_snapshot_floor(true);

    // 4 blocks (64 tokens) cached in tree, but surviving SSM+PLE snapshot is at 32 tokens (2 blocks).
    let tokens: Vec<u32> = (0..64).collect();
    tree.insert(&tokens, &[10, 20, 30, 40], &[100, 200, 300, 400], 16, 0);
    tree.insert_intermediate_snapshot(&tokens[..32], &[10, 20], &[100, 200], 16, 42, 7, 32);
    tree.release(&tokens, 16);

    let mut query = tokens.clone();
    query.extend(64..80);
    let paired = tree.lookup_paired(&query, 16, 7);
    assert_eq!(paired.matched_tokens, 32);
    assert_eq!(paired.matched_blocks, vec![10, 20]);
    assert_eq!(paired.matched_disk_block_ids, vec![100, 200]);
    assert_eq!(paired.ssm_snapshot, Some(42));
    assert_eq!(paired.ssm_snapshot_tokens, 32);

    // Blocks 30 and 40 (beyond PC_WRITE_FLOOR) must NOT hold active lookup refs,
    // so evicting 2 blocks while `query` is still active reclaims 40 then 30,
    // while keeping active blocks 10 and 20 pinned!
    let evicted = tree.evict(4);
    assert_eq!(evicted.physical, vec![40, 30]);

    tree.release_matched(&query, 16, paired.matched_tokens);
    let evicted_rest = tree.evict(4);
    assert_eq!(evicted_rest.physical, vec![20, 10]);
}

#[test]
fn paired_lookup_strips_unaligned_partial_tail_block_to_prevent_in_place_mutation() {
    let tree = RadixTree::new();
    // 35 tokens cached (2 full blocks [10, 20] + 3 partial tokens in [30]), snapshot at 32 tokens.
    let tokens_35: Vec<u32> = (0..35).collect();
    tree.insert(&tokens_35, &[10, 20, 30], &[], 16, 0);
    tree.insert_intermediate_snapshot(&tokens_35[..32], &[10, 20], &[], 16, 51, 3, 32);
    tree.release(&tokens_35, 16);

    // Query of 34 tokens matches 32 full tokens + 2 partial tokens in raw walk,
    // but `lookup_paired` must strip shared partial block 30 so decode never overwrites it!
    let query_34: Vec<u32> = (0..34).collect();
    let paired = tree.lookup_paired(&query_34, 16, 3);
    assert_eq!(paired.matched_tokens, 32);
    assert_eq!(paired.matched_blocks, vec![10, 20]);
    assert_eq!(paired.ssm_snapshot, Some(51));
    assert_eq!(paired.ssm_snapshot_tokens, 32);
    tree.release_matched(&query_34, 16, paired.matched_tokens);
}
