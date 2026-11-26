// SPDX-License-Identifier: AGPL-3.0-only

//! Pure-CPU unit tests for multi-rank (TP2/EP2) hybrid prefix-cache restore
//! agreement (`agree_restore` + `restored_prefix_skip_tokens` + `RadixTree`
//! shallower-boundary probe).

use super::prefix_lookup::{Agreed, agree_restore};
use super::{restored_prefix_skip_tokens, should_restore_ssm_snapshot};
use spark_runtime::prefix_cache::PrefixCache;
use spark_runtime::radix_tree::RadixTree;

type ProbeFn<'a> = Box<dyn FnOnce(usize) -> Option<usize> + 'a>;

/// Simulate lockstep execution of `agree_restore` across `N` ranks, asserting
/// that every rank issues the exact same number of collectives and arrives at
/// the exact same agreed `ssm_snapshot_tokens` (`marconi_skip_to`).
fn run_lockstep_ranks<'a, const N: usize>(
    local_depths: [usize; N],
    probes: [ProbeFn<'a>; N],
) -> ([(usize, Agreed); N], usize) {
    let round1_min = local_depths
        .iter()
        .map(|&d| u32::try_from(d).unwrap())
        .min()
        .unwrap();

    // Compute what each rank's second reduction input will be if round1_min > 0.
    let mut probe_opts: [Option<ProbeFn<'a>>; N] = probes.map(Some);
    let mut round2_inputs = [0u32; N];
    let mut cached_probe_res: [Option<Option<usize>>; N] = [None; N];
    if round1_min > 0 {
        let at = round1_min as usize;
        for r in 0..N {
            if local_depths[r] == at {
                round2_inputs[r] = 1;
            } else {
                let p = probe_opts[r].take().unwrap()(at);
                round2_inputs[r] = u32::from(p.is_some());
                cached_probe_res[r] = Some(p);
            }
        }
    }
    let round2_min = round2_inputs.into_iter().min().unwrap_or(0);

    let mut out = [(0usize, Agreed::None); N];
    let mut collective_counts = [0usize; N];

    for r in 0..N {
        let mut step = 0usize;
        let precomputed = cached_probe_res[r];
        let fallback_probe = probe_opts[r].take();
        let res = agree_restore(
            local_depths[r],
            |val| {
                let idx = step;
                step += 1;
                match idx {
                    0 => {
                        assert_eq!(val, u32::try_from(local_depths[r]).unwrap());
                        Ok(round1_min)
                    }
                    1 => {
                        assert_eq!(val, round2_inputs[r]);
                        Ok(round2_min)
                    }
                    _ => panic!("unexpected collective {idx} on rank {r}"),
                }
            },
            |at| {
                if let Some(res) = precomputed {
                    res
                } else {
                    fallback_probe.expect("probe must only run when agreed < depth")(at)
                }
            },
        )
        .unwrap();
        out[r] = res;
        collective_counts[r] = step;
    }

    for r in 1..N {
        assert_eq!(
            collective_counts[r], collective_counts[0],
            "collective count mismatch between rank 0 and rank {r}"
        );
        assert_eq!(
            out[r].0, out[0].0,
            "agreed restore depth mismatch between rank 0 and rank {r}"
        );
    }

    (out, collective_counts[0])
}

#[test]
fn divergent_intermediate_snapshots_16k_vs_8k_on_24k_kv_hit_agree_on_8k() {
    // Reproduces the live TP2/EP2 deadlock scenario:
    // Both ranks hit 24K KV tokens on a 32K prompt, Rank 0's deepest SSM
    // snapshot is at 16K, Rank 1's deepest is at 8K, and Rank 0 still holds
    // its 8K intermediate snapshot (id = 42).
    let matched = 24_576usize;
    let total = 32_768usize;
    assert!(should_restore_ssm_snapshot(16_384, matched, total));
    assert!(should_restore_ssm_snapshot(8_192, matched, total));

    let (results, collectives) = run_lockstep_ranks(
        [16_384, 8_192],
        [
            Box::new(|at| {
                assert_eq!(at, 8_192);
                Some(42)
            }),
            Box::new(|_at| panic!("rank at min depth must use Agreed::Local without probing")),
        ],
    );

    assert_eq!(collectives, 2);
    assert_eq!(results[0], (8_192, Agreed::At(42)));
    assert_eq!(results[1], (8_192, Agreed::Local));

    let skip0 = restored_prefix_skip_tokens(true, results[0].0, matched);
    let skip1 = restored_prefix_skip_tokens(true, results[1].0, matched);
    assert_eq!(skip0, 8_192);
    assert_eq!(skip1, 8_192);
}

#[test]
fn divergent_snapshots_fall_back_to_zero_when_shallower_snapshot_was_evicted() {
    // Rank 0 has a 16K snapshot but evicted its 8K snapshot; Rank 1 only has
    // the 8K snapshot. Both must fall back cleanly to 0 so neither restores
    // and both compute identical proc_count from token 0.
    let matched = 24_576usize;
    let (results, collectives) = run_lockstep_ranks(
        [16_384, 8_192],
        [Box::new(|_at| None), Box::new(|_at| None)],
    );

    assert_eq!(collectives, 2);
    assert_eq!(results[0], (0, Agreed::None));
    assert_eq!(results[1], (0, Agreed::None));

    for &(agreed_tok, choice) in &results {
        let skip = choice != Agreed::None;
        let skip_tokens = if skip {
            restored_prefix_skip_tokens(true, agreed_tok, matched)
        } else {
            0
        };
        assert_eq!(skip_tokens, 0);
    }
}

#[test]
fn one_rank_without_usable_snapshot_demotes_all_ranks_in_one_collective() {
    // Rank 0 has 16K snapshot, Rank 1 has 0 (e.g. session mismatch or evicted).
    // Must complete in 1 collective with zero probes and marconi_skip_to == 0.
    let (results, collectives) = run_lockstep_ranks(
        [16_384, 0],
        [
            Box::new(|_at| panic!("must not probe when agreed == 0")),
            Box::new(|_at| panic!("must not probe when agreed == 0")),
        ],
    );

    assert_eq!(collectives, 1);
    assert_eq!(results[0], (0, Agreed::None));
    assert_eq!(results[1], (0, Agreed::None));
}

#[test]
fn equal_nonzero_depths_use_local_without_probing_across_four_ranks() {
    let (results, collectives) = run_lockstep_ranks(
        [16_384, 16_384, 16_384, 16_384],
        [
            Box::new(|_at| panic!("equal depth must not probe")),
            Box::new(|_at| panic!("equal depth must not probe")),
            Box::new(|_at| panic!("equal depth must not probe")),
            Box::new(|_at| panic!("equal depth must not probe")),
        ],
    );

    assert_eq!(collectives, 2);
    for res in results {
        assert_eq!(res, (16_384, Agreed::Local));
    }
}

#[test]
fn radix_tree_probe_finds_agreed_shallower_snapshot_and_preserves_refcounts() {
    // End-to-end pure-CPU check of the exact RadixTree probe sequence used by
    // `prefill_b_prefix_lookup` on a 2-rank hybrid hit where Rank 0 holds
    // snapshots at 32 and 64 tokens while Rank 1 holds only the snapshot at 32
    // tokens on a 96-token KV hit (`bs = 16`, 128-token prompt).
    let bs = 16usize;
    let session = 0xCAFE_BABEu64;
    let prompt: Vec<u32> = (0..128u32).collect();
    let cached_blocks: Vec<u32> = (0..6u32).collect(); // 6 * 16 = 96 KV tokens

    let rank0_cache = RadixTree::new();
    let rank1_cache = RadixTree::new();

    // Seed Rank 0 with 96 KV tokens + intermediate snapshots at 32 and 64, then
    // release the seeding sequence's own refcount so only the cache's ref remains.
    rank0_cache.insert(&prompt[..96], &cached_blocks, &[], bs, 0);
    rank0_cache.insert_intermediate_snapshot(
        &prompt[..32],
        &cached_blocks[..2],
        &[],
        bs,
        10,
        session,
        0,
    );
    rank0_cache.insert_intermediate_snapshot(
        &prompt[..64],
        &cached_blocks[..4],
        &[],
        bs,
        20,
        session,
        0,
    );
    rank0_cache.release(&prompt[..96], bs);

    // Seed Rank 1 with 96 KV tokens + intermediate snapshot at 32 only.
    rank1_cache.insert(&prompt[..96], &cached_blocks, &[], bs, 0);
    rank1_cache.insert_intermediate_snapshot(
        &prompt[..32],
        &cached_blocks[..2],
        &[],
        bs,
        11,
        session,
        0,
    );
    rank1_cache.release(&prompt[..96], bs);

    let m0 = rank0_cache.lookup_paired(&prompt, bs, session);
    let m1 = rank1_cache.lookup_paired(&prompt, bs, session);
    assert_eq!(m0.matched_tokens, 96);
    assert_eq!(m1.matched_tokens, 96);
    assert_eq!(m0.ssm_snapshot, Some(20));
    assert_eq!(m0.ssm_snapshot_tokens, 64);
    assert_eq!(m1.ssm_snapshot, Some(11));
    assert_eq!(m1.ssm_snapshot_tokens, 32);

    let (results, _) = run_lockstep_ranks(
        [m0.ssm_snapshot_tokens, m1.ssm_snapshot_tokens],
        [
            Box::new(|at| {
                let probe = rank0_cache.lookup_paired(&prompt[..at], bs, session);
                if probe.matched_tokens > 0 {
                    rank0_cache.release_matched(&prompt, bs, probe.matched_tokens);
                }
                probe
                    .ssm_snapshot
                    .filter(|_| probe.ssm_snapshot_tokens == at)
            }),
            Box::new(|_at| None),
        ],
    );

    assert_eq!(results[0], (32, Agreed::At(10)));
    assert_eq!(results[1], (32, Agreed::Local));

    // Releasing the outer 96-token lookup must make all 6 blocks evictable on
    // Rank 0 (proving the inner probe released its temporary refcounts cleanly).
    rank0_cache.release_matched(&prompt, bs, m0.matched_tokens);
    let evicted = rank0_cache.evict(6);
    assert_eq!(evicted.physical.len(), 6);
}
