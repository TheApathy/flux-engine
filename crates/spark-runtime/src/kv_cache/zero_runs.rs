// SPDX-License-Identifier: AGPL-3.0-only

//! Coalesced KV block zeroing plan (`ATLAS_PREFILL_KV_ZERO_COALESCE`, K8 item A).
//!
//! `PagedKvCache::zero_block` issues two `cuMemsetD8Async` per attention layer per
//! block (K, then V). A 16384-row prefill chunk allocates 1024 blocks of 16 tokens,
//! so the serve recipe (12 attention layers) issues 24,576 memsets per chunk; at
//! ~2.9 us of host issue each that is the 70-90 ms `blocks` phase of the 12-e1
//! census (`PREFILL_PHASES ... blocks=`). Block `b` of a pool occupies bytes
//! `[b * stride, (b + 1) * stride)`, so a run of consecutive block indices is one
//! contiguous byte range per pool and the same bytes can be zeroed with one memset
//! per (layer, side, run). Every memset writes the value 0 on the same stream, so
//! the device state after the coalesced plan equals the state after the per-block
//! plan whatever the issue order (tests below simulate both on byte arrays).
//!
//! Dependency-free on purpose: `rustc --edition 2024 --test zero_runs.rs` runs the
//! unit tests without cargo.

/// A maximal run of consecutive block indices `first .. first + count`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockRun {
    pub first: u32,
    pub count: u32,
}

impl BlockRun {
    /// One past the last block of the run, widened so `u32::MAX` cannot wrap.
    pub fn end(&self) -> u64 {
        u64::from(self.first) + u64::from(self.count)
    }
}

/// Sorted, de-duplicated maximal runs covering exactly the blocks in `blocks`.
pub fn block_runs(blocks: &[u32]) -> Vec<BlockRun> {
    let mut sorted = blocks.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let mut runs: Vec<BlockRun> = Vec::new();
    for block in sorted {
        match runs.last_mut() {
            Some(run) if run.end() == u64::from(block) => run.count += 1,
            _ => runs.push(BlockRun {
                first: block,
                count: 1,
            }),
        }
    }
    runs
}

/// One memset: `pool` = 2 * layer + side (0 = K, 1 = V), byte `offset`, byte length.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ZeroSpan {
    pub pool: usize,
    pub offset: usize,
    pub bytes: usize,
}

/// The memsets `PagedKvCache::zero_block` issues for `blocks`, in its order:
/// block-major, then layer, K before V. `strides[layer] = (k_stride, v_stride)`.
/// Reference plan for the tests and the micro.
pub fn per_block_plan(blocks: &[u32], strides: &[(usize, usize)]) -> Option<Vec<ZeroSpan>> {
    let mut plan = Vec::with_capacity(blocks.len() * strides.len() * 2);
    for &block in blocks {
        for (layer, &(k_stride, v_stride)) in strides.iter().enumerate() {
            plan.push(ZeroSpan {
                pool: 2 * layer,
                offset: (block as usize).checked_mul(k_stride)?,
                bytes: k_stride,
            });
            plan.push(ZeroSpan {
                pool: 2 * layer + 1,
                offset: (block as usize).checked_mul(v_stride)?,
                bytes: v_stride,
            });
        }
    }
    Some(plan)
}

/// The coalesced plan: layer-major, K then V, one memset per run. `None` on
/// arithmetic overflow (the caller fails closed).
pub fn coalesced_plan(runs: &[BlockRun], strides: &[(usize, usize)]) -> Option<Vec<ZeroSpan>> {
    let mut plan = Vec::with_capacity(runs.len() * strides.len() * 2);
    for (layer, &(k_stride, v_stride)) in strides.iter().enumerate() {
        for (pool, stride) in [(2 * layer, k_stride), (2 * layer + 1, v_stride)] {
            for run in runs {
                plan.push(ZeroSpan {
                    pool,
                    offset: (run.first as usize).checked_mul(stride)?,
                    bytes: (run.count as usize).checked_mul(stride)?,
                });
            }
        }
    }
    Some(plan)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// xorshift64*: deterministic, dependency-free.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    /// Byte image of every pool: `pools[2 * layer + side]`, `num_blocks * stride` bytes,
    /// filled with a non-zero pattern so a missed or extra zero shows.
    fn pools(num_blocks: usize, strides: &[(usize, usize)], seed: u64) -> Vec<Vec<u8>> {
        let mut rng = Rng(seed | 1);
        let mut out = Vec::new();
        for &(k, v) in strides {
            for stride in [k, v] {
                out.push(
                    (0..num_blocks * stride)
                        .map(|_| 1 + (rng.next() % 255) as u8)
                        .collect(),
                );
            }
        }
        out
    }

    fn apply(pools: &mut [Vec<u8>], plan: &[ZeroSpan]) {
        for span in plan {
            pools[span.pool][span.offset..span.offset + span.bytes].fill(0);
        }
    }

    fn equivalent(blocks: &[u32], num_blocks: usize, strides: &[(usize, usize)], seed: u64) {
        let runs = block_runs(blocks);
        let mut a = pools(num_blocks, strides, seed);
        let mut b = a.clone();
        apply(&mut a, &per_block_plan(blocks, strides).unwrap());
        apply(&mut b, &coalesced_plan(&runs, strides).unwrap());
        assert_eq!(a, b, "blocks={blocks:?}");
        // Every listed block is zero on every pool; every other block is untouched.
        let fresh = pools(num_blocks, strides, seed);
        for (pool, image) in b.iter().enumerate() {
            let stride = if pool % 2 == 0 {
                strides[pool / 2].0
            } else {
                strides[pool / 2].1
            };
            for block in 0..num_blocks {
                let bytes = &image[block * stride..(block + 1) * stride];
                if blocks.contains(&(block as u32)) {
                    assert!(bytes.iter().all(|&x| x == 0));
                } else {
                    assert_eq!(bytes, &fresh[pool][block * stride..(block + 1) * stride]);
                }
            }
        }
    }

    #[test]
    fn runs_merge_sort_and_dedup() {
        assert_eq!(block_runs(&[]), vec![]);
        assert_eq!(block_runs(&[5]), vec![BlockRun { first: 5, count: 1 }]);
        assert_eq!(
            block_runs(&[3, 1, 2]),
            vec![BlockRun { first: 1, count: 3 }]
        );
        assert_eq!(
            block_runs(&[1, 3]),
            vec![
                BlockRun { first: 1, count: 1 },
                BlockRun { first: 3, count: 1 }
            ]
        );
        // The free list pops descending after a free_blocks(table) push.
        assert_eq!(
            block_runs(&[9, 8, 7]),
            vec![BlockRun { first: 7, count: 3 }]
        );
        assert_eq!(
            block_runs(&[4, 4, 5]),
            vec![BlockRun { first: 4, count: 2 }]
        );
        assert_eq!(
            block_runs(&[u32::MAX, u32::MAX - 1]),
            vec![BlockRun {
                first: u32::MAX - 1,
                count: 2
            }]
        );
        assert_eq!(block_runs(&[u32::MAX, u32::MAX - 1])[0].end(), 1u64 << 32);
    }

    #[test]
    fn coalesced_plan_shape() {
        let strides = [(16384, 16384); 12];
        let blocks: Vec<u32> = (0..1024).collect();
        let runs = block_runs(&blocks);
        let plan = coalesced_plan(&runs, &strides).unwrap();
        // The serve recipe's 16K chunk on a fresh cache: 24 memsets instead of 24,576.
        assert_eq!(plan.len(), 24);
        assert_eq!(per_block_plan(&blocks, &strides).unwrap().len(), 24_576);
        assert!(
            plan.iter()
                .all(|s| s.offset == 0 && s.bytes == 1024 * 16384)
        );
        let total: usize = plan.iter().map(|s| s.bytes).sum();
        let reference: usize = per_block_plan(&blocks, &strides)
            .unwrap()
            .iter()
            .map(|s| s.bytes)
            .sum();
        assert_eq!(total, reference);
    }

    #[test]
    fn overflow_fails_closed() {
        let runs = [BlockRun {
            first: u32::MAX,
            count: 1,
        }];
        assert!(coalesced_plan(&runs, &[(usize::MAX / 2, 1)]).is_none());
        assert!(per_block_plan(&[u32::MAX], &[(usize::MAX / 2, 1)]).is_none());
    }

    #[test]
    fn equivalent_on_handwritten_lists() {
        let strides = [(3, 5), (4, 4), (2, 7)];
        for blocks in [
            vec![],
            vec![0],
            vec![0, 1, 2, 3],
            vec![7, 6, 5, 4],
            vec![1, 3, 5, 7],
            vec![2, 3, 9, 10, 11, 0],
            vec![15],
        ] {
            equivalent(&blocks, 16, &strides, 0xA5A5);
        }
    }

    #[test]
    fn equivalent_on_random_lists() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for case in 0..400 {
            let num_blocks = 1 + rng.below(96) as usize;
            let layers = 1 + rng.below(4) as usize;
            let strides: Vec<(usize, usize)> = (0..layers)
                .map(|_| (1 + rng.below(9) as usize, 1 + rng.below(9) as usize))
                .collect();
            let mut perm: Vec<u32> = (0..num_blocks as u32).collect();
            for i in (1..perm.len()).rev() {
                perm.swap(i, rng.below(i as u64 + 1) as usize);
            }
            perm.truncate(rng.below(num_blocks as u64 + 1) as usize);
            equivalent(&perm, num_blocks, &strides, case);
        }
    }

    /// The allocator's free-list order: `PagedKvCache::new` pushes `(0..n).rev()`, `alloc` pops,
    /// `free_blocks(table)` pushes in table order, and a prefix cache keeps some blocks.
    #[test]
    fn allocator_sequences_stay_equivalent_and_coalesce() {
        let n = 512usize;
        let mut free: Vec<u32> = (0..n as u32).rev().collect();
        let mut rng = Rng(7);
        let strides = [(2, 2), (3, 1)];
        let mut held: Vec<u32> = Vec::new();
        for request in 0..60 {
            let want = 1 + rng.below(96) as usize;
            if free.len() < want {
                free.append(&mut held);
            }
            let table: Vec<u32> = (0..want).map(|_| free.pop().unwrap()).collect();
            equivalent(&table, n, &strides, request);
            let runs = block_runs(&table);
            assert!(runs.len() <= table.len());
            // A prefix cache keeps every other request's first quarter.
            let keep = if request % 2 == 0 { want / 4 } else { 0 };
            held.extend_from_slice(&table[..keep]);
            for &b in &table[keep..] {
                free.push(b);
            }
        }
        // Fresh cache: pops 0, 1, 2, ... -> one run.
        let mut fresh_list: Vec<u32> = (0..n as u32).rev().collect();
        let fresh: Vec<u32> = (0..64).map(|_| fresh_list.pop().unwrap()).collect();
        assert_eq!(fresh[..3], [0, 1, 2]);
        assert_eq!(
            block_runs(&fresh),
            vec![BlockRun {
                first: 0,
                count: 64
            }]
        );
    }
}
