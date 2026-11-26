// SPDX-License-Identifier: AGPL-3.0-only

//! `--kv-pool-tokens`: size the shared paged-KV pool apart from
//! `max_batch_size x max_seq_len`.
//!
//! The pool is one free list shared by every slot (blocks are taken on demand
//! as a sequence grows), but by default it is sized so that every slot can
//! reach `max_seq_len` at once, and startup refuses anything smaller. With
//! `kv_pool_tokens` set the pool holds that many tokens instead; startup only
//! requires one full `max_seq_len` sequence to fit, and the scheduler's depth
//! aware admission (`prompt + min(max_tokens, watermark)`) keeps the sum of
//! the in-flight reservations inside the pool, queueing the rest.
//!
//! These functions are pure so the pre-load fit planner can call them with
//! the same arguments the factory uses.

use anyhow::{Result, bail};

/// How the target KV pool was sized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvPoolMode {
    /// Legacy: `min(budget, max_batch x (blocks_per_seq + 1) + 1)`, and
    /// startup requires every slot to fit at `max_seq_len`.
    PerSlot,
    /// `--kv-pool-tokens`: a shared pool of an explicit size, of which one
    /// full `max_seq_len` sequence is the minimum.
    Shared,
}

/// Target KV pool plan in blocks (the dummy block included).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvPoolPlan {
    pub blocks: usize,
    pub mode: KvPoolMode,
}

/// Blocks every configured slot could address at once, including one spare
/// block per sequence (slide-then-alloc) and the permanent dummy block.
pub fn reachable_blocks(max_seq_len: usize, max_batch_size: usize, block_size: usize) -> usize {
    let blocks_per_seq = max_seq_len.div_ceil(block_size.max(1));
    max_batch_size
        .saturating_mul(blocks_per_seq.saturating_add(1))
        .saturating_add(1)
}

/// Blocks a `kv_pool_tokens` request asks for before the budget check:
/// the tokens themselves, one spare block per slot and the dummy block,
/// clamped to what active slots can address unless prefix caching is active.
pub fn requested_pool_blocks(
    kv_pool_tokens: usize,
    max_seq_len: usize,
    max_batch_size: usize,
    block_size: usize,
) -> usize {
    requested_pool_blocks_with_prefix_cache(
        kv_pool_tokens,
        max_seq_len,
        max_batch_size,
        block_size,
        false,
    )
}

/// Prefix-cache-aware variant of [`requested_pool_blocks`].
///
/// When `prefix_cache_active` is `true`, blocks beyond
/// `max_batch_size × max_seq_len` remain useful as cached Radix prefix blocks
/// across requests (e.g., `4.5M–16M` KV pools on `2×–4×` Spark clusters), so
/// the request is not clamped to active-slot reachability (it is still bounded
/// by `budget_blocks` in [`plan_kv_pool_with_prefix_cache`]).
pub fn requested_pool_blocks_with_prefix_cache(
    kv_pool_tokens: usize,
    max_seq_len: usize,
    max_batch_size: usize,
    block_size: usize,
    prefix_cache_active: bool,
) -> usize {
    let wanted = kv_pool_tokens
        .div_ceil(block_size.max(1))
        .saturating_add(max_batch_size.max(1))
        .saturating_add(1);
    if prefix_cache_active {
        wanted
    } else {
        wanted.min(reachable_blocks(max_seq_len, max_batch_size, block_size))
    }
}

/// Size the target KV pool. `budget_blocks` is what the post-construction
/// memory budget can hold. `None` is the legacy rule exactly (the caller keeps
/// the legacy per-slot startup check); `Some` fails closed when the budget
/// cannot hold the requested pool or the pool cannot hold one sequence.
pub fn plan_kv_pool(
    budget_blocks: usize,
    max_seq_len: usize,
    max_batch_size: usize,
    block_size: usize,
    kv_pool_tokens: Option<usize>,
) -> Result<KvPoolPlan> {
    plan_kv_pool_with_prefix_cache(
        budget_blocks,
        max_seq_len,
        max_batch_size,
        block_size,
        kv_pool_tokens,
        false,
    )
}

/// Prefix-cache-aware variant of [`plan_kv_pool`].
pub fn plan_kv_pool_with_prefix_cache(
    budget_blocks: usize,
    max_seq_len: usize,
    max_batch_size: usize,
    block_size: usize,
    kv_pool_tokens: Option<usize>,
    prefix_cache_active: bool,
) -> Result<KvPoolPlan> {
    let reachable = reachable_blocks(max_seq_len, max_batch_size, block_size);
    let Some(tokens) = kv_pool_tokens else {
        return Ok(KvPoolPlan {
            blocks: budget_blocks.min(reachable),
            mode: KvPoolMode::PerSlot,
        });
    };
    if tokens == 0 {
        bail!("--kv-pool-tokens must be positive (omit it for the per-slot pool)");
    }
    let bs = block_size.max(1);
    let blocks = requested_pool_blocks_with_prefix_cache(
        tokens,
        max_seq_len,
        max_batch_size,
        bs,
        prefix_cache_active,
    );
    // One full sequence, its spare and the dummy: a lone request at the
    // served ceiling must always be able to run.
    let one_sequence = reachable_blocks(max_seq_len, 1, bs);
    if blocks < one_sequence {
        bail!(
            "--kv-pool-tokens {tokens} cannot hold one --max-seq-len {max_seq_len} sequence: \
             the pool would have {blocks} block(s) of {bs} tokens and one sequence needs {one_sequence} \
             (including its spare and the dummy block). Use --kv-pool-tokens {} or more, or lower --max-seq-len",
            max_seq_len.div_ceil(bs) * bs,
        );
    }
    if budget_blocks < blocks {
        let budget_tokens = budget_blocks
            .saturating_sub(max_batch_size.max(1) + 1)
            .saturating_mul(bs);
        bail!(
            "--kv-pool-tokens {tokens} needs {blocks} KV block(s) but the memory budget after model \
             construction holds {budget_blocks} (about {budget_tokens} tokens at this KV dtype). \
             Lower --kv-pool-tokens, use a smaller --kv-cache-dtype, or raise --gpu-memory-utilization"
        );
    }
    Ok(KvPoolPlan {
        blocks,
        mode: KvPoolMode::Shared,
    })
}

/// Native MTP KV pool blocks (of the MTP cache's own `block_size`). Legacy:
/// one full sequence (plus a spare block) per slot. With a shared target pool
/// of `target_pool_tokens`, MTP rows never outnumber target rows by more than
/// one block per sequence, so the target capacity plus one block per slot
/// always suffices; never more than the legacy size.
pub fn mtp_kv_pool_blocks(
    max_seq_len: usize,
    max_batch_size: usize,
    block_size: usize,
    target_pool_tokens: Option<usize>,
) -> usize {
    let bs = block_size.max(1);
    let legacy = (max_seq_len.div_ceil(bs) + 1) * max_batch_size.max(1);
    match target_pool_tokens {
        None => legacy,
        Some(tokens) => tokens
            .div_ceil(bs)
            .saturating_add(max_batch_size.max(1))
            .min(legacy),
    }
}

/// Token capacity of a shared target pool (`None` for the per-slot pool),
/// the input [`mtp_kv_pool_blocks`] takes.
pub fn shared_pool_tokens(
    kv_pool_tokens: Option<usize>,
    max_seq_len: usize,
    max_batch_size: usize,
    block_size: usize,
) -> Option<usize> {
    kv_pool_tokens.map(|tokens| {
        requested_pool_blocks(tokens, max_seq_len, max_batch_size, block_size)
            .saturating_mul(block_size.max(1))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const BS: usize = 16;
    const SEQ_512K: usize = 524_288;

    #[test]
    fn unset_is_the_legacy_min_of_budget_and_reachable() {
        // rc7 256k C1: budget 65,296 blocks capped to 16,386.
        let plan = plan_kv_pool(65_296, 262_144, 1, BS, None).unwrap();
        assert_eq!(plan.blocks, 16_386);
        assert_eq!(plan.mode, KvPoolMode::PerSlot);
        let plan = plan_kv_pool(1_000, 262_144, 1, BS, None).unwrap();
        assert_eq!(plan.blocks, 1_000);
        assert_eq!(reachable_blocks(262_144, 1, BS), 16_386);
        assert_eq!(reachable_blocks(SEQ_512K, 2, BS), 2 * 32_769 + 1);
    }

    #[test]
    fn shared_pool_can_be_smaller_than_every_slot_at_max_seq() {
        // C4 at 512K, pool for two deep sessions.
        let tokens = 2 * SEQ_512K;
        let plan = plan_kv_pool(usize::MAX, SEQ_512K, 4, BS, Some(tokens)).unwrap();
        assert_eq!(plan.mode, KvPoolMode::Shared);
        assert_eq!(plan.blocks, tokens / BS + 4 + 1);
        assert!(plan.blocks < reachable_blocks(SEQ_512K, 4, BS));
    }

    #[test]
    fn shared_pool_never_exceeds_what_the_slots_can_address() {
        let plan = plan_kv_pool(usize::MAX, 8_192, 2, BS, Some(1 << 30)).unwrap();
        assert_eq!(plan.blocks, reachable_blocks(8_192, 2, BS));
    }

    #[test]
    fn shared_pool_with_prefix_cache_honors_multi_spark_radix_capacity_within_budget() {
        // 4x Spark TP=4 + EP=4 with C=16, 256K max_seq_len, and --prefix-cache:
        // active slots address 4M tokens, while 6M tokens (393,216 blocks + 17)
        // keeps 2M extra Radix prefix blocks resident when budget allows.
        let tokens_6m = 6_291_456;
        let active_4m = reachable_blocks(262_144, 16, BS);
        let plan_no_pc =
            plan_kv_pool_with_prefix_cache(500_000, 262_144, 16, BS, Some(tokens_6m), false)
                .unwrap();
        assert_eq!(plan_no_pc.blocks, active_4m);

        let plan_pc =
            plan_kv_pool_with_prefix_cache(500_000, 262_144, 16, BS, Some(tokens_6m), true)
                .unwrap();
        assert_eq!(plan_pc.mode, KvPoolMode::Shared);
        assert_eq!(plan_pc.blocks, tokens_6m / BS + 16 + 1);
        assert!(plan_pc.blocks > active_4m);

        // Still fails closed if memory budget cannot hold the requested prefix-cache pool.
        let err = plan_kv_pool_with_prefix_cache(
            plan_pc.blocks - 1,
            262_144,
            16,
            BS,
            Some(tokens_6m),
            true,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("memory budget"), "{err}");
    }

    #[test]
    fn shared_pool_must_hold_one_full_sequence() {
        let err = plan_kv_pool(usize::MAX, SEQ_512K, 2, BS, Some(SEQ_512K - 64))
            .unwrap_err()
            .to_string();
        assert!(err.contains("cannot hold one --max-seq-len"), "{err}");
        // Exactly one sequence (C1 rounding: +1 spare, +1 dummy) is accepted.
        let plan = plan_kv_pool(usize::MAX, SEQ_512K, 1, BS, Some(SEQ_512K)).unwrap();
        assert_eq!(plan.blocks, reachable_blocks(SEQ_512K, 1, BS));
        assert!(plan_kv_pool(usize::MAX, 8_192, 1, BS, Some(0)).is_err());
    }

    #[test]
    fn shared_pool_fails_closed_when_the_budget_is_short() {
        let need = requested_pool_blocks(2 * SEQ_512K, SEQ_512K, 2, BS);
        let err = plan_kv_pool(need - 1, SEQ_512K, 2, BS, Some(2 * SEQ_512K))
            .unwrap_err()
            .to_string();
        assert!(err.contains("memory budget"), "{err}");
        assert!(plan_kv_pool(need, SEQ_512K, 2, BS, Some(2 * SEQ_512K)).is_ok());
    }

    #[test]
    fn mtp_pool_follows_the_target_pool_only_when_shared() {
        assert_eq!(mtp_kv_pool_blocks(262_144, 1, BS, None), 16_385);
        assert_eq!(mtp_kv_pool_blocks(SEQ_512K, 4, BS, None), 4 * 32_769);
        let target = shared_pool_tokens(Some(2 * SEQ_512K), SEQ_512K, 4, BS).unwrap();
        assert_eq!(
            target,
            requested_pool_blocks(2 * SEQ_512K, SEQ_512K, 4, BS) * BS
        );
        assert_eq!(
            mtp_kv_pool_blocks(SEQ_512K, 4, BS, Some(target)),
            target / BS + 4
        );
        assert_eq!(shared_pool_tokens(None, SEQ_512K, 4, BS), None);
        // Never above the legacy size.
        assert_eq!(
            mtp_kv_pool_blocks(8_192, 2, BS, Some(1 << 20)),
            mtp_kv_pool_blocks(8_192, 2, BS, None)
        );
    }
}
