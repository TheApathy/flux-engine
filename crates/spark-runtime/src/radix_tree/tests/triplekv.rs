// SPDX-License-Identifier: AGPL-3.0-only

//! TripleKV 3-queue eviction priority (`Cached` < `Backup` < active `Prefetched`),
//! 120s TTL expiry, demand-pressure reclamation, and lookup consumption tests.

use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::prefix_cache::{DEFAULT_PREFETCH_TTL, PrefixCache, ProtectionClass};
use crate::radix_tree::RadixTree;

#[test]
fn triplekv_three_queue_eviction_order_respects_protection_tiers() {
    let tree = RadixTree::new();
    let now = Instant::now();

    // Insert 3 independent single-block prefixes with monotonically increasing last_access:
    // A (oldest) -> Prefetched (highest protection)
    // B (middle) -> Backup (medium protection)
    // C (newest) -> Cached (lowest protection)
    let tok_a: Vec<u32> = (0..16).collect();
    let tok_b: Vec<u32> = (100..116).collect();
    let tok_c: Vec<u32> = (200..216).collect();

    tree.insert(&tok_a, &[10], &[], 16, 0);
    tree.release(&tok_a, 16);
    tree.insert(&tok_b, &[20], &[], 16, 0);
    tree.release(&tok_b, 16);
    tree.insert(&tok_c, &[30], &[], 16, 0);
    tree.release(&tok_c, 16);

    assert_eq!(
        tree.set_prefix_protection(
            &tok_a,
            16,
            ProtectionClass::prefetched_until(now, DEFAULT_PREFETCH_TTL),
        ),
        1
    );
    assert_eq!(
        tree.set_prefix_protection(&tok_b, 16, ProtectionClass::Backup),
        1
    );

    // Even though C is NEWEST by LRU, Cached must evict first.
    let e1 = tree.evict_at(1, now);
    assert_eq!(e1.physical, vec![30], "Cached block C must evict first");

    // Next, Backup B must evict before active Prefetched A.
    let e2 = tree.evict_at(1, now);
    assert_eq!(e2.physical, vec![20], "Backup block B must evict second");

    // Under continued demand pressure, active Prefetched A IS still evictable (never OOM-pinned!).
    let e3 = tree.evict_at(1, now);
    assert_eq!(
        e3.physical,
        vec![10],
        "Prefetched block A must still be evictable under demand pressure"
    );
}

#[test]
fn triplekv_prefetch_ttl_expiry_demotes_before_backup_and_fresh_prefetch() {
    let tree = RadixTree::new();
    let t0 = Instant::now();

    let tok_expired: Vec<u32> = (0..16).collect();
    let tok_backup: Vec<u32> = (100..116).collect();
    let tok_fresh: Vec<u32> = (200..216).collect();

    tree.insert(&tok_expired, &[11], &[], 16, 0);
    tree.release(&tok_expired, 16);
    tree.insert(&tok_backup, &[22], &[], 16, 0);
    tree.release(&tok_backup, 16);
    tree.insert(&tok_fresh, &[33], &[], 16, 0);
    tree.release(&tok_fresh, 16);

    assert_eq!(
        tree.set_prefix_protection(
            &tok_expired,
            16,
            ProtectionClass::prefetched_until(t0, DEFAULT_PREFETCH_TTL),
        ),
        1
    );
    assert_eq!(
        tree.set_prefix_protection(&tok_backup, 16, ProtectionClass::Backup),
        1
    );
    assert_eq!(
        tree.set_prefix_protection(
            &tok_fresh,
            16,
            ProtectionClass::prefetched_until(t0 + Duration::from_secs(60), DEFAULT_PREFETCH_TTL),
        ),
        1
    );

    // At t0 + 120s: `tok_expired` TTL has elapsed -> demotes to tier 0 (Cached),
    // so it evicts ahead of `tok_backup` (tier 1) and `tok_fresh` (tier 2, expires at t0 + 180s).
    let t_after_ttl = t0 + DEFAULT_PREFETCH_TTL;
    let e1 = tree.evict_at(1, t_after_ttl);
    assert_eq!(
        e1.physical,
        vec![11],
        "expired Prefetched block must demote to Cached and evict before Backup"
    );

    let e2 = tree.evict_at(1, t_after_ttl);
    assert_eq!(
        e2.physical,
        vec![22],
        "Backup evicts before fresh Prefetched"
    );

    let e3 = tree.evict_at(1, t_after_ttl);
    assert_eq!(e3.physical, vec![33], "fresh Prefetched evicts last");
}

#[test]
fn triplekv_within_tier_lru_order_and_lookup_consumption() {
    let tree = Arc::new(RadixTree::new());
    let now = Instant::now();

    let tok_b1: Vec<u32> = (0..16).collect();
    let tok_b2: Vec<u32> = (32..48).collect();
    let tok_pf: Vec<u32> = (64..80).collect();

    tree.insert(&tok_b1, &[1], &[], 16, 0);
    tree.release(&tok_b1, 16);
    tree.insert(&tok_b2, &[2], &[], 16, 0);
    tree.release(&tok_b2, 16);
    tree.insert(&tok_pf, &[3], &[], 16, 0);
    tree.release(&tok_pf, 16);

    assert_eq!(
        tree.set_prefix_protection(&tok_b1, 16, ProtectionClass::Backup),
        1
    );
    assert_eq!(
        tree.set_prefix_protection(&tok_b2, 16, ProtectionClass::Backup),
        1
    );
    assert_eq!(
        tree.set_prefix_protection(
            &tok_pf,
            16,
            ProtectionClass::prefetched_until(now, DEFAULT_PREFETCH_TTL),
        ),
        1
    );

    // Lookup consumes `tok_pf`'s speculative Prefetched protection -> Cached once released.
    let m = PrefixCache::lookup(&tree, &tok_pf, 16, 0);
    assert_eq!(m.matched_tokens, 16);
    assert_eq!(
        tree.prefix_protection(&tok_pf, 16),
        vec![ProtectionClass::Cached]
    );
    tree.release(&tok_pf, 16);

    // Now `tok_pf` is Cached (tier 0) -> evicts before both Backup blocks (tier 1),
    // and among Backup blocks, older `tok_b1` evicts before newer `tok_b2`.
    let e1 = tree.evict_at(1, now);
    assert_eq!(e1.physical, vec![3]);
    let e2 = tree.evict_at(1, now);
    assert_eq!(e2.physical, vec![1]);
    let e3 = tree.evict_at(1, now);
    assert_eq!(e3.physical, vec![2]);
}
