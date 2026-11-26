// SPDX-License-Identifier: AGPL-3.0-only

//! Slot free-list contract: a slot is never simultaneously owned and free,
//! and a double release never puts it on the free list twice.
//!
//! Ported from upstream Atlas-Inf `11e83f5b` (`ssm_pool::slot_guard_tests`).
//! The fork has no RAII slot guard, so ownership is exercised through
//! `claim_slot` / `release_slot` / `claim_specific` directly.

use super::*;
use crate::model::qwen4_mtp_hidden::NativeMtpHiddenFrames;

/// A pool with `max_slots` claimable indices and no device memory.
fn bare_pool(max_slots: usize) -> SsmStatePool {
    SsmStatePool {
        allocation_geometry: Default::default(),
        h_state_pools: Vec::new(),
        conv_state_pools: Vec::new(),
        h_intermediate_pools: Vec::new(),
        conv_intermediate_pools: Vec::new(),
        h_verify_scratch: DevicePtr::NULL,
        h_checkpoint_pools: Vec::new(),
        conv_checkpoint_pools: Vec::new(),
        wy17_kv_retain_pools: Vec::new(),
        wy17_gate_retain_pools: Vec::new(),
        kv_retain_bytes: 0,
        gate_retain_bytes: 0,
        h_bytes: 0,
        conv_bytes: 0,
        max_slots,
        spec_slots: max_slots,
        num_ssm_layers: 0,
        has_mtp: false,
        compact_lazy_h: false,
        num_intermediates: 0,
        free_slots: Mutex::new((0..max_slots).rev().collect()),
    }
}

fn free_count(pool: &SsmStatePool) -> usize {
    pool.free_slots.lock().len()
}

fn sorted_free(pool: &SsmStatePool) -> Vec<usize> {
    let mut v = pool.free_slots.lock().clone();
    v.sort_unstable();
    v
}

#[test]
fn claim_hands_out_the_lowest_index_first() {
    let pool = bare_pool(3);
    assert_eq!(pool.claim_slot().unwrap(), 0);
    assert_eq!(pool.claim_slot().unwrap(), 1);
    assert_eq!(pool.claim_slot().unwrap(), 2);
    assert!(pool.claim_slot().is_err(), "no fourth slot");
}

#[test]
fn double_release_is_ignored_in_every_build() {
    // The guard is unconditional (not a `debug_assert!`): a second release
    // of the same index must not put it on the free list twice, or
    // `claim_slot` would hand it to two sequences from then on.
    let pool = bare_pool(2);
    let a = pool.claim_slot().unwrap();
    assert_eq!(a, 0);
    pool.release_slot(a);
    pool.release_slot(a); // double release: logged, NOT pushed
    assert_eq!(free_count(&pool), 2, "slot 0 must be free exactly once");
    // Two concurrent claims therefore still get DISTINCT slots.
    let x = pool.claim_slot().unwrap();
    let y = pool.claim_slot().unwrap();
    assert_ne!(x, y, "a duplicated free entry hands one slot to two seqs");
    assert!(pool.claim_slot().is_err(), "no phantom third slot");
}

#[test]
fn claim_specific_refuses_an_owned_slot() {
    // The prefill-overlap collision: L (still prefilling) owns slot 0, S
    // (active) owns slot 1, and compaction wants S on slot 0. The claim
    // must FAIL so `compact_sequence` never copies onto L's state.
    let pool = bare_pool(4);
    let l = pool.claim_slot().unwrap(); // slot 0
    let s = pool.claim_slot().unwrap(); // slot 1
    assert_eq!((l, s), (0, 1));
    assert!(!pool.claim_specific(0), "slot 0 is owned, not free");
    assert_eq!(free_count(&pool), 2, "a refused claim changes nothing");
    // Once L is gone its slot IS a legal target, claimable exactly once.
    pool.release_slot(l);
    assert!(pool.claim_specific(0));
    assert!(!pool.claim_specific(0));
    pool.release_slot(0); // undo the bare claim above
    pool.release_slot(s);
    assert_eq!(
        sorted_free(&pool),
        vec![0, 1, 2, 3],
        "every slot free exactly once"
    );
}

#[test]
fn claim_specific_takes_the_slot_out_of_the_free_list() {
    // A claimed migration target must not also be handed out by
    // `claim_slot`: the owned-and-free state is the double-ownership bug.
    let pool = bare_pool(3);
    assert!(pool.claim_specific(1));
    assert_eq!(sorted_free(&pool), vec![0, 2]);
    let a = pool.claim_slot().unwrap();
    let b = pool.claim_slot().unwrap();
    assert_ne!(a, 1);
    assert_ne!(b, 1);
    assert!(pool.claim_slot().is_err());
}

#[test]
fn retire_with_migration_frees_every_slot_exactly_once() {
    // Full scenario in pool terms: retired R owns slot 0; survivor S owns
    // slot 1. R releases its slot, S claims it specifically and releases
    // its old one. Then S is freed. Every slot ends up free exactly once.
    let pool = bare_pool(2);
    let r = pool.claim_slot().unwrap(); // 0
    let s = pool.claim_slot().unwrap(); // 1
    pool.release_slot(r);
    assert!(pool.claim_specific(r), "the just-vacated slot is claimable");
    pool.release_slot(s); // S migrated 1 -> 0, its old slot goes back
    pool.release_slot(r); // S finishes on slot 0
    assert_eq!(sorted_free(&pool), vec![0, 1]);
}

#[test]
fn hidden_session_mismatch_restores_the_claimed_destination_and_source_owner() {
    let pool = bare_pool(2);
    let old_slot = pool.claim_slot().unwrap();
    let new_slot = 1;
    let mut frames = NativeMtpHiddenFrames::new(2, 5, 16).unwrap();
    let write = frames.begin_write(old_slot, 11, 37, 1, 32).unwrap();
    frames.publish(write).unwrap();

    let error = pool
        .claim_specific_with_preflight(new_slot, || {
            frames
                .begin_move(old_slot, new_slot, 99)
                .map_err(anyhow::Error::msg)
        })
        .unwrap_err();

    assert!(error.to_string().contains("source owner mismatch"));
    assert_eq!(sorted_free(&pool), vec![new_slot]);
    assert!(frames.take(old_slot, 11, 37).is_ok());
    assert_eq!(pool.claim_slot().unwrap(), new_slot);
    assert!(
        pool.claim_slot().is_err(),
        "the source stays exclusively owned"
    );
}

#[test]
fn invalid_hidden_destination_restores_its_pool_claim_without_moving_the_source() {
    let pool = bare_pool(3);
    let old_slot = pool.claim_slot().unwrap();
    let new_slot = 2;
    // A deliberately mismatched hidden capacity exercises the production
    // policy's pre-copy error path without a device or a second implementation.
    let mut frames = NativeMtpHiddenFrames::new(2, 5, 16).unwrap();
    let write = frames.begin_write(old_slot, 11, 37, 1, 32).unwrap();
    frames.publish(write).unwrap();

    let error = pool
        .claim_specific_with_preflight(new_slot, || {
            frames
                .begin_move(old_slot, new_slot, 11)
                .map_err(anyhow::Error::msg)
        })
        .unwrap_err();

    assert!(
        error
            .to_string()
            .contains("slot is outside retained storage")
    );
    assert_eq!(sorted_free(&pool), vec![1, new_slot]);
    assert!(frames.take(old_slot, 11, 37).is_ok());
    assert!(
        pool.claim_specific(new_slot),
        "the reservation was returned"
    );
    assert!(!pool.claim_specific(new_slot), "it was returned only once");
}

#[test]
fn owned_destination_refuses_before_any_hidden_preflight_or_mutation() {
    let pool = bare_pool(2);
    let old_slot = pool.claim_slot().unwrap();
    let new_slot = pool.claim_slot().unwrap();
    let mut frames = NativeMtpHiddenFrames::new(2, 5, 16).unwrap();
    let source = frames.begin_write(old_slot, 11, 37, 1, 32).unwrap();
    frames.publish(source).unwrap();
    let destination = frames.begin_write(new_slot, 22, 48, 0, 16).unwrap();
    frames.publish(destination).unwrap();
    let prepared = std::cell::Cell::new(false);

    let result = pool
        .claim_specific_with_preflight(new_slot, || {
            prepared.set(true);
            frames
                .begin_move(old_slot, new_slot, 11)
                .map_err(anyhow::Error::msg)
        })
        .unwrap();

    assert!(result.is_none());
    assert!(!prepared.get());
    assert_eq!(free_count(&pool), 0);
    assert!(frames.take(old_slot, 11, 37).is_ok());
    assert!(frames.take(new_slot, 22, 48).is_ok());
}

#[test]
fn valid_hidden_preflight_keeps_the_destination_reserved_until_migration_finishes() {
    let pool = bare_pool(2);
    let old_slot = pool.claim_slot().unwrap();
    let new_slot = 1;
    let mut frames = NativeMtpHiddenFrames::new(2, 5, 16).unwrap();
    let write = frames.begin_write(old_slot, 11, 37, 1, 32).unwrap();
    frames.publish(write).unwrap();

    let movement = pool
        .claim_specific_with_preflight(new_slot, || {
            frames
                .begin_move(old_slot, new_slot, 11)
                .map_err(anyhow::Error::msg)
        })
        .unwrap()
        .unwrap();

    assert!(
        pool.claim_slot().is_err(),
        "both migration slots stay reserved"
    );
    frames.finish_move(movement).unwrap();
    pool.release_slot(old_slot);
    assert!(frames.take(new_slot, 11, 37).is_ok());
    assert_eq!(pool.claim_slot().unwrap(), old_slot);
    assert!(
        !pool.claim_specific(new_slot),
        "the survivor owns its destination"
    );
}
