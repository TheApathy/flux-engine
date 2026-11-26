// SPDX-License-Identifier: AGPL-3.0-only

//! Allocation-table transitions. Callers hold the same table mutex while
//! retaining a DMA destination or removing an allocation for a driver free.

use anyhow::{Result, ensure};
use std::collections::HashMap;

pub(super) struct Allocation {
    pub(super) bytes: usize,
    pending_h2d: usize,
    grouped: bool,
    quarantined: bool,
}

impl Allocation {
    pub(super) fn new(bytes: usize) -> Self {
        Self {
            bytes,
            pending_h2d: 0,
            grouped: false,
            quarantined: false,
        }
    }

    pub(super) fn retain(&mut self, offset: usize, bytes: usize) -> Result<()> {
        ensure!(!self.quarantined, "grouped allocation is quarantined");
        let end = offset
            .checked_add(bytes)
            .ok_or_else(|| anyhow::anyhow!("H2D destination range overflow"))?;
        ensure!(
            end <= self.bytes,
            "H2D destination range exceeds owning allocation"
        );
        self.pending_h2d = self
            .pending_h2d
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("H2D destination reference overflow"))?;
        Ok(())
    }

    pub(super) fn release(&mut self) -> Result<()> {
        self.pending_h2d = self
            .pending_h2d
            .checked_sub(1)
            .ok_or_else(|| anyhow::anyhow!("H2D destination reference underflow"))?;
        Ok(())
    }
}

pub(super) fn claim_grouped(live: &mut HashMap<u64, Allocation>, base: u64) -> Result<()> {
    let allocation = live
        .get_mut(&base)
        .ok_or_else(|| anyhow::anyhow!("grouped claim requires exact live allocation base"))?;
    ensure!(
        !allocation.grouped && !allocation.quarantined && allocation.pending_h2d == 0,
        "allocation cannot be claimed for grouped ownership"
    );
    allocation.grouped = true;
    Ok(())
}

/// Keep the table mutex held through the driver call and clean completion.
/// Quarantine is durable before any call that may fail or become uncertain.
pub(super) fn prepare_grouped_free(live: &mut HashMap<u64, Allocation>, base: u64) -> Result<()> {
    let allocation = live
        .get_mut(&base)
        .ok_or_else(|| anyhow::anyhow!("grouped free requires exact live allocation base"))?;
    ensure!(
        allocation.grouped && !allocation.quarantined,
        "grouped owner already quarantined or absent"
    );
    allocation.quarantined = true;
    ensure!(
        allocation.pending_h2d == 0,
        "grouped backing retained by pending H2D"
    );
    Ok(())
}

pub(super) fn complete_grouped_free(live: &mut HashMap<u64, Allocation>, base: u64) -> Result<()> {
    let allocation = live
        .get(&base)
        .ok_or_else(|| anyhow::anyhow!("completed grouped free lost owning record"))?;
    ensure!(
        allocation.grouped && allocation.quarantined && allocation.pending_h2d == 0,
        "completed grouped free has invalid ownership state"
    );
    live.remove(&base);
    Ok(())
}

pub(super) fn take_for_free(
    live: &mut HashMap<u64, Allocation>,
    base: u64,
) -> Result<Option<Allocation>> {
    if let Some(allocation) = live.get(&base) {
        ensure!(
            !allocation.grouped && !allocation.quarantined,
            "grouped backing requires its explicit owner release API"
        );
        ensure!(
            allocation.pending_h2d == 0,
            "device allocation retained by pending H2D"
        );
    } else {
        ensure!(
            !live.iter().any(|(&owner, allocation)| allocation.grouped
                && base > owner
                && base.checked_sub(owner).is_some_and(
                    |offset| u64::try_from(allocation.bytes).is_ok_and(|bytes| offset < bytes)
                )),
            "non-owning grouped view cannot be freed individually"
        );
    }
    Ok(live.remove(&base))
}

pub(super) fn take_releasable(
    live: &mut HashMap<u64, Allocation>,
) -> (Vec<(u64, Allocation)>, usize) {
    let bases: Vec<u64> = live
        .iter()
        .filter_map(|(&base, allocation)| {
            (allocation.pending_h2d == 0 && !allocation.grouped && !allocation.quarantined)
                .then_some(base)
        })
        .collect();
    let drained = bases
        .into_iter()
        .map(|base| (base, live.remove(&base).unwrap()))
        .collect();
    (drained, live.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_allocation_survives_both_free_paths_until_every_reference_completes() {
        let mut live = HashMap::new();
        let mut allocation = Allocation::new(16);
        allocation.retain(0, 8).unwrap();
        allocation.retain(8, 8).unwrap();
        live.insert(100, allocation);
        live.insert(200, Allocation::new(8));
        assert!(take_for_free(&mut live, 100).is_err());
        let (drained, retained) = take_releasable(&mut live);
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].0, 200);
        assert_eq!(retained, 1);
        live.get_mut(&100).unwrap().release().unwrap();
        assert!(take_for_free(&mut live, 100).is_err());
        live.get_mut(&100).unwrap().release().unwrap();
        assert_eq!(take_for_free(&mut live, 100).unwrap().unwrap().bytes, 16);
        assert!(
            live.get_mut(&100).is_none(),
            "retain cannot race an in-progress free"
        );
    }

    #[test]
    fn checked_ranges_and_reference_arithmetic_fail_without_changing_ownership() {
        let mut allocation = Allocation::new(16);
        assert!(allocation.retain(usize::MAX, 2).is_err());
        assert!(allocation.retain(8, 9).is_err());
        assert!(allocation.release().is_err());
        assert_eq!(allocation.pending_h2d, 0);
        allocation.pending_h2d = usize::MAX;
        assert!(allocation.retain(0, 1).is_err());
        assert_eq!(allocation.pending_h2d, usize::MAX);
    }

    #[test]
    fn grouped_claim_rejects_interior_unknown_and_active_dma_bases() {
        let mut live = HashMap::from([(0x1000, Allocation::new(1024))]);
        assert!(claim_grouped(&mut live, 0x1100).is_err());
        assert!(claim_grouped(&mut live, 0x9000).is_err());
        live.get_mut(&0x1000).unwrap().retain(0, 16).unwrap();
        assert!(claim_grouped(&mut live, 0x1000).is_err());
        live.get_mut(&0x1000).unwrap().release().unwrap();
        claim_grouped(&mut live, 0x1000).unwrap();
        assert!(claim_grouped(&mut live, 0x1000).is_err());
        assert!(take_for_free(&mut live, 0x1000).is_err());
        assert!(take_for_free(&mut live, 0x1100).is_err());
        assert!(take_releasable(&mut live).0.is_empty());
    }

    #[test]
    fn grouped_driver_failure_cannot_be_retried_by_any_free_or_h2d_path() {
        let mut live = HashMap::from([
            (0x1000, Allocation::new(1024)),
            (0x2000, Allocation::new(16)),
        ]);
        claim_grouped(&mut live, 0x1000).unwrap();
        prepare_grouped_free(&mut live, 0x1000).unwrap();
        // Simulate a driver error or panic by deliberately not completing.
        assert!(live.get(&0x1000).unwrap().quarantined);
        assert!(prepare_grouped_free(&mut live, 0x1000).is_err());
        assert!(take_for_free(&mut live, 0x1000).is_err());
        assert!(take_for_free(&mut live, 0x1100).is_err());
        assert!(live.get_mut(&0x1000).unwrap().retain(256, 16).is_err());
        let (drained, retained) = take_releasable(&mut live);
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].0, 0x2000);
        assert_eq!(retained, 1);
        assert!(live.contains_key(&0x1000));
    }

    #[test]
    fn pending_grouped_h2d_is_quarantined_without_driver_permission_even_after_completion() {
        let mut live = HashMap::from([(0x1000, Allocation::new(1024))]);
        claim_grouped(&mut live, 0x1000).unwrap();
        live.get_mut(&0x1000).unwrap().retain(256, 16).unwrap();
        assert!(prepare_grouped_free(&mut live, 0x1000).is_err());
        assert!(live.get(&0x1000).unwrap().quarantined);
        live.get_mut(&0x1000).unwrap().release().unwrap();
        assert!(prepare_grouped_free(&mut live, 0x1000).is_err());
        assert!(take_for_free(&mut live, 0x1000).is_err());
        assert!(take_releasable(&mut live).0.is_empty());
    }

    #[test]
    fn clean_grouped_completion_removes_old_record_and_permits_legitimate_address_reuse() {
        let mut live = HashMap::from([(0x1000, Allocation::new(1024))]);
        claim_grouped(&mut live, 0x1000).unwrap();
        prepare_grouped_free(&mut live, 0x1000).unwrap();
        complete_grouped_free(&mut live, 0x1000).unwrap();
        assert!(!live.contains_key(&0x1000));
        assert!(prepare_grouped_free(&mut live, 0x1000).is_err());
        live.insert(0x1000, Allocation::new(16));
        assert!(prepare_grouped_free(&mut live, 0x1000).is_err());
        assert_eq!(take_for_free(&mut live, 0x1000).unwrap().unwrap().bytes, 16);
    }
}

#[cfg(test)]
#[path = "allocation_lifetime_grouped_api_tests.rs"]
mod grouped_api_tests;
