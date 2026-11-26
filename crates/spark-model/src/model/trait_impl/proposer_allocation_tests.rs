// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;
use std::cell::RefCell;

struct State(DevicePtr);

impl ProposerState for State {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[test]
fn secondary_failure_frees_primary_before_the_claimed_slot() {
    let gpu = MockGpuBackend::new();
    let events = RefCell::new(Vec::new());
    let result = (|| -> Result<()> {
        let (_slot, _rollback) = claim_slot_transaction(
            || Ok(9),
            |slot| {
                events.borrow_mut().push(format!("release-slot:{slot}"));
            },
        )?;
        let _primary = allocate_guarded_state(
            || {
                Ok(Some(
                    Box::new(State(gpu.alloc(32)?)) as Box<dyn ProposerState>
                ))
            },
            |state| {
                gpu.synchronize_device().unwrap();
                events.borrow_mut().push("primary-sync".to_string());
                let pointer = state
                    .as_ref()
                    .unwrap()
                    .as_any()
                    .downcast_ref::<State>()
                    .unwrap()
                    .0;
                gpu.free(pointer).unwrap();
                events.borrow_mut().push("primary-free".to_string());
            },
        )?;
        let _secondary = allocate_guarded_state(
            || anyhow::bail!("injected secondary allocation failure"),
            |_| panic!("no secondary state exists to reclaim"),
        )?;
        Ok(())
    })();
    assert!(result.is_err());
    assert_eq!(
        events.into_inner(),
        ["primary-sync", "primary-free", "release-slot:9"]
    );
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn committed_state_is_transferred_without_running_rollback() {
    let gpu = MockGpuBackend::new();
    let mut guard = allocate_guarded_state(
        || {
            Ok(Some(
                Box::new(State(gpu.alloc(32)?)) as Box<dyn ProposerState>
            ))
        },
        |_| panic!("committed owner must not be reclaimed"),
    )
    .unwrap();
    let state = guard.take().unwrap();
    drop(guard);
    assert_eq!(gpu.alloc_count(), 1);
    let pointer = state.as_any().downcast_ref::<State>().unwrap().0;
    gpu.free(pointer).unwrap();
}

#[test]
fn production_keeps_both_arms_guarded_until_the_complete_sequence_exists() {
    let source = include_str!("meta.rs");
    let body = source.split("fn alloc_sequence_inner").nth(1).unwrap();
    assert_eq!(body.matches("self.allocate_proposer_state(").count(), 2);
    let completed = body.find("let mut state = SequenceState {").unwrap();
    let transfer = body.find("state_primary.take()").unwrap();
    let commit = body.find("slot_rollback.commit()").unwrap();
    assert!(completed < transfer && transfer < commit);
    assert!(source.contains("self.free_dflash_state_buffers(state)"));
    assert!(source.contains("proposer.free_state(state)"));
}
