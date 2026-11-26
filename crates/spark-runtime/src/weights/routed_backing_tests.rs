// SPDX-License-Identifier: AGPL-3.0-only

//! CPU-only unit tests for routed weight backing allocation and lookup semantics.

use super::super::WeightStore;
use super::*;

const LAYER: &str = "model.language_model.layers.0";

fn member(projection: RoutedProjection, expert: usize, scale: bool) -> RoutedBackingMember {
    RoutedBackingMember {
        name: format!(
            "{LAYER}.mlp.experts.{expert}.{}.{}",
            projection.name(),
            if scale { "weight_scale" } else { "weight" }
        ),
        expert,
        scale,
        offset: if scale { 512 } else { 256 },
        bytes: if scale { 8 } else { 16 },
        shape: vec![if scale { 8 } else { 16 }],
        dtype: if scale {
            WeightDtype::FP8E4M3
        } else {
            WeightDtype::UInt8
        },
    }
}

fn registry(
    projections: &[RoutedProjection],
) -> Result<(RoutedGroupRegistry, HashMap<String, WeightTensor>)> {
    let mut registry = RoutedGroupRegistry::default();
    let mut weights = HashMap::new();
    for (index, &projection) in projections.iter().enumerate() {
        let base = DevicePtr(0x1000 + (index as u64) * 0x1000);
        let members = vec![member(projection, 0, false), member(projection, 0, true)];
        for member in &members {
            weights.insert(
                member.name.clone(),
                WeightTensor {
                    ptr: base.offset(member.offset),
                    shape: member.shape.clone(),
                    dtype: member.dtype,
                },
            );
        }
        registry.register(LAYER.to_string(), projection, base, 768, members)?;
    }
    Ok((registry, weights))
}

#[derive(Default)]
struct StrictFree {
    live: HashSet<u64>,
    pending: HashSet<u64>,
    fail: HashSet<u64>,
    freed: Vec<u64>,
}

impl StrictFree {
    fn free(&mut self, base: DevicePtr) -> Result<()> {
        ensure!(self.live.contains(&base.0), "unknown/interior/double free");
        ensure!(!self.pending.contains(&base.0), "pending H2D retains base");
        ensure!(!self.fail.contains(&base.0), "injected driver free failure");
        self.live.remove(&base.0);
        self.freed.push(base.0);
        Ok(())
    }
}

#[test]
fn strict_fake_rejects_interior_unknown_and_double_free() -> Result<()> {
    let mut free = StrictFree {
        live: HashSet::from([0x1000]),
        ..Default::default()
    };
    assert!(free.free(DevicePtr(0x1100)).is_err());
    assert!(free.free(DevicePtr(0x9000)).is_err());
    free.free(DevicePtr(0x1000))?;
    assert!(free.free(DevicePtr(0x1000)).is_err());
    assert_eq!(free.freed, [0x1000]);
    Ok(())
}

#[test]
fn register_rejects_bad_bases_guards_alignment_ranges_and_scope() {
    for (base, bytes, offset) in [
        (0, 768, 256),
        (0x1001, 768, 256),
        (0x1000, 768, 0),
        (0x1000, 768, 257),
        (0x1000, 256, 256),
        (0x1000, 768, usize::MAX),
    ] {
        let mut registry = RoutedGroupRegistry::default();
        let mut view = member(RoutedProjection::Gate, 0, false);
        view.offset = offset;
        assert!(
            registry
                .register(
                    LAYER.into(),
                    RoutedProjection::Gate,
                    DevicePtr(base),
                    bytes,
                    vec![view]
                )
                .is_err()
        );
    }
    let mut registry = RoutedGroupRegistry::default();
    let mut view = member(RoutedProjection::Gate, 0, false);
    view.name = view.name.replace("gate_proj", "up_proj");
    assert!(
        registry
            .register(
                LAYER.into(),
                RoutedProjection::Gate,
                DevicePtr(0x1000),
                768,
                vec![view]
            )
            .is_err()
    );
}

#[test]
fn register_rejects_duplicate_names_and_overlapping_owners_or_views() -> Result<()> {
    let (mut registry, _) = registry(&[RoutedProjection::Gate])?;
    assert!(
        registry
            .register(
                LAYER.into(),
                RoutedProjection::Up,
                DevicePtr(0x1100),
                768,
                vec![member(RoutedProjection::Up, 0, false)]
            )
            .is_err()
    );
    assert!(
        registry
            .register(
                LAYER.into(),
                RoutedProjection::Gate,
                DevicePtr(0x2000),
                768,
                vec![member(RoutedProjection::Gate, 0, false)]
            )
            .is_err()
    );
    let mut registry = RoutedGroupRegistry::default();
    let mut scale = member(RoutedProjection::Gate, 0, true);
    scale.offset = 256;
    assert!(
        registry
            .register(
                LAYER.into(),
                RoutedProjection::Gate,
                DevicePtr(0x1000),
                768,
                vec![member(RoutedProjection::Gate, 0, false), scale]
            )
            .is_err()
    );
    Ok(())
}

#[test]
fn attach_requires_complete_exact_model_aliases() -> Result<()> {
    let (registry, _) = registry(&[RoutedProjection::Gate])?;
    let layer = registry.layer(LAYER).unwrap();
    layer.validate_projection(
        RoutedProjection::Gate,
        &[(DevicePtr(0x1100), DevicePtr(0x1200))],
    )?;
    assert!(
        layer
            .validate_projection(RoutedProjection::Gate, &[])
            .is_err()
    );
    assert!(
        layer
            .validate_projection(
                RoutedProjection::Gate,
                &[(DevicePtr(0x1100), DevicePtr(0x1201))]
            )
            .is_err()
    );
    assert!(
        layer
            .validate_projection(
                RoutedProjection::Up,
                &[(DevicePtr(0x1100), DevicePtr(0x1200))]
            )
            .is_err()
    );
    Ok(())
}

#[test]
fn store_rejects_missing_or_changed_view_and_individual_consumption() -> Result<()> {
    let (registry, mut weights) = registry(&[RoutedProjection::Gate])?;
    registry.validate_store(&weights)?;
    let name = format!("{LAYER}.mlp.experts.0.gate_proj.weight");
    weights.get_mut(&name).unwrap().ptr = DevicePtr(0x1101);
    assert!(registry.validate_store(&weights).is_err());
    weights.get_mut(&name).unwrap().ptr = DevicePtr(0x1100);
    let store = WeightStore::with_routed_groups(weights, registry)?;
    assert!(
        store
            .reserve_consumed_alias(&name, DevicePtr(0x1100), WeightDtype::UInt8)
            .is_err()
    );
    assert!(store.contains(&name));
    Ok(())
}

#[test]
fn release_frees_only_bases_in_projection_scope_and_tombstones_every_alias() -> Result<()> {
    let (registry, weights) = registry(&[
        RoutedProjection::Gate,
        RoutedProjection::Up,
        RoutedProjection::Down,
    ])?;
    let store = WeightStore::with_routed_groups(weights, registry)?;
    let groups = store.routed_original_groups(LAYER).unwrap();
    let mut free = StrictFree {
        live: HashSet::from([0x1000, 0x2000, 0x3000]),
        ..Default::default()
    };
    groups.release_projection_with(RoutedProjection::Gate, |base| free.free(base))?;
    assert_eq!(free.freed, [0x1000]);
    assert_eq!(store.len(), 4);
    for suffix in ["weight", "weight_scale"] {
        let name = format!("{LAYER}.mlp.experts.0.gate_proj.{suffix}");
        assert!(!store.contains(&name));
        assert!(store.get(&name).is_err());
        assert!(!store.names().any(|candidate| candidate == name));
    }
    groups.release_projection_with(RoutedProjection::Up, |base| free.free(base))?;
    assert!(free.live.contains(&0x3000));
    groups.release_projection_with(RoutedProjection::Down, |base| free.free(base))?;
    assert!(store.is_empty());
    assert_eq!(store.total_bytes(), 0);
    assert!(
        groups
            .release_projection_with(RoutedProjection::Gate, |base| free.free(base))
            .is_err()
    );
    assert_eq!(free.freed, [0x1000, 0x2000, 0x3000]);
    Ok(())
}

#[test]
fn driver_free_failure_or_pending_h2d_quarantines_without_retry() -> Result<()> {
    for pending in [false, true] {
        let (registry, weights) = registry(&[RoutedProjection::Gate])?;
        let store = WeightStore::with_routed_groups(weights, registry)?;
        let groups = store.routed_original_groups(LAYER).unwrap();
        let mut free = StrictFree {
            live: HashSet::from([0x1000]),
            ..Default::default()
        };
        if pending {
            free.pending.insert(0x1000);
        } else {
            free.fail.insert(0x1000);
        }
        assert!(
            groups
                .release_projection_with(RoutedProjection::Gate, |base| free.free(base))
                .is_err()
        );
        assert_eq!(*groups.groups[0].state.lock(), State::Quarantined);
        assert!(free.live.contains(&0x1000));
        assert!(free.freed.is_empty());
        assert!(store.is_empty());
        free.pending.clear();
        free.fail.clear();
        assert!(
            groups
                .release_projection_with(RoutedProjection::Gate, |base| free.free(base))
                .is_err()
        );
        assert!(free.live.contains(&0x1000));
    }
    Ok(())
}

#[test]
fn backend_panic_keeps_tombstones_and_quarantine_installed() -> Result<()> {
    let (registry, weights) = registry(&[RoutedProjection::Gate])?;
    let store = WeightStore::with_routed_groups(weights, registry)?;
    let groups = store.routed_original_groups(LAYER).unwrap();
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            groups.release_projection_with(RoutedProjection::Gate, |_| panic!("driver panic"))
        }))
        .is_err()
    );
    assert_eq!(*groups.groups[0].state.lock(), State::Quarantined);
    assert!(store.is_empty());
    assert!(
        groups
            .release_projection_with(RoutedProjection::Gate, |_| panic!("must not retry"))
            .is_err()
    );
    Ok(())
}

#[test]
fn consumed_group_handle_cannot_free_legitimately_reused_address() -> Result<()> {
    let (registry, _) = registry(&[RoutedProjection::Gate])?;
    let groups = registry.layer(LAYER).unwrap();
    let mut free = StrictFree {
        live: HashSet::from([0x1000]),
        ..Default::default()
    };
    groups.release_projection_with(RoutedProjection::Gate, |base| free.free(base))?;
    free.live.insert(0x1000);
    assert!(
        groups
            .release_projection_with(RoutedProjection::Gate, |base| free.free(base))
            .is_err()
    );
    assert!(free.live.contains(&0x1000));
    assert_eq!(free.freed, [0x1000]);
    Ok(())
}

#[test]
fn unregister_unclaimed_pops_only_the_latest_live_owner_with_that_base() -> Result<()> {
    let mut empty = RoutedGroupRegistry::default();
    assert!(empty.unregister_unclaimed(DevicePtr(0x1000)).is_err());
    let (mut registry, _) = registry(&[RoutedProjection::Gate, RoutedProjection::Up])?;
    let gate = member(RoutedProjection::Gate, 0, false).name;
    let up = member(RoutedProjection::Up, 0, true).name;
    // Not the latest owner, and an unknown base: refused, nothing removed.
    assert!(registry.unregister_unclaimed(DevicePtr(0x1000)).is_err());
    assert!(registry.unregister_unclaimed(DevicePtr(0x9000)).is_err());
    assert!(registry.contains(&gate) && registry.contains(&up));
    registry.unregister_unclaimed(DevicePtr(0x2000))?;
    assert!(!registry.contains(&up));
    assert!(!registry.contains(&member(RoutedProjection::Up, 0, false).name));
    assert!(registry.contains(&gate));
    registry.unregister_unclaimed(DevicePtr(0x1000))?;
    assert!(registry.is_empty());
    assert!(registry.unregister_unclaimed(DevicePtr(0x1000)).is_err());
    Ok(())
}

#[test]
fn unregister_unclaimed_refuses_a_released_latest_owner() -> Result<()> {
    let (mut registry, _) = registry(&[RoutedProjection::Gate])?;
    let groups = registry.layer(LAYER).unwrap();
    let mut free = StrictFree {
        live: HashSet::from([0x1000]),
        ..Default::default()
    };
    groups.release_projection_with(RoutedProjection::Gate, |base| free.free(base))?;
    assert!(registry.unregister_unclaimed(DevicePtr(0x1000)).is_err());
    assert!(!registry.is_empty());
    Ok(())
}

#[test]
fn target_group_release_refuses_released_names_and_leaves_native_mtp_tensors_readable() -> Result<()>
{
    let (registry, mut weights) = registry(&[RoutedProjection::Gate])?;
    let mtp = "mtp.layers.0.mlp.experts.0.gate_proj.weight".to_string();
    weights.insert(
        mtp.clone(),
        WeightTensor {
            ptr: DevicePtr(0x9000),
            shape: vec![16],
            dtype: WeightDtype::UInt8,
        },
    );
    let store = WeightStore::with_routed_groups(weights, registry)?;
    let groups = store.routed_original_groups(LAYER).unwrap();
    assert!(
        store.routed_original_groups("mtp.layers.0").is_none(),
        "MTP never attaches a group"
    );
    let mut free = StrictFree {
        live: HashSet::from([0x1000]),
        ..Default::default()
    };
    groups.release_projection_with(RoutedProjection::Gate, |base| free.free(base))?;
    let target = member(RoutedProjection::Gate, 0, false).name;
    // A target routed read after the release fails closed; the MTP tensor is
    // untouched and still readable at its own allocation.
    assert!(store.get(&target).is_err());
    assert!(!store.contains(&target));
    assert_eq!(store.get(&mtp)?.ptr, DevicePtr(0x9000));
    assert!(store.contains(&mtp));
    assert_eq!(
        free.freed,
        [0x1000],
        "the release never touches the MTP allocation"
    );
    Ok(())
}
