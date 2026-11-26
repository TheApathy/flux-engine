// SPDX-License-Identifier: AGPL-3.0-only

//! CPU-only unit tests for routed expert group metadata and grouping invariants.

use super::*;
use crate::gpu::KernelHandle;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

fn tensor(layer: usize, expert: usize, projection: RoutedProjection, scale: bool) -> TensorMeta {
    let (rows, cols) = if projection == RoutedProjection::Down {
        (2560, 640)
    } else {
        (640, 2560)
    };
    let shape = vec![rows, cols / if scale { 16 } else { 2 }];
    TensorMeta {
        name: format!(
            "{PREFIX}{layer}.mlp.experts.{expert}.{}.{}",
            projection.name(),
            if scale { "weight_scale" } else { "weight" }
        ),
        len: shape[0] * shape[1],
        shape,
        abs_offset: 0,
        dtype: if scale {
            WeightDtype::FP8E4M3
        } else {
            WeightDtype::UInt8
        },
    }
}

fn inventory() -> Vec<(PathBuf, Vec<TensorMeta>)> {
    let mut tensors = Vec::with_capacity(COMPONENTS);
    let mut offset = 0u64;
    for layer in 0..LAYERS {
        for projection in [
            RoutedProjection::Gate,
            RoutedProjection::Up,
            RoutedProjection::Down,
        ] {
            for expert in 0..EXPERTS {
                for scale in [false, true] {
                    let mut meta = tensor(layer, expert, projection, scale);
                    meta.abs_offset = offset;
                    offset += meta.len as u64;
                    tensors.push(meta);
                }
            }
        }
    }
    vec![(PathBuf::from("synthetic.safetensors"), tensors)]
}

#[test]
fn grouped_flag_is_off_default_and_only_accepts_exact_ascii_zero_or_one() -> Result<()> {
    assert!(!routed_grouping_requested(None)?);
    assert!(!routed_grouping_requested(Some(std::ffi::OsStr::new("0")))?);
    assert!(routed_grouping_requested(Some(std::ffi::OsStr::new("1")))?);
    for value in ["", "true", "yes", "01", "2", " 1", "1\n", "\u{ff11}"] {
        assert!(routed_grouping_requested(Some(std::ffi::OsStr::new(value))).is_err());
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        assert!(routed_grouping_requested(Some(std::ffi::OsStr::from_bytes(&[0xff]))).is_err());
    }
    Ok(())
}

#[test]
fn member_parser_accepts_only_exact_canonical_geometry_and_nvfp4_dtypes() -> Result<()> {
    for projection in [
        RoutedProjection::Gate,
        RoutedProjection::Up,
        RoutedProjection::Down,
    ] {
        for scale in [false, true] {
            let mut meta = tensor(47, 511, projection, scale);
            assert!(parse_member(&meta)?.is_some());
            meta.dtype = WeightDtype::BF16;
            assert!(parse_member(&meta).is_err());
            meta = tensor(47, 511, projection, scale);
            meta.shape[0] += 1;
            assert!(parse_member(&meta).is_err());
            meta = tensor(47, 511, projection, scale);
            meta.len -= 1;
            assert!(parse_member(&meta).is_err());
        }
    }
    for (layer, expert) in [(48, 0), (0, 512)] {
        assert!(parse_member(&tensor(layer, expert, RoutedProjection::Gate, false)).is_err());
    }
    let mut meta = tensor(0, 0, RoutedProjection::Gate, false);
    meta.name = meta.name.replacen("layers.0.", "layers.00.", 1);
    assert!(parse_member(&meta).is_err());
    Ok(())
}

#[test]
fn mtp_shared_scalars_and_other_targets_are_not_grouped() -> Result<()> {
    for name in [
        "mtp.layers.0.mlp.experts.0.gate_proj.weight",
        "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
        "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale_2",
        "model.language_model.layers.0.mlp.experts.0.gate_proj.input_scale",
        "model.layers.0.mlp.experts.0.gate_proj.weight",
    ] {
        let mut meta = tensor(0, 0, RoutedProjection::Gate, false);
        meta.name = name.into();
        assert!(parse_member(&meta)?.is_none());
    }
    Ok(())
}

#[test]
fn complete_fixture_has_bounded_phase_scoped_groups_and_exact_payload_and_guards() -> Result<()> {
    let plan = RoutedBackingLoad::from_inventories(inventory())?;
    assert_eq!(plan.by_name.len(), 147_456);
    assert_eq!(plan.groups.len(), 1152);
    assert_eq!(plan.overhead_bytes(), 1152 * 256);
    assert_eq!(
        plan.groups.iter().map(|g| g.bytes).sum::<usize>(),
        PAYLOAD_BYTES + 1152 * 256
    );
    for group in &plan.groups {
        assert!(group.bytes <= GROUP_LIMIT);
        assert!(group.base.is_none());
        let mut end = ALIGN;
        for view in &group.views {
            assert!(view.offset >= end && view.offset % ALIGN == 0);
            assert!(
                view.name
                    .starts_with(&format!("{}.mlp.experts.", group.layer_prefix))
            );
            assert!(
                view.name
                    .contains(&format!(".{}.", group.projection.name()))
            );
            end = view.offset + view.bytes;
        }
        assert_eq!(end, group.bytes);
    }
    Ok(())
}

#[test]
fn global_inventory_rejects_missing_filtered_duplicate_and_malformed_pairs_before_allocation() {
    let mut missing = inventory();
    missing[0].1.pop();
    assert!(RoutedBackingLoad::from_inventories(missing).is_err());
    let mut duplicate = inventory();
    duplicate[0]
        .1
        .push(tensor(0, 0, RoutedProjection::Gate, false));
    assert!(RoutedBackingLoad::from_inventories(duplicate).is_err());
    let mut wrong = inventory();
    wrong[0].1[0].dtype = WeightDtype::BF16;
    assert!(RoutedBackingLoad::from_inventories(wrong).is_err());
}

#[test]
fn bounded_layout_never_crosses_shards_even_for_one_projection() -> Result<()> {
    let mut all = inventory();
    let tail = all[0].1.split_off(1);
    all.push((PathBuf::from("second.safetensors"), tail));
    let plan = RoutedBackingLoad::from_inventories(all)?;
    let first = plan.by_name[&tensor(0, 0, RoutedProjection::Gate, false).name].0;
    let scale = plan.by_name[&tensor(0, 0, RoutedProjection::Gate, true).name].0;
    assert_ne!(first, scale);
    assert_ne!(plan.groups[first].shard, plan.groups[scale].shard);
    assert!(plan.groups.iter().all(|group| group.bytes <= GROUP_LIMIT));
    Ok(())
}

#[test]
fn layout_and_pointer_arithmetic_overflow_fail_closed() {
    assert!(align(usize::MAX).is_err());
    assert_eq!(align(0).unwrap(), 0);
    assert_eq!(align(256).unwrap(), 256);
    assert_eq!(align(257).unwrap(), 512);
}

/// Address-only grouped-ownership backend. Hands out fake 256-aligned bases
/// without host buffers (the canonical plan is ~68 GB), records every
/// allocation, claim and ordinary free, and can refuse claims.
struct OwnershipMock {
    next: AtomicU64,
    allocations: Mutex<Vec<u64>>,
    live: Mutex<HashSet<u64>>,
    claimed: Mutex<HashSet<u64>>,
    freed: Mutex<Vec<u64>>,
    refuse_claim: AtomicBool,
}

impl OwnershipMock {
    fn new() -> Self {
        Self {
            next: AtomicU64::new(0x4000_0000),
            allocations: Mutex::new(Vec::new()),
            live: Mutex::new(HashSet::new()),
            claimed: Mutex::new(HashSet::new()),
            freed: Mutex::new(Vec::new()),
            refuse_claim: AtomicBool::new(false),
        }
    }
    fn allocation_count(&self) -> usize {
        self.allocations.lock().unwrap().len()
    }
}

impl GpuBackend for OwnershipMock {
    fn supports_grouped_backing(&self) -> bool {
        true
    }
    fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
        let span = u64::try_from(align(bytes.max(1))?)?;
        let base = self.next.fetch_add(span, Ordering::SeqCst);
        self.allocations.lock().unwrap().push(base);
        self.live.lock().unwrap().insert(base);
        Ok(DevicePtr(base))
    }
    fn alloc_managed(&self, _bytes: usize) -> Result<DevicePtr> {
        anyhow::bail!("grouped owners never use managed memory")
    }
    fn free(&self, ptr: DevicePtr) -> Result<()> {
        ensure!(
            !self.claimed.lock().unwrap().contains(&ptr.0),
            "claimed base needs the grouped free API"
        );
        ensure!(
            self.live.lock().unwrap().remove(&ptr.0),
            "unknown or double free"
        );
        self.freed.lock().unwrap().push(ptr.0);
        Ok(())
    }
    fn claim_grouped_backing(&self, base: DevicePtr) -> Result<()> {
        ensure!(
            !self.refuse_claim.load(Ordering::SeqCst),
            "injected claim refusal"
        );
        ensure!(
            self.live.lock().unwrap().contains(&base.0),
            "claim requires a live base"
        );
        ensure!(self.claimed.lock().unwrap().insert(base.0), "double claim");
        Ok(())
    }
    fn copy_h2d(&self, _src: &[u8], _dst: DevicePtr) -> Result<()> {
        anyhow::bail!("unused by ownership tests")
    }
    fn copy_d2h(&self, _src: DevicePtr, _dst: &mut [u8]) -> Result<()> {
        anyhow::bail!("unused by ownership tests")
    }
    fn copy_d2d(&self, _src: DevicePtr, _dst: DevicePtr, _bytes: usize) -> Result<()> {
        anyhow::bail!("unused by ownership tests")
    }
    fn launch(
        &self,
        _func: KernelHandle,
        _grid: [u32; 3],
        _block: [u32; 3],
        _shared_mem: u32,
        _stream: u64,
        _params: &mut [*mut std::ffi::c_void],
    ) -> Result<()> {
        anyhow::bail!("unused by ownership tests")
    }
    fn synchronize(&self, _stream: u64) -> Result<()> {
        Ok(())
    }
    fn default_stream(&self) -> u64 {
        0
    }
    #[track_caller]
    fn kernel(&self, _module: &str, _func_name: &str) -> Result<KernelHandle> {
        anyhow::bail!("unused by ownership tests")
    }
    fn memset(&self, _ptr: DevicePtr, _value: u8, _bytes: usize) -> Result<()> {
        anyhow::bail!("unused by ownership tests")
    }
    fn memset_async(&self, _ptr: DevicePtr, _value: u8, _bytes: usize, _stream: u64) -> Result<()> {
        anyhow::bail!("unused by ownership tests")
    }
    fn total_memory(&self) -> Result<usize> {
        Ok(0)
    }
    fn free_memory(&self) -> Result<usize> {
        Ok(0)
    }
}

fn os(value: &str) -> Option<&std::ffi::OsStr> {
    Some(std::ffi::OsStr::new(value))
}

#[test]
fn eager_flag_accepts_only_unset_zero_or_one_and_requires_grouping() -> Result<()> {
    for grouped in [false, true] {
        assert!(!routed_grouping_eager_requested(None, grouped)?);
        assert!(!routed_grouping_eager_requested(os("0"), grouped)?);
    }
    assert!(routed_grouping_eager_requested(os("1"), true)?);
    let refused = routed_grouping_eager_requested(os("1"), false).unwrap_err();
    assert!(format!("{refused:#}").contains("requires ATLAS_FAST_LOAD_ROUTED_GROUPS=1"));
    for value in ["", "2", "true", " 1", "1\n", "01"] {
        assert!(
            routed_grouping_eager_requested(os(value), true).is_err(),
            "{value:?}"
        );
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let non_utf8 = Some(std::ffi::OsStr::from_bytes(&[0xff]));
        assert!(routed_grouping_eager_requested(non_utf8, true).is_err());
    }
    Ok(())
}

#[test]
fn refused_registration_frees_the_unclaimed_base_and_publishes_nothing() -> Result<()> {
    let mut load = RoutedBackingLoad::from_inventories(inventory())?;
    // A prior owner already holds the group's first member name, so the
    // group's registration is refused after its base is allocated.
    let first = load.groups[0].views[0].clone();
    let second = load.groups[0].views[1].name.clone();
    load.registry.as_mut().expect("live registry").register(
        "model.language_model.layers.0".to_string(),
        RoutedProjection::Gate,
        DevicePtr(0x7f00_0000_0000),
        ALIGN + first.bytes,
        vec![RoutedBackingMember {
            name: first.name.clone(),
            expert: first.expert,
            scale: first.scale,
            offset: ALIGN,
            bytes: first.bytes,
            shape: first.shape.clone(),
            dtype: first.dtype,
        }],
    )?;
    let gpu = OwnershipMock::new();
    assert!(load.materialize(0, &gpu).is_err());
    assert!(load.groups[0].base.is_none());
    let allocated = gpu.allocations.lock().unwrap().clone();
    assert_eq!(allocated.len(), 1);
    assert_eq!(*gpu.freed.lock().unwrap(), allocated);
    assert!(gpu.live.lock().unwrap().is_empty());
    assert!(gpu.claimed.lock().unwrap().is_empty());
    let registry = load.registry.as_ref().expect("live registry");
    assert!(!registry.contains(&second));
    assert!(
        registry.contains(&first.name),
        "only the prior owner remains"
    );
    Ok(())
}

#[test]
fn refused_claim_unregisters_then_frees_and_keeps_the_claim_error() -> Result<()> {
    let mut load = RoutedBackingLoad::from_inventories(inventory())?;
    let gpu = OwnershipMock::new();
    gpu.refuse_claim.store(true, Ordering::SeqCst);
    let error = load.materialize(0, &gpu).unwrap_err();
    assert!(format!("{error:#}").contains("injected claim refusal"));
    assert!(load.groups[0].base.is_none());
    let allocated = gpu.allocations.lock().unwrap().clone();
    assert_eq!(allocated.len(), 1);
    assert_eq!(*gpu.freed.lock().unwrap(), allocated);
    assert!(gpu.live.lock().unwrap().is_empty());
    assert!(gpu.claimed.lock().unwrap().is_empty());
    let registry = load.registry.as_ref().expect("live registry");
    assert!(registry.is_empty());
    assert!(
        load.groups[0]
            .views
            .iter()
            .all(|view| !registry.contains(&view.name))
    );
    // The group is retryable: nothing was left behind.
    gpu.refuse_claim.store(false, Ordering::SeqCst);
    load.materialize(0, &gpu)?;
    assert!(load.groups[0].base.is_some());
    Ok(())
}

#[test]
fn allocate_all_materializes_every_planned_base_once_and_first_views_reuse_them() -> Result<()> {
    let mut load = RoutedBackingLoad::from_inventories(inventory())?;
    let gpu = OwnershipMock::new();
    let sizes = load.allocate_all(&gpu)?;
    assert_eq!(sizes.len(), load.groups.len());
    assert_eq!(
        sizes,
        load.groups
            .iter()
            .map(|group| group.bytes)
            .collect::<Vec<_>>()
    );
    assert_eq!(gpu.allocation_count(), load.groups.len());
    assert_eq!(gpu.claimed.lock().unwrap().len(), load.groups.len());
    let bases: HashSet<u64> = load
        .groups
        .iter()
        .map(|group| group.base.expect("eager base").0)
        .collect();
    assert_eq!(bases.len(), load.groups.len());
    assert!(
        load.allocate_all(&gpu).is_err(),
        "a second eager pass must not reallocate"
    );
    assert_eq!(gpu.allocation_count(), load.groups.len());
    let shard = PathBuf::from("synthetic.safetensors");
    for meta in inventory().remove(0).1.into_iter().take(4) {
        let (group_index, view_index) = load.by_name[&meta.name];
        let (base, offset, allocated) = load
            .destination(&shard, &meta, &gpu)?
            .expect("planned routed member");
        assert_eq!(Some(base), load.groups[group_index].base);
        assert_eq!(offset, load.groups[group_index].views[view_index].offset);
        assert_eq!(allocated, None);
    }
    assert_eq!(gpu.allocation_count(), load.groups.len());
    assert!(gpu.freed.lock().unwrap().is_empty());
    Ok(())
}

#[test]
fn lazy_first_view_materializes_exactly_one_base() -> Result<()> {
    let mut load = RoutedBackingLoad::from_inventories(inventory())?;
    let gpu = OwnershipMock::new();
    let shard = PathBuf::from("synthetic.safetensors");
    let metas: Vec<TensorMeta> = inventory().remove(0).1.into_iter().take(2).collect();
    let first = load
        .destination(&shard, &metas[0], &gpu)?
        .expect("planned routed member");
    let group_index = load.by_name[&metas[0].name].0;
    assert_eq!(first.2, Some(load.groups[group_index].bytes));
    let second = load
        .destination(&shard, &metas[1], &gpu)?
        .expect("planned routed member");
    assert_eq!(load.by_name[&metas[1].name].0, group_index, "same group");
    assert_eq!((second.0, second.2), (first.0, None));
    assert_eq!(gpu.allocation_count(), 1);
    Ok(())
}

/// Strict owner mock for the tiny direct fixtures. Rejects unknown, interior
/// and double frees; an injected free failure keeps the base live; a claimed
/// base refuses the ordinary free; claims can be refused or panic at the n-th
/// call.
struct StrictOwnerMock {
    next: AtomicU64,
    live: Mutex<HashMap<u64, usize>>,
    claimed: Mutex<HashSet<u64>>,
    freed: Mutex<Vec<u64>>,
    allocations: Mutex<Vec<u64>>,
    claim_calls: AtomicU64,
    refuse_claim_at: AtomicU64,
    panic_claim_at: AtomicU64,
    fail_free: AtomicBool,
}

const NEVER: u64 = u64::MAX;

impl StrictOwnerMock {
    fn new() -> Self {
        Self {
            next: AtomicU64::new(0x5000_0000),
            live: Mutex::new(HashMap::new()),
            claimed: Mutex::new(HashSet::new()),
            freed: Mutex::new(Vec::new()),
            allocations: Mutex::new(Vec::new()),
            claim_calls: AtomicU64::new(0),
            refuse_claim_at: AtomicU64::new(NEVER),
            panic_claim_at: AtomicU64::new(NEVER),
            fail_free: AtomicBool::new(false),
        }
    }
    fn live_has(&self, base: u64) -> bool {
        self.live.lock().unwrap().contains_key(&base)
    }
    fn is_claimed(&self, base: u64) -> bool {
        self.claimed.lock().unwrap().contains(&base)
    }
    fn freed(&self) -> Vec<u64> {
        self.freed.lock().unwrap().clone()
    }
    fn allocations(&self) -> Vec<u64> {
        self.allocations.lock().unwrap().clone()
    }
}

impl GpuBackend for StrictOwnerMock {
    fn supports_grouped_backing(&self) -> bool {
        true
    }
    fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
        let span = u64::try_from(align(bytes.max(1))?)?;
        let base = self.next.fetch_add(span, Ordering::SeqCst);
        self.live.lock().unwrap().insert(base, bytes);
        self.allocations.lock().unwrap().push(base);
        Ok(DevicePtr(base))
    }
    fn alloc_managed(&self, _bytes: usize) -> Result<DevicePtr> {
        anyhow::bail!("grouped owners never use managed memory")
    }
    fn free(&self, ptr: DevicePtr) -> Result<()> {
        let mut live = self.live.lock().unwrap();
        if !live.contains_key(&ptr.0) {
            let interior = live
                .iter()
                .any(|(&base, &bytes)| ptr.0 > base && ptr.0 < base + bytes as u64);
            ensure!(!interior, "strict mock rejects an interior free");
            ensure!(
                !self.freed.lock().unwrap().contains(&ptr.0),
                "strict mock rejects a double free"
            );
            anyhow::bail!("strict mock rejects an unknown free");
        }
        ensure!(
            !self.claimed.lock().unwrap().contains(&ptr.0),
            "claimed base requires the grouped free API"
        );
        // An injected driver failure leaves the base live.
        ensure!(
            !self.fail_free.load(Ordering::SeqCst),
            "injected ordinary free failure"
        );
        live.remove(&ptr.0);
        self.freed.lock().unwrap().push(ptr.0);
        Ok(())
    }
    fn claim_grouped_backing(&self, base: DevicePtr) -> Result<()> {
        let call = self.claim_calls.fetch_add(1, Ordering::SeqCst);
        // Checked before any lock is taken, so a panic poisons nothing.
        assert!(
            call != self.panic_claim_at.load(Ordering::SeqCst),
            "injected claim panic"
        );
        ensure!(
            call != self.refuse_claim_at.load(Ordering::SeqCst),
            "injected claim refusal"
        );
        ensure!(self.live_has(base.0), "claim requires a live base");
        ensure!(self.claimed.lock().unwrap().insert(base.0), "double claim");
        Ok(())
    }
    fn copy_h2d(&self, _src: &[u8], _dst: DevicePtr) -> Result<()> {
        anyhow::bail!("unused by ownership tests")
    }
    fn copy_d2h(&self, _src: DevicePtr, _dst: &mut [u8]) -> Result<()> {
        anyhow::bail!("unused by ownership tests")
    }
    fn copy_d2d(&self, _src: DevicePtr, _dst: DevicePtr, _bytes: usize) -> Result<()> {
        anyhow::bail!("unused by ownership tests")
    }
    fn launch(
        &self,
        _func: KernelHandle,
        _grid: [u32; 3],
        _block: [u32; 3],
        _shared_mem: u32,
        _stream: u64,
        _params: &mut [*mut std::ffi::c_void],
    ) -> Result<()> {
        anyhow::bail!("unused by ownership tests")
    }
    fn synchronize(&self, _stream: u64) -> Result<()> {
        Ok(())
    }
    fn default_stream(&self) -> u64 {
        0
    }
    #[track_caller]
    fn kernel(&self, _module: &str, _func_name: &str) -> Result<KernelHandle> {
        anyhow::bail!("unused by ownership tests")
    }
    fn memset(&self, _ptr: DevicePtr, _value: u8, _bytes: usize) -> Result<()> {
        anyhow::bail!("unused by ownership tests")
    }
    fn memset_async(&self, _ptr: DevicePtr, _value: u8, _bytes: usize, _stream: u64) -> Result<()> {
        anyhow::bail!("unused by ownership tests")
    }
    fn total_memory(&self) -> Result<usize> {
        Ok(0)
    }
    fn free_memory(&self) -> Result<usize> {
        Ok(0)
    }
}

/// `count` one-view groups (layers 0..count, gate projection), built
/// directly: 272 bytes each (256-byte guard + a 16-byte packed view).
fn tiny_load(count: usize) -> RoutedBackingLoad {
    let mut groups = Vec::with_capacity(count);
    let mut by_name = HashMap::new();
    for layer in 0..count {
        let name = format!("{PREFIX}{layer}.mlp.experts.0.gate_proj.weight");
        by_name.insert(name.clone(), (layer, 0));
        groups.push(PlanGroup {
            shard: PathBuf::from("tiny.safetensors"),
            layer_prefix: format!("{PREFIX}{layer}"),
            projection: RoutedProjection::Gate,
            bytes: ALIGN + 16,
            views: vec![View {
                name,
                expert: 0,
                scale: false,
                offset: ALIGN,
                bytes: 16,
                shape: vec![16],
                dtype: WeightDtype::UInt8,
                abs_offset: 0,
            }],
            base: None,
        });
    }
    RoutedBackingLoad {
        groups,
        by_name,
        registry: Some(RoutedGroupRegistry::default()),
        copied: HashSet::new(),
    }
}

fn tiny_name(layer: usize) -> String {
    format!("{PREFIX}{layer}.mlp.experts.0.gate_proj.weight")
}

#[test]
fn strict_owner_mock_rejects_unknown_interior_and_double_frees_and_keeps_failed_bases() -> Result<()>
{
    let gpu = StrictOwnerMock::new();
    let base = gpu.alloc(272)?;
    assert!(gpu.free(DevicePtr(0x9000)).is_err(), "unknown");
    assert!(gpu.free(base.offset(ALIGN)).is_err(), "interior");
    gpu.fail_free.store(true, Ordering::SeqCst);
    let error = gpu.free(base).unwrap_err();
    assert!(format!("{error:#}").contains("injected ordinary free failure"));
    assert!(gpu.live_has(base.0), "a failed free keeps the base live");
    gpu.fail_free.store(false, Ordering::SeqCst);
    gpu.free(base)?;
    let error = gpu.free(base).unwrap_err();
    assert!(format!("{error:#}").contains("double free"));
    assert_eq!(gpu.freed(), [base.0]);
    let claimed = gpu.alloc(272)?;
    gpu.claim_grouped_backing(claimed)?;
    assert!(
        gpu.free(claimed).is_err(),
        "claimed bases refuse the ordinary free"
    );
    assert!(gpu.live_has(claimed.0));
    Ok(())
}

#[test]
fn ordinary_free_error_in_rollback_keeps_base_live_and_propagates() -> Result<()> {
    // Refused registration: a prior owner already holds the view's name.
    let mut load = tiny_load(1);
    load.registry.as_mut().expect("live registry").register(
        format!("{PREFIX}0"),
        RoutedProjection::Gate,
        DevicePtr(0x7f00_0000_0000),
        ALIGN + 16,
        vec![RoutedBackingMember {
            name: tiny_name(0),
            expert: 0,
            scale: false,
            offset: ALIGN,
            bytes: 16,
            shape: vec![16],
            dtype: WeightDtype::UInt8,
        }],
    )?;
    let gpu = StrictOwnerMock::new();
    gpu.fail_free.store(true, Ordering::SeqCst);
    let error = format!("{:#}", load.materialize(0, &gpu).unwrap_err());
    assert!(error.contains("duplicate grouped member"), "{error}");
    assert!(
        error.contains("releasing the unclaimed grouped base"),
        "{error}"
    );
    assert!(error.contains("injected ordinary free failure"), "{error}");
    let base = gpu.allocations()[0];
    assert!(gpu.live_has(base) && !gpu.is_claimed(base));
    assert!(gpu.freed().is_empty());
    assert!(load.groups[0].base.is_none());
    Ok(())
}

#[test]
fn partial_eager_prefix_failure_retains_claimed_prefix_and_frees_only_the_refused_base()
-> Result<()> {
    let mut load = tiny_load(3);
    let gpu = StrictOwnerMock::new();
    gpu.refuse_claim_at.store(2, Ordering::SeqCst);
    let error = format!("{:#}", load.allocate_all(&gpu).unwrap_err());
    assert!(error.contains("injected claim refusal"), "{error}");
    let allocations = gpu.allocations();
    assert_eq!(allocations.len(), 3);
    let registry = load.registry.as_ref().expect("live registry");
    for (layer, &base) in allocations.iter().enumerate().take(2) {
        // Retain-until-teardown: the claimed prefix is neither freed nor unregistered.
        assert_eq!(load.groups[layer].base, Some(DevicePtr(base)));
        assert!(gpu.live_has(base) && gpu.is_claimed(base));
        assert!(registry.contains(&tiny_name(layer)));
    }
    assert!(load.groups[2].base.is_none());
    assert!(!registry.contains(&tiny_name(2)));
    assert!(!gpu.live_has(allocations[2]) && !gpu.is_claimed(allocations[2]));
    assert_eq!(
        gpu.freed(),
        [allocations[2]],
        "exactly one free, of the refused base only"
    );
    Ok(())
}

#[test]
fn panic_mid_claim_retains_the_registered_owner_and_frees_nothing() {
    let mut load = tiny_load(2);
    let gpu = StrictOwnerMock::new();
    gpu.panic_claim_at.store(1, Ordering::SeqCst);
    let outcome =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| load.allocate_all(&gpu)));
    assert!(
        outcome.is_err(),
        "the injected claim panic unwinds out of allocate_all"
    );
    let allocations = gpu.allocations();
    assert_eq!(allocations.len(), 2);
    assert!(gpu.is_claimed(allocations[0]) && load.groups[0].base.is_some());
    // The panicking owner was registered but never claimed or published:
    // retained until teardown, distinct from an unclaimed rollback (freed).
    assert!(gpu.live_has(allocations[1]) && !gpu.is_claimed(allocations[1]));
    assert!(load.groups[1].base.is_none());
    let registry = load.registry.as_ref().expect("live registry");
    assert!(registry.contains(&tiny_name(0)) && registry.contains(&tiny_name(1)));
    assert!(
        gpu.freed().is_empty(),
        "no rollback free runs across a panic"
    );
}

#[test]
fn materialize_twice_is_refused_without_a_second_allocation() -> Result<()> {
    let mut load = tiny_load(1);
    let gpu = StrictOwnerMock::new();
    assert_eq!(load.materialize(0, &gpu)?, ALIGN + 16);
    let error = format!("{:#}", load.materialize(0, &gpu).unwrap_err());
    assert!(
        error.contains("grouped backing already allocated"),
        "{error}"
    );
    assert_eq!(gpu.allocations().len(), 1);
    assert!(gpu.freed().is_empty());
    Ok(())
}

#[test]
fn claim_refusal_is_transactional_and_a_rollback_free_error_is_reported() -> Result<()> {
    let mut load = tiny_load(1);
    let gpu = StrictOwnerMock::new();
    gpu.refuse_claim_at.store(0, Ordering::SeqCst);
    let error = format!("{:#}", load.materialize(0, &gpu).unwrap_err());
    assert!(error.contains("injected claim refusal"), "{error}");
    let first = gpu.allocations()[0];
    assert_eq!(gpu.freed(), [first]);
    assert!(load.registry.as_ref().expect("live registry").is_empty());
    assert!(load.groups[0].base.is_none());
    // Same refusal, but the rollback free also fails: both are reported, the
    // base stays live and unclaimed, and the registration was still undone.
    let mut load = tiny_load(1);
    let gpu = StrictOwnerMock::new();
    gpu.refuse_claim_at.store(0, Ordering::SeqCst);
    gpu.fail_free.store(true, Ordering::SeqCst);
    let error = format!("{:#}", load.materialize(0, &gpu).unwrap_err());
    assert!(error.contains("injected claim refusal"), "{error}");
    assert!(error.contains("injected ordinary free failure"), "{error}");
    let base = gpu.allocations()[0];
    assert!(gpu.live_has(base) && !gpu.is_claimed(base));
    assert!(gpu.freed().is_empty());
    assert!(load.registry.as_ref().expect("live registry").is_empty());
    assert!(load.groups[0].base.is_none());
    Ok(())
}

/// Native-MTP routed expert metadata in the target's canonical shapes, under
/// the `mtp.layers.0` prefix the MTP layer loads from.
fn native_mtp_experts() -> Vec<TensorMeta> {
    let mut out = Vec::new();
    for projection in [
        RoutedProjection::Gate,
        RoutedProjection::Up,
        RoutedProjection::Down,
    ] {
        for expert in 0..2 {
            for scale in [false, true] {
                let mut meta = tensor(0, expert, projection, scale);
                meta.name = meta.name.replacen(PREFIX, "mtp.layers.", 1);
                out.push(meta);
            }
        }
    }
    out
}

#[test]
fn native_mtp_experts_beside_the_target_inventory_are_never_planned_or_given_a_destination()
-> Result<()> {
    assert!(
        native_mtp_experts()
            .iter()
            .all(|meta| meta.name.starts_with("mtp.layers.0.mlp.experts."))
    );
    for meta in native_mtp_experts() {
        assert!(parse_member(&meta)?.is_none(), "{}", meta.name);
    }
    let mut all = inventory();
    all[0].1.extend(native_mtp_experts());
    let mut load = RoutedBackingLoad::from_inventories(all)?;
    assert_eq!(
        load.by_name.len(),
        147_456,
        "only the target routed originals are planned"
    );
    assert_eq!(load.groups.len(), 1152);
    assert!(load.by_name.keys().all(|name| !name.starts_with("mtp.")));
    assert!(
        load.groups
            .iter()
            .all(|group| group.layer_prefix.starts_with(PREFIX))
    );
    let gpu = OwnershipMock::new();
    let shard = PathBuf::from("synthetic.safetensors");
    for meta in native_mtp_experts() {
        assert!(
            load.destination(&shard, &meta, &gpu)?.is_none(),
            "{}",
            meta.name
        );
    }
    assert_eq!(
        gpu.allocation_count(),
        0,
        "an MTP tensor never allocates or claims a group"
    );
    assert!(gpu.claimed.lock().unwrap().is_empty());
    Ok(())
}
