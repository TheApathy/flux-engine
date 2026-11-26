// SPDX-License-Identifier: AGPL-3.0-only

//! CPU-only unit tests for fast-weight pinned bounce-copy scheduling.

use super::*;
use crate::gpu::{KernelHandle, PinnedHostSlice, PinnedHostStorage, mock::MockGpuBackend};
use std::collections::{HashMap, HashSet};
use std::ptr::NonNull;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
};

struct Storage {
    bytes: Vec<u8>,
    drops: Arc<AtomicUsize>,
}

impl PinnedHostStorage for Storage {
    fn ptr(&self) -> NonNull<u8> {
        NonNull::new(self.bytes.as_ptr() as *mut u8).unwrap()
    }
    fn len(&self) -> usize {
        self.bytes.len()
    }
}

impl Drop for Storage {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

struct Backend {
    inner: MockGpuBackend,
    failure: Mutex<Option<&'static str>>,
    ready: AtomicBool,
    panic_submit: AtomicBool,
    bad_poll: AtomicU64,
    panic_poll: AtomicU64,
    next_event: AtomicU64,
    recorded: Mutex<HashSet<u64>>,
    destroyed: Mutex<Vec<u64>>,
    polled: Mutex<Vec<u64>>,
    references: Mutex<HashMap<u64, usize>>,
    retained_ranges: Mutex<Vec<(DevicePtr, usize, usize)>>,
    released_bases: Mutex<Vec<DevicePtr>>,
    pending_submission: Mutex<Option<(DevicePtr, usize, usize)>>,
    freed_bases: Mutex<Vec<DevicePtr>>,
    submissions: AtomicUsize,
    pinned_requests: Mutex<Vec<usize>>,
    drops: Arc<AtomicUsize>,
}

impl Backend {
    fn new() -> Self {
        Self {
            inner: MockGpuBackend::new(),
            failure: Mutex::new(None),
            ready: AtomicBool::new(true),
            panic_submit: AtomicBool::new(false),
            bad_poll: AtomicU64::new(0),
            panic_poll: AtomicU64::new(0),
            next_event: AtomicU64::new(1),
            recorded: Mutex::new(HashSet::new()),
            destroyed: Mutex::new(Vec::new()),
            polled: Mutex::new(Vec::new()),
            references: Mutex::new(HashMap::new()),
            retained_ranges: Mutex::new(Vec::new()),
            released_bases: Mutex::new(Vec::new()),
            pending_submission: Mutex::new(None),
            freed_bases: Mutex::new(Vec::new()),
            submissions: AtomicUsize::new(0),
            pinned_requests: Mutex::new(Vec::new()),
            drops: Arc::new(AtomicUsize::new(0)),
        }
    }
    fn fail(&self, operation: &'static str) {
        *self.failure.lock().unwrap() = Some(operation);
    }
    fn check(&self, operation: &'static str) -> Result<()> {
        ensure!(
            *self.failure.lock().unwrap() != Some(operation),
            "injected {operation} failure"
        );
        Ok(())
    }
    fn refs(&self, base: DevicePtr) -> usize {
        *self.references.lock().unwrap().get(&base.0).unwrap_or(&0)
    }
}

impl GpuBackend for Backend {
    fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
        self.inner.alloc(bytes)
    }
    fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr> {
        self.inner.alloc_managed(bytes)
    }
    fn free(&self, ptr: DevicePtr) -> Result<()> {
        ensure!(
            self.inner.read_alloc(ptr).is_some(),
            "not a live owning base"
        );
        ensure!(self.refs(ptr) == 0, "retained destination");
        self.inner.free(ptr)?;
        self.freed_bases.lock().unwrap().push(ptr);
        Ok(())
    }
    fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> Result<()> {
        self.inner.copy_h2d(src, dst)
    }
    fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()> {
        self.inner.copy_d2h(src, dst)
    }
    fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
        self.inner.copy_d2d(src, dst, bytes)
    }
    fn launch(
        &self,
        func: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        shared: u32,
        stream: u64,
        params: &mut [*mut std::ffi::c_void],
    ) -> Result<()> {
        self.inner.launch(func, grid, block, shared, stream, params)
    }
    fn synchronize(&self, _stream: u64) -> Result<()> {
        panic!("blocking synchronization is forbidden")
    }
    fn default_stream(&self) -> u64 {
        0
    }
    fn kernel(&self, module: &str, func: &str) -> Result<KernelHandle> {
        self.inner.kernel(module, func)
    }
    fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()> {
        self.inner.memset(ptr, value, bytes)
    }
    fn memset_async(&self, ptr: DevicePtr, value: u8, bytes: usize, stream: u64) -> Result<()> {
        self.inner.memset_async(ptr, value, bytes, stream)
    }
    fn total_memory(&self) -> Result<usize> {
        self.inner.total_memory()
    }
    fn free_memory(&self) -> Result<usize> {
        self.inner.free_memory()
    }
    fn alloc_host_pinned(&self, bytes: usize) -> Result<PinnedHostBuffer> {
        self.pinned_requests.lock().unwrap().push(bytes);
        Ok(PinnedHostBuffer::from_storage(Box::new(Storage {
            bytes: vec![0; bytes],
            drops: self.drops.clone(),
        })))
    }
    fn retain_h2d_destination(&self, base: DevicePtr, offset: usize, bytes: usize) -> Result<()> {
        self.check("retain")?;
        let allocation = self
            .inner
            .read_alloc(base)
            .ok_or_else(|| anyhow::anyhow!("not an owning base"))?;
        ensure!(
            offset
                .checked_add(bytes)
                .is_some_and(|end| end <= allocation.len()),
            "invalid range"
        );
        let mut pending = self.pending_submission.lock().unwrap();
        ensure!(
            pending.is_none(),
            "previous retained range was not submitted"
        );
        *self.references.lock().unwrap().entry(base.0).or_default() += 1;
        self.retained_ranges
            .lock()
            .unwrap()
            .push((base, offset, bytes));
        *pending = Some((base, offset, bytes));
        Ok(())
    }
    fn release_h2d_destination(&self, base: DevicePtr) -> Result<()> {
        self.check("release")?;
        let mut refs = self.references.lock().unwrap();
        let count = refs.get_mut(&base.0).unwrap();
        *count = count.checked_sub(1).unwrap();
        self.released_bases.lock().unwrap().push(base);
        Ok(())
    }
    unsafe fn copy_h2d_pinned_async(
        &self,
        src: PinnedHostSlice<'_>,
        dst: DevicePtr,
        _stream: u64,
    ) -> Result<()> {
        self.submissions.fetch_add(1, Ordering::SeqCst);
        let (base, offset, bytes) = self
            .pending_submission
            .lock()
            .unwrap()
            .take()
            .expect("submission requires its own retained range");
        assert!(self.refs(base) > 0, "submission requires a live base lease");
        assert_eq!(bytes, src.len());
        assert_eq!(base.0.checked_add(u64::try_from(offset)?), Some(dst.0));
        if self.panic_submit.load(Ordering::SeqCst) {
            panic!("injected submission panic");
        }
        self.inner.copy_h2d(src.as_bytes(), dst)?;
        self.check("submit")
    }
    fn create_event(&self) -> Result<u64> {
        self.check("create")?;
        Ok(self.next_event.fetch_add(1, Ordering::SeqCst))
    }
    fn record_event(&self, event: u64, _stream: u64) -> Result<()> {
        self.check("record")?;
        self.recorded.lock().unwrap().insert(event);
        Ok(())
    }
    fn poll_event(&self, event: u64) -> Result<bool> {
        self.polled.lock().unwrap().push(event);
        assert!(
            self.recorded.lock().unwrap().contains(&event),
            "unrecorded event cannot prove completion"
        );
        if self.panic_poll.load(Ordering::SeqCst) == event {
            panic!("injected poll panic");
        }
        self.check("poll")?;
        ensure!(
            self.bad_poll.load(Ordering::SeqCst) != event,
            "injected event-specific poll failure"
        );
        Ok(self.ready.load(Ordering::SeqCst))
    }
    fn destroy_event(&self, event: u64) -> Result<()> {
        self.check("destroy")?;
        self.destroyed.lock().unwrap().push(event);
        Ok(())
    }
}

fn copier(gpu: &Backend, slots: usize) -> PinnedBounceCopier<'_> {
    PinnedBounceCopier::with_limits(gpu, slots, 4, Duration::ZERO).unwrap()
}

#[test]
fn shard_capacity_includes_scalar_image_and_never_exceeds_existing_chunk_cap() {
    for (tensor, slab, expected) in [
        (0, 0, 1),
        (1, 0, 1),
        (0, 1, 1),
        (819_200, 1_179_904, 1_179_904),
        (1_179_904, 819_200, 1_179_904),
        (CHUNK_BYTES - 1, 0, CHUNK_BYTES - 1),
        (CHUNK_BYTES, 0, CHUNK_BYTES),
        (CHUNK_BYTES + 1, 0, CHUNK_BYTES),
        (0, CHUNK_BYTES + 1, CHUNK_BYTES),
        (usize::MAX, usize::MAX, CHUNK_BYTES),
    ] {
        assert_eq!(shard_chunk_bytes(tensor, slab), expected);
    }
}

#[test]
fn shard_constructor_uses_one_slot_only_for_nonempty_single_chunk_payloads() {
    for (tensor, slab, expected, slots) in [
        (0, 0, 1, SLOTS),
        (3, 7, 7, 1),
        (7, 3, 7, 1),
        (819_200, 1_179_904, 1_179_904, 1),
    ] {
        let gpu = Backend::new();
        let copy = PinnedBounceCopier::new(&gpu, tensor, slab).unwrap();
        assert_eq!(copy.chunk_bytes, expected);
        assert_eq!(copy.slots.len(), slots);
        assert_eq!(*gpu.pinned_requests.lock().unwrap(), vec![expected; slots]);
        drop(copy);
        assert_eq!(gpu.drops.load(Ordering::SeqCst), slots);
    }
}

#[test]
fn slot_selection_preserves_empty_and_multichunk_paths_without_large_allocations() {
    for (tensor, slab, expected) in [
        (0, 0, SLOTS),
        (1, 0, 1),
        (0, 1, 1),
        (CHUNK_BYTES, 0, 1),
        (0, CHUNK_BYTES, 1),
        (CHUNK_BYTES + 1, 0, SLOTS),
        (0, CHUNK_BYTES + 1, SLOTS),
        (usize::MAX, usize::MAX, SLOTS),
    ] {
        assert_eq!(shard_slot_count(tensor, slab), expected);
    }
}

#[test]
fn invalid_explicit_slot_and_chunk_sizes_still_fail_before_pinned_allocation() {
    let gpu = Backend::new();
    assert!(PinnedBounceCopier::with_limits(&gpu, 0, 4, Duration::ZERO).is_err());
    assert!(PinnedBounceCopier::with_limits(&gpu, 1, 0, Duration::ZERO).is_err());
    assert!(gpu.pinned_requests.lock().unwrap().is_empty());
}

#[test]
fn selected_one_slot_constructor_keeps_uncertain_dma_owners_on_submission_error() {
    let gpu = Backend::new();
    gpu.fail("submit");
    let dst = gpu.alloc(4).unwrap();
    let mut copy = PinnedBounceCopier::new(&gpu, 4, 0).unwrap();
    assert_eq!(copy.slots.len(), 1);
    assert!(copy.copy(&[1; 4], dst).is_err());
    drop(copy);
    assert_eq!(gpu.refs(dst), 1);
    assert_eq!(gpu.drops.load(Ordering::SeqCst), 0);
    assert!(gpu.polled.lock().unwrap().is_empty());
    assert!(gpu.destroyed.lock().unwrap().is_empty());
    assert!(gpu.free(dst).is_err());
}

#[test]
fn small_capacities_copy_exact_tails_with_unchanged_destination_retention() {
    for capacity in [1, 2, 3, 4, 7, 16] {
        for len in [
            0,
            1,
            capacity,
            capacity + 1,
            capacity * SLOTS,
            capacity * SLOTS + 1,
        ] {
            let gpu = Backend::new();
            let src: Vec<u8> = (0..len).map(|index| (index % 251) as u8).collect();
            let dst = gpu.alloc(len.max(1)).unwrap();
            let mut copy = PinnedBounceCopier::new(&gpu, capacity, 0).unwrap();
            copy.copy(&src, dst).unwrap();
            assert_eq!(
                gpu.submissions.load(Ordering::SeqCst),
                len.div_ceil(capacity)
            );
            if len > 0 {
                assert!(gpu.free(dst).is_err());
            }
            copy.drain().unwrap();
            assert_eq!(gpu.refs(dst), 0);
            assert_eq!(&gpu.inner.read_alloc(dst).unwrap()[..len], src.as_slice());
            drop(copy);
            assert_eq!(gpu.drops.load(Ordering::SeqCst), 1);
            gpu.free(dst).unwrap();
        }
    }
}

#[test]
fn successful_copy_drains_and_releases_each_source_event_and_destination_reference() {
    let gpu = Backend::new();
    let dst = gpu.alloc(8).unwrap();
    let mut copy = copier(&gpu, 2);
    copy.copy(&[1; 8], dst).unwrap();
    assert_eq!(gpu.refs(dst), 2);
    assert!(gpu.free(dst).is_err());
    copy.drain().unwrap();
    assert_eq!(gpu.refs(dst), 0);
    assert_eq!(gpu.destroyed.lock().unwrap().len(), 2);
    drop(copy);
    assert_eq!(gpu.drops.load(Ordering::SeqCst), 2);
    gpu.free(dst).unwrap();
}

#[test]
fn timeout_keeps_event_and_bytes_and_forbids_retry_or_overwrite() {
    let gpu = Backend::new();
    gpu.ready.store(false, Ordering::SeqCst);
    let dst = gpu.alloc(8).unwrap();
    let mut copy = copier(&gpu, 1);
    assert!(copy.copy(&[1; 8], dst).is_err());
    assert!(matches!(
        copy.slots[0].flight,
        Flight::Recorded { event: 1, .. }
    ));
    assert!(copy.copy(&[2; 8], dst).is_err());
    assert_eq!(
        copy.slots[0]
            .buffer
            .as_ref()
            .unwrap()
            .pinned_slice(4)
            .unwrap()
            .as_bytes(),
        &[1; 4]
    );
    assert_eq!(gpu.submissions.load(Ordering::SeqCst), 1);
    drop(copy);
    assert_eq!(gpu.drops.load(Ordering::SeqCst), 0);
    assert_eq!(gpu.refs(dst), 1);
    assert!(gpu.destroyed.lock().unwrap().is_empty());
}

#[test]
fn submission_and_record_errors_never_poll_unrecorded_events_or_release_dma_owners() {
    for operation in ["submit", "record"] {
        let gpu = Backend::new();
        gpu.fail(operation);
        let dst = gpu.alloc(4).unwrap();
        let mut copy = copier(&gpu, 1);
        assert!(copy.copy(&[1; 4], dst).is_err());
        assert!(matches!(
            copy.slots[0].flight,
            Flight::Untracked { event: 1, .. }
        ));
        drop(copy);
        assert!(gpu.polled.lock().unwrap().is_empty());
        assert!(gpu.destroyed.lock().unwrap().is_empty());
        assert_eq!(gpu.refs(dst), 1);
        assert_eq!(gpu.drops.load(Ordering::SeqCst), 0);
        assert!(gpu.free(dst).is_err());
    }
}

#[test]
fn create_or_retain_failure_submits_nothing_and_drops_idle_source_normally() {
    for operation in ["create", "retain"] {
        let gpu = Backend::new();
        gpu.fail(operation);
        let dst = gpu.alloc(4).unwrap();
        let mut copy = copier(&gpu, 1);
        assert!(copy.copy(&[1; 4], dst).is_err());
        drop(copy);
        assert_eq!(gpu.submissions.load(Ordering::SeqCst), 0);
        assert_eq!(gpu.refs(dst), 0);
        assert_eq!(gpu.drops.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn poll_error_retains_owners_but_completed_destroy_or_release_errors_do_not_retain_source() {
    for operation in ["poll", "destroy", "release"] {
        let gpu = Backend::new();
        let dst = gpu.alloc(4).unwrap();
        let mut copy = copier(&gpu, 1);
        copy.copy(&[1; 4], dst).unwrap();
        gpu.fail(operation);
        assert!(copy.drain().is_err());
        assert!(copy.copy(&[2; 4], dst).is_err());
        drop(copy);
        assert_eq!(
            gpu.drops.load(Ordering::SeqCst),
            usize::from(operation != "poll")
        );
        assert_eq!(gpu.refs(dst), usize::from(operation != "destroy"));
    }
}

#[test]
fn caught_submission_panic_poison_prevents_reuse_and_drop_retains_dma_owners() {
    let gpu = Backend::new();
    gpu.panic_submit.store(true, Ordering::SeqCst);
    let dst = gpu.alloc(4).unwrap();
    let mut copy = copier(&gpu, 1);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| copy.copy(&[1; 4], dst))).is_err()
    );
    gpu.panic_submit.store(false, Ordering::SeqCst);
    assert!(copy.copy(&[2; 4], dst).is_err());
    drop(copy);
    assert_eq!(gpu.refs(dst), 1);
    assert_eq!(gpu.drops.load(Ordering::SeqCst), 0);
}

#[test]
fn drain_attempts_later_slots_after_error_or_panic_in_first_slot() {
    for panic in [false, true] {
        let gpu = Backend::new();
        let dst = gpu.alloc(8).unwrap();
        let mut copy = copier(&gpu, 2);
        copy.copy(&[1; 8], dst).unwrap();
        if panic {
            gpu.panic_poll.store(1, Ordering::SeqCst);
        } else {
            gpu.bad_poll.store(1, Ordering::SeqCst);
        }
        assert!(copy.drain().is_err());
        assert_eq!(gpu.refs(dst), 1);
        assert!(gpu.destroyed.lock().unwrap().contains(&2));
        drop(copy);
        assert_eq!(gpu.refs(dst), 1);
        assert_eq!(gpu.drops.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn later_proven_completion_releases_quarantine_but_never_unpoisons_failed_copier() {
    let gpu = Backend::new();
    let dst = gpu.alloc(4).unwrap();
    let mut copy = copier(&gpu, 1);
    copy.copy(&[1; 4], dst).unwrap();
    gpu.ready.store(false, Ordering::SeqCst);
    assert!(copy.drain().is_err());
    gpu.ready.store(true, Ordering::SeqCst);
    copy.drain().unwrap();
    assert_eq!(gpu.refs(dst), 0);
    assert!(copy.copy(&[2; 4], dst).is_err());
    drop(copy);
    assert_eq!(gpu.drops.load(Ordering::SeqCst), 1);
}

#[test]
fn exact_base_and_checked_subrange_are_required_before_submission() {
    let gpu = Backend::new();
    let dst = gpu.alloc(4).unwrap();
    assert!(gpu.retain_h2d_destination(dst.offset(1), 0, 1).is_err());
    assert!(gpu.retain_h2d_destination(dst, usize::MAX, 2).is_err());
    assert!(gpu.retain_h2d_destination(dst, 3, 2).is_err());
    assert_eq!(gpu.refs(dst), 0);
}

#[test]
fn grouped_views_copy_exact_tails_without_touching_guards_or_freeing_aliases() {
    let gpu = Backend::new();
    let base = gpu.alloc(32).unwrap();
    gpu.memset(base, 0xa5, 32).unwrap();
    let mut copy = copier(&gpu, 2);
    let first = [1, 2, 3, 4, 5, 6, 7];
    let second = [8, 9, 10, 11, 12, 13, 14, 15, 16];
    copy.copy_into_owned_range(&first, base, 3).unwrap();
    copy.copy_into_owned_range(&second, base, 17).unwrap();
    assert!(gpu.refs(base) > 0);
    assert!(gpu.free(base).is_err());
    assert!(gpu.free(base.offset(3)).is_err());
    assert!(gpu.free(DevicePtr(u64::MAX)).is_err());
    copy.drain().unwrap();
    assert_eq!(gpu.refs(base), 0);
    let mut expected = vec![0xa5; 32];
    expected[3..10].copy_from_slice(&first);
    expected[17..26].copy_from_slice(&second);
    assert_eq!(gpu.inner.read_alloc(base).unwrap(), expected);
    assert_eq!(
        *gpu.retained_ranges.lock().unwrap(),
        vec![
            (base, 3, 4),
            (base, 7, 3),
            (base, 17, 4),
            (base, 21, 4),
            (base, 25, 1)
        ]
    );
    assert_eq!(*gpu.released_bases.lock().unwrap(), vec![base; 5]);
    assert!(gpu.freed_bases.lock().unwrap().is_empty());
    assert!(gpu.free(base.offset(17)).is_err());
    drop(copy);
    assert_eq!(gpu.drops.load(Ordering::SeqCst), 2);
    gpu.free(base).unwrap();
    assert!(gpu.free(base).is_err());
    assert_eq!(*gpu.freed_bases.lock().unwrap(), vec![base]);
}

#[test]
fn grouped_range_arithmetic_overflow_submits_no_prefix_and_poison_is_sticky() {
    for (base, offset, src) in [
        (DevicePtr(1), usize::MAX, &[1][..]),
        (DevicePtr(u64::MAX), 0, &[1][..]),
        (DevicePtr(u64::MAX - 5), 2, &[1; 8][..]),
    ] {
        let gpu = Backend::new();
        let mut copy = copier(&gpu, 1);
        assert!(copy.copy_into_owned_range(src, base, offset).is_err());
        assert_eq!(gpu.submissions.load(Ordering::SeqCst), 0);
        assert_eq!(gpu.next_event.load(Ordering::SeqCst), 1);
        assert!(gpu.retained_ranges.lock().unwrap().is_empty());
        assert!(copy.copy_into_owned_range(&[], base, 0).is_err());
        drop(copy);
        assert_eq!(gpu.drops.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn grouped_invalid_base_or_first_chunk_range_submits_nothing() {
    for interior_base in [false, true] {
        let gpu = Backend::new();
        let base = gpu.alloc(8).unwrap();
        let mut copy = copier(&gpu, 1);
        let (destination, offset) = if interior_base {
            (base.offset(1), 0)
        } else {
            (base, 7)
        };
        assert!(
            copy.copy_into_owned_range(&[1; 2], destination, offset)
                .is_err()
        );
        assert_eq!(gpu.submissions.load(Ordering::SeqCst), 0);
        assert!(gpu.retained_ranges.lock().unwrap().is_empty());
        assert_eq!(gpu.refs(base), 0);
        assert_eq!(gpu.inner.read_alloc(base).unwrap(), vec![0; 8]);
        assert_eq!(*gpu.destroyed.lock().unwrap(), vec![1]);
        drop(copy);
        assert_eq!(gpu.drops.load(Ordering::SeqCst), 1);
        gpu.free(base).unwrap();
    }
}

#[test]
fn empty_owned_range_uses_no_backend_lease_or_submission() {
    let gpu = Backend::new();
    let base = gpu.alloc(8).unwrap();
    let mut copy = copier(&gpu, 1);
    copy.copy_into_owned_range(&[], base, 8).unwrap();
    copy.copy(&[], base).unwrap();
    assert_eq!(gpu.submissions.load(Ordering::SeqCst), 0);
    assert_eq!(gpu.next_event.load(Ordering::SeqCst), 1);
    assert!(gpu.retained_ranges.lock().unwrap().is_empty());
    copy.drain().unwrap();
    drop(copy);
    gpu.free(base).unwrap();
}

#[test]
fn grouped_uncertain_sibling_retains_only_its_base_lease_and_host_slot() {
    for operation in ["submit", "record"] {
        let gpu = Backend::new();
        let base = gpu.alloc(32).unwrap();
        let mut copy = copier(&gpu, 2);
        copy.copy_into_owned_range(&[1; 8], base, 2).unwrap();
        assert_eq!(gpu.refs(base), 2);
        gpu.fail(operation);
        assert!(copy.copy_into_owned_range(&[2; 4], base, 16).is_err());
        assert!(matches!(
            copy.slots[0].flight,
            Flight::Untracked { event: 3, destination } if destination == base
        ));
        assert!(copy.copy_into_owned_range(&[3; 4], base, 24).is_err());
        drop(copy);
        assert_eq!(gpu.refs(base), 1);
        assert_eq!(gpu.drops.load(Ordering::SeqCst), 1);
        assert!(!gpu.polled.lock().unwrap().contains(&3));
        assert!(!gpu.destroyed.lock().unwrap().contains(&3));
        assert_eq!(*gpu.released_bases.lock().unwrap(), vec![base; 2]);
        assert!(gpu.free(base).is_err());
        assert!(gpu.free(base.offset(16)).is_err());
        assert!(gpu.freed_bases.lock().unwrap().is_empty());
    }
}

#[test]
fn grouped_poll_failure_drains_completed_sibling_but_retains_uncertain_base() {
    let gpu = Backend::new();
    let base = gpu.alloc(16).unwrap();
    let mut copy = copier(&gpu, 2);
    copy.copy_into_owned_range(&[1; 8], base, 3).unwrap();
    gpu.bad_poll.store(1, Ordering::SeqCst);
    assert!(copy.drain().is_err());
    assert_eq!(gpu.refs(base), 1);
    assert_eq!(*gpu.released_bases.lock().unwrap(), vec![base]);
    assert_eq!(*gpu.destroyed.lock().unwrap(), vec![2]);
    drop(copy);
    assert_eq!(gpu.drops.load(Ordering::SeqCst), 1);
    assert!(gpu.free(base).is_err());
}

#[test]
fn grouped_timeout_can_later_release_base_without_allowing_retry() {
    let gpu = Backend::new();
    let base = gpu.alloc(16).unwrap();
    let mut copy = copier(&gpu, 2);
    copy.copy_into_owned_range(&[1; 8], base, 3).unwrap();
    gpu.ready.store(false, Ordering::SeqCst);
    assert!(copy.drain().is_err());
    assert_eq!(gpu.refs(base), 2);
    assert!(gpu.released_bases.lock().unwrap().is_empty());
    assert!(gpu.free(base).is_err());
    assert!(copy.copy_into_owned_range(&[2], base, 12).is_err());
    gpu.ready.store(true, Ordering::SeqCst);
    copy.drain().unwrap();
    assert_eq!(gpu.refs(base), 0);
    assert_eq!(*gpu.released_bases.lock().unwrap(), vec![base; 2]);
    assert!(copy.copy_into_owned_range(&[2], base, 12).is_err());
    drop(copy);
    assert_eq!(gpu.drops.load(Ordering::SeqCst), 2);
    gpu.free(base).unwrap();
}

#[test]
fn grouped_submission_panic_keeps_base_and_host_slot_quarantined() {
    let gpu = Backend::new();
    let base = gpu.alloc(16).unwrap();
    let mut copy = copier(&gpu, 1);
    gpu.panic_submit.store(true, Ordering::SeqCst);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            copy.copy_into_owned_range(&[1; 4], base, 5)
        }))
        .is_err()
    );
    gpu.panic_submit.store(false, Ordering::SeqCst);
    assert!(copy.copy_into_owned_range(&[2; 4], base, 5).is_err());
    drop(copy);
    assert_eq!(gpu.refs(base), 1);
    assert_eq!(gpu.drops.load(Ordering::SeqCst), 0);
    assert_eq!(*gpu.retained_ranges.lock().unwrap(), vec![(base, 5, 4)]);
    assert!(gpu.polled.lock().unwrap().is_empty());
    assert!(gpu.released_bases.lock().unwrap().is_empty());
    assert!(gpu.free(base).is_err());
}
