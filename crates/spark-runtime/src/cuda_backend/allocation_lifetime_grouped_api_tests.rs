// SPDX-License-Identifier: AGPL-3.0-only

//! CPU-only unit tests for grouped allocation lifetime ownership transitions.

use super::*;
use crate::gpu::{DevicePtr, GpuBackend, KernelHandle, mock::MockGpuBackend};
use crate::weights::{
    RoutedBackingMember, RoutedGroupRegistry, RoutedProjection, WeightDtype, WeightStore,
    WeightTensor,
};
use std::sync::{
    Mutex, MutexGuard,
    atomic::{AtomicBool, Ordering},
};

struct StrictApi {
    inner: MockGpuBackend,
    live: Mutex<HashMap<u64, Allocation>>,
    calls: Mutex<Vec<u64>>,
    fail: AtomicBool,
    panic: AtomicBool,
}

impl StrictApi {
    fn new() -> Self {
        Self {
            inner: MockGpuBackend::new(),
            live: Mutex::new(HashMap::new()),
            calls: Mutex::new(Vec::new()),
            fail: AtomicBool::new(false),
            panic: AtomicBool::new(false),
        }
    }
    fn table(&self) -> MutexGuard<'_, HashMap<u64, Allocation>> {
        self.live.lock().unwrap_or_else(|error| error.into_inner())
    }
    fn driver_free(&self, base: DevicePtr) -> Result<()> {
        ensure!(
            self.inner.read_alloc(base).is_some(),
            "strict driver rejects unknown/interior/double free"
        );
        self.calls.lock().unwrap().push(base.0);
        assert!(!self.panic.load(Ordering::SeqCst), "injected driver panic");
        ensure!(
            !self.fail.load(Ordering::SeqCst),
            "injected driver free failure"
        );
        self.inner.free(base)
    }
}

impl GpuBackend for StrictApi {
    fn supports_grouped_backing(&self) -> bool {
        true
    }
    fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
        let base = self.inner.alloc(bytes)?;
        self.table().insert(base.0, Allocation::new(bytes));
        Ok(base)
    }
    fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr> {
        self.alloc(bytes)
    }
    fn free(&self, base: DevicePtr) -> Result<()> {
        let allocation = take_for_free(&mut self.table(), base.0)?
            .ok_or_else(|| anyhow::anyhow!("strict backend rejects unknown free"))?;
        if let Err(error) = self.driver_free(base) {
            self.table().insert(base.0, allocation);
            return Err(error);
        }
        Ok(())
    }
    fn claim_grouped_backing(&self, base: DevicePtr) -> Result<()> {
        claim_grouped(&mut self.table(), base.0)
    }
    fn free_grouped_backing(&self, base: DevicePtr) -> Result<()> {
        let mut live = self.table();
        prepare_grouped_free(&mut live, base.0)?;
        self.driver_free(base)?;
        complete_grouped_free(&mut live, base.0)
    }
    fn free_all_allocations(&self) -> Result<usize> {
        let (allocations, retained) = take_releasable(&mut self.table());
        let mut bytes = 0;
        for (base, allocation) in allocations {
            if let Err(error) = self.driver_free(DevicePtr(base)) {
                self.table().insert(base, allocation);
                return Err(error);
            }
            bytes += allocation.bytes;
        }
        ensure!(retained == 0, "retained grouped allocations");
        Ok(bytes)
    }
    fn retain_h2d_destination(&self, base: DevicePtr, offset: usize, bytes: usize) -> Result<()> {
        self.table()
            .get_mut(&base.0)
            .ok_or_else(|| anyhow::anyhow!("H2D requires exact owning base"))?
            .retain(offset, bytes)
    }
    fn release_h2d_destination(&self, base: DevicePtr) -> Result<()> {
        self.table()
            .get_mut(&base.0)
            .ok_or_else(|| anyhow::anyhow!("H2D requires exact owning base"))?
            .release()
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
    fn synchronize(&self, stream: u64) -> Result<()> {
        self.inner.synchronize(stream)
    }
    fn default_stream(&self) -> u64 {
        self.inner.default_stream()
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
}

#[test]
fn unsupported_backend_defaults_reject_before_any_group_allocation() {
    let backend = MockGpuBackend::new();
    let gpu: &dyn GpuBackend = &backend;
    assert!(!gpu.supports_grouped_backing());
    assert!(gpu.claim_grouped_backing(DevicePtr::NULL).is_err());
    assert!(gpu.free_grouped_backing(DevicePtr::NULL).is_err());
}

#[test]
fn real_api_driver_failure_then_ordinary_bulk_or_grouped_retry_never_calls_driver_again()
-> Result<()> {
    let backend = StrictApi::new();
    let gpu: &dyn GpuBackend = &backend;
    let base = gpu.alloc(1024)?;
    gpu.claim_grouped_backing(base)?;
    backend.fail.store(true, Ordering::SeqCst);
    assert!(gpu.free_grouped_backing(base).is_err());
    backend.fail.store(false, Ordering::SeqCst);
    assert!(gpu.free(base).is_err());
    assert!(gpu.free(base.offset(256)).is_err());
    assert!(gpu.free_all_allocations().is_err());
    assert!(gpu.free_grouped_backing(base).is_err());
    assert!(gpu.retain_h2d_destination(base, 256, 8).is_err());
    assert_eq!(*backend.calls.lock().unwrap(), [base.0]);
    assert!(backend.inner.read_alloc(base).is_some());
    Ok(())
}

#[test]
fn real_api_live_group_rejects_individual_base_view_and_bulk_frees() -> Result<()> {
    let backend = StrictApi::new();
    let gpu: &dyn GpuBackend = &backend;
    let base = gpu.alloc(1024)?;
    gpu.claim_grouped_backing(base)?;
    assert!(gpu.free(base).is_err());
    assert!(gpu.free(base.offset(256)).is_err());
    assert!(gpu.free_all_allocations().is_err());
    assert!(backend.calls.lock().unwrap().is_empty());
    gpu.free_grouped_backing(base)?;
    assert_eq!(*backend.calls.lock().unwrap(), [base.0]);
    assert!(backend.inner.read_alloc(base).is_none());
    assert!(gpu.free_grouped_backing(base).is_err());
    assert_eq!(*backend.calls.lock().unwrap(), [base.0]);
    Ok(())
}

#[test]
fn real_api_pending_lease_prevents_driver_and_stays_quarantined_after_completion() -> Result<()> {
    let backend = StrictApi::new();
    let gpu: &dyn GpuBackend = &backend;
    let base = gpu.alloc(1024)?;
    gpu.claim_grouped_backing(base)?;
    gpu.retain_h2d_destination(base, 256, 8)?;
    assert!(gpu.free_grouped_backing(base).is_err());
    gpu.release_h2d_destination(base)?;
    assert!(gpu.free_grouped_backing(base).is_err());
    assert!(gpu.free_all_allocations().is_err());
    assert!(gpu.free(base).is_err());
    assert!(backend.calls.lock().unwrap().is_empty());
    Ok(())
}

#[test]
fn real_api_driver_panic_keeps_backend_quarantine_through_poisoned_mutex() -> Result<()> {
    let backend = StrictApi::new();
    let gpu: &dyn GpuBackend = &backend;
    let base = gpu.alloc(1024)?;
    gpu.claim_grouped_backing(base)?;
    backend.panic.store(true, Ordering::SeqCst);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(
            || gpu.free_grouped_backing(base)
        ))
        .is_err()
    );
    backend.panic.store(false, Ordering::SeqCst);
    assert!(gpu.free(base).is_err());
    assert!(gpu.free_all_allocations().is_err());
    assert!(gpu.free_grouped_backing(base).is_err());
    assert_eq!(*backend.calls.lock().unwrap(), [base.0]);
    Ok(())
}

#[test]
fn real_model_owner_tombstones_views_and_uses_only_grouped_backend_api() -> Result<()> {
    let backend = StrictApi::new();
    let gpu: &dyn GpuBackend = &backend;
    let base = gpu.alloc(768)?;
    gpu.claim_grouped_backing(base)?;
    let prefix = "model.language_model.layers.0";
    let mut registry = RoutedGroupRegistry::default();
    let mut weights = HashMap::new();
    let mut members = Vec::new();
    for (scale, offset, bytes, dtype) in [
        (false, 256, 16, WeightDtype::UInt8),
        (true, 512, 8, WeightDtype::FP8E4M3),
    ] {
        let name = format!(
            "{prefix}.mlp.experts.0.gate_proj.{}",
            if scale { "weight_scale" } else { "weight" }
        );
        weights.insert(
            name.clone(),
            WeightTensor {
                ptr: base.offset(offset),
                shape: vec![bytes],
                dtype,
            },
        );
        members.push(RoutedBackingMember {
            name,
            expert: 0,
            scale,
            offset,
            bytes,
            shape: vec![bytes],
            dtype,
        });
    }
    registry.register(prefix.into(), RoutedProjection::Gate, base, 768, members)?;
    let store = WeightStore::with_routed_groups(weights, registry)?;
    let groups = store.routed_original_groups(prefix).unwrap();
    backend.fail.store(true, Ordering::SeqCst);
    assert!(
        groups
            .release_projection(RoutedProjection::Gate, gpu)
            .is_err()
    );
    assert!(store.is_empty());
    backend.fail.store(false, Ordering::SeqCst);
    assert!(gpu.free(base).is_err());
    assert!(gpu.free_all_allocations().is_err());
    assert!(
        groups
            .release_projection(RoutedProjection::Gate, gpu)
            .is_err()
    );
    assert_eq!(*backend.calls.lock().unwrap(), [base.0]);
    Ok(())
}
