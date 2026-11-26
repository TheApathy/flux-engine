// SPDX-License-Identifier: AGPL-3.0-only

//! Transactional ownership for raw device allocations during construction.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::gpu_lifetime::require_completion;

pub(crate) struct DeviceAllocations<'a> {
    gpu: &'a dyn GpuBackend,
    pointers: Vec<DevicePtr>,
}

impl<'a> DeviceAllocations<'a> {
    pub(crate) fn new(gpu: &'a dyn GpuBackend) -> Self {
        Self {
            gpu,
            pointers: Vec::new(),
        }
    }

    pub(crate) fn alloc(&mut self, bytes: usize) -> Result<DevicePtr> {
        let pointer = self.gpu.alloc(bytes)?;
        Ok(self.adopt(pointer))
    }

    pub(crate) fn adopt(&mut self, pointer: DevicePtr) -> DevicePtr {
        if !pointer.is_null() {
            self.pointers.push(pointer);
        }
        pointer
    }

    /// Transfer every pointer into the completed owner, with no device free.
    pub(crate) fn commit(&mut self) {
        self.pointers.clear();
    }
}

impl Drop for DeviceAllocations<'_> {
    fn drop(&mut self) {
        if self.pointers.is_empty() {
            return;
        }
        require_completion(
            self.gpu.synchronize_device(),
            "partial device allocation rollback",
        );
        for pointer in self.pointers.drain(..).rev() {
            require_completion(self.gpu.free(pointer), "partial device allocation free");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::{KernelHandle, PinnedHostBuffer, mock::MockGpuBackend};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct FailingGpu {
        inner: MockGpuBackend,
        calls: AtomicUsize,
        fail_at: usize,
    }

    impl FailingGpu {
        fn allocation(&self) -> Result<()> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            anyhow::ensure!(call != self.fail_at, "injected allocation failure {call}");
            Ok(())
        }
    }

    impl GpuBackend for FailingGpu {
        fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
            self.allocation()?;
            self.inner.alloc(bytes)
        }
        fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr> {
            self.alloc(bytes)
        }
        fn alloc_host_pinned(&self, bytes: usize) -> Result<PinnedHostBuffer> {
            self.allocation()?;
            self.inner.alloc_host_pinned(bytes)
        }
        fn free(&self, ptr: DevicePtr) -> Result<()> {
            self.inner.free(ptr)
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
            kernel: KernelHandle,
            grid: [u32; 3],
            block: [u32; 3],
            shared: u32,
            stream: u64,
            params: &mut [*mut std::ffi::c_void],
        ) -> Result<()> {
            self.inner
                .launch(kernel, grid, block, shared, stream, params)
        }
        fn synchronize(&self, stream: u64) -> Result<()> {
            self.inner.synchronize(stream)
        }
        fn default_stream(&self) -> u64 {
            self.inner.default_stream()
        }
        fn kernel(&self, module: &str, name: &str) -> Result<KernelHandle> {
            self.inner.kernel(module, name)
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
    fn every_late_device_or_pinned_failure_reclaims_owned_device_buffers() {
        for fail_at in 0..7 {
            let gpu = FailingGpu {
                inner: MockGpuBackend::new(),
                calls: AtomicUsize::new(0),
                fail_at,
            };
            let result = (|| -> Result<()> {
                let mut allocations = DeviceAllocations::new(&gpu);
                for _ in 0..4 {
                    allocations.alloc(32)?;
                }
                let _position = gpu.alloc_host_pinned(16)?;
                let _tokens = gpu.alloc_host_pinned(16)?;
                let _draft = gpu.alloc_host_pinned(16)?;
                anyhow::bail!("fixture always rolls back");
            })();
            assert!(result.is_err());
            assert_eq!(gpu.inner.alloc_count(), 0, "failure {fail_at}");
        }
    }

    #[test]
    fn adopted_pool_pointer_is_reclaimed_on_failure_and_committed_pointer_is_retained() {
        let gpu = MockGpuBackend::new();
        let pooled = gpu.alloc(32).unwrap();
        {
            let mut allocations = DeviceAllocations::new(&gpu);
            allocations.adopt(pooled);
        }
        assert_eq!(gpu.alloc_count(), 0);
        let owned = {
            let mut allocations = DeviceAllocations::new(&gpu);
            let owned = allocations.alloc(32).unwrap();
            allocations.commit();
            owned
        };
        assert_eq!(gpu.alloc_count(), 1);
        gpu.free(owned).unwrap();
    }

    #[test]
    fn dflash_construction_uses_the_transaction_for_every_device_allocation() {
        let source = include_str!("layers/dflash_head.rs");
        let body = source
            .split("fn alloc_state(&self, gpu:")
            .nth(1)
            .unwrap()
            .split("fn propose(")
            .next()
            .unwrap();
        assert!(!body.contains("gpu.alloc("));
        assert!(body.contains("allocations.adopt(p)"));
        assert!(body.contains("allocations.commit()"));
    }
}
