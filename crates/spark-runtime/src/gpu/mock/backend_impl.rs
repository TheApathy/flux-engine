// SPDX-License-Identifier: AGPL-3.0-only

//! `GpuBackend` impl for [`MockGpuBackend`].

use super::*;

impl GpuBackend for MockGpuBackend {
    fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
        self.check_alloc_failure()?;
        let mut next = self.next_ptr.lock();
        let ptr = *next;
        *next += bytes as u64;
        // Align to 256 bytes
        *next = (*next + 255) & !255;
        self.allocs.lock().insert(
            ptr,
            MockAlloc {
                bytes,
                data: vec![0u8; bytes],
            },
        );
        if let Some(counters) = self.alloc_counters {
            counters.note_alloc(bytes);
        }
        Ok(DevicePtr(ptr))
    }

    fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr> {
        self.alloc(bytes) // Mock: same as regular alloc
    }

    fn free(&self, ptr: DevicePtr) -> Result<()> {
        let was_carveout = self.carveout_release(ptr);
        let removed = self.allocs.lock().remove(&ptr.0);
        if removed.is_none() && !ptr.is_null() {
            self.invalid_frees.fetch_add(1, Ordering::Relaxed);
        }
        if !was_carveout && let (Some(counters), Some(alloc)) = (self.alloc_counters, removed) {
            counters.note_free(alloc.bytes);
        }
        Ok(())
    }

    fn carveout_capacity(&self) -> usize {
        self.carveout_bytes()
    }

    fn alloc_carveout(&self, bytes: usize) -> Result<DevicePtr> {
        self.carveout_alloc(bytes)
    }

    fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> Result<()> {
        let mut allocs = self.allocs.lock();
        // Support offset pointers: find the allocation containing dst
        let (offset, alloc) = find_alloc_mut(&mut allocs, dst)
            .ok_or_else(|| anyhow::anyhow!("copy_h2d: ptr {dst} not allocated"))?;
        alloc.data[offset..offset + src.len()].copy_from_slice(src);
        Ok(())
    }

    fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()> {
        let allocs = self.allocs.lock();
        // Support offset pointers: find the allocation containing src
        let (offset, alloc) = find_alloc(&allocs, src)
            .ok_or_else(|| anyhow::anyhow!("copy_d2h: ptr {src} not allocated"))?;
        dst.copy_from_slice(&alloc.data[offset..offset + dst.len()]);
        Ok(())
    }

    fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
        // Copies when both ends are mock allocations; pointers the mock
        // never handed out stay a no-op, as before.
        let mut allocs = self.allocs.lock();
        let data = match find_alloc(&allocs, src) {
            Some((off, a)) if off + bytes <= a.bytes => a.data[off..off + bytes].to_vec(),
            _ => return Ok(()),
        };
        if let Some((off, a)) = find_alloc_mut(&mut allocs, dst)
            && off + bytes <= a.bytes
        {
            a.data[off..off + bytes].copy_from_slice(&data);
        }
        Ok(())
    }

    fn launch(
        &self,
        func: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        _shared_mem: u32,
        _stream: u64,
        _params: &mut [*mut std::ffi::c_void],
    ) -> Result<()> {
        self.launches.lock().push(MockLaunch {
            func: func.0,
            grid,
            block,
        });
        Ok(())
    }

    fn synchronize(&self, _stream: u64) -> Result<()> {
        Ok(())
    }

    fn default_stream(&self) -> u64 {
        0
    }

    #[track_caller]
    fn kernel(&self, _module: &str, _func_name: &str) -> Result<KernelHandle> {
        Ok(KernelHandle(0xDEAD))
    }

    fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()> {
        let mut allocs = self.allocs.lock();
        let (offset, alloc) = find_alloc_mut(&mut allocs, ptr)
            .ok_or_else(|| anyhow::anyhow!("memset: ptr {ptr} not allocated"))?;
        alloc.data[offset..offset + bytes].fill(value);
        *self.memset_count.lock() += 1;
        Ok(())
    }

    fn memset_async(&self, ptr: DevicePtr, value: u8, bytes: usize, _stream: u64) -> Result<()> {
        self.memset(ptr, value, bytes)
    }

    fn total_memory(&self) -> Result<usize> {
        Ok(128 * 1024 * 1024 * 1024) // 128 GB
    }

    fn free_memory(&self) -> Result<usize> {
        Ok(120 * 1024 * 1024 * 1024) // 120 GB
    }
}
