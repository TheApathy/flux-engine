// SPDX-License-Identifier: AGPL-3.0-only

//! `impl GpuBackend for AtlasCudaBackend` — production CUDA backend trait body.
//!
//! ## Safety invariants for FFI and kernel launch calls in this module
//!
//! Every `unsafe` block and `unsafe extern "C"` block carries a per-site
//! `// SAFETY:` comment. Across the module:
//!
//! - **FFI surface**: most calls invoke hand-written CUDA Driver API
//!   declarations in `super::*` matching `cuda.h`, plus
//!   `AtlasRegistry::launch_on_stream*` and the CUDA Runtime API
//!   `cudaMemcpy2DAsync` (where `CUstream` is ABI-compatible with
//!   `cudaStream_t`).
//! - **Context binding**: `AtlasCudaBackend::new` binds the CUDA primary
//!   context on the initializing thread via `registry.ctx.bind_to_thread()`.
//!   Other worker threads start unbound and must invoke
//!   [`GpuBackend::bind_to_thread`] before issuing CUDA calls.
//! - **Pointer provenance**: `DevicePtr` values originate from
//!   `cuMemAlloc_v2`, `cuMemAllocManaged`, or the VMM display carveout
//!   (`cuMemAddressReserve` / `cuMemMap`). `DevicePtr::NULL` denotes an
//!   unallocated pointer.
//! - **Resource teardown**: `AtlasCudaBackend` intentionally has no `Drop`
//!   impl; callers reclaim memory via `free` / `free_all_allocations` and
//!   explicitly destroy streams, events, and graphs once GPU work drains.

use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::OnceLock;

use anyhow::{Context, Result, bail};
use atlas_core::registry::{RawCudaFunc, cuda_error_text};
use cudarc::driver::LaunchConfig;

use super::{
    AtlasCudaBackend, cuCtxGetDevice, cuCtxSetCurrent, cuDeviceGraphMemTrim, cuEventCreate,
    cuEventDestroy_v2, cuEventRecord, cuGraphDestroy, cuGraphExecDestroy,
    cuGraphInstantiateWithFlags, cuGraphLaunch, cuMemAlloc_v2, cuMemAllocHost_v2,
    cuMemAllocManaged, cuMemFree_v2, cuMemFreeHost, cuMemGetAddressRange_v2, cuMemGetInfo_v2,
    cuMemcpyDtoDAsync_v2, cuMemcpyDtoH_v2, cuMemcpyHtoD_v2, cuMemcpyHtoDAsync_v2, cuMemsetD8Async,
    cuMemsetD32Async, cuPointerGetAttribute, cuStreamBeginCapture, cuStreamCreate,
    cuStreamEndCapture, cuStreamSynchronize, cuStreamWaitEvent,
};
use crate::gpu::{
    DevicePtr, GpuBackend, GraphHandle, HostToDeviceCopy, KernelHandle, PinnedHostBuffer,
    PinnedHostSlice, PinnedHostStorage,
};

static PINNED_HOST_POOL: std::sync::Mutex<Vec<(usize, usize)>> = std::sync::Mutex::new(Vec::new());
const PINNED_HOST_POOL_MAX_SLOTS: usize = 3;
/// OS page size used to round pinned host allocations without inflating tiny
/// metadata buffers to multi-MiB slabs.
const PINNED_HOST_PAGE_BYTES: usize = 4096;
/// Minimum pinned host buffer size (4 MiB) eligible for [`PINNED_HOST_POOL`]
/// reuse. Smaller allocations are allocated and freed directly so permanent
/// metadata buffers never hold multi-MiB bounce slabs.
const PINNED_HOST_POOL_MIN_BYTES: usize = 4 * 1024 * 1024;

/// Compute the page-aligned allocation capacity for `bytes` of pinned host memory.
/// Returns `None` if rounding up to [`PINNED_HOST_PAGE_BYTES`] overflows `usize`.
fn pinned_host_alloc_capacity(bytes: usize) -> Option<usize> {
    bytes
        .max(1)
        .checked_next_multiple_of(PINNED_HOST_PAGE_BYTES)
}

/// Whether a pinned host buffer of `size_bytes` qualifies for [`PINNED_HOST_POOL`].
fn pinned_host_pool_eligible(size_bytes: usize) -> bool {
    size_bytes >= PINNED_HOST_POOL_MIN_BYTES
}

/// Whether a pooled buffer of `pool_capacity` bytes may satisfy a request of
/// `requested_bytes` without exceeding 2x over-allocation.
fn pinned_host_pool_matches(requested_bytes: usize, pool_capacity: usize) -> bool {
    pinned_host_pool_eligible(requested_bytes)
        && pool_capacity >= requested_bytes
        && pool_capacity <= requested_bytes.saturating_mul(2)
}

/// Select the tightest-fitting slot in `pool` satisfying [`pinned_host_pool_matches`].
fn select_pinned_host_pool_slot(pool: &[(usize, usize)], requested_bytes: usize) -> Option<usize> {
    if !pinned_host_pool_eligible(requested_bytes) {
        return None;
    }
    pool.iter()
        .enumerate()
        .filter(|&(_, &(_, cap))| pinned_host_pool_matches(requested_bytes, cap))
        .min_by_key(|&(_, &(_, cap))| cap)
        .map(|(idx, _)| idx)
}

struct CudaPinnedHostStorage {
    ptr: NonNull<u8>,
    bytes: usize,
    capacity: usize,
}

// SAFETY: `CudaPinnedHostStorage` uniquely owns the CUDA page-locked host
// allocation returned by `cuMemAllocHost_v2`; sending or sharing immutable
// access across threads matches standard heap buffer semantics.
unsafe impl Send for CudaPinnedHostStorage {}
unsafe impl Sync for CudaPinnedHostStorage {}

impl PinnedHostStorage for CudaPinnedHostStorage {
    fn ptr(&self) -> NonNull<u8> {
        self.ptr
    }

    fn len(&self) -> usize {
        self.bytes
    }
}

impl Drop for CudaPinnedHostStorage {
    fn drop(&mut self) {
        if pinned_host_pool_eligible(self.capacity)
            && let Ok(mut pool) = PINNED_HOST_POOL.lock()
            && pool.len() < PINNED_HOST_POOL_MAX_SLOTS
        {
            pool.push((self.ptr.as_ptr() as usize, self.capacity));
            return;
        }
        // SAFETY: `self.ptr` was allocated by `cuMemAllocHost_v2`, is uniquely
        // owned by `self`, and is not returned to the pool above.
        let status = unsafe { cuMemFreeHost(self.ptr.as_ptr().cast()) };
        if status != 0 {
            tracing::warn!(
                "cuMemFreeHost failed while dropping pinned host buffer: status {status}"
            );
        } else {
            crate::alloc_receipt::GLOBAL.note_pinned_free(self.capacity);
        }
    }
}

impl AtlasCudaBackend {
    /// Record a successful `alloc`/`alloc_managed` in the live-allocation
    /// table that `free_all_allocations` drains at model teardown.
    fn track_allocation(&self, dptr: u64, bytes: usize) {
        self.live_allocations
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(dptr, super::allocation_lifetime::Allocation::new(bytes));
    }

    /// `ATLAS_ALLOC_POISON`: fill a fresh allocation on the default stream and
    /// wait for it, so no stream can read it before the fill lands. Skipped
    /// (and reported) while the default stream is capturing a graph.
    fn poison_new_allocation(&self, dptr: u64, bytes: usize) -> Result<()> {
        let Some(config) = super::alloc_poison::selected() else {
            return Ok(());
        };
        if bytes < config.min_bytes || bytes == 0 {
            return Ok(());
        }
        if self.stream_is_capturing(self.default_stream) {
            super::alloc_poison::note_skipped_in_capture(dptr, bytes);
            return Ok(());
        }
        let pattern = config.mode.pattern();
        let (words, tail) = super::alloc_poison::fill_plan(bytes);
        if words > 0 {
            // SAFETY: `dptr` is a fresh `bytes`-byte device allocation (`words * 4 <= bytes`)
            // and `self.default_stream` is a valid non-capturing CUDA stream.
            let status = unsafe { cuMemsetD32Async(dptr, pattern, words, self.default_stream) };
            if status != 0 {
                bail!("ATLAS_ALLOC_POISON cuMemsetD32Async failed: status {status}");
            }
        }
        if tail > 0 {
            let tail_bytes = pattern.to_le_bytes();
            for (i, &byte) in tail_bytes[..tail].iter().enumerate() {
                let byte_ptr = dptr + (words * 4 + i) as u64;
                // SAFETY: `words * 4 + i < bytes`, so `byte_ptr` lies within the fresh
                // `bytes`-byte allocation on `self.default_stream`.
                let status = unsafe { cuMemsetD8Async(byte_ptr, byte, 1, self.default_stream) };
                if status != 0 {
                    bail!("ATLAS_ALLOC_POISON cuMemsetD8Async failed: status {status}");
                }
            }
        }
        if super::bounded_copy::enabled() {
            super::bounded_copy::drain(self.default_stream, "alloc poison")?;
        } else {
            // SAFETY: `self.default_stream` is a valid CUDA stream on the active context.
            let sync = unsafe { cuStreamSynchronize(self.default_stream) };
            if sync != 0 {
                bail!("cuStreamSynchronize after ATLAS_ALLOC_POISON fill failed: status {sync}");
            }
        }
        super::alloc_poison::note_poisoned(dptr, bytes);
        Ok(())
    }

    /// Fill a fresh allocation per `ATLAS_ALLOC_POISON`, freeing it if the fill
    /// fails so the caller's error leaves no untracked device memory.
    fn poison_or_free(&self, dptr: u64, bytes: usize) -> Result<()> {
        if let Err(error) = self.poison_new_allocation(dptr, bytes) {
            // SAFETY: `dptr` was just allocated by `cuMemAlloc_v2` / `cuMemAllocManaged` and
            // has not been published to callers or `live_allocations`.
            let status = unsafe { cuMemFree_v2(dptr) };
            if status != 0 {
                tracing::warn!(
                    "cuMemFree_v2 after failed ATLAS_ALLOC_POISON fill: status {status}"
                );
            }
            return Err(error);
        }
        Ok(())
    }

    fn restore_allocation(&self, dptr: u64, allocation: super::allocation_lifetime::Allocation) {
        self.live_allocations
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .entry(dptr)
            .or_insert(allocation);
    }

    fn ensure_bounded_wait_outside_capture(&self, stream: u64, what: &str) -> Result<()> {
        if self.stream_is_capturing(stream) {
            let region = crate::gpu_capture_region::active_capture_region().unwrap_or("unknown");
            bail!(
                "{what}: ATLAS_BOUNDED_HOST_COPY refuses a blocking wait during CUDA graph capture (region={region})"
            );
        }
        Ok(())
    }
}

impl GpuBackend for AtlasCudaBackend {
    fn transform_cache_identity(&self) -> Option<&str> {
        self.transform_cache_identity.as_deref()
    }

    fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
        let mut dptr: u64 = 0;
        // SAFETY: `&mut dptr` is a valid writable out-pointer on the active CUDA context.
        let status = unsafe { cuMemAlloc_v2(&mut dptr, bytes) };
        if status != 0 {
            let mut free: usize = 0;
            let mut total: usize = 0;
            // SAFETY: `&mut free` and `&mut total` are valid out-pointers for memory query.
            unsafe { cuMemGetInfo_v2(&mut free, &mut total) };
            bail!(
                "cuMemAlloc_v2 failed: status {status}, requested {bytes} bytes \
                 (device reports {:.1} MB free / {:.1} GB total)",
                free as f64 / (1024.0 * 1024.0),
                total as f64 / (1024.0 * 1024.0 * 1024.0),
            );
        }
        self.poison_or_free(dptr, bytes)?;
        self.track_allocation(dptr, bytes);
        crate::alloc_receipt::GLOBAL.note_alloc(bytes);
        Ok(DevicePtr(dptr))
    }

    fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr> {
        let mut dptr: u64 = 0;
        const CU_MEM_ATTACH_GLOBAL: u32 = 0x1;
        // SAFETY: `&mut dptr` is a valid out-pointer; `CU_MEM_ATTACH_GLOBAL` is a valid attach flag.
        let status = unsafe { cuMemAllocManaged(&mut dptr, bytes, CU_MEM_ATTACH_GLOBAL) };
        if status != 0 {
            bail!(
                "cuMemAllocManaged failed: status {status}, requested {bytes} bytes. \
                 Check system swap space: swapon --show"
            );
        }
        self.poison_or_free(dptr, bytes)?;
        self.track_allocation(dptr, bytes);
        crate::alloc_receipt::GLOBAL.note_alloc(bytes);
        Ok(DevicePtr(dptr))
    }

    fn free(&self, ptr: DevicePtr) -> Result<()> {
        if ptr.is_null() {
            return Ok(());
        }
        if super::display_carveout::free(ptr) {
            return Ok(());
        }
        // Removal and DMA retention share one mutex. Once removed, a new
        // retain refuses this base before it can submit any transfer.
        let allocation = {
            let mut live = self
                .live_allocations
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            super::allocation_lifetime::take_for_free(&mut live, ptr.0)?
        };
        // SAFETY: `ptr.0` was removed from `live_allocations` with zero pending H2D retainers
        // and non-grouped status; caller ensures no GPU kernels still reference it.
        let status = unsafe { cuMemFree_v2(ptr.0) };
        if status != 0 {
            if let Some(allocation) = allocation {
                self.restore_allocation(ptr.0, allocation);
            }
            bail!("cuMemFree_v2 failed: status {status}, ptr {ptr}");
        }
        if let Some(allocation) = allocation {
            crate::alloc_receipt::GLOBAL.note_free(allocation.bytes);
        }
        Ok(())
    }

    fn carveout_capacity(&self) -> usize {
        super::display_carveout::capacity()
    }

    fn alloc_carveout(&self, bytes: usize) -> Result<DevicePtr> {
        super::display_carveout::alloc(bytes)
    }

    fn supports_grouped_backing(&self) -> bool {
        true
    }

    #[rustfmt::skip]
    fn claim_grouped_backing(&self, base: DevicePtr) -> Result<()> {
        let mut live = self.live_allocations.lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        super::allocation_lifetime::claim_grouped(&mut live, base.0)
    }

    #[rustfmt::skip]
    fn free_grouped_backing(&self, base: DevicePtr) -> Result<()> {
        // Unlike ordinary free, keep this mutex across the single driver call:
        // quarantine must survive errors/panics and a successful free must not
        // remove a new allocation that reused this address on another thread.
        let mut live = self.live_allocations.lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let bytes = live.get(&base.0).map(|allocation| allocation.bytes);
        super::allocation_lifetime::prepare_grouped_free(&mut live, base.0)?;
        // SAFETY: `prepare_grouped_free` verified `base.0` is a live grouped owner with zero
        // pending H2D retainers and installed quarantine while holding `live`.
        let status = unsafe { cuMemFree_v2(base.0) };
        if status != 0 {
            bail!("grouped cuMemFree_v2 failed: status {status}, base {base}; permanently quarantined");
        }
        if let Some(bytes) = bytes {
            crate::alloc_receipt::GLOBAL.note_free(bytes);
        }
        super::allocation_lifetime::complete_grouped_free(&mut live, base.0)
    }

    fn free_all_allocations(&self) -> Result<usize> {
        // Drain first, free after releasing the lock: cuMemFree_v2 never
        // re-enters `alloc`/`free` on this thread, but there is no reason to
        // hold the mutex across a run of driver calls either.
        let (drained, retained) = {
            let mut live = self
                .live_allocations
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            super::allocation_lifetime::take_releasable(&mut live)
        };
        let mut freed_bytes = 0usize;
        let mut failures = 0usize;
        for (dptr, allocation) in drained {
            let bytes = allocation.bytes;
            // SAFETY: Each `dptr` in `drained` is a non-grouped, unretained live allocation
            // removed from `live_allocations` at a quiescent teardown boundary.
            let status = unsafe { cuMemFree_v2(dptr) };
            if status != 0 {
                failures += 1;
                self.restore_allocation(dptr, allocation);
                tracing::error!(
                    "free_all_allocations: cuMemFree_v2 failed: status {status}, \
                     ptr {dptr:#x}, {bytes} bytes — leaving it allocated"
                );
                continue;
            }
            crate::alloc_receipt::GLOBAL.note_free(bytes);
            freed_bytes += bytes;
        }
        if failures > 0 || retained > 0 {
            bail!(
                "free_all_allocations: freed {:.2} GB but {failures} allocation(s) \
                 failed to free and {retained} remain retained by pending H2D or explicit grouped ownership",
                freed_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            );
        }
        super::display_carveout::free_all();
        Ok(freed_bytes)
    }

    fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> Result<()> {
        if super::bounded_copy::enabled() {
            self.ensure_bounded_wait_outside_capture(self.default_stream, "bounded H2D")?;
            return super::bounded_copy::h2d(src, dst.0, self.default_stream);
        }
        // This API accepts any Rust slice, including ordinary pageable memory.
        // CUDA's async host-copy APIs require page-locked storage; use the
        // synchronous driver call for this intentionally blocking interface.
        // Atlas uses a non-blocking CUDA stream, so drain it first to preserve
        // the old method's ordering against earlier work on that stream.
        // SAFETY: `self.default_stream` is a valid CUDA stream handle on the active context.
        let sync = unsafe { cuStreamSynchronize(self.default_stream) };
        if sync != 0 {
            bail!(
                "cuStreamSynchronize before H2D failed: {}",
                cuda_error_text(sync)
            );
        }
        // SAFETY: `src` is a valid readable slice of `src.len()` bytes; caller guarantees `dst`
        // points to at least `src.len()` bytes of valid device memory.
        let status = unsafe { cuMemcpyHtoD_v2(dst.0, src.as_ptr() as *const c_void, src.len()) };
        if status != 0 {
            bail!("cuMemcpyHtoD_v2 failed: status {status}");
        }
        drain_legacy_h2d_if_selected()
    }

    fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()> {
        if super::bounded_copy::enabled() {
            self.ensure_bounded_wait_outside_capture(self.default_stream, "bounded D2H")?;
            return super::bounded_copy::d2h(src.0, dst, self.default_stream);
        }
        // `dst` is not required to be page-locked. The synchronous API is the
        // CUDA-supported path for arbitrary pageable host buffers and returns
        // only after the bytes are safe for the caller to read.
        // Synchronize Atlas's non-blocking stream before the streamless driver
        // copy so prior kernels cannot race the host read.
        super::wfet_sync::park_stream_if_enabled(self.default_stream);
        // SAFETY: `self.default_stream` is a valid CUDA stream handle on the active context.
        let sync = unsafe { cuStreamSynchronize(self.default_stream) };
        if sync != 0 {
            bail!(
                "cuStreamSynchronize before D2H failed: {}",
                cuda_error_text(sync)
            );
        }
        // SAFETY: `dst` is a uniquely borrowed writable slice of `dst.len()` bytes; caller
        // guarantees `src` holds at least `dst.len()` bytes of valid device memory.
        let status = unsafe { cuMemcpyDtoH_v2(dst.as_mut_ptr() as *mut c_void, src.0, dst.len()) };
        if status != 0 {
            bail!("cuMemcpyDtoH_v2 failed: status {status}");
        }
        Ok(())
    }

    fn copy_d2h_on_stream(&self, src: DevicePtr, dst: &mut [u8], stream: u64) -> Result<()> {
        if super::bounded_copy::enabled() {
            self.ensure_bounded_wait_outside_capture(stream, "bounded D2H")?;
            return super::bounded_copy::d2h(src.0, dst, stream);
        }
        // This is a blocking API over an arbitrary host slice. Drain the
        // producer stream for ordering, then use the pageable-safe synchronous
        // copy. The coalesced pair API below uses the same safe ordering.
        super::wfet_sync::park_stream_if_enabled(stream);
        // SAFETY: `stream` is a caller-supplied CUDA stream handle on the active context.
        let sync = unsafe { cuStreamSynchronize(stream) };
        if sync != 0 {
            bail!(
                "cuStreamSynchronize before D2H on_stream failed: {}",
                cuda_error_text(sync)
            );
        }
        // SAFETY: `dst` is a uniquely borrowed writable slice of `dst.len()` bytes; caller
        // guarantees `src` holds at least `dst.len()` bytes of valid device memory.
        let status = unsafe { cuMemcpyDtoH_v2(dst.as_mut_ptr() as *mut c_void, src.0, dst.len()) };
        if status != 0 {
            bail!("cuMemcpyDtoH_v2 (on_stream) failed: status {status}");
        }
        Ok(())
    }

    fn copy_d2h_pair_on_stream(
        &self,
        first_src: DevicePtr,
        first_dst: &mut [u8],
        second_src: DevicePtr,
        second_dst: &mut [u8],
        stream: u64,
    ) -> Result<()> {
        if super::bounded_copy::enabled() {
            self.ensure_bounded_wait_outside_capture(stream, "bounded D2H pair")?;
            return super::bounded_copy::d2h_pair(
                first_src.0,
                first_dst,
                second_src.0,
                second_dst,
                stream,
            );
        }
        // Both destinations may be ordinary pageable Vec storage. Drain the
        // producer stream once, then use the synchronous driver entry point
        // for each copy. This preserves the hot paths' two-copy/one-sync
        // shape without passing pageable memory to cuMemcpyDtoHAsync_v2.
        super::wfet_sync::park_stream_if_enabled(stream);
        // SAFETY: `stream` is a caller-supplied CUDA stream handle on the active context.
        let sync = unsafe { cuStreamSynchronize(stream) };
        if sync != 0 {
            bail!(
                "cuStreamSynchronize before D2H pair failed: {}",
                cuda_error_text(sync)
            );
        }
        // SAFETY: `first_dst` is uniquely borrowed for `first_dst.len()` writable bytes; caller
        // guarantees `first_src` holds at least `first_dst.len()` valid device bytes.
        let first_status = unsafe {
            cuMemcpyDtoH_v2(
                first_dst.as_mut_ptr() as *mut c_void,
                first_src.0,
                first_dst.len(),
            )
        };
        if first_status != 0 {
            bail!("first cuMemcpyDtoH_v2 in pair failed: status {first_status}");
        }
        // SAFETY: `second_dst` is uniquely borrowed for `second_dst.len()` writable bytes; caller
        // guarantees `second_src` holds at least `second_dst.len()` valid device bytes.
        let second_status = unsafe {
            cuMemcpyDtoH_v2(
                second_dst.as_mut_ptr() as *mut c_void,
                second_src.0,
                second_dst.len(),
            )
        };
        if second_status != 0 {
            bail!("second cuMemcpyDtoH_v2 in pair failed: status {second_status}");
        }
        Ok(())
    }

    fn copy_h2d_group_on_stream(&self, copies: &[HostToDeviceCopy<'_>], stream: u64) -> Result<()> {
        if copies.is_empty() {
            return Ok(());
        }
        if super::bounded_copy::enabled() {
            self.ensure_bounded_wait_outside_capture(stream, "bounded H2D group")?;
            return super::bounded_copy::h2d_group(copies, stream);
        }
        // SAFETY: `stream` is a caller-supplied CUDA stream handle on the active context.
        let sync = unsafe { cuStreamSynchronize(stream) };
        if sync != 0 {
            bail!(
                "cuStreamSynchronize before H2D group failed: {}",
                cuda_error_text(sync)
            );
        }
        for (index, copy) in copies.iter().copied().enumerate() {
            let src = copy.src();
            // SAFETY: `src` is readable for `src.len()` bytes; caller guarantees `copy.dst()`
            // points to at least `src.len()` valid device bytes.
            let status =
                unsafe { cuMemcpyHtoD_v2(copy.dst().0, src.as_ptr() as *const c_void, src.len()) };
            if status != 0 {
                bail!("cuMemcpyHtoD_v2 group member {index} failed: status {status}");
            }
        }
        drain_legacy_h2d_if_selected()
    }

    fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
        if super::bounded_copy::enabled() {
            self.ensure_bounded_wait_outside_capture(self.default_stream, "bounded D2D")?;
            return super::bounded_copy::d2d(src.0, dst.0, bytes, self.default_stream);
        }
        // SAFETY: Caller supplies valid device pointers `src` and `dst` of at least `bytes` bytes
        // on `self.default_stream`.
        let status = unsafe { cuMemcpyDtoDAsync_v2(dst.0, src.0, bytes, self.default_stream) };
        if status != 0 {
            bail!("cuMemcpyDtoDAsync_v2 failed: status {status}");
        }
        // Synchronize to ensure copy completes before kernels on other streams read it.
        // SAFETY: `self.default_stream` is a valid CUDA stream handle on the active context.
        let sync = unsafe { cuStreamSynchronize(self.default_stream) };
        if sync != 0 {
            bail!(
                "cuStreamSynchronize after D2D failed: {}",
                cuda_error_text(sync)
            );
        }
        Ok(())
    }

    fn launch(
        &self,
        func: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        stream: u64,
        params: &mut [*mut c_void],
    ) -> Result<()> {
        let raw_func = RawCudaFunc(func.0 as *mut c_void);
        let cfg = LaunchConfig {
            grid_dim: (grid[0], grid[1], grid[2]),
            block_dim: (block[0], block[1], block[2]),
            shared_mem_bytes: shared_mem,
        };
        let registry = self.registry;
        // SAFETY: `func` was returned by `self.kernel(...)` on `self.registry`; caller supplies
        // kernel parameter pointers matching the PTX function's signature and valid `stream`.
        unsafe {
            registry
                .launch_on_stream(raw_func, cfg, stream, params)
                .map_err(|e| anyhow::anyhow!("Kernel launch failed: {e}"))
        }
    }

    fn launch_pdl(
        &self,
        func: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        stream: u64,
        params: &mut [*mut c_void],
        pdl: bool,
    ) -> Result<()> {
        let raw_func = RawCudaFunc(func.0 as *mut c_void);
        let cfg = LaunchConfig {
            grid_dim: (grid[0], grid[1], grid[2]),
            block_dim: (block[0], block[1], block[2]),
            shared_mem_bytes: shared_mem,
        };
        let registry = self.registry;
        // SAFETY: `func` was returned by `self.kernel(...)` on `self.registry`; caller supplies
        // kernel parameter pointers matching the PTX function's signature and valid `stream`.
        unsafe {
            registry
                .launch_on_stream_pdl(raw_func, cfg, stream, params, pdl)
                .map_err(|e| anyhow::anyhow!("Kernel launch (PDL={pdl}) failed: {e}"))
        }
    }

    fn launch_cooperative(
        &self,
        func: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        stream: u64,
        params: &mut [*mut c_void],
    ) -> Result<()> {
        // SAFETY: `func` is a loaded `CUfunction` from `self.registry`; caller guarantees `params`
        // matches the kernel ABI and `grid`/`block`/`shared_mem` fit cooperative launch limits.
        let status = unsafe {
            super::cuLaunchCooperativeKernel(
                func.0 as *mut c_void,
                grid[0],
                grid[1],
                grid[2],
                block[0],
                block[1],
                block[2],
                shared_mem,
                stream,
                params.as_mut_ptr(),
            )
        };
        if status != 0 {
            bail!(
                "cuLaunchCooperativeKernel failed: {}",
                cuda_error_text(status)
            );
        }
        Ok(())
    }

    fn set_kernel_max_dynamic_shared_memory(&self, func: KernelHandle, bytes: u32) -> Result<()> {
        // CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8.
        let value = i32::try_from(bytes).context("dynamic shared-memory size exceeds i32")?;
        // SAFETY: `func.0` is a loaded `CUfunction` from `self.registry`; attribute 8 takes `i32`.
        let status = unsafe { super::cuFuncSetAttribute(func.0 as *mut c_void, 8, value) };
        if status != 0 {
            bail!(
                "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES) failed: {}",
                cuda_error_text(status)
            );
        }
        Ok(())
    }

    fn synchronize(&self, stream: u64) -> Result<()> {
        if super::bounded_copy::enabled() {
            self.ensure_bounded_wait_outside_capture(stream, "synchronize")?;
            return super::bounded_copy::drain(stream, "synchronize");
        }
        super::wfet_sync::park_stream_if_enabled(stream);
        // SAFETY: `stream` is a valid CUDA stream handle on the active context.
        let status = unsafe { cuStreamSynchronize(stream) };
        if status != 0 {
            bail!("cuStreamSynchronize failed: {}", cuda_error_text(status));
        }
        Ok(())
    }

    fn synchronize_device(&self) -> Result<()> {
        // SAFETY: Synchronizes the calling thread's current CUDA context.
        let status = unsafe { super::cuCtxSynchronize() };
        if status != 0 {
            bail!("cuCtxSynchronize failed: {}", cuda_error_text(status));
        }
        Ok(())
    }

    fn default_stream(&self) -> u64 {
        self.default_stream
    }

    fn sm_count(&self) -> Option<u32> {
        self.device_props
            .filter(|props| props.sm_count > 0)
            .map(|props| props.sm_count)
            .or_else(super::AtlasCudaBackend::cached_sm_count)
    }

    fn is_unified_memory(&self) -> bool {
        self.device_props
            .map(|props| props.is_integrated)
            .or_else(super::current_device_is_integrated)
            .unwrap_or(false)
    }

    fn compute_capability(&self) -> Option<(u32, u32)> {
        self.device_props
            .map(|props| (props.compute_major, props.compute_minor))
            .or_else(atlas_core::device::active_compute_capability)
    }

    #[track_caller]
    fn kernel(&self, module: &str, func_name: &str) -> Result<KernelHandle> {
        // Ephemeral OnceLock — no cross-call caching, but kernel() is only
        // called at model init time. Layers store the returned KernelHandle.
        let cache: OnceLock<RawCudaFunc> = OnceLock::new();
        let registry = self.registry;
        let found = registry.raw_function_cached(&cache, module, func_name);
        // Record BEFORE the `?`: a failed lookup is the only kind worth
        // auditing, and `try_kernel` swallows the error.
        crate::kernel_audit::record(
            module,
            func_name,
            found.is_ok(),
            std::panic::Location::caller(),
        );
        let raw = found.map_err(|e| anyhow::anyhow!("Kernel lookup {module}::{func_name}: {e}"))?;
        Ok(KernelHandle(raw.0 as u64))
    }

    unsafe fn copy_h2d_pinned_async(
        &self,
        src: PinnedHostSlice<'_>,
        dst: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let src = src.as_bytes();
        // SAFETY: `src` has backend-issued page-locked provenance; caller upholds the trait's
        // `# Safety` lifetime contract through stream completion / graph replay.
        let status = unsafe {
            cuMemcpyHtoDAsync_v2(dst.0, src.as_ptr() as *const c_void, src.len(), stream)
        };
        if status != 0 {
            bail!("cuMemcpyHtoDAsync_v2 failed: status {status}");
        }
        Ok(())
    }

    fn bounded_host_copy(&self) -> bool {
        super::bounded_copy::enabled()
    }

    fn retain_h2d_destination(&self, base: DevicePtr, offset: usize, bytes: usize) -> Result<()> {
        let end = offset
            .checked_add(bytes)
            .ok_or_else(|| anyhow::anyhow!("H2D destination range overflow"))?;
        base.0
            .checked_add(u64::try_from(end)?)
            .ok_or_else(|| anyhow::anyhow!("H2D destination address overflow"))?;
        let mut live = self
            .live_allocations
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        live.get_mut(&base.0)
            .ok_or_else(|| anyhow::anyhow!("H2D destination is not a live owning base"))?
            .retain(offset, bytes)
    }

    fn release_h2d_destination(&self, base: DevicePtr) -> Result<()> {
        let mut live = self
            .live_allocations
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        live.get_mut(&base.0)
            .ok_or_else(|| anyhow::anyhow!("retained H2D destination is not a live owning base"))?
            .release()
    }

    fn stream_is_capturing(&self, stream: u64) -> bool {
        let mut status: u32 = 0;
        // CU_STREAM_CAPTURE_STATUS_NONE = 0; treat query failure as
        // capturing (conservative: the caller skips its eager work).
        // SAFETY: `&mut status` is a valid writable out-pointer; `stream` is a CUDA stream handle.
        let rc = unsafe { super::cuStreamIsCapturing(stream, &mut status) };
        rc != 0 || status != 0
    }

    fn copy_h2d_async(&self, src: &[u8], dst: DevicePtr, stream: u64) -> Result<()> {
        // SAFETY: `src` is readable for `src.len()` bytes; CUDA stages pageable host sources
        // before returning; caller guarantees `dst` has at least `src.len()` valid device bytes.
        let status = unsafe {
            super::cuMemcpyHtoDAsync_v2(dst.0, src.as_ptr() as *const c_void, src.len(), stream)
        };
        if status != 0 {
            bail!("cuMemcpyHtoDAsync_v2 failed: status {status}");
        }
        Ok(())
    }

    fn copy_d2d_async(
        &self,
        src: DevicePtr,
        dst: DevicePtr,
        bytes: usize,
        stream: u64,
    ) -> Result<()> {
        // SAFETY: Caller supplies valid device pointers `src` and `dst` of at least `bytes` bytes
        // on `stream`.
        let status = unsafe { cuMemcpyDtoDAsync_v2(dst.0, src.0, bytes, stream) };
        if status != 0 {
            bail!("cuMemcpyDtoDAsync_v2 failed: status {status}");
        }
        Ok(())
    }

    fn copy_d2d_2d_async(
        &self,
        src: DevicePtr,
        src_pitch: usize,
        dst: DevicePtr,
        dst_pitch: usize,
        width_bytes: usize,
        height: usize,
        stream: u64,
    ) -> Result<()> {
        // One pitched copy (cudaMemcpyDeviceToDevice = 3) on the caller's stream, replacing a
        // per-row copy_d2d_async loop. cudart is linked (cutlass/flashinfer use the runtime
        // API); a CUstream handle is a valid cudaStream_t. Ported from dsv41/integration.
        // SAFETY: Signature matches `cudaMemcpy2DAsync` in `cuda_runtime_api.h` in linked `libcudart`.
        unsafe extern "C" {
            fn cudaMemcpy2DAsync(
                dst: *mut c_void,
                dpitch: usize,
                src: *const c_void,
                spitch: usize,
                width: usize,
                height: usize,
                kind: i32,
                stream: u64,
            ) -> i32;
        }
        // SAFETY: Caller supplies valid pitched device regions `src`/`dst` and stream handle;
        // `kind = 3` is `cudaMemcpyDeviceToDevice` and `CUstream` is ABI-compatible with `cudaStream_t`.
        let status = unsafe {
            cudaMemcpy2DAsync(
                dst.0 as *mut c_void,
                dst_pitch,
                src.0 as *const c_void,
                src_pitch,
                width_bytes,
                height,
                3,
                stream,
            )
        };
        if status != 0 {
            bail!("cudaMemcpy2DAsync failed: status {status}");
        }
        Ok(())
    }

    fn begin_capture(&self, stream: u64) -> Result<()> {
        // CU_STREAM_CAPTURE_MODE_RELAXED = 2
        // Relaxed mode allows NCCL's internal streams to operate during
        // graph capture (required for EP all-reduce in CUDA graphs).
        // SAFETY: `stream` is a valid non-legacy CUDA stream on the active context.
        let status = unsafe { cuStreamBeginCapture(stream, 2) };
        if status != 0 {
            bail!("cuStreamBeginCapture failed: status {status}");
        }
        Ok(())
    }

    fn end_capture(&self, stream: u64) -> Result<GraphHandle> {
        let mut graph: u64 = 0;
        // SAFETY: `stream` is currently capturing; `&mut graph` is a valid out-pointer.
        let status = unsafe { cuStreamEndCapture(stream, &mut graph) };
        if status != 0 {
            bail!("cuStreamEndCapture failed: status {status}");
        }
        // Instantiate the graph into an executable
        let mut graph_exec: u64 = 0;
        // SAFETY: `graph` is a live `CUgraph` returned by `cuStreamEndCapture`; `&mut graph_exec` is valid.
        let status = unsafe { cuGraphInstantiateWithFlags(&mut graph_exec, graph, 0) };
        if status != 0 {
            // SAFETY: `graph` is uniquely owned and has not been destroyed yet.
            unsafe { cuGraphDestroy(graph) };
            bail!("cuGraphInstantiateWithFlags failed: status {status}");
        }
        // The graph template is no longer needed after instantiation
        // SAFETY: `graph` is uniquely owned and its executable `graph_exec` is independent.
        unsafe { cuGraphDestroy(graph) };
        Ok(GraphHandle(graph_exec))
    }

    fn launch_graph(&self, graph: GraphHandle, stream: u64) -> Result<()> {
        // SAFETY: `graph.0` is an instantiated `CUgraphExec` and `stream` is a valid CUDA stream.
        let status = unsafe { cuGraphLaunch(graph.0, stream) };
        if status != 0 {
            bail!("cuGraphLaunch failed: status {status}");
        }
        Ok(())
    }

    fn destroy_graph(&self, graph: GraphHandle) -> Result<()> {
        if graph.0 != 0 {
            // SAFETY: Caller guarantees `graph.0` is a live `CUgraphExec` with no in-flight replays.
            let status = unsafe { cuGraphExecDestroy(graph.0) };
            if status != 0 {
                bail!("cuGraphExecDestroy failed: status {status}");
            }
            // Return unused CUDA graph memory pool pages to the OS on UMA
            // hosts (GB10) so repeated per-sequence graph recaptures do not
            // accumulate unfreed pool residency across long runs.
            let mut dev: i32 = 0;
            // SAFETY: `&mut dev` is a valid out-pointer on the active CUDA context.
            if unsafe { cuCtxGetDevice(&mut dev) } == 0 {
                // SAFETY: `dev` is a valid `CUdevice` returned by `cuCtxGetDevice`.
                let _ = unsafe { cuDeviceGraphMemTrim(dev) };
            }
        }
        Ok(())
    }

    fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()> {
        let bounded = super::bounded_copy::enabled();
        if bounded {
            self.ensure_bounded_wait_outside_capture(self.default_stream, "memset")?;
        }
        // SAFETY: Caller supplies a valid device allocation `ptr` of at least `bytes` bytes
        // on `self.default_stream`.
        let status = unsafe { cuMemsetD8Async(ptr.0, value, bytes, self.default_stream) };
        if status != 0 {
            bail!("cuMemsetD8Async failed: status {status}");
        }
        if bounded {
            return super::bounded_copy::drain(self.default_stream, "memset");
        }
        // SAFETY: `self.default_stream` is a valid CUDA stream handle on the active context.
        let sync = unsafe { cuStreamSynchronize(self.default_stream) };
        if sync != 0 {
            bail!("cuStreamSynchronize after memset failed: status {sync}");
        }
        Ok(())
    }

    fn memset_async(&self, ptr: DevicePtr, value: u8, bytes: usize, stream: u64) -> Result<()> {
        // SAFETY: Caller supplies a valid device allocation `ptr` of at least `bytes` bytes on `stream`.
        let status = unsafe { cuMemsetD8Async(ptr.0, value, bytes, stream) };
        if status != 0 {
            bail!("cuMemsetD8Async failed: status {status}");
        }
        Ok(())
    }

    fn memset_u32_async(
        &self,
        ptr: DevicePtr,
        value: u32,
        count: usize,
        stream: u64,
    ) -> Result<()> {
        // SAFETY: Caller supplies a 4-byte-aligned device allocation `ptr` of at least `count * 4` bytes.
        let status = unsafe { cuMemsetD32Async(ptr.0, value, count, stream) };
        if status != 0 {
            bail!("cuMemsetD32Async failed: status {status}");
        }
        Ok(())
    }

    fn total_memory(&self) -> Result<usize> {
        let mut free: usize = 0;
        let mut total: usize = 0;
        // SAFETY: `&mut free` and `&mut total` are valid out-pointers on the active CUDA context.
        let status = unsafe { cuMemGetInfo_v2(&mut free, &mut total) };
        if status != 0 {
            bail!("cuMemGetInfo_v2 failed: status {status}");
        }
        Ok(total)
    }

    fn free_memory(&self) -> Result<usize> {
        let mut free: usize = 0;
        let mut total: usize = 0;
        // SAFETY: `&mut free` and `&mut total` are valid out-pointers on the active CUDA context.
        let status = unsafe { cuMemGetInfo_v2(&mut free, &mut total) };
        if status != 0 {
            bail!("cuMemGetInfo_v2 failed: status {status}");
        }
        // Only merge host `/proc/meminfo` `MemAvailable` on integrated/UMA
        // devices (GB10), where `cuMemGetInfo_v2` excludes reclaimable page
        // cache. On discrete GPUs, return exact VRAM free bytes.
        let is_integrated = self.is_unified_memory();
        let host_mem_available = is_integrated
            .then(super::system_available_memory_bytes)
            .flatten();
        let (effective_free, _) =
            super::effective_memory_bytes(free, total, is_integrated, host_mem_available);
        Ok(effective_free)
    }

    fn create_stream(&self) -> Result<u64> {
        let mut stream: u64 = 0;
        // CU_STREAM_NON_BLOCKING = 1 (does not synchronize with stream 0)
        // SAFETY: `&mut stream` is a valid out-pointer on the active CUDA context.
        let status = unsafe { cuStreamCreate(&mut stream, 1) };
        if status != 0 {
            bail!("cuStreamCreate failed: status {status}");
        }
        Ok(stream)
    }

    fn bind_to_thread(&self) -> Result<()> {
        // SAFETY: `self.cuda_ctx` is the non-null primary CUDA context owned by `self.registry`.
        let status = unsafe { cuCtxSetCurrent(self.cuda_ctx) };
        if status != 0 {
            bail!("cuCtxSetCurrent failed: status {status}");
        }
        Ok(())
    }

    fn create_event(&self) -> Result<u64> {
        let mut event: u64 = 0;
        // CU_EVENT_DISABLE_TIMING = 0x02 (skip timing overhead)
        // SAFETY: `&mut event` is a valid out-pointer on the active CUDA context.
        let status = unsafe { cuEventCreate(&mut event, 0x02) };
        if status != 0 {
            bail!("cuEventCreate failed: status {status}");
        }
        Ok(event)
    }

    fn record_event(&self, event: u64, stream: u64) -> Result<()> {
        // SAFETY: `event` and `stream` are valid handles on the active CUDA context.
        let status = unsafe { cuEventRecord(event, stream) };
        if status != 0 {
            bail!("cuEventRecord failed: status {status}");
        }
        Ok(())
    }

    fn stream_wait_event(&self, stream: u64, event: u64) -> Result<()> {
        // SAFETY: `stream` and `event` are valid handles on the active CUDA context.
        let status = unsafe { cuStreamWaitEvent(stream, event, 0) };
        if status != 0 {
            bail!("cuStreamWaitEvent failed: status {status}");
        }
        Ok(())
    }

    fn destroy_event(&self, event: u64) -> Result<()> {
        if event != 0 {
            // SAFETY: Caller guarantees `event` is a live event handle destroyed at most once.
            let status = unsafe { cuEventDestroy_v2(event) };
            if status != 0 {
                bail!("cuEventDestroy_v2 failed: status {status}");
            }
        }
        Ok(())
    }

    fn poll_event(&self, event: u64) -> Result<bool> {
        const CUDA_ERROR_NOT_READY: i32 = 600;
        // SAFETY: `event` is a live CUDA event handle on the active context.
        let status = unsafe { super::cuEventQuery(event) };
        match status {
            0 => Ok(true),
            CUDA_ERROR_NOT_READY => Ok(false),
            other => bail!("cuEventQuery failed: status {other}"),
        }
    }

    fn alloc_host_pinned(&self, bytes: usize) -> Result<PinnedHostBuffer> {
        if bytes == 0 {
            bail!("pinned host allocation must be non-empty");
        }
        if pinned_host_pool_eligible(bytes)
            && let Ok(mut pool) = PINNED_HOST_POOL.lock()
            && let Some(idx) = select_pinned_host_pool_slot(&pool, bytes)
        {
            let (raw_ptr, capacity) = pool.swap_remove(idx);
            if let Some(ptr) = NonNull::new(raw_ptr as *mut u8) {
                return Ok(PinnedHostBuffer::from_storage(Box::new(
                    CudaPinnedHostStorage {
                        ptr,
                        bytes,
                        capacity,
                    },
                )));
            }
        }
        let capacity = pinned_host_alloc_capacity(bytes).ok_or_else(|| {
            anyhow::anyhow!("pinned host allocation size overflow: {bytes} bytes")
        })?;
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: Active CUDA context is bound on this thread; `&mut ptr` is a valid out-pointer
        // and `capacity > 0`.
        let status = unsafe { cuMemAllocHost_v2(&mut ptr, capacity) };
        if status != 0 {
            bail!("cuMemAllocHost_v2 failed: status {status}, requested {capacity} bytes");
        }
        let ptr = NonNull::new(ptr.cast::<u8>())
            .ok_or_else(|| anyhow::anyhow!("cuMemAllocHost_v2 returned null"))?;
        // Zero freshly allocated pinned host memory so `PinnedHostBuffer::as_mut_slice` and
        // `pinned_slice_range` never expose uninitialized bytes (amortized by `PINNED_HOST_POOL`).
        // SAFETY: `ptr` points to `capacity` freshly allocated, uniquely owned writable bytes.
        unsafe { std::ptr::write_bytes(ptr.as_ptr(), 0, capacity) };
        crate::alloc_receipt::GLOBAL.note_pinned_alloc(capacity);
        Ok(PinnedHostBuffer::from_storage(Box::new(
            CudaPinnedHostStorage {
                ptr,
                bytes,
                capacity,
            },
        )))
    }

    fn copy_h2d_managed(&self, src: &[u8], dst: DevicePtr) -> Result<()> {
        if src.is_empty() {
            return Ok(());
        }
        if dst.is_null() {
            bail!("copy_h2d_managed: null destination pointer");
        }
        // `CU_POINTER_ATTRIBUTE_IS_MANAGED` = 8 (`cuda.h`).
        const CU_POINTER_ATTRIBUTE_IS_MANAGED: u32 = 8;
        let mut is_managed: u32 = 0;
        // SAFETY: `&mut is_managed` is a valid 4-byte out-pointer; `cuPointerGetAttribute`
        // queries pointer metadata without dereferencing `dst.0`.
        let attr_rc = unsafe {
            cuPointerGetAttribute(
                (&mut is_managed as *mut u32).cast(),
                CU_POINTER_ATTRIBUTE_IS_MANAGED,
                dst.0,
            )
        };
        if attr_rc != 0 || is_managed == 0 {
            return self.copy_h2d(src, dst);
        }
        let mut base: u64 = 0;
        let mut alloc_size: usize = 0;
        // SAFETY: `&mut base` and `&mut alloc_size` are valid out-pointers for querying the
        // CUDA allocation range containing `dst.0`.
        let range_rc = unsafe { cuMemGetAddressRange_v2(&mut base, &mut alloc_size, dst.0) };
        if range_rc != 0 {
            return self.copy_h2d(src, dst);
        }
        let offset = usize::try_from(dst.0.saturating_sub(base))
            .map_err(|_| anyhow::anyhow!("copy_h2d_managed: destination offset overflow"))?;
        let end = offset
            .checked_add(src.len())
            .ok_or_else(|| anyhow::anyhow!("copy_h2d_managed: destination length overflow"))?;
        if dst.0 < base || end > alloc_size {
            bail!(
                "copy_h2d_managed: write of {} bytes at {dst} (offset {offset}) exceeds managed allocation size {alloc_size}",
                src.len()
            );
        }
        let src_start = src.as_ptr() as usize;
        let src_end = src_start.saturating_add(src.len());
        let dst_start = dst.0 as usize;
        let dst_end = dst_start.saturating_add(src.len());
        if src_start < dst_end && dst_start < src_end {
            bail!("copy_h2d_managed: source slice overlaps managed destination range");
        }
        // SAFETY: `cuPointerGetAttribute` confirmed `dst.0` is CUDA managed (host-accessible)
        // memory, `cuMemGetAddressRange_v2` confirmed `dst..dst + src.len()` lies within the
        // allocation, `src` is a valid readable slice, and the ranges are verified non-overlapping.
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst.0 as *mut u8, src.len()) };
        Ok(())
    }
}

/// `ATLAS_H2D_LEGACY_DRAIN` (fn-mtp; default ON since fnd4, `=0` disables for
/// A/B only): after the synchronous
/// pageable `cuMemcpyHtoD_v2` in `copy_h2d` / `copy_h2d_group_on_stream`,
/// synchronize the legacy NULL stream before returning.
///
/// CUDA documents that a synchronous host-to-device copy from pageable memory
/// may return once the bytes are staged, before the DMA to the destination has
/// completed. Atlas's streams are non-blocking (cudarc `NonBlocking`), so a
/// kernel launched right after the call on the default stream is not ordered
/// after that DMA and can read the previous bytes (suspected cause of the rare
/// spec-vs-plain row-0 divergence after partial accepts, fnm1f 2026-10-08).
/// Draining the legacy stream makes "the bytes are on the device when the call
/// returns" true for consumers on every stream. fnd4 (2026-10-08) caught it in
/// the wild: the second of two identical 64-token requests prefilled a different
/// first token (13 / 279 vs 22149) and decoded garbage in all three arms; the
/// prompt token ids reach the device through `copy_h2d_group_on_stream`.
pub(super) fn h2d_legacy_drain_selected() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| {
        let on = std::env::var("ATLAS_H2D_LEGACY_DRAIN").ok().as_deref() != Some("0");
        if on {
            tracing::info!(
                "ENGAGED ATLAS_H2D_LEGACY_DRAIN: pageable H2D copies drain the legacy stream"
            );
        }
        on
    })
}

fn drain_legacy_h2d_if_selected() -> Result<()> {
    if !h2d_legacy_drain_selected() {
        return Ok(());
    }
    // Stream handle 0 is CU_STREAM_LEGACY for the driver API: the stream the
    // synchronous copy above was issued on.
    // SAFETY: Stream 0 is the CUDA legacy default stream on the active context.
    let status = unsafe { cuStreamSynchronize(0) };
    if status != 0 {
        bail!(
            "cuStreamSynchronize(legacy) after pageable H2D failed: {}",
            cuda_error_text(status)
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_host_capacity_is_page_aligned_without_64mib_floor() {
        assert_eq!(pinned_host_alloc_capacity(1), Some(4096));
        assert_eq!(pinned_host_alloc_capacity(512), Some(4096));
        assert_eq!(pinned_host_alloc_capacity(2048), Some(4096));
        assert_eq!(pinned_host_alloc_capacity(4096), Some(4096));
        assert_eq!(pinned_host_alloc_capacity(4097), Some(8192));
        assert_eq!(
            pinned_host_alloc_capacity(PINNED_HOST_POOL_MIN_BYTES),
            Some(PINNED_HOST_POOL_MIN_BYTES)
        );
        assert_eq!(pinned_host_alloc_capacity(usize::MAX), None);
    }

    #[test]
    fn pinned_host_pool_bounds_reject_small_and_oversized_matches() {
        const MIB: usize = 1024 * 1024;
        assert!(!pinned_host_pool_eligible(512));
        assert!(!pinned_host_pool_eligible(4 * MIB - 1));
        assert!(pinned_host_pool_eligible(4 * MIB));

        // Tiny metadata requests never match pooled bounce buffers.
        assert!(!pinned_host_pool_matches(512, 64 * MIB));
        assert!(!pinned_host_pool_matches(2048, 4 * MIB));
        assert!(!pinned_host_pool_matches(4 * MIB - 1, 4 * MIB));

        // Eligible requests match only within [requested, 2 * requested].
        assert!(!pinned_host_pool_matches(4 * MIB, 4 * MIB - 4096));
        assert!(pinned_host_pool_matches(4 * MIB, 4 * MIB));
        assert!(pinned_host_pool_matches(4 * MIB, 8 * MIB));
        assert!(!pinned_host_pool_matches(4 * MIB, 8 * MIB + 4096));
        assert!(!pinned_host_pool_matches(4 * MIB, 64 * MIB));

        // Tightest-fitting slot wins when multiple slots satisfy the 2x bound.
        let pool = [(1, 64 * MIB), (2, 8 * MIB), (3, 4 * MIB)];
        assert_eq!(select_pinned_host_pool_slot(&pool, 512), None);
        assert_eq!(select_pinned_host_pool_slot(&pool, 4 * MIB), Some(2));
        assert_eq!(select_pinned_host_pool_slot(&pool, 5 * MIB), Some(1));
        assert_eq!(select_pinned_host_pool_slot(&pool, 10 * MIB), None);
        assert_eq!(select_pinned_host_pool_slot(&pool, 32 * MIB), Some(0));
        assert_eq!(select_pinned_host_pool_slot(&pool, 32 * MIB - 1), None);
    }
}
