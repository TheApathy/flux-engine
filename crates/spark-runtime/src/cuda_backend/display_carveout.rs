// SPDX-License-Identifier: AGPL-3.0-only

//! The GB10 display carveout as device memory (see `crate::gpu::carveout`).
//!
//! `spark display-carveout` passes an RM object fd in
//! [`carveout::FD_ENV`]. The first [`alloc`] imports it with the CUDA VMM
//! API, maps it once for the life of the process, and zeroes it (it still
//! holds whatever the previous borrower wrote). That happens at KV
//! allocation, after the KV budget is measured, so the mapping never counts
//! as this process's footprint in that measurement.
//!
//! The launcher allocates it GPU-cacheable, so it behaves like `cuMemAlloc`
//! memory: on GB10 a 4 MiB region re-read runs at 1,759 against 1,762 GB/s
//! and a dependent load takes 156 against 156 ns. (The earlier memory-list
//! export was L2-uncached: 266 GB/s and 408 ns.)

use std::sync::OnceLock;

use anyhow::Result;
use parking_lot::Mutex;

use crate::gpu::DevicePtr;
use crate::gpu::carveout::{self, CarveoutArena};

enum CarveoutState {
    Unmapped,
    Mapped(CarveoutArena),
    Failed(String),
}

static ARENA: Mutex<CarveoutState> = Mutex::new(CarveoutState::Unmapped);

/// Partial VMM mappings unwind in reverse acquisition order on failure.
struct Lease {
    handle: Option<u64>,
    address: Option<u64>,
    mapped: bool,
    size: usize,
    unmap: unsafe extern "C" fn(u64, usize) -> i32,
    release: unsafe extern "C" fn(u64) -> i32,
    free_address: unsafe extern "C" fn(u64, usize) -> i32,
}

impl Drop for Lease {
    fn drop(&mut self) {
        if let Some(address) = self.address {
            if self.mapped {
                // SAFETY: `self.address` was mapped for `self.size` bytes via `cuMemMap`
                // and has not yet been unmapped.
                unsafe { (self.unmap)(address, self.size) };
            }
            // SAFETY: `self.address` was reserved for `self.size` bytes via `cuMemAddressReserve`
            // and any mapping on it has been removed above.
            unsafe { (self.free_address)(address, self.size) };
        }
        if let Some(handle) = self.handle {
            // SAFETY: `self.handle` was imported via `cuMemImportFromShareableHandle` and
            // has not yet been released via `cuMemRelease`.
            unsafe { (self.release)(handle) };
        }
    }
}

/// The fd and size from the environment, checked once: `None` without a
/// carveout, or when the fd is not open in this process (a child that only
/// inherited the environment).
#[cfg(target_os = "linux")]
fn exported() -> Option<(i32, usize)> {
    static EXPORTED: OnceLock<Option<(i32, usize)>> = OnceLock::new();
    *EXPORTED.get_or_init(|| match carveout::from_env() {
        // SAFETY: `F_GETFD` takes no variadic argument and queries kernel fd flags
        // without mutating descriptor or process state.
        Ok(Some((fd, size))) if unsafe { libc::fcntl(fd, libc::F_GETFD) } >= 0 => Some((fd, size)),
        Ok(Some((fd, _))) => {
            tracing::warn!(
                "{} names fd {fd}, which is not open; no carveout",
                carveout::FD_ENV
            );
            None
        }
        Ok(None) => None,
        Err(e) => {
            tracing::error!("display carveout environment: {e:#}; no carveout");
            None
        }
    })
}

#[cfg(not(target_os = "linux"))]
fn exported() -> Option<(i32, usize)> {
    let _ = OnceLock::<()>::new;
    None
}

pub(super) fn capacity() -> usize {
    exported().map_or(0, |(_, size)| size)
}

pub(super) fn alloc(bytes: usize) -> Result<DevicePtr> {
    let mut state = ARENA.lock();
    if matches!(*state, CarveoutState::Unmapped) {
        match map() {
            Ok(arena) => *state = CarveoutState::Mapped(arena),
            Err(err) => {
                let msg = format!("{err:#}");
                *state = CarveoutState::Failed(msg);
                return Err(err);
            }
        }
    }
    match &mut *state {
        CarveoutState::Mapped(arena) => Ok(DevicePtr(arena.alloc(bytes)?)),
        CarveoutState::Failed(msg) => {
            anyhow::bail!("display carveout mapping previously failed: {msg}")
        }
        CarveoutState::Unmapped => unreachable!("initialized above"),
    }
}

/// Releases `ptr` if it is a carveout piece.
pub(super) fn free(ptr: DevicePtr) -> bool {
    match &mut *ARENA.lock() {
        CarveoutState::Mapped(arena) => arena.free(ptr.0),
        CarveoutState::Unmapped | CarveoutState::Failed(_) => false,
    }
}

/// Releases all live carveout pieces at a quiescent model-teardown boundary.
pub(super) fn free_all() {
    if let CarveoutState::Mapped(arena) = &mut *ARENA.lock() {
        arena.clear();
    }
}

#[cfg(not(target_os = "linux"))]
fn map() -> Result<CarveoutArena> {
    let _ = std::mem::size_of::<Lease>();
    anyhow::bail!("the display carveout is a Linux GB10 feature")
}

#[cfg(target_os = "linux")]
fn map() -> Result<CarveoutArena> {
    use anyhow::{Context, bail};
    use std::ffi::c_void;

    const HANDLE_TYPE_POSIX_FD: u32 = 1;
    const LOCATION_DEVICE: u32 = 1;
    const ACCESS_READ_WRITE: u32 = 3;
    /// `CUmemAccessDesc`.
    #[repr(C)]
    struct AccessDesc {
        location_type: u32,
        location_id: i32,
        flags: u32,
    }
    // SAFETY: Symbols and C ABI signatures match the CUDA Virtual Memory Management API in `cuda.h`.
    unsafe extern "C" {
        fn cuMemImportFromShareableHandle(handle: *mut u64, os: *mut c_void, kind: u32) -> i32;
        fn cuMemAddressReserve(ptr: *mut u64, size: usize, align: usize, addr: u64, f: u64) -> i32;
        fn cuMemAddressFree(ptr: u64, size: usize) -> i32;
        fn cuMemMap(ptr: u64, size: usize, offset: usize, handle: u64, flags: u64) -> i32;
        fn cuMemUnmap(ptr: u64, size: usize) -> i32;
        fn cuMemRelease(handle: u64) -> i32;
        fn cuMemSetAccess(ptr: u64, size: usize, desc: *const AccessDesc, count: usize) -> i32;
    }
    let check = |status: i32, what: &str| -> Result<()> {
        if status != 0 {
            bail!(
                "display carveout {what}: {}",
                atlas_core::registry::cuda_error_text(status)
            );
        }
        Ok(())
    };

    let (fd, size) = exported().context("no display carveout was exported to this process")?;
    /// Closes the exported carveout fd exactly once on every exit path from `map()`,
    /// which runs at most once under `ARENA` (transitioning `Unmapped -> Mapped | Failed`).
    struct FdCloseGuard(i32);
    impl Drop for FdCloseGuard {
        fn drop(&mut self) {
            // SAFETY: `self.0` was verified open by `exported()`, `map()` is called at most
            // once while holding `ARENA`, and `FdCloseGuard` closes it exactly once.
            unsafe { libc::close(self.0) };
        }
    }
    let _fd_guard = FdCloseGuard(fd);
    let mut device = 0i32;
    // SAFETY: `device` is a valid stack `i32` and the backend bound a CUDA context on this thread.
    check(unsafe { super::cuCtxGetDevice(&mut device) }, "device")?;
    let mut handle = 0u64;
    // The fd is a plain integer handle for this call, not a pointer.
    let os_handle = fd as isize as *mut c_void;
    check(
        // SAFETY: `handle` points to a valid `u64`; for `CU_MEM_HANDLE_TYPE_POSIX_FD` the CUDA
        // driver API passes the integer fd cast to `*mut c_void` without dereferencing it.
        unsafe { cuMemImportFromShareableHandle(&mut handle, os_handle, HANDLE_TYPE_POSIX_FD) },
        "import",
    )?;

    let mut lease = Lease {
        handle: Some(handle),
        address: None,
        mapped: false,
        size,
        unmap: cuMemUnmap,
        release: cuMemRelease,
        free_address: cuMemAddressFree,
    };

    let mut va = 0u64;
    check(
        // SAFETY: `va` is a valid stack `u64` and `size` is the exported 2 MiB-aligned region size.
        unsafe { cuMemAddressReserve(&mut va, size, 2 << 20, 0, 0) },
        "address reserve",
    )?;
    lease.address = Some(va);

    // SAFETY: `va..va + size` was reserved above and `handle` is a live imported allocation of `size` bytes.
    check(unsafe { cuMemMap(va, size, 0, handle, 0) }, "map")?;
    lease.mapped = true;

    // The mapping keeps the memory alive; drop the handle's reference.
    // SAFETY: `handle` is valid; once mapped, the VA mapping retains the physical allocation reference.
    check(unsafe { cuMemRelease(handle) }, "release handle")?;
    lease.handle = None;

    let access = AccessDesc {
        location_type: LOCATION_DEVICE,
        location_id: device,
        flags: ACCESS_READ_WRITE,
    };
    check(
        // SAFETY: `va..va + size` is a mapped VMM range and `&access` points to 1 valid `CUmemAccessDesc`.
        unsafe { cuMemSetAccess(va, size, &access, 1) },
        "set access",
    )?;
    // SAFETY: `va..va + size` has read/write access on `device`; zeroing on legacy stream 0 and
    // synchronizing ensures previous contents cannot be observed before pieces are handed out.
    check(unsafe { super::cuMemsetD8Async(va, 0, size, 0) }, "zero")?;
    // SAFETY: stream 0 is the legacy default stream of the active CUDA context.
    check(unsafe { super::cuStreamSynchronize(0) }, "zero sync")?;

    std::mem::forget(lease);
    tracing::info!(
        "Display carveout: {} MiB mapped at 0x{va:x} for KV pools",
        size >> 20
    );
    Ok(CarveoutArena::new(va, size))
}

#[cfg(test)]
mod tests {
    use super::Lease;
    use std::sync::Mutex;

    static CALLS: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());

    unsafe extern "C" fn unmap(_: u64, _: usize) -> i32 {
        CALLS.lock().unwrap().push("unmap");
        0
    }
    unsafe extern "C" fn release(_: u64) -> i32 {
        CALLS.lock().unwrap().push("release");
        0
    }
    unsafe extern "C" fn free_address(_: u64, _: usize) -> i32 {
        CALLS.lock().unwrap().push("free_address");
        0
    }

    #[test]
    fn every_partial_mapping_releases_only_acquired_resources() {
        for (handle, address, mapped, expected) in [
            (Some(1), None, false, vec!["release"]),
            (Some(1), Some(2), false, vec!["free_address", "release"]),
            (
                Some(1),
                Some(2),
                true,
                vec!["unmap", "free_address", "release"],
            ),
            (None, Some(2), true, vec!["unmap", "free_address"]),
            (None, None, false, vec![]),
        ] {
            CALLS.lock().unwrap().clear();
            drop(Lease {
                handle,
                address,
                mapped,
                size: 4096,
                unmap,
                release,
                free_address,
            });
            assert_eq!(*CALLS.lock().unwrap(), expected);
        }
    }
}
