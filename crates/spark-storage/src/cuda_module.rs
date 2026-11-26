// SPDX-License-Identifier: AGPL-3.0-only
//
// CUDA module loading + event primitives. Split out of `cuda_min.rs` to
// keep that file focused on the core context/buffer/copy primitives.

use anyhow::{Context, Result, bail};
use std::ffi::c_void;

// SAFETY: External symbol signatures match the 64-bit CUDA Driver API C ABI in `cuda.h`.
unsafe extern "C" {
    fn cuModuleLoadData(module: *mut u64, image: *const c_void) -> i32;
    fn cuModuleUnload(module: u64) -> i32;
    fn cuModuleGetFunction(func: *mut u64, module: u64, name: *const std::ffi::c_char) -> i32;
    fn cuLaunchKernel(
        func: u64,
        grid_x: u32,
        grid_y: u32,
        grid_z: u32,
        block_x: u32,
        block_y: u32,
        block_z: u32,
        shared_bytes: u32,
        stream: u64,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> i32;
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

/// Loaded CUmodule with cached function lookups. Drop unloads.
pub struct CudaModule {
    handle: u64,
}

impl CudaModule {
    pub fn from_ptx(ptx: &str) -> Result<Self> {
        let cstr = std::ffi::CString::new(ptx).context("PTX contains NUL")?;
        let mut h = 0u64;
        // SAFETY: `&mut h` is a valid aligned out-pointer and `cstr` is a NUL-terminated PTX image.
        let s = unsafe { cuModuleLoadData(&mut h, cstr.as_ptr() as *const c_void) };
        if s != 0 {
            bail!("cuModuleLoadData failed: {s}");
        }
        Ok(Self { handle: h })
    }

    pub fn function(&self, name: &str) -> Result<u64> {
        let cstr = std::ffi::CString::new(name).context("function name has NUL")?;
        let mut f = 0u64;
        // SAFETY: `self.handle` is a live loaded `CUmodule`, `cstr` is NUL-terminated,
        // and `&mut f` is a valid aligned `CUfunction` out-pointer.
        let s = unsafe { cuModuleGetFunction(&mut f, self.handle, cstr.as_ptr()) };
        if s != 0 {
            bail!("cuModuleGetFunction({name}) failed: {s}");
        }
        Ok(f)
    }
}

impl Drop for CudaModule {
    fn drop(&mut self) {
        // SAFETY: `self.handle` was created by `cuModuleLoadData` and is unloaded once on Drop.
        unsafe {
            let _ = cuModuleUnload(self.handle);
        }
    }
}

/// CUDA event for cross-stream / host-side completion tracking.
pub struct CudaEvent {
    pub handle: u64,
}

impl CudaEvent {
    pub fn new() -> Result<Self> {
        let mut h = 0u64;
        // CU_EVENT_DISABLE_TIMING (0x2): we never query elapsed time.
        const CU_EVENT_DISABLE_TIMING: u32 = 0x2;
        // SAFETY: `&mut h` is a valid aligned `CUevent` out-pointer and `CU_EVENT_DISABLE_TIMING` is valid.
        let s = unsafe { cuEventCreate(&mut h, CU_EVENT_DISABLE_TIMING) };
        if s != 0 {
            bail!("cuEventCreate failed: {s}");
        }
        Ok(Self { handle: h })
    }
    pub fn record(&self, stream: u64) -> Result<()> {
        // SAFETY: `self.handle` is a live owned `CUevent` and `stream` is `0` or a live `CUstream`.
        let s = unsafe { cuEventRecord(self.handle, stream) };
        if s != 0 {
            bail!("cuEventRecord failed: {s}");
        }
        Ok(())
    }
    pub fn sync(&self) -> Result<()> {
        // SAFETY: `self.handle` is a live owned `CUevent` in the active CUDA context.
        let s = unsafe { cuEventSynchronize(self.handle) };
        if s != 0 {
            bail!("cuEventSynchronize failed: {s}");
        }
        Ok(())
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        // SAFETY: `self.handle` was created by `cuEventCreate` and is destroyed once on Drop.
        unsafe {
            let _ = cuEventDestroy_v2(self.handle);
        }
    }
}

/// Direct `cuLaunchKernel` wrapper.
///
/// # Safety
/// - `func` must be a valid `CUfunction` handle belonging to a loaded `CUmodule`
///   in the active CUDA context.
/// - `stream` must be `0` or a valid live `CUstream` handle in the active CUDA context.
/// - `params` must match the kernel's exact parameter count, order, types, and alignment,
///   and every pointer entry in `params` (as well as any device buffers referenced by the
///   kernel) must remain valid for the duration of the launch and asynchronous execution.
/// - `grid`, `block`, and `shared_bytes` must satisfy the kernel's preconditions so it
///   performs no out-of-bounds memory accesses or invalid warp operations.
pub unsafe fn launch_kernel(
    func: u64,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared_bytes: u32,
    stream: u64,
    params: &mut [*mut c_void],
) -> Result<()> {
    // SAFETY: Caller upholds the `# Safety` contract for `func`, `stream`, `params`,
    // launch geometry, and referenced device allocations.
    let s = unsafe {
        cuLaunchKernel(
            func,
            grid.0,
            grid.1,
            grid.2,
            block.0,
            block.1,
            block.2,
            shared_bytes,
            stream,
            params.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    };
    if s != 0 {
        bail!("cuLaunchKernel failed: status {s} grid={grid:?} block={block:?}");
    }
    Ok(())
}
