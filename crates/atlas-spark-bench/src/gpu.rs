// SPDX-License-Identifier: AGPL-3.0-only

//! GPU utilities for kernel microbenchmarks.
//!
//! Initializes `AtlasRegistry`, allocates device buffers, and provides
//! CUDA event-based timing for Criterion `iter_custom` benchmarks.

use std::ffi::c_void;
use std::sync::OnceLock;

use anyhow::{Result, bail};
use atlas_core::registry::{AtlasRegistry, RawCudaFunc};
use cudarc::driver::LaunchConfig;

// SAFETY: Standard CUDA Driver API symbols exported by `libcuda.so` using the C ABI.
unsafe extern "C" {
    fn cuMemAlloc_v2(dptr: *mut u64, bytesize: usize) -> i32;
    fn cuMemFree_v2(dptr: u64) -> i32;
    fn cuMemsetD8Async(dst: u64, value: u8, n: usize, stream: u64) -> i32;
    fn cuMemcpyHtoD_v2(dst: u64, src: *const c_void, bytesize: usize) -> i32;
    fn cuStreamSynchronize(stream: u64) -> i32;
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

static INIT: OnceLock<()> = OnceLock::new();

/// Ensure the AtlasRegistry is initialized (idempotent).
pub fn ensure_registry() -> &'static AtlasRegistry {
    INIT.get_or_init(|| {
        let ptx = atlas_kernels::ptx_modules();
        AtlasRegistry::get_or_init(0, &ptx).expect("AtlasRegistry init failed — is GPU available?");
    });
    AtlasRegistry::get()
}

/// Allocate `bytes` of GPU memory, zero-initialized.
pub fn gpu_alloc_zeroed(stream: u64, bytes: usize) -> Result<u64> {
    let mut dptr: u64 = 0;
    // SAFETY: `dptr` is a valid aligned out-pointer and CUDA context is bound via `ensure_registry`.
    let status = unsafe { cuMemAlloc_v2(&mut dptr, bytes) };
    if status != 0 {
        bail!("cuMemAlloc_v2 failed: status {status}, {bytes} bytes");
    }
    // SAFETY: `dptr` was just allocated with `bytes` writable device bytes; `stream` is valid or 0.
    let status = unsafe { cuMemsetD8Async(dptr, 0, bytes, stream) };
    if status != 0 {
        // SAFETY: `dptr` was allocated above and is not returned to the caller; release on failure.
        unsafe {
            cuMemFree_v2(dptr);
        }
        bail!("cuMemsetD8Async failed: status {status}");
    }
    Ok(dptr)
}

/// Synchronously upload `src` host bytes to a CUDA device buffer.
///
/// # Safety
/// `dst` must be a valid writable CUDA device allocation of at least `src.len()` bytes
/// in the active CUDA context, with no concurrent unsynchronized GPU writes.
pub unsafe fn gpu_upload_h2d(dst: u64, src: &[u8]) -> Result<()> {
    if src.is_empty() {
        return Ok(());
    }
    // SAFETY: Caller guarantees `dst` has at least `src.len()` writable device bytes.
    let status = unsafe { cuMemcpyHtoD_v2(dst, src.as_ptr().cast::<c_void>(), src.len()) };
    if status != 0 {
        bail!("cuMemcpyHtoD_v2 failed: status {status}");
    }
    Ok(())
}

/// Free GPU memory previously allocated by `gpu_alloc_zeroed` / `cuMemAlloc_v2`.
///
/// # Safety
/// `dptr` must be `0` or a live device allocation owned by the caller in the active
/// CUDA context, with all prior GPU operations referencing it completed and no
/// subsequent uses or double-frees.
pub unsafe fn gpu_free(dptr: u64) {
    if dptr != 0 {
        // SAFETY: Caller guarantees `dptr` is a valid live device pointer not freed again.
        unsafe { cuMemFree_v2(dptr) };
    }
}

/// Synchronize the stream.
pub fn gpu_sync(stream: u64) -> Result<()> {
    // SAFETY: `stream` is a valid CUstream handle (or 0) in the active CUDA context.
    let status = unsafe { cuStreamSynchronize(stream) };
    if status != 0 {
        bail!("cuStreamSynchronize failed: status {status}");
    }
    Ok(())
}

/// Look up a kernel function handle with caching.
pub fn get_kernel(
    registry: &'static AtlasRegistry,
    cache: &OnceLock<RawCudaFunc>,
    module: &str,
    func: &str,
) -> RawCudaFunc {
    registry
        .raw_function_cached(cache, module, func)
        .unwrap_or_else(|e| panic!("Kernel {module}::{func} not found: {e}"))
}

/// Launch a kernel with raw parameters.
///
/// # Safety
/// `params` must contain valid pointers matching the kernel signature.
pub unsafe fn launch(
    registry: &AtlasRegistry,
    func: RawCudaFunc,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared_mem: u32,
    stream: u64,
    params: &mut [*mut c_void],
) -> Result<()> {
    let cfg = LaunchConfig {
        grid_dim: grid,
        block_dim: block,
        shared_mem_bytes: shared_mem,
    };
    // SAFETY: Caller guarantees `func`, `stream`, `cfg`, and `params` satisfy `launch_on_stream`.
    unsafe { registry.launch_on_stream(func, cfg, stream, params) }
        .map_err(|e| anyhow::anyhow!("Kernel launch failed: {e}"))
}

/// Measure kernel execution time in milliseconds using CUDA events.
/// Runs `warmup` warmup iterations, then `iters` timed iterations (`iters > 0`),
/// returning the minimum per-iteration time across 3 rounds.
pub fn bench_kernel_ms(
    stream: u64,
    warmup: usize,
    iters: usize,
    mut kernel_fn: impl FnMut(),
) -> f32 {
    assert!(iters > 0, "bench_kernel_ms requires iters > 0");
    // Create CUDA events
    let mut start: u64 = 0;
    let mut end: u64 = 0;
    // SAFETY: `start` and `end` are valid aligned out-pointers in the active CUDA context.
    let (s1, s2) = unsafe { (cuEventCreate(&mut start, 0), cuEventCreate(&mut end, 0)) };
    assert!(
        s1 == 0 && s2 == 0,
        "cuEventCreate failed: start={s1}, end={s2}"
    );

    let mut best_ms = f32::MAX;

    for _ in 0..3 {
        // Warmup
        for _ in 0..warmup {
            kernel_fn();
        }
        gpu_sync(stream).unwrap();

        // Timed run
        // SAFETY: `start` was created above and `stream` is valid in the active context.
        let rec_start = unsafe { cuEventRecord(start, stream) };
        assert_eq!(rec_start, 0, "cuEventRecord(start) failed: {rec_start}");
        for _ in 0..iters {
            kernel_fn();
        }
        // SAFETY: `end` was created above and `stream` is valid in the active context.
        let rec_end = unsafe { cuEventRecord(end, stream) };
        assert_eq!(rec_end, 0, "cuEventRecord(end) failed: {rec_end}");
        // SAFETY: `end` was recorded on `stream` above.
        let sync_end = unsafe { cuEventSynchronize(end) };
        assert_eq!(sync_end, 0, "cuEventSynchronize(end) failed: {sync_end}");

        let mut elapsed_ms: f32 = 0.0;
        // SAFETY: Both `start` and `end` have completed and `elapsed_ms` is a valid out-pointer.
        let elapsed_status = unsafe { cuEventElapsedTime(&mut elapsed_ms, start, end) };
        assert_eq!(
            elapsed_status, 0,
            "cuEventElapsedTime failed: {elapsed_status}"
        );
        let per_iter = elapsed_ms / iters as f32;
        if per_iter < best_ms {
            best_ms = per_iter;
        }
    }

    // SAFETY: `start` and `end` were created above, synchronized, and are not used again.
    let (d1, d2) = unsafe { (cuEventDestroy_v2(start), cuEventDestroy_v2(end)) };
    debug_assert!(
        d1 == 0 && d2 == 0,
        "cuEventDestroy_v2 failed: start={d1}, end={d2}"
    );

    best_ms
}
