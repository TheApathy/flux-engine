// SPDX-License-Identifier: AGPL-3.0-only

//! Real CUDA GPU backend using AtlasRegistry.
//!
//! SBIO IORouter: all CUDA operations flow through `GpuBackend`.
//! Uses `AtlasRegistry` for kernel loading/launching and raw CUDA
//! driver API for memory management.

use std::ffi::c_void;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Result, bail};
use atlas_core::registry::AtlasRegistry;

mod alloc_poison;
mod allocation_lifetime;
mod bounded_copy;
mod display_carveout;
mod gpu_impl;
mod transform_cache_identity;
mod wfet_sync;

/// Process-wide CUDA primary context handle populated by [`AtlasCudaBackend::new`],
/// allowing background tasks (such as [`spawn_oom_watchdog`]) to bind the context on
/// arbitrary Tokio worker threads before querying driver memory info.
static PRIMARY_CUDA_CTX: AtomicU64 = AtomicU64::new(0);

// ── Raw CUDA driver API for memory operations ──

// SAFETY: Symbols and C ABI signatures match `cuda.h` in the linked `libcuda` driver library.
unsafe extern "C" {
    pub(super) fn cuMemAlloc_v2(dptr: *mut u64, bytesize: usize) -> i32;
    pub(super) fn cuMemFree_v2(dptr: u64) -> i32;
    pub(super) fn cuMemGetAddressRange_v2(pbase: *mut u64, psize: *mut usize, dptr: u64) -> i32;
    pub(super) fn cuPointerGetAttribute(data: *mut c_void, attribute: u32, ptr: u64) -> i32;
    pub(super) fn cuMemcpyHtoD_v2(dst: u64, src: *const c_void, bytes: usize) -> i32;
    pub(super) fn cuMemcpyDtoH_v2(dst: *mut c_void, src: u64, bytes: usize) -> i32;
    pub(super) fn cuMemcpyHtoDAsync_v2(
        dst: u64,
        src: *const c_void,
        bytes: usize,
        stream: u64,
    ) -> i32;
    pub(super) fn cuMemcpyDtoDAsync_v2(dst: u64, src: u64, bytes: usize, stream: u64) -> i32;
    pub(super) fn cuMemcpyDtoHAsync_v2(
        dst: *mut c_void,
        src: u64,
        bytes: usize,
        stream: u64,
    ) -> i32;
    pub(super) fn cuStreamIsCapturing(hStream: u64, captureStatus: *mut u32) -> i32;
    pub(super) fn cuStreamQuery(hStream: u64) -> i32;
    pub(super) fn cuStreamSynchronize(stream: u64) -> i32;
    pub(super) fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> i32;
    pub(super) fn cuMemsetD8Async(dst: u64, value: u8, n: usize, stream: u64) -> i32;
    pub(super) fn cuMemsetD32Async(dst: u64, value: u32, n: usize, stream: u64) -> i32;
    // CUDA graph capture/replay
    pub(super) fn cuStreamBeginCapture(hStream: u64, mode: u32) -> i32;
    pub(super) fn cuStreamEndCapture(hStream: u64, phGraph: *mut u64) -> i32;
    pub(super) fn cuGraphInstantiateWithFlags(
        phGraphExec: *mut u64,
        hGraph: u64,
        flags: u64,
    ) -> i32;
    pub(super) fn cuGraphLaunch(hGraphExec: u64, hStream: u64) -> i32;
    pub(super) fn cuGraphExecDestroy(hGraphExec: u64) -> i32;
    pub(super) fn cuGraphDestroy(hGraph: u64) -> i32;
    pub(super) fn cuDeviceGraphMemTrim(device: i32) -> i32;
    pub(super) fn cuCtxSetCurrent(ctx: u64) -> i32;
    pub(super) fn cuCtxSynchronize() -> i32;
    // Device properties. `CUdevice` is an opaque int handle, not a pointer.
    pub(super) fn cuCtxGetDevice(device: *mut i32) -> i32;
    pub(super) fn cuDeviceGetAttribute(pi: *mut i32, attrib: u32, dev: i32) -> i32;
    pub(super) fn cuDriverGetVersion(driver_version: *mut i32) -> i32;
    pub(super) fn cuStreamCreate(phStream: *mut u64, flags: u32) -> i32;
    // Page-locked host memory for efficient async transfers
    pub(super) fn cuMemAllocHost_v2(pp: *mut *mut c_void, bytesize: usize) -> i32;
    pub(super) fn cuMemHostAlloc(pp: *mut *mut c_void, bytesize: usize, flags: u32) -> i32;
    pub(super) fn cuMemHostGetDevicePointer_v2(pdptr: *mut u64, p: *mut c_void, flags: u32) -> i32;
    pub(super) fn cuStreamWriteValue32_v2(stream: u64, addr: u64, value: u32, flags: u32) -> i32;
    pub(super) fn cuMemFreeHost(p: *mut c_void) -> i32;
    pub(super) fn cuLaunchCooperativeKernel(
        f: *mut c_void,
        grid_dim_x: u32,
        grid_dim_y: u32,
        grid_dim_z: u32,
        block_dim_x: u32,
        block_dim_y: u32,
        block_dim_z: u32,
        shared_mem_bytes: u32,
        stream: u64,
        kernel_params: *mut *mut c_void,
    ) -> i32;
    pub(super) fn cuFuncSetAttribute(hfunc: *mut c_void, attrib: i32, value: i32) -> i32;
    // Managed (unified) memory — allows over-subscription with Linux swap paging
    pub(super) fn cuMemAllocManaged(dptr: *mut u64, bytesize: usize, flags: u32) -> i32;
    // CUDA events for inter-stream synchronization
    pub(super) fn cuEventCreate(phEvent: *mut u64, flags: u32) -> i32;
    pub(super) fn cuEventRecord(hEvent: u64, hStream: u64) -> i32;
    pub(super) fn cuStreamWaitEvent(hStream: u64, hEvent: u64, flags: u32) -> i32;
    pub(super) fn cuEventDestroy_v2(hEvent: u64) -> i32;
    // CUDA_ERROR_NOT_READY (600) on a still-pending event; CUDA_SUCCESS (0) once
    // it has completed. Never blocks — the polling counterpart to
    // cuEventSynchronize, used so a stalled H2D copy can be bounded by a
    // wall-clock timeout instead of hanging inside the driver indefinitely.
    pub(super) fn cuEventQuery(hEvent: u64) -> i32;
}

/// Production GPU backend wrapping AtlasRegistry + raw CUDA driver API.
///
/// Initialized once via [`AtlasCudaBackend::new`], which loads all PTX
/// modules from `atlas-kernels` into the global AtlasRegistry singleton.
pub struct AtlasCudaBackend {
    /// The module registry this backend was initialized with. Kernel lookups
    /// and launches go through it rather than the process-wide "current"
    /// registry, so a backend keeps resolving its own target's kernels even
    /// after a later swap initializes a different target.
    registry: &'static atlas_core::registry::AtlasRegistry,
    /// Default CUDA stream handle (from AtlasRegistry).
    default_stream: u64,
    /// CUDA context handle for cross-thread binding.
    cuda_ctx: u64,
    /// Exact ordered module-name/PTX receipt used to initialize the registry.
    transform_cache_identity: Option<String>,
    /// Runtime hardware properties queried from the CUDA driver at backend init.
    device_props: Option<atlas_core::device::GpuDeviceProps>,
    /// Every device pointer this instance has handed out via `alloc`/
    /// `alloc_managed` and not yet individually freed via `free`, keyed to
    /// its byte size. Model weights, KV cache, SSM pools and the buffer
    /// arena never call `free()` themselves (BUG #29 — see the doc on
    /// `GpuBackend::free_all_allocations`), so without this table nothing
    /// ever reclaims them until the process exits. Drained in one pass by
    /// `free_all_allocations` at model teardown.
    live_allocations:
        std::sync::Mutex<std::collections::HashMap<u64, allocation_lifetime::Allocation>>,
}

impl AtlasCudaBackend {
    /// Initialize the CUDA backend on the given GPU ordinal.
    ///
    /// Loads the provided PTX modules into AtlasRegistry.
    /// Use `atlas_kernels::ptx_for_model()` or `ptx_modules()` to
    /// obtain the correct module set for the target model.
    /// Subsequent calls reuse the cached singleton.
    pub fn new(ordinal: usize, ptx_modules: &[(&'static str, &str)]) -> Result<Self> {
        let registry = AtlasRegistry::get_or_init(ordinal, ptx_modules)
            .map_err(|e| anyhow::anyhow!("AtlasRegistry init failed: {e}"))?;
        registry
            .ctx
            .bind_to_thread()
            .map_err(|e| anyhow::anyhow!("AtlasRegistry context bind failed: {e}"))?;
        let default_stream = registry.raw_stream();

        // Capture the registry-owned CUDA context for cross-thread binding.
        let cuda_ctx = registry.ctx.cu_ctx() as u64;
        if cuda_ctx == 0 {
            bail!("AtlasRegistry returned a null CUDA context");
        }
        PRIMARY_CUDA_CTX.store(cuda_ctx, Ordering::Release);

        tracing::info!(
            "AtlasCudaBackend initialized on GPU {ordinal} with {} PTX modules",
            ptx_modules.len()
        );

        let device_props = match Self::query_current_device_props() {
            Ok(props) => {
                atlas_core::device::set_active_device_props(props);
                let topology = if props.is_integrated {
                    "integrated"
                } else {
                    "discrete"
                };
                tracing::info!(
                    "CUDA device {ordinal}: SM{}.{} ({} SMs, {} KB SMEM/SM, {})",
                    props.compute_major,
                    props.compute_minor,
                    props.sm_count,
                    props.smem_per_sm / 1024,
                    topology,
                );
                Some(props)
            }
            Err(error) => {
                tracing::warn!("CUDA device {ordinal} property query unavailable: {error:#}");
                None
            }
        };

        let transform_cache_identity =
            match transform_cache_identity::identity(registry.initialization_identity()) {
                Ok(identity) => Some(identity),
                Err(error) => {
                    tracing::warn!("transform-cache CUDA identity unavailable: {error:#}");
                    None
                }
            };
        Ok(Self {
            registry,
            default_stream,
            cuda_ctx,
            transform_cache_identity,
            device_props,
            live_allocations: std::sync::Mutex::new(std::collections::HashMap::new()),
        })
    }

    /// `CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT`.
    const ATTR_MULTIPROCESSOR_COUNT: u32 = 16;
    /// `CU_DEVICE_ATTRIBUTE_INTEGRATED` (`1` = unified/integrated, `0` = discrete).
    pub(super) const ATTR_INTEGRATED: u32 = 18;
    /// `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR`.
    const ATTR_COMPUTE_CAPABILITY_MAJOR: u32 = 75;
    /// `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR`.
    const ATTR_COMPUTE_CAPABILITY_MINOR: u32 = 76;
    /// `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR`.
    const ATTR_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR: u32 = 81;

    /// Query runtime hardware properties for the CUDA device bound to the
    /// current thread's context.
    fn query_current_device_props() -> Result<atlas_core::device::GpuDeviceProps> {
        let mut device: i32 = 0;
        // SAFETY: `AtlasCudaBackend::new` has bound a live CUDA context to the
        // calling thread and `device` is a valid out-pointer.
        let status = unsafe { cuCtxGetDevice(&mut device) };
        if status != 0 {
            bail!("cuCtxGetDevice failed: status {status}");
        }
        let get_attr = |attrib: u32, label: &str| -> Result<i32> {
            let mut value: i32 = 0;
            // SAFETY: `device` is a valid `CUdevice` handle from `cuCtxGetDevice`
            // and `value` is a valid out-pointer.
            let status = unsafe { cuDeviceGetAttribute(&mut value, attrib, device) };
            if status != 0 {
                bail!("cuDeviceGetAttribute({label}) failed: status {status}");
            }
            Ok(value)
        };
        let sm_count = get_attr(Self::ATTR_MULTIPROCESSOR_COUNT, "MULTIPROCESSOR_COUNT")?;
        let smem_per_sm = get_attr(
            Self::ATTR_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
            "MAX_SHARED_MEMORY_PER_MULTIPROCESSOR",
        )?;
        let integrated = get_attr(Self::ATTR_INTEGRATED, "INTEGRATED")?;
        let compute_major = get_attr(
            Self::ATTR_COMPUTE_CAPABILITY_MAJOR,
            "COMPUTE_CAPABILITY_MAJOR",
        )?;
        let compute_minor = get_attr(
            Self::ATTR_COMPUTE_CAPABILITY_MINOR,
            "COMPUTE_CAPABILITY_MINOR",
        )?;

        let mut _free: usize = 0;
        let mut total_mem_bytes: usize = 0;
        // SAFETY: Active CUDA context is bound; out-pointers are valid.
        let mem_status = unsafe { cuMemGetInfo_v2(&mut _free, &mut total_mem_bytes) };
        if mem_status != 0 {
            bail!("cuMemGetInfo_v2 failed: status {mem_status}");
        }

        Ok(atlas_core::device::GpuDeviceProps {
            compute_major: u32::try_from(compute_major).unwrap_or(0),
            compute_minor: u32::try_from(compute_minor).unwrap_or(0),
            sm_count: u32::try_from(sm_count).unwrap_or(0),
            smem_per_sm: usize::try_from(smem_per_sm).unwrap_or(0),
            is_integrated: integrated != 0,
            total_mem_bytes,
        })
    }

    /// Device SM count, queried once and cached for the process.
    ///
    /// The query needs only the current context's device handle — no
    /// allocation, no launch — but it still goes through the driver, so it is
    /// resolved on first use and never repeated. A failure is logged once and
    /// cached as `None` so a broken driver cannot turn into a per-launch log
    /// storm.
    pub(super) fn cached_sm_count() -> Option<u32> {
        static CACHE: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
        *CACHE.get_or_init(|| {
            let mut device: i32 = 0;
            // SAFETY: `&mut device` is a valid out-pointer; `cuCtxGetDevice` returns a non-zero
            // error code if no CUDA context is bound on the calling thread.
            let status = unsafe { cuCtxGetDevice(&mut device) };
            if status != 0 {
                tracing::warn!(
                    "cuCtxGetDevice failed (status {status}); occupancy sizing falls back to \
                     its compiled-in SM literal"
                );
                return None;
            }
            let mut count: i32 = 0;
            // SAFETY: `device` is a valid `CUdevice` handle from `cuCtxGetDevice` and `&mut count`
            // is a valid writable out-pointer.
            let status = unsafe {
                cuDeviceGetAttribute(&mut count, Self::ATTR_MULTIPROCESSOR_COUNT, device)
            };
            if status != 0 || count <= 0 {
                tracing::warn!(
                    "cuDeviceGetAttribute(MULTIPROCESSOR_COUNT) failed (status {status}, \
                     count {count}); occupancy sizing falls back to its compiled-in SM literal"
                );
                return None;
            }
            tracing::info!("GPU multiprocessor count: {count}");
            Some(count as u32)
        })
    }
}

// ── OOM Watchdog ────────────────────────────────────────────────────
//
// Background task that polls GPU free memory every `interval` and calls
// `std::process::exit(1)` if it drops below `threshold_bytes`.
// On GB10 unified memory, GPU OOM = system OOM = kernel freeze, so
// killing the process early prevents unrecoverable system hangs.

/// Determine whether the active CUDA context's device uses integrated/unified
/// memory (`Some(true)`) or discrete VRAM (`Some(false)`).
///
/// Checks the lock-free `atlas_core::device::active_device_is_integrated()`
/// cache first; if unset (e.g. before `AtlasCudaBackend::new`), queries the
/// driver directly via `cuCtxGetDevice` + `cuDeviceGetAttribute(INTEGRATED)`.
pub(super) fn current_device_is_integrated() -> Option<bool> {
    if let Some(is_integrated) = atlas_core::device::active_device_is_integrated() {
        return Some(is_integrated);
    }
    let mut device: i32 = 0;
    // SAFETY: `cuCtxGetDevice` returns an error code if no CUDA context is
    // bound on the calling thread; `device` is a valid out-pointer.
    if unsafe { cuCtxGetDevice(&mut device) } != 0 {
        return None;
    }
    let mut integrated: i32 = 0;
    // SAFETY: `device` is a valid `CUdevice` handle from `cuCtxGetDevice` and
    // `integrated` is a valid out-pointer.
    if unsafe { cuDeviceGetAttribute(&mut integrated, AtlasCudaBackend::ATTR_INTEGRATED, device) }
        != 0
    {
        return None;
    }
    Some(integrated != 0)
}

/// Compute effective `(free, total)` GPU memory bytes for unified vs discrete
/// devices.
///
/// On integrated/unified-memory devices (e.g. GB10), `cuMemGetInfo_v2` reports
/// Linux `MemFree` (excluding reclaimable page cache), under-reporting usable
/// memory by 30-50%, so host `MemAvailable` is merged with `cu_free.max(...)`.
/// On discrete GPUs, VRAM is disjoint from host RAM, so host `MemAvailable` is
/// never merged and exact `(cu_free, cu_total)` is returned.
#[must_use]
pub(crate) fn effective_memory_bytes(
    cu_free: usize,
    cu_total: usize,
    is_integrated: bool,
    host_mem_available: Option<usize>,
) -> (usize, usize) {
    let free = if is_integrated {
        host_mem_available.map_or(cu_free, |mem_available| cu_free.max(mem_available))
    } else {
        cu_free
    };
    (free, cu_total)
}

/// Query GPU free memory without requiring a GpuBackend reference.
/// Safe to call from any thread that shares the CUDA context.
///
/// On unified memory systems (GB10), `cuMemGetInfo` reports Linux's "free" memory
/// which excludes reclaimable buff/cache. This under-reports available memory by
/// 30-50%. We take the max of CUDA's report and `/proc/meminfo` MemAvailable
/// to get the true available memory. On discrete GPUs, exact VRAM free bytes
/// from `cuMemGetInfo_v2` are returned without merging host RAM.
pub fn cuda_free_memory_bytes() -> Option<usize> {
    cuda_memory_bytes().map(|(free, _)| free)
}

/// Query free and total GPU memory without requiring a backend reference.
pub fn cuda_memory_bytes() -> Option<(usize, usize)> {
    let mut free: usize = 0;
    let mut total: usize = 0;
    // SAFETY: `cuMemGetInfo_v2` writes to valid out-pointers when a CUDA
    // context is bound and returns a non-zero error status otherwise.
    let status = unsafe { cuMemGetInfo_v2(&mut free, &mut total) };
    if status != 0 {
        return None;
    }

    // Only merge host `/proc/meminfo` `MemAvailable` on integrated/UMA devices.
    // On discrete GPUs (or when topology cannot be confirmed), never inflate
    // VRAM with host RAM.
    let is_integrated = current_device_is_integrated().unwrap_or(false);
    let host_mem_available = is_integrated.then(system_available_memory_bytes).flatten();
    Some(effective_memory_bytes(
        free,
        total,
        is_integrated,
        host_mem_available,
    ))
}

/// Read MemAvailable from /proc/meminfo (Linux only).
/// Returns None on non-Linux or if parsing fails.
fn system_available_memory_bytes() -> Option<usize> {
    let contents = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in contents.lines() {
        if line.starts_with("MemAvailable:") {
            let kb: usize = line.split_whitespace().nth(1)?.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

/// Start a background OOM watchdog that polls GPU memory every `interval`.
///
/// Terminates the process via `std::process::exit(1)` after **3 consecutive**
/// readings where free memory is below `threshold_mb` MiB (hysteresis prevents
/// false positives during transient allocation spikes). Before each reading,
/// the task binds the primary CUDA context registered by
/// [`AtlasCudaBackend::new`] so polling works on arbitrary Tokio worker threads.
///
/// Returns a `tokio::task::JoinHandle`. Dropping a `JoinHandle` only detaches
/// the task — callers must invoke `handle.abort()` (e.g. via a drop guard) on
/// shutdown to stop the watchdog.
pub fn spawn_oom_watchdog(
    threshold_mb: usize,
    interval: std::time::Duration,
) -> tokio::task::JoinHandle<()> {
    let threshold_bytes = threshold_mb.saturating_mul(1024 * 1024);
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(interval);
        // Track consecutive low-memory readings to avoid false positives
        // during transient allocation spikes.
        let mut consecutive_low = 0u32;
        loop {
            tick.tick().await;
            let primary_ctx = PRIMARY_CUDA_CTX.load(Ordering::Acquire);
            if primary_ctx != 0 {
                // SAFETY: `primary_ctx` is the process-lifetime CUDA primary context published by
                // `AtlasCudaBackend::new`; binding it on the current Tokio worker thread makes
                // `cuMemGetInfo_v2` succeed instead of returning `CUDA_ERROR_INVALID_CONTEXT`.
                let _ = unsafe { cuCtxSetCurrent(primary_ctx) };
            }
            if let Some(free) = cuda_free_memory_bytes() {
                if free < threshold_bytes {
                    consecutive_low += 1;
                    let free_mb = free / (1024 * 1024);
                    tracing::error!(
                        "OOM watchdog: GPU free memory critically low: {} MB (threshold: {} MB) [{}/3]",
                        free_mb,
                        threshold_mb,
                        consecutive_low,
                    );
                    if consecutive_low >= 3 {
                        tracing::error!(
                            "OOM watchdog: 3 consecutive readings below threshold. \
                             Terminating to prevent system freeze."
                        );
                        // Flush stdio before terminating (note: async tracing buffers are not drained by process::exit).
                        let _ = std::io::stdout().flush();
                        let _ = std::io::stderr().flush();
                        std::process::exit(1);
                    }
                } else {
                    consecutive_low = 0;
                }
            }
        }
    })
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod memory_tests {
    use super::effective_memory_bytes;

    const GIB: usize = 1024 * 1024 * 1024;

    #[test]
    fn discrete_gpu_never_merges_host_mem_available() {
        // 32 GiB RTX 5090 (28 GiB free VRAM) in a 256 GiB host (240 GiB host MemAvailable).
        assert_eq!(
            effective_memory_bytes(28 * GIB, 32 * GIB, false, Some(240 * GIB)),
            (28 * GIB, 32 * GIB)
        );
        // 96 GiB RTX PRO 6000 nearly full (2 GiB free VRAM) in a 512 GiB host.
        assert_eq!(
            effective_memory_bytes(2 * GIB, 96 * GIB, false, Some(480 * GIB)),
            (2 * GIB, 96 * GIB)
        );
        // 288 GiB B300/GB300 discrete HBM in a 1 TiB host.
        assert_eq!(
            effective_memory_bytes(120 * GIB, 288 * GIB, false, Some(900 * GIB)),
            (120 * GIB, 288 * GIB)
        );
        // Host telemetry missing still returns exact CUDA VRAM.
        assert_eq!(
            effective_memory_bytes(10 * GIB, 32 * GIB, false, None),
            (10 * GIB, 32 * GIB)
        );
    }

    #[test]
    fn integrated_uma_merges_reclaimable_host_mem_available() {
        // GB10 UMA: cuMemGetInfo under-reports free memory because Linux page cache is reclaimable.
        assert_eq!(
            effective_memory_bytes(40 * GIB, 128 * GIB, true, Some(95 * GIB)),
            (95 * GIB, 128 * GIB)
        );
        // When CUDA free already exceeds host MemAvailable, keep CUDA free.
        assert_eq!(
            effective_memory_bytes(80 * GIB, 128 * GIB, true, Some(72 * GIB)),
            (80 * GIB, 128 * GIB)
        );
        // When /proc/meminfo is unavailable, fall back to CUDA free.
        assert_eq!(
            effective_memory_bytes(60 * GIB, 128 * GIB, true, None),
            (60 * GIB, 128 * GIB)
        );
    }
}
