// SPDX-License-Identifier: AGPL-3.0-only

//! Raw NCCL FFI bindings (minimal surface for EP all-reduce + broadcast).
//!
//! Only the functions Flux Engine actually calls are bound here — no attempt
//! at complete coverage. Baseline ABI targets NCCL 2.24+ on 64-bit Linux
//! (`x86_64` and `aarch64`); [`NcclConfig`] models the `ncclConfig_v22800`
//! layout (NCCL 2.28+), and symmetric memory (`ncclMemAlloc`/`ncclMemFree`)
//! requires NCCL 2.28+.
//!
//! ## Safety
//!
//! Every `unsafe` block and `unsafe extern "C"` block in this module carries a
//! per-site `// SAFETY:` justification. Callers of raw NCCL/CUDA functions must
//! supply live communicator/stream/event handles and device buffers owned by
//! the active CUDA context with element counts matching buffer sizes.

use std::ffi::c_void;

/// Opaque NCCL communicator handle.
pub type NcclComm = *mut c_void;

/// NCCL unique ID for bootstrapping (128 bytes, passed by value in C ABI).
#[repr(C)]
#[derive(Copy, Clone)]
pub struct NcclUniqueId {
    pub internal: [u8; 128],
}

/// Magic value required in [`NcclConfig::magic`] (`NCCL_API_MAGIC` in `nccl.h`).
pub const NCCL_API_MAGIC: u32 = 0xcafe_beef;

/// Sentinel value for unspecified integer attributes in [`NcclConfig`]
/// (`NCCL_CONFIG_UNDEF_INT = INT_MIN` in `nccl.h`).
pub const NCCL_CONFIG_UNDEF_INT: i32 = i32::MIN;

/// Highest NCCL version code whose exact `ncclConfig_t` layout (`ncclConfig_v22800`,
/// 96 bytes on 64-bit Linux) is modeled by [`NcclConfig`] (`NCCL_VERSION(2, 30, 4)`).
///
/// Advertising `min(runtime_version, NCCL_CONFIG_MAX_KNOWN_VERSION)` ensures:
/// - Older runtimes (2.24–2.29) accept the exact runtime version instead of rejecting
///   a hardcoded future version with `InvalidArgument` ("NCCL library is too old").
/// - Newer runtimes (2.31+ with additional trailing fields) treat this struct as a
///   valid `ncclConfig_v22800` layout bounded by [`NcclConfig::size`].
pub const NCCL_CONFIG_MAX_KNOWN_VERSION: u32 = 23_004;

/// Fallback `NCCL_VERSION_CODE` (`2.28.0`) used only if `ncclGetVersion` fails.
const FALLBACK_NCCL_VERSION: u32 = 22_800;

/// NCCL communicator configuration (`ncclConfig_t` / `ncclConfig_v22800` in `nccl.h`).
///
/// Set `blocking = 0` for non-blocking initialization (`ncclCommInitRankConfig`
/// returns `InProgress` immediately so init can be polled with a bounded timeout).
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct NcclConfig {
    /// Size of this struct in bytes (for ABI versioning).
    pub size: usize,
    /// Magic number (`NCCL_API_MAGIC = 0xcafebeef`).
    pub magic: u32,
    /// NCCL version code (`NCCL_VERSION(major, minor, patch)`).
    pub version: u32,
    /// `1` = blocking (default), `0` = non-blocking.
    pub blocking: i32,
    /// CGA cluster size (`NCCL_CONFIG_UNDEF_INT` = default).
    pub cga_cluster_size: i32,
    /// Min CTAs for launch (`NCCL_CONFIG_UNDEF_INT` = default).
    pub min_ctas: i32,
    /// Max CTAs for launch (`NCCL_CONFIG_UNDEF_INT` = default).
    pub max_ctas: i32,
    /// Network name (`NULL` = default).
    pub net_name: *const std::ffi::c_char,
    /// Split share (`NCCL_CONFIG_UNDEF_INT` = default).
    pub split_share: i32,
    /// Traffic class (`NCCL_CONFIG_UNDEF_INT` = default).
    pub traffic_class: i32,
    /// Communicator name (`NULL` = default).
    pub comm_name: *const std::ffi::c_char,
    /// CollNet enable (`NCCL_CONFIG_UNDEF_INT` = default).
    pub collnet_enable: i32,
    /// CTA policy (`NCCL_CONFIG_UNDEF_INT` = default).
    pub cta_policy: i32,
    /// Shrink share (`NCCL_CONFIG_UNDEF_INT` = default).
    pub shrink_share: i32,
    /// NVLS CTAs (`NCCL_CONFIG_UNDEF_INT` = default).
    pub nvls_ctas: i32,
    /// Channels per net peer (`NCCL_CONFIG_UNDEF_INT` = default).
    pub n_channels_per_net_peer: i32,
    /// NVLink-centric scheduler (`NCCL_CONFIG_UNDEF_INT` = default).
    pub nvlink_centric_sched: i32,
    /// CUDA Graph usage mode (`NCCL_CONFIG_UNDEF_INT` = default).
    pub graph_usage_mode: i32,
    /// Number of RMA contexts (`NCCL_CONFIG_UNDEF_INT` = default).
    pub num_rma_ctx: i32,
    /// Max P2P peers (`NCCL_CONFIG_UNDEF_INT` = default).
    pub max_p2p_peers: i32,
}

impl NcclConfig {
    /// Construct a non-blocking config for a specific runtime `NCCL_VERSION_CODE`.
    ///
    /// Matches `NCCL_CONFIG_INITIALIZER` from `nccl.h` with `blocking = 0`,
    /// `magic = NCCL_API_MAGIC`, `NCCL_CONFIG_UNDEF_INT` (`i32::MIN`) for all
    /// unspecified integer fields, and `NULL` for pointer fields.
    #[must_use]
    pub fn non_blocking_for_version(runtime_version: u32) -> Self {
        let version = if runtime_version == 0 {
            FALLBACK_NCCL_VERSION
        } else {
            runtime_version.min(NCCL_CONFIG_MAX_KNOWN_VERSION)
        };
        Self {
            size: std::mem::size_of::<Self>(),
            magic: NCCL_API_MAGIC,
            version,
            blocking: 0,
            cga_cluster_size: NCCL_CONFIG_UNDEF_INT,
            min_ctas: NCCL_CONFIG_UNDEF_INT,
            max_ctas: NCCL_CONFIG_UNDEF_INT,
            net_name: std::ptr::null(),
            split_share: NCCL_CONFIG_UNDEF_INT,
            traffic_class: NCCL_CONFIG_UNDEF_INT,
            comm_name: std::ptr::null(),
            collnet_enable: NCCL_CONFIG_UNDEF_INT,
            cta_policy: NCCL_CONFIG_UNDEF_INT,
            shrink_share: NCCL_CONFIG_UNDEF_INT,
            nvls_ctas: NCCL_CONFIG_UNDEF_INT,
            n_channels_per_net_peer: NCCL_CONFIG_UNDEF_INT,
            nvlink_centric_sched: NCCL_CONFIG_UNDEF_INT,
            graph_usage_mode: NCCL_CONFIG_UNDEF_INT,
            num_rma_ctx: NCCL_CONFIG_UNDEF_INT,
            max_p2p_peers: NCCL_CONFIG_UNDEF_INT,
        }
    }

    /// Create a non-blocking config dynamically matching the linked `libnccl`
    /// runtime version reported by `ncclGetVersion`.
    #[must_use]
    pub fn non_blocking() -> Self {
        let runtime_version = match nccl_version() {
            Ok(ver) => ver,
            Err(err) => {
                tracing::warn!(
                    "ncclGetVersion failed ({err}); falling back to version {FALLBACK_NCCL_VERSION}"
                );
                FALLBACK_NCCL_VERSION
            }
        };
        Self::non_blocking_for_version(runtime_version)
    }
}

/// NCCL result code (`ncclResult_t`).
///
/// Represented as a transparent `c_int` newtype rather than a Rust `enum` so
/// status codes added by newer `libnccl.so` releases cross FFI without
/// enum-validity undefined behavior.
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct NcclResult(pub std::ffi::c_int);

#[allow(non_upper_case_globals)]
impl NcclResult {
    /// Operation completed without error (`ncclSuccess`).
    pub const Success: Self = Self(0);
    /// Unhandled CUDA error (`ncclUnhandledCudaError`).
    pub const UnhandledCudaError: Self = Self(1);
    /// System/OS error (`ncclSystemError`).
    pub const SystemError: Self = Self(2);
    /// Internal NCCL error (`ncclInternalError`).
    pub const InternalError: Self = Self(3);
    /// Invalid argument (`ncclInvalidArgument`).
    pub const InvalidArgument: Self = Self(4);
    /// Invalid usage (`ncclInvalidUsage`).
    pub const InvalidUsage: Self = Self(5);
    /// Remote peer error (`ncclRemoteError`).
    pub const RemoteError: Self = Self(6);
    /// Asynchronous operation still in progress (`ncclInProgress`).
    pub const InProgress: Self = Self(7);
    /// Timeout (`ncclTimeout` / extension status 8).
    pub const Timeout: Self = Self(8);

    /// Construct an [`NcclResult`] from a raw C `ncclResult_t` integer.
    #[must_use]
    pub const fn from_raw(code: std::ffi::c_int) -> Self {
        Self(code)
    }

    /// Return the raw C `ncclResult_t` integer value.
    #[must_use]
    pub const fn raw(self) -> std::ffi::c_int {
        self.0
    }
}

impl std::fmt::Debug for NcclResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::Success => f.write_str("Success"),
            Self::UnhandledCudaError => f.write_str("UnhandledCudaError"),
            Self::SystemError => f.write_str("SystemError"),
            Self::InternalError => f.write_str("InternalError"),
            Self::InvalidArgument => f.write_str("InvalidArgument"),
            Self::InvalidUsage => f.write_str("InvalidUsage"),
            Self::RemoteError => f.write_str("RemoteError"),
            Self::InProgress => f.write_str("InProgress"),
            Self::Timeout => f.write_str("Timeout"),
            Self(code) => write!(f, "Unknown({code})"),
        }
    }
}

/// NCCL data types (matches nccl.h enum values).
#[repr(C)]
#[derive(Debug, Copy, Clone)]
#[allow(dead_code)]
pub enum NcclDataType {
    Int8 = 0,
    Uint8 = 1,
    Int32 = 2,
    Uint32 = 3,
    Int64 = 4,
    Uint64 = 5,
    Float16 = 6,
    Float32 = 7,
    Float64 = 8,
    Bfloat16 = 9,
}

/// NCCL reduction operations.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
#[allow(dead_code)]
pub enum NcclRedOp {
    Sum = 0,
    Prod = 1,
    Max = 2,
    Min = 3,
    Avg = 4,
}

// SAFETY: External symbol signatures match the C ABI of `nccl.h` for NCCL 2.24+
// (`ncclConfig_v22800` and symmetric memory APIs require NCCL 2.28+); `NcclResult`
// is `#[repr(transparent)]` over `c_int`, matching C `ncclResult_t`.
#[link(name = "nccl")]
unsafe extern "C" {
    pub fn ncclGetVersion(version: *mut i32) -> NcclResult;

    pub fn ncclGetUniqueId(id: *mut NcclUniqueId) -> NcclResult;

    pub fn ncclCommInitRank(
        comm: *mut NcclComm,
        nranks: i32,
        id: NcclUniqueId,
        rank: i32,
    ) -> NcclResult;

    /// Non-blocking variant: set config.blocking=0 so ncclGroupEnd returns immediately.
    /// Poll ncclCommGetAsyncError to check completion and detect hangs with timeout.
    pub fn ncclCommInitRankConfig(
        comm: *mut NcclComm,
        nranks: i32,
        id: NcclUniqueId,
        rank: i32,
        config: *const NcclConfig,
    ) -> NcclResult;

    pub fn ncclAllReduce(
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        count: usize,
        datatype: NcclDataType,
        op: NcclRedOp,
        comm: NcclComm,
        stream: u64, // cudaStream_t
    ) -> NcclResult;

    pub fn ncclBroadcast(
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        count: usize,
        datatype: NcclDataType,
        root: i32,
        comm: NcclComm,
        stream: u64,
    ) -> NcclResult;

    pub fn ncclCommDestroy(comm: NcclComm) -> NcclResult;

    pub fn ncclGetErrorString(result: NcclResult) -> *const std::ffi::c_char;

    // Buffer registration (pre-registers with IB HCA, avoids per-call ibv_reg_mr).
    pub fn ncclCommRegister(
        comm: NcclComm,
        buff: *mut c_void,
        size: usize,
        handle: *mut *mut c_void,
    ) -> NcclResult;

    pub fn ncclCommDeregister(comm: NcclComm, handle: *mut c_void) -> NcclResult;

    // NCCL 2.28+ symmetric memory window APIs.
    //
    // Buffers allocated via `ncclMemAlloc` participate in symmetric memory
    // windows across the communicator, enabling:
    //   1. Copy-engine offload on NVLink-connected ranks (frees SMs for
    //      compute during AllReduce/AllGather).
    //   2. The device-side communication API (kernels can issue collectives
    //      directly without host round-trip), which is the substrate for
    //      fused AllReduce+RMSNorm+Residual kernels (TokenWeave-style).
    //
    // For Atlas's 2-rank Spark over RoCE, the copy-engine path doesn't
    // apply (RoCE is not NVLink), but the symmetric-memory windows are
    // still needed to compose with future device-API fusions and reduce
    // NCCL setup overhead via pre-registered handles.
    //
    // Returns `InvalidArgument` if the linked NCCL is < 2.28.
    pub fn ncclMemAlloc(ptr: *mut *mut c_void, size: usize) -> NcclResult;

    pub fn ncclMemFree(ptr: *mut c_void) -> NcclResult;

    // Point-to-point (for custom 2-rank all-reduce).
    pub fn ncclSend(
        sendbuf: *const c_void,
        count: usize,
        datatype: NcclDataType,
        peer: i32,
        comm: NcclComm,
        stream: u64,
    ) -> NcclResult;

    pub fn ncclRecv(
        recvbuf: *mut c_void,
        count: usize,
        datatype: NcclDataType,
        peer: i32,
        comm: NcclComm,
        stream: u64,
    ) -> NcclResult;

    // Collective: all-gather (each rank sends `count`, recv gets `world_size * count`).
    pub fn ncclAllGather(
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        sendcount: usize,
        datatype: NcclDataType,
        comm: NcclComm,
        stream: u64,
    ) -> NcclResult;

    // Collective: reduce-scatter (send has `world_size * count`, each rank recvs `count`).
    pub fn ncclReduceScatter(
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        recvcount: usize,
        datatype: NcclDataType,
        op: NcclRedOp,
        comm: NcclComm,
        stream: u64,
    ) -> NcclResult;

    // Group API (batch multiple send/recv into one launch).
    pub fn ncclGroupStart() -> NcclResult;
    pub fn ncclGroupEnd() -> NcclResult;

    // Health check: retrieve asynchronous errors from the communicator.
    pub fn ncclCommGetAsyncError(comm: NcclComm, async_error: *mut NcclResult) -> NcclResult;

    // Abort: destroy a communicator that is in a failed state.
    // Unlike ncclCommDestroy, this does not block and cleans up immediately.
    pub fn ncclCommAbort(comm: NcclComm) -> NcclResult;
}

// CUDA driver API for inter-stream synchronization (async all-reduce).
// spark-comm already links libcuda transitively via NCCL.
// SAFETY: External symbol signatures match the 64-bit CUDA Driver API C ABI in `cuda.h`.
#[link(name = "cuda")]
unsafe extern "C" {
    fn cuStreamCreate(phStream: *mut u64, flags: u32) -> i32;
    fn cuEventCreate(phEvent: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(hEvent: u64, hStream: u64) -> i32;
    fn cuStreamWaitEvent(hStream: u64, hEvent: u64, flags: u32) -> i32;
    fn cuEventDestroy_v2(hEvent: u64) -> i32;
    fn cuStreamDestroy_v2(hStream: u64) -> i32;
    fn cuStreamSynchronize(hStream: u64) -> i32;
    fn cuMemcpyDtoH_v2(dstHost: *mut c_void, srcDevice: u64, byteCount: usize) -> i32;
    fn cuMemcpyHtoD_v2(dstDevice: u64, srcHost: *const c_void, byteCount: usize) -> i32;
    fn cuMemcpyDtoD_v2(dstDevice: u64, srcDevice: u64, byteCount: usize) -> i32;
}

pub fn create_stream() -> anyhow::Result<u64> {
    let mut stream: u64 = 0;
    // SAFETY: `&mut stream` is a valid, aligned writable `u64` out-pointer and
    // flag `1` (`CU_STREAM_NON_BLOCKING`) is a valid CUDA stream creation flag.
    let status = unsafe { cuStreamCreate(&mut stream, 1) }; // CU_STREAM_NON_BLOCKING
    if status != 0 {
        anyhow::bail!("cuStreamCreate failed: status {status}");
    }
    Ok(stream)
}

pub fn create_event() -> anyhow::Result<u64> {
    let mut event: u64 = 0;
    // SAFETY: `&mut event` is a valid, aligned writable `u64` out-pointer and
    // flag `0x02` (`CU_EVENT_DISABLE_TIMING`) is a valid CUDA event creation flag.
    let status = unsafe { cuEventCreate(&mut event, 0x02) }; // CU_EVENT_DISABLE_TIMING
    if status != 0 {
        anyhow::bail!("cuEventCreate failed: status {status}");
    }
    Ok(event)
}

/// Record CUDA event `event` on CUDA stream `stream`.
///
/// # Safety
/// `event` must be a valid live `CUevent` handle and `stream` must be `0` or a
/// valid live `CUstream` handle in the active CUDA context.
pub unsafe fn record_event(event: u64, stream: u64) -> anyhow::Result<()> {
    // SAFETY: Caller guarantees `event` and `stream` are valid live CUDA handles.
    let status = unsafe { cuEventRecord(event, stream) };
    if status != 0 {
        anyhow::bail!("cuEventRecord failed: status {status}");
    }
    Ok(())
}

/// Enqueue a stream-side wait on `stream` for `event`.
///
/// # Safety
/// `stream` must be `0` or a valid live `CUstream` handle and `event` must be
/// a valid live `CUevent` handle in the active CUDA context.
pub unsafe fn stream_wait_event(stream: u64, event: u64) -> anyhow::Result<()> {
    // SAFETY: Caller guarantees `stream` and `event` are valid live CUDA handles.
    let status = unsafe { cuStreamWaitEvent(stream, event, 0) };
    if status != 0 {
        anyhow::bail!("cuStreamWaitEvent failed: status {status}");
    }
    Ok(())
}

/// Destroy a CUDA event handle created by [`create_event`].
///
/// Passing `0` is a no-op.
///
/// # Safety
/// If non-zero, `event` must be a valid owned `CUevent` handle that has not
/// already been destroyed and will not be used again after this call.
pub unsafe fn destroy_event(event: u64) {
    if event != 0 {
        // SAFETY: Caller guarantees `event` is a valid, unfreed owned `CUevent`.
        unsafe { cuEventDestroy_v2(event) };
    }
}

/// Destroy a CUDA stream handle created by [`create_stream`].
///
/// Passing `0` is a no-op.
///
/// # Safety
/// If non-zero, `stream` must be a valid owned `CUstream` handle that has not
/// already been destroyed and will not be used again after this call.
pub unsafe fn destroy_stream(stream: u64) {
    if stream != 0 {
        // SAFETY: Caller guarantees `stream` is a valid, unfreed owned `CUstream`.
        unsafe { cuStreamDestroy_v2(stream) };
    }
}

/// Synchronize CUDA stream `stream` (`0` for the legacy default stream).
///
/// # Safety
/// `stream` must be `0` or a valid live `CUstream` handle in the active CUDA
/// context.
pub unsafe fn sync_stream(stream: u64) -> anyhow::Result<()> {
    // SAFETY: Caller guarantees `stream` is `0` or a valid live `CUstream`.
    let status = unsafe { cuStreamSynchronize(stream) };
    if status != 0 {
        anyhow::bail!("cuStreamSynchronize failed: status {status}");
    }
    Ok(())
}

/// Copy `dst.len()` bytes from CUDA device pointer `src_dev` into host slice `dst`.
///
/// # Safety
/// `src_dev` must point to at least `dst.len()` valid readable device bytes in the
/// active CUDA context.
pub unsafe fn memcpy_d2h(dst: &mut [u8], src_dev: u64) -> anyhow::Result<()> {
    if dst.is_empty() {
        return Ok(());
    }
    // SAFETY: `dst` is a valid writable host slice of `dst.len()` bytes, and the
    // caller guarantees `src_dev` points to at least `dst.len()` readable device bytes.
    let status = unsafe { cuMemcpyDtoH_v2(dst.as_mut_ptr().cast::<c_void>(), src_dev, dst.len()) };
    if status != 0 {
        anyhow::bail!(
            "cuMemcpyDtoH_v2 failed: status {status} (bytes={})",
            dst.len()
        );
    }
    Ok(())
}

/// Copy `src.len()` bytes from host slice `src` into CUDA device pointer `dst_dev`.
///
/// Drains the legacy stream (`0`) after `cuMemcpyHtoD_v2` because CUDA treats
/// small pageable H2D copies (`<= 64 KiB`) as host-asynchronous on the legacy
/// stream, which does not synchronize with Atlas's non-blocking compute stream
/// or `ATLAS_BOUNDED_HOST_COPY=1` async D2H staging.
///
/// # Safety
/// `dst_dev` must point to at least `src.len()` valid writable device bytes in the
/// active CUDA context.
pub unsafe fn memcpy_h2d(dst_dev: u64, src: &[u8]) -> anyhow::Result<()> {
    if src.is_empty() {
        return Ok(());
    }
    // SAFETY: `src` is a valid readable host slice of `src.len()` bytes, and the
    // caller guarantees `dst_dev` points to at least `src.len()` writable device bytes.
    let status = unsafe { cuMemcpyHtoD_v2(dst_dev, src.as_ptr().cast::<c_void>(), src.len()) };
    if status != 0 {
        anyhow::bail!(
            "cuMemcpyHtoD_v2 failed: status {status} (bytes={})",
            src.len()
        );
    }
    // SAFETY: `0` is the legacy CUDA stream, valid in the active CUDA context.
    unsafe { sync_stream(0)? };
    Ok(())
}

/// Copy `bytes` bytes between CUDA device pointers `src_dev` and `dst_dev`.
///
/// Drains the legacy stream (`0`) so subsequent work on non-blocking streams
/// observes the completed copy.
///
/// # Safety
/// Both `src_dev` and `dst_dev` must point to at least `bytes` valid device bytes
/// in the active CUDA context.
pub unsafe fn memcpy_d2d(dst_dev: u64, src_dev: u64, bytes: usize) -> anyhow::Result<()> {
    if bytes == 0 || dst_dev == src_dev {
        return Ok(());
    }
    // SAFETY: Caller guarantees `dst_dev` and `src_dev` each span at least `bytes`
    // valid device bytes in the active CUDA context.
    let status = unsafe { cuMemcpyDtoD_v2(dst_dev, src_dev, bytes) };
    if status != 0 {
        anyhow::bail!("cuMemcpyDtoD_v2 failed: status {status} (bytes={bytes})");
    }
    // SAFETY: `0` is the legacy CUDA stream, valid in the active CUDA context.
    unsafe { sync_stream(0)? };
    Ok(())
}

/// Allocate GPU memory backed by a symmetric memory window across the
/// communicator. NCCL 2.28+ only — older NCCL returns `InvalidArgument`.
///
/// Buffers from `ncclMemAlloc` enable copy-engine collectives over NVLink
/// and the device-side communication API. On Atlas's 2-rank Spark over
/// RoCE, the copy-engine offload is unavailable (RoCE != NVLink), but the
/// symmetric windows are required to compose with device-API fused kernels
/// (TokenWeave-style AR+RMSNorm).
///
/// # Safety
/// The returned pointer must be freed via [`nccl_mem_free`]. Passing the
/// pointer to non-NCCL allocators (e.g. `cudaFree`) is undefined behavior.
pub unsafe fn nccl_mem_alloc(size: usize) -> anyhow::Result<*mut c_void> {
    let mut ptr: *mut c_void = std::ptr::null_mut();
    // SAFETY: `&mut ptr` is a valid aligned out-pointer; the caller assumes
    // responsibility for freeing the returned allocation via `nccl_mem_free`.
    let result = unsafe { ncclMemAlloc(&mut ptr, size) };
    check_nccl(result, "ncclMemAlloc")?;
    Ok(ptr)
}

/// Free a buffer previously returned by `nccl_mem_alloc`.
///
/// # Safety
/// `ptr` must have been returned by [`nccl_mem_alloc`] and not yet freed.
pub unsafe fn nccl_mem_free(ptr: *mut c_void) -> anyhow::Result<()> {
    // SAFETY: Caller guarantees `ptr` is a live allocation returned by `nccl_mem_alloc`
    // and is not freed more than once.
    let result = unsafe { ncclMemFree(ptr) };
    check_nccl(result, "ncclMemFree")
}

/// Query the linked `libnccl` runtime version code (`NCCL_VERSION(major, minor, patch)`).
///
/// # Errors
/// Returns an error if `ncclGetVersion` fails or reports a negative version code.
pub fn nccl_version() -> anyhow::Result<u32> {
    let mut version: i32 = 0;
    // SAFETY: `version` is a valid, aligned writable `i32` out-pointer.
    let result = unsafe { ncclGetVersion(&mut version) };
    check_nccl(result, "ncclGetVersion")?;
    u32::try_from(version)
        .map_err(|_| anyhow::anyhow!("ncclGetVersion returned negative version code: {version}"))
}

/// Convert NCCL result to anyhow::Result.
pub fn check_nccl(result: NcclResult, context: &str) -> anyhow::Result<()> {
    if result == NcclResult::Success {
        Ok(())
    } else {
        // SAFETY: `ncclGetErrorString` accepts any `c_int` status code and returns
        // either NULL or a pointer to a static NUL-terminated C string.
        let msg = unsafe {
            let ptr = ncclGetErrorString(result);
            if ptr.is_null() {
                format!("{result:?}")
            } else {
                std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        };
        anyhow::bail!("NCCL error in {context}: {msg} ({result:?})")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, offset_of, size_of};

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn test_nccl_config_layout_matches_nccl_v22800_abi() {
        assert_eq!(size_of::<NcclConfig>(), 96);
        assert_eq!(align_of::<NcclConfig>(), 8);

        assert_eq!(offset_of!(NcclConfig, size), 0);
        assert_eq!(offset_of!(NcclConfig, magic), 8);
        assert_eq!(offset_of!(NcclConfig, version), 12);
        assert_eq!(offset_of!(NcclConfig, blocking), 16);
        assert_eq!(offset_of!(NcclConfig, cga_cluster_size), 20);
        assert_eq!(offset_of!(NcclConfig, min_ctas), 24);
        assert_eq!(offset_of!(NcclConfig, max_ctas), 28);
        assert_eq!(offset_of!(NcclConfig, net_name), 32);
        assert_eq!(offset_of!(NcclConfig, split_share), 40);
        assert_eq!(offset_of!(NcclConfig, traffic_class), 44);
        assert_eq!(offset_of!(NcclConfig, comm_name), 48);
        assert_eq!(offset_of!(NcclConfig, collnet_enable), 56);
        assert_eq!(offset_of!(NcclConfig, cta_policy), 60);
        assert_eq!(offset_of!(NcclConfig, shrink_share), 64);
        assert_eq!(offset_of!(NcclConfig, nvls_ctas), 68);
        assert_eq!(offset_of!(NcclConfig, n_channels_per_net_peer), 72);
        assert_eq!(offset_of!(NcclConfig, nvlink_centric_sched), 76);
        assert_eq!(offset_of!(NcclConfig, graph_usage_mode), 80);
        assert_eq!(offset_of!(NcclConfig, num_rma_ctx), 84);
        assert_eq!(offset_of!(NcclConfig, max_p2p_peers), 88);
    }

    #[test]
    fn test_nccl_config_non_blocking_sentinels_and_version_clamping() {
        for (input_ver, expected_ver) in [
            (22_501, 22_501),
            (22_602, 22_602),
            (22_703, 22_703),
            (22_809, 22_809),
            (22_907, 22_907),
            (23_004, 23_004),
            (23_100, NCCL_CONFIG_MAX_KNOWN_VERSION),
            (0, FALLBACK_NCCL_VERSION),
        ] {
            let cfg = NcclConfig::non_blocking_for_version(input_ver);
            assert_eq!(cfg.size, size_of::<NcclConfig>());
            assert_eq!(cfg.magic, NCCL_API_MAGIC);
            assert_eq!(cfg.version, expected_ver, "input_ver={input_ver}");
            assert_eq!(cfg.blocking, 0);
            assert_eq!(cfg.cga_cluster_size, NCCL_CONFIG_UNDEF_INT);
            assert_eq!(cfg.min_ctas, NCCL_CONFIG_UNDEF_INT);
            assert_eq!(cfg.max_ctas, NCCL_CONFIG_UNDEF_INT);
            assert!(cfg.net_name.is_null());
            assert_eq!(cfg.split_share, NCCL_CONFIG_UNDEF_INT);
            assert_eq!(cfg.traffic_class, NCCL_CONFIG_UNDEF_INT);
            assert!(cfg.comm_name.is_null());
            assert_eq!(cfg.collnet_enable, NCCL_CONFIG_UNDEF_INT);
            assert_eq!(cfg.cta_policy, NCCL_CONFIG_UNDEF_INT);
            assert_eq!(cfg.shrink_share, NCCL_CONFIG_UNDEF_INT);
            assert_eq!(cfg.nvls_ctas, NCCL_CONFIG_UNDEF_INT);
            assert_eq!(cfg.n_channels_per_net_peer, NCCL_CONFIG_UNDEF_INT);
            assert_eq!(cfg.nvlink_centric_sched, NCCL_CONFIG_UNDEF_INT);
            assert_eq!(cfg.graph_usage_mode, NCCL_CONFIG_UNDEF_INT);
            assert_eq!(cfg.num_rma_ctx, NCCL_CONFIG_UNDEF_INT);
            assert_eq!(cfg.max_p2p_peers, NCCL_CONFIG_UNDEF_INT);
        }
    }

    #[test]
    fn test_nccl_version_query_and_dynamic_config() {
        let ver = nccl_version().expect("linked libnccl must report a valid version");
        assert!(
            ver >= 21_400,
            "expected linked NCCL >= 2.14.0 (21400), got {ver}"
        );
        let cfg = NcclConfig::non_blocking();
        assert_eq!(cfg.magic, NCCL_API_MAGIC);
        assert_eq!(cfg.size, size_of::<NcclConfig>());
        assert_eq!(cfg.version, ver.min(NCCL_CONFIG_MAX_KNOWN_VERSION));
        assert_eq!(cfg.blocking, 0);
    }

    #[test]
    fn test_nccl_result_transparent_c_int_and_unknown_status_codes() {
        assert_eq!(size_of::<NcclResult>(), size_of::<std::ffi::c_int>());
        assert_eq!(align_of::<NcclResult>(), align_of::<std::ffi::c_int>());
        assert_eq!(NcclResult::Success.raw(), 0);
        assert_eq!(NcclResult::from_raw(7), NcclResult::InProgress);
        assert_eq!(format!("{:?}", NcclResult::Success), "Success");
        assert_eq!(format!("{:?}", NcclResult::from_raw(99)), "Unknown(99)");
        check_nccl(NcclResult::Success, "unit").expect("Success must map to Ok");
    }
}
