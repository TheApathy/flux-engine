// SPDX-License-Identifier: AGPL-3.0-only

//! Default-unrouted dynamic boundary for the native FlashInfer SM121 C ABI.
//!
//! Production call contract:
//! 1. Explicitly open an absolute, trusted `libatlas_fi_fp4_sm121.so` path.
//! 2. Construct a validated [`FlashInferSm121Shape`] and call [`FlashInferSm121::prepare`]
//!    before CUDA graph capture. Preparation binds the backend's CUDA context,
//!    queries tactic-specific workspace bytes, enforces a caller-supplied cap,
//!    and allocates the workspace through that same backend/context.
//! 3. Keep [`PreparedFlashInferSm121`] alive and use exactly its frozen stream,
//!    shape, tactic, library handle, and workspace. Every launch rebinds the
//!    backend context before entering the CUDA-runtime C ABI.
//! 4. For graph capture, create [`FlashInferSm121GraphPin`] before capture. It
//!    freezes all device addresses and rejects the legacy default stream. Keep
//!    the pin alive until the graph is destroyed; replay and destruction go
//!    through the pin so the prepared workspace/library cannot be dropped early.
//! 5. Explicit close or `Drop` synchronizes the frozen stream before freeing the
//!    workspace. No preparation, tactic change, allocation, or unload is legal
//!    while a captured graph can still reference this operation.
//! 6. Callers that only have `&dyn GpuBackend` may instead use
//!    [`FlashInferSm121::prepare_borrowed_zero_workspace`]. That eager-only seam
//!    binds the borrowed backend, requires the ABI query to return exactly zero,
//!    and never allocates or frees. It is deliberately not graph-capable; graph
//!    integration must use the `Arc`-owned preparation/pin lifetime above.
//!
//! Production projection paths (`dense_ffn`, `qwen3_ssm`, and
//! `qwen3_attention`) construct [`FlashInferSm121`] via
//! [`FlashInferSm121::open_qualified`] when enabled by their route and
//! qualification policies.

use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle};

use super::flashinfer_sm121_artifact::{
    MAX_LIBRARY_BYTES, QUALIFIED_FLASHINFER_SM121_SHA256, hex_sha256, sha256, verify_source,
};

const RTLD_NOW: c_int = 2;
const TACTIC_COUNT: u8 = 6;
const MFD_CLOEXEC: c_uint = 0x0001;
const MFD_ALLOW_SEALING: c_uint = 0x0002;
const F_ADD_SEALS: c_int = 1_033;
const F_GET_SEALS: c_int = 1_034;
const F_SEAL_SEAL: c_int = 0x0001;
const F_SEAL_SHRINK: c_int = 0x0002;
const F_SEAL_GROW: c_int = 0x0004;
const F_SEAL_WRITE: c_int = 0x0008;
const REQUIRED_MEMFD_SEALS: c_int = F_SEAL_SEAL | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE;

type WorkspaceFn = unsafe extern "C" fn(c_int, c_int, c_int, c_int, c_int, *mut usize) -> c_int;
type GemmFn = unsafe extern "C" fn(
    c_int,
    *mut c_void,
    *const c_void,
    *const c_void,
    *const c_void,
    *const c_void,
    *const f32,
    c_int,
    c_int,
    c_int,
    c_int,
    *mut c_void,
    usize,
    *mut c_void,
) -> c_int;
type LastErrorFn = unsafe extern "C" fn() -> *const c_char;

// SAFETY: Declarations match the POSIX/Linux libdl and libc C ABIs on 64-bit Linux.
#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
    fn dlerror() -> *const c_char;
    fn memfd_create(name: *const c_char, flags: c_uint) -> c_int;
    fn fcntl(fd: c_int, operation: c_int, ...) -> c_int;
}

struct RawApi {
    handle: *mut c_void,
    close_handle: bool,
    workspace: WorkspaceFn,
    gemm: GemmFn,
    last_error: LastErrorFn,
    path: PathBuf,
    // Keep the verified, write/grow/shrink-sealed anonymous copy alive for
    // exactly as long as any resolved symbol can be called. `dlopen` is
    // performed through this descriptor, never through the mutable source
    // pathname or inode.
    _sealed_library: Option<File>,
    sha256: [u8; 32],
    device: u64,
    inode: u64,
}

// SAFETY: The loaded library handle and function pointers are immutable after
// construction, its C wrapper constructs a fresh CUTLASS runner per call, and
// diagnostic storage is thread-local, so sending RawApi across threads is sound.
unsafe impl Send for RawApi {}
// SAFETY: Shared references only invoke thread-safe FFI entry points whose
// diagnostic buffer is thread-local and whose runner state is stack-local.
unsafe impl Sync for RawApi {}

impl Drop for RawApi {
    fn drop(&mut self) {
        if self.close_handle && !self.handle.is_null() {
            // SAFETY: RawApi holds exclusive ownership of this non-null dlopen
            // handle and Drop runs once after the final Arc<RawApi> is dropped.
            let status = unsafe { dlclose(self.handle) };
            if status != 0 {
                tracing::warn!(
                    path = %self.path.display(),
                    status,
                    "dlclose failed for FlashInfer SM121 C ABI"
                );
            }
        }
    }
}

/// Explicit handle to the default-unrouted native FlashInfer SM121 library.
#[derive(Clone)]
pub struct FlashInferSm121 {
    api: Arc<RawApi>,
}

impl FlashInferSm121 {
    /// Load the single artifact qualified for production SM121 projections.
    pub fn open_qualified(path: &Path) -> Result<Self> {
        Self::open_with_sha256(path, QUALIFIED_FLASHINFER_SM121_SHA256)
    }

    /// Load the exact three-symbol ABI from an absolute regular-file path and
    /// require its bytes to match a caller-frozen SHA-256 identity.
    ///
    /// The source file is opened and hashed, then its verified bytes are copied
    /// into an anonymous memfd with WRITE/GROW/SHRINK/SEAL all applied before
    /// `dlopen(/proc/self/fd/N)`. The sealed descriptor is retained. Path
    /// replacement and post-admission mutation of the source inode therefore
    /// cannot change any mapped or not-yet-faulted library page.
    pub fn open_with_sha256(path: &Path, expected_sha256: [u8; 32]) -> Result<Self> {
        ensure!(
            usize::BITS == 64,
            "FlashInfer SM121 ABI requires a 64-bit host"
        );
        let (canonical, library_bytes, sha256) =
            verify_source(path, expected_sha256).map_err(anyhow::Error::msg)?;

        let mut sealed_library = sealed_memfd(&library_bytes)?;
        let sealed_sha256 = sha256_file(&mut sealed_library)?;
        ensure!(
            sealed_sha256 == sha256,
            "sealed FlashInfer SM121 copy does not match admitted source SHA-256"
        );
        let sealed_metadata = sealed_library.metadata()?;
        let descriptor_path = format!("/proc/self/fd/{}", sealed_library.as_raw_fd());
        let encoded = CString::new(descriptor_path.as_bytes())
            .context("FlashInfer SM121 descriptor path contains a NUL byte")?;
        clear_dlerror();
        // SAFETY: `encoded` is a valid NUL-terminated path to the sealed memfd
        // descriptor and `RTLD_NOW` is a valid dlopen flag.
        let handle = unsafe { dlopen(encoded.as_ptr(), RTLD_NOW) };
        ensure!(
            !handle.is_null(),
            "dlopen admitted descriptor for {} failed: {}",
            canonical.display(),
            current_dlerror()
        );

        let loaded = (|| -> Result<(WorkspaceFn, GemmFn, LastErrorFn)> {
            let workspace_ptr = resolve_symbol(handle, b"atlas_fi_nvfp4_sm121_workspace_size\0")?;
            let gemm_ptr = resolve_symbol(handle, b"atlas_fi_nvfp4_sm121_bf16\0")?;
            let last_error_ptr = resolve_symbol(handle, b"atlas_fi_nvfp4_sm121_last_error\0")?;
            // SAFETY: The shared object bytes matched `expected_sha256` before
            // loading, pinning the exact `WorkspaceFn` C ABI signature.
            let workspace =
                unsafe { std::mem::transmute::<*mut c_void, WorkspaceFn>(workspace_ptr) };
            // SAFETY: The SHA-256-verified library pins `GemmFn`'s exact C ABI signature.
            let gemm = unsafe { std::mem::transmute::<*mut c_void, GemmFn>(gemm_ptr) };
            // SAFETY: The SHA-256-verified library pins `LastErrorFn`'s exact C ABI signature.
            let last_error =
                unsafe { std::mem::transmute::<*mut c_void, LastErrorFn>(last_error_ptr) };
            Ok((workspace, gemm, last_error))
        })();

        let (workspace, gemm, last_error) = match loaded {
            Ok(symbols) => symbols,
            Err(error) => {
                // SAFETY: `handle` is a non-null dlopen handle exclusively owned
                // by this function on the error path and has not been closed yet.
                let _ = unsafe { dlclose(handle) };
                return Err(error).with_context(|| {
                    format!("load FlashInfer SM121 ABI from {}", canonical.display())
                });
            }
        };

        Ok(Self {
            api: Arc::new(RawApi {
                handle,
                close_handle: true,
                workspace,
                gemm,
                last_error,
                path: canonical,
                _sealed_library: Some(sealed_library),
                sha256,
                device: sealed_metadata.dev(),
                inode: sealed_metadata.ino(),
            }),
        })
    }

    /// Bind the backend context, query workspace, and allocate it before capture.
    pub fn prepare(
        &self,
        gpu: Arc<dyn GpuBackend>,
        shape: FlashInferSm121Shape,
        stream: u64,
        max_workspace_bytes: usize,
    ) -> Result<PreparedFlashInferSm121> {
        ensure!(
            max_workspace_bytes > 0,
            "FlashInfer workspace cap must be non-zero"
        );
        gpu.bind_to_thread()
            .context("bind Flux CUDA context before FlashInfer workspace query")?;
        let mut workspace_bytes = 0usize;
        // SAFETY: The CUDA context is bound on this thread, `shape` fields are
        // validated `c_int` dimensions, and `&mut workspace_bytes` is a valid
        // aligned output pointer for the loaded library.
        let status = unsafe {
            (self.api.workspace)(
                c_int::from(shape.tactic),
                shape.m,
                shape.n,
                shape.k,
                shape.batch_count,
                &mut workspace_bytes,
            )
        };
        self.check_status("workspace_size", status)?;
        ensure!(
            workspace_bytes <= max_workspace_bytes,
            "FlashInfer workspace request {workspace_bytes} exceeds cap {max_workspace_bytes}"
        );
        let workspace = if workspace_bytes == 0 {
            DevicePtr::NULL
        } else {
            gpu.alloc(workspace_bytes).with_context(|| {
                format!("allocate FlashInfer workspace ({workspace_bytes} bytes)")
            })?
        };
        Ok(PreparedFlashInferSm121 {
            library: self.clone(),
            gpu,
            shape,
            stream,
            workspace,
            workspace_bytes,
        })
    }

    /// Prepare the eager-only borrowed-backend seam for a proven zero-workspace tactic.
    ///
    /// This is the minimally invasive route for model code whose forward context
    /// exposes only `&dyn GpuBackend`. A non-zero result is rejected rather than
    /// allocated, and this object intentionally provides no graph capture API.
    pub fn prepare_borrowed_zero_workspace<'gpu>(
        &self,
        gpu: &'gpu dyn GpuBackend,
        shape: FlashInferSm121Shape,
        stream: u64,
    ) -> Result<BorrowedZeroWorkspaceFlashInferSm121<'gpu>> {
        gpu.bind_to_thread()
            .context("bind Flux CUDA context before FlashInfer zero-workspace query")?;
        let mut workspace_bytes = usize::MAX;
        // SAFETY: The CUDA context is bound on this thread, `shape` fields are
        // validated `c_int` dimensions, and `&mut workspace_bytes` is a valid
        // aligned output pointer for the loaded library.
        let status = unsafe {
            (self.api.workspace)(
                c_int::from(shape.tactic),
                shape.m,
                shape.n,
                shape.k,
                shape.batch_count,
                &mut workspace_bytes,
            )
        };
        self.check_status("workspace_size", status)?;
        ensure!(
            workspace_bytes == 0,
            "borrowed FlashInfer route requires zero workspace, ABI requested {workspace_bytes} bytes"
        );
        Ok(BorrowedZeroWorkspaceFlashInferSm121 {
            library: self.clone(),
            gpu,
            shape,
            stream,
        })
    }

    fn check_status(&self, operation: &str, status: c_int) -> Result<()> {
        if status == 0 {
            return Ok(());
        }
        // SAFETY: `self.api` keeps the loaded library alive and `last_error`
        // returns a thread-local diagnostic pointer with no preconditions.
        let pointer = unsafe { (self.api.last_error)() };
        let detail = if pointer.is_null() {
            "<null last_error>".to_owned()
        } else {
            // SAFETY: `pointer` was checked non-null and points to a
            // NUL-terminated thread-local diagnostic C string valid until the
            // next FFI call on this thread.
            unsafe { CStr::from_ptr(pointer) }
                .to_string_lossy()
                .into_owned()
        };
        bail!("FlashInfer {operation} failed with status {status}: {detail}")
    }

    /// Canonical path retained by the loaded library owner.
    pub fn path(&self) -> &Path {
        &self.api.path
    }

    /// Frozen content identity of the retained library descriptor.
    pub fn sha256_hex(&self) -> String {
        hex_sha256(self.api.sha256)
    }

    /// Stable filesystem identity captured before descriptor-based loading.
    pub fn file_identity(&self) -> (u64, u64) {
        (self.api.device, self.api.inode)
    }

    #[cfg(test)]
    fn from_test_api(workspace: WorkspaceFn, gemm: GemmFn, last_error: LastErrorFn) -> Self {
        Self {
            api: Arc::new(RawApi {
                handle: std::ptr::null_mut(),
                close_handle: false,
                workspace,
                gemm,
                last_error,
                path: PathBuf::from("<test-api>"),
                _sealed_library: None,
                sha256: [0; 32],
                device: 0,
                inode: 0,
            }),
        }
    }
}

/// Eager-only zero-workspace launch owner borrowing Flux's CUDA backend.
///
/// It retains the dynamic library, tactic and stream, but cannot outlive the
/// backend reference and exposes no graph-capture operation. A mutable launch
/// borrow keeps one object from being launched concurrently.
pub struct BorrowedZeroWorkspaceFlashInferSm121<'gpu> {
    library: FlashInferSm121,
    gpu: &'gpu dyn GpuBackend,
    shape: FlashInferSm121Shape,
    stream: u64,
}

impl BorrowedZeroWorkspaceFlashInferSm121<'_> {
    pub fn stream(&self) -> u64 {
        self.stream
    }

    pub fn shape(&self) -> FlashInferSm121Shape {
        self.shape
    }

    /// Launch with a null workspace on the exact stream used for preparation.
    pub fn launch_eager(&mut self, buffers: FlashInferSm121Buffers, stream: u64) -> Result<()> {
        ensure!(
            stream == self.stream,
            "FlashInfer stream mismatch: prepared {:#x}, launch {stream:#x}",
            self.stream
        );
        buffers.validate()?;
        self.gpu
            .bind_to_thread()
            .context("bind Flux CUDA context before FlashInfer borrowed launch")?;
        // SAFETY: The CUDA context is bound on this thread, `buffers` passed
        // non-null/alignment validation, `prepare_borrowed_zero_workspace`
        // verified zero workspace bytes for `self.shape`, `self.library` keeps
        // the library loaded, and `&mut self` serializes launches.
        let status = unsafe {
            (self.library.api.gemm)(
                c_int::from(self.shape.tactic),
                device_mut(buffers.output_bf16),
                device_const(buffers.activation_fp4),
                device_const(buffers.weight_fp4),
                device_const(buffers.activation_scales),
                device_const(buffers.weight_scales),
                device_const(buffers.global_scale_f32).cast(),
                self.shape.m,
                self.shape.n,
                self.shape.k,
                self.shape.batch_count,
                std::ptr::null_mut(),
                0,
                self.stream as usize as *mut c_void,
            )
        };
        self.library.check_status("bf16", status)
    }
}

/// Validated, frozen GEMM geometry and CUTLASS tactic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlashInferSm121Shape {
    tactic: u8,
    m: c_int,
    n: c_int,
    k: c_int,
    batch_count: c_int,
}

impl FlashInferSm121Shape {
    pub fn new(tactic: u8, m: usize, n: usize, k: usize, batch_count: usize) -> Result<Self> {
        ensure!(tactic < TACTIC_COUNT, "FlashInfer tactic must be in 0..6");
        ensure!(m > 0 && n > 0 && k > 0, "FlashInfer M/N/K must be non-zero");
        ensure!(batch_count > 0, "FlashInfer batch_count must be non-zero");
        ensure!(
            k.is_multiple_of(32),
            "FlashInfer K must be a multiple of 32"
        );
        Ok(Self {
            tactic,
            m: c_int::try_from(m).context("FlashInfer M exceeds C ABI i32")?,
            n: c_int::try_from(n).context("FlashInfer N exceeds C ABI i32")?,
            k: c_int::try_from(k).context("FlashInfer K exceeds C ABI i32")?,
            batch_count: c_int::try_from(batch_count)
                .context("FlashInfer batch_count exceeds C ABI i32")?,
        })
    }

    pub fn tactic(self) -> u8 {
        self.tactic
    }
}

/// Fixed device addresses consumed by one prepared GEMM.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlashInferSm121Buffers {
    pub output_bf16: DevicePtr,
    pub activation_fp4: DevicePtr,
    pub weight_fp4: DevicePtr,
    pub activation_scales: DevicePtr,
    pub weight_scales: DevicePtr,
    pub global_scale_f32: DevicePtr,
}

impl FlashInferSm121Buffers {
    fn validate(self) -> Result<()> {
        let aligned_16 = [
            ("output_bf16", self.output_bf16),
            ("activation_fp4", self.activation_fp4),
            ("weight_fp4", self.weight_fp4),
            ("activation_scales", self.activation_scales),
            ("weight_scales", self.weight_scales),
        ];
        for (name, pointer) in aligned_16 {
            ensure!(!pointer.is_null(), "FlashInfer {name} pointer is null");
            ensure!(
                pointer.0 % 16 == 0,
                "FlashInfer {name} pointer is not 16-byte aligned"
            );
        }
        ensure!(
            !self.global_scale_f32.is_null(),
            "FlashInfer global_scale_f32 pointer is null"
        );
        ensure!(
            self.global_scale_f32.0.is_multiple_of(4),
            "FlashInfer global_scale_f32 pointer is not 4-byte aligned"
        );
        Ok(())
    }
}

/// Owns the tactic-specific workspace and freezes its context/stream lifetime.
pub struct PreparedFlashInferSm121 {
    library: FlashInferSm121,
    gpu: Arc<dyn GpuBackend>,
    shape: FlashInferSm121Shape,
    stream: u64,
    workspace: DevicePtr,
    workspace_bytes: usize,
}

impl PreparedFlashInferSm121 {
    pub fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }

    pub fn stream(&self) -> u64 {
        self.stream
    }

    pub fn shape(&self) -> FlashInferSm121Shape {
        self.shape
    }

    /// Eager launch. The mutable borrow prevents concurrent workspace reuse.
    pub fn launch_eager(&mut self, buffers: FlashInferSm121Buffers, stream: u64) -> Result<()> {
        ensure!(
            stream == self.stream,
            "FlashInfer stream mismatch: prepared {:#x}, launch {stream:#x}",
            self.stream
        );
        self.launch_fixed(buffers)
    }

    /// Freeze device addresses for capture. Must be created before capture starts.
    pub fn pin_for_graph(
        &mut self,
        buffers: FlashInferSm121Buffers,
    ) -> Result<FlashInferSm121GraphPin<'_>> {
        ensure!(
            self.stream != 0,
            "FlashInfer graph capture rejects the legacy default stream"
        );
        buffers.validate()?;
        Ok(FlashInferSm121GraphPin {
            prepared: self,
            buffers,
        })
    }

    /// Synchronize and release explicitly. `Drop` performs the same operation.
    pub fn close(mut self) -> Result<()> {
        self.release_workspace()
    }

    fn launch_fixed(&mut self, buffers: FlashInferSm121Buffers) -> Result<()> {
        buffers.validate()?;
        self.gpu
            .bind_to_thread()
            .context("bind Flux CUDA context before FlashInfer launch")?;
        // SAFETY: The CUDA context is bound on this thread, `buffers` passed
        // non-null/alignment validation, `self.workspace` holds
        // `self.workspace_bytes` allocated on this backend, `self.library` keeps
        // the library loaded, and `&mut self` prevents concurrent workspace use.
        let status = unsafe {
            (self.library.api.gemm)(
                c_int::from(self.shape.tactic),
                device_mut(buffers.output_bf16),
                device_const(buffers.activation_fp4),
                device_const(buffers.weight_fp4),
                device_const(buffers.activation_scales),
                device_const(buffers.weight_scales),
                device_const(buffers.global_scale_f32).cast(),
                self.shape.m,
                self.shape.n,
                self.shape.k,
                self.shape.batch_count,
                device_mut(self.workspace),
                self.workspace_bytes,
                self.stream as usize as *mut c_void,
            )
        };
        self.library.check_status("bf16", status)
    }

    fn release_workspace(&mut self) -> Result<()> {
        if self.workspace.is_null() {
            return Ok(());
        }
        self.gpu
            .bind_to_thread()
            .context("bind Flux CUDA context before FlashInfer workspace release")?;
        self.gpu
            .synchronize(self.stream)
            .context("synchronize FlashInfer stream before workspace release")?;
        let pointer = std::mem::replace(&mut self.workspace, DevicePtr::NULL);
        if let Err(error) = self.gpu.free(pointer) {
            self.workspace = pointer;
            return Err(error).context("free FlashInfer workspace");
        }
        Ok(())
    }
}

impl Drop for PreparedFlashInferSm121 {
    fn drop(&mut self) {
        if let Err(error) = self.release_workspace() {
            tracing::warn!(%error, "failed to release FlashInfer SM121 workspace");
        }
    }
}

/// Borrowed graph-lifetime token retaining fixed addresses, workspace and SO.
pub struct FlashInferSm121GraphPin<'a> {
    prepared: &'a mut PreparedFlashInferSm121,
    buffers: FlashInferSm121Buffers,
}

impl FlashInferSm121GraphPin<'_> {
    /// Record the fixed-address GEMM while the caller's stream is capturing.
    pub fn capture_launch(&mut self) -> Result<()> {
        self.prepared.launch_fixed(self.buffers)
    }

    /// Replay on the same non-default stream used during preparation/capture.
    pub fn replay(&self, graph: GraphHandle) -> Result<()> {
        self.prepared
            .gpu
            .bind_to_thread()
            .context("bind Flux CUDA context before FlashInfer graph replay")?;
        self.prepared.gpu.launch_graph(graph, self.prepared.stream)
    }

    /// Synchronize, destroy the graph, and consume the pin before workspace drop.
    pub fn destroy(self, graph: GraphHandle) -> Result<()> {
        self.prepared
            .gpu
            .bind_to_thread()
            .context("bind Flux CUDA context before FlashInfer graph destroy")?;
        self.prepared
            .gpu
            .synchronize(self.prepared.stream)
            .context("synchronize FlashInfer graph replay stream before destroy")?;
        self.prepared.gpu.destroy_graph(graph)
    }
}

fn device_mut(pointer: DevicePtr) -> *mut c_void {
    pointer.0 as usize as *mut c_void
}

fn device_const(pointer: DevicePtr) -> *const c_void {
    pointer.0 as usize as *const c_void
}

fn sealed_memfd(bytes: &[u8]) -> Result<File> {
    let name = CString::new("flux-fi-fp4-sm121")?;
    // SAFETY: `name` is a valid NUL-terminated CString and `MFD_CLOEXEC |
    // MFD_ALLOW_SEALING` are valid Linux `memfd_create` flags.
    let descriptor = unsafe { memfd_create(name.as_ptr(), MFD_CLOEXEC | MFD_ALLOW_SEALING) };
    ensure!(
        descriptor >= 0,
        "memfd_create for FlashInfer SM121 failed: {}",
        std::io::Error::last_os_error()
    );
    // SAFETY: memfd_create returned a fresh owned descriptor, transferred
    // exactly once into File for RAII close on all following error paths.
    let mut file = unsafe { File::from_raw_fd(descriptor) };
    file.write_all(bytes)
        .context("write verified FlashInfer SM121 bytes into memfd")?;
    file.flush()
        .context("flush verified FlashInfer SM121 memfd")?;
    file.seek(SeekFrom::Start(0))?;
    // SAFETY: `file.as_raw_fd()` is an open memfd created with
    // `MFD_ALLOW_SEALING` and `REQUIRED_MEMFD_SEALS` is a valid `c_int` mask.
    let status = unsafe { fcntl(file.as_raw_fd(), F_ADD_SEALS, REQUIRED_MEMFD_SEALS) };
    ensure!(
        status == 0,
        "seal FlashInfer SM121 memfd failed: {}",
        std::io::Error::last_os_error()
    );
    // SAFETY: `file.as_raw_fd()` is an open memfd descriptor and `F_GET_SEALS`
    // takes no variadic arguments.
    let seals = unsafe { fcntl(file.as_raw_fd(), F_GET_SEALS) };
    ensure!(
        seals >= 0 && seals & REQUIRED_MEMFD_SEALS == REQUIRED_MEMFD_SEALS,
        "FlashInfer SM121 memfd is missing required seals: expected {REQUIRED_MEMFD_SEALS:#x}, got {seals:#x}"
    );
    Ok(file)
}

fn sha256_file(file: &mut File) -> Result<[u8; 32]> {
    file.seek(SeekFrom::Start(0))?;
    let mut bytes = Vec::new();
    file.take(MAX_LIBRARY_BYTES + 1).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= MAX_LIBRARY_BYTES,
        "FlashInfer SM121 library exceeds {MAX_LIBRARY_BYTES} bytes while hashing"
    );
    Ok(sha256(&bytes))
}

fn clear_dlerror() {
    // SAFETY: `dlerror` takes no arguments and clears thread-local libdl error state.
    let _ = unsafe { dlerror() };
}

fn current_dlerror() -> String {
    // SAFETY: `dlerror` takes no arguments and returns either null or a
    // thread-local NUL-terminated diagnostic string.
    let pointer = unsafe { dlerror() };
    if pointer.is_null() {
        "<no dlerror>".to_owned()
    } else {
        // SAFETY: `pointer` was checked non-null and remains valid on this
        // thread until the next libdl call; `to_string_lossy` copies it now.
        unsafe { CStr::from_ptr(pointer) }
            .to_string_lossy()
            .into_owned()
    }
}

fn resolve_symbol(handle: *mut c_void, symbol: &'static [u8]) -> Result<*mut c_void> {
    clear_dlerror();
    // SAFETY: `handle` is a live dlopen handle and every call site passes a
    // NUL-terminated static C symbol byte slice.
    let pointer = unsafe { dlsym(handle, symbol.as_ptr().cast()) };
    let error = current_dlerror();
    ensure!(
        !pointer.is_null() && error == "<no dlerror>",
        "dlsym({}) failed: {error}",
        String::from_utf8_lossy(&symbol[..symbol.len() - 1])
    );
    Ok(pointer)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicI32, AtomicU64, AtomicUsize, Ordering};
    use std::sync::{LazyLock, Mutex};

    use spark_runtime::gpu::mock::MockGpuBackend;

    use super::*;

    static TEST_LOCK: Mutex<()> = Mutex::new(());
    static WORKSPACE_STATUS: AtomicI32 = AtomicI32::new(0);
    static GEMM_STATUS: AtomicI32 = AtomicI32::new(0);
    static WORKSPACE_BYTES: AtomicUsize = AtomicUsize::new(0);
    static GEMM_CALLS: AtomicUsize = AtomicUsize::new(0);
    static LAST_STREAM: AtomicU64 = AtomicU64::new(0);
    static LAST_WORKSPACE: AtomicU64 = AtomicU64::new(0);
    static LAST_OUTPUT: AtomicU64 = AtomicU64::new(0);
    static ERROR_TEXT: &[u8] = b"forced C ABI failure\0";
    static VALID_BUFFERS: LazyLock<FlashInferSm121Buffers> =
        LazyLock::new(|| FlashInferSm121Buffers {
            output_bf16: DevicePtr(0x2000),
            activation_fp4: DevicePtr(0x3000),
            weight_fp4: DevicePtr(0x4000),
            activation_scales: DevicePtr(0x5000),
            weight_scales: DevicePtr(0x6000),
            global_scale_f32: DevicePtr(0x7000),
        });

    /// Test stub matching `WorkspaceFn`.
    ///
    /// # Safety
    ///
    /// If `output` is non-null, it must be valid for aligned writes of `usize`.
    unsafe extern "C" fn fake_workspace(
        _tactic: c_int,
        _m: c_int,
        _n: c_int,
        _k: c_int,
        _batch: c_int,
        output: *mut usize,
    ) -> c_int {
        if !output.is_null() {
            // SAFETY: Caller upholds `output` is valid and aligned for `usize` writes.
            unsafe { *output = WORKSPACE_BYTES.load(Ordering::SeqCst) };
        }
        WORKSPACE_STATUS.load(Ordering::SeqCst)
    }

    /// Test stub matching `GemmFn`.
    ///
    /// # Safety
    ///
    /// Does not dereference any pointer arguments; records raw address values in
    /// atomics and returns `GEMM_STATUS`.
    unsafe extern "C" fn fake_gemm(
        _tactic: c_int,
        output: *mut c_void,
        _activation: *const c_void,
        _weight: *const c_void,
        _activation_scales: *const c_void,
        _weight_scales: *const c_void,
        _global_scale: *const f32,
        _m: c_int,
        _n: c_int,
        _k: c_int,
        _batch: c_int,
        workspace: *mut c_void,
        _workspace_bytes: usize,
        stream: *mut c_void,
    ) -> c_int {
        GEMM_CALLS.fetch_add(1, Ordering::SeqCst);
        LAST_STREAM.store(stream as usize as u64, Ordering::SeqCst);
        LAST_WORKSPACE.store(workspace as usize as u64, Ordering::SeqCst);
        LAST_OUTPUT.store(output as usize as u64, Ordering::SeqCst);
        GEMM_STATUS.load(Ordering::SeqCst)
    }

    /// Test stub matching `LastErrorFn`.
    ///
    /// # Safety
    ///
    /// Returns a pointer to the static NUL-terminated `ERROR_TEXT` byte slice.
    unsafe extern "C" fn fake_last_error() -> *const c_char {
        ERROR_TEXT.as_ptr().cast()
    }

    fn library() -> FlashInferSm121 {
        FlashInferSm121::from_test_api(fake_workspace, fake_gemm, fake_last_error)
    }

    fn shape() -> FlashInferSm121Shape {
        FlashInferSm121Shape::new(3, 2_079, 17_408, 5_120, 1).unwrap()
    }

    fn reset(workspace: usize) {
        WORKSPACE_STATUS.store(0, Ordering::SeqCst);
        GEMM_STATUS.store(0, Ordering::SeqCst);
        WORKSPACE_BYTES.store(workspace, Ordering::SeqCst);
        GEMM_CALLS.store(0, Ordering::SeqCst);
        LAST_STREAM.store(0, Ordering::SeqCst);
        LAST_WORKSPACE.store(0, Ordering::SeqCst);
        LAST_OUTPUT.store(0, Ordering::SeqCst);
    }

    #[test]
    fn rejects_hostile_shapes_and_tactics() {
        assert!(FlashInferSm121Shape::new(6, 1, 1, 32, 1).is_err());
        assert!(FlashInferSm121Shape::new(0, 0, 1, 32, 1).is_err());
        assert!(FlashInferSm121Shape::new(0, 1, 1, 31, 1).is_err());
        assert!(FlashInferSm121Shape::new(0, 1, 1, 32, 0).is_err());
        assert!(FlashInferSm121Shape::new(0, i32::MAX as usize + 1, 1, 32, 1).is_err());
    }

    #[test]
    fn explicit_open_rejects_relative_and_missing_paths() {
        assert!(FlashInferSm121::open_with_sha256(Path::new("relative.so"), [0; 32]).is_err());
        assert!(
            FlashInferSm121::open_with_sha256(
                Path::new("/definitely/missing/atlas-fi.so"),
                [0; 32],
            )
            .is_err()
        );
    }

    #[test]
    fn sealed_memfd_is_byte_exact_and_rejects_writes() {
        let mut file = sealed_memfd(b"immutable FlashInfer bytes").unwrap();
        assert_eq!(
            sha256_file(&mut file).unwrap(),
            sha256(b"immutable FlashInfer bytes")
        );
        file.seek(SeekFrom::Start(0)).unwrap();
        assert!(file.write_all(b"mutation").is_err());
        assert_eq!(
            sha256_file(&mut file).unwrap(),
            sha256(b"immutable FlashInfer bytes")
        );
    }

    #[test]
    fn workspace_error_and_cap_fail_before_allocation() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset(4_096);
        let gpu = Arc::new(MockGpuBackend::new());
        WORKSPACE_STATUS.store(-2, Ordering::SeqCst);
        let error = library()
            .prepare(gpu.clone(), shape(), 0x88, 8_192)
            .err()
            .unwrap();
        assert!(error.to_string().contains("forced C ABI failure"));
        assert_eq!(gpu.alloc_count(), 0);

        WORKSPACE_STATUS.store(0, Ordering::SeqCst);
        let error = library()
            .prepare(gpu.clone(), shape(), 0x88, 1_024)
            .err()
            .unwrap();
        assert!(error.to_string().contains("exceeds cap"));
        assert_eq!(gpu.alloc_count(), 0);
    }

    #[test]
    fn zero_workspace_avoids_allocation() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset(0);
        let gpu = Arc::new(MockGpuBackend::new());
        let prepared = library().prepare(gpu.clone(), shape(), 0x88, 1).unwrap();
        assert_eq!(prepared.workspace_bytes(), 0);
        assert_eq!(gpu.alloc_count(), 0);
        prepared.close().unwrap();
    }

    #[test]
    fn borrowed_route_fail_closes_nonzero_workspace_and_never_allocates() {
        let _guard = TEST_LOCK.lock().unwrap();
        let gpu = MockGpuBackend::new();

        reset(1);
        let error = library()
            .prepare_borrowed_zero_workspace(&gpu, shape(), 0x88)
            .err()
            .unwrap();
        assert!(error.to_string().contains("requires zero workspace"));
        assert_eq!(gpu.alloc_count(), 0);

        reset(0);
        {
            let mut prepared = library()
                .prepare_borrowed_zero_workspace(&gpu, shape(), 0x88)
                .unwrap();
            assert_eq!(prepared.stream(), 0x88);
            assert_eq!(prepared.shape(), shape());
            assert!(prepared.launch_eager(*VALID_BUFFERS, 0x99).is_err());
            assert_eq!(GEMM_CALLS.load(Ordering::SeqCst), 0);
            prepared.launch_eager(*VALID_BUFFERS, 0x88).unwrap();
            assert_eq!(GEMM_CALLS.load(Ordering::SeqCst), 1);
            assert_eq!(LAST_WORKSPACE.load(Ordering::SeqCst), 0);
            assert_eq!(LAST_STREAM.load(Ordering::SeqCst), 0x88);
        }
        assert_eq!(gpu.alloc_count(), 0);
    }

    #[test]
    fn launch_freezes_stream_workspace_and_propagates_errors() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset(4_096);
        let gpu = Arc::new(MockGpuBackend::new());
        let mut prepared = library()
            .prepare(gpu.clone(), shape(), 0x88, 8_192)
            .unwrap();
        assert_eq!(gpu.alloc_count(), 1);
        assert!(prepared.launch_eager(*VALID_BUFFERS, 0x99).is_err());
        assert_eq!(GEMM_CALLS.load(Ordering::SeqCst), 0);

        let mut misaligned = *VALID_BUFFERS;
        misaligned.weight_fp4 = DevicePtr(0x4001);
        assert!(prepared.launch_eager(misaligned, 0x88).is_err());
        assert_eq!(GEMM_CALLS.load(Ordering::SeqCst), 0);

        prepared.launch_eager(*VALID_BUFFERS, 0x88).unwrap();
        assert_eq!(GEMM_CALLS.load(Ordering::SeqCst), 1);
        assert_eq!(LAST_STREAM.load(Ordering::SeqCst), 0x88);
        assert_ne!(LAST_WORKSPACE.load(Ordering::SeqCst), 0);
        assert_eq!(LAST_OUTPUT.load(Ordering::SeqCst), 0x2000);

        GEMM_STATUS.store(-2, Ordering::SeqCst);
        let error = prepared.launch_eager(*VALID_BUFFERS, 0x88).unwrap_err();
        assert!(error.to_string().contains("forced C ABI failure"));
        prepared.close().unwrap();
        assert_eq!(gpu.alloc_count(), 0);
    }

    #[test]
    fn graph_pin_rejects_default_stream_and_retains_fixed_addresses() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset(128);
        let gpu = Arc::new(MockGpuBackend::new());
        let mut default_stream = library().prepare(gpu.clone(), shape(), 0, 1_024).unwrap();
        assert!(default_stream.pin_for_graph(*VALID_BUFFERS).is_err());
        default_stream.close().unwrap();

        let mut prepared = library()
            .prepare(gpu.clone(), shape(), 0x77, 1_024)
            .unwrap();
        {
            let mut pin = prepared.pin_for_graph(*VALID_BUFFERS).unwrap();
            pin.capture_launch().unwrap();
            pin.capture_launch().unwrap();
            assert_eq!(GEMM_CALLS.load(Ordering::SeqCst), 2);
            assert_eq!(LAST_STREAM.load(Ordering::SeqCst), 0x77);
            assert_eq!(LAST_OUTPUT.load(Ordering::SeqCst), 0x2000);
            pin.replay(GraphHandle(0x1234)).unwrap();
            pin.destroy(GraphHandle(0x1234)).unwrap();
        }
        prepared.close().unwrap();
        assert_eq!(gpu.alloc_count(), 0);
    }
}
