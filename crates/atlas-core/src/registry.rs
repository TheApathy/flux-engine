// SPDX-License-Identifier: AGPL-3.0-only

//! Global kernel registry — load PTX once, cache modules/functions/streams.
//!
//! Eliminates ~0.06-0.26ms overhead per kernel call from:
//! - CudaContext::new (driver init)
//! - CudaContext::load_module (PTX JIT compilation)
//! - CudaContext::new_stream (stream creation)
//! - cuModuleGetFunction (function lookup) — now cached after first call
//!
//! Usage:
//!   let reg = AtlasRegistry::get_or_init(ordinal, &[("gemm", PTX_SRC), ...])?;
//!   let func = reg.function("gemm", "dense_gemm_tc_bf16")?;
//!   unsafe { reg.stream.launch_builder(&func).arg(&ptr).launch(cfg)?; }
//!   reg.stream.synchronize()?;

use std::collections::HashMap;
use std::ffi::{CString, c_void};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig};
use cudarc::nvrtc::Ptx;
use ring::digest::{Context as DigestContext, SHA256};

use crate::error::{AtlasError, Result};
pub use crate::target::{
    AotCubinEncoding, AotCubinKey, decode_obf_bytes, lookup_aot_cubin, register_aot_cubin,
    sha256_bytes,
};

// SAFETY: Raw CUDA Driver API symbols exported by `libcuda.so` using the
// standard C ABI; all callers validate handles and status codes.
unsafe extern "C" {
    fn cuCtxSetCurrent(ctx: *mut c_void) -> i32;
    fn cuModuleLoadData(module: *mut *mut c_void, image: *const c_void) -> i32;
    fn cuModuleUnload(hmod: *mut c_void) -> i32;
    fn cuModuleGetFunction(hfunc: *mut *mut c_void, hmod: *mut c_void, name: *const i8) -> i32;
    fn cuLaunchKernel(
        f: *mut c_void,
        gridDimX: u32,
        gridDimY: u32,
        gridDimZ: u32,
        blockDimX: u32,
        blockDimY: u32,
        blockDimZ: u32,
        sharedMemBytes: u32,
        hStream: *mut c_void,
        kernelParams: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> i32;
    fn cuFuncSetAttribute(hfunc: *mut c_void, attrib: i32, value: i32) -> i32;
    fn cuGetErrorName(error: i32, pStr: *mut *const i8) -> i32;
    fn cuGetErrorString(error: i32, pStr: *mut *const i8) -> i32;
    // Resolve a `__device__` symbol in a loaded CUmodule into a device pointer
    // + size in bytes. Used by drivers that need to read/write device globals
    // (e.g. InnerQ calibration state) without round-tripping through a kernel.
    fn cuModuleGetGlobal_v2(
        dptr: *mut u64,
        bytes: *mut usize,
        hmod: *mut c_void,
        name: *const i8,
    ) -> i32;
    fn cuMemcpyHtoD_v2(dst: u64, src: *const c_void, bytes: usize) -> i32;
    fn cuMemcpyDtoH_v2(dst: *mut c_void, src: u64, bytes: usize) -> i32;
    fn cuStreamSynchronize(stream: u64) -> i32;
}

/// Resolve a CUresult status code into `"<NAME>: <description>"` via
/// cuGetErrorName + cuGetErrorString. Returns "CUDA_UNKNOWN" / "(no message)"
/// if the driver doesn't recognize the code.
pub fn cuda_error_text(status: i32) -> String {
    use std::ffi::CStr;
    let mut name_ptr: *const i8 = std::ptr::null();
    let mut msg_ptr: *const i8 = std::ptr::null();
    // SAFETY: `cuGetErrorName` accepts any `CUresult` status and writes a static
    // NUL-terminated string pointer on success (`0`), which we check for non-null
    // before passing to `CStr::from_ptr`.
    let name = unsafe {
        if cuGetErrorName(status, &mut name_ptr) == 0 && !name_ptr.is_null() {
            CStr::from_ptr(name_ptr as *const std::os::raw::c_char)
                .to_string_lossy()
                .into_owned()
        } else {
            "CUDA_UNKNOWN".to_string()
        }
    };
    // SAFETY: `cuGetErrorString` accepts any `CUresult` status and writes a static
    // NUL-terminated string pointer on success (`0`), checked non-null before use.
    let msg = unsafe {
        if cuGetErrorString(status, &mut msg_ptr) == 0 && !msg_ptr.is_null() {
            CStr::from_ptr(msg_ptr as *const std::os::raw::c_char)
                .to_string_lossy()
                .into_owned()
        } else {
            "(no message)".to_string()
        }
    };
    format!("{name} ({status}): {msg}")
}

/// Wrapper for raw CUfunction handle (Send+Sync safe — handles are context-wide).
#[derive(Clone, Copy)]
pub struct RawCudaFunc(pub *mut c_void);
// SAFETY: CUfunction handles returned by `cuModuleGetFunction` remain valid
// for the lifetime of the owning CUcontext (the Atlas registry binds the
// process-wide context once at startup and never destroys it). The handle
// itself is opaque metadata — actual kernel launches go through cuLaunchKernel
// with caller-supplied stream synchronisation, so `Sync` does not imply
// concurrent execution, only concurrent reads of an immutable pointer.
unsafe impl Send for RawCudaFunc {}
unsafe impl Sync for RawCudaFunc {}

/// Every registry this process has loaded, keyed by its initialization
/// identity (ordinal + the exact ordered module/PTX set). Leaked: modules are
/// never unloaded, so every `RawCudaFunc` handed out stays valid for the
/// process lifetime.
///
/// Keyed rather than a single `OnceLock` because a hot model swap
/// (`model_swap`) can load a model with a DIFFERENT kernel target (for
/// example qwen3.5-27b -> glm5.3-flash/exl3). A single registry refused that
/// with "initialization identity mismatch", so every cross-target swap failed
/// and restored the previous model. The old rule is kept where it matters: a
/// request is only ever served by a registry with exactly its identity, never
/// by one that loaded a different module set.
static REGISTRIES: std::sync::Mutex<Vec<&'static AtlasRegistry>> =
    std::sync::Mutex::new(Vec::new());
/// The most recently initialized/selected registry, for `get()` callers that
/// have no backend handle. Backends keep their own registry reference.
static CURRENT: std::sync::RwLock<Option<&'static AtlasRegistry>> = std::sync::RwLock::new(None);

/// Context handle wrapper for binding an existing `CUcontext` on scoped worker
/// threads during parallel PTX module compilation.
#[derive(Clone, Copy)]
struct SendCuContext(*mut c_void);

impl SendCuContext {
    #[inline]
    const fn as_ptr(self) -> *mut c_void {
        self.0
    }
}

// SAFETY: CUDA contexts are shareable across host threads when each thread
// binds the context with `cuCtxSetCurrent`. The owning `Arc<CudaContext>`
// outlives the `std::thread::scope` that borrows this handle.
unsafe impl Send for SendCuContext {}
unsafe impl Sync for SendCuContext {}

/// Module handle wrapper for transferring a freshly loaded `CUmodule` from a
/// scoped worker thread back to the owning `AtlasRegistry::init` call.
struct SendCuModule(*mut c_void);

// SAFETY: A `CUmodule` handle belongs to the owning `CUcontext`, not to the OS
// thread that called `cuModuleLoadData`. Returning the opaque handle to the
// parent thread after `cuModuleLoadData` completes is thread-safe.
unsafe impl Send for SendCuModule {}

/// Parse `ATLAS_DISABLE_AOT_CUBIN` opt-out values.
fn parse_disable_aot_cubin(val: Option<&str>) -> bool {
    matches!(
        val.map(str::trim),
        Some("1" | "true" | "TRUE" | "yes" | "YES")
    )
}

fn aot_cubin_enabled() -> bool {
    !parse_disable_aot_cubin(std::env::var("ATLAS_DISABLE_AOT_CUBIN").ok().as_deref())
}

/// Cached CUDA modules and a persistent stream.
pub struct AtlasRegistry {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    initialization_identity: String,
    /// Original PTX sources kept so legacy `function()` callers can lazily
    /// materialize a cudarc `CudaModule` on demand without duplicating JIT
    /// compilation at startup.
    ptx_sources: HashMap<&'static str, Box<str>>,
    /// Lazily-populated cudarc `CudaModule` cache (empty until `function()` is called).
    modules: Mutex<HashMap<&'static str, Arc<CudaModule>>>,
    /// Raw CUmodule handles for direct cuLaunchKernel access.
    raw_modules: HashMap<&'static str, *mut c_void>,
}

// SAFETY: Same rationale as `RawCudaFunc`: the `raw_modules` map holds
// CUmodule handles obtained at startup from a single CUcontext. Both
// `raw_modules` and `ptx_sources` are populated once during registry init and
// read-only from that point on, while `modules` is protected by a `Mutex`.
// CUDA itself serializes kernel launches via the stream the caller supplies —
// this impl only asserts that the *handle metadata* is shareable across threads.
unsafe impl Send for AtlasRegistry {}
unsafe impl Sync for AtlasRegistry {}

fn initialization_identity(ordinal: usize, ptx_sources: &[(&str, &str)]) -> String {
    let mut digest = DigestContext::new(&SHA256);
    digest.update(b"atlas-cuda-registry-modules.v1");
    digest.update(&(ptx_sources.len() as u64).to_le_bytes());
    for &(name, ptx) in ptx_sources {
        digest.update(&(name.len() as u64).to_le_bytes());
        digest.update(name.as_bytes());
        digest.update(&(ptx.len() as u64).to_le_bytes());
        digest.update(ptx.as_bytes());
    }
    let sha256: String = digest
        .finish()
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    format!("backend=cuda;ordinal={ordinal};loaded_modules_sha256={sha256}")
}

/// The registry in `slots` whose identity is exactly `requested`, if any.
fn select_registry<'a, T>(slots: &'a [(&str, T)], requested: &str) -> Option<&'a T> {
    slots
        .iter()
        .find(|(identity, _)| *identity == requested)
        .map(|(_, r)| r)
}

#[cfg(test)]
fn require_initialization_identity(loaded: &str, requested: &str) -> Result<()> {
    if loaded != requested {
        return Err(AtlasError::ModuleLoad(format!(
            "AtlasRegistry initialization identity mismatch: loaded {loaded}, requested \
             {requested}"
        )));
    }
    Ok(())
}

#[cfg(test)]
fn admit_initialization_request<'a>(
    identity: &'a OnceLock<String>,
    requested: &str,
) -> Result<&'a str> {
    let loaded = identity.get_or_init(|| requested.to_owned());
    require_initialization_identity(loaded, requested)?;
    Ok(loaded)
}

impl AtlasRegistry {
    /// Get or initialize the global registry.
    ///
    /// First call loads all PTX modules and creates the persistent stream.
    /// Subsequent calls return the cached registry instantly.
    pub fn get_or_init(
        ordinal: usize,
        ptx_sources: &[(&'static str, &str)],
    ) -> Result<&'static Self> {
        let requested_identity = initialization_identity(ordinal, ptx_sources);
        let mut loaded = REGISTRIES.lock().unwrap_or_else(|p| p.into_inner());
        let slots: Vec<(&str, &'static AtlasRegistry)> = loaded
            .iter()
            .map(|r| (r.initialization_identity.as_str(), *r))
            .collect();
        let registry = match select_registry(&slots, &requested_identity) {
            Some(existing) => *existing,
            None => {
                let fresh: &'static AtlasRegistry = Box::leak(Box::new(Self::init(
                    ordinal,
                    ptx_sources,
                    requested_identity,
                )?));
                loaded.push(fresh);
                fresh
            }
        };
        *CURRENT.write().unwrap_or_else(|p| p.into_inner()) = Some(registry);
        Ok(registry)
    }

    fn init(
        ordinal: usize,
        ptx_sources: &[(&'static str, &str)],
        initialization_identity: String,
    ) -> Result<AtlasRegistry> {
        let init_start = Instant::now();
        let ctx = CudaContext::new(ordinal).map_err(AtlasError::CudaDriver)?;
        ctx.bind_to_thread().map_err(AtlasError::CudaDriver)?;
        let stream = ctx.new_stream().map_err(AtlasError::CudaDriver)?;

        let use_aot_cubin = aot_cubin_enabled();
        let mut raw_modules: HashMap<&'static str, *mut c_void> =
            HashMap::with_capacity(ptx_sources.len());
        let mut ptx_fallback_list: Vec<(&'static str, &str)> = Vec::new();
        let mut aot_cubin_count = 0usize;
        let mut aot_fallback_warned = false;

        // Step 1: Fast-load any matching AOT CUBINs directly on the main thread.
        for &(name, src) in ptx_sources {
            if use_aot_cubin && let Some(cubin) = lookup_aot_cubin(name, src) {
                let mut raw_mod: *mut c_void = std::ptr::null_mut();
                // SAFETY: `cubin` is a valid ELF CUBIN image buffer whose pointer stays
                // valid for the duration of `cuModuleLoadData`, and `ctx` is bound to
                // the current thread above.
                let status =
                    unsafe { cuModuleLoadData(&mut raw_mod, cubin.as_ptr().cast::<c_void>()) };
                if status == 0 {
                    if let Some(prev) = raw_modules.insert(name, raw_mod) {
                        // SAFETY: `prev` is a live CUmodule owned by `ctx` replaced by
                        // a duplicate module entry.
                        unsafe {
                            let _ = cuModuleUnload(prev);
                        }
                    }
                    aot_cubin_count += 1;
                    continue;
                }
                if !aot_fallback_warned {
                    eprintln!(
                        "ATLAS_CUDA_AOT_CUBIN_FALLBACK ordinal={ordinal} module={name} status={} — falling back to PTX JIT",
                        cuda_error_text(status)
                    );
                    aot_fallback_warned = true;
                }
            }
            ptx_fallback_list.push((name, src));
        }

        // Step 2: Load remaining modules from PTX in parallel across CPU threads.
        let ptx_jit_count = ptx_fallback_list.len();
        if !ptx_fallback_list.is_empty() {
            match load_ptx_modules_parallel(&ctx, &ptx_fallback_list) {
                Ok(loaded) => {
                    for (name, module) in loaded {
                        if let Some(prev) = raw_modules.insert(name, module.0) {
                            // SAFETY: `prev` is a live CUmodule owned by `ctx` replaced by
                            // a duplicate module entry.
                            unsafe {
                                let _ = cuModuleUnload(prev);
                            }
                        }
                    }
                }
                Err(err) => {
                    let _ = ctx.bind_to_thread();
                    for raw_mod in raw_modules.into_values() {
                        // SAFETY: `raw_mod` was returned by `cuModuleLoadData` on `ctx`
                        // and is being discarded before returning the error.
                        unsafe {
                            let _ = cuModuleUnload(raw_mod);
                        }
                    }
                    return Err(err);
                }
            }
            // Safety belt: guarantee the main thread remains bound to `ctx`.
            if let Err(err) = ctx.bind_to_thread() {
                for raw_mod in raw_modules.into_values() {
                    // SAFETY: `raw_mod` was returned by `cuModuleLoadData` on `ctx`
                    // and is being discarded before returning the error.
                    unsafe {
                        let _ = cuModuleUnload(raw_mod);
                    }
                }
                return Err(AtlasError::CudaDriver(err));
            }
        }

        let stored_sources: HashMap<&'static str, Box<str>> = ptx_sources
            .iter()
            .map(|&(name, src)| (name, Box::<str>::from(src)))
            .collect();

        let elapsed_ms = init_start.elapsed().as_millis();
        eprintln!(
            "ATLAS_CUDA_MODULES_LOADED ordinal={ordinal} total={} aot_cubin={aot_cubin_count} ptx_jit={ptx_jit_count} elapsed_ms={elapsed_ms}",
            ptx_sources.len()
        );

        Ok(AtlasRegistry {
            ctx,
            stream,
            initialization_identity,
            ptx_sources: stored_sources,
            modules: Mutex::new(HashMap::new()),
            raw_modules,
        })
    }

    /// Exact ordinal and ordered module-name/PTX receipt loaded by this singleton.
    pub fn initialization_identity(&self) -> &str {
        &self.initialization_identity
    }

    /// The most recently initialized/selected registry (panics if none).
    /// Callers holding an `AtlasCudaBackend` should use its own registry.
    pub fn get() -> &'static Self {
        CURRENT
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .expect("AtlasRegistry not initialized — call get_or_init first")
    }

    /// Look up a cached function handle (cudarc safe API).
    ///
    /// Lazily loads the cudarc [`CudaModule`] on first call for `module_name` so
    /// startup never duplicates module compilation for callers that only use the
    /// raw CUDA driver API.
    pub fn function(&self, module_name: &str, func_name: &str) -> Result<CudaFunction> {
        let cached = {
            let guard = self
                .modules
                .lock()
                .expect("AtlasRegistry module cache lock poisoned");
            guard.get(module_name).cloned()
        };
        let module = match cached {
            Some(existing) => existing,
            None => {
                let (&static_name, src) =
                    self.ptx_sources.get_key_value(module_name).ok_or_else(|| {
                        AtlasError::ModuleLoad(format!("Module '{module_name}' not loaded"))
                    })?;
                let src: &str = src;
                let loaded = if aot_cubin_enabled()
                    && let Some(cubin) = lookup_aot_cubin(static_name, src)
                {
                    match self.ctx.load_module(Ptx::from_binary(cubin.into_owned())) {
                        Ok(m) => m,
                        Err(_) => self
                            .ctx
                            .load_module(Ptx::from_src(src))
                            .map_err(|e| AtlasError::ModuleLoad(format!("{static_name}: {e}")))?,
                    }
                } else {
                    self.ctx
                        .load_module(Ptx::from_src(src))
                        .map_err(|e| AtlasError::ModuleLoad(format!("{static_name}: {e}")))?
                };
                let mut guard = self
                    .modules
                    .lock()
                    .expect("AtlasRegistry module cache lock poisoned");
                Arc::clone(guard.entry(static_name).or_insert(loaded))
            }
        };
        module
            .load_function(func_name)
            .map_err(|e| AtlasError::ModuleLoad(format!("{module_name}::{func_name}: {e}")))
    }

    /// Look up a function handle with OnceLock caching (cudarc safe API).
    pub fn function_cached(
        &self,
        cache: &OnceLock<CudaFunction>,
        module_name: &str,
        func_name: &str,
    ) -> Result<CudaFunction> {
        if let Some(f) = cache.get() {
            return Ok(f.clone());
        }
        let func = self.function(module_name, func_name)?;
        let _ = cache.set(func.clone());
        Ok(func)
    }

    /// Look up a raw CUfunction handle with OnceLock caching.
    /// Uses the raw CUDA driver API — no cudarc struct layout dependency.
    pub fn raw_function_cached(
        &self,
        cache: &OnceLock<RawCudaFunc>,
        module_name: &str,
        func_name: &str,
    ) -> Result<RawCudaFunc> {
        if let Some(f) = cache.get() {
            return Ok(*f);
        }
        let raw_mod = self
            .raw_modules
            .get(module_name)
            .ok_or_else(|| AtlasError::ModuleLoad(format!("Module '{module_name}' not loaded")))?;
        let c_name = CString::new(func_name).map_err(|e| {
            AtlasError::ModuleLoad(format!("{module_name}::{func_name}: CString: {e}"))
        })?;
        let mut func: *mut c_void = std::ptr::null_mut();
        let status =
            // SAFETY: pointer cast handles the platform difference between
            // `c_char = i8` (x86_64) and `c_char = u8` (aarch64); we use
            // `.cast()` rather than `as *const i8` so clippy's
            // `unnecessary_cast` is satisfied on x86_64 builds while the
            // call still type-checks on aarch64 (Atlas's actual GB10 target).
            unsafe { cuModuleGetFunction(&mut func, *raw_mod, c_name.as_ptr().cast()) };
        if status != 0 {
            return Err(AtlasError::ModuleLoad(format!(
                "{module_name}::{func_name}: cuModuleGetFunction failed: {}",
                cuda_error_text(status)
            )));
        }
        let raw = RawCudaFunc(func);
        let _ = cache.set(raw);
        Ok(raw)
    }

    /// Get the raw CUstream handle for Atlas's own stream.
    pub fn raw_stream(&self) -> u64 {
        self.stream.cu_stream() as u64
    }

    /// Resolve a `__device__` symbol in a loaded PTX module to its device
    /// pointer + byte length. Required for drivers that read/write device
    /// globals without launching a kernel (e.g. InnerQ calibration state).
    /// `symbol` must be the linker-visible name — C++ namespace symbols are
    /// Itanium-mangled (`_ZN7tq_plus14d_innerq_scaleE`).
    pub fn device_symbol(&self, module_name: &str, symbol: &str) -> Result<(u64, usize)> {
        let raw_mod = self
            .raw_modules
            .get(module_name)
            .ok_or_else(|| AtlasError::ModuleLoad(format!("Module '{module_name}' not loaded")))?;
        let c_sym = CString::new(symbol).map_err(|e| {
            AtlasError::ModuleLoad(format!("{module_name}::{symbol}: CString: {e}"))
        })?;
        let mut dptr: u64 = 0;
        let mut bytes: usize = 0;
        // SAFETY: `*raw_mod` is a live CUmodule loaded into the registry's bound
        // CUDA context, `c_sym` is NUL-terminated, and `dptr`/`bytes` are valid out-ptrs.
        let status =
            unsafe { cuModuleGetGlobal_v2(&mut dptr, &mut bytes, *raw_mod, c_sym.as_ptr().cast()) };
        if status != 0 {
            return Err(AtlasError::ModuleLoad(format!(
                "{module_name}::{symbol}: cuModuleGetGlobal_v2 failed: {}",
                cuda_error_text(status)
            )));
        }
        Ok((dptr, bytes))
    }

    /// Copy a group of pageable host byte slices to device memory in order.
    ///
    /// The producer stream is drained once before the synchronous copies.
    /// A failing member suppresses every later member in the group.
    ///
    /// # Safety
    /// - Every `(dst, src)` destination `dst` must be a valid writable CUDA
    ///   device address of at least `src.len()` bytes in the active context.
    /// - `stream` must be `0` (legacy stream) or a valid `CUstream` handle in
    ///   the active CUDA context.
    pub unsafe fn copy_h2d_group(&self, copies: &[(u64, &[u8])], stream: u64) -> Result<()> {
        if copies.is_empty() {
            return Ok(());
        }
        // SAFETY: Caller guarantees `stream` is a valid CUstream handle (or 0).
        unsafe { self.stream_synchronize(stream)? };
        for (index, (dst, src)) in copies.iter().enumerate() {
            // SAFETY: Caller guarantees `*dst` points to at least `src.len()`
            // writable device bytes in the active CUDA context.
            let status = unsafe { cuMemcpyHtoD_v2(*dst, src.as_ptr().cast::<c_void>(), src.len()) };
            if status != 0 {
                return Err(AtlasError::KernelLaunch(format!(
                    "cuMemcpyHtoD_v2 group member {index} failed: {}",
                    cuda_error_text(status)
                )));
            }
        }
        Ok(())
    }

    /// Copy device bytes into ordinary pageable host storage.
    ///
    /// The producer stream is drained before the streamless synchronous copy.
    ///
    /// # Safety
    /// - `src` must be a valid readable CUDA device address of at least
    ///   `dst.len()` bytes in the active context.
    /// - `stream` must be `0` (legacy stream) or a valid `CUstream` handle in
    ///   the active CUDA context.
    pub unsafe fn copy_d2h(&self, dst: &mut [u8], src: u64, stream: u64) -> Result<()> {
        // SAFETY: Caller guarantees `stream` is a valid CUstream handle (or 0).
        unsafe { self.stream_synchronize(stream)? };
        // SAFETY: Caller guarantees `src` points to at least `dst.len()`
        // readable device bytes in the active CUDA context.
        let status = unsafe { cuMemcpyDtoH_v2(dst.as_mut_ptr().cast::<c_void>(), src, dst.len()) };
        if status != 0 {
            return Err(AtlasError::KernelLaunch(format!(
                "cuMemcpyDtoH_v2 failed: {}",
                cuda_error_text(status)
            )));
        }
        Ok(())
    }

    /// Block the calling thread until all prior work on `stream` completes.
    ///
    /// # Safety
    /// `stream` must be `0` (legacy stream) or a valid live `CUstream` handle
    /// belonging to the active CUDA context.
    pub unsafe fn stream_synchronize(&self, stream: u64) -> Result<()> {
        // SAFETY: Caller guarantees `stream` is 0 or a valid live CUstream handle.
        let status = unsafe { cuStreamSynchronize(stream) };
        if status != 0 {
            return Err(AtlasError::KernelLaunch(format!(
                "cuStreamSynchronize failed: {}",
                cuda_error_text(status)
            )));
        }
        Ok(())
    }

    /// Launch a kernel on a specified raw CUDA stream.
    ///
    /// Passing the caller's native `CUstream` handle ensures kernels are
    /// captured during CUDA graph recording and properly stream-ordered.
    ///
    /// # Safety
    /// - `kernel_params` must contain valid pointers to arguments matching the kernel signature.
    /// - `stream_ptr` must be a valid CUstream handle, or `0` for CUDA's legacy default stream.
    /// - `raw_func` must be a valid CUfunction obtained from `raw_function_cached`.
    pub unsafe fn launch_on_stream(
        &self,
        raw_func: RawCudaFunc,
        cfg: LaunchConfig,
        stream_ptr: u64,
        kernel_params: &mut [*mut c_void],
    ) -> Result<()> {
        // Always use the caller's stream directly. When stream_ptr=0, CUDA
        // treats it as the legacy default stream which has implicit
        // synchronization with all other streams in the same context.
        // Never fall back to the registry's private stream — that breaks
        // ordering with caller-stream operations and prevents CUDA graph capture.
        let stream = stream_ptr;
        // Opt in to >48KB dynamic shared memory when requested.
        if cfg.shared_mem_bytes > 48 * 1024 {
            const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: i32 = 8;
            // SAFETY: Caller guarantees `raw_func` is a valid CUfunction in the active context.
            let attr_status = unsafe {
                cuFuncSetAttribute(
                    raw_func.0,
                    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    cfg.shared_mem_bytes as i32,
                )
            };
            if attr_status != 0 {
                return Err(AtlasError::KernelLaunch(format!(
                    "cuFuncSetAttribute(MAX_DYNAMIC_SHARED={}) failed: {}",
                    cfg.shared_mem_bytes,
                    cuda_error_text(attr_status)
                )));
            }
        }
        // SAFETY: Caller upholds the `# Safety` contract for `raw_func`,
        // `stream_ptr`, `cfg`, and `kernel_params`.
        let status = unsafe {
            cuLaunchKernel(
                raw_func.0,
                cfg.grid_dim.0,
                cfg.grid_dim.1,
                cfg.grid_dim.2,
                cfg.block_dim.0,
                cfg.block_dim.1,
                cfg.block_dim.2,
                cfg.shared_mem_bytes,
                stream as *mut c_void,
                kernel_params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if status != 0 {
            return Err(AtlasError::KernelLaunch(format!(
                "cuLaunchKernel failed: {} (grid=[{},{},{}], block=[{},{},{}], shared_mem={})",
                cuda_error_text(status),
                cfg.grid_dim.0,
                cfg.grid_dim.1,
                cfg.grid_dim.2,
                cfg.block_dim.0,
                cfg.block_dim.1,
                cfg.block_dim.2,
                cfg.shared_mem_bytes
            )));
        }
        Ok(())
    }

    /// Launch a kernel with CUDA Programmatic Dependent Launch (`PDL`,
    /// `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION = 6`) when
    /// `pdl && cuda_pdl_enabled()`. Falls back cleanly to [`Self::launch_on_stream`]
    /// if disabled.
    ///
    /// # Safety
    /// Same safety requirements as [`Self::launch_on_stream`].
    pub unsafe fn launch_on_stream_pdl(
        &self,
        raw_func: RawCudaFunc,
        cfg: LaunchConfig,
        stream_ptr: u64,
        kernel_params: &mut [*mut c_void],
        pdl: bool,
    ) -> Result<()> {
        if !pdl || !cuda_pdl_enabled() {
            // SAFETY: Caller satisfies `launch_on_stream`'s `# Safety` contract.
            return unsafe { self.launch_on_stream(raw_func, cfg, stream_ptr, kernel_params) };
        }
        if cfg.shared_mem_bytes > 48 * 1024 {
            const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: i32 = 8;
            // SAFETY: Caller guarantees `raw_func` is a valid CUfunction in the active context.
            let attr_status = unsafe {
                cuFuncSetAttribute(
                    raw_func.0,
                    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    cfg.shared_mem_bytes as i32,
                )
            };
            if attr_status != 0 {
                return Err(AtlasError::KernelLaunch(format!(
                    "cuFuncSetAttribute(MAX_DYNAMIC_SHARED={}) failed: {}",
                    cfg.shared_mem_bytes,
                    cuda_error_text(attr_status)
                )));
            }
        }
        let mut attr = CuLaunchAttribute::pdl_serialization(1);
        let launch_cfg = CuLaunchConfig {
            grid_dim_x: cfg.grid_dim.0,
            grid_dim_y: cfg.grid_dim.1,
            grid_dim_z: cfg.grid_dim.2,
            block_dim_x: cfg.block_dim.0,
            block_dim_y: cfg.block_dim.1,
            block_dim_z: cfg.block_dim.2,
            shared_mem_bytes: cfg.shared_mem_bytes,
            h_stream: stream_ptr as *mut c_void,
            attrs: &mut attr,
            num_attrs: 1,
        };
        // SAFETY: `launch_cfg` and `attr` live for the duration of the call, match
        // CUDA 12's `CUlaunchConfig`/`CUlaunchAttribute` ABI (verified by size assertions),
        // and the caller guarantees `raw_func`, `stream_ptr`, and `kernel_params` are valid.
        let status = unsafe {
            cuLaunchKernelEx(
                &launch_cfg,
                raw_func.0,
                kernel_params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if status != 0 {
            return Err(AtlasError::KernelLaunch(format!(
                "cuLaunchKernelEx(PDL) failed: {} (grid=[{},{},{}], block=[{},{},{}], shared_mem={})",
                cuda_error_text(status),
                cfg.grid_dim.0,
                cfg.grid_dim.1,
                cfg.grid_dim.2,
                cfg.block_dim.0,
                cfg.block_dim.1,
                cfg.block_dim.2,
                cfg.shared_mem_bytes
            )));
        }
        Ok(())
    }
}

fn load_single_ptx_raw(name: &'static str, src: &str) -> Result<SendCuModule> {
    let src_nul =
        CString::new(src).map_err(|e| AtlasError::ModuleLoad(format!("{name}: CString: {e}")))?;
    let mut raw_mod: *mut c_void = std::ptr::null_mut();
    // SAFETY: `src_nul` is a NUL-terminated PTX string valid for the call, and
    // the caller has bound the owning CUcontext to the current thread.
    let status = unsafe { cuModuleLoadData(&mut raw_mod, src_nul.as_ptr().cast::<c_void>()) };
    if status != 0 {
        return Err(AtlasError::ModuleLoad(format!(
            "{name}: cuModuleLoadData failed: {}",
            cuda_error_text(status)
        )));
    }
    Ok(SendCuModule(raw_mod))
}

fn load_ptx_modules_parallel(
    ctx: &CudaContext,
    modules: &[(&'static str, &str)],
) -> Result<Vec<(&'static str, SendCuModule)>> {
    let cu_ctx = ctx.cu_ctx().cast::<c_void>();
    if cu_ctx.is_null() {
        return Err(AtlasError::ModuleLoad(
            "CudaContext returned null CUcontext during parallel PTX load".to_string(),
        ));
    }

    let available = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1);
    let worker_count = modules.len().min(available.clamp(1, 16));
    let send_ctx = SendCuContext(cu_ctx);

    // Longest-processing-time-first (largest PTX first) so multi-second modules
    // start at t=0 rather than tail-stalling behind smaller ones; original
    // indices are preserved so insertion order stays deterministic.
    let mut work: Vec<(usize, &'static str, &str)> = modules
        .iter()
        .enumerate()
        .map(|(i, &(name, src))| (i, name, src))
        .collect();
    work.sort_by(|a, b| b.2.len().cmp(&a.2.len()).then_with(|| a.0.cmp(&b.0)));

    if worker_count <= 1 {
        let mut loaded = Vec::with_capacity(work.len());
        for (orig_idx, name, src) in work {
            match load_single_ptx_raw(name, src) {
                Ok(raw_mod) => loaded.push((orig_idx, name, raw_mod)),
                Err(err) => {
                    for (_, _, m) in loaded {
                        // SAFETY: `m.0` was loaded on `ctx` and is being discarded on error.
                        unsafe {
                            let _ = cuModuleUnload(m.0);
                        }
                    }
                    return Err(err);
                }
            }
        }
        loaded.sort_by_key(|(orig_idx, _, _)| *orig_idx);
        return Ok(loaded.into_iter().map(|(_, name, m)| (name, m)).collect());
    }

    let next_idx = AtomicUsize::new(0);
    let failed = AtomicBool::new(false);

    type LoadedModuleSlot = (usize, &'static str, SendCuModule);
    type WorkerLoadResult = (Vec<LoadedModuleSlot>, Option<AtlasError>);

    let worker_results: Vec<WorkerLoadResult> = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for _ in 0..worker_count {
            let work_ref = &work;
            let next_ref = &next_idx;
            let failed_ref = &failed;
            let worker_ctx = send_ctx;
            handles.push(scope.spawn(move || {
                let mut local_loaded = Vec::new();
                // SAFETY: `worker_ctx` holds a valid CUcontext owned by `ctx`,
                // which outlives this `std::thread::scope`.
                let bind_status = unsafe { cuCtxSetCurrent(worker_ctx.as_ptr()) };
                if bind_status != 0 {
                    failed_ref.store(true, Ordering::Relaxed);
                    return (
                        local_loaded,
                        Some(AtlasError::ModuleLoad(format!(
                            "cuCtxSetCurrent failed on PTX loader worker: {}",
                            cuda_error_text(bind_status)
                        ))),
                    );
                }
                loop {
                    if failed_ref.load(Ordering::Relaxed) {
                        break;
                    }
                    let idx = next_ref.fetch_add(1, Ordering::Relaxed);
                    let Some(&(orig_idx, name, src)) = work_ref.get(idx) else {
                        break;
                    };
                    match load_single_ptx_raw(name, src) {
                        Ok(raw_mod) => local_loaded.push((orig_idx, name, raw_mod)),
                        Err(err) => {
                            failed_ref.store(true, Ordering::Relaxed);
                            return (local_loaded, Some(err));
                        }
                    }
                }
                (local_loaded, None)
            }));
        }
        handles
            .into_iter()
            .map(|h| match h.join() {
                Ok(res) => res,
                Err(_) => (
                    Vec::new(),
                    Some(AtlasError::ModuleLoad(
                        "PTX loader worker thread panicked".to_string(),
                    )),
                ),
            })
            .collect()
    });

    let mut first_error: Option<AtlasError> = None;
    let mut all_loaded: Vec<(usize, &'static str, SendCuModule)> =
        Vec::with_capacity(modules.len());
    for (loaded, err) in worker_results {
        all_loaded.extend(loaded);
        if first_error.is_none() {
            first_error = err;
        }
    }

    if let Some(err) = first_error {
        let _ = ctx.bind_to_thread();
        for (_, _, m) in all_loaded {
            // SAFETY: `m.0` was loaded on `ctx` during this failed init attempt and is
            // unloaded cleanly before returning the error.
            unsafe {
                let _ = cuModuleUnload(m.0);
            }
        }
        return Err(err);
    }

    all_loaded.sort_by_key(|(orig_idx, _, _)| *orig_idx);
    Ok(all_loaded
        .into_iter()
        .map(|(_, name, m)| (name, m))
        .collect())
}

const fn cuda_pdl_supported_for_cc(cc: Option<(u32, u32)>) -> bool {
    match cc {
        Some((major, _)) => major >= 9,
        None => true,
    }
}

pub fn cuda_pdl_enabled() -> bool {
    static ENV_ENABLED: OnceLock<bool> = OnceLock::new();
    let env_enabled =
        *ENV_ENABLED.get_or_init(|| std::env::var("ATLAS_CUDA_PDL").ok().as_deref() != Some("0"));
    env_enabled && cuda_pdl_supported_for_cc(crate::device::active_compute_capability())
}

#[repr(C)]
struct CuLaunchAttribute {
    id: u32,
    pad: [u8; 4],
    value_words: [u64; 8],
}

impl CuLaunchAttribute {
    fn pdl_serialization(allowed: i32) -> Self {
        // CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION = 6
        let mut value_words = [0u64; 8];
        value_words[0] = (allowed as u32) as u64;
        Self {
            id: 6,
            pad: [0u8; 4],
            value_words,
        }
    }
}

#[repr(C)]
struct CuLaunchConfig {
    grid_dim_x: u32,
    grid_dim_y: u32,
    grid_dim_z: u32,
    block_dim_x: u32,
    block_dim_y: u32,
    block_dim_z: u32,
    shared_mem_bytes: u32,
    h_stream: *mut c_void,
    attrs: *mut CuLaunchAttribute,
    num_attrs: u32,
}

const _: () = assert!(std::mem::size_of::<CuLaunchAttribute>() == 72);
const _: () = assert!(std::mem::size_of::<CuLaunchConfig>() == 56);

// SAFETY: `cuLaunchKernelEx` is exported by `libcuda.so` (CUDA 12+ Driver API)
// matching the C ABI verified by the `CuLaunchConfig`/`CuLaunchAttribute` size assertions above.
unsafe extern "C" {
    fn cuLaunchKernelEx(
        config: *const CuLaunchConfig,
        f: *mut c_void,
        kernelParams: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> i32;
}

#[cfg(test)]
mod initialization_identity_tests {
    use std::sync::OnceLock;

    use super::{admit_initialization_request, initialization_identity};

    #[test]
    fn identity_binds_offset_2048_order_count_and_ordinal() {
        let original = "x".repeat(2_049);
        let mut changed = original.clone();
        changed.replace_range(2_048..2_049, "y");
        let a = [("quantize_nvfp4", original.as_str())];
        let b = [("quantize_nvfp4", changed.as_str())];
        let reordered = [("other", "z"), ("quantize_nvfp4", original.as_str())];
        let loaded = initialization_identity(0, &a);
        let changed_offset = initialization_identity(0, &b);
        let changed_ordinal = initialization_identity(1, &a);
        let changed_order_count = initialization_identity(0, &reordered);
        let same_process = OnceLock::new();
        assert_eq!(
            admit_initialization_request(&same_process, &loaded).unwrap(),
            loaded
        );
        for requested in [&changed_offset, &changed_ordinal, &changed_order_count] {
            assert!(admit_initialization_request(&same_process, requested).is_err());
        }

        // Keyed selection: a request is served only by a registry with its
        // exact identity, never by one that loaded a different module set.
        let slots = [(loaded.as_str(), 1u32), (changed_offset.as_str(), 2u32)];
        assert_eq!(super::select_registry(&slots, &loaded), Some(&1));
        assert_eq!(super::select_registry(&slots, &changed_offset), Some(&2));
        assert_eq!(super::select_registry(&slots, &changed_ordinal), None);
        assert_eq!(super::select_registry(&slots, &changed_order_count), None);
    }
}

#[cfg(test)]
mod aot_cubin_tests {
    use super::{
        AotCubinEncoding, decode_obf_bytes, lookup_aot_cubin, parse_disable_aot_cubin,
        register_aot_cubin, sha256_bytes,
    };

    #[test]
    fn registration_exact_sha256_hit_and_mutated_ptx_miss() {
        const PTX: &str =
            ".version 8.8\n.target sm_121f\n.address_size 64\n.visible .entry k() { ret; }\n";
        const ELF: &[u8] = &[0x7f, b'E', b'L', b'F', 0x02, 0x01, 0x00, 0x00, 0xde, 0xad];

        let digest = sha256_bytes(PTX.as_bytes());
        register_aot_cubin(
            "__test_registry_exact_hit",
            PTX.len(),
            digest,
            ELF,
            AotCubinEncoding::Raw,
        );

        let found =
            lookup_aot_cubin("__test_registry_exact_hit", PTX).expect("exact PTX must match");
        assert_eq!(found.as_ref(), ELF);

        // Any PTX edit (even same length) must miss and fall back to PTX JIT.
        let mutated = PTX.replace("k()", "j()");
        assert_eq!(mutated.len(), PTX.len());
        assert!(lookup_aot_cubin("__test_registry_exact_hit", &mutated).is_none());
    }

    #[test]
    fn obfuscated_cubin_with_embedded_nul_bytes_decodes_verbatim() {
        const PTX: &str = ".version 8.8\n.target sm_121a\n.address_size 64\n";
        const ELF_WITH_NULS: &[u8] = &[
            0x7f, b'E', b'L', b'F', 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x79, 0x00, 0xff, 0x00,
        ];

        let obfuscated: &'static [u8] =
            Box::leak(decode_obf_bytes(ELF_WITH_NULS).into_boxed_slice());
        assert_ne!(obfuscated, ELF_WITH_NULS);

        register_aot_cubin(
            "__test_registry_obf_nul",
            PTX.len(),
            sha256_bytes(PTX.as_bytes()),
            obfuscated,
            AotCubinEncoding::Obfuscated,
        );

        let decoded =
            lookup_aot_cubin("__test_registry_obf_nul", PTX).expect("obfuscated CUBIN must match");
        assert_eq!(decoded.as_ref(), ELF_WITH_NULS);
    }

    #[test]
    fn disable_aot_cubin_env_parsing() {
        for enabled_val in [None, Some(""), Some("0"), Some("false"), Some("no")] {
            assert!(
                !parse_disable_aot_cubin(enabled_val),
                "expected AOT CUBIN enabled for {enabled_val:?}"
            );
        }
        for disabled_val in [
            Some("1"),
            Some(" 1 "),
            Some("true"),
            Some("TRUE"),
            Some("yes"),
            Some("YES"),
        ] {
            assert!(
                parse_disable_aot_cubin(disabled_val),
                "expected AOT CUBIN disabled for {disabled_val:?}"
            );
        }
    }

    #[test]
    fn cuda_pdl_requires_hopper_or_newer_compute_capability() {
        assert!(!super::cuda_pdl_supported_for_cc(Some((8, 0))));
        assert!(!super::cuda_pdl_supported_for_cc(Some((8, 6))));
        assert!(!super::cuda_pdl_supported_for_cc(Some((8, 9))));
        assert!(super::cuda_pdl_supported_for_cc(Some((9, 0))));
        assert!(super::cuda_pdl_supported_for_cc(Some((10, 0))));
        assert!(super::cuda_pdl_supported_for_cc(Some((10, 3))));
        assert!(super::cuda_pdl_supported_for_cc(Some((12, 0))));
        assert!(super::cuda_pdl_supported_for_cc(Some((12, 1))));
        assert!(super::cuda_pdl_supported_for_cc(None));
    }
}
